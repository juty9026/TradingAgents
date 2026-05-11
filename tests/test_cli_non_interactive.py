import datetime as dt

import pytest

import cli.main as cli_main
from cli.models import AnalystType
from tradingagents.default_config import DEFAULT_CONFIG


def analyst_values(analysts):
    return [analyst.value for analyst in analysts]


def test_openai_reasoning_effort_default_is_medium():
    assert DEFAULT_CONFIG["openai_reasoning_effort"] == "medium"


def test_parse_research_depth_accepts_names_and_round_aliases():
    assert cli_main._parse_research_depth(None) == 1
    assert cli_main._parse_research_depth("shallow") == 1
    assert cli_main._parse_research_depth("medium") == 3
    assert cli_main._parse_research_depth("deep") == 5
    assert cli_main._parse_research_depth("1") == 1
    assert cli_main._parse_research_depth("3") == 3
    assert cli_main._parse_research_depth("5") == 5


@pytest.mark.parametrize("value", ["", "2", "fast", "deepest"])
def test_parse_research_depth_rejects_unknown_values(value):
    with pytest.raises(ValueError, match="Invalid research depth"):
        cli_main._parse_research_depth(value)


def test_parse_analysts_defaults_to_full_analyst_team():
    assert cli_main._parse_analysts(None) == [
        AnalystType.MARKET,
        AnalystType.SOCIAL,
        AnalystType.NEWS,
        AnalystType.FUNDAMENTALS,
    ]


def test_parse_analysts_accepts_comma_separated_values_in_canonical_order():
    analysts = cli_main._parse_analysts("news,market,news")

    assert analyst_values(analysts) == ["market", "news"]


def test_parse_analysts_rejects_unknown_values():
    with pytest.raises(ValueError, match="Invalid analysts: macro"):
        cli_main._parse_analysts("market,macro")


def test_parse_analysis_date_defaults_to_today():
    today = dt.date(2026, 5, 9)

    assert cli_main._parse_analysis_date(None, today=today) == "2026-05-09"


def test_parse_analysis_date_accepts_valid_past_date():
    today = dt.date(2026, 5, 9)

    assert cli_main._parse_analysis_date("2026-05-01", today=today) == "2026-05-01"


def test_parse_analysis_date_rejects_invalid_format():
    today = dt.date(2026, 5, 9)

    with pytest.raises(ValueError, match="Invalid analysis date"):
        cli_main._parse_analysis_date("05/09/2026", today=today)


def test_parse_analysis_date_rejects_future_dates():
    today = dt.date(2026, 5, 9)

    with pytest.raises(ValueError, match="Analysis date cannot be in the future"):
        cli_main._parse_analysis_date("2026-05-10", today=today)


def test_parse_output_language_defaults_to_configured_default():
    assert cli_main._parse_output_language(None) == "English"


def test_parse_output_language_accepts_custom_language_and_trims():
    assert cli_main._parse_output_language(" Korean ") == "Korean"


@pytest.mark.parametrize("value", ["", "   "])
def test_parse_output_language_rejects_empty_values(value):
    with pytest.raises(
        ValueError,
        match="--output-language must be a non-empty language name",
    ):
        cli_main._parse_output_language(value)


def test_build_non_interactive_selections_requires_ticker():
    today = dt.date(2026, 5, 9)

    with pytest.raises(ValueError, match="--ticker is required"):
        cli_main.build_non_interactive_selections(
            ticker=None,
            analysis_date=None,
            analysts=None,
            research_depth=None,
            today=today,
        )


def test_build_non_interactive_selections_uses_safe_defaults():
    today = dt.date(2026, 5, 9)

    selections = cli_main.build_non_interactive_selections(
        ticker="spy",
        analysis_date=None,
        analysts=None,
        research_depth=None,
        today=today,
    )

    assert selections["ticker"] == "SPY"
    assert selections["analysis_date"] == "2026-05-09"
    assert analyst_values(selections["analysts"]) == [
        "market",
        "social",
        "news",
        "fundamentals",
    ]
    assert selections["research_depth"] == 1
    assert selections["llm_provider"] == "openai"
    assert selections["backend_url"] is None
    assert selections["shallow_thinker"] == "gpt-5.4-mini"
    assert selections["deep_thinker"] == "gpt-5.4"
    assert selections["openai_reasoning_effort"] == "medium"
    assert selections["output_language"] == "English"


def test_build_non_interactive_selections_accepts_output_language_override():
    today = dt.date(2026, 5, 9)

    selections = cli_main.build_non_interactive_selections(
        ticker="spy",
        analysis_date=None,
        analysts=None,
        research_depth=None,
        output_language=" Korean ",
        today=today,
    )

    assert selections["output_language"] == "Korean"


from pathlib import Path

from typer.testing import CliRunner


runner = CliRunner()


def test_resolve_report_save_path_uses_explicit_path():
    now = dt.datetime(2026, 5, 9, 14, 30, 5)

    assert cli_main._resolve_report_save_path("SPY", Path("reports/spy"), now=now) == Path("reports/spy")


def test_resolve_report_save_path_builds_timestamped_default_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    now = dt.datetime(2026, 5, 9, 14, 30, 5)

    assert cli_main._resolve_report_save_path("SPY", None, now=now) == (
        tmp_path / "reports" / "SPY_20260509_143005"
    )


def test_non_interactive_cli_requires_ticker(monkeypatch):
    calls = []

    def fake_run_analysis(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(cli_main, "run_analysis", fake_run_analysis)

    result = runner.invoke(cli_main.app, ["--non-interactive"])

    assert result.exit_code != 0
    assert "--ticker is required when --non-interactive is used" in result.output
    assert calls == []


def test_non_interactive_cli_forwards_defaults(monkeypatch):
    calls = []

    def fake_run_analysis(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(cli_main, "run_analysis", fake_run_analysis)

    result = runner.invoke(cli_main.app, ["--non-interactive", "--ticker", "spy"])

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["checkpoint"] is False
    assert calls[0]["selections"]["ticker"] == "SPY"
    assert calls[0]["selections"]["research_depth"] == 1
    assert analyst_values(calls[0]["selections"]["analysts"]) == [
        "market",
        "social",
        "news",
        "fundamentals",
    ]
    assert calls[0]["non_interactive"] is True
    assert calls[0]["save_report"] is True
    assert calls[0]["display_report"] is True
    assert calls[0]["save_path"] is None


def test_non_interactive_cli_forwards_overrides(monkeypatch):
    calls = []

    def fake_run_analysis(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(cli_main, "run_analysis", fake_run_analysis)

    result = runner.invoke(
        cli_main.app,
        [
            "--non-interactive",
            "--ticker",
            "7203.t",
            "--analysis-date",
            "2026-05-01",
            "--analysts",
            "market,news",
            "--research-depth",
            "deep",
            "--no-display-report",
            "--save-path",
            "reports/toyota",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    selections = calls[0]["selections"]
    assert selections["ticker"] == "7203.T"
    assert selections["analysis_date"] == "2026-05-01"
    assert analyst_values(selections["analysts"]) == ["market", "news"]
    assert selections["research_depth"] == 5
    assert calls[0]["save_report"] is True
    assert calls[0]["display_report"] is False
    assert calls[0]["save_path"] == Path("reports/toyota")


def test_non_interactive_cli_forwards_output_language(monkeypatch):
    calls = []

    def fake_run_analysis(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(cli_main, "run_analysis", fake_run_analysis)

    result = runner.invoke(
        cli_main.app,
        [
            "--non-interactive",
            "--ticker",
            "SPY",
            "--output-language",
            "Korean",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0]["selections"]["output_language"] == "Korean"


@pytest.mark.parametrize("value", ["", "   "])
def test_non_interactive_cli_rejects_empty_output_language(value, monkeypatch):
    calls = []

    def fake_run_analysis(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(cli_main, "run_analysis", fake_run_analysis)

    result = runner.invoke(
        cli_main.app,
        [
            "--non-interactive",
            "--ticker",
            "SPY",
            "--output-language",
            value,
        ],
    )

    assert result.exit_code != 0
    assert "--output-language must be a non-empty language name" in result.output
    assert calls == []


def test_save_path_cannot_be_used_when_report_saving_is_disabled(monkeypatch):
    calls = []

    def fake_run_analysis(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(cli_main, "run_analysis", fake_run_analysis)

    result = runner.invoke(
        cli_main.app,
        [
            "--non-interactive",
            "--ticker",
            "SPY",
            "--no-save-report",
            "--save-path",
            "reports/spy",
        ],
    )

    assert result.exit_code != 0
    assert "--save-path cannot be used with --no-save-report" in result.output
    assert calls == []


@pytest.mark.parametrize(
    ("args", "expected_error"),
    [
        (["--no-save-report"], "--no-save-report requires --non-interactive"),
        (["--no-display-report"], "--no-display-report requires --non-interactive"),
        (["--save-path", "reports/spy"], "--save-path requires --non-interactive"),
        (["--output-language", "Korean"], "--output-language requires --non-interactive"),
    ],
)
def test_report_control_options_require_non_interactive(args, expected_error, monkeypatch):
    calls = []

    def fake_run_analysis(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(cli_main, "run_analysis", fake_run_analysis)

    result = runner.invoke(cli_main.app, args)

    assert result.exit_code != 0
    assert expected_error in result.output
    assert calls == []


def test_clear_checkpoints_waits_for_valid_non_interactive_args(monkeypatch):
    cleared_paths = []
    analysis_calls = []

    def fake_clear_all_checkpoints(path):
        cleared_paths.append(path)
        return 1

    def fake_run_analysis(**kwargs):
        analysis_calls.append(kwargs)

    monkeypatch.setattr(
        "tradingagents.graph.checkpointer.clear_all_checkpoints",
        fake_clear_all_checkpoints,
    )
    monkeypatch.setattr(cli_main, "run_analysis", fake_run_analysis)

    result = runner.invoke(cli_main.app, ["--clear-checkpoints", "--non-interactive"])

    assert result.exit_code != 0
    assert "--ticker is required when --non-interactive is used" in result.output
    assert cleared_paths == []
    assert analysis_calls == []


def test_clear_checkpoints_runs_after_valid_non_interactive_args(monkeypatch):
    cleared_paths = []
    analysis_calls = []

    def fake_clear_all_checkpoints(path):
        cleared_paths.append(path)
        return 1

    def fake_run_analysis(**kwargs):
        analysis_calls.append(kwargs)

    monkeypatch.setattr(
        "tradingagents.graph.checkpointer.clear_all_checkpoints",
        fake_clear_all_checkpoints,
    )
    monkeypatch.setattr(cli_main, "run_analysis", fake_run_analysis)

    result = runner.invoke(
        cli_main.app,
        ["--clear-checkpoints", "--non-interactive", "--ticker", "SPY"],
    )

    assert result.exit_code == 0
    assert cleared_paths == [DEFAULT_CONFIG["data_cache_dir"]]
    assert len(analysis_calls) == 1
    assert analysis_calls[0]["selections"]["ticker"] == "SPY"


def test_help_exposes_non_interactive_options_without_llm_overrides():
    result = runner.invoke(cli_main.app, ["--help"])

    assert result.exit_code == 0
    assert "--non-interactive" in result.output
    assert "--ticker" in result.output
    assert "--analysis-date" in result.output
    assert "--analysts" in result.output
    assert "--research-depth" in result.output
    assert "--save-report" in result.output
    assert "--display-report" in result.output
    assert "--save-path" in result.output
    assert "--output-language" in result.output
    assert "--llm-provider" not in result.output
    assert "--quick-think-llm" not in result.output
    assert "--deep-think-llm" not in result.output
