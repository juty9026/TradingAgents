# Non-Interactive TradingAgents CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit `--non-interactive` CLI path for `uv run tradingagents` that runs from safe defaults plus command-line overrides without prompting for input.

**Architecture:** Keep the existing interactive path intact and introduce a small non-interactive selection builder in `cli/main.py`. Reuse `DEFAULT_CONFIG` for LLM provider/model settings instead of exposing new LLM flags, and route both interactive and non-interactive runs through the existing `run_analysis()` execution flow.

**Tech Stack:** Python 3.10+, Typer, pytest, Rich, existing TradingAgents CLI modules.

---

## File Structure

**Modify `cli/main.py`:**
Owns Typer options, interactive selection orchestration, non-interactive selection parsing, report save/display behavior, and validation for conflicting CLI options.

**Modify `tradingagents/default_config.py`:**
Owns default LLM configuration. Change OpenAI reasoning effort from `None` to `"medium"` so non-interactive runs match the current interactive default.

**Create `tests/test_cli_non_interactive.py`:**
Owns focused tests for non-interactive parser behavior, Typer option forwarding, save/display flags, and default config expectations.

**Already created `CONTEXT.md`:**
Captures the domain terms and decisions from the design grilling session. No implementation task needs to modify it unless requirements change.

## CLI Contract

`uv run tradingagents` remains interactive.

`uv run tradingagents --non-interactive --ticker SPY` runs without selection prompts.

`--ticker` is required when `--non-interactive` is set.

`--analysis-date` defaults to today's date and rejects future dates.

`--analysts` defaults to `market,social,news,fundamentals`.

`--research-depth` defaults to `shallow`, accepts `shallow|medium|deep`, and accepts `1|3|5` as aliases.

LLM provider and model options are not exposed as CLI options for non-interactive mode.

`--save-report / --no-save-report` defaults to saving.

`--display-report / --no-display-report` defaults to displaying the full report.

`--save-path PATH` is allowed only when report saving is enabled.

Non-interactive mode skips the welcome ASCII and announcements because those are part of the interactive questionnaire.

## Task 1: Add Non-Interactive Selection Parser Tests

**Files:**
- Create: `tests/test_cli_non_interactive.py`

- [ ] **Step 1: Write failing tests for research depth, analyst parsing, date parsing, and selection defaults**

Create `tests/test_cli_non_interactive.py` with this content:

```python
import datetime as dt

import pytest

import cli.main as cli_main
from cli.models import AnalystType


def analyst_values(analysts):
    return [analyst.value for analyst in analysts]


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
```

- [ ] **Step 2: Run the parser tests and verify they fail**

Run:

```bash
uv run pytest tests/test_cli_non_interactive.py -v
```

Expected: FAIL because `_parse_research_depth`, `_parse_analysts`, `_parse_analysis_date`, and `build_non_interactive_selections` do not exist yet.

- [ ] **Step 3: Commit the failing parser tests**

Run:

```bash
git add tests/test_cli_non_interactive.py
git commit -m "test: specify non-interactive cli selection defaults"
```

## Task 2: Implement Non-Interactive Selection Helpers

**Files:**
- Modify: `cli/main.py`
- Test: `tests/test_cli_non_interactive.py`

- [ ] **Step 1: Add helper imports and constants**

In `cli/main.py`, extend the top import from `typing` and add these constants after `app = typer.Typer(...)`:

```python
from typing import Optional, Sequence
```

```python
DEFAULT_ANALYST_TEAM = [
    AnalystType.MARKET,
    AnalystType.SOCIAL,
    AnalystType.NEWS,
    AnalystType.FUNDAMENTALS,
]

RESEARCH_DEPTH_ALIASES = {
    "shallow": 1,
    "1": 1,
    "medium": 3,
    "3": 3,
    "deep": 5,
    "5": 5,
}
```

- [ ] **Step 2: Add parser functions**

In `cli/main.py`, add these functions above `get_user_selections()`:

```python
def _parse_research_depth(value: str | int | None) -> int:
    raw_value = "shallow" if value is None else str(value).strip().lower()
    if raw_value not in RESEARCH_DEPTH_ALIASES:
        allowed = "shallow, medium, deep, 1, 3, 5"
        raise ValueError(f"Invalid research depth: {value}. Allowed values: {allowed}.")
    return RESEARCH_DEPTH_ALIASES[raw_value]


def _parse_analysts(value: str | None) -> list[AnalystType]:
    if value is None or not value.strip():
        return DEFAULT_ANALYST_TEAM.copy()

    requested = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not requested:
        return DEFAULT_ANALYST_TEAM.copy()

    valid_values = {analyst.value for analyst in DEFAULT_ANALYST_TEAM}
    invalid_values = sorted(set(requested) - valid_values)
    if invalid_values:
        allowed = ", ".join(analyst.value for analyst in DEFAULT_ANALYST_TEAM)
        invalid = ", ".join(invalid_values)
        raise ValueError(f"Invalid analysts: {invalid}. Allowed values: {allowed}.")

    requested_values = set(requested)
    return [analyst for analyst in DEFAULT_ANALYST_TEAM if analyst.value in requested_values]


def _parse_analysis_date(
    value: str | None,
    today: datetime.date | None = None,
) -> str:
    current_date = today or datetime.datetime.now().date()
    if value is None or not value.strip():
        return current_date.strftime("%Y-%m-%d")

    raw_value = value.strip()
    try:
        analysis_date = datetime.datetime.strptime(raw_value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("Invalid analysis date. Use YYYY-MM-DD.") from exc

    if analysis_date > current_date:
        raise ValueError("Analysis date cannot be in the future.")

    return raw_value


def build_non_interactive_selections(
    ticker: str | None,
    analysis_date: str | None,
    analysts: str | None,
    research_depth: str | int | None,
    today: datetime.date | None = None,
) -> dict:
    if ticker is None or not ticker.strip():
        raise ValueError("--ticker is required when --non-interactive is used.")

    return {
        "ticker": normalize_ticker_symbol(ticker),
        "analysis_date": _parse_analysis_date(analysis_date, today=today),
        "analysts": _parse_analysts(analysts),
        "research_depth": _parse_research_depth(research_depth),
        "llm_provider": DEFAULT_CONFIG["llm_provider"],
        "backend_url": DEFAULT_CONFIG.get("backend_url"),
        "shallow_thinker": DEFAULT_CONFIG["quick_think_llm"],
        "deep_thinker": DEFAULT_CONFIG["deep_think_llm"],
        "google_thinking_level": DEFAULT_CONFIG.get("google_thinking_level"),
        "openai_reasoning_effort": DEFAULT_CONFIG.get("openai_reasoning_effort"),
        "anthropic_effort": DEFAULT_CONFIG.get("anthropic_effort"),
        "output_language": DEFAULT_CONFIG.get("output_language", "English"),
    }
```

- [ ] **Step 3: Run the parser tests and verify the helper-related tests pass or narrow the remaining failure**

Run:

```bash
uv run pytest tests/test_cli_non_interactive.py -v
```

Expected: FAIL only on `test_build_non_interactive_selections_uses_safe_defaults` until `DEFAULT_CONFIG["openai_reasoning_effort"]` is changed to `"medium"`.

- [ ] **Step 4: Commit the selection helper implementation**

Run:

```bash
git add cli/main.py
git commit -m "feat: add non-interactive cli selection helpers"
```

## Task 3: Wire Non-Interactive Typer Options

**Files:**
- Modify: `cli/main.py`
- Test: `tests/test_cli_non_interactive.py`

- [ ] **Step 1: Append CLI forwarding tests**

Append these tests to `tests/test_cli_non_interactive.py`:

```python
from pathlib import Path

from typer.testing import CliRunner


runner = CliRunner()


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
    assert "--llm-provider" not in result.output
    assert "--quick-think-llm" not in result.output
    assert "--deep-think-llm" not in result.output
```

- [ ] **Step 2: Run the CLI forwarding tests and verify they fail**

Run:

```bash
uv run pytest tests/test_cli_non_interactive.py -v
```

Expected: FAIL because `analyze()` does not expose the new options and `run_analysis()` does not accept the forwarded keyword arguments yet.

- [ ] **Step 3: Update `analyze()` options and forwarding**

Replace the `analyze()` function signature and body in `cli/main.py` with this implementation:

```python
@app.command()
def analyze(
    checkpoint: bool = typer.Option(
        False,
        "--checkpoint",
        help="Enable checkpoint/resume: save state after each node so a crashed run can resume.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Run without prompts using safe defaults and command-line overrides.",
    ),
    ticker: Optional[str] = typer.Option(
        None,
        "--ticker",
        help="Ticker to analyze. Required with --non-interactive. Example: SPY.",
    ),
    analysis_date: Optional[str] = typer.Option(
        None,
        "--analysis-date",
        help="Analysis date in YYYY-MM-DD format. Defaults to today's date in non-interactive mode.",
    ),
    analysts: Optional[str] = typer.Option(
        None,
        "--analysts",
        help="Comma-separated analysts for non-interactive mode: market,social,news,fundamentals.",
    ),
    research_depth: Optional[str] = typer.Option(
        None,
        "--research-depth",
        help="Research depth for non-interactive mode: shallow, medium, deep, 1, 3, or 5. Defaults to shallow.",
    ),
    save_report: bool = typer.Option(
        True,
        "--save-report/--no-save-report",
        help="Save the final report after analysis. Defaults to saving.",
    ),
    display_report: bool = typer.Option(
        True,
        "--display-report/--no-display-report",
        help="Display the full final report after analysis. Defaults to displaying.",
    ),
    save_path: Optional[Path] = typer.Option(
        None,
        "--save-path",
        help="Directory for saved reports. Defaults to ./reports/{ticker}_{timestamp}.",
    ),
):
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")

    if save_path is not None and not save_report:
        raise typer.BadParameter("--save-path cannot be used with --no-save-report.")

    selections = None
    if non_interactive:
        try:
            selections = build_non_interactive_selections(
                ticker=ticker,
                analysis_date=analysis_date,
                analysts=analysts,
                research_depth=research_depth,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

    run_analysis(
        checkpoint=checkpoint,
        selections=selections,
        non_interactive=non_interactive,
        save_report=save_report,
        display_report=display_report,
        save_path=save_path,
    )
```

- [ ] **Step 4: Run the CLI forwarding tests and verify the remaining failure is isolated to `run_analysis()` signature**

Run:

```bash
uv run pytest tests/test_cli_non_interactive.py -v
```

Expected: FAIL because `run_analysis()` still has the old signature when tests do not monkeypatch it, or because the default config test still expects `"medium"`.

- [ ] **Step 5: Commit the Typer option wiring**

Run:

```bash
git add cli/main.py tests/test_cli_non_interactive.py
git commit -m "feat: wire non-interactive cli options"
```

## Task 4: Refactor `run_analysis()` for Interactive and Non-Interactive Inputs

**Files:**
- Modify: `cli/main.py`
- Test: `tests/test_cli_non_interactive.py`

- [ ] **Step 1: Append save path tests**

Append these tests to `tests/test_cli_non_interactive.py`:

```python
def test_resolve_report_save_path_preserves_existing_default_shape(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    now = dt.datetime(2026, 5, 9, 14, 30, 12)

    save_path = cli_main._resolve_report_save_path("SPY", None, now=now)

    assert save_path == tmp_path / "reports" / "SPY_20260509_143012"


def test_resolve_report_save_path_uses_explicit_path():
    save_path = cli_main._resolve_report_save_path(
        "SPY",
        Path("reports/custom-spy"),
        now=dt.datetime(2026, 5, 9, 14, 30, 12),
    )

    assert save_path == Path("reports/custom-spy")
```

- [ ] **Step 2: Run save path tests and verify they fail**

Run:

```bash
uv run pytest tests/test_cli_non_interactive.py::test_resolve_report_save_path_preserves_existing_default_shape tests/test_cli_non_interactive.py::test_resolve_report_save_path_uses_explicit_path -v
```

Expected: FAIL because `_resolve_report_save_path` does not exist yet.

- [ ] **Step 3: Add `_resolve_report_save_path()`**

Add this function above `run_analysis()` in `cli/main.py`:

```python
def _resolve_report_save_path(
    ticker: str,
    save_path: Path | None,
    now: datetime.datetime | None = None,
) -> Path:
    if save_path is not None:
        return save_path

    timestamp_source = now or datetime.datetime.now()
    timestamp = timestamp_source.strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / "reports" / f"{ticker}_{timestamp}"
```

- [ ] **Step 4: Refactor `run_analysis()` signature and selection source**

Change the `run_analysis()` definition in `cli/main.py` from:

```python
def run_analysis(checkpoint: bool = False):
    # First get all user selections
    selections = get_user_selections()
```

to:

```python
def run_analysis(
    checkpoint: bool = False,
    selections: dict | None = None,
    non_interactive: bool = False,
    save_report: bool = True,
    display_report: bool = True,
    save_path: Path | None = None,
):
    if selections is None:
        selections = get_user_selections()
```

- [ ] **Step 5: Replace the post-analysis prompt block**

In `cli/main.py`, replace the block beginning with:

```python
    # Prompt to save report
    save_choice = typer.prompt("Save report?", default="Y").strip().upper()
```

and ending with:

```python
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)
```

with:

```python
    if non_interactive:
        if save_report:
            resolved_save_path = _resolve_report_save_path(
                selections["ticker"],
                save_path,
            )
            try:
                report_file = save_report_to_disk(
                    final_state,
                    selections["ticker"],
                    resolved_save_path,
                )
                console.print(f"\n[green]✓ Report saved to:[/green] {resolved_save_path.resolve()}")
                console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
            except Exception as e:
                console.print(f"[red]Error saving report: {e}[/red]")

        if display_report:
            display_complete_report(final_state)

        return

    # Prompt to save report
    save_choice = typer.prompt("Save report?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        default_path = _resolve_report_save_path(selections["ticker"], None)
        save_path_str = typer.prompt(
            "Save path (press Enter for default)",
            default=str(default_path)
        ).strip()
        resolved_save_path = Path(save_path_str)
        try:
            report_file = save_report_to_disk(final_state, selections["ticker"], resolved_save_path)
            console.print(f"\n[green]✓ Report saved to:[/green] {resolved_save_path.resolve()}")
            console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")

    # Prompt to display full report
    display_choice = typer.prompt("\nDisplay full report on screen?", default="Y").strip().upper()
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)
```

- [ ] **Step 6: Run all non-interactive CLI tests and verify only the config default test remains failing**

Run:

```bash
uv run pytest tests/test_cli_non_interactive.py -v
```

Expected: FAIL only on assertions expecting `DEFAULT_CONFIG["openai_reasoning_effort"] == "medium"` until Task 5 changes the config.

- [ ] **Step 7: Commit the `run_analysis()` refactor**

Run:

```bash
git add cli/main.py tests/test_cli_non_interactive.py
git commit -m "feat: support non-interactive analysis execution"
```

## Task 5: Change OpenAI Reasoning Effort Default

**Files:**
- Modify: `tradingagents/default_config.py`
- Test: `tests/test_cli_non_interactive.py`

- [ ] **Step 1: Append the config default test**

Append this test to `tests/test_cli_non_interactive.py`:

```python
def test_openai_reasoning_effort_default_is_medium():
    from tradingagents.default_config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["openai_reasoning_effort"] == "medium"
```

- [ ] **Step 2: Run the config default test and verify it fails**

Run:

```bash
uv run pytest tests/test_cli_non_interactive.py::test_openai_reasoning_effort_default_is_medium -v
```

Expected: FAIL because `DEFAULT_CONFIG["openai_reasoning_effort"]` is currently `None`.

- [ ] **Step 3: Change the default config**

In `tradingagents/default_config.py`, change:

```python
    "openai_reasoning_effort": None,    # "medium", "high", "low"
```

to:

```python
    "openai_reasoning_effort": "medium",    # "medium", "high", "low"
```

- [ ] **Step 4: Run all non-interactive CLI tests**

Run:

```bash
uv run pytest tests/test_cli_non_interactive.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit the default config change**

Run:

```bash
git add tradingagents/default_config.py tests/test_cli_non_interactive.py
git commit -m "fix: default openai reasoning effort to medium"
```

## Task 6: Run Existing CLI-Focused Regression Checks

**Files:**
- Test: `tests/test_cli_non_interactive.py`
- Test: existing tests under `tests/`

- [ ] **Step 1: Run the focused non-interactive test file**

Run:

```bash
uv run pytest tests/test_cli_non_interactive.py -v
```

Expected: PASS.

- [ ] **Step 2: Run the existing unit and smoke tests**

Run:

```bash
uv run pytest -m "unit or smoke" -v
```

Expected: PASS, or SKIP for tests that are explicitly marked unavailable by local environment conditions.

- [ ] **Step 3: Run CLI help smoke check**

Run:

```bash
uv run tradingagents --help
```

Expected: command exits successfully and shows `--non-interactive`, `--ticker`, `--analysis-date`, `--analysts`, `--research-depth`, `--save-report`, `--display-report`, and `--save-path`.

- [ ] **Step 4: Run non-interactive validation smoke check without triggering analysis**

Run:

```bash
uv run tradingagents --non-interactive
```

Expected: command exits before analysis and prints an error containing `--ticker is required when --non-interactive is used`.

- [ ] **Step 5: Commit any test-only corrections required by actual Typer output**

If Typer formats an error with extra text but the semantic error is correct, update only the overly-specific assertion string in `tests/test_cli_non_interactive.py`.

Run:

```bash
git add tests/test_cli_non_interactive.py
git commit -m "test: align cli assertions with typer output"
```

## Task 7: Manual Full-Run Verification

**Files:**
- No source file changes expected.

- [ ] **Step 1: Verify required environment variables are present**

Run:

```bash
env | rg '^(OPENAI_API_KEY|TRADINGAGENTS_RESULTS_DIR|TRADINGAGENTS_CACHE_DIR)='
```

Expected: `OPENAI_API_KEY` is present if running the default OpenAI-backed analysis. `TRADINGAGENTS_RESULTS_DIR` and `TRADINGAGENTS_CACHE_DIR` may be absent because the app has defaults under `~/.tradingagents`.

- [ ] **Step 2: Run a non-interactive analysis with output suppressed**

Run:

```bash
uv run tradingagents --non-interactive --ticker SPY --research-depth shallow --analysts market --no-display-report --save-path reports/manual-spy-non-interactive
```

Expected: command does not show welcome ASCII or announcements, starts the analysis progress UI, saves the final report under `reports/manual-spy-non-interactive`, and does not print the full final report after completion.

- [ ] **Step 3: Verify default save and display behavior with a short run**

Run:

```bash
uv run tradingagents --non-interactive --ticker SPY --research-depth shallow --analysts market
```

Expected: command does not prompt for input, saves the final report under `./reports/SPY_<timestamp>`, and displays the full final report after completion.

- [ ] **Step 4: Commit no source changes after manual verification**

Run:

```bash
git status --short
```

Expected: no uncommitted source or test changes from manual verification.

## Self-Review

**Spec coverage:** The plan covers explicit `--non-interactive`, required `--ticker`, date default and future-date rejection, full analyst default, research-depth names and aliases, no LLM CLI options, `DEFAULT_CONFIG` OpenAI reasoning effort, save/display defaults, `--save-path` conflict detection, and omission of interactive welcome/announcements through bypassing `get_user_selections()`.

**Forbidden marker scan:** The plan avoids unspecified future work markers and gives concrete code, commands, and expected outcomes for each implementation step.

**Type consistency:** The helper names used in tests match the helper names introduced in implementation steps: `_parse_research_depth`, `_parse_analysts`, `_parse_analysis_date`, `build_non_interactive_selections`, and `_resolve_report_save_path`.

**Residual risk:** Full analysis requires external API credentials and market data access, so automated tests focus on parser and CLI behavior while manual verification covers actual execution.
