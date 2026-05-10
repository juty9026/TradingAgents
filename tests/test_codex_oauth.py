import base64
import json
import time
from pathlib import Path

import pytest


def _jwt_with_exp(exp: int, extra_claims: dict | None = None) -> str:
    header = {"alg": "none", "typ": "JWT"}
    payload = {"exp": exp}
    if extra_claims:
        payload.update(extra_claims)

    def _part(data: dict) -> str:
        raw = json.dumps(data, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{_part(header)}.{_part(payload)}."


def _write_codex_auth(
    codex_home: Path,
    *,
    access_token: str,
    account_id: str = "acct_test",
    auth_mode: str = "chatgpt",
) -> Path:
    codex_home.mkdir(parents=True, exist_ok=True)
    auth_path = codex_home / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": auth_mode,
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "refresh-token",
                    "account_id": account_id,
                },
            }
        )
    )
    return auth_path


@pytest.mark.unit
def test_codex_oauth_loader_reads_codex_home_auth_json(monkeypatch, tmp_path):
    from tradingagents.llm_clients.codex_oauth import load_codex_oauth_credentials

    access_token = _jwt_with_exp(int(time.time()) + 3600)
    auth_path = _write_codex_auth(tmp_path / "codex", access_token=access_token)
    monkeypatch.setenv("CODEX_HOME", str(auth_path.parent))

    credentials = load_codex_oauth_credentials()

    assert credentials.access_token == access_token
    assert credentials.account_id == "acct_test"
    assert credentials.auth_path == auth_path


@pytest.mark.unit
def test_codex_oauth_loader_uses_account_id_from_jwt_when_auth_file_omits_it(
    monkeypatch,
    tmp_path,
):
    from tradingagents.llm_clients.codex_oauth import load_codex_oauth_credentials

    access_token = _jwt_with_exp(
        int(time.time()) + 3600,
        {"https://api.openai.com/auth": {"chatgpt_account_id": "acct_from_jwt"}},
    )
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "refresh-token",
                },
            }
        )
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    credentials = load_codex_oauth_credentials()

    assert credentials.account_id == "acct_from_jwt"


@pytest.mark.unit
def test_codex_oauth_loader_rejects_expired_access_token(monkeypatch, tmp_path):
    from tradingagents.llm_clients.codex_oauth import (
        CodexOAuthCredentialError,
        load_codex_oauth_credentials,
    )

    auth_path = _write_codex_auth(
        tmp_path / "codex",
        access_token=_jwt_with_exp(int(time.time()) - 10),
    )
    monkeypatch.setenv("CODEX_HOME", str(auth_path.parent))
    monkeypatch.setenv("OPENAI_API_KEY", "api-key-that-must-not-win")

    with pytest.raises(CodexOAuthCredentialError) as exc_info:
        load_codex_oauth_credentials()

    message = str(exc_info.value)
    assert str(auth_path) in message
    assert "codex login" in message
    assert "OPENAI_API_KEY fallback is disabled" in message


@pytest.mark.unit
def test_codex_oauth_loader_rejects_non_object_auth_payload(monkeypatch, tmp_path):
    from tradingagents.llm_clients.codex_oauth import (
        CodexOAuthCredentialError,
        load_codex_oauth_credentials,
    )

    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    auth_path = codex_home / "auth.json"
    auth_path.write_text("[]")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    with pytest.raises(CodexOAuthCredentialError) as exc_info:
        load_codex_oauth_credentials()

    message = str(exc_info.value)
    assert str(auth_path) in message
    assert "codex login" in message
    assert "OPENAI_API_KEY fallback is disabled" in message


@pytest.mark.unit
def test_codex_oauth_requested_only_for_explicit_source(monkeypatch):
    from tradingagents.llm_clients.codex_oauth import codex_oauth_requested

    monkeypatch.delenv("TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE", raising=False)
    assert codex_oauth_requested() is False

    monkeypatch.setenv("TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE", "codex_oauth")
    assert codex_oauth_requested() is True

    monkeypatch.setenv("TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE", "api_key")
    assert codex_oauth_requested() is False


@pytest.mark.unit
def test_codex_oauth_profile_defaults_and_env_overrides(monkeypatch):
    from tradingagents.llm_clients.codex_oauth import (
        codex_oauth_reasoning_effort,
        codex_oauth_role_model,
        codex_oauth_service_tier,
        normalize_codex_service_tier,
    )

    config = {
        "codex_oauth_deep_think_llm": "gpt-5.5",
        "codex_oauth_quick_think_llm": "gpt-5.5",
        "codex_oauth_deep_reasoning_effort": "high",
        "codex_oauth_quick_reasoning_effort": "low",
        "codex_oauth_service_tier": "priority",
    }

    assert codex_oauth_role_model(config, "quick") == "gpt-5.5"
    assert codex_oauth_role_model(config, "deep") == "gpt-5.5"
    assert codex_oauth_reasoning_effort(config, "quick") == "low"
    assert codex_oauth_reasoning_effort(config, "deep") == "high"
    assert normalize_codex_service_tier(config["codex_oauth_service_tier"]) == "priority"
    assert normalize_codex_service_tier("fast") == "priority"
    for unsupported_tier in ("flex", "normal", "off", "none"):
        with pytest.raises(ValueError):
            normalize_codex_service_tier(unsupported_tier)

    monkeypatch.setenv("TRADINGAGENTS_CODEX_OAUTH_QUICK_MODEL", "gpt-5.5-mini")
    monkeypatch.setenv("TRADINGAGENTS_CODEX_OAUTH_DEEP_MODEL", "gpt-5.6")
    monkeypatch.setenv("TRADINGAGENTS_CODEX_OAUTH_QUICK_REASONING_EFFORT", "medium")
    monkeypatch.setenv("TRADINGAGENTS_CODEX_OAUTH_DEEP_REASONING_EFFORT", "minimal")
    monkeypatch.setenv("TRADINGAGENTS_CODEX_OAUTH_SERVICE_TIER", "fast")

    assert codex_oauth_role_model(config, "quick") == "gpt-5.5-mini"
    assert codex_oauth_role_model(config, "deep") == "gpt-5.6"
    assert codex_oauth_reasoning_effort(config, "quick") == "medium"
    assert codex_oauth_reasoning_effort(config, "deep") == "minimal"
    assert codex_oauth_service_tier(config) == "priority"
