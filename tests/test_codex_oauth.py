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


from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel


class Decision(BaseModel):
    rating: str
    rationale: str


def _structured_payload(client, schema: type[BaseModel]):
    structured = client.with_structured_output(schema)
    binding = structured.first
    return binding.bound._get_request_payload(
        [HumanMessage(content="Make a decision.")],
        **binding.kwargs,
    )


@pytest.mark.unit
def test_codex_chat_payload_uses_instructions_stream_and_sanitizes_params():
    from tradingagents.llm_clients.openai_client import CodexOAuthChatOpenAI

    client = CodexOAuthChatOpenAI(
        model="gpt-5.5",
        api_key="placeholder",
        base_url="https://chatgpt.com/backend-api/codex",
        use_responses_api=True,
        streaming=True,
        service_tier="priority",
    )

    payload = client._get_request_payload(
        [SystemMessage(content="System prompt"), HumanMessage(content="Analyze SPY")],
        temperature=0.2,
        max_tokens=100,
        service_tier="priority",
        metadata={"run": "test"},
        prompt_cache_retention="24h",
    )

    assert payload["instructions"] == "System prompt"
    assert payload["stream"] is True
    assert payload["store"] is False
    assert payload["service_tier"] == "priority"
    assert payload["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Analyze SPY"}],
        }
    ]
    assert "max_output_tokens" not in payload
    assert "temperature" not in payload
    assert "metadata" not in payload
    assert "prompt_cache_retention" not in payload


@pytest.mark.unit
def test_codex_chat_payload_preserves_explicit_and_message_instructions():
    from tradingagents.llm_clients.openai_client import _normalize_codex_responses_payload

    payload = _normalize_codex_responses_payload(
        {
            "instructions": "Explicit instructions",
            "input": [
                {
                    "type": "message",
                    "role": "system",
                    "content": "System prompt",
                },
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "text", "text": "Developer prompt"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": "Analyze SPY",
                },
            ],
        }
    )

    assert (
        payload["instructions"]
        == "Explicit instructions\n\nSystem prompt\n\nDeveloper prompt"
    )
    assert payload["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Analyze SPY"}],
        }
    ]


@pytest.mark.unit
def test_codex_chat_payload_keeps_tools_for_bind_tools():
    from tradingagents.llm_clients.openai_client import CodexOAuthChatOpenAI

    def get_price(ticker: str) -> str:
        """Return the current price for a ticker."""
        return "100"

    client = CodexOAuthChatOpenAI(
        model="gpt-5.5",
        api_key="placeholder",
        base_url="https://chatgpt.com/backend-api/codex",
        use_responses_api=True,
        streaming=True,
    )
    bound = client.bind_tools([get_price])
    payload = bound.bound._get_request_payload(
        [HumanMessage(content="Use the tool for SPY.")],
        **bound.kwargs,
    )

    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["name"] == "get_price"
    assert payload.get("tool_choice") in (None, "auto")


@pytest.mark.unit
def test_codex_structured_output_uses_function_calling_binding():
    from tradingagents.llm_clients.openai_client import CodexOAuthChatOpenAI

    client = CodexOAuthChatOpenAI(
        model="gpt-5.5",
        api_key="placeholder",
        base_url="https://chatgpt.com/backend-api/codex",
        use_responses_api=True,
        streaming=True,
    )

    payload = _structured_payload(client, Decision)

    assert "ls_structured_output_format" not in payload
    assert "structured_output_format" not in payload
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["name"] == "Decision"
    assert payload["tools"][0]["parameters"]["properties"]["rating"]["type"] == "string"
    assert payload["tools"][0]["parameters"]["properties"]["rationale"]["type"] == "string"


@pytest.mark.unit
def test_openai_client_prefers_codex_oauth_over_openai_api_key(monkeypatch, tmp_path):
    from tradingagents.llm_clients.openai_client import OpenAIClient

    access_token = _jwt_with_exp(int(time.time()) + 3600)
    auth_path = _write_codex_auth(tmp_path / "codex", access_token=access_token)
    monkeypatch.setenv("CODEX_HOME", str(auth_path.parent))
    monkeypatch.setenv("TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE", "codex_oauth")
    monkeypatch.setenv("OPENAI_API_KEY", "api-key-that-must-not-win")

    llm = OpenAIClient(
        "gpt-5.5",
        provider="openai",
        service_tier="priority",
        api_key="caller-api-key-that-must-not-win",
    ).get_llm()

    assert llm.__class__.__name__ == "CodexOAuthChatOpenAI"
    assert llm.openai_api_key.get_secret_value() == access_token
    assert str(llm.openai_api_base).rstrip("/") == "https://chatgpt.com/backend-api/codex"
    assert llm.default_headers["ChatGPT-Account-Id"] == "acct_test"
    assert llm.streaming is True
    assert llm.service_tier == "priority"


@pytest.mark.unit
def test_openai_client_codex_oauth_missing_auth_hard_fails(monkeypatch, tmp_path):
    from tradingagents.llm_clients.codex_oauth import CodexOAuthCredentialError
    from tradingagents.llm_clients.openai_client import OpenAIClient

    codex_home = tmp_path / "missing-codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE", "codex_oauth")
    monkeypatch.setenv("OPENAI_API_KEY", "api-key-that-must-not-win")

    with pytest.raises(CodexOAuthCredentialError) as exc_info:
        OpenAIClient("gpt-5.5", provider="openai").get_llm()

    message = str(exc_info.value)
    assert str(codex_home / "auth.json") in message
    assert "codex login" in message
    assert "OPENAI_API_KEY fallback is disabled" in message


@pytest.mark.unit
@pytest.mark.parametrize("credential_source", [None, "api_key"])
def test_openai_client_regular_openai_path_is_unchanged(
    monkeypatch,
    credential_source,
):
    from tradingagents.llm_clients.openai_client import OpenAIClient

    if credential_source is None:
        monkeypatch.delenv("TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE", raising=False)
    else:
        monkeypatch.setenv("TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE", credential_source)
    monkeypatch.setenv("OPENAI_API_KEY", "regular-api-key")

    llm = OpenAIClient("gpt-5.5", provider="openai").get_llm()

    assert llm.__class__.__name__ == "NormalizedChatOpenAI"
    assert llm.openai_api_key.get_secret_value() == "regular-api-key"
    assert llm.use_responses_api is True


@pytest.mark.unit
def test_openai_client_does_not_forward_service_tier_to_deepseek(monkeypatch):
    from tradingagents.llm_clients.openai_client import OpenAIClient

    monkeypatch.delenv("TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-api-key")

    llm = OpenAIClient(
        "deepseek-chat",
        provider="deepseek",
        service_tier="priority",
    ).get_llm()
    payload = llm._get_request_payload([HumanMessage(content="hi")])

    assert llm.__class__.__name__ == "DeepSeekChatOpenAI"
    assert "service_tier" not in payload
