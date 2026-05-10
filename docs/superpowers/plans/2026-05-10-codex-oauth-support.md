# Codex OAuth Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Codex OAuth support for TradingAgents OpenAI runs with file-based Codex credentials, Codex backend transport, role-specific GPT-5.5 profile defaults, analyst tool calling, and function-calling structured output.

**Architecture:** Add a small Codex OAuth support module for credential/profile resolution, then extend the existing OpenAI LangChain client with a Codex-specific `ChatOpenAI` subclass that normalizes Responses payloads for `chatgpt.com/backend-api/codex`. Wire the graph to choose Codex OAuth quick/deep model and reasoning profile only when `TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE=codex_oauth` is set; normal OpenAI API-key behavior stays unchanged.

**Tech Stack:** Python 3.11, LangChain `ChatOpenAI`, OpenAI Responses API via `langchain-openai`, pytest, existing TradingAgents LLM client factory.

---

## File Structure

- Create `tradingagents/llm_clients/codex_oauth.py`: owns Codex OAuth constants, auth file discovery, JWT expiry/account parsing, service-tier normalization, and Codex OAuth role profile resolution.
- Modify `tradingagents/llm_clients/openai_client.py`: adds `CodexOAuthChatOpenAI`, Codex Responses payload normalization, `service_tier` passthrough, and native OpenAI credential-source selection.
- Modify `tradingagents/default_config.py`: adds Codex OAuth profile defaults that can be overridden through environment variables.
- Modify `tradingagents/graph/trading_graph.py`: resolves quick/deep model and reasoning effort separately when Codex OAuth is requested.
- Create `tests/test_codex_oauth.py`: focused unit tests for credential discovery, profile defaults, Codex payload shape, client selection, and graph role profile wiring.

## References

- Domain decisions: `CONTEXT.md`
- ADR: `docs/adr/0001-codex-oauth-support.md`
- Existing OpenAI client: `tradingagents/llm_clients/openai_client.py`
- Existing graph wiring: `tradingagents/graph/trading_graph.py`
- Existing structured-output fallback: `tradingagents/agents/utils/structured.py`

---

### Task 1: Codex OAuth Credential and Profile Helpers

**Files:**
- Create: `tradingagents/llm_clients/codex_oauth.py`
- Modify: `tradingagents/default_config.py`
- Test: `tests/test_codex_oauth.py`

- [ ] **Step 1: Write failing tests for auth file loading, hard-fail errors, and profile defaults**

Create `tests/test_codex_oauth.py` with this initial content:

```python
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
        codex_oauth_service_tier,
        codex_oauth_reasoning_effort,
        codex_oauth_role_model,
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

    monkeypatch.setenv("TRADINGAGENTS_CODEX_OAUTH_QUICK_REASONING_EFFORT", "medium")
    monkeypatch.setenv("TRADINGAGENTS_CODEX_OAUTH_SERVICE_TIER", "fast")

    assert codex_oauth_reasoning_effort(config, "quick") == "medium"
    assert codex_oauth_service_tier(config) == "priority"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_codex_oauth.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tradingagents.llm_clients.codex_oauth'`.

- [ ] **Step 3: Implement the credential/profile helper module**

Create `tradingagents/llm_clients/codex_oauth.py`:

```python
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CODEX_RESPONSES_BASE_URL = "https://chatgpt.com/backend-api/codex"
TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE_ENV = "TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE"

_QUICK_MODEL_ENV = "TRADINGAGENTS_CODEX_OAUTH_QUICK_MODEL"
_DEEP_MODEL_ENV = "TRADINGAGENTS_CODEX_OAUTH_DEEP_MODEL"
_QUICK_REASONING_ENV = "TRADINGAGENTS_CODEX_OAUTH_QUICK_REASONING_EFFORT"
_DEEP_REASONING_ENV = "TRADINGAGENTS_CODEX_OAUTH_DEEP_REASONING_EFFORT"
_SERVICE_TIER_ENV = "TRADINGAGENTS_CODEX_OAUTH_SERVICE_TIER"


class CodexOAuthCredentialError(RuntimeError):
    """Raised when a usable Codex OAuth credential is unavailable."""


@dataclass(frozen=True)
class CodexOAuthCredentials:
    access_token: str
    account_id: str | None
    auth_path: Path
    expires_at: int


def codex_oauth_requested(value: Any | None = None) -> bool:
    source = (
        str(value).strip()
        if value is not None and str(value).strip()
        else os.getenv(TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE_ENV, "")
    )
    return source.strip().lower().replace("-", "_") == "codex_oauth"


def load_codex_oauth_credentials(
    *,
    codex_home: str | os.PathLike[str] | None = None,
    refresh_skew_seconds: int = 60,
) -> CodexOAuthCredentials:
    auth_path = _codex_auth_path(codex_home)
    if not auth_path.is_file():
        raise _credential_error(auth_path, "No Codex OAuth auth file was found.")

    try:
        payload = json.loads(auth_path.read_text())
    except Exception as exc:
        raise _credential_error(auth_path, "The Codex OAuth auth file could not be read.") from exc

    auth_mode = payload.get("auth_mode")
    if isinstance(auth_mode, str) and auth_mode and auth_mode != "chatgpt":
        raise _credential_error(auth_path, "The Codex auth file is not a ChatGPT login.")

    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        raise _credential_error(auth_path, "The Codex auth file is missing OAuth tokens.")

    access_token = tokens.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise _credential_error(auth_path, "The Codex auth file is missing access_token.")
    access_token = access_token.strip()

    claims = _decode_jwt_claims(access_token, auth_path)
    expires_at = _extract_expiry(claims, auth_path)
    if expires_at <= int(time.time()) + refresh_skew_seconds:
        raise _credential_error(auth_path, "The Codex OAuth access token is expired or too close to expiry.")

    return CodexOAuthCredentials(
        access_token=access_token,
        account_id=_extract_account_id(tokens, claims),
        auth_path=auth_path,
        expires_at=expires_at,
    )


def codex_oauth_default_headers(credentials: CodexOAuthCredentials) -> dict[str, str]:
    headers = {
        "Accept": "text/event-stream",
        "OpenAI-Beta": "responses=2026-02-06",
        "User-Agent": "codex_cli_rs/0.0.0 (TradingAgents)",
        "originator": "codex_cli_rs",
        "x-codex-installation-id": "tradingagents",
    }
    if credentials.account_id:
        headers["ChatGPT-Account-Id"] = credentials.account_id
    return headers


def codex_oauth_role_model(config: dict[str, Any], role: str) -> str:
    if role == "quick":
        return os.getenv(_QUICK_MODEL_ENV, "").strip() or str(
            config.get("codex_oauth_quick_think_llm") or "gpt-5.5"
        )
    if role == "deep":
        return os.getenv(_DEEP_MODEL_ENV, "").strip() or str(
            config.get("codex_oauth_deep_think_llm") or "gpt-5.5"
        )
    raise ValueError(f"Unknown Codex OAuth LLM role: {role}")


def codex_oauth_reasoning_effort(config: dict[str, Any], role: str) -> str:
    if role == "quick":
        return os.getenv(_QUICK_REASONING_ENV, "").strip() or str(
            config.get("codex_oauth_quick_reasoning_effort") or "low"
        )
    if role == "deep":
        return os.getenv(_DEEP_REASONING_ENV, "").strip() or str(
            config.get("codex_oauth_deep_reasoning_effort") or "high"
        )
    raise ValueError(f"Unknown Codex OAuth LLM role: {role}")


def codex_oauth_service_tier(config: dict[str, Any]) -> str | None:
    raw = os.getenv(_SERVICE_TIER_ENV, "").strip() or str(
        config.get("codex_oauth_service_tier") or ""
    ).strip()
    return normalize_codex_service_tier(raw)


def normalize_codex_service_tier(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized or normalized in {"normal", "off", "none"}:
        return None
    if normalized in {"fast", "priority"}:
        return "priority"
    if normalized == "flex":
        return "flex"
    raise ValueError(
        "Unsupported Codex OAuth service tier "
        f"{value!r}; expected one of: priority, fast, flex, normal, off."
    )


def _codex_auth_path(codex_home: str | os.PathLike[str] | None) -> Path:
    home = (
        Path(codex_home)
        if codex_home is not None
        else Path(os.getenv("CODEX_HOME", "") or Path.home() / ".codex")
    )
    return home.expanduser() / "auth.json"


def _credential_error(auth_path: Path, reason: str) -> CodexOAuthCredentialError:
    return CodexOAuthCredentialError(
        f"{reason} Checked {auth_path}. Run `codex login` with ChatGPT auth, "
        "or unset TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE to use OPENAI_API_KEY. "
        "OPENAI_API_KEY fallback is disabled because Codex OAuth was explicitly requested."
    )


def _decode_jwt_claims(token: str, auth_path: Path) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        raise _credential_error(auth_path, "The Codex OAuth access token is not a JWT.")
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload.encode()))
    except Exception as exc:
        raise _credential_error(auth_path, "The Codex OAuth access token could not be decoded.") from exc
    if not isinstance(claims, dict):
        raise _credential_error(auth_path, "The Codex OAuth access token has invalid claims.")
    return claims


def _extract_expiry(claims: dict[str, Any], auth_path: Path) -> int:
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)):
        raise _credential_error(auth_path, "The Codex OAuth access token is missing expiry.")
    return int(exp)


def _extract_account_id(tokens: dict[str, Any], claims: dict[str, Any]) -> str | None:
    token_account_id = tokens.get("account_id")
    if isinstance(token_account_id, str) and token_account_id.strip():
        return token_account_id.strip()

    nested_auth = claims.get("https://api.openai.com/auth")
    if isinstance(nested_auth, dict):
        nested_account_id = nested_auth.get("chatgpt_account_id")
        if isinstance(nested_account_id, str) and nested_account_id.strip():
            return nested_account_id.strip()

    dotted_account_id = claims.get("https://api.openai.com/auth.chatgpt_account_id")
    if isinstance(dotted_account_id, str) and dotted_account_id.strip():
        return dotted_account_id.strip()

    return None
```

- [ ] **Step 4: Add Codex OAuth defaults to `DEFAULT_CONFIG`**

In `tradingagents/default_config.py`, add these keys immediately after `quick_think_llm`:

```python
    "codex_oauth_deep_think_llm": os.getenv("TRADINGAGENTS_CODEX_OAUTH_DEEP_MODEL", "gpt-5.5"),
    "codex_oauth_quick_think_llm": os.getenv("TRADINGAGENTS_CODEX_OAUTH_QUICK_MODEL", "gpt-5.5"),
    "codex_oauth_deep_reasoning_effort": os.getenv("TRADINGAGENTS_CODEX_OAUTH_DEEP_REASONING_EFFORT", "high"),
    "codex_oauth_quick_reasoning_effort": os.getenv("TRADINGAGENTS_CODEX_OAUTH_QUICK_REASONING_EFFORT", "low"),
    "codex_oauth_service_tier": os.getenv("TRADINGAGENTS_CODEX_OAUTH_SERVICE_TIER", "priority"),
```

- [ ] **Step 5: Run tests to verify Task 1 passes**

Run:

```bash
uv run pytest tests/test_codex_oauth.py -q
```

Expected: PASS for the helper tests added in Task 1.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add tradingagents/llm_clients/codex_oauth.py tradingagents/default_config.py tests/test_codex_oauth.py
git commit -m "feat: add Codex OAuth credential helpers"
```

---

### Task 2: Codex Responses Payload Adapter

**Files:**
- Modify: `tradingagents/llm_clients/openai_client.py`
- Test: `tests/test_codex_oauth.py`

- [ ] **Step 1: Add failing tests for Codex payload normalization**

Append these tests to `tests/test_codex_oauth.py`:

```python
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel


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

    class Decision(BaseModel):
        rating: str
        rationale: str

    client = CodexOAuthChatOpenAI(
        model="gpt-5.5",
        api_key="placeholder",
        base_url="https://chatgpt.com/backend-api/codex",
        use_responses_api=True,
        streaming=True,
    )

    structured = client.with_structured_output(Decision)

    assert structured is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_codex_oauth.py::test_codex_chat_payload_uses_instructions_stream_and_sanitizes_params tests/test_codex_oauth.py::test_codex_chat_payload_keeps_tools_for_bind_tools tests/test_codex_oauth.py::test_codex_structured_output_uses_function_calling_binding -q
```

Expected: FAIL because `CodexOAuthChatOpenAI` does not exist.

- [ ] **Step 3: Add Codex payload helper functions and subclass**

In `tradingagents/llm_clients/openai_client.py`, add `Mapping` to imports:

```python
from typing import Any, Mapping, Optional
```

Add these helpers and class below `NormalizedChatOpenAI`:

```python
_CODEX_UNSUPPORTED_PARAMS = (
    "max_output_tokens",
    "metadata",
    "prompt_cache_retention",
    "temperature",
)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content) if content is not None else ""


def _normalize_codex_content(content: Any, *, assistant: bool = False) -> Any:
    if isinstance(content, str):
        return [{"type": "output_text" if assistant else "input_text", "text": content}]
    if not isinstance(content, list):
        text = str(content) if content is not None else ""
        return [{"type": "output_text" if assistant else "input_text", "text": text}]

    normalized = []
    for block in content:
        if isinstance(block, str):
            normalized.append({"type": "output_text" if assistant else "input_text", "text": block})
            continue
        if not isinstance(block, Mapping):
            continue
        block_type = block.get("type")
        if block_type == "text":
            normalized.append({"type": "output_text" if assistant else "input_text", "text": block.get("text", "")})
        elif block_type == "image_url":
            image_url = block.get("image_url")
            url = image_url.get("url") if isinstance(image_url, Mapping) else image_url
            entry = {"type": "input_image", "image_url": str(url or "")}
            detail = image_url.get("detail") if isinstance(image_url, Mapping) else None
            if detail:
                entry["detail"] = detail
            normalized.append(entry)
        elif block_type in {"input_text", "input_image", "input_file", "output_text", "refusal"}:
            normalized.append(dict(block))
    return normalized


def _normalize_codex_responses_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    instructions = []
    normalized_input = []

    for item in payload.get("input") or []:
        if not isinstance(item, Mapping):
            normalized_input.append(item)
            continue
        item_dict = dict(item)
        role = item_dict.get("role")
        if role in {"system", "developer"}:
            text = _content_to_text(item_dict.get("content"))
            if text:
                instructions.append(text)
            continue
        if item_dict.get("type") == "message" or role in {"user", "assistant"}:
            item_dict["type"] = "message"
            item_dict["content"] = _normalize_codex_content(
                item_dict.get("content"),
                assistant=(role == "assistant"),
            )
        normalized_input.append(item_dict)

    existing_instructions = payload.get("instructions")
    if instructions:
        payload["instructions"] = "\n\n".join(instructions)
    elif not existing_instructions:
        payload["instructions"] = "You are a helpful assistant."

    payload["input"] = normalized_input or [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": " "}],
        }
    ]
    payload["stream"] = True
    payload["store"] = False

    for key in _CODEX_UNSUPPORTED_PARAMS:
        payload.pop(key, None)

    if "tool_choice" not in payload and payload.get("tools"):
        payload["tool_choice"] = "auto"

    return payload


class CodexOAuthChatOpenAI(NormalizedChatOpenAI):
    """ChatOpenAI variant that targets the ChatGPT Codex Responses backend."""

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        return _normalize_codex_responses_payload(payload)
```

- [ ] **Step 4: Run tests to verify Task 2 passes**

Run:

```bash
uv run pytest tests/test_codex_oauth.py -q
```

Expected: PASS for Task 1 and Task 2 tests.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add tradingagents/llm_clients/openai_client.py tests/test_codex_oauth.py
git commit -m "feat: add Codex OAuth Responses adapter"
```

---

### Task 3: Wire OpenAIClient to Prefer Codex OAuth

**Files:**
- Modify: `tradingagents/llm_clients/openai_client.py`
- Test: `tests/test_codex_oauth.py`

- [ ] **Step 1: Add failing tests for OpenAIClient credential-source selection**

Append these tests to `tests/test_codex_oauth.py`:

```python
@pytest.mark.unit
def test_openai_client_prefers_codex_oauth_over_openai_api_key(monkeypatch, tmp_path):
    from tradingagents.llm_clients.openai_client import OpenAIClient

    access_token = _jwt_with_exp(int(time.time()) + 3600)
    auth_path = _write_codex_auth(tmp_path / "codex", access_token=access_token)
    monkeypatch.setenv("CODEX_HOME", str(auth_path.parent))
    monkeypatch.setenv("TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE", "codex_oauth")
    monkeypatch.setenv("OPENAI_API_KEY", "api-key-that-must-not-win")

    llm = OpenAIClient("gpt-5.5", provider="openai", service_tier="priority").get_llm()

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
def test_openai_client_regular_openai_path_is_unchanged(monkeypatch):
    from tradingagents.llm_clients.openai_client import OpenAIClient

    monkeypatch.setenv("TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE", "api_key")
    monkeypatch.setenv("OPENAI_API_KEY", "regular-api-key")

    llm = OpenAIClient("gpt-5.5", provider="openai").get_llm()

    assert llm.__class__.__name__ == "NormalizedChatOpenAI"
    assert llm.openai_api_key.get_secret_value() == "regular-api-key"
    assert llm.use_responses_api is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_codex_oauth.py::test_openai_client_prefers_codex_oauth_over_openai_api_key tests/test_codex_oauth.py::test_openai_client_codex_oauth_missing_auth_hard_fails tests/test_codex_oauth.py::test_openai_client_regular_openai_path_is_unchanged -q
```

Expected: FAIL because `OpenAIClient` still returns `NormalizedChatOpenAI` for native OpenAI even when Codex OAuth is requested.

- [ ] **Step 3: Add Codex imports and passthrough kwargs**

In `tradingagents/llm_clients/openai_client.py`, add this import block under existing local imports:

```python
from .codex_oauth import (
    CODEX_RESPONSES_BASE_URL,
    codex_oauth_default_headers,
    codex_oauth_requested,
    load_codex_oauth_credentials,
)
```

Update `_PASSTHROUGH_KWARGS`:

```python
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort", "service_tier",
    "api_key", "callbacks", "http_client", "http_async_client",
    "default_headers",
)
```

- [ ] **Step 4: Wire Codex OAuth selection in `OpenAIClient.get_llm`**

Inside `OpenAIClient.get_llm`, after `llm_kwargs = {"model": self.model}`, insert:

```python
        use_codex_oauth = self.provider == "openai" and codex_oauth_requested()
```

Replace the native OpenAI Responses block:

```python
        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True
```

with:

```python
        if use_codex_oauth:
            credentials = load_codex_oauth_credentials()
            llm_kwargs["base_url"] = CODEX_RESPONSES_BASE_URL
            llm_kwargs["api_key"] = credentials.access_token
            llm_kwargs["default_headers"] = codex_oauth_default_headers(credentials)
            llm_kwargs["use_responses_api"] = True
            llm_kwargs["streaming"] = True
        elif self.provider == "openai":
            llm_kwargs["use_responses_api"] = True
```

Replace the class selection:

```python
        chat_cls = DeepSeekChatOpenAI if self.provider == "deepseek" else NormalizedChatOpenAI
```

with:

```python
        if use_codex_oauth:
            chat_cls = CodexOAuthChatOpenAI
        elif self.provider == "deepseek":
            chat_cls = DeepSeekChatOpenAI
        else:
            chat_cls = NormalizedChatOpenAI
```

- [ ] **Step 5: Run tests to verify Task 3 passes**

Run:

```bash
uv run pytest tests/test_codex_oauth.py -q
```

Expected: PASS for Task 1 through Task 3 tests.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add tradingagents/llm_clients/openai_client.py tests/test_codex_oauth.py
git commit -m "feat: route OpenAI through Codex OAuth when requested"
```

---

### Task 4: Wire TradingGraph Quick/Deep Codex OAuth Profile

**Files:**
- Modify: `tradingagents/graph/trading_graph.py`
- Test: `tests/test_codex_oauth.py`

- [ ] **Step 1: Add failing tests for role-specific model and reasoning profile**

Append these tests to `tests/test_codex_oauth.py`:

```python
@pytest.mark.unit
def test_trading_graph_uses_codex_oauth_role_models_and_reasoning(monkeypatch):
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    monkeypatch.setenv("TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE", "codex_oauth")
    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "llm_provider": "openai",
            "deep_think_llm": "gpt-5.4",
            "quick_think_llm": "gpt-5.4-mini",
            "codex_oauth_deep_think_llm": "gpt-5.5",
            "codex_oauth_quick_think_llm": "gpt-5.5",
            "codex_oauth_deep_reasoning_effort": "high",
            "codex_oauth_quick_reasoning_effort": "low",
            "codex_oauth_service_tier": "priority",
        }
    )
    graph = object.__new__(TradingAgentsGraph)
    graph.config = config

    assert graph._get_llm_model("deep") == "gpt-5.5"
    assert graph._get_llm_model("quick") == "gpt-5.5"
    assert graph._get_provider_kwargs("deep") == {
        "reasoning_effort": "high",
        "service_tier": "priority",
    }
    assert graph._get_provider_kwargs("quick") == {
        "reasoning_effort": "low",
        "service_tier": "priority",
    }


@pytest.mark.unit
def test_trading_graph_regular_openai_model_and_reasoning_unchanged(monkeypatch):
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    monkeypatch.delenv("TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE", raising=False)
    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "llm_provider": "openai",
            "deep_think_llm": "gpt-5.4",
            "quick_think_llm": "gpt-5.4-mini",
            "openai_reasoning_effort": "medium",
        }
    )
    graph = object.__new__(TradingAgentsGraph)
    graph.config = config

    assert graph._get_llm_model("deep") == "gpt-5.4"
    assert graph._get_llm_model("quick") == "gpt-5.4-mini"
    assert graph._get_provider_kwargs("deep") == {"reasoning_effort": "medium"}
    assert graph._get_provider_kwargs("quick") == {"reasoning_effort": "medium"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_codex_oauth.py::test_trading_graph_uses_codex_oauth_role_models_and_reasoning tests/test_codex_oauth.py::test_trading_graph_regular_openai_model_and_reasoning_unchanged -q
```

Expected: FAIL because `_get_llm_model` does not exist and `_get_provider_kwargs` does not accept a role argument.

- [ ] **Step 3: Import Codex profile helpers in `trading_graph.py`**

Add this import below `from tradingagents.llm_clients import create_llm_client`:

```python
from tradingagents.llm_clients.codex_oauth import (
    codex_oauth_reasoning_effort,
    codex_oauth_requested,
    codex_oauth_role_model,
    codex_oauth_service_tier,
)
```

- [ ] **Step 4: Add `_get_llm_model` and role-aware `_get_provider_kwargs`**

Replace:

```python
    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
```

with:

```python
    def _get_llm_model(self, role: str) -> str:
        """Return the model for a quick or deep thinking role."""
        provider = self.config.get("llm_provider", "").lower()
        if provider == "openai" and codex_oauth_requested():
            return codex_oauth_role_model(self.config, role)
        if role == "deep":
            return self.config["deep_think_llm"]
        if role == "quick":
            return self.config["quick_think_llm"]
        raise ValueError(f"Unknown LLM role: {role}")

    def _get_provider_kwargs(self, role: str = "deep") -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
```

Inside the `elif provider == "openai":` block, replace:

```python
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
```

with:

```python
            if codex_oauth_requested():
                kwargs["reasoning_effort"] = codex_oauth_reasoning_effort(self.config, role)
                service_tier = codex_oauth_service_tier(self.config)
                if service_tier:
                    kwargs["service_tier"] = service_tier
            else:
                reasoning_effort = self.config.get("openai_reasoning_effort")
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort
```

- [ ] **Step 5: Use separate deep/quick kwargs and models in `__init__`**

Replace:

```python
        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
```

with:

```python
        # Initialize LLMs with provider-specific thinking configuration.
        # Codex OAuth can use the same model with different quick/deep reasoning effort.
        deep_kwargs = self._get_provider_kwargs("deep")
        quick_kwargs = self._get_provider_kwargs("quick")

        # Add callbacks to kwargs if provided (passed to LLM constructor).
        if self.callbacks:
            deep_kwargs["callbacks"] = self.callbacks
            quick_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self._get_llm_model("deep"),
            base_url=self.config.get("backend_url"),
            **deep_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self._get_llm_model("quick"),
            base_url=self.config.get("backend_url"),
            **quick_kwargs,
        )
```

- [ ] **Step 6: Run tests to verify Task 4 passes**

Run:

```bash
uv run pytest tests/test_codex_oauth.py -q
```

Expected: PASS for all Codex OAuth unit tests.

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add tradingagents/graph/trading_graph.py tests/test_codex_oauth.py
git commit -m "feat: apply Codex OAuth quick and deep LLM profile"
```

---

### Task 5: Integration Verification and Issue #1 Smoke Path

**Files:**
- Modify: `tests/test_codex_oauth.py`
- No production code changes expected in this task unless tests reveal a bug.

- [ ] **Step 1: Run focused LLM client tests**

Run:

```bash
uv run pytest tests/test_codex_oauth.py tests/test_deepseek_reasoning.py tests/test_cli_non_interactive.py -q
```

Expected: PASS. If a DeepSeek or CLI non-interactive test fails, inspect whether the failure is caused by Codex OAuth changes leaking into non-Codex paths; fix the leak before continuing.

- [ ] **Step 2: Run full test suite**

Run:

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run a Codex backend probe before the full analysis**

Run:

```bash
TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE=codex_oauth \
TRADINGAGENTS_CODEX_OAUTH_QUICK_MODEL=gpt-5.5 \
TRADINGAGENTS_CODEX_OAUTH_DEEP_MODEL=gpt-5.5 \
TRADINGAGENTS_CODEX_OAUTH_QUICK_REASONING_EFFORT=low \
TRADINGAGENTS_CODEX_OAUTH_DEEP_REASONING_EFFORT=high \
TRADINGAGENTS_CODEX_OAUTH_SERVICE_TIER=priority \
uv run python - <<'PY'
from langchain_core.messages import HumanMessage, SystemMessage
from tradingagents.llm_clients.openai_client import OpenAIClient

llm = OpenAIClient("gpt-5.5", provider="openai", reasoning_effort="low", service_tier="priority").get_llm()
response = llm.invoke([SystemMessage(content="Reply with exactly ok."), HumanMessage(content="Say ok.")])
print(response.content)
PY
```

Expected: prints `ok` or a short response containing `ok`. If it fails with credential text, run `codex login` and retry. If it fails with unsupported parameter text, inspect `CodexOAuthChatOpenAI._get_request_payload` and remove the unsupported field.

- [ ] **Step 4: Run the Issue #1 non-interactive manual command**

Run:

```bash
TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE=codex_oauth \
TRADINGAGENTS_CODEX_OAUTH_QUICK_MODEL=gpt-5.5 \
TRADINGAGENTS_CODEX_OAUTH_DEEP_MODEL=gpt-5.5 \
TRADINGAGENTS_CODEX_OAUTH_QUICK_REASONING_EFFORT=low \
TRADINGAGENTS_CODEX_OAUTH_DEEP_REASONING_EFFORT=high \
TRADINGAGENTS_CODEX_OAUTH_SERVICE_TIER=priority \
uv run tradingagents --non-interactive --ticker SPY --research-depth shallow --analysts market --no-display-report --save-path reports/manual-spy-non-interactive
```

Expected: command exits 0 and writes the manual SPY non-interactive report under `reports/manual-spy-non-interactive`.

- [ ] **Step 5: Run the default non-interactive smoke command requested by Issue #1**

Run:

```bash
TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE=codex_oauth \
TRADINGAGENTS_CODEX_OAUTH_QUICK_MODEL=gpt-5.5 \
TRADINGAGENTS_CODEX_OAUTH_DEEP_MODEL=gpt-5.5 \
TRADINGAGENTS_CODEX_OAUTH_QUICK_REASONING_EFFORT=low \
TRADINGAGENTS_CODEX_OAUTH_DEEP_REASONING_EFFORT=high \
TRADINGAGENTS_CODEX_OAUTH_SERVICE_TIER=priority \
uv run tradingagents --non-interactive --ticker SPY --no-display-report --save-path reports/manual-spy-default-non-interactive
```

Expected: command exits 0 and writes a report under `reports/manual-spy-default-non-interactive`.

- [ ] **Step 6: Commit verification notes if code changed during Task 5**

If Task 5 required code fixes, run:

```bash
git add tradingagents tests
git commit -m "fix: stabilize Codex OAuth integration path"
```

If Task 5 required no code fixes, do not create an empty commit.

---

## Self-Review

- Spec coverage: The plan covers hard-fail credential selection, file-only credential discovery, no API-key fallback, Codex backend Responses transport, internal streaming, `invoke`, `bind_tools`, function-calling structured output, configurable GPT-5.5 quick/deep profile, `low`/`high` reasoning defaults, and `service_tier=priority`.
- Placeholder scan: The plan has concrete file paths, exact tests, exact implementation snippets, exact verification commands, and exact expected outcomes.
- Type consistency: Helper names introduced in Task 1 are reused in Tasks 3 and 4 with the same signatures: `codex_oauth_requested()`, `load_codex_oauth_credentials()`, `codex_oauth_role_model(config, role)`, `codex_oauth_reasoning_effort(config, role)`, and `codex_oauth_service_tier(config)`.
