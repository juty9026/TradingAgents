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
        raise _credential_error(
            auth_path, "The Codex OAuth auth file could not be read."
        ) from exc

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
        raise _credential_error(
            auth_path, "The Codex OAuth access token is expired or too close to expiry."
        )

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
    if not normalized:
        return None
    if normalized in {"fast", "priority"}:
        return "priority"
    raise ValueError(
        "Unsupported Codex OAuth service tier "
        f"{value!r}; expected one of: priority, fast."
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
        raise _credential_error(
            auth_path, "The Codex OAuth access token could not be decoded."
        ) from exc
    if not isinstance(claims, dict):
        raise _credential_error(
            auth_path, "The Codex OAuth access token has invalid claims."
        )
    return claims


def _extract_expiry(claims: dict[str, Any], auth_path: Path) -> int:
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)):
        raise _credential_error(
            auth_path, "The Codex OAuth access token is missing expiry."
        )
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
