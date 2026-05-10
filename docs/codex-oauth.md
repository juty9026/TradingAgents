# Codex OAuth

Codex OAuth is an optional OpenAI LLM credential source for TradingAgents runs. It uses an existing `codex login` credential and is not a transparent replacement for every OpenAI API-key behavior.

## What It Is

Codex OAuth support applies only when OpenAI is the selected LLM provider and `TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE=codex_oauth` is set. In that mode, TradingAgents uses the Codex auth file created by `codex login`.

When Codex OAuth is requested, `OPENAI_API_KEY` is ignored. Missing, unreadable, expired, or otherwise invalid Codex credentials fail fast instead of falling back to an API key.

## Setup

Run Codex login first:

```bash
codex login
```

Then enable Codex OAuth for OpenAI provider runs:

```bash
export TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE=codex_oauth
export TRADINGAGENTS_CODEX_OAUTH_QUICK_MODEL=gpt-5.5
export TRADINGAGENTS_CODEX_OAUTH_DEEP_MODEL=gpt-5.5
export TRADINGAGENTS_CODEX_OAUTH_QUICK_REASONING_EFFORT=low
export TRADINGAGENTS_CODEX_OAUTH_DEEP_REASONING_EFFORT=high
export TRADINGAGENTS_CODEX_OAUTH_SERVICE_TIER=priority
```

These settings control credentials and the Codex OAuth LLM profile. They do not change report language, ticker selection, research depth, analyst selection, or other CLI run inputs.

## Environment Reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE` | unset | Set to `codex_oauth` to force OpenAI provider runs to use Codex OAuth credentials. Any other value preserves the normal API-key path. |
| `TRADINGAGENTS_CODEX_OAUTH_QUICK_MODEL` | `gpt-5.5` | Model used for quick-thinking OpenAI roles when Codex OAuth is enabled. |
| `TRADINGAGENTS_CODEX_OAUTH_DEEP_MODEL` | `gpt-5.5` | Model used for deep-thinking OpenAI roles when Codex OAuth is enabled. |
| `TRADINGAGENTS_CODEX_OAUTH_QUICK_REASONING_EFFORT` | `low` | Reasoning effort used for quick-thinking OpenAI roles when Codex OAuth is enabled. |
| `TRADINGAGENTS_CODEX_OAUTH_DEEP_REASONING_EFFORT` | `high` | Reasoning effort used for deep-thinking OpenAI roles when Codex OAuth is enabled. |
| `TRADINGAGENTS_CODEX_OAUTH_SERVICE_TIER` | `priority` | Service tier requested for Codex OAuth calls. Accepted values are `priority` and `fast`. |

## Credential Files

TradingAgents reads the Codex auth file from:

1. `${CODEX_HOME}/auth.json`, when `CODEX_HOME` is set.
2. `~/.codex/auth.json`, when `CODEX_HOME` is not set.

The auth file must come from a valid `codex login` session. TradingAgents does not create Codex logins, refresh Codex tokens, or read credentials from OS keychains.

## Service Tier

Use `TRADINGAGENTS_CODEX_OAUTH_SERVICE_TIER=priority` in examples and environment files. The user-facing alias `fast` is accepted and maps to the backend `priority` service tier.

Unsupported values fail during configuration instead of being silently ignored.

## Troubleshooting

**No Codex auth file was found**

Run `codex login`, then confirm the auth file exists at `${CODEX_HOME}/auth.json` or `~/.codex/auth.json`.

**The Codex OAuth access token is expired or too close to expiry**

Run `codex login` again to renew the Codex session. TradingAgents does not refresh Codex credentials.

**OPENAI_API_KEY is set but ignored**

This is expected when `TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE=codex_oauth` is set. Unset `TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE` to return to the normal OpenAI API-key path.

**Unsupported service tier**

Use `priority` or `fast`. `fast` is accepted as an alias for `priority`.

**A stale Codex login still fails**

Run `codex login` again and confirm TradingAgents is reading the auth file location you expect. If `CODEX_HOME` is set, TradingAgents reads only `${CODEX_HOME}/auth.json`.

## Current Limitations

Codex OAuth support targets the core TradingAgents analysis flow rather than full parity with every OpenAI API-key client feature. The first compatibility surface is normal invocation, analyst tool binding, and function-calling structured output for the Research Manager, Trader, and Portfolio Manager.

TradingAgents consumes an existing Codex login only. It does not create logins, refresh tokens, use OS keychains, or add `.env.local` loading behavior.

Codex OAuth does not change the output language. Interactive runs can prompt for report language, and non-interactive runs use the configured default language unless a separate language option is added later.

## Known Follow-Ups

- [#4](https://github.com/juty9026/TradingAgents/issues/4) tracks additional robustness for Codex SSE streams where assistant text may arrive through `response.output_item.done` events.
