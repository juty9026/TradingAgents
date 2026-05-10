# Codex OAuth Support

TradingAgents will support Codex OAuth as an explicit OpenAI **LLM Credential Source** for personal runs, not as a transparent replacement for every OpenAI API-key behavior. When `TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE=codex_oauth` is set, TradingAgents must use an already-valid Codex login from `CODEX_HOME/auth.json` or `~/.codex/auth.json`, hard fail if it is unavailable, and call the Codex backend with the Codex-specific Responses transport instead of falling back to `OPENAI_API_KEY`.

This chooses a pragmatic core-flow integration over full provider parity: first implementation compatibility must cover `invoke`, analyst `bind_tools`, and function-calling structured output for the Research Manager, Trader, and Portfolio Manager, while less-used OpenAI client features can become follow-up issues. The Codex OAuth LLM profile is configured outside interactive/non-interactive CLI inputs, defaults both quick and deep roles to `gpt-5.5`, uses `low` reasoning for quick and `high` for deep, and maps user-facing fast mode to Codex backend `service_tier=priority`.

Hard failures must be explicit: name the checked auth file, tell the user to run `codex login`, and explain that API-key fallback is disabled because Codex OAuth was explicitly requested.
