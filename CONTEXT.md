# TradingAgents CLI

TradingAgents CLI runs multi-agent financial analysis from terminal inputs. It supports both guided interactive runs and explicit non-interactive runs for scripts while preserving safe existing defaults where possible.

## Language

**Interactive Run**:
A TradingAgents CLI run that prompts the user for analysis inputs before execution.
_Avoid_: Wizard, questionnaire mode

**Non-Interactive Run**:
A TradingAgents CLI run that starts from defaults and command-line options without prompting for analysis inputs.
_Avoid_: Batch mode, headless mode

**Research Depth**:
The selected amount of debate and risk discussion used during analysis, expressed as shallow, medium, or deep.
_Avoid_: Debate count, recursion depth

**Analyst Team**:
The set of analyst agents selected for the first stage of a TradingAgents analysis.
_Avoid_: Analyst list, agent selection

**LLM Configuration**:
The provider and model settings used by TradingAgents to run analysis agents.
_Avoid_: Model flags, provider options

**LLM Credential Source**:
The source of credentials used to authenticate an LLM provider for a TradingAgents run.
_Avoid_: Auth mode, API key settings

**OAuth Preference**:
A setting that forces native OpenAI authentication to use Codex OAuth credentials even when an API key is available.
_Avoid_: Codex mode, subscription flag

**Codex OAuth Credential**:
A valid credential from an existing Codex login used as an OpenAI **LLM Credential Source**.
_Avoid_: Codex API key, refreshed session

**Codex OAuth Support**:
A pragmatic layer for running TradingAgents with a personal Codex login, prioritizing the core analysis flow over full provider-feature parity.
_Avoid_: OpenAI replacement, complete provider integration

**Codex OAuth LLM Profile**:
The model, reasoning effort, and service-tier settings used when **OAuth Preference** selects **Codex OAuth Support**.
_Avoid_: Default OpenAI model settings, API-key model profile

**Fast Service Tier**:
Priority processing requested for Codex OAuth calls. The user-facing fast setting maps to the Codex backend service tier request value `priority`.
_Avoid_: Fast model, reasoning effort

**Ticker**:
The exact market symbol selected as the analysis subject.
_Avoid_: Stock, asset, symbol

## Relationships

- A **Non-Interactive Run** uses the same safe default choices as an **Interactive Run** unless command-line options override them.
- A **Non-Interactive Run** requires an explicit **Ticker** because choosing an analysis subject implicitly can trigger unwanted paid analysis.
- A **Non-Interactive Run** uses the full **Analyst Team** by default: market, social, news, and fundamentals.
- A **Non-Interactive Run** does not expose **LLM Configuration** options; it uses the internal default configuration.
- OpenAI **LLM Configuration** defaults to medium reasoning effort.
- Native OpenAI **LLM Configuration** selects an **LLM Credential Source** without changing the provider or model.
- When **OAuth Preference** is set, native OpenAI **LLM Configuration** must use Codex OAuth credentials even if an API key is available.
- **OAuth Preference** is configured outside **Interactive Run** and **Non-Interactive Run** inputs through `TRADINGAGENTS_OPENAI_CREDENTIAL_SOURCE=codex_oauth`.
- A **Codex OAuth Credential** must already be available and valid; TradingAgents does not create or refresh a Codex login.
- In the first implementation, **Codex OAuth Credential** discovery is file-based through `CODEX_HOME/auth.json` or the default `~/.codex/auth.json`.
- If **OAuth Preference** is set and a valid **Codex OAuth Credential** is unavailable, the run fails instead of falling back to an API key.
- A missing or expired **Codex OAuth Credential** error should name the auth file that was checked, tell the user to run `codex login`, and explain that API-key fallback is disabled because **OAuth Preference** was set.
- **Codex OAuth Support** prioritizes keeping the main TradingAgents analysis flow working; compatibility gaps in less-used LLM features can be tracked for later instead of blocking the first implementation.
- The first **Codex OAuth Support** target is core TradingAgents analysis compatibility, not full parity with every OpenAI API-key client feature.
- Structured outputs used by the Research Manager, Trader, and Portfolio Manager are part of core TradingAgents analysis compatibility for **Codex OAuth Support**.
- **Codex OAuth Support** should handle core structured outputs through function-calling compatibility rather than prompt-only JSON parsing.
- The first **Codex OAuth Support** compatibility surface includes `invoke`, `bind_tools`, and function-calling structured output because those paths drive the core analysis flow.
- **Codex OAuth Support** may use Codex backend-required streaming internally while preserving the normal TradingAgents invocation shape.
- **Codex OAuth LLM Profile** is configurable separately from the existing OpenAI API-key defaults.
- The initial **Codex OAuth LLM Profile** uses `gpt-5.5` for both quick and deep thinking, with default reasoning effort `low` for quick thinking and `high` for deep thinking.
- **Codex OAuth LLM Profile** model and reasoning effort settings should be easy to override without changing code.
- **Fast Service Tier** is enabled in the initial **Codex OAuth LLM Profile** by sending the Codex backend request value `priority`, not the literal value `fast`.
- **Codex OAuth LLM Profile** is configured through environment variables and internal config keys, not through additional **Interactive Run** or **Non-Interactive Run** inputs.
- **Research Depth** maps to fixed internal round counts: shallow is 1, medium is 3, and deep is 5.

## Example dialogue

> **Dev:** "Should a non-interactive run skip saving reports because it is script-friendly?"
> **Domain expert:** "No — a **Non-Interactive Run** should preserve safe **Interactive Run** defaults unless the user overrides them, but it must still require an explicit **Ticker**."

> **Dev:** "If both an OpenAI API key and Codex OAuth credentials are present, which **LLM Credential Source** should OpenAI use?"
> **Domain expert:** "Use Codex OAuth credentials when **OAuth Preference** is set; otherwise preserve the existing API-key behavior."

> **Dev:** "Should TradingAgents refresh a **Codex OAuth Credential** during a long run?"
> **Domain expert:** "No — TradingAgents should consume an already-valid credential and tell the user to renew the Codex login if it is missing or expired."

> **Dev:** "If **OAuth Preference** is set but the **Codex OAuth Credential** is missing or expired, should TradingAgents silently fall back to the OpenAI API key?"
> **Domain expert:** "No — hard fail. The preference is close to a forced credential-source choice, so fallback would hide whether Codex OAuth was actually used."

> **Dev:** "What should a hard failure say when **OAuth Preference** is set but no valid **Codex OAuth Credential** exists?"
> **Domain expert:** "Name the auth file that was checked, tell the user to run `codex login`, and say API-key fallback is disabled because Codex OAuth was explicitly requested."

> **Dev:** "Does **Codex OAuth Support** need complete parity with the existing OpenAI API-key client before it is useful?"
> **Domain expert:** "No — it is a pragmatic personal layer. The core analysis flow matters first; compatibility gaps can become follow-up issues."

> **Dev:** "Can **Codex OAuth Support** defer structured output because it is an advanced LLM feature?"
> **Domain expert:** "No — the structured outputs used by the Research Manager, Trader, and Portfolio Manager are part of the core analysis flow and should be handled in the first implementation."

> **Dev:** "Should structured outputs under **Codex OAuth Support** be implemented by asking for JSON in the prompt and parsing the text?"
> **Domain expert:** "No — use function-calling compatibility for the core structured outputs so the behavior stays aligned with the existing OpenAI client."

> **Dev:** "Can analyst tool calling be deferred as a provider-parity detail?"
> **Domain expert:** "No — analyst reports depend on `bind_tools`, so tool calling is part of first-implementation core compatibility."

> **Dev:** "If the Codex backend requires streaming, should that change the way TradingAgents callers consume LLM results?"
> **Domain expert:** "No — the adapter can stream internally and still return the normal invocation result shape to TradingAgents."

> **Dev:** "Should fast mode be represented as a different model or reasoning effort?"
> **Domain expert:** "No — fast mode is a **Fast Service Tier** choice. The model and reasoning effort remain separately configurable."

> **Dev:** "Should Codex OAuth model and reasoning settings become CLI prompts or flags?"
> **Domain expert:** "No — keep them outside **Interactive Run** and **Non-Interactive Run** inputs. Use environment variables and internal config keys."

## Flagged ambiguities

- "research depth" could mean arbitrary numeric rounds or named levels — resolved: users should see shallow, medium, and deep, while 1, 3, and 5 remain accepted aliases.
- "Codex OAuth support" could mean replacing API-key authentication globally or adding an opt-in OpenAI credential source — resolved: it means an **OAuth Preference** for native OpenAI **LLM Configuration**, and that preference takes precedence over API keys when set.
- "support OAuth" could mean implementing the OAuth login lifecycle — resolved: **Codex OAuth Credential** handling is consume-only, and login/refresh remains owned by Codex.
- "consume Codex OAuth credentials" could include OS credential stores such as macOS Keychain — resolved: the first implementation only reads `CODEX_HOME/auth.json` or `~/.codex/auth.json`.
- "Codex OAuth support" could imply full OpenAI provider parity — resolved: **Codex OAuth Support** is initially a pragmatic layer for the core analysis flow, not a complete replacement for all OpenAI API-key behavior.
- "less-used LLM features" could include structured output — resolved: structured outputs in Research Manager, Trader, and Portfolio Manager are core, not deferrable.
- "support structured output" could mean prompt-only JSON parsing — resolved: for core structured outputs, **Codex OAuth Support** should use function-calling compatibility.
- "fast mode" could mean a faster model or lower reasoning effort — resolved: in **Codex OAuth Support**, it means **Fast Service Tier**, mapped to backend `service_tier=priority`.
