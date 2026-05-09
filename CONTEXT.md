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

**Ticker**:
The exact market symbol selected as the analysis subject.
_Avoid_: Stock, asset, symbol

## Relationships

- A **Non-Interactive Run** uses the same safe default choices as an **Interactive Run** unless command-line options override them.
- A **Non-Interactive Run** requires an explicit **Ticker** because choosing an analysis subject implicitly can trigger unwanted paid analysis.
- A **Non-Interactive Run** uses the full **Analyst Team** by default: market, social, news, and fundamentals.
- A **Non-Interactive Run** does not expose **LLM Configuration** options; it uses the internal default configuration.
- OpenAI **LLM Configuration** defaults to medium reasoning effort.
- **Research Depth** maps to fixed internal round counts: shallow is 1, medium is 3, and deep is 5.

## Example dialogue

> **Dev:** "Should a non-interactive run skip saving reports because it is script-friendly?"
> **Domain expert:** "No — a **Non-Interactive Run** should preserve safe **Interactive Run** defaults unless the user overrides them, but it must still require an explicit **Ticker**."

## Flagged ambiguities

- "research depth" could mean arbitrary numeric rounds or named levels — resolved: users should see shallow, medium, and deep, while 1, 3, and 5 remain accepted aliases.
