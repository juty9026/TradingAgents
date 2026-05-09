# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

This repo uses a single-context domain documentation layout.

## Before exploring, read these

- `CONTEXT.md` at the repo root, if it exists.
- `docs/adr/`, if it exists.

If these files don't exist, proceed silently. The producer skill (`/grill-with-docs`) creates them lazily when terms or decisions actually get resolved.

## File structure

Single-context repo:

```text
/
|-- CONTEXT.md
|-- docs/adr/
|   |-- 0001-example-decision.md
|   `-- 0002-another-decision.md
`-- tradingagents/
```

## Use the glossary's vocabulary

When output names a domain concept in an issue title, refactor proposal, hypothesis, or test name, use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, either reconsider whether the project uses that language or note the gap for `/grill-with-docs`.

## Flag ADR conflicts

If output contradicts an existing ADR, surface it explicitly rather than silently overriding.
