# Repository Guidelines

## Project Structure & Module Organization

Runtime code lives in `smc_trader/`; preserve the existing Eye, state, Brain,
execution, and research ownership boundaries. Version semantic authorities in
`semantics/`, settings in `configs/`, and reproducible materializers or bounded
runners in `scripts/`. Tests use `tests/test_*.py`. Frozen study contracts and
results live in `experiments/`; current explanations and receipts live in
`docs/` and `docs/evidence/`.

## Build, Test, and Development Commands

Create the environment with `uv sync --extra test`; there is no compile step.

- `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`
  runs the default suite; `historical_frozen` is excluded by `pyproject.toml`.
- `.venv/bin/python -m pytest tests/test_foundation_adapter.py -q -p no:cacheprovider`
  runs a focused contract file.
- `git diff --check` catches whitespace errors before commit.

Run formal research only through frozen manifests and documented validation
modes. Never casually add `--force` to a registered run.

## Coding Style & Naming Conventions

Target Python 3.10+, four-space indentation, explicit type hints, immutable
factual dataclasses, and deterministic serialization. Use `snake_case`,
`PascalCase`, and `UPPER_SNAKE_CASE` conventionally. Preserve `event_time`
versus `known_at`, exact source IDs, semantic versions, and fail-closed checks.
No formatter is configured; match adjacent code and group imports.

## Runtime Authority Boundaries

`ImmutableEventStore` owns atomic history; the in-memory Foundation ledger owns
Foundation revisions. Hot projections and snapshots carry current views,
counts, indexes, and rolling hashes—not full revision history. Do not restore
production `FOUNDATION_STATE_CHANGED` emission; its decoder is legacy-read-only.
The configured action authority remains `legacy_decision_risk_compat` until one
registered TradeIntent-to-FSM migration replaces it.

## Testing Guidelines

Name files `test_<area>.py` and cases `test_<behavior>`. Add boundary, replay,
checkpoint, provenance, and malformed-input cases for causal changes. Run
focused tests first, then the default suite. Never regenerate historical
governance artifacts during ordinary tests.

## Commit & Pull Request Guidelines

Use Conventional Commit prefixes (`feat:`, `fix:`, `refactor:`) and imperative
summaries. Keep commits scoped. Pull requests must explain causal/authority
impact, list checks, link issues, and report intentional artifact hash changes.

## Security & Data Authority

`data/`, `inputs/`, `outputs/`, and large ledgers are governed, not disposable.
Do not open sealed holdouts, overwrite hash-bound artifacts, weaken no-clobber
checks, or imply causal, profitability, trading, or live authority. Old frozen
receipts remain historical; refreeze a new identity instead of editing them.

Keep the two semantic identities explicit: atomic events use `smc_semantics_v1.2`; the additive `smc_semantic_foundation_v2.0` projection declares v1.2 as its parent. Do not describe this pairing as a unified full-stack v2 protocol.
