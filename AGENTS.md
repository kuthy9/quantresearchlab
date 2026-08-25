# Repository Guidelines

## Project Structure & Module Organization

Runtime code lives in `smc_trader/`; keep causal observation, immutable event/state, Brain, execution, and research responsibilities in their existing focused modules. Semantic authorities are versioned under `semantics/`, while runtime and experiment settings belong in `configs/`. Use `scripts/` for reproducible materializers, validators, and bounded runners. Tests mirror behavior in `tests/test_*.py`. Frozen study contracts and compact results live in `experiments/manifests/` and `experiments/results/`; explanatory and evidence documents live in `docs/` and `docs/evidence/`.

## Build, Test, and Development Commands

Create the supported environment from `uv.lock` with `uv sync --extra test`. This is a Python package with no separate compile step.

- `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider` runs the normal suite; `historical_frozen` tests are excluded by `pyproject.toml`.
- `.venv/bin/python -m pytest tests/test_foundation_adapter.py -q -p no:cacheprovider` runs one focused contract file.
- `git diff --check` catches whitespace errors before commit.

Run formal research scripts only through their frozen manifests and documented validation modes. Never add `--force` to a registered run casually.

## Coding Style & Naming Conventions

Target Python 3.10+, use four-space indentation, explicit type hints, immutable dataclasses where state is factual, and deterministic ordering/serialization. Follow `snake_case` for modules, functions, and variables; `PascalCase` for classes; `UPPER_SNAKE_CASE` for constants. Preserve `event_time` versus `known_at`, exact source IDs, semantic versions, and fail-closed validation. No formatter is configured; match nearby code and keep imports grouped.

## Testing Guidelines

Name files `test_<area>.py` and cases `test_<behavior>`. Add boundary, replay/determinism, checkpoint, provenance, and malformed-input cases for causal changes. Run focused tests first, then the full default suite. Do not regenerate historical governance artifacts during ordinary unit testing.

## Commit & Pull Request Guidelines

History uses Conventional Commit prefixes such as `feat:`, `fix:`, and `refactor:` with imperative summaries. Keep commits scoped and include changed manifests/results only when their identities are intentionally refreshed. Pull requests should explain causal and authority impact, list verification commands, link issues, and report artifact hashes; add screenshots only for visualization changes.

## Security & Data Authority

`data/`, `inputs/`, `outputs/`, and large ledgers are ignored but governed—not disposable. Never open sealed holdouts, overwrite hash-bound artifacts, weaken no-clobber checks, or imply causal, profitability, trading, or live-execution authority without the registered gates.

Keep the two semantic identities explicit: atomic events use `smc_semantics_v1.2`; the additive `smc_semantic_foundation_v2.0` projection declares v1.2 as its parent. Do not describe this pairing as a unified full-stack v2 protocol.
