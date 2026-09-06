# Repository Guidelines

## Project Structure & Module Organization

Runtime code lives in `smc_trader/`; preserve the existing Eye, state, Brain,
execution, and research ownership boundaries. Version semantic authorities in
`semantics/`, settings in `configs/`, and reproducible materializers or bounded
runners in `scripts/`. Tests use `tests/test_*.py`. Frozen study contracts and
results live in `experiments/`; current explanations and receipts live in
`docs/` and `docs/evidence/`.

## Runtime Architecture

One data entry, one Eye entry, one history authority, one current view:

```text
Bar
 └─ CausalMarketReader        causal.py            data normalizer
     ├─ market_clock.py       registered session calendar
     └─ scale_registry.py     ScaleSpec / scale_registry_id
     └─ CausalObserver        observation.py       semantic event engine
         ├─ structure.py / liquidity.py / displacement.py
         ├─ zone.py           (canonical owner of the former "Group 3")
         ├─ range_auction.py  (canonical owner of the former "Group 4")
         ├─ interaction.py    (Eye half of the former "Group 5")
         ├─ semantic_event_emitter.py
         │                    SemanticEventEmitter — sole event emitter and
         │                    owner of the cross-detector ancestry index
         ├─ event_memory.py   bounded causal working set
         ├─ event_store.py    EventStore — complete atomic history
         └─ market_state.py   MarketSnapshotPublisher
             ├─ TimeframeEventReducer → TimeframeState
             ├─ RelationResolver      → RelationState
             └─ SessionStateReducer   → SessionState
                 └─ MarketSnapshot    — current market view
 └─ ContinuousSMCEngine       engine.py            orchestration
     ├─ scene_graph.py           Engine-owned research/visualisation view
     ├─ execution.py             execution-reality scoring
     ├─ brain_entry_sequence.py  Brain interpretation of Eye facts
     ├─ neutral projection       one OpenMarketThesis per clock
     ├─ playbooks.py             PlaybookBrain.update(...)
     └─ decision.py → risk.py    sole runtime action authority
```

The Eye imports no downstream module. `scene_graph.py` is owned by
`ContinuousSMCEngine`, which advances it over one completed Eye observation and
stamps the resulting `scene_*` delta identities; a graph failure poisons the
observer through `CausalObserver.mark_terminal_failure` because the reducers
have already advanced. `execution.py` owns `ExecutionRealityInput` and the
cost/fillability score; `model.py` owns the inert not-evaluated value beside
`ExecutionObservation`; `ContinuousSMCEngine._score_execution` derives the
score and the Eye only transports the result. That boundary is enforced by
`tests/test_eye_module_boundary.py`. `eye_statistics.py`, `visualization.py`, `shadow_*`,
`*_research*`, `causal_cases.py` and `market_cases.py` are optional projections
and research consumers. None of them may become a second market-state
authority.

`EventStore` is the Eye's internal history authority and is not part of the
package's public surface. Research and replay tools that need read-only event
lineage import `smc_trader.event_store` directly.

Naming rule: `Zone`, `RangeAuction` and `Interaction` are the public concepts.
`Group3`/`Group4`/`Group5` and `Phase 4/5/6` survive only as historical or
internal migration names — in `group3.py`/`group4.py`/`group5.py` pickle shims,
in `MarketObservation` field names, and in frozen research-manifest binding
keys. Do not introduce them anywhere new.

The current implementation-versus-plan authority is
[docs/refactor/current_implementation_status.md](docs/refactor/current_implementation_status.md);
the registered semantic definitions are
[docs/refactor/preregistered_semantics_v1_3_2026-08-31.md](docs/refactor/preregistered_semantics_v1_3_2026-08-31.md)
for the atomic layer and
[docs/refactor/canonical_semantic_foundation_v2.1.md](docs/refactor/canonical_semantic_foundation_v2.1.md)
for the Foundation projection.

## Build, Test, and Development Commands

Create the environment with `uv sync --extra test`; there is no compile step.

- `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`
  runs daily semantic/runtime tests; historical and research-orchestration
  groups are excluded by `pyproject.toml`.
- `.venv/bin/python -m pytest -m 'research_runner or research_orchestration' -q -p no:cacheprovider`
  runs bounded study, publication, and operational-contract tests separately.
  These are excluded from the default loop, so run them after touching a
  research runner, a manifest template, or a runtime identity binding.
- `.venv/bin/python -m pytest tests/test_semantic_foundation_projection.py -q -p no:cacheprovider`
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

`EventStore` owns the complete atomic history; `MarketSnapshot` owns the current
market view. Hot projections and snapshots carry current views, counts, indexes,
and rolling hashes — not full revision history.

Foundation v2 is no longer a runtime component. `foundation_adapter.py` was
removed, no runtime module constructs `FoundationProjection`,
`FoundationProjectionReducer` or `FoundationRecordLedger`, and the Engine
carries no Foundation version in its checkpoint state. Foundation survives as
the hash-bound `smc_semantic_foundation_v2.1` registry identity, as
`foundation_version`-stamped Structural Leg evidence, and as cold definition
modules (`semantic_foundation.py`, `semantic_lifecycle.py`, `semantic_zones.py`,
and the geometry/cluster/range builders in `market_state.py`) whose only
consumers are their focused tests. Retain those definitions; do not present them
as hot state and do not build a second lifecycle engine beside them.

Do not restore production `FOUNDATION_STATE_CHANGED` emission; its decoder is
legacy-read-only. The configured action authority remains
`legacy_decision_risk_compat` until one registered TradeIntent-to-FSM migration
replaces it; `engine.py` rejects a non-zero `TradeIntent` before Decision/Risk.

Never create a second history, state, thesis, lifecycle, or execution authority.
A compatibility decoder or adapter may exist, but it may not become a production
authority, and it must not be what a research contract attests: bind runtime
provenance to the module that actually implements the semantics, not to a shim.

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

Keep the two semantic identities explicit: atomic events use `smc_semantics_v1.3`; the additive `smc_semantic_foundation_v2.1` projection declares v1.3 as its parent. Do not describe this pairing as a unified full-stack v2 protocol.
