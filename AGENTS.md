# Repository Guidelines

## Project Structure & Module Organization

Runtime code is split into four subsystem packages. Each owns its own `core/`,
`tests/`, `configs/` and `docs/`, and the split preserves the existing Eye,
state, Brain and execution ownership boundaries rather than redrawing them:

| package | owns |
| --- | --- |
| `eyes/` | the Trading Eye — normalization, the six detectors, semantic-event emission, the event store and market-state reduction |
| `brain/` | the naturally discovered hypothesis engine (proposer, pool, belief updater, forecast), decision and risk. The typed playbooks, DOL and Signal Policy were retired on 2026-09-07; the frozen six-path competition set on 2026-09-09 |
| `execution/` | execution reality, MBO reconstruction and sequential simulation (the order FSM and trade intent were retired on 2026-09-07) |
| `shares/` | data access, the session clock, the scale registry, orchestration (`engine.py`) and the study projections the other three consume |
| `contract/` | every payload that crosses a subsystem boundary, one package per boundary (`market`, `execution`, `eye`, `brain`, `decision`, `risk`, `research`) |

Each package also owns a `scripts/` directory holding its bounded studies,
materializers and throwaway probes; a script that drives the whole stack
(`shares/scripts/run_continuous_replay.py`) belongs to `shares/`.

`contract/` also sits at the repository root: it is consumed by all four
subsystems, so it cannot live inside any one of them.

Two more directories stay at the root because the atomic semantic identity
hashes their contents or their path strings, and moving either would force a
refreeze:

- `configs/` — exactly the eight sealed files: `model.json`, `data_splits.json`
  and the six `primitives_*.json` protocols. Every unsealed protocol lives with
  the subsystem that reads it (`brain/configs/`, `shares/configs/`).
- `semantics/` — `registry_v1_3.yaml` names its own parameters file by the
  repository-root-relative string `semantics/parameters_v1_3.yaml`, and the
  registry's bytes are `registry_sha256`. Relocating the directory rewrites that
  line and moves `atomic_definition_identity` off
  `f92b24c86bf942defc88de4edb7be16cc2a30dd64fde3b4432657780648b1f0c`.

Tests use `<subsystem>/tests/test_*.py`. Bounded studies write under `outputs/`;
current explanations and receipts live in `<subsystem>/docs/` and
`eyes/docs/evidence/`.
The frozen Phase 5 signal-research and Phase 6 MBO-mechanism runners, their
`smc_trader` primitives and their `configs/research/` manifest templates were
retired on 2026-09-06; no formal-research runner remains in the repository.

## Runtime Architecture

One data entry, one Eye entry, one history authority, one current view:

```text
Bar
 └─ CausalMarketReader     eyes/core/causal.py             data normalizer
     ├─ shares/core/market_clock.py    registered session calendar
     └─ shares/core/scale_registry.py  ScaleSpec / scale_registry_id
     └─ CausalObserver     eyes/core/observation.py         semantic event engine
         ├─ eyes/core/structure.py / liquidity.py / displacement.py
         ├─ eyes/core/zone.py           (canonical owner of the former "Group 3")
         ├─ eyes/core/range_auction.py  (canonical owner of the former "Group 4")
         ├─ eyes/core/interaction.py    (Eye half of the former "Group 5")
         ├─ eyes/core/semantic_event_emitter.py
         │                    SemanticEventEmitter — sole event emitter and
         │                    owner of the cross-detector ancestry index
         ├─ eyes/core/event_memory.py   bounded causal working set
         ├─ eyes/core/event_store.py    EventStore — complete atomic history
         └─ eyes/core/market_state.py   MarketSnapshotPublisher
             ├─ TimeframeEventReducer → TimeframeState
             ├─ RelationResolver      → RelationState
             └─ SessionStateReducer   → SessionState
                 └─ MarketSnapshot    — current market view
                     (published as contract/eye/observation.MarketObservation)
 └─ ContinuousSMCEngine    shares/core/engine.py            orchestration
     ├─ shares/core/scene_graph.py       Engine-owned research/visualisation view
     ├─ execution/core/execution.py      execution-reality scoring
     │                                   (contract: contract/execution/reality.py)
     ├─ brain/core/brain_entry_sequence.py
     │                                   Brain interpretation of Eye facts
     ├─ neutral projection               one OpenMarketThesis per clock
     ├─ brain/core/forecast.py           the belief producer: one
     │   ├─ hypothesis_proposer.py       MarketBeliefState per clock
     │   ├─ hypothesis_pool.py           (contract: contract/brain/forecast.py)
     │   └─ belief_updater.py
     └─ brain/core/decision.py → risk.py sole runtime action authority
```

**`shares/core/engine.py` cannot currently be imported.** It still imports
`brain.core.playbooks`, `brain.core.playbook_registry`,
`brain.core.dol_probability` and `brain.core.signal_policy`, all of which were
removed with the typed Brain. It also binds `model.path_hypotheses`, which went
with the six-path retirement on 2026-09-09. Eight test modules (135 tests)
cannot be collected until those import blocks and the code behind them are
removed or rebound to `brain/core/forecast.py`, the belief producer that
replaced them, and `shares.ContinuousSMCEngine` is unavailable until then. Run the suite with `--ignore` on those eight modules to exercise the
other 1297 tests.

The Eye imports no downstream module — no `eyes/core/` module imports `brain`,
`execution`, or the orchestration half of `shares`. `shares/core/scene_graph.py`
is owned by `ContinuousSMCEngine`, which advances it over one completed Eye
observation and stamps the resulting `scene_*` delta identities; a graph failure
poisons the observer through `CausalObserver.mark_terminal_failure` because the
reducers have already advanced. `execution/core/execution.py` owns
`ExecutionRealityInput` and the cost/fillability score; `contract/execution/reality.py`
owns the inert not-evaluated value beside `ExecutionObservation`;
`ContinuousSMCEngine._score_execution` derives the score and the Eye only
transports the result. That boundary is enforced by
`eyes/tests/test_eye_module_boundary.py`, which resolves both the intra-package
relative imports and the cross-package absolute ones. `eyes/core/eye_statistics.py`,
`shares/core/visualization.py` and `shares/core/market_cases.py` are optional
projections and study consumers. None of them may become a second market-state
authority.

`contract/` replaced `shares/core/model.py` on 2026-09-08. The type contracts
are now one package per boundary, strictly layered
`market -> execution -> eye -> brain -> decision -> risk -> research`; a package
may import an earlier one and never a later one, and
`eyes/tests/test_eye_module_boundary.py` enforces that per package. Import from
the layer that owns the type, so a consumer's imports state which boundaries it
depends on. See [contract/README.md](contract/README.md).

`shares/__init__.py` remains the aggregate public surface and resolves every
export lazily so the facade never drags the whole engine into an import.

`EventStore` is the Eye's internal history authority and is not part of the
package's public surface. Research and replay tools that need read-only event
lineage import `eyes.core.event_store` directly.

Naming rule: `Zone`, `RangeAuction` and `Interaction` are the public concepts.
`Group3`/`Group4`/`Group5` and `Phase 4/5/6` survive only as historical or
internal migration names — in `MarketObservation` field names and in the
test-local `shares/tests/legacy_group5.py` reducer. The frozen `phase6_*`
evidence-boundary keys went with `brain/configs/path_hypotheses.json` and
`brain/core/market_belief.py` when the six-path competition set was retired on
2026-09-09. The three `group3.py`/`group4.py`/
`group5.py` pickle shims no longer exist; they were removed before the subsystem
split. Do not introduce them anywhere new.

`semantics/registry_v1_3.yaml` and `semantics/parameters_v1_3.yaml` are hashed
into `atomic_definition_identity`, so the seven runtime provenance strings they
carry still name the pre-split modules: `smc_trader.market_state` is today
`eyes/core/market_state.py`, `smc_trader.structure` is `eyes/core/structure.py`,
and `smc_trader.zone` is `eyes/core/zone.py`. Those strings record which module
implemented v1.3 when the identity was frozen; rewriting them would break the
seal, so leave them and read them through this mapping. The same applies to the
`configs/` path strings inside `semantics/parameters_v1_3.yaml` and
`configs/data_splits.json`, which is why the eight sealed protocol files stay at
the repository root.

The current implementation-versus-plan authority is
[shares/docs/current_implementation_status.md](shares/docs/current_implementation_status.md);
the registered semantic definitions are
[eyes/docs/smc_semantic_specification_v1.3.md](eyes/docs/smc_semantic_specification_v1.3.md)
for the atomic layer and
[eyes/docs/canonical_semantic_foundation_v2.1.md](eyes/docs/canonical_semantic_foundation_v2.1.md)
for the Foundation projection.

## Build, Test, and Development Commands

Create the environment with `uv sync --all-extras`; there is no compile step.
`--extra test` alone prunes `torch` and `databento`, which the MBO protocol
tests need. The `data/` payload is gitignored, so a fresh worktree must link or
materialize it before `brain/tests/test_data_splits.py` and the MBO protocol
tests can pass.

- `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`
  runs daily semantic/runtime tests; the research-orchestration group is
  excluded by `pyproject.toml`.
- `.venv/bin/python -m pytest -m 'research_orchestration' -q -p no:cacheprovider`
  runs the remaining bounded study-orchestration tests, currently the Eye
  authority-scan group in `eyes/tests/test_eye_authority_scan.py`. They are excluded
  from the default loop, so run them after touching a study script or a runtime
  identity binding. `research_orchestration` is the only registered marker; the
  `historical_frozen` and `research_runner` markers were dropped on 2026-09-06
  once the retirements left them with no test.
- `.venv/bin/python -m pytest eyes/tests/test_semantic_foundation_projection.py -q -p no:cacheprovider`
  runs a focused contract file.
- `.venv/bin/python -m pytest eyes/tests -q -p no:cacheprovider` runs one
  subsystem's tests; swap in `brain/tests`, `execution/tests` or `shares/tests`.
- `git diff --check` catches whitespace errors before commit.

No formal-research runner or frozen manifest remains in the repository. Any new
formal study must be preregistered under a fresh experiment identity and its own
frozen manifest before a runner is reintroduced; the retired Phase 5/6 templates
are not a starting point. Never casually add `--force` to a registered run.

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
