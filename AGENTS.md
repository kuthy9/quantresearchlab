# Repository Guidelines

## Project Structure & Module Organization

Runtime code is split into five subsystem packages. Each owns its own `core/`,
`tests/`, `configs/` and `docs/`, and the split preserves the existing Eye,
state, Brain, risk and execution ownership boundaries rather than redrawing
them:

| package | owns |
| --- | --- |
| `eyes/` | the Trading Eye — normalization, the six detectors, semantic-event emission, the event store and market-state reduction |
| `brain/` | the LLM Trading Brain behind a Sleep Controller: the aliased Eye view, the controller, the Main Brain (DeepSeek, 14-step framework), the pure state reducer, the hash-chained journal and the SLEEP ↔ ACTIVE runtime. The typed playbooks, DOL and Signal Policy were retired on 2026-09-07; the frozen six-path set and the global mode library on 2026-09-09; the kNN hypothesis engine, its research gates and the old decision/risk on 2026-09-16 |
| `risk/` | the Risk gate (added 2026-09-16, v2 on 2026-09-17): sizes or vetoes the Brain's `TradePlan` against the broker's `AccountSnapshot` and the executor's positions — reward-to-risk floor and preferred ratio, a risk fraction per thesis grade, three same-direction positions, a leverage cap, staleness, a daily stop and a drawdown halt — with `risk/configs/risk.json` (schema 2) |
| `execution/` | execution reality and MBO reconstruction, plus (2026-09-16) the order machine that acts on an approved plan through the `Broker` boundary: `SimulatedExecutor` (OHLCV matching, a virtual account) for replay and tests, `IBKRBroker` (`ib_async`, paper accounts only) for TWS; since 2026-09-17 several intents (up to three positions), the `ThesisBook` (one expression per thesis, closed after a stop, a cooldown after any stop-out), the close-beyond exit and the drawdown halt at market; the FSM retired on 2026-09-07 was rebuilt on the LLM Brain's opportunity |
| `shares/` | data access, the session clock, the scale registry, orchestration (`engine.py`) and the study projections the other three consume |
| `contract/` | every payload that crosses a subsystem boundary, one package per boundary (`market`, `execution` — now also the account and order facts, `eye`, `brain`, `decision`, `risk` — now also `TradePlan` / `RiskVerdict`, `research`) |

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
  `144f1d6c6d6246931fda6c0f0e9cbc28d8260c851e41de7fdb9132214c3ee94d`.

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
 └─ BrainRuntime            brain/core/runtime.py            SLEEP ↔ ACTIVE, one step per bar
     ├─ brain/core/eye_view.py         EyeContext: aliased objects, events, price relations
     │   └─ object_registry.py         FVG_5m_3 ↔ Eye entity id, stable per episode
     ├─ brain/core/sleep_controller.py WAKE / UPDATE / TICK from transition events
     ├─ brain/core/main_brain.py       LLMInput → DeepSeek (llm_client.py) → LLMUpdate
     │   └─ reducer.py                 BrainState_t + evidence + update → BrainState_t+1
     │       └─ opportunity_geometry.py aliases → entry / stop / target prices, R
     ├─ brain/core/position_ledger.py  the engaged check (position or working order) and the executor's view for the LLM
     └─ brain/core/journal.py          hash-chained JSONL per episode; replayable
 └─ TradingStack             execution/core/stack.py          Brain step → plan → order machine, per bar
     ├─ execution/core/plan.py         ACTIONABLE opportunity → TradePlan (aliases, Eye entity ids, geometry)
     ├─ risk/core/gate.py              RiskGate: size or veto against the AccountSnapshot
     ├─ execution/core/order_fsm.py    OrderMachine: one bracket per intent, journal `trade` records, `execution_view()` → `prior_state.execution`
     └─ execution/core/broker.py       Broker protocol; simulated_executor.SimulatedExecutor (replay) / ibkr_broker.IBKRBroker (paper)
```

`shares/core/engine.py`, the orchestration bound to the typed Brain retired on
2026-09-07, was deleted on 2026-09-16 together with the kNN Brain
(`brain/core/forecast.py` and its research gates), `execution/core/simulation.py`
and the test modules that could not be collected without them. The Eye is
built through `shares/core/eye_factory.build_eye` from `configs/model.json`;
`brain/scripts/run_llm_brain.py` drives it over a window (`--broker
none|sim|ibkr` adds the Risk gate and the order machine) and
`brain/scripts/replay_journal.py` proves a journal reproduces from the Eye
alone (a `sim` run's trade records included); `brain/scripts/summarize_run.py`
reads a journal into one `summary.json` (calls, tokens, cost, sleeps, vetoes
and their repeats, the order lifecycle, the account, timings);
`brain/scripts/audit_scales.py` prints the change points of the per-scale
facts the Brain reads over a window, Eye only. The Eye-to-Brain link on the real tape is
`brain/tests/test_eye_link_real_tape.py` (`research_orchestration`), and the
frozen week backtest of 2026-09-17 is `brain/tests/test_regression_baseline.py`
(`research_orchestration`: replays the baseline journals named in
`brain/docs/evidence/regression_baselines.json` and compares their summaries). The
design and its receipts: [brain/docs/README.md](brain/docs/README.md),
[brain/docs/specs/2026-09-16-llm-brain-design.md](brain/docs/specs/2026-09-16-llm-brain-design.md).

The Eye imports no downstream module — no `eyes/core/` module imports `brain`,
`execution`, or the orchestration half of `shares`. `shares/core/scene_graph.py`
is a study projection with no runtime owner since the engine's retirement; the
Eye runs with `project_scene_graph=False`. `execution/core/execution.py` owns
`ExecutionRealityInput` and the cost/fillability score; `contract/execution/reality.py`
owns the inert not-evaluated value beside `ExecutionObservation`;
the retired engine derived the score and the Eye only transports the result. That boundary is enforced by
`eyes/tests/test_eye_module_boundary.py`, which resolves both the intra-package
relative imports and the cross-package absolute ones.
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
`--extra test` alone prunes `databento`, which only the raw DBN readers in
`shares/core/io.py` import. The `data/` payload is gitignored, so a fresh worktree must link or
materialize it before the real-tape tests and the MBO protocol tests can pass.
`pyproject.toml` ignores iCloud's `* 2.py`-style duplicates (`--ignore-glob`).

- `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`
  runs daily semantic/runtime tests; the research-orchestration group is
  excluded by `pyproject.toml`.
- `.venv/bin/python -m pytest -m 'research_orchestration' -q -p no:cacheprovider`
  runs the minutes-long real-tape tests, currently the Eye-to-Brain link in
  `brain/tests/test_eye_link_real_tape.py` (needs `data/`). They are excluded
  from the default loop, so run them after touching the controller, `eye_view`
  or a runtime identity binding. `research_orchestration` is the only
  registered marker (re-registered 2026-09-16; the Eye authority-scan group
  that first carried it went on 2026-09-13).
- `.venv/bin/python -m pytest eyes/tests/test_semantic_foundation_projection.py -q -p no:cacheprovider`
  runs a focused contract file.
- `.venv/bin/python -m pytest eyes/tests -q -p no:cacheprovider` runs one
  subsystem's tests; swap in `brain/tests`, `execution/tests`, `risk/tests` or
  `shares/tests`.
- `.venv/bin/python -m execution.scripts.ibkr_paper_check` is the read-only
  check of the IBKR paper session (`uv sync --all-extras` installs `ib_async`);
  it never places an order. `execution.scripts.ibkr_paper_exercise` does — by
  hand, with `--i-place-paper-orders`, on the paper account only — to record
  what TWS reports for working / cancel / replace / fill / flatten; nothing
  else outside `--broker ibkr` sends an order.
- `shares/tests/test_no_wall_clock.py` keeps every `*/core` module off the
  wall clock: the bar's `known_at` is the only time a component sees.
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
legacy-read-only. The LLM Brain names objects, `brain/core/opportunity_geometry.py`
resolves them to prices, `risk/core/gate.py` sizes or vetoes the resulting
`TradePlan`, and `execution/core/order_fsm.py` places at most one bracket per
intent through a `Broker` — the simulated one in replay, the IBKR paper
adapter live. `configs/model.json` keeps `live_execution_allowed: false`; the
IBKR adapter refuses to construct otherwise and refuses any non-`DU` account.
The LLM never writes a price; the reducer refuses any opportunity whose
objects are not visible or whose geometry is incoherent.

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
