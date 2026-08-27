# Trading Eye / Brain Consolidation Audit — 2026-08-26

This is a repository audit of the Eye/Brain consolidation and
overengineering-cleanup programme. It records, per registered item, whether the
work is **DONE**, **PARTIAL**, **STILL ACTIVE**, or **NO LONGER RELEVANT**, with
the exact code evidence that supports the verdict.

It changes no semantic definition, no event `known_at`/ordering rule, no
structure/liquidity/zone/range lifecycle, no Brain market interpretation, and no
Decision/Risk authority. The current implementation-versus-plan authority
remains [current_implementation_status.md](current_implementation_status.md);
this file is the item-by-item disposition that authority points at.

## 1. Current organisation

### 1.1 Authority map

| Authority | Sole owner | Notes |
|---|---|---|
| Complete atomic history | `EventStore` (`smc_trader/event_store.py`) | Named `EventStore`; immutability is a contract, not a class-name prefix. |
| Current market view | `MarketSnapshot` (`smc_trader/market_state.py`) | Schema 5. Holds `timeframe_states`, `relations`, `session`, `events_this_update`, counts and a prefix fingerprint. No Foundation slot. |
| Cross-timeframe relations | `RelationResolver` → `RelationState` | Owned by `MarketSnapshotPublisher`; child evidence never rewrites parent authority. |
| Deterministic market facts | Eye: `CausalMarketReader` → `CausalObserver` → typed semantics | See 1.2. |
| Interpretation / path / trade decision | Brain: `brain_entry_sequence.py` + `PlaybookBrain` (`playbooks.py`) | Qualification, setup, entry method, SL/TP, TradeIntent are Brain-only. |
| `OpenMarketThesis` | Neutral projection (`ContinuousSMCEngine._project_neutral`) | Built once per clock; Brain reuses the exact tuple under a capability token. Standalone Brain keeps one clearly-labelled fallback build. |
| Runtime action authority | legacy `Decision` → `Risk` (`legacy_decision_risk_compat`) | `TradeIntent` projection disabled; a non-zero intent is rejected in `engine.py` before Decision/Risk. |

### 1.2 Public pipeline

```text
Bar
 └─ CausalMarketReader                (causal.py)          data entry — one
     └─ CausalObserver                (observation.py)     Eye entry — one
         ├─ StructureSemantics        (structure.py)
         ├─ LiquiditySemantics        (liquidity.py + observer inventory)
         ├─ DisplacementSemantics     (displacement.py, displacement_observer.py)
         ├─ ZoneSemantics             (zone.py)            ← was "Group 3"
         ├─ RangeAuctionSemantics     (range_auction.py)   ← was "Group 4"
         ├─ InteractionSemantics      (interaction.py)     ← was "Group 5" (Eye half)
         ├─ EventStore                (event_store.py)     one history authority
         └─ MarketSnapshotPublisher   (market_state.py)
             ├─ TimeframeEventReducer → TimeframeState
             ├─ RelationResolver      → RelationState
             └─ SessionStateReducer   → SessionState
                 └─ MarketSnapshot                          one current view
 └─ ContinuousSMCEngine               (engine.py)          orchestration
     ├─ BrainObservationView          (brain_entry_sequence.py)
     ├─ Neutral projection            → GlobalMarketContext + OpenMarketThesis
     ├─ PlaybookBrain.update(...)     (playbooks.py)
     └─ Decision → Risk                                    execution authority
```

Optional, non-authoritative projections hanging off that path:
`TemporalMarketSceneGraph` (research/visualisation view), `eye_statistics.py`,
`visualization.py`, `shadow_*`, `*_research*`, `causal_cases.py`/`market_cases.py`.

### 1.3 Module dependency shape

`model.py` is the shared DTO hub (fan-in 47). It imports `market_state` only
under `TYPE_CHECKING` plus two deferred function-local imports, so there is no
runtime import cycle. `market_clock` (fan-in 10), `brain_entry_sequence`
(fan-in 9), `scene_graph` (fan-in 8) and `foundation_registry` (fan-in 8) are
the next hubs.

`scene_graph.py` is imported by `causal.py` and `observation.py` **only** for
the scale-registry primitives `ScaleSpec` / `parse_scale_specs` /
`scale_registry_id`, not for the graph. That is the remaining structural reason
the optional Scene Graph module still sits on the Eye's import path; see item 8.

## 2. Item-by-item disposition

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | Single data entry / Eye entry / EventStore / MarketSnapshot / Brain input | **PARTIAL** | One `CausalMarketReader`, one `CausalObserver`, one `EventStore`, one `MarketSnapshot`. Brain input is still `BrainObservationView`, which forwards the whole `MarketObservation` via `__getattr__` rather than only `MarketSnapshot + events`. See items 9 and 21. |
| 2 | Do not redo closed work | — | Applied. Nothing listed as closed below was reimplemented. |
| 3 | Foundation slimming | **DONE** | `smc_trader/foundation_adapter.py` removed (commit `66d6ecb`, −20,322 lines). No runtime module constructs `FoundationProjection`, `FoundationProjectionReducer` or `FoundationRecordLedger`; their only consumers are `tests/test_semantic_foundation_projection.py`. `ContinuousSMCEngine` no longer carries `_foundation_version` / `_foundation_registry_identity`. `MarketSnapshot` has no Foundation field. No per-clock completeness scan and no full geometry/cluster recomputation remain on the hot path, because none of that code runs there. |
| 4 | `EventStore` naming, history authority | **DONE** | Zero `ImmutableEventStore` references repository-wide. `FoundationRecord → technical MarketEvent` survives only as `market_state.foundation_record_from_projection_event`, a strict read-only decoder; production emits zero `FOUNDATION_STATE_CHANGED`. |
| 5 | `Group3 → Zone`, `Group4 → RangeAuction` | **PARTIAL** | Class/module owners are `ZoneProtocol`/`CausalZoneTracker` (`zone.py`) and `RangeAuctionProtocol`/`CausalRangeAuctionTracker` (`range_auction.py`); `group3.py`/`group4.py` are unexported pickle shims. Model config uses `zone_protocol`/`range_auction_protocol`. Residual public naming: `MarketObservation` still has `group3_*`/`group4_*` transition fields, and research manifests still use `group3_protocol`/`group4_protocol` binding keys. Both are serialisation identities — see §4. |
| 6 | Split Group 5 into Eye fact vs Brain interpretation | **DONE** | `InteractionUpdate` (Eye) carries `zone_interactions`, `reacceptance_interactions`, raw `micro_break_facts`, `interaction_paths`, `milestone_transitions` and nothing else; its docstring and `validate_canonical_bindings` enforce that. `qualified`, aligned/opposed outcome, path success and setup vocabulary are computed only in `brain_entry_sequence.py`. `interaction.py` contains no qualification logic. |
| 7 | `CausalObserver` back to orchestrator | **STILL ACTIVE** | `CausalObserver` is ~7,600 lines and still owns semantic emission (`_record_frame_events` ≈1,330 lines, `_record_group3_events` ≈445, `_record_group4_events` ≈850, `_record_interaction_events`), the whole liquidity inventory / reference-period subsystem (≈1,470 lines), execution-reality scoring (`_execution`), and snapshot invocation. It does delegate detection to `structure.py` / `liquidity.py` / `displacement.py` / `zone.py` / `range_auction.py` / `interaction.py`, and snapshot construction to `MarketSnapshotPublisher`. See §5 for the registered next step. |
| 8 | Scene Graph off the Eye main path | **PARTIAL** | Direct Eye construction defaults `project_scene_graph=false`, and as of this round `CausalObserver` no longer allocates a `TemporalMarketSceneGraph` at all when projection is disabled. Eye-only research runners are graph-free. **But** `ContinuousSMCEngine.from_config()` still defaults the flag to `True`, and the full Brain functionally depends on the graph for `GlobalMarketContext`, `build_open_market_theses`, obstruction views and `FocusState`. Flipping that default would change Brain market interpretation and is therefore out of scope for a semantics-preserving round. |
| 9 | `TrackerState` vs `TimeframeState` | **PARTIAL** | The production `MarketSnapshot` is `MarketSnapshotAuthority.ATOMIC_EVENT_REDUCER`: `TimeframeState` comes from `TimeframeEventReducer.states`, i.e. from canonical events. The frame-projection branch (`MarketSnapshotPublisher._timeframe_state`, `MarketSnapshotAuthority.FRAME_PROJECTION`) still builds `TimeframeState` from tracker `FrameObservation` and is reachable from exactly one bounded research scanner, `scripts/scan_mature_ranges.py` (`range_auction_projection_only=True`). `MarketObservation` still carries `frames` and typed transition tuples that Brain/Scene read. |
| 10 | Group lifecycle vs Foundation lifecycle | **NO LONGER RELEVANT** | Foundation lifecycle no longer executes in the runtime (item 3), so there is no second lifecycle truth to deduplicate. `semantic_lifecycle.py` remains a cold definition module with test-only consumers. |
| 11 | One canonical `OpenMarketThesis` per clock | **DONE** | `ContinuousSMCEngine._project_neutral` builds the thesis tuple once and hands `_precomputed_neutral_state` plus `_NEUTRAL_AUTHORITY_CAPABILITY` to `PlaybookBrain.update`, which validates clock/revision/epoch/root/order before reuse. The standalone build is explicitly documented as the isolated-evaluator fallback. |
| 12 | `CausalCase` / `MarketEpisode` — share infrastructure, not protocol | **DONE** | Protocols stay separate (`causal_cases.py` `entry-episode-causal-case-1.8.0`, `market_cases.py` `market-episode-input-only-1.4.0`). Shared infrastructure is `case_retrieval.py` (canonical hashing, no-clobber publication, immutable cosine matrix, OOD core) and `artifact_stream.py`. |
| 13 | No second execution authority | **DONE** | `engine.py` raises `legacy Decision/Risk compatibility mode rejects non-zero TradeIntent authority` whenever `belief.trade_intents` is non-empty. `ExecutionFSM` is constructed only inside `shadow_live.py`, behind `NullExecutionGateway`. No vertical migration was attempted. |
| 14 | Phase-runner de-engineering | **PARTIAL → addressed this round** | Two runners were broken at `HEAD` by rename residue and now execute again (§3.3). Otherwise:  `pyproject.toml` marks `research_runner` / `research_orchestration` / `historical_frozen`; the default suite selects 2,394 of 2,610 tests. No Phase 7/8/9 runner is exported from `smc_trader/__init__.py`. Runners were not deleted: Phase 7 is the sole materialise/fit/publish path, Phase 8 is bound by its runner contract and run template, Phase 9 v2 supplies the input codec shared by Phase 7/v3/Week-1. |
| 15 | Neutral-B2 stays retired | **DONE** | No `neutral_b2` code, config, CLI or test exists; only the retirement receipt and its references in `README.md` / inventory / status docs. `docs/codex提示词.pdf` is absent. |
| 16 | Documentation cleanup and authority state | **PARTIAL → addressed this round** | Broken-local-link scan: 0. Real drift found and fixed in §3. `current_implementation_status.md` remains the current authority; `repository_file_inventory.md` remains the historical-file reading rule; the four June comparison-v1 manifests remain inert, non-runnable preregistrations. |
| 17 | Semantic identity separation | **DONE** | `SMC_SEMANTIC_VERSION = "smc_semantics_v1.2"` and `FOUNDATION_VERSION = "smc_semantic_foundation_v2.0"` with `PARENT_ATOMIC_VERSION` cross-check. No `smc_semantics_v2` string exists anywhere in `smc_trader/`, `semantics/` or `configs/`. |
| 18 | Large responsibility modules | **STILL ACTIVE (deliberately deferred)** | `playbooks.py` 14,342 / `model.py` 11,204 / `observation.py` 9,489 / `scene_graph.py` 8,572 / `market_state.py` 5,155 lines. No split met all three of the registered conditions this round; splitting for line count alone was not done. |
| 19 | Hash-helper consolidation | **STILL ACTIVE (accepted debt)** | ~30 modules contain their own `_identity` / `_digest` / `_canonical_hash` / `sha256` helper. Each is bound to a frozen artifact or checkpoint identity; converging them would change artifact identities for no semantic gain. Recorded as debt. |
| 20 | Accepted debt | — | Current-view publication still copies O(current logical objects) references; a non-tail revision may rebuild the tuple/hash cursor; large modules, Phase 7/8/9 runner cleanup and hash-helper consolidation stay open. Unchanged. |
| 21 | Stop exposing legacy surfaces | **PARTIAL** | Fixed this round: the Group-5 runtime-provenance bindings no longer attest the frozen `group5.py` shim, and the current research templates no longer bind the frozen historical `primitives_entry.json` as the production interaction contract (§3.2, §3.3). Still exposed: `MarketObservation.group3_*/group4_*` fields, research-manifest `groupN_protocol` binding keys, `FrameObservation` tracker views on the Brain contract, and Scene Graph objects on the full-Engine Brain contract. |
| 22 | Audit first, then only PARTIAL / STILL ACTIVE | — | This document is that audit. Only the items marked PARTIAL or STILL ACTIVE were touched. |
| 23 | Core invariants | **HELD** | One `EventStore`, one `MarketSnapshot`, Eye = facts, Brain = interpretation, legacy Decision/Risk = execution authority. No second history/state/thesis/lifecycle/execution authority was introduced. |
| 24 | Verification | — | See §6. |
| 25 | Self-audit | — | See §7. |
| 26 | Completion standard | — | See §8. |

## 3. What changed in this round

### 3.1 Scene Graph is no longer allocated by a graph-free Eye

`CausalObserver.__init__` previously constructed a `TemporalMarketSceneGraph`
unconditionally, so every Eye-only research run carried — and checkpointed — a
second market-state container it never updated. The graph is now constructed
only when `project_scene_graph` is true, `observe()` fails closed if the flag
and the object ever disagree, and `ContinuousSMCEngine.compact_scene_graph_runtime`
fails closed when there is no graph to compact. Two `tests/test_observer.py`
assertions became the stronger `observer.scene_graph is None`.

This does not change any published fact, event, snapshot, or Brain result for a
graph-enabled run.

### 3.2 Group-5 runtime provenance no longer attests a frozen shim

The `Group 3/4/5 → Zone / RangeAuction / Interaction` rename repointed the
`runtime_group3` and `runtime_group4` code bindings to `zone.py` and
`range_auction.py` but left `runtime_group5` bound to `smc_trader/group5.py`,
which is a ~190-line legacy adapter. The actual Group-5 semantic owner,
`smc_trader/interaction.py` (2,764 lines), was therefore **not** covered by any
research runner's runtime-code hash: changing interaction semantics would not
have invalidated a frozen manifest. That fail-open binding is now closed in
`signal_research.py`, `mbo_mechanism_research.py` and
`scripts/run_eye_authority_scan.py`.

The two current templates
(`experiments/manifests/semantic_event_study_v3_template.yaml`,
`experiments/manifests/mbo_mechanism_phase6_template.yaml`) still declared the
pre-rename `group3.py`/`group4.py`/`group5.py` paths and would have failed
`identity binding runtime_group3 must bind smc_trader/zone.py` on any new frozen
run. They now declare the real owners. `semantic_event_study_v2_template.yaml`
deliberately keeps the historical numbered paths, and
`test_v2_template_preserves_historical_numbered_runtime_paths` was extended to
record `runtime_group5` as historical there too.

Already-frozen manifests and results are untouched. They continue to fail closed
against the current runtime, which is the documented and intended behaviour.

### 3.3 Two research runners were broken at `HEAD` and now run again

`scripts/run_semantic_signal_research.py` and
`scripts/run_mbo_mechanism_research.py` still read `model["observer"]["group5_protocol"]`,
but `configs/model.json` renamed that key to `interaction_protocol` and
`engine.py` explicitly rejects the old name. Both runners raised
`KeyError: 'group5_protocol'` before opening any data;
`tests/test_signal_research.py::test_research_eye_does_not_persist_redundant_state_projections`
was failing at `HEAD` (it is deselected from the default suite by its
`research_orchestration` marker, which is why it went unnoticed). The same stale
key appeared in the two model/manifest agreement checks
(`run_semantic_signal_research.py` `model_bindings`,
`mbo_mechanism_research.py` field/binding pairs). All four sites now read
`interaction_protocol`; the manifest-side binding *names* are unchanged.

The same rename left the two current templates binding the **frozen historical**
`configs/primitives_entry.json` as `group5_protocol` while production runs
`configs/primitives_interaction.json`, so the model-versus-manifest agreement
check could not pass either. Both templates and the hard-coded expected path in
`mbo_mechanism_research.load_frozen_phase6_contract` now name the production
contract. Verified end to end: every `*_protocol` binding in
`semantic_event_study_v3_template.yaml` and `mbo_mechanism_phase6_template.yaml`
now equals the corresponding `configs/model.json` observer value, and every
`runtime_*` binding equals `REQUIRED_RUNTIME_CODE_BINDINGS`.

`configs/primitives_entry.json` is deliberately kept: already-frozen manifests,
frozen case artifacts and the hash-bound `configs/data_splits.json`
authority-scan profiles bind it, and `scripts/run_eye_authority_scan.py` /
`scripts/scan_mature_ranges.py` must keep loading it to stay inside that
hash-bound historical profile. `README.md` now states that split explicitly.

### 3.4 Documentation brought back into agreement with the code

| Claim | Was | Now |
|---|---|---|
| Combined Engine checkpoint schema | 7 (and "schema-6 restore") | 11 (`NEUTRAL_ENGINE_CHECKPOINT_SCHEMA_VERSION`) |
| `MarketObservation` schema | 3 | 5 |
| `MarketSnapshot` schema | 2 | 5 |
| Shadow compact runner checkpoint | `shadow_compact_runtime_v4` | `shadow_compact_runtime_v8` |
| Shadow component digest | `phase9_shadow_component_digest_v2` | `phase9_shadow_component_digest_v3` |
| CausalCase protocol | 1.7 (contradicting 1.8 in the same file) | 1.8 |
| MarketCase protocol | 1.3 | 1.4 |
| Foundation hot state | `CanonicalFoundationAdapter` + hot `FoundationProjection` + `FoundationRecordLedger` | removed from the runtime; retained as cold definitions |

`docs/refactor/canonical_semantic_foundation_v2.md` keeps every registered
definition unchanged; only its architecture/authority section was corrected, and
the superseded hot-adapter paragraph is explicitly labelled as the historical
record of that migration. Frozen phase reports, receipts and result artifacts
were not edited.

`AGENTS.md` was also corrected: its focused-test example pointed at
`tests/test_foundation_adapter.py`, which commit `66d6ecb` deleted, and its
runtime-authority section still described the in-memory Foundation ledger as a
live authority.

### 3.5 Dead imports left by the Foundation removal

Commit `66d6ecb` removed ~20,300 lines and left four imports with no remaining
use: `PathSequenceStep` and `price_to_ticks` in `observation.py`,
`FOUNDATION_CANONICAL_IDENTITY` in `market_state.py`, and
`LiquidityInventoryItem` in `scene_graph.py`. None is re-exported — every test
that uses those names imports them from their defining module — so all four were
removed.

An AST scan of `smc_trader/` finds ten further unused imports that predate this
programme (`brain_entry_sequence.Direction`,
`causal_cases.INTERACTION_CURRENT_ARTIFACT_COLLECTION_NAMES`,
`execution_research_runner.MethodPriceSet`, `eye_statistics.ReaderUpdate`,
`market_representation.field`, `probability_admission.Iterable`,
`probability_fit.Iterable`/`defaultdict`, `range_auction.RANGE_PAIR_FUNNEL_COUNTS`,
`risk.DealingRangeLifecycle`, `shadow_outcome.field`,
`signal_policy.NO_TARGET_BEFORE_HORIZON`). They are recorded as debt rather than
swept up here, because each needs its own re-export check and none of them
affects authority or state. The `group3.py` shim's apparently unused constant
imports are deliberate: they are the legacy attribute surface the shim exists to
provide.

## 4. Deliberately not done, and why

| Candidate | Reason |
|---|---|
| Rename `MarketObservation.group3_*` / `group4_*` fields | They are part of the schema-5 dataclass, exact pickle surface and `to_primitive` transport that `causal_cases.py` and `market_cases.py` serialise into hash-bound artifacts. Renaming changes artifact identity and forces an Observation/checkpoint/Shadow migration for a naming gain only. |
| Rename research-manifest `group3_protocol` / `runtime_group3` binding keys | The key names appear inside frozen manifests and frozen result JSON as historical evidence. Item 5 permits legacy names as historical/internal migration names; changing them would churn every research-contract identity. |
| Remove `MarketSnapshotAuthority.FRAME_PROJECTION` | It is a labelled compatibility mode used by exactly one bounded research scanner, not a production authority. Removing it changes what `scripts/scan_mature_ranges.py` measures. |
| Delete `semantic_foundation.py` / `semantic_lifecycle.py` / `semantic_zones.py` | Item 3 requires the Foundation semantic definitions to be retained. Their focused tests are legitimate roots under the repository's own inventory rule. |
| Flip `ContinuousSMCEngine.from_config()` to `project_scene_graph=false` | The full Brain currently derives `GlobalMarketContext`, `OpenMarketThesis`, obstruction views and `FocusState` from the graph. Flipping the default changes Brain market interpretation, which this round must not do. |
| Split `observation.py` / `playbooks.py` / `model.py` / `scene_graph.py` / `market_state.py` | No split satisfied all three registered conditions without inventing a new module purely to hold moved code. |
| Consolidate hash helpers | Each is bound to a frozen artifact or checkpoint identity. |

## 5. Registered next steps (each needs its own task)

1. **Move the scale registry out of `scene_graph.py`.** `ScaleSpec`,
   `parse_scale_specs`, `scale_registry_id`, `ScaleRole` and `StructuralScale`
   are core registry primitives imported by `causal.py`, `observation.py` and
   every runner, and they are the only reason the optional Scene Graph module is
   on the Eye's import path. Blocking constraint: pickled `ScaleSpec` instances
   carry `smc_trader.scene_graph.ScaleSpec` as their class path, so the move
   needs a legacy class-lookup shim and a checkpoint-compatibility test.
2. **Migrate `GlobalMarketContext` / `OpenMarketThesis` / obstruction views onto
   `MarketSnapshot + RelationState`.** This is the real content of item 8 and it
   changes Brain interpretation, so it requires its own preregistered scope and
   parity evidence.
3. **Narrow `BrainObservationView` to `MarketSnapshot + events`.** Requires
   removing `frames`, typed transition tuples and `FrameObservation` from the
   Brain contract, with a versioned Observation/checkpoint/Shadow migration.
4. **Decompose `CausalObserver`** by delegating each `_record_*_events` block to
   the semantics module that already owns the corresponding tracker, and moving
   the liquidity inventory / reference-period subsystem behind `liquidity.py`.
   No new abstraction layer; delegation to existing owners only.
5. **`TradeIntent → RiskApproval → FSM` vertical migration.** Explicitly out of
   scope here and must be separately registered and validated.

## 6. Verification performed

Commands were run with
`env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -p no:warnings`.

| Check | Result |
|---|---|
| Default suite baseline at `HEAD` before changes | exit code 0 (2,393 passed, 1 skipped, 216 deselected) |
| Whole suite at `HEAD` with markers disabled (`-m ""`) | 1 failed — `test_research_eye_does_not_persist_redundant_state_projections`, `KeyError: 'group5_protocol'` |
| **Whole suite after changes (`-m ""`)** | **2,609 passed, 1 skipped, 0 failed** |
| Default suite after changes | 2,393 passed, 1 skipped, 216 deselected |
| `tests/test_observer.py` | 34 passed |
| `tests/test_phase234_atomic_reducer.py`, `test_calibration_stream_runner.py`, `test_visualization.py`, `test_neutral_market_state.py` | 264 passed |
| `tests/test_signal_research.py`, `test_signal_research_v3.py`, `test_mbo_mechanism_research.py`, `test_eye_authority_scan.py` (markers disabled) | 128 passed, including the test that failed at `HEAD` |
| `-m 'research_runner or research_orchestration'` | 215 passed, 2,395 deselected |
| `-m historical_frozen` | 1 passed, 2,609 deselected |
| `py_compile` on every changed Python file | clean |
| Unused-import AST scan on changed files | clean after §3.5 |
| `uv.lock` / `pyproject.toml` dependencies | unchanged; no dependency edit was made |
| `git diff --check` | clean |
| Broken local documentation links | 0 |
| Merge-conflict markers, `.orig`/`.rej`, broken symlinks, tracked `__pycache__` | none |

## 7. Self-audit

**Core logic.** No semantic definition, event `known_at`/ordering rule,
structure/liquidity/zone/range lifecycle, Brain market interpretation, or
Decision/Risk authority was changed. The Scene Graph change affects only runs
that had the projection disabled and therefore never read the object. The
provenance and model-key fixes change which files a *future* research contract
must attest and which config key a runner reads; neither alters what the Eye
computes.

**Authority.** No duplicate history, state, thesis, lifecycle or execution
authority remains in the runtime. The two labelled non-production seams are the
`FRAME_PROJECTION` snapshot mode (one research scanner) and the read-only
`FOUNDATION_STATE_CHANGED` decoder.

**Overengineering.** No wrapper, manager, adapter, projection, registry,
protocol or serialisation layer was added. The net change is one fewer object
constructed on the Eye path, four corrected string constants, and corrected
documentation.

**Cleanliness.** No dead code was introduced. Stale documentation was corrected
rather than deleted; frozen artifacts were not edited.

## 8. Distance to the completion standard

The target shape

```text
MarketPacket → TradingEye → canonical events → EventStore
             → MarketStateReducer → MarketSnapshot → Brain.update(events, snapshot)
```

is reached for the history and current-view halves: there is exactly one data
entry, one Eye entry, one `EventStore`, and one `MarketSnapshot`, and the
production snapshot is reduced from canonical events. It is **not** reached for
the Brain input contract: `Brain.update` still receives a view that forwards the
whole `MarketObservation` and, in the full Engine, a Scene Graph and delta. That
is the single largest remaining gap and is registered as next steps 1–3.
