> **Partly superseded.** The Phase-7 empirical pipeline audited here
> (`run_phase7_empirical_pipeline.py`, `probability_cohorts/fit/admission.py`,
> `signal_outcome_fit.py`, `configs/phase7_*.json` and its preregistration)
> has since been removed; references to those files are historical.

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
     ├─ registered session calendar   (market_clock.py)
     ├─ scale contract                (scale_registry.py)
     └─ CausalObserver                (observation.py)     Eye entry — one
         ├─ StructureSemantics        (structure.py)
         ├─ LiquiditySemantics        (liquidity.py + observer inventory)
         ├─ DisplacementSemantics     (displacement.py, displacement_observer.py)
         ├─ ZoneSemantics             (zone.py)            ← was "Group 3"
         ├─ RangeAuctionSemantics     (range_auction.py)   ← was "Group 4"
         ├─ InteractionSemantics      (interaction.py)     ← was "Group 5" (Eye half)
         ├─ SemanticEventEmitter      (semantic_event_emitter.py)
         │                                                 one emitter + one
         │                                                 ancestry index
         ├─ EventMemory               (event_memory.py)    bounded working set
         ├─ EventStore                (event_store.py)     one history authority
         └─ MarketSnapshotPublisher   (market_state.py)
             ├─ TimeframeEventReducer → TimeframeState
             ├─ RelationResolver      → RelationState
             └─ SessionStateReducer   → SessionState
                 └─ MarketSnapshot                          one current view
 └─ ContinuousSMCEngine               (engine.py)          orchestration
     ├─ TemporalMarketSceneGraph      (scene_graph.py)     Engine-owned view
     ├─ execution-reality scoring     (execution.py)
     ├─ BrainObservationView          (brain_entry_sequence.py)
     ├─ Neutral projection            → GlobalMarketContext + OpenMarketThesis
     ├─ PlaybookBrain.update(...)     (playbooks.py)
     └─ Decision → Risk                                    execution authority
```

Optional, non-authoritative projections: `TemporalMarketSceneGraph`
(Engine-owned research/visualisation view), `eye_statistics.py`,
`visualization.py`, `shadow_*`, `*_research*`, `causal_cases.py`/`market_cases.py`.

### 1.3 Module dependency shape

`model.py` is the shared DTO hub (fan-in 47). It imports `market_state` only
under `TYPE_CHECKING` plus two deferred function-local imports, so there is no
runtime import cycle. `market_clock`, `brain_entry_sequence`, `scene_graph` and
`foundation_registry` are the next hubs.

The Eye imports no downstream module. `causal.py` and `observation.py` take the
scale contract from `scale_registry.py`, and `scene_graph.py` is imported only
by Engine-side and research consumers.

## 2. Item-by-item disposition

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | Single data entry / Eye entry / EventStore / MarketSnapshot / Brain input | **PARTIAL** | One `CausalMarketReader`, one `CausalObserver`, one `EventStore`, one `MarketSnapshot`. Brain input is still `BrainObservationView`, which forwards the whole `MarketObservation` via `__getattr__` rather than only `MarketSnapshot + events`. See items 9 and 21. |
| 2 | Do not redo closed work | — | Applied. Nothing listed as closed below was reimplemented. |
| 3 | Foundation slimming | **DONE** | `smc_trader/foundation_adapter.py` removed (commit `66d6ecb`, −20,322 lines). No runtime module constructs `FoundationProjection`, `FoundationProjectionReducer` or `FoundationRecordLedger`; their only consumers are `tests/test_semantic_foundation_projection.py`. `ContinuousSMCEngine` no longer carries `_foundation_version` / `_foundation_registry_identity`. `MarketSnapshot` has no Foundation field. No per-clock completeness scan and no full geometry/cluster recomputation remain on the hot path, because none of that code runs there. |
| 4 | `EventStore` naming, history authority | **DONE** | Zero `ImmutableEventStore` references repository-wide. `FoundationRecord → technical MarketEvent` survives only as `market_state.foundation_record_from_projection_event`, a strict read-only decoder; production emits zero `FOUNDATION_STATE_CHANGED`. |
| 5 | `Group3 → Zone`, `Group4 → RangeAuction` | **PARTIAL** | Class/module owners are `ZoneProtocol`/`CausalZoneTracker` (`zone.py`) and `RangeAuctionProtocol`/`CausalRangeAuctionTracker` (`range_auction.py`); `group3.py`/`group4.py` are unexported pickle shims. Model config uses `zone_protocol`/`range_auction_protocol`. Residual public naming: `MarketObservation` still has `group3_*`/`group4_*` transition fields, and research manifests still use `group3_protocol`/`group4_protocol` binding keys. Both are serialisation identities — see §4. |
| 6 | Split Group 5 into Eye fact vs Brain interpretation | **DONE** | `InteractionUpdate` (Eye) carries `zone_interactions`, `reacceptance_interactions`, raw `micro_break_facts`, `interaction_paths`, `milestone_transitions` and nothing else; its docstring and `validate_canonical_bindings` enforce that. `qualified`, aligned/opposed outcome, path success and setup vocabulary are computed only in `brain_entry_sequence.py`. `interaction.py` contains no qualification logic. |
| 7 | `CausalObserver` back to orchestrator | **PARTIAL** (see §3b) | `EventMemory` and execution-reality scoring were moved out; `observation.py` is 8,386 lines. It still owns semantic emission (`_record_frame_events` ≈1,330 lines, `_record_group3_events` 445, `_record_group4_events` 854, `_record_interaction_events` 107), the liquidity inventory / reference-period subsystem (≈1,470 lines), and snapshot invocation. Measurement showed pushing the emitters into the detectors would increase coupling; §3b records the numbers and the registered alternative. |
| 8 | Scene Graph off the Eye main path | **DONE for the Eye; open for the Brain** | `CausalObserver` no longer constructs, holds, updates, checkpoints or imports a `TemporalMarketSceneGraph`; `ContinuousSMCEngine` owns it and stamps the `scene_*` delta identities (§3b). The Brain still functionally depends on the graph for `GlobalMarketContext`, `build_open_market_theses`, obstruction views and `FocusState`, so migrating those onto `MarketSnapshot + RelationState` remains next step 2 — it changes Brain market interpretation and needs its own scope. |
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

## 3b. Trading Eye decomposition round (2026-08-27)

A second round acted only on the Eye. Every step was validated by replaying the
registered `2022-02` OHLCV window through the graph-free Eye and comparing a
canonical fingerprint (event-store fingerprint, per-kind event histogram,
interaction totals, per-timeframe state digests, final snapshot identity)
against the same replay on the previous commit.

### What moved

| Change | Effect |
|---|---|
| **`scale_registry.py` (new, 164 lines)** — `ScaleRole`, `StructuralScale`, `ScaleSpec`, `parse_scale_specs`, `scale_registry_id` extracted from `scene_graph.py` | `causal.py` and `observation.py` no longer import the optional Scene Graph to describe their own scales. `scene_graph.py` re-exports the names, so a historical pickle naming `smc_trader.scene_graph.ScaleSpec` still resolves to the identical class object. |
| **`event_memory.py` (new, 1,037 lines)** — `EventMemory` extracted from `observation.py` | The Eye's bounded working set (latest-per-entity, per-entity lifecycle timelines and their legal transitions, closed durations, same-clock sequence counts, synthetic runs) is now its own component. `observation.py` re-imports the name, so observer checkpoints that recorded `smc_trader.observation.EventMemory` still resolve. |
| **`execution.py`** — `ExecutionRealityInput`, `observe_execution_reality()` and `execution_not_evaluated()` moved out of `CausalObserver` | Execution-reality scoring (spread/cost/fillability/deadline) is execution-layer interpretation, not a market fact. The Eye now transports a result it does not derive, and `execution.py` no longer imports `observation.py`. |
| **Scene Graph ownership moved to `ContinuousSMCEngine`** | `CausalObserver` no longer constructs, holds, updates or checkpoints a graph, and no longer imports `scene_graph`. The Engine owns `scene_graph`/`last_scene_delta`, advances the graph in `_project_scene_graph()`, and stamps the `scene_*` delta identities onto the observation it returns. `NEUTRAL_ENGINE_CHECKPOINT_SCHEMA_VERSION` 11 → 12. |
| **`CausalObserver.mark_terminal_failure()` (new, 12 lines)** | Preserves the previous fail-closed guarantee: the reducers have already committed the clock when the Engine-side graph runs, so a graph failure must poison the observer rather than leave a half-advanced Eye alive. |
| **`EventStore` removed from the package public surface** | `smc_trader/__init__.py` no longer exports it. Nothing imported it from the package root; research and replay tools import `smc_trader.event_store` directly for read-only lineage. |

`observation.py`: 9,498 → 8,386 lines. The Eye's only remaining reference to the
Scene Graph is the `ObserverConfig.project_scene_graph` flag, which now declares
that a downstream consumer will project one — which is what requires the
materialized event view.

One layering wrinkle is retained deliberately: `observation.py` still imports
`execution.py` for `ExecutionRealityInput` and the two scoring entry points,
because `observe(update, reality)` must place a value in
`MarketObservation.execution`. The interpretation now lives entirely in
`execution.py`; removing the import as well means changing the `observe()`
contract to accept an already-scored `ExecutionObservation`, which touches
fifteen call sites across the Engine, Shadow, simulation, MBO, calibration
replay, two scripts and their tests. That is registered rather than bundled
here.

Every moved definition was checked against its `HEAD` source at the AST level
(`ast.unparse` equality after parameter substitution): `EventMemory`,
`ScaleRole`, `StructuralScale`, `ScaleSpec`, `parse_scale_specs`,
`scale_registry_id`, `ExecutionRealityInput`, the execution scorer and the inert
value are all logically identical, and no top-level definition was lost from any
touched module.

### What was **not** moved, and the measurement that decided it

The registered direction included pushing each `_record_*_events` block down
into the semantics module that owns the corresponding tracker. Measuring the
actual coupling first showed that this would **increase** coupling, so it was
not done:

| Emitter | Lines | Distinct `self.*` attributes touched |
|---|---|---|
| `_record_group4_events` | 854 | **18**, including `_candidate_level_event_ids`, `_confirmed_swing_event_ids`, `_level_touch_event_ids`, `_penetration_event_ids`, `_crossing_generation_id`, `_prior` |
| `_record_group3_events` | 445 | **9**, including `_displacement_event_ids`, `_raw_break_event_ids`, `_origin_zone_created_event_ids` |
| `_record_interaction_events` | 107 | 1 (`self.memory`) |

`CausalObserver` carries **17 shared event-ancestry index attributes**
(`_candidate_level_event_ids` ×19 uses, `_penetration_event_ids` ×12,
`_confirmed_swing_event_ids` ×12, `_displacement_event_ids` ×10,
`_protected_swing_event_ids` ×9, …). They exist because a Range-Auction
manipulation event must cite the exact swing, level and penetration events that
Structure and Liquidity produced. Emission is centralised precisely because the
canonical emitter needs the semantic registry admission gate *and* this
cross-detector ancestry index; a detector only ever sees `Candle` objects.

Moving the emitters into the detectors would therefore require exporting that
shared index — or the observer itself — into `zone.py` and `range_auction.py`.
That is the "replace a simple problem with more abstraction" case the programme
rules require rejecting.

`_record_interaction_events` is the one cleanly separable emitter, but it emits
**legacy-transport** events (`entry_path_state`, `entry_path_step` are not in the
25 `canonical_emitted` bindings) through the module-level `_event()` factory.
Moving 107 lines would have dragged that shared factory into a new home for no
authority gain, so it stays.

**Registered next step instead:** extract the canonical emitter and its
cross-detector ancestry index (`_append_semantic_atomic`, the `_bar_event_id_*`
lookups, the crossing/penetration helpers, and the 17 `_*_event_ids` maps) into
one cohesive owner. That gives the emission concern a name without handing
mutable shared state to six detectors. It touches the observer's pickle surface,
so it needs its own task and checkpoint-compatibility evidence.

### Pre-existing defect found by the validation replay

The `2022-02` replay does **not** complete, on this commit or on the previous
one. At bar 1384 (`2022-02-02 00:05` ET) the Eye raises

```text
ValueError: protected assignment cannot be known after crossing resolution
```

from `_append_crossing_resolution`. The exact clocks are:

| Field | Value |
|---|---|
| crossing | swing `166bb8eb…`, crossed at `23:59` |
| terminal | `sweep_confirmed`, `known_at` = `resolved_at` = `00:00`, `event_time` = `23:59` |
| protected assignment | `protected_swing_assigned` for the same swing, `event_time` `23:56`, **`known_at` `00:04`** |

`post_break_at = item.accepted_at or item.rejected_at` is a market-time clock
from the structure tracker, and `_append_crossing_resolution` uses it as the
terminal's `known_at`. When BOS confirmation is delayed, the terminal is
back-dated to a clock at which the observer could not yet know it, and the guard
then correctly refuses to cite a protected assignment known four minutes later.

The root cause is therefore a `known_at` assignment, not the guard.
**Fixed on 2026-08-28 — see section 3c.**

Because the window stops there, the parity evidence below is the deterministic
prefix: 1,300 bars, 16,296 events, 36 distinct event kinds.

## 3c. Emitter extraction and the `known_at` fix (2026-08-28)

### What moved

`smc_trader/semantic_event_emitter.py` (new) owns `SemanticEventEmitter`: the
`_event` `MarketEvent` factory, `_append_semantic_atomic`, all 27 emission and
resolution methods, and all 37 cross-detector ancestry-index attributes that
`CausalObserver` used to carry alongside its orchestration state. The observer
constructs one emitter and delegates to it; `observation.py` drops from 8,386
to 3,821 lines and mints no `MarketEvent` of its own.

Three moved methods now take as parameters the observer state their bodies used
to read off `self`, all of them **required** keyword-only so no caller can
silently drop them:

| Method | Parameter |
|---|---|
| `_record_frame_events` | `prior` |
| `_record_group4_events` | `prior` |
| `_append_projected_pool_resolution_event` | `range_auction_tracker` |

The emitter grew four methods the observer no longer inlines: `rebind_memory`
(a contract boundary rebuilds `EventMemory`, and the emitter must follow it
*before* `MARKET_EPOCH_RESET` is appended), `reset_contract_state` (clear every
index, *after* that append, exactly where the inline block ran),
`emit_boundary_structure_break_failed`, and `emit_reference_period_retirements`.

Every other moved body is the pre-extraction body unchanged: 28 of 28 compare
`ast.unparse`-equal against the pre-move source, the three with new parameters
after substituting `self._prior` for `prior` and `self._range_auction_tracker`
for `range_auction_tracker`.

### The `known_at` defect registered in section 3 is fixed

`_append_crossing_resolution` now separates the two clocks it was conflating:

* `resolved_at` — the market clock that decided the crossing, preserved in
  `evidence["resolved_at"]` and frozen for the generation;
* `known_at` — the observation clock at which the Eye could first derive the
  terminal, now stamped on the event.

`known_at` is a required keyword at all seven call sites, so each one states the
clock it emits at. Six sites already passed their observation clock and are
unchanged. The seventh, the BOS post-break terminal in `_record_frame_events`,
passed `post_break_at`, a structure-tracker market clock, and now passes the
frame's `event_clock`.

Two guards moved with the clock they are about to stamp:

* the protected-assignment causality check compares against `known_at`;
* a generation re-derived on a later observation is idempotent
  (`prior.known_at > known_at` is the violation, not `!=`), while the market
  clock stays frozen (`prior.evidence["resolved_at"] != resolved_at` is a
  conflict).

The distribution confirms the fix does real work: crossing terminals now carry
`known_at - event_time` lags of 2, 3, 4, 5, 15 and 60 minutes where the old code
forced them onto the market clock.

### Replay result

The `2022-02` replay advanced in three steps this round: the `known_at` fix took
it from bar 1,384 to 4,979, the structural-leg stride fix to 15,169, and the
no-trade admission below to **the complete month, all 27,360 bars**. `2022-03`
then replayed complete on the first attempt, 31,740 bars. Over the full month of
February:

| Measure | Value |
|---|---|
| bars replayed | 27,360 (`2022-02-01 00:01` to `2022-03-01 00:00` ET) |
| events in `EventStore` | 351,801 across 42 kinds |
| by origin | 191,460 legacy transport, 125,111 canonical atomic, 35,230 normalized data |
| canonical kinds emitted | 24 |
| unresolved ancestry references | 0 |
| `EventStore` fingerprint | `66c02fc238c73211…` |

`scripts/scan_eye_event_statistics.py` produces this census;
`docs/evidence/eye_event_statistics_2022_02.json` and
`docs/evidence/eye_event_statistics_2022_03.json` are its receipts.
The scan is outcome-blind and read-only: it counts events by kind, timeframe,
direction, side, session phase, lifecycle and transition reason, and rebuilds
the market-structure relation graph from the ancestry every canonical event
already cites. What the window shows:

* Emission is dominated by the level subsystem: `level_touched` (14,437) and
  `level_penetrated` (14,261) each cite their `liquidity_level_created` level
  and their `bar_completed` root, and `level_penetrated` additionally cites the
  `level_touched` that preceded it. Every penetration resolves into exactly one
  terminal — 5,772 `sweep_confirmed` plus 5,655 `acceptance_confirmed`, summing
  to 11,427 against 14,261 penetrations — so crossings neither vanish nor
  double-resolve, and the 2,834 difference is the set still unresolved at the
  window edge.
* Structure is the second spine: 7,163 `swing_confirmed` feed 5,580
  `structural_leg_created` and 1,244 `structure_direction_confirmed`; 2,389
  `raw_boundary_break` narrow to 947 `mss_core_confirmed` and 618
  `qualified_bos`, which in turn produce 195 `protected_swing_assigned`. The
  funnel is monotone at every step, which is what a qualification chain should
  look like.
* Zones and ranges are rare by construction: 573 `fvg_created` against 304 fully
  filled and 248 invalidated, 24 `origin_zone_created`, and 22
  `dealing_range_created` against 21 invalidated and 21 replaced.
* Emission concentrates on M1 (5,504 of 5,772 sweeps) with a thin
  higher-timeframe tail, matching the M1-driven observation clock. The one
  exception is `displacement_observed`, which is M5-only (1,470 of 1,470) by
  its registered protocol.
* Every canonical event's cited ancestry resolves inside the same store: zero
  unresolved parent references across 69,599 canonical events.

### Newly reachable pre-existing defect: foundation structural-leg stride

At bar 4,979 the replay raises `foundation structural leg path endpoints do not
bind its confirmed Swing pivots and clocks` from
`EventStore._validate_structural_leg_contract`. Exactly one clause of that check
fails, on a 4H leg:

| Field | Value |
|---|---|
| start swing `event_time` (pivot start) | `2022-02-03 14:00` ET |
| start swing `pivot_end` | `2022-02-03 17:00` ET |
| start pivot BAR `known_at` | `2022-02-03 17:00` ET |
| clause expects `start_clock + interval` | `2022-02-03 18:00` ET |

The 4H bucket beginning `14:00` ET is truncated to three hours by the CME daily
maintenance break at `17:00`; the next 4H bar closes at `22:00`. The producer
(`market_state.py`) builds the leg path from the **actual** candle sequence,
while the validator reconstructs it arithmetically as `clock + interval`. The
two agree everywhere the session stride is uniform and disagree at every daily
break.

This was not a regression from this round: the structural-leg emission bodies
are AST-verbatim and the 1,300-bar parity prefix is identical to the
pre-refactor baseline. The defect was simply unreachable while the `known_at`
bug stopped the replay at bar 1,384.

**Fixed 2026-08-28 on explicit request.** `_validate_structural_leg_contract`
now derives both endpoint clocks from the eligible BAR sequence it already
builds — `start_pivot_terminal` and `path_terminal` are the first real completed
bars strictly after the start and end pivot clocks — instead of computing them
as `clock + interval`. The contract's meaning is unchanged: a leg's endpoint
pivot BAR is still required to be the bar that closes that swing's pivot window,
and the check now expresses that on a calendar whose stride is not uniform. It
fails closed when no such bar exists. The `_TIMEFRAME_MINUTES` stride table in
`event_store.py` had no other reader and was removed with it.

`tests/test_event_provenance_contract.py::test_foundation_structural_leg_spans_a_truncated_session_bucket`
locks the behaviour in: the fixture takes a `session_gap_minutes` argument that
reproduces a truncated bucket, and the test asserts the endpoint stride really
is four minutes rather than one. Restoring the arithmetic clause makes that test
fail with the original replay error, and the 1,300-bar replay fingerprint is
unchanged by the fix.

### The no-trade minute, and admitting it into definitional paths

At bar 15,169 (`2022-02-15 23:50` ET) the replay raises

```text
ValueError: foundation structural leg native BAR path is not contiguous
```

from `market_state._require_contiguous_native_candles`. This one is the Eye
failing closed on a source data hole, and the chain is fully measured:

1. The February 2022 front has exactly one unscheduled hole over a registered
   trading minute: `2022-02-15 23:47` ET is missing, and `scheduled_gap_kind`
   correctly reports `None` for it. The three other sub-minute discontinuities
   in the month are the `17:00 → 18:00` maintenance breaks.
2. `iter_completed_bars` densifies it exactly as designed — one missing minute
   is within `maximum_no_trade_gap_minutes=5` on an unchanged contract — into a
   `synthetic_no_trade` bar with zero volume.
3. That candle therefore has `synthetic_minutes == 1`, so `real_completed` is
   `False`.
4. A v2 definitional path admits only `real_completed` bars, so the synthetic
   bar cannot join it.
5. The remaining path has a hole on the registered calendar, and the contiguity
   check rejects it.

Unlike the stride defect, the contiguity check itself is calendar-aware: it
walks `next_registered_native_completion` rather than adding an interval. The
conflict is inside the registered v2 Foundation contract, which asks for two
things that a no-trade minute makes mutually exclusive — a definitional path of
real bars only, and no gap on the registered calendar. Any structure spanning a
no-trade minute is therefore unrepresentable.

Three resolutions were available. On 2026-08-28 the first was chosen and
implemented: **a densified no-trade bar is admitted into a definitional path,
and the object it produces records how much of that path was synthesized.**

What changed:

* `_require_contiguous_native_candles` now admits any registered, `complete`
  native bar. It no longer rejects `synthetic_minutes != 0`, and it no longer
  demands `real_minutes == registered_minutes` — full calendar coverage is
  already proven by `observed_minutes == registered_minutes`, and `Candle`
  itself guarantees `real + synthetic == observed`. Its error text drops the
  now-inaccurate word "real".
* Both definitional-path producers — `build_swing_geometry_nodes` and
  `build_structural_legs` — select on `candle.complete` instead of
  `candle.real_completed`.
* `StructuralLegState.synthetic_path_minutes` and
  `SwingGeometryNode.synthetic_window_minutes` carry the marker: the summed
  densified minutes across the admitted bars. Zero restores the old
  real-only guarantee, so a consumer that needs it filters on zero rather than
  losing the object entirely.
* **ATR ancestry stays real-only.** `prior_candles` gained a `real_completed`
  filter so the registered clause
  `atr_reference: strictly_prior_real_completed_native_bars_before_leg_start`
  remains literally true. This is deliberate: a no-trade bar has zero true
  range and would deflate ATR. Because a densified bar's close equals the
  preceding real close, skipping it leaves the true-range chain numerically
  unchanged.
* `semantics/foundation_v2_0.yaml` records the new admission rules, so the
  registry and the code agree. That moves the canonical identity from
  `ac04636919931d77…` to `0c49da28e103f051…`.

Live bindings of the identity were rebound: `foundation_registry.py`,
`configs/model.json`, `scripts/run_phase7_empirical_pipeline.py`,
`tests/test_semantic_selection.py`, and the two status documents.

**Every pending preregistration was rebound on 2026-08-29.** All five manifests
carried zero filled artifact slots, no authorized fit, and no opened sealed
holdout — they preregister experiments that have not been run, so rebinding them
to the current contract is ordinary re-preregistration and falsifies nothing:

| Rebound artifact | Now bound to |
|---|---|
| `experiments/manifests/phase7_foundation_v2_empirical_preregistration.yaml` | `0c49da28e103f051…` |
| `experiments/manifests/foundation_v2_2024_06_phase45_w1_development_comparison_v1.yaml` | `0c49da28e103f051…` |
| `experiments/manifests/foundation_v2_2024_06_phase45_w2_historical_validation_comparison_v1.yaml` | `0c49da28e103f051…` |
| `experiments/manifests/foundation_v2_2024_06_phase6_mbo_w1_development_comparison_v1.yaml` | `0c49da28e103f051…` |
| `experiments/manifests/foundation_v2_2024_06_phase6_mbo_w2_historical_validation_comparison_v1.yaml` | `0c49da28e103f051…` |
| `configs/phase7_foundation_v2_empirical.json` | `0c49da28e103f051…` |

Rebinding that config changes its bytes, so the `protocol_binding.sha256` inside
the Phase-7 preregistration was repaired in the same pass
(`0af7f448d632…` → `78d63b26e31c…`); the manifest's own hash chain stays intact.

One artifact deliberately keeps `ac04636919931d77…`:
`docs/evidence/phase9_foundation_v2_prefix_200_receipt.json`. Unlike the
manifests it records a run that actually happened — 200 rows, 200/200 parity,
fixed record and journal fingerprints — so the old identity is simply the true
answer to "what did that run execute against". Nothing validates against the
receipt, and the surrounding documentation already describes it as freezing
*then-current* bindings, so it needs no change to stay accurate.

Evidence that admission is inert wherever no bar was densified: on the
hole-free 1,300-bar prefix the event-store fingerprint, event counts and every
snapshot value are unchanged, and each per-timeframe state digest reproduces its
pre-change value exactly once `synthetic_path_minutes` is stripped from the
serialization — the digest moved because a field was added, not because a value
changed.

### One admission rule, shared by producer and contract

All three replay stops this round had the same shape: the producer builds a
definitional path, `EventStore` independently re-derives what that path should
have been, and the two implementations of "which bar may be admitted" disagreed.
Fixing one exposed the next, one bar later each time.

On 2026-08-29 the rule was converged into a single definition in `model.py`:

```python
class BarCoverage(str, Enum):
    REAL = "real"              # every minute carried price discovery
    DENSIFIED = "densified"    # bucket fully covered, some minutes synthesized
    INCOMPLETE = "incomplete"  # the bucket was never fully observed

    admits_definitional_path -> not INCOMPLETE
    admits_atr_window        -> REAL only
```

`classify_bar_coverage` decides from minute accounting alone; `candle_coverage`
and `bar_evidence_coverage` are the two thin readers that extract that
accounting from a producer-side `Candle` and from a BAR event's evidence. The
policy exists once; only the extraction differs, because the two sides genuinely
hold different types.

Five call sites now read it instead of restating it: the two definitional-path
producers, the contiguity validator, `_require_real_normalized_bar`, and both
endpoint/ATR derivations inside the structural-leg contract. The ad-hoc
`EventStore._is_densified_no_trade_bar` predicate was deleted. Producer and
contract can no longer drift apart, because there is no second copy to drift.

Replay parity after the convergence is byte-identical to the pre-convergence
run on the 1,300-bar prefix, and the full `2022-02` month still replays all
27,360 bars.

### The Eye stops deriving execution reality

`observation.py` was the last Eye module importing a downstream layer. It took
an `ExecutionRealityInput`, *scored* it through `observe_execution_reality`, and
published the result — which is exactly what `execution.py`'s own module
docstring forbids: "The Eye transports the result; it never derives it."

On 2026-08-29 the derivation moved to its caller:

* `CausalObserver.observe(update, execution: ExecutionObservation | None)` now
  transports an already-scored value and keeps only its own invariant — an
  eye-authority observer must not be handed one. When none is supplied it
  carries the inert `execution_not_evaluated()` value.
* `ContinuousSMCEngine._score_execution` does the scoring, immediately before
  the Eye is entered, so an invalid input still fails before any observer
  mutation — now more strongly, because the observer is never entered at all.
* `execution_not_evaluated()` moved to `model.py` beside the
  `ExecutionObservation` it constructs; `execution.py` re-exports it, so that
  module's public surface is unchanged.
* Thirteen modules that imported `ExecutionRealityInput` through the Eye's
  re-export now import it from `execution.py`, its real home.

Eye-authority replay never runs through the Engine — those callers drive
`CausalObserver` directly — so `_score_execution` always scores, and no engine
needs to know about that mode.

`tests/test_eye_module_boundary.py` now asserts the property permanently: none
of the eighteen Eye modules may import `decision`, `engine`, `execution`,
`execution_fsm`, `playbooks`, `risk`, `scene_graph` or `simulation`.

Two observer tests that exercised the derivation moved to the Engine, where the
behaviour now lives, rather than being deleted.

**Evidence.** The 1,300-bar Eye replay is byte-identical, and the Engine still
derives exactly what it derived before — `expected_round_trip_cost_points`
0.975, `source` `missing`, `minutes_to_deadline` 1440.

### Why a full replay takes hours

Measured, not estimated. Per-bar Eye latency grows linearly with accumulated
history, which makes a whole-month replay quadratic:

| bars | event store | mean ms | p50 | p95 | max |
|---|---|---|---|---|---|
| 0–999 | 1 | 44.4 | 35.7 | 92.5 | 141.8 |
| 2,000–2,999 | 24,667 | 90.3 | 94.6 | 175.5 | 257.0 |
| 5,000–5,999 | 63,783 | 149.8 | 136.1 | 322.0 | 510.9 |

The reader is negligible: 0.4% of the time against the observer's 99.6%.

`cProfile` at a 40k-event store attributes it to four places:

| Share | Where | Why |
|---|---|---|
| 30% | `build_structural_legs` | rebuilds every leg on every bar |
| 14% | `_require_contiguous_native_candles` | 21,930 calls per 400 bars |
| — | `registered_native_bar_bounds`, `next_registered_native_completion` | 128k and 52k calls, uncached pure functions over a frozen calendar |
| 25% | `dataclasses.replace` (799k calls), `deepcopy` (5.4M) | the cost of immutability |

Instrumenting `build_structural_legs` over 4,000 bars shows the shape plainly:

```
calls                    2,641
confirmed swings scanned   283,065   (107.2 per call)
legs actually kept         218,677   ( 82.8 per call)
candles scanned          2,198,416   (832.4 per call)
```

Every call rescans ~832 candles and rebuilds ~83 legs, though at most a couple
change per bar. The `output[-128:]` cap discards only 22.7%, so the cap is not
the waste — the **full recomputation** is. Every other detector in the Eye is
an incremental `CausalXTracker`; this one alone is not.

Two candidate fixes, measured rather than assumed:

* Memoising the two frozen-calendar functions: **12.4%** faster (115.9 → 101.5
  ms/bar) with a byte-identical event-store fingerprint, confirming purity.
* Making leg construction incremental: addresses the 30%, and is the only fix
  that changes the growth curve rather than its constant.

**Does this threaten live trading?** Not at M1 cadence: one bar per 60s against
150 ms of work is 0.25% duty cycle. The real exposures are that
`EventStore._events` is an unbounded list (only `EventMemory` is capped, at
512), that restart warm-up replays history quadratically, and that tail latency
is already 511 ms at 64k events. Sub-minute cadence or multi-symbol operation
would make it binding.

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

1. ~~**Move the scale registry out of `scene_graph.py`.**~~ Done 2026-08-27 —
   see §3b.
2. **Migrate `GlobalMarketContext` / `OpenMarketThesis` / obstruction views onto
   `MarketSnapshot + RelationState`.** This is the real content of item 8 and it
   changes Brain interpretation, so it requires its own preregistered scope and
   parity evidence.
3. **Narrow `BrainObservationView` to `MarketSnapshot + events`.** Requires
   removing `frames`, typed transition tuples and `FrameObservation` from the
   Brain contract, with a versioned Observation/checkpoint/Shadow migration.
4. ~~**Extract the canonical emitter and its cross-detector ancestry
   index.**~~ Done 2026-08-28 as `SemanticEventEmitter` — see section 3c. The
   liquidity inventory / reference-period subsystem is still on the observer and
   remains open as its own task.
5. ~~**Fix the back-dated crossing-terminal `known_at`.**~~ Done 2026-08-28 —
   see section 3c.
6. ~~**Make the foundation structural-leg contract session-aware.**~~ Done
   2026-08-28 — see section 3c.
7. ~~**Decide how a v2 definitional path crosses a no-trade minute.**~~ Decided
   and implemented 2026-08-28 — densified bars are admitted behind an explicit
   marker; see section 3c.
8. ~~**Re-preregister Phase-7 against the new foundation identity.**~~ Done
   2026-08-29 — every pending preregistration was rebound and the Phase-7 hash
   chain repaired; see section 3c.
9. **Dealing-range maturity is unreachable as specified.** Across `2022-02` and
   `2022-03` — 59,100 bars, 81 dealing ranges — not one reached `MATURE`. Every
   range went `forming → broken`: 68 by `close_beyond_frozen_range`, 11 by
   `maturity_deadline_elapsed`. `configs/primitives_range.json` requires ten
   conditions to hold simultaneously on one H1 close inside an 8–24 bar window,
   including `inside_close_fraction >= 0.80` together with
   `midpoint_crossings >= 2` and two touches on each boundary, while a single
   H1 close outside the frozen bounds terminates the candidate immediately.
   Whether the thresholds are too tight or maturity is genuinely a rare regime
   is an empirical question; either way `MATURE` is currently dead vocabulary,
   and changing it edits registered semantics, so it needs its own scope.
10. **`bos_state` transports one lifecycle value only.** Both months emit
   `bos_state` exclusively as `pending` — 33,424 events carrying no state
   distinction — while the confirmed and failed outcomes travel as separate
   `structure_break` and `structure_break_failed` kinds. Either `bos_state`
   should carry its terminal transitions or it is redundant transport.
11. **Make structural-leg construction incremental.** It is the last detector
   in the Eye that fully recomputes instead of reducing, and it is 30% of
   replay time. Section 4 carries the measurement. Behaviour must stay
   byte-identical, so it needs parity evidence of its own.
12. **`TradeIntent → RiskApproval → FSM` vertical migration.** Explicitly out of
   scope here and must be separately registered and validated.

## 6. Verification performed

Commands were run with
`env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider -p no:warnings`.

| Check | Result |
|---|---|
| Default suite baseline at `HEAD` before changes | exit code 0 (2,393 passed, 1 skipped, 216 deselected) |
| Whole suite after the 2026-08-27 Eye round (`-m ""`) | 2,609 passed, 1 skipped |
| `2022-02` Eye replay, 1,300 bars, before vs after every step | byte-identical fingerprint `63570ff885b0661847a85e3e871a84f6f0a2edfa231f5d332bfdd7617e9a2536` |
| `2022-02` full-window replay, before vs after the 2026-08-27 round | fails at the same bar (1384) with the same pre-existing error |
| `2022-02` Eye replay, 1,300 bars, after the `SemanticEventEmitter` extraction | identical fingerprint `63570ff885b0661847a85e3e871a84f6f0a2edfa231f5d332bfdd7617e9a2536` |
| Emitter method bodies vs the pre-extraction source (`ast.unparse` equality) | 28 of 28 identical; 0 definitions lost |
| Whole suite after the emitter extraction (`-m ""`) | 2,609 passed, 1 skipped |
| **Whole suite after the `known_at` fix (`-m ""`)** | **2,609 passed, 1 skipped, 0 failed** |
| `2022-02` full-window replay after the `known_at` fix | reaches bar 4,979 (was 1,384), then the pre-existing structural-leg stride defect |
| `2022-02` full-window replay after the structural-leg fix | reaches bar 15,169, then a single unscheduled source data hole |
| Unscheduled data holes over registered trading minutes in 2022-02 | exactly 1, at `2022-02-15 23:47` ET |
| `2022-02` full-window replay after no-trade admission | **completes all 27,360 bars** |
| `2022-03` full-window replay | **completes all 31,740 bars**, including the 2022-03-13 DST transition |
| Unscheduled data holes over registered trading minutes in 2022-03 | none |
| `2022-02` event statistics over the complete month | 351,801 events, 42 kinds, 0 unresolved ancestry references |
| `2022-03` event statistics over the complete month | 399,311 events, 43 kinds, 0 unresolved ancestry references |
| `2022-03` additional paths exercised | 1 `market_epoch_reset` (quarterly contract roll), 21 censored entry paths |
| `2022-02` replay after converging the admission rule | completes all 27,360 bars; 1,300-bar prefix byte-identical |
| Per-bar Eye latency (measured) | 44 ms at an empty store, 150 ms at 64k events; grows linearly with history |
| Replay cost split | reader 0.4%, observer 99.6% |
| Mutation check on this round's new tests | 9 of 11 catch a distinct defect; 2 were strictly subsumed and removed |
| Eye reverse dependencies on downstream layers | **0** of 18 modules, asserted by `tests/test_eye_module_boundary.py` |
| Eye replay parity after the execution seam moved | 1,300-bar fingerprint byte-identical; Engine still derives 0.975 |
| Calendar memoisation trial | 12.4% faster, identical fingerprint (not applied) |
| `2022-02` Eye replay, 1,300 bars, after the structural-leg contract fix | identical fingerprint `63570ff885b0661847a85e3e871a84f6f0a2edfa231f5d332bfdd7617e9a2536` |
| New session-gap regression test against the restored arithmetic clause | fails with the original replay error; passes with the fix |
| **Whole suite after the structural-leg contract fix (`-m ""`)** | **2,610 passed, 1 skipped, 0 failed** |
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

### 7b. Self-audit for the 2026-08-28 round

**Core functions and components.** The `SemanticEventEmitter` extraction moves
code without changing it: 28 of 28 moved bodies are `ast.unparse`-equal to the
pre-move source, no definition was lost, and the 1,300-bar replay fingerprint is
unchanged. The one deliberate behaviour change is the crossing-terminal
`known_at`, which is the defect section 3 registered.

**Code logic.** Two ordering hazards the extraction created were found and
closed before the suite was green. `MARKET_EPOCH_RESET` is appended after a
contract boundary rebuilds `EventMemory`, so the emitter is rebound to the new
memory *before* that append and its indexes are cleared *after* it, matching the
inline order exactly; getting this wrong appended the reset into the discarded
memory and left ten boundary tests failing. The three moved methods that lost
access to observer state take it as **required** keyword arguments rather than
defaulted ones, so a caller cannot silently emit without it.

**New bugs.** The `known_at` fix initially over-loosened the crossing-generation
conflict guard: dropping the `known_at` equality check also stopped the market
resolution clock from being compared. The guard now checks both properties
separately — `known_at` may only move forward, `evidence["resolved_at"]` may not
move at all — and the test that exercises it passes again.

**New overengineering.** One new module and one new class; no wrapper, manager,
adapter, registry or protocol. `observation.py` is 4,565 lines smaller and mints
no `MarketEvent`; the emitter never reaches back into the observer.

**Scope held.** `market_state.py`, the detectors and every `semantics/*.yaml`
registry are untouched. `event_store.py` was changed once, on explicit request,
to fix the structural-leg stride defect: the contract's meaning is unchanged and
its check now follows the session calendar instead of an arithmetic stride. The
1,300-bar replay fingerprint is identical before and after that change, and a
regression test reproduces the original failure against the old clause.

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
