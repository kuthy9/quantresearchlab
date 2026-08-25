# Current SMC Refactor Implementation Status

Status date: 2026-08-25
Runtime semantic identity: `smc_semantics_v1.2`
Canonical foundation identity: `smc_semantic_foundation_v2.0`
Canonical foundation registry identity: `ac04636919931d774309a0c306764fdf8eb53aee41df0f31d4d94e5b9125732b`

This is the current implementation-versus-plan authority. The
[Phase 2–5 completion report](phase_2_5_completion_report.md) remains the
immutable historical report for the January 2024 v1.1 diagnostic; its sample
counts, 24.04% matched-control coverage, and zero E3–E6 chain are baseline
results, not descriptions of the v1.2 implementation.
The [Phase 6–9 report](phase_6_9_completion_report.md) records the newer
engineering and validation boundary without changing any frozen artifact.
The [second-round semantic review](semantic_review_round_2.md) remains the
17-concept Definition-versus-Empirical validity authority for the historical
v1.2 review. The additive
[Canonical Semantic Foundation v2](canonical_semantic_foundation_v2.md) is the
current definition authority for geometry, generations, lifecycle,
reinteraction, ancestry, and factual outcomes. It does not rewrite the frozen
v1.2 runs or promote their empirical conclusions.
The [DOL/Belief/Temporal supplement](dol_belief_temporal_supplement.md) records
the current dependency-cluster rule, temporal/ancestry separation, FVG
first-lifecycle freeze, and the 2024-06 joint OHLCV/MBO data-role audit.
Against the repository-owned target architecture, the overall verdict remains
**partial**: the core ownership, causality, replay, and fail-closed interfaces
are present, while empirical fitting/calibration, formal Execution Research,
operational Shadow Live, rolling OOF, and sealed OOS remain incomplete.

## Current boundary

The Trading Eye is an event-sourced, replayable, auditable, deterministic
multi-timeframe market-state engine. The registered v1.2 surface remains the
atomic detector authority. Foundation v2 consumes only those admitted facts
and adds immutable geometry, generation/lifecycle, and first-reinteraction
records plus a compact current projection. A separate versioned
`StructuralOutcomeEngine` supplies factual research outcomes; those results
are not `FoundationRecord` or `MarketSnapshot` state. The implementation
reuses the existing detectors and event store; it is not a parallel Eye. The
Eye does not select a unique DOL or make a trade decision.

The checked-in model has one strict `semantic_selection` object. It selects the
atomic v1.2 registry/identity and the additive Foundation v2.0
registry/identity, then verifies that Foundation declares v1.2 as its parent.
There is no `smc_semantics_v2.0` runtime and no synthetic full-stack version.
`ContinuousSMCEngine` derives the internal projection gate from the validated
pair and freezes the existing identities into Engine, Shadow, and checkpoint
state. The current combined Engine checkpoint schema is 6; older schemas fail
closed on restore into the current Observation, Foundation, and Neutral-state
contracts. `MarketObservation` and `MarketSnapshot` schemas are 2. Shadow uses
`phase9_shadow_live_v1.3`, compact checkpoint `shadow_compact_runtime_v4`, and
component digest `phase9_shadow_component_digest_v2`. These
bindings grant no empirical, Brain, Trade Intent, execution, or live authority.

The current model schema is 3. Its observer configuration selects
`zone_protocol` and `range_auction_protocol`; canonical runtime ownership lives
in `smc_trader/zone.py` and `smc_trader/range_auction.py`. The old numbered
modules are unexported legacy pickle-lookup shims, not alternate implementations.

The compact-state migration is now implemented. `EventStore` owns
atomic history; one append-only, in-memory `FoundationRecordLedger` owns full
Foundation revisions. Hot Foundation state keeps the current logical-object
view, required indexes, counts, current-view/rolling hashes, and lifecycle
`fact_id → fingerprint` identities instead of full revision/transition DTO
histories. Snapshot fingerprint/replay transport uses the compact view. Adapter
transactions stage bounded suffix/write overlays over BAR, crossing, binding,
and index containers and reject stale sibling commits. Production emits zero
`FOUNDATION_STATE_CHANGED` events; its strict decoder remains legacy-read-only.
Cold replay and checkpoint restore rebuild and compare the hot view. The cold
ledger is still in memory, and an explicit checkpoint materializes its full
history; it is not yet an external durable ledger.

### Simplification audit and retained migration debt

The complete 288-file baseline census, static consumer links, exact-byte
duplicate check, and consolidation priorities are recorded in the
[repository file inventory](repository_file_inventory.md). It found no
unreferenced `smc_trader` runtime module, no unreferenced current config or
semantic authority, and no byte-identical tracked file pair. A zero inbound
reference is therefore treated only as triage: direct CLIs, tests, design
documents, and frozen governance artifacts are legitimate roots and leaves.

The removed `signal_empirical_admission` module had no runtime or script
consumer; its tests exercised only that module. It was nevertheless the sole,
never-integrated converter from research fit/receipt objects to production-
shape artifact payloads, so no equivalent converter is claimed to remain.
Research-side closed admission receipts remain owned by
`run_phase7_empirical_pipeline.py`, and production artifact validation/loading
remains owned by `signal_policy.py`. A future rolling-OOF admission must add one
explicitly governed conversion path rather than silently reviving this module.

The removed Phase-8 readiness checker and two old templates were consumed only
by their old tests, documentation, and one another; the fail-closed v2 runner
and sole run template retain the current evaluator/config/loader path. A zero
exit from `run_execution_research_v2.py` in its default validate-only mode means
only that the inert template is structurally valid. It does not override
`template_incomplete_not_authorized_to_run`, bind a non-zero ledger, resolve the
six blockers, or grant execution readiness.

Six focused admission, runner, and capacity-authority contracts pass after the
deletions; the default repository suite at the frozen simplification commit had
2,720 passes, one existing skip, and seven explicit deselections. Independent
review found no P0/P1 deletion regression. The deleted ignored Phase-9 capacity
directory had no repository consumer, but because Git never tracked its bytes,
future evidence-like ignored cleanup must record an inventory, hashes, and
recoverability before deletion.

The isolated three-config/three-CLI/three-test Neutral-B2 audit island and the
non-authoritative prompt PDF were subsequently retired as one receipt-bound
cleanup. Their exact preimage hashes and recovery command are preserved in the
[retirement receipt](../evidence/neutral_b2_retirement_receipt.md). No runtime,
current manifest, result, evidence, data, input, or output payload was deleted.

The identical Week-1 bytes at `inputs/phase9_week1_flat_v2.jsonl` and
`inputs/phase9_w1_foundation_v3_7465a04.jsonl` remain deliberately addressable:
their sidecars bind different manifest schemas and their legacy/v3 consumers
have not been migrated to one content-addressed identity. This is storage
duplication, not a duplicate semantic authority; neither path may be removed
independently yet.

The compact migration closes revision-history copying, full-transition scans,
and full-container staging copies. Normal tail revisions now validate and hash
only their write set; full current-view validation remains at checkpoint,
pickle, transport, and cold replay boundaries. Publishing an immutable
projection still copies a tuple and lookup-map references proportional to the
number of current logical objects, and a non-tail revision must rebuild the
canonical current-view hash. Removing those bounded costs would require a new
snapshot representation or persistent container and is not hidden behind a
compatibility facade. Other consolidation work remains lower priority: the
five large responsibility modules may be split along existing ownership seams;
Phase 7 may separate replay/cohort/fit/validation/publication; Phase 8 may share
only common contracts/serialization; Phase 9 v2/v3 may share checkpoint/parity
plumbing; and repeated hash helpers may converge after their frozen source
identities are released. The cold ledger also remains a full in-memory revision
list. No new Phase runner or facade is justified solely to rename these
boundaries.

The direct Eye path now defaults `project_scene_graph=false`. The full
development Engine retains its existing Scene Graph/Brain composition, while
`ContinuousSMCEngine.from_config()` now routes the existing
`project_scene_graph`, `materialize_event_view`, and
`persist_state_projections` flags for explicitly graph-free or projection-free
runs. This changes no checked-in model bytes, semantic identity, Snapshot
schema, or current development-Engine output. Phase 7 empirical, Phase 8 runner,
and Phase 9 file/operational runner tests are registered as the separate
`research_runner` group; they remain executable and are not deleted, but no
longer inflate the default semantic/runtime test loop.

This is only the safe boundary closure. `MarketSnapshot` already obtains
`TimeframeState` from canonical events, but compatibility Brain and Scene Graph
code still reads typed tracker projections and bounded observation histories.
Removing those fields requires a versioned Observation/checkpoint/Shadow
migration. Likewise, Foundation still owns registered generation/lifecycle,
range, FVG-context, cluster, and retirement projections; reducing it to only
owner/generation/relation views would change the Foundation registry and cold
replay contract. Neither migration is disguised as a flag or alias here.

The Neutral projection is now the single per-clock `OpenMarketThesis` owner.
It updates `GlobalMarketContext`, builds one canonical thesis tuple, and stores
that exact tuple in `NeutralMarketState`; full Engine Brain evaluation reuses
the precomputed context after strict clock/revision/epoch/root/order validation.
Standalone Brain calls use one fallback build. Terminal routing cannot replace
the Neutral thesis tuple, and checkpoint restore requires Brain/Neutral equality.

The existing `PlaybookBrain` maintains one shadow-only Hypothesis Manager
competition set containing six mutually exclusive path hypotheses:
`continuation`, `deeper_retracement`, `reversal`, `balance`,
`failed_breakout`, and `residual_unknown`. Exact-source evidence updates
preregistered log weights once within one shared horizon, terminal paths
receive zero weight, and remaining paths renormalize. `residual_unknown` makes
the registered alternatives exhaustive. Each update yields an exact
current-clock record; `MarketBelief` does not claim a complete historical
evidence journal. Only IDs resolvable to exact `MarketEvent` objects in the
current Observation are admitted; unresolved context/entity identities fail
closed. These values are explicitly
`normalized_development_weights_not_calibrated_posterior`: the configured
increments are not fitted likelihoods, so this is not a validated Bayesian
posterior and has no trading authority. The Bayesian Belief Updater is an
executable, bounded log-odds accounting API; its name does not imply empirical
calibration.

The published runtime configuration is deliberately neutral: all six path
priors are equal, both admitted evidence families have zero log-likelihood
increments, and real-bar decay is zero. Path protocol v1.2 makes
`correlation_key` a global dependency-cluster identity: multiple evidence
families in one cluster cannot become separate fitted likelihood multipliers
without one registered joint or history-conditioned contribution. The neutral
source ledger may retain both provenance records, but it performs no Bayesian
update. The pure reducer/API and the existing
`PlaybookBrain` adapter now implement an exact market-fact-to-path lifecycle
inside the current authority scope:
frozen-authority Qualified BOS/MSS facts can falsify registered paths;
production-shape BOS Acceptance/Rejection, connected lower-timeframe
Acceptance, and the legacy range-activation fact can settle registered winners;
terminal facts take precedence over a same-clock conflicting winner. The range
fact is now explicitly classified as the existing sparse Mature Balance Range,
not evidence that a generic Structural Dealing Range definition is valid.
Local `EntryEpisode` invalidation never terminals a market path. Authority or
scope rollover can still replace the current competition set without a
preregistered terminal-and-archive fact for the retired scope. Current-scope
lifecycle wiring is therefore executable, but cross-scope retirement remains
open; neither fact turns the equal/zero-weight configuration into a fitted
posterior or grants action authority.

Scene-Graph `PRECEDES` remains visible as a diagnostic temporal-neighborhood
edge, but explicit causal/open-thesis and action-connectivity allowlists exclude
it. Temporal proximity alone cannot satisfy DFP/LSR/FAVR action gates. Its
current 60-minute construction window is not a preregistered model relation and
must remain diagnostic until a separate temporal study freezes the definition.

The Brain produces shadow-only DOL rankings. Its public target inventory honors
foundation lifecycle state where available: a registered Sweep departure can
publish the distinct rearmed Generation 2, while Acceptance, retirement, and a
formed pool without its exact live source are excluded. Sources not yet managed
by the foundation retain the compatibility inventory path. The adapter reuses
the existing obstruction views, filters hard and soft obstacles to the strict
open interval between current price and target, excludes the target itself by
identity/source/co-location, and normalizes candidate scores across the eligible
candidates for each direction. The result references the exact associated
path-hypothesis weight; diagnostic joint quality is path weight multiplied by
candidate weight. Neither value is a posterior. Decision, Risk, and execution
do not consume either the new path competition or DOL ranking. The separate DOL
probability protocol marginalizes across paths and retains explicit no-target
mass, but the Brain publishes no probability result while its exact fitted/
admitted model artifact is absent.

`CausalCase` and neutral `MarketEpisode` remain distinct protocols.
`CausalCase` may join independently resolved outcomes after neighbour selection;
`MarketEpisode` is outcome-free and uses a run-independent physical episode
identity. Their estimands, eligibility, schemas, and leakage guards are not
merged. They share only technical infrastructure: canonical storage and hash
checks, safe no-clobber publication, immutable vector storage, and deterministic
cosine/OOD mechanics. Neutral retrieval additionally requires the same source
contract/epoch, a strictly prior clock, and a different physical episode ID, so
overlapping replays cannot retrieve themselves.

Signal Policy assessment and a standalone Trade Intent projector API are
integrated as shadow components with separate exact artifact admission. No
fitted/admitted path-likelihood, DOL-probability, or outcome-calibration
artifact exists. The configured
`legacy_decision_risk_compat` mode is the sole runtime action authority:
TradeIntent projection is disabled, and any non-zero prefilled intent is
rejected before Decision/Risk. The newer RiskApproval/FSM contracts remain
standalone research/shadow consumers; the vertical migration is not complete.
The Phase 7 fit-readiness checker is deliberately read-only. It verifies the
frozen Phase 6 manifest/result and all 7,381 compact ledger rows, reports 13
blockers with `ready_for_offline_fit=false`, returns `artifacts_written=[]`, and
does not fit any probability artifact. The separately frozen June Week-4
temporal/branching design passes its strict identity validator without opening
market data; its executable bindings, materialization, replay, and result are
still absent and unauthorized.

The retained development executor still provides the original conservative
pending/open/exit simulation. Separately, Phase 8 implements immutable order
FSM v1.5 with command/fact separation, partial fills, cancel/replace and OCO
races, stops, targets, reconciliation, reservations, position conservation,
checkpointing, and deterministic event replay. Its independent engineering
logic review is P0/P1=0. It has no broker/network submission authority.

Phase 8 Execution Research has a causal seven-entry-method same-intent core and
fail-closed `phase8_execution_research_v1.1` protocol. Entry good-til-time is
separate from the later analysis horizon; a target reached before a pending
remainder fills cancels that remainder; and secondary realized-spread/path
censoring does not suppress an otherwise complete primary implementation-
shortfall pair. Off-grid stop/invalidation or target prices fail intent
admission. The v2 protocol requires exact pre-outcome semantic provenance for
every method price and registers wait, cancel, stop, and target variants with
explicit evaluable-or-censored rules. A fail-closed formal runner and canonical
result writer exist. Its run manifest remains
`template_incomplete_not_authorized_to_run` because no non-zero
intent/research-case ledger or minute source is bound, outputs and experiment
identity are unset, and the manifest is not frozen. No formal empirical result
exists.
Phase 9 has a `phase9_shadow_live_v1.3` no-submission parity runner with exact
execution/account evidence, immutable evidence identities, frozen instrument
mapping, fail-stop journal/failure/gateway parity, and
`NullExecutionGateway`. Its current compact runner checkpoint is
`shadow_compact_runtime_v4`, and its component digest is
`phase9_shadow_component_digest_v2`. A retained tick-normalized schema-v2,
6,900-clock June Week-1 cold-start input was materialized (6,899 real plus one
synthetic; SHA-256
`fd9e48850d1657cf369e3e617e3e8b464790e9823f48c79b01f65bfc111a46e4`).
A bounded run checkpointed at 100 clocks, resumed to 200, and matched an
independent 200-clock cold replay exactly after v1.2 removed cross-process
hash-order dependence from revised Scene-Graph edge IDs. The full 6,900-clock
run and a real-time multi-day pilot have not run.
That receipt binds the earlier path/model identities `5213b3d6…` and
`4214da19…`. A separate Foundation-enabled 200-row receipt binds the model and
Shadow bytes that preceded the compact-state/action-policy migration. Both are
historical engineering evidence for their exact source snapshots; neither is a
current-runtime parity receipt. The single detailed historical
[release-verification table](canonical_semantic_foundation_v2.md#replay-test-and-empirical-boundary)
and its [machine receipt](../evidence/phase9_foundation_v2_prefix_200_receipt.json)
record that older bounded file-parity result. A new parity claim requires a new
frozen runtime/config identity. It does not complete or rematerialize the
6,900-clock input and is not a real-time pilot.
A read-only 6,900-clock capacity preflight verifies the historical
`COMPLETED.json` binding to the checkpoint-manifest SHA-256, creates no Engine,
and replays zero clocks. Its historical lower-bound estimates are 342,420,401 bytes
for the checkpoint, 370,193,384 bytes of retained output, and 723,016,956 bytes
for peak working set. The historical/current runtime identities differ,
nonlinear consistency cost remains unresolved, and the preflight explicitly
sets `full_6900_replay_authorized=false`.
These input/checkpoint files are local and Git-ignored; the
[Phase 6–9 gate report](phase_6_9_completion_report.md#phase-9-parity-harness-implemented-pilot-pending)
records their exact paths, hashes, commands, publication semantics, and the
limits of the small portable receipt for otherwise Git-ignored local evidence.

## Historical Foundation-v2 release verification

The foundation specification contains the single detailed
[release-verification table](canonical_semantic_foundation_v2.md#replay-test-and-empirical-boundary),
including the exact June-2024 OHLCV input identity, focused regressions,
the repository-regression handoff boundary, construction/replay performance
A/B, bounded final record census, five-view atomic replay parity, and the separate
Foundation-enabled 200-clock Engine file parity as of its recorded source
snapshot. The model, action policy, compact state, and Shadow identities have
since changed, so the table and receipt are historical and cannot authorize a
current run. Those checks are engineering evidence only: they are not a
6,900-clock or real-time multi-day pilot, MBO validation, model calibration,
rolling OOF, or sealed OOS.

## Plan items 1–20

| Item | Current status | Evidence boundary |
|---|---|---|
| 1. Eye / Brain / Executor definitions | **Complete as ownership definitions** | Eye publishes facts; Brain owns hypotheses/ranking/intent; Executor owns orders and positions. Shadow/research ownership is implemented while action admission remains closed. |
| 2. Event-sourced hierarchical state | **Implemented with separate role and geometric hierarchy** | Immutable normalized and semantic events reduce into dimensioned `TimeframeState`; Foundation v2 adds an append-only Swing containment tree without changing v1.2 role depth. There is no combinatorial master enum. |
| 3. Semantic provenance | **Producer implemented; immutable-store authority hardened in round 2** | `event_time`, `known_at`, version, immutable evidence, separated source namespaces, causal ancestry, deterministic ordering, crossing terminal uniqueness, and canonical cross-links are enforced. Frozen artifacts are not rewritten. |
| 4. Preregistered semantics | **v1.2 atomic and foundation-v2 contracts implemented** | Runtime binds `registry_v1_2.yaml`/`parameters_v1_2.yaml` and the hash-bound additive `foundation_v2_0.yaml`. Reserved v1.2 aliases and `FOUNDATION_STATE_CHANGED` are non-emitted; Foundation records live in their separate cold ledger and compact projection. The legacy event decoder grants no atomic authority. |
| 5. Eye organization | **Complete within the reused codebase** | Existing normalizer/detectors feed the event store, reducers, relation/session state, and snapshot publisher; no parallel Eye stack was created. |
| 6. Parent/child rules | **Complete for authority isolation and geometric nesting** | Only parent events change parent facts; child opposition remains evidence until the parent's own invalidation. The separate geometric tree uses only time/price containment and never BOS, protected role, or future importance. |
| 7. Independent relation object | **Complete with generation lifecycle** | `RelationState` remains the deterministic classifier; Foundation v2 binds it to persistent parent/child Structure Generations so unchanged snapshots update one generation rather than create independent samples. |
| 8. Cross-timeframe Session | **Complete** | `SessionState` is reduced from the completed M1 clock and is not embedded in a timeframe. |
| 9. Competing Brain hypotheses | **Current-scope lifecycle implemented; rollover and fitted model incomplete; shadow-only** | The six-path reducer, exact terminal/winner adapter, global dependency-cluster guard, ledger, common-horizon expiry, precedence rules, and residual mass are tested. Authority/scope rollover retirement is not preregistered, and runtime still uses equal priors and zero increments/decay with no fitted/admitted likelihood artifact. |
| 10. Signal / Execution Research separation | **Complete as an engineering boundary; execution study not run** | Signal Policy/Trade Intent, the seven-entry-method evaluator core, formal v2 research runner, order FSM, and retained simulator are distinct; none turns Eye diagnostics into action authority. Provenance and variant evaluability are enforced, but no non-zero ledger is bound and no empirical result exists. |
| 11. Nested and non-nested comparisons | **Full v1.2 protocol-v3 diagnostic complete** | The frozen r2 run separates source-only ancestry from normalized-M5-bar composition. Episode counts are E1–E6 = 1,124 / 317 / 17 / 1 / 1 / 0; E3–E6 remain underpowered and no semantic/model admission follows. |
| 12. Matched controls | **Measured; diagnostic coverage remains limited** | Quiet and non-sweep controls matched 372/1,124 (33.1%) and 62/1,124 (5.5%); pseudo and forward-shift matched 0. All four families stay separate, fixed-family Holm is non-significant, and cross-pair outcome overlap keeps inference descriptive/unvalidated. |
| 13. Structural outcomes before P&L | **Canonical definition complete; empirical programs remain** | One factual `StructuralOutcomeEngine` now owns target/invalidation precedence, same-bar ambiguity, native-bar gaps, horizon censoring, MFE and MAE. Structural Leg v2 freezes its full path, close/extreme efficiency, close/wick MAE, tick amplitude, duration, and strictly-prior `ATR_at_leg_start` ancestry. Protected-Swing survival and matched first-retest effects still require fresh independently frozen studies. |
| 14. OHLCV geometry / MBO mechanism | **Phase 6 two-week study complete** | The final registered extension supports only Acceptance continuation and Displacement impact for Phase 7 evidence. Sweep/MSS are underpowered; the historical FVG first-concrete-lifecycle proxy (`fvg_retest_response`) is unsupported and is not a true first-retest estimand. No Week 3 is authorized. |
| 15. Arrow-by-arrow causal chain | **Full diagnostic executed; sparse after E2** | v3 proves E2 through strict source ancestry and E3–E5 through separately labelled exact-BAR composition; E6 has no samples. Phase 6 MBO evidence remains a distinct study and cannot fill these sparse stages. |
| 16. Experiment preregistration | **v1.2 v3 r2 frozen and executed as a development diagnostic** | The r2 manifest froze definitions, controls, outcomes, inference, ledgers, identities, input census, and no-authority flags before the complete run. OOS, fitting, inference authority, and semantic acceptance remain closed. |
| 17. Brain organization | **Interfaces and current-scope path lifecycle integrated; target input/promotion path incomplete** | Neutral state owns one canonical `OpenMarketThesis` update per clock and Brain reuses it. Hypothesis Manager, Belief Updater, terminal/winner mapping, DOL ranking, fitted-artifact-only DOL probability, Signal Policy, Trade Intent, and artifact loaders remain integrated into `PlaybookBrain`. The compatibility Brain still consumes typed `MarketObservation`/Scene Graph views, scope retirement is unregistered, and no fitted artifacts are admitted. |
| 18. Execution state machine | **Standalone FSM engineering complete (v1.5)** | Immutable commands/facts, order/position conservation, partial fill, cancel/replace, OCO, stop/target, reconciliation, checkpoint and replay contracts passed focused development review. It has not replaced the current Engine/Decision/Risk/simulator path; no broker or empirical-fill authority is claimed. |
| 19. Recommended directories | **Adapted, not mechanically copied** | `semantics/`, `experiments/`, event/state/research/Brain modules, and focused tests exist inside the reused package. |
| 20. Refactor order | **Phase 6 passed; Phase 7–9 components exist with target gates closed** | See the phase matrix below; fitted Brain promotion, complete Execution Research, operational Shadow Live, and OOS are not complete. |

## Phase matrix

| Phase | Status | What is present | What remains |
|---|---|---|---|
| 1. Auditable foundation | **Complete for the active v1.2 plus additive-v2 path** | Causal clocks, exact tick admission, hash-bound semantic identities, immutable events/store, source-kind and cross-object validation, lifecycle uniqueness, production config admission, Engine checkpoint-schema-6 restore, compact hot-state/cold-ledger replay, fingerprints, and determinism tests. Phase 9 pickles the complete runner, including the active Hypothesis Manager ledger. | The Foundation cold ledger remains in memory, and a directly queryable belief-update archive across retired scopes is not persisted; registered input journals can replay both. |
| 2. Core atomic semantics | **v1.2 producer plus additive foundation-v2 lifecycle complete** | Existing Swing/candidate/touch/penetration/Sweep/Acceptance/Raw Break/FVG/Displacement producers are unchanged. Foundation v2 adds complete Structural Leg paths, same-level rearm, level retirement, competing interaction terminals, boundary attack, and multi-bar formation ancestry. | No arbitrary time TTL or new tutorial SMC detector was added. Any empirical expiry threshold still requires a later study/version. |
| 3. Derived structure | **Foundation definitions complete; empirical value untested** | Persistent internal/external Structure Generations and transitions, Base Origin Core versus Qualified OB, Structural versus Balance Range, and Delivery Phase Generation are independently represented. MSS starts or updates a forming challenger; it cannot itself confirm an opposite regime. | Range extension remains undefined. Foundation definitions do not validate predictive value or retroactively change v1.2 artifacts. |
| 4. Timeframe and relation state | **Complete with distinct geometry and relation generations** | Timeframe/Session/Snapshot and role hierarchy remain; Foundation v2 adds geometric Swing assignments, dual range locations, and persistent cross-timeframe Relation Generations. | Outcome value is not implied by deterministic nesting or relation state and remains an empirical question. |
| 5. Signal Research | **Full registered v1.2 protocol-v3 diagnostic complete** | Frozen r2 produced a complete 36,000-clock run, six hash-bound ledgers, E1–E6 and non-nested proofs, four separate controls, adjacent deltas, exact McNemar, and fixed-family Holm. | E3–E6 and two control families remain sparse/empty; preregister an independent development/validation design rather than relaxing thresholds. No OOS window is open. |
| 6. MBO mechanism | **Complete for the registered two-week development study** | Primary week plus the preregistered underpowered extension passed engineering/data/statistical audit. Phase 7 allowlist: `acceptance_continuation`, `displacement_impact`; no Week 3. | Keep underpowered Sweep/MSS and the unsupported historical FVG first-concrete-lifecycle proxy excluded. A true first-retest estimand requires a new preregistration. This association result is not causal, OOS, model-fit, or trading authority. |
| 7. Trading Brain | **Current-scope shadow lifecycle/interfaces integrated; fitted model not admitted** | Neutral state owns the canonical OpenMarketThesis tuple and Brain reuses it. Exact facts map to per-path falsification/winners; dependency guards, DOL ranking, fitted-artifact-only no-target projection, Signal Policy, Trade Intent, and loaders fail closed. The read-only checker verifies 7,381 Phase 6 rows and reports 13 blockers without fitting or writing. Equal priors and zero increments/decay remain neutral. | Preregister scope-rollover retirement/archive semantics; bind and execute the already-frozen June W4 design; narrow final Brain input to `MarketSnapshot + events`; fit, validate, load, and admit path/DOL/outcome artifacts. Pre-horizon per-path expiry/hazard/prior reversion also need separate definitions and fitted temporal evidence. |
| 8. Execution Research | **Standalone FSM, evaluator, and formal runner complete; study/vertical gates not passed** | Seven entry methods can be compared under one frozen intent. Evaluator v1.1 separates entry GTT from analysis end, cancels a remainder when the target resolves before its pending fill, keeps primary-pair eligibility independent of secondary censoring, and rejects off-grid stop/target prices. The v2 runner enforces provenance, variant evaluability/censoring, no-clobber outputs, and validate-only default behavior. | Resolve the six current run-template blockers: produce and bind a non-zero intent/research-case ledger and minute source, register outputs and experiment identity, freeze and execute the paired study, then connect exact risk-approved intents to the FSM. |
| 9. Shadow Live | **Deterministic parity harness implemented; current operational gate not passed** | v1.3/compact-v4/digest-v2 bind exact evidence/state identities. The 6,900-clock and Foundation 200-row receipts remain historical under their recorded pre-current bindings. The read-only capacity preflight evaluates historical evidence only and grants no run authority. | Re-freeze and rematerialize any new prefix or complete 6,900-clock rehearsal under the final source snapshot; preregister operational metrics, remove the full-week nonlinear capacity residual, then run the real-time multi-day no-order pilot. |
| Final OOS | **Not opened** | Split and sealed-holdout governance exist. | Open only after the vertical chain is stable and preregistered acceptance conditions are met. |

## Architecture-target conformance

The former non-authoritative prompt PDF was retired under the
[cleanup receipt](../evidence/neutral_b2_retirement_receipt.md). Its relevant
engineering invariants are now stated in repository-owned architecture and
semantic contracts: causal `known_at`, immutable
events, deterministic replay, parent/child isolation, Eye/Brain/Execution
ownership, shared research/production semantics, Signal/Execution separation,
and sealed-OOS discipline all have explicit tests or fail-closed boundaries.

The complete target state is **not** reached:

| Target area | Current verdict |
|---|---|
| Eye/event-state foundation | v1.2 producers remain the immutable atomic authority; foundation v2 supplies geometric Swing containment, complete Structural Leg paths, interaction/structure/relation/delivery generations, Origin/OB decomposition, dual ranges, structural FVG expiry, true first reinteraction, and a shared factual outcome engine. Arbitrary TTL and `DEALING_RANGE_EXTENDED` remain deliberately undefined. |
| Brain | Neutral state is the single OpenMarketThesis authority and Brain reuses it; current-scope lifecycle and admission interfaces are executable, but the final sole-input boundary, scope retirement archive, fitted probabilities, and non-zero intents are absent. |
| Signal Research | Full registered diagnostic executed; sparse later chains and empty controls prohibit fitting or semantic promotion. |
| Execution Research | Order FSM, seven-entry-method core, provenance/variant contracts, and formal v2 runner exist. The current run template has six unresolved bindings and no non-zero ledger or empirical result. |
| Shadow Live | The v1.3/compact-v4/digest-v2 harness is implemented, but existing 200/6,900-clock receipts bind earlier runtime/config bytes. A new frozen parity run, operational metrics, full-week capacity closure, and a real-time multi-day pilot are absent. |
| OOF/OOS/live execution | Intentionally unopened and unauthorized until the preceding vertical gates pass. |

## Remaining plan goals

The remaining work is not a request to build another Eye, Brain, or executor.
The canonical-definition gaps named above were closed by an additive,
hash-bound foundation instead of silently changing v1.2. Remaining work is
therefore empirical validation and promotion:

- Geometric nesting, Structural Leg v2, lifecycle generations, dual ranges,
  structural FVG expiry, and first reinteraction are now replayable
  definitions. Their market value is unknown; no frozen historical result is
  relabelled as validation of them.
- The legacy H1 `DEALING_RANGE_*` detector remains the historical Mature
  Balance Range. Foundation Structural Range is a separate object and location
  axis; old evidence is not renamed or reused as Structural Range evidence.
- `FVG_EXPIRED` remains non-emitted in the v1.2 atomic enum. Foundation v2 can
  terminate FVG availability only for an exact parent-structure/range/reset/
  rollover cause and never for an arbitrary bar TTL. `DEALING_RANGE_EXTENDED`
  remains undefined.
- Retained v1.2 Swing/pool Touch facts supply the source confirmation BAR rather
  than a separate pivot BAR. Foundation replay validates their available exact
  level/clock/lineage facts but does not claim to recompute contact geometry
  from that transport alone; direct and range-boundary Touch geometry remain
  independently checked. The other non-inferred transport limits are catalogued
  in the foundation specification's
  [explicit replay-seam limits](canonical_semantic_foundation_v2.md#explicit-replay-seam-limits).
- Protected-Swing survival and matched Origin-Zone first-retest remain
  registered research questions, not completed studies. Their estimands,
  pre-treatment rank/matching rules, competing-risk clocks, horizons, and
  inference must be frozen before running them. Origin Zone terminal
  provenance is already fail-closed, so no parallel detector is needed.
- Phase 7 still needs independently fitted, calibrated, validated, and
  admitted path-likelihood, DOL-probability, and target-before-invalidation
  artifacts. Authority/scope rollover retirement and a pre-horizon per-path
  expiry rule likewise need their own frozen market definitions. The W4
  temporal/branching design itself is already frozen; only a separately bound,
  still-unauthorized executable revision and its one bounded run remain at that
  construct gate.
- `MarketObservation` now embeds one `MarketSnapshot` and derives snapshot
  identity aliases from it, but still carries detector-facing/typed transition
  views consumed by Scene/Brain. Narrowing that contract and replacing the sole
  `legacy_decision_risk_compat` action path should occur only with the admitted
  TradeIntent-to-FSM vertical migration, not by creating a second Brain or
  action owner.
- Phase 8 needs a real non-zero Trade Intent/research-case ledger carrying the
  already-required method-price provenance, plus the registered minute source
  and outputs. The existing formal runner must then execute a newly frozen
  manifest before the exact `TradeIntent -> RiskApproval -> FSM` handoff.
- Phase 9 still needs a frozen operational acceptance protocol for semantic
  duplication, relation churn, evidence-belief consistency, signal expiry,
  and DOL stability. Its complete historical path also has an avoidable
  full-prefix/cold-run capacity cost; that engineering rehearsal must not be
  confused with the required real-time, multi-day, no-order pilot and durable
  feed/reconnect evidence. Rolling OOF, stability, and the sealed holdout remain
  closed until those gates pass.
- The large hash-bound research ledgers must be transferred to Git LFS or an
  immutable artifact store before an ordinary Git publication is complete.
  They are evidence and must not be deleted as cleanup.

## Playbook disposition

The three legacy typed playbooks are not physically deleted. DFP and LSR retain
their causal setup, entry-episode, invalidation, rearm, and managed-position
contracts, but are decoupled from market-path probability ownership. The Phase
7 Hypothesis Manager owns continuation, deeper retracement, reversal, balance,
failed breakout, and residual-unknown competition. FAVR remains registered but
parked/research-only because natural mature-range/value authority is still
sparse. Removing these definitions would discard useful typed episode and
replay semantics without advancing the target architecture.

## v1.2 executable semantic additions

Swing hierarchy is append-only and event-sourced. `SWING_CONFIRMED` assigns
`micro`; exact `STRUCTURAL_LEG_CREATED` endpoints promote to `internal`;
`STRUCTURE_DIRECTION_CONFIRMED` source swings promote to `structural`; and the
exact `PROTECTED_SWING_ASSIGNED` swing promotes to `external`. Each assignment
retains its `known_at` and source event IDs. The effective rank/nesting depth is
the highest causally known assignment; historical Swing events and Legs are
never rewritten with future authority.

Here the legacy `nesting_depth` field remains the frozen depth of that causal
structural-role assignment (`micro=0` through `external=3`). Foundation v2
adds a different `geometric_depth` and append-only parent assignment based only
on exact time/price containment. Neither depth is evidence that the other is
important, and no empirical threshold is inferred from the geometry.

IRL/ERL is an executable deterministic candidate classification, not a new
semantic event. Against the same-timeframe active/mature dealing range, strict
interior candidates are `irl`, boundary or outside candidates are `erl`, and
membership is `unresolved` when no active range is available. Continuous
`normalized_location_in_range` is retained where defined.

The following five `EventKind` values remain intentionally outside canonical
v1.2 emission:

- `FVG_TOUCHED`: compatibility alias; concrete partial/midpoint/full lifecycle
  events are canonical.
- `FVG_EXPIRED`: still reserved in atomic v1.2; Foundation v2 represents only
  exact structural/reset expiry causes in its independent lifecycle record.
- `ORIGIN_ZONE_TOUCHED`: compatibility alias; first intersection is represented
  by `ORIGIN_ZONE_MITIGATED`.
- `DEALING_RANGE_EXTENDED`: reserved until an extension rule is preregistered.
- `DELIVERY_PHASE_CHANGED`: compatibility projection alias in v1.2;
  Foundation v2 wraps the existing classifier in a persistent, parent-bound
  Delivery Phase Generation without emitting this atomic alias.

The canonical semantic emitter rejects a registry-bound semantic kind unless
its binding is `canonical_emitted`. Retaining an enum value or reducer import
path is not an emission claim.

## Research result boundary

The frozen v1.1 protocol-v2 diagnostic remains immutable historical evidence:
29,077 requested touches, 6,991 matches (24.04%), and zero source-linked
samples at E3–E6.

Tests marked `historical_frozen` require the then-frozen runtime checkout and
are excluded from the default suite. Against the current v1.2 working tree,
four marked checks still pass while two raw-Eye transport assertions correctly
fail closed on the changed `smc_trader/causal.py` SHA. The old binding is not
rewritten to make a current checkout impersonate that historical runtime.

Four June Foundation comparison-v1 manifests are also inert historical
preregistrations, not runnable current contracts. They were initially committed
with `frozen_before_run=true`, but their runtime bindings were subsequently
edited in place before any registered result was produced. The current files
and SHA-256 identities are:

| Historical manifest | Current file SHA-256 |
|---|---|
| `foundation_v2_2024_06_phase45_w1_development_comparison_v1.yaml` | `4d03649ceaea9a337fb8a95ed586c80e8b735f763c57cba8c635c73860d5bbc6` |
| `foundation_v2_2024_06_phase45_w2_historical_validation_comparison_v1.yaml` | `945f12fa366c982d1430ddc596556e990371a505142dcbaf10618e1efd7317f4` |
| `foundation_v2_2024_06_phase6_mbo_w1_development_comparison_v1.yaml` | `1b832e72084684735bfe95b827c83282d0285ecdc1a5bdd83643036019b68b98` |
| `foundation_v2_2024_06_phase6_mbo_w2_historical_validation_comparison_v1.yaml` | `4c6595c19bab5ed1d547edc4306e8c0647413242894ce7ffbf497df44be22584` |

Their validators now fail closed on current-runtime drift, and all four
registered result JSON files are absent. They carry no completed comparison,
mechanism, model, Phase-7, causal, profit, or trading authority. A future formal
comparison must use a new experiment identity, versioned manifest path and
`frozen_at` value after the runtime is final; these four files must not be
silently rebound again.

The two comparison wrappers, their no-input comparator, and their dedicated
tests have therefore been retired. The exact preimage hashes and recovery
command are in the
[tooling retirement receipt](../evidence/foundation_comparison_tooling_retirement_2026-08-25.md).
The generic semantic and MBO research runners remain; the historical manifests
above remain unchanged and are not redirected to those generic runners.

The separately frozen v1.2 protocol-v3 r2 development diagnostic completed.
Its [manifest](../../experiments/manifests/smc_semantics_v1_2_2024_01_phase5_diagnostic_v3_r2.yaml)
and [result](../../experiments/results/smc_semantics_v1_2_2024_01_phase5_diagnostic_v3_r2.json)
bind a complete 36,000-clock run, 30,477 real diagnostic bars, 157,802 atomic
events, and six audit ledgers. Episode counts are E1–E6 =
1,124 / 317 / 17 / 1 / 1 / 0. Quiet and non-sweep controls matched 372 and 62
episodes; pseudo and forward-shift controls matched zero. All fixed-family
Holm-adjusted p-values are 1.0. The result is diagnostic-only, carries no
inference, fit, OOS, semantic-acceptance, or trading authority, and cannot be
used to promote Phase 7.

The current checkout does not contain the result-bound 157,802-row
`event_study` JSONL, and no repository retrieval receipt points to an immutable
copy. The result summary and the other five declared ledgers remain, but local
byte-for-byte bundle verification is incomplete until that large ledger is
restored and its declared SHA-256 is checked. It must not be deleted or
recreated as disposable output.

Phase 6 is a separate completed MBO mechanism study. The frozen primary week
triggered and consumed its preregistered second-week extension. The
final registered result passed engineering/data checks and admits
only `acceptance_continuation` (matched n=39) and `displacement_impact`
(matched n=75). Sweep rejection and MSS flow shift are underpowered; the
historical FVG first-concrete-lifecycle proxy is unsupported and is not a true
first-retest estimand; the FVG pseudo-zone sensitivity is descriptive only. The
result says `further_extension_authorized=false`, so no Week 3 may be inferred.
That gate does not authorize a renamed extension. A separately scoped June
Week-4 temporal/branching diagnostic has passed input-quality preflight, and
its design manifest is frozen and identity-validator clean. The manifest keeps
execution authorization false and still lacks the bound mechanism artifact,
runtime identities, materialization, replay, and result. The generic
Signal Research library can now project deterministic typed branching episodes
without inferring missing edges or probabilities, but the frozen protocol-v3
r2 result remains a linear historical diagnostic and supplies no branching
evidence.
See the final [manifest](../../experiments/manifests/smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.yaml)
and [result](../../experiments/results/smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.json).

## Authority summary

- Eye state: deterministic v1.2 atomic facts plus the hash-bound additive
  foundation-v2 compact projection and cold in-memory revision ledger;
  production Foundation transport events are zero and all foundation
  empirical/Brain/action flags remain false.
- Phase 6 MBO: registered two-week development association passed; only two
  mechanisms are allowlisted and no further extension is authorized.
- Phase 7 path/DOL/Signal/Intent: reducer and current-scope path lifecycle
  mapping integrated; dependency clusters fail closed for fitted cross-family
  multipliers; auditable, `development_unvalidated`, `shadow_only`, with empty
  runtime DOL probabilities and zero production intents. Scope retirement is
  unregistered, and the runtime probability model remains neutral and lacks
  fitted/admitted artifacts. The read-only readiness audit inspected all 7,381
  compact Phase 6 rows and reports 13 blockers without fitting or writing.
- Legacy playbooks: DFP/LSR retained as setup generators; FAVR parked. None
  owns Phase 7 path probability.
- Phase 8: standalone order FSM v1.5 and seven-entry-method evaluator core are
  complete with focused P0/P1=0. Evaluator protocol v1.1 separates entry GTT
  from analysis end, cancels a still-pending remainder if target resolves
  first, does not let secondary censoring remove a valid primary pair, and
  rejects off-grid stop/target prices. The v2 protocol and formal runner require
  exact method provenance, register variant evaluability/censoring, and validate
  without opening data by default. The current run template still has six
  blockers; the empirical study and vertical TradeIntent-to-FSM handoff are not
  complete.
- Phase 9: the v1.3/compact-v4/digest-v2 deterministic harness is implemented.
  Existing checkpoint/resume/cold-replay and Foundation 200-clock receipts are
  exact only for their historical bindings; no current-runtime parity receipt,
  capacity-safe 6,900-clock replay, operational metrics, or real-time multi-day
  pilot exists. The full-window preflight withholds run authority.
- Decision/Risk is the sole runtime action authority in explicit legacy-compat
  mode; non-zero TradeIntent-to-FSM migration, live execution, and final OOS
  remain unopened and fail closed.

## Repository publication boundary

The 430,877,532-byte historical Phase 5 event-study ledger and the roughly
477-MiB v1.2 r2 event-study ledger are hash-bound formal evidence, not caches or
cleanup candidates. They exceed common Git-host object limits and must be
published through Git LFS or an immutable artifact store with path, SHA-256,
row count, and retrieval location preserved. This release commits the ordinary
foundation-v2 source/spec/test set; its commit ID is reported at handoff rather
than embedded in this document. Until the separate large-ledger handoff, the
evidence publication is not a complete portable bundle. Those ledgers remain
receipt-bound evidence and are not deleted as cleanup.

The tracked 2023 Eye-authority summary also points to the ignored local
`outputs/development/eye_authority_case_audit/2023_exact_contract/transmission_audit.json`
(107,989 bytes; SHA-256
`d34209116fa798e8b5932f7a39011cdd3ed722df29c90fa9903e605cba46b218`).
A small versionable [receipt](../evidence/eye_authority_case_audit_2023_receipt.json)
now preserves that path, byte count, hash, and no-authority boundary. The
ignored payload still requires an immutable retrieval location if clean-checkout
access to the full audit is required.
