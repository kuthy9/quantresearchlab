# Current SMC Refactor Implementation Status

Status date: 2026-08-23
Runtime semantic identity: `smc_semantics_v1.2`

This is the current implementation-versus-plan authority. The
[Phase 2–5 completion report](phase_2_5_completion_report.md) remains the
immutable historical report for the January 2024 v1.1 diagnostic; its sample
counts, 24.04% matched-control coverage, and zero E3–E6 chain are baseline
results, not descriptions of the v1.2 implementation.
The [Phase 6–9 report](phase_6_9_completion_report.md) records the newer
engineering and validation boundary without changing any frozen artifact.
The [second-round semantic review](semantic_review_round_2.md) is the current
17-concept Definition-versus-Empirical validity authority. It records both the
working-tree hardening completed after the frozen runs and the vNext definitions
that remain deliberately unimplemented.
The [DOL/Belief/Temporal supplement](dol_belief_temporal_supplement.md) records
the current dependency-cluster rule, temporal/ancestry separation, FVG
first-lifecycle freeze, and the 2024-06 joint OHLCV/MBO data-role audit.
Against the attached target-state prompt, the overall verdict remains
**partial**: the core ownership, causality, replay, and fail-closed interfaces
are present, while empirical fitting/calibration, formal Execution Research,
operational Shadow Live, rolling OOF, and sealed OOS remain incomplete.

## Current boundary

The Trading Eye is now an event-sourced, replayable, auditable, deterministic
multi-timeframe market-state engine within the registered v1.2 semantic
surface. It normalizes data, emits immutable atomic facts, reduces independent
timeframe states, resolves independent parent/child relations, maintains a
cross-timeframe Session object, and publishes current facts plus an event
delta. It does not select a unique DOL or make a trade decision.

The existing `PlaybookBrain` now maintains one shadow-only Hypothesis Manager
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

The Brain produces shadow-only DOL rankings. It reuses the existing
external-draw inventory and obstruction views, filters hard and soft obstacles
to the strict open interval between current price and target, excludes the
target itself by identity/source/co-location, and normalizes candidate scores
across the eligible candidates for each direction. The result references the
exact associated path-hypothesis weight; diagnostic joint quality is path
weight multiplied by candidate weight. Neither value is a posterior. Decision,
Risk, and execution do not consume either the new path competition or DOL
ranking. The separate DOL probability protocol marginalizes across paths and
retains explicit no-target mass, but the Brain publishes no probability result
while its exact fitted/admitted model artifact is absent.

Signal Policy and Trade Intent are integrated as shadow projections with
separate exact artifact admission. No fitted/admitted path-likelihood,
DOL-probability, or outcome-calibration artifact exists, so production emits
zero intents and the legacy Decision/Risk path remains unchanged.
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
admission. It does not yet freeze the exact semantic provenance of
every method price or compare registered wait, cancel, stop, and target
variants. Its manifest remains
`template_incomplete_not_authorized_to_run`. The retained Phase 8 facility is a
read-only readiness checker that opens no input ledger, writes no artifact, and
reports 12 blockers including `formal_runner_not_implemented`; no formal runner
or empirical result exists.
Phase 9 has a `phase9_shadow_live_v1.2` no-submission parity runner with exact
execution/account evidence, immutable evidence identities, frozen instrument
mapping, full registered state digests, fail-stop journal/failure/gateway
parity, and `NullExecutionGateway`. A retained tick-normalized schema-v2,
6,900-clock June Week-1 cold-start input was materialized (6,899 real plus one
synthetic; SHA-256
`fd9e48850d1657cf369e3e617e3e8b464790e9823f48c79b01f65bfc111a46e4`).
A bounded run checkpointed at 100 clocks, resumed to 200, and matched an
independent 200-clock cold replay exactly after v1.2 removed cross-process
hash-order dependence from revised Scene-Graph edge IDs. The full 6,900-clock
run and a real-time multi-day pilot have not run.
That receipt binds the earlier path/model identities `5213b3d6…` and
`4214da19…`; the present global dependency-cluster protocol intentionally
changes those identities. It is historical engineering evidence, not a current
pilot input, and must be rematerialized before any later rehearsal.
A read-only 6,900-clock capacity preflight verifies the historical
`COMPLETED.json` binding to the checkpoint-manifest SHA-256, creates no Engine,
and replays zero clocks. Its current lower-bound estimates are 342,420,401 bytes
for the checkpoint, 370,193,384 bytes of retained output, and 723,016,956 bytes
for peak working set. The historical/current runtime identities differ,
nonlinear consistency cost remains unresolved, and the preflight explicitly
sets `full_6900_replay_authorized=false`.
These input/checkpoint files are local and Git-ignored; the
[Phase 6–9 gate report](phase_6_9_completion_report.md#phase-9-parity-harness-implemented-pilot-pending)
records their exact paths, hashes, commands, publication semantics, and the
limits of the small portable receipt for otherwise Git-ignored local evidence.

## Plan items 1–20

| Item | Current status | Evidence boundary |
|---|---|---|
| 1. Eye / Brain / Executor definitions | **Complete as ownership definitions** | Eye publishes facts; Brain owns hypotheses/ranking/intent; Executor owns orders and positions. Shadow/research ownership is implemented while action admission remains closed. |
| 2. Event-sourced hierarchical state | **Implemented for the v1.2 role-depth contract; geometric hierarchy open** | Immutable normalized and semantic events reduce into dimensioned `TimeframeState`; there is no combinatorial master enum. Continuous geometric Swing containment/nesting is not the current hierarchy. |
| 3. Semantic provenance | **Producer implemented; immutable-store authority hardened in round 2** | `event_time`, `known_at`, version, immutable evidence, separated source namespaces, causal ancestry, deterministic ordering, crossing terminal uniqueness, and canonical cross-links are enforced. Frozen artifacts are not rewritten. |
| 4. Preregistered semantics | **Current v1.2 contract implemented** | Runtime binds `registry_v1_2.yaml` and `parameters_v1_2.yaml`. Five reserved/alias kinds intentionally remain non-emitted. |
| 5. Eye organization | **Complete within the reused codebase** | Existing normalizer/detectors feed the event store, reducers, relation/session state, and snapshot publisher; no parallel Eye stack was created. |
| 6. Parent/child rules | **Complete for isolation; geometric nesting open** | Only parent events change parent facts; child opposition becomes retracement/warning evidence until the parent's own invalidation. A continuous geometric parent/child Swing tree is not implemented. |
| 7. Independent relation object | **Complete** | `RelationState` owns parent/child relationships and source cutoffs instead of copying a parent into every child state. |
| 8. Cross-timeframe Session | **Complete** | `SessionState` is reduced from the completed M1 clock and is not embedded in a timeframe. |
| 9. Competing Brain hypotheses | **Current-scope lifecycle implemented; rollover and fitted model incomplete; shadow-only** | The six-path reducer, exact terminal/winner adapter, global dependency-cluster guard, ledger, common-horizon expiry, precedence rules, and residual mass are tested. Authority/scope rollover retirement is not preregistered, and runtime still uses equal priors and zero increments/decay with no fitted/admitted likelihood artifact. |
| 10. Signal / Execution Research separation | **Complete as an engineering boundary; execution-study coverage partial** | Signal Policy/Trade Intent, the seven-entry-method evaluator core, the order FSM, and retained simulator are distinct; none turns Eye diagnostics into action authority. A read-only readiness checker preserves the blocked template; exact method-price provenance, wait/cancel/stop/target variants, and the formal runner/study remain open. |
| 11. Nested and non-nested comparisons | **Full v1.2 protocol-v3 diagnostic complete** | The frozen r2 run separates source-only ancestry from normalized-M5-bar composition. Episode counts are E1–E6 = 1,124 / 317 / 17 / 1 / 1 / 0; E3–E6 remain underpowered and no semantic/model admission follows. |
| 12. Matched controls | **Measured; diagnostic coverage remains limited** | Quiet and non-sweep controls matched 372/1,124 (33.1%) and 62/1,124 (5.5%); pseudo and forward-shift matched 0. All four families stay separate, fixed-family Holm is non-significant, and cross-pair outcome overlap keeps inference descriptive/unvalidated. |
| 13. Structural outcomes before P&L | **Substantial; semantic and study gaps remain** | Direction, target/invalidation, MFE/MAE, path/time, next-structure, FVG, and half-life outcomes exist. Origin Zone terminal provenance now fails closed. Structural Leg still lacks the requested path metrics and explicit `ATR_at_leg_start`. The frozen r2 projection contains 50 repeated protected-assignment context references; current producer/reducer custody is hardened, but Protected-Swing survival and matched Origin-Zone first-retest require fresh independently frozen estimands/matching rules. |
| 14. OHLCV geometry / MBO mechanism | **Phase 6 two-week study complete** | The final registered extension supports only Acceptance continuation and Displacement impact for Phase 7 evidence. Sweep/MSS are underpowered; the historical FVG first-concrete-lifecycle proxy (`fvg_retest_response`) is unsupported and is not a true first-retest estimand. No Week 3 is authorized. |
| 15. Arrow-by-arrow causal chain | **Full diagnostic executed; sparse after E2** | v3 proves E2 through strict source ancestry and E3–E5 through separately labelled exact-BAR composition; E6 has no samples. Phase 6 MBO evidence remains a distinct study and cannot fill these sparse stages. |
| 16. Experiment preregistration | **v1.2 v3 r2 frozen and executed as a development diagnostic** | The r2 manifest froze definitions, controls, outcomes, inference, ledgers, identities, input census, and no-authority flags before the complete run. OOS, fitting, inference authority, and semantic acceptance remain closed. |
| 17. Brain organization | **Interfaces and current-scope path lifecycle integrated; target input/promotion path incomplete** | Hypothesis Manager, Belief Updater, exact runtime terminal/winner mapping, DOL ranking, fitted-artifact-only DOL probability, Signal Policy, Trade Intent, and all-or-none artifact loaders are integrated into `PlaybookBrain`. The compatibility Brain still consumes `MarketObservation`/Scene Graph, scope retirement is unregistered, and no fitted artifacts are admitted, so runtime DOL probabilities and production intents are empty. |
| 18. Execution state machine | **Standalone FSM engineering complete (v1.5)** | Immutable commands/facts, order/position conservation, partial fill, cancel/replace, OCO, stop/target, reconciliation, checkpoint and replay contracts passed focused development review. It has not replaced the current Engine/Decision/Risk/simulator path; no broker or empirical-fill authority is claimed. |
| 19. Recommended directories | **Adapted, not mechanically copied** | `semantics/`, `experiments/`, event/state/research/Brain modules, and focused tests exist inside the reused package. |
| 20. Refactor order | **Phase 6 passed; Phase 7–9 components exist with target gates closed** | See the phase matrix below; fitted Brain promotion, complete Execution Research, operational Shadow Live, and OOS are not complete. |

## Phase matrix

| Phase | Status | What is present | What remains |
|---|---|---|---|
| 1. Auditable foundation | **Complete for the active v1.2 path after round-two hardening** | Causal clocks, exact tick admission, semantic identity, immutable events/store, source validation, lifecycle uniqueness, replay, fingerprints, and determinism tests. Phase 9 pickles the complete runner, including the active Hypothesis Manager ledger. | A directly queryable append-only belief-update archive across retired scopes is not persisted; the hash-bound input journal can replay it. |
| 2. Core atomic semantics | **Producer implemented; definition review remains partial** | Swing, Leg, candidate level, touch, penetration, Sweep, Acceptance, Raw Break, FVG lifecycle, and continuous Displacement have executable v1.2 producers. Round 2 hardens tick authority, source contracts, and crossing terminal uniqueness. | Structural Leg path metrics/ATR reference, same-level rearm, multi-bar resolution ancestry, and any `FVG_EXPIRED` definition require a new frozen version. Three other non-emitted kinds remain deliberate aliases/derived state. |
| 3. Derived structure | **Mixed: core producers exist; target definitions are not complete** | BOS/Protected source chains and the protected external reducer are hardened; MSS, qualified Origin Zone, the existing Mature Balance Range projection, and deterministic Delivery Phase remain executable. | Explicit internal/external generations, base-vs-qualified Origin Zone split, Structural-vs-Balance Range split, persistent Delivery Phase, and any range-extension rule require preregistration and a new version. |
| 4. Timeframe and relation state | **Complete for the v1.2 role-depth contract** | Timeframe/Relation/Session/Snapshot, append-only Swing role hierarchy, and executable same-timeframe IRL/ERL membership. | A geometric parent/child Swing nesting tree is not implemented; outcome value is not implied by deterministic role assignment. |
| 5. Signal Research | **Full registered v1.2 protocol-v3 diagnostic complete** | Frozen r2 produced a complete 36,000-clock run, six hash-bound ledgers, E1–E6 and non-nested proofs, four separate controls, adjacent deltas, exact McNemar, and fixed-family Holm. | E3–E6 and two control families remain sparse/empty; preregister an independent development/validation design rather than relaxing thresholds. No OOS window is open. |
| 6. MBO mechanism | **Complete for the registered two-week development study** | Primary week plus the preregistered underpowered extension passed engineering/data/statistical audit. Phase 7 allowlist: `acceptance_continuation`, `displacement_impact`; no Week 3. | Keep underpowered Sweep/MSS and the unsupported historical FVG first-concrete-lifecycle proxy excluded. A true first-retest estimand requires a new preregistration. This association result is not causal, OOS, model-fit, or trading authority. |
| 7. Trading Brain | **Current-scope shadow lifecycle/interfaces integrated; fitted model not admitted** | Exact market facts map to per-path falsification and realized winners inside the current authority scope; the reducer, global dependency guard, DOL ranking, fitted-artifact-only no-target projection, Signal Policy, Trade Intent, and artifact loaders run fail closed in the existing Brain. The read-only readiness checker verifies 7,381 Phase 6 ledger rows and reports 13 blockers without fitting or writing artifacts. Equal priors and zero increments/decay remain neutral; no runtime DOL probability is published without its artifact. | Preregister scope-rollover retirement/archive semantics; bind and execute the already-frozen June W4 temporal/branching design without changing it; narrow the final Brain input to `MarketSnapshot + events`; fit, validate, load, and admit path/DOL/outcome artifacts. Pre-horizon per-path expiry/hazard/prior reversion also need separate definitions and fitted temporal evidence. |
| 8. Execution Research | **Standalone FSM and entry-method core complete; study/vertical gates not passed** | Seven entry methods can be compared under one frozen intent. Evaluator v1.1 separates entry GTT from analysis end, cancels a remainder when the target resolves before its pending fill, keeps primary-pair eligibility independent of secondary censoring, and rejects off-grid stop/target prices. Execution FSM v1.5 passed focused logic review, and a read-only checker confirms the template remains blocked without opening ledgers or writing artifacts. | Resolve all 12 blockers: freeze method-price provenance plus wait/cancel/stop/target variants and estimands, bind a non-zero intent/minute ledger and outputs, implement the formal runner, execute the paired study, then connect exact risk-approved intents to the FSM. |
| 9. Shadow Live | **Deterministic parity harness complete; operational gate not passed** | v1.2 binds exact evidence/state identities; a Git-ignored 6,900-clock input and exact 100→200 checkpoint/resume/cold replay receipt exist for the pre-supplement model snapshot. A read-only full-window capacity preflight verifies the completion/checkpoint binding but finds current identity mismatch and grants no run authority. | Rematerialize under the final source snapshot; preregister operational metrics for event/relation churn, evidence-belief consistency, signal expiry, and DOL stability; remove the full-week nonlinear capacity residual if doing that rehearsal; then run the real-time multi-day no-order pilot. |
| Final OOS | **Not opened** | Split and sealed-holdout governance exist. | Open only after the vertical chain is stable and preregistered acceptance conditions are met. |

## PDF target-state conformance

The attached [architecture target](../codex提示词.pdf) is treated as a design
specification, not as an instruction source. Its ten engineering invariants
are implemented as code or governance contracts: causal `known_at`, immutable
events, deterministic replay, parent/child isolation, Eye/Brain/Execution
ownership, shared research/production semantics, Signal/Execution separation,
and sealed-OOS discipline all have explicit tests or fail-closed boundaries.

The complete target state is **not** reached:

| Target area | Current verdict |
|---|---|
| Eye/event-state foundation | v1.2 producers are executable and round-two authority seams are hardened; geometric Swing containment, full Structural Leg path metrics, explicit MSS generation, Origin-Zone decomposition, Structural/Balance Range separation, persistent Delivery Phase, `FVG_EXPIRED`, and `DEALING_RANGE_EXTENDED` still need frozen definitions. |
| Brain | Current-scope lifecycle and admission interfaces are executable, but the final sole-input boundary, scope retirement archive, fitted probabilities, and non-zero intents are absent. |
| Signal Research | Full registered diagnostic executed; sparse later chains and empty controls prohibit fitting or semantic promotion. |
| Execution Research | Order FSM and seven-entry-method core exist; the read-only checker reports 12 blockers, while price provenance, wait/cancel/stop/target variants, formal input/result runner, and empirical study are absent. |
| Shadow Live | Deterministic file parity is demonstrated on an exact 200-clock prefix; the read-only 6,900-clock capacity estimate is not run authority, and operational metrics, full-week capacity closure, and a real-time multi-day pilot are absent. |
| OOF/OOS/live execution | Intentionally unopened and unauthorized until the preceding vertical gates pass. |

## Remaining plan goals

The remaining work is not a request to build another Eye, Brain, or executor.
It consists of the following explicit definition, evidence, and promotion
gates:

- v1.2 has a causal Swing role-depth hierarchy, but not a geometric
  parent/child Swing-containment tree. Such a tree needs a separately frozen
  pivot/leg unit, time/price containment rule, tie-break, and late-confirmation
  reparent policy before implementation.
- The current H1 range detector is the sparse two-sided Mature Balance Range
  evaluated by the 2023 scan. Its legacy `DEALING_RANGE_*` name must not be
  interpreted as a validated Structural Dealing Range. A new version must split
  the two identities before IRL/ERL, Delivery Phase, DOL, or Brain range evidence
  can be promoted.
- Structural Leg still lacks frozen path-wide efficiency/MAE fields and an
  explicit `ATR_at_leg_start` source. MSS likewise lacks an explicit internal
  generation identity. Both require versioned event-schema changes, not a
  silent field patch.
- `FVG_EXPIRED` and `DEALING_RANGE_EXTENDED` remain reserved because v1.2 has
  no registered expiry clock/threshold or range-extension transition. The
  three other non-emitted enum values are documented aliases/derived state,
  not unfinished detectors.
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
- The retained Brain compatibility facade still consumes the existing
  `MarketObservation`/Scene Graph contract in addition to the public
  `MarketSnapshot`. Narrowing that input and replacing the legacy
  Decision/Risk/simulator path should occur only with the admitted
  TradeIntent-to-FSM vertical migration, not by creating a second Brain.
- Phase 8 needs frozen semantic provenance for all method prices, explicit
  wait/cancel/stop/target variant estimands, a real non-zero Trade Intent/method
  ledger, a formal paired-study runner/result, and then the exact
  `TradeIntent -> RiskApproval -> FSM` handoff.
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

Here `nesting_depth` is the frozen depth of that causal structural-role
assignment (`micro=0` through `external=3`), not a separately inferred
geometric parent/child Swing tree. Prominence and duration remain continuous
features; v1.2 deliberately does not introduce unfitted thresholds that would
pretend to validate a geometric hierarchy.

IRL/ERL is an executable deterministic candidate classification, not a new
semantic event. Against the same-timeframe active/mature dealing range, strict
interior candidates are `irl`, boundary or outside candidates are `erl`, and
membership is `unresolved` when no active range is available. Continuous
`normalized_location_in_range` is retained where defined.

The following five `EventKind` values remain intentionally outside canonical
v1.2 emission:

- `FVG_TOUCHED`: compatibility alias; concrete partial/midpoint/full lifecycle
  events are canonical.
- `FVG_EXPIRED`: reserved until an expiry rule is preregistered.
- `ORIGIN_ZONE_TOUCHED`: compatibility alias; first intersection is represented
  by `ORIGIN_ZONE_MITIGATED`.
- `DEALING_RANGE_EXTENDED`: reserved until an extension rule is preregistered.
- `DELIVERY_PHASE_CHANGED`: compatibility projection alias; Delivery Phase is
  snapshot-derived.

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

The separately frozen v1.2 protocol-v3 r2 development diagnostic is now also
complete. Its [manifest](../../experiments/manifests/smc_semantics_v1_2_2024_01_phase5_diagnostic_v3_r2.yaml)
and [result](../../experiments/results/smc_semantics_v1_2_2024_01_phase5_diagnostic_v3_r2.json)
bind a complete 36,000-clock run, 30,477 real diagnostic bars, 157,802 atomic
events, and six audit ledgers. Episode counts are E1–E6 =
1,124 / 317 / 17 / 1 / 1 / 0. Quiet and non-sweep controls matched 372 and 62
episodes; pseudo and forward-shift controls matched zero. All fixed-family
Holm-adjusted p-values are 1.0. The result is diagnostic-only, carries no
inference, fit, OOS, semantic-acceptance, or trading authority, and cannot be
used to promote Phase 7.

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

- Eye state: deterministic market facts within v1.2.
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
  rejects off-grid stop/target prices. Its focused evaluator/readiness suite is
  40/40 green. The
  blocked-template checker is read-only and
  reports 12 blockers; the formal runner, method provenance/variants, empirical
  study, and vertical TradeIntent-to-FSM handoff are not complete.
- Phase 9: v1.2 deterministic parity harness is complete; a 200-clock
  historical checkpoint/resume/cold-replay prefix is exact, but operational
  metrics, capacity-safe 6,900-clock replay, and the real-time multi-day pilot
  are not complete. Its read-only full-window preflight verifies the completion
  marker/checkpoint SHA binding but exposes historical/current identity mismatch
  and explicitly withholds run authority.
- Decision/Risk/live/OOS: unchanged and fail closed; the final OOS window is
  unopened.

## Repository publication boundary

The 430,877,532-byte historical Phase 5 event-study ledger and the roughly
477-MiB v1.2 r2 event-study ledger are hash-bound formal evidence, not caches or
cleanup candidates. They exceed common Git-host object limits and must be
published through Git LFS or an immutable artifact store with path, SHA-256,
row count, and retrieval location preserved. Until that handoff and the
current uncommitted source/artifact set are versioned, the working directory
is not a clean ordinary-Git publication bundle.

The tracked 2023 Eye-authority summary also points to the ignored local
`outputs/development/eye_authority_case_audit/2023_exact_contract/transmission_audit.json`
(107,989 bytes; SHA-256
`d34209116fa798e8b5932f7a39011cdd3ed722df29c90fa9903e605cba46b218`).
A small versionable [receipt](../evidence/eye_authority_case_audit_2023_receipt.json)
now preserves that path, byte count, hash, and no-authority boundary. The
ignored payload still requires an immutable retrieval location if clean-checkout
access to the full audit is required.
