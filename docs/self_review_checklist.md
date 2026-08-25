# Mandatory pre-test self-review

Use this short gate before synthetic tests, bounded replay, calibration or
execution validation. Check the current change only; do not regenerate release
governance artifacts during ordinary development.

## Causal logic

- [ ] One newly completed 1m bar is processed exactly once.
- [ ] Higher-timeframe bars update only when complete.
- [ ] No future path, later extreme, PnL or action label enters observation.
- [ ] Raw OHLC prices are admitted on the exact integer tick grid before any
  detector/reducer mutation; banker rounding or silent snapping cannot create a
  semantic price. Derived midpoints/zone statistics are not falsely required to
  lie on the raw exchange grid.
- [ ] Event identity and common clocks survive the update; per-kind lifecycle
  and formation/confirmation/invalidation clocks survive where defined.
- [ ] `event_time` describes the occurrence/source pivot, `known_at` is the
  first usable clock, and no consumer admits the event before `known_at`.
- [ ] Event payloads cannot be mutated after append and a hot/audit store never
  mixes `semantic_version` values.
- [ ] Canonical Swing/Leg/level/touch/penetration/Sweep/Acceptance/raw-break/
  BOS/MSS/FVG/Displacement events preserve atomic definitions and real source
  IDs; context tags do not silently become definition gates.
- [ ] One Crossing Generation has at most one terminal across single append,
  batch append, checkpoint restore, and replay; Sweep and Acceptance cannot both
  terminalize the same generation.
- [ ] Runtime semantic identity, registry, parameters, event store, reducer,
  experiment manifest, and result all agree; v1.1 artifacts are never replayed
  or reported as v1.2 evidence.
- [ ] Foundation-v2 records bind the exact canonical registry identity; every
  cross-object reference resolves to an earlier compatible object and every
  critical serialized source ID resolves to its registered event kind,
  timeframe, entity, and causal clock.
- [ ] The production model has one exact `semantic_selection`: atomic
  `smc_semantics_v1.2` plus Foundation
  `smc_semantic_foundation_v2.0`, both registry paths and identities, and a
  Foundation parent equal to the atomic version. Engine construction
  strict-loads the pair once; Engine/Shadow/checkpoint state freezes and
  compares the existing identities. Missing, unknown, mismatched, or
  extra-field selections and every Engine checkpoint before current schema 5
  fail closed.
- [ ] Atomic history remains exclusively in `ImmutableEventStore`; Foundation
  revision history remains exclusively in `FoundationRecordLedger`. Hot
  Foundation/lifecycle state contains current views, required fact/index
  identities, counts, and rolling hashes—not full revision/transition DTOs.
- [ ] Production emits no `FOUNDATION_STATE_CHANGED`; legacy decoding cannot
  promote that transport into atomic ancestry or timeframe state.
- [ ] Swing role assignments (`micro → internal → structural → external`)
  and foundation-v2 geometric parent assignments are separate append-only
  histories. Geometry uses only exact time/price containment; BOS, protected
  role, and future importance cannot change `geometric_depth`.
- [ ] Candidate IRL/ERL uses only the same-timeframe active/mature frozen range;
  missing/inactive range yields unresolved and no invented location.
- [ ] The legacy Group 4 range remains Mature Balance Range evidence. The
  independent foundation Structural/Balance records and
  `x_structural_range`/`x_balance_range` never transfer that evidence or share
  boundaries by default.
- [ ] Liquidity clustering preserves every source-level identity, uses the
  frozen one-tick complete-link tolerance, and appends terminal/superseding
  cluster generations when membership changes; confluence never implies
  strength in the Eye.
- [ ] `FVG_TOUCHED`, `FVG_EXPIRED`, `ORIGIN_ZONE_TOUCHED`,
  `DEALING_RANGE_EXTENDED`, and `DELIVERY_PHASE_CHANGED` remain intentionally
  non-emitted unless a later semantic registry explicitly promotes them. The
  touch/phase values are compatibility or derived aliases. Foundation FVG
  expiry accepts exact structural/reset causes only, never an arbitrary TTL;
  range extension remains undefined.
- [ ] Liquidity levels have immutable source identity, explicit lifecycle and
  owner/supersession facts; terminal interaction generations cannot reopen.
  Sweep rearm creates a new generation only after the registered real-bar
  departure, while Acceptance and retirement remain excluded from DOL/target
  views.
- [ ] Sweep and Acceptance are competing terminals of one Liquidity
  Interaction Generation and retain the complete ordered formation BAR ledger.
  Formation ancestry is not confused with a later temporal response window.
- [ ] Validation does not overstate retained v1.2 Touch ancestry: legacy
  Swing/pool Touch facts bind a confirmation BAR, not an independently supplied
  pivot BAR. Transport validates the available exact lineage and fails closed;
  it never reconstructs missing contact geometry heuristically.
- [ ] External/internal Structure Generations persist until an exact terminal;
  MSS only starts or updates a forming challenger/transition and cannot itself
  confirm the opposite regime. Confirmed, failed, and censored transitions are
  retained.
- [ ] Relation and Delivery observations update one owner-bound generation;
  repeated snapshots do not create independent starts. Reset/rollover and
  reclassification append explicit terminal reasons.
- [ ] Reset, rollover, supersession, and retirement append explicit immutable
  terminals before current pointers change; no cache eviction or snapshot
  fluctuation silently deletes a live object or generation.
- [ ] Base Origin Core is outcome-blind and separate from Qualified OB. FVG/OB
  first reinteraction freezes only information available at the first strict
  future geometric return after departure; later fill, invalidation, and
  continuation cannot rewrite it.
- [ ] All semantic studies use the shared factual `StructuralOutcomeEngine` for
  horizon, native-bar gaps, MFE/MAE, and same-bar ambiguity. A conservative
  execution projection remains a separate policy.
- [ ] The canonical registry remains the frozen 24-object vocabulary and every
  empirical/Brain/intent/execution authority flag remains false. Tutorial-only
  concepts stay experimental unless a later version passes the promotion
  process; no second Eye or unnecessary infrastructure was introduced.
- [ ] Origin Zone terminal events resolve exact created-zone and real completed
  M5 BAR parents, preserve identity/scope/frozen geometry, and agree with
  intersection or close-through failure priority; an unknown terminal identity
  fails closed.
- [ ] Public protected structure changes on its registered acceptance standard;
  a child timeframe may raise a warning but cannot rewrite parent structure.
- [ ] While an exact protected assignment is live, an opposite Direction event
  does not overwrite external state, MSS changes internal only, and an opposite
  Q-BOS/assignment fails closed until the exact identity-linked Acceptance has
  terminalized the incumbent protection.
- [ ] `TimeframeState`, `RelationState`, and the independent `SessionState`
  replay deterministically into the published `MarketSnapshot`.
- [ ] Snapshot fingerprint/replay transport uses only the compact current
  Foundation view and component identities. An explicit checkpoint/cold replay
  materializes the ledger and reconstructs the identical hot view.
- [ ] Foundation adapter transactions use isolated suffix/write overlays; stale
  siblings fail closed and never alias or mutate committed BAR/crossing/binding
  containers.
- [ ] `MarketObservation` stores one exact `MarketSnapshot` and derives its
  as-of/instrument/price/event aliases; pickle restore rejects an incompatible
  or forged snapshot type.
- [ ] Scene Graph nodes/edges, focus and unknown/ambiguity match the observation
  actually passed to the Brain.
- [ ] Neutral projection constructs one canonical `OpenMarketThesis` tuple per
  clock; full Engine Brain evaluation reuses that tuple, standalone Brain builds
  it once, and checkpoint restore rejects Brain/Neutral disagreement.
- [ ] `Belief_t` starts from `Belief_t-1`; unchanged evidence is not assimilated
  repeatedly.
- [ ] The shadow path competition uses one instrument/epoch/authority/common-
  horizon scope, retains `residual_unknown`, normalizes active weights, and
  records exact rule/source/`known_at` contributions once for the current
  update without claiming a persisted historical evidence journal.
- [ ] Path and DOL outputs remain `development_unvalidated`/`shadow_only`, are
  described as normalized development weights rather than calibrated Bayesian
  posteriors, and do not enter Decision, Risk, execution, or position state.
- [ ] Exact path invalidation/winner mapping admits only its registered
  production-shape parents, gives same-clock terminals precedence, and never
  turns a local EntryEpisode failure into a global path terminal.
- [ ] Do not call current-scope path lifecycle final while authority/scope
  rollover retirement and archive semantics remain unregistered.
- [ ] Do not call the Phase 7 probability model fitted or calibrated while the
  runtime uses equal priors and zero likelihood increments/decay or lacks the
  admitted path/DOL/outcome artifact set.
- [ ] The Phase 7 readiness checker remains read-only: it verifies all 7,381
  compact Phase 6 ledger rows, reports the current 13 blockers and
  `ready_for_offline_fit=false`, writes no artifact, and performs no fit.
- [ ] `correlation_key` is a global dependency-cluster identity; changing an
  evidence-family label cannot turn one Sweep/Displacement/MSS impulse into
  independent likelihood multipliers. Any fitted shared cluster has one
  registered joint or history-conditioned contribution.
- [ ] Equal subtraction from every active logit is tested as softmax-invariant
  and is never reported as effective decay. Non-zero prior reversion, hazard,
  or path-specific expiry requires a separately frozen temporal artifact.
- [ ] The fitted DOL probability layer marginalizes registered paths, retains
  explicit no-target mass, and never forces a candidate. With no fitted model,
  the Brain publishes rankings and an empty DOL-probability map.
- [ ] Signal Policy and Trade Intent validate separate exact admitted artifacts;
  missing path-likelihood, DOL-probability, or outcome artifacts fail closed to
  zero production intents without falling back to legacy typed probability.
- [ ] DOL obstacles lie strictly between current price and target; target-self
  identity/source/co-location is excluded and ties/replay remain deterministic.
- [ ] DFP/LSR typed sequence, episode, terminal and rearm rules are intact and
  remain setup/entry generators rather than path-probability owners.
- [ ] FAVR is parked unless a natural mature range/value has authority; no
  physical playbook deletion silently changes replay or position management.
- [ ] Planned entry can differ from current close; first pullback binds one
  frozen zone and reacceptance has departure/reclaim/hold/failure semantics.

## Decision, risk and execution

- [ ] Thesis, sequence, location, readiness, delivery and uncertainty remain
  distinct inputs to action utility.
- [ ] An unclear utility advantage yields `abstain`.
- [ ] Entry freezes structural invalidation, draw/targets, deadline and risk;
  later extrema do not rewrite them.
- [ ] Spread, cost, stale/anomalous data, fillability, deadline and target/stop
  provenance retain hard-veto authority.
- [ ] Approved entry is first eligible at the next tradable clock.
- [ ] The current one-next-bar simulator emits only `filled`,
  `pending_right_censored`, or `not_filled_or_expired_next_bar`; each simulated
  fill creates at most one position and no label is misreported as an order FSM.
- [ ] Phase 8 FSM commands are distinct from broker/venue facts; the immutable
  stream conserves working/cancelled/expired/rejected/partial/filled state,
  aggregate entry/exit reservations, and position quantity across replay.
- [ ] Submit/request event-time causality, command identity, approval/intent
  expiry, whole-contract quantity, price/risk bounds, managed stops, OCO late
  fills, cancel/replace races, and reconciliation all fail closed.
- [ ] The FSM remains broker/network-free. An engineering P0/P1=0 review is not
  described as an empirical execution-quality result.
- [ ] Do not call the standalone FSM the authoritative executor until an exact
  admitted `TradeIntent -> RiskApproval -> FSM` handoff replaces the retained
  development execution path and passes replay parity.
- [ ] In `legacy_decision_risk_compat`, TradeIntent projection is disabled and
  any non-zero prefilled intent fails before Decision/Risk; no second action
  authority runs in parallel.
- [ ] Execution Research compares only the same frozen intent under the seven
  registered entry methods; every method price is bound to exact semantic
  provenance, and invalid/stale/crossed/reset/synthetic/missing/fractional
  capacity is censored rather than called signal failure.
- [ ] Phase 8 execution-research v1.1 keeps entry GTT distinct from the later
  analysis horizon: no post-GTT entry fill is admitted, while an existing
  position may continue through stop/target observation.
- [ ] A target reached before a pending remainder fills cancels that remainder;
  no later quote or passive-volume proxy can resurrect it.
- [ ] Phase 8 stop/invalidation and target prices are on the exact frozen tick
  grid; off-grid structural exits fail closed at intent admission.
- [ ] Primary implementation-shortfall pairing depends on full fill and the
  primary metric only; secondary realized-spread/path censoring does not erase
  an otherwise eligible primary pair.
- [ ] Wait-time, cancel-rule, stop, and target variants have separately frozen
  estimands and family membership before they enter Execution Research.
- [ ] The Phase 8 formal runner validates without opening input ledgers or
  writing artifacts by default; development execution requires an explicit
  flag plus a fully frozen manifest. It consumes a non-zero frozen
  TradeIntent/research-case ledger; tests or caller-supplied prices never
  substitute for evidence.
- [ ] Each Phase 8 method price has exact pre-outcome provenance, and each
  wait/cancel/stop/target variant is either evaluable from registered inputs or
  censored by its frozen rule; missing facts are never imputed.
- [ ] The Phase 8 research manifest is complete and frozen before any formal
  run; the current incomplete template cannot authorize an empirical claim.
- [ ] Same-bar stop/target ambiguity remains adverse-first.

## Data and replay

- [ ] Source and requested window match `configs/data_splits.json`.
- [ ] `CausalCase` and input-only `MarketEpisode` retain distinct schemas,
  estimands, identities, eligibility, and leakage guards even when they share
  canonical storage/publication and cosine/OOD mechanics. Neutral lookup cannot
  retrieve the same physical episode from an overlapping replay.
- [ ] Shared artifact publication validates exact regular-file identities and
  uses atomic no-clobber writes; symlinks, duplicates, cross-split physical
  identities, and outcome fields in neutral input fail closed.
- [ ] A development or calibration-trial event study remains explicitly
  diagnostic and cannot be mislabeled OOS or used to fit a trading artifact.
- [ ] Every semantic report lists Definition validity separately from empirical,
  predictive, OOS, and action-authority status; replay/coverage is not called
  empirical validation and an unsupported hypothesis is not generalized beyond
  its exact estimand.
- [ ] Protected-Swing survival and matched Origin-Zone first-retest are not
  run until their pre-treatment rank/matching rules, clocks, competing risks,
  horizons, censoring, and inference are frozen in independent manifests.
- [ ] A protocol-v3 run uses a fully frozen manifest; the incomplete template,
  a `--max-bars` smoke artifact, or the historical v1.1 result cannot be
  promoted into a complete v1.2 result. The complete r2 diagnostic still has
  no fit, inference, semantic-acceptance, OOS, or trading authority.
- [ ] Treatment episodes are canonicalized before matching; quiet,
  same-session non-sweep, pseudo-level, and forward-time-shift controls remain
  separate and are never pooled.
- [ ] Matching remains maximum-cardinality, forward-only, caliper/embargo and
  no-replacement constrained; inherited direction is known only at treatment
  time. Each control begins after its paired treatment horizon plus embargo;
  cross-pair outcome overlap remains an explicit unvalidated diagnostic limit.
- [ ] Exact McNemar and frozen-family Holm retain missing/underpowered tests as
  p=1; no family is dropped after results are seen.
- [ ] Previous-session contract selection and raw/processed manifests remain
  bound.
- [ ] MBO partition/execution manifests match their exact registered SHA-256;
  sealed MBO is not read during development.
- [ ] June Week 4 is treated only as a separately preregistered within-month
  development diagnostic: W1–W2 remain revealed design evidence, W3 remains
  parked for roll/holiday robustness, and W4 is not relabelled OOS or appended
  to the closed Phase 6 experiment.
- [ ] The frozen W4 design passes its strict identity-only validator without
  opening market data, while execution bindings, mechanism materialization,
  semantic replay, outputs, and run authority remain absent.
- [ ] Temporal predecessor remains a research relation, never semantic
  ancestry/composition or default Scene-Graph action connectivity. Its minimum,
  maximum, same-clock policy, completed-bar distance, and inference role are
  frozen before outcomes are inspected.
- [ ] The historical Phase 6 FVG first-concrete-lifecycle classification is
  frozen at its earliest unambiguous `known_at`; conflicting same-clock
  lifecycle payloads fail closed and later lifecycle changes cannot rewrite
  it. It is not relabelled as a geometric `first_retest_event`; that new
  estimand requires creation-zone/BAR overlap ancestry and a fresh manifest.
- [ ] Phase 6 reporting uses the final registered two-week extension result;
  only `acceptance_continuation` and `displacement_impact` enter the Phase 7
  allowlist, unsupported/underpowered mechanisms remain excluded, and no
  unregistered Week 3 is opened.
- [ ] Missing MBO remains missing and is not replaced with constant execution
  reality.
- [ ] Checkpoint, shards, progress, resume and portfolio before-bar /
  after-decision ordering remain intact.
- [ ] Daily replay writes light decisions/summary only; it does not emit full
  traces, audit packets, images or future-path artifacts.
- [ ] Any bounded 2024-06 foundation replay records its exact input identity,
  row selection, record count, payload parity, and elapsed time, and remains
  labelled engineering construct/replay evidence—not Phase 9, empirical
  validation, calibration, OOF, or OOS.

## Shadow-live parity

- [ ] Every Phase 9 clock binds one exact completed bar plus immutable causal
  execution/account evidence identities and the frozen instrument mapping.
- [ ] Runtime records bind the model configuration and every Phase 7/8 protocol
  fingerprint, full registered engine/Brain/FSM state digests, journal prefix,
  and external-submission-attempt count.
- [ ] Identical feed-event duplicates are idempotent; conflicting identity or
  content, out-of-order clocks, stale/future evidence, and partial mutation
  terminate fail-stop with deterministic failure evidence.
- [ ] Cold replay and checkpoint restart compare records, journal, failure,
  gateway, and final state exactly.
- [ ] Operational pilot metrics for semantic duplication, parent/child churn,
  evidence-belief consistency, signal expiry, and DOL stability are frozen
  before a multi-day run is evaluated.
- [ ] The runner accepts only the exact `ContinuousSMCEngine` and exact
  `NullExecutionGateway` classes; any external submission attempt fails.
  Engineering parity on 36 deterministic synthetic clocks and the historical
  Foundation-enabled 200-clock June prefix are not called current-runtime
  parity, a completed 6,900-clock run, or a real-time multi-day shadow pilot.
- [ ] Scene-Graph delta identity sets used in parity, including simultaneously
  revised edge IDs, have canonical ordering across different
  `PYTHONHASHSEED` values and process restarts.
- [ ] Historical Shadow input publication refuses every pre-existing output or
  sidecar and never uses overwrite/replace as a success path.
- [ ] One destination-scoped process lock covers interrupted-bundle recovery,
  random same-directory staging, sidecar-first/output-last publication, and
  device/inode-owned cleanup; a competing publisher's files are never removed.
- [ ] Local Git-ignored input/checkpoint evidence is labelled non-portable and
  records exact paths, commands, file hashes, protocol identities, and whether
  an independent cold-run receipt was actually retained.
- [ ] A full historical rehearsal does not rebuild every journal prefix or keep
  two unbounded runners resident without an explicit capacity budget; any
  fingerprint change bumps and refreezes the Shadow protocol.
- [ ] The 6,900-clock capacity preflight verifies the completed-prefix marker's
  checkpoint-manifest SHA binding, creates no Engine, replays zero clocks, and
  remains a lower-bound estimate bound to the historical receipt, with that
  receipt's historical/current identity mismatch and
  `full_6900_replay_authorized=false`. A separate later-but-now-historical
  200-clock parity run does not upgrade the preflight into full-run authority.

## Test scope

- [ ] Do not mark the repository target architecture complete while empirical
  fitting/calibration, formal Execution Research, operational Shadow Live,
  rolling OOF, or sealed OOS gates remain open; overall conformance is partial.

- [ ] Run syntax plus the unified synthetic/boundary/causal suite first.
- [ ] For a primitive change, run one bounded real OHLCV replay. If a stable
  chain later needs case inspection, use one small sampled diagnostic and allow
  at most one concept-level repair.
- [ ] Do not use PnL to repair primitive semantics or inspect the same OOF/
  holdout window repeatedly.
- [ ] Run Brain calibration only after primitives are frozen or parked.
- [ ] Run rolling OOF, MBO stability and sealed holdout once, in that order,
  only after the vertical chain is stable.
- [ ] Run the actual multi-day Phase 9 no-order pilot before calling the
  operational Shadow Live gate complete.
