# SMC Semantic Specification: v1.3 atomic layer + foundation v2.1

The current runtime semantic identity is `smc_semantics_v1.3`. Its
machine-readable authority is
[`semantics/registry_v1_3.yaml`](../semantics/registry_v1_3.yaml), with frozen
or deliberately unthresholded parameters in
[`semantics/parameters_v1_3.yaml`](../semantics/parameters_v1_3.yaml).
The frozen [`registry_v1_2.yaml`](../semantics/registry_v1_2.yaml) and
[`parameters_v1_2.yaml`](../semantics/parameters_v1_2.yaml) remain the identity
of every v1.2 artifact and are no longer the runtime. Sections below that still
say "v1.2" describe detector behaviour v1.3 inherits unchanged; the v1.3 delta
is stated in the next section.

The current additive object/lifecycle projection is
`smc_semantic_foundation_v2.1`, bound to the v1.3 atomic stream by
[`semantics/foundation_v2_1.yaml`](../semantics/foundation_v2_1.yaml). Its
human-readable authority is the
[Canonical Semantic Foundation v2](refactor/canonical_semantic_foundation_v2.md).
It adds no parallel detector and does not rewrite a v1.2 `MarketEvent` or any
frozen experiment. Production emits no `FOUNDATION_STATE_CHANGED`; the enum and
strict decoder remain only for legacy technical replay transport and grant no
atomic-market or trading authority. Foundation revision history lives in its
separate in-memory ledger, while snapshots carry a compact current projection.
The checked-in model selects both authorities through one strict
`semantic_selection`: atomic v1.3 version/path/definition identity and
Foundation v2.1 version/path/registry identity. Construction loads the pair
once and requires the Foundation parent to equal the atomic version; missing,
unknown, extra, or mismatched fields fail closed. The Observer's enable flag is
derived internally. This does not define a unified `smc_semantics_v2.0`.
Engine/Shadow/checkpoint state retains and compares the existing identities,
and the current combined Engine checkpoint schema is 7. Earlier schemas cannot
resume into the current Observation, Foundation, and Neutral-state contracts.

[`semantics/registry.yaml`](../semantics/registry.yaml) and
[`semantics/parameters.yaml`](../semantics/parameters.yaml) remain the immutable
v1.1 definitions required to verify the historical January 2024 protocol-v2
artifact. A v1.1 result does not establish v1.2 behavior, and a v1.2 runtime
does not retroactively upgrade a v1.1 event ledger.

## What v1.3 changed

v1.3 keeps every canonical emitted EventKind of v1.2 byte-identical. It changes
identity, naming and reservation only:

- **Object identity split.** `active_dealing_range` became `structural_range`
  (frozen anchors, location, normalized_location, Premium/Discount, IRL/ERL) and
  `balance_range` (two-sided testing, midpoint crossings, inside fraction,
  compression, maturity). `order_block_origin_zone` became `base_origin_core`
  (geometry only) and `qualified_origin_zone` (core + active displacement +
  qualified BOS). Order-block interpretation belongs to the Brain.
- **`structure_regime` is now a registered concept** and is snapshot-derived. It
  and `delivery_phase` are two independent dimensions; neither is derived from
  the other. Every phase fact carries the regime beside it as evidence, never
  as its cause.
- **`delivery_phase` is an entered/updated/exited lifecycle and is emitted.**
  `MarketSnapshotPublisher` tracks one occupancy per timeframe. An entry
  freezes `entered_at`, `age_bars` (real completed M1 bars since entry),
  `origin_event`, `parent_structure_generation` and `previous_phase`; an exit
  names `next_phase`. `DELIVERY_PHASE_UPDATED` is published **only** when a
  registered phase input (`structure_regime`, `active_leg_direction`,
  `protected_swing_intact`, range availability) moves while the phase itself
  does not, so a quiet bar publishes nothing and the phase stream stays far
  below one event per bar. `parent_structure_generation` is the event id of the
  most recent `STRUCTURE_DIRECTION_CONFIRMED` / `QUALIFIED_BOS` /
  `MSS_CORE_CONFIRMED` on that timeframe: the hot path carries no separate
  generation object, and this is the causal anchor a generation id would hash.
  The three kinds are admitted to the reducer and are recorded as seen without
  being reduced back into the state they were derived from.
- **The Origin Zone is split and `ORIGIN_ZONE_CREATED` is retired.** At one
  clock the Eye publishes `BASE_ORIGIN_CORE_CREATED` (frozen candle geometry
  plus the impulse identity that located it, citing only its anchor bars) and
  then `QUALIFIED_ORIGIN_ZONE_CREATED` (that core plus the active displacement
  and the qualified BOS). Origin-zone terminals cite the qualified zone. The
  compact current view tracks the qualified zone; the bare geometry stays
  readable as a fact the Brain interprets. The detector materialises the core
  only when the BOS confirms, so publishing it earlier would be a Group-3
  protocol change and was not made.
- **The Dealing Range is split and `DEALING_RANGE_ACTIVATED` is retired.**
  `DEALING_RANGE_CREATED` / `_INVALIDATED` / `_REPLACED` remain the structural
  range's location lifecycle. `BALANCE_RANGE_OBSERVED` is published once per
  range, carrying `balance_lower_test_generations`,
  `balance_upper_test_generations`, their per-generation
  `balance_*_test_kinds`, the structural `lower_touch_count` /
  `upper_touch_count` beside them, `midpoint_crossings`,
  `inside_close_fraction`, `compression_ratio`, `candidate_real_h1_bars` and
  `age_h1_bars`, as soon as price has tested both frozen boundaries at least
  `balance_range_bilateral_price_tests_each` (2) times.
  Failing to balance no longer costs the location.
- **The Structural Range lifecycle is `active` -> `broken`, and nothing else.**
  A range is a location from the moment it is created and ends only when price
  closes outside it. `mature` was a grade the interval earned by balancing and
  `forming` meant it had not earned one yet; both are gone, and
  `BALANCE_RANGE_MATURED` is reserved rather than emitted. Three months of data
  removed the grounds for the inheritance: 0 of 84 ranges reached a two-sided
  test, an arbitrary rolling H1 window reaches one 4.06% of the time, and the
  sample cannot separate the two. Balance is rare in this market -- the range
  was never failing at anything. Balance evidence is still collected and still
  published once per range as `BALANCE_RANGE_OBSERVED`; settling the claim is
  recorded on the range as the `balance_claim_confirmed` reason and the
  `balance_confirmed_at` clock, which is what promotes the frozen boundaries to
  candidate liquidity levels. Those levels now descend from
  `DEALING_RANGE_CREATED`, the interval that froze them, rather than from a
  maturity event that no longer exists.
- **Balance evidence is price interacting with the boundary
  (`balance_range_v1.2`).** A test is a completed H1 bar reaching into the
  boundary's tolerance band, `max(1 tick, 0.25 x the ATR known before that
  bar)`, classified `touch_only`, `shallow_penetration`, `deep_penetration` or
  `close_outside` and escalating to the deepest reached. One *generation* is
  one continuous visit: price must leave the band before the next test can
  open, so a run of bars hugging a level counts once. `close_outside` remains
  the range's own invalidation rather than a second concept. This replaced two
  earlier clocks, each of which made the claim unreachable for a different
  reason. Until 2026-09-03 the emitter saw a range only when its lifecycle
  changed, so a candidate that simply kept forming had no clock at all; the
  observation clock is now every completed H1 bar on which the range is live.
  That alone changed nothing, because the evidence was the source zone's
  `total_touch_count`, and a `structural_swing` zone is skipped by the
  price-contact counter outright -- it only counts a touch when *another
  confirmed swing* forms inside it. Over 2022-02, 17 of 18 H1 zones therefore
  stayed at one touch for life, and one touch per side is structural because a
  range's source pair is built from tested zones. The structural count is
  untouched and still carries its own meaning; balance now counts its own
  evidence.
- **`FVG_FIRST_RETEST` is emitted.** Exactly one per gap, on the first bar
  where price re-enters the frozen gap, published before the revisable fill
  observation it shares that bar with. It freezes `fill_depth_at_entry`,
  `age_bars`, `age_seconds`, the entry lifecycle and reason, the formation
  qualification, the session, the source displacement, and
  `approach_speed_atr` = the distance still separating the last completed M5
  close from the near edge, divided by that entry bar's causal ATR. Partial,
  mitigated and invalidated all require entry, so the first of them is the
  first re-entry; the reserved age-based expiry is not an entry. The Group-3
  protocol lists MBO among its prohibited inputs, so the event carries no book
  evidence. It is admitted to the reducer as an authoritative fact but has no
  reducer branch: it never revises the compact current view.
- **`FVG_EXPIRED`, `DEALING_RANGE_EXTENDED` remain reserved**, and
  `FVG_TOUCHED`, `ORIGIN_ZONE_TOUCHED`, `ORIGIN_ZONE_CREATED`,
  `DEALING_RANGE_ACTIVATED` and `DELIVERY_PHASE_CHANGED` are compatibility
  aliases that are not emitted.
- **`bos_state` was removed.** It was an unregistered transport carrying only
  `pending`, already stated by the absence of a terminal on
  `RAW_BOUNDARY_BREAK`. The BOS timeline now begins at `CONFIRMED`/`FAILED`.
- **Structural Leg `rank` became `path_class`**, and the lookup is strict: a
  missing key now fails instead of silently defaulting.
- **Densified no-trade bars are admitted into definitional paths** with
  `synthetic_path_minutes` / `synthetic_window_minutes` markers, under one
  shared `BarCoverage` admission rule.

Phase 7 stays preregistered against `smc_semantic_foundation_v2.0` and refuses
the current model config; running it under v1.3 requires a new preregistration.

New event studies must start from a separately frozen manifest and fail closed
when a required semantic identity, parameter, code/data hash, input census, or
acceptance field is absent. The completed Signal Research protocol-v3 r2
manifest authorizes only its full-window development diagnostic; the template,
smoke run, and failed first attempt authorize nothing. That boundary is
separate from the completed, frozen Phase 6 MBO study.

## Existing-system migration decision

The implementation reuses the repository's causal multi-timeframe aggregation
and existing Swing, structure, liquidity, displacement, FVG/Origin Zone, and
dealing-range detectors. v1.2 does not introduce a parallel detector or market
state stack. Detector output becomes immutable canonical atomic events, and
those events are the accepted inputs to hierarchical state reducers.
Compatibility/state-projection events may remain readable for transport or
historical replay, but they are not a second semantic authority.

The [second-round semantic review](refactor/semantic_review_round_2.md)
separates **definition validity** from **empirical validity** for the 17 named
concepts. A causal, deterministic, replayable definition may remain empirically
unknown or unsupported. Conversely, coverage or a contemporaneous association
cannot repair an ambiguous definition or grant predictive/OOS/action authority.
The frozen v1.2 registry remains the identity of historical artifacts. The
foundation is deliberately a separate versioned identity, so current
store/reducer/tick hardening and lifecycle completion do not rewrite those
artifacts or silently change their definitions.
The [DOL/Belief/Temporal supplement](refactor/dol_belief_temporal_supplement.md)
is the current authority for dependency-cluster handling, research-only
temporal links, FVG first-lifecycle freezing, and the 2024-06 data role. It does
not add a canonical market event or rewrite this semantic version.

## Common provenance contract

The common audit envelope contains:

- `event_time`: when the market occurrence or source pivot happened;
- `known_at`: the first completed-data clock at which the event may be used;
- `semantic_version`: the definition identity used to construct the event;
- immutable `source_event_ids`, `source_data_ids`, and `source_entity_ids` in
  distinct namespaces;
- immutable evidence, optional zone, origin classification, and event payload;
- deterministic identity bound to clocks, provenance, payload, and semantic
  version; and
- append-only storage, causal source validation, fingerprinting, and
  deterministic as-of replay.

`MarketEvent.observed_at` remains a compatibility alias of `known_at`. Legacy
`source_ids` may be read only on the explicit legacy path. Canonical atomic
events must satisfy their per-kind contract and generic source-graph rules; a
raw candle or domain entity cannot masquerade as a semantic event parent.

An event may have `event_time` earlier than confirmation, but cannot enter a
reducer, Brain, or study before `known_at`. Canonical order is
`(known_at, sequence_no, event_id)`. Given identical normalized input,
semantic identity, ordered atomic events, initial state, and reducer code,
replay must produce identical events, state, and fingerprints.

A market clock recorded inside evidence is never a substitute for `known_at`.
A crossing terminal (`SWEEP_CONFIRMED`, `ACCEPTANCE_CONFIRMED`) carries three
distinct clocks: `event_time` is the crossing, `evidence["resolved_at"]` is the
market clock that decided the outcome, and `known_at` is the observation clock
at which the Eye could first derive the terminal. When a reducer only reaches
its verdict on a later observation, `known_at` is that later clock — stamping
the market clock instead backdates the terminal behind ancestry it is required
to cite. A generation's `resolved_at` is frozen once recorded; its `known_at`
is the first clock it was knowable at and may never move earlier.

Research temporal proximity is not semantic ancestry. Scene-Graph `PRECEDES`
may remain visible as a diagnostic edge, but it is excluded from canonical
causal/open-thesis closure and action connectivity. It never enters
`source_event_ids` merely because two facts occur inside a time window.

The bounded `EventMemory` remains the hot view used by retained consumers.
`EventStore` is the append-only audit/research journal and supports
deterministic as-of replay; the runtime need not keep an unbounded duplicate
archive in every state object.

## Ownership boundaries

The Eye normalizes data, recognizes preregistered atomic semantics, reduces
each timeframe, resolves deterministic parent/child relations, maintains a
cross-timeframe Session object, and publishes market facts and candidate
targets. It does not choose a unique DOL, assign a trading probability, choose
an action, or select an execution method.

The Brain owns competing path and DOL hypotheses, evidence assimilation,
invalidation/expiry, signal assessment, and trade/no-trade intent. The current
Brain includes a normalized six-path Hypothesis Manager, Bayesian-shaped
Belief Updater, DOL ranking, a fitted-artifact-only DOL probability layer with
explicit no-target mass, Signal
Policy, and a standalone Trade Intent projector API. They are
`development_unvalidated` and `shadow_only`. Exact production-shape market
facts now map to registered path
invalidation and realized-winner events; same-clock terminal facts take
precedence and local EntryEpisode failure is excluded. No fitted/admitted
path-likelihood, DOL-probability, or outcome artifact exists. The current
legacy-compat Engine does not invoke the standalone Trade Intent projector and
rejects a non-zero prefilled map before Decision/Risk. Only exact
`MarketEvent` IDs resolvable from the current Observation are admitted as path
evidence; unresolved context/entity identities fail closed. `MarketBelief`
carries the current set and this-clock update records, not a complete evidence
journal. Decision does not consume them.

Path protocol v1.2 treats each `correlation_key` as one global evidence
dependency cluster. Different evidence-family labels do not create independent
likelihood factors. A future admitted model must provide one precombined or
history-conditioned contribution for a shared cluster; the current neutral
ledger applies no fitted likelihood. The Brain rejects a likelihood-enabled
protocol without a complete externally pinned artifact set. Since the current
artifact schema has no registered dependency resolver, fitted diagnostic facts
are conservatively scoped to one unresolved competition-set cluster and cannot
accumulate multiple factors. Equal subtraction from every active logit
is probability-invariant and is not called decay. Hypothesis-specific expiry,
hazard, and prior reversion remain unavailable until separately preregistered.

Execution owns market/limit choice, working-order lifecycle, fills,
cancellation, stops, targets, position management, and account risk. The
retained replay portfolio still has its conservative pending/open/exit
simulation. Separately, Phase 8 implements immutable order FSM v1.5 with
command/fact separation, partial-fill/cancel-replace custody, rejection,
reconciliation, reservations, position conservation, checkpoint, and replay.
It has no broker/network submission authority. Execution-research protocol
v1.1 separately defines entry GTT and the later analysis horizon, cancels a
pending remainder when target resolution precedes its fill, and keeps primary
implementation-shortfall pairing independent from secondary realized-spread/
path censoring. Stop/invalidation and target prices fail closed unless they are
on the frozen tick grid. The formal study remains unauthorized to run, so no
empirical fill-quality claim exists.
Market and order state remain separate ownership domains.

## Executable v1.2 producer surface and review boundary

- The audit foundation provides causal clocks, semantic identity binding,
  immutable event payloads, an append-only journal, source-graph validation,
  checkpoints, and deterministic replay.
- Core atomic semantics cover Confirmed Swing, Structural Leg, Candidate
  Liquidity Level, Touch, Penetration, mutually resolved Sweep or Acceptance,
  continuous Displacement observations, Raw Boundary Break, and registered FVG
  lifecycle transitions.
- Derived structure producers cover Structure Direction, Qualified BOS,
  Protected Swing assignment/invalidation through registered same-timeframe
  evidence, MSS Core, a qualified Origin Zone composite, the legacy
  `DEALING_RANGE_*` lifecycle, and deterministic Delivery Phase projection.
  Sweep, Displacement, and FVG remain MSS context rather than definition gates.
- The authoritative timeframe reducer accepts registered atomic events.
  Independent `TimeframeState`, `RelationState`, and `SessionState` objects are
  published through `MarketSnapshot`; parent state changes only on the parent's
  own qualifying event. A child may be classified as a retracement or warning
  without rewriting its parent.
- Session remains one cross-timeframe context object, not a field owned by a
  single timeframe.

## v1.2 Swing hierarchy

Confirmed pivots are immutable. Hierarchy is an append-only reducer projection
over exact canonical source identities:

| Causal fact | Assigned role | Nesting depth |
|---|---|---:|
| `SWING_CONFIRMED` | `micro` | 0 |
| exact `STRUCTURAL_LEG_CREATED` endpoint | `internal` | 1 |
| exact `STRUCTURE_DIRECTION_CONFIRMED` source swing | `structural` | 2 |
| exact `PROTECTED_SWING_ASSIGNED` swing | `external` | 3 |

Each `SwingRankAssignment` carries its own `known_at` and source event IDs.
The current role is the highest causally available assignment; a later
promotion never rewrites or backdates the original Swing or a historical Leg.
Prominence remains a continuous feature with no positive v1.2 hard cutoff.
This table remains the v1.2 role-depth projection. Foundation v2 now publishes
a separate pure time-and-price containment tree with `geometric_depth`,
parent/child identities, deterministic late-parent supersession, and no BOS or
future-importance input. Role depth and geometric depth remain independent.

## Legacy v1.2 range location and IRL/ERL

The Eye preserves continuous normalized location
`x=(price-low)/(high-low)`. Candidate liquidity is classified only against the
same-timeframe frozen range while that range is active/mature:

- `low < candidate_price < high` → `irl`;
- candidate on either boundary or outside → `erl`;
- no active range, forming/replaced/invalidated range, or invalid geometry →
  `unresolved`, with null `normalized_location_in_range`.

This membership is deterministic snapshot state, not a canonical event and not
a claim that the candidate will be delivered.
The production Group 4 detector is the sparse two-sided Mature Balance Range
measured by the 2023 natural scan. The legacy `DEALING_RANGE_*` name does not
establish a separately defined Structural Dealing Range. Within the v1.2 field
alone, this location must not be promoted as structural-range, DOL, or
predictive evidence.
Foundation v2 now publishes independent `StructuralRange` and `BalanceRange`
records plus `x_structural_range` and `x_balance_range`; both may coexist, and
Premium/Discount authority belongs to the Structural Range view. This additive
view does not retroactively change the legacy dealing-range field or validate
predictive value.

## DOL boundary

The Eye publishes visible candidate facts: side, price, timeframe, source kind,
strength, event-sourced Swing role, age, and IRL/ERL membership. It neither
chooses a unique DOL nor publishes a path probability or authoritative obstacle
ranking.

The shadow Brain joins exact external-draw inventory to competing paths and
reuses existing directional obstruction views. Obstacles must lie in the
strict open interval from current price to target; target-self identity,
shared-source, and co-located obstructions are excluded. Candidate softmax
weights and their linked path weights are explicitly diagnostic and
uncalibrated. The separate DOL-probability projection marginalizes registered
path mass and retains an explicit no-target outcome; it cannot force a draw or
become action authority while its fitted artifact is absent.

## Canonical-emission governance

The registry is the emission allowlist. A registry-bound semantic kind marked
`canonical_emitted` may enter the canonical emitter; aliases, reserved kinds,
and snapshot-derived concepts are rejected from canonical emission. Retaining
an `EventKind` enum value or a reducer compatibility path is not an emission
claim.

Five kinds intentionally remain non-emitted in v1.2:

- `FVG_TOUCHED`: compatibility alias; the canonical lifecycle emits partial
  fill, midpoint touch, full fill, and invalidation.
- `FVG_EXPIRED`: reserved; v1.2 has no age-based expiry threshold.
- `ORIGIN_ZONE_TOUCHED`: compatibility alias; first intersection is the
  canonical `ORIGIN_ZONE_MITIGATED` transition.
- `DEALING_RANGE_EXTENDED`: reserved; no extension rule is registered.
- `DELIVERY_PHASE_CHANGED`: compatibility projection alias; Delivery Phase is
  deterministically snapshot-derived.

These remain deliberate v1.2 atomic-emission decisions. Foundation v2 supplies
versioned persistent Delivery and structural FVG expiry records over exact
atomic parents; it does not enable the reserved v1.2 aliases. Range extension
still has no registered definition.

## v1.2 compatibility limits and v2 resolution

- The v1.2 Delivery value remains a deterministic snapshot classifier.
  Foundation v2 wraps that existing classifier in a persistent generation with
  owner, entered/updated/terminal clocks, native-bar age, extrema, and exact
  transition sources. It does not invent range-extension or compression
  inputs.
- FVG and Origin Zone detection stays on the existing M5 path. Foundation v2
  separates Base Origin Core from Qualified OB, emits a strictly future
  geometric First Retest, and allows only structural/reset FVG expiry with no
  arbitrary TTL.
- A v1.2 Leg remains readable. A foundation-v2 Leg requires the full native
  path, close/extreme efficiency, close/wick MAE, duration, tick amplitude,
  exactly 14 strictly prior ATR source bars, and frozen `ATR_at_leg_start`.
  Missing ancestry remains compatibility v1.2 rather than being backfilled.
- MSS remains an atomic evidence fact. Foundation v2 starts one FORMING
  internal challenger/Structure Transition and requires a strictly later
  independent confirmation; MSS alone never confirms the opposite regime.
- Deterministic Swing rank and IRL/ERL implementation do not establish outcome
  value; they still require preregistered research.
- The path/DOL/Signal/Intent shadow path is normalized and auditable but not
  fitted, calibrated, Bayesian-validated, action-authorized, or live-authorized.

## Research and promotion boundary

The frozen January 2024 v1.1 protocol-v2 run is historical diagnostic evidence.
Its 24.04% matched-control coverage and zero E3–E6 samples must not be presented
as current v1.2 results.

Research protocol v3 implements an M5 E1–E6 chain, non-nested partitions,
canonical treatment-episode de-duplication, four separate control families,
deterministic forward-only maximum-cardinality matching with caliper/embargo,
pseudo-level construction, exact-stratum forward time shifts, adjacent-stage
deltas, exact McNemar, and fixed-family Holm. Strict ancestry uses only
`source_event_ids`; exact normalized M5 source-bar composition is retained as a
different proof and context-only links prove neither. Cross-pair outcome
windows may overlap, so the inference remains descriptive and unvalidated.
The frozen r2 run completed the registered January development window. It
contains E1–E6 episode counts 1,124 / 317 / 17 / 1 / 1 / 0; quiet and non-sweep
controls matched 372 and 62, while pseudo and forward-shift matched zero. This
is a complete diagnostic sample but not inference, semantic admission, model
fit, OOS, or trading evidence.

Phase 6 MBO mechanism validation is complete for the frozen June 2024 primary
week plus its one preregistered second-week extension. Only
`acceptance_continuation` and `displacement_impact` are admitted as Phase 7
evidence; underpowered/unsupported mechanisms remain excluded and no Week 3 is
authorized. This is development association, not a new semantic definition,
causal proof, OOS evidence, model fit, or trading authority.

Phase 7 reducer/interfaces and current-scope market-fact path lifecycle are
integrated, but scope-retirement semantics are unregistered and the probability
model remains neutral with no fitted or admitted path/DOL/outcome artifacts.
The Brain therefore emits rankings but an empty DOL-probability map. June Week
4 has passed joint OHLCV/MBO input preflight, and its separately scoped
temporal/branching development design is frozen. The strict identity validator
passes without opening market data, but execution bindings, materialization,
replay, and results remain absent and unauthorized; it is not a Phase 6
extension. The Phase 7 read-only readiness checker inspects all 7,381 compact
Phase 6 ledger rows and reports 13 blockers with no fit or artifact write.
Phase 8 standalone order-FSM engineering and seven-entry-method evaluator core
are complete. Evaluator v1.1 separates entry GTT from analysis end, cancels a
pending remainder after target-first resolution, and preserves eligible primary
pairs despite secondary-metric censoring. It rejects off-grid stop/target
prices. The v2 research protocol requires method-price provenance and registers
wait/cancel/stop/target variants with explicit evaluable-or-censored rules. Its
formal runner exists and validates without opening data by default, but the
current template has no bound non-zero intent ledger or minute source, is not
frozen, and has no empirical result. The authoritative end-to-end handoff
remains open. The current Phase 9 harness is `phase9_shadow_live_v1.3` with
`shadow_compact_runtime_v8` and `phase9_shadow_component_digest_v3`.
Earlier deterministic parity ran behind `NullExecutionGateway`; its 6,900-clock
historical cold-start input and exact 200-clock checkpoint/resume/cold-replay
prefix are retained evidence for the earlier path/model bindings
`5213b3d6…`/`4214da19…`. The current dependency protocol changed those bindings,
so that output remains historical. A separate Foundation-enabled Engine run
passed the same exact 200-row prefix under later, but now also historical,
model/registry bindings; its detailed boundary is in the
[Foundation release table](refactor/canonical_semantic_foundation_v2.md#replay-test-and-empirical-boundary).
The read-only 6,900-clock capacity preflight still verifies the historical
`COMPLETED.json` to checkpoint-manifest SHA binding, finds that receipt's
historical/current identity mismatch, and explicitly withholds full-run
authority. Operational acceptance metrics, a rematerialized complete
historical run, and an actual real-time multi-day pilot remain open. Calibrated
promotion, empirical Execution Research, operational Shadow Live, rolling OOF,
and final OOS remain later gates; conformance to the complete target-state
prompt is therefore **partial**.
The input and checkpoint are ignored local engineering files; their paths,
hashes, commands, no-replace publication rules, and receipt limitation are
listed in the
[Phase 6–9 gate report](refactor/phase_6_9_completion_report.md#phase-9-parity-harness-implemented-pilot-pending).
See the
[current implementation status](refactor/current_implementation_status.md).
