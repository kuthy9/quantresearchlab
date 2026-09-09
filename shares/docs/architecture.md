# Causal continuous architecture

Document schema: **1**

## Repository layout

Runtime code lives in four subsystem packages — `eyes/`, `brain/`, `execution/`
and `shares/` — each with its own `core/`, `tests/`, `configs/` and `docs/`.
The ownership boundaries this document describes are unchanged by that split;
it relocated files and rewrote imports only. Each package also owns a
`scripts/` directory for its bounded studies and materializers. `semantics/` and
the eight sealed `configs/` files stay at the repository root because the atomic
semantic identity hashes their contents or their path strings. See
[current_implementation_status.md](current_implementation_status.md) for what
the identity seal allowed to move.

## Current implementation boundary

This document describes the incremental/development data path and later
subsystem ownership. Live mode currently fails closed.
The current Eye semantic identity is `smc_semantics_v1.3`; its exact executable
and reserved surface is recorded in the
[semantic specification](../../eyes/docs/smc_semantic_specification_v1.3.md). The Phase 2–4
producer/state surface is executable within that frozen boundary; round-two
review distinguishes this from completion of every target definition.
The additive `smc_semantic_foundation_v2.1` projection now completes the
registered geometry, generation, lifecycle, relation, transition, first-
reinteraction, ancestry, and factual-outcome layer over those immutable facts.
See the
[foundation specification](../../eyes/docs/canonical_semantic_foundation_v2.1.md).
It reuses the existing detectors and immutable store; it is not a parallel Eye
or a new action path.
The current implementation-versus-plan matrix is recorded separately in the
[current implementation status](current_implementation_status.md), and
the registered atomic definitions — concepts, formulas, evidence, the
v1.2 → v1.3 delta and every defect the replay exposed — in the
[semantic specification](../../eyes/docs/smc_semantic_specification_v1.3.md).
The historical Phase 2–5, Phase 6–9, DOL/Belief/Temporal, round-two-review and
separate preregistered-semantics documents were removed on 2026-09-06; their
still-current conclusions live in those two documents. Executable coverage is
never treated as predictive validation.
The numbered implementation modules are no longer runtime owners: Zone logic
lives in `eyes/core/zone.py`, and Range Auction logic lives in
`eyes/core/range_auction.py`. `group3.py` and `group4.py` are unexported,
warning-only compatibility shims for historical pickle class lookup.
The repository-owned target architecture is therefore only partially
satisfied: the auditable ownership and fail-closed interfaces are substantially
present, but the fitted empirical model, Execution Research, operational Shadow
Live, rolling OOF, and sealed OOS gates remain unopened and closed.

The existing Brain retains multiple root-specific candidates and raw,
calibratable scores. Phase 7 adds a six-path Hypothesis Manager,
Bayesian-shaped belief updater, obstacle-aware DOL ranking, fitted-artifact-only
DOL probability, Signal
Policy, and Trade Intent projection to that same Brain. The complete path stays
shadow-only because no fitted/admitted likelihood, DOL, or outcome artifact
exists; production therefore emits zero Trade Intents and Decision/Risk remain
unchanged. Phase 8 adds a separate immutable order FSM v1.5 and a causal
seven-entry-method research core. Its v1.1 clock/metric contract separates
entry GTT from the later analysis horizon, cancels a pending remainder when a
target resolves before that fill, and keeps primary implementation-shortfall
pair eligibility independent of secondary-metric censoring. Stop/invalidation
and target prices fail closed unless they lie on the frozen tick grid. Method
variants and provenance remain incomplete, the manifest is unauthorized, and
no empirical result exists.
Phase 9 adds a
no-submission parity runner, not a completed real-time operational pilot.

## Runtime flow

```text
completed M1 clock (real or clock-only synthetic)
        ↓
normalized BAR_COMPLETED root + causal 1m / 5m / 15m / 1H / 4H aggregation
        ↓
pre-registered SEMANTIC_ATOMIC events + in-memory immutable audit store
        ↓
deterministic TimeframeState reducers + SessionState
        ↓
independent parent/child RelationState
        ↓
MarketSnapshot (atomic event-reducer authority + compact replay projection)
        ├── versioned factual StructuralOutcomeEngine (research output,
        │   not MarketSnapshot state)
        ───── optional existing development-trader downstream ─────
Temporal Market Scene Graph + GlobalMarketContext
        ↓
playbook-neutral OpenMarketThesis + FocusState
        ↓
        ✕ no belief producer — the typed Brain was retired 2026-09-07
        ↓
enter / wait / hold / protect / exit / abstain utility comparison   (no input)
        ↓
independent structural, cost, deadline, data and fillability development vetoes
        ↓
optional one-next-bar conservative simulation → position feedback

the six-path competition set survives standalone in
brain/core/market_belief.py; nothing consumes it
```

### Reasoning responsibilities

The model uses a one-way reasoning chain. Each layer answers a different
question and may consume only already-known outputs from the layer above it:

| Reasoning mode | Components | Question answered | Prohibited authority |
|---|---|---|---|
| Data fact admission | `io`, `market_clock`, `CausalMarketReader` | Which completed clocks and prices are legally knowable now? | No structure, probability, or action interpretation. |
| Atomic observation | `CausalObserver`, Group 1–5 trackers, `EventStore` | Which preregistered v1.2 market facts occurred, and from which exact sources? | No rewriting history and no trade decision. |
| State and relation projection | timeframe/session reducers, `RelationResolver`, Foundation v2, `MarketSnapshotPublisher` | What is the deterministic current state, lifecycle, geometry, and cross-frame relation? | Foundation is an additive v1.2 projection, not a new detector or full-stack v2 authority. |
| Competing-hypothesis reasoning | Scene Graph, `market_belief` (the typed Brain, DOL and Signal Policy were retired 2026-09-07) | Which still-falsifiable path or target is better supported by admitted evidence? | With no fitted/admitted artifacts, outputs remain neutral or shadow-only. |
| Action constraint and validation | Decision, Risk, simulator (Trade Intent and the execution FSM were retired 2026-09-07) | Is an already-described candidate allowed to become a simulated action or a no-order audit fact? | No retroactive semantic change, broker authority, profit claim, or sealed-OOS access. |

The `Decision`/`Risk`/sequential simulator path remains the only registered
runtime action authority under `legacy_decision_risk_compat`, and it currently
has no belief to act on. The `TradeIntent`/risk-approval/execution-FSM contracts
that used to sit beside it were retired on 2026-09-07. The rule they existed to
enforce still holds and must hold for whatever replaces them: this prevents two
simultaneous action owners without claiming the vertical migration is complete.
The per-file owner/consumer census that used to live in a repository file
inventory was removed on 2026-09-06; `git ls-files` and this document are the
current answer.

### Trading Eye responsibility layers

The requested five layers exist as responsibilities in the reused runtime;
they are not five duplicate same-named stacks:

| Layer | Current implementation | Contract |
|---|---|---|
| Data Normalizer | `CausalMarketReader` and `_TimeframeAggregator` (`causal.py`), the registered session calendar (`market_clock.py`), the scale contract (`scale_registry.py`), and normalized-root publication in `CausalObserver` | Accept one completed M1 clock, reject off-grid prices before any state advances, distinguish real/clock-only/no-trade data, surface every clock defect explicitly, and complete higher frames without future data. |
| Semantic Event Engine | `CausalObserver` plus the typed detectors (`structure.py`, `liquidity.py`, `displacement.py`, `zone.py`, `range_auction.py`, `interaction.py`), `semantic_event_emitter.py`, `event_memory.py` and `EventStore` | Detectors own registered state; `CausalObserver` orchestrates one clock; `SemanticEventEmitter` is the single canonical emitter and the sole owner of the cross-detector event-ancestry index that binds every semantic fact to its normalized BAR root. |
| Timeframe State Reducer | pure `reduce_timeframe_state()` and `TimeframeEventReducer` | Reduce ordered normalized/atomic events into independent deterministic timeframe state. |
| Cross-Timeframe Relation Resolver | `RelationResolver` owned by `MarketSnapshotPublisher` | Compute parent/child relations without allowing child votes to rewrite parent authority. |
| Market Snapshot Publisher | `MarketSnapshotPublisher` | Publish timeframe, relation, session and event-delta views at one causal clock. |

`CausalObserver` is the event-engine facade; the publisher owns the relation,
session, and snapshot assembly seam. Creating additional pipelines solely to
match these labels would split authority rather than simplify it.

`SemanticEventEmitter` (`semantic_event_emitter.py`) is the only place a
`MarketEvent` is minted for the Eye. It holds the canonical atomic append path,
the crossing-terminal and protected-assignment custody rules, and the whole
cross-detector ancestry index, so the observer no longer carries emission state
alongside orchestration state. The observer passes it the clock and the
detector state each emission needs; the emitter never reaches back into the
observer. A crossing terminal separates the two clocks it depends on: evidence
`resolved_at` keeps the market clock that decided the crossing, while `known_at`
is the observation clock at which the Eye could first derive the terminal. They
differ whenever a reducer only reaches its verdict on a later observation, and
conflating them backdated a terminal behind ancestry it is required to cite.

The Eye imports no downstream module. `scale_registry.py` owns `ScaleSpec`,
`parse_scale_specs` and `scale_registry_id`, so neither the reader nor the
observer imports the optional Scene Graph in order to describe its own scales;
`scene_graph.py` re-exports those names only so historical pickles resolve to
the same class objects. `execution.py` owns `ExecutionRealityInput` and the
cost/fillability score, and `contract/execution/reality.py` owns the inert not-evaluated value
beside the `ExecutionObservation` it constructs, so the Eye transports an
execution observation without deriving one; `ContinuousSMCEngine._score_execution`
derives it. `eyes/tests/test_eye_module_boundary.py` asserts that no Eye module
imports a downstream layer. `ContinuousSMCEngine` owns the
`TemporalMarketSceneGraph`, advances it over one completed Eye observation, and
stamps the resulting `scene_*` delta identities onto that observation; a graph
failure calls `CausalObserver.mark_terminal_failure` because the reducers have
already committed the clock. `EventStore` is the Eye's internal history
authority and is not exported from the package root.

Production construction is explicit and fail closed. One root
`semantic_selection` chooses the atomic `smc_semantics_v1.3` registry and the
additive `smc_semantic_foundation_v2.1` registry, including both identities.
The strict loader requires the Foundation registry's parent version to equal
the selected atomic version. It does not mint a composite or “full-stack v2”
identity. Engine construction loads this pair once and derives the internal
Foundation-enabled flag; Engine, Shadow, and checkpoint state freeze and
compare the existing version/identity fields. The current combined Engine
checkpoint schema is 12; earlier schemas are rejected rather than restored into
an incompatible Observation, Foundation, or Neutral-state contract.

### Hot-state boundary

`EventStore` is the sole authoritative atomic history and `MarketSnapshot` is
the sole current-market-view authority. There is no hot Foundation projection:
`smc_trader/foundation_adapter.py` was removed, no runtime module constructs
`FoundationProjection`, `FoundationProjectionReducer` or
`FoundationRecordLedger`, and the Engine no longer carries a Foundation version
or registry identity in its checkpoint state. `MarketSnapshot` fingerprinting
and replay transport serialize only the compact current view and component
identities; publishing an immutable view still copies current tuple/map
references, bounded by live logical objects.

Production emits no `FOUNDATION_STATE_CHANGED`; its enum, encoder helper, and
`market_state.foundation_record_from_projection_event` decoder remain read-only
compatibility seams for historical journals. Foundation v2 now persists only as
the hash-bound `smc_semantic_foundation_v2.1` registry identity in
`semantic_selection`, as `foundation_version`-stamped Structural Leg evidence,
and as cold definition modules (`semantic_foundation.py`,
`semantic_lifecycle.py`, `semantic_zones.py`, and the geometry/cluster/range
builders in `market_state.py`) exercised only by their focused tests. Those
definitions are retained on purpose; they are not a second lifecycle or state
authority. Formal Signal and MBO research runners remain Eye-only.

`ContinuousSMCEngine` has one incremental trader path, and its trader replay
calls that same engine. The frozen Phase 5 diagnostic was deliberately a
separate Reader + Eye-only runner that never constructed Brain, Decision, Risk,
MBO, or execution; that runner was retired on 2026-09-06 and only its recorded
conclusions survive. Multi-timeframe bars update only when complete.

The separator in the diagram is a sequencing boundary, not a claim that the
Scene Graph/Brain consumes `MarketSnapshot` as its sole evidence contract.
`MarketObservation` now stores one embedded `MarketSnapshot` as the owner of
`asof`, instrument, price, and this-clock semantic events, exposing read-only
compatibility properties instead of duplicate fields. It still carries
detector-facing and typed transition views for Scene/Brain consumers. The Phase
7 path/DOL/Signal/Intent components are integrated into that same Brain and do
not establish a second runtime trader.

The public Eye contract is event-sourced and hierarchical rather than one
combinatorial enum. `FrameObservation` remains an existing detector-facing
view used to produce atomic events; it is not the authority for public state.
`TimeframeState` is reduced from ordered normalized and atomic events;
`RelationState` owns cross-timeframe facts without copying a parent into every
child; `SessionState` is a separate completed-1m context object; and
`MarketSnapshot` publishes both the current state and the immutable event delta
available at that clock. v1.2 additionally maintains append-only Swing role
assignments and same-timeframe IRL/ERL target membership as deterministic state
projections.
Round-two v1.2 hardening makes raw-price tick admission and immutable-event
source/lifecycle authority fail closed.

v1.3 promotes six of those facts from per-bar rows to lifecycle entities the
snapshot publishes alongside the states:

- `MarketSnapshot.structure_generations` — one continuous structural claim per
  timeframe and scope, absorbing every BOS and MSS while its protected swing
  holds. Delivery phases and relations cite the generation, not the last break.
- `MarketSnapshot.relation_generations` — one continuous occupancy per
  cross-timeframe relation, keyed by
  `(parent_structure_generation_id, child_structure_generation_id, role)` and
  carrying `observation_count`, so a role held across many bars is one episode.
- `DOLCandidateView.generation_ordinal` — a swept level is disarmed rather than
  deleted, and is re-armed as the next generation of the same identity once
  price has left it behind and come back.
- `SwingHierarchyView.geometric_parent_id` / `geometric_depth` / `child_ids` —
  the geometric nesting tree, settled across every timeframe at once from
  window containment alone, fully independent of `semantic_rank`.
- `BaseOriginCoreState` — frozen impulse geometry published on the bar that
  locks it, before and independently of any qualification.
- The structural interval locates price from creation; the balance claim is a
  separate, gated fact that can be abandoned without ending the interval.

The legacy role-depth Swing projection and the H1 balance-range detector rule
itself remain unchanged. Additive foundation v2 remains the cold definition of
the same nesting model and Structural Range; the hot entities above are the
v1.3 projections of those definitions, not a second authority.

## Eyes and Scene Graph

Each completed minute updates one shared observation across five enabled
frames. A higher-timeframe frame changes only when a bar for that frame has
completed:

| Frame | Descriptive state |
|---|---|
| 4H | confirmed structure/BOS and external liquidity; directional displacement, efficiency and range position remain explicitly descriptive proxies |
| 1H | confirmed swings/BOS, support/resistance and the legacy v1.2 Mature Balance Range atomic slot; foundation v2 publishes Structural Range separately |
| 15m | bridge-scale candle, structure and liquidity context |
| 5m | displacement episode, raw/displacement-linked FVG, qualified order block; rolling compression remains an explicitly descriptive proxy |
| 1m | candle description, manipulation resolution, exact first return, qualified entry-zone reacceptance, micro BOS and ordered path steps |

Every event carries the common envelope: event identity and kind, timeframe,
`event_time`, `known_at`, semantic version, origin, sequence, provenance, and
evidence. Direction, strength, lifecycle, zone, and specialized transition
clocks are nullable or governed by per-kind contracts rather than universally
required. Event memory retains order and duration, not just current scores. In
particular:

- `event_time` is the market occurrence/source-pivot clock and `known_at` is
  the first completed-data availability clock; no consumer may use an event
  before `known_at`;
- `observed_at` is the compatibility alias of `known_at` and must equal it;
- a neighbouring bar clock is derived from the completed BAR sequence, never as
  `clock + timeframe_interval`: the registered session calendar truncates the
  bucket that closes at the daily maintenance break and restarts one gap later,
  so an arithmetic stride disagrees with the producer at every break;
- event identity and one EventMemory are bound to one `semantic_version`;
- semantic details/evidence are immutable after construction;
- bounded EventMemory is the hot view, while `EventStore` is the
  append-only in-memory audit/research store with deterministic as-of replay
  and a journal API. The formal January bundle persists three research ledgers,
  not the full 471,045-event audit journal.

- first pullback is the first return to one frozen qualified FVG/OB entry
  zone; a mature range is context, not the entry zone itself;
- qualified entry-zone reacceptance requires departure, reclaim, hold and
  explicit failure; manipulation reacceptance remains the separate Group4
  multi-bar lifecycle;
- path sequence is ordered event identity, not swing progression;
- value comes from a mature dealing range, not 4H range position;
- planned entry is a frozen zone price and need not equal the current close.

The scene graph links events, zones, draws, invalidations and competing
interpretations. `GlobalMarketContext` incrementally summarizes structural
authority, cross-scale relations, external draw candidates, path blockers and
identity-bound conflicts from the current graph delta. It does not choose a
draw or action. Focus identifies what the Brain should inspect next while
preserving ambiguity and unknown authority. The eyes and graph cannot choose an
action.

Swing role hierarchy is a reducer-owned append-only role history, not a future
rank written back onto a pivot. Confirmation creates the `micro` assignment; exact
Structural Leg endpoints can add `internal`; Structure Direction source swings
can add `structural`; and Protected Swing assignment can add `external`. The
effective nesting depth follows that highest causally known role, while every
assignment retains its own `known_at` and source IDs.
That legacy depth is structural-role depth. Foundation v2 independently
publishes an append-only geometric parent/child tree using only exact time and
price containment. Its `geometric_depth` never consumes BOS, protected role,
or future importance; neither depth is promoted into the other.

Candidate liquidity also carries executable range membership. When the same
timeframe has an active/mature frozen dealing range, strict interior candidates
are IRL and boundary/outside candidates are ERL; otherwise the role is
unresolved. This is deterministic Eye state, not a target probability. The Eye
does not duplicate path-obstacle ranking: the Brain reuses
`GlobalMarketContext.external_draw_candidates` and directional obstruction
views for its shadow ranking.

The legacy range input above remains the sparse two-sided Mature Balance Range
despite the frozen `DEALING_RANGE_*` compatibility name. Foundation v2 does
not rename it: it publishes independent Structural Range and Balance Range
records plus `x_structural_range`/`x_balance_range`. Delivery Phase Generation
binds the existing classifier to an exact parent Structure Generation. Old
Balance evidence cannot be transferred to Structural Range or persistent
Delivery evidence.

Five enum values are intentionally not canonical v1.2 emissions:
`FVG_TOUCHED`, `FVG_EXPIRED`, `ORIGIN_ZONE_TOUCHED`,
`DEALING_RANGE_EXTENDED`, and `DELIVERY_PHASE_CHANGED`. They are governed as
compatibility aliases or reserved future semantics; the registry-bound emitter
fails closed if they are presented as canonical semantic atoms.

When supplied, MBO adds only execution reality: bid/ask, displayed size/depth,
spread, cost, freshness and fillability. Missing MBO stays missing and is not
replaced by a constant market assumption. Phase 6 completed the registered
June 2024 primary week plus the one preregistered second-week extension. Only
`acceptance_continuation` and `displacement_impact` are admitted to the Phase 7
evidence allowlist. Sweep and MSS are underpowered, the historical FVG
first-concrete-lifecycle proxy (originally labelled `fvg_retest_response`) is
unsupported,
and the pseudo-zone comparison is descriptive only. No Week 3 is authorized.

## Retired typed Brain, DOL and Signal Policy

**Retired on 2026-09-07.** The whole typed-playbook Brain and the Phase 7
shadow layer above it are removed from the repository:

| removed | was |
| --- | --- |
| `brain/core/playbooks.py` | `PlaybookBrain`, the DFP/LSR/FAVR typed playbooks, the Context/EntryEpisode lifecycles, `PlanFeasibility` and Focus binding |
| `brain/core/playbook_registry.py` | the preregistered playbook set and its versioned protocol loader |
| `brain/core/dol_ranking.py` | deterministic shadow-only DOL candidate ranking |
| `brain/core/dol_probability.py` | the path-marginalized shadow-only DOL probability boundary |
| `brain/core/signal_policy.py` | the fail-closed shadow-only Signal Policy assessment |
| `brain/core/shadow_outcome.py` | outcome-blind shadow candidates for one frozen replay |
| `execution/core/trade_intent.py` | the never-submit Trade Intent, whose only entry point consumed a `SignalAssessment` |
| `execution/core/execution_fsm.py` | order FSM v1.5, which existed only to take that intent into custody |
| `shares/core/causal_cases.py`, `market_representation.py`, `case_retrieval.py` | the case-library chain that recorded and retrieved Brain output |
| `shares/scripts/run_continuous_replay.py` | the 7,519-line full-stack replay runner built on `PlaybookBrain` output |
| `brain/configs/playbooks.json`, `dol_probability.json`, `signal_policy.json` | their protocol files |

None of it was published or fitted. `configs/model.json` had
`calibration_artifact: null`, every Signal Policy artifact slot `null`, and
`release_readiness.live_execution_allowed: false`; `path_hypotheses.json`
declared `authority: shadow_only` with equal priors, zero decay and zero
likelihood increments. Removing the layer therefore retires unfitted machinery,
not a validated model.

### What this leaves

`PlaybookBrain` was the only producer of `MarketBelief`. With it gone:

- `brain/core/decision.py` and `risk.py` keep their contracts but have no input.
  Decision reads `MarketBelief.thesis_candidates` and
  `position_management_candidates`, which nothing now fills.
- `MarketBelief` (now `contract/brain/belief.py`) keeps its shape, but its DOL, Signal Policy and
  Trade Intent fields (`dol_rankings`, `dol_probabilities`,
  `dol_candidate_exclusions`, `dol_probability_protocol_fingerprint`,
  `signal_policy_protocol_fingerprint`, `signal_assessments`, `trade_intents`,
  `shadow_signal_rejections`) are removed along with the roughly 200 lines of
  `__post_init__` that validated their shadow-authority scope. What survives is
  the path-diagnostic scope: `path_competition_state`,
  `path_update_records_this_clock`, `path_protocol_status` and `path_authority`.
- `shares/core/engine.py` still imports `brain.core.playbooks`,
  `brain.core.playbook_registry`, `brain.core.dol_probability` and
  `brain.core.signal_policy`, so **it cannot be imported**. Eight test modules
  (135 tests) cannot be collected, and `shares/__init__.py` no longer exports
  `ContinuousSMCEngine`, `ExecutionFSM` or `RiskApprovedTradeIntent`. Rebinding
  that orchestration to a new belief producer is the next piece of work.

### What survives on the belief side

`brain/core/market_belief.py` — renamed from `path_belief.py` — is now a
self-contained real-time path-probability component. Nothing else imports its
`PathKind` vocabulary; `contract/brain/belief.py` takes only
`PathBeliefUpdateRecord` and `PathCompetitionSetState` from it.

It owns one competition set scoped by instrument, market epoch, dominant
authority structure and a shared session horizon, over six mutually exclusive
paths: continuation, deeper retracement, reversal, balance, failed breakout and
residual unknown. Exact-source evidence contributes a configured log-weight
increment once; unchanged evidence is deduplicated, and `correlation_key` is a
global dependency-cluster identity rather than a family-local token. Real
completed bars apply registered decay, terminal paths receive zero probability,
and active survivors are normalized with log-sum-exp. `residual_unknown`
preserves mass for paths the named hypotheses do not represent. Update records
retain the rule, source event IDs, `known_at`, model version and protocol
fingerprint, and the manager checkpoints and restores.

The published model remains neutral — equal priors, zero log-likelihood
increments, zero real-bar decay — and no fitted artifact is admitted. This is
executable accounting with an audit trail, not a calibrated posterior. The
protocol declares `development_unvalidated` / `shadow_only` and
`action_authority_ready: false`, and `load_path_belief_protocol` fails closed if
any of that changes without a separately pinned artifact set.

`path_hypotheses.json` lost its `dol_diagnostic_ranking` block, whose only
loader lived in the retired `dol_ranking.py`. Because the protocol fingerprint
is `sha256` over the whole file, that edit moved it from `d897635c…b0482` to
`61417d9f…3152e0`; `configs/model.json` records the new value, and its
`playbook_registry`, `dol_probability` and `signal_policy` bindings are gone.

The six paths are still a frozen `PathKind` enum with order-sensitive
validation throughout the module. Making that set protocol-driven, so the
module no longer names its own hypotheses, is a separate task.

## Decision, risk and the retired execution boundary

Flat states compare enter, wait and abstain. Open states compare hold, protect,
exit and abstain. An unclear best-versus-second-best advantage resolves to
abstain.

Risk is a separate hard boundary. Entry freezes planned entry, structural
invalidation, selected draw/targets, deadline and maximum risk. Later extrema
cannot rewrite the original thesis. Cost, stale or anomalous data, insufficient
depth, consumed draw, invalid stop provenance and deadline can veto entry.

In the current optional OHLCV simulator, an approved development candidate is
first eligible on the next tradable clock. Its recorded resolution is
`filled`, `pending_right_censored`, or `not_filled_or_expired_next_bar`; the
last label intentionally combines non-fill and expiry. A simulated fill may
create one position, and same-bar stop/target ambiguity is adverse-first.
Position/risk feedback entered the next Brain update.

Decision and Risk are intact as code and untouched by the Brain retirement, but
they have no input: Decision reads `MarketBelief.thesis_candidates` and
`position_management_candidates`, which no module now fills. The simulator in
`execution/core/simulation.py` is likewise intact but imports
`shares/core/engine.py`, which cannot be imported. Both wait on a new belief
producer.

**Order FSM v1.5 was retired on 2026-09-07** together with the Trade Intent it
took into custody. `execution/core/execution_fsm.py` implemented immutable
risk-approved intent custody, command/fact separation,
working/cancelled/expired/rejected/partial/filled transitions, cancel/replace
races, OCO reconciliation, managed stops, aggregate entry/exit reservations,
position conservation, checkpoints and deterministic event replay, and its
independent logic review finished with P0/P1=0. That was engineering validation
only: it never had a broker or network submission path, and it was never the
authoritative executor — the development action path always used Decision/Risk
and the simulator, with `TradeIntent -> RiskApproval -> FSM` a standalone shadow
adapter. It went because `execution/core/trade_intent.py` went, and that went
because its only entry point consumed a `SignalAssessment` from the retired
Signal Policy. Restoring an executor means registering a new intent contract
against the new belief producer, not restoring these files.

**Phase 8 Execution Research was retired on 2026-09-06.** Its seven-entry-method
core, the v1.1/v1.2 protocols, the fail-closed formal runner and result
serializer, their four configs and the run template are removed
(`execution_research.py`, `execution_research_v2.py`,
`execution_research_runner.py`, `run_execution_research_v2.py`,
`configs/execution_research_v{1,2}.json`,
`configs/execution_research_runner_v1.json`, `configs/risk_admission_v1.json`).
It never produced an empirical result: the run manifest stayed at
`template_incomplete_not_authorized_to_run` with no bound non-zero
intent/research-case ledger, no minute source, unset output identities and no
freeze, so nothing measured is lost with it. The Phase 8 research gate is
closed, not passed, and reopening it means registering a new protocol rather
than restoring these files.

The order FSM was unaffected by that retirement — it was a separate deliverable
from the research core — but it has since been retired in its own right, above.

## Phase 9 shadow-live parity boundary

**Retired on 2026-09-06.** The `phase9_shadow_live_v1.3` runner, the
operational v3 contract, the two file-pilot scripts and the three configs
(`shadow_live_v1.json`, `phase9_shadow_operational_v1.json`,
`phase9_current_contract_mapping_v1.template.json`) are removed, together with
the `inputs/`/`outputs/` payloads their bounded runs read and wrote. Nothing in
the current runtime depended on them: `shadow_live.py` and
`shadow_operational.py` were imported only by each other, by those scripts and
by their own tests. `shadow_outcome.py` was a different module and survived that
retirement; it was itself retired on 2026-09-07 with the case-library projection
it served.

An operational pilot therefore has no harness in this repository. Re-opening one
needs a newly frozen runtime/config identity, not a restored file. The rest of
this section is the record of what that harness had established before it was
retired.

The runner drove the production development engine
from one exact completed bar plus causal execution/account evidence. It bound
immutable feed, execution, and account evidence IDs; a frozen logical-to-vendor
instrument mapping; the model and Phase 7/8 protocol fingerprints; complete
registered state digests; and the journal prefix. Duplicate identical feed IDs
were idempotent, conflicting identity/content or causal-order violations failed
stop, and the first failure became terminal evidence.

Its last compact runner checkpoint was `shadow_compact_runtime_v8`, and its
component identity `phase9_shadow_component_digest_v3`. Every run required
the exact `ContinuousSMCEngine` and exact
`NullExecutionGateway` classes, so an external submission attempt failed and was
included in parity state. Cold replay and checkpoint restart compared records,
journal, failure, gateway, and final engine/FSM state exactly. The synthetic
baseline used 36 deterministic minute clocks. A retained schema-v2 June Week-1
cold-start input materialized 6,900 hash-bound, tick-normalized clocks (6,899
real plus one synthetic), flat engineering account snapshots, and zero
approvals/events. A
bounded run checkpointed at 100 clocks, resumed to 200, and matched an
independent 200-clock cold replay exactly. That cross-process rehearsal found
the former hash-order dependence of `scene_revised_edge_ids`; v1.2
canonicalizes the identity set and different `PYTHONHASHSEED` values now yield
the same prefix digest. The 200-row input has a local completion marker; the
full 6,900-clock run was not completed, and this
cold-start file path does not restore the Phase 6 warmup or validate real-feed
disconnect, latency, out-of-order arrival, reconnect, or a durable production
journal. A real-time multi-day no-order pilot has not run, so Phase 9's
operational gate is not passed.
The original receipt binds the earlier path/model identities `5213b3d6…` and
`4214da19…`. A later Foundation-enabled 200-row receipt binds the pre-compact
model and Shadow configuration. Both outputs remain historical evidence for
their exact source snapshots: the compact runtime, action-policy, and Shadow
schema changes require a new frozen identity before another parity claim. See
the historical
[release-verification table](../../eyes/docs/canonical_semantic_foundation_v2.1.md#replay-test-and-empirical-boundary).
That result closed only its bound prefix. A new prefix or complete 6,900-clock
rehearsal must be rematerialized, and neither is a real-time pilot.
The current protocol tests deterministic equality and fail-stop behavior; it
does not yet define acceptance metrics for semantic duplication, relation
churn, evidence-belief consistency, signal expiry, or DOL stability. The
full-week historical path also repeats full-prefix consistency work and keeps
live/cold runners resident together, so capacity closure remains an engineering
task rather than a hardware-only limitation.
A historical read-only 6,900-clock capacity preflight verifies the completed-prefix
marker's bound checkpoint-manifest SHA-256 before extrapolating from the exact
200-clock receipt. It creates no Engine and replays zero clocks. Its recorded
estimate is a 342,420,401-byte checkpoint, 370,193,384 bytes of retained output,
and a 723,016,956-byte peak-working-set lower bound. Historical and current
runtime bindings differ, nonlinear consistency cost remains unresolved, and
the preflight explicitly returns `full_6900_replay_authorized=false`; it does
not close the engineering or operational gate.
The retained input and bounded checkpoint were local `inputs/` and `outputs/`
payloads, never portable repository evidence; both directories were deleted on
2026-09-06 together with the Phase 6–9 gate report that listed their commands,
paths and hashes.

## Replay, audit and validation

Normal trader replay emits only light decision rows, aggregate summaries,
progress, checkpoints and resumable shards. Brain calibration mode adds typed
calibration rows. The prior frozen-packet, sealing, and full-trace stack is
intentionally absent. This does not remove the current in-memory atomic event
store or its journal API.

The frozen January 2024 v1.1 Signal Research artifact is Eye-only and writes a
result, derived Markdown, and three hash-bound research ledgers. It is
diagnostic-only, does not persist the complete audit store, and has no
Brain/MBO/execution or OOS authority. Its exact source-linked nested population
stops at E2; E3–E6 have zero samples, and its matched-control coverage is
24.04%. Those are immutable historical v1.1 results; the completion report that
recorded them was removed on 2026-09-06.

Research protocol v3 is a separately frozen and executed v1.2 development
diagnostic. It
canonicalizes treatment episodes, uses an M5 E1–E6 chain with registered
source-only ancestry or exact normalized-M5-source-bar composition, retains
non-nested proof records, and keeps four control families separate: quiet
rows, same-session non-sweep touches, pseudo-level touches, and forward time
shifts. Context-only links cannot prove either source ancestry or source-bar
composition. Matching is deterministic, maximum-cardinality, forward-only,
and constrained by a caliper, outcome horizon, embargo, exact registered
strata, and no replacement. Every adjacent nested stage retains its explicit
delta when both stages have resolved samples. Paired binary outcomes use exact
McNemar, followed by a frozen-family Holm adjustment in which missing or
underpowered tests enter with p=1. Outcome windows may overlap across distinct
pairs, so those statistics remain descriptive and unvalidated. The complete
r2 run reports E1–E6 episode counts of 1,124 / 317 / 17 / 1 / 1 / 0; quiet and
non-sweep controls match 372 and 62, while pseudo and forward-shift match zero.
It grants no inference, fit, semantic-acceptance, OOS, or trading authority.
The generic research library separately supports deterministic rooted branching
projection from explicit typed link specifications and retains every qualifying
sibling edge. That utility does not alter this frozen linear r2 protocol and is
not itself evidence that any branching relationship is stable.
See the
[current implementation status](current_implementation_status.md).

Phase 6 is a different, completed MBO mechanism study. Its frozen primary week
triggered the preregistered underpowered extension; the final two-week result
passed engineering/data audit and admits only `acceptance_continuation` and
`displacement_impact`. The registered Week 2 extension is consumed and the
result explicitly forbids a further extension. This is development association
only—not causality, OOS, model fit, or trading authority. The final `smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3` manifest
and result was retired on 2026-09-06 with the rest of `experiments/`; the conclusion above is what
survives it, and no document rewrites the identities they carried.

The executable side of both studies was retired on 2026-09-06 as well:
`smc_trader/signal_research.py`, `smc_trader/mbo_mechanism.py`,
`smc_trader/mbo_mechanism_research.py`, the
`scripts/run_semantic_signal_research.py`,
`scripts/run_mbo_mechanism_research.py` and
`scripts/materialize_mbo_mechanism.py` runners, and the three
`configs/research/` manifest templates are gone. What the studies concluded
still binds Phase 7 through the `phase6_*` identities and evidence allowlist in
`brain/configs/path_hypotheses.json`, which `brain/core/market_belief.py` still
validates fail-closed. `execution/core/mbo.py` and
`execution/scripts/materialize_mbo_execution.py` are unaffected: they belong to the
execution-reality path, not to the mechanism study. The replay runner that drove
it (`shares/scripts/run_continuous_replay.py --mbo-execution`) was retired on
2026-09-07 with the Brain it replayed.

**The EntryEpisode causal case library was retired on 2026-09-07.**
`shares/core/causal_cases.py`, `shares/core/market_representation.py`,
`shares/core/case_retrieval.py`, `shares/scripts/query_causal_cases.py`,
`shares/scripts/train_market_representation.py` and
`shares/scripts/evaluate_market_episode_retrieval.py` are gone: `CausalCase`
recorded `PlaybookBrain` EntryEpisodes, so it lost its subject with the Brain.
`shares/core/market_cases.py` is retained — the input-only MarketEpisode stream
has a different physical identity, estimand and outcome-free boundary, and does
not depend on the Brain — but nothing loads it today, and
`shares/configs/market_case_input_profiles_v2.json` is likewise kept as a frozen
preregistration artifact with no current reader.

The rules those protocols established still bind any replacement: a case may
join independently resolved outcomes only after neighbour selection; shared
storage/hash/no-clobber infrastructure cannot weaken either eligibility or
leakage rule; OHLCV is read by prefix boundary from the canonical source; and an
AI comment must be translated into a computable sequence primitive and can never
become an action label.

[`../configs/data_splits.json`](../../configs/data_splits.json) separates
development, calibration, rolling OOF and sealed OHLCV, plus MBO development
and sealed execution holdout. It binds causal artifacts and their manifests by
exact SHA-256.
