# Causal continuous architecture

Document schema: **1**

## Current implementation boundary

This document describes the incremental/development data path and later
subsystem ownership. Live mode currently fails closed.
The current Eye semantic identity is `smc_semantics_v1.3`; its exact executable
and reserved surface is recorded in the
[semantic specification](smc_semantic_specification_v1.md). The Phase 2–4
producer/state surface is executable within that frozen boundary; round-two
review distinguishes this from completion of every target definition.
The additive `smc_semantic_foundation_v2.1` projection now completes the
registered geometry, generation, lifecycle, relation, transition, first-
reinteraction, ancestry, and factual-outcome layer over those immutable facts.
See the
[foundation specification](refactor/canonical_semantic_foundation_v2.1.md).
It reuses the existing detectors and immutable store; it is not a parallel Eye
or a new action path.
The current implementation-versus-plan matrix is recorded separately in the
[current implementation status](refactor/current_implementation_status.md), and
the registered atomic definitions in the
[v1.3 preregistered semantics](refactor/preregistered_semantics_v1_3_2026-08-31.md).
The historical Phase 2–5, Phase 6–9, DOL/Belief/Temporal and round-two-review
documents were removed on 2026-09-06; their still-current conclusions live in
those two documents. Executable coverage is never treated as predictive
validation.
The numbered implementation modules are no longer runtime owners: Zone logic
lives in `smc_trader/zone.py`, and Range Auction logic lives in
`smc_trader/range_auction.py`. `group3.py` and `group4.py` are unexported,
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
independent root-specific DFP / LSR projections
        ↓
long-lived Context Thesis + short-lived Entry Episodes
        ↓
Belief(t) = update(Belief(t-1), Observation(t), SceneDelta(t))
        ├── shadow six-path beliefs + DOL ranking
        ├── fitted-artifact-only DOL probability/no-target layer
        ├── fail-closed Signal Policy + Trade Intent projection
        └── retained typed action candidates
        ↓
typed DFP / LSR / FAVR episodes and phases
        ↓
enter / wait / hold / protect / exit / abstain utility comparison
        ↓
independent structural, cost, deadline, data and fillability development vetoes
        ↓
optional one-next-bar conservative simulation → position feedback

separate research/shadow paths (no order submission):
Trade Intent → Phase 8 execution evaluator / immutable order FSM v1.5
exact bar + execution + account evidence → Phase 9 parity journal
```

### Reasoning responsibilities

The model uses a one-way reasoning chain. Each layer answers a different
question and may consume only already-known outputs from the layer above it:

| Reasoning mode | Components | Question answered | Prohibited authority |
|---|---|---|---|
| Data fact admission | `io`, `market_clock`, `CausalMarketReader` | Which completed clocks and prices are legally knowable now? | No structure, probability, or action interpretation. |
| Atomic observation | `CausalObserver`, Group 1–5 trackers, `EventStore` | Which preregistered v1.2 market facts occurred, and from which exact sources? | No rewriting history and no trade decision. |
| State and relation projection | timeframe/session reducers, `RelationResolver`, Foundation v2, `MarketSnapshotPublisher` | What is the deterministic current state, lifecycle, geometry, and cross-frame relation? | Foundation is an additive v1.2 projection, not a new detector or full-stack v2 authority. |
| Competing-hypothesis reasoning | Scene Graph, `PlaybookBrain`, path belief, DOL, Signal Policy | Which still-falsifiable path or target is better supported by admitted evidence? | With no fitted/admitted artifacts, outputs remain neutral or shadow-only. |
| Action constraint and validation | Trade Intent, Decision, Risk, execution FSM, simulator, Phase 7–9 runners | Is an already-described candidate allowed to become an intent, simulated action, or no-order audit fact? | No retroactive semantic change, broker authority, profit claim, or sealed-OOS access. |

The legacy `Decision`/`Risk`/sequential simulator path is the sole current
runtime action authority under `legacy_decision_risk_compat`. The newer
`TradeIntent`/risk-approval/execution-FSM contracts remain separate research and
shadow consumers; Engine disables TradeIntent projection in compatibility mode
and rejects a non-zero prefilled intent before Decision/Risk. This prevents two
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
cost/fillability score, and `model.py` owns the inert not-evaluated value
beside the `ExecutionObservation` it constructs, so the Eye transports an
execution observation without deriving one; `ContinuousSMCEngine._score_execution`
derives it. `tests/test_eye_module_boundary.py` asserts that no Eye module
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
calls that same engine. The frozen Phase 5 diagnostic is deliberately a
separate Reader + Eye-only runner; it does not construct Brain, Decision, Risk,
MBO, or execution. Multi-timeframe bars update only when complete.

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

## Existing Development Typed Brain

Exactly three legacy typed playbooks remain registered; they were not replaced
or physically removed:

1. DFP: directional structure/draw → displacement zone → first pullback →
   exact rejection, held reacceptance or aligned 1m micro BOS; an accepted H1
   continuation BOS is supporting evidence rather than a duplicate hard gate.
   The H4-timeframe draw establishes thesis and terminal context (external to
   the M5 setup, without a new structural-rank gate); the executable primary
   target is a visible unconsumed registered structural-liquidity kind before
   that draw and any relevant hard barrier, with planned and remaining path
   each at least 1R;
2. LSR: visible liquidity pool → sweep/failed outside acceptance → one frozen
   opposite-displacement reversal Context → multiple independent
   displacement-linked FVG/OB Entry Episodes → each zone's own first pullback
   and three-way typed trigger; an accepted M5 OPPOSED MSS strengthens the
   Context but is neither a mandatory gate nor a separate entry mechanism;
3. FAVR: mature accumulation/dealing range → failed outside auction → re-entry
   and displacement back inside → first pullback/trigger → midpoint or opposite
   boundary liquidity.

DFP and LSR now serve only as typed setup and entry-episode generators; neither
owns the six-path market probability model. FAVR remains parked whenever a
mature range and value cannot be established with natural market evidence. A
general rejection is not a FAVR substitute. This keeps proven episode and
position-management semantics without preserving competing probability
authority.

Each hypothesis exposes separate typed dimensions:

- `thesis_strength`;
- deterministic `sequence_progress`;
- `location_quality`;
- `entry_readiness`;
- `delivery_quality`;
- `uncertainty`.

The common phase vocabulary is:

`inactive → forming → armed → waiting_location → waiting_trigger → executable
→ entered → delivering/weakening → completed/invalidated`

Episode, terminal and rearm semantics prevent a later event from silently
rewriting the active thesis. A stable evidence revision is assimilated once,
not repeatedly every minute.

The Scene Graph first emits identity-bound, playbook-neutral market theses.
These theses are descriptive analysis candidates and never possess action
authority by themselves. Their `ThesisEvidenceState` incrementally records
new support, opposition and the forming/active/weakening/invalidated lifecycle;
an unchanged evidence revision is not assimilated again. Every compatible thesis root is evaluated
independently by DFP or LSR. Its root-specific typed projection becomes an
action candidate only when the exact-root graph binding, causal hard gates,
frozen entry/invalidation/draw/target/deadline and delivery path are complete.
An unmatched or incomplete thesis remains available to Focus and unexplained-
episode diagnostics but cannot enter. FAVR projections remain parked.

`PRECEDES` remains a diagnostic temporal-neighborhood edge (the current
producer still uses an unregistered 60-minute construction window). Generic
graph inspection may display it, but causal/open-thesis closure and action
connectivity use explicit relation allowlists that exclude `PRECEDES`,
`RESPONDS_TO`, and path-blocking edges. A temporal neighbor therefore cannot
manufacture an action path or FAVR eligibility. A future temporal research
definition must freeze its identity, completed-bar window, same-clock policy,
and observed delta before it may enter any modeled relation.

`thesis_candidates` is the sole development action-candidate identity set. The
six
playbook-direction slots are read-only summaries projected from those roots and
carry `summary_source_candidate_id`; they keep no independent prior and cannot
authorize an action. Explicit shadow diagnostics read the root-specific
candidate and its common plan-feasibility result directly; no parallel
thesis-comparison object is stored in the development `MarketBelief`. Focus
binds to the dominant root candidate and is recomputed only for a semantic graph
revision, candidate/phase/terminal change, or related conflict/ambiguity.
Unexplained episodes are offline diagnostics and never override Decision.

Each playbook supplies the allowed invalidation and draw identities. The shared
`PlanFeasibility` view only validates their entry/stop/target geometry, remaining
path, obstruction and deadline; it does not invent a stop or target. Risk remains
the final independent veto. For DFP, the primary target may be re-evaluated only
before Risk approval; Risk approval freezes it for order and position lifecycle
management. Once a validated calibration artifact is installed, delivery
calibration is intended to resolve against that primary target before frozen
invalidation/deadline, not against the farther H4 terminal draw. The current
configuration has no such artifact.

Candidate prior, terminal and rearm state are isolated by thesis root. Once an
approved candidate owns a position, its unique frozen root projection remains
available for position management until completion, invalidation or boundary
exit even if the descriptive root leaves the current open-thesis set; this
retention cannot authorize a second entry.

The lifecycle is deliberately split. `ContextThesisState` freezes the market
epoch, higher-timeframe authority identity, direction, context/terminal draw
and structural invalidation. It may remain active across an interval with no
current entry child. `EntryEpisodeState` owns one local mechanism root, zone,
path, first pullback, frozen first trigger, plan and short deadline. Multiple
independent episodes may be children of one Context; one child's terminal does
not weaken or close the Context, while a causal Context invalidation or context
draw delivery closes all current/dormant/position children. Discovery-root
visibility is not itself an Entry Episode lifetime signal: if the current
observation uniquely resolves the identical frozen setup, location and active
path, that same root-specific candidate continues typed gate evaluation and may
remain in the action map. It cannot borrow a zone, path or trigger from another
root. If any frozen identity is missing or ambiguous, it moves to
`retained_episode_candidates`, which is resolution-only and excluded from
Focus, Decision entry candidates and the six-slot projection. Explicit
structure/draw/deadline/boundary terminal evidence still closes it before any
action evaluation. `HypothesisBelief.entry_path_id` and
`EntryEpisodeState.entry_path_id` carry the exact frozen zone-return path even
before a trigger or plan exists; setup/root identity is never used as a path
fallback. The bounded `child_episode_ids` field contains current
children only; replay diagnostics own historical counts. DFP Context lifetime
is governed by its frozen structure/draw and market epoch rather than the local
plan clock, while each DFP episode freezes its own plan deadline. LSR retains a
local Context horizon because its sweep is itself the mechanism root.

DFP freezes terminal authority by semantic role rather than by a broad source-ID
closure. The exact H4 structure, protected raw swing and context draw can close
the Context; current H4 high/low projections can revise supporting evidence but
cannot close an Entry Episode, the parent Context or a managed position. A
local zone/path/trigger terminal applies only to its owning child. Scene Graph
connectivity still records projection provenance, but shared provenance does
not grant a projection the terminal authority of its source structure.

LSR separates its parent reversal mechanism from local entry opportunity.
`FrozenLSRContext` carries the exact manipulation, pool-path protocol,
reacceptance and displacement clocks, sweep extreme and direction even after a
completed Group5 path is compacted from the current observation. Each eligible
zone gets a stable Episode identity derived from manipulation root,
displacement, zone and direction. Siblings never share first pullback, trigger,
entry path or terminal state. A failed child leaves the Context and other
children alive; a Context terminal cascades once. The first uniquely timed
executable and plan-valid child freezes plan/action-candidate ownership together
with its entry, original sweep stop, primary target, deadline, route, trigger and first-
executable clock; only current remaining-path, target-visibility and hard-
obstruction diagnostics remain dynamic. A same-clock tie fails closed. Before
or after terminal resolution, no newly visible intermediate liquidity may
rewrite that owner route; completed or invalidated phase keeps the frozen plan
and first trigger as non-actionable historical custody. Before
that owner freeze, the target in a child's route is provisional; its
consumption cannot complete an Episode that is still waiting for its own first
pullback or same-zone trigger. Once ownership and the complete plan are frozen,
delivery of that exact primary target retains its existing completion authority.
Context termination, position management, Decision and Risk semantics are
unchanged. Accepted outside rejects a not-yet-established Context, but cannot
retrospectively erase a frozen reacceptance and reverse displacement. Risk
validates the frozen parent
provenance and the exact child location/path independently; it does not infer
the parent from a zone-specific setup ID. No threshold, target geometry or hard
gate is relaxed.

Child discovery is gated by the frozen parent lifecycle. A closed, terminal,
deadline-expired, or formerly known but absent LSR Context cannot spawn a later
zone Episode; retained children remain available only for causal settlement.
When multiple live Contexts claim one physical location or entry path, Episode
materialization and action publication both fail closed while the Contexts and
diagnostic evidence remain visible. Rearm requires a new manipulation root.

LSR source authority is tiered without deleting 1m observations: H4/H1 or
typed external/intermediate liquidity may establish Tier A; connected 15m/5m
internal liquidity may establish Tier B; isolated or nested 1m internal
liquidity is Tier C trigger/refinement evidence only. A mature balance range is
rare optional context for LSR, never its hard gate, and FAVR remains parked.

## Phase 7 shadow hypotheses, DOL probability and intent

The neutral projection is the single per-clock authority for
`OpenMarketThesis`: it updates `GlobalMarketContext`, builds the thesis tuple
once, and stores that exact tuple in `NeutralMarketState`. The full Engine hands
the precomputed context to `PlaybookBrain`; the Brain validates its clock,
revision, epoch, root uniqueness, and canonical order and reuses it. Standalone
Brain use performs the same construction once as a fallback. Terminal routing
may change candidates and lifecycle state, but cannot rebuild or replace the
neutral thesis tuple. Checkpoint restore requires Brain and Neutral views to
agree.

The same `PlaybookBrain` owns one current Hypothesis Manager competition set
scoped by instrument, market epoch, dominant authority structure, and a shared
session horizon. Its mutually exclusive paths are continuation, deeper
retracement, reversal, balance, failed breakout, and residual unknown.
Exact-source evidence contributes a configured log-weight increment once;
unchanged evidence is deduplicated, and `correlation_key` is a global
dependency-cluster identity rather than a family-local token. An admitted
model cannot multiply cross-family contributions from one cluster unless one
registered joint or history-conditioned contribution represents it. The Brain
rejects a likelihood-enabled path protocol without a complete externally pinned
artifact set. Because the current admission schema has no registered dependency
resolver, fitted runtime facts are conservatively scoped to one unresolved
competition-set cluster and cannot accumulate independent multipliers. Real
completed bars apply registered decay,
terminal paths receive zero probability, and active survivors are normalized
with log-sum-exp. `residual_unknown` preserves mass for paths not represented by
the named hypotheses. Update records retain the rule, source event IDs,
`known_at`, model version, and protocol fingerprint. Only exact `MarketEvent`
IDs resolvable in the current Observation are admitted; unresolved
context/entity identities fail closed. `MarketBelief` carries the current
competition set and this-clock records, not a complete persisted evidence
journal.

The runtime now uses those invalidation and realization contracts inside the
current authority scope. The `PlaybookBrain` adapter admits only
production-shape, exact-parent facts:
frozen-authority Qualified BOS/MSS may falsify paths; BOS
Acceptance/Rejection, graph-connected lower-frame Acceptance, and the legacy
range-activation fact may settle a winner. A same-clock
terminal precedes a conflicting winner, and local EntryEpisode failure has no
global path authority. The published probability model is still neutral—equal
priors, zero log-likelihood increments, zero real-bar decay—and no fitted
artifact is admitted. Equal-logit subtraction is explicitly softmax-invariant,
not effective decay. Earlier per-path expiry, hazard, or prior reversion
remains unavailable unless a separately frozen temporal definition/artifact is
admitted; the shared horizon is unchanged.
The read-only Phase 7 readiness checker verifies the frozen Phase 6 source and
all three compact ledgers, inspects 7,381 rows, and reports 13 blockers with
`ready_for_offline_fit=false` and `artifacts_written=[]`. It does not fit a
likelihood, temporal, path-calibration, or DOL-calibration artifact. The June
Week-4 temporal/branching design is separately frozen and passes its strict
identity validator without opening market data; execution bindings,
materialization, replay, and results remain absent and unauthorized.
Authority/scope rollover can replace the current competition set, but no
registered fact yet classifies and archives the retired scope as realized,
invalidated, or expired. The Brain input is still
`MarketObservation + SceneGraph/SceneDelta`, not solely
`MarketSnapshot + events`; both gaps belong in the eventual vertical migration
rather than a second Brain.
The range-driven balance mapping remains shadow-only. Foundation v2 now keeps
Structural and Balance Range identities/locations separate, but no empirical
artifact admits either as path evidence. DOL target-map compatibility consumes
the foundation-active liquidity lifecycle where present, including a distinct
rearmed interaction generation, while preserving the legacy view only for
sources not yet managed by the foundation. This is still a candidate/ranking
adapter, not a fitted DOL probability model.

The Bayesian Belief Updater is executable accounting, not a calibrated
Bayesian posterior. The configuration declares
`normalized_development_weights_not_calibrated_posterior` and
`development_unvalidated`/`shadow_only`; the increments have not been fitted
as likelihood ratios. The engine binds exact path, DOL-ranking,
DOL-probability, and Signal Policy protocol fingerprints, and live/action
consumers remain isolated.

DOL ranking joins only exact visible external-draw candidates to one path. For
each direction it evaluates target distance, strength, timeframe, structural
rank, age, and hard/soft obstructions lying strictly between current price and
target. Target-self identity, shared-source, and co-located obstructions are
excluded. A softmax normalizes candidate weights within the diagnostic cohort,
and the output also records the linked normalized path-hypothesis weight and
their product as diagnostic joint quality. The separate DOL-probability layer
can marginalize across paths and retains an explicit no-target alternative
instead of forcing a draw. Neither the Brain nor the standalone marginalizer
publishes a DOL-probability result while an exact fitted/admitted artifact bound
to the active path protocol/model is absent; only the diagnostic ranking weight
remains visible.

Signal Policy assessment and a standalone shadow Trade Intent projector retain
separate exact artifact admission. `legacy_decision_risk_compat` Engine updates
do not invoke that projector. Without fitted/admitted path-likelihood,
DOL-probability, and outcome-calibration artifacts, assessment also fails
closed and the belief intent map remains empty. It does not fall back to a legacy
`HypothesisBelief.probability`, and none of these fields changes Decision,
Risk, an order, or a position.

## Decision, risk and Phase 8 execution boundary

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
Position/risk feedback enters the next Brain update.

The retained simulator above is not the Phase 8 Executor. Phase 8 separately
implements immutable order FSM v1.5 with risk-approved intent custody, command/
fact separation, working/cancelled/expired/rejected/partial/filled transitions,
cancel/replace races, OCO reconciliation, managed stops, aggregate entry/exit
reservations, position conservation, checkpoints, and deterministic event
replay. Venue facts are checked against submit/request causality, and order and
position states remain separate from market state. The independent logic review
finished with P0/P1=0. This is engineering validation only: the FSM has no
broker/network submission path and is not evidence of fill quality. It is also
not yet the authoritative end-to-end executor: the current development action
path still uses Decision/Risk and the retained simulator, while the exact
`TradeIntent -> RiskApproval -> FSM` handoff remains a standalone shadow
adapter.

Execution Research is also separate. Its implemented core compares seven
same-intent entry methods using causal BBO, a Phase 6 MBO displayed-volume
proxy, and OHLC adverse-first resolution; invalid, stale, crossed, reset,
synthetic, missing, or fractional-capacity observations are censored rather
than relabeled as failures. Exact logical/vendor instrument mapping,
tick/point value, whole contracts, source-intent identity, and protocol hashes
are required. Protocol v1.1 separates entry GTT from a later analysis horizon,
forbids new fills after GTT while allowing an existing position's observation
to continue, cancels an unfilled remainder if the target resolves first, and
does not suppress a complete primary implementation-shortfall pair merely
because a secondary realized-spread/path metric is censored. Stop/invalidation
and target prices that are off the frozen tick grid reject the research intent.
The v2 protocol requires exact pre-outcome semantic provenance for every method
price and registers wait/cancel/stop/target variants. A variant whose required
future-frozen cancel state or entry-zone boundary is absent is censored by its
registered evaluability rule; it is never backfilled. The formal runner and
result serializer exist, but the current run manifest remains
`template_incomplete_not_authorized_to_run`: no non-zero intent/research-case
ledger or minute source is bound, output identities are unset, and the manifest
is not frozen. Therefore no empirical result exists and the Phase 8 gate is not
passed.

## Phase 9 shadow-live parity boundary

The `phase9_shadow_live_v1.3` runner drives the production development engine
from one exact completed bar plus causal execution/account evidence. It binds
immutable feed, execution, and account evidence IDs; a frozen logical-to-vendor
instrument mapping; the model and Phase 7/8 protocol fingerprints; complete
registered state digests; and the journal prefix. Duplicate identical feed IDs
are idempotent, conflicting identity/content or causal-order violations fail
stop, and the first failure becomes terminal evidence.

Its current compact runner checkpoint is `shadow_compact_runtime_v8`, and its
component identity is `phase9_shadow_component_digest_v3`. Every run requires
the exact `ContinuousSMCEngine` and exact
`NullExecutionGateway` classes, so an external submission attempt fails and is
included in parity state. Cold replay and checkpoint restart compare records,
journal, failure, gateway, and final engine/FSM state exactly. The synthetic
baseline uses 36 deterministic minute clocks. A retained schema-v2 June Week-1
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
[release-verification table](refactor/canonical_semantic_foundation_v2.1.md#replay-test-and-empirical-boundary).
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
[current implementation status](refactor/current_implementation_status.md).

Phase 6 is a different, completed MBO mechanism study. Its frozen primary week
triggered the preregistered underpowered extension; the final two-week result
passed engineering/data audit and admits only `acceptance_continuation` and
`displacement_impact`. The registered Week 2 extension is consumed and the
result explicitly forbids a further extension. This is development association
only—not causality, OOS, model fit, or trading authority. The final
[manifest](../experiments/manifests/smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.yaml)
and [result](../experiments/results/smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.json)
remain the evidence authority; no document rewrites their identities.

The optional [EntryEpisode causal case library](causal_case_library.md) and
input-only [MarketEpisode stream](market_episode_case_library.md) consume this
same replay without a second replay loop. `CausalCase` may join independently
resolved outcomes only after neighbour selection; `MarketEpisode` has a
different physical identity, estimand, and outcome-free boundary. They remain
separate protocols while sharing canonical storage/hash/no-clobber publication,
immutable vector storage, and cosine/OOD mechanics. Shared infrastructure
cannot weaken either eligibility or leakage rule. OHLCV remains in the
canonical source and is read by prefix boundary. Any AI comment must still be
translated into a computable sequence primitive and can never become an action
label.

[`../configs/data_splits.json`](../configs/data_splits.json) separates
development, calibration, rolling OOF and sealed OHLCV, plus MBO development
and sealed execution holdout. It binds causal artifacts and their manifests by
exact SHA-256.
