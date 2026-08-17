# Causal continuous architecture

Schema: **1**

## Runtime flow

```text
newly completed 1m OHLCV bar
        ↓
causal 1m / 5m / 15m / 1H / 4H aggregation
        ↓
descriptive primitives + ordered event memory
        ↓
Temporal Market Scene Graph + GlobalMarketContext
        ↓
playbook-neutral OpenMarketThesis + FocusState
        ↓
independent root-specific DFP / LSR projections
        ↓
long-lived Context Thesis + short-lived Entry Episodes
        ↓
Belief(t) = update(Belief(t-1), Observation(t), SceneDelta(t))
        ↓
typed DFP / LSR / FAVR episodes and phases
        ↓
enter / wait / hold / protect / exit / abstain utility comparison
        ↓
independent structural, cost, deadline, data and fillability vetoes
        ↓
next-bar conservative execution → position feedback
```

There is one runtime path. Replay calls the same engine used for incremental
operation; it does not reproduce reader → observer → Brain → Decision → Risk in
a second implementation. Multi-timeframe bars update only when complete.

## Eyes and Scene Graph

Each completed minute updates one shared observation across five enabled
frames. A higher-timeframe frame changes only when a bar for that frame has
completed:

| Frame | Descriptive state |
|---|---|
| 4H | confirmed structure/BOS and external liquidity; directional displacement, efficiency and range position remain explicitly descriptive proxies |
| 1H | confirmed swings/BOS, support/resistance and typed dealing range; rolling acceptance/rejection remains an explicitly descriptive proxy |
| 15m | bridge-scale candle, structure and liquidity context |
| 5m | displacement episode, raw/displacement-linked FVG, qualified order block; rolling compression remains an explicitly descriptive proxy |
| 1m | candle description, manipulation resolution, exact first return, qualified entry-zone reacceptance, micro BOS and ordered path steps |

Every semantic event carries identity, formation/confirmation/invalidation
clocks, lifecycle, direction, strength and source IDs. Event memory retains
order and duration, not just current scores. In particular:

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

MBO adds only execution reality: bid/ask, displayed size/depth, spread, cost,
freshness and fillability. Missing MBO stays missing and is not replaced by a
constant market assumption.

## Typed Brain

Exactly three mechanisms remain registered:

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

FAVR remains parked whenever a mature range and value cannot be established
with natural market evidence. A general rejection is not a FAVR substitute.

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

`thesis_candidates` is the sole live action identity set. The six
playbook-direction slots are read-only summaries projected from those roots and
carry `summary_source_candidate_id`; they keep no independent prior and cannot
authorize an action. Explicit shadow diagnostics read the root-specific
candidate and its common plan-feasibility result directly; no parallel
thesis-comparison object is stored in the live `MarketBelief`. Focus binds to
the dominant root candidate and is recomputed only for a semantic graph
revision, candidate/phase/terminal change, or related conflict/ambiguity.
Unexplained episodes are offline diagnostics and never override Decision.

Each playbook supplies the allowed invalidation and draw identities. The shared
`PlanFeasibility` view only validates their entry/stop/target geometry, remaining
path, obstruction and deadline; it does not invent a stop or target. Risk remains
the final independent veto. For DFP, the primary target may be re-evaluated only
before Risk approval; Risk approval freezes it for order and position lifecycle
management. Delivery calibration resolves against that primary target before
frozen invalidation/deadline, not against the farther H4 terminal draw.

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
executable and plan-valid child freezes execution ownership together with its
entry, original sweep stop, primary target, deadline, route, trigger and first-
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

## Decision, risk and execution

Flat states compare enter, wait and abstain. Open states compare hold, protect,
exit and abstain. An unclear best-versus-second-best advantage resolves to
abstain.

Risk is a separate hard boundary. Entry freezes planned entry, structural
invalidation, selected draw/targets, deadline and maximum risk. Later extrema
cannot rewrite the original thesis. Cost, stale or anomalous data, insufficient
depth, consumed draw, invalid stop provenance and deadline can veto entry.

An approved order becomes eligible only at the next tradable clock. An attempt
must become filled, expired or cancelled. A fill creates one position, which
ends completed, invalidated or at a data boundary. Same-bar stop/target
ambiguity is adverse-first, and position/risk feedback enters the next Brain
update.

## Replay, audit and validation

Normal historical replay emits only light decision rows, aggregate summaries,
progress, checkpoints and resumable shards. Brain calibration mode adds typed
calibration rows. The prior frozen-packet, sealing, full-trace and audit-record
stack is intentionally absent from the current runtime.

The optional [EntryEpisode causal case library](causal_case_library.md)
consumes this same replay without a second replay loop. It writes sparse
decision-time revisions and later Shadow-derived labels to separate,
hash-bound Arrow streams; OHLCV remains in the canonical source and is read by
prefix boundary. Any AI comment must still be translated into a computable
sequence primitive and can never become an action label.

[`../configs/data_splits.json`](../configs/data_splits.json) separates
development, calibration, rolling OOF and sealed OHLCV, plus MBO development
and sealed execution holdout. It binds causal artifacts and their manifests by
exact SHA-256.
