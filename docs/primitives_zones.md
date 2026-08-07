# Displacement-created zone primitives

The executable definition is `configs/primitives_zones.json`.

Current protocol: `3.2.0-group3.4`. Status:
`implementation_complete_finite_real_replay_passed_ob_failure_coverage_sparse`.

This contract freezes the causal meaning of the 5m fair value gap and order
block. Both are descriptive location entities. Neither is an entry signal,
playbook, stop, target, probability adjustment, or profitability claim.

The executable companion is
`configs/primitives_zones.json`. If this document and that file ever
disagree, development stops until both are versioned together. Implementations
must bind the exact file hash in replay and checkpoint provenance.

## Common input and clock

- Group 3 consumes ordered `real_completed` 5m candles on the frozen `0.25`
  tick grid. Displacement and structure/BOS must use that same tick size.
  Forming candles, partial 5m buckets and synthetic candles
  cannot create, touch, age, mitigate or fail an entity.
- All source candles must have the same symbol and `instrument_id` and belong
  to one uninterrupted semantic epoch.
- A new 1m decision observation may carry the unchanged current Group 3
  snapshot, but Group 3 state advances only when a new real completed 5m
  candle arrives.
- Event times are knowledge times. A source candle may be older, but an FVG is
  not known before `c3.end`, and an order block is not known before the
  confirming BOS `resolved_at`. No event time is backfilled.
- Unknown gaps, data anomalies, tick-size disagreement and contract changes
  terminate nonterminal entities, expose the terminal transitions, then reset
  candidate windows. Scheduled same-contract closures and synthetic intervals
  clear unfinished windows but do not age or rewrite already-created zones.
- For a paired candle boundary, a censored displacement source must belong to
  the retained pre-boundary Group 3 identity. `contract_change_reset` requires
  that retained prior identity and a boundary candle with a different symbol
  or `instrument_id`. Every other paired boundary reason requires the candle
  to retain the old identity; a foreign candle cannot masquerade as a gap,
  data anomaly, registered closure or synthetic interruption. A contract
  change without provable prior identity fails closed.
- Cold attachment cannot infer a missing source transition from a current
  snapshot. If the retained prefix does not contain the required three bars,
  displacement episode, frozen order-block search window and BOS transition,
  it waits for a fresh setup rather than backfilling one.
- Exact retry is idempotent. A failure after any paired reducer may have
  advanced makes the observer unusable; resume from the last checkpoint.

For each real completed 5m candle, processing order is:

1. validate candle, contract, tick grid and boundary provenance;
2. admit the candle to the configured displacement and structure/BOS reducers;
3. update only FVG and order-block entities created before this candle;
4. form new entities whose complete evidence first exists at this candle;
5. emit one immutable snapshot and ordered transition set.

This order is mandatory. In particular, a creation candle never tests,
mitigates or invalidates the entity that it creates.

## Qualified displacement

Group 3 does not define another impulse score. It consumes the exact typed
5m displacement episode produced by the frozen upstream protocol.

`started` is not qualified. The episode must be `active`, its direction must
match the new entity, and its exact `entity_id` and `protocol_hash` are stored.
Activation on the current completed bar is causal and allowed; activation on
a later bar never backfills an earlier FVG or order block.

Membership must be proved by the episode's admitted candle provenance.
Matching only a time interval or a legacy `impulse_strength` value is
insufficient.

## Fair value gap

### Formation

Let `c1`, `c2`, and `c3` be three consecutive real completed 5m candles.
Formation is evaluated only after `c3` and the same-bar upstream displacement
update are complete.

| Direction | Strict formation test | Lower bound | Upper bound | Near edge | Far edge |
| --- | --- | --- | --- | --- | --- |
| Long | `c3.low_ticks > c1.high_ticks` | `c1.high` | `c3.low` | `c3.low` | `c1.high` |
| Short | `c3.high_ticks < c1.low_ticks` | `c3.high` | `c1.low` | `c3.high` | `c1.low` |

Equality is not a gap. The midpoint is the arithmetic mean of the frozen
bounds; it may lie between ticks and is descriptive only.

Every strict three-bar geometry is materialized as one FVG with qualification
`raw`. At `c3.end`, it becomes `displacement_linked` only when a
same-direction displacement is `active`, `c2` is an exact admitted member of
that episode, and:

`displacement.started_at <= c2.end <= c3.end <=
displacement.prefix_last_admitted_at`.

The displacement `active_at` must be no later than `c3.end`. A linked FVG
stores the displacement identity, clocks and prefix commitment. A geometric
gap without that source remains visible as `raw`; it cannot register a Group5
entry location. A displacement that activates after `c3.end` never upgrades
or backfills the frozen qualification.

`formed_at` and `confirmed_at` both equal `c3.end`. Identity includes the
Group 3 protocol hash, contract, direction and all three candle IDs; it does
not include qualification or displacement identity. Direction, boundaries,
midpoint, formation ATR and formation-time qualification never change.

### Lifecycle

Lifecycle is:

`open → partial|mitigated|invalidated`

`partial → mitigated|invalidated`

`mitigated` and `invalidated` are terminal.

Only a later real completed 5m candle can change the lifecycle.

For a long FVG:

- `invalidated` when the close is strictly below the far edge;
- otherwise `mitigated` when the low reaches or crosses the far edge;
- otherwise `partial` when the low is strictly inside the gap;
- a low equal to the near edge leaves it `open`.

For a short FVG, use the exact mirror:

- `invalidated` when the close is strictly above the far edge;
- otherwise `mitigated` when the high reaches or crosses the far edge;
- otherwise `partial` when the high is strictly inside the gap;
- a high equal to the near edge leaves it `open`.

A close equal to the far edge is mitigation, not invalidation. On one bar the
priority is `invalidated`, then `mitigated`, then `partial`; at most one
transition is emitted. Partial revisions may increase a causal maximum
penetration fraction, but cannot move any boundary or overwrite the first
`partial_at`.

Unknown-gap, contract, data and tick-grid boundaries use `invalidated` with
the registered reason.

## Order block

### Causal qualification

An order block requires both:

- one same-direction 5m displacement episode that remains `active`; and
- one newly observed same-direction 5m BOS whose lifecycle is `confirmed`.

Their clocks must satisfy:

`displacement.started_at <= bos.resolved_at <=
displacement.prefix_last_admitted_at`.

No BOS scope is excluded. `local`, `continuation`, and `opposed` remain
descriptive source fields; they are not strategy filters in this contract.
The BOS arrives in a contract-bound envelope carrying the exact structure
protocol SHA and tick size. Group 3 rejects cross-contract, cross-protocol,
future-clock, or mismatched-grid provenance before changing state.

The order block is created at `bos.resolved_at`, after the confirming candle
has been admitted to both upstream reducers. Its `formed_at`,
`confirmed_at`, and `created_at` all equal that clock. It stores the exact
displacement ID and protocol, and the exact BOS protocol, ID, target swing,
source structure, scope and resolution clock.

### Frozen source search

When the displacement seed completes, freeze a window containing up to the
64 immediately preceding, contiguous, same-contract real completed 5m
candles. The window ends no later than the seed start and is never expanded
later.

- For a long displacement, a candidate must have
  `close_ticks < open_ticks`.
- For a short displacement, a candidate must have
  `close_ticks > open_ticks`.
- Doji candles are not candidates.
- Select the candidate with the latest unique end clock.

The selected candle's full low-to-high range, including wicks, is the order
block. For long, the high is the near edge and the low is the distal edge.
For short, the low is the near edge and the high is the distal edge. The
midpoint is descriptive.

The candidate is frozen at the seed. Later bars cannot replace it with a more
convenient candle. If the search finds no candidate, if provenance is
incomplete, or if the displacement/BOS/candidate association is not unique,
no order-block entity and no failed placeholder are emitted.

Identity includes the Group 3 protocol hash, contract, direction, selected
candle ID, displacement ID and BOS ID. All geometry and source fields are
immutable.

### Lifecycle

Lifecycle is:

`created → untested|mitigated|failed`

`untested → mitigated|failed`

`mitigated` and `failed` are terminal.

The confirming BOS bar only creates the entity; it never tests it. On the
first later real completed 5m candle:

- a strict close through the distal edge produces `failed`;
- otherwise a bar whose range intersects the frozen full range produces
  `mitigated`;
- otherwise `created` becomes `untested`.

Later non-intersecting bars leave `untested` unchanged. For long, failure is
`close_ticks < distal_edge_ticks`; for short it is
`close_ticks > distal_edge_ticks`. A close equal to the distal edge is not
failure and an intersecting bar is mitigated.

Failure has priority over mitigation, and one entity emits at most one
transition per completed bar. Unknown-gap, contract, data and tick-grid
boundaries use `failed` with the registered reason.

## Incremental state and capacity

The implementation needs only:

- the last three eligible candles for FVG formation;
- one frozen order-block candidate for the currently live displacement
  episode;
- bounded current and terminal FVG/order-block collections; and
- bounded per-entity transition histories.

It must not scan full history on each minute. The registered realtime
collection limits are 256 total FVG states and 128 total order-block states,
including both live and retained terminal states. Each entity has at most four
lifecycle transitions, and each live displacement episode has at most one
frozen order-block candidate.

Live entities are never evicted. A terminal entity may be compacted only
after its terminal transition appeared in an earlier successful observation.
Same-bar terminal transitions remain visible in the observation that created
them. If safe compaction cannot admit a new entity, the whole completed-bar
update fails closed transactionally; it may not partially advance clocks,
zones, ages or histories.

At a hard epoch boundary the tracker still exposes those new terminal states
in that boundary update and retains them for later safe compaction. The
observer keeps them out of the next contract/current frame, publishes them
through dedicated typed boundary-transition fields, and the development
replay persists that payload in the boundary minute's decision row. These
terminal transitions do not enter the ordinary EventMemory for the new epoch;
the dedicated fields are the causal audit channel across that boundary.

Age counts later real completed 5m bars only. Terminal age and duration stop
at the terminal clock. Frozen price and source fields are never recalculated
from later extremes.

## Required output and consumers

Every entity exposes stable identity, protocol identity, contract,
timeframe, direction, lifecycle, formation/confirmation/update/terminal
clocks, age, duration, frozen bounds, midpoint, transition reason and exact
source identities.

FVG additionally exposes near/far edges, width, maximum penetration, all
three candle IDs, qualification and optional linked-displacement provenance.
Order block additionally
exposes near/distal edges, source candle body direction, displacement
provenance and BOS provenance.

Typed observation, event memory, brain and visualization must consume these
fields directly. Ordinary within-epoch lifecycle transitions enter
EventMemory. Hard-boundary terminal transitions instead remain isolated in
the typed boundary fields and replay row so an old contract or invalid epoch
cannot contaminate new-epoch memory. Consumers must not reconstruct a zone
from natural-language annotations, current price, future extremes or the
legacy impulse proxy.

This protocol does not itself choose DFP admission, belief weights, entry
readiness, planned entry, invalidation, target, action utility or risk. Typed
playbooks may consume only displacement-linked FVGs and qualified order blocks
through their own causal gates; raw FVG remains descriptive geometry.

## Formalization gate

Formalization, incremental implementation, static review and the unified
synthetic/boundary/causal suite are complete for `3.2.0-group3.4`. FVG natural
lifecycle coverage passed the finite 2023 replay and stratified review. Order
blocks also formed and mitigated naturally, but only one failed lifecycle was
observed, so OB failure coverage remains sparse. There is no profitability or
broad-regime claim.

Any change to a gap inequality, source association, order-block search
window, lifecycle transition, same-bar priority, boundary behavior or
capacity rule requires a new protocol version. No numeric rule may be tuned
against PnL.
