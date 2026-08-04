# SMC primitives v3 Group 4 contract

Version: `3.1.0-group4.0`

Status: formalized; implementation and primitive-level validation are pending.

Protocol SHA-256:
`14b049facadb815c3fdc0d134275ee3f1efb3ca7c775a5f18ec4efad61663bc5`.

This contract freezes the causal meaning of:

1. one H1 `DealingRangeState` with lifecycle
   `forming → mature → broken`; and
2. one completed-1m `ManipulationState` with lifecycle
   `swept → reaccepted|accepted_outside`.

The executable companion is
`configs/smc_primitives_v3_group4.json`. If this document and that file ever
disagree, development stops until both are versioned together.

Group 4 is descriptive market-eye infrastructure. It does not authorize an
entry, FAVR, a probability change, a state-machine advance, a stop, a target
or a profitability claim. In particular, the existing legacy H1 dealing
range, H1 acceptance/rejection, H4 range position, 5m compression and 1m path
scores are not Group 4 sources.

## Minimal implementation boundary

The later implementation should add one incremental Group 4 reducer, not
separate accumulation, range, value, manipulation, custody, controller or
wrapper layers.

- Accumulation is the evidence accumulated while one typed range is
  `forming`.
- A dealing range is the same entity after it becomes `mature`.
- Value is the mature range's frozen arithmetic midpoint.
- A manipulation is a completed-1m excursion through one already-visible
  mature range boundary or one already-formed typed liquidity pool.

The reducer consumes immutable typed Group 1–2 sources, returns one atomic
Group 4 update, and leaves playbooks untouched.

## Frozen upstream dependency

Group 4 binds exactly:

- Group 1–2 version `3.1.0-group12.4`;
- Group 1–2 SHA-256
  `189b6af3bff631c3985fa37bcf9f5f82528296800886d9c9bd4cbe123ea4c701`;
- tick size `0.25`;
- H1 `SupportResistanceState`;
- authoritative `LiquidityPoolState`;
- authoritative `LiquidityInventoryItem`;
- ordered `real_completed` H1 candles; and
- ordered `real_completed` 1m candles.

An input envelope must preserve symbol, `instrument_id`, protocol hash,
source identity, source lifecycle, source clocks and frozen bounds. Matching
a price approximately, copying a natural-language label or reconstructing a
source from legacy scalar metrics is insufficient.

Support/resistance is reused as the frozen range-boundary source:

- an active/tested support supplies the lower outer bound;
- an active/tested resistance supplies the upper outer bound;
- typed touch counts and touch clocks supply repeated boundary tests.

The existing pool lifecycle is also reused as source provenance. Group 4
does not create a second equal-high/equal-low detector.

## One-time descriptive engineering thresholds

All numeric values below were chosen once for causal description and bounded
engineering. They were not selected from PnL, MFE, MAE, target delivery,
future paths or action labels.

| Parameter | Frozen value | Scale rationale |
| --- | ---: | --- |
| H1 ATR period | 14 real H1 bars | Matches the existing causal observer scale and removes price-unit dependence. |
| 1m ATR period | 14 real 1m bars | Normalizes sweep penetration without a contract-specific point threshold. |
| Minimum candidate duration | 8 real H1 bars | Supports non-overlapping first-four and latest-four range samples. |
| Maximum forming duration | 24 real H1 bars | Bounds one attempt to approximately one electronic-session cycle. |
| Minimum touches per side | 2 | The minimum repeat test already represented by a typed `tested` zone. |
| Minimum midpoint crossings | 2 | Requires a two-sided internal auction, not one pass. |
| Minimum inside-close fraction | 0.80 | Requires sustained containment while allowing one excursion in the minimum sample. |
| Maximum frozen width | 4.0 formation H1 ATR | Excludes very broad directional areas while allowing multi-hour balance. |
| Compression windows | first 4 / latest 4 real H1 bars | Non-overlapping at the first maturity clock and bounded thereafter. |
| Maximum compression ratio | 0.80 | Requires at least a 20% median true-range contraction. |
| Manipulation resolution delay | 1 later real 1m bar | Prevents the sweep bar from resolving itself and matches typed pool timing. |
| Retained ranges | 64 | Memory-safety limit, not a market threshold. |
| Retained manipulations | 256 | Memory-safety limit, not a market threshold. |

No Group 4 implementation may search alternative values against trade
outcomes. A threshold change is the one allowed concept-level repair only
when finite causal replay and blind review demonstrate a systematic semantic
error.

## Completed-bar and knowledge clock

Every engine update begins from exactly one newly completed 1m bar.

- Only a `real_completed` H1 candle may form, mature, break or age a range.
- Only a `real_completed` 1m candle may sweep, resolve or age a
  manipulation.
- Candles containing any synthetic minutes may advance the raw cutoff but
  cannot change ATR, candidate statistics, source selection, lifecycle,
  strength, semantic age or duration.
- Scheduled maintenance and weekend closures preserve state and do not add
  semantic bars or market minutes.
- An event clock is always the completed-bar end when its full evidence first
  becomes knowable. It is never the earlier pivot, source formation or
  intrabar extreme time.

For a source to be swept by a completed 1m bar:

`source.mature_at_or_confirmed_at <= current_1m.start`.

Equality here means the source was fully known at the instant the new 1m
interval began. A source first known at `current_1m.end` is ineligible for
that bar. This is the explicit prevention of a newly matured H1 range being
retroactively swept by its own final minute.

## DealingRangeState

### Candidate source pair

At most one range is live per contract. A pair is eligible only when:

1. the lower source is an H1 support in `active` or `tested`;
2. the upper source is an H1 resistance in `active` or `tested`;
3. `support.upper_bound < resistance.lower_bound`;
4. the selecting completed H1 close lies inside
   `[support.lower_bound, resistance.upper_bound]`;
5. both sources belong to the same current H1 frame, contract, uninterrupted
   epoch, Group 1–2 protocol hash and tick-size binding; and
6. a full 14-real-H1 ATR prefix with one predecessor close exists.

One ordered support/resistance identity pair may be admitted only once.
Its terminal range remains source-pinned and cannot be compacted while both
source IDs remain in the authoritative upstream snapshot. A nominally new
range therefore needs at least one new source-zone identity; no separate
unbounded tombstone registry is needed.

### Deterministic selection

An existing `forming` or `mature` range is never replaced by a more convenient
later pair. When no range is live, eligible pairs are ordered by:

1. smallest frozen outer width;
2. most recent pair eligibility clock;
3. lower source zone ID; then
4. upper source zone ID.

The first pair is selected. `formed_at` is the selecting completed H1 end,
not either source's older clock. Identity is the hash of the Group 4 protocol,
contract, both source IDs and `formed_at`.

No new range is selected on the same H1 bar that terminalizes the previous
range. This preserves same-bar terminal visibility and avoids a hidden
terminal/create ordering choice.

### Frozen geometry and value

At candidate formation:

- `lower_bound = support.lower_bound`;
- `upper_bound = resistance.upper_bound`;
- `midpoint = (lower_bound + upper_bound) / 2`;
- `value_price = midpoint`;
- `width_points = upper_bound - lower_bound`;
- `formation_atr` is the arithmetic mean of the last 14 positive
  real-completed H1 true ranges through `formed_at`, with `tick_size` as the
  positive floor; and
- `width_atr_at_formation = width_points / formation_atr`.

Source IDs, source clocks, source bounds, outer bounds, midpoint,
`value_price`, formation ATR and normalized width never change.

The midpoint exists geometrically while forming, but it becomes authoritative
value only when lifecycle is `mature`. A broken range retains the value as
historical provenance; consumers must not treat it as current value.

This definition deliberately avoids a volume-profile proxy. Bar-level OHLCV
does not reveal the within-bar volume distribution required to claim a true
volume value area.

### Incremental accumulation statistics

The reducer retains no more than the 24 candidate H1 bars. For each real H1
bar from `formed_at` through the current clock it incrementally maintains:

- lower and upper exact typed touch counts;
- count of closes within the frozen outer bounds, inclusive;
- close-side sequence relative to the midpoint;
- midpoint crossing count;
- first-four true-range median;
- latest-four true-range median; and
- candidate real-H1-bar count.

Closes exactly at the midpoint are removed from the side sequence. Remaining
closes below midpoint are `-1`, closes above are `+1`, and an adjacent sign
change is one midpoint crossing.

Definitions are:

`inside_close_fraction = inside_closes / candidate_real_h1_bars`

`compression_ratio = latest_four_median_true_range /
max(first_four_median_true_range, tick_size)`

`compression_strength = clamp(1 - compression_ratio, 0, 1)`

`narrowness_strength =
clamp(1 - width_atr_at_formation / 4.0, 0, 1)`

`boundary_test_strength =
clamp(min(lower_touch_count, upper_touch_count) / 3, 0, 1)`

`crossing_strength = clamp(midpoint_crossings / 4, 0, 1)`

Range strength is the unweighted mean of narrowness strength, compression
strength, boundary-test strength, crossing strength and the inside-close
fraction. It describes the maturity evidence; it is not an entry score.

### Maturity

A forming range becomes mature at the first real completed H1 end when all
conditions are simultaneously true:

- candidate count is from 8 through 24;
- lower touch count is at least 2;
- upper touch count is at least 2;
- midpoint crossing count is at least 2;
- inside-close fraction is at least 0.80;
- frozen width is no more than 4.0 formation ATR;
- compression ratio is no more than 0.80;
- both source zones remain active/tested; and
- the current completed H1 close remains within the frozen bounds.

`mature_at` is that H1 end. Touch counts and IDs, crossings, containment,
compression, component strengths and aggregate strength freeze at that clock.
Later price action cannot improve the original maturity evidence.

Maturity is descriptive authority to say that a dealing range and its value
exist. It is not FAVR authority.

### Break and lifecycle priority

Allowed transitions are:

`forming → mature|broken`

`mature → broken`

`broken` is terminal.

A strict completed H1 close below the lower bound or above the upper bound
breaks a forming or mature range. Equality remains inside. A wick outside
with a close inside does not break it.

Before maturity, either source leaving `active|tested` breaks the candidate.
After maturity, source retirement alone does not rewrite or break the frozen
range; only its completed-close rule or a hard epoch boundary can do so.

On candidate bar 24, maturity is evaluated before the forming deadline.
If maturity still fails, the state becomes broken with reason
`maturity_deadline_elapsed`.

Same-H1-bar priority is:

1. hard epoch boundary;
2. strict close break;
3. pre-maturity source invalidation;
4. maturity;
5. forming deadline; then
6. ordinary statistic and age update.

Every transition exposes its event time, state-start time, last-update time,
real-H1 age and exact reason.

## ManipulationState

### Eligible typed sources

A manipulation may originate from exactly one of:

1. a prior-observation mature range boundary whose matching
   `range_boundary` inventory item is visible; or
2. a prior-observation formed liquidity pool whose matching
   `equal_highs|equal_lows` inventory item is visible.

For a mature range, the prior real completed 1m close must lie inside the full
range. For an above pool the prior close must be at or below its upper bound;
for a below pool it must be at or above its lower bound.

The state freezes:

- source kind and ID;
- exact source protocol;
- source timeframe and eligibility clock;
- source symbol and instrument;
- source side and frozen bounds;
- source inventory ID and lifecycle; and
- any exact same-price coincident source IDs.

A wick that lacks such a pre-existing typed source is not manipulation. It
remains only candle structure or generic liquidity/rejection description.

### Sweep

For an above source:

`current_1m.high > source.upper_bound`.

For a below source:

`current_1m.low < source.lower_bound`.

Equality is not a sweep. `formed_at`, `confirmed_at` and `swept_at` all equal
the sweep bar end. The sweep extreme is that first bar's high or low and is
immutable; a later more extreme price must not rewrite it.

Penetration is normalized by a frozen 14-real-1m ATR known at bar start:

`penetration_atr = strict_distance_beyond_boundary / prior_14_bar_atr`

`strength = clamp(penetration_atr, 0, 1)`.

No minimum penetration threshold exists beyond strict crossing.

### Multiple source and direction handling

If one bar crosses eligible sources on both sides, OHLC cannot establish which
side occurred first. The reducer emits no manipulation, consumes every
crossed inventory item, and emits `ambiguous_dual_side_sweep`.

When only one side is crossed, the primary source is the crossed boundary with
the smallest nonnegative distance from the prior real 1m close. Exact ties are
ordered by source kind, eligibility clock and source ID. Same-price ties are
retained as `coincident_source_ids`, but they produce one manipulation state
and may not be double-counted downstream.

Only one manipulation may be live. If an older one resolves on the current
bar, no new manipulation is created from that bar's high/low. A new attempt
waits for the next real completed minute.

One source inventory identity can create at most one manipulation. Its first
crossing consumes that source inventory identity, so a new attempt requires a
new range, range-boundary or pool-generation identity rather than a separate
source-use tombstone.

### Resolution

The sweep bar cannot resolve itself. Resolution occurs on the first later
real completed 1m bar.

For a mature range source:

- a close within the full frozen range, including equality, is
  `reaccepted`;
- a close strictly outside the full range is `accepted_outside`, and
  `resolved_side` records which side.

For an above pool:

- close at or below the upper bound is `reaccepted`;
- close above it is `accepted_outside`.

For a below pool:

- close at or above the lower bound is `reaccepted`;
- close below it is `accepted_outside`.

`reentry_price` is the terminal close only for `reaccepted`.
`outside_completed_bars` counts sweep and later real bars whose closes satisfy
the source-specific outside rule. Duration is reported both in real 1m bars
and through EventMemory market minutes; synthetic and closure minutes are
excluded.

`reaccepted` and `accepted_outside` are terminal. Later paths cannot relabel
the event. Qualified leave/return/hold entry reacceptance remains a Group 5
concept; this one-bar resolution is only the minimum manipulation description.

### Per-minute processing order

For each completed 1m update:

1. validate contract, clock, tick-size and boundary provenance;
2. resolve an already-swept manipulation from the current completed close;
3. project every prior-visible inventory crossing;
4. apply an optional current real completed H1 range transition;
5. if an older manipulation resolved this minute, do not create another;
6. select at most one new manipulation from prior-visible sources that remain
   eligible; and
7. after range maturity, create its boundary inventory without allowing a
   same-bar sweep.

This order ensures a source was visible at the start of the interval and that
one bar cannot secretly both create and validate it.

## Range-boundary liquidity inventory

At maturity the range emits exactly two typed inventory items:

- lower item: side `below`, kind `range_boundary`, price and both item bounds
  equal the frozen range lower bound;
- upper item: side `above`, kind `range_boundary`, price and both item bounds
  equal the frozen range upper bound.

Each identity binds the Group 4 hash, range ID, side and `mature_at`.
`formed_at` is the range formation clock; `confirmed_at` is `mature_at`.
Source IDs contain the range and corresponding source zone.

The eye emits only `visible` or `consumed`; `targeted` remains owned by belief
and planning.

- Above is consumed on the first later real 1m `high > price`.
- Below is consumed on the first later real 1m `low < price`.
- Equality does not consume.
- `consumed_at` is the crossing bar end.
- Reason is `range_boundary_consumed`.

Range break does not retrospectively remove or consume the opposite unswept
boundary. Range lifecycle and liquidity consumption answer different
questions: whether the old range remains current, and whether its liquidity
has already been taken.

## Cold attachment

No current snapshot may be used to invent missing history.

### Range reconstruction

Cold reconstruction replays ordered real H1 candles together with the exact
Group 1–2 typed zone snapshot at each clock. The prefix starts no later than
candidate selection and contains one predecessor plus the full 14-real-H1 ATR
window.

If unavailable, the reducer does not backfill a mature range. Already-present
source pairs remain ineligible until at least one source identity changes,
then normal fresh formation may begin.

### Manipulation and inventory reconstruction

For each eligible mature range boundary or formed pool, the real completed 1m
prefix starts no later than source eligibility and contains one predecessor
plus the 14-real-1m ATR window. The reducer reconstructs:

1. the first strict crossing;
2. the exact range-boundary/pool inventory consumption clock; and
3. the first later real-bar manipulation resolution.

A prefix beginning after eligibility cannot prove that an earlier sweep did
not occur and fails closed. Failure for one retained source rejects the whole
cold attachment; partial state is forbidden.

A hash-bound checkpoint containing exact Group 4 state and its last committed
source envelope may resume directly without rediscovery.

## Soft and hard boundaries

Scheduled same-contract closures and synthetic/partially-synthetic candles are
soft boundaries. They preserve all state and do not age, resolve, mature,
compact or update statistics.

Hard reasons are:

- `data_gap_reset`;
- `contract_change_reset`;
- `data_anomaly`; and
- `tick_size_mismatch`.

At a hard boundary:

- every forming or mature range becomes broken at the boundary clock with the
  exact reason;
- a swept manipulation is not falsely relabeled as market acceptance. It
  remains lifecycle `swept` in a dedicated typed boundary transition, records
  `censored_at` and the hard reason, then leaves the new epoch;
- all same-clock terminal/censored transitions remain visible in dedicated
  typed boundary fields and the replay row;
- old-epoch transitions do not enter the new epoch's ordinary EventMemory;
  and
- no old contract source appears in the new frame.

This boundary censorship is outside the ordinary manipulation lifecycle. It
preserves the requested `swept → reaccepted|accepted_outside` market semantics
without claiming that missing data or a contract roll was acceptance.

## Incremental state, capacity and transaction

Normal operation retains only:

- one live H1 range and at most 24 candidate bars;
- up to 64 total retained range states;
- one live manipulation;
- up to 256 total retained manipulation states;
- prior real 1m close;
- bounded H1 and 1m ATR windows;
- at most three range or two manipulation lifecycle transitions per entity.

It must not rescan full history each minute.

Live state is never evicted. A terminal state may be compacted only after its
terminal transition appeared in an earlier successful observation. A terminal
range also remains pinned while both frozen source zone IDs remain in the
authoritative upstream snapshot. Manipulation reuse is prevented by the
consumed source-inventory identity. Same-bar terminals remain visible;
temporary overflow is bounded by the count of those same-bar terminals and is
compacted before the next real semantic update.

If safe compaction cannot admit required state, the entire completed-bar update
fails closed. It may not evict live state, change thresholds, discard source
provenance or grow without bound.

Range state, manipulation state, used-source memory and range-boundary
inventory additions/consumptions update in one shadow copy and commit once.
Any exception leaves the previous clock, state, ATR, candidate window, ages,
inventory and histories unchanged.

Exact retry requires the same:

- `asof`;
- completed 1m provenance;
- optional completed H1 provenance; and
- typed upstream source envelopes.

It returns the exact previously committed result. A non-identical same-clock
input or an earlier clock is rejected before mutation.

If later observer or EventMemory projection fails after an upstream or Group 4
commit, the observer is terminal and must be discarded in favor of the last
checkpoint. Partially paired state cannot be reused.

## Required consumers and display

Every state exposes identity, protocol, contract, lifecycle, formation,
knowledge, state-start, update and terminal clocks, semantic age, duration,
strength, transition reason, frozen geometry and exact source provenance.

EventMemory uses:

- `range:<range_id>` for `forming`, `mature`, `broken`; and
- `manipulation:<manipulation_id>` for
  `swept`, `reaccepted|accepted_outside`.

Hard-boundary censorship remains in the typed boundary channel rather than
the new epoch's ordinary timeline.

The H1 frame displays typed ranges at its completed cutoff. Current
completed-1m manipulation and range-boundary inventory live at top-level
observation authority, so they cannot rewrite an unfinished H1 frame.

Visualization must show frozen bounds, midpoint/value, range lifecycle and
age, manipulation sweep extreme, duration, reentry price and source IDs
without expanding the price axis.

Consumers must read these typed outputs directly. They may not reconstruct
Group 4 from old rolling ranges, current price, annotations or future extrema.

## FAVR remains shadow

This formalization does not modify `configs/playbooks_v3.json`,
`smc_trader/playbooks.py`, the decision layer or the risk engine.

Failed-auction value return remains shadow and inactive. It may be reconsidered
only after:

1. Group 4 implementation passes synthetic, boundary and causal tests;
2. one finite real OHLCV replay and stratified blind review show no systematic
   mature-range/manipulation misread; and
3. Group 5 supplies qualified reacceptance, entry location, micro BOS and true
   path sequence.

If mature ranges remain unreliable after the one permitted concept repair,
FAVR stays disabled and Group 4 is parked rather than threshold-tuned.

## Formalization gate

This document completes the single formalization step only. Pending work is:

- one implementation;
- one static trading-logic/code self-review;
- the unified synthetic/boundary/causal suite;
- one finite real OHLCV replay;
- one small stratified blind review; and
- freeze or park.

Primitive profitability, MBO stability, rolling OOF and sealed holdout are not
required at this gate.
