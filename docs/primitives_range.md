# Range and manipulation primitives

The executable definition is `configs/primitives_range.json`.

Current protocol: `3.2.0-group4.1`. Status:
`implementation_complete_limited_natural_observation_passed_mature_range_coverage_sparse`.

This contract freezes the causal meaning of:

1. one H1 `DealingRangeState` with lifecycle
   `active → broken`; and
2. one completed-1m `ManipulationState` with lifecycle
   `swept → reaccepted|accepted_outside`.

This file is the retained v1.2 Group-4/Mature Balance Range contract. Its
`DealingRangeState` name is not Structural Range authority. The additive
[`smc_semantic_foundation_v2.1`](canonical_semantic_foundation_v2.1.md)
projection preserves this identity as `BalanceRange` and publishes a separate
Structure-Generation-owned `StructuralRange`; the two may coexist and have
independent normalized locations.

The executable companion is
`configs/primitives_range.json`. If this document and that file ever
disagree, development stops until both are versioned together.

Group 4 is descriptive market-eye infrastructure. It never selects an entry,
phase, stop, target, risk result, execution result or profitability claim.
Typed playbooks may consume its exact identities and lifecycles through their
own causal gates. The legacy H1 dealing range, H1 acceptance/rejection, H4
range position, 5m compression and 1m path scores are not Group 4 sources.

A registered outcome-blind 2023 full-year coverage scan was run under the
`group4_natural_authority_2023_full_year` profile in `configs/data_splits.json`;
its result file was retired on 2026-09-06 as v1.2-era evidence, and the profile
no longer binds a permanent path. The recorded counts were 448 in-window range
formations, two ranges whose balance claim
settled (recorded then as "mature") and 19,465 typed
manipulations from all five enabled pool-source timeframes plus mature-range
boundaries. Manipulation outcome conservation passed. Mature-range coverage is
still only two cases, so this evidence does not change the sparse-coverage
status or enable FAVR.

> **Lifecycle change, 2026-09-06.** `forming -> mature -> broken` became
> `active -> broken`. Balance was a grade the interval earned, and three months
> of data showed the grade recorded a failure that was not occurring: 0 of 84
> ranges reached a two-sided test against a 4.06% base rate on arbitrary H1
> windows, which the sample cannot separate. Balance evidence is unchanged and
> is still published as `BALANCE_RANGE_OBSERVED`. Candidate selection,
> thresholds and the `mature_range_boundary` source kind are unchanged. Figures
> below that predate this are historical measurements and are left as recorded.

## Minimal implementation boundary

The implementation uses one incremental Group 4 reducer, not separate
accumulation, range, value, manipulation, custody, controller or wrapper
layers.

- Accumulation is the evidence accumulated while one typed range is
  `active`.
- A dealing range is the same entity for its whole life. Since 2026-09-06
  it has no maturity state: balance was a grade the interval earned, and
  the data showed the grade recorded a failure that was not occurring.
  Settling the balance claim is recorded as `balance_claim_confirmed` and
  `balance_confirmed_at`, and is not a transition.
- Value is the range's frozen arithmetic midpoint.
- A manipulation is a completed-1m excursion through one already-visible
  mature range boundary or one already-formed typed liquidity pool.

The reducer consumes immutable typed Group 1–2 sources, returns one atomic
Group 4 update, and leaves playbooks untouched.

## Frozen upstream dependency

Group 4 binds exactly:

- Group 1–2 version `3.2.0-group12.7` and the exact hash declared by the
  executable Group 4 config;
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
| Reacceptance hold | 1 additional inside real 1m bar | The first inside close is only a reentry candidate; the next inside close proves the hold. |
| Outside acceptance | 2 consecutive outside closes | Distinguishes sustained outside acceptance from a one-close excursion; the sweep close counts when outside. |
| Resolution deadline | 5 later real 1m bars | Censors unresolved observation without relabelling it accepted or rejected. |
| Retained ranges | 64 | Memory-safety limit, not a market threshold. |
| Retained manipulations | 256 | Memory-safety limit, not a market threshold. |

No Group 4 implementation may search alternative values against trade
outcomes. A threshold change is the one allowed concept-level repair only
when finite causal replay and blind review demonstrate a systematic semantic
error.

## Completed-bar and knowledge clock

Every engine update begins from exactly one newly completed 1m bar.

- Only a `real_completed` H1 candle may form, break or age a range, or
  settle its balance claim.
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

`source.balance_confirmed_at_or_confirmed_at <= current_1m.start`.

Equality here means the source was fully known at the instant the new 1m
interval began. A source first known at `current_1m.end` is ineligible for
that bar. This is the explicit prevention of a newly promoted H1 boundary being
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

An existing `active` range is never replaced by a more convenient
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

The midpoint is authoritative value for as long as the range is `active`. It
used to be withheld until the range matured, which made a two-sided-test
statistic a precondition for arithmetic the interval could always do. A broken
range retains the value as historical provenance; consumers must not treat it
as current value.

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

### Balance claim settlement

The range has no maturity state. What follows settles the *balance claim* over
the interval at the first real completed H1 end when all conditions are
simultaneously true; the range's own lifecycle does not move, and settling is
recorded as `balance_claim_confirmed` with a `balance_confirmed_at` clock. It
is what promotes the frozen boundaries to candidate liquidity levels.

- candidate count is from 8 through 24;
- lower price test generations are at least 2;
- upper price test generations are at least 2;
- midpoint crossing count is at least 2;
- inside-close fraction is at least 0.80;
- frozen width is no more than 4.0 formation ATR;
- compression ratio is no more than 0.80;
- both source zones remain active/tested; and
- the current completed H1 close remains within the frozen bounds.

`balance_confirmed_at` is that H1 end. Touch counts and IDs, crossings, containment,
compression, component strengths and aggregate strength freeze at that clock.
Later price action cannot improve the original maturity evidence.

Maturity is descriptive authority to say that a dealing range and its value
exist. It is not FAVR authority.

### Break and lifecycle priority

Allowed transitions are:

`active → broken`


`broken` is terminal.

A strict completed H1 close below the lower bound or above the upper bound
breaks an active range. Equality remains inside. A wick outside
with a close inside does not break it.

Before maturity, either source leaving `active|tested` breaks the candidate.
After maturity, source retirement alone does not rewrite or break the frozen
range; only its completed-close rule or a hard epoch boundary can do so.

On candidate bar 24, the balance claim is evaluated before the claim deadline.
If maturity still fails, the state becomes broken with reason
`maturity_deadline_elapsed`.

Same-H1-bar priority is:

1. hard epoch boundary;
2. strict close break;
3. pre-maturity source invalidation;
4. maturity;
5. balance claim deadline; then
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

The sweep bar cannot confirm reacceptance. For an above-side source, a close
at or below the frozen upper boundary is inside; for a below-side source, a
close at or above the frozen lower boundary is inside. Equality is inside.

The first later inside close freezes `reentry_candidate_at` and
`reentry_price`. One additional real completed 1m close must remain inside;
only then does lifecycle become `reaccepted`. An outside close before that
hold clears the candidate, records `reentry_failed_at`, and restarts the
outside run.

`accepted_outside` requires two consecutive source-specific outside closes.
The sweep close counts as the first when it closed outside, so acceptance can
complete on the first later real bar only in that case. Otherwise it completes
on the second close of a later consecutive outside pair.

If neither result resolves within five later real completed 1m bars, lifecycle
remains `swept` while `deadline_elapsed`, `deadline_at` and `censored_at` record
the actual fifth real-bar clock. A terminal market result on that fifth bar
takes priority. Synthetic and scheduled-closure bars do not count.

`reaccepted` and `accepted_outside` are terminal. The original source,
boundary and sweep extreme never change. This multi-bar pool/range
reacceptance belongs to Group 4. Group 5 does not build a second pool
leave/reclaim/hold state; it only records the Group4 outcome in the ordered
pool path and later handles qualified entry-zone reacceptance for a frozen
FVG/OB.

### Per-minute processing order

For each completed 1m update:

1. validate contract, clock, tick-size and boundary provenance;
2. advance or resolve an already-swept manipulation from the current
   completed close;
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

Each identity binds the Group 4 hash, range ID, side and
`balance_confirmed_at`. `formed_at` is the range formation clock;
`confirmed_at` is `balance_confirmed_at`.
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

If unavailable, the reducer does not backfill a promoted boundary. Already-present
source pairs remain ineligible until at least one source identity changes,
then normal fresh formation may begin.

### Manipulation and inventory reconstruction

For each eligible mature range boundary or formed pool, the real completed 1m
prefix starts no later than source eligibility and contains one predecessor
plus the 14-real-1m ATR window. The reducer reconstructs:

1. the first strict crossing;
2. the exact range-boundary/pool inventory consumption clock; and
3. every later real bar needed to reproduce reentry candidate, hold,
   outside-run, terminal result or five-bar deadline.

A prefix beginning after eligibility cannot prove that an earlier sweep did
not occur and fails closed. Failure for one retained source rejects the whole
cold attachment; partial state is forbidden.

A hash-bound checkpoint containing exact Group 4 state and its last committed
source envelope may resume directly without rediscovery.

## Soft and hard boundaries

Scheduled same-contract closures and synthetic/partially-synthetic candles are
soft boundaries. They preserve all state and do not age, resolve, settle,
compact or update statistics.

Hard reasons are:

- `data_gap_reset`;
- `contract_change_reset`;
- `data_anomaly`; and
- `tick_size_mismatch`.

At a hard boundary:

- every active range becomes broken at the boundary clock with the
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

- `range:<range_id>` for `active`, `broken`; and
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

## FAVR remains runtime parked

The typed FAVR evaluator and vertical data path are implemented, but runtime
execution remains parked. Group 4 owns the mature range, boundary liquidity,
sweep and multi-bar failed-outside/reacceptance sequence. After that outcome,
Group 5 may register only the reverse-displacement FVG/OB entry location and
its first-pullback/trigger path; it does not duplicate Group4 manipulation
reacceptance.

Finite real OHLCV replay and stratified blind review showed that natural mature
ranges and both manipulation resolutions are observable, but mature-range
coverage was sparse: the registered full-year scan found only two mature
ranges. FAVR may be enabled only after the complete causal
sequence naturally remains connected through range identity, manipulation
identity, reverse displacement, zone identity, first pullback and trigger.

If mature ranges remain unreliable after the one permitted concept repair,
FAVR stays disabled and Group 4 is parked rather than threshold-tuned.

## Formalization gate

Formal definition, incremental implementation, static review and the unified
synthetic/boundary/causal suite, finite real OHLCV replay and small stratified
review are complete for `3.2.0-group4.1`. Limited natural observation passed;
the permanent full-year coverage evidence is linked above, broader
mature-range stability remains pending, and FAVR stays parked.

Primitive profitability, MBO stability, rolling OOF and sealed holdout are not
required at this gate.
