# Preregistered Semantics v1.3: definitions, formulas, and what the data showed

Status date: 2026-08-31
Atomic identity: `smc_semantics_v1.3` ·
`f2f70377f10c0715256882370ad69fb60fb33d80e606472533387c8ffb32dc9f`
Foundation identity: `smc_semantic_foundation_v2.1` ·
`69428dbfd2a9b2aa19f0254391fca2da17aedb8d0206829572e69c0cc212a715`
Status: `preregistered_development_contract_not_oos_trading_authority`

This supersedes
[the v1.2 status document](preregistered_semantics_status_2026-08-29.md) for the
runtime. That document's measurements keep their v1.2 identity and remain valid
as v1.2 evidence.

## What binds what

The atomic identity is a SHA-256 over five things, not over the registry alone:

```
sha256( registry_v1_3.yaml
      + parameters_v1_3.yaml
      + configs/data_splits.json
      + sha256(configs/primitives_structure_liquidity.json)
      + sha256(configs/primitives_displacement.json)
      + sha256(configs/primitives_zones.json)
      + sha256(configs/primitives_range.json) )
```

Changing any detector threshold therefore changes the semantic identity and, by
the freeze policy, requires a new semantic version and fresh validation. That
guarantee had a hole, now closed:

- **`configs/primitives_interaction.json` was loaded by the Eye and was not in
  the bound set.** Protocols are *discovered* by regex-scanning the registered
  parameters' `source` strings, so a protocol no parameter happened to mention
  was silently excluded — its thresholds could change while the identity, and
  therefore the freeze policy, reported that nothing had changed. Demonstrated
  by mutating `later_hold_bars` from 1 to 99 and watching the identity not move.
  Interaction parameters are now registered, the file is bound, and the identity
  moves when it changes. `tests/test_semantic_protocol_binding.py` asserts that
  every protocol `configs/model.json` loads is hashed into the identity, so the
  same hole cannot reopen for a different file.
- **`configs/primitives_entry.json` is correctly *not* bound.** It is not loaded
  by the Eye at all; it parameterises entry sequencing, which is Execution's
  concern. Binding it to the Eye's identity would make an Execution change
  invalidate the Eye's semantic version for no reason. It belongs to an
  execution identity, and this is a deliberate exclusion rather than an
  oversight.

The Foundation identity is a SHA-256 over its whole payload and declares
`smc_semantics_v1.3` as its parent. The strict loader refuses any pair whose
parent does not equal the selected atomic version.

## The evidence base

| window | data | bars | what it measures |
|---|---|---:|---|
| 2022-02-01 → 03-01 | NQ 1m OHLCV | 27,360 | full event population, 356,449 events |
| 2022-03-01 → 04-01 | NQ 1m OHLCV | 31,740 | full event population, 404,656 events |
| 2024-06-01 → 07-01 | NQ 1m OHLCV + MBO top-of-book | 27,720 | crossing-generation outcomes against the book |

Machine-readable:
[`v1_3_two_month_replay_comparison.json`](../evidence/v1_3_two_month_replay_comparison.json),
[`balance_range_gate_study_2022_02.json`](../evidence/balance_range_gate_study_2022_02.json),
[`balance_range_gate_study_2022_03.json`](../evidence/balance_range_gate_study_2022_03.json),
[`unresolved_interaction_study_2024_06.json`](../evidence/unresolved_interaction_study_2024_06.json).

## The 20 registered concepts

31 canonical emitted EventKinds, 5 compatibility aliases, 2 reserved, 4
snapshot-derived bindings.

### Structure

**`confirmed_swing`** — a local price pivot confirmed by a registered
completed-bar span, never by a live extreme.
Formula: `k_left = k_right = 2` on 4H/1H/15m/5m and `1` on 1m
(`primitives_structure_liquidity.json:swing_span_by_timeframe`).
`prominence_ATR` has **no positive hard cutoff**: the executable minimum is 0.0
and prominence is retained as a continuous field.
Emits `SWING_CONFIRMED`. 2022-02: 12,933 · 2022-03: 14,537.

**`structural_leg`** — the frozen path between two confirmed opposite swings,
with its complete native bar lineage and strict-prior ATR.
Formula: `path_class = MICRO if len(path) <= 2 else INTERNAL` (renamed from
`rank` in v1.3, and now a strict lookup that fails rather than defaulting).
Emits `STRUCTURAL_LEG_CREATED`. 2022-02: 10,081 · 2022-03: 11,309.

**`structure_direction`** — which direction the current structure is
established in, from confirmed breaks only.
Emits `STRUCTURE_DIRECTION_CONFIRMED`. 2022-02: 2,272 · 2022-03: 2,589.

**`raw_boundary_break`** — a completed close strictly beyond a boundary, before
any qualification. Formula: `break_buffer_ticks = 0`, comparison
`strict_close_beyond`. Emits `RAW_BOUNDARY_BREAK`.
2022-02: 4,380 · 2022-03: 5,072.

**`qualified_bos`** — a raw break that survived its registered post-break
acceptance. Formula: `bos_post_break_later_bars = 1`.
Emits `QUALIFIED_BOS`. 2022-02: 1,066 · 2022-03: 1,195.
Qualification rate: 24.3% and 23.6% of raw breaks.

**`protected_swing`** — the swing whose violation would end the current
structural claim. Emits `PROTECTED_SWING_ASSIGNED`.
2022-02: 222 · 2022-03: 263.

**`mss_core`** — an opposed break that qualifies as a market-structure shift.
Emits `MSS_CORE_CONFIRMED`. 2022-02: 1,763 · 2022-03: 2,036.
MSS outnumbers qualified BOS 1.65:1 and 1.70:1.

**`structure_regime`** *(new in v1.3, snapshot-derived)* — which direction is
established. Registered explicitly so that it and `delivery_phase` are two
independent dimensions: the same regime spans many phases and the same phase
occurs under opposite regimes. Collapsing them destroys both.

### Liquidity

**`candidate_liquidity_level`** — a level that is a candidate for interaction,
not yet a claim about what will happen there.
Emits `LIQUIDITY_LEVEL_CREATED`. 2022-02: 15,000 · 2022-03: 16,772.

**`liquidity_sweep`** — the crossing-generation family: touch, penetration, and
the sweep terminal. Formula: `pool_post_cross_resolution_bars = 1`,
`range_manipulation_real_1m_bars = 5`.
Emits `LEVEL_TOUCHED` (25,837 / 28,852), `LEVEL_PENETRATED` (25,563 / 28,540),
`SWEEP_CONFIRMED` (10,188 / 11,499).

**`acceptance`** — the opposite terminal: price stayed.
Formula by family: `bos_post_break_later_bars = 1`,
`range_manipulation_real_1m_outside_closes = 2`,
`active_dealing_range_external_acceptance_completed_h1_closes = 1`.
Emits `ACCEPTANCE_CONFIRMED`. 2022-02: 10,238 · 2022-03: 11,264.

### Delivery

**`displacement`** — a registered impulse episode with a frozen lifecycle.
`displacement_score_threshold` is deliberately **null**: the features and
lifecycle are retained, no score cutoff is applied.
Emits `DISPLACEMENT_OBSERVED`. 2022-02: 2,697 · 2022-03: 3,067.

**`delivery_phase`** *(reworked in v1.3)* — an entered/updated/exited lifecycle,
not a value recomputed each bar. Each occupancy freezes `entered_at`,
`age_bars`, `origin_event`, `parent_structure_generation` and `previous_phase`
at entry, and `next_phase` at exit.
Derived from same-timeframe confirmed structure, the active structural leg,
protected-swing intactness and active-range availability. Range extension and
volatility compression are **not** v1.3 phase inputs.
`age_bars` counts real completed M1 bars since entry.
`parent_structure_generation` is the event id of the most recent
`STRUCTURE_DIRECTION_CONFIRMED` / `QUALIFIED_BOS` / `MSS_CORE_CONFIRMED` on that
timeframe — the hot path carries no separate generation object.
`DELIVERY_PHASE_UPDATED` fires **only** when a registered phase input
(`structure_regime`, `active_leg_direction`, `protected_swing_intact`, range
availability) moves while the phase itself does not.
Emits `DELIVERY_PHASE_ENTERED` (7,953 / 9,004), `_EXITED` (7,948 / 8,994),
`_UPDATED` (3,420 / 3,731). That is ~0.42 events per bar across five
timeframes, against the ~137,000 a per-bar-per-timeframe projection would emit.
Entries exceed exits by 5 and 10 — the timeframes still occupying a phase when
the window ended.

**`premium_discount_irl_erl`** *(snapshot-derived)* —
`normalized_location = (price − low) / (high − low)` over the active structural
range, with the Premium/Discount label derived from it and same-timeframe
IRL/ERL membership. Formula for membership:
active lifecycles `{active, mature}`, inside comparison
`low < candidate_price < high`, boundary and outside → `erl`, inactive →
`unresolved`.

**`dol_candidate`** *(snapshot-derived)* — the candidate target inventory the
Brain ranks. The Eye publishes the inventory; it does not rank it.

### Zones

**`fvg`** — a strict three-completed-bar gap with frozen formation
qualification. Formula: `min_fvg_ticks = 1`, comparison
`strict_non_overlap_on_tick_grid`; bullish requires
`c3.low_ticks > c1.high_ticks`, bearish `c3.high_ticks < c1.low_ticks`;
equality does not form a gap.
`formation_atr` = mean positive true range of at most 14 real 5m bars completed
before c3 (c3 excluded, tick_size as the positive fallback).
`fvg_expiry_bars` is **null**: v1.3 infers no expiry from age and
`FVG_EXPIRED` is reserved and unemitted.

*New in v1.3:* `FVG_FIRST_RETEST`, exactly one per gap, on the first bar where
price re-enters the frozen gap, published before the revisable fill observation
it shares that bar with. Partial, mitigated and invalidated all require entry,
so the first of them is the first re-entry; the reserved age-based expiry is not
an entry. It freezes `fill_depth_at_entry` (= `max_fill_fraction` at that bar),
`age_bars`, `age_seconds`, the entry lifecycle and reason, the formation
qualification, the session, the source displacement, and

```
approach_speed_atr = max(0, signed distance from the last completed M5 close
                            to the near edge) / ATR(entry bar)
```

which is distance covered in one bar, normalized by that bar's own causal ATR.
The Group-3 protocol lists MBO among its prohibited inputs, so the event carries
no book evidence.

Counts — 2022-02: created 1,044, **first retest 1,031 (98.8%)**, partial 292,
midpoint 322, fully filled 543, invalidated 469.
2022-03: created 1,275, **first retest 1,234 (96.8%)**, partial 366, midpoint
391, fully filled 658, invalidated 560.

**`base_origin_core`** *(new in v1.3)* — the configured geometry of the last
opposite candle or opposite-candle cluster preceding a directional impulse, with
the exact impulse identity that located it. It exists whether or not the impulse
went on to displace or break structure, is never an independent causal agent,
and carries **no order-block interpretation** — that reading belongs to the
Brain. It cites only its anchor bars.
Emits `BASE_ORIGIN_CORE_CREATED`. 2022-02: 43 · 2022-03: 57.

**`qualified_origin_zone`** *(new in v1.3)* — binds exactly one
`base_origin_core`, the active `DISPLACEMENT_OBSERVED` episode, and the
`QUALIFIED_BOS` that together qualify it, keeping all three source identities.
Qualification is append-only and never backdated into the core.
Emits `QUALIFIED_ORIGIN_ZONE_CREATED` (43 / 57), `ORIGIN_ZONE_MITIGATED`
(35 / 51), `ORIGIN_ZONE_INVALIDATED` (5 / 4).
The detector materialises the core only when the BOS confirms, so both facts are
published at one clock; publishing the core earlier is a Group-3 protocol change
and was not made.

### Ranges

**`structural_range`** — the external interval whose frozen boundaries locate
current price. Usable for location the moment it is created; it has **no**
maturity condition and no two-sided test requirement, because location does not
depend on how price behaved inside.
Four production-reachable invalidation variants stay distinct under
`DEALING_RANGE_INVALIDATED`: three forming variants
(`close_beyond_frozen_range_before_activation`, `forming_source_invalidated`,
`maturity_deadline_elapsed`) and one `active_acceptance` variant that
additionally requires the completed H1 break BAR and `ACCEPTANCE_CONFIRMED`.
External acceptance is exactly the first completed H1 close strictly outside a
frozen boundary; a wick outside with an H1 close inside does not qualify, and no
M-bar hold is claimed. `DEALING_RANGE_EXTENDED` is reserved and unemitted.
Emits `DEALING_RANGE_CREATED` (40 / 41), `_INVALIDATED` (39 / 40),
`_REPLACED` (39 / 39).

**`balance_range`** *(separated in v1.3)* — evidence that price actually
two-sided-tested an interval rather than merely occupying it. Observed **over** a
structural range that already exists; failing to mature never invalidates the
underlying structural range.

`BALANCE_RANGE_OBSERVED` is published once per range, as soon as both frozen
boundaries have been tested at least `balance_range_boundary_touches_each = 2`
times. One touch per side is structural, not evidence: `DealingRangeState`
requires `lower_touch_count >= len(lower_source_member_swing_ids) >= 1` and the
same for upper, so every valid range satisfies a one-touch test by construction.
Because the Group-4 detector surfaces no intra-forming update, the observation
clock is the range's next registered transition rather than the second boundary
test itself; moving it earlier is a Group-4 protocol change.

`BALANCE_RANGE_MATURED` replaces the retired `DEALING_RANGE_ACTIVATED` and is
what the compact current view now calls an active range. The maturity gate is a
conjunction of six registered conditions at one completed real H1 close:

```
candidate_real_h1_bars  >= 8      lower_touch_count      >= 2
midpoint_crossings      >= 2      upper_touch_count      >= 2
inside_close_fraction   >= 0.8    width_atr_at_formation <= 4.0
                                  compression_ratio      <= 0.8
```

with

```
candidate_bar_set     = real completed H1 candles from formed_at through the
                        current H1 end, retained up to 24
touch counts          = each frozen source zone's total_touch_count at the
                        current H1 clock, frozen at mature_at
inside_close_fraction = count(low <= close <= high) / candidate_real_h1_bars
midpoint_crossings    = drop closes equal to the midpoint, encode the rest as
                        -1 below / +1 above, count adjacent sign changes
compression_ratio     = late_true_range / max(early_true_range, tick_size)
                        early = first 4 candidate bars, late = last 4
width_atr_at_formation= width_points / formation_atr   (H1 ATR-14, frozen)
```

A candidate that reaches 24 real H1 bars without maturing is terminalized with
`maturity_deadline_elapsed`.

## What the data said about balance

40 structural ranges in February and 41 in March. **One** per month reached the
two-touch standard. **None** matured, in either month, and
`sole_blocking_gate` is empty in both: no range was ever one gate away.

| gate | threshold | unmet 2022-02 | unmet 2022-03 | observed median (02 / 03) |
|---|---|---:|---:|---|
| bilateral_touches | ≥ 2 | 39/40 | 40/41 | 1.0 / 1.0 (max 2.0) |
| compression | ≤ 0.8 | 37/40 | 37/41 | 1.0 / 1.0 |
| midpoint_crossing | ≥ 2 | 34/40 | 30/41 | 0.0 / 0.0 |
| width | ≤ 4.0 | 22/40 | 20/41 | 4.14 / 3.99 |
| duration | ≥ 8 | 20/40 | 21/41 | 8.0 / 7.0 |
| inside_close_fraction | ≥ 0.8 | 11/40 | 13/41 | 0.875 / 0.875 |

Unmet gates per range: never fewer than two, most often three to five.

The terminal reasons explain why. 33 of 40 and 35 of 41 ranges ended with
`close_beyond_frozen_range` — price left the interval — against only 6 and 5
that ran out the 24-bar deadline. Median duration is 8 and 7 H1 bars against a
minimum of 8. **These ranges are broken by price leaving before they can be
tested twice**, and the boundary-touch, midpoint-crossing and duration gates are
all downstream of that single fact. The compression gate is separately
uninformative here: a median ratio of 1.0 means the late window's true range
equals the early window's, so most candidates are not contracting at all.

This is a distribution, not a recommendation. It does not say the thresholds are
wrong; it says the candidate selection produces intervals that mostly do not
survive long enough for any of them to bind.

## What the data said about unresolved penetrations

The 2024-06 window classified 23,488 crossing generations against the top of
book: 9,574 sweeps, 9,082 acceptances, **4,832 unresolved** (20.6%), with zero
right-censoring after a 5-minute tail exclusion.

Both hypotheses for the unresolved population were tested and refuted:

- *The resolution window is too short.* Refuted: measured M1 resolution lag is
  p90 = 1 minute and p99 = 2–3 minutes; widening the window from 5 to 120
  minutes moves the population by 0.1%.
- *They are superseded by a later crossing of the same level.* Refuted:
  **4,832 of 4,832** carry `no_later_crossing_on_that_level`.

Top-of-book features do not separate the three groups. Spread is 2.32 / 2.30 /
2.32 ticks and total top-5 depth 49.2 / 47.5 / 48.4 across
acceptance / sweep / unresolved; every Cohen's *d* is below 0.25. The only
notable difference is timeframe composition: unresolved penetrations are 99.98%
M1 against 87.5% and 94.8%.

The signature is an **absence of future interaction** — price penetrates, never
returns, and the level is never touched again. That conflicts with the Eye's
causal contract, which cannot condition a fact on the future not happening.
UNRESOLVED is therefore treated as a snapshot censoring marker and is
deliberately **not** a registered v1.3 concept.

## v1.2 → v1.3

Both months were replayed bar for bar under both versions on identical data.
**Every registered v1.2 event kind returned exactly its v1.2 count** — 40
unchanged kinds in February, 41 in March. v1.3 is an identity and naming change,
not a detector change.

| change | 2022-02 | 2022-03 | why |
|---|---:|---:|---|
| `bos_state` removed | 15,748 → 0 | 17,676 → 0 | unregistered transport carrying only `pending`, already stated by the absence of a terminal on `RAW_BOUNDARY_BREAK` |
| `origin_zone_created` retired | 43 → 0 | 57 → 0 | replaced by the split below |
| `base_origin_core_created` | → 43 | → 57 | geometry alone |
| `qualified_origin_zone_created` | → 43 | → 57 | core + displacement + BOS |
| `fvg_first_retest` | → 1,031 | → 1,234 | the first re-entry, with its entry context frozen |
| `delivery_phase_entered` | → 7,953 | → 9,004 | one per occupancy per timeframe |
| `delivery_phase_exited` | → 7,948 | → 8,994 | names its successor |
| `delivery_phase_updated` | → 3,420 | → 3,731 | registered-input changes only |
| `balance_range_observed` | → 1 | → 1 | the two-sided test standard |
| `dealing_range_activated` retired | 0 → 0 | 0 → 0 | was a balance claim wearing a location lifecycle; never fired under either version |

Non-event changes: Structural Leg `rank` → `path_class` with a strict lookup;
densified no-trade bars admitted into definitional paths under one shared
`BarCoverage` rule with `synthetic_path_minutes` / `synthetic_window_minutes`
markers; `ObservedExecutionReality` separated from `AssumedExecutionModel` with
`execution_source = assumed_default` when no real input exists.

The origin-zone split is exactly 1:1:1 in both months. Nothing was invented and
nothing was lost.

## v1.3 lifecycle entities

Six facts the Eye already produced were being republished every bar, renamed by
every break, or dated by the wrong clock.  Each is now a named object with a
life, so a consumer counts episodes instead of rows.

| entity | identity | opens | ends | why it exists |
|---|---|---|---|---|
| **Structure Generation** | `{tf}_{scope}_generation_{n:04d}` | a confirmed direction with an originating break on that bar | protection accepted through, else direction reversed | `parent_structure_generation` used to be whichever break fired last, so a phase spanning four breaks reported four parents |
| **Relation Generation** | `{parent}__{child}_generation_{n:04d}` | a role established under a named pair of structural claims | the role or either structural claim changes | `RelationState` is recomputed every minute, so one 50-bar parent retracement was 50 observations |
| **Liquidity Level Generation** | `{level_id}_generation_{n:04d}` | level created, or re-armed after a departure | swept | a swept level used to be deleted, so the same price offered again minted an unrelated identity |
| **Swing Geometry Node** | the swing's own id | confirmation freezes its definitional window | never; the tree is re-settled as windows appear | geometric nesting and semantic role were the same number |
| **Base Origin Core** | `base-origin-core-v1(...)` over the frozen cluster and its impulse | the impulse locks the preceding opposite candles | (geometry does not expire) | published only at qualification, so every core ever seen was one that worked |
| **Delivery Phase Occupancy** | per timeframe | a phase is entered | the phase changes | (v1.3, earlier in this round) |

Definitions in force:

- **Structure Generation termination precedence.** Accepting through the
  protected swing also clears `external_direction`, so both conditions fire on
  one bar. The protection failure is the cause and wins; naming it a reversal
  loses the only fact that explains it.
- **Relation Generation signature** is
  `(parent_structure_generation_id, child_structure_generation_id, role)`. A
  parent retracement under H1 generation 42 is a different episode from one
  under generation 43 even though the role text is identical.
- **Level taxonomy.** *Same Level, New Generation* when the same source object
  at the same price is armed again after a departure. *New Level* when a
  different structural source produces a level at that price — the
  discriminator is the source identity, never the price. So "same price + same
  source + new structure" is a **New Generation** if the source swing survives
  the structure change, and a **New Level** if the new structure produced a new
  swing.
- **Rearm condition** (no new threshold; the frozen foundation rule): a
  disarmed level becomes `rearmable` on the first close strictly outside its own
  band on the side it was offered from, and `rearmed` on the first close that
  comes back inside that recorded reach. The reach is the extreme, so a still-
  receding close extends it rather than arming the level early. Only `armed`
  levels appear in `unswept_bsl` / `unswept_ssl`.
- **Geometric containment** is the whole rule: parent window starts no later,
  ends no earlier, low no higher, high no lower, and is strictly longer. It is
  settled across every timeframe at once — two swings on one timeframe share a
  window length and can never enclose each other, so a per-timeframe tree is
  always flat. `geometric_depth` and `semantic_rank` are independent.
- **Crossing terminals** carry `constituent_bar_ids`, `penetration_bar_id`,
  `reentry_bar_id`, `hold_bar_id` and `outside_close_ids`, derived once in
  `_append_crossing_resolution` from the terminal's own ancestry so all eight
  emission sites report them identically.
- **Structural Range and Balance Range are separate claims.** A registered
  structural interval locates price — boundaries, normalized location,
  premium/discount, IRL/ERL — from creation, with no maturity check. Running
  out of room to prove balance now ends the balance claim alone
  (`balance_claim_abandoned`) and leaves the interval live; only a close outside
  the interval ends the interval.

## What the entities changed, measured

Two full months replayed bar for bar and independently: 2022-02 (27,360 bars,
12,933 confirmed Swings) and 2022-03 (31,740 bars, 14,537 confirmed Swings),
sources `docs/evidence/v1_3_structure_reading_2022_0{2,3}.json`.  The second
month is a replication, not a larger sample: every figure below is reported for
both so that a number which moved between them is visible as such.

**Efficiency — how many rows a consumer reads to learn one thing.**

| | v1.2 rows | v1.3 episodes | compression |
|---|---:|---:|---:|
| cross-timeframe relations, 2022-02 | 136,800 | **2,333** | 58.6× |
| cross-timeframe relations, 2022-03 | 158,700 | **2,794** | 56.8× |
| structural claims, 2022-02 | 2,829 breaks | **1,927** generations | longest absorbed 1,466 |
| structural claims, 2022-03 | 3,231 breaks | **2,226** generations | longest absorbed 1,364 |

Relation occupancy spans agree closely across the two months — median 16 bars
both, p90 135 and 130, longest 5,040 and 3,120.  This is the pseudo-replication
the work exists to remove: a parent retracement held for 5,040 bars used to
enter a study 5,040 times, and the inflation factor is ~58×, not a rounding
error.

**Precision — whether a fact is dated when it became true.**

| | 2022-02 | 2022-03 |
|---|---:|---:|
| base origin cores published | 634 | 731 |
| of those, ever qualified | 43 (6.8%) | 57 (7.8%) |
| qualification lead, median | 300 s | 300 s |
| qualification lead, max | 1,200 s | 900 s |
| levels tracked | 7,166 | 7,952 |
| levels re-armed after a sweep | 4,603 | 5,089 |
| reaching generation 2 / 3 | 3,673 / 930 | 4,015 / 1,074 |
| Swings geometrically nested | 12,849 / 12,933 (99.4%) | 14,285 / 14,433 (99.0%) |

- The **lead is strictly positive**, which is the whole point of task 8: the
  core is published a median of one M5 bar *before* the break that qualifies
  it, so it can no longer be created in hindsight.  v1.2 published only the 43
  and 57 that worked, hiding 93% and 92% of the impulse geometry the Eye
  actually saw.
- **Every semantic rank — micro, internal, structural, external — occurs at
  every geometric depth 0–4** in both months.  That is the decoupling of task 7
  stated as a measured fact rather than an intention.
- The 2022-03 geometry population (14,433) is smaller than that month's
  confirmed Swings (14,537) because the month contains one
  `MARKET_EPOCH_RESET`, which clears the tree by design; the 104 Swings
  confirmed before it keep the geometry they had and leave the live population.

**Structural range without a balance gate.** 25 and 27 structural intervals
were registered; 24 and 26 were invalidated by a close outside the interval.
All of them locate price from creation, which under v1.2 they could not do
until a balance claim matured.

**FVG first retest — the evidence that settles the expiry question.**

| | 2022-02 | 2022-03 |
|---|---:|---:|
| gaps created | 1,044 | 1,275 |
| ever first-retested | 1,031 (98.8%) | 1,234 (96.8%) |
| median age at first retest | 2 bars | 2 bars |
| p90 / max age | 34 / 1,642 | 33 / 842 |
| first retests crossing a session | 151 | 180 |

Time to first retest, as a share of retests:

| within | 1 bar | 3 | 6 | 12 | 24 |
|---|---:|---:|---:|---:|---:|
| 2022-02 | 45.4% | 64.9% | 73.1% | 81.5% | 87.2% |
| 2022-03 | 46.6% | 65.2% | 75.2% | 81.9% | 88.5% |

Mean fill depth at first retest, by age:

| age at first retest | ≤1 | ≤3 | ≤6 | ≤12 | ≤24 | >24 |
|---|---:|---:|---:|---:|---:|---:|
| 2022-02 | 0.75 | 0.70 | 0.73 | 0.76 | 0.78 | 0.72 |
| 2022-03 | 0.73 | 0.71 | 0.68 | 0.76 | 0.75 | 0.74 |

Flat in both months, and the two months agree to within a few points in every
bucket.  An old gap behaves like a young one on the only measure recorded so
far, so there is no age at which the population separates and no basis for a
TTL.  `fvg_expiry_bars` stays `null`.

## Three defects the replay exposed

Neither is a semantic change; both are recorded because they invalidated
earlier runs and because one of them was a live correctness bug.

1. **The cross-timeframe geometry settle was quadratic per bar.** Task 7's
   first implementation re-derived the whole tree on every published bar, over a
   population that only grows.  Measured on three days of real data: 1,190 s of
   a 1,694 s replay, **70.3% of wall time**, and rising as O(nodes² × bars) — a
   month could not finish.  The tree is now maintained incrementally: a newly
   confirmed Swing searches only the windows that can contain it and adopts
   only those that fit inside it, both bounded by its own window rather than by
   history, and a bar that confirms nothing does no work at all.  The same
   three days: **2.6 s, 0.3% of wall**, with the settled views persisted so the
   reducer carries them forward.  Adoption stays retroactive by necessity — an
   enclosing higher-timeframe window is confirmed at or after the Swing it
   contains, so an append-only assignment would leave almost everything a root.
2. **`TemporalMarketSceneGraph._neighbors` yielded in set-iteration order.**
   Adjacency is stored in `dict[str, set[str]]`, whose order varies with the
   interpreter's hash seed and with the insertion history a pickle round trip
   rebuilds.  Two of its four callers already wrapped it in `sorted()`; the
   breadth-first path search did not, and it returns the *first* route it
   reaches — so the same event log could read out a different path in a
   different process, or after a checkpoint resume.  For a system whose claim is
   replay determinism this is a real hole, not a test artefact.  The order is
   now imposed on `edge_id` at the source.  Reproduced across 8 hash seeds
   before the fix and green across the same 8 after.
3. **The per-bar rollback point deep-copied the history it protected.**
   `publish` snapshots the publisher before running so a raising bar leaves no
   half-applied state, and it took that snapshot with `copy.deepcopy` over two
   containers that grow with every confirmed Swing. A profile of the expensive
   tail of a replay attributed **56% of all time to `copy.deepcopy`** — 290
   million calls, 627 s of 1,117 s — so the cost of being able to undo one bar
   grew without bound. Both containers hold values that are never mutated in
   place (frozen views; primitives built fresh by `to_primitive` and only ever
   replaced wholesale), so copying the containers alone restores exactly as much
   state. Measured over the same 4,000 bars: **883 s → 473 s, 1.9× faster**, and
   the per-bar cost now grows 5.5× rather than 7.8× across that range.

## Known semantic problems

1. **The balance claim cannot be reached, and the cause is structural rather
   than a threshold.** Both months report `balance_observed: 0` and
   `balance_matured: 0`. Instrumenting every evaluation over six days (13
   evaluations across 7 ranges) shows why, and neither reason is a gate value:

   - **The gate is only ever evaluated at birth and at death.** A range's only
     registered transitions are formation and breaking — the month's counts
     agree exactly (25 created + 24 invalidated = 49 `dealing_range_state`
     events) — and the observed lifecycles are `forming` (7) and `broken` (6),
     nothing between. There is no intra-forming update to hang an observation
     on, so a range that *did* get tested twice a side mid-life has no clock at
     which to say so.
   - **At those two clocks the count cannot have moved.** Boundary touches are
     read from the source S/R zone's `total_touch_count`. The registry states
     the consequence itself — "a range whose source pair exists always has at
     least one touch per side and that alone is structure rather than evidence
     of balance" — which is exactly why the threshold is 2: the second touch is
     the evidence. But formation is where the count is 1 by construction, and
     breaking is price leaving the interval, which is not a touch. Observed
     distribution over the six days: `lower` was **1 in all 13** evaluations,
     `upper` was 1 in 12 and 2 in one.

   The registry already records the cause as a known limitation: "The Group-4
   detector publishes no intra-forming update, so the observation clock is that
   next registered transition rather than the second boundary test itself;
   moving it earlier is a Group-4 protocol change." So this is not an
   implementation that fails to do what it declares — it is a declared concept
   whose clock makes it unreachable in practice. Tuning
   `balance_range_boundary_touches_each` would not change that, and it was
   correctly out of scope. The structural half of task 10 is done and measured
   (25 and 27 intervals locate price from creation, which they could not do
   under v1.2); **the balance half remains unverified, because a detector that
   never fires cannot be said to work.**
2. **`BALANCE_RANGE_OBSERVED` is published one transition late.** The detector
   has no intra-forming update to hang it on. Given problem 1 this has not been
   observable in production data.
3. **UNRESOLVED penetrations (20.6% of crossings) have no registered concept**,
   by design, because their signature is the non-occurrence of a future event.
4. **`FVG_EXPIRED` and `DEALING_RANGE_EXTENDED` remain reserved** with no
   detector rule in either version. `fvg_expiry_bars` is registered with
   `value: null`: v1.3 does not infer expiry from age, and no threshold will be
   set until the retest statistics show high-age gaps approaching a random
   baseline.
5. **The Eye's hot state still grows without bound.** Sampled every 500 bars:
   `swing_hierarchy` (and with it the geometry population) grows linearly and
   forever — 231 nodes at 500 bars, 1,875 at 4,000 — as does the event store
   (6,188 → 51,535) and the liquidity candidate set (91 → 785). `event_memory`
   is correctly capped at 512 and `structural_legs` is bounded, so the
   accumulators are identifiable rather than diffuse. Because `to_primitive` and
   `TimeframeState.__post_init__` walk the whole Swing population on every
   published bar, per-bar cost stays proportional to history even after defect 3
   was fixed, making a month O(bars²). Bounding `swing_hierarchy` would change
   v1.2 behaviour and is not attempted here.
6. **`smc_trader/model.py` imports Brain modules** (`path_belief`,
   `dol_probability`, `dol_ranking`). It is a shared type module, not an
   Eye-only module, and the layering test does not catch this.

## What none of this establishes

These are event-population counts and one book study over three development
windows. The two months agree closely with each other, which is evidence that
the *measurement* is stable — it is not evidence that any of it predicts
anything. Nothing here validates predictive value, tests a trading rule, fits a
model, or opens an out-of-sample window. In particular the compression figures
say a consumer now reads 58× fewer rows; they say nothing about whether those
rows carry an edge. Status remains
`preregistered_development_contract_not_oos_trading_authority`.
