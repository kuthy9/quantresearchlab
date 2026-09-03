# Preregistered Semantics: Definitions, Formulas, and What Two Months Showed

Status date: 2026-08-29
Atomic identity: `smc_semantics_v1.2` · Foundation identity: `smc_semantic_foundation_v2.0`

> **Superseded for the runtime.** Every measurement in this document was
> produced by a `smc_semantics_v1.2` / `smc_semantic_foundation_v2.0` replay and
> keeps that identity. The runtime has since moved to `smc_semantics_v1.3` /
> `smc_semantic_foundation_v2.1`; v1.3 keeps every canonical emitted EventKind
> identical, so the counts below remain the current expectation until the v1.3
> two-month replay has now been re-run under v1.3 and confirms every v1.2
> count unchanged; see
> [the v1.3 comparison](../evidence/v1_3_two_month_replay_comparison.md). Also see
> [the v1.3 delta](../smc_semantic_specification_v1.md#what-v13-changed) and
> [the v1.3 preregistered-semantics document](preregistered_semantics_v1_3_2026-08-31.md),
> which replaces this one for the runtime.
Foundation canonical-JSON SHA-256: `0c49da28e103f0515d3eb93ab03e8659e334d2477449f5174df3b5e8b0b84cc6`

This document states what the Eye is registered to mean, the exact frozen
parameters and formulas behind each meaning, and what a complete replay of
`2022-02` and `2022-03` measured against them. Every number here comes from
`docs/evidence/eye_event_statistics_2022_02.json` and
`docs/evidence/eye_event_statistics_2022_03.json`.

When first written, nothing in this document changed a registered definition.
Sections 1.2 and 1.3 have since been acted on and record what was done; the
remaining problems are still findings, not edits.

---

## 1. Known semantic problems

### 1.1 `MATURE` is dead vocabulary in the dealing-range lifecycle

`DealingRangeLifecycle` registers three states. Across both months — 59,100
bars and 81 dealing ranges — the middle state was never reached once:

| Month | created | → mature | → broken |
|---|---|---|---|
| 2022-02 | 40 | **0** | 39 |
| 2022-03 | 41 | **0** | 40 |

Every range died one of two ways:

| Transition reason | 2022-02 | 2022-03 | Total |
|---|---|---|---|
| `close_beyond_frozen_range` | 33 | 35 | 68 (84%) |
| `maturity_deadline_elapsed` | 6 | 5 | 11 (14%) |

**Why.** `configs/primitives_range.json:maturity` requires *ten* conditions to
be simultaneously true at one completed H1 close, inside an 8–24 bar window:

```
candidate_real_h1_bars           >= 8   and <= 24
lower source total_touch_count   >= 2
upper source total_touch_count   >= 2
midpoint_crossings               >= 2
inside_close_fraction            >= 0.80
width_atr_at_formation           <= 4.0
compression_ratio                <= 0.80
both source zones remain active or tested at the maturity clock
the current completed H1 close remains inside the frozen outer bounds
```

Two of these pull against each other. `inside_close_fraction >= 0.80` demands
price stay inside the range on 80% of closes, while `midpoint_crossings >= 2`
plus two touches on *each* boundary demands it traverse the range repeatedly —
all within 8 to 24 hours, while a single H1 close outside the frozen bounds
terminates the candidate immediately.

**Open question, not a proposed fix.** Whether the thresholds are too tight or
maturity is genuinely a rare regime is empirical, and 81 samples over two
months cannot settle it. What is certain is that `MATURE`, and every
downstream rule that reads it, is currently unreachable.

### 1.2 `bos_state` transports a single lifecycle value

`BOSLifecycle` registers `pending / confirmed / failed`. Both months emit only
the first:

| Month | `bos_state` events | distinct lifecycle values |
|---|---|---|
| 2022-02 | 15,748 | `{pending}` |
| 2022-03 | 17,676 | `{pending}` |

The terminal outcomes travel as *separate event kinds* instead —
`structure_break` (confirmed) and `structure_break_failed` (failed). So 33,424
`bos_state` events across two months carry no state distinction at all: they
say only "a BOS exists".

Either `bos_state` should carry its own terminal transitions, or it is
redundant transport that duplicates what `structure_break*` already says.

**Resolved 2026-08-30 — removed.** It was never a registered semantic: the 33
`event_bindings` contain no `bos_state`, and it was cited as neither source nor
context by any event in either month. `raw_boundary_break` cites the
`STRUCTURE_BREAK` variant of the same emitter variable, not the pending one.

Removal needed four coordinated edits, because a pending BOS does exist in
snapshot state and `bos_state` was its timeline anchor:

| Location | Change |
|---|---|
| `semantic_event_emitter.py` | the pending branch emits nothing |
| `event_memory.py` | the `bos` timeline now begins at `{confirmed, failed}`; `pending` leaves the transition table; `_TIMELINE_LIMITS` 2 → 1 |
| `observation.py` | a pending BOS no longer claims a retained timeline |
| `model.py`, `scene_graph.py`, `visualization.py` | the enum member and its display mappings |

The BOS timeline now begins at its terminal. "Pending" is no longer a state
needing an event: it is exactly "a raw break was emitted and no terminal has
followed", which the presence of `raw_boundary_break` without
`structure_break*` already states.

### 1.3 Execution reality is a fabricated constant, not an observation

Not a registered-semantics defect, but it reaches the same consumers. No
caller anywhere supplies an `ExecutionRealityInput`. Every Engine bar therefore
scores the *default* input, producing a fixed value that simulation bills every
trade against:

```
source                          'missing'
spread_points                   0.25     (tick size, substituted)
expected_slippage_points        0.25
expected_round_trip_cost_points 0.975
anomalies  ('spread_missing_used_one_tick', 'deadline_missing')
```

The anomalies are truthful and nobody reads them. Cost accounting that looks
observed is in fact a constant.

**Partly addressed 2026-08-29.** The Eye no longer derives this — scoring moved
to `ContinuousSMCEngine._score_execution`, so the Eye transports a value it does
not compute. The constant itself is unchanged and still reaches simulation;
separating an observed reality from an assumed default model remains open.

### 1.4 Sample poverty on the higher timeframes

Emission concentrates almost entirely on M1. Two-month totals:

| Concept | 1m | 5m | 15m | 1H | 4H |
|---|---|---|---|---|---|
| `swing_confirmed` | 23,125 | 3,011 | 1,004 | 264 | 66 |
| `structural_leg_created` | 17,905 | 2,429 | 802 | 207 | 47 |
| `sweep_confirmed` | 20,693 | 764 | 177 | 48 | **5** |
| `qualified_bos` | 1,835 | 272 | 114 | 31 | **9** |

Any 4H or 1H statistic here is arithmetic, not evidence. Two months yields five
4H sweeps.

`displacement_observed` (5,764) and `fvg_created` (2,319) are **M5-only by
protocol** — that is a registered choice, not a gap.

---

## 2. Data evidence

### 2.1 Replay coverage

| | 2022-02 | 2022-03 |
|---|---|---|
| Bars replayed | 27,360 | 31,740 |
| Result | complete | complete |
| Events | 351,801 | 399,311 |
| Distinct kinds | 42 | 43 |
| `EventStore` fingerprint | `66c02fc238c73211…` | `8735ad932b1997f6…` |
| Unresolved ancestry references | **0** | **0** |
| Extra paths exercised | — | contract roll, DST transition |

Origin split: 2022-02 — 191,460 legacy transport / 125,111 canonical atomic /
35,230 normalized data; 2022-03 — 217,963 / 140,478 / 40,870.

### 2.2 Canonical emission counts

| Event kind | 2022-02 | 2022-03 |
|---|---|---|
| `level_touched` | 25,837 | 28,852 |
| `level_penetrated` | 25,563 | 28,540 |
| `liquidity_level_created` | 15,000 | 16,772 |
| `swing_confirmed` | 12,933 | 14,537 |
| `acceptance_confirmed` | 10,238 | 11,264 |
| `sweep_confirmed` | 10,188 | 11,499 |
| `structural_leg_created` | 10,081 | 11,309 |
| `raw_boundary_break` | 4,380 | 5,072 |
| `displacement_observed` | 2,697 | 3,067 |
| `structure_direction_confirmed` | 2,272 | 2,589 |
| `mss_core_confirmed` | 1,763 | 2,036 |
| `qualified_bos` | 1,066 | 1,195 |
| `fvg_created` | 1,044 | 1,275 |
| `fvg_fully_filled` | 543 | 658 |
| `fvg_invalidated` | 469 | 560 |
| `fvg_midpoint_touched` | 322 | 391 |
| `fvg_partially_filled` | 292 | 366 |
| `protected_swing_assigned` | 222 | 263 |
| `origin_zone_created` | 43 | 57 |
| `dealing_range_created` | 40 | 41 |
| `dealing_range_invalidated` | 39 | 40 |
| `dealing_range_replaced` | 39 | 39 |
| `origin_zone_mitigated` | 35 | 51 |
| `origin_zone_invalidated` | 5 | 4 |
| `market_epoch_reset` | 0 | 1 |

### 2.3 Two invariants that held

**Crossing resolution is conserved and never doubles.** Every penetration
resolves into at most one terminal:

| | penetrations | sweeps | acceptances | unresolved at window edge |
|---|---|---|---|---|
| 2022-02 | 25,563 | 10,188 | 10,238 | 5,137 |
| 2022-03 | 28,540 | 11,499 | 11,264 | 5,777 |

**Qualification is a branch, not a ladder.** A raw break resolves into a
qualified BOS *or* an MSS core, never both: the emitter selects on
`item.scope` in a single `if`/`elif`, so `CONTINUATION` produces
`QUALIFIED_BOS` and `OPPOSED` produces `MSS_CORE_CONFIRMED`.

```
raw_boundary_break        4,380 / 5,072
  → structure_direction   2,272 / 2,589
       ├── CONTINUATION → qualified_bos  1,066 / 1,195   (24.3% / 23.6% of raw)
       │                    └── protected_swing 222 / 263 ( 5.1% /  5.2% of raw)
       └── OPPOSED      → mss_core       1,763 / 2,036   (40.3% / 40.1% of raw)

          neither branch taken           1,551 / 1,841   (35.4% / 36.3% of raw)
```

The two branches are mutually exclusive and sum with the unqualified
remainder to the raw-break total (1,066 + 1,763 + 1,551 = 4,380). An earlier
revision of this document drew them as consecutive funnel stages, which
implied MSS was a subset of qualified BOS; the implementation has always
treated them as competing outcomes. Only the diagram was wrong.

### 2.4 Session distribution

Identical in both months: **62%** of penetrations occur in
`overnight_delivery`, with sweeps, qualified BOS, displacement and FVG all
between 61% and 64%. RTH (`morning` + `midday` + `afternoon`) accounts for 26%.
Part of this is clock length — the overnight session spans more minutes — but
the stability across two independent months makes it structural rather than
incidental.

### 2.5 Anomaly census

All benign and all explained (2022-02):

| Tag | Count | Explanation |
|---|---|---|
| `warmup_4H` | 3,779 | 15.7 native bars against `minimum_bars=16` |
| `warmup_1H` | 1,439 | 24.0 against 24 |
| `warmup_15m` | 359 | 23.9 against 24 |
| `warmup_5m` | 119 | 23.8 against 24 |
| `warmup_1m` | 29 | 29.0 against 30 |
| `scheduled_market_closure` | 9 | nine weekday 17:00→18:00 breaks |
| `scheduled_weekend_closure` | 2 | two weekends |
| `group4_ambiguous_dual_side_sweep` | 1 | a real dual-side ambiguity, flagged not guessed |

Zero `data_gap_history_reset`, zero contract-change resets, zero provenance
anomalies.

---

## 3. Registered concepts: definition and formula

Seventeen concepts, all owned by the Eye. Definitions are the registered
`operational_definition`; formulas are the frozen parameters they read.

### 3.1 Structure

**`confirmed_swing`** — a local price turning point; the atomic node of the
structure tree. A pivot must dominate the frozen left and right windows and is
emitted only after the right window completes.

```
k_left  = k_right = 2   (4H, 1H, 15m, 5m)
k_left  = k_right = 1   (1m)
prominence_ATR = null   → continuous feature, no positive hard threshold
```

At confirmation the pivot takes the `MICRO` role. Append-only same-timeframe
assignments later promote it to `INTERNAL` (exact leg endpoint), `STRUCTURAL`
(direction source), or `EXTERNAL` (protected swing). Later roles never rewrite
or backdate the confirmed swing.

**`structural_leg`** — the complete movement between opposite confirmed swings.

```
efficiency = abs(close_end - close_start) / sum(abs(delta_close))
path_class = MICRO if path length <= 2 bars else INTERNAL
```

`path_class` was named `rank` through v1.2. The repository holds four unrelated
`rank` fields — a leg's path length, a swing-role assignment, a liquidity
candidate's structural rank, and a DOL candidate's ordering — and the name is
the only thing they had in common. v1.2 keeps `rank`: its registry file is
sha256-bound by fourteen experiment manifests, several of which record runs that
actually happened.

Foundation v2 additionally freezes `amplitude_ticks`, `atr_at_leg_start` (from
exactly 14 strictly-prior **real** completed bars), `close_efficiency`,
`extreme_path_efficiency`, close/wick MAE, the full ordered path BAR ancestry,
and `synthetic_path_minutes`.

**`structure_direction`** — the confirmed same-timeframe direction that exists
*before* a break is classified. Emitted once per structure generation from the
exact confirmed high and low swings knowable at that generation's
`confirmed_at`. A child timeframe never rewrites a parent's direction.

**`raw_boundary_break`** — a completed close beyond a frozen confirmed
boundary, with no directional qualification.

```
break_buffer_ticks = 0, comparison = strict_close_beyond
```

**`qualified_bos`** — a raw break aligned with the same-timeframe confirmed
structure direction that existed at `known_at`. Displacement and FVG are
context, never definition fields.

**`mss_core`** — the first confirmed structural-boundary break *opposite* the
internal direction immediately preceding it.

**`protected_swing`** — the opposite structural swing at the origin of the most
recent same-timeframe BOS, held protected until the registered acceptance
break completes.

```
bos_post_break_later_bars = 1
```

Wick, close and acceptance breaks stay distinct observations.

### 3.2 Liquidity

**`candidate_liquidity_level`** — an observable level that may attract orders.
Explicitly *not* a claim about institutional stops. Created only from confirmed
swings, equal highs/lows, previous completed session/day/week extremes, and
frozen dealing-range boundaries.

**`liquidity_sweep`** — temporary penetration followed by return to the prior
side. `LEVEL_TOUCHED` and `LEVEL_PENETRATED` are emitted first for one frozen
level and `crossing_generation_id`; `SWEEP_CONFIRMED` only when a later
completed close returns within the frozen window.

```
pool_post_cross_resolution_bars      = 1
range_manipulation_real_1m_bars      = 5
```

**`acceptance`** — sustained trading outside a crossed level. Sweep and
acceptance are **mutually exclusive** terminals for one crossing generation.

```
range_manipulation_real_1m_outside_closes                    = 2
active_dealing_range_external_acceptance_completed_h1_closes = 1
```

### 3.3 Delivery

**`displacement`** — fast, efficient, directionally coherent delivery. Emitted
per lifecycle transition (started / active / exhausted / censored) with frozen
continuous metrics; no isolated ATR threshold defines it.

```
net_points      = direction_sign * (last_admitted_close - frozen_origin_price)
travel_points   = seed_body_travel + sum(abs(close-to-close) of later admitted bars)
efficiency      = max(net_points, 0) / max(travel_points, tick_size)
speed_atr_per_bar = relative_atr / admitted_real_episode_bar_count
body_continuity = same-direction body points
                  / (same-direction + opposite-direction body points)
displacement_score_threshold = null   → continuous, unthresholded
```

**`delivery_phase`** — balancing, expanding, retracing, reversal-attempting or
transitioning. In v1.2 this is **snapshot-derived, not an event**: computed
from same-timeframe confirmed structure, the active leg, protected-swing
intactness, and range availability. Range-extension and volatility-compression
are *not* implemented v1.2 inputs.

### 3.4 Zones

**`fvg`** — a three-candle non-overlapping interval.

```
bullish  when low[c3] - high[c1] >= min_fvg_ticks
bearish  symmetric
min_fvg_ticks = 1, strict non-overlap on the tick grid
fvg_expiry_bars = null  → no age-based expiry in v1.2
```

Confirmed at c3 close; source bars and bounds frozen. `FVG_EXPIRED` is
**reserved and not emitted**.

**`order_block_origin_zone`** — the price origin of a qualifying impulse: the
frozen geometry of the last opposite candle or cluster before it. Derived from
exact impulse and BOS identities, never an independent causal agent.
`ORIGIN_ZONE_MITIGATED` is canonical; `ORIGIN_ZONE_TOUCHED` is a
compatibility alias and is not emitted.

### 3.5 Range and location

**`active_dealing_range`** — the external interval through which structure is
currently delivering, created from a frozen valid pair of opposite structural
anchors. Four production-reachable invalidation variants stay distinct. See
§1.1 for the maturity conditions and the fact that none is ever satisfied.

```
dealing_range_extension = null   → no extension rule in v1.2
```

**`premium_discount_irl_erl`** — continuous position within the active range.

```
x = (price - low) / (high - low)          stored continuously
IRL  when strictly inside the frozen bounds
ERL  when equal to a boundary or outside
```

Without an *active* range — including forming, replaced, invalidated, or
descriptive envelopes — membership is unresolved and
`normalized_location_in_range` is null.

**`dol_candidate`** — visible candidate inventory with side, price, timeframe,
source kind, strength, event-sourced rank, age and IRL/ERL membership. The Eye
**does not** select a unique DOL, populate a path probability, or claim an
authoritative obstacle ranking.

### 3.6 Emission population

| Status | Count | Meaning |
|---|---|---|
| `canonical_emitted` | 25 | production-emitted event kinds |
| `snapshot_derived` | 3 | computed into the snapshot, never an event |
| `compatibility_alias_not_emitted` | 3 | accepted name, never emitted |
| `reserved_not_emitted` | 2 | registered for a future version |

Both months emitted 24 of the 25 canonical kinds; the 25th
(`market_epoch_reset`) appeared only in March, at the quarterly contract roll.

---

## 4. Freeze policy

From `semantics/parameters_v1_2.yaml`:

> Freeze semantic definitions, split, primary outcome, controls, search range,
> and acceptance rule before final out-of-sample evaluation. Any semantic
> parameter change requires a new semantic version and fresh validation.

`null` means "retain the continuous feature without thresholding in v1.2" —
it is a deliberate non-threshold, not an unset value. Four parameters are null
by design: `prominence_ATR`, `fvg_expiry_bars`, `dealing_range_extension`,
`displacement_score_threshold`.

Consequently every item in section 1 that implies a parameter change —
notably the dealing-range maturity conditions — requires a **new semantic
version**, not an edit to v1.2.
