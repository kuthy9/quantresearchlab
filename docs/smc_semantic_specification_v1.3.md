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
[Canonical Semantic Foundation v2.1](canonical_semantic_foundation_v2.1.md).
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
and the current combined Engine checkpoint schema is 12
(`NEUTRAL_ENGINE_CHECKPOINT_SCHEMA_VERSION` in `smc_trader/engine.py`). Earlier
schemas cannot resume into the current Observation, Foundation, and
Neutral-state contracts.

This document is the whole atomic-layer contract. It absorbed the separate
preregistered-semantics document on 2026-09-06, so the registered concepts and
their formulas, the identity binding, the two-month and 2024-06 evidence, the
v1.2 → v1.3 delta, the lifecycle entities, and every defect the replay exposed
are stated here rather than split across two files. Its status is
`preregistered_development_contract_not_oos_trading_authority`: atomic identity
`smc_semantics_v1.3` ·
`f92b24c86bf942defc88de4edb7be16cc2a30dd64fde3b4432657780648b1f0c`, Foundation
identity `smc_semantic_foundation_v2.1` ·
`69428dbfd2a9b2aa19f0254391fca2da17aedb8d0206829572e69c0cc212a715`.

[`semantics/registry.yaml`](../semantics/registry.yaml) and
[`semantics/parameters.yaml`](../semantics/parameters.yaml) remain the immutable
v1.1 definitions required to verify the historical January 2024 protocol-v2
artifact. A v1.1 result does not establish v1.2 behavior, and a v1.2 runtime
does not retroactively upgrade a v1.1 event ledger.

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

The machine-readable ledgers behind these windows -- the two-month replay
comparison, the two balance-range gate studies and the 2024-06
unresolved-interaction study -- were retired on 2026-09-06. Every count this
document relies on is stated here in full; re-running a study reproduces a new
artifact, not those bytes.

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
sources, retired on 2026-09-06 with the rest of `docs/evidence/`'s JSON ledgers.  The second
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

## What the data said about balance

40 structural ranges in February and 41 in March. **One** per month reached the
two-touch standard. **None** matured, in either month, and
`sole_blocking_gate` is empty in both: no range was ever one gate away.

These are the pre-`balance_range_v1.2` numbers, and the first row is the gate
under its old evidence and old name — `bilateral_touches`, counting the source
zone's structural touches. It is now `bilateral_price_tests`. The rest of the
table is unaffected by that change.

| gate | threshold | unmet 2022-02 | unmet 2022-03 | observed median (02 / 03) |
|---|---|---:|---:|---|
| bilateral_touches (old evidence) | ≥ 2 | 39/40 | 40/41 | 1.0 / 1.0 (max 2.0) |
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

## Two v1.3 defects closed after the first two-month replay

Both were recorded above as known problems of v1.3 and are now fixed. Neither
moves a registered threshold: `balance_range_boundary_touches_each` is still 2,
and no Group-4 maturity gate was touched. The atomic definition identity moves
anyway, because the parameter file records both changes, so the freeze policy's
drift detector still reports them rather than hiding them.

### 1. `BALANCE_RANGE_OBSERVED` was unreachable, for two stacked reasons

Both months reported `balance_observed: 0`. The diagnosis took two passes,
because fixing the first cause exposed a second one underneath it that the
first had been hiding.

**The clock.** Group 4 recomputes every balance statistic on **every completed
H1 bar** and writes them straight back onto the live range, but only a
*lifecycle change* ever reached the emitter. A candidate that simply keeps
forming has no next transition, so the earliest clock at which the Eye could
say "both boundaries have now been tested" was, for most ranges, the transition
that destroys them. The two clocks that did exist are exactly the two at which
the count cannot have moved: formation is where the count is 1 by construction,
and breaking is price leaving the interval, which is not a touch. Instrumenting
every evaluation over six days (13 evaluations across 7 ranges) found `lower`
was **1 in all 13** and `upper` 1 in 12 and 2 in one.

The observation clock is now every completed H1 bar on which the range is live.
Re-instrumented over 4,000 bars that turned 2 evaluation clocks per range into
**1,841 evaluations across 6 ranges**, up to 13 distinct H1 clocks for a single
range — and changed the result not at all. Every one of the 1,841 still read
`(1, 1)`.

**The evidence.** The counter could not move, and not because the market was
quiet. Boundary touches were read from the source S/R zone's
`total_touch_count`, and `_update_reference_contacts` skips
`source_kind == "structural_swing"` zones outright — which is precisely the
kind Group 4 builds range boundaries from. For those zones the only path that
increments is `_add_touch`, which fires when *another confirmed swing* forms
inside the zone. Price trading into the boundary, however many times, never
counted. Measured over 2022-02: of 18 H1 zones the detector saw, **17 peaked at
one touch for life** and one reached two; of the 9 ever consumed as a range
boundary, **all 9 stayed at one**. A second confirmed H1 swing landing inside
one narrow band during a range's 8–24 bar life is rare by construction, so the
registered standard of 2 was unreachable in principle rather than merely
demanding.

That `continue` is deliberate, not a slip: it separates structural zones from
reference zones. So the fix is not to delete it but to give balance its own
evidence, which is what `balance_range_v1.2` does — see below. The structural
touch count is untouched and keeps its meaning; it still feeds
`boundary_test_strength` and the scene graph, and it travels beside the balance
evidence on every `BALANCE_RANGE_OBSERVED` so a consumer can tell the two
apart.

### 1b. `balance_range_v1.2`: what a boundary test is

Moving the clock proved the clock was not the binding constraint, so the
evidence itself was re-registered. The sub-protocol carries its own version and
the global semantic version stays at `smc_semantics_v1.3`, because no
registered threshold moved: the two-sided standard is still 2, and no maturity
gate parameter was tuned.

A **test** is a completed H1 bar reaching into a boundary's tolerance band,

    eps = max(1 tick, 0.25 x the ATR known before that bar)

classified `touch_only`, `shallow_penetration`, `deep_penetration` or
`close_outside`, escalating to the deepest reached. A **generation** is one
continuous visit: price must leave the band before the next test can open, so a
run of bars hugging a level counts once rather than once per bar.
`close_outside` remains the range's own invalidation and is not a second
concept. The maturity gate `bilateral_touches` became `bilateral_price_tests`
and reads these generations; `boundary_test_strength` and the scene graph keep
reading the structural touch count, which is untouched.

The band is sized correctly, which had to be checked before any conclusion
about rarity — an over-wide band would keep price permanently in-band, never
close a generation, and produce the same "everything stuck at 1" signature as a
rare phenomenon. Measured over 6,000 bars of 2022-02:

| | min | median | max |
|---|---:|---:|---:|
| band / range width | 0.021 | **0.056** | 0.157 |
| band (points) | 8.85 | 21.86 | 32.79 |
| range width (points) | 131 | 442 | 474 |
| prior H1 ATR (points) | 35.4 | 87.4 | 131.2 |

In-band occupancy was 10 of 61 live H1 evaluations on the upper boundary and 5
of 61 on the lower. Generations close constantly; the band is not the problem.

### What the two months then said

Each month was replayed alone, through the Eye-authority configuration, and
compared kind by kind against the same month replayed immediately before this
change.

| | 2022-02 | 2022-03 |
|---|---|---|
| bars | 27,360 | 31,740 |
| wall time | 3h36m (12,972 s) | 2h48m (10,092 s) |
| total events | 356,943 | 405,240 |
| **event kinds whose count changed** | **1 of 47** | **0 of 47** |
| `balance_range_observed` | 1 → **0** | 0 → 0 |
| `balance_range_matured` | 0 → 0 | 0 → 0 |
| range lifecycles | forming 25 / broken 24, unchanged | forming 27 / broken 26, unchanged |

Across two months and 762,000 events, exactly one event changed. The store
fingerprint moves in both months regardless, because it binds the atomic
definition identity and that identity necessarily changed with the protocol —
it is not evidence that the stream moved, and the kind-by-kind comparison is.

The one event moved *down*. That is the
intended semantics rather than a regression: February's single observation came
from a range whose source zone happened to gain a second confirmed swing, which
under `balance_range_v1.2` is not evidence that price tested the boundary. The
old counter was not merely late, it was counting the wrong thing.

**Balance remains unreachable in this data, and the reason is now measured
rather than inferred.** The per-bar record shows price *traversing* ranges
rather than oscillating inside them: over 6,000 bars, 6 of the 10 recorded test
generations were `close_outside` — the interaction that ends the range. One
candidate spent thirteen consecutive live H1 bars 300+ points below its upper
boundary and 25-90 points above its lower one, never reaching either band.

The arithmetic says this is structural, not incidental. Median range width is
442 points against a median H1 ATR of 87, so width/ATR is about **5.1** —
already past the registered `maximum_width_atr_at_formation` of 4.0, which is
why the width gate was unmet in 22 of 40 ranges before any of this. Testing
both sides twice inside a 24-bar window would need roughly 4 x 442 = 1,768
points of traversal, about 0.85 ATR of net directional travel every hour for a
day.

This reads as a candidate-selection problem, and that was the first
conclusion drawn here. **It did not survive its own control.** Measuring the
same evidence on arbitrary rolling H1 windows — intervals no filter has
touched — gives a bilateral revisit rate of 4.06%. Group 4's ranges behave like
arbitrary windows, and at that base rate a run of zeroes is exactly what a
sample this size produces. Balance is rare in this market; the width arithmetic
above describes *why* two-sided trade is uncommon at H1, not a defect in how
Group 4 picks intervals. See
`docs/evidence/balance_candidate_hypotheses_2026-09-05.md` for the three-arm
comparison that settles this, and for the one generator that does beat the base
rate.

### 2. The hot Swing set grew for the life of the process

`swing_hierarchy` kept every confirmed Swing forever, so
`TimeframeState.__post_init__`, the rank projection onto live liquidity, the
geometry settle and `to_primitive` all walked a population proportional to
elapsed bars. A month therefore cost O(bars²) however cheap any single step
became.

A Swing leaves the hot set once nothing can still read it. Exactly two reads
can reach an older Swing, and both were measured over 2022-02:

| read | maximum reach | behaviour |
|---|---:|---|
| a later rank assignment naming an older Swing | **4** Swings back | flat across 3,500 bars |
| the geometry tree's retroactive adoption | **447** Swings back | plateaued from bar 2,500 |

The geometric reach is bounded by the span of the largest enabled timeframe's
confirmation window, not by history, which is why it plateaus rather than
grows. `swing_hierarchy_hot_retention` is registered at **2048** per timeframe:
500× the assignment reach and 4.6× the geometric one. The complete role history
stays in the immutable event stream, exactly as it already does for
`structural_legs`.

Two supporting changes were needed and are worth recording:

- **The settle short circuit had to stop being a length test.** While the
  hierarchy grew forever, "the tuple got longer" was an exact "this timeframe
  confirmed nothing" test. Once the hot set is bounded its length stops
  changing while Swings keep arriving, so that test would have silently stopped
  admitting every Swing confirmed after the bound was reached. It is now tuple
  identity, which the reducer already guarantees for an event that confirms
  nothing.
- **The per-bar rollback point no longer copies the tree.** Defect 3 above
  replaced a deep copy with a shallow one, which was far cheaper but still
  proportional to the whole Swing population on every bar. The tree now
  journals the entries a publish is about to overwrite and replays them
  backwards, so undoing one bar costs what that bar changed. `_starts` is
  maintained beside the id order for the same reason: the bisect that bounds
  every parent search used to rebuild it from every node.

### What the bound does and does not change

Replaying the same bars with the hot set bounded and unbounded, through the
Eye-authority configuration (`persist_state_projections=False`), 1,200 bars at
a deliberately punitive retention of 64:

| | unbounded | bounded (64) |
|---|---:|---:|
| canonical semantic events | 15,224 | 15,224 |
| events present in only one arm | — | **0** |
| shared events differing in `strength` | — | **0** |
| event id order | — | **identical, position by position** |

So the bound changes no semantic fact the Eye publishes. It does change the
**state projection** events, necessarily and by construction: those are a hash
of the reduced state itself, and `swing_hierarchy` is part of that state. A
3,000-bar run through `ContinuousSMCEngine` (which persists projections by
default) returns an identical event count of 70,859 in both arms with a
different `event_prefix_fingerprint` for exactly that reason. An identical
count with a different fingerprint is the signature of projection transport
moving, not of a detector moving.

## The Structural Range lifecycle drops its balance-derived states

**Date: 2026-09-06.** `DealingRangeLifecycle` was `forming -> mature -> broken`
and is now `active -> broken`. `BALANCE_RANGE_MATURED` moves to
`reserved_not_emitted`. No Group-4 candidate selection rule changed, no
threshold moved, and no new detector was built.

### Why

`mature` was a grade the interval earned by balancing, and `forming` meant it
had not earned one yet, so 96% of ranges were recorded as having failed. The
three-arm comparison removed the grounds for that: 0 of 84 ranges reached a
two-sided test, an arbitrary rolling H1 window reaches one 4.06% of the time,
and the 95% interval on the range arm (0, 4.37%) still contains the base rate.
The grade was recording a failure that was not occurring.

### What the range keeps

`created` is the event and `active` is the state. A range is a location from
the moment it is created, so a separate `CREATED` lifecycle state would have
zero duration and no observation could ever catch it; the birth is published as
`DEALING_RANGE_CREATED` and the state it enters is `ACTIVE`. Exactly two range
transitions exist per range — creation and break — and the entity timeline is
now `("active", "broken")` rather than `("forming", "mature", "broken")`.

Balance evidence is untouched. It is still collected on every completed H1 bar
and still published once per range as `BALANCE_RANGE_OBSERVED`. When it meets
the registered standard the claim *settles*, which is recorded on the range as
the `balance_claim_confirmed` reason and the `balance_confirmed_at` clock
(previously `mature_at`) and is deliberately **not** a transition: the interval
is the same location it was on the bar before.

### Two defects the change exposed

1. **Boundary promotion lost its publisher.** The candidate liquidity levels a
   settled claim mints were emitted from inside the `MATURE` transition. With
   no such transition the inventory was still built but the levels were never
   published, so inventory and event stream would have diverged silently.
   Promotion now runs as its own pass over the live population
   (`_record_confirmed_range_boundaries`), anchored on the range's creation
   event — which is the truer provenance anyway: a boundary descends from the
   interval that froze it, not from the claim that promoted it. The event
   store's contract was updated to require `DEALING_RANGE_CREATED` as that
   parent instead of `BALANCE_RANGE_MATURED`.
2. **A broken range kept supplying Brain context.** `lifecycle is MATURE` had
   carried two meanings at once — the claim is confirmed *and* the range is
   alive — and replacing it with `balance_confirmed_at is not None` silently
   dropped the second, because that clock survives the break. Three call sites
   in `playbooks.py` now test both. This is the failure mode of splitting an
   enum whose members carried more than one fact: every meaning has to be
   written out again explicitly.

## Known semantic problems

1. **Balance is a maturity state of an object that does not exhibit it, and
   width is not the reason.** After `balance_range_v1.2` the detector records
   real evidence, but no range reached two test generations on both sides in
   either month. A width-stratified study of all 52 ranges
   (`docs/evidence/structural_range_width_strata_2026-09-05.md`) tested the
   obvious explanation and falsified it: `P(bilateral)` is **0 in every
   stratum** from ≤2 ATR to >5 ATR, with no gradient, and the single-sided rate
   moves the wrong way — `upper ≥2` *rises* with width. Median lifetime rises
   monotonically with width (2.0 → 12.0 H1 bars), so the narrowest ranges are
   the ones that die fastest rather than the ones that get tested. Even the far
   weaker "touch each side once, ever" is met by only 9 of 52. 50 of 52 ranges
   ended with `close_beyond_frozen_range`, and 40 of the 71 recorded
   interaction generations were `close_outside` — the interaction that destroys
   the range. Price transits these intervals at every width instead of
   oscillating inside them. This is consistent with Structural Range and
   Balance Range being separate market objects, and with the forced
   parent → maturity relation measuring a property the object does not have.
   Removing that relation is a semantic change and is not attempted here. The
   band itself was checked and is not the constraint (median 5.6% of range
   width, in-band on 12% of live bars). A later three-arm comparison
   (`docs/evidence/balance_candidate_hypotheses_2026-09-05.md`) put numbers on
   it: an arbitrary rolling H1 window shows a bilateral revisit 4.06% of the
   time, an independent shape scan 8.36%, and the Structural Range 0 of 84 with
   an upper bound of 4.37%. Balance is simply rare; the Structural Range is not
   worse than random at finding it, only no better, and the shape scan is
   separated from both. Decoupling is supported but not cleanly, because a
   window's extremes are touched by construction and a frozen boundary is not.
2. **UNRESOLVED penetrations (20.6% of crossings) have no registered concept**,
   by design, because their signature is the non-occurrence of a future event.
3. **`FVG_EXPIRED` and `DEALING_RANGE_EXTENDED` remain reserved** with no
   detector rule in either version. `fvg_expiry_bars` is registered with
   `value: null`: v1.3 does not infer expiry from age, and no threshold will be
   set until the retest statistics show high-age gaps approaching a random
   baseline.
4. **The event store and the liquidity candidate set still grow without
   bound.** Sampled every 500 bars, the event store went 6,188 → 51,535 and
   the liquidity candidate set 91 → 785 across 4,000 bars. `event_memory` is
   capped at 512, `structural_legs` is bounded, and `swing_hierarchy` is now
   bounded too (below), so the remaining accumulators are identifiable rather
   than diffuse. The Swing geometry tree also still retains every node it ever
   admitted: that is memory and checkpoint size only — no per-bar step walks
   it any more — and it is deliberate, because evicting a node is the one thing
   that could change a later geometric adoption.
5. **`smc_trader/model.py` imports Brain modules** (`path_belief`,
   `dol_probability`, `dol_ranking`). It is a shared type module, not an
   Eye-only module, and the layering test does not catch this.

## Existing-system migration decision

The implementation reuses the repository's causal multi-timeframe aggregation
and existing Swing, structure, liquidity, displacement, FVG/Origin Zone, and
dealing-range detectors. v1.2 does not introduce a parallel detector or market
state stack. Detector output becomes immutable canonical atomic events, and
those events are the accepted inputs to hierarchical state reducers.
Compatibility/state-projection events may remain readable for transport or
historical replay, but they are not a second semantic authority.

**Definition validity** and **empirical validity** are separate for every named
concept. A causal, deterministic, replayable definition may remain empirically
unknown or unsupported. Conversely, coverage or a contemporaneous association
cannot repair an ambiguous definition or grant predictive/OOS/action authority.
The frozen v1.2 registry remains the identity of historical artifacts. The
foundation is deliberately a separate versioned identity, so current
store/reducer/tick hardening and lifecycle completion do not rewrite those
artifacts or silently change their definitions.
The separate DOL/Belief/Temporal supplement that carried dependency-cluster
handling, research-only temporal links, FVG first-lifecycle freezing, and the
2024-06 data role was removed on 2026-09-06. None of it added a canonical
market event or rewrote this semantic version; the executable contracts remain
in `configs/dol_probability.json` and the runtime modules themselves.

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
formal runner validated without opening data by default, but the template never
bound a non-zero intent ledger or minute source, was never frozen, and produced
no empirical result. That whole surface, and the Phase 9 harness
(`phase9_shadow_live_v1.3`, `shadow_compact_runtime_v8`,
`phase9_shadow_component_digest_v3`), were retired on 2026-09-06. The
authoritative end-to-end handoff remains open.
Earlier deterministic parity ran behind `NullExecutionGateway`; its 6,900-clock
historical cold-start input and exact 200-clock checkpoint/resume/cold-replay
prefix are retained evidence for the earlier path/model bindings
`5213b3d6…`/`4214da19…`. The current dependency protocol changed those bindings,
so that output remains historical. A separate Foundation-enabled Engine run
passed the same exact 200-row prefix under later, but now also historical,
model/registry bindings; its detailed boundary is in the
[Foundation release table](canonical_semantic_foundation_v2.1.md#replay-test-and-empirical-boundary).
The read-only 6,900-clock capacity preflight still verifies the historical
`COMPLETED.json` to checkpoint-manifest SHA binding, finds that receipt's
historical/current identity mismatch, and explicitly withholds full-run
authority. Operational acceptance metrics, a rematerialized complete
historical run, and an actual real-time multi-day pilot remain open. Calibrated
promotion, empirical Execution Research, operational Shadow Live, rolling OOF,
and final OOS remain later gates; conformance to the complete target-state
prompt is therefore **partial**.
The input and checkpoint were ignored local engineering files under `inputs/`
and `outputs/`; both directories, and the Phase 6–9 gate report that listed
their paths, hashes, commands and no-replace publication rules, were removed on
2026-09-06.
See the
[current implementation status](current_implementation_status.md).

## What none of this establishes

These are event-population counts and one book study over three development
windows. The two months agree closely with each other, which is evidence that
the *measurement* is stable — it is not evidence that any of it predicts
anything. Nothing here validates predictive value, tests a trading rule, fits a
model, or opens an out-of-sample window. In particular the compression figures
say a consumer now reads 58× fewer rows; they say nothing about whether those
rows carry an edge. Status remains
`preregistered_development_contract_not_oos_trading_authority`.
