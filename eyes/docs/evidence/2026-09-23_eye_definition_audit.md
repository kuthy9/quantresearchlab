# Eye definition audit — receipt (2026-09-23)

A read-only audit of the Trading Eye's SMC definitions against standard
practitioner usage, the registered v1.3 texts (spec, registry, parameters,
protocols) and the running code, with every high/medium claim checked against
the 2023 Eye journal. No code was changed. No strategy is proposed.

## How it ran

| item | value |
| --- | --- |
| code | `c1a62f2` plus the uncommitted `eyes/core/range_auction.py` synthetic-bar compaction fix (the same tree the 2023 scan ran on) |
| static audit | 7 independent auditors by concept group (structure; BOS/MSS; displacement/leg/phase; liquidity; FVG/OB; dealing range; boundaries), each reading spec → registry → parameters → protocol → code → tests |
| empirical | one streaming pass over the 2023 scan's cold journal `outputs/eye_scans/edge_2023/_state/journal/*.evlog`: 4,975,893 `MarketEvent`s, 2022-12-01 → 2023-12-26 16:59 ET (the last ~3 days live only in `eye.pkl`, not opened), plus the scan's bar rows |
| verification | one adversarial verifier re-read the cited code for all 83 high/medium claims: 0 refuted, 11 severities or numbers corrected (applied below); 8 issues added that the auditors missed |
| spot checks by hand | the 15m lock chain (below), `displacement.py:379-389,819-850`, `observation.py:3462-3471,3712-3722`, `semantic_event_emitter.py:1233-1320,3184-3232,3281,5451-5474`, `market_state.py:1605-1624,2544-2700,2963-2966` |

Production authority: `build_eye` sets `range_auction_projection_only=False`,
so the published structure, delivery and range state come from the atomic
event reducer `reduce_timeframe_state` (`market_state.py:2421-3040`), not from
`MarketSnapshotPublisher._structure/_formalize_structure`, which runs only in
the frame-projection mode (`shares/core/eye_factory.py:62-66`,
`observation.py:910-918`).

Every scale (1m/5m/15m/1H/4H) uses **micro** fractal pivots (span 2, 1 on 1m).
`SWING_CONFIRMED` is `micro` on 100 % of 2023 events; `nesting_depth` is
always 0; `STRUCTURAL_LEG_CREATED.path_class` is never `structural` or
`external`. "External" and "internal" below are therefore labels on one pivot
size, not a swing hierarchy.

## The four most consequential findings

### F1. External direction and its protected level can deadlock until the contract roll (confirmed)

Chain (code, verified):

1. A continuation `QUALIFIED_BOS` assigns `PROTECTED_SWING_ASSIGNED` = the
   start swing of the leg that ends at the BOS target
   (`semantic_event_emitter.py:3136-3245`), only if it does not loosen the live
   protection (`protection_is_monotonic`, 3184-3197); the reducer raises on a
   loosening assignment (`market_state.py:2603-2613`).
2. While a protection is live, an opposite `STRUCTURE_DIRECTION_CONFIRMED` is
   ignored (`market_state.py:2544-2572`), an opposite continuation BOS is
   downgraded to `RAW_BOUNDARY_BREAK` only (`semantic_event_emitter.py:3070-3095`),
   and `MSS_CORE_CONFIRMED` changes only `internal_direction` / `last_mss`
   (`market_state.py:2632-2638`). The tracker's own structure `BROKEN` has no
   reducer branch.
3. The only release is an opposite `ACCEPTANCE_CONFIRMED` that names the exact
   swing *and* the live assignment event (`market_state.py:2646-2697`;
   emitter custody `semantic_event_emitter.py:1185-1231`).
4. That acceptance can come only from the swing's single crossing generation:
   a swing becomes `BROKEN` once, on its first close beyond (`structure.py:970-1002`),
   the generation is keyed on `broken_at` (`semantic_event_emitter.py:1233-1260`),
   and the next native close decides sweep vs acceptance once; the 1m
   inventory route is one-shot per item id (`observation.py:2984-2992`).

So if the protected swing's first crossing resolves as a **sweep** (close
beyond, next close back inside), or resolved before the assignment existed,
or the swing left the tracker's 256-record window, no fact can ever release
the protection. In a counter-trend rally every new same-direction origin
loosens the protection, so re-assignment is refused too. Only the roll reset
clears it.

2023 evidence (journal):

- 15m short protection on swing `71a785fdddb71ad013cad29a`, high 14,327.75,
  assigned 2023-10-26 03:15 ET. 04:07 a 1m wick above with a same-bar close
  back inside (`sweep_confirmed`, 1m, `same_bar_close_returned_inside`);
  04:15 a 15m close above (`raw_boundary_break` long); 04:30 the next 15m close
  back inside → `SWEEP_CONFIRMED` (`later_close_returned_inside_confirmed_swing_price`);
  11-02 `LIQUIDITY_RETIRED candidate_aged_out`. Then 2,854 15m closes above the
  level, up to +1,918 pts, with no further event on it. `ext_15m` short and
  `protected_intact_15m` true on 2,982 / 2,982 15m bars until the 12-11 roll.
- In that window 15m emitted 67 long `STRUCTURE_DIRECTION_CONFIRMED`, 60 long
  MSS and 67 long continuation raw breaks; 0 became `QUALIFIED_BOS` long, and
  51 short QBOS produced 0 new 15m protections.
- Protections closed beyond on their own scale but never released:
  15m 27, 1H 9, 4H 3, 5m 44, 1m 201. Every one had a prior sweep (or a
  pre-assignment acceptance: 1H 1 → live 312 h, 4H 1 → live 844 h). None was
  attributed to swing eviction (that path is real but latent in 2023).
- Longest runs: `ext_15m` short 1,145 h (ended by the 12-11 roll); `ext_5m`
  long 1,499 h (09-13 → 11-15, through a low of 14,147.75 from 15,567);
  `ext_4H` short 1,172 h; the 1m `PROTECTED_SWING_ASSIGNED` gap 2,031 h.
- When the acceptance path fires it works: `protected_intact` flipped within
  20 min for 15m 54/54 native and 38/39 1m-route acceptances, 1H 24/24 and
  17/17, 4H 7/7 and 4/4.

### F2. 4H displacement cannot exist; 1H only 10:00–17:00 ET (confirmed)

The displacement baseline is exactly 14 prior true ranges
(`displacement.py:379-389`), cleared by `on_boundary` (`819-850`), which every
registered session closure triggers (`displacement_observer.py:102-111,248-253`).
The first possible `STARTED` is the 16th real bar of a session. A session has
6 4H bars and 23 1H bars. 2023: 4H 0 events (4H is registered in
`configs/primitives_displacement.json:7`); 1H earliest start 15 h into the
session. Downstream: no 4H base-origin core, OB or displacement-linked FVG;
1H cores/OBs only in the RTH afternoon; `displacement_score` on 4H is None on
100 % of bars; 15m has 4 h and 5m 80 min of dead time after each 18:00 open.

### F3. MSS carries no sweep link and displacement context only on 5m (confirmed)

`prior_sweep` is hard-coded `None` (`semantic_event_emitter.py:3281`) on all
22,089 MSS, although a same-scale same-direction `SWEEP_CONFIRMED` preceded
770 / 892 15m MSS within 24 h. `displacement_context_present` and
`legacy_mss_qualified_context` are set only by `_enrich_mss_breaks`, which runs
only on the 5m frame and only on OPPOSED breaks (`observation.py:3465-3471,
1521-1587`); they are always equal; `QUALIFIED_BOS.displacement_context_present`
is False on all 13,213 events.

### F4. Contract rolls drop open zones silently and reset every scale at once (confirmed)

Zone hard-boundary terminals are built (`zone.py:664-711`) but the observer
skips emitting them for hard reasons (`observation.py:3712-3722`): 531 open
FVGs and ~10 origin zones vanish at the 5 resets with no terminal event,
against `configs/primitives_zones.json:55` ("emits the terminal states").
Dealing ranges get only the legacy `DEALING_RANGE_STATE`; liquidity pools,
zones and pending crossings are cleared with no terminal
(`observation.py:1452-1459`). Warm-up after each 2023 roll to the first
non-None value: `ext_5m` 1.3–2.1 h, `ext_15m` 5–13 h, `ext_1H` 15–31 h,
`ext_4H` 47–148 h (12–35 4H bars); `leg_4H`/`int_4H` 28–60 h. The Brain gets
no epoch or warm-up flag.

## Per concept: expected → actual → difference → impact → test

Severity is the verifier's. "Test" names the check that would expose the
difference; where a test already exists it is named.

### Swing

- **Expected.** Practitioner: an N-bar fractal (N = 2), significance separating
  internal from swing/external pivots, EQH/EQL as liquidity. Registered:
  `k_left = k_right = 2` (1 on 1m), `prominence_ATR` continuous with no cutoff,
  break = first later close strictly beyond, rank MICRO promoted append-only to
  INTERNAL/STRUCTURAL/EXTERNAL.
- **Actual.** Centre of the last `2·span+1` real bars, strict on both sides in
  ticks (`structure.py:386-418,1163-1216`); an outside bar yields neither side
  (1192-1196); `BROKEN` on the first later real close beyond, once
  (970-1002); rank MICRO at confirmation and never promoted in practice.
  `_swings` holds 256 records including FORMATION_FAILED ones; eviction pins
  only BOS targets (`structure.py:60,1004-1030`) — a swing is observed for
  ~334–387 real bars (≈3.9 trading days on 15m, 15 on 1H, 60 on 4H).
- **Difference.** (SW-1, medium, latent) the retention bound is unregistered
  and ends a non-target swing's break/crossing lifecycle; (SW-2, medium) one
  crossing verdict per swing per path; (HH-1, medium) the hierarchy is dead —
  no rank above MICRO; (SW-3, low) equal extremes and outside bars drop pivots.
- **Impact.** The protected swing can become permanently unobservable (F1
  path 3); "external" structure has no larger pivot behind it.
- **Test.** Unit: V to a swing high P, 400 real bars below P, then closes above
  → assert P is absent and never `BROKEN`. Contract: after a multi-day trend,
  some swing ranked above MICRO (fails today). Existing:
  `test_outside_bar_never_confirms_both_swing_sides`
  (`test_v3_structure_bos.py:201`).

### HH / HL / LH / LL

- **Expected.** Compare with the previous *structural* swing of the same side;
  EQH/EQL within a tolerance.
- **Actual.** Compared with the previous confirmed micro pivot of the same side
  in exact ticks (`structure.py:369-384,461-471`); EH/EL only on exact equality;
  the first swing per side after a reset is `NONE`; relations live in
  `SWING_CONFIRMED` evidence, not in the published structure state.
- **Difference.** (HH-1, medium) relations are noise-level, so structure
  direction whipsaws (15m: 67 long / 71 short SDC in 6.5 weeks); (HH-2, low)
  tie handling is exact ticks, inconsistent with the liquidity equal-level
  tolerance.
- **Impact.** SDC and QBOS fire at micro frequency; `internal_direction`
  flickers.
- **Test.** 15m sequence H110, L100, H108, L101, H109, L99 → relations
  `[NONE, NONE, LH, HL, HH, LL]`, a LONG structure CONFIRMED at H109 (the
  verifier's correction), then BROKEN at L99. Existing:
  `test_tick_relation_table_is_exact_and_atr_independent`.

### Protected level

- **Expected.** Practitioner: the extreme swing that originated the impulse
  that broke structure; trails after each BOS; a close through it is a CHoCH.
  Registered: bind the opposite swing of the latest same-timeframe BOS origin
  and keep it until the registered acceptance completes; same-timeframe
  evidence.
- **Actual.** The start swing of the structural leg ending at the BOS target;
  same-side runs collapse to the **latest** swing, not the extreme
  (`market_state.py:3972-3985`); no leg → no assignment; monotonic (tighten
  only); origin lifecycle not checked; release only by the exact acceptance.
  Two different "protected" swings coexist: the tracker's (latest HL/LH,
  `structure.py:584-586,673-714`, used for MSS scope and structure BROKEN) and
  the emitter's (used for external direction).
- **Difference.** (PR-1, high) release can become impossible — F1; (PR-4 / QB-4,
  medium, confirmed) an already-BROKEN swing can be assigned (produced the
  844 h 4H lock); (PR-2, medium) latest not extreme; (PR-3 / LIQ-S2, medium) a
  1m two-close hold releases a 15m/1H/4H protection — ~43 % of 15m releases and
  4 / 11 4H releases in 2023, against "same-timeframe evidence".
- **Impact.** The Brain's governing-scale regime and every consumer of
  `protected_swing_intact` can be wrong for weeks; HTF regime can flip on 2
  minutes of 1m closes.
- **Test.** Observer test (F1): SHORT PSA at H; one 15m close above H, next close
  back below (SWEEP); twenty closes above H + 5 ATR → assert today external
  stays SHORT and intact True; the registered reading expects release.
  Variant: origin already BROKEN before assignment. Variant: 400 chop bars
  before the break. 4H test: 1m close below P twice while the 4H bar closes
  above → external None today.

### Internal vs external structure

- **Expected.** External = trend of major swings, flipped by a close through
  the external protected swing and confirmed by the next BOS; internal = the
  sub-structure of the current range on smaller pivots; BOS/CHoCH counted in
  both directions. Registered: an external generation persists until exact
  protected-break acceptance, rollover, reset or supersession; MSS alone
  never confirms the opposite regime.
- **Actual.** External set by an applied SDC (no live protection, or same
  direction) or QBOS; cleared to None only by the exact acceptance
  (`market_state.py:2544-2700`). Internal = last applied SDC/QBOS/MSS
  direction, or the last leg when None.
- **Difference.** (EXT-1, high) external follows micro SDC/QBOS until a
  continuation BOS latches it, then locks (F1); (EXT-2 / QB-1, high) opposite
  BOS is suppressed while any protection lives; (SD-1, high) the practitioner
  trend-change event (tracker BROKEN) never touches external; (EXT-4, medium)
  internal is not nested; (EXT-3 / TR-4, low) the two producers define
  external/internal differently with no parity test.
- **Impact.** `ext_15m` short 64.5 % of 2023; BOS counted only in the locked
  direction (the 2023 edge study's BOS family and htf strata measured this
  state machine); `DeliveryPhase` inherits the lock.
- **Test.** Replay the 2023 journal through `replay_atomic_market_snapshot` and
  log every `(external, protected id, intact)` change with its causing event
  kind per scale. Existing: `test_live_protected_regime_ignores_opposite_direction_until_mss`
  (`test_phase234_atomic_reducer.py:1399`) — its name contradicts its
  assertions (external stays after the MSS).

### BOS (raw break, scope, qualified BOS, post-break)

- **Expected.** A close beyond the last swing that made a new extreme, in the
  trend's direction; the spec says QBOS = a raw break that survived one later
  bar of acceptance.
- **Actual.** One pending target per direction = the latest same-side
  confirmed swing with relation ≠ NONE (`structure.py:1079-1161`); strict close
  beyond in ticks; scope OPPOSED if the target is the opposite structure's
  (tracker) protected swing, CONTINUATION if it is the same structure's latest
  high/low, else LOCAL (`851-886`). `QUALIFIED_BOS` = CONTINUATION and no
  opposite live protection, emitted **on the break bar**
  (`semantic_event_emitter.py:3071-3134`). The post-break state is resolved one
  bar later and shares the swing's single crossing generation.
- **Difference.** (QB-2, medium) no acceptance is awaited, against the spec;
  (PB-1, high) a one-bar pullback makes the crossing a SWEEP permanently;
  (SC-1, medium) a CONTINUATION BOS can fire on an LH/HL inside the trend;
  (SC-2, medium) scope uses tracker structures, not the published regime;
  (RB-1, medium) older or protected swings are never raw-break targets, so a
  structure can break with no BOS/MSS event (MSS-2); (RB-2, low-medium) a bar
  with one synthetic minute is invisible (~44–50 bars per scale per 13 months);
  (QB-3, medium) displacement context always False.
- **Impact.** QBOS includes breaks that failed on the next bar; failed breaks
  of the protected swing lock the regime (F1).
- **Test.** Unit: bull structure, close above target, next close back inside →
  assert QBOS exists and SWEEP follows (documents QB-2). Journal: share of
  QBOS whose post-break state is rejected. Existing:
  `test_opposite_continuation_keeps_raw_fact_without_rewriting_live_regime`
  (`test_semantic_v1_2_eye_extensions.py:732`).

### MSS / BOS → MSS transition

- **Expected.** Practitioner: the first close through the current trend's
  protected swing, often after a sweep and with displacement; a BOS in the new
  direction confirms the change. Registry: "the first confirmed
  structural-boundary break opposite the internal direction"; sweep,
  displacement, FVG and parent location as separate context tags.
- **Actual.** OPPOSED scope against the tracker's protected HL/LH
  (`structure.py:869-875`); emitted on the break bar with no sweep, displacement
  or protection gate; at most one per opposite structure generation; the
  reducer sets internal only. After it, new-direction breaks are LOCAL until
  the new tracker structure confirms, and even then SDC/QBOS are ignored or
  suppressed while the old protection lives.
- **Difference.** (MSS-1 / TR-2, high) MSS never transitions the regime and is
  judged on a different swing from the one guarding external; (TR-1, high)
  the handover can deadlock (F1); (MSS-3, medium) definition differs from the
  registry ("opposite the internal direction"); (EV-1, medium) `prior_sweep`
  None; (EV-2, medium) displacement context 5m-only; (MSS-4, medium) no FVG or
  parent-location tags; (TR-3, medium, inferred) MSS > QBOS (1.65:1) partly by
  construction (suppressed continuation, OPPOSED precedence).
- **Impact.** 29–51 % of 15m–4H MSS never see external flip within 72 h; the
  sweep → MSS combination (C2 in the 2023 study) cannot be measured from the
  Eye's own evidence.
- **Test.** Reducer+emitter test: live SHORT assignment H = 120, tracker SHORT
  protected LH = 110; close at 111 → MSS LONG, external still SHORT; acceptance
  at 110 does not release, only at 120. Contract test: MSS evidence keys cover
  the registry's context list (fails today). Journal: join each 15m/1H MSS
  break bar to an ACTIVE same-scale displacement; count how many would qualify.

### Displacement

- **Expected.** A fast one-sided large-body move judged on the scale being
  read; a 4H displacement is ordinary. Registered: STARTED/ACTIVE/EXHAUSTED/
  CENSORED with versioned thresholds, no score cutoff.
- **Actual.** STARTED needs a full 14-TR baseline, a body, ≥ 1 tick close
  progress, CLV ≥ 0.5 (`displacement.py:391-408,864-882`); ACTIVE when all six
  activation ratios ≥ 1; terminal priority qualified-opposite → protection
  broken → progress loss. One eye each on 5m/15m/1H/4H; baseline reset every
  session, synthetic candle and roll. `displacement_score` / `_direction` /
  `_at` are overwritten by **any** lifecycle event (`market_state.py:2720-2741`)
  and frozen in between; the Brain sees no lifecycle (`brain/core/eye_view.py:222-248`).
- **Difference.** (DISP-1 / B1, high) F2; (DSC-1 / B2, high) a CENSORED event at
  18:01 re-dates `displacement_at` and direction, so a dead episode looks fresh
  at every session open (192 / 214 / 200 such events on 5m / 15m / 1H); (DSC-2,
  medium) the score does not track an ACTIVE episode or decay; (DISP-2, medium)
  STARTED clusters at the first eligible bar of each session (e.g. 1H 10:00);
  (DISP-3, medium) no same-bar opposite seed on progress loss, which the
  protocol allows; (DISP-4, low) the protocol was widened to 4 scales without a
  new version. Memory correction: the 2026-09-18 note "non-5m score updates only
  on that scale's event" is true of every scale, 5m included.
- **Impact.** No HTF displacement evidence at all on 4H; a time-of-day artifact
  in 1H/15m displacement statistics; stale displacement read as fresh.
- **Test.** Unit: feed a 4H tracker a full session with a clear impulse on bar
  5 → assert no STARTED (documents F2). Reducer: CENSORED event → assert
  `displacement_at` changes today. Journal: STARTED counts by minute-of-session.

### Active leg (structural leg, forming leg) and delivery phase

- **Expected.** The active leg is the swing in progress, low lag, no flicker at
  one price; phase: expansion with the trend, retracement against it with
  protection intact, reversal attempt for an unconfirmed counter MSS.
  Registered: leg from one confirmed swing to the next opposite one; phase is
  an entered/updated/exited lifecycle, not a per-bar value.
- **Actual.** Legs fold over swings in pivot order; a same-side swing
  **replaces** the anchor (`market_state.py:3972-3974`); known at end-swing
  confirmation (median 3 native bars after the pivot). Forming leg (since
  `e0581ed`) = sign(last close − last leg's end price), recomputed on every real
  1m close for every scale (`market_state.py:1626-1698,3399-3415`). Phase:
  REVERSAL_ATTEMPT checked first (internal ≠ external and last MSS = internal),
  then EXPANSION if leg = external, RETRACEMENT if leg set and protection intact,
  else TRANSITION; BALANCE only when external is None and price is inside a
  live range.
- **Difference.** (PH-1, high) under an F1 lock whole rallies read
  REVERSAL_ATTEMPT / RETRACEMENT / short EXPANSION (15m 10-26 → 12-11: 1,469
  reversal-attempt, 838 expansion, 669 retracement bars); (LEG-1, medium)
  latest-not-extreme anchors, and BOS targets at replaced swings get no origin
  leg hence no protection; (LEG-A1, medium) the reference is the last leg's
  end, which can stay stale; (LEG-A2, medium) no hysteresis — the 4H leg and
  phase can flip within a 4H bar on 1-tick crossings, close == swing gives
  None/TRANSITION; (PH-2, medium) phase recomputed per minute, against the
  registered lifecycle intent; (PH-3, medium) after a fresh SDC or an
  acceptance a counter-leg reads TRANSITION, not RETRACEMENT; (LEG-A3, medium)
  `leg_4H` None up to 60 h after a roll; (PH-4, low) no BALANCE on 5m/1m, epoch
  reset leaves an occupancy without EXITED.
- **Impact.** Leg and phase inherit the external lock and add minute-level
  flicker on HTF scales (leg runs median 3 bars on every scale).
- **Test.** Reducer test: 4H leg end at P; 1m closes P+0.25, P−0.25, P+0.25 →
  assert three active-leg flips inside one 4H bar. Journal replay: phase
  occupancy per scale inside each F1 lock.

### Liquidity pool / level

- **Expected.** BSL/SSL beyond swing highs/lows, relatively equal highs/lows,
  PDH/PDL, PWH/PWL, Asia/London/NY session extremes; a run level is spent until
  rebuilt.
- **Actual.** Four sources: every confirmed swing (`swing:<id>`), equal pools
  (two same-side swings inside one frozen S/R band ± max(1 tick, 0.1 ATR)),
  M1 reference levels (previous Globex session, civil ET day, ISO week), and
  mature range boundaries (never reached in 2023). Retired by age, distance,
  pool resolution or period replacement — all explicit.
- **Difference.** (LIQ-C1, high) the reducer re-arms a swept level on a
  retracing close, but the observer never offers a consumed id again — a
  re-armed level is published as armed liquidity the Eye can never cross
  (0 of 265,425 resolved pairs had a second generation); (LIQ-C3 / LIQ-P2,
  medium) FORMED pools never expire; (LIQ-C4, medium) no Asia/London/NY levels,
  "day" is the civil date, nothing until a full period after each reset;
  (LIQ-C2, low, dormant) a range-boundary touch can retire as INVALIDATED;
  (LIQ-C5, LIQ-C6, low) HTF ages count 1m fan-out bars; `LEVEL_TOUCHED` means
  two different things for pools.
- **Impact.** The Brain's DOL inventory lists "unswept" levels that are dead.
- **Test.** Observer: swing low L swept by a 1m wick (same-bar close back
  above), price rallies and returns below L → assert no second
  `LEVEL_PENETRATED`, yet the reducer lists L armed (fails the expected
  contract).

### Liquidity sweep and acceptance

- **Expected.** Sweep = trade beyond, close back on the original side on the
  candle of the scale being read; acceptance = closes beyond that hold.
  Registered: touch/penetration per crossing generation, `SWEEP_CONFIRMED` only
  when a registered later close returns within the family's window.
- **Actual.** Four families. Swing and reference items on the 1m path: a strict
  1m wick beyond; close back inside on the same minute → same-bar SWEEP
  (resolution 0, no `source_timeframe`); else the next real 1m close decides.
  Native swing path: close beyond, next native close decides. Pools: first 1m
  wick, next 1m close. Pools and mature range boundaries under Group 4: the
  manipulation decides within 5 real 1m bars, or never (censored).
- **Difference.** (LIQ-S1, high) the same-bar 1m sweep reaches only the M1
  reducer — on 5m/15m/1H/4H the swept level stays armed as `penetrated`;
  (LIQ-S3, medium) rules differ by family, equality counts as inside;
  (LIQ-S4 / LIQ-M2, low in size) censored manipulations and non-primary range
  crossings leave penetrations unresolved (108 of ~520k); (LIQ-S5 / B6, medium
  rule, low count) resolution ignores sessions — the reopening gap bar decides;
  (LIQ-M1, medium) one live manipulation globally, concurrent sweeps BLOCKED;
  (LIQ-P1, medium) Group-4 pools get two verdicts; (LIQ-P4, low)
  `pool_post_cross_resolution_bars` is not wired; (LIQ-S6, low) exact tags are
  neither touched nor reached.
- **Impact.** HTF "unswept liquidity" misstates what was run; HTF sweep
  semantics are 1m-wick semantics.
- **Test.** Observer: 15m swing high, a 1m bar wicks above and closes below →
  assert the 15m candidate is still `penetrated`/armed (documents LIQ-S1).
  Journal: resolutions whose deciding bar is the 18:01 reopen.

### FVG

- **Expected.** Three-candle imbalance; graded by its displacement; revisits
  read as first touch, CE (50 %), full fill, close-through; old gaps dropped.
  Registered: strict non-overlap on the tick grid, `formation_atr` of ≤ 14
  real 5m bars, no age expiry, `FVG_FIRST_RETEST` once per gap.
- **Actual.** Contiguous real same-contract c1–c3 (`zone.py:1351-1480`); linked
  only if an ACTIVE same-direction displacement admitted c2; lifecycle on later
  real native bars (invalidated on a close through the far edge, fully filled at
  the far edge). Runs on 5m/15m/1H/4H.
- **Difference.** (ZN-FVG-1, medium) no linked FVG on 4H, 1H only 10:00–17:00;
  (ZN-FVG-2, medium) no expiry, capacity 256 fail-closed (2023 max open: 92 on
  5m), lives up to 1,446 h, the Brain gets every live FVG without age; F4
  silent drop at rolls; (ZN-FVG-3, low) formation ATR from as few as 2
  in-session bars; (ZN-FVG-4, low) `approach_speed_atr` takes the prior close
  from the 5m index for every scale; (ZN-FVG-5, low) one synthetic minute
  freezes a candle's fills; (ZN-FVG-6, low) docs still describe a 5m-only FVG;
  `FVG_EXPIRED` and `FVG_TOUCHED` are never emitted.
- **Impact.** HTF FVG grading is missing or biased to RTH; zone inventory grows
  with stale gaps until a roll.
- **Test.** Observer: create a 5m FVG, change `instrument_id` next minute →
  assert an `FVG_INVALIDATED` with `contract_change_reset` in the audit store
  (fails today).

### Order block (base origin core, qualified origin zone)

- **Expected.** The last opposite candle (or cluster) before the displacement
  that breaks structure (BOS or MSS); mitigated on return, invalidated on a
  close through the distal edge; often kept until invalidation. Registered:
  core = geometry of that cluster, no OB reading; qualified zone binds one core,
  the active displacement and "the QUALIFIED_BOS".
- **Actual.** Core frozen at displacement STARTED from the strictly opposite
  bar before the seed (`zone.py:1154-1288`); zone requires an ACTIVE
  displacement, a tracker BOS resolved on this bar (CONTINUATION, or OPPOSED
  with `mss_qualified` — 5m only) and the clock order
  `pending_at ≤ started_at ≤ resolved_at ≤ prefix_last_admitted_at`
  (`1482-1674`); terminal MITIGATED on the first intersecting bar.
- **Difference.** (ZN-CORE-1, high) no cores on 4H, 1H only 10:00–17:00;
  (ZN-OB-1, medium — verifier's correction) MSS-sourced OBs exist only on 5m;
  15m/1H OBs come only from CONTINUATION breaks; the claim that the external
  lock starves the opposite side was refuted — zones follow tracker scope, not
  the reducer lock; (ZN-OB-2, medium) the "qualified" zone is not bound to a
  `QUALIFIED_BOS` event and can be qualified by a break the atomic layer
  suppressed; (ZN-OB-3, medium, inferred) the strict clock likely rejects the
  textbook OB whose origin swing confirms after the seed; (ZN-OB-4, medium) a
  wick touching the near edge ends the OB (602 mitigated vs 37 invalidated);
  `ORIGIN_ZONE_CREATED/TOUCHED` never emitted; (ZN-CORE-2, low) the core is
  published one bar before its qualification, not "1:1:1 at one clock".
- **Impact.** OBs are rare (649 in 13 months, 7 on 1H, 0 on 4H) and live until
  first touch only.
- **Test.** 15m observer: bear structure, LONG displacement ACTIVE on the bar
  that closes above the protected LH → assert no qualified zone today
  (OPPOSED needs 5m enrichment). Journal: count BOS/MSS bars with an ACTIVE
  same-scale displacement and no zone, by failing funnel step.

### Dealing range (structural range, balance, premium/discount)

- **Expected.** The swing low to swing high of the current external leg,
  re-anchored on BOS; premium/discount at 50 %; range extremes and beyond = ERL.
  Registered: external interval from a frozen valid pair of structural
  anchors; four invalidation variants; balance observed at ≥ 2 tests per side,
  matured by a six-condition gate.
- **Actual.** The narrowest unbroken bracket of same-scale S/R swing zones of
  any rank around the close (`range_auction.py:1069-1178`); created on the first
  eligible native bar with no live range; one live range per scale; balance
  claim settles by maturity or is abandoned at 24 bars. Location re-priced on
  every 1m close.
- **Difference.** (RNG-SR-1, medium) not the external/protected range, never
  re-anchored on BOS/MSS; (RNG-SR-2, high) once the claim settles only a close
  outside or a hard boundary ends it, with no width cap — wide ranges live
  days (max 306 / 231 / 812 h on 15m / 1H / 4H); (RNG-PD-1, high) after
  `DEALING_RANGE_INVALIDATED` the reducer keeps publishing location and a
  premium/discount label on the dead interval (6.8 % / 10.3 % / 14.2 % of
  15m / 1H / 4H bars in 2023); (RNG-BR-1, medium) balance is almost
  unreachable (8 / 2 / 0 observed, 0 matured, 0 activated) so range-boundary
  liquidity, range manipulation and active acceptance are dormant; (RNG-SR-3,
  medium) `maturity_deadline_elapsed` has no producer, `active_acceptance`
  never occurs; (RNG-PD-2, medium) creation publishes x = 0.5 "equilibrium"
  until the next 1m bar; (RNG-PD-4, medium) the membership lifecycle set in
  code is wider than the frozen parameter; (RNG-BR-4, medium)
  `DeliveryPhase.BALANCE` is unrelated to the balance claim; (RNG-SR-4, medium)
  no range for 60–146 h on 4H after a roll; 5m/1m never have one.
- **Impact.** Premium/discount and IRL/ERL describe a micro bracket, sometimes
  a dead one.
- **Test.** Reducer: DEALING_RANGE_CREATED then INVALIDATED then a 1m bar →
  assert `location_label` is still set today (expected None or flagged).
  Journal: invalidated-to-next-created gaps per scale.

## Boundaries and warm-up (all concepts)

| event | structure / liquidity | displacement | zones | range | reducer |
| --- | --- | --- | --- | --- | --- |
| synthetic minute in an HTF bar | whole bar skipped | CENSORED + baseline reset | windows cleared, entities frozen | cutoff only | — |
| daily 17:00–18:00 break, weekend, holiday | nothing | CENSORED + baseline reset (F2) | windows cleared | nothing | displacement re-dated (DSC-1) |
| contract roll / data gap | all cleared, no terminals for pools/crossings | CENSORED | terminals built, not emitted (F4) | BROKEN (legacy only) | `MARKET_EPOCH_RESET` clears all |

Other boundary facts: 4H buckets anchor at 18:00 with a 3 h 14:00–17:00 tail
(B7, low); a densified 18:00 first minute would crash the session reducer
(B8, low); range auction and interaction skip the new contract's first 1m bar
(B12, low); ATR seeds differ by detector (tick size vs 1.0 vs None, B14, low);
candidate retirements and phase lifecycle events reach the snapshot one bar
late (B10, low); the Brain's ages count closed-market minutes
(`eye_view._age_bars`, B9, low). Only a hard boundary clears a stuck external
direction (B16, medium); the roll ends 19 of 310 15m runs and hides how long
they would have lasted.

## Never-emitted and dead definitions

Never emitted in 13 months: `fvg_touched`, `fvg_expired`,
`balance_range_matured`, `dealing_range_activated`, `dealing_range_extended`,
`delivery_phase_changed`, `origin_zone_created`, `origin_zone_touched`,
`swing_formed` (and the state-change kinds `timeframe_state_changed`,
`relation_state_changed`, `session_state_changed`, `foundation_state_changed`).
Dead ranks: swing INTERNAL/STRUCTURAL/EXTERNAL, leg STRUCTURAL/EXTERNAL.
Dormant chain: balance maturity → range-boundary liquidity → range
manipulation on range sources → active acceptance. `prior_sweep` is a field
with no producer.

## Dependencies between definitions

```text
swing ─┬─ relations ─ tracker structures ─┬─ BOS target / scope ─┬─ QUALIFIED_BOS ─ PROTECTED_SWING_ASSIGNED ─┐
       │                                   │                      └─ MSS (tracker protected) ─ internal only     │
       │                                   └─ structure BROKEN (no reducer effect)                              │
       ├─ structural legs ─ origin leg (PSA) ─ forming leg ─ delivery phase ◄─ external / internal ◄───────────┤
       ├─ liquidity items ─ crossing generation (one-shot) ─ SWEEP / ACCEPTANCE ─ release of protection ───────┘
       └─ S/R zones ─ dealing range ─ premium/discount, IRL/ERL, BALANCE phase
displacement (session-reset baseline) ─ base core ─ qualified zone (tracker BOS) ; 5m-only MSS context ; FVG grade
```

No code-level circular dependency was found; the range does not feed
structure or liquidity. F1 is a circular wait in state: the live protection
suppresses the opposite QBOS that would create a new protection, refuses a
looser same-direction one, and can be released only by a crossing generation
that is already spent.

## Doc / registry / runtime drift (low unless noted)

- Spec says QBOS "survived its post-break acceptance"; code emits on the break
  bar (medium).
- Spec says protection uses "same-timeframe evidence"; a 1m route releases HTF
  protection (medium).
- Registry MSS = "first break opposite the internal direction"; code uses the
  opposite tracker structure's protected swing (medium).
- `primitives_zones.json:55` "emits the terminal states" at hard boundaries;
  not emitted (medium).
- Spec "structure direction from confirmed breaks only"; code and registry use
  aligned relations.
- Zone protocol, parameters and spec still describe 5m-only FVG/OB; runtime
  runs 4 scales. Displacement protocol widened to 4 scales with no new version
  and advertises a 4H it cannot produce.
- Spec/registry say the base core is published only with its qualification;
  it is published at STARTED.
- Balance texts describe observation at the next transition, a MATURED event
  and "active" = matured — none of which occurs.
- `retained_swings = 256` and the tracker-level eviction are unregistered.
- `sweep_window.pool_post_cross_resolution_bars` is registered but not wired.
- `test_live_protected_regime_ignores_opposite_direction_until_mss` — the name
  contradicts the assertions.
- `PROTECTED_SWING_ASSIGNED` and `STRUCTURE_DIRECTION_CONFIRMED` carry an
  `event_time` hours before `known_at` (the origin pivot / first swing); any
  study keyed on `event_time` looks ahead (medium, a consumer risk).

## Limits

- One year (2023), NQ only, one Eye revision; the journal misses 2023-12-26 →
  12-29. The range-auction counts reflect the uncommitted compaction fix.
- Standard definitions are practitioner consensus, not a registered authority;
  a documented registered deviation is reported as a difference, not a bug.
- The eviction path of F1 (SW-1) and several inferred items (ZN-OB-3, RNG-PD-3,
  TR-3 share) are mechanisms without a 2023 occurrence or without a
  reproduction; each has a named test.
- No test was written or run; the tests above are specifications.

## Scratch material

Auditor reports, the verifier's verdicts and the empirical tables
(`events.parquet` 378 MB, `protection_custody.parquet`, scripts `a1`–`a9`) are
in the session scratchpad, not in the repository.
