# v1.3 versus v1.2: two complete months of NQ replay

Status date: 2026-08-31
Atomic identity at run time: `smc_semantics_v1.3` ·
`7f4e790bfcd43e9a7077e738cb33be4e2f6f1a4731c219a947c7807d23b3aa7f`
Foundation identity: `smc_semantic_foundation_v2.1` ·
`69428dbfd2a9b2aa19f0254391fca2da17aedb8d0206829572e69c0cc212a715`

The atomic identity above is the one this replay ran under and is kept as the
run's own binding. The runtime has since moved twice — `balance_range_v1.2` and
the 2026-09-06 configuration cleanup — so it is deliberately not the current
identity; the current value lives in
[the semantic specification](../smc_semantic_specification_v1.3.md).

The v1.2 / foundation-v2.0 baselines this compares against, and the machine
ledger of the comparison itself, were retired on 2026-09-06. The tables below
are self-contained: every changed kind carries both its v1.2 and its v1.3
count, and an unchanged kind is by definition equal to the v1.3 count shown.

## What was compared

Both months were replayed bar for bar against the same OHLCV source and the
same `configs/model.json` composition, once under v1.2 and once under v1.3:
27,360 completed M1 bars in February and 31,740 in March, identical counts on
both sides.

## The v1.2 surface did not move

Every registered v1.2 event kind returned **exactly** its v1.2 count in both
months — 40 unchanged kinds in February and 41 in March. Swings, structural
legs, liquidity, sweeps, acceptances, displacements, FVG lifecycle, dealing
ranges and origin-zone terminals are all byte-for-byte the same population.
v1.3 is an identity and naming change, not a detector change, and the replay
says so.

## What changed, and only what changed

| kind | 2022-02 | 2022-03 | why |
|---|---:|---:|---|
| `bos_state` | 15,748 → 0 | 17,676 → 0 | removed: an unregistered transport carrying only `pending`, already stated by the absence of a terminal on `RAW_BOUNDARY_BREAK` |
| `origin_zone_created` | 43 → 0 | 57 → 0 | retired in favour of the split below |
| `base_origin_core_created` | 0 → 43 | 0 → 57 | the frozen candle geometry, on its own |
| `qualified_origin_zone_created` | 0 → 43 | 0 → 57 | that core plus the active displacement and the qualified BOS |
| `fvg_first_retest` | 0 → 1,031 | 0 → 1,234 | the first re-entry into each gap |
| `delivery_phase_entered` | 0 → 7,953 | 0 → 9,004 | one per phase occupancy per timeframe |
| `delivery_phase_exited` | 0 → 7,948 | 0 → 8,994 | its successor-naming close |
| `delivery_phase_updated` | 0 → 3,420 | 0 → 3,731 | only when a registered phase input moved |
| `balance_range_observed` | 0 → 1 | 0 → 1 | the two-sided test standard, reached once per month |

The origin-zone split is exactly 1:1:1 — 43 old creations became 43 cores and
43 qualifications in February, 57 and 57 in March. Nothing was invented and
nothing was lost.

## What the new facts measured

**Almost every gap gets re-entered.** 1,031 of 1,044 February gaps (98.8%) and
1,234 of 1,275 March gaps (96.8%) produced a first retest. The interesting
population is therefore not "which gaps are revisited" — nearly all are — but
*how* they are revisited, which is what `fill_depth_at_entry`, `age_bars` and
`approach_speed_atr` now freeze at the instant of entry. Under v1.2 that
context was unrecoverable: the fill fraction only ratchets, so by the time a
gap was read its entry conditions had already been overwritten.

**The phase lifecycle is sparse, not a per-bar stream.** Across five timeframes
and 27,360 bars, February produced 7,953 entries and 3,420 updates: about 0.42
events per bar in total, against the ~137,000 a naive per-bar-per-timeframe
projection would have emitted. Entries and exits differ by 5 in February and 10
in March — exactly the timeframes still occupying a phase when the replay
window ended, which is the expected right-censoring.

> **Superseded 2026-09-05 for the balance rows only.** The counts below were
> produced when balance evidence was the source zone's structural touch count.
> `balance_range_v1.2` replaced that with price interaction against the frozen
> boundary; February's single observation was a second confirmed swing inside a
> source zone, which is not a price test, and it is gone. Both months now report
> zero. Everything else in this document is unaffected — a kind-by-kind replay
> comparison found exactly one changed event across both months. The machine
> ledger that recorded it was retired on 2026-09-06.

**Balance is genuinely rare, and that is the point of splitting it.** February
registered 40 structural ranges and March 41; in each month exactly **one**
reached two confirmed touches on each frozen boundary, and **none** matured.
Under v1.2 those 80 remaining ranges were simply "created and never activated"
— an absence with nothing positive to say. Under v1.3 all 81 keep a usable
location claim and only one carries a balance claim. The v1.2 falsification
condition recorded for this concept ("maturity is unreachable in production
data") is confirmed a second time, now with the location dimension no longer
paying for it.

## What this does not establish

These are event-population counts from a two-month development window. Nothing
here validates predictive value, tests a trading rule, or opens any OOS window.
The `balance_range` maturity gate remains unreached; whether its thresholds are
wrong or the concept is rare is not answered by two months. `FVG_EXPIRED` and
`DEALING_RANGE_EXTENDED` remain reserved and unemitted in both months.
