# Does Structural Range width explain the unreachable Balance claim?

**Date:** 2026-09-05
**Window:** 2022-02-01 → 2022-04-01, replayed one month at a time
**Semantics:** `smc_semantics_v1.3`, balance sub-protocol `balance_range_v1.2`
**Changes made to run this study:** none. Group 3's touch counter and the
balance interaction-generation logic are frozen exactly as committed; the
script reads published state and restores the method it hooks.

Raw data: `structural_range_width_strata_2022_02.json`,
`structural_range_width_strata_2022_03.json`.

## The question, and what would answer it either way

Balance is currently a maturity state *of* a Structural Range, so every range
that does not balance is recorded as a range that failed. Two months of zero
observations are consistent with two very different readings, and the
discriminating evidence was declared before the run:

* If `P(bilateral)` climbs steeply as ranges get narrower, **width is the
  mechanism** — Group 4 selects intervals too wide for a two-sided test to
  complete, and the fix belongs in candidate selection.
* If `P(bilateral)` is near zero in **every** stratum including the narrowest,
  width is not the explanation, and the parent → maturity coupling is measuring
  a property these objects do not have.

Ranges are grouped by `width_atr_at_formation`, which is frozen when the range
is created, so the stratification carries no look-ahead. Each range contributes
its **peak** evidence over its whole life — the question is whether it ever got
there, not where it happened to end.

## Result

52 Structural Ranges (25 in February, 27 in March).

| stratum | n | bilateral ≥2/2 | bilateral ≥1/1 | lower ≥2 | upper ≥2 | median lifetime (H1 bars) | Wilson 95% on bilateral |
|---|---:|---:|---:|---:|---:|---:|---|
| ≤2 ATR | 4 | **0/4** | 2/4 | 0/4 | 0/4 | 2.0 | (0, 0.490) |
| 2–3 ATR | 6 | **0/6** | 2/6 | 1/6 | 1/6 | 7.0 | (0, 0.390) |
| 3–4 ATR | 16 | **0/16** | 4/16 | 0/16 | 2/16 | 6.5 | (0, 0.194) |
| 4–5 ATR | 8 | **0/8** | 0/8 | 1/8 | 1/8 | 11.0 | (0, 0.324) |
| >5 ATR | 18 | **0/18** | 1/18 | 1/18 | 4/18 | 12.0 | (0, 0.176) |
| **pooled** | **52** | **0/52** | 9/52 | 3/52 | 8/52 | — | **(0, 0.069)** |

## What this says

**1. The width hypothesis is not supported.** `P(bilateral)` is zero in every
stratum, and there is no gradient. Where a single-sided rate moves at all it
moves the *wrong* way for the hypothesis: `upper ≥2` rises with width
(0/4, 1/6, 2/16, 1/8, 4/18) rather than falling. Narrow ranges do not do
better.

**2. Median lifetime rises monotonically with width** — 2.0, 7.0, 6.5, 11.0,
12.0 H1 bars. This is the opposite of what the width story needs. The narrowest
ranges are not patient intervals waiting to be tested twice; they are ranges
that form and die within two bars. Their `0/4` says almost nothing about
balance and almost everything about lifetime.

**3. Even the far weaker standard mostly fails.** Asking only that price touch
*each* side once, ever — `bilateral ≥1/1` — is met by 9 of 52 ranges, 17%,
Wilson (0.094, 0.297). Five ranges recorded no boundary interaction at all.
The two-sided test is not failing at its second touch; most of these intervals
never see both sides.

**4. The mechanism is visible in the terminal statistics.** 50 of the 52 ranges
ended with `close_beyond_frozen_range`, and only 2 were still forming when the
window closed. Of the 71 interaction generations recorded across all ranges,
**40 were `close_outside`** — the interaction that destroys the range. A
Structural Range's typical, and often only, meaningful contact with its own
boundary is the one that ends it.

Price **transits** these intervals rather than oscillating inside them, and it
does so at every width.

> **Read with `balance_candidate_hypotheses_2026-09-05.md`.** That study added
> the missing control: an arbitrary rolling H1 window shows a bilateral revisit
> only 4.06% of the time. At that base rate, seeing 0 in 52 is unremarkable
> (P = 0.21), so the zeroes below are not evidence that Group 4's ranges are
> unusual. Pooled to 84 ranges the upper bound is 4.37%, still above the base
> rate: the Structural Range is not worse than random at finding balance, only
> no better.

## What follows, and what does not

This is consistent with Structural Range and Balance Range being different
market objects. A Structural Range as Group 4 currently selects one is a
*location* that price passes through; two-sided rejection is a different
phenomenon that these objects essentially never exhibit. Making balance a
maturity state of the range means ~96% of ranges are recorded as having failed
a test they were never candidates for.

The data supports dropping the forced parent → maturity relation. It does
**not** say what should replace it, and it does not identify what a genuine
Balance Range candidate looks like — nothing here searched for intervals that
*do* show two-sided rejection, because the only intervals examined were the
ones Group 4 already admits.

**Limits.** 52 ranges over two months of one instrument. The two narrow strata
carry n=4 and n=6, so their individual zeroes are weak; the pooled bound —
true bilateral rate at most about 7% — and the absence of a gradient across the
well-populated 3–4 ATR (n=16) and >5 ATR (n=18) strata are the load-bearing
results. Nothing here validates predictive value or opens an OOS window.
