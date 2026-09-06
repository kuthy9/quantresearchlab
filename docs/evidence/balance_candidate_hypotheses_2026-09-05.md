# Where should a Balance Candidate come from?

**Date:** 2026-09-05
**Data:** 2022-01, 2022-02, 2022-03 (Structural Range arm, 84 ranges);
2022-01 fit / 2022-02 test (independent scan arm, out of sample)
**Changes made to run this:** none. No semantics, no detector, no Group-3 rule.

## The two hypotheses

**H1 — the Structural Range is already the right candidate.** A range forms,
width and overlap develop inside it, bilateral interactions accumulate, balance
is observed.

**H2 — balance has to be found on its own terms.** A narrow, overlapping
episode appears anywhere, is nominated on shape alone, and is then confirmed
bilaterally.

The number both have to beat is the base rate of an arbitrary rolling H1
window.

## Result

| arm | P(balance) | 95% CI | unit |
|---|---:|---|---|
| Baseline — any rolling window | 4.06% | (3.68, 4.48) | 9,237 windows |
| **H2 — independent shape scan** | **8.36%** | **(6.78, 10.27)** | 969 nominated windows |
| **H1 — Structural Range** | **0.00%** | **(0, 4.37)** | 84 ranges |

* **H2 vs H1: separated.** The intervals do not overlap — 6.78% against 4.37%.
  An independent scan is a materially better candidate generator than the
  Structural Range.
* **H2 vs baseline: separated.** 2.06× lift, intervals disjoint. The shape
  features carry real information.
* **H1 vs baseline: not separated.** The upper bound, 4.37%, still sits above
  the 4.06% base rate. The Structural Range cannot be shown to be worse than
  picking a window at random — only that it is no better.

Zero of 84 ranges reached two interactions on both sides. 11 of 84 reached even
one per side, and 8 of 84 recorded no boundary interaction at all.

## The episode-level metrics carry no evidence, in either direction

Precision, recall and lead time looked excellent for H2 — 78.4% precision, 100%
recall, median 10 bars of lead. They are artefacts. Against a random-nomination
control at the same horizon:

| horizon (H1 bars) | H2 precision | chance precision | lift |
|---:|---:|---:|---:|
| 2 | 0.098 | 0.112 | 0.88 |
| 4 | 0.157 | 0.186 | 0.84 |
| 6 | 0.255 | 0.261 | 0.98 |
| 8 | 0.294 | 0.336 | 0.88 |
| 12 | 0.510 | 0.480 | 1.06 |
| 24 | 0.784 | 0.789 | 0.99 |

At every horizon H2's precision equals chance. The 17 ground-truth episodes
have a median span of 19 H1 bars in a 456-bar month, so most of the month sits
inside some episode's lead window and a hit is close to automatic. **Lead time
is meaningless for the same reason** — a generator that hits at the chance rate
has not anticipated anything. The episode definition is too permissive (21
window lengths ending on one bar merge into a single episode) and would have to
be tightened before any of these three metrics can be used.

Only the window-level rate above survives this control, and it is what the
verdict rests on.

## Verdict, and what qualifies it

**The data supports decoupling, but not cleanly.**

The supporting result is that H1 is separated from H2 and indistinguishable
from random. A candidate generator that performs no better than an arbitrary
window is not adding information, so binding balance to it as a maturity state
buys nothing an independent scan would not buy better.

**The qualification is a definitional asymmetry between the arms, and it favours
H2.** A rolling window's extremes are its own maximum and minimum, so each is
touched at least once by construction. A Structural Range's boundaries are
frozen from source zones and price may never reach them — 8 of 84 ranges never
touched either. Part of H2's advantage is therefore built into how the two
units are defined rather than earned.

Controlling for it exhausts the sample: restricting to ranges where price
reached both boundaries at least once leaves 11 ranges, of which 0 reached
2/2 — CI (0, 25.9%), which excludes nothing. **A clean test needs either
matched definitions (nominate windows, then test against frozen levels) or
several more months.**

## What is not claimed

Not that balance is unreachable — the independent scan finds it at 8.4%, and
the market base rate is 4.06%. Not that the Structural Range is defective — it
was never shown to be worse than random. Not that any detector should be built
yet: the shape features that separate the classes are `width_atr` (inverted)
and `overlap_ratio`, while `directional_efficiency` and `path_efficiency` carry
AUC ≈ 0.5 at windows of 12 bars or more and would be noise in a rule.

Three months of one instrument. Nothing here validates predictive value or
opens an out-of-sample window.
