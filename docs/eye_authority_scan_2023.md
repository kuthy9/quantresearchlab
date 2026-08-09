# 2023 full-eye natural-authority scan

Status: **annual lightweight scan and sampled transport review complete**.

This scan evaluates the causal market eye only:

`CausalMarketReader -> CausalObserver -> Group 1-5 + Displacement -> lightweight counters`

Brain, Decision, Risk, execution, MBO, PnL, future path, annual Scene Graph
projection, images and per-minute artifacts are excluded. Reducer-owned identity,
lifecycle and event ordering remain enabled. The public EventMemory query view is
not copied into each observation during the annual pass.

All production pool-source timeframes are included: `4H`, `1H`, `15m`, `5m` and
`1m`. The scan does not substitute an H1-only liquidity contract. PnL, MBO and
future data are not merely omitted from the report; they are unavailable to the
scan path.

The exact interval, seven-calendar-day New York warmup, canonical previous-session
OHLCV source, protocol files, sampling strata and stopping rules are frozen in
`configs/data_splits.json` under
`eye_group1_5_natural_authority_2023_full_year`.

## Mature range target: definition A

The existing `DealingRangeState` is evaluated as a **MatureBalanceRange**: a rare,
high-quality and persistent H1 balance/accumulation. It is not expected to describe
every ordinary H1 dealing range. This is frozen as target definition **A**; the
more common generic H1 dealing range described by alternative definition B is not
the object under review.

A systematic miss requires repeated cases from different months in which all of
the following are visible at the causal decision clock:

- stable boundaries on both sides;
- multiple internal rotations;
- no sustained one-way delivery;
- persistence across multiple completed H1 bars;
- rejection for the same maturity gate or source-identity reason.

An isolated visual disagreement does not authorize a definition change. A repeated
semantic miss permits at most one conceptual repair, tested on a different frozen
window. Sparse but accurate recognition is accepted as rare context. Unstable
recognition remains parked; thresholds are not relaxed to manufacture samples.

## FAVR and LSR decision rule

- Keep FAVR as a rare independent playbook only if MatureBalanceRange is reliable
  and the complete identity-bound FAVR chain forms naturally.
- If the range is reliable but FAVR repeatedly shares LSR's sweep, invalidation and
  target, merge it into an optional LSR `range_context`.
- If MatureBalanceRange remains systematically unreadable after the one permitted
  repair, keep FAVR parked.

The LSR core remains pool -> sweep -> reacceptance -> opposite displacement ->
micro BOS -> entry. Optional range context may explain the mature range identity,
swept boundary, midpoint/value and opposing boundary liquidity; it may enhance
thesis, target selection and delivery context, but is never an LSR hard gate.

## Completed annual pass

The annual pass ran from clean commit
`f64427f31bb61066fe69dc6cd25349cf721d8b15`; its scan identity is
`effe9ab4c0045ad41303cce64c9d5a2672ac92f760f61ad7ecac713f828a221b`.
The registered causal source SHA-256 is
`84c9ed4d1de379382bdc41e0fe02e3611373ba182cfeebe09e3832d29cbafd7b`.
The permanent artifacts are
[`eye_group1_5_natural_authority_2023_summary.json`](evidence/eye_group1_5_natural_authority_2023_summary.json)
and
[`eye_group1_5_natural_authority_2023_cases.json`](evidence/eye_group1_5_natural_authority_2023_cases.json).

The run read 358,917 source rows, emitted 358,965 completed 1m clocks including
synthetic boundary bars, and retained 353,445 observations inside the registered
2023 interval. Duplicate, out-of-order, data-gap reset, reducer exception and
fatal exception counts were all zero; four contract resets were processed.

| Timeframe | Real completed bars | Synthetic completed bars |
|---|---:|---:|
| 4H | 1,499 | 38 |
| 1H | 5,850 | 43 |
| 15m | 23,519 | 44 |
| 5m | 70,644 | 45 |
| 1m | 353,400 | 45 |

| Module | Annual funnel |
|---|---|
| Group 1-2 | 155,653 confirmed swings; 189,002 BOS identities -> 53,592 confirmed / 135,404 failed / 6 right-censored pending; 25,690 pools formed -> 25,412 swept; 182,559 liquidity items -> 170,491 consumed |
| Displacement | 15,702 started -> 2,836 ever active -> 15,498 exhausted; 203 boundary-censored, 1 right-censored and 4,024 same-bar terminal -> restart |
| FVG | 13,693 raw geometries -> 4,904 displacement-linked and 8,789 raw-only; 6,029 partial, 7,453 mitigated and 6,214 invalidated transitions |
| Order block | 70,644 attempts -> 10,707 active-displacement candidates -> 875 compatible BOS/break-member candidates -> 531 created; 494 mitigated and 36 failed, so failure coverage remains sparse |
| MatureBalanceRange | 225,102 geometry-valid pair evaluations -> 15,640 eligible -> 448 forming; 2 reached mature, 447 later broke and 1 remained right-censored |
| Manipulation | 25,719 visible eligible sources -> 25,445 crossed source IDs -> 19,465 episodes -> 8,840 reaccepted / 10,549 accepted outside / 76 deadline-censored |
| Group 5 | 5,435 qualified zones = 4,904 FVG + 531 OB -> exactly 5,435 EntryLocation and path identities; 5,325 first pullbacks, 3,710 with trigger; 6,468 complete and 18,413 interrupted paths; 0 path-order errors |
| FAVR | 0 complete natural identity-bound chains; the sole reaccepted mature-range root stopped at `opposite_displacement_zone_inside_range`, so FAVR remains parked |

The crossed-source difference is fully classified. The 25,445 identities equal
19,465 selected primary sources + 2,311 same-side secondaries + 749 coincident
secondaries + 1,641 sources blocked by a live episode resolved on the same bar +
1,132 blocked by an existing live episode + 145 ambiguous dual-side sources + 2
missing/stale sources. ATR-unready, prior-close rejection and same-clock range
invalidation were all zero. Episode outcomes also conserve exactly:
`19,465 = 8,840 + 10,549 + 76`.

`TARGETED` liquidity is a Brain/Plan overlay. Brain was disabled, so that state
is `not_evaluated`, not an Eye failure. Likewise, execution reality is explicitly
`not_evaluated`; missing spread or fillability was not injected as a primitive
anomaly.

In the registered 2023 scan, only 2 of 448 forming episodes reached mature
(0.45%), so the current definition is sparse. The blinded sampled review found
both natural mature ranges visually credible and found no repeated same-gate miss
across the selected near-mature and multi-gate cases. The correct verdict is
therefore **rare context, not a generic H1 dealing range**, rather than
systematically over-strict. The zero complete FAVR chain keeps FAVR parked. This
Eye-only pass cannot decide whether FAVR and LSR share an invalidation or target
because it did not construct those plans.

## Sampled transport and blinded image result

Thirty unique frozen clocks were replayed with EventMemory and the public Scene
Graph enabled; 29 distinct images were rendered and 28 were reviewed without
future price, PnL or later extrema. This stays inside the registered 20–40 case
range and covers every available stratum. No visual category mismatch, systematic
MatureBalanceRange miss or Group5 path-order error was found. Representative
complete Group5 paths preserved 8/8 and 2/2 typed steps through EventMemory and
Scene Graph, including exact step clocks and order.

A stricter nine-case identity audit observed and rendered all nine clocks. Seven
were exactly evaluable and all seven passed their full transport contracts. Two
dependent July cases (`c1555fbdada20e3a` and `1031ba5242f70bd8`) are explicitly
`prefix_censored`: the isolated seven-day replay reconstructed the same S/R source
pair and frozen geometry, but selected the range one hour earlier than the annual
continuous prefix. Because range, boundary and manipulation IDs bind their parent
identity and formation clock, the annual frozen IDs were absent. They are neither
counted as exact-ID passes nor interpreted as reducer semantic failures. Exact
reproduction would require an annual reducer checkpoint or replay from the earliest
relevant causal ancestor, not a threshold change.

The independent November case passed exactly: the mature range, boundary source
and manipulation identities were preserved through `swept -> reaccepted`,
EventMemory and two `SWEEPS` graph relations. This proves that the natural
mature-range manipulation transport chain is reachable; it does not create the
missing opposite displacement/entry sequence and therefore does not unpark FAVR.
No concept or threshold repair was authorized. Optional LSR `range_context`
remains non-gating, and this Eye-only evidence does not justify merging FAVR into
LSR.

## Evidence retained

The annual run retains one summary JSON and one compact blinded case index. It does
not retain observations, minute traces or future outcomes. Sampled images and one
small transmission-audit JSON remain only in the ignored development output
directory; they are not additional annual minute artifacts. Future price remained
hidden throughout the first review.

The nine frozen sampling categories are:

1. all recognized mature cases;
2. obvious mature-looking but rejected cases;
3. near-mature, single-gate rejections;
4. multiple-gate rejections;
5. forming ranges that reasonably broke;
6. source-identity or reset failures;
7. mature-range manipulations;
8. complete Group5 paths;
9. interrupted Group5 paths.

Selection is deterministic and seedless. Each candidate ID is the first 16 lower-
case hexadecimal characters of SHA-256 over
`stratum|group|primitive|entity_id|lifecycle_or_outcome|event_clock_iso8601`.
Within a category, the smallest candidate hashes are retained. Final selection uses
the implementation's frozen
`all_recognized_mature_first_smallest_sha256_then_registered_strata_round_robin_with_global_episode_uniqueness`
method: every recognized mature-range episode is selected first (up to the global
40-case cap), then the remaining categories are visited in registered order, taking
hash rank 0 from each, then rank 1, and so on. Range snapshots and range-gate
evaluations sharing one range ID count as the same episode; otherwise the same
`(group, primitive, entity_id)` is selected at most once. Selection stops at 40
cases or when candidates are exhausted. A result with fewer than 20 unique cases is
explicitly insufficient rather than silently accepted as passed.
