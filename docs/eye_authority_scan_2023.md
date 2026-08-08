# 2023 full-eye natural-authority scan

Status: **registered before replay; outcome blind**.

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

## Evidence retained

The annual run retains one summary JSON and one compact blinded case index. It does
not retain observations, minute traces or future outcomes. After the annual pass,
20-40 predeclared stratified clocks are replayed with EventMemory, Scene Graph and
images enabled solely to check information transport. Future price is hidden in the
first review.

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
