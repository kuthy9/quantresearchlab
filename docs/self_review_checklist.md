# Mandatory pre-test self-review

Use this short gate before synthetic tests, bounded replay, calibration or
execution validation. Check the current change only; do not regenerate release
governance artifacts during ordinary development.

## Causal logic

- [ ] One newly completed 1m bar is processed exactly once.
- [ ] Higher-timeframe bars update only when complete.
- [ ] No future path, later extreme, PnL or action label enters observation.
- [ ] Event identity, formation/confirmation/invalidation clocks, lifecycle and
  event order survive the update.
- [ ] Scene Graph nodes/edges, focus and unknown/ambiguity match the observation
  actually passed to the Brain.
- [ ] `Belief_t` starts from `Belief_t-1`; unchanged evidence is not assimilated
  repeatedly.
- [ ] DFP/LSR/FAVR typed sequence, episode, terminal and rearm rules are intact.
- [ ] FAVR is parked unless a natural mature range/value has authority.
- [ ] Planned entry can differ from current close; first pullback binds one
  frozen zone and reacceptance has departure/reclaim/hold/failure semantics.

## Decision, risk and execution

- [ ] Thesis, sequence, location, readiness, delivery and uncertainty remain
  distinct inputs to action utility.
- [ ] An unclear utility advantage yields `abstain`.
- [ ] Entry freezes structural invalidation, draw/targets, deadline and risk;
  later extrema do not rewrite them.
- [ ] Spread, cost, stale/anomalous data, fillability, deadline and target/stop
  provenance retain hard-veto authority.
- [ ] Approved entry is first eligible at the next tradable clock.
- [ ] Attempts conserve state (`filled | expired | cancelled`), and each fill
  creates one position with terminal feedback on the next minute.
- [ ] Same-bar stop/target ambiguity remains adverse-first.

## Data and replay

- [ ] Source and requested window match `configs/data_splits.json`.
- [ ] Previous-session contract selection and raw/processed manifests remain
  bound.
- [ ] MBO partition/execution manifests match their exact registered SHA-256;
  sealed MBO is not read during development.
- [ ] Missing MBO remains missing and is not replaced with constant execution
  reality.
- [ ] Checkpoint, shards, progress, resume and portfolio before-bar /
  after-decision ordering remain intact.
- [ ] Daily replay writes light decisions/summary only; full trace, images and
  future reveal are enabled only for a fixed stratified sample.

## Test scope

- [ ] Run syntax plus the unified synthetic/boundary/causal suite first.
- [ ] For a primitive change, run one bounded real OHLCV replay and one small
  blind review; allow at most one concept-level repair.
- [ ] Do not use PnL to repair primitive semantics or inspect the same OOF/
  holdout window repeatedly.
- [ ] Run Brain calibration only after primitives are frozen or parked.
- [ ] Run rolling OOF, MBO stability and sealed holdout once, in that order,
  only after the vertical chain is stable.
