# Execution

Execution owns order and position reality. It never submits.

**The order FSM and Trade Intent were retired on 2026-09-07.**
`execution_fsm.py` and `trade_intent.py` are gone: `trade_intent.py`'s only
entry point took a `SignalAssessment` from the retired `brain.core.signal_policy`,
and `execution_fsm.py` was built entirely on the `TradeIntent` it produced. What
remains is execution reality, MBO reconstruction and the sequential simulator —
the parts that never needed a Brain intent.

## Core modules — `execution/core/`

| module | owns |
| --- | --- |
| `execution.py` | `ExecutionRealityInput`, the cost/fillability score and the causal top-of-book adapter |
| `simulation.py` | the causal sequential execution adapter for historical and shadow replay |
| `mbo.py` | memory-bounded MBO book reconstruction and minute execution reality |

`simulation.py` imports `shares/core/engine.py`, which cannot currently be
imported because the Brain it orchestrates was retired. The simulator itself is
unchanged and will work again as soon as that orchestration is rebound.

`ContinuousSMCEngine._score_execution` derives the execution score; the Eye only
transports the result. `contract/execution/reality.py` owns `ExecutionObservation`,
its inert not-evaluated value, `PositionSnapshot` and `AccountState`.

## Protocols — `execution/configs/`

No protocol file of its own (the directory holds only its README): its cost, spread and
fillability gates live in the `risk` block of `configs/model.json`
(`maximum_spread_ticks`, `maximum_cost_R`, `minimum_fillability`,
`same_bar_resolution`), and the MBO data windows live in
`configs/data_splits.json`. Both are sealed by the atomic semantic identity and
stay at the repository root.

## Tests — `execution/tests/`

`test_sequential_replay.py` only. It cannot be
collected while `shares/core/engine.py` is broken. `test_execution_fsm.py` was
removed with the FSM. The MBO protocol cases live in
`shares/tests/test_v2_protocols.py` because they span validation and the data
authority as well.

## Authority documents

[shares/docs/architecture.md](../../shares/docs/architecture.md) for the runtime
boundaries and
[shares/docs/current_implementation_status.md](../../shares/docs/current_implementation_status.md)
for implementation-versus-plan status.

## Scripts — `execution/scripts/`

`materialize_mbo_execution.py` streams level-3 MBO into one causal
execution-reality row per OHLCV minute. Throwaway probes belong here too.
