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
| `mbo.py` | memory-bounded MBO book reconstruction and minute execution reality |

`simulation.py`, the sequential execution adapter bound to the retired typed
vertical (`EngineSnapshot`, `TradePlan`, the old decision / risk pair), was
deleted on 2026-09-16 with `shares/core/engine.py`. The Execution that will
act on the LLM Brain's `OpportunityGeometry` (`contract/decision/opportunity.py`)
is a later phase; `brain/core/position_ledger.py` is the boundary it attaches to.

The retired engine derived the execution score; the Eye only transports the result. `contract/execution/reality.py` owns `ExecutionObservation`,
its inert not-evaluated value, `PositionSnapshot` and `AccountState`.

## Protocols — `execution/configs/`

No protocol file of its own (the directory holds only its README). The
`risk` block of `configs/model.json` that held the cost, spread and
fillability gates was removed on 2026-09-16 with the old risk engine; the
Risk phase will define its own protocol. The MBO data windows live in
`configs/data_splits.json`, sealed by the atomic semantic identity at the
repository root.

## Tests — `execution/tests/`

None at present: `test_sequential_replay.py` went with `simulation.py` on
2026-09-16, `test_execution_fsm.py` with the FSM on 2026-09-07, and
`shares/tests/test_v2_protocols.py` (the MBO protocol cases, bound to the
retired validation protocol) on 2026-09-16. `execution/core/execution.py` is
exercised through the Eye's observation tests.

## Authority documents

[shares/docs/architecture.md](../../shares/docs/architecture.md) for the runtime
boundaries and
[shares/docs/current_implementation_status.md](../../shares/docs/current_implementation_status.md)
for implementation-versus-plan status.

## Scripts — `execution/scripts/`

`materialize_mbo_execution.py` streams level-3 MBO into one causal
execution-reality row per OHLCV minute. Throwaway probes belong here too.
