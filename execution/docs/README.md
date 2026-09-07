# Execution

Execution owns order and position reality. It never submits: `trade_intent.py`
constructs a deterministic, immutable, never-submit Trade Intent, and
`shares/core/engine.py` rejects a non-zero `TradeIntent` before Decision/Risk
while the action authority stays `legacy_decision_risk_compat`.

## Core modules — `execution/core/`

| module | owns |
| --- | --- |
| `execution.py` | `ExecutionRealityInput`, the cost/fillability score and the causal top-of-book adapter |
| `execution_fsm.py` | immutable shadow execution facts and the explicit order/position FSM |
| `trade_intent.py` | deterministic, immutable, never-submit Trade Intent construction |
| `simulation.py` | the causal sequential execution adapter for historical and shadow replay |
| `mbo.py` | memory-bounded MBO book reconstruction and minute execution reality |

`ContinuousSMCEngine._score_execution` derives the execution score; the Eye only
transports the result. `shares/core/model.py` owns the inert not-evaluated value
beside `ExecutionObservation`.

## Protocols — `execution/configs/`

No protocol file of its own (the directory holds only its README): its cost, spread and
fillability gates live in the `risk` block of `configs/model.json`
(`maximum_spread_ticks`, `maximum_cost_R`, `minimum_fillability`,
`same_bar_resolution`), and the MBO data windows live in
`configs/data_splits.json`. Both are sealed by the atomic semantic identity and
stay at the repository root.

## Tests — `execution/tests/`

`test_execution_fsm.py` and `test_sequential_replay.py`. The MBO protocol cases
live in `shares/tests/test_v2_protocols.py` because they span validation, the
playbook registry and the data authority as well.

## Authority documents

[shares/docs/architecture.md](../../shares/docs/architecture.md) for the runtime
boundaries and
[shares/docs/current_implementation_status.md](../../shares/docs/current_implementation_status.md)
for implementation-versus-plan status.

## Scripts — `execution/scripts/`

`materialize_mbo_execution.py` streams level-3 MBO into one causal
execution-reality row per OHLCV minute. Throwaway probes belong here too.
