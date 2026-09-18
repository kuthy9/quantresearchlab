# Risk Gate and Execution FSM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Act on the LLM Brain's ACTIONABLE opportunities through a deterministic risk gate and a stateful order machine over a broker boundary, with a simulated broker for replay and an `ib_async` paper adapter for IBKR.

**Architecture:** Same per-bar loop as the Brain. `plan_from_state` turns an ACTIONABLE `BrainState` into a `TradePlan` (aliases + Eye entity ids + geometry, with a signature). `RiskGate` sizes or vetoes it against an `AccountSnapshot`. `OrderMachine` owns the one intent per episode and drives `Broker` (`SimulatedBroker` / `IBKRBroker`); every transition is a journal `trade` record. `ExecutionLedger` implements the Brain's `PositionLedger` so an engaged Brain cannot sleep.

**Tech Stack:** Python 3.12, pandas, `ib_async` 2.1 (new optional extra `ibkr`), pytest. No new runtime dependency for `none` / `sim`.

**Spec:** [../specs/2026-09-16-risk-execution-design.md](../specs/2026-09-16-risk-execution-design.md)

## Global Constraints

- `configs/model.json` `live_execution_allowed` stays `false`; `IBKRBroker` refuses to construct when it is true and refuses any account id not starting with `DU`.
- No secret, host credential or account id in code or tracked config; `execution/configs/ibkr.json` holds host / port / client_id / contract month only.
- Contract types are frozen dataclasses with `to_dict` / `from_dict` and `__post_init__` validation, in the style of `contract/brain/state.py`; `contract` layering `market -> execution -> eye -> brain -> decision -> risk` is kept (a `risk` contract may import `decision` and `brain`; an `execution` contract may not import `brain`).
- The Eye imports nothing from `risk/` or `execution/`; `eyes/tests/test_eye_module_boundary.py` must stay green.
- Every new module gets its tests first; the Brain suite (`brain/tests`) stays at 139 passed.
- Journal record kind `trade` (already reserved in `brain/core/journal.py`) is the only new record kind written.

---

## File structure

| file | responsibility |
| --- | --- |
| `contract/execution/account.py` | `AccountSnapshot`, `Position`, `OrderStatus`, `OrderRole`, `OrderState`, `Fill`, `BracketIntent`, `BrokerEvent` |
| `contract/risk/plan.py` | `ObjectRef`, `TradePlan` (+ `signature`), `RiskVerdict`; `VetoCode` extended in `assessment.py` |
| `risk/core/gate.py`, `risk/configs/risk.json`, `risk/tests/test_gate.py`, `risk/docs/README.md` | `RiskConfig.from_json`, `RiskGate.assess` |
| `execution/core/broker.py` | `Broker` protocol, `SimulatedBroker` |
| `execution/core/ibkr_broker.py` | `IBKRBroker` over `ib_async` (`IBKRConfig.from_json`) |
| `execution/core/plan.py` | `plan_from_state(state, context, tick)` |
| `execution/core/order_fsm.py` | `MachineState`, `OrderMachine`, `ExecutionLedger` |
| `execution/scripts/ibkr_paper_check.py` | read-only paper connectivity check |
| `execution/configs/ibkr.json` | connection parameters |
| `brain/core/position_ledger.py` | `has_working_order()` on the protocol |
| `brain/core/runtime.py` | engaged = open position or working order |
| `brain/scripts/_run_identity.py`, `run_llm_brain.py`, `replay_journal.py` | the `Bar` in the drive callback, `--broker`, replay of `sim` runs |
| `execution/docs/README.md`, `AGENTS.md`, `shares/docs/current_implementation_status.md` | docs |

---

### Task 1: account and order contracts

**Files:** Create `contract/execution/account.py`; modify `contract/execution/__init__.py`; test `execution/tests/test_account_contract.py`.

**Produces:** the types in spec §2 with `to_dict` / `from_dict`; `OrderState.remaining` property; `AccountSnapshot.open_entry_orders()` helper; `Position.quantity` signed (LONG > 0).

- [ ] Test: round trip of every type through JSON; `OrderState` with `filled_quantity > quantity` raises; `AccountSnapshot.open_entry_orders()` returns only ENTRY roles in SUBMITTED / WORKING / PARTIAL.
- [ ] Implement; run `pytest execution/tests/test_account_contract.py`.

### Task 2: plan and verdict contracts

**Files:** Create `contract/risk/plan.py`; modify `contract/risk/assessment.py` (`VetoCode` += `EXPOSURE`, `WORKING_ORDER`, `POSITION_SIZE`), `contract/risk/__init__.py`; test `risk/tests/test_plan_contract.py`.

**Produces:** `ObjectRef(alias, entity_id, kind, timeframe)`, `TradePlan(episode_id, revision, known_at, direction, entry, invalidation, target, geometry, close)` with `signature` (sha256 of direction + the three entity ids, first 16 hex), `RiskVerdict(passed, vetoes, reasons, quantity=0, limit_price=None, stop_price=None, target_price=None, risk_amount=None, reward_risk=None, equity=None)`.

- [ ] Test: same objects → same signature; a different target → different signature; `RiskVerdict(passed=True)` without quantity raises; round trips.
- [ ] Implement; run.

### Task 3: risk gate

**Files:** Create `risk/__init__.py`, `risk/core/__init__.py`, `risk/core/gate.py`, `risk/configs/risk.json`, `risk/tests/__init__.py`, `risk/tests/test_gate.py`, `risk/docs/README.md`; modify `pyproject.toml` (`testpaths` += `risk/tests`, packages += `risk*`).

**Produces:** `RiskConfig.from_json(path)` (fields per spec §3 plus `sha256`), `RiskGate(config).assess(plan, account, *, asof) -> RiskVerdict`.

- [ ] Tests: sizing example (equity 100 000, risk 0.5 %, entry 16387.5, stop 16411.5 → 24 points × 20 = 480 per contract → quantity 1; equity 250 000 → 2); every veto in spec §3 individually; `max_quantity` cap; tick rounding of limit / stop / target; a WORKING entry order → `WORKING_ORDER`; an open position → `EXPOSURE`.
- [ ] Implement; run `pytest risk/tests`.

### Task 4: simulated broker

**Files:** Create `execution/core/broker.py`; test `execution/tests/test_simulated_broker.py`.

**Produces:** `Broker` protocol (spec §4); `SimulatedBroker(*, equity, tick_size, point_value, account_id="SIM")` — `submit_bracket` returns the ENTRY `OrderState` (SUBMITTED) and registers STOP / TARGET children; `poll(asof, bar)` advances: SUBMITTED → WORKING on the first bar after submission, fills per the rules, PnL into equity; `snapshot(asof)`.

- [ ] Tests: no fill on the submission bar; LONG entry fills when a later bar's low ≤ limit; SHORT mirrored; stop-first when a bar touches both; target fill closes and cancels the stop; equity moves by `(exit − entry) × qty × point_value` signed; `cancel` of a WORKING entry → CANCELLED and the children are dropped; the same bars twice give identical event lists.
- [ ] Implement; run.

### Task 5: plan from state

**Files:** Create `execution/core/plan.py`; test `execution/tests/test_plan_from_state.py`.

**Produces:** `plan_from_state(state: BrainState, context: EyeContext, *, tick: float) -> TradePlan | None` — None unless `state.opportunity.state is ACTIONABLE`; geometry via `brain.core.opportunity_geometry.resolve_geometry(state.opportunity, context.geometries(), close=context.close, tick=tick)`; entity ids from `state.object_registry`; `GeometryError` → None (the reducer already vetted; a vanished object is a None plan).

- [ ] Tests on the synthetic Eye context (`shares.tests.helpers.session_bars`): NONE → None; DEVELOPING → None; ACTIONABLE with visible objects → a plan whose signature is stable across two contexts on the same bar; an object missing from the context → None.
- [ ] Implement; run.

### Task 6: order machine and execution ledger

**Files:** Create `execution/core/order_fsm.py`; modify `brain/core/position_ledger.py` (protocol + `InMemoryPositionLedger.has_working_order() -> False`), `brain/core/runtime.py` (`has_open_position=self._ledger.has_open_position() or self._ledger.has_working_order()` in both `ReduceContext`s), `brain/core/main_brain.py` (same expression); test `execution/tests/test_order_fsm.py`, `brain/tests/test_runtime.py` (a ledger with a working order blocks sleep and the idle rule).

**Produces:** `MachineState` enum (IDLE, WORKING, PARTIAL, IN_POSITION); `ExecutionConfig(order_ttl_bars=15)`; `OrderMachine(broker, gate, *, config, journal: BrainJournal | None, symbol)` with `on_bar(asof, bar, plan, *, episode_id, visible: Callable[[str], bool]) -> tuple[str, ...]` (the transition kinds this bar), `state`, `ledger: ExecutionLedger`; `ExecutionLedger` implements `PositionLedger` (+ `has_working_order`).

- [ ] Tests with `SimulatedBroker` and hand-made plans (helper building a `TradePlan` from prices): submit once per signature; re-signal no-op; TTL cancel at 15 bars; signature change cancels then submits the new one next bar; entry invisible cancels; fill → IN_POSITION and `has_open_position`; stop fill → IDLE and `position_closed`; veto recorded once per (signature, vetoes); journal chain verifies with the `trade` records; a `plan=None` while IN_POSITION does nothing.
- [ ] Implement; run `pytest execution/tests brain/tests`.

### Task 7: IBKR adapter

**Files:** Create `execution/core/ibkr_broker.py`, `execution/configs/ibkr.json`, `execution/scripts/ibkr_paper_check.py`; modify `pyproject.toml` (extra `ibkr = ["ib_async>=2.1"]`); test `execution/tests/test_ibkr_broker.py`.

**Produces:** `IBKRConfig.from_json` (host, port, client_id, contract: symbol / exchange / currency / last_trade_month, `paper_only=True`), `IBKRBroker(ib, config, *, live_execution_allowed: bool)` where `ib` is an `ib_async.IB`-shaped object (injected for tests); `IBKRBroker.connect(config, live_execution_allowed)` classmethod does the real import and connect.

- [ ] Tests with `FakeIB` (managedAccounts, accountSummary, positions, openTrades, fills, bracketOrder, placeOrder, cancelOrder, waitOnUpdate): construction refused for a `U…` account and when `live_execution_allowed` is true; `snapshot` maps NetLiquidation / AvailableFunds / BuyingPower and statuses; `submit_bracket` places three orders with the right actions, prices and `orderRef`; `poll` diffs statuses into events; `cancel` calls `cancelOrder` on the entry.
- [ ] Implement; run.

### Task 8: wiring, replay and docs

**Files:** modify `brain/scripts/_run_identity.py` (`on_observation(observation, emitting, bar)`), `brain/scripts/run_llm_brain.py` (`--broker`, `--risk-config`, machine per bar, `run.json` fields), `brain/scripts/replay_journal.py` (drive the machine for `sim` runs; `trade` records compared by kind sequence), `execution/docs/README.md`, `AGENTS.md`, `shares/docs/current_implementation_status.md`, `brain/docs/README.md`; test `execution/tests/test_stack_e2e.py`.

- [ ] Test: synthetic Eye + `ScriptedClient` whose first reply names an ACTIONABLE LONG at the nearest zone (as `test_opportunity_naming_a_visible_object_survives_reduce` builds it) + `SimulatedBroker` → the journal has `trade` records submitted → filled → position_closed (or cancelled after TTL), the Brain's sleep was refused with `open_position` while engaged, and `replay_run` over the same observations reproduces the states.
- [ ] Implement; run `pytest brain/tests execution/tests risk/tests`; echo run with `--broker sim` on 2022-01-03; update docs.
