# Phase review fixes and the simulated executor — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the confirmed defects of the 2026-09-16 sleep / risk / execution phase and replace the minimal `SimulatedBroker` with a local simulated executor that keeps the order interface and the FSM, replacing only IBKR's account and matching.

**Architecture:** The `Broker` protocol (`snapshot`, `submit_bracket`, `cancel`, `poll`) stays the one seam. `OrderMachine` gains three small rules (reject blocks the signature; the block is per episode; a lost exit leg is journaled). `IBKRBroker` fails loudly when disconnected and the runner refuses to start it over a non-flat account or a stale tape. `SimulatedExecutor` (new module) fills against completed 1m bars with fixed OHLCV rules, supports partial fills, margin rejects and cancel-after-partial, and keeps a `VirtualAccount` (cash, positions, pending / filled / cancelled / rejected orders) starting at 100 000 USD.

**Tech Stack:** Python 3.12, pandas, pytest; `ib_async` only behind `IBKRBroker.connect`.

**Spec:** [../specs/2026-09-16-risk-execution-design.md](../specs/2026-09-16-risk-execution-design.md) §4–5, amended by this plan's findings (recorded in `execution/docs/evidence/2026-09-17_phase_review.md`).

## Global Constraints

- `configs/model.json` keeps `live_execution_allowed: false`; `IBKRBroker` keeps both guards (paper `DU…` accounts only, refuses the live flag).
- No wall clock in `brain/core`, `execution/core`, `risk/core`; the simulated executor is driven by bar timestamps only so a `sim` journal replays.
- The FSM semantics (one intent per episode, one bracket per intent, cancel + resubmit as the only "replace") are identical for the simulated executor and IBKR; the simulator never gets a code path the FSM does not drive.
- No new credential, account id or absolute path anywhere; simulator parameters live in `execution/configs/simulated_executor.json`.
- Test command: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest <paths> -p no:cacheprovider`.

---

## Findings this plan acts on

| id | severity | where | defect | fix |
| --- | --- | --- | --- | --- |
| H1 | high | `brain/scripts/run_llm_brain.py` | `--broker ibkr` prices orders from the historical tape against the live paper market (a 2022 SELL limit is marketable today) | refuse `ibkr` unless the window end is within `IBKR_MAX_TAPE_AGE` (1 day) of now |
| H2 | high | `run_llm_brain.py`, `ibkr_broker.py` | no restart recovery: the machine starts IDLE beside an existing position / bracket, the Brain's ledger reads not engaged | refuse to start `ibkr` over a non-flat account (`require_flat`); recovery itself stays unimplemented and documented |
| M1 | medium | `order_fsm._apply` | entry REJECTED → IDLE, signature not blocked → resubmitted every bar | block the signature on `rejected` as on `expired` |
| M2 | medium | `order_fsm._apply` | a cancelled / rejected stop or target while IN_POSITION is ignored — naked position, no journal trace | journal `exit_leg_lost` once per leg |
| M3 | medium | `order_fsm` | `_blocked_signature` survives the episode: a new episode's identical plan is never traded | block per `(episode_id, signature)` |
| M4 | medium | `ibkr_broker.poll/snapshot` | after a socket drop ib_async's cache is read silently | raise `IBKRDisconnected` when `ib.isConnected()` is false |
| M6 | medium | `broker.SimulatedBroker` | a cancelled PARTIAL leaves its exits dead; no partial / reject model | superseded by `SimulatedExecutor` |

Reported, not changed: M5 (a vetoed ACTIONABLE keeps the episode awake; the LLM never learns of the veto), M7 (the gate ignores margin), L1 (`STALE_DATA` inert), L2 (incident bars advance `_last_llm_known_at`), L3 (IBKR journals one fill per order per poll), L4 (one-bar lag between machine and Brain, conservative), L5 (no in-place replace; only a range's value price can move under the same signature).

---

## File structure

| file | responsibility |
| --- | --- |
| `execution/core/order_fsm.py` | M1, M2, M3 |
| `execution/core/ibkr_broker.py` | M4 (`IBKRDisconnected`), `require_flat(snapshot, symbol)` |
| `brain/scripts/run_llm_brain.py` | H1 (`tape_is_current`), H2 wiring, `--sim-config`, `--sim-equity` override |
| `execution/core/simulated_executor.py` | `SimulatorConfig`, `VirtualAccount`, `SimulatedExecutor` |
| `execution/configs/simulated_executor.json` | initial equity 100 000, margin per contract, max fill per bar |
| `execution/core/broker.py` | the `Broker` protocol only (`SimulatedBroker` removed) |
| `brain/scripts/replay_journal.py` | build the executor from the config + `run.json` |
| `execution/tests/test_order_fsm.py`, `test_ibkr_broker.py`, `test_simulated_executor.py` (replaces `test_simulated_broker.py`), `test_stack_e2e.py`, `brain/tests/test_run_guards.py` | tests |
| `execution/docs/README.md`, `execution/docs/specs/…`, `AGENTS.md`, `shares/docs/current_implementation_status.md`, `execution/docs/evidence/2026-09-17_phase_review.md` | docs |

---

### Task 1: FSM — reject blocks, block per episode, lost exit legs

**Files:** modify `execution/core/order_fsm.py`; test `execution/tests/test_order_fsm.py`.

**Produces:** `OrderMachine._blocked: tuple[str, str] | None` (episode id, signature); journal `trade.kind == "exit_leg_lost"` with `exit_role`, `order`, `reason`.

- [ ] Test `test_a_rejected_entry_blocks_the_signature_until_the_plan_changes`: a broker subclass whose first poll after submission emits `rejected` for the entry; after the reject the machine is IDLE, three more bars with the same plan submit nothing; a plan with another target submits.
- [ ] Test `test_an_expired_signature_is_tradeable_again_in_a_new_episode`: TTL expiry in `EP_…_001`, then the same plan with `episode_id="EP_…_002"` submits (journal opened for both).
- [ ] Test `test_a_lost_exit_leg_is_journaled_once`: fill the entry, then a broker subclass emits `cancelled` for the STOP leg; the machine stays IN_POSITION, the journal holds one `exit_leg_lost` with `exit_role == "stop"`; a second poll with nothing new adds nothing.
- [ ] Run: expect 3 failures (`exit_leg_lost` missing, resubmission, no submit in EP2).
- [ ] Implement: in `_apply`, `kind in ("expired","rejected")` sets `self._blocked = (intent.plan.episode_id, intent.plan.signature)`; IDLE branch compares `(plan.episode_id, plan.signature)`; non-entry branch: `cancelled` / `rejected` / `expired` → `self._record(kinds, "exit_leg_lost", …, {**base, "exit_role": order.role.value})` unless the event batch also closed the position (record after the poll loop: collect lost legs, emit only if `self._intent` is still that intent).
- [ ] Run `execution/tests/test_order_fsm.py`: all pass.

### Task 2: IBKR — disconnected is an error, flat-account check

**Files:** modify `execution/core/ibkr_broker.py`; test `execution/tests/test_ibkr_broker.py` (FakeIB gains `connected: bool = True`, `isConnected()`).

**Produces:** `class IBKRDisconnected(RuntimeError)`; `require_flat(snapshot: AccountSnapshot, symbol: str) -> None` raising `IBKRRefused` naming the positions / open entry orders found.

- [ ] Test `test_poll_and_snapshot_raise_when_the_socket_is_gone`: `ib.connected = False` → `poll` and `snapshot` raise `IBKRDisconnected`.
- [ ] Test `test_require_flat_names_what_is_in_the_way`: a snapshot with a position → `IBKRRefused` mentioning `NQ`; with an open entry order → refused; empty → returns None.
- [ ] Run: fail (no `isConnected` check, no `require_flat`). Implement. Run: pass.

### Task 3: runner guards

**Files:** modify `brain/scripts/run_llm_brain.py`; test `brain/tests/test_run_guards.py`.

**Produces:** `IBKR_MAX_TAPE_AGE = pd.Timedelta(days=1)`; `tape_is_current(end: str, *, now: pd.Timestamp, max_age=IBKR_MAX_TAPE_AGE) -> bool`; `main` returns 3 with a message when `--broker ibkr` and the tape is stale or the account is not flat.

- [ ] Test: `tape_is_current("2022-01-03 12:00", now=2026-09-17…)` is False; a window ending 2 hours before `now` is True.
- [ ] Run: fail (name missing). Implement; wire in `main` before connecting (`tape_is_current`) and right after (`require_flat(broker.snapshot(now), risk.contract.symbol)`). Run: pass.

### Task 4: `SimulatedExecutor` and its `VirtualAccount`

**Files:** create `execution/core/simulated_executor.py`, `execution/configs/simulated_executor.json`, `execution/tests/test_simulated_executor.py`; delete `execution/tests/test_simulated_broker.py` after its cases are carried over.

**Produces:**
```python
@dataclass(frozen=True)
class SimulatorConfig:
    initial_equity: float          # 100000.0
    margin_per_contract: float     # USD held per open or working contract
    max_fill_per_bar: int | None   # None: the whole remaining quantity fills on a touch
    sha256: str
    @classmethod
    def from_json(cls, path) -> "SimulatorConfig"

class VirtualAccount:
    cash: float                    # initial equity + realized PnL
    positions: dict[str, Position]
    orders: dict[str, OrderState]  # every order ever submitted, latest state
    fills: list[Fill]
    def pending(self) -> tuple[OrderState, ...]     # open statuses
    def filled(self) -> tuple[OrderState, ...]
    def cancelled(self) -> tuple[OrderState, ...]   # CANCELLED / EXPIRED
    def rejected(self) -> tuple[OrderState, ...]
    def margin_held(self, margin_per_contract) -> float
    def available_funds(self, margin_per_contract) -> float
    def summary(self) -> dict

class SimulatedExecutor:            # implements Broker; paper = True
    def __init__(self, config: SimulatorConfig, *, tick_size, point_value, equity: float | None = None, account_id="SIM")
    account: VirtualAccount
```
Rules (all at OHLCV level, no bid/ask, no queue): an entry limit works from the bar after submission; BUY LIMIT fills when `bar.low <= limit`, SELL LIMIT when `bar.high >= limit`, at the limit price, `min(remaining, max_fill_per_bar)` per touching bar (`partial` events until the last, then `filled`); the stop and target work from the bar after the first fill for the filled quantity; a bar touching both fills the stop; the loser is cancelled; realized PnL moves `cash`; `submit_bracket` accepts every intent and the next `poll` emits `rejected` when `quantity × margin_per_contract > available_funds`; `cancel` on a working or partial entry emits `cancelled` on the next poll, and a partial's exits keep working for the filled quantity.

- [ ] Tests (each one behaviour): fill timing and price; no fill on the submission bar; LONG mirror with target PnL; stop wins on a both-touch; cancel drops the children; determinism (carried over); `partial fills across two touching bars then filled`; `cancel after a partial keeps the position and its exits`; `margin reject on the next poll`; `available_funds counts held margin`; `account summary lists pending, filled, cancelled, rejected`; `config loads from json with initial_equity 100000`.
- [ ] Run: fail (module missing). Implement. Run: pass.

### Task 5: wire the executor, remove `SimulatedBroker`

**Files:** modify `execution/core/broker.py` (protocol only), `brain/scripts/run_llm_brain.py` (`--sim-config`, `--sim-equity` default None → config; run.json `simulator_config_sha256`, `simulated_account` summary at the end), `brain/scripts/replay_journal.py`, `execution/tests/test_order_fsm.py`, `execution/tests/test_stack_e2e.py`.

- [ ] Replace every `SimulatedBroker(...)` with `SimulatedExecutor(SimulatorConfig…, tick_size=…, point_value=…, equity=…)`; grep confirms no reference remains.
- [ ] Run `execution/tests risk/tests brain/tests/test_runtime.py brain/tests/test_run_guards.py`: pass.

### Task 6: docs, evidence, full suite

- [ ] `execution/docs/README.md` (module table, rules table with `exit_leg_lost`, the simulator section, the two runner guards), spec §4–5 amendments, `AGENTS.md` tree line, `shares/docs/current_implementation_status.md`, `risk/docs/README.md` unchanged.
- [ ] `execution/docs/evidence/2026-09-17_phase_review.md`: the findings table with locations, triggers, impact and what changed; the non-changed items.
- [ ] Full suite: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider`; echo run `--broker sim` on the audit window and its replay.
