# Review of the sleep, controller, risk and execution phase — 2026-09-17

Scope: the components built on 2026-09-16 ([../specs/2026-09-16-risk-execution-design.md](../specs/2026-09-16-risk-execution-design.md),
[../../brain/docs/evidence/2026-09-16_controller_eye_audit_2022-01-03.md](../../brain/docs/evidence/2026-09-16_controller_eye_audit_2022-01-03.md)),
read for logic holes, edge cases, state inconsistencies between components
and races — not for redesign. Fixes were limited to what the code confirms;
the plan is [../plans/2026-09-17-review-fixes-and-simulated-executor.md](../plans/2026-09-17-review-fixes-and-simulated-executor.md).

## 1. Findings

Severity: **high** = wrong or unmanaged orders at a real (paper) session;
**medium** = a wrong or stuck state in a run; **low** = a gap that does not
change what the machine does.

| id | severity | location | trigger | impact | change |
| --- | --- | --- | --- | --- | --- |
| H1 | high | `brain/scripts/run_llm_brain.py` (`--broker ibkr`) | the runner drives the Eye over the historical parquet; an `ibkr` run on a 2022 window submits limits at 2022 prices to today's paper market | a SELL limit far below the market is marketable and fills at once; a BUY limit never fills | `tape_is_current`: `ibkr` is refused (exit 3) when the window ends more than `IBKR_MAX_TAPE_AGE` (1 day) before now |
| H2 | high | `execution/core/ibkr_broker.py`, `order_fsm.py` | a restart, or a session with a bracket / position already at TWS | `IBKRBroker` forgets its orders, `poll` skips unknown ones, the machine starts IDLE, the Brain's ledger reads not engaged — the gate blocks a duplicate (`EXPOSURE` / `WORKING_ORDER`) but nothing manages the existing bracket (no TTL, no cancel on plan change) and the Brain may sleep with a position open | `require_flat`: the runner refuses to start over a position or a working entry order in the contract; recovery is not implemented and is documented as such |
| M1 | medium | `order_fsm._apply` | the broker rejects the entry (e.g. margin) | IDLE with the signature unblocked → the same plan is resubmitted on every bar: a reject loop against TWS | `rejected` blocks the signature as `expired` does |
| M2 | medium | `order_fsm._apply` | TWS cancels or rejects the stop or target while the position is open | silently ignored: a naked position with no journal trace | journaled once as `exit_leg_lost` (no automatic flatten) |
| M3 | medium | `order_fsm` | a signature expired in one episode; a later episode proposes the same three objects | blocked until a different plan appears — the new episode's reasoning is never traded | blocks are per `(episode_id, signature)` |
| M4 | medium | `ibkr_broker.poll` / `snapshot` | the socket to TWS drops | ib_async's cache is read silently: orders look working, fills are missed, until the next `placeOrder` / `cancelOrder` raises `ConnectionError` | `IBKRDisconnected` on the next `snapshot` / `poll`; no reconnect (the GTC bracket stays at TWS) |
| M5 | medium | `order_fsm` + `reducer` | the gate vetoes an ACTIONABLE plan on every bar | the LLM is never told; the opportunity blocks sleep and the idle rule, so the episode stays awake and keeps calling while the LLM insists | reported only — feeding the veto into the LLM input is a contract change |
| M6 | medium | `broker.SimulatedBroker` (removed) | a partial fill then a cancel | the exits were evaluated only while the entry was `FILLED`: a cancelled partial left a position whose stop and target never fired; no partial or reject model existed | superseded by `SimulatedExecutor` |
| M7 | medium | `risk/core/gate.py` | 5 contracts on a 100 000 account | `available_funds` / `buying_power` are never read: TWS would reject for margin (with M1 the loop stops after one reject) | reported; the simulator models margin per contract |
| L1 | low | `risk/core/gate.py` | any bar | `STALE_DATA` is inert: both brokers stamp `account.asof = asof` | reported |
| L2 | low | `brain/core/runtime.py` | an UPDATE that ends in an incident | `_last_llm_known_at` still advances, so an interaction step between the last good call and the failed one reads as seen by the sleep gate | reported (a fix changes the input shas of journals with incidents) |
| L3 | low | `ibkr_broker.poll` | two executions of one order between polls | one event per order per poll carries only the last unseen execution as `fill`; the state is right, the journal loses one fill payload | reported |
| L4 | low | `execution/core/stack.py` | every bar | the Brain steps before the machine polls, so it reads the machine's state of the previous bar (a fill on bar t is engaged from t+1) — conservative by construction | none |
| L5 | low | `order_fsm` | same signature, moved geometry | no in-place replace anywhere; only a range's value price can move under an unchanged signature, and the working limit stays until TTL or a cancel | documented |
| L6 | low | `ibkr_broker.snapshot` | orders placed from another client id | `ib.trades()` holds this client's orders plus the open orders fetched at connect; positions are account-wide | reported |

Checked and found consistent: the wake / UPDATE / TICK / idle-archive
interplay (rule 6b uses the count before the update and `_count_idle` the
result after it; an engaged Brain blocks both the LLM's sleep and the idle
rule; a refused understanding resets the idle run); Eye → controller →
Brain delivery (1m tape between calls, 5m+ bookkeeping deferred on TICK
bars and delivered with the next call, pending items re-offered, relation
changes of watched objects trigger a call — nothing is dropped while awake;
asleep, only wake kinds fire, by design); the gate reads
`broker.snapshot(asof)` after the bar's poll on every assessment (re-run
each bar, journaled once per (signature, vetoes)); cancel vs fill: the
simulator applies a cancel on the next poll before it evaluates that bar's
touch, deterministically, and at IBKR the broker's truth wins (a `filled`
after a cancel request opens the position, a `cancelled` with a filled part
keeps it); a re-signal is a no-op, a changed signature is cancel then submit
on the next bar, the machine never holds two entries; no wall clock in
`brain/core`, `execution/core`, `risk/core` (grep) — the simulator is driven
by bar timestamps only, `run.json`'s `started_at` / `finished_at` are not
part of the run identity, and only `IBKRBroker.poll` waits on the clock.

## 2. The simulated executor

`execution/core/simulated_executor.py` replaces `SimulatedBroker` behind the
unchanged `Broker` protocol; the order machine, the plan builder and the
Risk gate are the same objects a `--broker ibkr` run uses. Rules and the
virtual account: [../README.md](../README.md) § *The simulated executor's
rules*. It is an OHLCV approximation (a limit fills when the bar's range
reaches it, at the limit, from the bar after submission; no bid / ask, no
queue, no microstructure). "Replace" is what the machine does at both
brokers: cancel, then a new submission — the simulator shows it as a
cancelled order followed by a new pending one.

## 3. Verification

| check | result |
| --- | --- |
| new tests written first and watched fail: `test_order_fsm.py` (reject blocks, new episode trades an expired signature, lost exit leg), `test_ibkr_broker.py` (disconnected, `require_flat`), `brain/tests/test_run_guards.py`, `test_simulated_executor.py` (13 cases) | each failed on the missing behaviour, then passed |
| `execution/tests risk/tests brain/tests/test_run_guards.py` | 57 passed |
| `brain/tests` + `eyes/tests/test_eye_module_boundary.py` | 167 passed, 1 deselected |
| full default suite | **1380 passed, 1 deselected** (26 min 27 s) |
| `run_llm_brain --client echo --broker sim` on 2022-01-03 09:00–12:00 (warm-up from 2021-12-27) | 24 episodes, 35 calls, 0 trade records (the echo client proposes no opportunity); `run.json` carries `sim_equity` 100 000, `simulator_config_sha256` and the virtual account summary |
| replay of that echo run | replay OK — 24 episodes, 35 llm calls, 62 revisions, 0 trade records reproduced |
| replay of the real run 3 (`46a8bcf33f5f1f04`, Brain path only) | replay OK — 6 episodes, 78 llm calls, 168 revisions |

The echo journal was deleted afterwards. No real-model run was made in this
review; the fixes touch no LLM input, so run 3 still replays sha-equal.
