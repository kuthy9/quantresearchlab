# Execution

Execution owns order and position reality, and — since 2026-09-16 — the
order machine that acts on the LLM Brain's opportunities through a broker
boundary. Design: [specs/2026-09-16-risk-execution-design.md](specs/2026-09-16-risk-execution-design.md);
plans: [plans/2026-09-16-risk-execution.md](plans/2026-09-16-risk-execution.md),
[plans/2026-09-17-review-fixes-and-simulated-executor.md](plans/2026-09-17-review-fixes-and-simulated-executor.md);
review: [evidence/2026-09-17_phase_review.md](evidence/2026-09-17_phase_review.md).

```text
BrainRuntime.step ─► state.opportunity ACTIONABLE ─► plan_from_state ─► TradePlan
                                                                          │
                              risk/core/gate.py  RiskGate.assess(plan, AccountSnapshot) ─► RiskVerdict
                                                                          │ passed
                              order_fsm.py       OrderMachine.on_bar ─► Broker.submit_bracket / cancel / poll
                                                                          │
                              simulated_executor.py  SimulatedExecutor (replay, tests)   ibkr_broker.py  IBKRBroker (paper)
```

`configs/model.json` keeps `live_execution_allowed: false`; the IBKR adapter
refuses to construct when it is true and refuses any account whose id does
not start with `DU` (paper).

## Core modules — `execution/core/`

| module | owns |
| --- | --- |
| `broker.py` | the `Broker` protocol (`snapshot`, `submit_bracket`, `cancel` — an entry, or a stop / target of a filled bracket —, `flatten` — a market order —, `poll`) — one seam, two implementations, no in-place order modification in either (the machine replaces by cancel + submit) |
| `simulated_executor.py` | `SimulatedExecutor`, the local executor for historical replay: IBKR's account and matching replaced by a `VirtualAccount` (cash, positions, every order at its latest state — pending / filled / cancelled / rejected — and the fills) and fixed OHLCV rules (below); `SimulatorConfig` from `execution/configs/simulated_executor.json` |
| `ibkr_broker.py` | `IBKRBroker` over `ib_async` (TWS / IB Gateway socket API): `accountSummary` / `positions` / `trades` / `fills` → `AccountSnapshot`, `bracketOrder` + `placeOrder` (GTC, `orderRef` = the intent's `client_ref`), `cancelOrder`, `waitOnUpdate` then a status diff → `BrokerEvent`s; a dropped socket raises `IBKRDisconnected` on the next `snapshot` / `poll` (no silent cache, no reconnect); `require_flat` refuses to start over a position or a working entry order in the contract (no recovery); `IBKRConfig` from `execution/configs/ibkr.json`; `ib_async` is imported only in `IBKRBroker.connect` |
| `plan.py` | `plan_from_state(state, context, tick)`: an ACTIONABLE opportunity → `TradePlan` with the three aliases, their Eye entity ids (the `data_id` a reader resolves back to the Eye), the geometry resolved on the current bar (the 1m ATR feeds a `CLOSE_BEYOND` buffer), the thesis fields and the invalidation object's far edge; None otherwise |
| `thesis.py` | `ThesisBook` (2026-09-17): one record per `thesis_id` per episode; `admit` refuses `direction_changed`, `thesis_closed` (after a stop, an invalidation exit or a target), `expressions_exhausted` (`thesis.max_expressions`; since 2026-09-19 an expiry gives its expression back; since 2026-09-20 so does a cancel that replaced the entry object (`signature_changed`) or lost it to the Eye (`entry_object_not_visible`) — `REPLACEMENT_REASONS`; a dropped plan keeps it), `thesis_engaged` (one expression at a time) and `stop_cooldown` (`thesis.stop_cooldown_bars` after any stop-out); a bias-reversal flatten closes the thesis `bias_reversed` without a cooldown; `view` is the `theses` / `cooldown_bars_left` the LLM reads |
| `order_fsm.py` | `OrderMachine`: several intents keyed by the plan's `signature` — one working entry at a time, positions up to the gate's `max_open_positions` in one direction; a re-signal is a no-op; a changed signature, a dropped plan, an invisible entry object or `order_ttl_bars` bars of the entry object's scale without a fill (2026-09-20: 75 1m bars for a 5m object, 225 for a 15m one; `execution_view.order.ttl_bars` reports the 1m figure) cancels the working entry; a plan whose limit the market is already past is held, not submitted (`thesis_refused` with reason `entry_marketable`, 2026-09-20 — a chase is not an expression); an expired, rejected or closed signature is not resubmitted in that episode; the `ThesisBook` and the gate are asked before every submission (`thesis_refused` / `veto` journaled per LLM proposal); a `CLOSE_BEYOND` position is flattened at market when a bar of its invalidation object's scale closes beyond the object (`invalidation_close`); a position on the other side of the Brain's bias is flattened at market (`bias_reversed`, 2026-09-19); the gate's drawdown halt cancels and flattens everything (`halted`) and nothing is submitted again; a stop or target the broker cancels or rejects on its own is journaled `exit_leg_lost`; every transition is a journal `trade` record and `stats` counts them; `ExecutionLedger` is the Brain's `PositionLedger` (`has_open_position`, `has_working_order`, `execution_view` → `prior_state.execution`: status, order, positions, theses, cooldown, daily stop, halted, last outcome, last veto) |
| `stack.py` | `TradingStack.step(observation, bar)`: Brain step → plan → machine, one bar at a time, telling the machine whether the LLM was called (`llm_called`) and which scales completed a bar (`closed_timeframes`, from the Eye's `bar_completed` events); `halted` mirrors the machine's; optional `Timings` (`plan`, `machine`) |
| `execution.py`, `mbo.py` | execution reality, MBO reconstruction (unchanged) |

Contracts: `contract/execution/account.py` (`AccountSnapshot`, `Position`,
`OrderState`, `OrderStatus`, `OrderRole`, `Fill`, `BracketIntent`,
`BrokerEvent`) and `contract/risk/plan.py` (`ObjectRef`, `TradePlan`,
`RiskVerdict`).

## The order machine's rules

| situation | action | journal `trade.kind` |
| --- | --- | --- |
| no entry working, a new plan (its signature is not an intent, not blocked), entry object visible, the `ThesisBook` admits it, the gate passes | one bracket (entry limit + stop + target); up to `max_open_positions` positions, all in one direction | `submitted`, then `working` |
| the `ThesisBook` refuses (`direction_changed`, `thesis_closed`, `expressions_exhausted`, `thesis_engaged`, `stop_cooldown`) | nothing; recorded on the first bar of a (thesis, reason) pair in the episode and on every bar on which the LLM proposed it again | `thesis_refused` |
| the gate refuses | nothing; counted every bar (`stats.veto_bars`), recorded on the first bar of a (signature, vetoes) pair in the episode and on every bar on which the LLM, having seen `prior_state.execution.last_veto`, proposed it again (`proposals_vetoed`) | `veto` |
| WORKING / PARTIAL, same signature again; or a plan whose signature is an open position | nothing | — |
| WORKING / PARTIAL, plan gone or signature changed | cancel the entry | `cancel_requested` → `cancelled` |
| WORKING / PARTIAL, entry object no longer visible | cancel the entry | `cancel_requested` → `cancelled` |
| WORKING / PARTIAL, `order_ttl_bars` (15 bars of the entry object's scale, 2026-09-20) without a full fill | cancel; the signature stays blocked in this episode; the thesis gets the expression back | `cancel_requested` → `expired` |
| the broker rejects the entry | the intent ends; the signature stays blocked in this episode | `rejected` |
| entry fills | position open; the Brain cannot sleep | `filled`, `position_opened` |
| stop or target fills | position closed; the signature stays blocked in this episode, the thesis is closed (`stopped` / `achieved`), a stop starts the cooldown, and nothing at all is submitted until the Brain's next call (it then sees `last_outcome`) | `position_closed` |
| a `CLOSE_BEYOND` position and a bar of the invalidation object's scale closes beyond the object's far edge | cancel the stop and target, flatten at market; the fill closes the position with `exit_role` `invalidation` (a stop for the thesis book) | `invalidation_close`, `cancel_requested` ×2, `position_closed`, `flattened` |
| the Brain's bias is LONG or SHORT and a position is on the other side (`bias_direction` from `TradingStack`; NEUTRAL is not a reversal) | cancel the stop and target, flatten at market; the fill closes the position with `exit_role` `bias_reversed` (the thesis closes `bias_reversed`, no cooldown) | `bias_reversed`, `cancel_requested` ×2, `position_closed`, `flattened` |
| the gate's drawdown halt (`observe` on the bar's snapshot) | cancel the working entry and every exit leg, flatten every position (`exit_role` `flatten`), refuse everything afterwards; the runner stops | `halted`, then `position_closed`, `flattened` |
| the broker cancels or rejects a stop or target on its own while the position is open | nothing automatic; the leg is journaled once | `exit_leg_lost` |

A cancelled PARTIAL keeps its filled part as a position with the stop and
target still working. A new episode is a new intent: its blocks and its
thesis book start empty. There is no in-place replace; a plan whose
signature changed is cancelled and the new one submitted on the next bar,
at the simulator and at IBKR alike.

## The simulated executor's rules

OHLCV-level matching only — no bid / ask, no queue position, no market
microstructure. The same bars give the same events, so a `sim` journal
replays.

| rule | |
| --- | --- |
| entry works | from the bar after its submission |
| `BUY LIMIT` fills | when a later bar's `low <= limit`, at the limit |
| `SELL LIMIT` fills | when a later bar's `high >= limit`, at the limit |
| quantity per touching bar | `max_fill_per_bar` (`null`: the whole remainder) — `partial` events until the last bar, then `filled` |
| stop and target | work from the bar after the first fill, for the filled quantity; a bar touching both fills the stop; the other leg and any unfilled entry remainder are cancelled |
| reject | `quantity × margin_per_contract > available funds` at submission → `rejected` on the next poll (asynchronous, as at TWS) |
| cancel | takes effect on the next poll; a partial keeps its filled part as a position with its exits working; a stop or target of a filled bracket can be cancelled too (a position whose both exits are cancelled stays open with none) |
| flatten | a market order (`OrderRole.FLATTEN`) fills on the next poll at that bar's open, before any limit or stop of the bar is matched |
| marketable limit | an entry whose limit is through the market (a BUY at or above the bar's open, a SELL at or below) fills at the open, not at the limit (2026-09-19) |
| account | `cash` = initial equity + realized PnL; `equity` = cash + open positions marked at the last polled close; `available_funds` = cash − margin held for open and working contracts; orders kept in submission order at their latest state |

`execution/configs/simulated_executor.json`: `initial_equity` 100 000,
`margin_per_contract` 20 000 (an approximation of NQ's initial margin;
override per run), `max_fill_per_bar` null.

## Running

```bash
.venv/bin/python -m brain.scripts.run_llm_brain --client deepseek --broker sim \
  --warmup-start 2021-12-27 --emit-start "2022-01-03 09:00" --end "2022-01-03 12:00" \
  --max-llm-calls 80 --sim-equity 100000
```

`--broker none` (default) runs the Brain alone; `sim` runs the gate and the
machine against the simulated executor (`--sim-config`, `--sim-equity`
overrides the starting cash) and `replay_journal.py` re-drives the machine
from `run.json`'s `sim_equity` and the simulator config, comparing every
`trade` record; `run.json` ends with the virtual account's summary
(`simulated_account`). `ibkr` connects to the paper session in
`execution/configs/ibkr.json` (an `ibkr` journal replays its Brain path only
— fills are the broker's) and is refused, exit 3, when the window ends more
than a day before now (its limits would belong to another market) or when
the account already holds a position or a working entry order in the
contract (there is no recovery of orders the machine did not place).

Before an `ibkr` run, with TWS or IB Gateway logged into the paper account
and the API enabled on the configured port:

```bash
.venv/bin/python -m execution.scripts.ibkr_paper_check
```

prints the `AccountSnapshot` the gate would read and never places an order.

```bash
.venv/bin/python -m execution.scripts.ibkr_paper_exercise --reference-price <last NQ price> --i-place-paper-orders [--marketable]
```

places, by hand and on the paper account only, what the machine would
place and records what TWS reports: an unmarketable SELL bracket 1 % above
the reference until `working`, its cancel until `cancelled`, a replacement
(cancel + resubmit, 1.5 % above) and, with `--marketable`, a SELL bracket
0.5 % below the reference until `filled`, the cancel of its stop and
target, and a market flatten (`IBKRBroker.flatten`, which the machine never
calls). The receipt — every `BrokerEvent` and snapshot — lands in
`outputs/ibkr_paper/<timestamp>.json`; it exits 0 only when every step
reported what was expected and the account is flat with no open order at
the end. A partial fill cannot be forced on the paper session.
Set `contract.last_trade_month` in `execution/configs/ibkr.json` to the front
month before running.

## Protocols — `execution/configs/`

`ibkr.json`: host, port, `client_id`, `paper_only` (must stay true),
`poll_timeout_s`, the contract (symbol, exchange, currency, last trade
month). No secret: TWS is already logged in. `simulated_executor.json`: the
virtual account's starting cash, margin per contract, fill quantity per bar.
Risk parameters live in `risk/configs/risk.json` (schema 2: the risk
fraction per thesis grade, three positions, the RR floor and the preferred
RR, the daily stop, the drawdown halt, the leverage cap, `max_quantity`, TTL,
`margin_per_contract`, the thesis book's limits — see
[risk/docs/README.md](../../risk/docs/README.md)).

## Tests — `execution/tests/`

| file | covers |
| --- | --- |
| `test_account_contract.py` | round trips, order invariants, open entry orders, bracket price sides |
| `test_simulated_executor.py` | config (100 000 start), fill timing and price, the `low <= limit` rule, LONG / SHORT mirror, stop-first, PnL, cancel drops children, determinism, partial fills across bars, cancel after a partial, margin reject, margin held in available funds, the account summary |
| `test_plan_from_state.py` | NONE / DEVELOPING → no plan; ACTIONABLE → entity ids and a stable signature; a vanished object → no plan |
| `test_order_fsm.py` | submit once, re-signal no-op, fill → position → stop closes, TTL expiry blocks the signature, signature change cancels then submits, invisible entry cancels, veto recorded once per bar-run and again per LLM re-proposal, partial then filled, a reject blocks until the plan changes, a new episode may trade an expired signature, a lost exit leg is journaled once, the execution view through the intent's life, veto memory per episode, a closed position is not re-entered, an expiry gives the expression back, a bias reversal flattens the position and closes the thesis, the TTL counts bars of the entry object's scale, a marketable entry is refused not chased |
| `test_order_scenarios_real_tape.py` | the nine order scenarios on the real 2022-01-03 RTH bars with scripted plans and the account checked after each: normal fill (SHORT and the LONG mirror), wait → cancel, wait → cancel → resubmit, wait → re-analyse (keep / cancel), partial → filled, cancel after a partial, risk veto, funds / position / quantity limits, duplicate signal, the cancel-versus-fill race |
| `test_ibkr_paper_exercise.py` | the paper exercise through the `Broker` protocol against a scripted fake TWS: working → cancel → replace, the marketable fill → cancel exits → flatten, refusal of a non-flat account, `flatten` places a market order |
| `test_ibkr_broker.py` | `FakeIB`: paper and live-flag guards, snapshot mapping, bracket shape (LMT / LMT / STP, GTC, `orderRef`), status diff into events, cancel, disconnected → `IBKRDisconnected`, `require_flat` |
| `test_stack_e2e.py` | synthetic Eye + a scripted ACTIONABLE LONG + `SimulatedExecutor`: bracket, fill, close, journal chain, replay with the machine, one open entry at most |
| `brain/tests/test_run_guards.py` | `tape_is_current` for `--broker ibkr`, the effort label, `drive` timings |

## Authority documents

[shares/docs/architecture.md](../../shares/docs/architecture.md) for the runtime
boundaries and
[shares/docs/current_implementation_status.md](../../shares/docs/current_implementation_status.md)
for implementation-versus-plan status.

## Scripts — `execution/scripts/`

`materialize_mbo_execution.py` streams level-3 MBO into one causal
execution-reality row per OHLCV minute; `ibkr_paper_check.py` is the
read-only paper connectivity check; `ibkr_paper_exercise.py` is the
by-hand paper order exercise described above. Throwaway probes belong here too.
