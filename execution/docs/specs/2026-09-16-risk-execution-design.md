# Risk gate and Execution FSM — design

Status: approved in chat on 2026-09-16; implemented by the plan in
[../plans/2026-09-16-risk-execution.md](../plans/2026-09-16-risk-execution.md).
Upstream: the LLM Brain ([../../../brain/docs/specs/2026-09-16-llm-brain-design.md](../../../brain/docs/specs/2026-09-16-llm-brain-design.md)).

## 1. What this adds

The Brain names objects; `opportunity_geometry` turns them into prices. Nothing
acted on them. This phase adds the two components downstream of the Brain in
the same per-bar loop:

```text
Eye ─► BrainRuntime.step ─► (state.opportunity ACTIONABLE?) ─► TradePlan
                                                                  │
                                    RiskGate.assess(plan, AccountSnapshot) ─► RiskVerdict
                                                                  │ passed
                                    OrderMachine.on_bar ─► Broker.submit_bracket / cancel / poll
                                                                  │
                                    PositionLedger (execution-backed) ─► the Brain cannot sleep while engaged
```

`live_execution_allowed` in `configs/model.json` stays `false`. The IBKR
adapter refuses any account whose id does not start with `DU` (paper), and
refuses to construct at all when `live_execution_allowed` is true — a later
phase decides what "live" means.

## 2. Contracts (`contract/`)

- `contract/risk/plan.py` — `TradePlan`: `episode_id`, `revision`, `known_at`,
  `direction`, the three aliases with their Eye entity ids (`entry`, `invalidation`,
  `target`: `ObjectRef(alias, entity_id, kind, timeframe)`), `geometry`
  (`OpportunityGeometry`), `close`, and `signature` — sha256 over
  (direction, entry entity id, invalidation entity id, target entity id),
  the identity of an intent across bars and revisions. `RiskVerdict`:
  `passed`, `vetoes: tuple[VetoCode, ...]`, `reasons`, and when passed
  `quantity`, `limit_price`, `stop_price`, `target_price`, `risk_amount`,
  `reward_risk`, `equity`. `VetoCode` gains `EXPOSURE`, `WORKING_ORDER`,
  `POSITION_SIZE`. The retired `RiskAssessment` (bound to `Action`) stays
  inert, as `contract/decision/action.py` does.
- `contract/execution/account.py` — `AccountSnapshot(account_id, asof, equity,
  available_funds, buying_power, positions, open_orders, fills, cancelled,
  source)`, `Position(symbol, quantity (signed), average_price)`, `OrderStatus`
  (`SUBMITTED`, `WORKING`, `PARTIAL`, `FILLED`, `CANCELLED`, `EXPIRED`,
  `REJECTED`), `OrderRole` (`ENTRY`, `STOP`, `TARGET`), `OrderState(order_id,
  client_ref, role, side, quantity, limit_price, stop_price, filled_quantity,
  average_fill_price, status, submitted_at, updated_at, parent_id)`,
  `Fill(order_id, quantity, price, at)`, `BracketIntent(client_ref, symbol,
  side, quantity, limit_price, stop_price, target_price, signature)`,
  `BrokerEvent(kind, order, fill, at)`.

Every contract is a frozen dataclass with `to_dict` / `from_dict`, validated in
`__post_init__`, following `contract/brain/state.py`.

## 3. Risk gate (`risk/`)

A fifth package with the same shape as the others: `risk/core/gate.py`,
`risk/configs/risk.json`, `risk/tests/`, `risk/docs/README.md`.

`RiskGate(config).assess(plan, account, *, asof) -> RiskVerdict`, in order,
stopping at the first veto group that fails:

| check | veto |
| --- | --- |
| no plan | `NO_PLAN` |
| `asof - account.asof > account_max_age_s` | `STALE_DATA` |
| open positions ≥ `max_open_positions`, or any open ENTRY order | `EXPOSURE` / `WORKING_ORDER` |
| stop on the wrong side of entry for the direction, or target on the wrong side | `INVALID_STOP` / `INVALID_TARGET` |
| `geometry.reward_risk < min_reward_risk` | `REWARD_RISK` |
| `quantity = floor(equity × risk_fraction ÷ (|entry − stop| × point_value)) < 1` | `POSITION_SIZE` |

`quantity` is capped at `max_quantity`; `limit_price` is the entry rounded to
the nearest tick (a limit must be reachable, so no bias either way);
`stop_price` / `target_price` are the geometry's, rounded to the tick.
`risk_amount = quantity × |entry − stop| × point_value`.

`risk/configs/risk.json`: `risk_fraction` 0.005, `max_open_positions` 1,
`min_reward_risk` 1.5, `max_quantity` 5, `account_max_age_s` 120, `contract`
{`symbol` NQ, `exchange` CME, `currency` USD, `point_value` 20.0,
`tick_size` 0.25}. The file's sha256 is part of a run's identity.

The gate never reads the Eye. The prices it reads are the geometry that
`opportunity_geometry` resolved on the same bar from the objects the Brain
named; the `data_id` the user asked for is the alias plus its Eye entity id in
the plan, which a reader resolves back through the episode's registry and the
Eye's journal.

Amended 2026-09-17: `margin_per_contract` (20 000) joins the config and the
quantity is also capped by `floor(available_funds / margin_per_contract)`
(a `POSITION_SIZE` veto when that is zero). The week run at high effort had
submitted fourteen 5-contract orders in 80 minutes that the executor
rejected on margin — the gate sized them against equity alone (review
finding M7).

Amended 2026-09-17 (Risk v2): superseded by
[2026-09-17-thesis-lifecycle-risk-v2-design.md](2026-09-17-thesis-lifecycle-risk-v2-design.md)
§5 — the fraction is per thesis grade (BASE 1.5 %, A_PLUS 2 % at or above
the preferred 3 R), three positions in one direction counted from the
executor, a leverage cap, a daily stop and a drawdown halt kept by
`observe`, schema 2.

## 4. Broker boundary (`execution/core/broker.py`)

```python
class Broker(Protocol):
    paper: bool
    def snapshot(self, asof) -> AccountSnapshot: ...
    def submit_bracket(self, intent: BracketIntent, asof) -> OrderState: ...
    def cancel(self, order_id: str, asof) -> None: ...
    def poll(self, asof, bar: Bar | None) -> tuple[BrokerEvent, ...]: ...
```

`SimulatedExecutor(config, tick_size, point_value, equity=None)` —
`execution/core/simulated_executor.py`, amended 2026-09-17 (it replaced the
first `SimulatedBroker`) — for historical replay and tests. It keeps this
protocol and the machine; only IBKR's account and matching are replaced:

- a `VirtualAccount`: cash (100 000 USD from
  `execution/configs/simulated_executor.json`, plus realized PnL), positions,
  every order at its latest state (pending / filled / cancelled / rejected),
  the fills; available funds = cash − `margin_per_contract` × (open +
  working contracts);
- an ENTRY limit works from the bar after submission; `BUY LIMIT` fills when
  a later bar's low ≤ limit, `SELL LIMIT` when high ≥ limit, at the limit,
  `max_fill_per_bar` contracts per touching bar (`null`: the whole
  remainder) — `partial` events, then `filled`;
- STOP and TARGET are live from the bar after the first fill for the filled
  quantity; when a bar touches both, the stop fills first (conservative); the
  other leg and any unfilled entry remainder are cancelled;
- `quantity × margin_per_contract` above the available funds → `rejected` on
  the next poll; a cancel takes effect on the next poll and a partial keeps
  its filled part with its exits working;
- OHLCV-level approximation only: no bid / ask, no queue, no microstructure;
- deterministic: the same bars give the same events, so a `--broker sim`
  journal replays.

`IBKRBroker` (`execution/core/ibkr_broker.py`, `ib_async`):

- `connect(host, port, client_id)` from `execution/configs/ibkr.json`
  (host, port, client_id, account, contract month; no secrets — TWS is
  already logged in);
- refuses unless `managedAccounts` contains an id starting with `DU` and
  `live_execution_allowed` is false;
- `snapshot`: `accountSummary` (NetLiquidation, AvailableFunds, BuyingPower),
  `positions`, `openTrades`, `fills`, and the cancelled trades of this session;
- `submit_bracket`: `bracketOrder(action, quantity, limitPrice,
  takeProfitPrice, stopLossPrice)` with `tif=GTC`, `transmit` on the last leg,
  `orderRef = client_ref`;
- `poll`: `waitOnUpdate(timeout)` then diff order statuses since the last poll
  into events;
- unit-tested against a `FakeIB`; the paper connection is verified read-only
  by `execution/scripts/ibkr_paper_check.py` (account, positions, open
  orders — it never places an order);
- amended 2026-09-17: a dropped socket raises `IBKRDisconnected` on the next
  `snapshot` / `poll` (never a stale cache; no reconnect, no recovery), and
  `require_flat` makes the runner refuse to start over an existing position
  or entry order in the contract; the runner also refuses `--broker ibkr`
  over a tape ending more than a day ago.

## 5. Order FSM (`execution/core/order_fsm.py`)

One `OrderMachine` per run. Per episode at most one intent; its identity is
`TradePlan.signature`.

```text
IDLE ──plan ACTIONABLE, risk passed, bracket submitted──► WORKING
WORKING ──entry fill──► IN_POSITION          WORKING ──cancel / expire / reject──► IDLE
WORKING ──partial fill──► PARTIAL ──rest fills──► IN_POSITION
IN_POSITION ──stop or target fill──► IDLE
```

`on_bar(asof, bar, plan, visible)`:

1. `broker.poll` → apply every event (fills open the position in the ledger;
   a stop/target fill closes it; cancel / expire / reject return to IDLE).
2. WORKING or PARTIAL: cancel when `bars_since_submit ≥ order_ttl_bars`
   (15), when `plan` is None or `plan.signature ≠ working.signature`, or when
   the entry object is no longer visible (`visible(entry alias)` false).
   A cancelled PARTIAL leaves the filled part as a position with its stop
   and target still working.
3. IDLE and `plan` is ACTIONABLE: `RiskGate.assess`; passed → `submit_bracket`
   → WORKING; vetoed → one `trade` record per (signature, vetoes) pair, not
   per bar.
4. The same signature arriving again while WORKING or IN_POSITION is a no-op:
   a re-signal never places a second order.
5. Amended 2026-09-17: a rejected entry blocks its signature like an expired
   one; blocks are per `(episode_id, signature)` so a new episode's identical
   plan is a new intent; a stop or target the broker cancels or rejects while
   the position is open is journaled `exit_leg_lost` once (no automatic
   flatten). There is no in-place replace: a changed signature is cancel +
   submit, at the simulator and at IBKR alike.
6. Amended 2026-09-17 (execution feedback): a closed position blocks its
   `(episode_id, signature)` like an expiry — the plan the Brain still
   holds is not re-entered on the exit bar — and after any close the
   machine submits nothing until the Brain has been called again (the
   2022-01-03 low-effort day showed a different SHORT, chosen while in the
   position, entered on the exit bar one minute after a stop-out); a veto is journaled on the
   first bar of a (signature, vetoes) pair *and* on every bar on which the
   LLM re-proposed it (`on_bar(..., llm_called=True)`), per episode; the
   machine exposes `execution_view()` (status, order, position, last
   outcome, last veto — aliases, never prices) through `ExecutionLedger`,
   and the Main Brain copies it into `prior_state.execution`. Design:
   `brain/docs/specs/2026-09-17-veto-feedback-week-backtest-design.md`.

7. Amended 2026-09-17 (thesis lifecycle): the machine holds several
   intents — one working entry at a time, positions up to the gate's
   limit in one direction; a `ThesisBook` (`execution/core/thesis.py`)
   is asked before the gate and its refusals are journaled
   `thesis_refused`; a `CLOSE_BEYOND` position exits at market when a bar
   of the invalidation object's scale closes beyond the object; the gate's
   drawdown halt cancels and flattens everything (`Broker.flatten`, a
   market order) and stops the run. Design:
   [2026-09-17-thesis-lifecycle-risk-v2-design.md](2026-09-17-thesis-lifecycle-risk-v2-design.md).

Every transition writes a journal `trade` record (`kind` ∈ plan, veto,
thesis_refused, submitted, working, partial, filled, cancelled, expired,
rejected, position_opened, position_closed, exit_leg_lost,
invalidation_close, flattened, halted) with the signature, ids, prices and
quantities. The Brain's `PositionLedger` is now `ExecutionLedger`, backed by
the machine: `has_open_position()` is true in IN_POSITION / PARTIAL, and a
new `has_working_order()` is true in WORKING. The runtime treats either as the
`open_position` sleep blocker, and the idle rule does not fire while engaged.

## 6. Wiring

- `brain/scripts/_run_identity.drive` hands the callback the `Bar` as well.
- `run_llm_brain.py --broker none|sim|ibkr` (default `none`): with a broker,
  every emitted bar runs `runtime.step`, builds the plan from
  `runtime.state.opportunity` when ACTIONABLE (`execution/core/plan.py:
  plan_from_state(state, context, tick)` — geometry resolved on the current
  context, entity ids from the state's registry), then `machine.on_bar`.
  `run.json` records the broker kind and the risk config sha.
- `replay_journal.py` re-drives `sim` runs with the same machine; an `ibkr`
  run replays its Brain path only (fills are the broker's, not reproducible).

## 7. Testing

- `risk/tests/test_gate.py`: sizing arithmetic, each veto, tick rounding.
- `execution/tests/test_simulated_executor.py` (2026-09-17, replacing
  `test_simulated_broker.py`): limit fill timing, stop-first precedence, PnL,
  determinism, partials, margin rejects, the account summary.
- `execution/tests/test_order_scenarios_real_tape.py` (2026-09-17): the
  nine order scenarios on the real 2022-01-03 RTH bars with scripted plans —
  normal fill, wait → cancel, wait → cancel → resubmit, wait → re-analyse
  (keep / cancel), partial → filled and cancel after a partial, risk veto,
  funds / position limits, duplicate signal, the cancel-versus-fill race.
- `execution/tests/test_order_fsm.py`: submit once per signature, re-signal
  no-op, TTL cancel, signature change cancel, entry-invisible cancel, partial
  then fill, stop/target close → IDLE, veto recorded once, journal chain.
- `execution/tests/test_ibkr_broker.py`: `FakeIB` — snapshot mapping, paper
  guard, `live_execution_allowed` guard, bracket shape, event diffing.
- `execution/tests/test_stack_e2e.py`: synthetic Eye + a scripted reply with an
  ACTIONABLE opportunity + `SimulatedExecutor` → a bracket, a fill, a close,
  the ledger blocking sleep in between, `prior_state.execution` in the
  journaled inputs, and the journal replaying.
