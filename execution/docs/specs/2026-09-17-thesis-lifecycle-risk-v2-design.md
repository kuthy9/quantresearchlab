# Thesis lifecycle and Risk v2 — design, 2026-09-17

Implements the five fixes of
[brain/docs/evidence/2026-09-17_trade_quality_root_cause_2022-01-03.md](../../../brain/docs/evidence/2026-09-17_trade_quality_root_cause_2022-01-03.md)
and the owner's new risk parameters. Written in an autonomous session: the
decisions below are stated, not negotiated; each one names the alternative
it rejected.

## 1. What changes, in one screen

| fix | where | what |
| --- | --- | --- |
| 1 thesis identity + lifecycle | contract, execution | the LLM names a `thesis_id` on every opportunity; a `ThesisBook` in the executor holds one record per id per episode, allows one expression at a time and at most `max_expressions` per thesis, closes the thesis after a stop or a target, and holds every new expression for `stop_cooldown_bars` after any stop-out |
| 2 invalidation on the thesis's scale, typed | contract, brain, execution | `governing_timeframe` and `invalidation_mode` (`TOUCH` / `CLOSE_BEYOND`) on the opportunity; the reducer refuses an invalidation object more than one scale below the governing one; `CLOSE_BEYOND` puts the hard stop one scaled 1m-ATR beyond the object and exits at market when a bar of the object's scale closes beyond it |
| 3 sizing veto never moves the stop | prompt, gate | `position_size` reads "the contract is too large for this stop"; the prompt tells the Brain to keep the invalidation where the thesis puts it and skip the trade |
| 4 leverage bound | gate | `max_leverage` caps contracts by notional ÷ equity; the risk budget, the margin and `max_quantity` still apply |
| 5 re-reason on the thesis's scale | controller, prompt | a watched object's relation flip triggers an UPDATE only for objects on `relation_change_timeframes` (15m and up); the prompt asks for `watch_next` on the governing scale or one below |
| risk v2 | gate | per-grade risk fraction (BASE 1.5 %, A_PLUS 2 %), `max_open_positions` 3, `min_reward_risk` 2, `preferred_reward_risk` 3 (A_PLUS sizing only at or above it), daily stop 2.5 % of the session's opening equity, hard stop 6.5 % from the equity peak (halts the run and flattens), TTL 15 bars unchanged |
| 3 positions | execution | the order machine holds several intents: one working entry at a time, up to `max_open_positions` open positions, never opposite directions |

## 2. Contract (`contract/`)

### 2.1 `Opportunity` (`contract/brain/state.py`)

Four fields join `state`, `direction` and the three object ids; all are
`None` when `state` is `NONE` and required otherwise:

| field | values | meaning |
| --- | --- | --- |
| `thesis_id` | `^[A-Za-z0-9_-]{1,32}$` | the reading this trade expresses; the same id while the reading holds |
| `governing_timeframe` | `4H` `1H` `15m` `5m` | the scale whose structure the thesis rests on |
| `grade` | `BASE` `A_PLUS` | the Brain's conviction tier; `A_PLUS` only when structure, delivery and liquidity agree across scales |
| `invalidation_mode` | `TOUCH` `CLOSE_BEYOND` | how the invalidation object falsifies the thesis: any trade through its far edge, or a bar of its scale closing beyond it |

`parse_update` requires the four keys inside `opportunity` (the reply
schema stays closed). `Opportunity.from_dict` reads them with defaults
(`None`, `None`, `BASE`, `TOUCH`) so a journal state written before this
design still loads; a reply written before it does not parse, so the three
regression baselines of 2026-09-17 are retired (§9).

### 2.2 `TradePlan` (`contract/risk/plan.py`)

Carries `thesis_id`, `governing_timeframe`, `grade`, `invalidation_mode`
(serialized, read back with the same defaults). `signature` is unchanged:
direction and the three entity ids. Two expressions of one thesis with
different objects have different signatures; the `ThesisBook` is what ties
them together.

### 2.3 Vocabulary

`VetoCode` gains `DAILY_STOP`, `HALTED`, `LEVERAGE`. `OrderRole` gains
`FLATTEN` (a market order that closes a position: the close-beyond exit
and the halt). `RiskVerdict` gains `grade_applied` (`BASE` / `A_PLUS`) and
`risk_fraction` (the fraction sized with).

## 3. Geometry (`brain/core/opportunity_geometry.py`)

`resolve_geometry(opportunity, objects, *, close, tick, atr_1m)`:

- `TOUCH`: stop = the invalidation object's far edge ± one tick (as today;
  rule `stop.<family>.far_edge`).
- `CLOSE_BEYOND`: stop = far edge ± `CLOSE_BEYOND_BUFFER_ATR × atr_1m ×
  sqrt(minutes(timeframe))`, rounded away from the entry to the tick; rule
  `stop.<family>.close_beyond`. `CLOSE_BEYOND_BUFFER_ATR = 1.0` is a named
  module constant and a keyword argument. Rationale: without a per-scale
  ATR in the Eye view, one 1m ATR scaled by the square root of the scale's
  minutes is the standard estimate of one bar's range on that scale — the
  room a wick needs before the bar closes. The reward-to-risk and the
  quantity are computed on this hard stop, never on the object's edge, so a
  close-beyond thesis is sized for the distance it can actually lose.

The scale check is the reducer's (§4), not the geometry's.

## 4. Reducer (`brain/core/reducer.py`), rule 4

Two new rejections drop the opportunity to `NONE` like the existing ones:

- `opportunity_invalidation_scale:<alias>` — the invalidation object's
  timeframe is more than one step below `governing_timeframe` on the ladder
  4H → 1H → 15m → 5m → 1m (4H allows 4H/1H, 1H allows 1H/15m, 15m allows
  15m/5m, 5m allows 5m/1m). The entry and target objects are free: a
  thesis is expressed where price is, but it is falsified on its own scale.
- `thesis_direction_changed:<id>` — the prior state's opportunity carries
  the same `thesis_id` with the other direction. A reading that flips is a
  new thesis with a new id. (The `ThesisBook` enforces the same across the
  whole episode; the reducer catches the immediate case so the state never
  records it.)

## 5. Risk gate v2 (`risk/core/gate.py`, `risk/configs/risk.json` schema 2)

```json
{
  "schema_version": 2,
  "risk_fraction": {"BASE": 0.015, "A_PLUS": 0.02},
  "max_open_positions": 3,
  "min_reward_risk": 2.0,
  "preferred_reward_risk": 3.0,
  "daily_loss_fraction": 0.025,
  "max_drawdown_fraction": 0.065,
  "max_leverage": 8.0,
  "max_quantity": 5,
  "order_ttl_bars": 15,
  "account_max_age_s": 120,
  "margin_per_contract": 20000.0,
  "thesis": {"max_expressions": 2, "stop_cooldown_bars": 30},
  "contract": {"symbol": "NQ", "exchange": "CME", "currency": "USD", "point_value": 20.0, "tick_size": 0.25}
}
```

`RiskGate.observe(account, asof)` runs once per bar before anything else
(the machine calls it): it records the session's opening equity (NY
session date = `(asof in New York + 6 h).date()`, so 18:00 Sunday belongs
to Monday), the equity peak across the run, and latches `halted` when
`equity ≤ peak × (1 − max_drawdown_fraction)`. `daily_stopped(asof)` is
true for the rest of the session date once `equity ≤ opening × (1 −
daily_loss_fraction)`.

`RiskGate.assess(plan, account, *, asof, positions)` — `positions` are the
executor's own open `PositionRecord`s (the account nets contracts per
symbol, so it cannot count brackets). In order, stopping at the first
group that fails:

| check | veto |
| --- | --- |
| no plan | `NO_PLAN` |
| `halted` | `HALTED` |
| `daily_stopped(asof)` | `DAILY_STOP` |
| `asof − account.asof > account_max_age_s` | `STALE_DATA` |
| `len(positions) ≥ max_open_positions`, or a position in the other direction, or (no positions of ours) a net position in the contract | `EXPOSURE` |
| an entry order already working | `WORKING_ORDER` |
| stop / target on the wrong side | `INVALID_STOP` / `INVALID_TARGET` |
| reward-to-risk on tick-rounded prices < `min_reward_risk` | `REWARD_RISK` |
| `floor(equity × fraction ÷ (risk_points × point_value)) < 1` | `POSITION_SIZE` — reason: "risk budget X buys no contract at Y per contract: the contract is too large for this stop" |
| `floor(equity × max_leverage ÷ (limit_price × point_value)) − contracts already open < 1` | `LEVERAGE` (amended after the first run: the cap counts the account's open contracts, else three positions could hold 6 contracts at 8× each) |
| `floor(available_funds ÷ margin_per_contract) < 1` | `POSITION_SIZE` |

`fraction` = `risk_fraction["A_PLUS"]` when `plan.grade` is `A_PLUS` and the
reward-to-risk is at or above `preferred_reward_risk`, else
`risk_fraction["BASE"]` (`grade_applied` says which). Rejected alternative:
letting `A_PLUS` size 2 % at 2 R — the owner's "preferred 3 R+" would then
change nothing; tying the larger budget to the preferred ratio is the one
place the ratio can act without reshaping a trade.

`quantity = min(budget contracts, max_quantity, leverage contracts, margin
contracts)`. With 100 000 USD and NQ at 16 400 (328 000 USD a contract)
`max_leverage` 8 allows two contracts; the BASE budget of 1 500 USD holds
a 75-point stop on one contract (the old 0.5 % held 25).

## 6. Executor

### 6.1 `SimulatedExecutor`

- `equity` becomes cash plus the open positions marked at the last polled
  close of their symbol (the account summary and the day / peak tracking
  need mark-to-market; `available_funds` stays cash less margin held).
- `cancel(order_id)` accepts a stop or a target of a filled bracket (the
  leg is cancelled on the next poll; a bracket whose both exits are
  cancelled keeps its position with no exits, as TWS would).
- `flatten(symbol, quantity, side, asof, client_ref)` submits a market
  order (`OrderRole.FLATTEN`) that fills on the next poll at that bar's
  open, before any limit or stop of the bar is matched; it is reported
  `filled` with its fill and moves the account like any fill.
- Amended 2026-09-19 (direction fix §3): an entry limit through the market
  (a BUY at or above the bar's open, a SELL at or below) fills at the open
  rather than at the limit — the fill a marketable limit gets at TWS.

### 6.2 `IBKRBroker`

`flatten` takes the same signature and sets `orderRef = client_ref`; cancel
already accepts any order id. The paper exercise keeps using it.

### 6.3 `ThesisBook` (`execution/core/thesis.py`)

Per episode, one `ThesisRecord` per `thesis_id`: `direction`,
`governing_timeframe`, `opened_at`, `expressions` (orders submitted),
`status` (`OPEN` / `CLOSED`), `closed_reason` (`stopped`, `achieved`,
`expressions_exhausted`, `direction_changed`), `last_outcome`. The book
also keeps `cooldown_until_bar` (bars since the last stop-out).

`admit(plan, bar_index) -> str | None` returns the refusal, or `None`:

| condition | refusal |
| --- | --- |
| a record with this id has the other direction | `direction_changed` (the record is closed) |
| the record is `CLOSED` | `thesis_closed` |
| `expressions ≥ max_expressions` | `expressions_exhausted` (closes the record) |
| a stop-out happened fewer than `stop_cooldown_bars` bars ago | `stop_cooldown` |
| another intent of this thesis is working or in position | `thesis_engaged` |

`expressed(plan)` counts an order; `outcome(plan, kind, exit_role)` closes
the record on a stop (`stopped`) or a target (`achieved`) and starts the
cooldown on a stop; an expiry or a cancel leaves it open. The book resets
with the episode. `view()` is what the LLM reads (§7.3). Amended
2026-09-19 (direction fix §3): an expiry gives its expression back (the
entry was never reached), a cancel keeps it; the bias-reversal flatten
(§6.7) closes the record `bias_reversed` without a cooldown.

Rejected alternative: keying the lifecycle on a code-derived thesis
(direction + governing scale + destination). The Brain's own id is what
lets it be told "T3 is closed"; the cooldown after any stop is what stops
a renamed thesis from re-entering the same minute.

### 6.4 `OrderMachine` — several intents

```text
per intent:  WORKING ──fill──► IN_POSITION ──stop / target / flatten──► done
             WORKING ──partial──► PARTIAL ──rest──► IN_POSITION
             WORKING ──cancel / expire / reject──► done
machine.state = WORKING if an entry works, else IN_POSITION if a position is open, else IDLE
```

- `_intents: dict[signature, _Intent]`; at most one intent in WORKING /
  PARTIAL (the working entry); positions up to the gate's limit, checked by
  the gate on every submission with the machine's own `positions()`.
- Every bar: poll and route events by `client_ref` to their intent;
  `gate.observe(snapshot)`; the halt (§6.5); the working entry's cancel
  rules as today (plan dropped, signature changed, entry invisible, TTL);
  the close-beyond monitor (§6.6); then, when no entry works, the plan
  is new (no intent with its signature, not blocked, not awaiting the
  Brain), the `ThesisBook` admits it and the gate passes, one bracket.
- Blocked signatures (`expired`, `rejected`, `position_closed`) are a set
  per episode, as before per signature. `_await_brain` after any position
  closes, as before.
- A plan whose signature matches an open position is that position's
  description: nothing is submitted, nothing is cancelled. A plan with a
  new signature while positions are open goes to the book and the gate
  (same direction, limit, budget) — this is how the third position opens.
- Journal kinds added to `STAT_KINDS`: `thesis_refused` (journaled on the
  first bar of a (thesis id, refusal) pair in the episode and on every
  `llm_called` bar that re-proposes it, like `veto`), `invalidation_close`
  (the close-beyond exit was requested), `flattened` (a flatten filled;
  the `position_closed` record carries `exit_role: "flatten"`), `halted`.
- `execution_view()` (§7.3) reports `positions` (a list), the working
  `order`, `theses`, `cooldown_bars_left`, `daily_stop`, `halted`.

### 6.5 Halt

When `gate.halted` turns true (observed on the bar's snapshot): cancel the
working entry, cancel every open stop and target of ours, flatten every
position of ours with one market order per position (`client_ref =
"<episode>:<signature>:halt"`), journal `halted` with the equity, the peak
and the drawdown, and refuse everything afterwards. `OrderMachine.halted`
is true from then on; `TradingStack.halted` mirrors it; `drive(...,
stop=lambda: stack.halted)` ends the pass after the bar on which the
flatten fills are polled (one more bar), and `run.json` records `halted`
(`at`, `equity`, `peak`, `drawdown`). A live run stops the same way; the
owner restarts it by hand.

### 6.6 Close-beyond exit

`TradingStack` passes the machine the scales whose bar completed on this
1m bar (`bar_completed` events of the observation) — on that bar the
1m close is the scale's close. For each position whose plan is
`CLOSE_BEYOND`, when its invalidation object's scale completed a bar
whose close is beyond the object's far edge (above for a SHORT, below for a
LONG), the machine cancels the bracket's stop and target, submits a
flatten (`client_ref = "<episode>:<signature>:invalidation"`), journals
`invalidation_close`, and the flatten's fill closes the position with
`exit_role: "flatten"`. The hard stop stays at the broker until the cancel
is confirmed, so a runaway bar is still stopped.

### 6.7 Bias-reversal exit (2026-09-19)

`TradingStack` also passes the machine the Brain's bias direction
(`state.bias.direction`, `None` when NEUTRAL). A position whose plan
direction is the other side is flattened at market the same way as §6.6:
`bias_reversed` journaled with the bias and the position, the stop and
target cancelled, the flatten's fill closing the position with
`exit_role: "bias_reversed"`, and the thesis closed `bias_reversed` with no
cooldown. Reducer rule 4b already refuses a *new* opportunity against the
bias; this is the same rule applied to the open position (design
[2026-09-18-direction-eye-brain-execution-design.md](../../../brain/docs/specs/2026-09-18-direction-eye-brain-execution-design.md) §3).

### 6.8 The entry model (2026-09-20)

Amended by [brain/docs/specs/2026-09-20-entry-model-design.md](../../../brain/docs/specs/2026-09-20-entry-model-design.md):
`order_ttl_bars` counts bars of the entry object's own scale (`risk.json`
schema 3; the machine converts to 1m bars and reports that figure as
`ttl_bars`); a cancel that replaced the entry object (`signature_changed`)
or lost it to the Eye (`entry_object_not_visible`) gives the thesis its
expression back like an expiry; a plan whose limit the market is already
past is held with `thesis_refused` reason `entry_marketable`.

## 7. Brain

### 7.1 Sleep Controller (`brain/configs/sleep_controller.json` schema 3)

`relation_change_timeframes: ["15m", "1H", "4H"]`. The runtime remembers
relations only for watched / opportunity objects on those scales; a 5m
pool flipping side no longer triggers an UPDATE (49 % of the day run's
calls). Evidence-triggered UPDATEs are unchanged.

### 7.2 Prompt (`brain/configs/prompts/main_brain_system.md`)

- Opportunity: the four fields, their vocabulary and rules — a `thesis_id`
  names one reading on one governing scale and stays while the reading
  holds; a flipped or replaced reading is a new id; the invalidation object
  lives on the governing scale or one below and is the thesis's own
  falsification, never the nearest pool; `CLOSE_BEYOND` when the
  falsification is acceptance, `TOUCH` when any trade through the level
  ends the thesis; `A_PLUS` is earned by cross-scale agreement and is sized
  larger only when code finds the reward-to-risk at or above the preferred
  ratio.
- `prior_state.execution`: `positions` (a list), `order`, `theses` with
  their status and closed reason, `cooldown_bars_left`, `daily_stop`,
  `halted`, `last_outcome`, `last_veto`. A closed thesis is not proposed
  again in the episode; during the cooldown or the daily stop no
  opportunity is `ACTIONABLE`.
- `last_veto` guidance rewritten: `reward_risk` → a nearer entry or a
  farther target, never a nearer invalidation; `position_size` /
  `leverage` → the contract is too large for this stop at the account's
  budget; the trade is skipped, the invalidation stays where the thesis
  puts it; `daily_stop` / `halted` → no trade today / the model is stopped.
- `watch_next` on the governing scale or one below; the thesis is
  re-examined when that scale delivers, not when a 5m pool is crossed.

### 7.3 `execution_view()` shape (`brain/core/position_ledger.py`)

```json
{"status": "IDLE|WORKING|PARTIAL|IN_POSITION", "order": null | {…as today…, "thesis_id": "T3"},
 "positions": [{"thesis_id": "T3", "direction": "SHORT", "entry_object_id": "FVG_15m_2", "opened_at": "…", "quantity": 1, "invalidation_mode": "CLOSE_BEYOND"}],
 "theses": [{"thesis_id": "T3", "direction": "SHORT", "governing_timeframe": "1H", "status": "CLOSED", "closed_reason": "stopped", "expressions": 1}],
 "cooldown_bars_left": 0, "daily_stop": false, "halted": false,
 "last_outcome": null | {…}, "last_veto": null | {…}}
```

`IDLE_VIEW` is this with empty lists and nulls.

## 8. Runner and summary

`run_llm_brain.py`: `drive(..., stop=lambda: stack.halted)`; `run.json`
gains `halted`. `summarize_run.py`: the new stat kinds appear in `orders`
and a `risk` line gains `thesis_refused` by reason, `daily_stop_bars`,
`halted`. The regression test replays the runs named in
`regression_baselines.json` as before.

## 9. Retired and replaced

- The three regression baselines (`302c54bb21d9d77e`, `04e7302f3cb72bbd`,
  `7ec17f066d232ba4`) cannot replay under the new reply schema and gate;
  their journals stay under `outputs/brain_journal/` as evidence and
  `regression_baselines.json` is rewritten with the first complete run of
  this design.
- `risk.json` schema 1 is refused (`unsupported risk schema_version`).
- `sleep_controller.json` schema 2 is refused.

## 10. Tests

| file | proves |
| --- | --- |
| `brain/tests/test_llm_contract.py` | the four fields are required when the state is not NONE, null when it is; bad values refused |
| `brain/tests/test_brain_state.py` | `Opportunity.from_dict` defaults for old states |
| `brain/tests/test_opportunity_geometry.py` | close-beyond buffer arithmetic and rounding; TOUCH unchanged |
| `brain/tests/test_reducer.py` | invalidation scale rule; thesis direction rule |
| `risk/tests/test_gate.py` | grade sizing, preferred ratio, leverage cap, positions count, opposite direction, daily stop by session date, drawdown halt latch, veto wording |
| `execution/tests/test_simulated_executor.py` | mark-to-market equity, exit-leg cancel, flatten fill at next open |
| `execution/tests/test_thesis.py` | every admit refusal, outcomes, cooldown, episode reset, view |
| `execution/tests/test_order_fsm.py` | three positions, opposite direction refused, thesis refusals journaled, close-beyond exit, halt flattens everything, view shape |
| `execution/tests/test_order_scenarios_real_tape.py` | the nine scenarios still hold with the v2 config |
| `execution/tests/test_stack_e2e.py` | the scripted Brain names the new fields; `positions` in `prior_state.execution`; replay |
| `brain/tests/test_sleep_controller.py`, `test_runtime.py` | relation changes filtered by scale |
| `brain/tests/test_main_brain.py` | the prior view carries the new execution shape |
| `brain/tests/test_summarize_run.py` | the new counters |
| `brain/tests/test_run_guards.py` | `drive` stops on the predicate |

## 11. Documentation

This spec; amendments in `2026-09-16-risk-execution-design.md` (§3, §5)
and `brain/docs/specs/2026-09-16-llm-brain-design.md` (§7.2); the
`execution/`, `risk/` and `brain/` READMEs; `AGENTS.md`;
`shares/docs/current_implementation_status.md`; the prompt; the run's
receipt under `brain/docs/evidence/`.
