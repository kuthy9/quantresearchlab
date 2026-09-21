# Risk

The Risk gate sizes or vetoes the Brain's trade plan against the account. It
is a separate hard boundary: the Brain names objects, `opportunity_geometry`
turns them into prices, `plan_from_state` packs them into a `TradePlan`, and
only this gate decides quantity — or refuses with named `VetoCode`s. It never
reads the Eye and never moves a price the geometry did not give it; a sizing
veto means the contract is too large for this stop, never that the stop
should move.
Design: [../../execution/docs/specs/2026-09-16-risk-execution-design.md](../../execution/docs/specs/2026-09-16-risk-execution-design.md) §3,
amended by [../../execution/docs/specs/2026-09-17-thesis-lifecycle-risk-v2-design.md](../../execution/docs/specs/2026-09-17-thesis-lifecycle-risk-v2-design.md) §5 (Risk v2).

## `risk/core/gate.py`

`RiskGate(config).observe(account, asof)` runs once per bar (the order
machine calls it): it records the session's opening equity (a CME session
opens at 18:00 New York the evening before its date), latches the daily stop
for the rest of that session once equity is `daily_loss_fraction` below the
opening, tracks the equity peak and latches `halted` for good once equity is
`max_drawdown_fraction` below it (`halt_record`: at, equity, peak, drawdown).

`RiskGate(config).assess(plan, account, *, asof, positions) -> RiskVerdict`
— `positions` are the executor's own open positions (the account nets
contracts per symbol, so it cannot count brackets) — in order, stopping at
the first group that fails:

| check | veto |
| --- | --- |
| no plan | `NO_PLAN` |
| the run halted | `HALTED` |
| the session's daily stop | `DAILY_STOP` |
| `asof − account.asof > account_max_age_s` | `STALE_DATA` |
| positions ≥ `max_open_positions`, or one in the opposite direction, or (none of ours) a net position in the contract | `EXPOSURE` |
| an entry order already working | `WORKING_ORDER` |
| stop not on the losing side of the entry | `INVALID_STOP` |
| target not on the winning side of the entry | `INVALID_TARGET` |
| reward-to-risk (on tick-rounded prices) < `min_reward_risk` | `REWARD_RISK` |
| `floor(equity × fraction ÷ (|entry − stop| × point_value)) < 1` | `POSITION_SIZE` ("the contract is too large for this stop") |
| `floor(equity × max_leverage ÷ (entry × point_value)) − contracts already open < 1` | `LEVERAGE` |
| `floor(available_funds ÷ margin_per_contract) < 1` | `POSITION_SIZE` |

`fraction` is `risk_fraction["A_PLUS"]` when the plan's grade is `A_PLUS`
and the reward-to-risk is at or above `preferred_reward_risk`, else
`risk_fraction["BASE"]` (`grade_applied` and `risk_fraction` are in the
verdict). When it passes: `quantity = min(budget, max_quantity, leverage,
margin)`, `limit_price`, `stop_price` and `target_price` rounded to the tick,
`risk_amount = quantity × |entry − stop| × point_value`, `reward_risk`,
`equity`.

## `risk/configs/risk.json` (schema 3)

| field | default | meaning |
| --- | --- | --- |
| `risk_fraction` | `{"BASE": 0.015, "A_PLUS": 0.02}` | equity fraction at risk per trade, by thesis grade |
| `max_open_positions` | 3 | the executor's positions, all in one direction |
| `min_reward_risk` | 2.0 | |
| `preferred_reward_risk` | 3.0 | the ratio at which an `A_PLUS` thesis earns its larger fraction |
| `daily_loss_fraction` | 0.025 | of the session's opening equity; no new entries below it for the session |
| `max_drawdown_fraction` | 0.065 | from the equity peak; the run halts and flattens |
| `max_leverage` | 8.0 | open notional over equity, counting the contracts already open (100 000 USD holds two NQ at 16 400 in total) |
| `max_quantity` | 5 | contracts |
| `order_ttl_bars` | 15 | bars of the entry object's *own scale* an unfilled entry may work (schema 3, 2026-09-20; the order machine converts — 75 1m bars for a 5m object, 225 for a 15m one — and reports the 1m figure as `ttl_bars`) |
| `account_max_age_s` | 120 | staleness of the account snapshot |
| `margin_per_contract` | 20 000 | held per contract; the available funds cap the quantity at it |
| `thesis.max_expressions` | 2 | orders one thesis may place in an episode (the `ThesisBook`) |
| `thesis.stop_cooldown_bars` | 30 | 1m bars every new expression waits after a stop-out |
| `contract` | NQ / CME / USD, point value 20, tick 0.25 | |

The file's sha256 is written to a run's `run.json` (`risk_config_sha256`);
schema 1 is refused. With 100 000 USD the BASE budget (1 500 USD) holds a
75-point stop on one NQ contract; the 0.5 % of schema 1 held 25.

## Tests — `risk/tests/`

`test_plan_contract.py` (signature = direction + the three entity ids, the
thesis fields, round trips, verdict invariants) and `test_gate.py` (grade
sizing and the preferred ratio, the leverage cap, tick rounding, every veto,
positions and the opposite direction, the daily stop by session date, the
drawdown halt, the LONG mirror, a config override).
