# Execution feedback to the Brain, the week backtest and its measurements — design, 2026-09-17

Follows [2026-09-16-llm-brain-design.md](2026-09-16-llm-brain-design.md) and
[../../execution/docs/specs/2026-09-16-risk-execution-design.md](../../execution/docs/specs/2026-09-16-risk-execution-design.md).
The 2026-09-17 review ([../../execution/docs/evidence/2026-09-17_phase_review.md](../../execution/docs/evidence/2026-09-17_phase_review.md))
left finding M5 open: the Risk gate vetoes an ACTIONABLE opportunity on every
bar, the LLM never learns of it, the opportunity blocks sleep and the idle
rule, and the episode stays awake proposing the same trade. This design
closes M5, measures whether the closure works on a week of real tape, and
fixes the measurements as the regression baseline for the four components
downstream of the Eye.

## 1. Scope

In:

1. `prior_state.execution` — what the executor did with the Brain's last
   opportunity (working order, position, last outcome, last veto), built by
   the Main Brain from the order machine on every UPDATE call, with prompt
   rules on how to use it and how not to be anchored by it.
2. Veto bookkeeping that makes re-proposals countable: a `veto` trade record
   on the first bar a (signature, vetoes) pair is refused *and* on every bar
   on which the LLM re-proposed it after seeing the feedback.
3. `brain/scripts/summarize_run.py` — one summary per journal (LLM calls,
   tokens, cost, decisions, sleeps and wakes, wake coverage of sharp moves,
   opportunities, vetoes and their repeats, orders through their lifecycle,
   duplicate / conflicting order checks, the account, timings), written to
   `summary.json` in the run directory and printed side by side for several
   runs.
4. Per-component timings in `run.json` (Eye, controller, input build, LLM
   call, reduce, journal, plan + gate + machine, per bar end to end).
5. `--reasoning-effort` on the runner, folded into the run identity.
6. The order-scenario tests on the real 2022-01-03 … 01-07 tape
   (`execution/tests/test_order_scenarios_real_tape.py`): the nine
   scenarios the request names, driven by scripted plans whose prices are
   read from the bars, so every fill, cancel, partial and race is
   deterministic and the account is checked after each.
7. The week backtest: `--broker sim` over 2022-01-02 18:00 → 2022-01-07
   17:00 NY (6 900 bars) at three reasoning efforts, its receipt
   (`brain/docs/evidence/2026-09-17_week_backtest_2022-01-03_07.md`), and the
   regression test that replays the baseline journals and compares their
   summaries (`brain/tests/test_regression_baseline.py`, marked
   `research_orchestration`).
8. A guard that the core modules never read the wall clock
   (`shares/tests/test_no_wall_clock.py`): the bar's `known_at` is the only
   clock the Eye, controller, Brain, gate and executor see.
9. `execution/scripts/ibkr_paper_exercise.py` — the paper-session exercise
   (working → cancel → replace, optionally a marketable fill and flatten)
   the user runs against TWS; this session does not place orders.

Out: no change to the Eye, to the controller's wake set, to the reducer's
sleep rules, to the gate's checks or to the executor's matching rules unless
the week's evidence shows a defect; no in-place order modification; no
recovery of orders after a restart (unchanged from the review).

## 2. `prior_state.execution`

Built in `MainBrain.build_input` from `ledger.execution_view()` (a new
`PositionLedger` method; `InMemoryPositionLedger` returns the IDLE view,
`ExecutionLedger` reads the machine). Present on every call whose
`prior_state` is not null; a WAKE call starts an episode and the machine's
per-episode memory (last outcome, last veto) is empty for it.

```json
"execution": {
  "status": "IDLE | WORKING | PARTIAL | IN_POSITION",
  "order": null | {
    "direction": "SHORT", "entry_object_id": "FVG_5m_10",
    "invalidation_object_id": "BSL_5m_3", "target_object_id": "SSL_4H_2",
    "submitted_at": "2022-01-03T15:31:00Z", "bars_working": 4,
    "quantity": 1, "filled_quantity": 0, "ttl_bars": 15
  },
  "position": null | {
    "direction": "SHORT", "entry_object_id": "FVG_5m_10",
    "opened_at": "2022-01-03T15:35:00Z", "quantity": 1
  },
  "last_outcome": null | {
    "kind": "expired | cancelled | rejected | position_closed",
    "at": "2022-01-03T15:46:00Z", "reason": "ttl | plan_dropped | signature_changed | entry_object_not_visible | null",
    "exit_role": "stop | target | null"
  },
  "last_veto": null | {
    "direction": "SHORT", "entry_object_id": "FVG_5m_10",
    "invalidation_object_id": "BSL_5m_3", "target_object_id": "SSL_4H_2",
    "vetoes": ["reward_risk"], "reasons": ["reward-to-risk 1.12 < 1.5"],
    "reward_risk": 1.12,
    "first_vetoed_at": "2022-01-03T15:31:00Z", "last_vetoed_at": "2022-01-03T15:40:00Z",
    "bars_vetoed": 10, "proposals_vetoed": 2
  }
}
```

- `status`, `order`, `position` are the machine's current state; `order`
  and `position` name the plan's aliases (the LLM's vocabulary), never
  prices. `bars_working` counts the bars since submission against
  `ttl_bars`, so the LLM knows an unreached entry is about to expire.
- `last_outcome` is the last terminal event of this episode's intent
  (cleared when a new intent is submitted).
- `last_veto` is the last refused plan of this episode: cleared when a plan
  passes the gate, never when the LLM merely stops proposing it —
  `last_vetoed_at` against `known_at` tells the LLM how old it is.
  `bars_vetoed` counts bars on which the gate refused that signature;
  `proposals_vetoed` counts the LLM calls after which it was refused again
  (the first proposal included).
- Everything resets when the episode changes (`on_bar` compares the
  `episode_id`), so nothing leaks across episodes. A position or a working
  order never crosses an episode boundary: while engaged the Brain cannot
  sleep.

The reply contract does not change. The prompt gains a section:

> `prior_state.execution` is what the executor did with your last
> opportunity. `last_veto`: the Risk gate refused it; `reasons` names the
> failing check. A veto is a statement about the geometry and the account,
> not about the market: it does not favour a direction and it never binds
> an opportunity built on different objects. While its cause stands, do not
> propose the same three objects again as ACTIONABLE — for `reward_risk`
> the target is too near or the invalidation too far for the entry you
> chose; for `position_size` the invalidation is too far for the account's
> risk budget; for `exposure` / `working_order` a trade is already on and
> there is nothing to add. When evidence changes your reading, re-evaluate
> freely. `order`: your opportunity is at the broker as a limit order;
> keeping the same three objects keeps it working, changing any of them or
> downgrading the state cancels it (a new ACTIONABLE submits a new one).
> `position`: you are in the trade; the stop and target are at the broker
> and code exits there; track (step 14) and keep the opportunity that
> describes the trade.

`RecordedClient` replay: a `sim` journal replays with the machine, so the
feedback reproduces byte for byte. An `ibkr` journal replays its Brain path
only, as before; a call whose input carried a working order, a position or a
veto will not reproduce there (documented, unchanged decision).

## 3. Veto bookkeeping

`OrderMachine.on_bar` gains `llm_called: bool = False` (the stack passes
`StepResult.llm_called`). A refused plan writes a `veto` record when the
(episode, signature, vetoes) triple is new **or** when `llm_called` is true —
one record per LLM proposal that ended in a veto, none for the TICK bars in
between. `_vetoed` is keyed per episode (today it is never cleared, so a
later episode's identical veto is silent). The machine keeps `stats`
keyed by record kind (`veto`, `submitted`, `working`, `partial`, `filled`,
`position_opened`, `position_closed`, `cancel_requested`, `cancelled`,
`expired`, `rejected`, `exit_leg_lost`) plus `veto_bars`, for `run.json`.

Found while writing the scenario tests (§6) and fixed with its own failing
test: after a stop or target closed the position, an unchanged ACTIONABLE
plan was resubmitted on the exit bar — a re-entry the Brain never decided.
A closed position now blocks its `(episode, signature)` like an expiry or a
reject; the prompt says so under `last_outcome`. The second launch then
showed a different SHORT, proposed while the position was open, entered on
the exit bar one minute after a stop-out: after a close the machine now
waits for the Brain's next call (`llm_called`) before submitting anything,
so every order rests on a decision made with the outcome in view.

Metrics the summary derives from the journal:

| metric | definition |
| --- | --- |
| `vetoes` | `veto` records |
| `veto_signatures` | distinct (episode, signature) among them |
| `reproposals_after_veto` | `veto` records for a signature already vetoed in the episode — the LLM saw `last_veto` and proposed the same three objects again |
| `veto_repeat_rate` | `reproposals_after_veto / vetoes` |
| `calls_with_veto_feedback` | LLM calls whose input carried `last_veto` |
| `reanalysed_after_veto` | of those, accepted replies whose opportunity differs from the vetoed one (state, direction or an object) |
| `kept_after_veto` | of those, accepted replies that kept it unchanged |
| `veto_bars` | bars on which the gate refused (from `run.json`) |
| `awake_bars_with_vetoed_opportunity` | bars in ACTIVE with an ACTIONABLE opportunity whose plan is being vetoed — the cost M5 measures |
| `cost_usd_after_veto` | the cost of `calls_with_veto_feedback` |

The mechanism is judged to work when, on the same tape,
`veto_repeat_rate` and `awake_bars_with_vetoed_opportunity` fall against the
run without feedback (run 3 had no vetoes at all — its opportunities died in
geometry — so the reference is the week's own distribution: a repeat rate
near zero while `reanalysed_after_veto` shows fresh opportunities appearing
after market changes means "no loop, no anchoring").

## 4. Summary and regression

`brain/scripts/summarize_run.py --run-dir <run> [--run-dir <run> …]
[--write]` reads the journals and prints one column per run:

- run: id, window, model, effort, prompt / controller / risk shas, minutes;
- LLM: calls, incidents, repairs, prompt tokens (hit / miss), completion
  tokens, cost in USD at DeepSeek's published peak and off-peak rates
  (`brain/configs/llm_pricing.json`), latency p50 / p95 / max, input
  characters median;
- controller: WAKE / UPDATE / TICK / STAY_ASLEEP counts, episodes, sleeps by
  reason (`continue_active=false`, `idle`), median awake bars, calls per
  awake bar, trigger reasons by kind (what re-analyses cost);
- coverage: sharp moves (15-bar range > 2.5 × ATR(14), the audit's rule)
  and how many had a WAKE or UPDATE within [−3, +10] minutes — the
  "missed market while asleep" measure, computed from the tape the run
  names;
- Brain: opportunities proposed by state, survived geometry, confidence
  counts, understanding changes;
- risk and orders: the veto metrics of §3, vetoes by code, submitted /
  working / partial / filled / cancelled / expired / rejected /
  position_opened / position_closed / exit_leg_lost counts, cancel reasons,
  bars from submission to fill, `account_asof == known_at` on every gate
  decision (stale-snapshot count), and the invariants: never two open
  entries, never a `submitted` while an intent is open, never a position
  without a fill;
- account: final cash, positions, realized PnL, unrealized PnL at the last
  close of the window, fills;
- timings from `run.json`.

`--write` saves `summary.json` beside `run.json`. The receipt document
copies the tables. `brain/tests/test_regression_baseline.py`
(`research_orchestration`) reads `brain/docs/evidence/regression_baselines.json`
— run ids and the committed summaries — replays each run that exists under
`outputs/brain_journal/` with `replay_run` and requires the replay to
reproduce and the fresh summary to equal the committed one on every
deterministic field. A change to the reducer, the controller, the gate or
the executor that alters a decision fails it; a change to the summarizer
shows up as a summary diff.

## 5. Timings

`shares/core/timing.py`: `Timings.record(phase, seconds)` and `summary()`
(count, total, mean, p50, p95, max per phase). Phases: `eye`
(`reader.on_bar` + `observer.observe`, in `drive`), `bar` (the whole
callback, in `drive`), `controller`, `input`, `llm` (the call under the
retry policy, wall time), `reduce`, `journal` (runtime), `plan`,
`machine` (stack; gate and broker inside). A `Timings` is optional
everywhere and defaults to a no-op, so tests and replay are untouched.
`run.json` gains `timings`; the summary prints them, and the receipt names
the dominant term.

## 6. Order scenarios on the real tape

`execution/tests/test_order_scenarios_real_tape.py` loads the week's bars
once (`load_ohlcv` + `iter_completed_bars`, skipped without the tape) and
drives `OrderMachine` + `RiskGate` + `SimulatedExecutor` with scripted
`TradePlan`s whose limit sits on a price the tape reaches on a known later
bar (`SELL LIMIT` = the first bar whose high reaches it after k quiet bars;
`BUY LIMIT` mirrored). Each scenario asserts the trade records, the machine
state and the virtual account (cash, positions, pending / filled /
cancelled / rejected counts, margin held):

| scenario | driver | expected |
| --- | --- | --- |
| normal fill | plan held; tape touches the limit on bar k | submitted → working → filled at the limit; one position; cash unchanged until the exit; exit at stop or target moves cash by the tick-exact PnL; no pending orders |
| wait → cancel | plan dropped before the touch | cancel_requested → cancelled; flat; margin released |
| wait → cancel → resubmit | signature changes before the touch | cancelled then submitted next bar; never two open entries |
| wait → reanalyse → no new order | plan None (cancel) and plan unchanged (keep) | cancelled without a new order; unchanged plan keeps the same order id |
| partial | `max_fill_per_bar` 1, quantity 2 | Submitted → Partially Filled → Filled; then cancel after the partial keeps a position of 1 with exits of 1; the account's position equals the filled quantity throughout |
| risk reject | RR below the floor | `veto`; the broker holds no order |
| funds / position limits | margin above available funds; a second plan while in position; quantity above `max_quantity` | `rejected` with no position; `exposure` veto; quantity capped |
| duplicate signal | the same plan on every bar while working | one order id, one `submitted` |
| cancel race | a broker that fills the entry on the poll that carries the cancel | the fill wins: IN_POSITION, position and cash consistent, no pending entry, exits working |

## 7. The week backtest

Window: warm-up 2021-12-27, emit 2022-01-02 18:00 → 2022-01-07 17:00 NY
(6 900 bars: 360 on Sunday evening, 1 380 on each of Monday to Thursday,
1 020 on Friday). `--broker sim`, the default risk and simulator configs,
`--max-llm-calls 4000` as the spend guard.

A probe of one journaled input (run 3, call 4) before scheduling: `low`
13.8 s and 2.5 k completion tokens, `high` 24.8 s and 5.2 k, `max` 216 s
and 44 k (42.7 k of them reasoning) — `max` also overruns the 32 768
`max_tokens` of `main_brain.json` and returns an empty reply, so it runs
from `brain/configs/main_brain_max.json` (65 536 tokens, 900 s timeout).
A week at `max` would take about 100 hours; the schedule is therefore:

| run | window | effort |
| --- | --- | --- |
| week | 2022-01-02 18:00 → 01-07 17:00 | `low` |
| week | same | `high` (DeepSeek's default: what `main_brain.json` sends when unset) |
| day | 2022-01-03 09:00 → 17:00 NY (the RTH session, 480 bars, `--max-llm-calls 400`) | `low`, `high`, `max` |

The three day runs start from the same warm-up and the same SLEEP state,
so they are the like-for-like comparison of the efforts; the two week runs
carry the veto mechanism, the order lifecycle and the regression baseline.
All five run concurrently (the Eye costs about a minute per 2 000 bars;
the rest is waiting on DeepSeek). `main_brain.json`'s `timeout_s` was
raised from 120 to 300 so a long `high` reply is not an incident.

What the receipt answers, run by run and side by side:

1. the veto metrics of §3 and whether the feedback stops the loop without
   anchoring;
2. sleep / wake counts and the coverage of sharp moves; what triggered each
   LLM call (wake kinds, update kinds, relation changes) and which calls
   changed nothing;
3. low / high / max: trades proposed and taken, sharp moves recognised,
   confidence, calls, tokens, cost, latency — the precision-versus-cost
   verdict, decided by the rule the request states (precision first, cost
   only when precision holds);
4. the order lifecycle as it happened: submitted / filled / cancelled /
   expired / rejected, partials, replacements, duplicates (must be zero),
   FSM anomalies (must be zero), stale snapshots (must be zero);
5. final cash, positions, realized and unrealized PnL;
6. timings per component and end to end, and where the latency is.

`main_brain.json` keeps whichever effort the verdict chooses.

## 8. The clock

`brain/scripts/_run_identity.drive` feeds the Eye one completed bar at a
time; every downstream component receives `observation.asof` (the bar's
`known_at`) and nothing else: the controller sees `events_this_update`, the
Brain's input is asserted causal (`assert_causal`), the gate's `asof` and
the executor's `poll(asof, bar)` are the same bar, and fills use only that
bar's high and low. `shares/tests/test_no_wall_clock.py` scans
`eyes/core`, `brain/core`, `execution/core`, `risk/core` and `shares/core`
for `Timestamp.now`, `datetime.now`, `utcnow` and `time.time` and allows
only the two places that measure latency (`llm_client.py`) and the IBKR
adapter's socket wait. `run.json`'s `started_at` / `finished_at` are
metadata written by the script, never read by a component.

## 9. IBKR paper exercise

`execution/scripts/ibkr_paper_exercise.py --reference-price <last NQ
price> --i-place-paper-orders` connects with `IBKRBroker`, refuses a
non-flat account, then: submits a SELL bracket 1 % above the reference
(unmarketable) and polls until `working`; cancels it and polls until
`cancelled`; submits again 1.5 % above and cancels (the machine's
"replace"); with `--marketable` also submits a SELL bracket 0.5 % below the
reference (fills), polls until `filled`, then cancels its stop and target
and flattens with a market BUY. Every `BrokerEvent` and snapshot is written
to `outputs/ibkr_paper/<timestamp>.json`. The script's driver is unit-tested
against `FakeIB`; this session never runs it against TWS.

## 10. Files

| file | change |
| --- | --- |
| `brain/core/position_ledger.py` | `PositionLedger.execution_view()`; `InMemoryPositionLedger` returns the IDLE view |
| `execution/core/order_fsm.py` | per-episode veto memory and `stats`, `execution_view()`, `on_bar(..., llm_called)` |
| `execution/core/stack.py` | passes `llm_called`; timings |
| `brain/core/main_brain.py` | `prior_state.execution`; timings |
| `brain/core/runtime.py` | `StepResult.llm_latency_ms`; timings |
| `brain/configs/prompts/main_brain_system.md` | the execution-feedback section |
| `brain/configs/llm_pricing.json` | DeepSeek's published rates |
| `brain/scripts/run_llm_brain.py` | `--reasoning-effort`; `timings`, `machine_stats`, `reasoning_effort` in `run.json` |
| `brain/scripts/_run_identity.py` | `drive(..., timings)` |
| `brain/scripts/summarize_run.py` | new |
| `shares/core/timing.py` | new |
| `shares/tests/test_no_wall_clock.py` | new |
| `execution/tests/test_order_scenarios_real_tape.py` | new |
| `brain/tests/test_regression_baseline.py`, `brain/docs/evidence/regression_baselines.json` | new |
| `execution/scripts/ibkr_paper_exercise.py`, `execution/tests/test_ibkr_paper_exercise.py` | new |
| `brain/docs/evidence/2026-09-17_week_backtest_2022-01-03_07.md` | the receipt |
| READMEs, `AGENTS.md`, `shares/docs/current_implementation_status.md`, both earlier specs | updated |

Approval gate: this session runs autonomously (the request is explicit down
to the scenarios and the window); the decisions above are stated so they can
be reversed, and the final report lists them.
