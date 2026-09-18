# Execution feedback, week backtest and measurements — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tell the LLM what the executor did with its last opportunity (veto, order, position), count whether that stops the propose → veto loop without anchoring, then measure Brain, controller, gate and executor over a week of real tape at three reasoning efforts and freeze the result as the regression baseline.

**Architecture:** The order machine already knows every veto and every order; it exposes one `execution_view()` through the `PositionLedger` protocol, the Main Brain copies it into `prior_state.execution`, and the prompt says how to read it. Everything else is measurement: timings threaded through the existing objects as an optional `Timings`, a summarizer over the journal, deterministic scenario tests over real bars, and the runner gaining an effort flag.

**Tech Stack:** Python 3.10+, pandas, pytest; DeepSeek chat completions through the existing `DeepSeekClient`; the venv at `.venv`.

**Spec:** `brain/docs/specs/2026-09-17-veto-feedback-week-backtest-design.md`

## Global Constraints

- Tests: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest <paths> -p no:cacheprovider` (never add `-q`: pyproject already sets it).
- No wall clock in `*/core`; the bar's `known_at` is the clock (spec §8).
- No secrets, no absolute paths in code; the DeepSeek key stays in the gitignored key file.
- The LLM reply contract (`LLMUpdate`) does not change; `BrainState`'s schema does not change (old journals must still load).
- Replay of a `sim` journal must reproduce the new input byte for byte: the feedback is derived from the machine the replay re-drives.
- Nothing is committed; no file is deleted without approval except session scratch.
- This session never places an IBKR order.

---

### Task 1: `Timings`

**Files:**
- Create: `shares/core/timing.py`
- Test: `shares/tests/test_timing.py`

**Interfaces:**
- Produces: `class Timings: record(phase: str, seconds: float) -> None; summary() -> dict[str, dict[str, float | int]]; NO_TIMINGS` (a `Timings` subclass whose `record` is a no-op); `@contextmanager timed(timings, phase)`.

- [x] **Step 1: failing test**

```python
from shares.core.timing import NO_TIMINGS, Timings, timed

def test_summary_has_count_total_and_percentiles() -> None:
    t = Timings()
    for s in (0.010, 0.020, 0.030, 0.040):
        t.record("eye", s)
    s = t.summary()["eye"]
    assert s["count"] == 4 and abs(s["total_s"] - 0.1) < 1e-9 and s["max_ms"] == 40.0 and s["p50_ms"] == 25.0

def test_timed_records_and_no_timings_ignores() -> None:
    t = Timings()
    with timed(t, "llm"):
        pass
    assert t.summary()["llm"]["count"] == 1
    with timed(NO_TIMINGS, "llm"):
        pass
    assert NO_TIMINGS.summary() == {}
```

- [x] **Step 2: run, expect ImportError**
- [x] **Step 3: implement** — `Timings` keeps `dict[str, list[float]]`; `summary` uses `statistics`/sorted percentiles (nearest-rank), ms rounded to 3 decimals; `NO_TIMINGS = _NoTimings()` whose `record` does nothing and `summary` returns `{}`; `timed` uses `time.perf_counter` (allowed: the only wall-clock read is a duration).
- [x] **Step 4: run, expect pass**

### Task 2: the machine's memory — `execution_view`, per-episode vetoes, stats, `llm_called`

**Files:**
- Modify: `execution/core/order_fsm.py`, `brain/core/position_ledger.py`
- Test: `execution/tests/test_order_fsm.py`, `brain/tests/test_runtime.py` (ledger view)

**Interfaces:**
- `PositionLedger.execution_view() -> Mapping[str, Any]` — the dict of spec §2; `InMemoryPositionLedger.execution_view()` returns `{"status": "IDLE", "order": None, "position": None, "last_outcome": None, "last_veto": None}`; `IDLE_VIEW` constant exported.
- `OrderMachine.on_bar(asof, bar, plan, *, episode_id, visible, llm_called=False)`.
- `OrderMachine.stats: dict[str, int]`.

- [x] **Step 1: failing tests**

```python
def test_execution_view_follows_the_intent(tmp_path):
    m, broker, journal = machine(tmp_path)
    assert m.ledger.execution_view() == IDLE_VIEW
    m.on_bar(at(1), bar(1, 16370.0, 16380.0), short_plan(), episode_id=EP, visible=lambda a: True)
    view = m.ledger.execution_view()
    assert view["status"] == "WORKING" and view["order"]["entry_object_id"] == "FVG_5m_10" and view["order"]["bars_working"] == 0 and view["order"]["ttl_bars"] == CONFIG.order_ttl_bars
    m.on_bar(at(2), bar(2, 16380.0, 16390.0), short_plan(), episode_id=EP, visible=lambda a: True)
    view = m.ledger.execution_view()
    assert view["status"] == "IN_POSITION" and view["order"] is None and view["position"]["direction"] == "SHORT" and view["position"]["quantity"] == 1
    m.on_bar(at(3), bar(3, 16400.0, 16420.0), None, episode_id=EP, visible=lambda a: True)
    view = m.ledger.execution_view()
    assert view["status"] == "IDLE" and view["last_outcome"] == {"kind": "position_closed", "at": "2022-01-03T14:15:00Z", "reason": None, "exit_role": "stop"}

def test_a_veto_is_counted_per_bar_and_per_llm_proposal(tmp_path):
    m, broker, journal = machine(tmp_path, equity=5_000.0)
    kinds = m.on_bar(at(1), bar(1, 16370.0, 16380.0), short_plan(), episode_id=EP, visible=lambda a: True, llm_called=True)
    assert kinds == ("veto",)
    assert quiet_bars(m, short_plan(), 2, 3) == ()          # TICK bars: no record
    view = m.ledger.execution_view()["last_veto"]
    assert view["vetoes"] == ["position_size"] and view["bars_vetoed"] == 4 and view["proposals_vetoed"] == 1 and view["entry_object_id"] == "FVG_5m_10"
    kinds = m.on_bar(at(5), bar(5, 16370.0, 16380.0), short_plan(), episode_id=EP, visible=lambda a: True, llm_called=True)
    assert kinds == ("veto",) and m.ledger.execution_view()["last_veto"]["proposals_vetoed"] == 2
    assert m.stats["veto_bars"] == 5 and m.stats["vetoes"] == 2

def test_veto_memory_is_per_episode(tmp_path): ...  # same signature in EP2 → recorded again, view reset
```

- [x] **Step 2: run, expect failures on `execution_view` / `llm_called`**
- [x] **Step 3: implement** — `_Intent` gains nothing; the machine gains `_episode: str | None`, `_last_outcome`, `_veto: _VetoMemory | None` (dataclass: plan aliases, direction, vetoes, reasons, reward_risk from the plan geometry, first/last asof, bars, proposals), `_vetoed: set` cleared with the episode, `stats`. `_record` increments `stats[kind]`; the IDLE branch increments `veto_bars`, updates the memory, and journals when the key is new or `llm_called`. Terminal events set `_last_outcome`; a submit clears `_last_outcome` and `_veto`. `execution_view()` assembles the dict; `ExecutionLedger.execution_view()` delegates.
- [x] **Step 4: run the execution suite, expect pass**

### Task 3: `prior_state.execution`, the prompt, `llm_called` through the stack, timings through Brain and runtime

**Files:**
- Modify: `brain/core/main_brain.py`, `brain/core/runtime.py`, `execution/core/stack.py`, `brain/configs/prompts/main_brain_system.md`
- Test: `brain/tests/test_main_brain.py`, `brain/tests/test_runtime.py`, `execution/tests/test_stack_e2e.py`

**Interfaces:**
- `MainBrain(..., timings: Timings = NO_TIMINGS)`; `build_input` writes `prior_state["execution"] = ledger.execution_view()` when `prior` is not None.
- `BrainRuntime(..., timings: Timings = NO_TIMINGS)`; `StepResult.llm_latency_ms: int | None` (last field, default None).
- `TradingStack(runtime, machine, *, tick, timings=NO_TIMINGS)` passes `llm_called=result.llm_called`.

- [x] **Step 1: failing tests**

```python
def test_prior_state_carries_the_execution_view(context):
    ctx, registry = context
    class Engaged(InMemoryPositionLedger):
        def execution_view(self):
            return {**IDLE_VIEW, "status": "WORKING", "order": {"direction": "LONG"}}
    brain = MainBrain(client=ScriptedClient([]), config=CONFIG, ledger=Engaged())
    prior = ...  # a state from a good reply as the other tests build it
    payload = brain.build_input(episode_id="EP_1", context=ctx, trigger_kind="UPDATE", reasons=[], tape=EMPTY_TAPE, prior=prior).to_dict()
    assert payload["prior_state"]["execution"]["status"] == "WORKING"
    wake = brain.build_input(..., prior=None).to_dict()
    assert wake["prior_state"] is None
```

Runtime: `test_step_result_reports_llm_latency` — with `EchoClient` (latency 1 ms) the UPDATE step's `llm_latency_ms == 1` and a TICK's is None; timings summary has `controller`, `input`, `llm`, `reduce`, `journal` keys after a run. Stack e2e: the `veto`/`submitted` path unchanged; add an assertion that the journal's llm_call inputs after the first carry `prior_state.execution.status`.

- [x] **Step 2: run, expect KeyError on `execution`**
- [x] **Step 3: implement** — `_prior_view` adds `payload["execution"] = dict(self._ledger.execution_view())`; MainBrain.step wraps input / call / reduce in `timed`; runtime wraps `decide` and the journal writes, fills `llm_latency_ms` from `step.outcome.reply.latency_ms` (None on incident); stack wraps `plan_from_state` and `machine.on_bar`. Prompt: the section of spec §2 after "Incremental update", plus one hard rule line. `test_config_and_prompt_load` gains `"prior_state.execution" in CONFIG.system_prompt`.
- [x] **Step 4: run brain + execution suites, expect pass** (the prompt sha changes; nothing asserts its value)

### Task 4: the runner — `--reasoning-effort`, `run.json` fields, `drive` timings, pricing file

**Files:**
- Modify: `brain/scripts/run_llm_brain.py`, `brain/scripts/_run_identity.py`
- Create: `brain/configs/llm_pricing.json`
- Test: `brain/tests/test_run_guards.py`

- [x] **Step 1: failing test** — `effective_model_label("deepseek", "deepseek-flash", "max") == "deepseek:deepseek-flash@max"` and `... None) == "deepseek:deepseek-flash"`; `drive(..., timings=t)` on a tiny window records `eye` and `bar` (use the synthetic session bars through `build_eye`? `drive` reads a parquet — test only the label helper and let the echo smoke run prove `timings` lands in `run.json`).
- [x] **Step 2: implement** — `--reasoning-effort {low,high,max}` overrides `config.reasoning_effort` (`dataclasses.replace`), the label enters `run_identity(model=…)`, `run.json` gets `reasoning_effort`, `timings` (summary), `machine_stats`. `drive(..., timings=NO_TIMINGS)` records `eye` around `reader.on_bar` + `observer.observe` and `bar` around the callback. Pricing file: `{"schema_version": 1, "currency": "USD", "per_million_tokens": {"deepseek-flash": {"peak": {"input_cache_hit": 0.006, "input_cache_miss": 0.3, "output": 1.2}, "off_peak": {"input_cache_hit": 0.003, "input_cache_miss": 0.15, "output": 0.6}}}, "peak_hours_utc": "01:00-04:00 and 06:00-10:00 Monday-Friday", "source": "https://api-docs.deepseek.com/quick_start/pricing, read 2026-09-17"}`.
- [x] **Step 3: echo smoke** — `--client echo --broker sim` over 2022-01-03 09:00–10:00, then `replay_journal.py` on it; delete the echo journal.

### Task 5: `summarize_run.py`

**Files:**
- Create: `brain/scripts/summarize_run.py`
- Test: `brain/tests/test_summarize_run.py` (a journal written by the stack e2e fixture on the synthetic tape; the coverage table needs a tape → tested on the synthetic bars passed explicitly)

**Interfaces:**
- `summarize(run_dir: Path, *, pricing: Mapping, bars: Sequence[Bar] | None = None) -> dict` (pure); `render(summaries: Sequence[dict]) -> str`; `main(argv)` with `--run-dir` (repeatable), `--write`, `--pricing`, `--no-coverage`.

- [x] **Step 1: failing tests** — on the e2e journal: `summary["llm"]["calls"]` equals the journal's llm_call count; `summary["orders"]["submitted"] >= 1`; `summary["invariants"]["double_entry"] == 0`; `summary["risk"]["vetoes"]` counts; `summary["account"]["cash"]` equals `run.json`'s or the executor's; cost equals tokens × rates for a hand-built journal with known usage.
- [x] **Step 2: implement** the sections of spec §4. Sharp-move coverage: from the bars, ATR(14) on 1m closes-to-range, 15-bar forward range > 2.5 × ATR, WAKE/UPDATE `known_at` within [−3, +10] minutes; the tape is loaded from `run.json`'s window when `--no-coverage` is absent.
- [x] **Step 3: run, expect pass**

### Task 6: no wall clock in the core

**Files:**
- Create: `shares/tests/test_no_wall_clock.py`

- [x] one test: walk `*/core/*.py`, regex `Timestamp\.now|datetime\.now|utcnow|time\.time\(`, allow-list `brain/core/llm_client.py` and `execution/core/ibkr_broker.py` (`ib_async` waits), assert nothing else matches. Run: expect pass on the current tree (grep already showed only `llm_client.py`).

### Task 7: order scenarios on the real tape

**Files:**
- Create: `execution/tests/test_order_scenarios_real_tape.py`

- [x] fixture `week()` loads `data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet` for 2022-01-03 09:30 → 2022-01-03 16:00 (RTH of the first day; enough touches) via `load_ohlcv` + `iter_completed_bars`; `skipif` without the file. Helpers: `first_touch(bars, start, side, quiet)`: returns `(k, price)` — the limit price `p` such that bars `start+1 … start+quiet` do not reach it and bar `start+quiet+1` does (SELL: `p = max(high[start+1 … start+quiet+1])` only if that max is the last bar's high; scan forward until such a `start` exists). `plan_at(direction, entry, stop_points, target_points, target_id)` builds a `TradePlan` with tick-rounded prices. Nine tests as in spec §6; the cancel-race broker: `class FillsDespiteCancel(SimulatedExecutor)` whose `cancel` records the request but whose `poll` applies the touch first on that one bar (override `poll`: temporarily clear `cancel_requested`, call super, then if the entry is still open re-set it).
- [x] run, expect pass (`< 30 s`).

### Task 8: regression baseline test

**Files:**
- Create: `brain/tests/test_regression_baseline.py`, `brain/docs/evidence/regression_baselines.json` (filled after the week runs)

- [x] `research_orchestration`; for each entry `{run_id, summary}` in the JSON: skip if `outputs/brain_journal/<run_id>` is absent; `replay_run` with the sim broker (as `replay_journal.main` builds it) must be `ok`; `summarize(run_dir, pricing, bars=None)` restricted to the deterministic keys (everything except latency and timings) must equal the committed summary.

### Task 9: the paper exercise script

**Files:**
- Create: `execution/scripts/ibkr_paper_exercise.py`, `execution/tests/test_ibkr_paper_exercise.py`

- [x] `exercise(broker, *, reference_price, marketable, clock, sleep, log) -> dict` drives the steps of spec §9 through the `Broker` protocol only (so `FakeIB`-backed `IBKRBroker` tests it: working → cancelled → working → cancelled, and with `marketable` a fill then a flatten); `main` builds `IBKRBroker.connect`, `require_flat`, refuses without `--i-place-paper-orders`, writes the receipt JSON. The flatten uses a new `IBKRBroker.flatten(symbol, quantity, side, asof)` (market order) — added with its `FakeIB` test.

### Task 10: the week runs, the receipt, the docs

- [x] launch five `nohup` runs (week low / high, day low / high / max — see the spec's §7 schedule) (`--reasoning-effort low|high|max`, `--broker sim`, `--max-llm-calls 4000`) with `-u`, logs under the scratchpad; monitor with `until`-loops on `run.json`'s `finished_at`.
- [x] while they run: Tasks 5–9.
- [x] (partially: the DeepSeek balance ran out; the two complete day runs are the baselines, the week reruns after a top-up) when done: `replay_journal.py` on each; `summarize_run.py --write` on all three; `regression_baselines.json`; the receipt `brain/docs/evidence/2026-09-17_week_backtest_2022-01-03_07.md` with every table of spec §7; decide `reasoning_effort` in `main_brain.json` by the precision-first rule; any evidence-backed fix goes through its own failing test first.
- [x] docs: `brain/docs/README.md` (input, prompt, scripts, tests), `execution/docs/README.md` (machine rules, scripts, tests), both specs (amendment notes), `AGENTS.md`, `shares/docs/current_implementation_status.md`; the memory file.
- [ ] full suite; `git diff --check`; self-review against `shares/docs/self_review_checklist.md`'s causal items and the request's checklist.
