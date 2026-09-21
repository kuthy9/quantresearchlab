# Entry model implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate the bias (which side) from the entry (which price): a resting limit at a retracement object that code cannot turn into a chase, a wait the order can afford, and feedback the Brain can read.

**Architecture:** Geometry rules in `brain/core/opportunity_geometry.py` (inside-zone midpoint, no range entry, the side rule in `coherence_error`), the reducer's rejections carried in `LastUpdate.rejections` into `prior_state`, a scale-aware TTL and an `entry_marketable` refusal in the order machine, replacement refunds in the thesis book, the prompt's Expression section, `entry_quality` in the summarizer, and a benchmark runner over ten windows.

**Tech Stack:** Python 3.12, pytest, pandas; DeepSeek via `run_llm_brain`.

**Spec:** [brain/docs/specs/2026-09-20-entry-model-design.md](../specs/2026-09-20-entry-model-design.md)

## Global Constraints

- Test command: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest <paths> -p no:cacheprovider` (never `-q`).
- No threshold is added; `order_ttl_bars` keeps its value (15) and changes its unit.
- Nothing is committed until asked; nothing under `outputs/` is deleted.
- The DeepSeek key stays in the gitignored key file; runs launch from scratchpad scripts.

---

### Task 1: Geometry — inside a zone the midpoint, no range entry, the side rule at proposal time

**Files:**
- Modify: `brain/core/opportunity_geometry.py` (`_entry`, `resolve_geometry`, `coherence_error`, new `entry_side_error`)
- Test: `brain/tests/test_opportunity_geometry.py`

**Interfaces:**
- Produces: `entry_side_error(direction: TradeDirection, entry: float, close: float) -> str | None` (None when the limit rests: LONG entry ≤ close, SHORT entry ≥ close); `coherence_error` returns it after the geometry resolves; `resolve_geometry` raises `GeometryError` for a `range` entry; rule ids `entry.zone.inside_midpoint`, `entry.zone.inside_far_edge`.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_zone_that_contains_price_is_entered_at_its_midpoint_or_far_edge() -> None:
    # FVG_5m_3 is 100–102, midpoint 101
    long_inside = Opportunity(OpportunityState.ACTIONABLE, TradeDirection.LONG, "FVG_5m_3", "SSL_5m_2", "BSL_1H_1")
    g = resolve_geometry(long_inside, OBJ, close=101.5, tick=0.25)
    assert g.entry_price == 101.0 and g.rule_ids[0] == "entry.zone.inside_midpoint"
    g = resolve_geometry(long_inside, OBJ, close=100.5, tick=0.25)  # price already under the midpoint
    assert g.entry_price == 100.0 and g.rule_ids[0] == "entry.zone.inside_far_edge"
    short_inside = Opportunity(OpportunityState.ACTIONABLE, TradeDirection.SHORT, "OB_15m_1", "BSL_1H_1", "SSL_5m_2")  # 104–106
    g = resolve_geometry(short_inside, OBJ, close=104.5, tick=0.25)
    assert g.entry_price == 105.0 and g.rule_ids[0] == "entry.zone.inside_midpoint"
    g = resolve_geometry(short_inside, OBJ, close=105.5, tick=0.25)
    assert g.entry_price == 106.0 and g.rule_ids[0] == "entry.zone.inside_far_edge"


def test_a_range_is_not_an_entry_object() -> None:
    with pytest.raises(GeometryError, match="range is not an entry"):
        resolve_geometry(Opportunity(OpportunityState.DEVELOPING, TradeDirection.LONG, "DR_15m_1", "SWING_L_5m_1", "BSL_1H_1"), OBJ, close=103.0, tick=0.25)


def test_an_entry_the_market_is_past_is_incoherent_at_proposal_but_still_resolves() -> None:
    long_above = Opportunity(OpportunityState.ACTIONABLE, TradeDirection.LONG, "OB_15m_1", "SSL_5m_2", "BSL_1H_1")  # OB 104–106 above a 103 close
    assert resolve_geometry(long_above, OBJ, close=103.0, tick=0.25).entry_price == 106.0
    reason = coherence_error(long_above, OBJ, close=103.0, tick=0.25)
    assert reason is not None and "above price" in reason and "pullback" in reason
    assert coherence_error(long_above, OBJ, close=106.0, tick=0.25) is None
    assert entry_side_error(TradeDirection.SHORT, 98.0, 103.0) is not None and entry_side_error(TradeDirection.SHORT, 103.0, 103.0) is None
```

Update `test_swing_stop_and_range_entry` (a range entry now raises) to use `FVG_5m_3` as the entry, keeping the swing stop assertion.

- [ ] **Step 2: Run to verify failure** — `pytest brain/tests/test_opportunity_geometry.py`: three fail (attribute / no raise / wrong price).
- [ ] **Step 3: Implement** `_entry(obj, direction, close)`: zone containing price → midpoint if on the resting side of the close else the far edge; range → `GeometryError`; `entry_side_error`; `coherence_error` calls it on the resolved entry.
- [ ] **Step 4: Run to verify pass**; run `brain/tests/test_reducer.py execution/tests/test_plan_from_state.py` too.

### Task 2: Feedback — `LastUpdate.rejections` into `prior_state`

**Files:**
- Modify: `contract/brain/state.py` (`LastUpdate`), `brain/core/reducer.py` (three `LastUpdate(...)` sites), `contract/brain/llm.py` (`LLM_INPUT_SCHEMA_VERSION = 3`)
- Test: `brain/tests/test_brain_state.py`, `brain/tests/test_reducer.py`, `brain/tests/test_main_brain.py`

- [ ] **Step 1: Failing tests**

```python
# test_brain_state.py
def test_last_update_carries_the_rejections_and_old_journals_read_as_none() -> None:
    state = make_state(last_update=LastUpdate(T1, True, {}, None, rejections=("opportunity_incoherent:x",)))
    payload = state.to_dict()
    assert payload["last_update"]["rejections"] == ["opportunity_incoherent:x"]
    assert BrainState.from_dict(payload).last_update.rejections == ("opportunity_incoherent:x",)
    del payload["last_update"]["rejections"]
    assert BrainState.from_dict(payload).last_update.rejections == ()

# test_reducer.py
def test_the_state_records_why_the_opportunity_was_dropped() -> None:
    prev = quiet()
    bad = Opportunity(OpportunityState.ACTIONABLE, TradeDirection.LONG, "FVG_5m_3", "SSL_5m_2", "BSL_1H_1", thesis_id="T1", governing_timeframe="5m")
    r = apply(prev, episode_id=prev.episode_id, evidence=[], update=upd(opportunity=bad), ctx=ctx(coherence=lambda o: "LONG entry 105 lies above price 103"))
    assert r.state.last_update.rejections == ("opportunity_incoherent:LONG entry 105 lies above price 103",) == r.rejections

# test_main_brain.py (the prior view test)
    assert view["last_update"]["rejections"] == [...]  and LLM_INPUT_SCHEMA_VERSION == 3
```

- [ ] **Step 2: Verify failure** (`TypeError: unexpected keyword rejections`).
- [ ] **Step 3: Implement** the field (`tuple[str, ...] = ()`, `_texts` validated, in `to_dict`/`from_dict` with `payload.get("rejections", ())`), pass `tuple(rejections)` at the reducer's final `LastUpdate` (the update path) and `()` on the carry-forward / empty paths, bump the input schema.
- [ ] **Step 4: Verify pass**; `pytest brain/tests -p no:cacheprovider` for the fallout (journal fixtures, replay).

### Task 3: The wait — scale TTL, `entry_marketable`, replacement refunds

**Files:**
- Modify: `risk/configs/risk.json` (`schema_version` 3), `risk/core/gate.py` (`RISK_SCHEMA_VERSION`), `execution/core/order_fsm.py`, `execution/core/thesis.py`
- Test: `risk/tests/test_gate.py`, `execution/tests/test_order_fsm.py`, `execution/tests/test_thesis.py`

- [ ] **Step 1: Failing tests**

```python
# test_order_fsm.py — the plan's entry is a 5m object: 15 × 5 = 75 bars
def test_the_ttl_counts_bars_of_the_entry_objects_scale(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    kinds = quiet_bars(m, short_plan(), 1, CONFIG.order_ttl_bars + 2)
    assert "expired" not in kinds and m.state is MachineState.WORKING
    assert m.execution_view()["order"]["ttl_bars"] == CONFIG.order_ttl_bars * 5
    kinds = quiet_bars(m, short_plan(), CONFIG.order_ttl_bars + 3, CONFIG.order_ttl_bars * 5)
    assert "expired" in kinds and m.state is MachineState.IDLE


def test_a_marketable_entry_is_refused_not_chased(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    chase = short_plan(entry=16387.5, close=16380.0)  # a SELL limit under the market
    kinds = m.on_bar(at(1), bar(1, 16375.0, 16385.0), chase, episode_id=EP, visible=lambda a: True, llm_called=True)
    assert kinds == ("thesis_refused",) and last_trade(tmp_path, "thesis_refused")["reason"] == "entry_marketable"
    assert not m.ledger.has_working_order()

# test_thesis.py
def test_a_replacement_gives_the_expression_back_but_a_dropped_plan_does_not() -> None:
    b = book()
    for reason, left in (("signature_changed", 0), ("entry_object_not_visible", 0), ("plan_dropped", 1)):
        p = plan(target_id=f"swing:{reason}")
        assert b.admit(p, 0) is None
        b.expressed(p)
        b.outcome(p, "cancelled", exit_role=None, bar_index=1, reason=reason)
        assert b.view(1)["theses"][0]["expressions"] == left
```

`short_plan` needs a `close` override (it already forwards `**fields`; make `close` explicit).

- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement**: `_ttl_bars(plan)` = `order_ttl_bars × TIMEFRAME_MINUTES[plan.entry.timeframe]` (unknown scale → 1m) used at the TTL check and in the view; `entry_side_error(plan.direction, verdict-independent geometry.entry_price, plan.close)` before `admit` → `_refuse(plan, "entry_marketable", ...)`; `ThesisBook.outcome(..., reason=None)` refunds on `expired` and on cancels with reason in `("signature_changed", "entry_object_not_visible")`; the FSM passes `intent.cancel_reason`; `REFUSALS` gains `entry_marketable`; risk schema 3 with a config test.
- [ ] **Step 4: Verify pass**; `pytest execution/tests risk/tests brain/tests/test_summarize_run.py`.

### Task 4: The prompt

**Files:**
- Modify: `brain/configs/prompts/main_brain_system.md`
- Test: `brain/tests/test_main_brain.py`

- [ ] **Step 1: Failing test**

```python
def test_the_prompt_separates_the_bias_from_the_entry() -> None:
    text = CONFIG.system_prompt
    for word in ("## Expression", "the pullback picks the price", "Nearest first", "Follow the leg", "last_update.rejections", "ttl_bars", "midpoint", "not an entry"):
        assert word in text, word
    assert "the order fills now" not in text and "entry.range" not in text
```

- [ ] **Step 2: Verify failure**; **Step 3:** write the section (spec §1.4) and rewrite the geometry hard rule and the `order` bullet; **Step 4:** verify pass.

### Task 5: Metrics — `entry_quality` and the ACTIONABLE accuracy

**Files:**
- Modify: `brain/scripts/summarize_run.py`
- Test: `brain/tests/test_summarize_run.py`

- [ ] **Step 1: Failing test**

```python
def test_entry_quality_reads_location_wait_and_excursions_from_the_tape() -> None:
    start = pd.Timestamp("2022-01-03T14:00:00Z")
    bars = [Bar(start=start + pd.Timedelta(minutes=i), open=16400.0 + i, high=16401.0 + i, low=16399.0 + i, close=16400.5 + i, volume=1.0, symbol="NQ", instrument_id=1) for i in range(400)]
    fills = [{"direction": "LONG", "fill_price": 16460.0, "limit_price": 16460.0, "stop_price": 16450.0, "submitted_at": start + pd.Timedelta(minutes=55), "filled_at": start + pd.Timedelta(minutes=60)}]
    q = entry_quality(fills, bars, submitted=2)
    assert q["fills"] == 1 and q["fill_rate"] == 0.5 and q["median_wait_minutes"] == 5.0
    assert q["median_location_60m"] == pytest.approx(1.0, abs=0.05) and q["chased"] == 1  # bought the top of a rising hour
    assert q["right_60m"] == 1 and q["median_mfe_r"] == pytest.approx(6.1, abs=0.2) and q["median_mae_r"] == pytest.approx(0.1, abs=0.1)
```

- [ ] **Step 2–4:** implement `entry_quality(fills, bars, *, submitted)`; wire fills from `filled` records (with the submission's verdict), `orders.entry_quality` (None without bars), `brain.actionable_direction_accuracy_60m` from ACTIONABLE replies; verify.

### Task 6: The benchmark runner

**Files:**
- Create: `brain/configs/benchmark_windows.json`, `brain/scripts/run_benchmark.py`
- Test: `brain/tests/test_run_benchmark.py`

- [ ] **Step 1: Failing test**

```python
def test_the_windows_file_builds_one_command_per_window_with_the_warmup() -> None:
    windows = load_windows(ROOT / "brain" / "configs" / "benchmark_windows.json")
    assert len(windows["windows"]) == 10 and windows["warmup_days"] == 7
    commands = build_commands(windows, client="deepseek", broker="sim", reasoning_effort="high", label="entry-model", max_llm_calls=400)
    first = commands[0]
    assert first[:3] == [sys.executable, "-u", "-m"] and "brain.scripts.run_llm_brain" in first
    assert first[first.index("--warmup-start") + 1] == "2022-01-17" and first[first.index("--emit-start") + 1] == "2022-01-24T12:00"
```

- [ ] **Step 2–4:** implement `load_windows`, `build_commands`, `main` (`--dry-run`, `--parallel`, `--only`), logs under `outputs/brain_journal/benchmark_logs/`; verify.

### Task 7: Docs, then the runs

- [ ] Update `brain/docs/README.md` (prompt row, reducer/feedback, summarizer, benchmark, receipt index), `execution/docs/README.md` (TTL row, thesis row, refusal), `risk/docs/README.md` (`order_ttl_bars` unit, schema 3), the two design specs' amendment notes.
- [ ] Full suite green: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest eyes brain contract shares risk execution -p no:cacheprovider`.
- [ ] Frozen window run (label `entry-model`), summarize against run X, receipt `brain/docs/evidence/2026-09-20_entry_model_frozen_window_2022-01-03.md`.
- [ ] Benchmark: `run_benchmark.py --parallel 3`, summarize, receipt `brain/docs/evidence/2026-09-20_entry_model_benchmark_2022.md`.
- [ ] Re-freeze `regression_baselines.json`; research test green.
