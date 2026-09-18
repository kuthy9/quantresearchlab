# Thesis lifecycle and Risk v2 — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the Brain names a thesis with a scale, a grade and an invalidation type; the executor holds up to three same-direction positions, refuses re-expressions of a closed thesis, exits close-beyond theses at market, and the gate sizes by grade, caps leverage, stops the day at −2.5 % and halts the run at −6.5 % from the peak.

**Architecture:** contract fields first (they gate everything), then geometry and reducer rules, then the gate, the simulator, the `ThesisBook`, the multi-intent machine, the stack / runner wiring, the controller filter, the prompt and view, the summary and baselines, the docs.

**Tech Stack:** Python 3.12, pandas, pytest; no new dependencies.

**Spec:** `execution/docs/specs/2026-09-17-thesis-lifecycle-risk-v2-design.md`

## Global Constraints

- Test command: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest <paths> -p no:cacheprovider` (never `-q`).
- No secrets, no absolute paths, no placeholders; defaults are named constants or config fields.
- `live_execution_allowed` stays false; nothing in the tests talks to TWS.
- Every rule the spec states has a test that fails before the code exists.
- Schema versions: `risk.json` → 2, `sleep_controller.json` → 3; the old versions are refused.

---

### Task 1: Contract fields

**Files:**
- Modify: `contract/brain/state.py` (`Opportunity`, new enums `ThesisGrade`, `InvalidationMode`, `GOVERNING_TIMEFRAMES`)
- Modify: `contract/brain/llm.py` (`_OPPORTUNITY_KEYS`, `LLM_UPDATE_EXAMPLE`, `parse_update`)
- Modify: `contract/risk/plan.py` (`TradePlan` fields), `contract/risk/assessment.py` (`VetoCode`), `contract/execution/account.py` (`OrderRole.FLATTEN`)
- Test: `brain/tests/test_llm_contract.py`, `brain/tests/test_brain_state.py`, `risk/tests/test_plan_contract.py`

**Interfaces:**
- Produces: `Opportunity(state, direction, entry_object_id, invalidation_object_id, target_object_id, thesis_id=None, governing_timeframe=None, grade=ThesisGrade.BASE, invalidation_mode=InvalidationMode.TOUCH)`; `ThesisGrade.BASE|A_PLUS`; `InvalidationMode.TOUCH|CLOSE_BEYOND`; `GOVERNING_TIMEFRAMES = ("4H", "1H", "15m", "5m")`; `THESIS_ID_PATTERN`; `TradePlan(..., thesis_id: str, governing_timeframe: str, grade: ThesisGrade, invalidation_mode: InvalidationMode)`; `VetoCode.DAILY_STOP|HALTED|LEVERAGE`; `OrderRole.FLATTEN`; `RiskVerdict(..., grade_applied: str | None = None, risk_fraction: float | None = None)`.

- [ ] **Step 1: failing tests**

```python
# brain/tests/test_llm_contract.py — replace the ACTIONABLE cases' dicts with the helper and add
def _actionable(**over):
    payload = {"state": "ACTIONABLE", "direction": "LONG", "entry_object_id": "FVG_5m_3", "invalidation_object_id": "SSL_5m_2",
               "target_object_id": "BSL_1H_1", "thesis_id": "T1", "governing_timeframe": "15m", "grade": "BASE", "invalidation_mode": "TOUCH"}
    payload.update(over)
    return payload

def test_actionable_opportunity_carries_thesis_scale_grade_and_mode() -> None:
    update = parse_update(_reply(opportunity=_actionable(grade="A_PLUS", invalidation_mode="CLOSE_BEYOND")), evidence_ids=EVIDENCE, object_ids=OBJECTS)
    o = update.opportunity
    assert o.thesis_id == "T1" and o.governing_timeframe == "15m" and o.grade is ThesisGrade.A_PLUS and o.invalidation_mode is InvalidationMode.CLOSE_BEYOND

@pytest.mark.parametrize("bad", [
    _reply(opportunity=_actionable(thesis_id=None)), _reply(opportunity=_actionable(thesis_id="has space")),
    _reply(opportunity=_actionable(governing_timeframe="1m")), _reply(opportunity=_actionable(grade="A+")),
    _reply(opportunity=_actionable(invalidation_mode="CLOSE")),
    _reply(opportunity={**_actionable(), "state": "NONE", "direction": None, "entry_object_id": None, "invalidation_object_id": None, "target_object_id": None}),  # NONE with a thesis id
    _reply(opportunity={k: v for k, v in _actionable().items() if k != "thesis_id"}),
])
def test_thesis_fields_are_validated(bad: str) -> None:
    with pytest.raises(MalformedReply):
        parse_update(bad, evidence_ids=EVIDENCE, object_ids=OBJECTS)

def test_example_none_opportunity_carries_null_thesis_fields() -> None:
    assert LLM_UPDATE_EXAMPLE["opportunity"] == {"state": "NONE", "direction": None, "entry_object_id": None, "invalidation_object_id": None,
        "target_object_id": None, "thesis_id": None, "governing_timeframe": None, "grade": None, "invalidation_mode": None}
```

```python
# brain/tests/test_brain_state.py
def test_opportunity_from_dict_defaults_the_thesis_fields_of_an_old_state() -> None:
    o = Opportunity.from_dict({"state": "ACTIONABLE", "direction": "SHORT", "entry_object_id": "a", "invalidation_object_id": "b", "target_object_id": "c"})
    assert o.thesis_id is None and o.governing_timeframe is None and o.grade is ThesisGrade.BASE and o.invalidation_mode is InvalidationMode.TOUCH
    assert Opportunity.from_dict(o.to_dict()) == o
```

```python
# risk/tests/test_plan_contract.py
def test_plan_carries_the_thesis_fields_and_the_signature_ignores_them() -> None:
    a = plan(thesis_id="T1", grade=ThesisGrade.BASE); b = plan(thesis_id="T2", grade=ThesisGrade.A_PLUS, invalidation_mode=InvalidationMode.CLOSE_BEYOND)
    assert a.signature == b.signature and TradePlan.from_dict(b.to_dict()) == b and b.to_dict()["invalidation_mode"] == "CLOSE_BEYOND"

def test_vetoes_and_roles_gained_the_v2_names() -> None:
    assert {VetoCode.DAILY_STOP.value, VetoCode.HALTED.value, VetoCode.LEVERAGE.value} == {"daily_stop", "halted", "leverage"} and OrderRole.FLATTEN.value == "flatten"
```

- [ ] **Step 2: run, watch them fail** (`ImportError` / `MalformedReply` not raised / `TypeError`).
- [ ] **Step 3: implement.** `Opportunity.__post_init__`: when `NONE` all four must be `None`/defaults (`grade`/`invalidation_mode` may be their default enum); otherwise `thesis_id` must match `THESIS_ID_PATTERN = r"^[A-Za-z0-9_-]{1,32}$"`, `governing_timeframe in GOVERNING_TIMEFRAMES`, enums coerced with `_enum`. `to_dict` writes the four keys (`None` for the enums when state is NONE). `parse_update`: `_OPPORTUNITY_KEYS` gains the four; when `state == NONE` each must be `None`; otherwise `_str` / `_enum`. `TradePlan`: four fields after `close` with defaults (`thesis_id="", governing_timeframe="", grade=BASE, invalidation_mode=TOUCH`) — `__post_init__` coerces enums; `to_dict`/`from_dict` round-trip. `RiskVerdict`: two optional fields, in `to_dict`/`from_dict`.
- [ ] **Step 4: run the three test files green; run `brain/tests execution/tests risk/tests` to see what else broke (expected: nothing yet — the fields default).**

---

### Task 2: Geometry buffer and reducer rules

**Files:**
- Modify: `brain/core/opportunity_geometry.py`, `brain/core/reducer.py`, `brain/core/main_brain.py` (pass `atr_1m`), `brain/core/runtime.py` (`_journal_state` geometry call), `execution/core/plan.py`
- Test: `brain/tests/test_opportunity_geometry.py`, `brain/tests/test_reducer.py`, `execution/tests/test_plan_from_state.py`

**Interfaces:**
- Produces: `resolve_geometry(opportunity, objects, *, close, tick, atr_1m, buffer_atr=CLOSE_BEYOND_BUFFER_ATR)`; `coherence_error(..., atr_1m)`; `TIMEFRAME_MINUTES = {"4H": 240, "1H": 60, "15m": 15, "5m": 5, "1m": 1}`; `scale_gap(governing, invalidation) -> int`; reducer rejections `opportunity_invalidation_scale:<alias>`, `thesis_direction_changed:<id>`; `ReduceContext.timeframe_of: Callable[[str], str | None]`.

- [ ] **Step 1: failing tests**

```python
# brain/tests/test_opportunity_geometry.py
def test_close_beyond_puts_the_hard_stop_one_scaled_atr_past_the_edge() -> None:
    objects = {"FVG_15m_1": ObjectGeometry("FVG_15m_1", "fvg", "15m", 100.0, 102.0, 101.0), "BSL_15m_1": ObjectGeometry("BSL_15m_1", "bsl", "15m", 104.0, 104.0, 104.0), "SSL_5m_1": ObjectGeometry("SSL_5m_1", "ssl", "5m", 80.0, 80.0, 80.0)}
    opp = Opportunity("ACTIONABLE", "SHORT", "FVG_15m_1", "BSL_15m_1", "SSL_5m_1", thesis_id="T1", governing_timeframe="1H", invalidation_mode="CLOSE_BEYOND")
    g = resolve_geometry(opp, objects, close=99.0, tick=0.25, atr_1m=2.0)
    assert g.stop_price == 104.0 + 2.0 * 15 ** 0.5 // 0.25 * 0.25 + (0.25 if (2.0 * 15 ** 0.5) % 0.25 else 0) or g.stop_price == 111.75  # 104 + 7.746 → 111.75, rounded away
    assert g.rule_ids[1] == "stop.pool.close_beyond" and g.reward_risk == (100.0 - 80.0) / (111.75 - 100.0)
    touch = resolve_geometry(replace(opp, invalidation_mode="TOUCH"), objects, close=99.0, tick=0.25, atr_1m=2.0)
    assert touch.stop_price == 104.25 and touch.rule_ids[1] == "stop.pool.far_edge"
```

(Write the assertion as the literal `111.75`; the line above shows the arithmetic: 104 + 1.0 × 2.0 × √15 = 111.746, rounded away from the entry to 111.75.)

```python
# brain/tests/test_reducer.py
def test_invalidation_more_than_one_scale_below_the_governing_one_is_refused() -> None:
    res = apply(None, episode_id="EP", evidence=(), update=upd(opportunity=Opportunity("ACTIONABLE", "SHORT", "FVG_5m_1", "BSL_5m_2", "SSL_1H_1", thesis_id="T1", governing_timeframe="1H")),
                ctx=ctx(timeframe_of=lambda a: a.split("_")[-2]))
    assert "opportunity_invalidation_scale:BSL_5m_2" in res.rejections and res.state.opportunity.state is OpportunityState.NONE

def test_invalidation_one_scale_below_is_accepted() -> None:
    res = apply(None, episode_id="EP", evidence=(), update=upd(opportunity=Opportunity("ACTIONABLE", "SHORT", "FVG_5m_1", "BSL_15m_2", "SSL_1H_1", thesis_id="T1", governing_timeframe="1H")),
                ctx=ctx(timeframe_of=lambda a: a.split("_")[-2]))
    assert res.state.opportunity.state is OpportunityState.ACTIONABLE

def test_a_thesis_id_that_flips_direction_is_refused() -> None:
    first = apply(None, episode_id="EP", evidence=(), update=upd(opportunity=Opportunity("DEVELOPING", "SHORT", "FVG_5m_1", "BSL_5m_2", "SSL_1H_1", thesis_id="T1", governing_timeframe="5m")), ctx=ctx())
    res = apply(first.state, episode_id="EP", evidence=(), update=upd(opportunity=Opportunity("ACTIONABLE", "LONG", "SSL_1H_1", "BSL_5m_2", "FVG_5m_1", thesis_id="T1", governing_timeframe="5m")), ctx=ctx())
    assert "thesis_direction_changed:T1" in res.rejections and res.state.opportunity.state is OpportunityState.NONE
```

- [ ] **Step 2: run, watch them fail.**
- [ ] **Step 3: implement.** Geometry: `_stop(obj, direction, tick, *, mode, atr_1m, buffer_atr)`; for `CLOSE_BEYOND` `buffer = buffer_atr * atr_1m * sqrt(TIMEFRAME_MINUTES[obj.timeframe])`, price = edge ± buffer, then `_round_away(price, tick, direction)` (ceil for SHORT, floor for LONG). Reducer: `ReduceContext.timeframe_of` (default `lambda alias: None` — unknown scale is not checked); rule 4 adds the two rejections before coherence. `main_brain.py` passes `atr_1m=context.atr_1m` (the `EyeContext` has it — check the attribute name; add it if the context only exposes it inside `bar`) and `timeframe_of=lambda alias: registry.get(alias).timeframe`. `runtime._journal_state` and `plan_from_state` pass `atr_1m`.
- [ ] **Step 4: green; run `brain/tests execution/tests`.**

---

### Task 3: Risk gate v2

**Files:**
- Modify: `risk/core/gate.py`, `risk/configs/risk.json`
- Test: `risk/tests/test_gate.py` (rewrite the config-dependent assertions), `execution/tests/test_stack_e2e.py` (`RISK` override keeps working via `dataclasses.replace`)

**Interfaces:**
- Produces: `RiskConfig(risk_fraction: Mapping[str, float], max_open_positions, min_reward_risk, preferred_reward_risk, daily_loss_fraction, max_drawdown_fraction, max_leverage, max_quantity, order_ttl_bars, account_max_age_s, margin_per_contract, thesis: ThesisConfig(max_expressions, stop_cooldown_bars), contract, sha256)`; `RiskGate.observe(account, asof) -> None`; `RiskGate.halted: bool`; `RiskGate.halt_record: dict | None`; `RiskGate.daily_stopped(asof) -> bool`; `RiskGate.session_date(asof) -> date`; `RiskGate.assess(plan, account, *, asof, positions: Sequence[PositionRecord] = ())`.

- [ ] **Step 1: failing tests** (the existing helpers `snapshot(...)`, `plan(...)` in `risk/tests/test_gate.py` are reused; `positions` uses `PositionRecord`)

```python
def test_base_grade_risks_one_and_a_half_percent_and_a_plus_two_only_at_the_preferred_ratio() -> None:
    gate = RiskGate(CONFIG)
    base = gate.assess(plan(entry=16400.0, stop=16350.0, target=16500.0, grade="BASE"), snapshot(), asof=T0)      # 2 R, 50 pts
    assert base.passed and base.risk_fraction == 0.015 and base.grade_applied == "BASE" and base.quantity == 1        # 1500 / 1000
    demoted = gate.assess(plan(entry=16400.0, stop=16350.0, target=16500.0, grade="A_PLUS"), snapshot(), asof=T0)  # 2 R < preferred 3
    assert demoted.grade_applied == "BASE" and demoted.risk_fraction == 0.015
    a_plus = gate.assess(plan(entry=16400.0, stop=16350.0, target=16550.0, grade="A_PLUS"), snapshot(), asof=T0)   # 3 R
    assert a_plus.grade_applied == "A_PLUS" and a_plus.risk_fraction == 0.02 and a_plus.quantity == 2                # 2000 / 1000

def test_reward_risk_below_two_is_vetoed() -> None:
    v = RiskGate(CONFIG).assess(plan(entry=16400.0, stop=16350.0, target=16490.0), snapshot(), asof=T0)
    assert v.vetoes == (VetoCode.REWARD_RISK,) and "1.80 < 2.0" in v.reasons[0]

def test_leverage_caps_the_contracts_by_notional() -> None:
    v = RiskGate(CONFIG).assess(plan(entry=16400.0, stop=16395.0, target=16450.0), snapshot(equity=100_000.0), asof=T0)  # budget 1500/100 = 15, margin 5, leverage 800k/328k = 2
    assert v.passed and v.quantity == 2
    tiny = RiskGate(CONFIG).assess(plan(entry=16400.0, stop=16395.0, target=16450.0), snapshot(equity=30_000.0), asof=T0)  # 240k/328k < 1
    assert tiny.vetoes == (VetoCode.LEVERAGE,)

def test_position_size_reason_says_the_contract_is_too_large_for_this_stop() -> None:
    v = RiskGate(CONFIG).assess(plan(entry=16400.0, stop=16300.0, target=16700.0), snapshot(), asof=T0)  # 100 pts = 2000 > 1500
    assert v.vetoes == (VetoCode.POSITION_SIZE,) and "too large for this stop" in v.reasons[0]

def test_exposure_counts_the_machines_positions_and_refuses_the_opposite_direction() -> None:
    gate = RiskGate(CONFIG)
    three = [PositionRecord(f"p{i}", TradeDirection.SHORT, T0, "FVG_5m_1") for i in range(3)]
    assert gate.assess(plan(direction=TradeDirection.SHORT), snapshot(), asof=T0, positions=three).vetoes == (VetoCode.EXPOSURE,)
    two = three[:2]
    assert gate.assess(plan(direction=TradeDirection.SHORT), snapshot(), asof=T0, positions=two).passed
    opposite = gate.assess(plan(direction=TradeDirection.LONG), snapshot(), asof=T0, positions=two)
    assert opposite.vetoes == (VetoCode.EXPOSURE,) and "opposite" in opposite.reasons[0]

def test_daily_stop_latches_for_the_session_date_and_resets_at_the_next() -> None:
    gate = RiskGate(CONFIG)
    monday_open = pd.Timestamp("2022-01-02T23:00:00Z")  # 18:00 NY Sunday = Monday's session
    gate.observe(snapshot(equity=100_000.0, asof=monday_open), monday_open)
    later = monday_open + pd.Timedelta(hours=12)
    gate.observe(snapshot(equity=97_400.0, asof=later), later)
    assert gate.daily_stopped(later) and gate.assess(plan(), snapshot(equity=97_400.0, asof=later), asof=later).vetoes == (VetoCode.DAILY_STOP,)
    recovered = later + pd.Timedelta(hours=1)
    gate.observe(snapshot(equity=99_000.0, asof=recovered), recovered)
    assert gate.daily_stopped(recovered), "the daily stop holds for the rest of the session even if equity recovers"
    tuesday = pd.Timestamp("2022-01-03T23:30:00Z")
    gate.observe(snapshot(equity=97_400.0, asof=tuesday), tuesday)
    assert not gate.daily_stopped(tuesday)

def test_drawdown_from_the_peak_halts_and_stays_halted() -> None:
    gate = RiskGate(CONFIG)
    for i, equity in enumerate((100_000.0, 104_000.0, 98_000.0, 97_240.0, 120_000.0)):
        at = T0 + pd.Timedelta(minutes=i)
        gate.observe(snapshot(equity=equity, asof=at), at)
        if equity == 98_000.0:
            assert not gate.halted  # 5.8 % from 104k
    assert gate.halted and gate.halt_record["peak"] == 104_000.0 and gate.halt_record["equity"] == 97_240.0
    assert gate.assess(plan(), snapshot(equity=120_000.0), asof=T0 + pd.Timedelta(minutes=5)).vetoes == (VetoCode.HALTED,)

def test_v1_config_is_refused() -> None:
    with pytest.raises(ValueError):
        RiskConfig.from_json(ROOT / "risk" / "tests" / "risk_v1.json")   # write this fixture: the old file's content
```

- [ ] **Step 2: run, watch them fail.**
- [ ] **Step 3: implement** per spec §5. `session_date(asof) = (asof.tz_convert("America/New_York") + 6h).date()`. `observe`: `self._peak = max(self._peak or equity, equity)`; day start reset on a new session date; latches. The old `risk_fraction` float is gone; `from_json` requires the mapping with both grades in (0, 1). Update `risk/configs/risk.json` to schema 2 and copy the old content to `risk/tests/risk_v1.json` as the refused fixture.
- [ ] **Step 4: green; then `execution/tests` — the machine tests will fail on the new `assess` signature only if they call it directly (they don't).**

---

### Task 4: Simulated executor: mark-to-market, exit-leg cancel, flatten

**Files:**
- Modify: `execution/core/simulated_executor.py`, `execution/core/broker.py` (`flatten` in the protocol), `execution/core/ibkr_broker.py` (`flatten(symbol, quantity, side, asof, client_ref)`), `execution/scripts/ibkr_paper_exercise.py` (call site), `execution/tests/test_ibkr_paper_exercise.py`, `execution/tests/test_ibkr_broker.py`
- Test: `execution/tests/test_simulated_executor.py`

- [ ] **Step 1: failing tests**

```python
def test_equity_marks_open_positions_at_the_last_polled_close() -> None:
    ex = executor()
    run(ex, short(quantity=2), [bar(1, 16370.0, 16380.0), bar(2, 16375.0, 16390.0, close=16385.0), bar(3, 16380.0, 16400.0, close=16395.0)])
    snap = ex.snapshot(at(3))
    assert snap.positions[0].quantity == -2 and snap.equity == 100_000.0 + (16387.5 - 16395.0) * 2 * 20.0 and ex.account.cash == 100_000.0

def test_a_filled_brackets_exits_can_be_cancelled_and_a_flatten_fills_at_the_next_open() -> None:
    ex = executor()
    events = run(ex, short(quantity=2), [bar(1, 16370.0, 16380.0), bar(2, 16375.0, 16390.0, close=16385.0)])
    entry = ex.snapshot(at(2)).positions and [o for o in ex.account.orders.values() if o.role is OrderRole.ENTRY][0]
    exits = [o for o in ex.snapshot(at(2)).open_orders if o.parent_id == entry.order_id]
    for o in exits:
        ex.cancel(o.order_id, at(2))
    flat = ex.flatten("NQ", 2, "BUY", at(2), "EP_1:sig:invalidation")
    assert flat.role is OrderRole.FLATTEN and flat.client_ref == "EP_1:sig:invalidation"
    b3 = Bar(start=T0 + pd.Timedelta(minutes=3), open=16392.0, high=16420.0, low=16390.0, close=16410.0, volume=1.0, symbol="NQ", instrument_id=1)
    events = ex.poll(at(3), b3)
    kinds = [(e.kind, e.order.role.value) for e in events]
    assert ("cancelled", "stop") in kinds and ("cancelled", "target") in kinds and ("filled", "flatten") in kinds
    fill = next(e for e in events if e.order.role is OrderRole.FLATTEN).fill
    assert fill.price == 16392.0 and not ex.snapshot(at(3)).positions and ex.account.cash == 100_000.0 + (16387.5 - 16392.0) * 2 * 20.0
    assert ("filled", "stop") not in kinds, "the flatten fills at the open before the bar's range is matched; the stop was cancelled"
```

- [ ] **Step 2: run, watch them fail** (`AttributeError: flatten`, `ValueError: not an open entry`).
- [ ] **Step 3: implement.** `VirtualAccount.last_price: dict[str, float]`, `equity(point_value)`. `poll` records `bar.close` per symbol first. `cancel`: entry (as today) or an open exit leg of a filled bracket → `bracket.cancel_exits.add(leg)`. `_flattens: list[_Flatten(order, symbol)]`; `poll` processes flattens before brackets: fill at `bar.open`, `apply_fill`, event `filled`. `_Bracket` exit cancels applied at the top of the bracket loop, emitting `cancelled`; when both exits are cancelled and the entry is not open the bracket is deleted (the position stays in the account). IBKR `flatten` signature; `_order_state` maps `orderRef` ending in `:halt` / `:invalidation` or `"flatten"` to `OrderRole.FLATTEN`.
- [ ] **Step 4: green; `execution/tests` green (the paper-exercise test's `ScriptedIB` may need the new signature).**

---

### Task 5: `ThesisBook`

**Files:**
- Create: `execution/core/thesis.py`
- Test: `execution/tests/test_thesis.py`

**Interfaces:**
- Produces: `ThesisConfig(max_expressions: int, stop_cooldown_bars: int)` (lives in `risk/core/gate.py` as part of `RiskConfig.thesis`); `ThesisBook(config)`; `.start_episode(episode_id)`; `.admit(plan, bar_index) -> str | None`; `.expressed(plan)`; `.engaged(signature, engaged: bool)`; `.outcome(plan, kind, *, exit_role, bar_index)`; `.cooldown_bars_left(bar_index) -> int`; `.view(bar_index) -> dict` (`theses` list + `cooldown_bars_left`); `ThesisRecord` dataclass.

- [ ] **Step 1: failing tests**

```python
def test_first_expression_is_admitted_then_a_second_of_the_same_thesis_while_engaged_is_refused() -> None:
    book = ThesisBook(ThesisConfig(2, 30)); book.start_episode("EP")
    p = plan(thesis_id="T1")
    assert book.admit(p, 0) is None
    book.expressed(p); book.engaged(p.signature, True)
    assert book.admit(plan(thesis_id="T1", target_id="swing:d"), 1) == "thesis_engaged"

def test_a_stop_closes_the_thesis_and_starts_the_cooldown_for_every_thesis() -> None:
    book = ThesisBook(ThesisConfig(2, 30)); book.start_episode("EP")
    p = plan(thesis_id="T1"); book.admit(p, 0); book.expressed(p); book.engaged(p.signature, True)
    book.outcome(p, "position_closed", exit_role="stop", bar_index=10); book.engaged(p.signature, False)
    assert book.admit(plan(thesis_id="T1", target_id="swing:d"), 11) == "thesis_closed"
    assert book.admit(plan(thesis_id="T2"), 11) == "stop_cooldown" and book.cooldown_bars_left(11) == 29
    assert book.admit(plan(thesis_id="T2"), 40) is None

def test_expressions_are_capped_and_a_target_closes_as_achieved() -> None:
    book = ThesisBook(ThesisConfig(2, 30)); book.start_episode("EP")
    for i in range(2):
        p = plan(thesis_id="T1", target_id=f"swing:{i}"); assert book.admit(p, i) is None; book.expressed(p)
        book.outcome(p, "expired", exit_role=None, bar_index=i + 1)
    assert book.admit(plan(thesis_id="T1", target_id="swing:z"), 5) == "expressions_exhausted"
    assert book.view(5)["theses"][0]["closed_reason"] == "expressions_exhausted"
    q = plan(thesis_id="T2"); book.admit(q, 6); book.expressed(q); book.outcome(q, "position_closed", exit_role="target", bar_index=9)
    assert book.view(9)["theses"][1] == {"thesis_id": "T2", "direction": "SHORT", "governing_timeframe": "1H", "status": "CLOSED", "closed_reason": "achieved", "expressions": 1}

def test_a_thesis_that_flips_direction_is_closed_and_a_new_episode_forgets_everything() -> None:
    book = ThesisBook(ThesisConfig(2, 30)); book.start_episode("EP")
    book.admit(plan(thesis_id="T1"), 0)
    assert book.admit(plan(thesis_id="T1", direction=TradeDirection.LONG), 1) == "direction_changed"
    book.start_episode("EP2")
    assert book.admit(plan(thesis_id="T1", direction=TradeDirection.LONG), 2) is None and book.view(2)["cooldown_bars_left"] == 0
```

- [ ] **Step 2: run, watch them fail.**
- [ ] **Step 3: implement** per spec §6.3 (a record is created on the first `admit`; `admit` closes on `direction_changed` / `expressions_exhausted` and returns the refusal).
- [ ] **Step 4: green.**

---

### Task 6: Order machine with several intents, the book, close-beyond and the halt

**Files:**
- Modify: `execution/core/order_fsm.py` (rewrite), `brain/core/position_ledger.py` (`IDLE_VIEW`, docstring)
- Test: `execution/tests/test_order_fsm.py` (existing tests adapted to the v2 config: `short_plan` needs `thesis_id="T1", governing_timeframe="1H"`, stop 16411.5 → 24 pts × 20 = 480 fits 1500), `execution/tests/test_order_scenarios_real_tape.py` (same), new tests below

**Interfaces:**
- Produces: `OrderMachine(broker, gate, *, journal=None)`; `.on_bar(asof, bar, plan, *, episode_id, visible, llm_called=False, closed_timeframes: frozenset[str] = frozenset())`; `.state`; `.positions() -> tuple[PositionRecord, ...]`; `.halted: bool`; `.execution_view()`; `STAT_KINDS` + `thesis_refused`, `invalidation_close`, `flattened`, `halted`; `PositionRecord(position_id, direction, opened_at, entry_object_id, thesis_id="", invalidation_mode="TOUCH")`.

- [ ] **Step 1: failing tests**

```python
def test_three_same_direction_theses_open_three_positions_and_a_fourth_is_vetoed(tmp_path):
    m, broker, journal = machine(tmp_path)
    for i in range(3):
        p = short_plan(target_id=f"swing:{i}", thesis_id=f"T{i}")
        assert "submitted" in m.on_bar(at(2 * i + 1), bar(2 * i + 1, 16370.0, 16380.0), p, episode_id=EP, visible=lambda a: True)
        assert "position_opened" in m.on_bar(at(2 * i + 2), bar(2 * i + 2, 16380.0, 16390.0), p, episode_id=EP, visible=lambda a: True)
    assert len(m.positions()) == 3 and m.state is MachineState.IN_POSITION
    kinds = m.on_bar(at(7), bar(7, 16370.0, 16380.0), short_plan(target_id="swing:x", thesis_id="T9"), episode_id=EP, visible=lambda a: True)
    assert "veto" in kinds and last_trade(journal)["verdict"]["vetoes"] == ["exposure"]

def test_an_opposite_direction_plan_is_vetoed_while_a_position_is_open(tmp_path): ...

def test_a_closed_thesis_is_refused_and_journaled_once_per_proposal(tmp_path):
    # stop out T1, re-propose T1 with another target: "thesis_refused" with reason thesis_closed; the same plan on a quiet bar is not journaled again; with llm_called=True it is
    # and a fresh thesis T2 is refused with stop_cooldown for 30 bars, then admitted

def test_close_beyond_exits_at_market_when_the_scale_closes_beyond_the_object(tmp_path):
    # plan invalidation_mode=CLOSE_BEYOND on a 5m BSL at 16411.5, hard stop far above (buffer); bars trade up to 16412 (wick) without a 5m close beyond → nothing;
    # a bar with closed_timeframes={"5m"} and close 16413 → cancel_requested×2 (stop, target) + invalidation_close + flatten submitted; next bar → position_closed exit_role "flatten"

def test_the_hard_stop_halts_cancels_and_flattens_everything(tmp_path):
    # equity 100k, open a position, then a bar whose close marks equity ≤ 93.5k → "halted" record, cancel_requested for the exits, flatten submitted; next bar: position_closed(flatten); m.halted is True; a later plan → veto halted

def test_execution_view_lists_positions_theses_and_the_cooldown(tmp_path): ...
```

- [ ] **Step 2: run, watch them fail.**
- [ ] **Step 3: implement** per spec §6.4–6.6. Keep the file's docstring current. Bar index for the book = a counter of `on_bar` calls in the episode.
- [ ] **Step 4: `execution/tests` green, including the nine real-tape scenarios (adapt their plans to the v2 fields and budget).**

---

### Task 7: Stack, plan builder, runner halt, IBKR call sites

**Files:**
- Modify: `execution/core/stack.py` (closed timeframes from `observation.events_this_update` where `kind is BAR_COMPLETED`; `halted` property), `execution/core/plan.py` (fields + `atr_1m`), `brain/scripts/_run_identity.py` (`drive(..., stop=None)`), `brain/scripts/run_llm_brain.py` (`stop`, `run["halted"]`), `brain/scripts/replay_journal.py` (unchanged API; verify)
- Test: `execution/tests/test_stack_e2e.py` (scripted client emits the four fields: `thesis_id "T1"`, `governing_timeframe "5m"`, `grade "BASE"`, `invalidation_mode "TOUCH"`; assert `prior_state.execution.positions`), `execution/tests/test_plan_from_state.py`, `brain/tests/test_run_guards.py` (`drive` stops when the predicate turns true: count bars)

- [ ] **Step 1: failing tests; Step 2: fail; Step 3: implement; Step 4: green.**

---

### Task 8: Controller relation filter

**Files:**
- Modify: `brain/configs/sleep_controller.json` (schema 3, `relation_change_timeframes`), `brain/core/sleep_controller.py` (`ControllerConfig.relation_change_timeframes`), `brain/core/runtime.py` (`_watched` filters by `registry.get(alias).timeframe`)
- Test: `brain/tests/test_sleep_controller.py` (schema 2 refused; field parsed), `brain/tests/test_runtime.py` (a 5m watched object's flip is a TICK; a 15m one's is an UPDATE)

- [ ] **Steps 1–4 as above.**

---

### Task 9: Prompt and prior view

**Files:**
- Modify: `brain/configs/prompts/main_brain_system.md` (spec §7.2), `brain/core/main_brain.py` (`_prior_view` copies the new view as is), `brain/core/position_ledger.py` (`IDLE_VIEW`)
- Test: `brain/tests/test_main_brain.py` (`prior_state.execution` has `positions`, `theses`, `cooldown_bars_left`, `daily_stop`, `halted`), `brain/tests/test_run_guards.py` if it checks the prompt's `{EXAMPLE}` token

- [ ] **Steps 1–4.**

---

### Task 10: Summary and baselines

**Files:**
- Modify: `brain/scripts/summarize_run.py` (`risk` section: `thesis_refused` by reason from the journal, `daily_stop_vetoes`, `halted`; `orders` picks up the new stat kinds), `brain/docs/evidence/regression_baselines.json` (emptied to `[]` until the new run completes; the test skips with a reason when the list is empty), `brain/tests/test_regression_baseline.py`
- Test: `brain/tests/test_summarize_run.py`

- [ ] **Steps 1–4.**

---

### Task 11: Docs and cleanup

- Amend `execution/docs/specs/2026-09-16-risk-execution-design.md` §3 and §5 with a dated note pointing to the new spec; `brain/docs/specs/2026-09-16-llm-brain-design.md` §7.2 likewise.
- `execution/docs/README.md` (module table, rules table, protocols), `risk/docs/README.md` (config table, checks), `brain/docs/README.md` (receipts, controller), `AGENTS.md` (risk row), `shares/docs/current_implementation_status.md`.
- Remove nothing tracked without approval; list the stale `.claude/worktrees/*` checkouts in the report as a cleanup candidate.

### Task 12: The day run, the receipt, the baseline, the parameter judgment

- Launch `run_llm_brain.py --client deepseek --broker sim --emit-start 2022-01-02T18:00 --end 2022-01-03T17:00 --max-llm-calls 1000 --reasoning-effort high` in the background (logs in the scratchpad); summarize; replay; add to `regression_baselines.json`; write `brain/docs/evidence/2026-09-17_risk_v2_day_run_2022-01-03.md`; judge the parameters from the arithmetic (§5) and the run.
