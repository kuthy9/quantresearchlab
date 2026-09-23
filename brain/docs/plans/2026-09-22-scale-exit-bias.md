# Thesis scale, structural exit and bias decay implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One NQ contract fundable at the floored stop with the day's and the run's limits validated against it; the thesis scale set by code from the bias; the target on that scale; positions that leave on their own invalidation or a structural reversal on their scale, never on a bias flip; a bias that decays to NEUTRAL when the scales below it deliver against it.

**Architecture:** Risk schema 4 with two load-time policy checks and an open-risk veto; `governing_timeframe` leaves the reply and the reducer derives it from the bias (`THESIS_SCALE_OF_BIAS`); reducer rules 4c (target scale) and 4d (bias continuity, re-assertion, decay) on `Bias.since` / `Bias.decayed` (state schema 3); the stack hands the machine this bar's MSS / BOS events and the machine flattens a position whose thesis scale reversed.

**Tech Stack:** Python 3.12, pytest, pandas; DeepSeek via `run_llm_brain`.

**Spec:** [brain/docs/specs/2026-09-22-scale-exit-bias-design.md](../specs/2026-09-22-scale-exit-bias-design.md)

## Global Constraints

- Test command: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest <paths> -p no:cacheprovider` (never `-q`).
- Risk: `BASE 0.025`, `A_PLUS 0.03`, `daily_loss_fraction 0.05`, `max_drawdown_fraction 0.10`, `max_open_positions 2`; `max_open_positions × BASE ≤ daily`, `2 × daily ≤ drawdown`; the open-risk veto is `ACCOUNT_RISK`.
- `THESIS_SCALE_OF_BIAS = {"4H": "1H", "1H": "15m", "15m": "15m"}`; `BIAS_DECAY_SCALES = {"4H": ("4H", "1H", "15m"), "1H": ("1H", "15m"), "15m": ("15m", "5m")}`; `BIAS_DECAY_EVENTS = 2`; structural kinds `mss_core_confirmed`, `qualified_bos`, `displacement_observed`; the exit watches `mss_core_confirmed` and `qualified_bos` only.
- No entry-timing change; no single-day tuning; nothing committed until asked; nothing under `outputs/` deleted.

---

### Task 1: The risk budget

**Files:**
- Modify: `risk/configs/risk.json`, `risk/core/gate.py`, `execution/core/order_fsm.py`
- Test: `risk/tests/test_gate.py`, `execution/tests/test_order_fsm.py`

**Interfaces:**
- Produces: `RISK_SCHEMA_VERSION = 4`; `RiskConfig.from_json` raises `ValueError("risk policy: ...")` on the two checks; `RiskGate.assess(plan, account, *, asof, positions=(), open_risk=0.0)`; `RiskGate.session_budget_left(equity) -> float`; veto `VetoCode.ACCOUNT_RISK` with reason `"open risk {open:.2f} + {new:.2f} exceeds the session's remaining loss budget {left:.2f}"`; `OrderMachine.open_risk() -> float`.

- [x] Step 1: Write the failing tests in `risk/tests/test_gate.py` — replace `test_config_loads_the_v2_defaults_and_hashes` and `test_the_config_is_schema_3_...` with schema-4 values; add:

```python
def _policy(tmp_path: Path, **over) -> Path:
    import json
    payload = json.loads((ROOT / "risk" / "configs" / "risk.json").read_text(encoding="utf-8"))
    payload.update(over)
    path = tmp_path / "risk.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_the_policy_checks_bind_the_three_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="risk policy"):
        RiskConfig.from_json(_policy(tmp_path, max_open_positions=3))  # 3 × 0.025 > 0.05
    with pytest.raises(ValueError, match="risk policy"):
        RiskConfig.from_json(_policy(tmp_path, max_drawdown_fraction=0.09))  # 2 × 0.05 > 0.09
    assert RiskConfig.from_json(_policy(tmp_path)).max_open_positions == 2


def test_open_risk_is_held_to_the_sessions_remaining_budget() -> None:
    gate = RiskGate(CONFIG)
    gate.observe(account(100_000.0), T)  # the session opens at 100 000: budget 5 000
    verdict = gate.assess(plan(), account(100_000.0), asof=T, positions=positions(1), open_risk=2_500.0)
    assert verdict.passed and verdict.risk_amount == 2_400.0  # 2 500 + 2 400 ≤ 5 000
    verdict = gate.assess(plan(), account(100_000.0), asof=T, positions=positions(1), open_risk=2_700.0)
    assert not verdict.passed and verdict.vetoes == (VetoCode.ACCOUNT_RISK,) and "remaining loss budget 5000.00" in verdict.reasons[0]
    gate.observe(account(98_000.0), T + pd.Timedelta(minutes=5))  # 2 000 lost: 3 000 left
    assert gate.session_budget_left(98_000.0) == 3_000.0
    verdict = gate.assess(plan(), account(98_000.0, asof=T + pd.Timedelta(minutes=5)), asof=T + pd.Timedelta(minutes=5), open_risk=1_000.0)
    assert not verdict.passed and verdict.vetoes == (VetoCode.ACCOUNT_RISK,)
```

and in `execution/tests/test_order_fsm.py`:

```python
def test_the_machine_passes_the_open_risk_to_the_gate(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    first = short_plan(thesis_id="T1")
    m.on_bar(at(1), bar(1, 16370.0, 16380.0), first, episode_id=EP, visible=lambda a: True)
    m.on_bar(at(2), bar(2, 16380.0, 16390.0), first, episode_id=EP, visible=lambda a: True)
    assert m.state is MachineState.IN_POSITION and m.open_risk() == 480.0  # one contract, 24 points
```

- [x] Step 2: Run both files; expected: the schema-4 assertions fail, `open_risk` is an unexpected keyword, `open_risk` does not exist.
- [x] Step 3: `risk/configs/risk.json`: `schema_version 4`, the five values. `gate.py`: `RISK_SCHEMA_VERSION = 4`; in `from_json` after the range check:

```python
        base = config.risk_fraction[ThesisGrade.BASE.value]
        if config.max_open_positions * base > config.daily_loss_fraction + 1e-12:
            raise ValueError("risk policy: max_open_positions × risk_fraction.BASE must not exceed daily_loss_fraction")
        if 2.0 * config.daily_loss_fraction > config.max_drawdown_fraction + 1e-12:
            raise ValueError("risk policy: max_drawdown_fraction must hold two daily stops")
```

`session_budget_left`:

```python
    def session_budget_left(self, equity: float) -> float:
        """What the session may still lose before its daily stop: the
        opening equity's ``daily_loss_fraction`` less what it lost so far."""
        opening = self._session_open_equity if self._session_open_equity is not None else float(equity)
        return round(opening * self._config.daily_loss_fraction - max(0.0, opening - float(equity)), 2)
```

and in `assess` (signature `open_risk: float = 0.0`), after `quantity = min(...)`:

```python
        new_risk = quantity * per_contract
        left = self.session_budget_left(account.equity)
        if open_risk + new_risk > left + 1e-9:
            return RiskVerdict(False, (VetoCode.ACCOUNT_RISK,), (
                f"open risk {open_risk:.2f} + {new_risk:.2f} exceeds the session's remaining loss budget {left:.2f}",
            ))
```

`order_fsm.py`: `def open_risk(self) -> float: return round(sum(float(intent.verdict.risk_amount or 0.0) for intent in self._intents.values() if intent.position is not None), 2)` and `self._gate.assess(plan, account, asof=asof, positions=self.positions(), open_risk=self.open_risk())`.

- [x] Step 4: Run `risk/tests`, `execution/tests/test_order_fsm.py`, `execution/tests/test_stack_e2e.py`, `brain/tests/test_run_guards.py`; fix the quantity assertions the new fractions change (the tests' `CONFIG` in `test_order_fsm.py` overrides the fractions to 0.005, so its quantities hold).

### Task 2: The contract — thesis scale, `Bias.since` / `Bias.decayed`, the reply without `governing_timeframe`

**Files:**
- Modify: `contract/brain/state.py`, `contract/brain/llm.py`
- Test: `brain/tests/test_brain_state.py`, `brain/tests/test_llm_contract.py`

**Interfaces:**
- Produces: `THESIS_SCALE_OF_BIAS`, `BIAS_DECAY_SCALES`, `BIAS_DECAY_EVENTS`, `STRUCTURAL_EVIDENCE_KINDS = frozenset({"mss_core_confirmed", "qualified_bos", "displacement_observed"})`, `REVERSAL_EVIDENCE_KINDS = frozenset({"mss_core_confirmed", "qualified_bos"})` in `contract/brain/state.py`; `Bias(direction, scale, basis, since: pd.Timestamp | None = None, decayed: str | None = None)` with `pair` property (`"LONG@15m"`); `BRAIN_STATE_SCHEMA_VERSION = 3`; `parse_update` refuses a reply whose opportunity carries `governing_timeframe`.

- [x] Step 1: Failing tests. `test_brain_state.py`:

```python
def test_bias_since_and_decayed_round_trip_and_schema_2_reads_back_unset() -> None:
    bias = Bias(BiasDirection.SHORT, "4H", "4H leg short", since=T1, decayed=None)
    state = make_state(bias=bias)
    again = BrainState.from_json(state.to_json())
    assert again.bias == bias and again.bias.pair == "SHORT@4H" and again.schema_version == 3
    decayed = Bias(BiasDirection.NEUTRAL, "4H", "code: 2 events", since=T1, decayed="SHORT@4H")
    assert Bias.from_dict(decayed.to_dict()) == decayed and decayed.to_dict()["since"] == "2022-01-04T14:41:00Z"
    payload = state.to_dict()
    payload["schema_version"] = 2
    payload["bias"] = {"direction": "LONG", "scale": "15m", "basis": "old"}
    old = BrainState.from_dict(payload)
    assert old.bias == Bias(BiasDirection.LONG, "15m", "old") and old.bias.since is None and old.bias.decayed is None


def test_the_thesis_scale_of_each_bias_scale() -> None:
    assert THESIS_SCALE_OF_BIAS == {"4H": "1H", "1H": "15m", "15m": "15m"}
    assert BIAS_DECAY_SCALES == {"4H": ("4H", "1H", "15m"), "1H": ("1H", "15m"), "15m": ("15m", "5m")} and BIAS_DECAY_EVENTS == 2
```

`test_llm_contract.py`: the `_actionable` helper and the NONE example lose `governing_timeframe`; the `1m` case becomes:

```python
def test_a_reply_naming_a_governing_timeframe_is_refused() -> None:
    with pytest.raises(MalformedReply, match="governing_timeframe"):
        parse_update(_reply(opportunity={**_actionable(), "governing_timeframe": "15m"}), evidence_ids=EVIDENCE, object_ids=OBJECTS)
```

and the parsed opportunity asserts `o.governing_timeframe is None`.

- [x] Step 2: Run; expected: `TypeError` on `since`, `ImportError` on the constants, the refusal test fails (the key is accepted today).
- [x] Step 3: `state.py`: the constants beside `BIAS_SCALES`; `Bias` fields `since`, `decayed`, `pair`, `to_dict` (`"since": None | isoformat_utc`, `"decayed"`), `from_dict` reading both with `.get`; `BRAIN_STATE_SCHEMA_VERSION = 3`; `from_dict` accepting `(1, 2, 3)`. `llm.py`: drop the key from `_OPPORTUNITY_KEYS`, `_THESIS_KEYS`, `LLM_UPDATE_EXAMPLE`, the parser (the `_require_keys` check already refuses unknown keys — confirm the message names the key).
- [x] Step 4: Run `brain/tests/test_brain_state.py`, `brain/tests/test_llm_contract.py`, `brain/tests/test_journal.py`.

### Task 3: The reducer — thesis scale, target scale, bias continuity / re-assertion / decay

**Files:**
- Modify: `brain/core/reducer.py`
- Test: `brain/tests/test_reducer.py`

**Interfaces:**
- Consumes: Task 2's constants and `Bias` fields.
- Produces: `effective_bias(prev: Bias | None, reply: Bias, ledger: EvidenceLedger, known_at) -> tuple[Bias, tuple[str, ...]]` (pure, exported); rejections `bias_decayed:<PAIR>:<n>`, `bias_reassert_refused:<PAIR>`, `opportunity_target_scale:<alias>`; the state's `opportunity.governing_timeframe` = `THESIS_SCALE_OF_BIAS[bias.scale]`.

- [x] Step 1: Failing tests (the `upd()` helper's bias becomes `Bias(BiasDirection.LONG, "1H", "scripted")` so the tests' 15m theses stay valid; `test_a_thesis_above_the_bias_scale_is_dropped` is replaced):

```python
def struct(i: str, tf: str, direction: str, kind: str = "displacement_observed", minutes: int = 0) -> EvidenceItem:
    return EvidenceItem(i, T2 + pd.Timedelta(minutes=minutes), kind, tf, None, direction=direction)


def test_the_thesis_scale_is_the_biass_and_the_reply_cannot_set_it() -> None:
    opp = Opportunity(OpportunityState.ACTIONABLE, TradeDirection.LONG, "FVG_5m_3", "SSL_5m_2", "BSL_1H_1", thesis_id="T1")
    res = apply(None, episode_id="EP", evidence=[], update=upd(opportunity=opp, bias=Bias(BiasDirection.LONG, "1H", "b")), ctx=ctx(timeframe_of=lambda a: REG[a].timeframe))
    assert res.state.opportunity.governing_timeframe == "15m" and res.state.opportunity.state is OpportunityState.ACTIONABLE
    res = apply(None, episode_id="EP", evidence=[], update=upd(opportunity=opp, bias=Bias(BiasDirection.LONG, "4H", "b")), ctx=ctx(timeframe_of=lambda a: REG[a].timeframe))
    assert "opportunity_invalidation_scale:SSL_5m_2" in res.rejections  # a 1H thesis is not falsified by a 5m pool


def test_a_target_below_the_thesis_scale_is_refused() -> None:
    reg = {**REG, "BSL_5m_9": RegisteredObject("d" * 24, "bsl", "5m")}
    opp = Opportunity(OpportunityState.ACTIONABLE, TradeDirection.LONG, "FVG_5m_3", "SSL_5m_2", "BSL_5m_9", thesis_id="T1")
    res = apply(None, episode_id="EP", evidence=[], update=upd(opportunity=opp, bias=Bias(BiasDirection.LONG, "15m", "b")),
                ctx=ctx(registry=reg, visible_aliases=frozenset(reg), timeframe_of=lambda a: reg[a].timeframe))
    assert "opportunity_target_scale:BSL_5m_9" in res.rejections and res.state.opportunity == Opportunity()


def test_a_bias_keeps_its_since_while_the_pair_holds_and_restarts_on_a_new_pair() -> None:
    first = apply(None, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.SHORT, "4H", "a")), ctx=ctx())
    assert first.state.bias.since == T2 and first.state.bias.decayed is None
    later = ctx(known_at=T2 + pd.Timedelta(minutes=5))
    second = apply(first.state, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.SHORT, "4H", "b")), ctx=later)
    assert second.state.bias.since == T2 and second.state.bias.basis == "b"
    third = apply(second.state, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.LONG, "15m", "c")), ctx=later)
    assert third.state.bias.since == later.known_at


def test_two_structural_events_against_the_bias_on_the_scales_below_decay_it() -> None:
    prev = apply(None, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.SHORT, "4H", "a")), ctx=ctx()).state
    later = ctx(known_at=T2 + pd.Timedelta(minutes=30))
    evidence = [struct("ev_1", "15m", "long", minutes=10), struct("ev_2", "15m", "long", minutes=20)]
    verdicts = tuple(EvidenceVerdict(i, Verdict.CONTRADICT, "n") for i in ("ev_1", "ev_2"))
    res = apply(prev, episode_id="EP", evidence=evidence, update=upd(evidence_verdicts=verdicts, bias=Bias(BiasDirection.SHORT, "4H", "a")), ctx=later)
    bias = res.state.bias
    assert bias.direction is BiasDirection.NEUTRAL and bias.scale == "4H" and bias.decayed == "SHORT@4H" and bias.since == later.known_at
    assert "bias_decayed:SHORT@4H:2" in res.rejections and bias.basis.startswith("code: 2 structural events against SHORT")


def test_a_same_direction_event_resets_the_count_and_the_5m_counts_only_under_a_15m_bias() -> None:
    prev = apply(None, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.SHORT, "4H", "a")), ctx=ctx()).state
    later = ctx(known_at=T2 + pd.Timedelta(minutes=40))
    evidence = [struct("ev_1", "15m", "long", minutes=10), struct("ev_2", "1H", "short", minutes=20), struct("ev_3", "15m", "long", minutes=30), struct("ev_4", "5m", "long", minutes=35)]
    verdicts = tuple(EvidenceVerdict(i, Verdict.NEUTRAL, "n") for i in ("ev_1", "ev_2", "ev_3", "ev_4"))
    res = apply(prev, episode_id="EP", evidence=evidence, update=upd(evidence_verdicts=verdicts, bias=Bias(BiasDirection.SHORT, "4H", "a")), ctx=later)
    assert res.state.bias.direction is BiasDirection.SHORT and not any(r.startswith("bias_") for r in res.rejections)
    prev15 = apply(None, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.LONG, "15m", "a")), ctx=ctx()).state
    evidence = [struct("ev_5", "5m", "short", minutes=10), struct("ev_6", "5m", "short", minutes=20)]
    verdicts = tuple(EvidenceVerdict(i, Verdict.NEUTRAL, "n") for i in ("ev_5", "ev_6"))
    res = apply(prev15, episode_id="EP", evidence=evidence, update=upd(evidence_verdicts=verdicts, bias=Bias(BiasDirection.LONG, "15m", "a")), ctx=later)
    assert res.state.bias.decayed == "LONG@15m"


def test_an_mss_against_the_bias_on_its_own_scale_ends_it_at_once() -> None:
    prev = apply(None, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.LONG, "15m", "a")), ctx=ctx()).state
    later = ctx(known_at=T2 + pd.Timedelta(minutes=15))
    res = apply(prev, episode_id="EP", evidence=[struct("ev_1", "15m", "short", kind="mss_core_confirmed", minutes=10)],
                update=upd(evidence_verdicts=(EvidenceVerdict("ev_1", Verdict.NEUTRAL, "n"),), bias=Bias(BiasDirection.LONG, "15m", "a")), ctx=later)
    assert res.state.bias.decayed == "LONG@15m" and "bias_decayed:LONG@15m:1" in res.rejections


def test_a_decayed_bias_is_reasserted_by_structure_on_its_scale_only() -> None:
    decayed = make_state(bias=Bias(BiasDirection.NEUTRAL, "4H", "code", since=T2, decayed="SHORT@4H"), watch_next=(), destination_candidates=())
    later = ctx(known_at=T2 + pd.Timedelta(minutes=30))
    res = apply(decayed, episode_id="EP", evidence=[struct("ev_1", "15m", "short", minutes=10)],
                update=upd(evidence_verdicts=(EvidenceVerdict("ev_1", Verdict.SUPPORT, "n"),), bias=Bias(BiasDirection.SHORT, "4H", "again")), ctx=later)
    assert res.state.bias == decayed.bias and "bias_reassert_refused:SHORT@4H" in res.rejections
    res = apply(decayed, episode_id="EP", evidence=[struct("ev_2", "4H", "short", kind="qualified_bos", minutes=20)],
                update=upd(evidence_verdicts=(EvidenceVerdict("ev_2", Verdict.SUPPORT, "n"),), bias=Bias(BiasDirection.SHORT, "4H", "again")), ctx=later)
    assert res.state.bias.direction is BiasDirection.SHORT and res.state.bias.since == later.known_at and res.state.bias.decayed is None
    res = apply(decayed, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.LONG, "15m", "other")), ctx=later)
    assert res.state.bias.direction is BiasDirection.LONG and res.state.bias.decayed is None


def test_an_opportunity_under_a_decayed_bias_is_dropped() -> None:
    prev = apply(None, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.LONG, "15m", "a")), ctx=ctx()).state
    later = ctx(known_at=T2 + pd.Timedelta(minutes=15), timeframe_of=lambda a: REG[a].timeframe)
    opp = Opportunity(OpportunityState.ACTIONABLE, TradeDirection.LONG, "FVG_5m_3", "SSL_5m_2", "BSL_1H_1", thesis_id="T1")
    res = apply(prev, episode_id="EP", evidence=[struct("ev_1", "15m", "short", kind="mss_core_confirmed", minutes=10)],
                update=upd(evidence_verdicts=(EvidenceVerdict("ev_1", Verdict.NEUTRAL, "n"),), bias=Bias(BiasDirection.LONG, "15m", "a"), opportunity=opp), ctx=later)
    assert res.state.opportunity == Opportunity() and "opportunity_against_bias:NEUTRAL" in res.rejections
```

- [x] Step 2: Run; expected: `AttributeError` / wrong scale / missing rejections.
- [x] Step 3: Implement `effective_bias` and call it after rule 2 with the post-bookkeeping ledger; set `opportunity = replace(opportunity, governing_timeframe=THESIS_SCALE_OF_BIAS[bias.scale])` for a non-NONE opportunity before rule 4; add rule 4c after the invalidation-scale check; rule 4b reads the effective bias; drop the `opportunity_scale_above_bias` branch; the draft state carries the effective bias.

```python
def _structural_items(ledger: EvidenceLedger) -> list[EvidenceItem]:
    items = [item for item in ledger.supporting + ledger.contradicting + ledger.unresolved
             if item.kind in STRUCTURAL_EVIDENCE_KINDS and item.direction]
    return sorted(items, key=lambda item: (item.known_at, item.evidence_id))


def effective_bias(prev: Bias | None, reply: Bias, ledger: EvidenceLedger, known_at: pd.Timestamp) -> tuple[Bias, tuple[str, ...]]:
    """Rule 4d: the bias the state carries — the reply's, with ``since``
    carried while the pair holds, a decayed pair refused until its scale
    prints structure in its direction again, and a decay when the scales
    below deliver against it (``BIAS_DECAY_SCALES`` / ``BIAS_DECAY_EVENTS``)."""
    rejections: list[str] = []
    if reply.direction is BiasDirection.NEUTRAL:
        if prev is not None and prev.direction is BiasDirection.NEUTRAL:
            return replace(reply, scale=prev.scale, since=prev.since, decayed=prev.decayed), ()
        return replace(reply, since=known_at, decayed=None), ()
    if prev is not None and prev.decayed == reply.pair and prev.since is not None:
        confirmed = any(
            item.timeframe == reply.scale and item.kind in REVERSAL_EVIDENCE_KINDS
            and item.direction.upper() == reply.direction.value and item.known_at >= prev.since
            for item in _structural_items(ledger)
        )
        if not confirmed:
            return prev, (f"bias_reassert_refused:{reply.pair}",)
    since = prev.since if prev is not None and prev.pair == reply.pair and prev.since is not None else known_at
    bias = replace(reply, since=since, decayed=None)
    scales = BIAS_DECAY_SCALES[bias.scale]
    against = 0
    for item in _structural_items(ledger):
        if item.timeframe not in scales or item.known_at < since:
            continue
        if item.direction.upper() == bias.direction.value:
            against = 0
            continue
        against = BIAS_DECAY_EVENTS if item.timeframe == bias.scale and item.kind in REVERSAL_EVIDENCE_KINDS else against + 1
        if against >= BIAS_DECAY_EVENTS:
            decayed = Bias(
                BiasDirection.NEUTRAL, bias.scale,
                f"code: {against} structural events against {bias.direction.value} on {'/'.join(scales)} since {isoformat_utc(since)}",
                since=known_at, decayed=bias.pair,
            )
            return decayed, (f"bias_decayed:{bias.pair}:{against}",)
    return bias, ()
```

- [x] Step 4: Run `brain/tests/test_reducer.py`, `brain/tests/test_main_brain.py`, `brain/tests/test_runtime.py`, `execution/tests/test_plan_from_state.py`, `execution/tests/test_stack_e2e.py` (the scripted replies lose `governing_timeframe`; the e2e THESIS dict too).

### Task 4: The structural exit

**Files:**
- Modify: `execution/core/stack.py`, `execution/core/order_fsm.py`, `execution/core/thesis.py`
- Test: `execution/tests/test_order_fsm.py`, `execution/tests/test_thesis.py`, `execution/tests/test_stack_e2e.py`

**Interfaces:**
- Produces: `TradingStack.structure_events(observation) -> frozenset[tuple[str, str]]`; `OrderMachine.on_bar(..., structure_events: frozenset[tuple[str, str]] = frozenset())` (no `bias_direction`); `EXIT_STRUCTURE = "structure_reversed"`; trade record `structure_reversed` (`signature`, `thesis_id`, `timeframe`, `direction`, `position`); `position_closed` with `exit_role` `structure_reversed`; the thesis closes `structure_reversed` without a cooldown; `STAT_KINDS` without `bias_reversed`.

- [x] Step 1: Failing tests — replace `test_a_bias_reversal_flattens_the_open_position_and_closes_the_thesis` with:

```python
def test_a_structural_reversal_on_the_thesis_scale_flattens_the_position_and_closes_the_thesis(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    plan = short_plan(thesis_id="T1")  # governing 15m
    m.on_bar(at(1), bar(1, 16370.0, 16380.0), plan, episode_id=EP, visible=lambda a: True)
    kinds = m.on_bar(at(2), bar(2, 16380.0, 16390.0), plan, episode_id=EP, visible=lambda a: True)
    assert "position_opened" in kinds
    # a 5m MSS long, a 15m MSS short and a 1H MSS long change nothing
    kinds = m.on_bar(at(3), bar(3, 16380.0, 16390.0), None, episode_id=EP, visible=lambda a: True,
                     structure_events=frozenset({("5m", "LONG"), ("15m", "SHORT"), ("1H", "LONG")}))
    assert kinds == () and m.state is MachineState.IN_POSITION
    kinds = m.on_bar(at(4), bar(4, 16380.0, 16390.0), None, episode_id=EP, visible=lambda a: True, structure_events=frozenset({("15m", "LONG")}))
    assert "structure_reversed" in kinds and "position_closed" not in kinds
    kinds = m.on_bar(at(5), bar(5, 16380.0, 16390.0), None, episode_id=EP, visible=lambda a: True)  # the open is 16385
    assert "position_closed" in kinds and "flattened" in kinds and m.state is MachineState.IDLE
    closed = [r for r in JournalReader(tmp_path).records(EP) if r.record == "trade" and r.payload["kind"] == "position_closed"][0]
    assert closed.payload["exit_role"] == "structure_reversed" and closed.payload["exit_price"] == 16385.0
    view = m.execution_view()
    assert view["theses"][0]["closed_reason"] == "structure_reversed" and view["cooldown_bars_left"] == 0
    JournalReader(tmp_path).verify_chain(EP, run_id="fsm")
```

and in `test_stack_e2e.py` a test that `TradingStack.structure_events` maps a `MarketEvent` of kind `MSS_CORE_CONFIRMED` on `Timeframe.M15` with `Direction.LONG` to `{("15m", "LONG")}` and ignores a `DISPLACEMENT_OBSERVED`; `test_thesis.py`: `outcome(..., exit_role="structure_reversed")` closes the record `structure_reversed` with `cooldown_bars_left == 0`.

- [x] Step 2: Run; expected: unexpected keyword `structure_events`, `bias_direction` still required, no `structure_events` on the stack.
- [x] Step 3: Implement — `stack.py`:

```python
    @staticmethod
    def structure_events(observation: MarketObservation) -> frozenset[tuple[str, str]]:
        """The (scale, direction) of this bar's MSS / qualified-BOS events."""
        return frozenset(
            (event.timeframe.value, event.direction.value.upper())
            for event in observation.events_this_update
            if event.kind in (EventKind.MSS_CORE_CONFIRMED, EventKind.QUALIFIED_BOS) and event.direction is not None
        )
```

passed as `structure_events=self.structure_events(observation)`; `order_fsm.py`: `EXIT_STRUCTURE`, `_structure_reversed(events, asof, kinds, episode_id)` replacing `_bias_reversed` (a position whose `plan.governing_timeframe == tf` and `plan.direction.value != direction`), `STAT_KINDS`, the flatten's `exit_role` set, the `on_bar` signature; `thesis.py`: `("structure_reversed", "event_sleep")` in `outcome`, docstrings.

- [x] Step 4: Run `execution/tests`, `brain/tests/test_runtime.py`.

### Task 5: Prompt, summarizer, memory of the words

**Files:**
- Modify: `brain/configs/prompts/main_brain_system.md`, `brain/scripts/summarize_run.py`
- Test: `brain/tests/test_main_brain.py`, `brain/tests/test_summarize_run.py`

- [x] Step 1: Failing tests: the prompt names `bias_decayed`, `bias_reassert_refused`, `structure_reversed`, "thesis scale", and no longer `governing_timeframe`; the summary's `bias` block has `decays` and `reasserts_refused`, the order counts `structure_reversed`.
- [x] Step 2: Run; expected failures on the words and keys.
- [x] Step 3: Rewrite the prompt sections per spec §1.6; the summarizer per §1.7.
- [x] Step 4: Run `brain/tests`, then the full suite (`pytest -p no:cacheprovider`).

### Task 6: Docs

- [x] Patch `risk/docs/README.md`, `execution/docs/README.md`, `brain/docs/README.md` per spec §1.8; run `brain/tests/test_main_brain.py` and `brain/tests/test_run_guards.py` once more.

### Task 7: Runs and receipts

- [x] Frozen window and the ten benchmark windows with `--label scale-exit-bias` (pass 1, superseded by the episode-boundary fix and the review) and `--label scale-exit-bias-2` (pass 2: `2cd6fd64acccccfa` and the ten ids in `benchmark_ids_scale_exit_bias.txt`) (`launch_scale_exit_bias_day.sh`, `launch_benchmark_scale_exit_bias.sh` in the scratchpad).
- [x] Summaries against `f4f6998a6ed59eb8` and `benchmark_ids_stop_floor.txt`; receipts `brain/docs/evidence/2026-09-22_scale_exit_bias_frozen_window_2022-01-03.md` and `…_benchmark_2022.md`; README receipts index.
- [x] Re-freeze `brain/docs/evidence/regression_baselines.json` (eleven pass-2 runs, `write_baselines_scale_exit_bias.py frozen=2cd6fd64acccccfa`); research regression test (13 passed, 27.7 min); memory note; final report.
