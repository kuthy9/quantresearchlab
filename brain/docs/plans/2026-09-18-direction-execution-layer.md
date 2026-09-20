# Direction fix — Execution layer implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a right-side thesis is not killed by limits price never revisits, and a limit through the market fills where the market is.

**Architecture:** three small rule changes in the executor stack (`ThesisBook.outcome` refunds an expiry; `SimulatedExecutor.poll` fills a marketable limit at the open; `OrderMachine.on_bar` flattens a position the bias has turned against — Task 4, added 2026-09-19 after run B′, spec §3) and one metric (`missed_trends`) in the summarizer; validated by unit tests and run X, which becomes the regression baseline.

**Tech Stack:** Python 3.12, pytest (`env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest <paths> -p no:cacheprovider`, never `-q`).

**Spec:** [brain/docs/specs/2026-09-18-direction-eye-brain-execution-design.md](../specs/2026-09-18-direction-eye-brain-execution-design.md) §3.

## Global Constraints

- Starts only after run B′ is summarized and its receipt written (done 2026-09-19).
- No new order types; IBKR untouched (a limit through the market fills at the market there already).
- A cancel by the Brain (plan dropped, signature changed) still counts as an expression.

---

### Task 1: An expiry gives its expression back

**Files:**
- Modify: `execution/core/thesis.py` (`outcome`, the module docstring)
- Test: `execution/tests/test_thesis.py`

- [ ] **Step 1: Write the failing test**

```python
def test_an_expiry_gives_the_expression_back_but_a_cancel_does_not() -> None:
    b = book()
    first = plan()
    assert b.admit(first, 0) is None
    b.expressed(first)
    b.outcome(first, "expired", exit_role=None, bar_index=16)
    second = plan(target_id="swing:d")
    assert b.admit(second, 17) is None
    b.expressed(second)
    b.outcome(second, "expired", exit_role=None, bar_index=33)
    third = plan(target_id="swing:e")
    assert b.admit(third, 34) is None, "two expiries spent nothing"
    b.expressed(third)
    b.outcome(third, "cancelled", exit_role=None, bar_index=40)
    fourth = plan(target_id="swing:f")
    assert b.admit(fourth, 41) is None
    b.expressed(fourth)
    assert b.admit(plan(target_id="swing:g"), 42) == "thesis_engaged"
    b.outcome(fourth, "cancelled", exit_role=None, bar_index=43)
    assert b.admit(plan(target_id="swing:h"), 44) == "expressions_exhausted", "two cancels are two expressions"
```

- [ ] **Step 2: Run to verify it fails** — the third `admit` returns `expressions_exhausted`.
- [ ] **Step 3: Implement** — in `outcome`, after the record's `last_outcome`: `if kind == "expired" and record is not None: record.expressions = max(0, record.expressions - 1)`. Docstring: "An expiry gives its expression back (the entry was never reached); a cancel keeps it (the Brain changed its mind)."
- [ ] **Step 4: Run** `execution/tests/test_thesis.py execution/tests/test_order_fsm.py` — all passed.

---

### Task 2: A marketable limit fills at the open

**Files:**
- Modify: `execution/core/simulated_executor.py` (`poll`, the entry fill: `_touched` + fill price; the module docstring's fill rule)
- Test: `execution/tests/test_simulated_executor.py`

- [ ] **Step 1: Write the failing test**

```python
def test_a_limit_through_the_market_fills_at_the_open() -> None:
    ex = executor()
    intent = BracketIntent("EP_1:sig", "NQ", "BUY", 1, 16360.0, 16330.0, 16400.0, "sig")  # limit above the next open
    events = run(ex, intent, [bar(1, 16352.0, 16358.0, close=16355.0)])
    fills = [e for e in events if e.kind == "filled"]
    assert fills and fills[0].fill.price == 16355.0, "a BUY limit above the open fills at the open, not at the limit"
    ex = executor()
    intent = BracketIntent("EP_1:sig", "NQ", "SELL", 1, 16350.0, 16380.0, 16300.0, "sig")  # limit below the next open
    events = run(ex, intent, [bar(1, 16352.0, 16358.0, close=16355.0)])
    fills = [e for e in events if e.kind == "filled"]
    assert fills and fills[0].fill.price == 16355.0
```

(`bar()` sets `open = close`; check `BrokerEvent`'s field names in `contract/execution` before asserting on `e.kind` / `e.fill`.)

- [ ] **Step 2: Run to verify it fails** — the fill price equals the limit.
- [ ] **Step 3: Implement** — the entry's fill price becomes `min(limit, open)` for a BUY and `max(limit, open)` for a SELL:

```python
                limit = float(entry.limit_price)
                price = min(limit, float(bar.open)) if entry.side == "BUY" else max(limit, float(bar.open))
                filled, fill = self._fill(entry, quantity, price, asof)
```

Docstring rule: "a limit through the market (BUY at or above the open, SELL at or below) fills at the open".

- [ ] **Step 4: Run** `execution/tests` — all passed.

---

### Task 3: `missed_trends` in the summarizer

**Files:**
- Modify: `brain/scripts/summarize_run.py` (collect per expired entry: the limit, the stop distance, the direction, the submission and expiry times; with bars, count those where the tape travelled ≥ 1 R in the thesis direction from the limit without touching it)
- Test: `brain/tests/test_summarize_run.py`

- [ ] **Step 1: Write the failing test**

```python
def test_missed_trends_count_expiries_the_tape_ran_away_from() -> None:
    start = pd.Timestamp("2022-01-03T14:00:00Z")
    bars = [Bar(start=start + pd.Timedelta(minutes=i), open=16450.0 + i, high=16451.0 + i, low=16449.5 + i, close=16450.5 + i, volume=1.0, symbol="NQ", instrument_id=1) for i in range(40)]
    expiries = [
        {"direction": "LONG", "limit_price": 16449.25, "stop_price": 16439.5, "submitted_at": start + pd.Timedelta(minutes=1), "expired_at": start + pd.Timedelta(minutes=16)},   # never touched, ran +10 pts ≥ 1 R (9.75)
        {"direction": "SHORT", "limit_price": 16470.0, "stop_price": 16480.0, "submitted_at": start + pd.Timedelta(minutes=1), "expired_at": start + pd.Timedelta(minutes=16)},  # ran against it
    ]
    assert missed_trends(expiries, bars) == {"expired": 2, "missed": 1}
```

- [ ] **Step 2: Run to verify it fails** — ImportError.
- [ ] **Step 3: Implement** — `missed_trends(expiries, bars)` (module function): for each expiry, the bars in `[submitted_at, expired_at]`; touched = any low ≤ limit (LONG) / high ≥ limit (SHORT); travelled = max high − first open (LONG) / first open − min low (SHORT), the first open being the price at submission; missed when not touched and travelled ≥ |limit − stop|. In `summarize()`, collect the expiries from `submitted` (limit, stop, direction, time) matched to their `expired` record by signature; `orders` gains `"missed_trends": missed_trends(expiries, bars) if bars else None`; `render` prints it.
- [ ] **Step 4: Run** `brain/tests/test_summarize_run.py` — all passed.

---

### Task 4: A bias reversal against an open position flattens it

**Files:**
- Modify: `execution/core/order_fsm.py` (`STAT_KINDS` += `"bias_reversed"`; `EXIT_BIAS = "bias_reversed"`; `on_bar(..., bias_direction: str | None = None)`; `_bias_reversed()` called after `_close_beyond`; the FLATTEN fill maps `exit_requested` to its exit role), `execution/core/thesis.py` (`outcome`: `exit_role == "bias_reversed"` closes the thesis with `closed_reason = "bias_reversed"`, no cooldown), `execution/core/stack.py` (passes `state.bias.direction.value` when LONG/SHORT, else `None`)
- Test: `execution/tests/test_order_fsm.py`, `execution/tests/test_thesis.py`

**Interfaces:**
- Consumes: `BrainState.bias` (`contract/brain/state.py`, `BiasDirection`).
- Produces: journal trade kind `bias_reversed` `{signature, thesis_id, bias, position}`; `position_closed` with `exit_role = "bias_reversed"`; `ThesisRecord.closed_reason = "bias_reversed"`.

- [ ] **Step 1: Write the failing test** (`test_order_fsm.py`)

```python
def test_a_bias_reversal_flattens_the_open_position_and_closes_the_thesis(tmp_path: Path) -> None:
    m, broker, journal = machine(tmp_path)
    plan = short_plan(thesis_id="T1")
    m.on_bar(at(1), bar(1, 16370.0, 16380.0), plan, episode_id=EP, visible=lambda a: True, bias_direction="SHORT")
    kinds = m.on_bar(at(2), bar(2, 16380.0, 16390.0), plan, episode_id=EP, visible=lambda a: True, bias_direction="SHORT")
    assert "position_opened" in kinds
    # NEUTRAL is not a reversal, nor is the Brain dropping the opportunity
    kinds = m.on_bar(at(3), bar(3, 16380.0, 16390.0), None, episode_id=EP, visible=lambda a: True, bias_direction=None)
    assert kinds == () and m.state is MachineState.IN_POSITION
    # LONG against a SHORT position: flattened at market, filled at the next open
    kinds = m.on_bar(at(4), bar(4, 16380.0, 16390.0), None, episode_id=EP, visible=lambda a: True, bias_direction="LONG")
    assert "bias_reversed" in kinds and "position_closed" not in kinds
    kinds = m.on_bar(at(5), bar(5, 16380.0, 16390.0, close=16385.0), None, episode_id=EP, visible=lambda a: True, bias_direction="LONG")
    assert "position_closed" in kinds and "flattened" in kinds and m.state is MachineState.IDLE
    closed = [r for r in JournalReader(tmp_path).records(EP) if r.record == "trade" and r.payload["kind"] == "position_closed"][0]
    assert closed.payload["exit_role"] == "bias_reversed" and closed.payload["exit_price"] == 16385.0
    view = m.execution_view()
    assert view["theses"][0]["closed_reason"] == "bias_reversed" and view["cooldown_bars_left"] == 0
```

- [ ] **Step 2: Run to verify it fails** — `TypeError: on_bar() got an unexpected keyword argument 'bias_direction'`.
- [ ] **Step 3: Implement** — in `on_bar`, after `_close_beyond`:

```python
        if bias_direction in ("LONG", "SHORT"):
            self._bias_reversed(bias_direction, asof, kinds, episode_id)
```

```python
    def _bias_reversed(self, bias_direction: str, asof: pd.Timestamp, kinds: list[str], episode_id: str | None) -> None:
        """The Brain's bias is the opposite side of an open position: the
        thesis has lost its direction (rule 4b would refuse it as a new
        opportunity), so the position is flattened at market."""
        for intent in list(self._intents.values()):
            if intent.position is None or intent.exit_requested is not None or intent.plan.direction.value == bias_direction:
                continue
            self._record(kinds, "bias_reversed", episode_id, asof, {
                "signature": intent.plan.signature, "thesis_id": intent.plan.thesis_id, "bias": bias_direction, "position": intent.position.to_dict(),
            })
            self._flatten(intent, asof, EXIT_BIAS, kinds, episode_id)
```

and the FLATTEN fill: `exit_role = intent.exit_requested if intent.exit_requested in (EXIT_INVALIDATION, EXIT_BIAS) else EXIT_HALT`. In `ThesisBook.outcome`: `elif exit_role == "bias_reversed" and record is not None and record.closed_reason is None: record.closed_reason = "bias_reversed"`. In `TradingStack.step`: `bias_direction=None if state is None or state.bias.direction is BiasDirection.NEUTRAL else state.bias.direction.value`.

- [ ] **Step 4: Run** `execution/tests` — all passed.

---

### Task 5: Docs, verification, run X

- [ ] `execution/docs/README.md` (the thesis-book rules table: expiry refunds, the bias-reversal exit; the simulator's fill rules), `execution/docs/specs/2026-09-17-thesis-lifecycle-risk-v2-design.md` §6 amendment, `brain/docs/README.md` (summarizer `missed_trends`, the `bias_reversed` kind), `git diff --check`.
- [ ] Full suite.
- [ ] Launch run **X** with the frozen command; summarize against runs B, E and `72ea13c7`; replay it (`replay_journal.py --run-dir …`) and, when it reproduces, add it to `brain/docs/evidence/regression_baselines.json` (its note explains the field shape); run `brain/tests/test_regression_baseline.py -m research_orchestration -o addopts=''`.
- [ ] Launch the 2022-01-04 run (`--emit-start 2022-01-03T18:00 --end 2022-01-04T17:00`, warm-up 2021-12-28) and summarize it — the day none of this was tuned on.
- [ ] Receipt `brain/docs/evidence/2026-09-18_execution_entry_day_run_2022-01-03.md` (+ the 2022-01-04 numbers), index, memory; the layered comparison table (72ea13c7 → E → B → B′ → X) closes the build.
