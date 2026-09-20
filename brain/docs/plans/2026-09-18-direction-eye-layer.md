# Direction fix — Eye layer implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** make the per-scale facts the Brain reads describe the leg that is forming, carry the age and direction of a displacement, and name a broken protection — then measure what those facts alone do on the frozen backtest.

**Architecture:** `TimeframeDeliveryState` / `TimeframeStructureState` gain fields; both producers of a timeframe state (`reduce_timeframe_state`, the atomic authority in production, and `MarketSnapshotPublisher._timeframe_state`, the projection) apply one rule; `brain/core/eye_view.py` publishes the derived keys (ages, ATR-scaled excursion, reset) under input schema 2; `brain/scripts/audit_scales.py` prints the change points of those facts over a window without an LLM.

**Tech Stack:** Python 3.12, dataclasses, pandas, pytest (`env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest <paths> -p no:cacheprovider`, never `-q`).

**Spec:** [brain/docs/specs/2026-09-18-direction-eye-brain-execution-design.md](../specs/2026-09-18-direction-eye-brain-execution-design.md) §1 (and §2.3's phase-bookkeeping item, moved here because this layer changes what a phase transition means).

## Global Constraints

- No change to swings, BOS, MSS, dealing ranges or their protocols.
- New dataclass fields default to `None` so every existing constructor keeps working.
- Both producers implement the same rule; `test_snapshot_projection_replays_to_same_hierarchical_state` must stay green.
- No prices in the LLM contract: the view publishes ages and ATR-scaled excursions only.
- Frozen backtest window and command: `--warmup-start 2021-12-27 --emit-start 2022-01-02T18:00 --end 2022-01-03T17:00 --max-llm-calls 1000 --reasoning-effort high --client deepseek --broker sim`.

---

### Task 1: The delivery-state fields and the phase rule (pure functions)

**Files:**
- Modify: `eyes/core/market_state.py` — `TimeframeDeliveryState` (~line 903), `TimeframeStructureState` (~884), `_delivery_phase` (~1607)
- Test: `eyes/tests/test_forming_leg.py` (new)

**Interfaces:**
- Produces: `forming_leg_direction(points: float | None, fallback: Direction | None) -> Direction | None`; fields `TimeframeDeliveryState.last_leg_direction`, `.forming_leg_points`, `.last_close`, `.displacement_direction`, `.displacement_at`; fields `TimeframeStructureState.protection_broken_direction`, `.protection_broken_at`.

- [ ] **Step 1: Write the failing tests**

```python
"""The active leg is the leg price is in now; a range price has left is not balance."""
from __future__ import annotations

import pandas as pd

from contract.eye import DeliveryPhase
from contract.market import Direction
from eyes.core.market_state import (
    TimeframeDeliveryState,
    TimeframeRangeState,
    TimeframeStructureState,
    _delivery_phase,
    forming_leg_direction,
)


def _structure(external, internal, *, intact=True, last_mss=None):
    return TimeframeStructureState(
        external_direction=external, internal_direction=internal,
        protected_low=None, protected_high=None, protected_low_id=None, protected_high_id=None,
        protected_swing_intact=intact, last_bos=None, last_bos_direction=None, last_mss=None, last_mss_direction=last_mss,
    )


def _range(location):
    return TimeframeRangeState("range", 90.0, 110.0, location, "equilibrium", "active", ("low", "high"), "active_dealing_range")


def test_forming_leg_direction_follows_the_sign_of_the_excursion() -> None:
    assert forming_leg_direction(3.0, None) is Direction.LONG
    assert forming_leg_direction(-0.25, None) is Direction.SHORT
    assert forming_leg_direction(0.0, Direction.LONG) is None
    assert forming_leg_direction(None, Direction.SHORT) is Direction.SHORT


def test_new_fields_default_to_none_and_are_validated() -> None:
    state = TimeframeDeliveryState(phase=DeliveryPhase.TRANSITION, active_leg_direction=None, displacement_score=None)
    assert state.last_leg_direction is None and state.forming_leg_points is None and state.last_close is None
    assert state.displacement_direction is None and state.displacement_at is None
    structure = _structure(None, None)
    assert structure.protection_broken_direction is None and structure.protection_broken_at is None


def test_no_external_direction_is_balance_only_inside_the_active_range() -> None:
    structure = _structure(None, Direction.SHORT)
    assert _delivery_phase(structure, _range(0.5), Direction.LONG) is DeliveryPhase.BALANCE
    assert _delivery_phase(structure, _range(1.3), Direction.LONG) is DeliveryPhase.TRANSITION
    assert _delivery_phase(structure, _range(None), Direction.LONG) is DeliveryPhase.TRANSITION


def test_the_phase_reads_the_active_leg_against_the_external_direction() -> None:
    short = _structure(Direction.SHORT, Direction.SHORT)
    assert _delivery_phase(short, _range(0.5), Direction.SHORT) is DeliveryPhase.EXPANSION
    assert _delivery_phase(short, _range(0.5), Direction.LONG) is DeliveryPhase.RETRACEMENT
    assert _delivery_phase(_structure(Direction.SHORT, Direction.SHORT, intact=False), _range(0.5), Direction.LONG) is DeliveryPhase.TRANSITION
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest eyes/tests/test_forming_leg.py -p no:cacheprovider`
Expected: ImportError on `forming_leg_direction` (and `TypeError`/`AttributeError` for the new fields).

- [ ] **Step 3: Implement**

In `eyes/core/market_state.py`:

```python
def forming_leg_direction(points: float | None, fallback: Direction | None) -> Direction | None:
    """The leg price is in now: the sign of the excursion from the last
    confirmed leg's end swing; ``fallback`` (the internal direction) when no
    leg has been confirmed; ``None`` when the close sits on the swing."""
    if points is None:
        return fallback
    if points > 0.0:
        return Direction.LONG
    if points < 0.0:
        return Direction.SHORT
    return None
```

`TimeframeStructureState`: append `protection_broken_direction: Direction | None = None` and `protection_broken_at: pd.Timestamp | None = None` after the event-id fields.

`TimeframeDeliveryState`: append

```python
    # 2026-09-18: the leg price is in now (``active_leg_direction`` is its
    # direction), the confirmed leg it left, and the last displacement's
    # direction and time so a reader can age it.
    last_leg_direction: Direction | None = None
    forming_leg_points: float | None = None
    last_close: float | None = None
    displacement_direction: Direction | None = None
    displacement_at: pd.Timestamp | None = None
```

and in `__post_init__` reject a non-finite `forming_leg_points` / `last_close`.

`_delivery_phase`: the `external is None` branch becomes

```python
    if external is None:
        inside = (
            range_state.normalized_location is not None
            and 0.0 <= float(range_state.normalized_location) <= 1.0
        )
        return (
            DeliveryPhase.BALANCE
            if inside
            and range_state.range_id is not None
            and range_state.range_kind == "active_dealing_range"
            and range_state.lifecycle not in {"broken", "invalidated"}
            else DeliveryPhase.TRANSITION
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: the same command. Expected: 4 passed. Then `eyes/tests/test_hierarchical_market_state.py` (its `_empty_state` helper uses the old constructors): all passed.

---

### Task 2: The reducer path (atomic authority)

**Files:**
- Modify: `eyes/core/market_state.py` — `reduce_timeframe_state`: BAR_COMPLETED (~2440), STRUCTURE_DIRECTION_CONFIRMED (~2477), STRUCTURAL_LEG_CREATED (~2500), QUALIFIED_BOS (~2508), the exact-protection acceptance (~2607), DISPLACEMENT_OBSERVED (~2646), the phase recompute (~2903)
- Test: `eyes/tests/test_forming_leg.py`

**Interfaces:**
- Produces: `_settle_delivery(delivery, structure, range_state, legs) -> TimeframeDeliveryState` (module function) used by the recompute; the reducer keeps `last_close` on every BAR_COMPLETED (including the M1 owner fan-out) and `last_leg_direction` on every leg.

- [ ] **Step 1: Write the failing tests** (append to `eyes/tests/test_forming_leg.py`; the event helper mirrors `eyes/tests/test_liquidity_level_rearm.py`)

```python
from contract.eye import EventKind, EventOrigin, MarketEvent
from contract.market import SMC_SEMANTIC_VERSION, Timeframe
from eyes.core.market_state import reduce_timeframe_state

TZ = "America/New_York"


def _clock(minutes: int) -> pd.Timestamp:
    return pd.Timestamp("2025-01-06 10:00", tz=TZ) + pd.Timedelta(minutes=minutes)


def _event(kind, minutes, *, price=100.0, side=None, direction=None, evidence=None):
    known_at = _clock(minutes)
    return MarketEvent(
        event_id=f"{minutes:04d}:{kind.value}", kind=kind, observed_at=known_at, timeframe=Timeframe.M5,
        side=side, price=price, strength=0.75, details={} if evidence is None else evidence, sequence_no=0,
        event_time=known_at, known_at=known_at, semantic_version=SMC_SEMANTIC_VERSION,
        origin=EventOrigin.LEGACY_TRANSPORT, direction=direction,
    )


def _bar(minutes, close):
    return _event(EventKind.BAR_COMPLETED, minutes, price=close, evidence={
        "close": close, "atr": 2.0, "data_complete": True, "real_completed": True, "clock_only": False,
        "event_category": "normalized_data", "source_data_ids": (f"bar:m5:{minutes}",),
    })


def _leg(minutes, direction, *, start_price, end_price):
    return _event(EventKind.STRUCTURAL_LEG_CREATED, minutes, price=end_price, direction=direction,
                  side="above" if direction is Direction.LONG else "below", evidence={
        "leg_id": f"leg:{minutes}", "start_swing_id": "swing:a", "end_swing_id": "swing:b",
        "start_event_time": _clock(minutes - 30).isoformat(), "end_event_time": _clock(minutes - 10).isoformat(),
        "start_price": start_price, "end_price": end_price, "start_close": start_price, "end_close": end_price,
        "amplitude_points": abs(end_price - start_price), "amplitude_atr": abs(end_price - start_price) / 2.0,
        "duration_bars": 4, "duration_minutes": 20, "efficiency": 0.8, "max_retracement_points": 0.5,
        "max_retracement_atr": 0.25, "path_class": "external",
    })


def _displacement(minutes, direction):
    return _event(EventKind.DISPLACEMENT_OBSERVED, minutes, direction=direction,
                  side="above" if direction is Direction.LONG else "below",
                  evidence={"state_metrics": {"efficiency": 0.8, "relative_atr": 1.0}})


def _reduce(events):
    state = None
    for event in events:
        state = reduce_timeframe_state(state, event, semantic_registry_identity="definition-test")
    assert state is not None
    return state


def test_the_active_leg_is_the_leg_price_is_in_now() -> None:
    state = _reduce((_bar(0, 104.0), _leg(5, Direction.SHORT, start_price=108.0, end_price=100.0), _bar(10, 103.0)))
    assert state.delivery.last_leg_direction is Direction.SHORT
    assert state.delivery.forming_leg_points == 3.0
    assert state.delivery.active_leg_direction is Direction.LONG
    state = reduce_timeframe_state(state, _bar(15, 99.0), semantic_registry_identity="definition-test")
    assert state.delivery.active_leg_direction is Direction.SHORT and state.delivery.last_close == 99.0


def test_a_leg_created_after_the_bar_reads_the_last_close() -> None:
    state = _reduce((_bar(0, 104.0), _leg(5, Direction.SHORT, start_price=108.0, end_price=100.0)))
    assert state.delivery.forming_leg_points == 4.0 and state.delivery.active_leg_direction is Direction.LONG


def test_a_displacement_carries_its_direction_and_time() -> None:
    state = _reduce((_bar(0, 100.0), _displacement(5, Direction.SHORT)))
    assert state.delivery.displacement_direction is Direction.SHORT
    assert state.delivery.displacement_at == _clock(5)
    assert state.delivery.displacement_score is not None
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest eyes/tests/test_forming_leg.py -p no:cacheprovider`
Expected: the three new tests fail on `last_leg_direction is None` / `forming_leg_points is None` / `displacement_direction is None`. If `MarketEvent` has no `direction` keyword, read `contract/eye/observation.py:25` and pass it the way that class expects.

- [ ] **Step 3: Implement**

Module function (next to `_delivery_phase`):

```python
def _settle_delivery(
    delivery: TimeframeDeliveryState,
    structure: TimeframeStructureState,
    range_state: TimeframeRangeState,
    legs: Sequence[StructuralLegState],
) -> TimeframeDeliveryState:
    """Derive the active (forming) leg and the phase from the last close, the
    confirmed legs and the structure — the one rule both producers use."""
    points = (
        None
        if not legs or delivery.last_close is None
        else float(delivery.last_close) - float(legs[-1].end_price)
    )
    active = forming_leg_direction(points, None if legs else structure.internal_direction)
    return replace(
        delivery,
        forming_leg_points=points,
        last_leg_direction=legs[-1].direction if legs else None,
        active_leg_direction=active,
        phase=_delivery_phase(structure, range_state, active),
    )
```

Reducer edits:
- BAR_COMPLETED: right after `close = float(...)`: `delivery = replace(delivery, last_close=close)` (for the M1 fan-out too — it is a price update).
- STRUCTURAL_LEG_CREATED: drop `delivery = replace(delivery, active_leg_direction=leg.direction)`; `_settle_delivery` derives it.
- DISPLACEMENT_OBSERVED: add `displacement_direction=event.direction, displacement_at=event.known_at` to the `replace`.
- Exact-protection acceptance (`external_direction=None, protected_swing_intact=False`): add `protection_broken_direction=event.direction, protection_broken_at=event.known_at`.
- STRUCTURE_DIRECTION_CONFIRMED (both `replace` calls that set `external_direction=event.direction`) and QUALIFIED_BOS: add `protection_broken_direction=None, protection_broken_at=None`.
- The recompute at the end: replace `delivery = replace(delivery, phase=_delivery_phase(structure, range_state, delivery.active_leg_direction))` with `delivery = _settle_delivery(delivery, structure, range_state, legs)`.

- [ ] **Step 4: Run to verify they pass**

Run: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest eyes/tests/test_forming_leg.py eyes/tests/test_hierarchical_market_state.py eyes/tests/test_liquidity_level_rearm.py -p no:cacheprovider`
Expected: all passed.

---

### Task 3: The projection path and the protection break in the publisher

**Files:**
- Modify: `eyes/core/market_state.py` — `MarketSnapshotPublisher._phase` (~5857), `_timeframe_state` delivery construction (~5958), `_displacement` (~5776), `_formalize_structure` (~5640–5760)
- Test: `eyes/tests/test_hierarchical_market_state.py`, `eyes/tests/test_facts_on_every_scale.py`

- [ ] **Step 1: Write the failing tests**

In `eyes/tests/test_hierarchical_market_state.py`, next to `test_delivery_phase_uses_one_rule_for_invalidated_active_range`:

```python
def _leg_state(direction: Direction, *, start_price: float, end_price: float) -> StructuralLegState:
    when = _clock("2025-01-06 09:30")
    return StructuralLegState(
        leg_id="leg:test", timeframe=Timeframe.H1, direction=direction, start_swing_id="a", end_swing_id="b",
        start_event_time=when, end_event_time=when, known_at=when, start_price=start_price, end_price=end_price,
        start_close=start_price, end_close=end_price, amplitude_points=abs(end_price - start_price),
        amplitude_atr=1.0, duration_bars=4, duration_minutes=240, efficiency=0.8, max_retracement_points=0.5,
        max_retracement_atr=0.25, path_class=SwingRank.EXTERNAL, source_swing_ids=("a", "b"),
    )


def test_the_projected_phase_reads_the_forming_leg_against_price() -> None:
    short = _empty_state(Timeframe.H1, direction=Direction.SHORT, internal=Direction.SHORT, protected_low=None)
    structure = replace(short.structure, protected_high=110.0, protected_high_id="protected-high", protected_swing_intact=True)
    leg = _leg_state(Direction.SHORT, start_price=108.0, end_price=100.0)
    assert MarketSnapshotPublisher._phase(structure, short.range, (leg,), price=103.0) is DeliveryPhase.RETRACEMENT
    assert MarketSnapshotPublisher._phase(structure, short.range, (leg,), price=97.0) is DeliveryPhase.EXPANSION
```

(`StructuralLegState` and `SwingRank` are imported from where `eyes/core/market_state.py` gets them; check the fields against `_leg_from_event` at `market_state.py:2323` and adjust the helper's keyword names to that dataclass.)

In `eyes/tests/test_facts_on_every_scale.py` (the `replay` fixture runs the full Eye on a noisy walk, so the published states come from the atomic authority):

```python
def test_the_active_leg_is_the_sign_of_the_excursion_from_the_last_leg(replay) -> None:
    _, observations = replay
    checked = 0
    for observation in observations:
        for state in observation.market_snapshot.timeframe_states.values():
            delivery = state.delivery
            if not state.structural_legs or delivery.last_close is None:
                continue
            points = float(delivery.last_close) - float(state.structural_legs[-1].end_price)
            assert delivery.forming_leg_points == points
            assert delivery.last_leg_direction is state.structural_legs[-1].direction
            expected = Direction.LONG if points > 0 else Direction.SHORT if points < 0 else None
            assert delivery.active_leg_direction is expected
            if delivery.phase is DeliveryPhase.EXPANSION:
                assert delivery.active_leg_direction is state.structure.external_direction
            checked += 1
    assert checked, "no scale ever carried a confirmed leg"


def test_a_displacement_is_dated_and_directed(replay) -> None:
    _, observations = replay
    dated = 0
    for observation in observations:
        for state in observation.market_snapshot.timeframe_states.values():
            if state.delivery.displacement_score is None:
                continue
            assert state.delivery.displacement_direction is not None
            assert state.delivery.displacement_at is not None
            assert state.delivery.displacement_at <= observation.asof
            dated += 1
    assert dated


def test_a_broken_protection_is_named_until_a_structure_confirms(replay) -> None:
    _, observations = replay
    broken = 0
    for observation in observations:
        for state in observation.market_snapshot.timeframe_states.values():
            structure = state.structure
            if structure.external_direction is None and structure.protected_swing_intact is False:
                assert structure.protection_broken_direction is not None
                assert structure.protection_broken_at is not None
                broken += 1
            if structure.external_direction is not None:
                assert structure.protection_broken_direction is None
    assert broken, "the walk never broke a protected swing; widen the walk before weakening the test"
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest eyes/tests/test_hierarchical_market_state.py eyes/tests/test_facts_on_every_scale.py -p no:cacheprovider`
Expected: the projected-phase test fails (`_phase` takes no `price`); the replay tests fail on `last_leg_direction`/`displacement_direction`/`protection_broken_direction` being `None`. If the broken-protection test's `assert broken` is what fails, extend the walk (`_noisy(..., seed)` or more sessions) until a protection breaks; do not drop the assertion.

- [ ] **Step 3: Implement**

`_phase(structure, range_state, legs, price=None)`:

```python
        points = None if not legs or price is None else float(price) - float(legs[-1].end_price)
        active = forming_leg_direction(points, None if legs else structure.internal_direction)
        return _delivery_phase(structure, range_state, active)
```

`_timeframe_state`: compute `points`/`active` the same way with `price`, and build

```python
            delivery=TimeframeDeliveryState(
                phase=self._phase(structure, range_state, frame.structural_legs, price=price),
                active_leg_direction=active,
                displacement_score=score,
                displacement_features=features,
                last_leg_direction=frame.structural_legs[-1].direction if frame.structural_legs else None,
                forming_leg_points=points,
                last_close=float(price),
                displacement_direction=displacement_direction,
                displacement_at=displacement_at,
            ),
```

`_displacement` returns `(score, features, direction, at)`: for the 5m live path `direction = displacement.current_direction`, `at = displacement.asof`; otherwise `(None, {}, None, None)`.

`_formalize_structure`: in the branch that preserves the invalidated protection through the directionless stretch, also carry `protection_broken_direction` / `protection_broken_at` from `prior`; in the `accepted and candidate.external_direction in {None, prior.external_direction}` branch set `protection_broken_direction=opposite, protection_broken_at=<the accepting event's known_at>` (take it from the event `accepts_exact_protection` matched); every branch that yields a non-`None` external direction leaves both `None`.

- [ ] **Step 4: Run to verify they pass**

Run: the same command plus `eyes/tests/test_forming_leg.py`. Expected: all passed, including `test_snapshot_projection_replays_to_same_hierarchical_state`.

---

### Task 4: The Brain's view of the facts (input schema 2)

**Files:**
- Modify: `brain/core/eye_view.py` (`_scale_objects` summary ~294, the 1m block ~424, `_session_payload` ~318, `build_eye_context` ~394), `contract/brain/llm.py` (`LLM_INPUT_SCHEMA_VERSION`)
- Test: `brain/tests/test_eye_view.py`, `brain/tests/test_llm_contract.py` (if it pins the version)

**Interfaces:**
- Produces per scale: `delivery.last_leg_direction`, `delivery.forming_leg_atr` (points ÷ `state.quality.atr`, `None` when unknown), `delivery.displacement_direction`, `delivery.displacement_age_bars`; `structure.reset` = `{"direction": "long"|"short", "bars_ago": int}` or `None`; `session.drift_atr`.

- [ ] **Step 1: Write the failing tests** (append to `brain/tests/test_eye_view.py`)

```python
def test_scales_publish_the_forming_leg_the_displacement_age_and_the_reset(synthetic_observations) -> None:
    registry = ObjectRegistry()
    seen_forming = False
    for observation in synthetic_observations:
        context = build_eye_context(observation, registry, rule=RULE)
        for name, scale in context.scales.items():
            delivery = scale["delivery"]
            for key in ("phase", "active_leg_direction", "last_leg_direction", "forming_leg_atr", "displacement_score", "displacement_direction", "displacement_age_bars"):
                assert key in delivery, (name, key)
            assert "reset" in scale["structure"]
            if delivery["forming_leg_atr"] is not None:
                seen_forming = True
                sign = delivery["active_leg_direction"]
                assert (delivery["forming_leg_atr"] > 0) == (sign == "long") or delivery["forming_leg_atr"] == 0
            if delivery["displacement_age_bars"] is not None:
                assert delivery["displacement_age_bars"] >= 0 and delivery["displacement_direction"] in ("long", "short")
    assert seen_forming


def test_session_drift_is_the_close_against_the_session_open_in_1m_atrs(synthetic_observations) -> None:
    observation = synthetic_observations[-1]
    context = build_eye_context(observation, ObjectRegistry(), rule=RULE)
    drift = context.session["drift_atr"]
    if context.atr_1m is None or context.session["session_open"] is None:
        assert drift is None
    else:
        assert drift == round((context.close - context.session["session_open"]) / context.atr_1m, 4)


def test_input_schema_is_2() -> None:
    from contract.brain.llm import LLM_INPUT_SCHEMA_VERSION
    assert LLM_INPUT_SCHEMA_VERSION == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest brain/tests/test_eye_view.py -p no:cacheprovider`
Expected: KeyError `last_leg_direction`, KeyError `drift_atr`, schema 1 ≠ 2.

- [ ] **Step 3: Implement**

In `eye_view.py` add

```python
from brain.core.opportunity_geometry import TIMEFRAME_MINUTES

def _age_bars(known_at: pd.Timestamp, at: pd.Timestamp | None, timeframe: Timeframe) -> int | None:
    if at is None:
        return None
    minutes = (pd.Timestamp(known_at) - pd.Timestamp(at)).total_seconds() / 60.0
    return max(0, int(minutes // TIMEFRAME_MINUTES[timeframe.value]))


def _delivery_payload(state: TimeframeState, known_at: pd.Timestamp) -> dict[str, Any]:
    delivery = state.delivery
    atr = state.quality.atr
    return {
        "phase": delivery.phase.value,
        "active_leg_direction": _direction_value(delivery.active_leg_direction),
        "last_leg_direction": _direction_value(delivery.last_leg_direction),
        "forming_leg_atr": (
            None if delivery.forming_leg_points is None or atr is None or float(atr) <= 0.0
            else _round(float(delivery.forming_leg_points) / float(atr))
        ),
        "displacement_score": _round(delivery.displacement_score),
        "displacement_direction": _direction_value(delivery.displacement_direction),
        "displacement_age_bars": _age_bars(known_at, delivery.displacement_at, state.timeframe),
    }


def _reset_payload(state: TimeframeState, known_at: pd.Timestamp) -> dict[str, Any] | None:
    structure = state.structure
    if structure.protection_broken_direction is None:
        return None
    return {
        "direction": _direction_value(structure.protection_broken_direction),
        "bars_ago": _age_bars(known_at, structure.protection_broken_at, state.timeframe),
    }
```

Use them in `_scale_objects` (`"delivery": _delivery_payload(state, known_at)`, `"reset": _reset_payload(state, known_at)` inside `"structure"`) and in the 1m block; `_scale_objects` needs `known_at` (pass it from `build_eye_context`). `_session_payload(observation, *, close, atr_1m)` adds `"drift_atr": None if atr_1m is None or session.session_open is None else _round((close - session.session_open) / atr_1m)`. `LLM_INPUT_SCHEMA_VERSION = 2`.

- [ ] **Step 4: Run to verify they pass**

Run: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest brain/tests -p no:cacheprovider`
Expected: all passed (fix any test that pinned the exact `scales` keys or the schema version by updating its expectation, not by weakening it).

---

### Task 5: Phase transitions are bookkeeping

**Files:**
- Modify: `brain/configs/sleep_controller.json` (`bookkeeping_kinds` += `delivery_phase_entered`, `delivery_phase_exited`)
- Test: `brain/tests/test_sleep_controller.py`

- [ ] **Step 1: Write the failing test**

```python
def test_phase_transitions_are_bookkeeping_not_reactions() -> None:
    for kind in ("delivery_phase_entered", "delivery_phase_exited"):
        for tf in ("5m", "15m", "1H", "4H"):
            assert not CONFIG.is_update_event(_event(kind, tf)), (kind, tf)
            assert not CONFIG.is_wake_event(_event(kind, tf)), (kind, tf)
```

(use the file's existing `CONFIG` and event helper names; read the top of the test file.)

- [ ] **Step 2: Run to verify it fails** — `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest brain/tests/test_sleep_controller.py -p no:cacheprovider` — expected: assertion on `delivery_phase_entered@15m` being an update event.
- [ ] **Step 3: Implement** — add the two kinds to `bookkeeping_kinds` in the JSON (schema stays 3; only the list changes; the config's sha256 changes and `run.json` records it).
- [ ] **Step 4: Run to verify it passes**, then `brain/tests` whole.

---

### Task 6: `brain/scripts/audit_scales.py`

**Files:**
- Create: `brain/scripts/audit_scales.py`
- Test: `brain/tests/test_audit_scales.py` (new)

**Interfaces:**
- `scale_facts(snapshot, timeframe) -> dict` — the tuple printed per scale: `external, internal, last_bos, last_mss, protected_intact, reset, active_leg, last_leg, forming_leg_atr, phase, displacement_direction, displacement_age_bars`.
- `change_points(rows) -> list[dict]` — rows `(known_at, close, facts)`; returns the rows whose facts differ from the previous row.
- CLI: `.venv/bin/python -m brain.scripts.audit_scales --warmup-start 2021-12-27 --emit-start 2022-01-02T18:00 --end 2022-01-03T17:00 [--scales 4H,1H,15m] [--source …] [--model …]` prints, per scale, one line per change point: NY time, close, the facts.

- [ ] **Step 1: Write the failing test**

```python
from brain.scripts.audit_scales import change_points, scale_facts


def test_change_points_keep_only_rows_whose_facts_moved() -> None:
    rows = [("t0", 1.0, {"a": 1}), ("t1", 2.0, {"a": 1}), ("t2", 3.0, {"a": 2})]
    assert [row[0] for row in change_points(rows)] == ["t0", "t2"]


def test_scale_facts_read_the_published_state(synthetic_observations) -> None:
    snapshot = synthetic_observations[-1].market_snapshot
    facts = scale_facts(snapshot, Timeframe.M5, known_at=snapshot.asof)
    assert set(facts) >= {"external", "active_leg", "last_leg", "phase", "forming_leg_atr", "reset", "displacement_age_bars"}
```

(reuse the `synthetic_observations` fixture by importing it from `brain/tests/test_eye_view.py` or moving it to a `conftest.py` in `brain/tests`.)

- [ ] **Step 2: Run to verify it fails** — ModuleNotFoundError.
- [ ] **Step 3: Implement** — the script drives the Eye with `brain.scripts._run_identity.drive` (as `run_llm_brain.py` does, `on_observation` collecting `(asof, close, facts)` for emitting observations), prints change points per scale; `scale_facts` reads `snapshot.timeframe_states[tf]` and uses the same `_age_bars`/ATR arithmetic as `eye_view` (import them from `brain.core.eye_view` rather than duplicating).
- [ ] **Step 4: Run to verify it passes**; then run the script on the frozen window and on `--emit-start 2022-01-03T18:00 --end 2022-01-04T17:00` and keep both outputs for the receipt.

---

### Task 7: Docs

**Files:**
- Modify: `eyes/docs/README.md` (a "Delivery and structure facts (2026-09-18)" paragraph: active/forming leg, last leg, displacement age, protection break, BALANCE inside the range), `brain/docs/README.md` (input schema 2 keys in the `scales`/`session` description; a row for `audit_scales.py`; link the spec and plan; receipt entry after the run), `brain/docs/specs/2026-09-16-llm-brain-design.md` (an amendment note on the input schema), `AGENTS.md` (script row if scripts are listed there), `brain/docs/evidence/regression_baselines.json` note unchanged.

- [ ] **Step 1: Write the doc changes.** — [ ] **Step 2: `git diff --check`.**

---

### Task 8: Verification and the frozen backtest (run E)

- [ ] **Step 1: Full suite**

Run: `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest eyes brain contract shares risk execution -p no:cacheprovider` (research-marked tests stay deselected). Expected: all passed.

- [ ] **Step 2: Deterministic audit** — `audit_scales.py` on 2022-01-03 and 2022-01-04; check §1.5's acceptance readings (4H active leg long by 02:00 NY, 1H active leg long at 05:00 NY, 1H `reset: long` from 09:39 NY) and record the outputs.

- [ ] **Step 3: Launch run E** in the background with the frozen command (`nohup … > <scratchpad>/run_eye_day.log`), monitor every 30 min, then

```bash
.venv/bin/python -m brain.scripts.summarize_run --run-dir outputs/brain_journal/<run E> --run-dir outputs/brain_journal/72ea13c7fcbc1cff --write
```

- [ ] **Step 4: Receipt** `brain/docs/evidence/2026-09-18_eye_forming_leg_day_run_2022-01-03.md`: the audit readings, run E against `72ea13c7` (direction of ACTIONABLE replies against the 60-min drift, calls, coverage, orders, fills, P&L), what the facts alone moved and what they did not. Index it in `brain/docs/README.md`; update the memory file.

- [ ] **Step 5: Commit** the layer (`feat(eyes,brain): the forming leg, displacement age and protection break in the facts the Brain reads`), then start the Brain-layer plan.
