from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from smc_trader.causal import CausalMarketReader
from smc_trader.model import EventKind, MarketEvent, Timeframe
from smc_trader.observation import (
    CausalObserver,
    EventMemory,
    ExecutionRealityInput,
)

from .helpers import session_bars


def test_observer_exposes_all_requested_descriptive_primitives() -> None:
    reader = CausalMarketReader()
    update = None
    for bar in session_bars(4, first_trade_date="2025-01-13"):
        update = reader.on_bar(bar)
    assert update is not None
    observer = CausalObserver()
    observation = observer.observe(
        update,
        ExecutionRealityInput(
            spread_points=0.25,
            deadline=update.asof + pd.Timedelta(minutes=60),
            size_available=10,
        ),
    )
    expected = {
        Timeframe.H4: {
            "directional_displacement",
            "path_efficiency",
            "structure_age_bars",
            "range_position",
            "external_above_distance_atr",
        },
        Timeframe.H1: {
            "swing_progression",
            "acceptance_direction",
            "rejection_direction",
            "dealing_range_position",
            "up_path_obstruction_atr",
        },
        Timeframe.M5: {
            "impulse_strength",
            "pullback_depth",
            "pullback_completeness",
            "reacceptance_direction",
            "compression",
        },
        Timeframe.M1: {
            "path_sequence",
            "acceleration",
            "counter_pressure",
            "trigger_hold_direction",
        },
    }
    for timeframe, columns in expected.items():
        frame = observation.frame(timeframe)
        assert frame.ready
        assert columns <= set(frame.metrics)
        assert frame.cutoff <= observation.asof


def test_missing_deadline_remains_visible_to_risk_layer() -> None:
    reader = CausalMarketReader()
    update = reader.on_bar(session_bars(1)[0])
    observation = CausalObserver().observe(update)
    assert "deadline_missing" in observation.anomalies


def test_invalid_execution_reality_fails_before_observer_mutation() -> None:
    reader = CausalMarketReader()
    update = reader.on_bar(session_bars(1)[0])
    observer = CausalObserver()
    with pytest.raises(ValueError, match="spread cannot be negative"):
        observer.observe(
            update,
            ExecutionRealityInput(spread_points=-0.25),
        )
    assert observer._prior is None
    assert observer.memory.last_minute_end is None
    assert observer.memory.clock_coverage_start is None


def test_event_memory_reports_state_persistence_not_instantaneous_event_age() -> None:
    start = pd.Timestamp("2025-01-06 10:00", tz="America/New_York")
    memory = EventMemory(16)
    memory.append(
        MarketEvent(
            "impulse",
            EventKind.IMPULSE,
            start,
            Timeframe.M5,
            "above",
            None,
            0.8,
        )
    )
    memory.append(
        MarketEvent(
            "sweep",
            EventKind.LIQUIDITY_SWEEP,
            start,
            Timeframe.M1,
            "above",
            101.0,
            0.5,
        )
    )
    memory.close_state(
        EventKind.IMPULSE,
        Timeframe.M5,
        start + pd.Timedelta(minutes=15),
    )
    durations = memory.durations(start + pd.Timedelta(minutes=30))
    assert durations["impulse"] == 15
    assert durations["sweep"] == 0


def test_active_state_survives_recent_deque_eviction_and_closes_stably() -> None:
    start = pd.Timestamp("2025-01-06 10:00", tz="America/New_York")
    memory = EventMemory(1)
    impulse = MarketEvent(
        "long-lived-impulse",
        EventKind.IMPULSE,
        start,
        Timeframe.M5,
        "above",
        None,
        0.8,
    )
    memory.append(impulse)
    memory.append(
        MarketEvent(
            "later-sweep",
            EventKind.LIQUIDITY_SWEEP,
            start + pd.Timedelta(minutes=1),
            Timeframe.M1,
            "above",
            101.0,
            0.5,
        )
    )

    assert impulse.event_id in {
        event.event_id for event in memory.recent(limit=1)
    }
    assert memory.durations(start + pd.Timedelta(minutes=5))[
        impulse.event_id
    ] == 5

    memory.close_state(
        EventKind.IMPULSE,
        Timeframe.M5,
        start + pd.Timedelta(minutes=6),
    )
    assert impulse.event_id in {
        event.event_id for event in memory.recent(limit=1)
    }
    assert memory.durations(start + pd.Timedelta(minutes=30))[
        impulse.event_id
    ] == 6

    closed_memory = EventMemory(1)
    closed_impulse = replace(impulse, event_id="closed-before-eviction")
    closed_memory.append(closed_impulse)
    closed_memory.close_state(
        EventKind.IMPULSE,
        Timeframe.M5,
        start + pd.Timedelta(minutes=6),
    )
    closed_memory.append(
        MarketEvent(
            "post-close-sweep",
            EventKind.LIQUIDITY_SWEEP,
            start + pd.Timedelta(minutes=7),
            Timeframe.M1,
            "above",
            102.0,
            0.5,
        )
    )
    assert closed_impulse.event_id in {
        event.event_id for event in closed_memory.recent(limit=1)
    }
    assert closed_memory.durations(start + pd.Timedelta(minutes=30))[
        closed_impulse.event_id
    ] == 6


def test_continuous_state_identity_is_not_closed_and_reopened() -> None:
    start = pd.Timestamp("2025-01-06 10:00", tz="America/New_York")
    memory = EventMemory(8)
    first = MarketEvent(
        "impulse-start",
        EventKind.IMPULSE,
        start,
        Timeframe.M5,
        "above",
        None,
        0.7,
    )
    continued = replace(
        first,
        event_id="impulse-update",
        observed_at=start + pd.Timedelta(minutes=5),
        strength=0.8,
    )
    reversed_state = replace(
        continued,
        event_id="impulse-reversed",
        observed_at=start + pd.Timedelta(minutes=10),
        side="below",
    )
    memory.append(first)
    memory.append(continued)
    active = [
        event
        for event in memory.recent()
        if event.kind is EventKind.IMPULSE
    ]
    assert [event.event_id for event in active] == [first.event_id]
    assert active[0].observed_at == continued.observed_at
    assert active[0].strength == continued.strength
    assert memory.durations(start + pd.Timedelta(minutes=10))[
        first.event_id
    ] == 10

    memory.append(reversed_state)
    closed = next(
        event
        for event in memory.recent()
        if event.event_id == first.event_id
    )
    assert closed.ended_at == start + pd.Timedelta(minutes=10)
    assert closed.transition_reason == "state_identity_changed"
    current_active = next(
        event
        for event in memory.recent()
        if event.event_id == reversed_state.event_id
    )
    assert closed.sequence_no < current_active.sequence_no
    assert memory.durations(start + pd.Timedelta(minutes=15))[
        first.event_id
    ] == 10
    assert memory.durations(start + pd.Timedelta(minutes=15))[
        reversed_state.event_id
    ] == 5
    assert [
        event.event_id
        for event in memory.recent()
        if event.kind is EventKind.IMPULSE and event.ended_at is None
    ] == [reversed_state.event_id]
