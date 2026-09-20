"""The active leg is the leg price is in now, not the confirmed leg it left;
a displacement carries its direction and time; a range price has left is not
balance (2026-09-18, the direction fix's Eye layer)."""
from __future__ import annotations

import pandas as pd


from contract.market import Direction
from eyes.core.market_state import (
    DeliveryPhase,
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
    empty = TimeframeRangeState(None, None, None, None, None, None)
    assert _delivery_phase(structure, empty, Direction.LONG) is DeliveryPhase.TRANSITION


def test_the_phase_reads_the_active_leg_against_the_external_direction() -> None:
    short = _structure(Direction.SHORT, Direction.SHORT)
    assert _delivery_phase(short, _range(0.5), Direction.SHORT) is DeliveryPhase.EXPANSION
    assert _delivery_phase(short, _range(0.5), Direction.LONG) is DeliveryPhase.RETRACEMENT
    assert _delivery_phase(_structure(Direction.SHORT, Direction.SHORT, intact=False), _range(0.5), Direction.LONG) is DeliveryPhase.TRANSITION


# ------------------------------------------------------------ the reducer path

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
