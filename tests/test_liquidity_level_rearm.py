"""A swept level is disarmed, not deleted.

Before v1.3 ``SWEEP_CONFIRMED`` removed the candidate from the inventory
outright.  When price later left the level behind and returned, the Eye had no
way to say that this was the *same* level being offered again -- the next swing
minted a fresh identity with no ancestry, so a level that was tested four times
looked like four unrelated levels.  A level now survives its sweep as a
disarmed object and counts generations: sweep closes generation 1, a close
strictly beyond the level's own band records the departure, and a re-approach
arms generation 2 at the same identity.
"""
from __future__ import annotations

import pandas as pd
import pytest

from smc_trader.market_state import reduce_timeframe_state
from smc_trader.model import (
    EventKind,
    EventOrigin,
    MarketEvent,
    SMC_SEMANTIC_VERSION,
    Timeframe,
)


TZ = "America/New_York"
IDENTITY = "definition-test"
LEVEL_ID = "level:equal-highs"


def _clock(minutes: int) -> pd.Timestamp:
    return pd.Timestamp("2025-01-06 10:00", tz=TZ) + pd.Timedelta(minutes=minutes)


def _event(
    kind: EventKind,
    minutes: int,
    *,
    price: float | None = 100.0,
    side: str | None = None,
    evidence: dict[str, object] | None = None,
) -> MarketEvent:
    known_at = _clock(minutes)
    return MarketEvent(
        event_id=f"{minutes:04d}:{kind.value}",
        kind=kind,
        observed_at=known_at,
        timeframe=Timeframe.M5,
        side=side,
        price=price,
        strength=0.75,
        details={} if evidence is None else evidence,
        sequence_no=0,
        event_time=known_at,
        known_at=known_at,
        semantic_version=SMC_SEMANTIC_VERSION,
        origin=EventOrigin.LEGACY_TRANSPORT,
    )


def _bar(minutes: int, close: float) -> MarketEvent:
    return _event(
        EventKind.BAR_COMPLETED,
        minutes,
        price=close,
        evidence={
            "close": close,
            "atr": 2.0,
            "data_complete": True,
            "real_completed": True,
            "clock_only": False,
            "event_category": "normalized_data",
            "source_data_ids": (f"bar:m5:{minutes}",),
        },
    )


def _level_created(minutes: int, *, price: float = 100.0) -> MarketEvent:
    return _event(
        EventKind.LIQUIDITY_LEVEL_CREATED,
        minutes,
        price=price,
        side="above",
        evidence={
            "level_id": LEVEL_ID,
            "source_kind": "equal_highs",
            "rank": "external",
        },
    )


def _sweep(minutes: int) -> MarketEvent:
    return _event(
        EventKind.SWEEP_CONFIRMED,
        minutes,
        evidence={"level_id": LEVEL_ID},
    )


def _reduce(events):
    state = None
    for event in events:
        state = reduce_timeframe_state(
            state,
            event,
            semantic_registry_identity=IDENTITY,
        )
    assert state is not None
    return state


def _candidate(state):
    found = [
        item
        for item in state.liquidity.candidates
        if item.candidate_id == LEVEL_ID
    ]
    assert len(found) == 1, "the level lost its identity"
    return found[0]


def test_a_created_level_is_armed_as_its_first_generation() -> None:
    state = _reduce((_bar(0, 98.0), _level_created(1)))

    candidate = _candidate(state)
    assert candidate.generation_ordinal == 1
    assert candidate.generation_id == f"{LEVEL_ID}_generation_0001"
    assert candidate.is_armed
    assert state.liquidity.candidate_bsl_ids == (LEVEL_ID,)


def test_a_sweep_disarms_the_level_without_destroying_it() -> None:
    state = _reduce((_bar(0, 98.0), _level_created(1), _sweep(2)))

    candidate = _candidate(state)
    assert candidate.lifecycle == "disarmed"
    assert not candidate.is_armed
    assert candidate.generation_ordinal == 1
    assert candidate.disarmed_at == _clock(2)
    # A disarmed level is no longer offered as resting liquidity.
    assert state.liquidity.candidate_bsl_ids == ()
    assert state.liquidity.unswept_bsl == ()
    assert LEVEL_ID in state.liquidity.recently_swept_ids


def test_a_close_beyond_the_level_records_the_departure() -> None:
    state = _reduce(
        (_bar(0, 98.0), _level_created(1), _sweep(2), _bar(3, 96.0))
    )

    candidate = _candidate(state)
    assert candidate.lifecycle == "rearmable"
    assert not candidate.is_armed
    assert candidate.rearm_departure == pytest.approx(4.0)
    assert candidate.rearm_departure_bar_event_id == _bar(3, 96.0).event_id


def test_a_close_that_does_not_leave_the_level_behind_keeps_it_disarmed() -> None:
    """A sweep that price never leaves is still one unresolved generation."""

    state = _reduce(
        (_bar(0, 98.0), _level_created(1), _sweep(2), _bar(3, 101.0))
    )

    candidate = _candidate(state)
    assert candidate.lifecycle == "disarmed"
    assert candidate.rearm_departure is None
    assert candidate.generation_ordinal == 1


def test_a_re_approach_arms_the_same_level_as_its_next_generation() -> None:
    state = _reduce(
        (
            _bar(0, 98.0),
            _level_created(1),
            _sweep(2),
            _bar(3, 96.0),
            _bar(4, 97.0),
        )
    )

    candidate = _candidate(state)
    assert candidate.lifecycle == "rearmed"
    assert candidate.is_armed
    assert candidate.generation_ordinal == 2
    assert candidate.generation_id == f"{LEVEL_ID}_generation_0002"
    # The identity is unchanged: this is the same level, offered again.
    assert candidate.candidate_id == LEVEL_ID
    assert candidate.price == 100.0
    assert state.liquidity.candidate_bsl_ids == (LEVEL_ID,)


def test_a_rearmed_level_can_be_swept_again_into_a_third_generation() -> None:
    state = _reduce(
        (
            _bar(0, 98.0),
            _level_created(1),
            _sweep(2),
            _bar(3, 96.0),
            _bar(4, 97.0),
            _sweep(5),
            _bar(6, 95.0),
            _bar(7, 96.0),
        )
    )

    candidate = _candidate(state)
    assert candidate.generation_ordinal == 3
    assert candidate.is_armed
    assert state.liquidity.recently_swept_ids.count(LEVEL_ID) == 1


def test_a_deeper_departure_extends_the_reach_it_must_return_from() -> None:
    """The departure is the extreme, so a still-receding close is not a return."""

    state = _reduce(
        (
            _bar(0, 98.0),
            _level_created(1),
            _sweep(2),
            _bar(3, 96.0),
            _bar(4, 94.0),
        )
    )

    candidate = _candidate(state)
    assert candidate.lifecycle == "rearmable"
    assert candidate.rearm_departure == pytest.approx(6.0)
    assert candidate.generation_ordinal == 1
