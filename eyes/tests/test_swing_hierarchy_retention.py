"""The hot Swing set is a working set; the event stream is the history.

``swing_hierarchy`` grew for the life of the process: every confirmed Swing
stayed in the reduced state forever, so ``TimeframeState.__post_init__``, the
rank projection onto live liquidity, the geometry settle and ``to_primitive``
all walked a population proportional to elapsed bars.  A month therefore cost
O(bars squared) no matter how cheap any single step became.

A Swing leaves the hot set once nothing can still read it.  Two reads exist and
both are local, measured over 2022-02:

* a later rank assignment naming an older Swing reached at most **4** Swings
  back on its own timeframe;
* the geometry tree's retroactive adoption reached at most **447**, bounded by
  the span of the largest enabled timeframe's confirmation window rather than
  by history.

The retained window is far larger than either, so the bound removes only Swings
that nothing reads.  Their complete role history stays in the immutable event
stream, exactly as ``structural_legs`` already works.
"""
from __future__ import annotations

import pandas as pd
import pytest

import eyes.core.market_state as market_state
from eyes.core.market_state import (
    SWING_HIERARCHY_HOT_RETENTION,
    reduce_timeframe_state,
)
from shares.core.model import Direction, EventKind, EventOrigin, MarketEvent, Timeframe
from eyes.core.semantics import SemanticRegistry

TZ = "America/New_York"


def _clock(minutes: int) -> pd.Timestamp:
    return pd.Timestamp("2026-08-19 10:00", tz=TZ) + pd.Timedelta(minutes=minutes)


def _confirmed(index: int) -> MarketEvent:
    return MarketEvent(
        event_id=f"swing-event-{index:05d}",
        kind=EventKind.SWING_CONFIRMED,
        observed_at=_clock(index),
        timeframe=Timeframe.M5,
        side="below",
        price=100.0,
        strength=0.75,
        details={
            "source_entity_id": f"swing-{index:05d}",
            "semantic_rank": "micro",
        },
        sequence_no=index,
        known_at=_clock(index),
        origin=EventOrigin.SEMANTIC_ATOMIC,
        source_entity_ids=(f"swing-{index:05d}",),
    )


def _protected(index: int, target: int) -> MarketEvent:
    return MarketEvent(
        event_id=f"protected-event-{index:05d}",
        kind=EventKind.PROTECTED_SWING_ASSIGNED,
        observed_at=_clock(index),
        timeframe=Timeframe.M5,
        side="below",
        price=100.0,
        strength=0.75,
        details={"protected_swing_id": f"swing-{target:05d}"},
        direction=Direction.LONG,
        sequence_no=index,
        known_at=_clock(index),
        origin=EventOrigin.SEMANTIC_ATOMIC,
        source_entity_ids=(f"swing-{target:05d}",),
    )


def _replay(events, *, retention: int, monkeypatch) -> object:
    monkeypatch.setattr(
        market_state, "SWING_HIERARCHY_HOT_RETENTION", retention
    )
    state = None
    for event in events:
        state = reduce_timeframe_state(
            state,
            event,
            semantic_registry_identity="retention-test",
        )
    return state


def test_the_hot_swing_set_stops_growing_at_the_retained_window(
    monkeypatch,
) -> None:
    retention = 16
    state = _replay(
        [_confirmed(index) for index in range(retention * 3)],
        retention=retention,
        monkeypatch=monkeypatch,
    )

    assert len(state.swing_hierarchy) == retention


def test_the_retained_window_keeps_the_newest_confirmations(
    monkeypatch,
) -> None:
    retention = 16
    total = retention * 3
    state = _replay(
        [_confirmed(index) for index in range(total)],
        retention=retention,
        monkeypatch=monkeypatch,
    )

    assert {view.swing_id for view in state.swing_hierarchy} == {
        f"swing-{index:05d}" for index in range(total - retention, total)
    }


def test_a_later_assignment_inside_the_window_still_lands(
    monkeypatch,
) -> None:
    """The reach that matters is measured in Swings, not in bars."""

    retention = 16
    total = retention * 3
    events = [_confirmed(index) for index in range(total)]
    events.append(_protected(total, total - 4))

    state = _replay(events, retention=retention, monkeypatch=monkeypatch)

    protected = next(
        view
        for view in state.swing_hierarchy
        if view.swing_id == f"swing-{total - 4:05d}"
    )
    assert [item.source_kind for item in protected.assignments] == [
        EventKind.SWING_CONFIRMED.value,
        EventKind.PROTECTED_SWING_ASSIGNED.value,
    ]


def test_an_assignment_beyond_the_window_fails_loudly(monkeypatch) -> None:
    """Silently inventing an evicted Swing would forge its role history."""

    retention = 16
    total = retention * 3
    events = [_confirmed(index) for index in range(total)]
    events.append(_protected(total, 0))

    with pytest.raises(ValueError, match="unknown confirmed swing"):
        _replay(events, retention=retention, monkeypatch=monkeypatch)


def test_the_retained_window_is_registered_and_covers_both_measured_reads(
    monkeypatch,
) -> None:
    registry = SemanticRegistry.from_file()
    registered = registry.parameters.parameters["swing_hierarchy_hot_retention"]

    assert registered["value"] == SWING_HIERARCHY_HOT_RETENTION
    # 4 rank-assignment Swings back and 447 geometric adoptions back, both
    # measured over 2022-02; the window carries a wide margin over each.
    assert SWING_HIERARCHY_HOT_RETENTION >= 447 * 2
