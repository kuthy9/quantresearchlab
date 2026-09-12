"""Δₜ reads only the past, keeps order, and keeps one track per scale."""
from __future__ import annotations

import numpy as np
import pandas as pd

from brain.research.event_log import EVENT_COLUMNS, TRANSITION_KINDS
from brain.research.event_sequence import (
    SCALE_MINUTES,
    clock_mask,
    sequence_features,
    without_kind,
)

T0 = pd.Timestamp("2022-01-03T10:00", tz="America/New_York")


def _clocks(n: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([T0 + pd.Timedelta(minutes=i) for i in range(n)])


def _events(rows: list[tuple[int, str, str, str | None]]) -> pd.DataFrame:
    """rows: (minute offset, kind, timeframe, direction)."""

    return pd.DataFrame(
        [
            {
                "known_at": T0 + pd.Timedelta(minutes=m), "kind": k, "timeframe": tf,
                "direction": d, "side": None, "strength": 1.0, "price": None,
                "entity_id": f"e{i}", "lifecycle": None, "event_id": f"ev{i}",
            }
            for i, (m, k, tf, d) in enumerate(rows)
        ],
        columns=list(EVENT_COLUMNS),
    )


def test_scale_minutes_cover_the_five_registered_scales() -> None:
    assert SCALE_MINUTES == {"1m": 1, "5m": 5, "15m": 15, "1H": 60, "4H": 240}


def test_clock_membership_follows_the_timeframe_threshold() -> None:
    index = _clocks(5)
    events = _events([(1, "sweep_confirmed", "1m", "long"), (3, "qualified_bos", "15m", "short")])
    assert clock_mask(index, events, min_timeframe_minutes=1).tolist() == [False, True, False, True, False]
    assert clock_mask(index, events, min_timeframe_minutes=5).tolist() == [False, False, False, True, False]
    assert clock_mask(index, events, min_timeframe_minutes=15).tolist() == [False, False, False, True, False]
    assert clock_mask(index, events, min_timeframe_minutes=0).all()


def test_a_future_event_leaves_the_vector_unchanged() -> None:
    index = _clocks(3)
    past = _events([(0, "sweep_confirmed", "5m", "long")])
    with_future = _events([(0, "sweep_confirmed", "5m", "long"), (2, "qualified_bos", "5m", "short")])
    a, _ = sequence_features(index, past)
    b, _ = sequence_features(index, with_future)
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[2], b[2])


def test_order_of_the_same_three_events_is_distinguishable() -> None:
    index = _clocks(4)
    forward = _events([(0, "sweep_confirmed", "5m", "long"), (1, "acceptance_confirmed", "5m", "long"), (2, "displacement_observed", "5m", "long")])
    reverse = _events([(0, "displacement_observed", "5m", "long"), (1, "sweep_confirmed", "5m", "long"), (2, "acceptance_confirmed", "5m", "long")])
    a, names = sequence_features(index, forward)
    b, _ = sequence_features(index, reverse)
    track = [i for i, n in enumerate(names) if n.startswith("track_5m_")]
    assert not np.array_equal(a[3, track], b[3, track])
    recency = [i for i, n in enumerate(names) if n.endswith("_count")]
    assert np.array_equal(a[3, recency], b[3, recency])


def test_each_scale_keeps_its_own_track() -> None:
    index = _clocks(60)
    rows = [(0, "qualified_bos", "1H", "long")] + [(m, "level_touched", "1m", None) for m in range(1, 41)]
    x, names = sequence_features(index, _events(rows))
    assert x[59, names.index("track_1H_0_empty")] == 0.0
    assert x[59, names.index("track_1H_0_kind")] == float(TRANSITION_KINDS.index("qualified_bos"))
    assert x[59, names.index("track_1H_0_since")] == 59.0
    assert x[59, names.index("track_1H_1_empty")] == 1.0
    assert x[59, names.index("track_1m_0_kind")] == float(TRANSITION_KINDS.index("level_touched"))
    assert x[59, names.index("track_1m_3_empty")] == 0.0


def test_recency_count_and_direction_block() -> None:
    index = _clocks(10)
    x, names = sequence_features(
        index, _events([(2, "sweep_confirmed", "5m", "short"), (5, "sweep_confirmed", "5m", "long")]), window_minutes=120
    )
    since = names.index("sweep_confirmed@5m_since")
    count = names.index("sweep_confirmed@5m_count")
    direction = names.index("sweep_confirmed@5m_dir")
    assert x[1, since] == 120.0 and x[1, count] == 0.0 and x[1, direction] == 0.0
    assert x[2, since] == 0.0 and x[2, count] == 1.0 and x[2, direction] == -1.0
    assert x[9, since] == 4.0 and x[9, count] == 2.0 and x[9, direction] == 1.0
    assert x[9, names.index("qualified_bos@1H_since")] == 120.0


def test_count_forgets_events_older_than_the_window() -> None:
    index = _clocks(200)
    x, names = sequence_features(index, _events([(0, "fvg_created", "5m", None)]), window_minutes=120)
    count = names.index("fvg_created@5m_count")
    since = names.index("fvg_created@5m_since")
    assert x[100, count] == 1.0 and x[100, since] == 100.0
    assert x[150, count] == 0.0 and x[150, since] == 120.0


def test_trigger_one_hot_marks_the_kinds_fired_at_t() -> None:
    index = _clocks(3)
    x, names = sequence_features(index, _events([(1, "fvg_created", "5m", None)]))
    assert x[:, names.index("trigger_fvg_created")].tolist() == [0.0, 1.0, 0.0]


def test_feature_width_is_fixed() -> None:
    x, names = sequence_features(_clocks(2), _events([]))
    assert x.shape == (2, len(names))
    assert len(names) == 49 * 5 * 3 + 5 * 4 * 5 + 49


def test_without_kind_drops_only_that_kind() -> None:
    events = _events([(0, "sweep_confirmed", "5m", "long"), (1, "qualified_bos", "5m", "long")])
    assert list(without_kind(events, "sweep_confirmed")["kind"]) == ["qualified_bos"]
