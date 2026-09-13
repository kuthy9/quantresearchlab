"""Which of a Setup's two levels the tape reaches first, scanned from the bar
after the one that published the step, censored at the horizon or the
session's end."""
from __future__ import annotations

import numpy as np
import pandas as pd

from brain.research.setup_labels import (
    CENSORED,
    FAILURE_FIRST,
    HORIZON_MINUTES,
    LABEL_COLUMNS,
    TARGET_FIRST,
    label_instances,
    nearest_target,
    session_end_positions,
)

START = pd.Timestamp("2022-01-04T15:00", tz="UTC")


def _tape(highs, lows, *, start=START, gap_after: int | None = None) -> pd.DataFrame:
    stamps = [start + pd.Timedelta(minutes=i) for i in range(len(highs))]
    if gap_after is not None:  # a maintenance break after position gap_after
        stamps = stamps[: gap_after + 1] + [t + pd.Timedelta(hours=1) for t in stamps[gap_after + 1 :]]
    return pd.DataFrame({"high": highs, "low": lows}, index=pd.DatetimeIndex(stamps, name="ts"))


def _path(**overrides) -> pd.DataFrame:
    row = {
        "known_at": START, "sequence_id": "s", "context_kind": "zone_return", "direction": 1.0,
        "close": 100.0, "atr_1m": 1.0, "failure_boundary": 98.0,
        "bsl_5m": 103.0, "ssl_5m": 97.0, "bsl_15m": 102.0, "ssl_15m": np.nan, "bsl_1h": np.nan, "ssl_1h": 90.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _label(paths, tape, **kw):
    out = label_instances(paths, tape, **kw)
    assert set(LABEL_COLUMNS) <= set(out.columns)
    return out.iloc[0]


def test_nearest_target_takes_the_closest_level_in_direction_across_scales() -> None:
    row = _path().iloc[0]
    assert nearest_target(row, 1.0) == 102.0
    assert nearest_target(row, -1.0) == 97.0
    assert np.isnan(nearest_target(_path(bsl_5m=np.nan, bsl_15m=np.nan).iloc[0], 1.0))


def test_target_first_with_time_and_excursions() -> None:
    tape = _tape([100.5, 101.0, 102.2, 100.0], [99.7, 99.0, 101.0, 99.5])
    row = _label(_path(), tape)
    assert row["label"] == TARGET_FIRST and row["drop_reason"] == ""
    assert row["target_price"] == 102.0 and row["d_target_points"] == 2.0 and row["d_failure_points"] == 2.0
    assert row["unit_atr60"] == np.sqrt(60) and np.isclose(row["d_target_atr"], 2.0 / np.sqrt(60))
    assert row["time_to_resolve"] == 3  # third scanned bar
    assert np.isclose(row["mae_atr"], (100.0 - 99.0) / np.sqrt(60))
    assert np.isclose(row["mfe_atr"], (102.2 - 100.0) / np.sqrt(60))
    assert row["same_bar"] == False  # noqa: E712


def test_failure_first_and_same_bar_is_failure() -> None:
    tape = _tape([100.5, 100.8], [99.7, 97.9])
    assert _label(_path(), tape)["label"] == FAILURE_FIRST
    tape = _tape([102.5], [97.5])
    row = _label(_path(), tape)
    assert row["label"] == FAILURE_FIRST and row["same_bar"] == True  # noqa: E712


def test_short_direction_is_mirrored() -> None:
    tape = _tape([100.3, 100.4, 100.2], [99.8, 99.5, 96.9])
    row = _label(_path(direction=-1.0, failure_boundary=101.0), tape)
    assert row["label"] == TARGET_FIRST and row["target_price"] == 97.0
    assert row["d_target_points"] == 3.0 and row["d_failure_points"] == 1.0


def test_censored_at_the_horizon_and_at_the_session_end() -> None:
    n = HORIZON_MINUTES + 20
    quiet = _tape([100.5] * n, [99.5] * n)
    row = _label(_path(), quiet)
    assert row["label"] == CENSORED and row["time_to_resolve"] == HORIZON_MINUTES
    assert row["minutes_to_session_end"] == n
    # the target is reached only after the horizon: still censored
    late = quiet.copy(); late.iloc[HORIZON_MINUTES, 0] = 105.0
    assert _label(_path(), late)["label"] == CENSORED
    # a break after ten bars ends the session: the hit on bar twelve is unseen
    broken = _tape([100.5] * 30, [99.5] * 30, gap_after=9)
    broken.iloc[12, 0] = 105.0
    row = _label(_path(), broken)
    assert row["label"] == CENSORED and row["time_to_resolve"] == 10 and row["minutes_to_session_end"] == 10


def test_session_end_positions_follow_gaps() -> None:
    tape = _tape([1.0] * 6, [1.0] * 6, gap_after=2)
    assert session_end_positions(tape.index).tolist() == [2, 2, 2, 5, 5, 5]


def test_drop_reasons() -> None:
    tape = _tape([100.5] * 5, [99.5] * 5)
    assert _label(_path(context_kind=None), tape)["drop_reason"] == "no_path"
    assert _label(_path(atr_1m=np.nan), tape)["drop_reason"] == "no_atr"
    assert _label(_path(failure_boundary=np.nan), tape)["drop_reason"] == "no_failure"
    assert _label(_path(bsl_5m=np.nan, bsl_15m=np.nan), tape)["drop_reason"] == "no_target"
    assert _label(_path(failure_boundary=100.5), tape)["drop_reason"] == "past_failure"
    assert _label(_path(bsl_5m=99.0, bsl_15m=99.0), tape)["drop_reason"] == "no_target"  # levels below close are not targets for a long
    assert _label(_path(known_at=START - pd.Timedelta(days=1)), tape)["drop_reason"] == "no_tape"
    dropped = label_instances(_path(atr_1m=np.nan), tape).iloc[0]
    assert np.isnan(dropped["label"])
