"""Classifier fitting shared by the gates: a purge by clock when the rows are
not one per minute, the row gap otherwise."""
from __future__ import annotations

import numpy as np
import pandas as pd

from brain.research.gate_models import (
    DEFAULT_GAP_ROWS,
    fit_predict_proba,
    minutes_of,
    purged_after,
    purged_before,
    select_logistic_c,
)


def test_row_gap_purge_matches_the_every_minute_gates() -> None:
    rows = np.arange(100)
    kept = purged_before(rows, 100, times=None, embargo_minutes=240, gap_rows=DEFAULT_GAP_ROWS)
    assert kept.tolist() == list(range(40))
    after = purged_after(np.arange(100, 200), 99, times=None, embargo_minutes=240, gap_rows=DEFAULT_GAP_ROWS)
    assert after.tolist() == list(range(160, 200))


def test_time_purge_drops_rows_whose_window_reaches_the_boundary() -> None:
    # instances ten minutes apart: a row is kept before the boundary only if
    # its 240-minute window ends by the boundary's clock, and after the test
    # rows only once their windows can no longer reach it.
    index = pd.date_range("2022-01-03T09:30", periods=100, freq="10min", tz="UTC")
    times = minutes_of(index)
    kept = purged_before(np.arange(50), 50, times=times, embargo_minutes=240, gap_rows=DEFAULT_GAP_ROWS)
    assert kept.tolist() == list(range(27))  # 10·r + 240 ≤ 500 → r ≤ 26
    after = purged_after(np.arange(50, 100), 49, times=times, embargo_minutes=240, gap_rows=DEFAULT_GAP_ROWS)
    assert after.tolist() == list(range(73, 100))  # 10·r ≥ 490 + 240 → r ≥ 73


def test_minutes_of_is_integer_minutes() -> None:
    index = pd.DatetimeIndex(["2022-01-03T09:30Z", "2022-01-03T09:31Z"])
    assert (np.diff(minutes_of(index)) == 1).all()
    assert minutes_of(index).dtype == np.int64


def test_fit_predict_proba_returns_three_columns_with_or_without_times() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 3))
    y = (x[:, 0] > 0).astype(int)  # two classes seen in training
    index = pd.date_range("2022-01-03T09:30", periods=600, freq="1min", tz="UTC")
    plain = fit_predict_proba("logistic", x, y, x[:10], c=1.0)
    timed = fit_predict_proba("logistic", x, y, x[:10], c=1.0, times=minutes_of(index))
    assert plain.shape == (10, 3) and timed.shape == (10, 3)
    assert np.allclose(plain.sum(axis=1), 1.0)
    assert select_logistic_c(x, y, times=minutes_of(index)) in (0.01, 0.1, 1.0)
