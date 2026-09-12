"""First-passage labels: which ATR barrier the next sixty minutes reach first."""
from __future__ import annotations

import numpy as np

from brain.research.first_passage import (
    FIRST_PASSAGE_TARGETS,
    LOWER_FIRST,
    NEITHER,
    UPPER_FIRST,
    first_passage_labels,
)

PRICES = np.array([[100.0, 100.0, 100.0, 1.0]])  # close, high, low, atr


def _future(highs: list[float], lows: list[float], horizon: int = 60):
    h = np.full((1, horizon), 100.0)
    l = np.full((1, horizon), 100.0)
    h[0, : len(highs)] = highs
    l[0, : len(lows)] = lows
    return h, l


def _label(h, l, up=1.0, down=0.5, prices=PRICES):
    return first_passage_labels(prices=prices, future_highs=h, future_lows=l, up_atr=up, down_atr=down)[0]


def test_upper_first() -> None:
    assert _label(*_future([100.5, 101.2], [99.8, 99.9])) == UPPER_FIRST


def test_lower_first() -> None:
    assert _label(*_future([100.2, 100.3], [99.4, 99.9])) == LOWER_FIRST


def test_neither_within_horizon() -> None:
    assert _label(*_future([100.4] * 60, [99.7] * 60)) == NEITHER


def test_a_barrier_reached_on_the_last_bar_still_counts() -> None:
    h, l = _future([100.4] * 60, [99.7] * 60)
    h[0, 59] = 101.0
    assert _label(h, l) == UPPER_FIRST


def test_a_barrier_beyond_the_horizon_does_not_count() -> None:
    h, l = _future([100.4] * 70, [99.7] * 70, horizon=70)
    h[0, 65] = 101.0
    assert _label(h, l) == NEITHER


def test_same_bar_tie_is_lower_first() -> None:
    assert _label(*_future([101.0], [99.5])) == LOWER_FIRST


def test_barriers_scale_with_the_anchor_atr() -> None:
    prices = np.array([[100.0, 100.0, 100.0, 4.0]])
    assert _label(*_future([102.0], [99.0]), up=1.0, down=1.0, prices=prices) == NEITHER


def test_three_registered_targets() -> None:
    assert [t[0] for t in FIRST_PASSAGE_TARGETS] == ["fp_1.0_1.0", "fp_1.0_0.5", "fp_0.5_1.0"]
    assert FIRST_PASSAGE_TARGETS[2][1:] == (0.5, 1.0)
