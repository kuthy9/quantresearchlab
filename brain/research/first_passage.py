"""Which barrier the next sixty minutes reach first, in ATR units.

Three classes: neither barrier inside the horizon, the upper first, the lower
first. A bar that touches both is read as the lower first: the conservative
reading for a long claim, the same convention ``configs/model.json`` names
``same_bar_resolution: conservative`` for the risk engine. The three
registered barrier pairs are the gate's verdict targets
(brain/docs/specs/2026-09-11-information-gain-gate-design.md §5.3).
"""
from __future__ import annotations

import numpy as np

NEITHER, UPPER_FIRST, LOWER_FIRST = 0, 1, 2
CLASS_COUNT = 3
FIRST_PASSAGE_TARGETS: tuple[tuple[str, float, float], ...] = (
    ("fp_1.0_1.0", 1.0, 1.0),
    ("fp_1.0_0.5", 1.0, 0.5),
    ("fp_0.5_1.0", 0.5, 1.0),
)


def first_passage_labels(
    *,
    prices: np.ndarray,
    future_highs: np.ndarray,
    future_lows: np.ndarray,
    up_atr: float,
    down_atr: float,
    horizon: int = 60,
) -> np.ndarray:
    """``prices`` columns are close, high, low, ATR at the anchor clock."""

    anchor = prices[:, 0].reshape(-1, 1)
    scale = prices[:, 3].reshape(-1, 1)
    up_hit = (future_highs[:, :horizon] - anchor) / scale >= up_atr
    down_hit = (anchor - future_lows[:, :horizon]) / scale >= down_atr
    first_up = np.where(up_hit.any(axis=1), up_hit.argmax(axis=1), horizon)
    first_down = np.where(down_hit.any(axis=1), down_hit.argmax(axis=1), horizon)
    labels = np.full(prices.shape[0], NEITHER, dtype=np.int64)
    labels[first_up < first_down] = UPPER_FIRST
    labels[(first_down <= first_up) & (first_down < horizon)] = LOWER_FIRST
    return labels


__all__ = [
    "CLASS_COUNT",
    "FIRST_PASSAGE_TARGETS",
    "LOWER_FIRST",
    "NEITHER",
    "UPPER_FIRST",
    "first_passage_labels",
]
