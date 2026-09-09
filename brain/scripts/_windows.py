"""Shared window slicing for the forecast studies.

The fit and holdout windows are named in exchange-local time because trading
sessions are: a session runs 18:00 to 17:00 New York, so "the three sessions
2022-01-03/04/05" is the half-open clock range 01-02 18:00 to 01-05 17:00.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXCHANGE_TZ = "America/New_York"


@dataclass(frozen=True)
class Window:
    """One contiguous slice of an observation-point dataset."""

    name: str
    index: pd.DatetimeIndex
    features: np.ndarray
    prices: np.ndarray
    future_closes: np.ndarray
    future_highs: np.ndarray
    future_lows: np.ndarray

    def __len__(self) -> int:
        return len(self.index)

    @property
    def local(self) -> pd.DatetimeIndex:
        return self.index.tz_convert(EXCHANGE_TZ)

    @property
    def sessions(self) -> list[str]:
        """Session dates, using the 18:00 New York boundary."""

        shifted = self.local + pd.Timedelta(hours=6)
        return sorted({str(value) for value in shifted.date})

    def describe(self) -> str:
        return (
            f"{self.name}: {len(self)} clocks  "
            f"{self.local.min()} -> {self.local.max()}  "
            f"sessions {', '.join(self.sessions)}"
        )


def load_dataset(path: Path) -> dict:
    stored = np.load(path, allow_pickle=False)
    required = {"index", "features", "prices", "future_closes", "future_highs", "future_lows"}
    missing = required - set(stored.files)
    if missing:
        raise RuntimeError(
            f"{path} predates the raw-future cache and lacks {sorted(missing)}; "
            "rebuild it with brain/scripts/build_forecast_index.py"
        )
    return {
        "index": pd.DatetimeIndex(pd.to_datetime(stored["index"], utc=True), name="asof"),
        "features": stored["features"],
        "prices": stored["prices"],
        "future_closes": stored["future_closes"],
        "future_highs": stored["future_highs"],
        "future_lows": stored["future_lows"],
    }


def slice_window(data: dict, *, name: str, start: str, end: str) -> Window:
    """Half-open [start, end) slice, both named in exchange-local time."""

    local = data["index"].tz_convert(EXCHANGE_TZ)
    keep = (local >= pd.Timestamp(start, tz=EXCHANGE_TZ)) & (
        local < pd.Timestamp(end, tz=EXCHANGE_TZ)
    )
    if not keep.any():
        raise RuntimeError(f"window {name} ({start} -> {end}) selected no clocks")
    return Window(
        name=name,
        index=data["index"][keep],
        features=data["features"][keep],
        prices=data["prices"][keep],
        future_closes=data["future_closes"][keep],
        future_highs=data["future_highs"][keep],
        future_lows=data["future_lows"][keep],
    )


__all__ = ["EXCHANGE_TZ", "Window", "load_dataset", "slice_window"]
