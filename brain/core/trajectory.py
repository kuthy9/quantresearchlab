"""The geometry of a price path, defined once for both runtime and research.

A trajectory is the ATR-normalized cumulative return curve over the next sixty
completed minutes.  That curve is the whole identity input: no hand-picked
subset of it decides what a future *is*.  The attributes in
``PathAttributes`` describe a trajectory once it is already identified, and
realized volatility appears only there.

The runtime updater reads a partially realized path through the same functions
the offline builder reads a complete one, so a node's representative curve and a
live hypothesis's realized path can never be measured by different arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import numpy as np

from contract.brain.forecast import (
    ATTRIBUTE_EXCURSION_WINDOWS,
    ATTRIBUTE_RETURN_HORIZONS,
    ATTRIBUTE_VOLATILITY_HORIZONS,
    TRAJECTORY_CURVE_LENGTH,
    PathAttributes,
)


class TrajectoryError(RuntimeError):
    """A path was asked for something it cannot yet answer."""


@dataclass(frozen=True)
class RealizedPath:
    """What has actually happened since a hypothesis was anchored.

    Prices are stored raw and normalized on read, so one path can be scored
    against several nodes without re-deriving anything.  ``anchor_atr`` is the
    one-minute ATR at the anchor: the same scale every curve is expressed in.
    """

    anchor_price: float
    anchor_atr: float
    closes: tuple[float, ...] = ()
    highs: tuple[float, ...] = ()
    lows: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.anchor_price):
            raise ValueError("anchor_price must be finite")
        if not math.isfinite(self.anchor_atr) or self.anchor_atr <= 0.0:
            raise ValueError("anchor_atr must be finite and positive")
        if not (len(self.closes) == len(self.highs) == len(self.lows)):
            raise ValueError("realized close/high/low series must be the same length")

    @property
    def age(self) -> int:
        """Completed bars observed since the anchor."""

        return len(self.closes)

    def extend(self, *, close: float, high: float, low: float) -> "RealizedPath":
        return replace(
            self,
            closes=self.closes + (float(close),),
            highs=self.highs + (float(high),),
            lows=self.lows + (float(low),),
        )

    def curve(self) -> tuple[float, ...]:
        """The realized part of the trajectory curve, one point per elapsed bar."""

        return tuple(
            (close - self.anchor_price) / self.anchor_atr for close in self.closes
        )

    def full_curve(self) -> tuple[float, ...]:
        """The complete sixty-point curve; raises if the horizon has not elapsed."""

        if self.age < TRAJECTORY_CURVE_LENGTH:
            raise TrajectoryError(
                f"a full curve needs {TRAJECTORY_CURVE_LENGTH} bars, got {self.age}"
            )
        return self.curve()[:TRAJECTORY_CURVE_LENGTH]

    def attributes(self) -> PathAttributes:
        """Read the descriptive geometry of a fully realized path."""

        if self.age < TRAJECTORY_CURVE_LENGTH:
            raise TrajectoryError(
                f"attributes need {TRAJECTORY_CURVE_LENGTH} bars, got {self.age}"
            )
        return path_attributes(
            curve=self.full_curve(),
            highs=tuple(
                (high - self.anchor_price) / self.anchor_atr
                for high in self.highs[:TRAJECTORY_CURVE_LENGTH]
            ),
            lows=tuple(
                (low - self.anchor_price) / self.anchor_atr
                for low in self.lows[:TRAJECTORY_CURVE_LENGTH]
            ),
        )


def path_attributes(
    *,
    curve: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
) -> PathAttributes:
    """Describe one complete ATR-normalized trajectory.

    ``curve``, ``highs`` and ``lows`` are all already anchored and ATR-scaled,
    so every number below is in ATR units and nothing here re-derives a price.
    """

    if not (len(curve) == len(highs) == len(lows) == TRAJECTORY_CURVE_LENGTH):
        raise TrajectoryError(
            f"path attributes need {TRAJECTORY_CURVE_LENGTH} aligned points"
        )
    values = {f"r_{h}": float(curve[h - 1]) for h in ATTRIBUTE_RETURN_HORIZONS}

    # Incremental excursions: each window reports only what it added beyond the
    # running extreme, so the three numbers are not three restatements of one.
    running_high = 0.0
    running_low = 0.0
    for start, end in ATTRIBUTE_EXCURSION_WINDOWS:
        window_high = max(highs[start:end])
        window_low = min(lows[start:end])
        values[f"mfe_{start}_{end}"] = max(0.0, window_high - running_high)
        values[f"mae_{start}_{end}"] = min(0.0, window_low - running_low)
        running_high = max(running_high, window_high)
        running_low = min(running_low, window_low)

    horizon = float(TRAJECTORY_CURVE_LENGTH)
    values["time_to_mfe"] = (int(np.argmax(np.asarray(highs))) + 1) / horizon
    values["time_to_mae"] = (int(np.argmin(np.asarray(lows))) + 1) / horizon

    # Net displacement over distance travelled. Zero-length paths are perfectly
    # inefficient rather than a division by zero.
    steps = np.diff(np.concatenate(([0.0], np.asarray(curve, dtype=float))))
    travelled = float(np.abs(steps).sum())
    values["path_efficiency"] = (
        min(1.0, abs(float(curve[-1])) / travelled) if travelled > 0 else 0.0
    )

    for horizon_minutes in ATTRIBUTE_VOLATILITY_HORIZONS:
        segment = steps[:horizon_minutes]
        values[f"rv_{horizon_minutes}"] = float(
            math.sqrt(float(np.dot(segment, segment)))
        )
    return PathAttributes(**values)


def curve_from_future(
    *,
    anchor_price: float,
    anchor_atr: float,
    closes: Sequence[float],
) -> tuple[float, ...]:
    """The ATR-normalized cumulative return curve of a known future window."""

    if len(closes) < TRAJECTORY_CURVE_LENGTH:
        raise TrajectoryError(
            f"a curve needs {TRAJECTORY_CURVE_LENGTH} future bars, got {len(closes)}"
        )
    if not math.isfinite(anchor_atr) or anchor_atr <= 0.0:
        raise TrajectoryError("anchor_atr must be finite and positive")
    return tuple(
        (float(close) - float(anchor_price)) / float(anchor_atr)
        for close in closes[:TRAJECTORY_CURVE_LENGTH]
    )


def attributes_from_future(
    *,
    anchor_price: float,
    anchor_atr: float,
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
) -> PathAttributes:
    """Describe a known future window without going through ``RealizedPath``."""

    scale = float(anchor_atr)
    anchor = float(anchor_price)
    return path_attributes(
        curve=curve_from_future(
            anchor_price=anchor, anchor_atr=scale, closes=closes
        ),
        highs=tuple(
            (float(v) - anchor) / scale for v in highs[:TRAJECTORY_CURVE_LENGTH]
        ),
        lows=tuple(
            (float(v) - anchor) / scale for v in lows[:TRAJECTORY_CURVE_LENGTH]
        ),
    )


def curve_matrix(
    *,
    anchor_prices: np.ndarray,
    anchor_atrs: np.ndarray,
    future_closes: np.ndarray,
) -> np.ndarray:
    """Vectorized curves for a whole dataset: one row per observation point."""

    closes = np.asarray(future_closes, dtype=float)[:, :TRAJECTORY_CURVE_LENGTH]
    anchors = np.asarray(anchor_prices, dtype=float).reshape(-1, 1)
    scales = np.asarray(anchor_atrs, dtype=float).reshape(-1, 1)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise TrajectoryError("every anchor ATR must be finite and positive")
    return (closes - anchors) / scales


__all__ = [
    "RealizedPath",
    "TrajectoryError",
    "attributes_from_future",
    "curve_from_future",
    "curve_matrix",
    "path_attributes",
]
