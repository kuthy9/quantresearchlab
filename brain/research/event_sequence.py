"""Δₜ: the Eye's recent transition events as a fixed-width, causal vector.

Three blocks, all read from events with ``known_at <= t``:

1. per (kind × scale): minutes since the last occurrence (capped at the
   window, the window when none), the count inside the window, and the
   direction of the last occurrence (+1 / -1 / 0);
2. one ordered track per scale holding the last ``slots`` events published
   on that scale — kind id, minutes ago, direction, strength, empty flag —
   because one-minute transitions arrive every ~1.3 minutes on the real tape
   and a shared track would never show a 15-minute or 1-hour sequence;
3. a one-hot over the kinds fired at ``t`` itself.

Nothing here carries a weight. Every coefficient that reads these columns is
fitted inside the training window of the gate. L and K are fixed by the
spec (§5.5) so they cannot be tuned to the out-of-sample result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from brain.research.event_log import TRANSITION_KINDS

SCALE_MINUTES: dict[str, int] = {"1m": 1, "5m": 5, "15m": 15, "1H": 60, "4H": 240}
SCALES: tuple[str, ...] = tuple(SCALE_MINUTES)
TRACK_FIELDS: tuple[str, ...] = ("kind", "since", "dir", "strength", "empty")
WINDOW_MINUTES = 120
TRACK_SLOTS = 4
_KIND_ID = {kind: float(i) for i, kind in enumerate(TRANSITION_KINDS)}
_DIRECTION = {"long": 1.0, "short": -1.0}


def _direction(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return _DIRECTION.get(str(value), 0.0)


def _utc_naive(stamps: pd.DatetimeIndex) -> np.ndarray:
    stamps = pd.DatetimeIndex(stamps)
    if stamps.tz is not None:
        stamps = stamps.tz_convert("UTC").tz_localize(None)
    return stamps.to_numpy(dtype="datetime64[ns]")


def _prepared(events: pd.DataFrame) -> pd.DataFrame:
    frame = events[events["kind"].isin(TRANSITION_KINDS) & events["timeframe"].isin(SCALES)]
    return frame.sort_values(["known_at", "event_id"], kind="stable").reset_index(drop=True)


def feature_names(*, slots: int = TRACK_SLOTS) -> tuple[str, ...]:
    names: list[str] = []
    for kind in TRANSITION_KINDS:
        for scale in SCALES:
            names += [f"{kind}@{scale}_since", f"{kind}@{scale}_count", f"{kind}@{scale}_dir"]
    for scale in SCALES:
        for slot in range(slots):
            names += [f"track_{scale}_{slot}_{field}" for field in TRACK_FIELDS]
    names += [f"trigger_{kind}" for kind in TRANSITION_KINDS]
    return tuple(names)


def clock_mask(
    index: pd.DatetimeIndex, events: pd.DataFrame, *, min_timeframe_minutes: int
) -> np.ndarray:
    """True where at least one transition event on a scale of at least
    ``min_timeframe_minutes`` was published at that clock; zero means every
    clock."""

    if min_timeframe_minutes <= 0:
        return np.ones(len(index), dtype=bool)
    frame = _prepared(events)
    minutes = frame["timeframe"].map(SCALE_MINUTES)
    hits = _utc_naive(pd.DatetimeIndex(frame.loc[minutes >= min_timeframe_minutes, "known_at"]))
    return np.isin(_utc_naive(index), np.unique(hits))


def sequence_features(
    index: pd.DatetimeIndex,
    events: pd.DataFrame,
    *,
    window_minutes: int = WINDOW_MINUTES,
    slots: int = TRACK_SLOTS,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Vectorised per (kind, scale) and per scale with ``searchsorted``: for
    151k clocks and a million events this runs in seconds, which the ablation
    (one encoding per kind) depends on."""

    names = feature_names(slots=slots)
    column = {name: i for i, name in enumerate(names)}
    frame = _prepared(events)
    stamps = _utc_naive(index)
    known = (
        _utc_naive(pd.DatetimeIndex(frame["known_at"]))
        if len(frame)
        else np.array([], dtype="datetime64[ns]")
    )
    kinds = frame["kind"].to_numpy()
    scales = frame["timeframe"].to_numpy()
    directions = np.array([_direction(v) for v in frame["direction"].to_numpy()], dtype=float)
    strengths = frame["strength"].to_numpy(dtype=float) if len(frame) else np.array([])
    kind_ids = np.array([_KIND_ID[k] for k in kinds], dtype=float)

    # float32: 151k clocks x 884 columns is 0.5 GB this way and 1.1 GB in
    # float64, and every value here is a small integer, a minute count or a
    # strength in [0, 1].
    out = np.zeros((len(index), len(names)), dtype=np.float32)
    since_default = float(window_minutes)
    minute = np.timedelta64(1, "m")
    cutoff = stamps - np.timedelta64(window_minutes, "m")

    # Block 1: per (kind, scale).
    for kind in TRANSITION_KINDS:
        for scale in SCALES:
            base = column[f"{kind}@{scale}_since"]
            picked = np.flatnonzero((kinds == kind) & (scales == scale))
            if picked.size == 0:
                out[:, base] = since_default
                continue
            times = known[picked]
            upto = np.searchsorted(times, stamps, side="right")   # events with known_at <= t
            seen = upto > 0
            last = times[np.maximum(upto - 1, 0)]
            since = np.where(seen, (stamps - last) / minute, since_default)
            out[:, base] = np.minimum(since, since_default)
            # count inside (t - window, t]: known_at >= cutoff and <= t
            before = np.searchsorted(times, cutoff, side="left")
            out[:, base + 1] = (upto - before).astype(float)
            out[:, base + 2] = np.where(seen, directions[picked][np.maximum(upto - 1, 0)], 0.0)

    # Block 2: one track per scale, newest first.
    for scale in SCALES:
        picked = np.flatnonzero(scales == scale)
        times = known[picked]
        upto = np.searchsorted(times, stamps, side="right")
        for slot in range(slots):
            base = column[f"track_{scale}_{slot}_kind"]
            position = upto - 1 - slot
            filled = position >= 0
            safe = np.maximum(position, 0)
            if picked.size == 0:
                out[:, base : base + 5] = (-1.0, since_default, 0.0, 0.0, 1.0)
                continue
            out[:, base] = np.where(filled, kind_ids[picked][safe], -1.0)
            out[:, base + 1] = np.where(filled, (stamps - times[safe]) / minute, since_default)
            out[:, base + 2] = np.where(filled, directions[picked][safe], 0.0)
            out[:, base + 3] = np.where(filled, strengths[picked][safe], 0.0)
            out[:, base + 4] = np.where(filled, 0.0, 1.0)

    # Block 3: the kinds fired at t.
    if len(frame):
        clock_position = {stamp: i for i, stamp in enumerate(stamps.tolist())}
        for at, kind in zip(known.tolist(), kinds.tolist()):
            row = clock_position.get(at)
            if row is not None:
                out[row, column[f"trigger_{kind}"]] = 1.0
    return out, names


def without_kind(events: pd.DataFrame, kind: str) -> pd.DataFrame:
    """The ablation's event stream: everything but ``kind``, on every scale."""

    return events[events["kind"] != kind].reset_index(drop=True)


__all__ = [
    "SCALE_MINUTES",
    "SCALES",
    "TRACK_FIELDS",
    "TRACK_SLOTS",
    "WINDOW_MINUTES",
    "clock_mask",
    "feature_names",
    "sequence_features",
    "without_kind",
]
