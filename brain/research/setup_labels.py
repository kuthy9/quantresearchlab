"""Which of a Setup's two levels the tape reaches first.

The scan starts at the bar whose ``ts`` equals ``known_at``: the completed bar
that published the step closed at ``known_at``, so that is the first bar not
yet seen. It stops at ``min(horizon, session end)``, the session end being the
last bar before the next gap longer than a minute in the tape. Both levels on
one bar read as ``failure``, the conservative reading for the claim
(spec §5.3). Distances are also expressed in ATR₆₀ = ATR₁ₘ·√60.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from brain.research.first_passage import HORIZON_ATR_SCALE
from brain.research.path_log import LEVEL_SCALES

HORIZON_MINUTES = 240
TARGET_FIRST, FAILURE_FIRST, CENSORED = 0, 1, 2
LABEL_NAMES: tuple[str, ...] = ("target", "failure", "censored")
LABEL_COLUMNS: tuple[str, ...] = (
    "label", "target_price", "d_target_points", "d_failure_points", "unit_atr60", "d_target_atr",
    "d_failure_atr", "time_to_resolve", "mae_atr", "mfe_atr", "minutes_to_session_end", "same_bar", "drop_reason",
)
DROP_REASONS: tuple[str, ...] = ("no_path", "no_atr", "no_failure", "no_target", "past_failure", "no_tape")
_MINUTE_NS = 60_000_000_000


def session_end_positions(index: pd.DatetimeIndex) -> np.ndarray:
    """For every row, the position of the last bar of its contiguous run."""

    stamps = pd.DatetimeIndex(index).asi8
    if stamps.size == 0:
        return np.zeros(0, dtype=int)
    gap_after = np.flatnonzero(np.diff(stamps) != _MINUTE_NS)
    ends = np.append(gap_after, stamps.size - 1)
    return ends[np.searchsorted(ends, np.arange(stamps.size))]


def nearest_target(row: Mapping[str, object], sign: float) -> float:
    close = float(row["close"])
    side = "bsl" if sign > 0 else "ssl"
    levels = []
    for _, name in LEVEL_SCALES:
        value = row.get(f"{side}_{name}")
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value) and sign * (value - close) > 0:
            levels.append(value)
    if not levels:
        return float("nan")
    return min(levels) if sign > 0 else max(levels)


def _blank(reason: str) -> dict[str, object]:
    out: dict[str, object] = {name: float("nan") for name in LABEL_COLUMNS}
    out["same_bar"] = False
    out["drop_reason"] = reason
    return out


def label_instances(
    paths: pd.DataFrame, tape: pd.DataFrame, *, horizon_minutes: int = HORIZON_MINUTES
) -> pd.DataFrame:
    index = pd.DatetimeIndex(tape.index)
    highs = tape["high"].to_numpy(dtype=float)
    lows = tape["low"].to_numpy(dtype=float)
    ends = session_end_positions(index)
    known = pd.to_datetime(paths["known_at"], utc=True)
    if index.tz is not None:
        known = known.dt.tz_convert(index.tz)
    positions = index.get_indexer(pd.DatetimeIndex(known))
    labelled: list[dict[str, object]] = []
    for row, start in zip(paths.to_dict("records"), positions):
        kind = row.get("context_kind")
        if kind is None or (isinstance(kind, float) and math.isnan(kind)):
            labelled.append(_blank("no_path"))
            continue
        atr = float(row.get("atr_1m", float("nan")))
        if not math.isfinite(atr) or atr <= 0.0:
            labelled.append(_blank("no_atr"))
            continue
        sign = float(row["direction"])
        failure = float(row.get("failure_boundary", float("nan")))
        if not math.isfinite(failure) or sign == 0.0:
            labelled.append(_blank("no_failure"))
            continue
        target = nearest_target(row, sign)
        if not math.isfinite(target):
            labelled.append(_blank("no_target"))
            continue
        close = float(row["close"])
        d_failure = sign * (close - failure)
        if d_failure <= 0.0:
            labelled.append(_blank("past_failure"))
            continue
        d_target = sign * (target - close)
        if start < 0:
            labelled.append(_blank("no_tape"))
            continue
        stop = min(start + horizon_minutes - 1, int(ends[start]))
        h = highs[start : stop + 1]
        l = lows[start : stop + 1]
        if sign > 0:
            hit_t, hit_f = h >= target, l <= failure
        else:
            hit_t, hit_f = l <= target, h >= failure
        first_t = int(hit_t.argmax()) if hit_t.any() else len(h)
        first_f = int(hit_f.argmax()) if hit_f.any() else len(h)
        if first_f <= first_t and first_f < len(h):
            label, k, same_bar = FAILURE_FIRST, first_f, bool(first_f == first_t)
        elif first_t < len(h):
            label, k, same_bar = TARGET_FIRST, first_t, False
        else:
            label, k, same_bar = CENSORED, len(h) - 1, False
        unit = atr * HORIZON_ATR_SCALE
        seen_h, seen_l = h[: k + 1], l[: k + 1]
        if sign > 0:
            mae, mfe = (close - seen_l.min()) / unit, (seen_h.max() - close) / unit
        else:
            mae, mfe = (seen_h.max() - close) / unit, (close - seen_l.min()) / unit
        labelled.append({
            "label": label, "target_price": target, "d_target_points": d_target, "d_failure_points": d_failure,
            "unit_atr60": unit / atr, "d_target_atr": d_target / unit, "d_failure_atr": d_failure / unit,
            "time_to_resolve": int(k + 1), "mae_atr": max(float(mae), 0.0), "mfe_atr": max(float(mfe), 0.0),
            "minutes_to_session_end": int(ends[start] - start + 1), "same_bar": same_bar, "drop_reason": "",
        })
    out = paths.reset_index(drop=True).copy()
    for name in LABEL_COLUMNS:
        out[name] = [item[name] for item in labelled]
    return out


__all__ = [
    "CENSORED", "DROP_REASONS", "FAILURE_FIRST", "HORIZON_MINUTES", "LABEL_COLUMNS", "LABEL_NAMES",
    "TARGET_FIRST", "label_instances", "nearest_target", "session_end_positions",
]
