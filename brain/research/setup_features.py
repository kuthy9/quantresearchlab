"""The two feature sets of the Setup gate and the zero-parameter reference.

M₀ is geometry only: under a driftless walk the target-first probability is
``d_f / (d_t + d_f)``, so a model that sees the two distances, the remaining
session and the volatility already holds everything a Setup-free forecast can
hold. M₁ adds what the Eye knows about the Setup. Columns are fixed here
(spec §5.4); nothing is chosen on the data.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from brain.core.hypothesis_proposer import FEATURE_NAMES
from brain.research.first_passage import HORIZON_ATR_SCALE
from contract.eye.vocabulary import (
    GROUP5_CONTEXT_KINDS,
    INTERACTION_PHYSICAL_PATH_STEP_KINDS,
    _INTERACTION_PHYSICAL_PATH_STEP_REASONS,
)
from contract.market import Timeframe

SESSION_MINUTES = 1380
GEOMETRY_COLUMNS: tuple[str, ...] = (
    "log_ratio", "span_atr", "d_target_atr", "d_failure_atr", "minutes_to_session_end",
    "rv_30", "rv_60", "tod_sin", "tod_cos",
)
STEP_KINDS: tuple[str, ...] = tuple(sorted(INTERACTION_PHYSICAL_PATH_STEP_KINDS))
STEP_REASONS: tuple[str, ...] = tuple(
    sorted({reason for reasons in _INTERACTION_PHYSICAL_PATH_STEP_REASONS.values() for reason in reasons})
)
# Column identity comes from the contract where it names the values; zone
# kinds and entry modes have no vocabulary and are read from the instances
# (sorted unique values — an identity, not a fitted statistic).
FIXED_CATEGORIES: dict[str, tuple[str, ...]] = {
    "context_kind": tuple(sorted(GROUP5_CONTEXT_KINDS)),
    "source_timeframe": tuple(tf.value for tf in Timeframe),
}
CATEGORICALS: tuple[tuple[str, str], ...] = (
    ("context_kind", "context_kind"), ("source_zone_kind", "zone_kind"),
    ("entry_mode", "entry_mode"), ("source_timeframe", "source_tf"),
)
ALIGNMENT: tuple[tuple[str, str], ...] = (("ext_dir", "ext"), ("int_dir", "int"), ("last_bos_dir", "bos"))
SCALES: tuple[str, ...] = ("5m", "15m", "1h")


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def geometry_matrix(instances: pd.DataFrame) -> np.ndarray:
    d_t, d_f = _column(instances, "d_target_atr"), _column(instances, "d_failure_atr")
    phase = _column(instances, "minutes_since_open") / SESSION_MINUTES * 2.0 * np.pi
    columns = [
        np.log(d_t / d_f), d_t + d_f, d_t, d_f, _column(instances, "minutes_to_session_end"),
        _column(instances, "rv_30"), _column(instances, "rv_60"), np.sin(phase), np.cos(phase),
    ]
    return np.column_stack(columns).astype(float)


def analytic_target_probability(instances: pd.DataFrame) -> np.ndarray:
    d_t, d_f = _column(instances, "d_target_atr"), _column(instances, "d_failure_atr")
    return d_f / (d_t + d_f)


def _categories(frame: pd.DataFrame, name: str) -> tuple[str, ...]:
    if name in FIXED_CATEGORIES:
        return FIXED_CATEGORIES[name]
    if name not in frame:
        return ()
    values = {str(v) for v in frame[name].tolist() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    return tuple(sorted(values))


def _steps(frame: pd.DataFrame) -> list[list[list]]:
    out = []
    for value in frame.get("steps_so_far", pd.Series([None] * len(frame))).tolist():
        try:
            out.append(json.loads(value) if isinstance(value, str) else [])
        except json.JSONDecodeError:
            out.append([])
    return out


def setup_matrix(instances: pd.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    rows = len(instances)
    names: list[str] = []
    columns: list[np.ndarray] = []
    for source, prefix in CATEGORICALS:
        for category in _categories(instances, source):
            names.append(f"{prefix}={category}")
            columns.append((instances[source].astype(object).map(lambda v, c=category: 1.0 if str(v) == c else 0.0)).to_numpy(dtype=float))
    steps = _steps(instances)
    for kind in STEP_KINDS:
        names.append(f"step_strength:{kind}")
        columns.append(np.array([max((float(s[2]) for s in items if s[0] == kind), default=0.0) for items in steps]))
    for reason in STEP_REASONS:
        names.append(f"reason={reason}")
        columns.append(np.array([1.0 if any(str(s[1]) == reason for s in items) else 0.0 for items in steps]))
    known = pd.to_datetime(instances["known_at"], utc=True)
    formed = pd.to_datetime(instances["path_formed_at"], utc=True, errors="coerce")
    names.append("path_age")
    columns.append(((known - formed) / pd.Timedelta(minutes=1)).to_numpy(dtype=float))
    unit = _column(instances, "atr_1m") * HORIZON_ATR_SCALE
    names.append("zone_width_atr")
    columns.append((_column(instances, "upper_bound") - _column(instances, "lower_bound")) / unit)
    for name in ("first_penetration_fraction", "penetration_atr", "reclaim_margin_atr", "hold_margin_atr"):
        names.append(name)
        columns.append(_column(instances, name))
    sign = _column(instances, "direction")
    for field, short in ALIGNMENT:
        for scale in SCALES:
            names.append(f"align_{short}_{scale}")
            columns.append(sign * _column(instances, f"{field}_{scale}"))
    matrix = np.column_stack(columns).astype(float) if columns else np.zeros((rows, 0))
    return matrix, tuple(names)


def eye_state_matrix(instances: pd.DataFrame) -> np.ndarray:
    return np.column_stack([_column(instances, name) for name in FEATURE_NAMES]).astype(float)


__all__ = [
    "GEOMETRY_COLUMNS", "STEP_KINDS", "analytic_target_probability", "eye_state_matrix",
    "geometry_matrix", "setup_matrix",
]
