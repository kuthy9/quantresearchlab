"""One row per new Group-5 path step, with the geometry the Setup gate labels from.

Rows are read from ``InteractionUpdate.milestone_transitions`` at the bar the
step is published, so ``known_at`` is the completed bar's ``asof`` and nothing
is known earlier than the Eye knew it. The Eye's own DTOs are read by
attribute and never rebuilt; a step whose context is not on the same
observation is logged with NaN geometry and ``context_found = False`` so the
receipt can count it (spec §5.2).
"""
from __future__ import annotations

import json
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from brain.core.hypothesis_proposer import FEATURE_NAMES
from contract.market import Direction, Timeframe

EXCHANGE_TZ = "America/New_York"
SESSION_OPEN_HOUR = 18
LEVEL_SCALES: tuple[tuple[Timeframe, str], ...] = (
    (Timeframe.M5, "5m"), (Timeframe.M15, "15m"), (Timeframe.H1, "1h"),
)
IDENTITY_COLUMNS: tuple[str, ...] = (
    "known_at", "sequence_id", "context_kind", "context_id", "direction", "path_formed_at",
    "path_lifecycle", "step_id", "step_kind", "step_reason", "step_strength", "step_observed_at",
    "step_ordinal", "steps_so_far", "context_found",
)
ZONE_COLUMNS: tuple[str, ...] = (
    "lower_bound", "upper_bound", "near_edge", "far_edge", "failure_boundary", "source_zone_kind",
    "entry_mode", "first_penetration_fraction", "eye_draw_distance_points",
)
POOL_COLUMNS: tuple[str, ...] = (
    "source_lower_bound", "source_upper_bound", "sweep_extreme", "penetration_atr", "source_timeframe",
    "reference_price", "reclaim_margin_atr", "hold_margin_atr",
)
TAPE_COLUMNS: tuple[str, ...] = ("close", "high", "low", "atr_1m", "rv_30", "rv_60", "minutes_since_open")
LEVEL_COLUMNS: tuple[str, ...] = tuple(f"{side}_{name}" for _, name in LEVEL_SCALES for side in ("bsl", "ssl"))
STRUCTURE_COLUMNS: tuple[str, ...] = tuple(
    f"{field}_{name}" for _, name in LEVEL_SCALES for field in ("ext_dir", "int_dir", "last_bos_dir")
)
PATH_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS + ZONE_COLUMNS + POOL_COLUMNS + TAPE_COLUMNS + LEVEL_COLUMNS + STRUCTURE_COLUMNS + FEATURE_NAMES
)
_NAN = float("nan")


def direction_sign(value: object) -> float:
    text = value.value if isinstance(value, Direction) else value
    if text == Direction.LONG.value:
        return 1.0
    if text == Direction.SHORT.value:
        return -1.0
    return 0.0


def _text(value: object) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _number(obj: object, name: str) -> float:
    value = getattr(obj, name, None)
    if value is None:
        return _NAN
    try:
        return float(value)
    except (TypeError, ValueError):
        return _NAN


def minutes_since_open(asof: pd.Timestamp) -> float:
    local = pd.Timestamp(asof).tz_convert(EXCHANGE_TZ)
    session_day = (local + pd.Timedelta(hours=24 - SESSION_OPEN_HOUR)).normalize()
    open_at = session_day - pd.Timedelta(hours=24 - SESSION_OPEN_HOUR)
    return float((local - open_at) / pd.Timedelta(minutes=1))


def realized_volatility(closes: Sequence[float], minutes: int, atr: float) -> float:
    """``sqrt(sum of squared one-minute close changes)`` over the last
    ``minutes`` changes, in ATR units; NaN when the history is too short."""

    values = np.asarray(closes[-(minutes + 1):], dtype=float)
    if values.size < minutes + 1 or not atr or not math.isfinite(atr):
        return _NAN
    return float(np.sqrt(np.sum(np.diff(values) ** 2)) / atr)


def _nearest_levels(snapshot: Any, close: float) -> dict[str, float]:
    out: dict[str, float] = {}
    states = getattr(snapshot, "timeframe_states", None) or {}
    for timeframe, name in LEVEL_SCALES:
        liquidity = getattr(states.get(timeframe), "liquidity", None)
        above = [float(level) for level in (getattr(liquidity, "unswept_bsl", None) or ()) if float(level) > close]
        below = [float(level) for level in (getattr(liquidity, "unswept_ssl", None) or ()) if float(level) < close]
        out[f"bsl_{name}"] = min(above) if above else _NAN
        out[f"ssl_{name}"] = max(below) if below else _NAN
    return out


def _structure(snapshot: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    states = getattr(snapshot, "timeframe_states", None) or {}
    for timeframe, name in LEVEL_SCALES:
        structure = getattr(states.get(timeframe), "structure", None)
        out[f"ext_dir_{name}"] = direction_sign(getattr(structure, "external_direction", None))
        out[f"int_dir_{name}"] = direction_sign(getattr(structure, "internal_direction", None))
        out[f"last_bos_dir_{name}"] = direction_sign(getattr(structure, "last_bos_direction", None))
    return out


def _zone(location: Any) -> dict[str, Any]:
    return {
        "lower_bound": _number(location, "lower_bound"),
        "upper_bound": _number(location, "upper_bound"),
        "near_edge": _number(location, "near_edge"),
        "far_edge": _number(location, "far_edge"),
        "failure_boundary": _number(location, "failure_boundary"),
        "source_zone_kind": _text(getattr(location, "source_zone_kind", None)),
        "entry_mode": _text(getattr(location, "entry_mode", None)),
        "first_penetration_fraction": _number(location, "first_penetration_fraction"),
        "eye_draw_distance_points": _number(location, "nearest_visible_draw_distance_points"),
    }


def _reacceptance(reacceptance: Any) -> dict[str, Any]:
    """A zone's own Group-5 reacceptance, when the update carries one."""

    return {
        "reference_price": _number(reacceptance, "reference_price"),
        "reclaim_margin_atr": _number(reacceptance, "reclaim_margin_atr"),
        "hold_margin_atr": _number(reacceptance, "hold_margin_atr"),
    }


def _pool(manipulation: Any, atr: float) -> dict[str, Any]:
    """A pool path's context is a Group-4 ``ManipulationState``. Its
    ``timeframe`` is the one-minute clock it is maintained on; the swept
    pool's scale is ``source_timeframe``. Group 4 owns the reclaim, so the
    reference is the swept boundary and the reclaim margin is how far the
    re-entry close came back through it; there is no hold margin."""

    side = _text(getattr(manipulation, "side", None))
    reference = _number(manipulation, "source_upper_bound" if side == "above" else "source_lower_bound")
    reentry = _number(manipulation, "reentry_price")
    reclaim = abs(reentry - reference) / atr if math.isfinite(reentry) and math.isfinite(reference) and atr > 0 else _NAN
    return {
        "source_lower_bound": _number(manipulation, "source_lower_bound"),
        "source_upper_bound": _number(manipulation, "source_upper_bound"),
        "sweep_extreme": _number(manipulation, "sweep_extreme"),
        "penetration_atr": _number(manipulation, "penetration_atr"),
        "source_timeframe": _text(getattr(manipulation, "source_timeframe", None)),
        "reference_price": reference,
        "reclaim_margin_atr": reclaim,
        "hold_margin_atr": _NAN,
        "failure_boundary": _number(manipulation, "sweep_extreme"),
    }


def _blank() -> dict[str, Any]:
    row: dict[str, Any] = {name: _NAN for name in PATH_COLUMNS}
    for name in ("known_at", "sequence_id", "context_kind", "context_id", "path_formed_at", "path_lifecycle",
                 "step_id", "step_kind", "step_reason", "step_observed_at", "steps_so_far",
                 "source_zone_kind", "entry_mode", "source_timeframe"):
        row[name] = None
    row["context_found"] = False
    return row


def remember_pools(observation: Any, pool_memory: dict[str, Any]) -> None:
    """Keep the last Group-4 state seen for every live pool path.

    Cheap enough for every bar the Eye observes, and it must run on every
    one: a pool swept before the emit window, or before the recorder is
    warm, can still be the context of a step inside it. Idempotent —
    ``path_rows`` calls it again for the bar it is given.
    """

    update = getattr(observation, "interaction_update", None)
    if update is None:
        return
    manipulations = {item.manipulation_id: item for item in getattr(observation, "manipulations", ())}
    # Live paths and the bar's transition copies alike: a step is looked up
    # by the path that carries it, whichever tuple published that path.
    live_pools = {
        path.context_id
        for path in (*getattr(update, "interaction_paths", ()), *getattr(update, "interaction_path_transitions", ()))
        if _text(path.context_kind) == "pool_reversal"
    }
    for context_id in live_pools:
        if context_id in manipulations:
            pool_memory[context_id] = manipulations[context_id]
    for context_id in tuple(pool_memory):
        if context_id not in live_pools:
            del pool_memory[context_id]


def path_rows(
    observation: Any, *, close: float, high: float, low: float, atr: float,
    history: Sequence[float], features: Sequence[float],
    pool_memory: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """The rows for this bar's new path steps.

    ``pool_memory`` is the caller's, kept across bars: the last Group-4
    ``ManipulationState`` seen for each live pool path, keyed by
    ``manipulation_id``. Group 4 compacts a resolved manipulation one
    completed bar after it resolves, and a pool path's later steps come after
    that; the remembered state is the geometry those steps are measured
    against. Entries follow the live paths and leave with them.
    """

    update = getattr(observation, "interaction_update", None)
    if update is None:
        return []
    manipulations = {item.manipulation_id: item for item in getattr(observation, "manipulations", ())}
    if pool_memory is not None:
        remember_pools(observation, pool_memory)
    transitions = tuple(getattr(update, "milestone_transitions", ()) or ())
    if not transitions:
        return []
    snapshot = getattr(observation, "market_snapshot", None)
    asof = pd.Timestamp(getattr(snapshot, "asof", None) or getattr(observation, "asof"))
    # The live set is authoritative; a transition copy can predate a step the
    # reducer appended on the same bar, so it only fills in a path the live
    # set does not carry.
    paths = {path.sequence_id: path for path in getattr(update, "interaction_path_transitions", ())}
    paths.update({path.sequence_id: path for path in getattr(update, "interaction_paths", ())})
    locations = {loc.location_id: loc for loc in getattr(update, "zone_interactions", ())}
    reacceptances = {item.context_id: item for item in getattr(update, "reacceptance_interactions", ())}
    common: dict[str, Any] = {
        "known_at": asof, "close": float(close), "high": float(high), "low": float(low), "atr_1m": float(atr),
        "rv_30": realized_volatility(history, 30, atr), "rv_60": realized_volatility(history, 60, atr),
        "minutes_since_open": minutes_since_open(asof),
        **_nearest_levels(snapshot, float(close)), **_structure(snapshot),
        **dict(zip(FEATURE_NAMES, (float(v) for v in features))),
    }
    rows: list[dict[str, Any]] = []
    for sequence_id, step in transitions:
        row = _blank()
        row.update(common)
        row.update({
            "sequence_id": sequence_id, "step_id": getattr(step, "step_id", None),
            "step_kind": _text(getattr(step, "kind", None)), "step_reason": _text(getattr(step, "reason", None)),
            "step_strength": _number(step, "strength"),
            "step_observed_at": pd.Timestamp(step.observed_at) if getattr(step, "observed_at", None) is not None else None,
        })
        path = paths.get(sequence_id)
        if path is None:
            rows.append(row)
            continue
        steps = tuple(getattr(path, "steps", ()))
        # -1 when the published path does not show the step: recorded, not guessed.
        ordinal = next((i for i, item in enumerate(steps) if item.step_id == row["step_id"]), -1)
        shown = steps if ordinal < 0 else steps[: ordinal + 1]
        row.update({
            "context_kind": _text(path.context_kind), "context_id": path.context_id,
            "direction": direction_sign(path.direction),
            "path_formed_at": pd.Timestamp(path.formed_at) if getattr(path, "formed_at", None) is not None else None,
            "path_lifecycle": _text(getattr(path, "lifecycle", None)), "step_ordinal": int(ordinal),
            "steps_so_far": json.dumps(
                [[_text(item.kind), _text(item.reason), float(item.strength)] for item in shown]
            ),
        })
        if row["context_kind"] == "zone_return":
            location = locations.get(path.context_id)
            if location is not None:
                row.update(_zone(location))
                row["context_found"] = True
            reacceptance = reacceptances.get(path.context_id)
            if reacceptance is not None:
                row.update(_reacceptance(reacceptance))
        elif row["context_kind"] == "pool_reversal":
            manipulation = manipulations.get(path.context_id)
            if manipulation is None and pool_memory is not None:
                manipulation = pool_memory.get(path.context_id)
            if manipulation is not None:
                row.update(_pool(manipulation, float(atr)))
                row["context_found"] = True
        rows.append(row)
    return rows


def empty_path_log() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype="object") for name in PATH_COLUMNS})


__all__ = [
    "IDENTITY_COLUMNS", "LEVEL_COLUMNS", "LEVEL_SCALES", "PATH_COLUMNS", "POOL_COLUMNS",
    "STRUCTURE_COLUMNS", "TAPE_COLUMNS", "ZONE_COLUMNS", "direction_sign", "empty_path_log",
    "minutes_since_open", "path_rows", "realized_volatility", "remember_pools",
]
