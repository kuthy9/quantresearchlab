"""The Eye's transition events, one row each, as the Brain's Δₜ source.

A transition is an ``EventKind`` that reports a change: not the ``*_state``
re-publication of a retained entity, not the ``bar_completed`` heartbeat, not
the epoch reset. On the real tape the excluded kinds fire on 67-100 % of
bars; the 49 that remain are what an event clock is built from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from brain.research.path_log import PATH_COLUMNS, empty_path_log
from contract.eye import EventKind, MarketEvent

HEARTBEAT_KINDS: frozenset[str] = frozenset({"bar_completed", "market_epoch_reset"})
TRANSITION_KINDS: tuple[str, ...] = tuple(
    kind.value
    for kind in EventKind
    if not kind.value.endswith("_state") and kind.value not in HEARTBEAT_KINDS
)
EVENT_COLUMNS: tuple[str, ...] = (
    "known_at", "kind", "timeframe", "direction", "side", "strength",
    "price", "entity_id", "lifecycle", "event_id",
)
DATASET_KEYS: tuple[str, ...] = (
    "index", "features", "prices", "future_closes", "future_highs", "future_lows",
)


def is_transition(kind: EventKind) -> bool:
    return kind.value in TRANSITION_KINDS


def event_row(event: MarketEvent) -> dict[str, Any]:
    known_at = event.known_at if event.known_at is not None else event.observed_at
    return {
        "known_at": pd.Timestamp(known_at),
        "kind": event.kind.value,
        "timeframe": event.timeframe.value,
        "direction": None if event.direction is None else event.direction.value,
        "side": event.side,
        "strength": float(event.strength),
        "price": None if event.price is None else float(event.price),
        "entity_id": event.entity_id,
        "lifecycle": event.lifecycle,
        "event_id": event.event_id,
    }


def empty_event_log() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype="object") for name in EVENT_COLUMNS})


def save_block(dataset, out_dir: Path, *, emit_end: str | None) -> None:
    """Write one block the way ``build_forecast_index`` caches a dataset, plus the log."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = np.ones(len(dataset.index), dtype=bool)
    if emit_end is not None:
        local = dataset.index.tz_convert("America/New_York")
        keep = np.asarray(local < pd.Timestamp(emit_end, tz="America/New_York"), dtype=bool)
    np.savez_compressed(
        out_dir / "dataset.npz",
        index=dataset.index[keep].tz_convert("UTC").tz_localize(None).to_numpy(),
        features=dataset.features[keep],
        prices=dataset.prices.to_numpy(dtype=float)[keep],
        future_closes=dataset.future_closes[keep],
        future_highs=dataset.future_highs[keep],
        future_lows=dataset.future_lows[keep],
    )
    events = dataset.events.copy()
    if len(events):
        events["known_at"] = pd.to_datetime(events["known_at"], utc=True)
        if emit_end is not None:
            limit = pd.Timestamp(emit_end, tz="America/New_York")
            events = events[events["known_at"] < limit]
    events.to_parquet(out_dir / "events.parquet", index=False)
    paths = getattr(dataset, "paths", None)
    paths = paths.copy() if paths is not None else empty_path_log()
    if len(paths):
        paths["known_at"] = pd.to_datetime(paths["known_at"], utc=True)
        if emit_end is not None:
            paths = paths[paths["known_at"] < pd.Timestamp(emit_end, tz="America/New_York")]
    paths.to_parquet(out_dir / "paths.parquet", index=False)


def load_blocks(blocks_root: Path) -> dict:
    """Concatenate every block directory under ``blocks_root`` in name order."""

    from brain.scripts._windows import load_dataset

    parts = sorted(
        path for path in Path(blocks_root).iterdir() if (path / "dataset.npz").exists()
    )
    if not parts:
        raise FileNotFoundError(f"no block under {blocks_root}")
    loaded = [load_dataset(path / "dataset.npz") for path in parts]
    merged: dict[str, Any] = {
        "index": pd.DatetimeIndex(
            np.concatenate([part["index"].asi8 for part in loaded]), tz="UTC", name="asof"
        )
    }
    for name in DATASET_KEYS[1:]:
        merged[name] = np.concatenate([part[name] for part in loaded], axis=0)
    logs = [pd.read_parquet(path / "events.parquet") for path in parts]
    events = pd.concat(logs, ignore_index=True) if logs else empty_event_log()
    if len(events):
        events["known_at"] = pd.to_datetime(events["known_at"], utc=True)
        events = events.sort_values(["known_at", "event_id"], kind="stable").reset_index(drop=True)
    merged["events"] = events[list(EVENT_COLUMNS)]
    path_logs = [
        pd.read_parquet(path / "paths.parquet") for path in parts if (path / "paths.parquet").exists()
    ]
    paths = pd.concat(path_logs, ignore_index=True) if path_logs else empty_path_log()
    if len(paths):
        paths["known_at"] = pd.to_datetime(paths["known_at"], utc=True)
        paths = paths.sort_values(["known_at", "sequence_id", "step_id"], kind="stable").reset_index(drop=True)
    merged["paths"] = paths[list(PATH_COLUMNS)] if len(paths) else paths
    merged["blocks"] = tuple(path.name for path in parts)
    return merged
