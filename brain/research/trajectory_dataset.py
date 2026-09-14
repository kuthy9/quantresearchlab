"""Offline construction of the (context, trajectory) pairs the Brain learns from.

One observation point per completed one-minute bar: ``X_t`` is what the Eye had
published by ``t``, and the sixty bars that followed are kept **raw**.

Storing the raw future rather than a derived summary is deliberate. Deriving the
trajectory representation is cheap; re-driving the Eye to change it costs an hour
per window. Every curve, attribute and principal score downstream is computed
from these bars, so the representation can be revised without touching the Eye.

This is a research surface.  It drives the Eye, reads the future and writes
study artifacts; nothing here may become a runtime authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from brain.core.hypothesis_proposer import (
    FEATURE_DIM,
    FEATURE_NAMES,
    observation_features,
)
from contract.brain.forecast import TRAJECTORY_CURVE_LENGTH
from contract.market import Timeframe
from eyes.core.causal import CausalMarketReader
from eyes.core.observation import CausalObserver, ObserverConfig
from eyes.core.semantics import load_semantic_selection
from shares.core.io import iter_completed_bars, load_ohlcv
from shares.core.scale_registry import parse_scale_specs

FUTURE_HORIZON_MINUTES = TRAJECTORY_CURVE_LENGTH

# The proposer's tape features look back an hour, so a context vector is only
# complete once that much one-minute history has accumulated.
CONTEXT_LOOKBACK_MINUTES = 60

_REGISTERED_TIMEFRAMES = (
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M15,
    Timeframe.M5,
    Timeframe.M1,
)


class TrajectoryDatasetError(RuntimeError):
    """The dataset builder refuses to fabricate an observation point."""


def build_eye(model_path: str | Path, *, root: Path) -> tuple[CausalMarketReader, CausalObserver]:
    """Construct the registered graph-free Eye described by ``model_path``.

    ``persist_state_projections`` stays off: it only adds projection *events* to
    the Eye's memory, the authoritative ``MarketSnapshot`` is published either
    way, and leaving it on degrades a multi-day scan from ~30 bars/s to under 5.
    """

    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    selection = load_semantic_selection(model.get("semantic_selection"), root=root)
    raw = model["observer"]
    specs = parse_scale_specs(model["scales"])
    minimum = raw["minimum_bars"]
    observer = CausalObserver(
        ObserverConfig(
            atr_period=int(raw["atr_period"]),
            memory_events=int(raw["memory_events"]),
            minimum_bars={tf: int(minimum[tf.value]) for tf in _REGISTERED_TIMEFRAMES},
            tick_size=float(model["tick_size"]),
            point_value=float(model["point_value"]),
            structure_protocol=str(root / raw["structure_protocol"]),
            liquidity_protocol=str(root / raw["liquidity_protocol"]),
            displacement_protocol=str(root / raw["displacement_protocol"]),
            zone_protocol=str(root / raw["zone_protocol"]),
            range_auction_protocol=str(root / raw["range_auction_protocol"]),
            interaction_protocol=str(root / raw["interaction_protocol"]),
            semantic_registry=str(selection.atomic_registry.source_path),
            scale_specs=specs,
            project_scene_graph=False,
            materialize_event_view=False,
            range_auction_projection_only=False,
            eye_authority_mode=True,
            persist_state_projections=False,
            audit_journal_dir=(
                None
                if raw.get("audit_journal_dir") is None
                else str(root / raw["audit_journal_dir"])
            ),
            audit_hot_window_minutes=int(
                raw.get(
                    "audit_hot_window_minutes",
                    ObserverConfig.audit_hot_window_minutes,
                )
            ),
        ),
        semantic_registry=selection.atomic_registry,
    )
    reader = CausalMarketReader(scale_specs=specs, tick_size=float(model["tick_size"]))
    return reader, observer


@dataclass(frozen=True)
class TrajectoryDataset:
    """Aligned contexts and futures, plus the bars they were derived from."""

    index: pd.DatetimeIndex
    features: np.ndarray
    prices: pd.DataFrame
    future_closes: np.ndarray
    future_highs: np.ndarray
    future_lows: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        rows = len(self.index)
        if self.features.shape != (rows, FEATURE_DIM):
            raise TrajectoryDatasetError(
                f"features must be ({rows}, {FEATURE_DIM}), got {self.features.shape}"
            )
        for name in ("future_closes", "future_highs", "future_lows"):
            shape = getattr(self, name).shape
            if shape != (rows, FUTURE_HORIZON_MINUTES):
                raise TrajectoryDatasetError(
                    f"{name} must be ({rows}, {FUTURE_HORIZON_MINUTES}), got {shape}"
                )
        if len(self.prices) != rows:
            raise TrajectoryDatasetError("prices and observation points disagree in length")

    def __len__(self) -> int:
        return len(self.index)


def build_dataset(
    *,
    source: Path,
    warmup_start: str,
    emit_start: str,
    end: str,
    model_path: Path,
    root: Path,
    progress_every: int = 0,
) -> TrajectoryDataset:
    """Run the Eye once and assemble every observation point with a full future."""

    loaded = load_ohlcv(source, start=warmup_start, end=end)
    frame = loaded.frame
    closes = frame["close"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    position = {timestamp: i for i, timestamp in enumerate(frame.index)}

    reader, observer = build_eye(model_path, root=root)
    emit_from = pd.Timestamp(emit_start, tz=frame.index.tz)

    stamps: list[pd.Timestamp] = []
    feature_rows: list[tuple[float, ...]] = []
    price_rows: list[dict[str, float]] = []
    future_close_rows: list[np.ndarray] = []
    future_high_rows: list[np.ndarray] = []
    future_low_rows: list[np.ndarray] = []
    history: list[float] = []
    seen = 0

    for bar in iter_completed_bars(frame):
        observation = observer.observe(reader.on_bar(bar))
        seen += 1
        history.append(float(bar.close))
        if progress_every and seen % progress_every == 0:
            print(f"  {seen}/{len(frame)} bars, kept {len(stamps)}", flush=True)
        snapshot = observation.market_snapshot
        if snapshot is None or snapshot.asof < emit_from:
            continue
        index = position.get(snapshot.asof)
        if index is None or index + FUTURE_HORIZON_MINUTES >= len(frame):
            continue
        if len(history) <= CONTEXT_LOOKBACK_MINUTES:
            continue
        state = snapshot.timeframe_states.get(Timeframe.M1)
        atr = getattr(getattr(state, "quality", None), "atr", None)
        if atr is None or float(atr) <= 0.0:
            continue
        atr = float(atr)
        window = slice(index + 1, index + 1 + FUTURE_HORIZON_MINUTES)
        future_close_rows.append(closes[window].copy())
        future_high_rows.append(highs[window].copy())
        future_low_rows.append(lows[window].copy())
        feature_rows.append(
            observation_features(
                snapshot,
                closes=history[-(CONTEXT_LOOKBACK_MINUTES + 1):],
                bar_high_low=(float(bar.high), float(bar.low)),
            )
        )
        stamps.append(snapshot.asof)
        price_rows.append(
            {
                "close": float(bar.close),
                "high": float(bar.high),
                "low": float(bar.low),
                "atr": atr,
            }
        )

    if not stamps:
        raise TrajectoryDatasetError(
            "no observation point in the emit window had both a warm Eye and a "
            "complete sixty-minute future"
        )
    index = pd.DatetimeIndex(stamps, name="asof")
    return TrajectoryDataset(
        index=index,
        features=np.asarray(feature_rows, dtype=float),
        prices=pd.DataFrame(price_rows, index=index),
        future_closes=np.asarray(future_close_rows, dtype=float),
        future_highs=np.asarray(future_high_rows, dtype=float),
        future_lows=np.asarray(future_low_rows, dtype=float),
    )


__all__ = [
    "CONTEXT_LOOKBACK_MINUTES",
    "FUTURE_HORIZON_MINUTES",
    "TrajectoryDataset",
    "TrajectoryDatasetError",
    "build_dataset",
    "build_eye",
]
