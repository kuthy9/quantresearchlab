"""Offline construction of the (context, trajectory) pairs the Brain learns from.

One observation point per completed one-minute bar: ``X_t`` is what the Eye had
published by ``t``, and ``T_t`` summarizes the sixty minutes that followed it in
ATR units.  ``T_t`` is computed by replaying those minutes through the same
``RealizedPath`` the runtime updater uses, so a mode's medoid and a live
hypothesis's realized path are measured by identical arithmetic.

This is a research surface.  It drives the Eye, reads the future and writes
study artifacts; nothing here may become a runtime authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from brain.core.belief_updater import RealizedPath
from brain.core.hypothesis_proposer import FEATURE_DIM, FEATURE_NAMES, observation_features
from contract.brain.forecast import TRAJECTORY_COMPONENTS, TRAJECTORY_DIM
from contract.market import Timeframe
from eyes.core.causal import CausalMarketReader
from eyes.core.observation import CausalObserver, ObserverConfig
from eyes.core.semantics import load_semantic_selection
from shares.core.io import iter_completed_bars, load_ohlcv
from shares.core.scale_registry import parse_scale_specs

FUTURE_HORIZON_MINUTES = max(
    int(name.rsplit("_", 1)[1]) for name in TRAJECTORY_COMPONENTS
)

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
        ),
        semantic_registry=selection.atomic_registry,
    )
    reader = CausalMarketReader(scale_specs=specs, tick_size=float(model["tick_size"]))
    return reader, observer


def trajectory_vector(
    *,
    anchor_price: float,
    anchor_atr: float,
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
) -> tuple[float, ...]:
    """Summarize the realized future as the contract's trajectory vector.

    The future bars are replayed through ``RealizedPath`` so this offline
    definition and the runtime's partial-path reading can never drift apart.
    """

    if len(closes) < FUTURE_HORIZON_MINUTES:
        raise TrajectoryDatasetError(
            f"a trajectory needs {FUTURE_HORIZON_MINUTES} future bars, got {len(closes)}"
        )
    path = RealizedPath(anchor_price=float(anchor_price), anchor_atr=float(anchor_atr))
    for close, high, low in zip(
        closes[:FUTURE_HORIZON_MINUTES],
        highs[:FUTURE_HORIZON_MINUTES],
        lows[:FUTURE_HORIZON_MINUTES],
    ):
        path = path.extend(close=close, high=high, low=low)
    values = []
    for name in TRAJECTORY_COMPONENTS:
        realized = path.realized(name)
        if realized is None:
            raise TrajectoryDatasetError(f"component {name} is undecided at full horizon")
        values.append(float(realized))
    return tuple(values)


@dataclass(frozen=True)
class TrajectoryDataset:
    """Aligned contexts and futures, plus the bars they were derived from."""

    index: pd.DatetimeIndex
    features: np.ndarray
    trajectories: np.ndarray
    prices: pd.DataFrame
    feature_names: tuple[str, ...] = FEATURE_NAMES
    component_names: tuple[str, ...] = TRAJECTORY_COMPONENTS

    def __post_init__(self) -> None:
        rows = len(self.index)
        if self.features.shape != (rows, FEATURE_DIM):
            raise TrajectoryDatasetError(
                f"features must be ({rows}, {FEATURE_DIM}), got {self.features.shape}"
            )
        if self.trajectories.shape != (rows, TRAJECTORY_DIM):
            raise TrajectoryDatasetError(
                f"trajectories must be ({rows}, {TRAJECTORY_DIM}), "
                f"got {self.trajectories.shape}"
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
    trajectory_rows: list[tuple[float, ...]] = []
    price_rows: list[dict[str, float]] = []
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
        trajectory_rows.append(
            trajectory_vector(
                anchor_price=float(bar.close),
                anchor_atr=atr,
                closes=closes[window],
                highs=highs[window],
                lows=lows[window],
            )
        )
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
        trajectories=np.asarray(trajectory_rows, dtype=float),
        prices=pd.DataFrame(price_rows, index=index),
    )


__all__ = [
    "CONTEXT_LOOKBACK_MINUTES",
    "FUTURE_HORIZON_MINUTES",
    "TrajectoryDataset",
    "TrajectoryDatasetError",
    "build_dataset",
    "build_eye",
    "trajectory_vector",
]
