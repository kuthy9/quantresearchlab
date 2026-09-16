"""The registered Eye, built from ``configs/model.json`` — the one constructor
every Brain-side and script consumer shares.

``persist_state_projections`` stays off: it only adds projection *events* to
the Eye's memory, the authoritative ``MarketSnapshot`` is published either way,
and leaving it on degrades a multi-day scan from ~30 bars/s to under 5.
"""
from __future__ import annotations

import json
from pathlib import Path

from contract.market import Timeframe
from eyes.core.causal import CausalMarketReader
from eyes.core.observation import CausalObserver, ObserverConfig
from eyes.core.semantics import load_semantic_selection
from shares.core.scale_registry import parse_scale_specs

# ``build_eye``'s marker for "the journal directory the model configures".
CONFIGURED: str = "<configured>"

_REGISTERED_TIMEFRAMES = (
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M15,
    Timeframe.M5,
    Timeframe.M1,
)


def build_eye(
    model_path: str | Path, *, root: Path, audit_journal_dir: str | Path | None = CONFIGURED,
) -> tuple[CausalMarketReader, CausalObserver]:
    """Construct the registered graph-free Eye described by ``model_path``.

    ``audit_journal_dir`` overrides the model's shared journal directory (the
    one the runtime spills cold events into and never empties) for a bounded
    pass that owns its Eye and removes the journal with it; ``None`` disables
    the journal.
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
                (None if raw.get("audit_journal_dir") is None else str(root / raw["audit_journal_dir"]))
                if audit_journal_dir is CONFIGURED
                else (None if audit_journal_dir is None else str(audit_journal_dir))
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


__all__ = ["CONFIGURED", "build_eye"]
