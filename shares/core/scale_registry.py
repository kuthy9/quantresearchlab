"""Registered observation scale contract shared by the Eye and its consumers.

The scale registry is a small, standalone contract: which timeframes are
observed, in what role, with what history bound, and one deterministic hash of
that ordered contract.  It deliberately lives outside
:mod:`shares.core.scene_graph` so the data reader and the Trading Eye do not
import the optional research/visualisation graph in order to describe their own
scales.  :mod:`shares.core.scene_graph` re-exports these names so historical
pickles that recorded ``shares.core.scene_graph.ScaleSpec`` still resolve to this
exact class.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .model import CORE_TIMEFRAMES, Timeframe, content_hash, to_primitive


class ScaleRole(str, Enum):
    MACRO = "macro"
    CONTEXT = "context"
    CONTEXT_BRIDGE = "context_bridge"
    SETUP = "setup"
    TRIGGER = "trigger"


class StructuralScale(str, Enum):
    INTERNAL = "internal"
    INTERMEDIATE = "intermediate"
    EXTERNAL = "external"


_TIMEFRAME_MINUTES: Mapping[Timeframe, int] = {
    Timeframe.H4: 240,
    Timeframe.H1: 60,
    Timeframe.M15: 15,
    Timeframe.M5: 5,
    Timeframe.M1: 1,
}


@dataclass(frozen=True)
class ScaleSpec:
    """One explicitly registered observation scale.

    ``1D`` is accepted as a parked declaration so the contract can describe
    the intended macro scale without silently enabling an unimplemented daily
    aggregation rule.
    """

    timeframe: Timeframe | str
    role: ScaleRole | str
    session_anchor: str = "exchange_session"
    completed_bar_only: bool = True
    history_limit: int = 1024
    enabled: bool = True
    structural_scales: tuple[StructuralScale, ...] = (
        StructuralScale.INTERNAL,
        StructuralScale.INTERMEDIATE,
        StructuralScale.EXTERNAL,
    )

    def __post_init__(self) -> None:
        timeframe = (
            self.timeframe.value
            if isinstance(self.timeframe, Timeframe)
            else str(self.timeframe)
        )
        role = self.role if isinstance(self.role, ScaleRole) else ScaleRole(self.role)
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self,
            "structural_scales",
            tuple(StructuralScale(value) for value in self.structural_scales),
        )
        if (
            timeframe not in {item.value for item in Timeframe} | {"1D"}
            or self.session_anchor != "exchange_session"
            or type(self.completed_bar_only) is not bool
            or type(self.enabled) is not bool
            or type(self.history_limit) is not int
            or self.history_limit < 8
            or not self.structural_scales
            or len(self.structural_scales) != len(set(self.structural_scales))
        ):
            raise ValueError("invalid scale specification")
        if self.enabled and not self.completed_bar_only:
            raise ValueError("enabled market scales must use completed bars only")
        if timeframe == "1D" and self.enabled:
            raise ValueError("1D aggregation is registered but remains parked")

    @property
    def key(self) -> str:
        return self.timeframe.value if isinstance(self.timeframe, Timeframe) else str(self.timeframe)

    @property
    def native_timeframe(self) -> Timeframe | None:
        if isinstance(self.timeframe, Timeframe):
            return self.timeframe
        return next((item for item in Timeframe if item.value == self.timeframe), None)

    @property
    def minutes(self) -> int | None:
        timeframe = self.native_timeframe
        return None if timeframe is None else _TIMEFRAME_MINUTES[timeframe]


def parse_scale_specs(
    payload: Sequence[Mapping[str, Any]] | None,
    *,
    history_limit: int = 1024,
) -> tuple[ScaleSpec, ...]:
    if not payload:
        raise ValueError("scale registry must be supplied explicitly")
    specs = tuple(
        ScaleSpec(
            timeframe=str(item.get("timeframe", "")),
            role=str(item.get("role", "")),
            session_anchor=str(item.get("session_anchor", "exchange_session")),
            completed_bar_only=item.get("completed_bar_only", True),
            history_limit=int(item.get("history_limit", history_limit)),
            enabled=item.get("enabled", True),
            structural_scales=tuple(
                item.get(
                    "structural_scales",
                    tuple(value.value for value in StructuralScale),
                )
            ),
        )
        for item in payload
    )
    keys = tuple(item.key for item in specs)
    if len(keys) != len(set(keys)):
        raise ValueError("scale registry contains duplicate timeframes")
    enabled = {item.native_timeframe for item in specs if item.enabled}
    if not set(CORE_TIMEFRAMES).issubset(enabled):
        raise ValueError("scale registry must retain the causal core frames")
    return specs


def scale_registry_id(scale_specs: Sequence[ScaleSpec]) -> str:
    """Hash the complete ordered scale contract, not only its display role."""

    specs = tuple(scale_specs)
    if not specs:
        raise ValueError("scale registry cannot be empty")
    payload = {
        "registry_version": "scale-registry-v1",
        "scales": [to_primitive(item) for item in specs],
    }
    return f"scale:{content_hash(payload)[:24]}"




__all__ = [
    "ScaleRole",
    "ScaleSpec",
    "StructuralScale",
    "parse_scale_specs",
    "scale_registry_id",
]
