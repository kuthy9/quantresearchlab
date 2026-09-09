"""Research contracts: the per-clock engine snapshots replay and study consume.

These aggregate the other layers for audit and replay. Nothing in the
runtime action path reads them."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from contract.market.primitives import _exact_dataclass_pickle_state, _restore_exact_dataclass_pickle_state
from contract.eye.observation import MarketObservation
from contract.brain.context import NeutralMarketState
from contract.brain.belief import MarketBelief
from contract.decision.action import Decision
from contract.risk.assessment import RiskAssessment


ENGINE_SNAPSHOT_SCHEMA_VERSION = 4


NEUTRAL_ENGINE_SNAPSHOT_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class EngineSnapshot:
    observation: MarketObservation
    belief: MarketBelief
    decision: Decision
    risk: RiskAssessment
    neutral_market_state: NeutralMarketState | None = None

    schema_version: ClassVar[int] = ENGINE_SNAPSHOT_SCHEMA_VERSION

    @property
    def market_snapshot(self) -> "MarketSnapshot | None":
        return self.observation.market_snapshot

    def __post_init__(self) -> None:
        if (
            self.neutral_market_state is not None
            and (
                not isinstance(self.neutral_market_state, NeutralMarketState)
                or self.neutral_market_state.asof != self.observation.asof
                or self.neutral_market_state.scene_revision_id
                != self.observation.scene_revision_id
            )
        ):
            raise ValueError("Engine snapshot observation and neutral state differ")

    def __getstate__(self) -> Mapping[str, Any]:
        return _exact_dataclass_pickle_state(
            self,
            schema_version=ENGINE_SNAPSHOT_SCHEMA_VERSION,
            label="EngineSnapshot",
        )

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        _restore_exact_dataclass_pickle_state(
            self,
            state,
            schema_version=ENGINE_SNAPSHOT_SCHEMA_VERSION,
            label="EngineSnapshot",
        )


@dataclass(frozen=True)
class NeutralEngineSnapshot:
    """Action-free Engine projection for neutral market-case input."""

    observation: MarketObservation
    neutral_market_state: NeutralMarketState

    schema_version: ClassVar[int] = NEUTRAL_ENGINE_SNAPSHOT_SCHEMA_VERSION

    @property
    def market_snapshot(self) -> "MarketSnapshot | None":
        return self.observation.market_snapshot

    def __post_init__(self) -> None:
        if (
            not isinstance(self.neutral_market_state, NeutralMarketState)
            or self.observation.asof != self.neutral_market_state.asof
            or self.observation.scene_revision_id
            != self.neutral_market_state.scene_revision_id
        ):
            raise ValueError(
                "neutral Engine snapshot observation and state differ"
            )

    def __getstate__(self) -> Mapping[str, Any]:
        return _exact_dataclass_pickle_state(
            self,
            schema_version=NEUTRAL_ENGINE_SNAPSHOT_SCHEMA_VERSION,
            label="NeutralEngineSnapshot",
        )

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        _restore_exact_dataclass_pickle_state(
            self,
            state,
            schema_version=NEUTRAL_ENGINE_SNAPSHOT_SCHEMA_VERSION,
            label="NeutralEngineSnapshot",
        )


__all__ = [
    "ENGINE_SNAPSHOT_SCHEMA_VERSION",
    "EngineSnapshot",
    "NEUTRAL_ENGINE_SNAPSHOT_SCHEMA_VERSION",
    "NeutralEngineSnapshot",
]
