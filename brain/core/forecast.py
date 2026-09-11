"""The Trading Brain's published information stream.

One call per completed one-minute bar, one ``MarketBeliefState`` out. This is
the only surface downstream consumers are meant to read; the proposer, the pool
and the updater are its internals.

The forecaster owns no market state and no history. It reads the Eye's published
snapshot, asks the proposer what the conditional future cloud looks like from
here, advances the pool, and publishes. It is a belief producer, not an action
authority: every state it emits is ``shadow_only``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from contract.brain.forecast import (
    BeliefUncertainty,
    MarketBeliefState,
    belief_revision_id,
    mode_ambiguity,
    retrieval_confidence,
)
from contract.market import Timeframe

from .belief_updater import BeliefUpdaterConfig
from .hypothesis_pool import HypothesisPool, PoolConfig
from .hypothesis_proposer import HypothesisProposer, observation_features


class ForecastError(RuntimeError):
    """The forecaster refuses to publish a belief it cannot stand behind."""


@dataclass(frozen=True)
class ForecastInput:
    """One completed bar, as the forecaster needs to see it.

    ``closes`` is the completed one-minute close history ending at this bar,
    oldest first. The forecaster keeps no price history of its own — that would
    be a second market-state authority.

    ``context`` and ``atr`` let a replay hand over values a study already derived
    from the live snapshot instead of re-deriving them. Supplying both makes
    ``snapshot`` unnecessary; supplying neither requires it.
    """

    asof: pd.Timestamp
    close: float
    high: float
    low: float
    snapshot: Any = None
    closes: Sequence[float] = ()
    context: Sequence[float] | None = None
    atr: float | None = None


class HypothesisForecaster:
    """Turns one Eye observation per minute into one published belief."""

    def __init__(
        self,
        *,
        proposer: HypothesisProposer,
        protocol_fingerprint: str,
        pool_config: PoolConfig | None = None,
        updater_config: BeliefUpdaterConfig | None = None,
    ) -> None:
        if not protocol_fingerprint:
            raise ForecastError("a published belief must cite its protocol fingerprint")
        self.proposer = proposer
        self.protocol_fingerprint = protocol_fingerprint
        self.pool = HypothesisPool(config=pool_config, updater_config=updater_config)

    @property
    def index_fingerprint(self) -> str:
        return self.proposer.index.fingerprint

    def reset(self) -> None:
        self.pool.reset()

    def observe(self, payload: ForecastInput) -> MarketBeliefState:
        """Advance one clock and publish the resulting belief."""

        context = payload.context
        atr = payload.atr
        if context is None or atr is None:
            snapshot = payload.snapshot
            if snapshot is None:
                raise ForecastError(
                    "the Brain cannot form a belief without an Eye snapshot"
                )
            if atr is None:
                states = getattr(snapshot, "timeframe_states", None) or {}
                m1 = states.get(Timeframe.M1)
                atr = getattr(getattr(m1, "quality", None), "atr", None)
            if context is None:
                context = observation_features(
                    snapshot,
                    closes=payload.closes,
                    bar_high_low=(float(payload.high), float(payload.low)),
                )
        if atr is None:
            raise ForecastError(
                "a one-minute ATR is required: every curve is expressed in ATR units"
            )

        cloud = self.proposer.propose(context, asof=payload.asof)
        advance = self.pool.advance(
            asof=payload.asof,
            close=float(payload.close),
            high=float(payload.high),
            low=float(payload.low),
            atr=float(atr),
            cloud=cloud,
        )

        # The three ways this belief can be wrong, kept separate because they
        # have different remedies. Mode ambiguity says the covered future is
        # contested; representation coverage says the pool does not speak for
        # much of the cloud; retrieval confidence says the cloud itself rests on
        # thin precedent. A belief can be sharp and confident on all three
        # counts and still be built on twelve remote analogues, which is exactly
        # the failure the third number exists to surface.
        by_node = {node.node_id: node for node in cloud.nodes}
        covered = tuple(
            (by_node[item.node_id], item.probability)
            for item in advance.hypotheses
            if item.node_id in by_node
        )
        uncertainty = BeliefUncertainty(
            mode_ambiguity=mode_ambiguity(
                tuple(node.components for node, _ in covered),
                tuple(probability for _, probability in covered),
                scale=self.proposer.index.component_scale,
            ),
            # Measured from the cloud, not from the normalized posterior: how
            # much of what actually followed similar contexts the live claims
            # speak for is a different question from how the probability mass is
            # spread over the claims that do exist.
            representation_coverage=sum(node.mass for node, _ in covered),
            retrieval_confidence=retrieval_confidence(
                neighbour_count=cloud.neighbour_count,
                mean_distance=cloud.mean_neighbour_distance,
                target_count=self.proposer.config.neighbours,
                distance_scale=self.proposer.index.context_scale,
            ),
        )
        revision_id = belief_revision_id(
            asof=advance.asof,
            hypotheses=advance.hypotheses,
            residual_probability=advance.residual_probability,
            index_fingerprint=self.index_fingerprint,
            protocol_fingerprint=self.protocol_fingerprint,
        )
        return MarketBeliefState(
            asof=advance.asof,
            hypotheses=advance.hypotheses,
            residual_probability=advance.residual_probability,
            uncertainty=uncertainty,
            revision_id=revision_id,
            cloud=cloud,
            lifecycle_records=advance.records,
            index_fingerprint=self.index_fingerprint,
            protocol_fingerprint=self.protocol_fingerprint,
        )


__all__ = ["ForecastError", "ForecastInput", "HypothesisForecaster"]
