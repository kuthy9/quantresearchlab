"""The Trading Brain's published information stream.

One call per completed one-minute bar, one ``MarketBeliefState`` out.  This is
the only surface downstream consumers are meant to read; the proposer, the pool
and the updater are its internals.

The forecaster owns no market state and no history.  It reads the Eye's
published snapshot, asks the proposer what historically followed contexts like
this one, advances the pool, and publishes.  It is a belief producer, not an
action authority: every state it emits is ``shadow_only``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from contract.brain.forecast import (
    MarketBeliefState,
    ModeLibrary,
    belief_revision_id,
    normalized_entropy,
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
    oldest first, and it must end with ``close``.  The forecaster does not keep
    a price history of its own — that would be a second market-state authority.

    ``context`` and ``atr`` let a replay hand over values a study already derived
    from the live snapshot, instead of re-deriving them.  Supplying both makes
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
        library: ModeLibrary,
        protocol_fingerprint: str,
        pool_config: PoolConfig | None = None,
        updater_config: BeliefUpdaterConfig | None = None,
    ) -> None:
        if proposer.library is not library:
            raise ForecastError(
                "the proposer and the forecaster must share one mode library"
            )
        if not protocol_fingerprint:
            raise ForecastError("a published belief must cite its protocol fingerprint")
        self.proposer = proposer
        self.library = library
        self.protocol_fingerprint = protocol_fingerprint
        self.pool = HypothesisPool(
            library=library, config=pool_config, updater_config=updater_config
        )

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
                "a one-minute ATR is required: the mode library is fitted in ATR units"
            )
        proposals = self.proposer.propose(context)
        advance = self.pool.advance(
            asof=payload.asof,
            close=float(payload.close),
            high=float(payload.high),
            low=float(payload.low),
            atr=float(atr),
            proposals=proposals,
        )

        probabilities = tuple(item.probability for item in advance.hypotheses)
        uncertainty = normalized_entropy(probabilities + (advance.residual_probability,))
        revision_id = belief_revision_id(
            asof=advance.asof,
            hypotheses=advance.hypotheses,
            residual_probability=advance.residual_probability,
            mode_library_fingerprint=self.library.fingerprint,
            protocol_fingerprint=self.protocol_fingerprint,
        )
        return MarketBeliefState(
            asof=advance.asof,
            hypotheses=advance.hypotheses,
            residual_probability=advance.residual_probability,
            uncertainty=uncertainty,
            revision_id=revision_id,
            lifecycle_records=advance.records,
            mode_library_fingerprint=self.library.fingerprint,
            protocol_fingerprint=self.protocol_fingerprint,
        )


__all__ = ["ForecastError", "ForecastInput", "HypothesisForecaster"]
