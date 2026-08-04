"""Isolated causal projection of the frozen 5m displacement reducer."""
from __future__ import annotations

from collections import deque

import pandas as pd

from .causal import ReaderUpdate
from .displacement import (
    CausalDisplacementTracker,
    DisplacementLifecycle,
    DisplacementProtocol,
    DisplacementState,
    DisplacementTransition,
    DisplacementUpdate,
)
from .model import (
    Candle,
    DisplacementObservation,
    DisplacementTransitionObservation,
    Timeframe,
    aware_timestamp,
)


DATA_GAP_BOUNDARY = "data_gap_history_reset"
CONTRACT_BOUNDARY = "contract_change_history_reset"
REGISTERED_SESSION_BOUNDARY = "registered_session_reset"
REGISTERED_CLOSURE_ANOMALIES = frozenset(
    {
        "scheduled_market_closure",
        "scheduled_weekend_closure",
        "registered_full_session_closure",
        "registered_abbreviated_good_friday_closure",
        "registered_special_session_closure",
        "historical_settlement_pause",
        "registered_exchange_closure",
    }
)
READER_ANOMALY_WHITELIST = frozenset(
    {
        DATA_GAP_BOUNDARY,
        CONTRACT_BOUNDARY,
        *REGISTERED_CLOSURE_ANOMALIES,
    }
)
_STATE_CLOCK_FIELDS = (
    "prefix_last_admitted_at", "started_at", "active_at", "state_started_at",
    "last_updated_at", "observed_at", "terminal_at",
    "favorable_extreme_first_observed_at", "last_favorable_close_at",
)
_CURRENT_METRIC_FIELDS = (
    "age_minutes_at_last_admitted", "atr0", "efficiency",
    "favorable_extreme", "mean_body_fraction", "min_directional_clv",
    "nested_seed_observed", "net_points", "net_ticks", "origin_price",
    "real_episode_bar_count", "relative_atr", "speed_atr_per_bar",
    "travel_points", "v0", "volume_ratio", "volume_ready",
    "protection_price", "last_favorable_close", "interruption_run",
    "total_interruption_bars", "directional_bar_count",
    "neutral_bar_count", "opposite_bar_count",
    "directional_body_points", "opposite_body_points",
    "body_continuity", "activation_gate_count",
    "activation_weakest_ratio",
)


def _boundary_reason(anomalies: tuple[str, ...]) -> str | None:
    if CONTRACT_BOUNDARY in anomalies:
        return CONTRACT_BOUNDARY
    if DATA_GAP_BOUNDARY in anomalies:
        return DATA_GAP_BOUNDARY
    if any(value in REGISTERED_CLOSURE_ANOMALIES for value in anomalies):
        return REGISTERED_SESSION_BOUNDARY
    return None


def _validate_state_clocks(
    state: DisplacementState,
    asof: pd.Timestamp,
) -> None:
    for name in _STATE_CLOCK_FIELDS:
        value = getattr(state, name)
        if (
            value is not None
            and aware_timestamp(value, name=name) > asof
        ):
            raise ValueError("displacement state contains a future clock")


def _transition_observation(
    transition: DisplacementTransition,
    asof: pd.Timestamp,
    ordinal: int,
) -> DisplacementTransitionObservation:
    _validate_state_clocks(transition.state, asof)
    state = transition.state
    return DisplacementTransitionObservation(
        transition_id=transition.transition_id,
        entity_id=state.entity_id,
        lifecycle=state.lifecycle.value,
        reason=state.terminal_reason,
        observed_at=state.observed_at,
        direction=state.direction,
        ordinal=ordinal,
        started_at=state.started_at,
        active_at=state.active_at,
        state_started_at=state.state_started_at,
        terminal_at=state.terminal_at,
        prefix_last_admitted_at=state.prefix_last_admitted_at,
        terminal_evidence_candle_id=state.terminal_evidence_candle_id,
        admitted_candle_ids=state.admitted_candle_ids,
        state_metrics=_current_metrics(state),
    )


def _current_metrics(
    state: DisplacementState,
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (name, float(getattr(state, name)))
        for name in _CURRENT_METRIC_FIELDS
        if getattr(state, name) is not None
    )


class CausalDisplacementEye:
    """Consume reader outputs once and expose an isolated shadow view."""

    def __init__(self, protocol: DisplacementProtocol) -> None:
        if not isinstance(protocol, DisplacementProtocol):
            raise TypeError("a frozen displacement protocol is required")
        self._tracker = CausalDisplacementTracker(protocol)
        self._transitions: deque[DisplacementTransitionObservation] = deque(
            maxlen=64
        )
        self._last_observation: DisplacementObservation | None = None
        self._last_update: DisplacementUpdate | None = None
        self._last_batch: tuple[
            tuple[Candle, DisplacementUpdate], ...
        ] = ()

    @property
    def tracker(self) -> CausalDisplacementTracker:
        return self._tracker

    @property
    def last_observation(self) -> DisplacementObservation | None:
        return self._last_observation

    @property
    def last_update(self) -> DisplacementUpdate | None:
        return self._last_update

    @property
    def last_batch(
        self,
    ) -> tuple[tuple[Candle, DisplacementUpdate], ...]:
        """Return per-candle raw updates from the last successful call."""
        return getattr(self, "_last_batch", ())

    def on_update(self, update: ReaderUpdate) -> DisplacementObservation:
        anomalies = tuple(update.anomalies)
        unknown = tuple(
            value
            for value in anomalies
            if not isinstance(value, str) or value not in READER_ANOMALY_WHITELIST
        )
        if unknown:
            raise ValueError(f"unknown reader anomalies: {unknown!r}")
        asof = aware_timestamp(update.asof, name="reader update asof")
        boundary = _boundary_reason(anomalies)
        transitions: list[DisplacementTransition] = []
        batch: list[tuple[Candle, DisplacementUpdate]] = []

        if boundary is not None:
            result = self._tracker.on_boundary(boundary, asof)
            state = result.state
            transitions.extend(result.transitions)
        else:
            candles = tuple(
                update.newly_completed.get(Timeframe.M5, ())
            )
            invalid_candle = any(
                candle.timeframe is not Timeframe.M5
                or not candle.complete
                or candle.end > asof
                for candle in candles
            )
            invalid_order = any(
                right.end <= left.end
                for left, right in zip(candles[:-1], candles[1:])
            )
            if invalid_candle or invalid_order:
                raise ValueError("reader supplied invalid ordered completed M5")
            state = self._tracker.snapshot()
            for candle in candles:
                result = self._tracker.on_completed_5m(candle)
                batch.append((candle, result))
                state = result.state
                transitions.extend(result.transitions)

        projected = tuple(
            _transition_observation(transition, asof, ordinal)
            for ordinal, transition in enumerate(transitions)
        )
        recent = (tuple(self._transitions) + projected)[-64:]
        if state is not None:
            _validate_state_clocks(state, asof)
            if state.lifecycle not in {
                DisplacementLifecycle.STARTED,
                DisplacementLifecycle.ACTIVE,
            }:
                raise ValueError("terminal displacement cannot be current")

        observation = DisplacementObservation(
            asof=asof,
            lifecycle=(
                DisplacementLifecycle.IDLE.value
                if state is None
                else state.lifecycle.value
            ),
            current_entity_id=None if state is None else state.entity_id,
            current_direction=None if state is None else state.direction,
            current_state_observed_at=None if state is None else state.observed_at,
            current_started_at=None if state is None else state.started_at,
            current_active_at=None if state is None else state.active_at,
            current_last_admitted_at=(
                None if state is None else state.prefix_last_admitted_at
            ),
            current_metrics=None if state is None else _current_metrics(state),
            latest_transition=recent[-1] if recent else None,
            recent_transitions=recent,
            transitions_this_update=projected,
            reader_anomalies=anomalies,
        )
        self._transitions.extend(projected)
        self._last_observation = observation
        self._last_update = DisplacementUpdate(state, tuple(transitions))
        self._last_batch = tuple(batch)
        return observation


__all__ = [
    "CausalDisplacementEye",
    "READER_ANOMALY_WHITELIST",
]
