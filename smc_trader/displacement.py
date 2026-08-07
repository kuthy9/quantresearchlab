"""Forward-only 5m displacement episodes.

One reducer owns the complete lifecycle.  It starts at the first causally
visible directional candidate, evaluates activation from cumulative episode
geometry, admits one finite interruption, and emits every same-clock
transition in causal order.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from .model import (
    Candle,
    Direction,
    Timeframe,
    aware_timestamp,
    candle_identity,
)


EPISODE_PROTOCOL_VERSION = "3.2.0-displacement-episode.3"


class DisplacementLifecycle(str, Enum):
    IDLE = "idle"
    STARTED = "started"
    ACTIVE = "active"
    EXHAUSTED = "exhausted"
    CENSORED = "censored"


@dataclass(frozen=True)
class DisplacementProtocol:
    protocol_hash: str
    tick_size: float
    protocol_version: str = EPISODE_PROTOCOL_VERSION
    atr_baseline_bars: int = 14

    # A strong one-bar impulse seeds the cumulative opposite-displacement probe.
    seed_body_fraction: float = 0.60
    seed_directional_clv: float = 0.75
    seed_tr_atr: float = 0.80

    activation_min_bar: int = 2
    activation_max_bar: int = 0  # zero means that no activation deadline exists
    activation_relative_atr: float = 1.00
    activation_efficiency: float = 0.70
    activation_speed: float = 0.30
    activation_mean_body_fraction: float = 0.60
    activation_min_directional_clv: float = 0.0  # descriptive only
    continuation_progress_ticks: int = 1

    candidate_directional_clv: float = 0.50
    activation_body_continuity: float = 0.60
    consecutive_interruptions_max: int = 1
    downstream_authoritative: bool = False

    def __post_init__(self) -> None:
        if self.protocol_version != EPISODE_PROTOCOL_VERSION:
            raise ValueError("unsupported displacement protocol version")
        if (
            not isinstance(self.protocol_hash, str)
            or len(self.protocol_hash) != 64
            or any(value not in "0123456789abcdef" for value in self.protocol_hash)
        ):
            raise ValueError("displacement protocol hash is invalid")
        finite = (
            self.tick_size,
            self.seed_body_fraction,
            self.seed_directional_clv,
            self.seed_tr_atr,
            self.activation_relative_atr,
            self.activation_efficiency,
            self.activation_speed,
            self.activation_mean_body_fraction,
            self.activation_min_directional_clv,
            self.candidate_directional_clv,
            self.activation_body_continuity,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("displacement protocol contains a non-finite value")
        if (
            self.tick_size <= 0
            or self.atr_baseline_bars < 1
            or self.activation_min_bar < 1
            or self.activation_max_bar < 0
            or self.continuation_progress_ticks < 1
            or self.consecutive_interruptions_max < 0
            or self.activation_relative_atr <= 0.0
            or self.activation_speed <= 0.0
            or self.activation_efficiency <= 0.0
            or self.activation_mean_body_fraction <= 0.0
            or self.activation_body_continuity <= 0.0
            or not 0.0 <= self.candidate_directional_clv <= 1.0
            or not 0.0 <= self.activation_body_continuity <= 1.0
            or not 0.0 <= self.activation_efficiency <= 1.0
            or not 0.0 <= self.activation_mean_body_fraction <= 1.0
            or not 0.0 <= self.seed_body_fraction <= 1.0
            or not 0.0 <= self.seed_directional_clv <= 1.0
        ):
            raise ValueError("displacement protocol values are outside their domains")
        if type(self.downstream_authoritative) is not bool:
            raise ValueError("downstream authority must be explicit")

    @classmethod
    def from_file(cls, path: str | Path) -> "DisplacementProtocol":
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = Path(__file__).resolve().parents[1] / source
        raw = source.read_bytes()
        payload = json.loads(raw)
        thresholds = payload["thresholds"]
        version = str(payload["protocol_version"])
        if version != EPISODE_PROTOCOL_VERSION:
            raise ValueError("unsupported displacement protocol version")
        if payload.get("timeframe") != "5m":
            raise ValueError("displacement episode protocol requires 5m")
        tick_size = payload.get("tick_size")
        if tick_size is None:
            tick_size = payload["input_contract"]["tick_size"]

        protocol = cls(
            protocol_hash=hashlib.sha256(raw).hexdigest(),
            protocol_version=version,
            tick_size=float(tick_size),
            atr_baseline_bars=int(
                thresholds["strictly_prior_baseline_observations"]
            ),
            seed_body_fraction=float(
                thresholds["strong_reverse_body_fraction_min"]
            ),
            seed_directional_clv=float(
                thresholds["strong_reverse_directional_clv_min"]
            ),
            seed_tr_atr=float(
                thresholds["strong_reverse_tr_over_atr_min"]
            ),
            activation_min_bar=int(thresholds["activation_episode_bar_min"]),
            activation_max_bar=0,
            activation_relative_atr=float(
                thresholds["activation_relative_atr_min"]
            ),
            activation_efficiency=float(
                thresholds["activation_efficiency_min"]
            ),
            activation_speed=float(
                thresholds["activation_speed_atr_per_bar_min"]
            ),
            activation_mean_body_fraction=float(
                thresholds["activation_mean_body_fraction_min"]
            ),
            activation_min_directional_clv=0.0,
            continuation_progress_ticks=int(
                thresholds["candidate_close_progress_ticks_min"]
            ),
            candidate_directional_clv=float(
                thresholds["candidate_directional_clv_min"]
            ),
            activation_body_continuity=float(
                thresholds["activation_body_continuity_min"]
            ),
            consecutive_interruptions_max=int(
                thresholds["consecutive_interruptions_max"]
            ),
            downstream_authoritative=payload["downstream_authoritative"],
        )
        return protocol


@dataclass(frozen=True)
class DisplacementState:
    entity_id: str
    protocol_hash: str
    symbol: str
    instrument_id: int
    timeframe: Timeframe
    direction: Direction
    lifecycle: DisplacementLifecycle
    terminal_reason: str | None
    seed_candle_id: str
    origin_price: float
    last_valid_candle_id: str
    prefix_last_admitted_at: pd.Timestamp
    started_at: pd.Timestamp
    active_at: pd.Timestamp | None
    state_started_at: pd.Timestamp
    last_updated_at: pd.Timestamp
    observed_at: pd.Timestamp
    terminal_at: pd.Timestamp | None
    terminal_evidence_candle_id: str | None
    atr0: float
    v0: float | None
    volume_ready: bool
    real_episode_bar_count: int
    age_minutes_at_last_admitted: int
    net_points: float
    net_ticks: int
    relative_atr: float
    travel_points: float
    efficiency: float
    speed_atr_per_bar: float
    mean_body_fraction: float
    mean_overlap_ratio: float
    max_overlap_ratio: float
    mean_directional_clv: float
    min_directional_clv: float
    volume_ratio: float | None
    favorable_extreme: float
    favorable_extreme_first_observed_at: pd.Timestamp
    nested_seed_observed: bool
    prefix_commitment: str

    protection_price: float
    last_favorable_close: float
    last_favorable_close_at: pd.Timestamp
    interruption_run: int
    total_interruption_bars: int
    directional_bar_count: int
    neutral_bar_count: int
    opposite_bar_count: int
    directional_body_points: float
    opposite_body_points: float
    body_continuity: float
    activation_gate_count: int
    activation_weakest_ratio: float
    admitted_candle_ids: tuple[str, ...]


@dataclass(frozen=True)
class DisplacementTransition:
    transition_id: str
    state: DisplacementState


@dataclass(frozen=True)
class DisplacementUpdate:
    state: DisplacementState | None
    transitions: tuple[DisplacementTransition, ...] = ()


@dataclass
class _OpenEpisode:
    state: DisplacementState
    last_close: float
    last_high: float
    last_low: float
    body_fraction_sum: float
    overlap_ratio_sum: float
    overlap_pair_count: int
    directional_clv_sum: float
    volume_sum: float


def _canonical(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, float):
        return format(value, ".17g")
    if value is None:
        return ""
    return str(value)


def _identity(*parts: Any) -> str:
    raw = json.dumps(
        [_canonical(value) for value in parts],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CausalDisplacementTracker:
    """The single incremental displacement-episode reducer."""

    def __init__(self, protocol: DisplacementProtocol) -> None:
        if protocol.protocol_version != EPISODE_PROTOCOL_VERSION:
            raise ValueError(
                "legacy displacement protocols are archived and cannot "
                "drive the episode reducer"
            )
        self.protocol = protocol
        self._trs: deque[float] = deque(
            maxlen=protocol.atr_baseline_bars
        )
        self._volumes: deque[float] = deque(
            maxlen=protocol.atr_baseline_bars
        )
        self._prior_close: float | None = None
        self._identity: tuple[str, int] | None = None
        self._open: _OpenEpisode | None = None
        self._reverse_probe: _OpenEpisode | None = None
        self._last_clock: pd.Timestamp | None = None
        self._last_input_kind: str | None = None
        self._failed = False

    def snapshot(self) -> DisplacementState | None:
        return None if self._open is None else self._open.state

    def _ticks(self, value: float) -> int:
        scaled = float(value) / self.protocol.tick_size
        rounded = round(scaled)
        if not math.isfinite(scaled) or abs(scaled - rounded) > 1e-6:
            raise ValueError("off-grid price")
        return int(rounded)

    def _candle_id(self, candle: Candle) -> str:
        return candle_identity(
            candle,
            tick_size=self.protocol.tick_size,
        )

    def _geometry(
        self,
        candle: Candle,
        direction_sign: float,
    ) -> tuple[int, float, float]:
        body_ticks = self._ticks(candle.close) - self._ticks(candle.open)
        direction = 1 if body_ticks >= 1 else (-1 if body_ticks <= -1 else 0)
        width = max(
            float(candle.high - candle.low),
            self.protocol.tick_size,
        )
        body_fraction = abs(float(candle.close - candle.open)) / width
        clv = (
            (candle.close - candle.low) / width
            if direction_sign > 0
            else (candle.high - candle.close) / width
        )
        return direction, float(body_fraction), float(clv)

    def _baseline(self) -> tuple[float | None, float | None]:
        if len(self._trs) != self.protocol.atr_baseline_bars:
            return None, None
        atr0 = sum(self._trs) / self.protocol.atr_baseline_bars
        positive = [value for value in self._volumes if value > 0]
        v0 = (
            float(median(positive))
            if len(positive) == self.protocol.atr_baseline_bars
            else None
        )
        return float(atr0), v0

    def _candidate_direction(
        self,
        candle: Candle,
        prior_close: float,
    ) -> int:
        body_direction, _, _ = self._geometry(candle, 1.0)
        if body_direction == 0:
            return 0
        _, _, clv = self._geometry(candle, float(body_direction))
        close_progress = body_direction * (
            self._ticks(candle.close) - self._ticks(prior_close)
        )
        if (
            close_progress < self.protocol.continuation_progress_ticks
            or clv < self.protocol.candidate_directional_clv
        ):
            return 0
        return body_direction

    def _strong_impulse(
        self,
        candle: Candle,
        tr: float,
        atr0: float | None,
        direction: int,
    ) -> bool:
        if atr0 is None or direction == 0:
            return False
        body_direction, body_fraction, _ = self._geometry(candle, 1.0)
        if body_direction != direction:
            return False
        _, _, clv = self._geometry(candle, float(direction))
        return bool(
            body_fraction >= self.protocol.seed_body_fraction
            and clv >= self.protocol.seed_directional_clv
            and tr / atr0 >= self.protocol.seed_tr_atr
        )

    def _transition(
        self,
        state: DisplacementState,
    ) -> DisplacementTransition:
        transition_id = _identity(
            "displacement-transition-v2",
            state.entity_id,
            state.lifecycle,
            state.state_started_at,
            state.terminal_reason,
        )
        return DisplacementTransition(transition_id, state)

    def _activation_components(
        self,
        state: DisplacementState,
    ) -> tuple[float, ...]:
        protocol = self.protocol
        return (
            state.real_episode_bar_count / protocol.activation_min_bar,
            state.relative_atr / protocol.activation_relative_atr,
            state.efficiency / protocol.activation_efficiency,
            state.speed_atr_per_bar / protocol.activation_speed,
            state.mean_body_fraction
            / protocol.activation_mean_body_fraction,
            state.body_continuity
            / protocol.activation_body_continuity,
        )

    def _with_activation_diagnostics(
        self,
        state: DisplacementState,
    ) -> DisplacementState:
        components = self._activation_components(state)
        return replace(
            state,
            activation_gate_count=sum(
                int(value >= 1.0) for value in components
            ),
            activation_weakest_ratio=min(components),
        )

    def _seed(
        self,
        candle: Candle,
        atr0: float,
        v0: float | None,
        direction_i: int,
    ) -> tuple[_OpenEpisode, DisplacementTransition]:
        direction = Direction.LONG if direction_i > 0 else Direction.SHORT
        q = direction.sign
        _, body_fraction, clv = self._geometry(candle, q)
        candle_id = self._candle_id(candle)
        net = q * float(candle.close - candle.open)
        travel = abs(float(candle.close - candle.open))
        relative = net / atr0
        prefix = _identity("displacement-prefix-v2", "", candle_id)
        entity_id = _identity(
            "displacement-entity-v2",
            self.protocol.protocol_hash,
            candle.symbol,
            candle.instrument_id,
            Timeframe.M5,
            direction,
            candle_id,
            candle.end,
        )
        state = DisplacementState(
            entity_id=entity_id,
            protocol_hash=self.protocol.protocol_hash,
            symbol=candle.symbol,
            instrument_id=candle.instrument_id,
            timeframe=Timeframe.M5,
            direction=direction,
            lifecycle=DisplacementLifecycle.STARTED,
            terminal_reason=None,
            seed_candle_id=candle_id,
            origin_price=float(candle.open),
            last_valid_candle_id=candle_id,
            prefix_last_admitted_at=candle.end,
            started_at=candle.end,
            active_at=None,
            state_started_at=candle.end,
            last_updated_at=candle.end,
            observed_at=candle.end,
            terminal_at=None,
            terminal_evidence_candle_id=None,
            atr0=atr0,
            v0=v0,
            volume_ready=v0 is not None,
            real_episode_bar_count=1,
            age_minutes_at_last_admitted=0,
            net_points=net,
            net_ticks=int(round(net / self.protocol.tick_size)),
            relative_atr=relative,
            travel_points=travel,
            efficiency=max(net, 0.0)
            / max(travel, self.protocol.tick_size),
            speed_atr_per_bar=relative,
            mean_body_fraction=body_fraction,
            mean_overlap_ratio=0.0,
            max_overlap_ratio=0.0,
            mean_directional_clv=clv,
            min_directional_clv=clv,
            volume_ratio=(
                None if v0 is None else float(candle.volume) / v0
            ),
            favorable_extreme=float(
                candle.high if q > 0 else candle.low
            ),
            favorable_extreme_first_observed_at=candle.end,
            nested_seed_observed=False,
            prefix_commitment=prefix,
            protection_price=float(
                candle.low if q > 0 else candle.high
            ),
            last_favorable_close=float(candle.close),
            last_favorable_close_at=candle.end,
            interruption_run=0,
            total_interruption_bars=0,
            directional_bar_count=1,
            neutral_bar_count=0,
            opposite_bar_count=0,
            directional_body_points=max(net, 0.0),
            opposite_body_points=max(-net, 0.0),
            body_continuity=1.0,
            activation_gate_count=0,
            activation_weakest_ratio=0.0,
            admitted_candle_ids=(candle_id,),
        )
        state = self._with_activation_diagnostics(state)
        opened = _OpenEpisode(
            state=state,
            last_close=float(candle.close),
            last_high=float(candle.high),
            last_low=float(candle.low),
            body_fraction_sum=body_fraction,
            overlap_ratio_sum=0.0,
            overlap_pair_count=0,
            directional_clv_sum=clv,
            volume_sum=float(candle.volume),
        )
        return opened, self._transition(state)

    def _favorable_close_progress(
        self,
        candle: Candle,
        state: DisplacementState,
    ) -> bool:
        return (
            state.direction.sign
            * (
                self._ticks(candle.close)
                - self._ticks(state.last_favorable_close)
            )
            >= self.protocol.continuation_progress_ticks
        )

    def _body_kind(
        self,
        candle: Candle,
        direction: Direction,
    ) -> str:
        body_direction, _, _ = self._geometry(candle, direction.sign)
        if body_direction == 0:
            return "neutral"
        if body_direction == int(direction.sign):
            return "directional"
        return "opposite"

    def _breaks_protection(
        self,
        candle: Candle,
        state: DisplacementState,
    ) -> bool:
        return (
            state.direction.sign
            * (
                self._ticks(candle.close)
                - self._ticks(state.protection_price)
            )
            < 0
        )

    def _advanced(
        self,
        opened: _OpenEpisode,
        candle: Candle,
        *,
        nested_strong_impulse: bool = False,
    ) -> _OpenEpisode:
        prior = opened.state
        q = prior.direction.sign
        _, body_fraction, clv = self._geometry(candle, q)
        kind = self._body_kind(candle, prior.direction)
        favorable_progress = self._favorable_close_progress(candle, prior)
        count = prior.real_episode_bar_count + 1
        body_sum = opened.body_fraction_sum + body_fraction
        prior_range = max(
            opened.last_high - opened.last_low,
            self.protocol.tick_size,
        )
        current_range = max(
            float(candle.high - candle.low),
            self.protocol.tick_size,
        )
        overlap_points = max(
            0.0,
            min(opened.last_high, float(candle.high))
            - max(opened.last_low, float(candle.low)),
        )
        overlap_ratio = min(
            1.0,
            overlap_points / min(prior_range, current_range),
        )
        overlap_sum = opened.overlap_ratio_sum + overlap_ratio
        overlap_count = opened.overlap_pair_count + 1
        directional_clv_sum = opened.directional_clv_sum + clv
        volume_sum = opened.volume_sum + float(candle.volume)
        net = q * float(candle.close - prior.origin_price)
        travel = prior.travel_points + abs(
            float(candle.close - opened.last_close)
        )
        relative = net / prior.atr0
        candle_id = self._candle_id(candle)
        favorable = float(candle.high if q > 0 else candle.low)
        improves = (
            favorable > prior.favorable_extreme
            if q > 0
            else favorable < prior.favorable_extreme
        )
        signed_body = q * float(candle.close - candle.open)
        directional_body = (
            prior.directional_body_points + max(signed_body, 0.0)
        )
        opposite_body = (
            prior.opposite_body_points + max(-signed_body, 0.0)
        )
        body_total = directional_body + opposite_body
        body_continuity = (
            1.0 if body_total <= 0.0 else directional_body / body_total
        )
        interruption_run = (
            0 if favorable_progress else prior.interruption_run + 1
        )
        state = replace(
            prior,
            last_valid_candle_id=candle_id,
            prefix_last_admitted_at=candle.end,
            last_updated_at=candle.end,
            observed_at=candle.end,
            real_episode_bar_count=count,
            age_minutes_at_last_admitted=int(
                (candle.end - prior.started_at).total_seconds() // 60
            ),
            net_points=net,
            net_ticks=int(round(net / self.protocol.tick_size)),
            relative_atr=relative,
            travel_points=travel,
            efficiency=max(net, 0.0)
            / max(travel, self.protocol.tick_size),
            speed_atr_per_bar=relative / count,
            mean_body_fraction=body_sum / count,
            mean_overlap_ratio=overlap_sum / overlap_count,
            max_overlap_ratio=max(prior.max_overlap_ratio, overlap_ratio),
            mean_directional_clv=directional_clv_sum / count,
            min_directional_clv=min(prior.min_directional_clv, clv),
            volume_ratio=(
                None if prior.v0 is None else volume_sum / count / prior.v0
            ),
            favorable_extreme=(
                favorable if improves else prior.favorable_extreme
            ),
            favorable_extreme_first_observed_at=(
                candle.end
                if improves
                else prior.favorable_extreme_first_observed_at
            ),
            nested_seed_observed=(
                prior.nested_seed_observed or nested_strong_impulse
            ),
            prefix_commitment=_identity(
                "displacement-prefix-v2",
                prior.prefix_commitment,
                candle_id,
            ),
            last_favorable_close=(
                float(candle.close)
                if favorable_progress
                else prior.last_favorable_close
            ),
            last_favorable_close_at=(
                candle.end
                if favorable_progress
                else prior.last_favorable_close_at
            ),
            interruption_run=interruption_run,
            total_interruption_bars=(
                prior.total_interruption_bars
                + int(not favorable_progress)
            ),
            directional_bar_count=(
                prior.directional_bar_count + int(kind == "directional")
            ),
            neutral_bar_count=(
                prior.neutral_bar_count + int(kind == "neutral")
            ),
            opposite_bar_count=(
                prior.opposite_bar_count + int(kind == "opposite")
            ),
            directional_body_points=directional_body,
            opposite_body_points=opposite_body,
            body_continuity=body_continuity,
            admitted_candle_ids=(
                *prior.admitted_candle_ids,
                candle_id,
            ),
        )
        state = self._with_activation_diagnostics(state)
        return _OpenEpisode(
            state=state,
            last_close=float(candle.close),
            last_high=float(candle.high),
            last_low=float(candle.low),
            body_fraction_sum=body_sum,
            overlap_ratio_sum=overlap_sum,
            overlap_pair_count=overlap_count,
            directional_clv_sum=directional_clv_sum,
            volume_sum=volume_sum,
        )

    def _apply_started_recovery_rule(
        self,
        prior: DisplacementState,
        proposed: _OpenEpisode,
        *,
        favorable_progress: bool,
    ) -> _OpenEpisode:
        state = proposed.state
        if not (
            prior.lifecycle is DisplacementLifecycle.STARTED
            and prior.interruption_run > 0
            and favorable_progress
            and not self._activates(state)
            and state.activation_gate_count
            <= prior.activation_gate_count
        ):
            return proposed
        return replace(
            proposed,
            state=replace(
                state,
                interruption_run=prior.interruption_run + 1,
                total_interruption_bars=(
                    prior.total_interruption_bars + 1
                ),
            ),
        )

    def _terminal(
        self,
        lifecycle: DisplacementLifecycle,
        reason: str,
        observed_at: pd.Timestamp,
        evidence_id: str | None,
    ) -> DisplacementState:
        assert self._open is not None
        return replace(
            self._open.state,
            lifecycle=lifecycle,
            terminal_reason=reason,
            state_started_at=observed_at,
            last_updated_at=observed_at,
            observed_at=observed_at,
            terminal_at=observed_at,
            terminal_evidence_candle_id=evidence_id,
        )

    def _activates(self, state: DisplacementState) -> bool:
        return bool(
            state.lifecycle is DisplacementLifecycle.STARTED
            and state.activation_gate_count
            == len(self._activation_components(state))
        )

    def _reset(self) -> None:
        self._trs.clear()
        self._volumes.clear()
        self._prior_close = None
        self._identity = None
        self._open = None
        self._reverse_probe = None

    def on_boundary(
        self,
        reason: str,
        observed_at: pd.Timestamp,
    ) -> DisplacementUpdate:
        if self._failed:
            raise RuntimeError("displacement tracker is terminally failed")
        if not reason:
            raise ValueError("boundary reason is required")
        clock = aware_timestamp(observed_at, name="displacement.boundary")
        if self._last_clock is not None and clock < self._last_clock:
            raise ValueError("boundary clock is out of order")
        if clock == self._last_clock and self._last_input_kind != "candle":
            raise ValueError("duplicate boundary clock")
        transitions: tuple[DisplacementTransition, ...] = ()
        if self._open is not None:
            terminal = self._terminal(
                DisplacementLifecycle.CENSORED,
                reason,
                clock,
                None,
            )
            transitions = (self._transition(terminal),)
        self._reset()
        self._last_clock = clock
        self._last_input_kind = "boundary"
        return DisplacementUpdate(None, transitions)

    def on_data_anomaly(
        self,
        observed_at: pd.Timestamp,
        reason: str,
    ) -> DisplacementUpdate:
        update = self.on_boundary(reason, observed_at)
        self._failed = True
        return update

    def _start_if_candidate(
        self,
        candle: Candle,
        *,
        atr0: float | None,
        v0: float | None,
        prior_close: float,
        transitions: list[DisplacementTransition],
    ) -> None:
        direction = self._candidate_direction(candle, prior_close)
        if direction == 0 or atr0 is None:
            return
        self._open, transition = self._seed(
            candle,
            float(atr0),
            v0,
            direction,
        )
        transitions.append(transition)

    def _update_reverse_probe(
        self,
        candle: Candle,
        *,
        prior_close: float,
        atr0: float | None,
        v0: float | None,
        primary: DisplacementState,
    ) -> None:
        if primary.lifecycle is not DisplacementLifecycle.ACTIVE:
            self._reverse_probe = None
            return
        desired = -int(primary.direction.sign)
        candidate = self._candidate_direction(candle, prior_close)
        if self._reverse_probe is None:
            if candidate != desired or atr0 is None:
                return
            self._reverse_probe, _ = self._seed(
                candle,
                float(atr0),
                v0,
                desired,
            )
            return
        if int(self._reverse_probe.state.direction.sign) != desired:
            self._reverse_probe = None
            return
        prior_probe = self._reverse_probe.state
        proposed = self._advanced(
            self._reverse_probe,
            candle,
            nested_strong_impulse=self._strong_impulse(
                candle,
                max(
                    float(candle.high - candle.low),
                    abs(float(candle.high - prior_close)),
                    abs(float(candle.low - prior_close)),
                    self.protocol.tick_size,
                ),
                atr0,
                desired,
            ),
        )
        proposed = self._apply_started_recovery_rule(
            prior_probe,
            proposed,
            favorable_progress=self._favorable_close_progress(
                candle,
                prior_probe,
            ),
        )
        if (
            self._breaks_protection(candle, prior_probe)
            or proposed.state.interruption_run
            > self.protocol.consecutive_interruptions_max
        ):
            self._reverse_probe = None
            return
        self._reverse_probe = proposed

    def _promote_probe(
        self,
        probe: _OpenEpisode,
        *,
        observed_at: pd.Timestamp,
        activate: bool,
        transitions: list[DisplacementTransition],
    ) -> None:
        started = replace(
            probe.state,
            lifecycle=DisplacementLifecycle.STARTED,
            active_at=None,
            state_started_at=observed_at,
            last_updated_at=observed_at,
            observed_at=observed_at,
        )
        opened = replace(probe, state=started)
        transitions.append(self._transition(started))
        if activate:
            active = replace(
                started,
                lifecycle=DisplacementLifecycle.ACTIVE,
                active_at=observed_at,
                state_started_at=observed_at,
            )
            opened = replace(opened, state=active)
            transitions.append(self._transition(active))
        self._open = opened

    def on_completed_5m(self, candle: Candle) -> DisplacementUpdate:
        if self._failed:
            raise RuntimeError("displacement tracker is terminally failed")
        if candle.timeframe is not Timeframe.M5 or not candle.complete:
            raise ValueError("a completed 5m candle is required")
        if self._last_clock is not None and candle.end <= self._last_clock:
            self._failed = True
            raise ValueError("duplicate or out-of-order completed candle")
        if (
            self._last_clock is not None
            and self._last_input_kind == "candle"
            and candle.start < self._last_clock
        ):
            self._failed = True
            raise ValueError("completed candle overlaps the prior causal clock")
        if (
            self._last_input_kind == "candle"
            and candle.start > self._last_clock
        ):
            return self.on_boundary("data_gap_history_reset", candle.end)
        if not candle.real_completed:
            return self.on_boundary("synthetic_interruption", candle.end)
        if (
            candle.expected_minutes,
            candle.real_minutes,
            candle.observed_minutes,
        ) != (5, 5, 5):
            return self.on_boundary("registered_session_reset", candle.end)

        values = (
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        )
        try:
            if (
                not all(math.isfinite(float(value)) for value in values)
                or candle.volume < 0
            ):
                raise ValueError("invalid OHLCV")
            for value in (
                candle.open,
                candle.high,
                candle.low,
                candle.close,
            ):
                self._ticks(value)
        except ValueError:
            return self.on_data_anomaly(candle.end, "data_anomaly")

        identity = (candle.symbol, int(candle.instrument_id))
        if self._identity is not None and identity != self._identity:
            return self.on_boundary(
                "contract_change_history_reset",
                candle.end,
            )
        self._identity = identity
        if self._prior_close is None:
            self._prior_close = float(candle.close)
            self._last_clock = candle.end
            self._last_input_kind = "candle"
            return DisplacementUpdate(self.snapshot())

        prior_close = float(self._prior_close)
        tr = max(
            float(candle.high - candle.low),
            abs(float(candle.high - prior_close)),
            abs(float(candle.low - prior_close)),
            self.protocol.tick_size,
        )
        atr0, v0 = self._baseline()
        transitions: list[DisplacementTransition] = []

        if self._open is None:
            self._start_if_candidate(
                candle,
                atr0=atr0,
                v0=v0,
                prior_close=prior_close,
                transitions=transitions,
            )
        else:
            prior = self._open.state
            candidate_direction = self._candidate_direction(
                candle,
                prior_close,
            )
            protection_broken = self._breaks_protection(candle, prior)
            favorable_progress = self._favorable_close_progress(
                candle,
                prior,
            )
            proposed = self._advanced(
                self._open,
                candle,
                nested_strong_impulse=self._strong_impulse(
                    candle,
                    tr,
                    atr0,
                    int(prior.direction.sign),
                ),
            )
            proposed = self._apply_started_recovery_rule(
                prior,
                proposed,
                favorable_progress=favorable_progress,
            )
            self._update_reverse_probe(
                candle,
                prior_close=prior_close,
                atr0=atr0,
                v0=v0,
                primary=prior,
            )
            qualified_reverse = bool(
                self._reverse_probe is not None
                and self._activates(self._reverse_probe.state)
            )

            terminal_reason: str | None = None
            if qualified_reverse:
                terminal_reason = "qualified_opposite_displacement"
            elif protection_broken:
                terminal_reason = "protection_broken"
            elif (
                proposed.state.interruption_run
                > self.protocol.consecutive_interruptions_max
            ):
                terminal_reason = "confirmed_progress_loss"

            if terminal_reason is not None:
                terminal = self._terminal(
                    DisplacementLifecycle.EXHAUSTED,
                    terminal_reason,
                    candle.end,
                    self._candle_id(candle),
                )
                transitions.append(self._transition(terminal))
                probe = self._reverse_probe
                self._open = None
                self._reverse_probe = None
                if terminal_reason == "qualified_opposite_displacement":
                    assert probe is not None
                    self._promote_probe(
                        probe,
                        observed_at=candle.end,
                        activate=True,
                        transitions=transitions,
                    )
                elif terminal_reason == "protection_broken":
                    if probe is not None:
                        self._promote_probe(
                            probe,
                            observed_at=candle.end,
                            activate=False,
                            transitions=transitions,
                        )
                    elif (
                        candidate_direction
                        == -int(prior.direction.sign)
                    ):
                        self._start_if_candidate(
                            candle,
                            atr0=atr0,
                            v0=v0,
                            prior_close=prior_close,
                            transitions=transitions,
                        )
            else:
                self._open = proposed
                state = proposed.state
                if self._activates(state):
                    state = replace(
                        state,
                        lifecycle=DisplacementLifecycle.ACTIVE,
                        active_at=candle.end,
                        state_started_at=candle.end,
                    )
                    assert self._open is not None
                    self._open = replace(self._open, state=state)
                    transitions.append(self._transition(state))

        self._trs.append(tr)
        self._volumes.append(float(candle.volume))
        self._prior_close = float(candle.close)
        self._last_clock = candle.end
        self._last_input_kind = "candle"
        return DisplacementUpdate(self.snapshot(), tuple(transitions))


__all__ = [
    "CausalDisplacementTracker",
    "DisplacementLifecycle",
    "DisplacementProtocol",
    "DisplacementState",
    "DisplacementTransition",
    "DisplacementUpdate",
    "EPISODE_PROTOCOL_VERSION",
]
