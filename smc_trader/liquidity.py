"""Incremental swing clusters, equal-liquidity pools and draw inventory.

The detector is descriptive only.  Cluster bounds are frozen when the first
confirmed swing is admitted.  Later bars may change lifecycle state, but may
not rewrite the original zone.
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import pandas as pd

from .model import (
    Candle,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityPoolLifecycle,
    LiquidityPoolState,
    SUPPORT_RESISTANCE_RETIREMENT_REASON,
    SupportResistanceLifecycle,
    SupportResistanceState,
    SwingLifecycle,
    SwingPoint,
    SwingSide,
    Timeframe,
    clamp,
)


class LiquidityProtocolError(ValueError):
    """Raised when the incremental liquidity contract is violated."""


@dataclass(frozen=True)
class LiquidityConfig:
    """Single development definition shared by S/R and equal pools."""

    protocol_version: str = "3.1.0-group12.4"
    protocol_hash: str = ""
    tick_size: float = 0.25
    atr_period: int = 14
    cluster_tolerance_atr: float = 0.10
    cluster_tolerance_ticks: int = 1
    retained_zones: int = 128
    retained_pools: int = 128
    retained_touches: int = 128

    def __post_init__(self) -> None:
        try:
            tick_size = float(self.tick_size)
            tolerance_atr = float(self.cluster_tolerance_atr)
        except (TypeError, ValueError) as error:
            raise LiquidityProtocolError(
                "invalid liquidity protocol"
            ) from error
        object.__setattr__(self, "tick_size", tick_size)
        object.__setattr__(
            self,
            "cluster_tolerance_atr",
            tolerance_atr,
        )
        protocol_hash = self.protocol_hash or hashlib.sha256(
            self.protocol_version.encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "protocol_hash", protocol_hash)
        if (
            not self.protocol_version
            or len(protocol_hash) != 64
            or any(value not in "0123456789abcdef" for value in protocol_hash)
            or not math.isfinite(float(self.tick_size))
            or self.tick_size <= 0
            or type(self.atr_period) is not int
            or self.atr_period < 1
            or not math.isfinite(float(self.cluster_tolerance_atr))
            or not 0 < self.cluster_tolerance_atr <= 1
            or type(self.cluster_tolerance_ticks) is not int
            or self.cluster_tolerance_ticks < 1
            or type(self.retained_zones) is not int
            or self.retained_zones < 8
            or type(self.retained_pools) is not int
            or self.retained_pools < 8
            or type(self.retained_touches) is not int
            or self.retained_touches < 8
        ):
            raise LiquidityProtocolError("invalid liquidity protocol")

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        tick_size: float,
        atr_period: int,
    ) -> "LiquidityConfig":
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = Path(__file__).resolve().parents[1] / source
        raw = source.read_bytes()
        payload = json.loads(raw)
        parameters = payload.get("liquidity_parameters")
        if not isinstance(parameters, dict):
            raise LiquidityProtocolError(
                "liquidity protocol lacks executable parameters"
            )
        return cls(
            protocol_version=str(payload.get("protocol_version", "")),
            protocol_hash=hashlib.sha256(raw).hexdigest(),
            tick_size=float(tick_size),
            atr_period=int(atr_period),
            cluster_tolerance_atr=float(
                parameters.get("cluster_tolerance_atr", -1.0)
            ),
            cluster_tolerance_ticks=int(
                parameters.get("cluster_tolerance_ticks", -1)
            ),
            retained_zones=int(parameters.get("retained_zones", -1)),
            retained_pools=int(parameters.get("retained_pools", -1)),
            retained_touches=int(
                parameters.get("retained_touches", -1)
            ),
        )


@dataclass
class _ZoneRecord:
    state: SupportResistanceState
    formed_index: int
    total_touches: int
    reaction_total_atr: float


@dataclass
class _PoolRecord:
    state: LiquidityPoolState
    confirmed_index: int
    zone_id: str
    generation_number: int
    earliest_suppressed_touch_at: pd.Timestamp | None = None


@dataclass
class _PoolGeneration:
    number: int
    materialized: bool
    terminal: bool
    member_swing_ids: tuple[str, ...]
    touch_times: tuple[pd.Timestamp, ...]
    reactions_atr: tuple[float, ...]
    formed_at: pd.Timestamp
    total_touches: int
    reaction_total_atr: float
    confirmed_index: int | None


def _identity(*parts: object) -> str:
    raw = "|".join(str(value) for value in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class CausalLiquidityTracker:
    """One-timeframe bounded state updated only by completed candles."""

    def __init__(
        self,
        timeframe: Timeframe,
        config: LiquidityConfig | None = None,
    ) -> None:
        self.timeframe = timeframe
        self.config = config or LiquidityConfig()
        self.reset()

    def reset(self) -> None:
        self._zones: dict[str, _ZoneRecord] = {}
        self._pools: dict[str, _PoolRecord] = {}
        self._pool_generations: dict[str, _PoolGeneration] = {}
        self._known_swing_ids: set[str] = set()
        self._known_swing_order: deque[str] = deque(
            maxlen=max(1024, self.config.retained_zones * 8)
        )
        self._latest_swings: dict[str, SwingPoint] = {}
        self._true_ranges: deque[float] = deque(
            maxlen=self.config.atr_period
        )
        self._prior_close: float | None = None
        self._last_end: pd.Timestamp | None = None
        self._contract: tuple[str, int] | None = None
        self._bar_index = -1

    @property
    def last_end(self) -> pd.Timestamp | None:
        return self._last_end

    def _atr(self) -> float:
        positive = [value for value in self._true_ranges if value > 0]
        return (
            float(sum(positive) / len(positive))
            if positive
            else self.config.tick_size
        )

    def _update_atr(self, candle: Candle) -> None:
        true_range = float(candle.high - candle.low)
        if self._prior_close is not None:
            true_range = max(
                true_range,
                abs(candle.high - self._prior_close),
                abs(candle.low - self._prior_close),
            )
        self._true_ranges.append(max(0.0, true_range))
        self._prior_close = float(candle.close)

    @staticmethod
    def _zone_side(swing: SwingPoint) -> str:
        return (
            "resistance"
            if swing.side is SwingSide.HIGH
            else "support"
        )

    @staticmethod
    def _pool_side(swing: SwingPoint) -> str:
        return "above" if swing.side is SwingSide.HIGH else "below"

    @staticmethod
    def _strength(
        touch_count: int,
        reactions: Sequence[float],
    ) -> float:
        touch_component = min(1.0, max(0, touch_count) / 3.0)
        reaction_component = min(
            1.0,
            (
                sum(float(value) for value in reactions) / len(reactions)
                if reactions
                else 0.0
            ),
        )
        return clamp(0.5 * touch_component + 0.5 * reaction_component)

    @staticmethod
    def _strength_from_summary(
        touch_count: int,
        reaction_total_atr: float,
    ) -> float:
        touch_component = min(1.0, max(0, touch_count) / 3.0)
        reaction_component = min(
            1.0,
            max(0.0, float(reaction_total_atr))
            / max(1, int(touch_count)),
        )
        return clamp(0.5 * touch_component + 0.5 * reaction_component)

    def _bounded_touch_history(
        self,
        members: Sequence[str],
        touch_times: Sequence[pd.Timestamp],
        reactions: Sequence[float],
    ) -> tuple[tuple[str, ...], tuple[pd.Timestamp, ...], tuple[float, ...]]:
        """Retain formation evidence plus the newest bounded touch history."""

        values = tuple(zip(members, touch_times, reactions))
        limit = self.config.retained_touches
        if len(values) <= limit:
            retained = values
        elif limit == 2:
            retained = (values[0], values[-1])
        else:
            retained = (*values[:2], *values[-(limit - 2) :])
        return (
            tuple(item[0] for item in retained),
            tuple(item[1] for item in retained),
            tuple(float(item[2]) for item in retained),
        )

    def _reaction_atr(
        self,
        swing: SwingPoint,
        candle: Candle,
    ) -> float:
        atr = max(self._atr(), self.config.tick_size)
        favorable = (
            swing.price - candle.close
            if swing.side is SwingSide.HIGH
            else candle.close - swing.price
        )
        return max(0.0, float(favorable) / atr)

    def _match_zone(self, swing: SwingPoint) -> _ZoneRecord | None:
        side = self._zone_side(swing)
        matches = [
            record
            for record in self._zones.values()
            if (
                record.state.lifecycle
                in {
                    SupportResistanceLifecycle.ACTIVE,
                    SupportResistanceLifecycle.TESTED,
                }
                and record.state.side == side
                and record.state.lower_bound
                <= swing.price
                <= record.state.upper_bound
            )
        ]
        if not matches:
            return None
        return min(
            matches,
            key=lambda item: (
                abs(item.state.anchor_price - swing.price),
                item.state.confirmed_at,
                item.state.zone_id,
            ),
        )

    @staticmethod
    def _zone_terminal_at(
        record: _ZoneRecord,
    ) -> pd.Timestamp | None:
        state = record.state
        if (
            state.lifecycle
            is SupportResistanceLifecycle.REACCEPTED
        ):
            return state.reaccepted_at
        if state.lifecycle is SupportResistanceLifecycle.RETIRED:
            return state.retired_at
        return None

    def _prune_zones(self) -> None:
        while len(self._zones) >= self.config.retained_zones:
            pinned_zone_ids = {
                record.zone_id
                for record in self._pools.values()
                if record.state.lifecycle
                in {
                    LiquidityPoolLifecycle.FORMED,
                    LiquidityPoolLifecycle.SWEPT,
                }
            }
            candidates = [
                record
                for record in self._zones.values()
                if (
                    record.state.zone_id not in pinned_zone_ids
                    and record.state.lifecycle
                    in {
                        SupportResistanceLifecycle.REACCEPTED,
                        SupportResistanceLifecycle.RETIRED,
                    }
                    and self._zone_terminal_at(record)
                    != self._last_end
                )
            ]
            if not candidates:
                current_terminal_count = sum(
                    record.state.zone_id not in pinned_zone_ids
                    and self._zone_terminal_at(record)
                    == self._last_end
                    for record in self._zones.values()
                )
                current_overflow = max(
                    0,
                    len(self._zones) - self.config.retained_zones,
                )
                if current_overflow < current_terminal_count:
                    # Keep same-bar terminal states visible until the next
                    # real completed bar compacts the exposure buffer.
                    return
                raise LiquidityProtocolError(
                    "zone retention is exhausted by nonterminal state"
                )
            victim = min(
                candidates,
                key=lambda item: (
                    self._zone_terminal_at(item),
                    item.state.zone_id,
                ),
            )
            zone_id = victim.state.zone_id
            self._zones.pop(zone_id, None)
            self._pool_generations.pop(zone_id, None)

    def _compact_observed_zone_overflow(self) -> None:
        overflow = len(self._zones) - self.config.retained_zones
        if overflow <= 0:
            return
        pinned_zone_ids = {
            record.zone_id
            for record in self._pools.values()
            if record.state.lifecycle
            in {
                LiquidityPoolLifecycle.FORMED,
                LiquidityPoolLifecycle.SWEPT,
            }
        }
        candidates = sorted(
            (
                record
                for record in self._zones.values()
                if (
                    record.state.zone_id not in pinned_zone_ids
                    and self._zone_terminal_at(record) is not None
                    and self._zone_terminal_at(record)
                    != self._last_end
                )
            ),
            key=lambda item: (
                self._zone_terminal_at(item),
                item.state.zone_id,
            ),
        )
        if len(candidates) < overflow:
            raise LiquidityProtocolError(
                "zone exposure overflow lacks observed terminal states"
            )
        for victim in candidates[:overflow]:
            zone_id = victim.state.zone_id
            self._zones.pop(zone_id, None)
            self._pool_generations.pop(zone_id, None)

    def _retire_unreferenced_zones(self, candle: Candle) -> None:
        """Archive zones whose retained structural evidence is no longer live."""

        authoritative_swing_ids = set(self._latest_swings)
        prospective_touch_zone_ids = {
            matched.state.zone_id
            for swing in self._latest_swings.values()
            if swing.swing_id not in self._known_swing_ids
            for matched in (self._match_zone(swing),)
            if matched is not None
        }
        pinned_zone_ids = {
            record.zone_id
            for record in self._pools.values()
            if record.state.lifecycle
            in {
                LiquidityPoolLifecycle.FORMED,
                LiquidityPoolLifecycle.SWEPT,
            }
        }
        for record in self._zones.values():
            state = record.state
            if (
                state.lifecycle
                in {
                    SupportResistanceLifecycle.REACCEPTED,
                    SupportResistanceLifecycle.RETIRED,
                }
                or state.zone_id in pinned_zone_ids
                or state.zone_id in prospective_touch_zone_ids
                or not authoritative_swing_ids.isdisjoint(
                    state.member_swing_ids
                )
            ):
                continue
            latest_evidence_at = max(
                value
                for value in (
                    state.confirmed_at,
                    state.tested_at,
                    state.broken_at,
                )
                if value is not None
            )
            if candle.end <= latest_evidence_at:
                continue
            record.state = replace(
                state,
                lifecycle=SupportResistanceLifecycle.RETIRED,
                retired_at=candle.end,
                transition_reason=(
                    SUPPORT_RESISTANCE_RETIREMENT_REASON
                ),
            )
            self._pool_generations.pop(state.zone_id, None)

    def _prune_pools(self) -> None:
        while len(self._pools) >= self.config.retained_pools:
            candidates = [
                record
                for record in self._pools.values()
                if (
                    record.state.lifecycle
                    in {
                        LiquidityPoolLifecycle.ACCEPTED,
                        LiquidityPoolLifecycle.REJECTED,
                    }
                    and record.state.resolved_at != self._last_end
                )
            ]
            if not candidates:
                current_terminal_count = sum(
                    record.state.lifecycle
                    in {
                        LiquidityPoolLifecycle.ACCEPTED,
                        LiquidityPoolLifecycle.REJECTED,
                    }
                    and record.state.resolved_at == self._last_end
                    for record in self._pools.values()
                )
                current_overflow = max(
                    0,
                    len(self._pools) - self.config.retained_pools,
                )
                if current_overflow < current_terminal_count:
                    return
                raise LiquidityProtocolError(
                    "pool retention is exhausted by unresolved state"
                )
            victim = min(
                candidates,
                key=lambda item: (
                    item.state.resolved_at,
                    item.state.pool_id,
                ),
            )
            self._pools.pop(victim.state.pool_id, None)

    def _compact_observed_pool_overflow(self) -> None:
        overflow = len(self._pools) - self.config.retained_pools
        if overflow <= 0:
            return
        candidates = sorted(
            (
                record
                for record in self._pools.values()
                if (
                    record.state.lifecycle
                    in {
                        LiquidityPoolLifecycle.ACCEPTED,
                        LiquidityPoolLifecycle.REJECTED,
                    }
                    and record.state.resolved_at != self._last_end
                )
            ),
            key=lambda item: (
                item.state.resolved_at,
                item.state.pool_id,
            ),
        )
        if len(candidates) < overflow:
            raise LiquidityProtocolError(
                "pool exposure overflow lacks observed terminal states"
            )
        for victim in candidates[:overflow]:
            self._pools.pop(victim.state.pool_id, None)

    def _mark_generation_terminal(self, record: _PoolRecord) -> None:
        generation = self._pool_generations.get(record.zone_id)
        if (
            generation is None
            or generation.number != record.generation_number
        ):
            return
        self._pool_generations[record.zone_id] = replace(
            generation,
            terminal=True,
        )

    def _new_zone(
        self,
        swing: SwingPoint,
        candle: Candle,
    ) -> _ZoneRecord:
        tolerance = max(
            self.config.cluster_tolerance_ticks * self.config.tick_size,
            self.config.cluster_tolerance_atr * self._atr(),
        )
        lower = max(self.config.tick_size, swing.price - tolerance)
        upper = swing.price + tolerance
        zone_id = _identity(
            self.config.protocol_hash,
            self.timeframe.value,
            self._zone_side(swing),
            swing.swing_id,
            f"{lower:.10f}",
            f"{upper:.10f}",
        )
        reaction = self._reaction_atr(swing, candle)
        state = SupportResistanceState(
            zone_id=zone_id,
            timeframe=self.timeframe,
            side=self._zone_side(swing),
            lower_bound=lower,
            upper_bound=upper,
            anchor_price=swing.price,
            formed_at=swing.pivot_end,
            confirmed_at=swing.confirmed_at,
            lifecycle=SupportResistanceLifecycle.ACTIVE,
            member_swing_ids=(swing.swing_id,),
            touch_times=(swing.confirmed_at,),
            reaction_magnitudes_atr=(reaction,),
            age_bars=0,
            strength=self._strength(1, (reaction,)),
            total_touch_count=1,
        )
        return _ZoneRecord(
            state=state,
            formed_index=self._bar_index,
            total_touches=1,
            reaction_total_atr=reaction,
        )

    def _add_touch(
        self,
        record: _ZoneRecord,
        swing: SwingPoint,
        candle: Candle,
    ) -> None:
        state = record.state
        if swing.swing_id in state.member_swing_ids:
            return
        reaction = self._reaction_atr(swing, candle)
        members, touch_times, reactions = self._bounded_touch_history(
            (*state.member_swing_ids, swing.swing_id),
            (*state.touch_times, swing.confirmed_at),
            (*state.reaction_magnitudes_atr, reaction),
        )
        total_touches = record.total_touches + 1
        reaction_total = record.reaction_total_atr + reaction
        lifecycle = state.lifecycle
        tested_at = state.tested_at
        if lifecycle is SupportResistanceLifecycle.ACTIVE:
            lifecycle = SupportResistanceLifecycle.TESTED
            tested_at = swing.confirmed_at
        record.state = replace(
            state,
            lifecycle=lifecycle,
            member_swing_ids=members,
            touch_times=touch_times,
            reaction_magnitudes_atr=reactions,
            tested_at=tested_at,
            strength=self._strength_from_summary(
                total_touches,
                reaction_total,
            ),
            total_touch_count=total_touches,
        )
        record.total_touches = total_touches
        record.reaction_total_atr = reaction_total
        self._sync_pool(record, swing)

    def _pool_id(
        self,
        zone_id: str,
        generation: _PoolGeneration,
    ) -> str:
        parts: tuple[object, ...] = (
            self.config.protocol_hash,
            "equal_pool",
            zone_id,
        )
        if generation.number:
            parts = (
                *parts,
                generation.number,
                generation.member_swing_ids[0],
            )
        return _identity(*parts)

    def _sync_pool(
        self,
        zone: _ZoneRecord,
        swing: SwingPoint,
    ) -> None:
        state = zone.state
        generation = self._pool_generations.get(state.zone_id)
        if generation is None:
            generation = _PoolGeneration(
                number=0,
                materialized=False,
                terminal=False,
                member_swing_ids=state.member_swing_ids,
                touch_times=state.touch_times,
                reactions_atr=state.reaction_magnitudes_atr,
                formed_at=state.formed_at,
                total_touches=zone.total_touches,
                reaction_total_atr=zone.reaction_total_atr,
                confirmed_index=None,
            )
            self._pool_generations[state.zone_id] = generation
        pool_id = self._pool_id(state.zone_id, generation)
        existing = self._pools.get(pool_id)
        if generation.terminal:
            generation = _PoolGeneration(
                number=generation.number + 1,
                materialized=False,
                terminal=False,
                member_swing_ids=(swing.swing_id,),
                touch_times=(swing.confirmed_at,),
                reactions_atr=(state.reaction_magnitudes_atr[-1],),
                formed_at=swing.pivot_end,
                total_touches=1,
                reaction_total_atr=state.reaction_magnitudes_atr[-1],
                confirmed_index=None,
            )
            self._pool_generations[state.zone_id] = generation
            return
        if (
            existing is not None
            and existing.state.lifecycle is LiquidityPoolLifecycle.SWEPT
        ):
            if (
                existing.earliest_suppressed_touch_at is None
                or swing.confirmed_at
                < existing.earliest_suppressed_touch_at
            ):
                existing.earliest_suppressed_touch_at = (
                    swing.confirmed_at
                )
            return
        if swing.swing_id not in generation.member_swing_ids:
            members, touch_times, reactions = self._bounded_touch_history(
                (*generation.member_swing_ids, swing.swing_id),
                (*generation.touch_times, swing.confirmed_at),
                (
                    *generation.reactions_atr,
                    state.reaction_magnitudes_atr[-1],
                ),
            )
            generation = _PoolGeneration(
                number=generation.number,
                materialized=generation.materialized,
                terminal=generation.terminal,
                member_swing_ids=members,
                touch_times=touch_times,
                reactions_atr=reactions,
                formed_at=generation.formed_at,
                total_touches=generation.total_touches + 1,
                reaction_total_atr=(
                    generation.reaction_total_atr
                    + state.reaction_magnitudes_atr[-1]
                ),
                confirmed_index=generation.confirmed_index,
            )
            self._pool_generations[state.zone_id] = generation
        if len(generation.member_swing_ids) < 2:
            return
        pool_id = self._pool_id(state.zone_id, generation)
        existing = self._pools.get(pool_id)
        generation_strength = self._strength_from_summary(
            generation.total_touches,
            generation.reaction_total_atr,
        )
        if existing is None:
            self._prune_pools()
            pool = LiquidityPoolState(
                pool_id=pool_id,
                timeframe=self.timeframe,
                side=(
                    "above"
                    if state.side == "resistance"
                    else "below"
                ),
                lower_bound=state.lower_bound,
                upper_bound=state.upper_bound,
                midpoint=(
                    state.lower_bound + state.upper_bound
                )
                / 2.0,
                formed_at=generation.formed_at,
                confirmed_at=generation.touch_times[1],
                lifecycle=LiquidityPoolLifecycle.FORMED,
                member_swing_ids=generation.member_swing_ids,
                touch_times=generation.touch_times,
                age_bars=0,
                strength=generation_strength,
                total_touch_count=generation.total_touches,
            )
            self._pools[pool_id] = _PoolRecord(
                state=pool,
                confirmed_index=(
                    self._bar_index
                    if generation.confirmed_index is None
                    else generation.confirmed_index
                ),
                zone_id=state.zone_id,
                generation_number=generation.number,
            )
            self._pool_generations[state.zone_id] = replace(
                generation,
                materialized=True,
                confirmed_index=(
                    self._bar_index
                    if generation.confirmed_index is None
                    else generation.confirmed_index
                ),
            )
            return
        if existing.state.lifecycle is LiquidityPoolLifecycle.FORMED:
            existing.state = replace(
                existing.state,
                member_swing_ids=generation.member_swing_ids,
                touch_times=generation.touch_times,
                strength=generation_strength,
                total_touch_count=generation.total_touches,
            )

    def project_pool_sweep(
        self,
        pool_id: str,
        *,
        observed_at: pd.Timestamp,
        sweep_extreme: float,
        close_outside_on_sweep: bool,
    ) -> LiquidityPoolState:
        """Advance one formed pool from the completed 1m path."""

        record = self._pools.get(pool_id)
        if record is None:
            raise LiquidityProtocolError("projected pool is not retained")
        state = record.state
        if state.lifecycle is LiquidityPoolLifecycle.SWEPT:
            if state.swept_at != observed_at:
                raise LiquidityProtocolError(
                    "pool already swept at a different clock"
                )
            return state
        if state.lifecycle is not LiquidityPoolLifecycle.FORMED:
            raise LiquidityProtocolError(
                "terminal pool cannot be swept again"
            )
        if observed_at <= state.confirmed_at:
            raise LiquidityProtocolError(
                "projected pool sweep must follow confirmation"
            )
        record.state = replace(
            state,
            lifecycle=LiquidityPoolLifecycle.SWEPT,
            swept_at=observed_at,
            sweep_extreme=float(sweep_extreme),
            close_outside_on_sweep=bool(close_outside_on_sweep),
        )
        return record.state

    def project_pool_resolution(
        self,
        pool_id: str,
        *,
        observed_at: pd.Timestamp,
        accepted_outside: bool,
    ) -> LiquidityPoolState:
        """Resolve a projected sweep from the next real completed 1m bar."""

        record = self._pools.get(pool_id)
        if record is None:
            raise LiquidityProtocolError("projected pool is not retained")
        state = record.state
        lifecycle = (
            LiquidityPoolLifecycle.ACCEPTED
            if accepted_outside
            else LiquidityPoolLifecycle.REJECTED
        )
        if state.lifecycle is lifecycle:
            if state.resolved_at != observed_at:
                raise LiquidityProtocolError(
                    "pool already resolved at a different clock"
                )
            self._mark_generation_terminal(record)
            return state
        if (
            state.lifecycle is not LiquidityPoolLifecycle.SWEPT
            or state.swept_at is None
            or observed_at <= state.swept_at
        ):
            raise LiquidityProtocolError(
                "pool resolution requires an earlier projected sweep"
            )
        record.state = replace(
            state,
            lifecycle=lifecycle,
            resolved_at=observed_at,
            resolution_reason=(
                "close_held_outside"
                if accepted_outside
                else "close_returned_inside"
            ),
        )
        self._mark_generation_terminal(record)
        return record.state

    def bootstrap_pool_projection(
        self,
        pool_id: str,
        *,
        swept_at: pd.Timestamp,
        sweep_extreme: float,
        close_outside_on_sweep: bool,
        resolved_at: pd.Timestamp | None = None,
        accepted_outside: bool | None = None,
    ) -> LiquidityPoolState:
        """Rebuild a retained pool from a completed causal 1m prefix.

        This replaces only coarse native lifecycle clocks. Frozen formation
        bounds, identity, members and confirmation clocks are untouched.
        """

        record = self._pools.get(pool_id)
        if record is None:
            raise LiquidityProtocolError(
                "bootstrap pool is not retained"
            )
        native_swept_at = record.state.swept_at
        if (
            resolved_at is not None
            and native_swept_at is not None
            and record.earliest_suppressed_touch_at is not None
            and resolved_at
            <= record.earliest_suppressed_touch_at
            <= native_swept_at
        ):
            raise LiquidityProtocolError(
                "cold native replay swallowed a post-resolution "
                "liquidity-generation touch"
            )
        record.state = replace(
            record.state,
            lifecycle=LiquidityPoolLifecycle.FORMED,
            swept_at=None,
            sweep_extreme=None,
            close_outside_on_sweep=None,
            resolved_at=None,
            resolution_reason=None,
        )
        generation = self._pool_generations.get(record.zone_id)
        if (
            generation is not None
            and generation.number == record.generation_number
        ):
            self._pool_generations[record.zone_id] = replace(
                generation,
                terminal=False,
            )
        self.project_pool_sweep(
            pool_id,
            observed_at=swept_at,
            sweep_extreme=sweep_extreme,
            close_outside_on_sweep=close_outside_on_sweep,
        )
        if resolved_at is None:
            if accepted_outside is not None:
                raise LiquidityProtocolError(
                    "unresolved bootstrap pool cannot have resolution"
                )
            return record.state
        if accepted_outside is None:
            raise LiquidityProtocolError(
                "resolved bootstrap pool needs acceptance state"
            )
        return self.project_pool_resolution(
            pool_id,
            observed_at=resolved_at,
            accepted_outside=accepted_outside,
        )

    def _ingest_swing(
        self,
        swing: SwingPoint,
        candle: Candle,
    ) -> None:
        if (
            swing.lifecycle
            not in {SwingLifecycle.CONFIRMED, SwingLifecycle.BROKEN}
            or swing.confirmed_at is None
            or swing.swing_id in self._known_swing_ids
        ):
            return
        if swing.confirmed_at != candle.end:
            raise LiquidityProtocolError(
                "new swing must be admitted at its confirmation clock"
            )
        if (
            len(self._known_swing_order)
            == self._known_swing_order.maxlen
            and self._known_swing_order
        ):
            self._known_swing_ids.discard(
                self._known_swing_order[0]
            )
        self._known_swing_order.append(swing.swing_id)
        self._known_swing_ids.add(swing.swing_id)
        zone = self._match_zone(swing)
        if zone is None:
            self._prune_zones()
            zone = self._new_zone(swing, candle)
            self._zones[zone.state.zone_id] = zone
        else:
            self._add_touch(zone, swing, candle)

    def _update_zones(self, candle: Candle) -> None:
        for record in self._zones.values():
            state = record.state
            if state.lifecycle in {
                SupportResistanceLifecycle.ACTIVE,
                SupportResistanceLifecycle.TESTED,
            }:
                broken = (
                    candle.close < state.lower_bound
                    if state.side == "support"
                    else candle.close > state.upper_bound
                )
                if broken and candle.end > state.confirmed_at:
                    record.state = replace(
                        state,
                        lifecycle=SupportResistanceLifecycle.BROKEN,
                        broken_at=candle.end,
                        transition_reason="close_beyond_frozen_zone",
                    )
            elif (
                state.lifecycle is SupportResistanceLifecycle.BROKEN
                and state.broken_at is not None
                and candle.end > state.broken_at
            ):
                reentered = (
                    candle.close >= state.lower_bound
                    if state.side == "support"
                    else candle.close <= state.upper_bound
                )
                if reentered:
                    record.state = replace(
                        state,
                        lifecycle=SupportResistanceLifecycle.REACCEPTED,
                        reaccepted_at=candle.end,
                        transition_reason="close_reentered_frozen_zone",
                    )

    def _update_pools(self, candle: Candle) -> None:
        for record in self._pools.values():
            state = record.state
            if state.lifecycle is LiquidityPoolLifecycle.FORMED:
                swept = (
                    candle.high > state.upper_bound
                    if state.side == "above"
                    else candle.low < state.lower_bound
                )
                if not swept or candle.end <= state.confirmed_at:
                    continue
                outside_close = (
                    candle.close > state.upper_bound
                    if state.side == "above"
                    else candle.close < state.lower_bound
                )
                record.state = replace(
                    state,
                    lifecycle=LiquidityPoolLifecycle.SWEPT,
                    swept_at=candle.end,
                    sweep_extreme=(
                        candle.high
                        if state.side == "above"
                        else candle.low
                    ),
                    close_outside_on_sweep=outside_close,
                )
            elif (
                state.lifecycle is LiquidityPoolLifecycle.SWEPT
                and state.swept_at is not None
                and candle.end > state.swept_at
            ):
                outside_close = (
                    candle.close > state.upper_bound
                    if state.side == "above"
                    else candle.close < state.lower_bound
                )
                record.state = replace(
                    state,
                    lifecycle=(
                        LiquidityPoolLifecycle.ACCEPTED
                        if outside_close
                        else LiquidityPoolLifecycle.REJECTED
                    ),
                    resolved_at=candle.end,
                    resolution_reason=(
                        "close_held_outside"
                        if outside_close
                        else "close_returned_inside"
                    ),
                )
                self._mark_generation_terminal(record)

    def on_candle(
        self,
        candle: Candle,
        swings: Sequence[SwingPoint],
    ) -> None:
        if candle.timeframe is not self.timeframe or not candle.complete:
            raise LiquidityProtocolError(
                "liquidity tracker requires a complete same-timeframe candle"
            )
        contract = (candle.symbol, int(candle.instrument_id))
        if self._contract is not None and contract != self._contract:
            raise LiquidityProtocolError(
                "contract changed without a liquidity reset"
            )
        if self._last_end is not None and candle.end <= self._last_end:
            raise LiquidityProtocolError(
                "duplicate or out-of-order liquidity candle"
            )
        if any(item.timeframe is not self.timeframe for item in swings):
            raise LiquidityProtocolError(
                "liquidity tracker received a foreign-timeframe swing"
            )
        if any(
            item.lifecycle
            in {SwingLifecycle.CONFIRMED, SwingLifecycle.BROKEN}
            and item.swing_id not in self._known_swing_ids
            and item.confirmed_at != candle.end
            for item in swings
        ):
            raise LiquidityProtocolError(
                "new swing must be admitted at its confirmation clock"
            )
        new_swing_ids = {
            item.swing_id
            for item in swings
            if (
                item.lifecycle
                in {SwingLifecycle.CONFIRMED, SwingLifecycle.BROKEN}
                and item.swing_id not in self._known_swing_ids
            )
        }
        rollback_state = (
            copy.deepcopy(self.__dict__)
            if (
                candle.real_completed
                and new_swing_ids
                and (
                    len(self._zones) + len(new_swing_ids)
                    > self.config.retained_zones
                    or len(self._pools) + len(new_swing_ids)
                    > self.config.retained_pools
                )
            )
            else None
        )
        try:
            self._contract = contract
            self._last_end = candle.end
            if not candle.real_completed:
                return
            self._bar_index += 1
            self._compact_observed_zone_overflow()
            self._compact_observed_pool_overflow()
            self._update_zones(candle)
            self._update_pools(candle)
            self._update_atr(candle)
            self._latest_swings = {
                item.swing_id: item
                for item in swings
                if item.lifecycle
                in {SwingLifecycle.CONFIRMED, SwingLifecycle.BROKEN}
            }
            self._retire_unreferenced_zones(candle)
            for swing in sorted(
                self._latest_swings.values(),
                key=lambda item: (
                    item.confirmed_at,
                    item.pivot_start,
                    item.side.value,
                ),
            ):
                self._ingest_swing(swing, candle)
        except Exception:
            if rollback_state is not None:
                self.__dict__.clear()
                self.__dict__.update(rollback_state)
            raise

    def snapshot(
        self,
    ) -> tuple[
        tuple[SupportResistanceState, ...],
        tuple[LiquidityPoolState, ...],
        tuple[LiquidityInventoryItem, ...],
    ]:
        zones = tuple(
            replace(
                record.state,
                age_bars=max(0, self._bar_index - record.formed_index),
            )
            for record in sorted(
                self._zones.values(),
                key=lambda item: (
                    item.state.confirmed_at,
                    item.state.zone_id,
                ),
            )
        )
        pools = tuple(
            replace(
                record.state,
                age_bars=max(
                    0,
                    self._bar_index - record.confirmed_index,
                ),
            )
            for record in sorted(
                self._pools.values(),
                key=lambda item: (
                    item.state.confirmed_at,
                    item.state.pool_id,
                ),
            )
        )
        inventory: list[LiquidityInventoryItem] = []
        for swing in self._latest_swings.values():
            lifecycle = (
                LiquidityInventoryLifecycle.CONSUMED
                if swing.lifecycle is SwingLifecycle.BROKEN
                else LiquidityInventoryLifecycle.VISIBLE
            )
            inventory.append(
                LiquidityInventoryItem(
                    item_id=f"swing:{swing.swing_id}",
                    timeframe=self.timeframe,
                    side=(
                        "above"
                        if swing.side is SwingSide.HIGH
                        else "below"
                    ),
                    kind="swing",
                    price=swing.price,
                    lower_bound=swing.price,
                    upper_bound=swing.price,
                    formed_at=swing.pivot_end,
                    confirmed_at=swing.confirmed_at,
                    lifecycle=lifecycle,
                    source_ids=(swing.swing_id,),
                    age_bars=swing.age_bars,
                    strength=clamp(swing.magnitude_atr),
                    consumed_at=(
                        swing.broken_at
                        if lifecycle
                        is LiquidityInventoryLifecycle.CONSUMED
                        else None
                    ),
                    lifecycle_reason=(
                        "close_beyond_swing"
                        if lifecycle
                        is LiquidityInventoryLifecycle.CONSUMED
                        else None
                    ),
                )
            )
        for pool in pools:
            consumed = pool.swept_at is not None
            inventory.append(
                LiquidityInventoryItem(
                    item_id=f"pool:{pool.pool_id}",
                    timeframe=self.timeframe,
                    side=pool.side,
                    kind=(
                        "equal_highs"
                        if pool.side == "above"
                        else "equal_lows"
                    ),
                    price=(
                        pool.upper_bound
                        if pool.side == "above"
                        else pool.lower_bound
                    ),
                    lower_bound=pool.lower_bound,
                    upper_bound=pool.upper_bound,
                    formed_at=pool.formed_at,
                    confirmed_at=pool.confirmed_at,
                    lifecycle=(
                        LiquidityInventoryLifecycle.CONSUMED
                        if consumed
                        else LiquidityInventoryLifecycle.VISIBLE
                    ),
                    source_ids=pool.member_swing_ids,
                    age_bars=pool.age_bars,
                    strength=pool.strength,
                    consumed_at=pool.swept_at,
                    lifecycle_reason="pool_swept" if consumed else None,
                )
            )
        return (
            zones,
            pools,
            tuple(
                sorted(
                    inventory,
                    key=lambda item: (
                        item.confirmed_at,
                        item.timeframe.value,
                        item.item_id,
                    ),
                )
            ),
        )


__all__ = [
    "CausalLiquidityTracker",
    "LiquidityConfig",
    "LiquidityProtocolError",
]
