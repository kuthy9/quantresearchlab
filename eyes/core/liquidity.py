"""Incremental swing clusters, equal-liquidity pools and draw inventory.

The detector is descriptive only.  Cluster bounds are frozen when the first
confirmed swing is admitted.  Later bars may change lifecycle state, but may
not rewrite the original zone.
"""
from __future__ import annotations

from collections import deque
import dataclasses
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from contract.market import (
    Candle,
    Timeframe,
    clamp,
)
from contract.eye import (
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityPoolLifecycle,
    LiquidityPoolState,
    SUPPORT_RESISTANCE_RETIREMENT_REASON,
    StructureLifecycle,
    StructureSequenceState,
    SupportResistanceLifecycle,
    SupportResistanceState,
    SwingLifecycle,
    SwingPoint,
    SwingSide,
)


class LiquidityProtocolError(ValueError):
    """Raised when the incremental liquidity contract is violated."""


_REFERENCE_INVENTORY_KINDS = {
    "previous_session_high",
    "previous_session_low",
    "previous_day_high",
    "previous_day_low",
    "previous_week_high",
    "previous_week_low",
}


@dataclass(frozen=True)
class LiquidityConfig:
    """Single development definition shared by S/R and equal pools."""

    protocol_version: str = "3.2.0-group12.7"
    protocol_hash: str = ""
    tick_size: float = 0.25
    atr_period: int = 14
    cluster_tolerance_atr: float = 0.10
    cluster_tolerance_ticks: int = 1
    retained_zones: int = 128
    retained_pools: int = 128
    retained_touches: int = 128
    # A candidate retires past this many bars of its own scale, or beyond
    # this many of its scale's ATRs from the close; see the protocol's
    # ``candidate_retirement`` note.
    candidate_retirement_max_native_age_bars: int = 480
    candidate_retirement_max_distance_atr: float = 20.0
    # A terminal zone or pool stays in the tracker for this many completed
    # native bars, counting the bar that made it terminal, then is
    # compacted; ``retained_zones``/``retained_pools`` remain the capacity
    # that live state may not exhaust.
    terminal_state_retention_native_bars: int = 1

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
            or type(self.candidate_retirement_max_native_age_bars) is not int
            or self.candidate_retirement_max_native_age_bars < 1
            or not math.isfinite(float(self.candidate_retirement_max_distance_atr))
            or self.candidate_retirement_max_distance_atr <= 0.0
            or type(self.terminal_state_retention_native_bars) is not int
            or self.terminal_state_retention_native_bars < 1
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
            source = Path(__file__).resolve().parents[2] / source
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
            candidate_retirement_max_native_age_bars=int(
                parameters.get("candidate_retirement_max_native_age_bars", -1)
            ),
            candidate_retirement_max_distance_atr=float(
                parameters.get("candidate_retirement_max_distance_atr", -1.0)
            ),
            terminal_state_retention_native_bars=int(
                parameters.get("terminal_state_retention_native_bars", -1)
            ),
        )


@dataclass
class _ZoneRecord:
    state: SupportResistanceState
    formed_index: int
    total_touches: int
    reaction_total_atr: float
    contact_active: bool = False
    terminal_index: int | None = None


@dataclass
class _PoolRecord:
    state: LiquidityPoolState
    confirmed_index: int
    zone_id: str
    generation_number: int
    structural_rank: str
    is_protected_swing: bool
    visibility_strength: float
    earliest_suppressed_touch_at: pd.Timestamp | None = None
    terminal_index: int | None = None


@dataclass
class _PoolGeneration:
    number: int
    materialized: bool
    terminal: bool
    member_swing_ids: tuple[str, ...]
    member_prices: tuple[float, ...]
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
        self._strict_prior_atr_by_end: dict[pd.Timestamp, float] = {}
        self._strict_prior_atr_order: deque[pd.Timestamp] = deque(
            maxlen=max(128, self.config.atr_period * 8)
        )
        self._prior_close: float | None = None
        self._last_end: pd.Timestamp | None = None
        self._last_real_end: pd.Timestamp | None = None
        # Completed-period reference levels remain part of the current eye
        # state until their source period is replaced.  A zone may already
        # have reached a terminal descriptive lifecycle (for example,
        # REACCEPTED), but pruning it while the exact reference source is
        # still live would make the next bar look like a late first admission.
        self._live_reference_source_ids: set[str] = set()
        self._contract: tuple[str, int] | None = None
        self._bar_index = -1
        self._group4_source_snapshot_cache: dict[
            bool,
            tuple[tuple[object, ...], tuple],
        ] = {}

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

    def _remember_strict_prior_atr(self, candle: Candle) -> None:
        if (
            len(self._strict_prior_atr_order)
            == self._strict_prior_atr_order.maxlen
            and self._strict_prior_atr_order
        ):
            self._strict_prior_atr_by_end.pop(
                self._strict_prior_atr_order[0],
                None,
            )
        self._strict_prior_atr_order.append(candle.end)
        self._strict_prior_atr_by_end[candle.end] = self._atr()

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

    def _visibility_strength(
        self,
        *,
        external: bool,
    ) -> float:
        scale_visibility = {
            Timeframe.M1: 0.20,
            Timeframe.M5: 0.40,
            Timeframe.M15: 0.55,
            Timeframe.H1: 0.75,
            Timeframe.H4: 1.00,
        }[self.timeframe]
        return 1.0 if external else scale_visibility

    @staticmethod
    def _reaction_quality(
        touch_count: int,
        reaction_total_atr: float,
    ) -> float:
        return clamp(
            max(0.0, float(reaction_total_atr))
            / max(1, int(touch_count))
        )

    @staticmethod
    def _freshness(age_bars: int) -> float:
        return 1.0 / (1.0 + max(0, int(age_bars)))

    @staticmethod
    def _depletion_risk(touch_count: int) -> float:
        count = max(1, int(touch_count))
        return (count - 1.0) / count

    def _refresh_structural_metadata(
        self,
        structures: Sequence[StructureSequenceState],
    ) -> None:
        protected_ids = {
            item.protected_swing_id
            for item in structures
            if (
                item.lifecycle is StructureLifecycle.CONFIRMED
                and item.protected_swing_id is not None
            )
        }
        for record in self._zones.values():
            state = record.state
            if state.source_kind != "structural_swing":
                continue
            protected = not protected_ids.isdisjoint(
                state.member_swing_ids
            )
            external = (
                protected or state.structural_rank == "external"
            )
            structural_rank = "external" if external else "internal"
            visibility_strength = self._visibility_strength(
                external=external,
            )
            reaction_quality = self._reaction_quality(
                record.total_touches,
                record.reaction_total_atr,
            )
            depletion_risk = self._depletion_risk(record.total_touches)
            if (
                state.structural_rank != structural_rank
                or state.is_protected_swing != protected
                or state.visibility_strength != visibility_strength
                or state.reaction_quality != reaction_quality
                or state.depletion_risk != depletion_risk
            ):
                record.state = replace(
                    state,
                    structural_rank=structural_rank,
                    is_protected_swing=protected,
                    visibility_strength=visibility_strength,
                    reaction_quality=reaction_quality,
                    depletion_risk=depletion_risk,
                )
        for record in self._pools.values():
            source_zone = self._zones.get(record.zone_id)
            if source_zone is None:
                continue
            source_state = source_zone.state
            if record.structural_rank != source_state.structural_rank:
                record.structural_rank = source_state.structural_rank
            if (
                record.is_protected_swing
                != source_state.is_protected_swing
            ):
                record.is_protected_swing = (
                    source_state.is_protected_swing
                )
            if (
                record.visibility_strength
                != source_state.visibility_strength
            ):
                record.visibility_strength = (
                    source_state.visibility_strength
                )

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
                and record.state.source_kind == "structural_swing"
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
                    and self._live_reference_source_ids.isdisjoint(
                        record.state.source_ids
                    )
                    and record.state.lifecycle
                    in {
                        SupportResistanceLifecycle.REACCEPTED,
                        SupportResistanceLifecycle.RETIRED,
                    }
                    and self._terminal_exposure_complete(record)
                )
            ]
            if not candidates:
                current_terminal_count = sum(
                    record.state.zone_id not in pinned_zone_ids
                    and self._live_reference_source_ids.isdisjoint(
                        record.state.source_ids
                    )
                    and record.terminal_index is not None
                    for record in self._zones.values()
                )
                current_overflow = max(
                    0,
                    len(self._zones) - self.config.retained_zones,
                )
                if current_overflow < current_terminal_count:
                    # Keep terminal states inside their retention visible
                    # until a later real completed bar compacts them.
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

    def _terminal_exposure_complete(
        self,
        record: "_ZoneRecord | _PoolRecord",
    ) -> bool:
        """True once a terminal record has had its retention of native bars."""

        return (
            record.terminal_index is not None
            and self._bar_index - record.terminal_index
            >= self.config.terminal_state_retention_native_bars
        )

    def _compact_terminal_zones(self) -> None:
        """Drop terminal zones past their retention; fail closed on overflow.

        Every terminal zone was exposed on the bar that made it terminal, so
        after ``terminal_state_retention_native_bars`` it is history the event
        log owns.  A zone a live pool or a live reference source still cites
        is kept regardless.  The capacity rule stays: an overflow that only
        live or same-bar state could absorb is an error, never an eviction.
        """

        overflow = len(self._zones) - self.config.retained_zones
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
                    and self._live_reference_source_ids.isdisjoint(
                        record.state.source_ids
                    )
                    and self._terminal_exposure_complete(record)
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
        for victim in candidates:
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
                state.source_kind != "structural_swing"
                or
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
            record.terminal_index = self._bar_index
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
                    and self._terminal_exposure_complete(record)
                )
            ]
            if not candidates:
                current_terminal_count = sum(
                    record.terminal_index is not None
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

    def _compact_terminal_pools(self) -> None:
        """Drop resolved pools past retention; fail closed on overflow."""

        overflow = len(self._pools) - self.config.retained_pools
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
                    and self._terminal_exposure_complete(record)
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
        for victim in candidates:
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
            source_kind="structural_swing",
            structural_rank="internal",
            is_protected_swing=False,
            zone_role="both",
            visibility_strength=self._visibility_strength(
                external=False,
            ),
            reaction_quality=self._reaction_quality(1, reaction),
            freshness=1.0,
            depletion_risk=0.0,
        )
        return _ZoneRecord(
            state=state,
            formed_index=self._bar_index,
            total_touches=1,
            reaction_total_atr=reaction,
        )

    @staticmethod
    def _reference_source_kind(item: LiquidityInventoryItem) -> str:
        if item.kind not in _REFERENCE_INVENTORY_KINDS:
            raise LiquidityProtocolError(
                "generic S/R source is not a completed-period reference"
            )
        return item.kind.rsplit("_", 1)[0]

    def _new_reference_zone(
        self,
        item: LiquidityInventoryItem,
    ) -> _ZoneRecord:
        """Materialize a reaction band around an exact reference level."""

        strict_prior_atr = self._strict_prior_atr_by_end.get(
            item.confirmed_at
        )
        if strict_prior_atr is None:
            raise LiquidityProtocolError(
                "reference S/R lacks ATR frozen before its source "
                "confirmation candle"
            )
        tolerance = max(
            self.config.cluster_tolerance_ticks * self.config.tick_size,
            self.config.cluster_tolerance_atr * strict_prior_atr,
        )
        lower = max(self.config.tick_size, float(item.price) - tolerance)
        upper = float(item.price) + tolerance
        source_kind = self._reference_source_kind(item)
        zone_id = _identity(
            self.config.protocol_hash,
            "reference_zone",
            self.timeframe.value,
            item.item_id,
            f"{lower:.10f}",
            f"{upper:.10f}",
        )
        state = SupportResistanceState(
            zone_id=zone_id,
            timeframe=self.timeframe,
            side=("resistance" if item.side == "above" else "support"),
            lower_bound=lower,
            upper_bound=upper,
            anchor_price=float(item.price),
            formed_at=item.formed_at,
            confirmed_at=item.confirmed_at,
            lifecycle=SupportResistanceLifecycle.ACTIVE,
            member_swing_ids=(),
            touch_times=(item.confirmed_at,),
            reaction_magnitudes_atr=(0.0,),
            age_bars=0,
            strength=self._strength(1, (0.0,)),
            total_touch_count=1,
            source_kind=source_kind,
            structural_rank="external",
            is_protected_swing=False,
            zone_role="both",
            visibility_strength=item.visibility_strength,
            reaction_quality=0.0,
            freshness=1.0,
            depletion_risk=0.0,
            metadata_observed_at=item.confirmed_at,
            source_ids=item.source_ids,
        )
        return _ZoneRecord(
            state=state,
            formed_index=self._bar_index,
            total_touches=1,
            reaction_total_atr=0.0,
        )

    def _ingest_reference_sources(
        self,
        sources: Sequence[LiquidityInventoryItem],
    ) -> None:
        existing_by_source_id: dict[str, _ZoneRecord] = {}
        for record in self._zones.values():
            if record.state.source_kind == "structural_swing":
                continue
            for source_id in record.state.source_ids:
                prior = existing_by_source_id.get(source_id)
                if prior is not None and prior is not record:
                    raise LiquidityProtocolError(
                        "reference source identity maps to multiple S/R zones"
                    )
                existing_by_source_id[source_id] = record
        for item in sources:
            matches = {
                id(existing_by_source_id[source_id]): existing_by_source_id[
                    source_id
                ]
                for source_id in item.source_ids
                if source_id in existing_by_source_id
            }
            if matches:
                if len(matches) != 1:
                    raise LiquidityProtocolError(
                        "reference source identity maps to multiple S/R zones"
                    )
                existing = next(iter(matches.values())).state
                expected_side = (
                    "resistance" if item.side == "above" else "support"
                )
                if (
                    existing.source_ids != item.source_ids
                    or existing.source_kind
                    != self._reference_source_kind(item)
                    or existing.side != expected_side
                    or existing.anchor_price != item.price
                    or existing.formed_at != item.formed_at
                    or existing.confirmed_at != item.confirmed_at
                    or existing.lifecycle
                    is SupportResistanceLifecycle.RETIRED
                ):
                    raise LiquidityProtocolError(
                        "reference source attempted to rewrite frozen S/R facts"
                    )
                continue
            self._prune_zones()
            record = self._new_reference_zone(item)
            self._zones[record.state.zone_id] = record
            for source_id in item.source_ids:
                existing_by_source_id[source_id] = record

    def _add_reference_touch(
        self,
        record: _ZoneRecord,
        candle: Candle,
    ) -> None:
        state = record.state
        atr = max(self._atr(), self.config.tick_size)
        favorable = (
            state.anchor_price - candle.close
            if state.side == "resistance"
            else candle.close - state.anchor_price
        )
        reaction = max(0.0, float(favorable) / atr)
        values = tuple(
            zip(
                (*state.touch_times, candle.end),
                (*state.reaction_magnitudes_atr, reaction),
            )
        )
        limit = self.config.retained_touches
        if len(values) > limit:
            values = (
                values[:2]
                if limit == 2
                else (*values[:2], *values[-(limit - 2) :])
            )
        touch_times = tuple(value[0] for value in values)
        reactions = tuple(float(value[1]) for value in values)
        total_touches = record.total_touches + 1
        reaction_total = record.reaction_total_atr + reaction
        lifecycle = state.lifecycle
        tested_at = state.tested_at
        if lifecycle is SupportResistanceLifecycle.ACTIVE:
            lifecycle = SupportResistanceLifecycle.TESTED
            tested_at = candle.end
        record.state = replace(
            state,
            lifecycle=lifecycle,
            touch_times=touch_times,
            reaction_magnitudes_atr=reactions,
            tested_at=tested_at,
            total_touch_count=total_touches,
            strength=self._strength_from_summary(
                total_touches,
                reaction_total,
            ),
            reaction_quality=self._reaction_quality(
                total_touches,
                reaction_total,
            ),
            depletion_risk=self._depletion_risk(total_touches),
            metadata_observed_at=candle.end,
        )
        record.total_touches = total_touches
        record.reaction_total_atr = reaction_total

    def _update_reference_contacts(self, candle: Candle) -> None:
        for record in self._zones.values():
            state = record.state
            if state.source_kind == "structural_swing":
                continue
            intersects = (
                candle.high >= state.lower_bound
                and candle.low <= state.upper_bound
            )
            if (
                state.lifecycle
                in {
                    SupportResistanceLifecycle.ACTIVE,
                    SupportResistanceLifecycle.TESTED,
                }
                and candle.end > state.confirmed_at
                and intersects
                and not record.contact_active
            ):
                self._add_reference_touch(record, candle)
            record.contact_active = intersects

    def _retire_unreferenced_source_zones(
        self,
        sources: Sequence[LiquidityInventoryItem],
        candle: Candle,
    ) -> None:
        live_source_ids = {
            source_id for item in sources for source_id in item.source_ids
        }
        for record in self._zones.values():
            state = record.state
            if (
                state.source_kind == "structural_swing"
                or state.lifecycle
                in {
                    SupportResistanceLifecycle.REACCEPTED,
                    SupportResistanceLifecycle.RETIRED,
                }
                or not live_source_ids.isdisjoint(state.source_ids)
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
                transition_reason=SUPPORT_RESISTANCE_RETIREMENT_REASON,
                metadata_observed_at=candle.end,
            )
            record.terminal_index = self._bar_index
            record.contact_active = False

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
            reaction_quality=self._reaction_quality(
                total_touches,
                reaction_total,
            ),
            depletion_risk=self._depletion_risk(total_touches),
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
            # The caller may provide only newly confirmed swings.  The
            # zone's frozen anchor is therefore the causal price authority
            # for its first member; the current swing supplies the second.
            # A generation is created on the second touch, before any older
            # swing is allowed to disappear from the caller's retained view.
            if len(state.member_swing_ids) != 2:
                raise LiquidityProtocolError(
                    "initial equal-pool generation requires two members"
                )
            member_prices = (
                float(state.anchor_price),
                float(swing.price),
            )
            generation = _PoolGeneration(
                number=0,
                materialized=False,
                terminal=False,
                member_swing_ids=state.member_swing_ids,
                member_prices=member_prices,
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
                member_prices=(float(swing.price),),
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
        if (
            existing is not None
            and existing.state.lifecycle is LiquidityPoolLifecycle.FORMED
            and (
                (
                    existing.state.side == "above"
                    and swing.price > existing.state.upper_bound
                )
                or (
                    existing.state.side == "below"
                    and swing.price < existing.state.lower_bound
                )
            )
        ):
            # Matching uses the broader frozen S/R tolerance, while an
            # equal-pool touch must not cross its actual member extreme.
            # The completed 1m projection owns the separate sweep event.
            return
        if swing.swing_id not in generation.member_swing_ids:
            all_member_ids = (
                *generation.member_swing_ids,
                swing.swing_id,
            )
            all_member_prices = (
                *generation.member_prices,
                float(swing.price),
            )
            members, touch_times, reactions = self._bounded_touch_history(
                all_member_ids,
                (*generation.touch_times, swing.confirmed_at),
                (
                    *generation.reactions_atr,
                    state.reaction_magnitudes_atr[-1],
                ),
            )
            price_by_id = dict(zip(all_member_ids, all_member_prices))
            member_prices = tuple(price_by_id[item] for item in members)
            generation = _PoolGeneration(
                number=generation.number,
                materialized=generation.materialized,
                terminal=generation.terminal,
                member_swing_ids=members,
                member_prices=member_prices,
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
            # Matching uses the frozen S/R tolerance, but the stop pool is
            # bounded by the actual first two member extremes.  Later touches
            # may strengthen the pool without moving its sweep boundary.
            first_two_prices = generation.member_prices[:2]
            if len(first_two_prices) != 2:
                raise LiquidityProtocolError(
                    "equal pool requires two frozen member prices"
                )
            pool_lower = min(first_two_prices)
            pool_upper = max(first_two_prices)
            pool = LiquidityPoolState(
                pool_id=pool_id,
                timeframe=self.timeframe,
                side=(
                    "above"
                    if state.side == "resistance"
                    else "below"
                ),
                lower_bound=pool_lower,
                upper_bound=pool_upper,
                midpoint=(pool_lower + pool_upper) / 2.0,
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
                structural_rank=state.structural_rank,
                is_protected_swing=state.is_protected_swing,
                visibility_strength=state.visibility_strength,
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
        # A completed-1m projection lands before the native bar that carries
        # its clock is processed; the terminal bar is that native bar, so the
        # record is still exposed in its snapshot before retention counts.
        record.terminal_index = self._bar_index + (
            1
            if self._last_end is not None and observed_at > self._last_end
            else 0
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
                    record.terminal_index = self._bar_index

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
                record.terminal_index = self._bar_index
                self._mark_generation_terminal(record)

    def on_candle(
        self,
        candle: Candle,
        swings: Sequence[SwingPoint],
        structures: Sequence[StructureSequenceState] = (),
        reference_sources: Sequence[LiquidityInventoryItem] = (),
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
        if any(item.timeframe is not self.timeframe for item in structures):
            raise LiquidityProtocolError(
                "liquidity tracker received a foreign-timeframe structure"
            )
        reference_sources = tuple(reference_sources)
        if reference_sources and self.timeframe is not Timeframe.M1:
            raise LiquidityProtocolError(
                "completed-period reference S/R is projected on 1m only"
            )
        if any(
            item.timeframe is not Timeframe.M1
            or item.kind not in _REFERENCE_INVENTORY_KINDS
            or item.lifecycle is not LiquidityInventoryLifecycle.VISIBLE
            or item.lower_bound != item.price
            or item.upper_bound != item.price
            or item.confirmed_at > candle.start
            or len(item.source_ids) != 1
            or (
                item.kind.endswith("_high")
                and item.side != "above"
            )
            or (
                item.kind.endswith("_low")
                and item.side != "below"
            )
            for item in reference_sources
        ):
            raise LiquidityProtocolError(
                "completed-period reference S/R source is invalid"
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
        retained_reference_source_ids = {
            source_id
            for record in self._zones.values()
            if record.state.source_kind != "structural_swing"
            for source_id in record.state.source_ids
        }
        new_reference_items = tuple(
            item
            for item in reference_sources
            if retained_reference_source_ids.isdisjoint(item.source_ids)
        )
        if any(
            self._last_real_end is None
            or item.confirmed_at != self._last_real_end
            or item.confirmed_at
            not in self._strict_prior_atr_by_end
            for item in new_reference_items
        ):
            raise LiquidityProtocolError(
                "reference S/R must be admitted on the first real candle "
                "after its source confirmation"
            )
        new_reference_count = sum(
            1 for _ in new_reference_items
        )
        rollback_state = (
            self._rollback_snapshot()
            if (
                candle.real_completed
                and (new_swing_ids or new_reference_count)
                and (
                    len(self._zones)
                    + len(new_swing_ids)
                    + new_reference_count
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
            self._live_reference_source_ids = {
                source_id
                for item in reference_sources
                for source_id in item.source_ids
            }
            self._bar_index += 1
            self._remember_strict_prior_atr(candle)
            self._compact_terminal_zones()
            self._compact_terminal_pools()
            # A replaced completed-period source becomes cold before this
            # bar's price is interpreted.  The replacement bar cannot
            # retroactively test, break or reaccept the old reference zone.
            self._retire_unreferenced_source_zones(
                reference_sources,
                candle,
            )
            self._ingest_reference_sources(reference_sources)
            self._update_zones(candle)
            self._update_reference_contacts(candle)
            self._update_pools(candle)
            self._latest_swings = {
                item.swing_id: item
                for item in swings
                if item.lifecycle
                in {SwingLifecycle.CONFIRMED, SwingLifecycle.BROKEN}
            }
            self._refresh_structural_metadata(structures)
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
            self._refresh_structural_metadata(structures)
            # Any zone first materialized on this bar uses only ATR available
            # before the bar.  The completed bar joins the ATR window last.
            self._update_atr(candle)
            self._last_real_end = candle.end
        except Exception:
            if rollback_state is not None:
                self.__dict__.clear()
                self.__dict__.update(rollback_state)
            raise

    def _rollback_snapshot(self) -> dict[str, Any]:
        """A bounded copy of the state a failed update must restore.

        Every container is copied one level deep and every mutable record
        inside it is copied; the frozen states, tuples and timestamps under
        them are shared, because nothing in an update mutates them.  A deep
        copy of the whole ``__dict__`` walked all of that on every new swing
        once retention was at capacity, and grew with the retained history.
        """

        snapshot: dict[str, Any] = {}
        for name, value in self.__dict__.items():
            if isinstance(value, dict):
                snapshot[name] = {
                    key: (
                        dataclasses.replace(item)
                        if isinstance(item, (_ZoneRecord, _PoolRecord, _PoolGeneration))
                        else item
                    )
                    for key, item in value.items()
                }
            elif isinstance(value, deque):
                snapshot[name] = deque(value, maxlen=value.maxlen)
            elif isinstance(value, (set, list)):
                snapshot[name] = type(value)(value)
            else:
                snapshot[name] = value
        return snapshot

    def snapshot(
        self,
        *,
        range_auction_sources_only: bool = False,
        include_support_resistance: bool = True,
    ) -> tuple[
        tuple[SupportResistanceState, ...],
        tuple[LiquidityPoolState, ...],
        tuple[LiquidityInventoryItem, ...],
    ]:
        if type(range_auction_sources_only) is not bool:
            raise TypeError("Group 4 source-only flag must be boolean")
        if type(include_support_resistance) is not bool:
            raise TypeError("support/resistance inclusion flag must be boolean")
        if not range_auction_sources_only and not include_support_resistance:
            raise ValueError(
                "support/resistance may be omitted only from a Group 4 "
                "source projection"
            )

        pool_records = tuple(self._pools.values())
        if range_auction_sources_only:
            pool_signature = tuple(
                (
                    record.state,
                    record.structural_rank,
                    record.is_protected_swing,
                    record.visibility_strength,
                )
                for record in pool_records
            )
            zone_signature = (
                tuple(record.state for record in self._zones.values())
                if include_support_resistance
                else ()
            )
            signature: tuple[object, ...] = (
                pool_signature,
                zone_signature,
            )
            cached = self._group4_source_snapshot_cache.get(
                include_support_resistance
            )
            if cached is not None and cached[0] == signature:
                return cached[1]

        zones = (
            tuple(
                (
                    record.state
                    if range_auction_sources_only
                    else replace(
                        record.state,
                        age_bars=(
                            age := max(
                                0,
                                self._bar_index - record.formed_index,
                            )
                        ),
                        freshness=self._freshness(age),
                        depletion_risk=self._depletion_risk(
                            record.total_touches,
                        ),
                        metadata_observed_at=(
                            self._last_end or record.state.confirmed_at
                        ),
                    )
                )
                for record in sorted(
                    self._zones.values(),
                    key=lambda item: (
                        item.state.confirmed_at,
                        item.state.zone_id,
                    ),
                )
            )
            if include_support_resistance
            else ()
        )
        pools = tuple(
            (
                record.state
                if range_auction_sources_only
                else replace(
                    record.state,
                    age_bars=max(
                        0,
                        self._bar_index - record.confirmed_index,
                    ),
                )
            )
            for record in sorted(
                pool_records,
                key=lambda item: (
                    item.state.confirmed_at,
                    item.state.pool_id,
                ),
            )
        )
        inventory: list[LiquidityInventoryItem] = []
        if range_auction_sources_only:
            for pool in pools:
                consumed = pool.swept_at is not None
                record = self._pools[pool.pool_id]
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
                        lifecycle_reason=(
                            "pool_swept" if consumed else None
                        ),
                        structural_rank=record.structural_rank,
                        is_protected_swing=record.is_protected_swing,
                        visibility_strength=record.visibility_strength,
                    )
                )
            result = (zones, pools, tuple(inventory))
            self._group4_source_snapshot_cache[
                include_support_resistance
            ] = (signature, result)
            return result

        zone_by_swing: dict[str, SupportResistanceState] = {}
        for zone in zones:
            for swing_id in zone.member_swing_ids:
                current = zone_by_swing.get(swing_id)
                if (
                    current is None
                    or (
                        zone.structural_rank == "external"
                        and current.structural_rank != "external"
                    )
                ):
                    zone_by_swing[swing_id] = zone
        for swing in self._latest_swings.values():
            source_zone = zone_by_swing.get(swing.swing_id)
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
                    # Structure break remains a close-based structural fact.
                    # Draw consumption is a separate completed-1m wick-cross
                    # fact and is projected only by CausalObserver.
                    lifecycle=LiquidityInventoryLifecycle.VISIBLE,
                    source_ids=(swing.swing_id,),
                    age_bars=swing.age_bars,
                    strength=clamp(swing.magnitude_atr),
                    structural_rank=(
                        "internal"
                        if source_zone is None
                        else source_zone.structural_rank
                    ),
                    is_protected_swing=(
                        False
                        if source_zone is None
                        else source_zone.is_protected_swing
                    ),
                    visibility_strength=(
                        0.0
                        if source_zone is None
                        else source_zone.visibility_strength
                    ),
                )
            )
        for pool in pools:
            consumed = pool.swept_at is not None
            record = self._pools[pool.pool_id]
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
                    structural_rank=record.structural_rank,
                    is_protected_swing=record.is_protected_swing,
                    visibility_strength=record.visibility_strength,
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
