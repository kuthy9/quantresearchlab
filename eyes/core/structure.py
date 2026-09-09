"""Frozen causal HH/HL, LH/LL and BOS state.

The tracker consumes only real completed candles from one timeframe. Swing
confirmation is delayed by that timeframe's configured right-hand span;
structure and BOS clocks are never backfilled to the pivot candle.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence, TypeVar

import pandas as pd

from contract.market import (
    Candle,
    Direction,
    Timeframe,
    candle_identity,
    price_to_ticks,
)
from contract.eye import (
    BOSLifecycle,
    BOSPostBreakState,
    BOSScope,
    BreakOfStructureState,
    STRUCTURE_BREAK_FAILURE_REASON,
    STRUCTURE_FORMATION_FAILURE_REASON,
    StructureLifecycle,
    StructureSequenceState,
    SwingLifecycle,
    SwingPoint,
    SwingRank,
    SwingRelation,
    SwingSide,
)


class StructureProtocolError(ValueError):
    """Raised when a structure protocol or causal update is invalid."""


@dataclass(frozen=True)
class StructureConfig:
    protocol_version: str = "3.0.0-structure-bos.1"
    protocol_hash: str = "unregistered-structure-protocol"
    swing_spans: tuple[tuple[Timeframe, int], ...] = (
        (Timeframe.H4, 2),
        (Timeframe.H1, 2),
        (Timeframe.M15, 2),
        (Timeframe.M5, 2),
        (Timeframe.M1, 1),
    )
    atr_period: int = 14
    tick_size: float = 0.25
    retained_swings: int = 256
    retained_bos: int = 64
    minimum_prominence_atr: float = 0.0

    def __post_init__(self) -> None:
        if not self.protocol_version or not self.protocol_hash:
            raise StructureProtocolError("structure protocol identity is required")
        spans = tuple(self.swing_spans)
        if (
            len(spans) != len({timeframe for timeframe, _ in spans})
            or {timeframe for timeframe, _ in spans}
            != {
                Timeframe.H4,
                Timeframe.H1,
                Timeframe.M15,
                Timeframe.M5,
                Timeframe.M1,
            }
            or any(
                not isinstance(timeframe, Timeframe)
                or type(span) is not int
                or span < 1
                for timeframe, span in spans
            )
        ):
            raise StructureProtocolError(
                "structure swing spans require one positive integer for "
                "every enabled timeframe"
            )
        if (
            self.atr_period < 1
            or self.tick_size <= 0
            or self.retained_swings < 8
            or self.retained_bos < 4
            or self.retained_swings < self.retained_bos + 2
            or not math.isfinite(float(self.minimum_prominence_atr))
            or self.minimum_prominence_atr < 0.0
        ):
            raise StructureProtocolError("invalid structure tracker bounds")

    def span_for(self, timeframe: Timeframe) -> int:
        try:
            return dict(self.swing_spans)[timeframe]
        except KeyError as exc:
            raise StructureProtocolError(
                f"no swing span is registered for {timeframe.value}"
            ) from exc

    @classmethod
    def from_file(
        cls,
        path: str | Path = "configs/primitives_structure_liquidity.json",
        *,
        atr_period: int = 14,
        tick_size: float = 0.25,
    ) -> "StructureConfig":
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = Path(__file__).resolve().parents[2] / source
        raw = source.read_bytes()
        payload = json.loads(raw)
        raw_spans = payload.get("swing_span_by_timeframe", {})
        if not isinstance(raw_spans, dict):
            raise StructureProtocolError(
                "structure swing span registry must be an object"
            )
        return cls(
            protocol_version=str(payload.get("protocol_version", "")),
            protocol_hash=hashlib.sha256(raw).hexdigest(),
            swing_spans=tuple(
                (Timeframe(value), int(span))
                for value, span in raw_spans.items()
            ),
            atr_period=int(atr_period),
            tick_size=float(tick_size),
            minimum_prominence_atr=float(
                payload.get("swing_prominence_atr", 0.0)
            ),
        )


@dataclass
class _SwingRecord:
    point: SwingPoint
    confirmed_index: int | None


@dataclass
class _StructureRecord:
    state: StructureSequenceState
    formed_index: int | None
    confirmed_index: int | None
    broken_index: int | None


@dataclass
class _BosRecord:
    state: BreakOfStructureState
    pending_index: int
    resolved_index: int | None


def _identity(*parts: object) -> str:
    raw = "|".join(str(value) for value in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


_AgeState = TypeVar(
    "_AgeState",
    SwingPoint,
    StructureSequenceState,
    BreakOfStructureState,
)
_AGE_STATE_TYPES = (
    SwingPoint,
    StructureSequenceState,
    BreakOfStructureState,
)


def _copy_with_age(
    value: _AgeState,
    age_bars: int,
) -> _AgeState:
    """Copy an already-validated frozen state and change only its age."""

    value_type = type(value)
    if value_type not in _AGE_STATE_TYPES:
        raise TypeError("snapshot age copy received an untrusted state type")
    if type(age_bars) is not int or age_bars < 0:
        raise ValueError("snapshot age must be a non-negative integer")
    if (
        value_type is StructureSequenceState
        and value.lifecycle is StructureLifecycle.INACTIVE
        and age_bars != 0
    ):
        raise ValueError("inactive structure snapshot age must remain zero")
    output = object.__new__(value_type)
    state = output.__dict__
    state.update(value.__dict__)
    state["age_bars"] = age_bars
    return output


class StructureTracker:
    """Incremental, checkpoint-safe structure state for exactly one timeframe."""

    def __init__(
        self,
        timeframe: Timeframe,
        config: StructureConfig | None = None,
    ) -> None:
        self.timeframe = timeframe
        self.config = config or StructureConfig()
        self.swing_span = self.config.span_for(timeframe)
        self.reset()

    def reset(self) -> None:
        width = self.swing_span * 2 + 1
        self._recent: deque[tuple[int, Candle]] = deque(maxlen=width)
        self._true_ranges: deque[float] = deque(maxlen=self.config.atr_period)
        self._swings: deque[_SwingRecord] = deque()
        self._last_same_side: dict[SwingSide, _SwingRecord] = {}
        self._latest_relation: dict[SwingSide, _SwingRecord] = {}
        self._runs: dict[Direction, dict[SwingSide, int]] = {
            direction: {SwingSide.HIGH: 0, SwingSide.LOW: 0}
            for direction in Direction
        }
        self._structures: dict[Direction, _StructureRecord] = {
            direction: self._inactive_structure(direction)
            for direction in Direction
        }
        self._pending_bos: dict[Direction, _BosRecord] = {}
        self._recent_bos: deque[_BosRecord] = deque(
            maxlen=self.config.retained_bos
        )
        self._last_end: pd.Timestamp | None = None
        self._prior_close: float | None = None
        self._contract: tuple[str, int] | None = None
        self._bar_index = -1

    def reset_for_boundary(
        self,
        *,
        reason: str,
        observed_at: pd.Timestamp,
    ) -> tuple[BreakOfStructureState, ...]:
        """Fail pending BOS at a known boundary, then clear all state."""

        if reason not in {
            "data_gap_reset",
            "contract_change_reset",
        }:
            raise StructureProtocolError(
                "structure reset reason is not registered"
            )
        if observed_at.tzinfo is None:
            raise StructureProtocolError(
                "structure reset clock must be timezone aware"
            )
        pending = tuple(self._pending_bos)
        for direction in pending:
            record = self._pending_bos[direction]
            if observed_at <= record.state.pending_at:
                raise StructureProtocolError(
                    "structure reset cannot predate pending BOS"
                )
            self._fail_pending(
                direction,
                resolved_at=observed_at,
                reason=reason,
            )
        failed = tuple(
            _copy_with_age(
                record.state,
                max(
                    0,
                    (
                        self._bar_index
                        - record.pending_index
                    ),
                ),
            )
            for record in self._recent_bos
            if (
                record.state.lifecycle is BOSLifecycle.FAILED
                and record.state.resolved_at == observed_at
                and record.state.failure_reason == reason
            )
        )
        self.reset()
        return failed

    @property
    def last_end(self) -> pd.Timestamp | None:
        """Clock of the latest completed candle admitted by this tracker."""

        return self._last_end

    def _inactive_structure(self, direction: Direction) -> _StructureRecord:
        return _StructureRecord(
            StructureSequenceState(
                structure_id=None,
                timeframe=self.timeframe,
                direction=direction,
                lifecycle=StructureLifecycle.INACTIVE,
                formed_at=None,
                confirmed_at=None,
                broken_at=None,
                high_run=0,
                low_run=0,
                sequence_count=0,
                latest_high_id=None,
                latest_low_id=None,
                protected_swing_id=None,
                protected_price=None,
                cumulative_magnitude_atr=0.0,
                age_bars=0,
            ),
            None,
            None,
            None,
        )

    def _ticks(self, value: float) -> int:
        return price_to_ticks(
            value,
            self.config.tick_size,
            name="structure price",
        )

    def _atr(self) -> float:
        positive = [value for value in self._true_ranges if value > 0]
        return float(sum(positive) / len(positive)) if positive else self.config.tick_size

    def _update_atr(self, candle: Candle) -> None:
        value = candle.high - candle.low
        if self._prior_close is not None:
            value = max(
                value,
                abs(candle.high - self._prior_close),
                abs(candle.low - self._prior_close),
            )
        self._true_ranges.append(max(0.0, float(value)))
        self._prior_close = float(candle.close)

    def _swing_id(
        self,
        candle: Candle,
        side: SwingSide,
        price_ticks: int,
    ) -> str:
        return _identity(
            self.config.protocol_hash,
            candle.symbol,
            candle.instrument_id,
            self.timeframe.value,
            side.value,
            candle.start.isoformat(),
            price_ticks,
        )

    def _candle_id(self, candle: Candle) -> str:
        return candle_identity(
            candle,
            tick_size=self.config.tick_size,
        )

    @staticmethod
    def _relation(
        side: SwingSide,
        current_ticks: int,
        prior_ticks: int,
    ) -> SwingRelation:
        if side is SwingSide.HIGH:
            if current_ticks > prior_ticks:
                return SwingRelation.HH
            if current_ticks < prior_ticks:
                return SwingRelation.LH
            return SwingRelation.EH
        if current_ticks > prior_ticks:
            return SwingRelation.HL
        if current_ticks < prior_ticks:
            return SwingRelation.LL
        return SwingRelation.EL

    def _candidate_is_left_extreme(
        self,
        side: SwingSide,
        pivot: Candle,
        left: Sequence[Candle],
    ) -> bool:
        pivot_ticks = self._ticks(
            pivot.high if side is SwingSide.HIGH else pivot.low
        )
        values = [
            self._ticks(item.high if side is SwingSide.HIGH else item.low)
            for item in left
        ]
        if side is SwingSide.HIGH:
            return all(pivot_ticks > value for value in values)
        return all(pivot_ticks < value for value in values)

    def _right_confirms(
        self,
        side: SwingSide,
        pivot: Candle,
        right: Sequence[Candle],
    ) -> bool:
        pivot_ticks = self._ticks(
            pivot.high if side is SwingSide.HIGH else pivot.low
        )
        values = [
            self._ticks(item.high if side is SwingSide.HIGH else item.low)
            for item in right
        ]
        if side is SwingSide.HIGH:
            return all(value < pivot_ticks for value in values)
        return all(value > pivot_ticks for value in values)

    def _local_prominence_atr(
        self,
        side: SwingSide,
        pivot: Candle,
        left: Sequence[Candle],
        right: Sequence[Candle],
    ) -> float:
        """Return causal two-sided local pivot prominence at confirmation.

        For a high, both sides must descend from the pivot; for a low, both
        sides must ascend from it.  The weaker side is the registered local
        prominence.  The ATR is the value available at the confirmation
        clock, never a later full-sample statistic.
        """

        if not left or not right:
            return 0.0
        if side is SwingSide.HIGH:
            pivot_price = float(pivot.high)
            left_excursion = pivot_price - min(float(item.low) for item in left)
            right_excursion = pivot_price - min(float(item.low) for item in right)
        else:
            pivot_price = float(pivot.low)
            left_excursion = max(float(item.high) for item in left) - pivot_price
            right_excursion = max(float(item.high) for item in right) - pivot_price
        prominence = max(0.0, min(left_excursion, right_excursion))
        return prominence / max(self._atr(), self.config.tick_size)

    def _build_resolved_swing(
        self,
        side: SwingSide,
        pivot: Candle,
        *,
        confirmed: bool,
        observed_at: pd.Timestamp,
        prominence_atr: float,
        failure_reason: str | None = None,
    ) -> _SwingRecord:
        price = float(pivot.high if side is SwingSide.HIGH else pivot.low)
        price_ticks = self._ticks(price)
        swing_id = self._swing_id(pivot, side, price_ticks)
        prior = self._last_same_side.get(side)
        relation = SwingRelation.NONE
        prior_id = None
        delta_ticks = 0
        if confirmed and prior is not None:
            relation = self._relation(
                side,
                price_ticks,
                prior.point.price_ticks,
            )
            prior_id = prior.point.swing_id
            delta_ticks = price_ticks - prior.point.price_ticks
        delta_points = delta_ticks * self.config.tick_size
        atr = self._atr()
        point = SwingPoint(
            swing_id=swing_id,
            timeframe=self.timeframe,
            symbol=pivot.symbol,
            instrument_id=pivot.instrument_id,
            side=side,
            price=price,
            price_ticks=price_ticks,
            pivot_start=pivot.start,
            pivot_end=pivot.end,
            observed_at=observed_at,
            confirmed_at=observed_at if confirmed else None,
            lifecycle=(
                SwingLifecycle.CONFIRMED
                if confirmed
                else SwingLifecycle.FORMATION_FAILED
            ),
            relation=relation,
            prior_same_side_id=prior_id,
            delta_ticks=delta_ticks,
            delta_points=delta_points,
            magnitude_atr=abs(delta_points) / max(atr, self.config.tick_size),
            prominence_atr=prominence_atr,
            confirmation_delay_bars=self.swing_span,
            # Every causally confirmed pivot is the immutable MICRO atom.
            # Higher roles are assigned later by the event-sourced timeframe
            # hierarchy; the detector never rewrites this source object.
            semantic_rank=(
                SwingRank.MICRO if confirmed else SwingRank.UNRESOLVED
            ),
            age_bars=0,
            failure_reason=None if confirmed else failure_reason,
        )
        return _SwingRecord(
            point=point,
            # This is the causal resolution index for both CONFIRMED and
            # FORMATION_FAILED states.  The internal name is retained for
            # checkpoint compatibility.
            confirmed_index=self._bar_index,
        )

    def _update_runs(self, record: _SwingRecord) -> None:
        side = record.point.side
        relation = record.point.relation
        positive = (
            relation is SwingRelation.HH
            if side is SwingSide.HIGH
            else relation is SwingRelation.HL
        )
        negative = (
            relation is SwingRelation.LH
            if side is SwingSide.HIGH
            else relation is SwingRelation.LL
        )
        self._runs[Direction.LONG][side] = (
            self._runs[Direction.LONG][side] + 1 if positive else 0
        )
        self._runs[Direction.SHORT][side] = (
            self._runs[Direction.SHORT][side] + 1 if negative else 0
        )
        self._latest_relation[side] = record

    @staticmethod
    def _aligned_relation(
        direction: Direction,
        side: SwingSide,
        relation: SwingRelation,
    ) -> bool:
        if direction is Direction.LONG:
            return relation is (
                SwingRelation.HH
                if side is SwingSide.HIGH
                else SwingRelation.HL
            )
        return relation is (
            SwingRelation.LH
            if side is SwingSide.HIGH
            else SwingRelation.LL
        )

    def _new_structure(
        self,
        direction: Direction,
        *,
        lifecycle: StructureLifecycle,
        high: _SwingRecord | None,
        low: _SwingRecord | None,
        forming_anchor: _StructureRecord | None = None,
    ) -> _StructureRecord:
        present = [item for item in (high, low) if item is not None]
        if not present:
            return self._inactive_structure(direction)
        preserve_forming_identity = bool(
            forming_anchor is not None
            and forming_anchor.state.lifecycle is StructureLifecycle.FORMING
            and forming_anchor.state.structure_id is not None
            and forming_anchor.state.formed_at is not None
            and forming_anchor.formed_index is not None
        )
        formed_at = (
            forming_anchor.state.formed_at
            if preserve_forming_identity
            else min(item.point.confirmed_at for item in present)
        )
        confirmed_at = (
            max(item.point.confirmed_at for item in present)
            if lifecycle is StructureLifecycle.CONFIRMED and len(present) == 2
            else None
        )
        protected = (
            low if direction is Direction.LONG else high
        ) if lifecycle is StructureLifecycle.CONFIRMED else None
        high_run = self._runs[direction][SwingSide.HIGH]
        low_run = self._runs[direction][SwingSide.LOW]
        structure_id = (
            forming_anchor.state.structure_id
            if preserve_forming_identity
            else _identity(
                self.config.protocol_hash,
                self.timeframe.value,
                direction.value,
                formed_at.isoformat(),
            )
        )
        cumulative_magnitude = sum(
            item.point.magnitude_atr
            for item in self._swings
            if (
                item.point.confirmed_at is not None
                and item.point.confirmed_at >= formed_at
                and self._aligned_relation(
                    direction,
                    item.point.side,
                    item.point.relation,
                )
            )
        )
        return _StructureRecord(
            StructureSequenceState(
                structure_id=structure_id,
                timeframe=self.timeframe,
                direction=direction,
                lifecycle=lifecycle,
                formed_at=formed_at,
                confirmed_at=confirmed_at,
                broken_at=None,
                high_run=high_run,
                low_run=low_run,
                sequence_count=min(high_run, low_run),
                latest_high_id=None if high is None else high.point.swing_id,
                latest_low_id=None if low is None else low.point.swing_id,
                protected_swing_id=(
                    None if protected is None else protected.point.swing_id
                ),
                protected_price=(
                    None if protected is None else protected.point.price
                ),
                cumulative_magnitude_atr=cumulative_magnitude,
                age_bars=0,
            ),
            (
                forming_anchor.formed_index
                if preserve_forming_identity
                else min(
                    item.confirmed_index
                    for item in present
                    if item.confirmed_index is not None
                )
            ),
            (
                self._bar_index
                if lifecycle is StructureLifecycle.CONFIRMED
                else None
            ),
            None,
        )

    def _update_structures(self, new_records: Sequence[_SwingRecord]) -> None:
        high = self._latest_relation.get(SwingSide.HIGH)
        low = self._latest_relation.get(SwingSide.LOW)
        for direction in Direction:
            current = self._structures[direction]
            high_aligned = bool(
                high is not None
                and self._aligned_relation(
                    direction,
                    SwingSide.HIGH,
                    high.point.relation,
                )
            )
            low_aligned = bool(
                low is not None
                and self._aligned_relation(
                    direction,
                    SwingSide.LOW,
                    low.point.relation,
                )
            )
            if current.state.lifecycle is StructureLifecycle.CONFIRMED:
                protected = None
                if direction is Direction.LONG:
                    protected = next(
                        (
                            item
                            for item in new_records
                            if item.point.side is SwingSide.LOW
                            and item.point.relation is SwingRelation.HL
                        ),
                        None,
                    )
                else:
                    protected = next(
                        (
                            item
                            for item in new_records
                            if item.point.side is SwingSide.HIGH
                            and item.point.relation is SwingRelation.LH
                        ),
                        None,
                    )
                if (
                    protected is not None
                    and current.state.protected_price is not None
                    and (
                        (
                            direction is Direction.LONG
                            and protected.point.price
                            <= current.state.protected_price
                        )
                        or (
                            direction is Direction.SHORT
                            and protected.point.price
                            >= current.state.protected_price
                        )
                    )
                ):
                    # A later relation may be locally HL/LH after an intervening
                    # extreme while still loosening the frozen structural
                    # invalidation.  Locked protection can only tighten.
                    protected = None
                aligned_magnitude = sum(
                    item.point.magnitude_atr
                    for item in new_records
                    if self._aligned_relation(
                        direction,
                        item.point.side,
                        item.point.relation,
                    )
                )
                state = replace(
                    current.state,
                    high_run=self._runs[direction][SwingSide.HIGH],
                    low_run=self._runs[direction][SwingSide.LOW],
                    sequence_count=min(
                        self._runs[direction][SwingSide.HIGH],
                        self._runs[direction][SwingSide.LOW],
                    ),
                    latest_high_id=(
                        current.state.latest_high_id
                        if high is None
                        else high.point.swing_id
                    ),
                    latest_low_id=(
                        current.state.latest_low_id
                        if low is None
                        else low.point.swing_id
                    ),
                    protected_swing_id=(
                        current.state.protected_swing_id
                        if protected is None
                        else protected.point.swing_id
                    ),
                    protected_price=(
                        current.state.protected_price
                        if protected is None
                        else protected.point.price
                    ),
                    cumulative_magnitude_atr=(
                        current.state.cumulative_magnitude_atr
                        + aligned_magnitude
                    ),
                )
                self._structures[direction] = replace(current, state=state)
                continue

            after_broken = current.state.broken_at
            eligible_high = bool(
                high_aligned
                and high is not None
                and (
                    after_broken is None
                    or high.point.confirmed_at > after_broken
                )
            )
            eligible_low = bool(
                low_aligned
                and low is not None
                and (
                    after_broken is None
                    or low.point.confirmed_at > after_broken
                )
            )
            if eligible_high and eligible_low:
                self._structures[direction] = self._new_structure(
                    direction,
                    lifecycle=StructureLifecycle.CONFIRMED,
                    high=high,
                    low=low,
                    forming_anchor=current,
                )
            elif eligible_high or eligible_low:
                self._structures[direction] = self._new_structure(
                    direction,
                    lifecycle=StructureLifecycle.FORMING,
                    high=high if eligible_high else None,
                    low=low if eligible_low else None,
                    forming_anchor=current,
                )
            elif current.state.lifecycle is StructureLifecycle.FORMING:
                failed_at = max(
                    item.point.confirmed_at
                    for item in new_records
                    if item.point.confirmed_at is not None
                )
                failed_state = replace(
                    current.state,
                    lifecycle=StructureLifecycle.FORMATION_FAILED,
                    formation_failed_at=failed_at,
                    high_run=self._runs[direction][SwingSide.HIGH],
                    low_run=self._runs[direction][SwingSide.LOW],
                    sequence_count=min(
                        self._runs[direction][SwingSide.HIGH],
                        self._runs[direction][SwingSide.LOW],
                    ),
                    latest_high_id=(
                        current.state.latest_high_id
                        if high is None
                        else high.point.swing_id
                    ),
                    latest_low_id=(
                        current.state.latest_low_id
                        if low is None
                        else low.point.swing_id
                    ),
                    failure_reason=STRUCTURE_FORMATION_FAILURE_REASON,
                )
                self._structures[direction] = replace(
                    current,
                    state=failed_state,
                )
            elif current.state.lifecycle not in {
                StructureLifecycle.BROKEN,
                StructureLifecycle.FORMATION_FAILED,
            }:
                self._structures[direction] = self._inactive_structure(direction)

    def _fail_pending(
        self,
        direction: Direction,
        *,
        resolved_at: pd.Timestamp,
        reason: str,
    ) -> None:
        record = self._pending_bos.pop(direction, None)
        if record is None:
            return
        state = replace(
            record.state,
            lifecycle=BOSLifecycle.FAILED,
            resolved_at=resolved_at,
            failure_reason=reason,
        )
        self._recent_bos.append(
            _BosRecord(state, record.pending_index, self._bar_index)
        )

    def _bos_scope(
        self,
        direction: Direction,
        target_swing_id: str,
    ) -> tuple[BOSScope, str | None]:
        target = next(
            (
                record.point
                for record in self._swings
                if record.point.swing_id == target_swing_id
            ),
            None,
        )
        if target is None:
            raise StructureProtocolError("BOS target swing is not retained")
        same = self._structures[direction].state
        opposite_direction = (
            Direction.SHORT if direction is Direction.LONG else Direction.LONG
        )
        opposite = self._structures[opposite_direction].state
        if (
            opposite.lifecycle is StructureLifecycle.CONFIRMED
            and opposite.protected_swing_id == target_swing_id
        ):
            return BOSScope.OPPOSED, opposite.structure_id
        if (
            same.lifecycle is StructureLifecycle.CONFIRMED
            and target_swing_id
            == (
                same.latest_high_id
                if direction is Direction.LONG
                else same.latest_low_id
            )
        ):
            return BOSScope.CONTINUATION, same.structure_id
        return BOSScope.LOCAL, None

    def _advance_post_break_states(self, candle: Candle) -> None:
        close_ticks = self._ticks(candle.close)
        updated: deque[_BosRecord] = deque(maxlen=self._recent_bos.maxlen)
        for record in self._recent_bos:
            state = record.state
            if (
                state.lifecycle is not BOSLifecycle.CONFIRMED
                or state.post_break_state is not BOSPostBreakState.PENDING
                or state.resolved_at is None
                or candle.end <= state.resolved_at
            ):
                updated.append(record)
                continue
            accepted = (
                close_ticks > state.target_ticks
                if state.direction is Direction.LONG
                else close_ticks < state.target_ticks
            )
            state = replace(
                state,
                post_break_state=(
                    BOSPostBreakState.ACCEPTED
                    if accepted
                    else BOSPostBreakState.REJECTED
                ),
                accepted_at=candle.end if accepted else None,
                rejected_at=None if accepted else candle.end,
            )
            updated.append(replace(record, state=state))
        self._recent_bos = updated

    def _resolve_bos(self, candle: Candle) -> None:
        close_ticks = self._ticks(candle.close)
        high_ticks = self._ticks(candle.high)
        low_ticks = self._ticks(candle.low)
        for direction, record in tuple(self._pending_bos.items()):
            state = record.state
            if candle.end <= state.pending_at:
                continue
            confirmed = (
                close_ticks > state.target_ticks
                if direction is Direction.LONG
                else close_ticks < state.target_ticks
            )
            if confirmed:
                break_distance_atr = (
                    abs(candle.close - state.target_price)
                    / max(self._atr(), self.config.tick_size)
                )
                resolved = replace(
                    state,
                    lifecycle=BOSLifecycle.CONFIRMED,
                    resolved_at=candle.end,
                    strength=min(1.0, break_distance_atr),
                    break_bar_id=self._candle_id(candle),
                    break_distance_atr=break_distance_atr,
                    post_break_state=BOSPostBreakState.PENDING,
                )
                self._recent_bos.append(
                    _BosRecord(resolved, record.pending_index, self._bar_index)
                )
                self._pending_bos.pop(direction)
                continue
            attempted = (
                high_ticks > state.target_ticks
                if direction is Direction.LONG
                else low_ticks < state.target_ticks
            )
            if attempted:
                self._pending_bos[direction] = replace(
                    record,
                    state=replace(
                        state,
                        attempt_count=state.attempt_count + 1,
                        last_attempt_at=candle.end,
                        attempt_clocks=(
                            *state.attempt_clocks,
                            candle.end,
                        ),
                    ),
                )

    def _mark_swings_broken(self, candle: Candle) -> None:
        close_ticks = self._ticks(candle.close)
        updated: deque[_SwingRecord] = deque()
        by_id: dict[str, _SwingRecord] = {}
        for record in self._swings:
            point = record.point
            broken = bool(
                point.lifecycle is SwingLifecycle.CONFIRMED
                and point.confirmed_at is not None
                and candle.end > point.confirmed_at
                and (
                    close_ticks > point.price_ticks
                    if point.side is SwingSide.HIGH
                    else close_ticks < point.price_ticks
                )
            )
            if broken:
                record = replace(
                    record,
                    point=replace(
                        point,
                        lifecycle=SwingLifecycle.BROKEN,
                        broken_at=candle.end,
                        failure_reason="close_beyond_swing",
                    ),
                )
            updated.append(record)
            by_id[record.point.swing_id] = record
        self._swings = updated
        for side, record in tuple(self._last_same_side.items()):
            self._last_same_side[side] = by_id.get(record.point.swing_id, record)
        for side, record in tuple(self._latest_relation.items()):
            self._latest_relation[side] = by_id.get(record.point.swing_id, record)

    def _append_swing(self, record: _SwingRecord) -> None:
        """Keep bounded output history without dropping active BOS targets."""

        self._swings.append(record)
        while len(self._swings) > self.config.retained_swings:
            protected_ids = {
                item.state.target_swing_id
                for item in (
                    *self._recent_bos,
                    *self._pending_bos.values(),
                )
            }
            values = list(self._swings)
            removable = next(
                (
                    index
                    for index, item in enumerate(values)
                    if item.point.swing_id not in protected_ids
                ),
                None,
            )
            if removable is None:
                raise StructureProtocolError(
                    "retained swing bound is smaller than active BOS evidence"
                )
            values.pop(removable)
            self._swings = deque(values)

    def _break_structures(self, candle: Candle) -> None:
        close_ticks = self._ticks(candle.close)
        for direction, record in tuple(self._structures.items()):
            state = record.state
            if (
                state.lifecycle is not StructureLifecycle.CONFIRMED
                or state.protected_price is None
            ):
                continue
            protected_ticks = self._ticks(state.protected_price)
            broken = (
                close_ticks < protected_ticks
                if direction is Direction.LONG
                else close_ticks > protected_ticks
            )
            if not broken:
                continue
            broken_state = replace(
                state,
                lifecycle=StructureLifecycle.BROKEN,
                broken_at=candle.end,
                failure_reason=STRUCTURE_BREAK_FAILURE_REASON,
            )
            self._structures[direction] = _StructureRecord(
                broken_state,
                record.formed_index,
                record.confirmed_index,
                self._bar_index,
            )
            pending = self._pending_bos.get(direction)
            if (
                pending is not None
                and (
                    pending.state.source_structure_id == state.structure_id
                    or (
                        pending.state.source_structure_id is None
                        and state.confirmed_at is not None
                        and pending.state.pending_at <= state.confirmed_at
                    )
                )
            ):
                self._fail_pending(
                    direction,
                    resolved_at=candle.end,
                    reason="opposite_structure_break",
                )

    def _refresh_pending_targets(
        self,
        observed_at: pd.Timestamp,
    ) -> None:
        for direction in Direction:
            side = (
                SwingSide.HIGH
                if direction is Direction.LONG
                else SwingSide.LOW
            )
            target = self._last_same_side.get(side)
            if (
                target is None
                or target.point.confirmed_at is None
                or target.point.lifecycle is not SwingLifecycle.CONFIRMED
                or target.point.relation is SwingRelation.NONE
            ):
                continue
            scope, source_structure = self._bos_scope(
                direction,
                target.point.swing_id,
            )
            current = self._pending_bos.get(direction)
            if (
                current is not None
                and current.state.target_swing_id == target.point.swing_id
                and current.state.scope is scope
                and current.state.source_structure_id == source_structure
            ):
                continue
            if current is not None:
                # The target price can stay unchanged while a newly confirmed
                # structure gives that target different semantic authority.
                # Close the old generation and start a fresh one at the
                # current evidence clock; never backdate CONTINUATION/OPPOSED
                # authority into the earlier LOCAL attempt.
                self._fail_pending(
                    direction,
                    resolved_at=observed_at,
                    reason="superseded",
                )
            source_state = next(
                (
                    record.state
                    for record in self._structures.values()
                    if record.state.structure_id == source_structure
                ),
                None,
            )
            # A terminal BOS attempt cannot be re-armed retroactively from
            # the target's original confirmation clock.  The new attempt
            # starts when the evidence refresh makes it pending again.
            pending_at = max(target.point.confirmed_at, observed_at)
            if (
                source_state is not None
                and source_state.confirmed_at is not None
            ):
                pending_at = max(pending_at, source_state.confirmed_at)
            bos_id = _identity(
                self.config.protocol_hash,
                self.timeframe.value,
                direction.value,
                target.point.swing_id,
                pending_at.isoformat(),
            )
            self._pending_bos[direction] = _BosRecord(
                BreakOfStructureState(
                    bos_id=bos_id,
                    timeframe=self.timeframe,
                    direction=direction,
                    lifecycle=BOSLifecycle.PENDING,
                    scope=scope,
                    target_swing_id=target.point.swing_id,
                    source_structure_id=source_structure,
                    target_price=target.point.price,
                    target_ticks=target.point.price_ticks,
                    pending_at=pending_at,
                    resolved_at=None,
                    age_bars=0,
                ),
                self._bar_index,
                None,
            )

    def _resolve_pivot(self, observed_at: pd.Timestamp) -> None:
        width = self.swing_span * 2 + 1
        if len(self._recent) < width:
            return
        values = list(self._recent)
        pivot_position = self.swing_span
        pivot = values[pivot_position][1]
        left = [item[1] for item in values[:pivot_position]]
        right = [item[1] for item in values[pivot_position + 1 :]]
        candidates = tuple(
            side
            for side in (SwingSide.HIGH, SwingSide.LOW)
            if self._candidate_is_left_extreme(side, pivot, left)
        )
        right_confirmations = {
            side: self._right_confirms(side, pivot, right)
            for side in candidates
        }
        prominences = {
            side: self._local_prominence_atr(side, pivot, left, right)
            for side in candidates
        }
        confirmations = {
            side: bool(
                right_confirmations[side]
                and prominences[side] >= self.config.minimum_prominence_atr
            )
            for side in candidates
        }
        if sum(confirmations.values()) > 1:
            # An outside bar can geometrically dominate both sides of the
            # window.  That is an ambiguous expansion, not two independent
            # pivots at one candle; declining both avoids inventing an order.
            return
        new_records: list[_SwingRecord] = []
        for side in candidates:
            confirmed = confirmations[side]
            record = self._build_resolved_swing(
                side,
                pivot,
                confirmed=confirmed,
                observed_at=observed_at,
                prominence_atr=prominences[side],
                failure_reason=(
                    "right_side_invalidated"
                    if not right_confirmations[side]
                    else "insufficient_prominence"
                ),
            )
            self._append_swing(record)
            if confirmed:
                self._last_same_side[side] = record
                self._update_runs(record)
                new_records.append(record)
        if new_records:
            self._update_structures(new_records)
            self._refresh_pending_targets(observed_at)

    def on_candle(self, candle: Candle) -> None:
        if candle.timeframe is not self.timeframe or not candle.complete:
            raise StructureProtocolError(
                "structure tracker accepts only complete candles of its timeframe"
            )
        # Detector admission is exact and precedes every tracker mutation.
        # All subsequent swing/raw-break comparisons use integer coordinates.
        candle.ohlc_ticks_for(self.config.tick_size)
        contract = (candle.symbol, int(candle.instrument_id))
        if self._contract is not None and contract != self._contract:
            raise StructureProtocolError(
                "contract changed without an explicit structure reset"
            )
        if self._last_end is not None and candle.end <= self._last_end:
            raise StructureProtocolError("duplicate or out-of-order structure candle")
        self._contract = contract
        if not candle.real_completed:
            # Registered synthetic market time advances the causal clock, but
            # cannot alter real-candle structure semantics or semantic age.
            self._last_end = candle.end
            return
        self._bar_index += 1
        self._advance_post_break_states(candle)
        self._resolve_bos(candle)
        self._mark_swings_broken(candle)
        self._break_structures(candle)
        self._update_atr(candle)
        self._recent.append((self._bar_index, candle))
        self._resolve_pivot(candle.end)
        self._last_end = candle.end

    def sync(self, candles: Sequence[Candle]) -> None:
        values = tuple(candles)
        if not values:
            return
        if any(item.timeframe is not self.timeframe for item in values):
            raise StructureProtocolError("structure history mixes timeframes")
        if self._last_end is None:
            for candle in values:
                self.on_candle(candle)
            return
        for candle in values:
            if candle.end > self._last_end:
                self.on_candle(candle)

    def _forming_swings(self) -> tuple[SwingPoint, ...]:
        values = list(self._recent)
        output: list[SwingPoint] = []
        for position in range(self.swing_span, len(values)):
            _, pivot = values[position]
            right = [item[1] for item in values[position + 1 :]]
            if len(right) >= self.swing_span:
                continue
            left = [
                item[1]
                for item in values[
                    position - self.swing_span : position
                ]
            ]
            for side in (SwingSide.HIGH, SwingSide.LOW):
                if not self._candidate_is_left_extreme(side, pivot, left):
                    continue
                price = float(
                    pivot.high if side is SwingSide.HIGH else pivot.low
                )
                ticks = self._ticks(price)
                output.append(
                    SwingPoint(
                        swing_id=self._swing_id(pivot, side, ticks),
                        timeframe=self.timeframe,
                        symbol=pivot.symbol,
                        instrument_id=pivot.instrument_id,
                        side=side,
                        price=price,
                        price_ticks=ticks,
                        pivot_start=pivot.start,
                        pivot_end=pivot.end,
                        observed_at=pivot.end,
                        confirmed_at=None,
                        lifecycle=SwingLifecycle.FORMING,
                        age_bars=len(right),
                    )
                )
        return tuple(
            sorted(
                output,
                key=lambda item: (item.pivot_start, item.side.value),
            )
        )

    def snapshot(
        self,
    ) -> tuple[
        tuple[SwingPoint, ...],
        tuple[StructureSequenceState, ...],
        tuple[BreakOfStructureState, ...],
    ]:
        swings = [
            _copy_with_age(
                record.point,
                (
                    0
                    if record.confirmed_index is None
                    else max(0, self._bar_index - record.confirmed_index)
                ),
            )
            for record in self._swings
        ]
        swings.extend(self._forming_swings())
        structures = tuple(
            _copy_with_age(
                record.state,
                max(
                    0,
                    self._bar_index
                    - (
                        record.confirmed_index
                        if record.confirmed_index is not None
                        else record.formed_index
                        if record.formed_index is not None
                        else self._bar_index
                    ),
                ),
            )
            for direction, record in sorted(
                self._structures.items(),
                key=lambda item: item[0].value,
            )
        )
        bos_records = list(self._recent_bos) + list(self._pending_bos.values())
        bos = tuple(
            _copy_with_age(
                record.state,
                max(
                    0,
                    (
                        record.resolved_index
                        if record.resolved_index is not None
                        else self._bar_index
                    )
                    - record.pending_index,
                ),
            )
            for record in sorted(
                bos_records,
                key=lambda item: (
                    item.state.pending_at,
                    item.state.direction.value,
                    item.state.bos_id,
                ),
            )
        )
        return (
            tuple(
                sorted(
                    swings,
                    key=lambda item: (
                        item.observed_at,
                        item.pivot_start,
                        item.side.value,
                    ),
                )
            ),
            structures,
            bos,
        )


__all__ = [
    "StructureConfig",
    "StructureProtocolError",
    "StructureTracker",
]
