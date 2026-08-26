"""Causal 5m FVG and order-block primitives derived from frozen sources."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .displacement import (
    DisplacementLifecycle,
    DisplacementState,
    DisplacementUpdate,
)
from .model import (
    BOSLifecycle,
    BOSScope,
    BreakOfStructureState,
    Candle,
    Direction,
    FVGQualification,
    FairValueGapLifecycle,
    FairValueGapState,
    ORDER_BLOCK_FUNNEL_STAGES,
    OrderBlockAttemptOutcome,
    OrderBlockFunnelSnapshot,
    OrderBlockLifecycle,
    OrderBlockState,
    Timeframe,
    aware_timestamp,
    candle_identity,
    price_to_ticks,
)
FVG_BOUNDARY_REASONS = frozenset(
    {
        "data_gap_reset",
        "contract_change_reset",
        "data_anomaly",
        "tick_size_mismatch",
        "semantic_reset",
    }
)
ORDER_BLOCK_BOUNDARY_REASONS = FVG_BOUNDARY_REASONS
WINDOW_RESET_REASONS = frozenset(
    {
        "registered_session_reset",
        "synthetic_interruption",
    }
)
ZONE_TRACKER_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ZoneProtocol:
    """Executable mirror of the frozen Group 3 descriptive contract."""

    protocol_hash: str
    tick_size: float
    protocol_version: str = "3.2.0-group3.4"
    timeframe: str = "5m"
    fvg_source_bars: int = 3
    fvg_formation_atr_period: int = 14
    ob_anchor_history_bars: int = 64
    maximum_fvg_states: int = 256
    maximum_order_block_states: int = 128

    def __post_init__(self) -> None:
        if (
            not isinstance(self.protocol_hash, str)
            or len(self.protocol_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.protocol_hash
            )
            or self.protocol_version != "3.2.0-group3.4"
            or self.timeframe != "5m"
            or not math.isclose(
                float(self.tick_size),
                0.25,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or self.fvg_source_bars != 3
            or self.fvg_formation_atr_period != 14
            or self.ob_anchor_history_bars != 64
            or self.maximum_fvg_states != 256
            or self.maximum_order_block_states != 128
        ):
            raise ValueError("Group 3 protocol differs from its frozen contract")

    @classmethod
    def from_file(cls, path: str | Path) -> "ZoneProtocol":
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = Path(__file__).resolve().parents[1] / source
        raw = source.read_bytes()
        payload = json.loads(raw)
        return cls(
            protocol_hash=hashlib.sha256(raw).hexdigest(),
            protocol_version=payload["protocol_version"],
            tick_size=payload["tick_size"],
            timeframe=payload["timeframe"],
            fvg_source_bars=payload["fvg_source_bars"],
            fvg_formation_atr_period=payload[
                "fvg_formation_atr_period"
            ],
            ob_anchor_history_bars=payload[
                "ob_anchor_history_bars"
            ],
            maximum_fvg_states=payload["maximum_fvg_states"],
            maximum_order_block_states=payload[
                "maximum_order_block_states"
            ],
        )


@dataclass(frozen=True)
class ZoneUpdate:
    fair_value_gaps: tuple[FairValueGapState, ...]
    order_blocks: tuple[OrderBlockState, ...]
    fvg_transitions: tuple[FairValueGapState, ...] = ()
    order_block_transitions: tuple[OrderBlockState, ...] = ()
    order_block_funnel: tuple[OrderBlockFunnelSnapshot, ...] = ()
    boundary_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "fair_value_gaps",
            "order_blocks",
            "fvg_transitions",
            "order_block_transitions",
            "order_block_funnel",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            self.boundary_reason is not None
            and self.boundary_reason
            not in FVG_BOUNDARY_REASONS | WINDOW_RESET_REASONS
        ):
            raise ValueError("unregistered Group 3 update boundary")
        if self.boundary_reason in WINDOW_RESET_REASONS and (
            self.fvg_transitions
            or self.order_block_transitions
            or self.order_block_funnel
        ):
            raise ValueError(
                "soft Group 3 boundary cannot emit terminal transitions"
            )
        if self.boundary_reason in FVG_BOUNDARY_REASONS and (
            any(
                state.lifecycle
                is not FairValueGapLifecycle.INVALIDATED
                or state.transition_reason != self.boundary_reason
                for state in self.fvg_transitions
            )
            or any(
                state.lifecycle is not OrderBlockLifecycle.FAILED
                or state.transition_reason != self.boundary_reason
                for state in self.order_block_transitions
            )
        ):
            raise ValueError(
                "hard Group 3 boundary transition is inconsistent"
            )
        if any(
            not isinstance(item, OrderBlockFunnelSnapshot)
            for item in self.order_block_funnel
        ):
            raise TypeError("Group 3 OB funnel record is not typed")
        if self.boundary_reason is not None and self.order_block_funnel:
            raise ValueError("Group 3 boundary cannot emit an OB funnel")
        funnel_clocks = tuple(
            item.observed_at for item in self.order_block_funnel
        )
        if (
            funnel_clocks != tuple(sorted(funnel_clocks))
            or len(funnel_clocks) != len(set(funnel_clocks))
        ):
            raise ValueError("Group 3 OB funnel clocks are invalid")


@dataclass(frozen=True)
class _FrozenOrderBlockCandidate:
    candle: Candle
    candle_id: str
    cluster: tuple[Candle, ...]
    cluster_ids: tuple[str, ...]
    source_displacement_state: DisplacementState
    source_displacement_transition_identity: str


@dataclass(frozen=True)
class ZoneBOSSource:
    """Contract-bound envelope for one typed M5 BOS state."""

    state: BreakOfStructureState
    symbol: str
    instrument_id: int
    protocol_hash: str
    tick_size: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.state, BreakOfStructureState)
            or self.state.timeframe is not Timeframe.M5
            or not isinstance(self.symbol, str)
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or not isinstance(self.protocol_hash, str)
            or len(self.protocol_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.protocol_hash
            )
            or not math.isfinite(float(self.tick_size))
            or self.tick_size <= 0
        ):
            raise ValueError("invalid contract-bound BOS source")


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


class CausalZoneTracker:
    """Incrementally maintain qualified FVG and order-block zones."""

    _CHECKPOINT_FIELDS = frozenset(
        {
            "protocol",
            "_history",
            "_episode_membership",
            "_active_transition_ids",
            "_ob_candidates",
            "_fair_value_gaps",
            "_fvg_order",
            "_order_blocks",
            "_order_block_order",
            "_exposed_terminal_ids",
            "_identity",
            "_window_epoch_known",
            "_source_displacement_protocol_hash",
            "_source_structure_protocol_hash",
            "_last_clock",
            "_last_input_kind",
            "_last_boundary_input",
            "_last_candle_input",
            "_last_output",
            "_failed",
            "_zone_tracker_checkpoint_schema_version",
        }
    )

    def __init__(
        self,
        protocol: ZoneProtocol,
        *,
        displacement_protocol_hash: str | None = None,
        structure_protocol_hash: str | None = None,
    ) -> None:
        if not isinstance(protocol, ZoneProtocol):
            raise TypeError("a frozen Group 3 protocol is required")
        self.protocol = protocol
        self._history: deque[Candle] = deque(
            maxlen=protocol.ob_anchor_history_bars
        )
        self._episode_membership: dict[str, str] = {}
        self._active_transition_ids: dict[str, str] = {}
        self._ob_candidates: dict[
            str,
            _FrozenOrderBlockCandidate | None,
        ] = {}
        self._fair_value_gaps: dict[str, FairValueGapState] = {}
        self._fvg_order: deque[str] = deque()
        self._order_blocks: dict[str, OrderBlockState] = {}
        self._order_block_order: deque[str] = deque()
        self._exposed_terminal_ids: set[str] = set()
        self._identity: tuple[str, int] | None = None
        self._window_epoch_known = False
        self._source_displacement_protocol_hash = (
            self._validated_source_hash(
                displacement_protocol_hash,
                name="displacement",
            )
        )
        self._source_structure_protocol_hash = (
            self._validated_source_hash(
                structure_protocol_hash,
                name="structure",
            )
        )
        self._last_clock: pd.Timestamp | None = None
        self._last_input_kind: str | None = None
        self._last_boundary_input: tuple[
            str,
            pd.Timestamp,
        ] | None = None
        self._last_candle_input: tuple[object, ...] | None = None
        self._last_output: ZoneUpdate | None = None
        self._failed = False

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_zone_tracker_checkpoint_schema_version"] = (
            ZONE_TRACKER_CHECKPOINT_SCHEMA_VERSION
        )
        if set(state) != self._CHECKPOINT_FIELDS:
            raise ValueError("Group 3 tracker checkpoint state is not exact")
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        schema_version = (
            state.get("_zone_tracker_checkpoint_schema_version")
            if isinstance(state, Mapping)
            else None
        )
        if (
            not isinstance(state, Mapping)
            or set(state) != self._CHECKPOINT_FIELDS
            or type(schema_version) is not int
            or schema_version != ZONE_TRACKER_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError("Group 3 tracker checkpoint schema changed")
        restored = dict(state)
        restored.pop("_zone_tracker_checkpoint_schema_version")
        self.__dict__.clear()
        self.__dict__.update(restored)

    @staticmethod
    def _validated_source_hash(
        value: str | None,
        *,
        name: str,
    ) -> str | None:
        if value is None:
            return None
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value
            )
        ):
            raise ValueError(f"invalid Group 3 {name} protocol hash")
        return value

    def _bind_source_hash(
        self,
        actual: str,
        *,
        attribute: str,
        name: str,
    ) -> None:
        validated = self._validated_source_hash(actual, name=name)
        expected = getattr(self, attribute)
        if expected is None:
            setattr(self, attribute, validated)
        elif validated != expected:
            self._failed = True
            raise ValueError(
                f"Group 3 {name} protocol provenance disagrees"
            )

    def _transaction_clone(self) -> "CausalZoneTracker":
        candidate = object.__new__(type(self))
        candidate.__dict__.update(self.__dict__)
        candidate._history = deque(
            self._history,
            maxlen=self._history.maxlen,
        )
        candidate._episode_membership = dict(
            self._episode_membership
        )
        candidate._active_transition_ids = dict(
            self._active_transition_ids
        )
        candidate._ob_candidates = dict(self._ob_candidates)
        candidate._fair_value_gaps = dict(self._fair_value_gaps)
        candidate._fvg_order = deque(self._fvg_order)
        candidate._order_blocks = dict(self._order_blocks)
        candidate._order_block_order = deque(
            self._order_block_order
        )
        candidate._exposed_terminal_ids = set(
            self._exposed_terminal_ids
        )
        return candidate

    def _commit(self, candidate: "CausalZoneTracker") -> None:
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)

    def snapshot(
        self,
    ) -> tuple[
        tuple[FairValueGapState, ...],
        tuple[OrderBlockState, ...],
    ]:
        return self._snapshots()

    def current_update(self) -> ZoneUpdate:
        """Return the current physical Zone state."""

        return self._update()

    def _snapshots(
        self,
    ) -> tuple[
        tuple[FairValueGapState, ...],
        tuple[OrderBlockState, ...],
    ]:
        fair_value_gaps = tuple(
            self._fair_value_gaps[entity_id]
            for entity_id in self._fvg_order
            if entity_id in self._fair_value_gaps
        )
        order_blocks = tuple(
            self._order_blocks[entity_id]
            for entity_id in self._order_block_order
            if entity_id in self._order_blocks
        )
        return fair_value_gaps, order_blocks

    def _update(
        self,
        fvg_transitions: Iterable[FairValueGapState] = (),
        order_block_transitions: Iterable[OrderBlockState] = (),
        order_block_funnel: Iterable[OrderBlockFunnelSnapshot] = (),
        *,
        boundary_reason: str | None = None,
    ) -> ZoneUpdate:
        fair_value_gaps, order_blocks = self._snapshots()
        return ZoneUpdate(
            fair_value_gaps=fair_value_gaps,
            order_blocks=order_blocks,
            fvg_transitions=tuple(fvg_transitions),
            order_block_transitions=tuple(order_block_transitions),
            order_block_funnel=tuple(order_block_funnel),
            boundary_reason=boundary_reason,
        )

    def _ticks(self, value: float) -> int:
        return price_to_ticks(
            value,
            self.protocol.tick_size,
            name="Group 3 price",
        )

    def _candle_id(self, candle: Candle) -> str:
        return candle_identity(
            candle,
            tick_size=self.protocol.tick_size,
        )

    def _strict_prior_atr(self, current: Candle) -> float:
        """Formation ATR from completed bars strictly preceding current."""

        prior = tuple(
            candle
            for candle in self._history
            if candle.real_completed and candle.end <= current.start
        )
        true_ranges: list[float] = []
        for index, candle in enumerate(prior):
            if index == 0:
                value = float(candle.high - candle.low)
            else:
                prior_close = float(prior[index - 1].close)
                value = max(
                    float(candle.high - candle.low),
                    abs(float(candle.high) - prior_close),
                    abs(float(candle.low) - prior_close),
                )
            if math.isfinite(value) and value > 0.0:
                true_ranges.append(value)
        window = true_ranges[-self.protocol.fvg_formation_atr_period :]
        return (
            sum(window) / len(window)
            if window
            else self.protocol.tick_size
        )

    @staticmethod
    def _is_fvg_terminal(state: FairValueGapState) -> bool:
        return state.lifecycle in {
            FairValueGapLifecycle.MITIGATED,
            FairValueGapLifecycle.INVALIDATED,
            FairValueGapLifecycle.EXPIRED,
        }

    @staticmethod
    def _is_order_block_terminal(state: OrderBlockState) -> bool:
        return state.lifecycle in {
            OrderBlockLifecycle.MITIGATED,
            OrderBlockLifecycle.FAILED,
        }

    def _mark_terminals_exposed(self) -> None:
        self._exposed_terminal_ids.update(
            state.fvg_id
            for state in self._fair_value_gaps.values()
            if self._is_fvg_terminal(state)
        )
        self._exposed_terminal_ids.update(
            state.order_block_id
            for state in self._order_blocks.values()
            if self._is_order_block_terminal(state)
        )

    def _admit_capacity(
        self,
        *,
        states: dict[str, FairValueGapState] | dict[str, OrderBlockState],
        order: deque[str],
        maximum: int,
        terminal,
    ) -> None:
        if len(states) < maximum:
            return
        evictable: list[tuple[pd.Timestamp, str]] = []
        for entity_id, state in states.items():
            if (
                entity_id not in self._exposed_terminal_ids
                or not terminal(state)
            ):
                continue
            if isinstance(state, FairValueGapState):
                terminal_clock = (
                    state.mitigated_at or state.invalidated_at
                )
            else:
                terminal_clock = state.mitigated_at or state.failed_at
            if terminal_clock is None:
                raise RuntimeError(
                    "terminal Group 3 entity lacks its terminal clock"
                )
            evictable.append((terminal_clock, entity_id))
        if not evictable:
            self._failed = True
            raise RuntimeError(
                "Group 3 capacity cannot admit a new entity safely; "
                "discard this tracker and resume from a prior checkpoint"
            )
        _, evicted_id = min(evictable)
        states.pop(evicted_id)
        order.remove(evicted_id)
        self._exposed_terminal_ids.discard(evicted_id)

    def _clear_windows(self, *, clear_identity: bool) -> None:
        self._history.clear()
        self._episode_membership.clear()
        self._active_transition_ids.clear()
        self._ob_candidates.clear()
        if clear_identity:
            self._identity = None

    def _apply_boundary(
        self,
        reason: str,
        clock: pd.Timestamp,
    ) -> ZoneUpdate:
        fvg_transitions: list[FairValueGapState] = []
        order_block_transitions: list[OrderBlockState] = []
        hard_boundary = reason in FVG_BOUNDARY_REASONS
        if hard_boundary:
            for entity_id, state in tuple(
                self._fair_value_gaps.items()
            ):
                if self._is_fvg_terminal(state):
                    continue
                terminal = replace(
                    state,
                    lifecycle=FairValueGapLifecycle.INVALIDATED,
                    state_started_at=clock,
                    last_updated_at=clock,
                    invalidated_at=clock,
                    transition_reason=reason,
                )
                self._fair_value_gaps[entity_id] = terminal
                fvg_transitions.append(terminal)
            for entity_id, state in tuple(self._order_blocks.items()):
                if self._is_order_block_terminal(state):
                    continue
                terminal = replace(
                    state,
                    lifecycle=OrderBlockLifecycle.FAILED,
                    state_started_at=clock,
                    last_updated_at=clock,
                    failed_at=clock,
                    transition_reason=reason,
                )
                self._order_blocks[entity_id] = terminal
                order_block_transitions.append(terminal)
        self._clear_windows(clear_identity=hard_boundary)
        self._window_epoch_known = True
        self._last_clock = clock
        output = self._update(
            fvg_transitions,
            order_block_transitions,
            boundary_reason=reason,
        )
        self._mark_terminals_exposed()
        return output

    def on_boundary(
        self,
        reason: str,
        observed_at: pd.Timestamp,
    ) -> ZoneUpdate:
        if self._failed:
            raise RuntimeError("Group 3 tracker is terminally failed")
        clock = aware_timestamp(observed_at, name="group3.boundary")
        boundary_input = (reason, clock)
        if (
            self._last_input_kind == "boundary"
            and self._last_boundary_input == boundary_input
            and self._last_output is not None
        ):
            return self._last_output
        if (
            self._last_clock is not None
            and clock <= self._last_clock
        ):
            self._failed = True
            raise ValueError("Group 3 boundary clock is out of order")
        if reason not in FVG_BOUNDARY_REASONS | WINDOW_RESET_REASONS:
            raise ValueError("unregistered Group 3 boundary reason")
        candidate = self._transaction_clone()
        try:
            output = candidate._apply_boundary(reason, clock)
            candidate._last_input_kind = "boundary"
            candidate._last_boundary_input = boundary_input
            candidate._last_candle_input = None
            candidate._last_output = output
        except Exception:
            self._failed = True
            raise
        self._commit(candidate)
        return output

    def _validate_candle(
        self,
        candle: Candle,
        displacement: DisplacementUpdate,
        bos_sources: tuple[ZoneBOSSource, ...],
    ) -> None:
        if (
            candle.timeframe is not Timeframe.M5
            or not candle.complete
            or (candle.expected_minutes, candle.observed_minutes)
            != (5, 5)
        ):
            raise ValueError("Group 3 requires a completed 5m candle")
        if (
            self._last_clock is not None
            and candle.end <= self._last_clock
        ):
            self._failed = True
            raise ValueError(
                "duplicate or out-of-order Group 3 completed candle"
            )
        if (
            self._last_input_kind == "candle"
            and candle.start != self._last_clock
        ):
            self._failed = True
            raise ValueError(
                "Group 3 completed candle is not contiguous"
            )
        for value in (
            candle.open,
            candle.high,
            candle.low,
            candle.close,
        ):
            self._ticks(value)
        identity = (candle.symbol, int(candle.instrument_id))
        if self._identity is not None and identity != self._identity:
            self._failed = True
            raise ValueError(
                "Group 3 contract changed without a registered boundary"
            )
        state = displacement.state
        if state is not None:
            self._bind_source_hash(
                state.protocol_hash,
                attribute="_source_displacement_protocol_hash",
                name="displacement",
            )
            if (
                state.timeframe is not Timeframe.M5
                or state.symbol != candle.symbol
                or state.instrument_id != candle.instrument_id
                or state.observed_at > candle.end
                or state.prefix_last_admitted_at > candle.end
            ):
                self._failed = True
                raise ValueError(
                    "Group 3 displacement provenance disagrees with candle"
                )
        for transition in displacement.transitions:
            transition_state = transition.state
            self._bind_source_hash(
                transition_state.protocol_hash,
                attribute="_source_displacement_protocol_hash",
                name="displacement",
            )
            if (
                transition_state.observed_at > candle.end
                or transition_state.prefix_last_admitted_at > candle.end
                or transition_state.timeframe is not Timeframe.M5
                or transition_state.symbol != candle.symbol
                or transition_state.instrument_id != candle.instrument_id
            ):
                self._failed = True
                raise ValueError(
                    "Group 3 displacement transition contains future "
                    "or cross-contract evidence"
                )
        for source in bos_sources:
            bos = source.state
            self._bind_source_hash(
                source.protocol_hash,
                attribute="_source_structure_protocol_hash",
                name="structure",
            )
            if (
                source.symbol != candle.symbol
                or source.instrument_id != candle.instrument_id
                or not math.isclose(
                    source.tick_size,
                    self.protocol.tick_size,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or (
                    bos.resolved_at is not None
                    and bos.resolved_at > candle.end
                )
                or self._ticks(bos.target_price) != bos.target_ticks
            ):
                self._failed = True
                raise ValueError(
                    "Group 3 BOS provenance disagrees with candle"
                )

    def _validate_censored_boundary(
        self,
        candle: Candle,
        displacement: DisplacementUpdate,
        *,
        mapped_reason: str,
    ) -> None:
        censored = tuple(
            transition
            for transition in displacement.transitions
            if (
                transition.state.lifecycle
                is DisplacementLifecycle.CENSORED
            )
        )
        if (
            displacement.state is not None
            or len(censored) != 1
            or len(displacement.transitions) != 1
        ):
            self._failed = True
            raise ValueError(
                "Group 3 received ambiguous displacement boundary"
            )
        state = censored[0].state
        self._bind_source_hash(
            state.protocol_hash,
            attribute="_source_displacement_protocol_hash",
            name="displacement",
        )
        source_identity = (state.symbol, state.instrument_id)
        expected_terminal_reason = {
            "data_gap_reset": "data_gap_history_reset",
            "contract_change_reset": "contract_change_history_reset",
            "data_anomaly": "data_anomaly",
            "registered_session_reset": "registered_session_reset",
            "synthetic_interruption": "synthetic_interruption",
        }[mapped_reason]
        if (
            self._identity is None
            or source_identity != self._identity
            or state.timeframe is not Timeframe.M5
            or state.terminal_reason != expected_terminal_reason
            or state.observed_at != candle.end
            or state.terminal_at != candle.end
            or state.state_started_at != candle.end
            or state.last_updated_at != candle.end
            or state.started_at > state.prefix_last_admitted_at
            or state.prefix_last_admitted_at > candle.end
            or (
                state.active_at is not None
                and not (
                    state.started_at
                    <= state.active_at
                    <= state.prefix_last_admitted_at
                )
            )
            or state.favorable_extreme_first_observed_at
            > state.prefix_last_admitted_at
            or state.terminal_evidence_candle_id is not None
        ):
            self._failed = True
            raise ValueError(
                "Group 3 displacement boundary provenance disagrees"
            )

    def _validate_boundary_candle_identity(
        self,
        candle: Candle,
        *,
        mapped_reason: str,
    ) -> None:
        if self._identity is None:
            if mapped_reason == "contract_change_reset":
                self._failed = True
                raise ValueError(
                    "Group 3 cannot prove a contract-change boundary "
                    "without prior contract identity"
                )
            return
        candle_identity = (
            candle.symbol,
            int(candle.instrument_id),
        )
        if mapped_reason == "contract_change_reset":
            valid = candle_identity != self._identity
        else:
            valid = candle_identity == self._identity
        if not valid:
            self._failed = True
            raise ValueError(
                "Group 3 boundary candle identity disagrees with its "
                "registered reason"
            )

    def _advance_fvgs(
        self,
        candle: Candle,
    ) -> list[FairValueGapState]:
        transitions: list[FairValueGapState] = []
        for entity_id, state in tuple(self._fair_value_gaps.items()):
            if self._is_fvg_terminal(state):
                continue
            age = state.age_bars + 1
            if state.direction is Direction.LONG:
                penetration = (
                    state.upper_bound
                    - min(float(candle.low), state.upper_bound)
                ) / state.width_points
                invalidated = self._ticks(candle.close) < self._ticks(
                    state.lower_bound
                )
                mitigated = self._ticks(candle.low) <= self._ticks(
                    state.lower_bound
                )
                partial = (
                    self._ticks(state.lower_bound)
                    < self._ticks(candle.low)
                    < self._ticks(state.upper_bound)
                )
            else:
                penetration = (
                    max(float(candle.high), state.lower_bound)
                    - state.lower_bound
                ) / state.width_points
                invalidated = self._ticks(candle.close) > self._ticks(
                    state.upper_bound
                )
                mitigated = self._ticks(candle.high) >= self._ticks(
                    state.upper_bound
                )
                partial = (
                    self._ticks(state.lower_bound)
                    < self._ticks(candle.high)
                    < self._ticks(state.upper_bound)
                )
            fill = max(
                state.max_fill_fraction,
                min(1.0, max(0.0, float(penetration))),
            )
            midpoint_crossed = bool(
                state.midpoint_touched_at is None and fill >= 0.5
            )
            if invalidated:
                updated = replace(
                    state,
                    lifecycle=FairValueGapLifecycle.INVALIDATED,
                    state_started_at=candle.end,
                    last_updated_at=candle.end,
                    age_bars=age,
                    max_fill_fraction=1.0,
                    midpoint_touched_at=(
                        state.midpoint_touched_at or candle.end
                    ),
                    invalidated_at=candle.end,
                    transition_reason="close_through_far_edge",
                )
                transitions.append(updated)
            elif mitigated:
                updated = replace(
                    state,
                    lifecycle=FairValueGapLifecycle.MITIGATED,
                    state_started_at=candle.end,
                    last_updated_at=candle.end,
                    age_bars=age,
                    max_fill_fraction=1.0,
                    midpoint_touched_at=(
                        state.midpoint_touched_at or candle.end
                    ),
                    mitigated_at=candle.end,
                    transition_reason="far_edge_reached",
                )
                transitions.append(updated)
            elif partial:
                updated = replace(
                    state,
                    lifecycle=FairValueGapLifecycle.PARTIAL,
                    state_started_at=(
                        candle.end
                        if state.lifecycle
                        is FairValueGapLifecycle.OPEN
                        else state.state_started_at
                    ),
                    last_updated_at=candle.end,
                    age_bars=age,
                    max_fill_fraction=fill,
                    partial_at=state.partial_at or candle.end,
                    midpoint_touched_at=(
                        candle.end
                        if midpoint_crossed
                        else state.midpoint_touched_at
                    ),
                    transition_reason=(
                        "midpoint_touched"
                        if midpoint_crossed
                        else "near_edge_penetrated"
                    ),
                )
                if (
                    state.lifecycle is FairValueGapLifecycle.OPEN
                    or midpoint_crossed
                ):
                    transitions.append(updated)
            else:
                updated = replace(
                    state,
                    last_updated_at=candle.end,
                    age_bars=age,
                )
            self._fair_value_gaps[entity_id] = updated
        return transitions

    def _advance_order_blocks(
        self,
        candle: Candle,
    ) -> list[OrderBlockState]:
        transitions: list[OrderBlockState] = []
        for entity_id, state in tuple(self._order_blocks.items()):
            if self._is_order_block_terminal(state):
                continue
            age = state.age_bars + 1
            intersects = bool(
                self._ticks(candle.low)
                <= self._ticks(state.upper_bound)
                and self._ticks(candle.high)
                >= self._ticks(state.lower_bound)
            )
            failed = (
                self._ticks(candle.close)
                < self._ticks(state.lower_bound)
                if state.direction is Direction.LONG
                else self._ticks(candle.close)
                > self._ticks(state.upper_bound)
            )
            if failed:
                updated = replace(
                    state,
                    lifecycle=OrderBlockLifecycle.FAILED,
                    state_started_at=candle.end,
                    last_updated_at=candle.end,
                    age_bars=age,
                    first_test_at=(
                        state.first_test_at
                        or (candle.end if intersects else None)
                    ),
                    failed_at=candle.end,
                    transition_reason="close_through_distal_edge",
                )
                transitions.append(updated)
            elif intersects:
                updated = replace(
                    state,
                    lifecycle=OrderBlockLifecycle.MITIGATED,
                    state_started_at=candle.end,
                    last_updated_at=candle.end,
                    age_bars=age,
                    first_test_at=state.first_test_at or candle.end,
                    mitigated_at=candle.end,
                    transition_reason="zone_intersected",
                )
                transitions.append(updated)
            elif state.lifecycle is OrderBlockLifecycle.CREATED:
                updated = replace(
                    state,
                    lifecycle=OrderBlockLifecycle.UNTESTED,
                    state_started_at=candle.end,
                    last_updated_at=candle.end,
                    age_bars=age,
                    transition_reason="first_later_bar_no_touch",
                )
                transitions.append(updated)
            else:
                updated = replace(
                    state,
                    last_updated_at=candle.end,
                    age_bars=age,
                )
            self._order_blocks[entity_id] = updated
        return transitions

    @staticmethod
    def _open_displacement_state(
        displacement: DisplacementUpdate,
    ) -> DisplacementState | None:
        state = displacement.state
        if (
            state is None
            or state.lifecycle
            not in {
                DisplacementLifecycle.STARTED,
                DisplacementLifecycle.ACTIVE,
            }
        ):
            return None
        return state

    def _freeze_new_displacement_sources(
        self,
        candle: Candle,
        candle_id: str,
        displacement: DisplacementUpdate,
    ) -> _FrozenOrderBlockCandidate | None:
        for transition in displacement.transitions:
            state = transition.state
            if state.lifecycle is DisplacementLifecycle.ACTIVE:
                self._active_transition_ids[
                    state.entity_id
                ] = transition.transition_id
            elif state.lifecycle in {
                DisplacementLifecycle.EXHAUSTED,
                DisplacementLifecycle.CENSORED,
            }:
                self._active_transition_ids.pop(state.entity_id, None)
                self._ob_candidates.pop(state.entity_id, None)
        started = tuple(
            transition
            for transition in displacement.transitions
            if transition.state.lifecycle
            is DisplacementLifecycle.STARTED
        )
        if len(started) > 1:
            self._failed = True
            raise ValueError(
                "one candle cannot start multiple displacement episodes"
            )
        if not started:
            return None
        started_transition = started[0]
        state = started_transition.state
        history = (*tuple(self._history), candle)
        candle_ids = tuple(self._candle_id(item) for item in history)
        if (
            state.seed_candle_id not in candle_ids
            or not set(state.admitted_candle_ids).issubset(candle_ids)
        ):
            self._failed = True
            raise ValueError(
                "promoted displacement prefix is absent from Group 3 history"
            )
        for admitted_id in state.admitted_candle_ids:
            self._episode_membership[admitted_id] = state.entity_id
        seed_index = candle_ids.index(state.seed_candle_id)
        prior_history = history[:seed_index]
        if not prior_history:
            self._ob_candidates[state.entity_id] = None
            return None
        anchor = prior_history[-1]
        seed = history[seed_index]
        valid_anchor = (
            anchor.end == seed.start
            and anchor.real_completed
            and anchor.symbol == seed.symbol
            and anchor.instrument_id == seed.instrument_id
            and (
                self._ticks(anchor.close) < self._ticks(anchor.open)
                if state.direction is Direction.LONG
                else self._ticks(anchor.close) > self._ticks(anchor.open)
            )
        )
        if not valid_anchor:
            self._ob_candidates[state.entity_id] = None
            return None

        # The seed-adjacent bar must strictly oppose the displacement.  Only
        # its immediately contiguous reverse/doji predecessors may extend the
        # frozen cluster; an older unrelated reverse candle is never selected.
        reverse_cluster = [anchor]
        next_start = anchor.start
        for item in reversed(prior_history[:-1]):
            if (
                item.end != next_start
                or not item.real_completed
                or item.symbol != seed.symbol
                or item.instrument_id != seed.instrument_id
            ):
                break
            body_sign = self._ticks(item.close) - self._ticks(item.open)
            extends = (
                body_sign <= 0
                if state.direction is Direction.LONG
                else body_sign >= 0
            )
            if not extends:
                break
            reverse_cluster.append(item)
            next_start = item.start
        cluster = tuple(reversed(reverse_cluster))
        cluster_ids = tuple(self._candle_id(item) for item in cluster)
        candidate = _FrozenOrderBlockCandidate(
            candle=anchor,
            candle_id=self._candle_id(anchor),
            cluster=cluster,
            cluster_ids=cluster_ids,
            source_displacement_state=state,
            source_displacement_transition_identity=(
                started_transition.transition_id
            ),
        )
        self._ob_candidates[state.entity_id] = candidate
        return candidate

    def _remember_membership(
        self,
        candle_id: str,
        candle: Candle,
        displacement: DisplacementUpdate,
    ) -> None:
        state = self._open_displacement_state(displacement)
        if state is None:
            return
        if (
            state.last_valid_candle_id != candle_id
            or state.prefix_last_admitted_at != candle.end
        ):
            self._failed = True
            raise ValueError(
                "displacement did not prove current candle membership"
            )
        self._episode_membership[candle_id] = state.entity_id

    def _create_fvg(
        self,
        displacement: DisplacementUpdate,
    ) -> FairValueGapState | None:
        if len(self._history) < self.protocol.fvg_source_bars:
            return None
        c1, c2, c3 = tuple(self._history)[-3:]
        if not (
            c1.end == c2.start
            and c2.end == c3.start
            and all(
                candle.real_completed
                and candle.symbol == c3.symbol
                and candle.instrument_id == c3.instrument_id
                for candle in (c1, c2, c3)
            )
        ):
            return None
        c1_id, c2_id, c3_id = (
            self._candle_id(candle)
            for candle in (c1, c2, c3)
        )
        if self._ticks(c3.low) > self._ticks(c1.high):
            direction = Direction.LONG
            lower_bound = float(c1.high)
            upper_bound = float(c3.low)
        elif self._ticks(c3.high) < self._ticks(c1.low):
            direction = Direction.SHORT
            lower_bound = float(c3.high)
            upper_bound = float(c1.low)
        else:
            return None
        source = displacement.state
        linked = bool(
            source is not None
            and source.lifecycle is DisplacementLifecycle.ACTIVE
            and source.direction is direction
            and source.active_at is not None
            and source.active_at <= c3.end
            and c2_id in source.admitted_candle_ids
            and self._episode_membership.get(c2_id) == source.entity_id
            and self._active_transition_ids.get(source.entity_id) is not None
        )
        active_transition_id = (
            self._active_transition_ids[source.entity_id]
            if linked and source is not None
            else None
        )
        width_points = upper_bound - lower_bound
        width_ticks = self._ticks(upper_bound) - self._ticks(
            lower_bound
        )
        fvg_id = _identity(
            "group3-fvg-v2",
            self.protocol.protocol_hash,
            c3.symbol,
            c3.instrument_id,
            Timeframe.M5,
            direction,
            c1_id,
            c2_id,
            c3_id,
        )
        if fvg_id in self._fair_value_gaps:
            return None
        self._admit_capacity(
            states=self._fair_value_gaps,
            order=self._fvg_order,
            maximum=self.protocol.maximum_fvg_states,
            terminal=self._is_fvg_terminal,
        )
        formation_atr = self._strict_prior_atr(c3)
        width_atr = width_points / formation_atr
        state = FairValueGapState(
            fvg_id=fvg_id,
            protocol_hash=self.protocol.protocol_hash,
            symbol=c3.symbol,
            instrument_id=c3.instrument_id,
            timeframe=Timeframe.M5,
            direction=direction,
            lifecycle=FairValueGapLifecycle.OPEN,
            qualification=(
                FVGQualification.DISPLACEMENT_LINKED
                if linked
                else FVGQualification.RAW
            ),
            source_displacement_id=(
                source.entity_id if linked and source is not None else None
            ),
            source_active_transition_id=active_transition_id,
            source_displacement_protocol_hash=(
                source.protocol_hash if linked and source is not None else None
            ),
            source_displacement_started_at=(
                source.started_at if linked and source is not None else None
            ),
            source_displacement_active_at=(
                source.active_at if linked and source is not None else None
            ),
            source_displacement_prefix_commitment=(
                source.prefix_commitment
                if linked and source is not None
                else None
            ),
            source_candle_ids=(c1_id, c2_id, c3_id),
            source_candle_starts=(c1.start, c2.start, c3.start),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            midpoint=(lower_bound + upper_bound) / 2.0,
            invalidation_price=(
                lower_bound
                if direction is Direction.LONG
                else upper_bound
            ),
            width_points=width_points,
            width_ticks=width_ticks,
            formation_atr=formation_atr,
            width_atr=width_atr,
            # Width is descriptive geometry, not a trade-quality score.
            strength=0.0,
            formed_at=c3.end,
            confirmed_at=c3.end,
            state_started_at=c3.end,
            last_updated_at=c3.end,
            age_bars=0,
            max_fill_fraction=0.0,
        )
        self._fair_value_gaps[fvg_id] = state
        self._fvg_order.append(fvg_id)
        return state

    def _create_order_block(
        self,
        candle: Candle,
        displacement: DisplacementUpdate,
        bos_sources: Iterable[ZoneBOSSource],
    ) -> tuple[OrderBlockState | None, OrderBlockFunnelSnapshot]:
        stage_counts = dict.fromkeys(ORDER_BLOCK_FUNNEL_STAGES, 0)

        def finish(
            outcome: OrderBlockAttemptOutcome,
            state: OrderBlockState | None = None,
        ) -> tuple[OrderBlockState | None, OrderBlockFunnelSnapshot]:
            return state, OrderBlockFunnelSnapshot(
                observed_at=candle.end,
                stages=tuple(
                    (name, stage_counts[name])
                    for name in ORDER_BLOCK_FUNNEL_STAGES
                ),
                outcome=outcome,
            )

        source = displacement.state
        if (
            source is None
            or source.lifecycle is not DisplacementLifecycle.ACTIVE
            or source.active_at is None
            or source.active_at > candle.end
        ):
            return finish(
                OrderBlockAttemptOutcome.NO_ACTIVE_DISPLACEMENT
            )
        stage_counts["active_displacement"] = 1
        compatible = tuple(
            bos_source
            for bos_source in bos_sources
            if (
                bos_source.state.timeframe is Timeframe.M5
                and bos_source.state.lifecycle is BOSLifecycle.CONFIRMED
                and bos_source.state.direction is source.direction
                and bos_source.state.resolved_at == candle.end
                and bos_source.state.scope
                in {BOSScope.CONTINUATION, BOSScope.OPPOSED}
                and (
                    bos_source.state.scope is BOSScope.CONTINUATION
                    or (
                        bos_source.state.mss_qualified
                        and bos_source.state.source_displacement_id
                        == source.entity_id
                    )
                )
                and bos_source.state.pending_at <= source.started_at
                and source.started_at
                <= bos_source.state.resolved_at
                <= source.prefix_last_admitted_at
            )
        )
        stage_counts["compatible_bos"] = len(compatible)
        if not compatible:
            return finish(OrderBlockAttemptOutcome.NO_COMPATIBLE_BOS)
        eligible = tuple(
            bos_source
            for bos_source in compatible
            if bos_source.state.break_bar_id
            in source.admitted_candle_ids
        )
        stage_counts[
            "break_bar_belongs_to_displacement"
        ] = len(eligible)
        if not eligible:
            return finish(
                OrderBlockAttemptOutcome.BREAK_BAR_NOT_IN_DISPLACEMENT
            )
        candidate = self._ob_candidates.get(source.entity_id)
        if candidate is None:
            return finish(
                OrderBlockAttemptOutcome.REVERSE_ANCHOR_CLUSTER_MISSING
            )
        stage_counts["reverse_anchor_cluster_found"] = 1
        if len(eligible) != 1:
            return finish(
                OrderBlockAttemptOutcome.DUPLICATE_ELIGIBLE_BOS
            )
        stage_counts["unique_eligible_bos"] = 1
        bos_source = eligible[0]
        bos = bos_source.state
        active_transition_id = self._active_transition_ids.get(
            source.entity_id
        )
        if active_transition_id is None:
            return finish(
                OrderBlockAttemptOutcome.ACTIVE_TRANSITION_MISSING
            )
        anchor = candidate.candle
        lower_bound = min(float(item.low) for item in candidate.cluster)
        upper_bound = max(float(item.high) for item in candidate.cluster)
        body_lower_bound = min(
            min(float(item.open), float(item.close))
            for item in candidate.cluster
        )
        body_upper_bound = max(
            max(float(item.open), float(item.close))
            for item in candidate.cluster
        )
        width_points = upper_bound - lower_bound
        width_ticks = self._ticks(upper_bound) - self._ticks(
            lower_bound
        )
        if width_ticks <= 0:
            return finish(
                OrderBlockAttemptOutcome.INVALID_ANCHOR_WIDTH
            )
        order_block_id = _identity(
            "group3-order-block-v1",
            self.protocol.protocol_hash,
            candle.symbol,
            candle.instrument_id,
            Timeframe.M5,
            source.direction,
            *candidate.cluster_ids,
            source.entity_id,
            bos.bos_id,
        )
        if order_block_id in self._order_blocks:
            return finish(
                OrderBlockAttemptOutcome.DUPLICATE_ORDER_BLOCK
            )
        self._admit_capacity(
            states=self._order_blocks,
            order=self._order_block_order,
            maximum=self.protocol.maximum_order_block_states,
            terminal=self._is_order_block_terminal,
        )
        state = OrderBlockState(
            order_block_id=order_block_id,
            protocol_hash=self.protocol.protocol_hash,
            symbol=candle.symbol,
            instrument_id=candle.instrument_id,
            timeframe=Timeframe.M5,
            direction=source.direction,
            lifecycle=OrderBlockLifecycle.CREATED,
            source_displacement_id=source.entity_id,
            source_active_transition_id=active_transition_id,
            source_displacement_protocol_hash=source.protocol_hash,
            source_displacement_seed_candle_id=source.seed_candle_id,
            source_displacement_started_at=source.started_at,
            source_displacement_active_at=source.active_at,
            source_displacement_prefix_commitment=(
                source.prefix_commitment
            ),
            source_bos_id=bos.bos_id,
            source_bos_protocol_hash=bos_source.protocol_hash,
            source_bos_target_swing_id=bos.target_swing_id,
            source_bos_structure_id=bos.source_structure_id,
            source_bos_scope=bos.scope,
            source_bos_pending_at=bos.pending_at,
            source_bos_resolved_at=bos.resolved_at,
            source_bos_break_bar_id=bos.break_bar_id,
            source_bos_mss_qualified=bos.mss_qualified,
            anchor_candle_id=candidate.candle_id,
            anchor_candle_ids=candidate.cluster_ids,
            anchor_start=anchor.start,
            anchor_end=anchor.end,
            anchor_open=float(anchor.open),
            anchor_close=float(anchor.close),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            body_lower_bound=body_lower_bound,
            body_upper_bound=body_upper_bound,
            midpoint=(lower_bound + upper_bound) / 2.0,
            invalidation_price=(
                lower_bound
                if source.direction is Direction.LONG
                else upper_bound
            ),
            width_points=width_points,
            width_ticks=width_ticks,
            width_atr=width_points / source.atr0,
            strength=bos.strength,
            formed_at=bos.resolved_at,
            confirmed_at=bos.resolved_at,
            state_started_at=bos.resolved_at,
            last_updated_at=bos.resolved_at,
            age_bars=0,
        )
        self._order_blocks[order_block_id] = state
        self._order_block_order.append(order_block_id)
        stage_counts["ob_created"] = 1
        return finish(OrderBlockAttemptOutcome.CREATED, state)

    def _trim_membership(self) -> None:
        retained_ids = {
            self._candle_id(candle)
            for candle in self._history
        }
        self._episode_membership = {
            candle_id: entity_id
            for candle_id, entity_id
            in self._episode_membership.items()
            if candle_id in retained_ids
        }

    def _apply_completed_5m(
        self,
        candle: Candle,
        displacement: DisplacementUpdate,
        bos_sources: tuple[ZoneBOSSource, ...],
    ) -> ZoneUpdate:
        identity = (candle.symbol, int(candle.instrument_id))
        candle_id = self._candle_id(candle)
        fvg_transitions = self._advance_fvgs(candle)
        order_block_transitions = self._advance_order_blocks(candle)
        self._freeze_new_displacement_sources(
            candle,
            candle_id,
            displacement,
        )
        self._history.append(candle)
        if len(self._history) == self.protocol.ob_anchor_history_bars:
            self._window_epoch_known = True
        self._identity = identity
        self._remember_membership(
            candle_id,
            candle,
            displacement,
        )
        self._trim_membership()

        created_fvg = self._create_fvg(displacement)
        if created_fvg is not None:
            fvg_transitions.append(created_fvg)
        created_order_block, order_block_funnel = self._create_order_block(
            candle,
            displacement,
            bos_sources,
        )
        if created_order_block is not None:
            order_block_transitions.append(created_order_block)

        self._last_clock = candle.end
        output = self._update(
            fvg_transitions,
            order_block_transitions,
            (order_block_funnel,),
        )
        self._mark_terminals_exposed()
        return output

    def on_completed_5m(
        self,
        candle: Candle,
        displacement: DisplacementUpdate,
        confirmed_bos: Iterable[ZoneBOSSource] = (),
    ) -> ZoneUpdate:
        if self._failed:
            raise RuntimeError("Group 3 tracker is terminally failed")
        if not isinstance(candle, Candle):
            raise TypeError("Group 3 requires a Candle input")
        candle.ohlc_ticks_for(self.protocol.tick_size)
        bos_sources = tuple(confirmed_bos)
        if any(
            not isinstance(source, ZoneBOSSource)
            for source in bos_sources
        ):
            raise TypeError(
                "Group 3 requires contract-bound BOS source envelopes"
            )
        if (
            candle.timeframe is not Timeframe.M5
            or not candle.complete
            or (candle.expected_minutes, candle.observed_minutes)
            != (5, 5)
        ):
            raise ValueError("Group 3 requires a completed 5m candle")
        candle_input = (candle, displacement, bos_sources)
        if (
            self._last_input_kind in {"candle", "candle_boundary"}
            and self._last_candle_input == candle_input
            and self._last_output is not None
        ):
            return self._last_output
        if (
            self._last_clock is not None
            and candle.end <= self._last_clock
        ):
            self._failed = True
            raise ValueError(
                "duplicate or out-of-order Group 3 completed candle"
            )

        censored_reasons = tuple(
            transition.state.terminal_reason
            for transition in displacement.transitions
            if (
                transition.state.lifecycle
                is DisplacementLifecycle.CENSORED
            )
        )
        mapped_reason: str | None = None
        if censored_reasons:
            if len(censored_reasons) != 1:
                self._failed = True
                raise ValueError(
                    "Group 3 received ambiguous displacement boundary"
                )
            mapped_reason = {
                "data_gap_history_reset": "data_gap_reset",
                "contract_change_history_reset": (
                    "contract_change_reset"
                ),
                "data_anomaly": "data_anomaly",
                "registered_session_reset": (
                    "registered_session_reset"
                ),
                "synthetic_interruption": "synthetic_interruption",
            }.get(censored_reasons[0])
            if mapped_reason is None:
                self._failed = True
                raise ValueError(
                    "Group 3 received an unregistered displacement boundary"
                )
        elif not candle.real_completed:
            if (
                displacement.state is not None
                or displacement.transitions
            ):
                self._failed = True
                raise ValueError(
                    "synthetic Group 3 boundary lacks exact "
                    "displacement censorship"
                )
            mapped_reason = "synthetic_interruption"

        candidate = self._transaction_clone()
        try:
            if mapped_reason is not None:
                candidate._validate_boundary_candle_identity(
                    candle,
                    mapped_reason=mapped_reason,
                )
                if censored_reasons:
                    candidate._validate_censored_boundary(
                        candle,
                        displacement,
                        mapped_reason=mapped_reason,
                    )
                output = candidate._apply_boundary(
                    mapped_reason,
                    candle.end,
                )
                candidate._last_input_kind = "candle_boundary"
            else:
                candidate._validate_candle(
                    candle,
                    displacement,
                    bos_sources,
                )
                output = candidate._apply_completed_5m(
                    candle,
                    displacement,
                    bos_sources,
                )
                candidate._last_input_kind = "candle"
            candidate._last_candle_input = candle_input
            candidate._last_boundary_input = None
            candidate._last_output = output
        except Exception:
            self._failed = True
            raise
        self._commit(candidate)
        return output


__all__ = [
    "CausalZoneTracker",
    "FVG_BOUNDARY_REASONS",
    "ZoneBOSSource",
    "ZoneProtocol",
    "ZoneUpdate",
    "ORDER_BLOCK_BOUNDARY_REASONS",
    "WINDOW_RESET_REASONS",
]
