"""Causal 5m FVG and order-block primitives derived from frozen sources."""
from __future__ import annotations

from collections import deque
from copy import copy
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
from .semantic_zones import (
    BaseOriginCore,
    CompatibleStructureKind,
    CompletedZoneBar,
    FVGAvailability,
    FVGStructuralLifecycle,
    FVGTerminationCause,
    QualifiedOrderBlock,
    ZoneFirstReinteractionTracker,
    ZoneFirstRetest,
    ZoneFirstRetestSpec,
    ZoneObjectKind,
    bind_fvg_structural_context,
    qualify_order_block,
    reduce_fvg_termination,
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
FOUNDATION_FVG_CENSOR_REASONS = frozenset(
    {
        "data_gap_reset",
        "data_anomaly",
        "tick_size_mismatch",
        "synthetic_interruption",
    }
)
FOUNDATION_FVG_EXPIRY_REASONS = frozenset(
    {
        "contract_change_reset",
        "semantic_reset",
    }
)


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
    base_origin_cores: tuple[BaseOriginCore, ...] = ()
    qualified_order_blocks: tuple[QualifiedOrderBlock, ...] = ()
    first_retests: tuple[ZoneFirstRetest, ...] = ()
    fvg_structural_lifecycles: tuple[FVGStructuralLifecycle, ...] = ()
    first_retest_transitions: tuple[ZoneFirstRetest, ...] = ()
    fvg_structural_transitions: tuple[FVGStructuralLifecycle, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "fair_value_gaps",
            "order_blocks",
            "fvg_transitions",
            "order_block_transitions",
            "order_block_funnel",
            "base_origin_cores",
            "qualified_order_blocks",
            "first_retests",
            "fvg_structural_lifecycles",
            "first_retest_transitions",
            "fvg_structural_transitions",
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
        typed_foundation = (
            (self.base_origin_cores, BaseOriginCore),
            (self.qualified_order_blocks, QualifiedOrderBlock),
            (self.first_retests, ZoneFirstRetest),
            (self.fvg_structural_lifecycles, FVGStructuralLifecycle),
            (self.first_retest_transitions, ZoneFirstRetest),
            (self.fvg_structural_transitions, FVGStructuralLifecycle),
        )
        if any(
            not isinstance(item, expected)
            for collection, expected in typed_foundation
            for item in collection
        ):
            raise TypeError("Group 3 foundation projection is not typed")
        identity_sets = (
            tuple(item.core_id for item in self.base_origin_cores),
            tuple(
                item.qualified_ob_id
                for item in self.qualified_order_blocks
            ),
            tuple(item.object_id for item in self.first_retests),
            tuple(
                item.fvg_id for item in self.fvg_structural_lifecycles
            ),
        )
        if any(len(values) != len(set(values)) for values in identity_sets):
            raise ValueError("Group 3 foundation projection repeats identity")
        retests_by_id = {
            item.first_retest_event_id: item for item in self.first_retests
        }
        fvg_lifecycles_by_id = {
            item.fvg_id: item for item in self.fvg_structural_lifecycles
        }
        if any(
            retests_by_id.get(item.first_retest_event_id) != item
            for item in self.first_retest_transitions
        ) or any(
            fvg_lifecycles_by_id.get(item.fvg_id) != item
            for item in self.fvg_structural_transitions
        ):
            raise ValueError(
                "Group 3 foundation transition is absent from snapshot"
            )
        expected_foundation_disposition = (
            FVGAvailability.CENSORED
            if self.boundary_reason in FOUNDATION_FVG_CENSOR_REASONS
            else FVGAvailability.EXPIRED
            if self.boundary_reason in FOUNDATION_FVG_EXPIRY_REASONS
            else None
        )
        if self.boundary_reason is not None and (
            (
                expected_foundation_disposition is None
                and self.fvg_structural_transitions
            )
            or any(
                item.availability is not expected_foundation_disposition
                for item in self.fvg_structural_transitions
            )
        ):
            raise ValueError(
                "Group 3 foundation boundary disposition is inconsistent"
            )


@dataclass(frozen=True)
class _FrozenOrderBlockCandidate:
    candle: Candle
    candle_id: str
    cluster: tuple[Candle, ...]
    cluster_ids: tuple[str, ...]
    source_displacement_state: DisplacementState
    source_displacement_transition_identity: str


@dataclass(frozen=True)
class _QualifiedOrderBlockSeed:
    legacy_state: OrderBlockState
    candidate: _FrozenOrderBlockCandidate
    compatible_structure_entity_id: str
    compatible_structure_kind: CompatibleStructureKind


@dataclass(frozen=True)
class _FoundationCompletedSeed:
    candle: Candle
    candle_id: str
    new_base_origin_candidates: tuple[_FrozenOrderBlockCandidate, ...]
    new_qualified_order_blocks: tuple[_QualifiedOrderBlockSeed, ...]
    new_fvgs: tuple[FairValueGapState, ...]
    price_invalidated_fvg_ids: tuple[str, ...]
    terminal_order_block_ids: tuple[str, ...]


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


@dataclass(frozen=True)
class ZoneRawOnlyStructureDisposition:
    """Producer proof that one continuation BOS intentionally stayed raw."""

    bos_id: str
    raw_break_event_id: str
    protected_assignment_event_id: str
    timeframe: Timeframe
    direction: Direction
    resolved_at: pd.Timestamp
    source_structure_id: str
    target_swing_id: str
    break_bar_id: str
    bos_source_displacement_id: str | None

    def __post_init__(self) -> None:
        if (
            any(
                not isinstance(value, str) or not value
                for value in (
                    self.bos_id,
                    self.raw_break_event_id,
                    self.protected_assignment_event_id,
                    self.source_structure_id,
                    self.target_swing_id,
                    self.break_bar_id,
                )
            )
            or (
                self.bos_source_displacement_id is not None
                and (
                    not isinstance(self.bos_source_displacement_id, str)
                    or not self.bos_source_displacement_id
                )
            )
            or self.timeframe is not Timeframe.M5
            or self.direction not in {Direction.LONG, Direction.SHORT}
        ):
            raise ValueError("invalid raw-only Group 3 structure disposition")
        object.__setattr__(
            self,
            "resolved_at",
            aware_timestamp(
                self.resolved_at,
                name="group3.raw_only_structure.resolved_at",
            ),
        )


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
        self._base_origin_cores: dict[str, BaseOriginCore] = {}
        self._qualified_order_blocks: dict[
            str,
            QualifiedOrderBlock,
        ] = {}
        self._fvg_structural_lifecycles: dict[
            str,
            FVGStructuralLifecycle,
        ] = {}
        self._zone_reinteraction_trackers: dict[
            str,
            ZoneFirstReinteractionTracker,
        ] = {}
        self._pending_foundation_completed: list[
            _FoundationCompletedSeed
        ] = []
        self._pending_foundation_boundary: tuple[
            str,
            pd.Timestamp,
        ] | None = None
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
            str | None,
        ] | None = None
        self._last_candle_input: tuple[object, ...] | None = None
        self._last_output: ZoneUpdate | None = None
        self._failed = False

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
        candidate = copy(self)
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
        candidate._base_origin_cores = dict(self._base_origin_cores)
        candidate._qualified_order_blocks = dict(
            self._qualified_order_blocks
        )
        candidate._fvg_structural_lifecycles = dict(
            self._fvg_structural_lifecycles
        )
        candidate._zone_reinteraction_trackers = dict(
            self._zone_reinteraction_trackers
        )
        candidate._pending_foundation_completed = list(
            self._pending_foundation_completed
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
        """Return the compatible legacy + foundation-v2 projection."""

        return self._update()

    def bind_fvg_foundation_context(
        self,
        *,
        fvg_id: str,
        parent_structure_generation_id: str,
        structural_range_id: str | None,
        source_event_ids: Sequence[str],
    ) -> ZoneUpdate:
        """Bind one FVG to structure visible at its creation clock."""

        if self._failed:
            raise RuntimeError("Group 3 tracker is terminally failed")
        candidate = self._transaction_clone()
        try:
            state = candidate._fvg_structural_lifecycles[fvg_id]
            candidate._fvg_structural_lifecycles[fvg_id] = (
                bind_fvg_structural_context(
                    state,
                    parent_structure_generation_id=(
                        parent_structure_generation_id
                    ),
                    structural_range_id=structural_range_id,
                    source_event_ids=source_event_ids,
                )
            )
            output = candidate._update()
            candidate._last_output = output
        except Exception:
            self._failed = True
            raise
        self._commit(candidate)
        return output

    def expire_fvg_foundation_context(
        self,
        *,
        cause: FVGTerminationCause,
        related_entity_id: str,
        known_at: pd.Timestamp,
        cause_event_id: str,
    ) -> ZoneUpdate:
        """Expire every live FVG owned by one terminated structural object."""

        cause = FVGTerminationCause(cause)
        if cause not in {
            FVGTerminationCause.PARENT_STRUCTURE_TERMINATED,
            FVGTerminationCause.STRUCTURAL_RANGE_REPLACED,
        }:
            raise ValueError("FVG contextual expiry cause is not structural")
        if (
            not isinstance(related_entity_id, str)
            or not related_entity_id
            or not isinstance(cause_event_id, str)
            or not cause_event_id
        ):
            raise ValueError("FVG contextual expiry identity is invalid")
        clock = aware_timestamp(
            known_at,
            name="FVG contextual expiry known_at",
        )
        candidate = self._transaction_clone()
        try:
            transitions: list[FVGStructuralLifecycle] = []
            for fvg_id, state in tuple(
                candidate._fvg_structural_lifecycles.items()
            ):
                owner = (
                    state.parent_structure_generation_id
                    if cause
                    is FVGTerminationCause.PARENT_STRUCTURE_TERMINATED
                    else state.structural_range_id
                )
                if (
                    owner != related_entity_id
                    or state.availability is not FVGAvailability.ACTIVE
                ):
                    continue
                terminal = candidate._terminate_fvg_foundation(
                    entity_id=fvg_id,
                    cause=cause,
                    known_at=clock,
                    cause_event_id=cause_event_id,
                    terminal_event_id=cause_event_id,
                    related_entity_id=related_entity_id,
                )
                if terminal is not None:
                    transitions.append(terminal)
            output = candidate._update(
                fvg_structural_transitions=transitions,
            )
            candidate._last_output = output
        except Exception:
            self._failed = True
            raise
        self._commit(candidate)
        return output

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

    def _foundation_snapshots(
        self,
    ) -> tuple[
        tuple[BaseOriginCore, ...],
        tuple[QualifiedOrderBlock, ...],
        tuple[ZoneFirstRetest, ...],
        tuple[FVGStructuralLifecycle, ...],
    ]:
        base_origin_cores = tuple(
            self._base_origin_cores[displacement_id]
            for displacement_id in self._base_origin_cores
        )
        qualified_order_blocks = tuple(
            self._qualified_order_blocks[entity_id]
            for entity_id in self._order_block_order
            if entity_id in self._qualified_order_blocks
        )
        first_retests = tuple(
            tracker.first_retest
            for tracker in self._zone_reinteraction_trackers.values()
            if tracker.first_retest is not None
        )
        fvg_structural_lifecycles = tuple(
            self._fvg_structural_lifecycles[entity_id]
            for entity_id in self._fvg_order
            if entity_id in self._fvg_structural_lifecycles
        )
        return (
            base_origin_cores,
            qualified_order_blocks,
            first_retests,
            fvg_structural_lifecycles,
        )

    def _update(
        self,
        fvg_transitions: Iterable[FairValueGapState] = (),
        order_block_transitions: Iterable[OrderBlockState] = (),
        order_block_funnel: Iterable[OrderBlockFunnelSnapshot] = (),
        first_retest_transitions: Iterable[ZoneFirstRetest] = (),
        fvg_structural_transitions: Iterable[
            FVGStructuralLifecycle
        ] = (),
        *,
        boundary_reason: str | None = None,
    ) -> ZoneUpdate:
        fair_value_gaps, order_blocks = self._snapshots()
        (
            base_origin_cores,
            qualified_order_blocks,
            first_retests,
            fvg_structural_lifecycles,
        ) = self._foundation_snapshots()
        return ZoneUpdate(
            fair_value_gaps=fair_value_gaps,
            order_blocks=order_blocks,
            fvg_transitions=tuple(fvg_transitions),
            order_block_transitions=tuple(order_block_transitions),
            order_block_funnel=tuple(order_block_funnel),
            boundary_reason=boundary_reason,
            base_origin_cores=base_origin_cores,
            qualified_order_blocks=qualified_order_blocks,
            first_retests=first_retests,
            fvg_structural_lifecycles=fvg_structural_lifecycles,
            first_retest_transitions=tuple(first_retest_transitions),
            fvg_structural_transitions=tuple(
                fvg_structural_transitions
            ),
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

    @staticmethod
    def _zone_tracker_key(
        object_kind: ZoneObjectKind,
        legacy_entity_id: str,
    ) -> str:
        return f"{object_kind.value}:{legacy_entity_id}"

    @staticmethod
    def _bar_source_id(
        candle_id: str,
        bar_event_ids_by_candle_id: Mapping[str, str],
    ) -> str:
        try:
            event_id = bar_event_ids_by_candle_id[candle_id]
        except KeyError as error:
            raise ValueError(
                "foundation BAR source is not canonically bound"
            ) from error
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("foundation BAR source identity is invalid")
        return event_id

    def _prune_foundation_companions(self) -> None:
        retained_fvg_ids = set(self._fair_value_gaps)
        retained_order_block_ids = set(self._order_blocks)
        self._fvg_structural_lifecycles = {
            entity_id: state
            for entity_id, state in self._fvg_structural_lifecycles.items()
            if entity_id in retained_fvg_ids
        }
        self._qualified_order_blocks = {
            entity_id: state
            for entity_id, state in self._qualified_order_blocks.items()
            if entity_id in retained_order_block_ids
        }
        retained_tracker_keys = {
            self._zone_tracker_key(ZoneObjectKind.FVG, entity_id)
            for entity_id in retained_fvg_ids
        } | {
            self._zone_tracker_key(
                ZoneObjectKind.QUALIFIED_ORDER_BLOCK,
                entity_id,
            )
            for entity_id in retained_order_block_ids
        }
        self._zone_reinteraction_trackers = {
            key: tracker
            for key, tracker in self._zone_reinteraction_trackers.items()
            if key in retained_tracker_keys
        }
        # Base Origin is a frozen observation at displacement STARTED, not a
        # live qualification candidate.  In particular, exhaustion without a
        # compatible Q-BOS/MSS must not hindsight-delete the unqualified core.

    def _register_fvg_foundation(
        self,
        state: FairValueGapState,
        *,
        creation_event_id: str,
        departure_event_id: str,
    ) -> None:
        lifecycle = FVGStructuralLifecycle(
            fvg_id=state.fvg_id,
            source_creation_event_id=creation_event_id,
            symbol=state.symbol,
            instrument_id=state.instrument_id,
            timeframe=state.timeframe,
            created_at=state.formed_at,
            known_at=state.confirmed_at,
        )
        spec = ZoneFirstRetestSpec(
            object_kind=ZoneObjectKind.FVG,
            object_id=state.fvg_id,
            creation_event_id=creation_event_id,
            symbol=state.symbol,
            instrument_id=state.instrument_id,
            timeframe=state.timeframe,
            direction=state.direction,
            lower_bound=state.lower_bound,
            upper_bound=state.upper_bound,
            tick_size=self.protocol.tick_size,
            object_created_at=state.formed_at,
            object_known_at=state.confirmed_at,
            departure_confirmed_at=state.confirmed_at,
            departure_source_event_id=departure_event_id,
            creation_declared_departed=True,
        )
        self._fvg_structural_lifecycles[state.fvg_id] = lifecycle
        self._zone_reinteraction_trackers[
            self._zone_tracker_key(ZoneObjectKind.FVG, state.fvg_id)
        ] = ZoneFirstReinteractionTracker(spec)

    def _register_order_block_foundation(
        self,
        *,
        legacy_state: OrderBlockState,
        qualified: QualifiedOrderBlock,
        candle: Candle,
        creation_event_id: str,
        departure_event_id: str,
    ) -> None:
        self._qualified_order_blocks[
            legacy_state.order_block_id
        ] = qualified
        departed = (
            self._ticks(candle.close)
            > self._ticks(legacy_state.upper_bound)
            if legacy_state.direction is Direction.LONG
            else self._ticks(candle.close)
            < self._ticks(legacy_state.lower_bound)
        )
        spec = ZoneFirstRetestSpec(
            object_kind=ZoneObjectKind.QUALIFIED_ORDER_BLOCK,
            object_id=qualified.qualified_ob_id,
            creation_event_id=creation_event_id,
            symbol=legacy_state.symbol,
            instrument_id=legacy_state.instrument_id,
            timeframe=legacy_state.timeframe,
            direction=legacy_state.direction,
            lower_bound=legacy_state.lower_bound,
            upper_bound=legacy_state.upper_bound,
            tick_size=self.protocol.tick_size,
            object_created_at=legacy_state.formed_at,
            object_known_at=legacy_state.confirmed_at,
            departure_confirmed_at=(
                legacy_state.confirmed_at if departed else None
            ),
            departure_source_event_id=(
                departure_event_id if departed else None
            ),
            creation_declared_departed=departed,
        )
        self._zone_reinteraction_trackers[
            self._zone_tracker_key(
                ZoneObjectKind.QUALIFIED_ORDER_BLOCK,
                legacy_state.order_block_id,
            )
        ] = ZoneFirstReinteractionTracker(spec)

    def _drop_unresolved_order_block_reinteraction(
        self,
        entity_id: str,
    ) -> None:
        tracker_key = self._zone_tracker_key(
            ZoneObjectKind.QUALIFIED_ORDER_BLOCK,
            entity_id,
        )
        tracker = self._zone_reinteraction_trackers.get(tracker_key)
        if tracker is not None and tracker.first_retest is None:
            self._zone_reinteraction_trackers.pop(tracker_key)

    def _advance_first_reinteractions(
        self,
        candle: Candle,
        *,
        candle_id: str,
        bar_event_ids_by_candle_id: Mapping[str, str],
        session: str,
        context_event_ids: Sequence[str],
    ) -> list[ZoneFirstRetest]:
        source = CompletedZoneBar(
            bar_event_id=self._bar_source_id(
                candle_id,
                bar_event_ids_by_candle_id,
            ),
            symbol=candle.symbol,
            instrument_id=candle.instrument_id,
            timeframe=candle.timeframe,
            known_at=candle.end,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            session=session,
            context_event_ids=tuple(context_event_ids),
        )
        transitions: list[ZoneFirstRetest] = []
        for key, tracker in tuple(
            self._zone_reinteraction_trackers.items()
        ):
            prior = tracker.first_retest
            updated = tracker.on_completed_bar(source)
            self._zone_reinteraction_trackers[key] = updated
            if prior is None and updated.first_retest is not None:
                transitions.append(updated.first_retest)
        return transitions

    def _terminate_fvg_foundation(
        self,
        *,
        entity_id: str,
        cause: FVGTerminationCause,
        known_at: pd.Timestamp,
        cause_event_id: str,
        terminal_event_id: str,
        related_entity_id: str | None = None,
    ) -> FVGStructuralLifecycle | None:
        state = self._fvg_structural_lifecycles.get(entity_id)
        if state is None or state.availability is not FVGAvailability.ACTIVE:
            return None
        terminal = reduce_fvg_termination(
            state,
            cause=cause,
            known_at=known_at,
            cause_event_ids=(cause_event_id,),
            related_entity_id=related_entity_id,
        )
        terminal_sources = tuple(
            dict.fromkeys(
                (*terminal.terminal_source_event_ids, terminal_event_id)
            )
        )
        terminal = replace(
            terminal,
            terminal_event_id=terminal_event_id,
            terminal_source_event_ids=terminal_sources,
        )
        self._fvg_structural_lifecycles[entity_id] = terminal
        tracker_key = self._zone_tracker_key(
            ZoneObjectKind.FVG,
            entity_id,
        )
        tracker = self._zone_reinteraction_trackers.get(tracker_key)
        if tracker is not None and tracker.first_retest is None:
            self._zone_reinteraction_trackers.pop(tracker_key)
        return terminal

    def _advance_fvg_foundation_age(
        self,
        candle: Candle,
        *,
        bar_event_id: str,
    ) -> None:
        """Advance descriptive age from one exact real native BAR.

        This state never expires an FVG.  Elapsed BARs are temporal
        observations rather than definitional ancestry, so only the
        continuous count/elapsed time are retained here.  The caller still
        supplies the exact canonical BAR identity to prevent an unbound
        runtime heartbeat from advancing the clock.
        """

        if not candle.real_completed or candle.timeframe is not Timeframe.M5:
            return
        if not isinstance(bar_event_id, str) or not bar_event_id:
            raise ValueError("FVG age update lacks its exact BAR identity")
        for fvg_id, state in tuple(
            self._fvg_structural_lifecycles.items()
        ):
            if (
                state.availability is not FVGAvailability.ACTIVE
                or candle.end <= state.known_at
            ):
                continue
            if candle.end <= state.last_updated_at:
                raise ValueError("FVG age observations must be strictly ordered")
            self._fvg_structural_lifecycles[fvg_id] = replace(
                state,
                age_bars=state.age_bars + 1,
                age_seconds=int(
                    (candle.end - state.known_at).total_seconds()
                ),
                last_updated_at=candle.end,
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
        self._pending_foundation_completed.clear()
        self._prune_foundation_companions()
        if clear_identity:
            self._identity = None

    def _apply_foundation_boundary(
        self,
        reason: str,
        clock: pd.Timestamp,
        *,
        boundary_event_id: str | None,
    ) -> list[FVGStructuralLifecycle]:
        if reason in FOUNDATION_FVG_CENSOR_REASONS:
            cause = FVGTerminationCause.DATA_GAP
        elif reason == "contract_change_reset":
            cause = FVGTerminationCause.CONTRACT_ROLLOVER
        elif reason == "semantic_reset":
            cause = FVGTerminationCause.SEMANTIC_RESET
        else:
            return []
        live_foundation = any(
            state.availability is FVGAvailability.ACTIVE
            for state in self._fvg_structural_lifecycles.values()
        )
        if live_foundation and boundary_event_id is None:
            pending = (reason, clock)
            if (
                self._pending_foundation_boundary is not None
                and self._pending_foundation_boundary != pending
            ):
                raise ValueError(
                    "foundation FVG boundary binding is already pending"
                )
            self._pending_foundation_boundary = pending
            return []
        if boundary_event_id is not None and (
            not isinstance(boundary_event_id, str)
            or not boundary_event_id
        ):
            raise ValueError(
                "foundation FVG boundary provenance is invalid"
            )
        if boundary_event_id is None:
            return []
        transitions: list[FVGStructuralLifecycle] = []
        for entity_id in tuple(self._fvg_structural_lifecycles):
            terminal = self._terminate_fvg_foundation(
                entity_id=entity_id,
                cause=cause,
                known_at=clock,
                cause_event_id=boundary_event_id,
                terminal_event_id=boundary_event_id,
            )
            if terminal is not None:
                transitions.append(terminal)
        self._pending_foundation_boundary = None
        return transitions

    def _apply_boundary(
        self,
        reason: str,
        clock: pd.Timestamp,
        *,
        boundary_event_id: str | None,
    ) -> ZoneUpdate:
        fvg_transitions: list[FairValueGapState] = []
        order_block_transitions: list[OrderBlockState] = []
        fvg_structural_transitions = self._apply_foundation_boundary(
            reason,
            clock,
            boundary_event_id=boundary_event_id,
        )
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
                self._drop_unresolved_order_block_reinteraction(entity_id)
        elif reason == "synthetic_interruption":
            for entity_id in tuple(self._qualified_order_blocks):
                self._drop_unresolved_order_block_reinteraction(entity_id)
        self._clear_windows(clear_identity=hard_boundary)
        self._window_epoch_known = True
        self._last_clock = clock
        output = self._update(
            fvg_transitions,
            order_block_transitions,
            fvg_structural_transitions=fvg_structural_transitions,
            boundary_reason=reason,
        )
        self._mark_terminals_exposed()
        return output

    def on_boundary(
        self,
        reason: str,
        observed_at: pd.Timestamp,
        *,
        foundation_boundary_event_id: str | None = None,
    ) -> ZoneUpdate:
        if self._failed:
            raise RuntimeError("Group 3 tracker is terminally failed")
        clock = aware_timestamp(observed_at, name="group3.boundary")
        boundary_input = (
            reason,
            clock,
            foundation_boundary_event_id,
        )
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
            output = candidate._apply_boundary(
                reason,
                clock,
                boundary_event_id=foundation_boundary_event_id,
            )
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
        new_base_origin_candidate = self._freeze_new_displacement_sources(
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
        qualified_seed: _QualifiedOrderBlockSeed | None = None
        if created_order_block is not None:
            candidate = self._ob_candidates.get(
                created_order_block.source_displacement_id
            )
            if candidate is None:
                raise RuntimeError(
                    "created OB lost its frozen Base Origin candidate"
                )
            qualified_seed = _QualifiedOrderBlockSeed(
                legacy_state=created_order_block,
                candidate=candidate,
                compatible_structure_entity_id=(
                    created_order_block.source_bos_id
                ),
                compatible_structure_kind=(
                    CompatibleStructureKind.MSS_CORE_CONFIRMED
                    if created_order_block.source_bos_scope
                    is BOSScope.OPPOSED
                    else CompatibleStructureKind.QUALIFIED_BOS
                ),
            )
        price_invalidated_fvg_ids = tuple(
            state.fvg_id
            for state in fvg_transitions
            if (
                state.lifecycle is FairValueGapLifecycle.INVALIDATED
                and state.transition_reason == "close_through_far_edge"
            )
        )
        terminal_order_block_ids = tuple(
            state.order_block_id
            for state in order_block_transitions
            if self._is_order_block_terminal(state)
        )
        if (
            new_base_origin_candidate is not None
            or qualified_seed is not None
            or created_fvg is not None
            or price_invalidated_fvg_ids
            or self._zone_reinteraction_trackers
        ):
            self._pending_foundation_completed.append(
                _FoundationCompletedSeed(
                    candle=candle,
                    candle_id=candle_id,
                    new_base_origin_candidates=(
                        ()
                        if new_base_origin_candidate is None
                        else (new_base_origin_candidate,)
                    ),
                    new_qualified_order_blocks=(
                        () if qualified_seed is None else (qualified_seed,)
                    ),
                    new_fvgs=(
                        () if created_fvg is None else (created_fvg,)
                    ),
                    price_invalidated_fvg_ids=(
                        price_invalidated_fvg_ids
                    ),
                    terminal_order_block_ids=(
                        terminal_order_block_ids
                    ),
                )
            )

        self._last_clock = candle.end
        output = self._update(
            fvg_transitions,
            order_block_transitions,
            (order_block_funnel,),
        )
        self._mark_terminals_exposed()
        return output

    def _base_origin_from_candidate(
        self,
        candidate: _FrozenOrderBlockCandidate,
        *,
        bar_event_ids_by_candle_id: Mapping[str, str],
        displacement_event_ids_by_identity: Mapping[str, str],
        displacement_event_known_at_by_identity: Mapping[
            str, pd.Timestamp
        ],
    ) -> BaseOriginCore:
        state = candidate.source_displacement_state
        try:
            displacement_event_id = (
                displacement_event_ids_by_identity[
                    candidate.source_displacement_transition_identity
                ]
            )
        except KeyError as error:
            raise ValueError(
                "Base Origin Core lacks canonical displacement provenance"
            ) from error
        try:
            displacement_known_at = aware_timestamp(
                displacement_event_known_at_by_identity[
                    candidate.source_displacement_transition_identity
                ],
                name="Base Origin displacement event known_at",
            )
        except KeyError as error:
            raise ValueError(
                "Base Origin Core lacks its displacement knowledge clock"
            ) from error
        cluster = candidate.cluster
        return BaseOriginCore(
            symbol=state.symbol,
            instrument_id=state.instrument_id,
            timeframe=state.timeframe,
            direction=state.direction,
            source_displacement_id=state.entity_id,
            source_displacement_event_id=displacement_event_id,
            anchor_bar_event_ids=tuple(
                self._bar_source_id(
                    candle_id,
                    bar_event_ids_by_candle_id,
                )
                for candle_id in candidate.cluster_ids
            ),
            anchor_candle_ids=candidate.cluster_ids,
            anchor_completed_at=tuple(item.end for item in cluster),
            lower_bound=min(float(item.low) for item in cluster),
            upper_bound=max(float(item.high) for item in cluster),
            body_lower_bound=min(
                min(float(item.open), float(item.close))
                for item in cluster
            ),
            body_upper_bound=max(
                max(float(item.open), float(item.close))
                for item in cluster
            ),
            tick_size=self.protocol.tick_size,
            formed_at=cluster[-1].end,
            # Detector lifecycle ``started_at`` is event time.  The core is
            # knowable only after the exact canonical displacement fact has
            # been appended, which can be a later completed-bar clock.
            known_at=displacement_known_at,
        )

    @staticmethod
    def _required_event_id(
        mapping: Mapping[str, str],
        identity: str,
        *,
        name: str,
    ) -> str:
        try:
            event_id = mapping[identity]
        except KeyError as error:
            raise ValueError(f"{name} is not canonically bound") from error
        if not isinstance(event_id, str) or not event_id:
            raise ValueError(f"{name} canonical event identity is invalid")
        return event_id

    def _apply_finalize_foundation(
        self,
        update: ZoneUpdate,
        *,
        bar_event_ids_by_candle_id: Mapping[str, str],
        displacement_event_ids_by_identity: Mapping[str, str],
        displacement_event_known_at_by_identity: Mapping[
            str, pd.Timestamp
        ],
        structure_event_ids_by_entity: Mapping[str, str],
        raw_only_structure_dispositions: Iterable[
            ZoneRawOnlyStructureDisposition
        ],
        fvg_creation_event_ids_by_entity: Mapping[str, str],
        fvg_terminal_event_ids_by_entity: Mapping[str, str],
        order_block_creation_event_ids_by_entity: Mapping[str, str],
        sessions_by_clock: Mapping[pd.Timestamp, str],
        context_event_ids_by_clock: Mapping[
            pd.Timestamp,
            Sequence[str],
        ],
    ) -> ZoneUpdate:
        if isinstance(raw_only_structure_dispositions, (str, bytes)):
            raise ValueError(
                "raw-only Group 3 structure dispositions must be a sequence"
            )
        raw_only_disposition_order = tuple(
            raw_only_structure_dispositions
        )
        if (
            any(
                not isinstance(
                    disposition,
                    ZoneRawOnlyStructureDisposition,
                )
                for disposition in raw_only_disposition_order
            )
        ):
            raise ValueError(
                "raw-only Group 3 structure dispositions are invalid"
            )
        raw_only_by_id = {
            disposition.bos_id: disposition
            for disposition in raw_only_disposition_order
        }
        if len(raw_only_by_id) != len(raw_only_disposition_order):
            raise ValueError(
                "raw-only Group 3 structure disposition is ambiguous"
            )
        raw_only_ids = frozenset(raw_only_by_id)
        if raw_only_ids.intersection(structure_event_ids_by_entity):
            raise ValueError(
                "raw-only Group 3 structure identity conflicts with a "
                "qualified structure binding"
            )
        pending_raw_only_seeds: dict[
            str,
            tuple[_FoundationCompletedSeed, _QualifiedOrderBlockSeed],
        ] = {}
        for completed in self._pending_foundation_completed:
            for seed in completed.new_qualified_order_blocks:
                identity = seed.compatible_structure_entity_id
                if identity not in raw_only_ids:
                    continue
                if identity in pending_raw_only_seeds:
                    raise ValueError(
                        "raw-only Group 3 structure identity is ambiguous"
                    )
                state = seed.legacy_state
                candidate = seed.candidate
                disposition = raw_only_by_id[identity]
                if (
                    seed.compatible_structure_kind
                    is not CompatibleStructureKind.QUALIFIED_BOS
                    or state.source_bos_id != identity
                    or state.source_bos_scope is not BOSScope.CONTINUATION
                    or state.source_bos_mss_qualified
                    or state.timeframe is not Timeframe.M5
                    or state.direction
                    is not candidate.source_displacement_state.direction
                    or state.source_displacement_id
                    != candidate.source_displacement_state.entity_id
                    or not state.source_bos_structure_id
                    or not state.source_bos_target_swing_id
                    or state.source_bos_resolved_at != state.confirmed_at
                    or state.confirmed_at != completed.candle.end
                    or state.source_bos_break_bar_id
                    != completed.candle_id
                    or disposition.timeframe is not state.timeframe
                    or disposition.direction is not state.direction
                    or disposition.resolved_at
                    != state.source_bos_resolved_at
                    or disposition.source_structure_id
                    != state.source_bos_structure_id
                    or disposition.target_swing_id
                    != state.source_bos_target_swing_id
                    or disposition.break_bar_id
                    != state.source_bos_break_bar_id
                ):
                    raise ValueError(
                        "raw-only Group 3 structure disposition does not "
                        "match its exact provisional QOB seed"
                    )
                pending_raw_only_seeds[identity] = (completed, seed)
        if set(pending_raw_only_seeds) != set(raw_only_ids):
            raise ValueError(
                "raw-only Group 3 structure identity has no exact "
                "provisional QOB seed"
            )
        first_retest_transitions: list[ZoneFirstRetest] = list(
            update.first_retest_transitions
        )
        fvg_structural_transitions: list[
            FVGStructuralLifecycle
        ] = list(update.fvg_structural_transitions)
        for completed in self._pending_foundation_completed:
            candle = completed.candle
            current_bar_event_id = self._bar_source_id(
                completed.candle_id,
                bar_event_ids_by_candle_id,
            )
            self._advance_fvg_foundation_age(
                candle,
                bar_event_id=current_bar_event_id,
            )
            for candidate in completed.new_base_origin_candidates:
                core = self._base_origin_from_candidate(
                    candidate,
                    bar_event_ids_by_candle_id=(
                        bar_event_ids_by_candle_id
                    ),
                    displacement_event_ids_by_identity=(
                        displacement_event_ids_by_identity
                    ),
                    displacement_event_known_at_by_identity=(
                        displacement_event_known_at_by_identity
                    ),
                )
                existing = self._base_origin_cores.get(
                    core.source_displacement_id
                )
                if existing is not None and existing != core:
                    raise ValueError(
                        "Base Origin Core canonical rebinding drifted"
                    )
                self._base_origin_cores[
                    core.source_displacement_id
                ] = core

            try:
                session = sessions_by_clock[candle.end]
            except KeyError as error:
                raise ValueError(
                    "first reinteraction lacks its frozen session"
                ) from error
            context_event_ids = tuple(
                context_event_ids_by_clock.get(candle.end, ())
            )
            first_retest_transitions.extend(
                self._advance_first_reinteractions(
                    candle,
                    candle_id=completed.candle_id,
                    bar_event_ids_by_candle_id=(
                        bar_event_ids_by_candle_id
                    ),
                    session=session,
                    context_event_ids=context_event_ids,
                )
            )
            for fvg_id in completed.price_invalidated_fvg_ids:
                terminal_event_id = self._required_event_id(
                    fvg_terminal_event_ids_by_entity,
                    fvg_id,
                    name="FVG invalidation terminal",
                )
                terminal = self._terminate_fvg_foundation(
                    entity_id=fvg_id,
                    cause=FVGTerminationCause.CLOSE_THROUGH_FAR_EDGE,
                    known_at=candle.end,
                    cause_event_id=current_bar_event_id,
                    terminal_event_id=terminal_event_id,
                )
                if terminal is not None:
                    fvg_structural_transitions.append(terminal)
            for order_block_id in completed.terminal_order_block_ids:
                self._drop_unresolved_order_block_reinteraction(
                    order_block_id
                )

            for seed in completed.new_qualified_order_blocks:
                displacement_id = (
                    seed.legacy_state.source_displacement_id
                )
                core = self._base_origin_cores.get(displacement_id)
                if core is None:
                    core = self._base_origin_from_candidate(
                        seed.candidate,
                        bar_event_ids_by_candle_id=(
                            bar_event_ids_by_candle_id
                        ),
                        displacement_event_ids_by_identity=(
                            displacement_event_ids_by_identity
                        ),
                        displacement_event_known_at_by_identity=(
                            displacement_event_known_at_by_identity
                        ),
                    )
                    self._base_origin_cores[displacement_id] = core
                if seed.compatible_structure_entity_id in raw_only_ids:
                    # The observer has proven that the exact typed BOS ended
                    # at RAW_BOUNDARY_BREAK and deliberately did not become a
                    # canonical QUALIFIED_BOS/MSS_CORE fact.  Keep the legacy
                    # OB and displacement-backed BaseOriginCore, but do not
                    # invent a compatible structure source or a QOB/retest
                    # companion from direction alone.
                    continue
                compatible_event_id = self._required_event_id(
                    structure_event_ids_by_entity,
                    seed.compatible_structure_entity_id,
                    name="Qualified OB structure source",
                )
                qualified = qualify_order_block(
                    core,
                    source_displacement_id=displacement_id,
                    source_displacement_event_id=(
                        core.source_displacement_event_id
                    ),
                    compatible_structure_event_id=compatible_event_id,
                    compatible_structure_kind=(
                        seed.compatible_structure_kind
                    ),
                    qualified_at=seed.legacy_state.confirmed_at,
                    known_at=seed.legacy_state.confirmed_at,
                )
                creation_event_id = self._required_event_id(
                    order_block_creation_event_ids_by_entity,
                    seed.legacy_state.order_block_id,
                    name="Qualified OB creation source",
                )
                departure_event_id = self._required_event_id(
                    displacement_event_ids_by_identity,
                    seed.legacy_state.source_active_transition_id,
                    name="Qualified OB departure source",
                )
                self._register_order_block_foundation(
                    legacy_state=seed.legacy_state,
                    qualified=qualified,
                    candle=candle,
                    creation_event_id=creation_event_id,
                    departure_event_id=departure_event_id,
                )

            for state in completed.new_fvgs:
                creation_event_id = self._required_event_id(
                    fvg_creation_event_ids_by_entity,
                    state.fvg_id,
                    name="FVG creation source",
                )
                departure_event_id = (
                    self._required_event_id(
                        displacement_event_ids_by_identity,
                        state.source_active_transition_id,
                        name="FVG departure source",
                    )
                    if state.source_active_transition_id is not None
                    else self._bar_source_id(
                        state.source_candle_ids[-1],
                        bar_event_ids_by_candle_id,
                    )
                )
                self._register_fvg_foundation(
                    state,
                    creation_event_id=creation_event_id,
                    departure_event_id=departure_event_id,
                )
        self._pending_foundation_completed.clear()
        self._prune_foundation_companions()
        return self._update(
            update.fvg_transitions,
            update.order_block_transitions,
            update.order_block_funnel,
            first_retest_transitions,
            fvg_structural_transitions,
            boundary_reason=update.boundary_reason,
        )

    def finalize_foundation(
        self,
        update: ZoneUpdate,
        *,
        bar_event_ids_by_candle_id: Mapping[str, str],
        displacement_event_ids_by_identity: Mapping[str, str],
        displacement_event_known_at_by_identity: Mapping[
            str, pd.Timestamp
        ],
        structure_event_ids_by_entity: Mapping[str, str],
        raw_only_structure_dispositions: Iterable[
            ZoneRawOnlyStructureDisposition
        ] = (),
        fvg_creation_event_ids_by_entity: Mapping[str, str],
        fvg_terminal_event_ids_by_entity: Mapping[str, str],
        order_block_creation_event_ids_by_entity: Mapping[str, str],
        sessions_by_clock: Mapping[pd.Timestamp, str],
        context_event_ids_by_clock: Mapping[
            pd.Timestamp,
            Sequence[str],
        ] | None = None,
    ) -> ZoneUpdate:
        """Bind provisional reducer identities to canonical event facts."""

        if self._failed:
            raise RuntimeError("Group 3 tracker is terminally failed")
        if not isinstance(update, ZoneUpdate) or update.boundary_reason:
            raise ValueError(
                "foundation finalization requires an ordinary ZoneUpdate"
            )
        candidate = self._transaction_clone()
        try:
            output = candidate._apply_finalize_foundation(
                update,
                bar_event_ids_by_candle_id=bar_event_ids_by_candle_id,
                displacement_event_ids_by_identity=(
                    displacement_event_ids_by_identity
                ),
                displacement_event_known_at_by_identity=(
                    displacement_event_known_at_by_identity
                ),
                structure_event_ids_by_entity=(
                    structure_event_ids_by_entity
                ),
                raw_only_structure_dispositions=(
                    raw_only_structure_dispositions
                ),
                fvg_creation_event_ids_by_entity=(
                    fvg_creation_event_ids_by_entity
                ),
                fvg_terminal_event_ids_by_entity=(
                    fvg_terminal_event_ids_by_entity
                ),
                order_block_creation_event_ids_by_entity=(
                    order_block_creation_event_ids_by_entity
                ),
                sessions_by_clock=sessions_by_clock,
                context_event_ids_by_clock=(
                    {}
                    if context_event_ids_by_clock is None
                    else context_event_ids_by_clock
                ),
            )
            candidate._last_output = output
        except Exception:
            self._failed = True
            raise
        self._commit(candidate)
        return output

    def finalize_foundation_boundary(
        self,
        update: ZoneUpdate,
        *,
        boundary_event_id: str | None,
    ) -> ZoneUpdate:
        """Bind a deferred uncertain boundary to its canonical event fact."""

        if self._failed:
            raise RuntimeError("Group 3 tracker is terminally failed")
        if not isinstance(update, ZoneUpdate) or update.boundary_reason is None:
            raise ValueError(
                "foundation boundary finalization requires a boundary update"
            )
        pending = self._pending_foundation_boundary
        if pending is None:
            return update
        reason, clock = pending
        if reason != update.boundary_reason:
            self._failed = True
            raise ValueError("foundation boundary reason drifted")
        candidate = self._transaction_clone()
        try:
            transitions = candidate._apply_foundation_boundary(
                reason,
                clock,
                boundary_event_id=boundary_event_id,
            )
            if not transitions:
                raise ValueError(
                    "foundation boundary lacks a canonical terminal fact"
                )
            output = candidate._update(
                update.fvg_transitions,
                update.order_block_transitions,
                update.order_block_funnel,
                first_retest_transitions=(
                    update.first_retest_transitions
                ),
                fvg_structural_transitions=transitions,
                boundary_reason=reason,
            )
            candidate._last_output = output
        except Exception:
            self._failed = True
            raise
        self._commit(candidate)
        return output

    def on_completed_5m(
        self,
        candle: Candle,
        displacement: DisplacementUpdate,
        confirmed_bos: Iterable[ZoneBOSSource] = (),
    ) -> ZoneUpdate:
        if self._failed:
            raise RuntimeError("Group 3 tracker is terminally failed")
        if self._pending_foundation_boundary is not None:
            raise RuntimeError(
                "Group 3 foundation boundary requires canonical finalization"
            )
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
                    boundary_event_id=None,
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
