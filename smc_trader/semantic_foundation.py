"""Version-bound projection and cold ledger for foundation-v2 DTOs.

The hot :class:`FoundationProjection` retains only the deterministic current
view, record count, and rolling chain identity.  Immutable revision history
lives in :class:`FoundationRecordLedger` and is materialized only for an
explicit checkpoint or cold replay.  Neither layer detects market semantics.
"""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields as dataclass_fields
from enum import Enum
from functools import lru_cache
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

import pandas as pd

from .foundation_registry import (
    FOUNDATION_CANONICAL_IDENTITY,
    FOUNDATION_VERSION,
)
from .market_state import (
    BalanceRangeState,
    LiquidityClusterState,
    LiquidityClusterSupersession,
    StructuralRangeState,
    SwingGeometryAssignment,
    SwingGeometryNode,
)
from .model import (
    DealingRangeLifecycle,
    Direction,
    FrozenDict,
    StructuralLegState,
    SwingRank,
    Timeframe,
    aware_timestamp,
    to_primitive,
)
from .semantic_lifecycle import (
    BoundaryAttackFact,
    canonical_semantic_id,
    DeliveryPhaseGeneration,
    GenerationLifecycle,
    InteractionConstituent,
    LiquidityInteractionGeneration,
    LiquidityInteractionLifecycle,
    LiquidityLevelLifecycle,
    LiquidityLevelState,
    RelationGeneration,
    StructureGeneration,
    StructureGenerationLifecycle,
    StructureTransition,
    StructureTransitionLifecycle,
)
from .semantic_zones import (
    BaseOriginCore,
    FVGAvailability,
    FVGStructuralLifecycle,
    QualifiedOrderBlock,
    ZoneObjectKind,
    ZoneFirstRetest,
)


FOUNDATION_COMPONENT_FINGERPRINT_VERSION = (
    "foundation_projection_append_chain_v1"
)
FOUNDATION_CURRENT_VIEW_FINGERPRINT_VERSION = (
    "foundation_projection_current_view_v1"
)
FOUNDATION_PROJECTION_STATE_SCHEMA_VERSION = 2
FOUNDATION_PROJECTION_CHECKPOINT_SCHEMA_VERSION = 2
FOUNDATION_RECORD_LEDGER_SCHEMA_VERSION = 1
FOUNDATION_RECORD_DELTA_SCHEMA_VERSION = 2
FOUNDATION_PROJECTION_OWNER_STATE_SCHEMA_VERSION = 1

# These capabilities are call-local implementation authorities, never schema
# fields or serialized identities.  They keep trusted hot/cold construction
# paths distinct from public projection admission.
_FOUNDATION_INCREMENTAL_CAPABILITY = object()
_FOUNDATION_COLD_PREFIX_CAPABILITY = object()


class FoundationObjectType(str, Enum):
    LIQUIDITY_LEVEL = "liquidity_level"
    LIQUIDITY_INTERACTION_GENERATION = "liquidity_interaction_generation"
    STRUCTURE_GENERATION = "structure_generation"
    STRUCTURE_TRANSITION = "structure_transition"
    RELATION_GENERATION = "relation_generation"
    DELIVERY_PHASE_GENERATION = "delivery_phase_generation"
    BOUNDARY_ATTACK = "boundary_attack"
    STRUCTURAL_LEG = "structural_leg"
    SWING_GEOMETRY_NODE = "swing_geometry_node"
    SWING_GEOMETRY_ASSIGNMENT = "swing_geometry_assignment"
    LIQUIDITY_CLUSTER = "liquidity_cluster"
    LIQUIDITY_CLUSTER_SUPERSESSION = "liquidity_cluster_supersession"
    STRUCTURAL_RANGE = "structural_range"
    BALANCE_RANGE = "balance_range"
    BASE_ORIGIN_CORE = "base_origin_core"
    QUALIFIED_ORDER_BLOCK = "qualified_order_block"
    ZONE_FIRST_RETEST = "zone_first_retest"
    FVG_STRUCTURAL_LIFECYCLE = "fvg_structural_lifecycle"


class FoundationRecordStatus(str, Enum):
    FACT = "fact"
    ACTIVE = "active"
    INACTIVE = "inactive"
    TERMINAL = "terminal"


_FACT_TYPES = frozenset(
    {
        FoundationObjectType.BOUNDARY_ATTACK,
        FoundationObjectType.STRUCTURAL_LEG,
        FoundationObjectType.SWING_GEOMETRY_NODE,
        FoundationObjectType.SWING_GEOMETRY_ASSIGNMENT,
        FoundationObjectType.LIQUIDITY_CLUSTER_SUPERSESSION,
        FoundationObjectType.BASE_ORIGIN_CORE,
        FoundationObjectType.QUALIFIED_ORDER_BLOCK,
        FoundationObjectType.ZONE_FIRST_RETEST,
    }
)

_OBJECT_ID_FIELDS = {
    FoundationObjectType.LIQUIDITY_LEVEL: "level_id",
    FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION: "generation_id",
    FoundationObjectType.STRUCTURE_GENERATION: "structure_generation_id",
    FoundationObjectType.STRUCTURE_TRANSITION: "structure_transition_id",
    FoundationObjectType.RELATION_GENERATION: "relation_generation_id",
    FoundationObjectType.DELIVERY_PHASE_GENERATION: "delivery_generation_id",
    FoundationObjectType.BOUNDARY_ATTACK: "boundary_attack_id",
    FoundationObjectType.STRUCTURAL_LEG: "leg_id",
    FoundationObjectType.SWING_GEOMETRY_NODE: "swing_id",
    FoundationObjectType.SWING_GEOMETRY_ASSIGNMENT: "assignment_id",
    FoundationObjectType.LIQUIDITY_CLUSTER: "cluster_id",
    FoundationObjectType.STRUCTURAL_RANGE: "range_id",
    FoundationObjectType.BALANCE_RANGE: "range_id",
    FoundationObjectType.BASE_ORIGIN_CORE: "core_id",
    FoundationObjectType.QUALIFIED_ORDER_BLOCK: "qualified_ob_id",
    FoundationObjectType.ZONE_FIRST_RETEST: "first_retest_event_id",
    FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE: "fvg_id",
}

_VERSION_FIELDS = {
    FoundationObjectType.LIQUIDITY_LEVEL: "semantic_version",
    FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION: "semantic_version",
    FoundationObjectType.STRUCTURE_GENERATION: "semantic_version",
    FoundationObjectType.STRUCTURE_TRANSITION: "semantic_version",
    FoundationObjectType.RELATION_GENERATION: "semantic_version",
    FoundationObjectType.DELIVERY_PHASE_GENERATION: "semantic_version",
    FoundationObjectType.BOUNDARY_ATTACK: "semantic_version",
    FoundationObjectType.STRUCTURAL_LEG: "foundation_version",
    FoundationObjectType.SWING_GEOMETRY_NODE: "foundation_version",
    FoundationObjectType.SWING_GEOMETRY_ASSIGNMENT: "foundation_version",
    FoundationObjectType.LIQUIDITY_CLUSTER: "foundation_version",
    FoundationObjectType.LIQUIDITY_CLUSTER_SUPERSESSION: "foundation_version",
    FoundationObjectType.STRUCTURAL_RANGE: "foundation_version",
    FoundationObjectType.BASE_ORIGIN_CORE: "semantic_version",
    FoundationObjectType.QUALIFIED_ORDER_BLOCK: "semantic_version",
    FoundationObjectType.ZONE_FIRST_RETEST: "semantic_version",
    FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE: "semantic_version",
}


def _identities(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of identities")
    result = tuple(values)
    if (
        not result
        or len(result) != len(set(result))
        or any(not isinstance(value, str) or not value.strip() for value in result)
    ):
        raise ValueError(f"{name} must contain unique non-empty identities")
    return result


def _freeze_primitive(value: Any, *, path: str = "payload") -> Any:
    """Validate and deeply freeze a strict JSON-primitive value tree."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            frozen[key] = _freeze_primitive(item, path=f"{path}.{key}")
        return FrozenDict(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_primitive(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} must contain only JSON primitives, lists, and mappings"
    )


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _foundation_component_fingerprint_seed(
    *,
    foundation_version: str,
    registry_identity: str,
) -> str:
    return _canonical_digest(
        {
            "component_fingerprint_version": (
                FOUNDATION_COMPONENT_FINGERPRINT_VERSION
            ),
            "foundation_version": foundation_version,
            "registry_identity": registry_identity,
            "record_chain": "empty",
        }
    )


def _extend_foundation_component_fingerprint(
    previous: str,
    record_id: str,
) -> str:
    return _canonical_digest(
        {
            "component_fingerprint_version": (
                FOUNDATION_COMPONENT_FINGERPRINT_VERSION
            ),
            "previous": previous,
            "record_id": record_id,
        }
    )


def _canonical_json_scalar(value: str) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _foundation_current_view_hash_suffix(
    *,
    foundation_version: str,
    registry_identity: str,
) -> bytes:
    """Return the exact suffix emitted by ``_canonical_digest`` above."""

    return b"".join(
        (
            b'],"current_view_fingerprint_version":',
            _canonical_json_scalar(
                FOUNDATION_CURRENT_VIEW_FINGERPRINT_VERSION
            ),
            b',"foundation_version":',
            _canonical_json_scalar(foundation_version),
            b',"registry_identity":',
            _canonical_json_scalar(registry_identity),
            b"}",
        )
    )


class _FoundationCurrentViewHashCursor:
    """Private immutable-use SHA cursor for append/move-to-tail updates.

    The published digest remains byte-for-byte identical to the registered
    canonical-JSON digest.  Keeping only two SHA prefix states avoids retaining
    another copy of current record identities or a persistent container.  The
    cursor is deliberately excluded from every transport and pickle contract;
    those boundaries rebuild and fully validate it from the current view.
    """

    __slots__ = (
        "__before_last",
        "__records",
        "__record_count",
        "__fingerprint",
        "__foundation_version",
        "__registry_identity",
    )

    def __init__(
        self,
        *,
        _capability: object,
        before_last: Any,
        records: Any,
        record_count: int,
        fingerprint: str,
        foundation_version: str,
        registry_identity: str,
    ) -> None:
        if _capability is not _FOUNDATION_INCREMENTAL_CAPABILITY:
            raise ValueError("foundation hash cursor authority is invalid")
        self.__before_last = before_last.copy()
        self.__records = records.copy()
        self.__record_count = record_count
        self.__fingerprint = fingerprint
        self.__foundation_version = foundation_version
        self.__registry_identity = registry_identity

    @classmethod
    def rebuild(
        cls,
        record_ids: Sequence[str],
        *,
        _capability: object,
        foundation_version: str,
        registry_identity: str,
    ) -> "_FoundationCurrentViewHashCursor":
        if _capability is not _FOUNDATION_INCREMENTAL_CAPABILITY:
            raise ValueError("foundation hash cursor authority is invalid")
        state = hashlib.sha256(b'{"current_record_ids":[')
        before_last = state.copy()
        count = 0
        for record_id in record_ids:
            if count:
                state.update(b",")
            before_last = state.copy()
            state.update(_canonical_json_scalar(record_id))
            count += 1
        fingerprint_state = state.copy()
        fingerprint_state.update(
            _foundation_current_view_hash_suffix(
                foundation_version=foundation_version,
                registry_identity=registry_identity,
            )
        )
        return cls(
            _capability=_FOUNDATION_INCREMENTAL_CAPABILITY,
            before_last=before_last,
            records=state,
            record_count=count,
            fingerprint=fingerprint_state.hexdigest(),
            foundation_version=foundation_version,
            registry_identity=registry_identity,
        )

    @property
    def record_count(self) -> int:
        return self.__record_count

    @property
    def fingerprint(self) -> str:
        return self.__fingerprint

    @property
    def foundation_version(self) -> str:
        return self.__foundation_version

    @property
    def registry_identity(self) -> str:
        return self.__registry_identity

    def _with_records_state(
        self,
        *,
        before_last: Any,
        records: Any,
        record_count: int,
    ) -> "_FoundationCurrentViewHashCursor":
        fingerprint_state = records.copy()
        fingerprint_state.update(
            _foundation_current_view_hash_suffix(
                foundation_version=self.foundation_version,
                registry_identity=self.registry_identity,
            )
        )
        return type(self)(
            _capability=_FOUNDATION_INCREMENTAL_CAPABILITY,
            before_last=before_last,
            records=records,
            record_count=record_count,
            fingerprint=fingerprint_state.hexdigest(),
            foundation_version=self.foundation_version,
            registry_identity=self.registry_identity,
        )

    def append(self, record_id: str) -> "_FoundationCurrentViewHashCursor":
        state = self.__records.copy()
        if self.record_count:
            state.update(b",")
        before_last = state.copy()
        state.update(_canonical_json_scalar(record_id))
        return self._with_records_state(
            before_last=before_last,
            records=state,
            record_count=self.record_count + 1,
        )

    def replace_last(
        self,
        record_id: str,
    ) -> "_FoundationCurrentViewHashCursor":
        if not self.record_count:
            raise ValueError("cannot replace the last record of an empty view")
        state = self.__before_last.copy()
        state.update(_canonical_json_scalar(record_id))
        return self._with_records_state(
            before_last=self.__before_last,
            records=state,
            record_count=self.record_count,
        )


def _payload_clock(
    object_type: FoundationObjectType,
    payload: Mapping[str, Any],
) -> pd.Timestamp:
    terminal_clock = {
        FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION: "terminal_at",
        FoundationObjectType.STRUCTURE_GENERATION: "terminated_at",
        FoundationObjectType.STRUCTURE_TRANSITION: "terminal_at",
        FoundationObjectType.RELATION_GENERATION: "terminated_at",
        FoundationObjectType.DELIVERY_PHASE_GENERATION: "terminated_at",
        FoundationObjectType.LIQUIDITY_CLUSTER: "terminated_at",
        FoundationObjectType.STRUCTURAL_RANGE: "terminated_at",
    }.get(object_type)
    if terminal_clock is not None and payload.get(terminal_clock) is not None:
        key = terminal_clock
    else:
        key = {
            FoundationObjectType.LIQUIDITY_LEVEL: "updated_at",
            FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION: "updated_at",
            FoundationObjectType.STRUCTURE_GENERATION: "updated_at",
            FoundationObjectType.STRUCTURE_TRANSITION: "updated_at",
            FoundationObjectType.RELATION_GENERATION: "last_updated_at",
            FoundationObjectType.DELIVERY_PHASE_GENERATION: "last_updated_at",
            FoundationObjectType.BOUNDARY_ATTACK: "known_at",
            FoundationObjectType.STRUCTURAL_LEG: "known_at",
            FoundationObjectType.SWING_GEOMETRY_NODE: "known_at",
            FoundationObjectType.SWING_GEOMETRY_ASSIGNMENT: "assigned_at",
            FoundationObjectType.LIQUIDITY_CLUSTER: "updated_at",
            FoundationObjectType.LIQUIDITY_CLUSTER_SUPERSESSION: "known_at",
            FoundationObjectType.STRUCTURAL_RANGE: "updated_at",
            FoundationObjectType.BALANCE_RANGE: "last_updated_at",
            FoundationObjectType.BASE_ORIGIN_CORE: "known_at",
            FoundationObjectType.QUALIFIED_ORDER_BLOCK: "known_at",
            FoundationObjectType.ZONE_FIRST_RETEST: "known_at",
            FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE: "last_updated_at",
        }[object_type]
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"foundation payload lacks primitive {key} clock")
    return aware_timestamp(value, name=f"foundation_payload.{key}")


def _payload_status(
    object_type: FoundationObjectType,
    payload: Mapping[str, Any],
) -> FoundationRecordStatus:
    if object_type in _FACT_TYPES:
        return FoundationRecordStatus.FACT
    if object_type is FoundationObjectType.LIQUIDITY_LEVEL:
        lifecycle = LiquidityLevelLifecycle(payload.get("lifecycle"))
        if lifecycle in {
            LiquidityLevelLifecycle.ACTIVE,
            LiquidityLevelLifecycle.REARMED,
        }:
            return FoundationRecordStatus.ACTIVE
        if lifecycle in {
            LiquidityLevelLifecycle.DISARMED,
            LiquidityLevelLifecycle.REARMABLE,
        }:
            return FoundationRecordStatus.INACTIVE
        return FoundationRecordStatus.TERMINAL
    if object_type is FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION:
        lifecycle = LiquidityInteractionLifecycle(payload.get("lifecycle"))
        return (
            FoundationRecordStatus.TERMINAL
            if lifecycle is LiquidityInteractionLifecycle.TERMINAL
            else FoundationRecordStatus.ACTIVE
        )
    if object_type is FoundationObjectType.STRUCTURE_GENERATION:
        lifecycle = StructureGenerationLifecycle(payload.get("lifecycle"))
        return (
            FoundationRecordStatus.TERMINAL
            if lifecycle is StructureGenerationLifecycle.TERMINATED
            else FoundationRecordStatus.ACTIVE
        )
    if object_type is FoundationObjectType.STRUCTURE_TRANSITION:
        lifecycle = StructureTransitionLifecycle(payload.get("lifecycle"))
        return (
            FoundationRecordStatus.ACTIVE
            if lifecycle is StructureTransitionLifecycle.STARTED
            else FoundationRecordStatus.TERMINAL
        )
    if object_type in {
        FoundationObjectType.RELATION_GENERATION,
        FoundationObjectType.DELIVERY_PHASE_GENERATION,
    }:
        lifecycle = GenerationLifecycle(payload.get("lifecycle"))
        return (
            FoundationRecordStatus.TERMINAL
            if lifecycle is GenerationLifecycle.TERMINATED
            else FoundationRecordStatus.ACTIVE
        )
    if object_type in {
        FoundationObjectType.LIQUIDITY_CLUSTER,
        FoundationObjectType.STRUCTURAL_RANGE,
    }:
        return (
            FoundationRecordStatus.TERMINAL
            if payload.get("terminated_at") is not None
            else FoundationRecordStatus.ACTIVE
        )
    if object_type is FoundationObjectType.BALANCE_RANGE:
        lifecycle = DealingRangeLifecycle(payload.get("lifecycle"))
        return (
            FoundationRecordStatus.TERMINAL
            if lifecycle is DealingRangeLifecycle.BROKEN
            else FoundationRecordStatus.ACTIVE
        )
    if object_type is FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE:
        availability = FVGAvailability(payload.get("availability"))
        return (
            FoundationRecordStatus.ACTIVE
            if availability is FVGAvailability.ACTIVE
            else FoundationRecordStatus.TERMINAL
        )
    raise TypeError(f"unsupported foundation object type: {object_type.value}")


def _validate_structure_provenance_payload(
    object_type: FoundationObjectType,
    payload: Mapping[str, Any],
    source_event_ids: Sequence[str],
) -> None:
    """Reject direct/transport records that bypass typed DTO invariants."""

    sources = frozenset(source_event_ids)
    if object_type is FoundationObjectType.STRUCTURE_GENERATION:
        protected_swing_id = payload.get("protected_swing_id")
        assignment_event_id = payload.get(
            "protected_swing_assignment_event_id"
        )
        if (protected_swing_id is None) != (assignment_event_id is None):
            raise ValueError(
                "protected Swing assignment provenance must be complete"
            )
        if assignment_event_id is not None and assignment_event_id not in sources:
            raise ValueError(
                "protected Swing assignment event must be exact ancestry"
            )
        return
    if object_type is not FoundationObjectType.STRUCTURE_TRANSITION:
        return

    lifecycle = StructureTransitionLifecycle(payload.get("lifecycle"))
    acceptance_event_id = payload.get("protected_acceptance_event_id")
    opposite_generation_id = payload.get("opposite_structure_generation_id")
    opposite_confirmation_id = payload.get("opposite_confirmation_event_id")
    resumption_event_id = payload.get("resumption_event_id")
    if (opposite_generation_id is None) != (opposite_confirmation_id is None):
        raise ValueError(
            "structure-transition opposite confirmation is incomplete"
        )
    cited = tuple(
        value
        for value in (
            acceptance_event_id,
            opposite_confirmation_id,
            resumption_event_id,
        )
        if value is not None
    )
    if any(
        not isinstance(value, str) or not value or value not in sources
        for value in cited
    ):
        raise ValueError("structure-transition evidence must be exact ancestry")
    if lifecycle is StructureTransitionLifecycle.STARTED:
        if any(
            value is not None
            for value in (
                opposite_generation_id,
                opposite_confirmation_id,
                resumption_event_id,
            )
        ):
            raise ValueError("started transition contains terminal evidence")
    elif lifecycle is StructureTransitionLifecycle.CONFIRMED:
        if (
            acceptance_event_id is None
            or opposite_generation_id is None
            or opposite_confirmation_id is None
            or resumption_event_id is not None
        ):
            raise ValueError(
                "confirmed transition lacks exact confirmation evidence"
            )
    elif lifecycle is StructureTransitionLifecycle.FAILED:
        if (
            resumption_event_id is None
            or opposite_generation_id is not None
            or opposite_confirmation_id is not None
        ):
            raise ValueError(
                "failed transition lacks exclusive resumption evidence"
            )
    elif lifecycle is StructureTransitionLifecycle.CENSORED and (
        opposite_generation_id is not None
        or opposite_confirmation_id is not None
        or resumption_event_id is not None
    ):
        raise ValueError(
            "censored transition contains invented terminal evidence"
        )


def _validate_typed_payload_contract(
    object_type: FoundationObjectType,
    payload: Mapping[str, Any],
) -> None:
    """Re-run the canonical DTO constructor at the serialized-record seam."""

    dto_type: type[object] | None = {
        FoundationObjectType.LIQUIDITY_LEVEL: LiquidityLevelState,
        FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION: (
            LiquidityInteractionGeneration
        ),
        FoundationObjectType.STRUCTURE_GENERATION: StructureGeneration,
        FoundationObjectType.STRUCTURE_TRANSITION: StructureTransition,
        FoundationObjectType.RELATION_GENERATION: RelationGeneration,
        FoundationObjectType.DELIVERY_PHASE_GENERATION: DeliveryPhaseGeneration,
        FoundationObjectType.BOUNDARY_ATTACK: BoundaryAttackFact,
        FoundationObjectType.STRUCTURAL_LEG: StructuralLegState,
        FoundationObjectType.SWING_GEOMETRY_NODE: SwingGeometryNode,
        FoundationObjectType.SWING_GEOMETRY_ASSIGNMENT: SwingGeometryAssignment,
        FoundationObjectType.LIQUIDITY_CLUSTER: LiquidityClusterState,
        FoundationObjectType.LIQUIDITY_CLUSTER_SUPERSESSION: (
            LiquidityClusterSupersession
        ),
        FoundationObjectType.STRUCTURAL_RANGE: StructuralRangeState,
        FoundationObjectType.BALANCE_RANGE: BalanceRangeState,
        FoundationObjectType.BASE_ORIGIN_CORE: BaseOriginCore,
        FoundationObjectType.QUALIFIED_ORDER_BLOCK: QualifiedOrderBlock,
        FoundationObjectType.ZONE_FIRST_RETEST: ZoneFirstRetest,
        FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE: FVGStructuralLifecycle,
    }.get(object_type)
    if dto_type is None:
        raise TypeError(f"unsupported foundation object type: {object_type.value}")
    kwargs = {
        item.name: payload[item.name]
        for item in dataclass_fields(dto_type)
        if item.init and item.name in payload
    }
    if object_type is FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION:
        kwargs["constituents"] = tuple(
            InteractionConstituent(**dict(item))
            for item in payload.get("constituents", ())
        )
    elif object_type is FoundationObjectType.STRUCTURAL_LEG:
        kwargs["timeframe"] = Timeframe(payload["timeframe"])
        kwargs["direction"] = Direction(payload["direction"])
        kwargs["rank"] = SwingRank(payload["rank"])
    elif object_type is FoundationObjectType.BALANCE_RANGE:
        kwargs["timeframe"] = Timeframe(payload["timeframe"])
        kwargs["lifecycle"] = DealingRangeLifecycle(payload["lifecycle"])
    rebuilt = dto_type(**kwargs)
    if _freeze_primitive(to_primitive(rebuilt)) != payload:
        raise ValueError("foundation serialized payload disagrees with canonical DTO")


def _dto_contract(dto: object) -> tuple[FoundationObjectType, str, pd.Timestamp]:
    """Return the exact supported type, identity, and record clock."""

    if isinstance(dto, LiquidityLevelState):
        return FoundationObjectType.LIQUIDITY_LEVEL, dto.level_id, dto.updated_at
    if isinstance(dto, LiquidityInteractionGeneration):
        return (
            FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION,
            dto.generation_id,
            dto.terminal_at or dto.updated_at,
        )
    if isinstance(dto, StructureGeneration):
        return (
            FoundationObjectType.STRUCTURE_GENERATION,
            dto.generation_id,
            dto.terminated_at or dto.updated_at,
        )
    if isinstance(dto, StructureTransition):
        return (
            FoundationObjectType.STRUCTURE_TRANSITION,
            dto.structure_transition_id,
            dto.terminal_at or dto.updated_at,
        )
    if isinstance(dto, RelationGeneration):
        return (
            FoundationObjectType.RELATION_GENERATION,
            dto.generation_id,
            dto.terminated_at or dto.updated_at,
        )
    if isinstance(dto, DeliveryPhaseGeneration):
        return (
            FoundationObjectType.DELIVERY_PHASE_GENERATION,
            dto.generation_id,
            dto.terminated_at or dto.updated_at,
        )
    if isinstance(dto, BoundaryAttackFact):
        return FoundationObjectType.BOUNDARY_ATTACK, dto.boundary_attack_id, dto.known_at
    if isinstance(dto, StructuralLegState):
        if dto.foundation_version != FOUNDATION_VERSION:
            raise ValueError("only complete foundation-v2 StructuralLegState is accepted")
        return FoundationObjectType.STRUCTURAL_LEG, dto.leg_id, dto.known_at
    if isinstance(dto, SwingGeometryNode):
        return FoundationObjectType.SWING_GEOMETRY_NODE, dto.swing_id, dto.known_at
    if isinstance(dto, SwingGeometryAssignment):
        return (
            FoundationObjectType.SWING_GEOMETRY_ASSIGNMENT,
            dto.assignment_id,
            dto.assigned_at,
        )
    if isinstance(dto, LiquidityClusterState):
        return (
            FoundationObjectType.LIQUIDITY_CLUSTER,
            dto.cluster_id,
            dto.terminated_at or dto.updated_at,
        )
    if isinstance(dto, LiquidityClusterSupersession):
        # Its semantic identity is the immutable supersession fact itself.
        primitive = _freeze_primitive(to_primitive(dto))
        identity = _canonical_digest(primitive)
        return (
            FoundationObjectType.LIQUIDITY_CLUSTER_SUPERSESSION,
            f"liquidity-cluster-supersession:{identity}",
            dto.known_at,
        )
    if isinstance(dto, StructuralRangeState):
        return (
            FoundationObjectType.STRUCTURAL_RANGE,
            dto.range_id,
            dto.terminated_at or dto.updated_at,
        )
    if isinstance(dto, BalanceRangeState):
        return FoundationObjectType.BALANCE_RANGE, dto.range_id, dto.last_updated_at
    if isinstance(dto, BaseOriginCore):
        return FoundationObjectType.BASE_ORIGIN_CORE, dto.core_id, dto.known_at
    if isinstance(dto, QualifiedOrderBlock):
        return (
            FoundationObjectType.QUALIFIED_ORDER_BLOCK,
            dto.qualified_ob_id,
            dto.known_at,
        )
    if isinstance(dto, ZoneFirstRetest):
        return (
            FoundationObjectType.ZONE_FIRST_RETEST,
            dto.first_retest_event_id,
            dto.known_at,
        )
    if isinstance(dto, FVGStructuralLifecycle):
        return (
            FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE,
            dto.fvg_id,
            dto.last_updated_at,
        )
    raise TypeError(f"unsupported semantic-foundation DTO: {type(dto).__name__}")


def _dto_sources(dto: object) -> tuple[str, ...] | None:
    if isinstance(dto, FVGStructuralLifecycle):
        return (
            dto.terminal_source_event_ids
            if dto.availability is not FVGAvailability.ACTIVE
            else (
                dto.source_creation_event_id,
                *dto.context_source_event_ids,
            )
        )
    if isinstance(
        dto,
        (
            LiquidityLevelState,
            LiquidityInteractionGeneration,
            StructureGeneration,
            StructureTransition,
            RelationGeneration,
            DeliveryPhaseGeneration,
            BoundaryAttackFact,
            BaseOriginCore,
            QualifiedOrderBlock,
            ZoneFirstRetest,
        ),
    ):
        return tuple(dto.source_event_ids)
    return None


@dataclass(frozen=True)
class FoundationRecord:
    """One immutable version/registry-bound revision of a foundation DTO."""

    object_type: FoundationObjectType
    object_id: str
    status: FoundationRecordStatus
    known_at: pd.Timestamp
    payload: Mapping[str, Any]
    source_event_ids: tuple[str, ...]
    foundation_version: str = FOUNDATION_VERSION
    registry_identity: str = FOUNDATION_CANONICAL_IDENTITY
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        object_type = FoundationObjectType(self.object_type)
        status = FoundationRecordStatus(self.status)
        clock = aware_timestamp(self.known_at, name="foundation_record.known_at")
        payload = _freeze_primitive(self.payload)
        if not isinstance(payload, FrozenDict):
            raise TypeError("foundation record payload must be a mapping")
        sources = _identities(self.source_event_ids, name="source_event_ids")
        object.__setattr__(self, "object_type", object_type)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "known_at", clock)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "source_event_ids", sources)
        if (
            not isinstance(self.object_id, str)
            or not self.object_id.strip()
            or self.foundation_version != FOUNDATION_VERSION
            or self.registry_identity != FOUNDATION_CANONICAL_IDENTITY
        ):
            raise ValueError("foundation record identity or version is invalid")
        id_field = _OBJECT_ID_FIELDS.get(object_type)
        if id_field is not None and payload.get(id_field) != self.object_id:
            raise ValueError("foundation record object identity disagrees with payload")
        version_field = _VERSION_FIELDS.get(object_type)
        if version_field is not None and payload.get(version_field) != FOUNDATION_VERSION:
            raise ValueError("foundation DTO payload is not frozen foundation v2")
        if _payload_clock(object_type, payload) != clock:
            raise ValueError("foundation record clock disagrees with DTO payload")
        if _payload_status(object_type, payload) is not status:
            raise ValueError("foundation record status disagrees with DTO lifecycle")
        _validate_typed_payload_contract(object_type, payload)
        embedded_sources = payload.get("source_event_ids")
        if embedded_sources is not None and tuple(embedded_sources) != sources:
            raise ValueError("foundation record source ancestry disagrees with payload")
        _validate_structure_provenance_payload(
            object_type,
            payload,
            sources,
        )
        digest = _canonical_digest(
            {
                "foundation_version": self.foundation_version,
                "registry_identity": self.registry_identity,
                "object_type": object_type.value,
                "object_id": self.object_id,
                "status": status.value,
                "known_at": clock.isoformat(),
                "payload": payload,
                "source_event_ids": sources,
            }
        )
        object.__setattr__(self, "record_id", f"foundation-record:{digest}")

    @classmethod
    def from_dto(
        cls,
        dto: object,
        *,
        source_event_ids: Sequence[str] | None = None,
    ) -> "FoundationRecord":
        object_type, object_id, known_at = _dto_contract(dto)
        expected_sources = _dto_sources(dto)
        if source_event_ids is None:
            if expected_sources is None:
                raise ValueError(
                    "DTO has no exact event ancestry; source_event_ids are required"
                )
            sources = _identities(expected_sources, name="source_event_ids")
        else:
            sources = _identities(source_event_ids, name="source_event_ids")
            if expected_sources is not None and sources != expected_sources:
                raise ValueError("source_event_ids differ from exact DTO ancestry")
        payload = _freeze_primitive(to_primitive(dto))
        status = _payload_status(object_type, payload)
        return cls(
            object_type=object_type,
            object_id=object_id,
            status=status,
            known_at=known_at,
            payload=payload,
            source_event_ids=sources,
        )


def _foundation_record_fingerprint(record: FoundationRecord) -> str:
    digest = _canonical_digest(
        {
            "foundation_version": record.foundation_version,
            "registry_identity": record.registry_identity,
            "object_type": record.object_type.value,
            "object_id": record.object_id,
            "status": record.status.value,
            "known_at": record.known_at.isoformat(),
            "payload": record.payload,
            "source_event_ids": record.source_event_ids,
        }
    )
    if record.record_id != f"foundation-record:{digest}":
        raise ValueError("foundation record identity differs from its bytes")
    return digest


def _require_foundation_record_identity_integrity(
    records: Sequence[FoundationRecord],
) -> None:
    """Fail closed when a frozen record envelope no longer binds its bytes."""

    for record in records:
        if not isinstance(record, FoundationRecord):
            raise ValueError("foundation current record identity is invalid")
        try:
            _foundation_record_fingerprint(record)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "foundation current record identity is invalid"
            ) from error


@dataclass(frozen=True, slots=True)
class _SwingGeometryView:
    """Parsed, identity-neutral geometry used only by cross-link validation."""

    object_id: str
    symbol: str
    instrument_id: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    lower_bound: float
    upper_bound: float

    @property
    def duration_seconds(self) -> float:
        return float((self.window_end - self.window_start).total_seconds())

    @property
    def price_span(self) -> float:
        return self.upper_bound - self.lower_bound


def _swing_geometry_view(record: FoundationRecord) -> _SwingGeometryView:
    """Parse one already DTO-validated node once for projection-local reuse."""

    if record.object_type is not FoundationObjectType.SWING_GEOMETRY_NODE:
        raise TypeError("Swing geometry cache accepts only geometry nodes")
    payload = record.payload
    return _SwingGeometryView(
        object_id=record.object_id,
        symbol=str(payload.get("symbol")),
        instrument_id=int(payload.get("instrument_id")),
        window_start=aware_timestamp(
            payload.get("window_start"),
            name="swing_geometry.window_start",
        ),
        # SwingGeometryNode freezes known_at == window_end.
        window_end=record.known_at,
        lower_bound=float(payload.get("lower_bound")),
        upper_bound=float(payload.get("upper_bound")),
    )


def _canonical_swing_parent(
    node: _SwingGeometryView,
    views: Mapping[str, _SwingGeometryView],
) -> _SwingGeometryView | None:
    best: _SwingGeometryView | None = None
    best_key: tuple[float, float, str] | None = None
    for parent in views.values():
        if (
            parent.object_id == node.object_id
            or parent.symbol != node.symbol
            or parent.instrument_id != node.instrument_id
            or parent.window_start > node.window_start
            or parent.window_end < node.window_end
            or parent.lower_bound > node.lower_bound
            or parent.upper_bound < node.upper_bound
            or parent.duration_seconds <= node.duration_seconds
        ):
            continue
        key = (
            parent.duration_seconds,
            parent.price_span,
            parent.object_id,
        )
        if best_key is None or key < best_key:
            best = parent
            best_key = key
    return best


def _canonical_swing_depth(
    node: _SwingGeometryView,
    views: Mapping[str, _SwingGeometryView],
) -> int:
    depth = 0
    cursor = node
    seen: set[str] = set()
    while (parent := _canonical_swing_parent(cursor, views)) is not None:
        if parent.object_id in seen:
            raise ValueError("Swing geometry contains a parent cycle")
        seen.add(parent.object_id)
        depth += 1
        cursor = parent
    return depth


def _permitted_liquidity_archive(
    previous: FoundationRecord,
    current: FoundationRecord,
) -> bool:
    if not (
        current.object_type is FoundationObjectType.LIQUIDITY_LEVEL
        and previous.status is FoundationRecordStatus.TERMINAL
        and current.status is FoundationRecordStatus.TERMINAL
        and previous.payload.get("lifecycle")
        == LiquidityLevelLifecycle.RETIRED.value
        and current.payload.get("lifecycle")
        == LiquidityLevelLifecycle.ARCHIVED.value
        and previous.payload.get("retired_at")
        == current.payload.get("retired_at")
        and previous.payload.get("retirement_reason")
        == current.payload.get("retirement_reason")
    ):
        return False
    mutable_archive_fields = frozenset(
        {
            "lifecycle",
            "updated_at",
            "archived_at",
            "archive_reason",
            "source_event_ids",
        }
    )
    previous_stable = {
        key: value
        for key, value in previous.payload.items()
        if key not in mutable_archive_fields
    }
    current_stable = {
        key: value
        for key, value in current.payload.items()
        if key not in mutable_archive_fields
    }
    return (
        previous_stable == current_stable
        and current.source_event_ids[: len(previous.source_event_ids)]
        == previous.source_event_ids
    )


def _validate_object_revision(
    previous: FoundationRecord,
    current: FoundationRecord,
) -> None:
    # One completed BAR may authoritatively expose touch, penetration and a
    # terminal resolution at the same knowledge clock. Their event/record
    # order remains append-only; only backward knowledge time is forbidden.
    if current.known_at < previous.known_at:
        raise ValueError("object revisions cannot move backward in knowledge time")
    if previous.status is FoundationRecordStatus.FACT or (
        previous.status is FoundationRecordStatus.TERMINAL
        and not _permitted_liquidity_archive(previous, current)
    ):
        raise ValueError("foundation fact or terminal object is immutable")
    lifecycle_field = {
        FoundationObjectType.LIQUIDITY_LEVEL: "lifecycle",
        FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION: "lifecycle",
        FoundationObjectType.STRUCTURE_GENERATION: "lifecycle",
        FoundationObjectType.STRUCTURE_TRANSITION: "lifecycle",
        FoundationObjectType.RELATION_GENERATION: "lifecycle",
        FoundationObjectType.DELIVERY_PHASE_GENERATION: "lifecycle",
        FoundationObjectType.BALANCE_RANGE: "lifecycle",
        FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE: "availability",
    }.get(current.object_type)
    permitted = {
        FoundationObjectType.LIQUIDITY_LEVEL: {
            "active": {"active", "disarmed", "retired", "archived"},
            "disarmed": {"disarmed", "rearmable", "rearmed", "retired", "archived"},
            "rearmable": {"rearmable", "rearmed", "retired", "archived"},
            "rearmed": {"rearmed", "disarmed", "retired", "archived"},
            "retired": {"archived"},
        },
        FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION: {
            "armed": {"armed", "touched", "penetrated", "terminal"},
            "touched": {"touched", "penetrated", "terminal"},
            "penetrated": {"penetrated", "terminal"},
        },
        FoundationObjectType.STRUCTURE_GENERATION: {
            "forming": {"forming", "confirmed", "terminated"},
            "confirmed": {"confirmed", "terminated"},
        },
        FoundationObjectType.STRUCTURE_TRANSITION: {
            "started": {"started", "confirmed", "failed", "censored"},
        },
        FoundationObjectType.RELATION_GENERATION: {
            "active": {"active", "terminated"},
        },
        FoundationObjectType.DELIVERY_PHASE_GENERATION: {
            "active": {"active", "terminated"},
        },
        FoundationObjectType.BALANCE_RANGE: {
            "forming": {"forming", "mature", "broken"},
            "mature": {"mature", "broken"},
        },
        FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE: {
            "active": {"active", "invalidated", "expired", "censored"},
        },
    }.get(current.object_type)
    if lifecycle_field is not None and permitted is not None:
        prior_value = previous.payload.get(lifecycle_field)
        current_value = current.payload.get(lifecycle_field)
        if current_value not in permitted.get(prior_value, set()):
            raise ValueError("foundation lifecycle revision is not preregistered")
    cumulative_types = {
        FoundationObjectType.LIQUIDITY_LEVEL,
        FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION,
        FoundationObjectType.STRUCTURE_GENERATION,
        FoundationObjectType.STRUCTURE_TRANSITION,
        FoundationObjectType.RELATION_GENERATION,
        FoundationObjectType.DELIVERY_PHASE_GENERATION,
        FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE,
    }
    if (
        current.object_type in cumulative_types
        and current.source_event_ids[: len(previous.source_event_ids)]
        != previous.source_event_ids
    ):
        raise ValueError("foundation lifecycle ancestry cannot be rewritten")
    if (
        current.object_type
        is FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION
    ):
        immutable_generation_fields = (
            "level_id",
            "source_timeframe",
            "interaction_timeframe",
            "level_side",
            "lower_bound_ticks",
            "upper_bound_ticks",
            "generation_number",
            "armed_at",
            "known_at",
            "armed_real_bar_ordinal",
            "previous_generation_id",
            "rearm_fact_id",
            "semantic_version",
        )
        if any(
            previous.payload.get(field) != current.payload.get(field)
            for field in immutable_generation_fields
        ):
            raise ValueError(
                "liquidity interaction generation scope is immutable"
            )
        once_set_path_fields = (
            "first_touch_at",
            "first_penetration_at",
            "first_inside_close_at",
            "first_outside_close_at",
        )
        if any(
            previous.payload.get(field) is not None
            and previous.payload.get(field) != current.payload.get(field)
            for field in once_set_path_fields
        ) or aware_timestamp(
            current.payload.get("updated_at"),
            name="liquidity_interaction.updated_at",
        ) < aware_timestamp(
            previous.payload.get("updated_at"),
            name="liquidity_interaction.updated_at",
        ):
            raise ValueError(
                "liquidity interaction path facts are immutable once set"
            )
        if (
            tuple(current.payload.get("touch_bar_ids", ()))
            [: len(tuple(previous.payload.get("touch_bar_ids", ())))]
            != tuple(previous.payload.get("touch_bar_ids", ()))
            or tuple(current.payload.get("constituents", ()))
            [: len(tuple(previous.payload.get("constituents", ())))]
            != tuple(previous.payload.get("constituents", ()))
            or int(current.payload.get("max_penetration_ticks", 0))
            < int(previous.payload.get("max_penetration_ticks", 0))
        ):
            raise ValueError(
                "liquidity interaction path history cannot be rewritten"
            )
    if current.object_type is FoundationObjectType.LIQUIDITY_LEVEL:
        immutable_level_fields = (
            "source_timeframe",
            "source_kind",
            "source_identity",
            "side",
            "price_ticks",
            "lower_bound_ticks",
            "upper_bound_ticks",
            "tick_size",
            "created_at",
            "price_anchor_rule",
            "semantic_version",
        )
        if any(
            previous.payload.get(field) != current.payload.get(field)
            for field in immutable_level_fields
        ):
            raise ValueError("liquidity level creation scope is immutable")
        previous_history = tuple(
            previous.payload.get("interaction_generation_ids", ())
        )
        current_history = tuple(
            current.payload.get("interaction_generation_ids", ())
        )
        if current_history[: len(previous_history)] != previous_history:
            raise ValueError(
                "liquidity interaction generation history cannot be rewritten"
            )
        prior_owner = previous.payload.get("owner_structure_generation_id")
        current_owner = current.payload.get("owner_structure_generation_id")
        if prior_owner is not None and current_owner != prior_owner:
            raise ValueError("liquidity structure ownership is immutable")


def _required_prior_record(
    prior: Mapping[tuple[FoundationObjectType, str], FoundationRecord],
    object_type: FoundationObjectType,
    object_id: object,
    *,
    role: str,
) -> FoundationRecord:
    if not isinstance(object_id, str) or not object_id:
        raise ValueError(f"{role} identity is missing")
    record = prior.get((object_type, object_id))
    if record is None:
        raise ValueError(f"{role} must reference an earlier foundation record")
    return record


def _matching_payload_fields(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    fields: Sequence[str],
    *,
    role: str,
) -> None:
    if any(left.get(field) != right.get(field) for field in fields):
        raise ValueError(f"{role} scope or geometry does not match its owner")


def _validate_generation_owner(
    prior: Mapping[tuple[FoundationObjectType, str], FoundationRecord],
    object_id: object,
    *,
    timeframe: object,
    role: str,
    require_live_confirmed: bool,
) -> FoundationRecord:
    owner = _required_prior_record(
        prior,
        FoundationObjectType.STRUCTURE_GENERATION,
        object_id,
        role=role,
    )
    if owner.payload.get("timeframe") != timeframe:
        raise ValueError(f"{role} timeframe does not match its owner")
    if require_live_confirmed and (
        owner.payload.get("lifecycle")
        != StructureGenerationLifecycle.CONFIRMED.value
    ):
        raise ValueError(f"{role} requires an earlier confirmed live owner")
    return owner


def _validate_zone_first_retest_cross_link(
    records: Mapping[tuple[FoundationObjectType, str], FoundationRecord],
    record: FoundationRecord,
    *,
    require_currently_interactable: bool,
) -> None:
    payload = record.payload
    object_kind = ZoneObjectKind(payload.get("object_kind"))
    target_type = {
        ZoneObjectKind.FVG: FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE,
        ZoneObjectKind.QUALIFIED_ORDER_BLOCK: (
            FoundationObjectType.QUALIFIED_ORDER_BLOCK
        ),
        ZoneObjectKind.BASE_ORIGIN_CORE: FoundationObjectType.BASE_ORIGIN_CORE,
        ZoneObjectKind.STRUCTURAL_RANGE: FoundationObjectType.STRUCTURAL_RANGE,
        ZoneObjectKind.LIQUIDITY_ZONE: FoundationObjectType.LIQUIDITY_LEVEL,
    }[object_kind]
    target = _required_prior_record(
        records,
        target_type,
        payload.get("object_id"),
        role="first-retest target",
    )
    if (
        require_currently_interactable
        and target_type
        in {
            FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE,
            FoundationObjectType.STRUCTURAL_RANGE,
            FoundationObjectType.LIQUIDITY_LEVEL,
        }
        and target.status is not FoundationRecordStatus.ACTIVE
    ):
        raise ValueError("first retest target is no longer interactable")
    creation_clock_field = {
        FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE: "known_at",
        FoundationObjectType.QUALIFIED_ORDER_BLOCK: "known_at",
        FoundationObjectType.BASE_ORIGIN_CORE: "known_at",
        FoundationObjectType.STRUCTURAL_RANGE: "known_at",
        FoundationObjectType.LIQUIDITY_LEVEL: "created_at",
    }[target_type]
    creation_clock = aware_timestamp(
        target.payload.get(creation_clock_field),
        name="first_retest.target_creation_clock",
    )
    if record.known_at <= creation_clock:
        raise ValueError("first retest must be strictly later than its target")
    common_fields = tuple(
        field
        for field in ("symbol", "instrument_id", "timeframe", "direction")
        if field in target.payload
    )
    _matching_payload_fields(
        payload,
        target.payload,
        common_fields,
        role="first retest",
    )
    if (
        object_kind is ZoneObjectKind.FVG
        and payload.get("creation_event_id")
        != target.payload.get("source_creation_event_id")
    ):
        raise ValueError("FVG first retest creation identity is incompatible")


def _validate_record_cross_links(
    prior: Mapping[tuple[FoundationObjectType, str], FoundationRecord],
    record: FoundationRecord,
    *,
    swing_geometry_views: Mapping[str, "_SwingGeometryView"] | None = None,
    swing_assignment_incumbents: Mapping[str, FoundationRecord] | None = None,
    current_static: bool = False,
) -> None:
    """Validate mandatory prior-object links at the append boundary.

    Append admission uses a prior-only map.  Cold current-graph admission may
    pass the complete current map with ``current_static=True``; in that mode
    only active dependants require a currently live target.
    """

    kind = record.object_type
    payload = record.payload
    if kind is FoundationObjectType.BOUNDARY_ATTACK:
        bos_generation_id = payload.get("bos_generation_id")
        bar_event_id = payload.get("bar_event_id")
        expected_id = canonical_semantic_id(
            "boundary-attack",
            payload.get("timeframe"),
            bos_generation_id,
            bar_event_id,
            payload.get("direction"),
            payload.get("boundary_ticks"),
        )
        prior_attacks = tuple(
            item
            for (object_type, _), item in prior.items()
            if object_type is FoundationObjectType.BOUNDARY_ATTACK
            and item.payload.get("bos_generation_id") == bos_generation_id
        )
        if (
            record.object_id != expected_id
            or record.source_event_ids
            != (
                payload.get("target_swing_event_id"),
                bar_event_id,
            )
            or any(item.payload.get("bar_event_id") == bar_event_id for item in prior_attacks)
            or payload.get("attempt_ordinal") != len(prior_attacks) + 1
        ):
            raise ValueError("boundary-attack identity or ordinal is incompatible")
        return

    if kind is FoundationObjectType.LIQUIDITY_LEVEL:
        owner_id = payload.get("owner_structure_generation_id")
        if owner_id is not None:
            owner = _validate_generation_owner(
                prior,
                owner_id,
                timeframe=payload.get("source_timeframe"),
                role="liquidity level structure owner",
                require_live_confirmed=(
                    record.status is FoundationRecordStatus.ACTIVE
                ),
            )
            if owner.payload.get("scope") != "external":
                raise ValueError("liquidity level owner must be external structure")
        active_generation_id = payload.get("active_generation_id")
        if active_generation_id is None:
            return
        interaction = _required_prior_record(
            prior,
            FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION,
            active_generation_id,
            role="liquidity level active interaction",
        )
        if (
            interaction.payload.get("level_id") != record.object_id
            or interaction.payload.get("lifecycle")
            == LiquidityInteractionLifecycle.TERMINAL.value
            or interaction.payload.get("source_timeframe")
            != payload.get("source_timeframe")
        ):
            raise ValueError("liquidity level active interaction is incompatible")
        return

    if kind is FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION:
        previous_generation_id = payload.get("previous_generation_id")
        if previous_generation_id is None:
            return
        previous = _required_prior_record(
            prior,
            FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION,
            previous_generation_id,
            role="rearmed liquidity prior interaction",
        )
        generation_number = payload.get("generation_number")
        if type(generation_number) is not int or (
            previous.payload.get("level_id") != payload.get("level_id")
            or previous.payload.get("lifecycle")
            != LiquidityInteractionLifecycle.TERMINAL.value
            or previous.payload.get("generation_number")
            != generation_number - 1
        ):
            raise ValueError("rearmed liquidity prior interaction is incompatible")
        return

    if kind is FoundationObjectType.STRUCTURE_TRANSITION:
        incumbent = _validate_generation_owner(
            prior,
            payload.get("incumbent_structure_generation_id"),
            timeframe=payload.get("timeframe"),
            role="structure-transition incumbent",
            require_live_confirmed=False,
        )
        if (
            incumbent.payload.get("scope") != payload.get("scope")
            or incumbent.payload.get("direction")
            != payload.get("incumbent_direction")
        ):
            raise ValueError("structure-transition incumbent is incompatible")
        acceptance_event_id = payload.get("protected_acceptance_event_id")
        if acceptance_event_id is None:
            if (
                (
                    not current_static
                    and payload.get("lifecycle")
                    in {
                        StructureTransitionLifecycle.STARTED.value,
                        StructureTransitionLifecycle.FAILED.value,
                    }
                )
                or (
                    current_static
                    and record.status is FoundationRecordStatus.ACTIVE
                )
            ) and incumbent.payload.get(
                "lifecycle"
            ) != StructureGenerationLifecycle.CONFIRMED.value:
                raise ValueError(
                    "live or failed transition requires a confirmed incumbent"
                )
        elif (
            incumbent.payload.get("lifecycle")
            != StructureGenerationLifecycle.TERMINATED.value
            or incumbent.payload.get("termination_reason")
            != "protected_break_accepted"
            or incumbent.payload.get("protected_acceptance_event_id")
            != acceptance_event_id
        ):
            raise ValueError(
                "structure-transition Acceptance conflicts with incumbent terminal"
            )
        opposite_id = payload.get("opposite_structure_generation_id")
        if opposite_id is not None:
            opposite = _validate_generation_owner(
                prior,
                opposite_id,
                timeframe=payload.get("timeframe"),
                role="structure-transition opposite generation",
                require_live_confirmed=(
                    not current_static
                    or record.status is FoundationRecordStatus.ACTIVE
                ),
            )
            confirmed_at = aware_timestamp(
                opposite.payload.get("confirmed_at"),
                name="structure_transition.opposite_confirmed_at",
            )
            started_at = aware_timestamp(
                payload.get("started_at"),
                name="structure_transition.started_at",
            )
            if (
                opposite.payload.get("scope") != payload.get("scope")
                or opposite.payload.get("direction")
                != payload.get("challenger_direction")
                or opposite.payload.get("confirmation_event_id")
                != payload.get("opposite_confirmation_event_id")
                or confirmed_at <= started_at
            ):
                raise ValueError(
                    "structure-transition opposite confirmation is incompatible"
                )
        return

    if kind is FoundationObjectType.RELATION_GENERATION:
        parent = _validate_generation_owner(
            prior,
            payload.get("parent_structure_generation_id"),
            timeframe=payload.get("parent_tf"),
            role="relation parent",
            require_live_confirmed=(
                record.status is FoundationRecordStatus.ACTIVE
            ),
        )
        child = _validate_generation_owner(
            prior,
            payload.get("child_structure_generation_id"),
            timeframe=payload.get("child_tf"),
            role="relation child",
            require_live_confirmed=(
                record.status is FoundationRecordStatus.ACTIVE
            ),
        )
        if parent.object_id == child.object_id:
            raise ValueError("relation parent and child owners must be distinct")
        return

    if kind is FoundationObjectType.DELIVERY_PHASE_GENERATION:
        _validate_generation_owner(
            prior,
            payload.get("parent_structure_generation_id"),
            timeframe=payload.get("timeframe"),
            role="delivery parent",
            require_live_confirmed=(
                record.status is FoundationRecordStatus.ACTIVE
            ),
        )
        return

    if kind is FoundationObjectType.SWING_GEOMETRY_ASSIGNMENT:
        child = _required_prior_record(
            prior,
            FoundationObjectType.SWING_GEOMETRY_NODE,
            payload.get("child_swing_id"),
            role="Swing assignment child",
        )
        views = (
            swing_geometry_views
            if swing_geometry_views is not None
            else {
                item.object_id: _swing_geometry_view(item)
                for (object_type, _), item in prior.items()
                if object_type is FoundationObjectType.SWING_GEOMETRY_NODE
            }
        )
        child_view = views.get(child.object_id)
        if child_view is None:
            raise ValueError("Swing assignment child geometry is unavailable")

        expected_parent = _canonical_swing_parent(child_view, views)
        expected_parent_id = (
            None if expected_parent is None else expected_parent.object_id
        )
        expected_depth = _canonical_swing_depth(child_view, views)
        if (
            payload.get("parent_swing_id") != expected_parent_id
            or payload.get("geometric_depth") != expected_depth
        ):
            raise ValueError("Swing assignment is not the canonical geometry")
        incumbent = (
            swing_assignment_incumbents.get(child.object_id)
            if swing_assignment_incumbents is not None
            else max(
                (
                    item
                    for (object_type, _), item in prior.items()
                    if object_type
                    is FoundationObjectType.SWING_GEOMETRY_ASSIGNMENT
                    and item.payload.get("child_swing_id") == child.object_id
                ),
                key=lambda item: (item.known_at, item.object_id),
                default=None,
            )
        )
        supersedes = payload.get("supersedes_assignment_id")
        if incumbent is None:
            if supersedes is not None:
                raise ValueError("first Swing assignment cannot supersede history")
        elif (
            supersedes != incumbent.object_id
            or (
                incumbent.payload.get("parent_swing_id") == expected_parent_id
                and incumbent.payload.get("geometric_depth")
                == expected_depth
            )
        ):
            raise ValueError("Swing assignment supersession is not canonical")
        return

    if kind is FoundationObjectType.LIQUIDITY_CLUSTER:
        levels = tuple(
            _required_prior_record(
                prior,
                FoundationObjectType.LIQUIDITY_LEVEL,
                level_id,
                role="liquidity cluster member level",
            )
            for level_id in payload.get("member_level_ids", ())
        )
        member_prices = tuple(payload.get("member_prices", ()))
        if len(levels) != len(member_prices):
            raise ValueError("liquidity cluster member prices are incomplete")
        if any(
            level.payload.get("side") != payload.get("side")
            or not math.isclose(
                float(member_price),
                float(level.payload.get("price_ticks"))
                * float(level.payload.get("tick_size")),
            )
            or not math.isclose(
                float(level.payload.get("tick_size")),
                float(payload.get("tick_size")),
            )
            for level, member_price in zip(levels, member_prices, strict=True)
        ) or frozenset(payload.get("member_source_ids", ())) != frozenset(
            level.payload.get("source_identity") for level in levels
        ):
            raise ValueError("liquidity cluster membership does not match its levels")
        if record.status is FoundationRecordStatus.ACTIVE and any(
            level.status is not FoundationRecordStatus.ACTIVE for level in levels
        ):
            raise ValueError("active liquidity cluster contains an inactive level")
        return

    if kind is FoundationObjectType.LIQUIDITY_CLUSTER_SUPERSESSION:
        _required_prior_record(
            prior,
            FoundationObjectType.LIQUIDITY_CLUSTER,
            payload.get("superseded_cluster_id"),
            role="superseded liquidity cluster",
        )
        for replacement_id in payload.get("replacement_cluster_ids", ()):
            _required_prior_record(
                prior,
                FoundationObjectType.LIQUIDITY_CLUSTER,
                replacement_id,
                role="replacement liquidity cluster",
            )
        return

    if kind is FoundationObjectType.STRUCTURAL_RANGE:
        owner = _validate_generation_owner(
            prior,
            payload.get("structure_generation_id"),
            timeframe=payload.get("timeframe"),
            role="StructuralRange owner",
            require_live_confirmed=(
                record.status is FoundationRecordStatus.ACTIVE
            ),
        )
        if (
            owner.payload.get("scope") != "external"
            or owner.payload.get("direction") != payload.get("direction")
        ):
            raise ValueError("StructuralRange owner must be external structure")
        range_known_at = aware_timestamp(
            payload.get("known_at"),
            name="structural_range.known_at",
        )
        owner_confirmed_at = aware_timestamp(
            owner.payload.get("confirmed_at"),
            name="structural_range.owner_confirmed_at",
        )
        if range_known_at < owner_confirmed_at:
            raise ValueError("StructuralRange predates its confirmed owner")
        for field in ("lower_swing_id", "upper_swing_id"):
            node = _required_prior_record(
                prior,
                FoundationObjectType.SWING_GEOMETRY_NODE,
                payload.get(field),
                role=f"StructuralRange {field}",
            )
            if (
                node.payload.get("timeframe") != payload.get("timeframe")
                or node.payload.get("symbol") != payload.get("symbol")
                or node.payload.get("instrument_id")
                != payload.get("instrument_id")
                or range_known_at < node.known_at
            ):
                raise ValueError("StructuralRange Swing timeframe is incompatible")
        supersedes = payload.get("supersedes_range_id")
        if supersedes is not None:
            previous = _required_prior_record(
                prior,
                FoundationObjectType.STRUCTURAL_RANGE,
                supersedes,
                role="superseded StructuralRange",
            )
            if previous.payload.get("timeframe") != payload.get("timeframe"):
                raise ValueError("superseded StructuralRange scope is incompatible")
        return

    if kind is FoundationObjectType.QUALIFIED_ORDER_BLOCK:
        core = _required_prior_record(
            prior,
            FoundationObjectType.BASE_ORIGIN_CORE,
            payload.get("base_origin_core_id"),
            role="Qualified OB Base Origin Core",
        )
        _matching_payload_fields(
            payload,
            core.payload,
            (
                "source_displacement_id",
                "source_displacement_event_id",
                "symbol",
                "instrument_id",
                "timeframe",
                "direction",
                "lower_bound",
                "upper_bound",
                "body_lower_bound",
                "body_upper_bound",
                "tick_size",
                "semantic_version",
            ),
            role="Qualified OB",
        )
        if record.known_at < core.known_at or any(
            event_id not in record.source_event_ids
            for event_id in (
                payload.get("source_displacement_event_id"),
                payload.get("compatible_structure_event_id"),
            )
        ):
            raise ValueError("Qualified OB clock or qualification ancestry is invalid")
        return

    if kind is FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE:
        creation_event_id = payload.get("source_creation_event_id")
        if creation_event_id not in record.source_event_ids:
            raise ValueError("FVG creation event is absent from exact ancestry")
        parent_id = payload.get("parent_structure_generation_id")
        if parent_id is None:
            return
        parent = _validate_generation_owner(
            prior,
            parent_id,
            timeframe=payload.get("timeframe"),
            role="FVG parent structure",
            require_live_confirmed=(
                record.status is FoundationRecordStatus.ACTIVE
            ),
        )
        if parent.payload.get("scope") != "external":
            raise ValueError("FVG parent structure must be external")
        confirmation_event_id = parent.payload.get("confirmation_event_id")
        context_sources = frozenset(payload.get("context_source_event_ids", ()))
        if (
            not isinstance(confirmation_event_id, str)
            or confirmation_event_id not in context_sources
        ):
            raise ValueError("FVG context does not cite its parent confirmation")
        range_id = payload.get("structural_range_id")
        if range_id is not None:
            structural_range = _required_prior_record(
                prior,
                FoundationObjectType.STRUCTURAL_RANGE,
                range_id,
                role="FVG StructuralRange",
            )
            if (
                structural_range.payload.get("structure_generation_id")
                != parent.object_id
                or structural_range.payload.get("timeframe")
                != payload.get("timeframe")
            ):
                raise ValueError("FVG StructuralRange context is incompatible")
        return

    if kind is FoundationObjectType.ZONE_FIRST_RETEST:
        _validate_zone_first_retest_cross_link(
            prior,
            record,
            require_currently_interactable=not current_static,
        )


def _validate_projection_completeness(
    latest: Mapping[tuple[FoundationObjectType, str], FoundationRecord],
) -> None:
    """Reject a finalized graph with owner-first batch prefixes left orphaned."""

    for (object_type, _), record in latest.items():
        if object_type is FoundationObjectType.LIQUIDITY_LEVEL:
            _validate_liquidity_level_interaction_graph(latest, record)
        elif (
            object_type
            is FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION
        ):
            _validate_liquidity_interaction_registration(latest, record)


def _validate_liquidity_interaction_registration(
    latest: Mapping[tuple[FoundationObjectType, str], FoundationRecord],
    interaction: FoundationRecord,
) -> None:
    level = latest.get(
        (
            FoundationObjectType.LIQUIDITY_LEVEL,
            interaction.payload.get("level_id"),
        )
    )
    if (
        level is None
        or interaction.object_id
        not in tuple(level.payload.get("interaction_generation_ids", ()))
        or interaction.payload.get("source_timeframe")
        != level.payload.get("source_timeframe")
        or interaction.payload.get("level_side") != level.payload.get("side")
        or interaction.payload.get("lower_bound_ticks")
        != level.payload.get("lower_bound_ticks")
        or interaction.payload.get("upper_bound_ticks")
        != level.payload.get("upper_bound_ticks")
    ):
        raise ValueError(
            "finalized liquidity interaction lacks its registered level"
        )


def _validate_liquidity_level_interaction_graph(
    latest: Mapping[tuple[FoundationObjectType, str], FoundationRecord],
    level: FoundationRecord,
    *,
    tail_only: bool = False,
) -> None:
    payload = level.payload
    history_ids = tuple(payload.get("interaction_generation_ids", ()))
    interactions: list[FoundationRecord] = []
    created_at = aware_timestamp(
        payload.get("created_at"),
        name="liquidity_level.created_at",
    )
    start = max(0, len(history_ids) - 2) if tail_only else 0
    for ordinal, generation_id in enumerate(
        history_ids[start:],
        start=start + 1,
    ):
        interaction = _required_prior_record(
            latest,
            FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION,
            generation_id,
            role="liquidity level interaction history",
        )
        _validate_liquidity_interaction_registration(latest, interaction)
        interaction_payload = interaction.payload
        previous_id = None if ordinal == 1 else history_ids[ordinal - 2]
        armed_at = aware_timestamp(
            interaction_payload.get("armed_at"),
            name="liquidity_interaction.armed_at",
        )
        expected_generation_id = canonical_semantic_id(
            "liquidity-interaction",
            level.object_id,
            ordinal,
            armed_at,
            previous_id,
        )
        if (
            interaction.object_id != expected_generation_id
            or interaction_payload.get("generation_number") != ordinal
            or interaction_payload.get("previous_generation_id")
            != previous_id
            or interaction.known_at < created_at
        ):
            raise ValueError(
                "liquidity interaction generation chain is incompatible"
            )
        if previous_id is not None and interactions and (
            interactions[-1].payload.get("lifecycle")
            != LiquidityInteractionLifecycle.TERMINAL.value
        ):
            raise ValueError(
                "liquidity interaction generation predecessor is live"
            )
        interactions.append(interaction)

    active_id = payload.get("active_generation_id")
    if tail_only:
        tail = interactions[-1]
        previous = None if len(interactions) == 1 else interactions[-2]
        if active_id is None:
            if (
                tail.payload.get("lifecycle")
                != LiquidityInteractionLifecycle.TERMINAL.value
                or payload.get("last_terminal_generation_id")
                != tail.object_id
            ):
                raise ValueError(
                    "inactive liquidity level retains a live interaction"
                )
        elif (
            active_id != tail.object_id
            or tail.payload.get("lifecycle")
            == LiquidityInteractionLifecycle.TERMINAL.value
            or level.status is not FoundationRecordStatus.ACTIVE
            or payload.get("last_terminal_generation_id")
            != (None if previous is None else previous.object_id)
        ):
            raise ValueError(
                "active liquidity level interaction is not the live chain tail"
            )
        return

    live_ids = tuple(
        interaction.object_id
        for interaction in interactions
        if interaction.payload.get("lifecycle")
        != LiquidityInteractionLifecycle.TERMINAL.value
    )
    if active_id is None:
        if live_ids:
            raise ValueError(
                "inactive liquidity level retains a live interaction"
            )
    elif (
        active_id != history_ids[-1]
        or live_ids != (active_id,)
        or level.status is not FoundationRecordStatus.ACTIVE
    ):
        raise ValueError(
            "active liquidity level interaction is not the live chain tail"
        )
    terminal_ids = tuple(
        interaction.object_id
        for interaction in interactions
        if interaction.payload.get("lifecycle")
        == LiquidityInteractionLifecycle.TERMINAL.value
    )
    expected_terminal_id = terminal_ids[-1] if terminal_ids else None
    if payload.get("last_terminal_generation_id") != expected_terminal_id:
        raise ValueError(
            "liquidity level last terminal interaction is incompatible"
        )


def _validate_current_record_cross_links(
    latest: Mapping[tuple[FoundationObjectType, str], FoundationRecord],
    record: FoundationRecord,
) -> None:
    """Validate static links against the complete canonical current map."""

    if record.object_type in {
        FoundationObjectType.BOUNDARY_ATTACK,
        FoundationObjectType.SWING_GEOMETRY_NODE,
        FoundationObjectType.SWING_GEOMETRY_ASSIGNMENT,
    }:
        return
    _validate_record_cross_links(
        latest,
        record,
        current_static=True,
    )


def _validate_current_swing_graph(
    projection: "FoundationProjection",
) -> None:
    latest = projection._latest_records_by_key_cache
    histories: dict[str, list[FoundationRecord]] = {}
    for record in projection.current_records:
        if (
            record.object_type
            is not FoundationObjectType.SWING_GEOMETRY_ASSIGNMENT
        ):
            continue
        child_id = str(record.payload.get("child_swing_id"))
        _required_prior_record(
            latest,
            FoundationObjectType.SWING_GEOMETRY_NODE,
            child_id,
            role="Swing assignment child",
        )
        parent_id = record.payload.get("parent_swing_id")
        if parent_id is not None:
            _required_prior_record(
                latest,
                FoundationObjectType.SWING_GEOMETRY_NODE,
                parent_id,
                role="Swing assignment parent",
            )
        histories.setdefault(child_id, []).append(record)

    for child_id, history in histories.items():
        ordered = sorted(
            history,
            key=lambda item: (item.known_at, item.object_id),
        )
        incumbent = ordered[-1]
        previous = None if len(ordered) == 1 else ordered[-2]
        _validate_record_cross_links(
            latest,
            incumbent,
            swing_geometry_views=projection._swing_geometry_views_cache,
            swing_assignment_incumbents=(
                {} if previous is None else {child_id: previous}
            ),
            current_static=True,
        )


def _validate_projection_current_graph(
    projection: "FoundationProjection",
) -> None:
    """Validate the static current graph, never historical admission state."""

    latest = projection._latest_records_by_key_cache
    _validate_projection_completeness(latest)
    for record in projection.current_records:
        _validate_current_record_cross_links(latest, record)
    _validate_current_swing_graph(projection)


@dataclass(frozen=True)
class FoundationProjection:
    """Frozen current view published by the single-writer hot owner."""

    current_records: tuple[FoundationRecord, ...] = ()
    record_count: int = 0
    component_fingerprint: str = ""
    current_view_fingerprint: str = ""
    asof: pd.Timestamp | None = None
    foundation_version: str = FOUNDATION_VERSION
    registry_identity: str = FOUNDATION_CANONICAL_IDENTITY
    schema_version: int = FOUNDATION_PROJECTION_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        records = tuple(self.current_records)
        object.__setattr__(self, "current_records", records)
        _require_foundation_record_identity_integrity(records)
        empty_fingerprint = _foundation_component_fingerprint_seed(
            foundation_version=self.foundation_version,
            registry_identity=self.registry_identity,
        )
        if not self.component_fingerprint:
            object.__setattr__(self, "component_fingerprint", empty_fingerprint)
        view_cursor = _FoundationCurrentViewHashCursor.rebuild(
            tuple(record.record_id for record in records),
            _capability=_FOUNDATION_INCREMENTAL_CAPABILITY,
            foundation_version=self.foundation_version,
            registry_identity=self.registry_identity,
        )
        expected_view_fingerprint = view_cursor.fingerprint
        if not self.current_view_fingerprint:
            object.__setattr__(
                self,
                "current_view_fingerprint",
                expected_view_fingerprint,
            )
        keys = tuple((record.object_type, record.object_id) for record in records)
        if (
            self.schema_version != FOUNDATION_PROJECTION_STATE_SCHEMA_VERSION
            or self.foundation_version != FOUNDATION_VERSION
            or self.registry_identity != FOUNDATION_CANONICAL_IDENTITY
            or type(self.record_count) is not int
            or self.record_count < len(records)
            or len(keys) != len(set(keys))
            or any(
                record.foundation_version != self.foundation_version
                or record.registry_identity != self.registry_identity
                for record in records
            )
            or len(self.component_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.component_fingerprint)
            or self.current_view_fingerprint != expected_view_fingerprint
            or (self.record_count == 0) != (not records)
            or (self.record_count == 0 and self.component_fingerprint != empty_fingerprint)
        ):
            raise ValueError("foundation current projection is invalid")
        latest = dict(zip(keys, records))
        geometry_views = {
            record.object_id: _swing_geometry_view(record)
            for record in records
            if record.object_type is FoundationObjectType.SWING_GEOMETRY_NODE
        }
        assignment_incumbents: dict[str, FoundationRecord] = {}
        interaction_ids_by_level: dict[str, set[str]] = {}
        for record in records:
            if (
                record.object_type
                is FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION
            ):
                level_id = str(record.payload.get("level_id"))
                interaction_ids_by_level.setdefault(level_id, set()).add(
                    record.object_id
                )
            if record.object_type is not FoundationObjectType.SWING_GEOMETRY_ASSIGNMENT:
                continue
            child_id = str(record.payload.get("child_swing_id"))
            incumbent = assignment_incumbents.get(child_id)
            if incumbent is None or (record.known_at, record.object_id) > (
                incumbent.known_at,
                incumbent.object_id,
            ):
                assignment_incumbents[child_id] = record
        object.__setattr__(self, "_latest_records_by_key_cache", MappingProxyType(latest))
        object.__setattr__(
            self,
            "_current_record_ids_cache",
            frozenset(record.record_id for record in records),
        )
        object.__setattr__(self, "_swing_geometry_views_cache", MappingProxyType(geometry_views))
        object.__setattr__(self, "_swing_assignment_incumbents_cache", MappingProxyType(assignment_incumbents))
        object.__setattr__(
            self,
            "_liquidity_interaction_ids_by_level_cache",
            MappingProxyType(
                {
                    level_id: frozenset(interaction_ids)
                    for level_id, interaction_ids
                    in interaction_ids_by_level.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "_current_view_hash_cursor_cache",
            view_cursor,
        )
        expected_asof = max((record.known_at for record in records), default=None)
        if self.asof is None:
            object.__setattr__(self, "asof", expected_asof)
        else:
            asof = aware_timestamp(self.asof, name="foundation_projection.asof")
            object.__setattr__(self, "asof", asof)
            if expected_asof is None or asof < expected_asof:
                raise ValueError("foundation projection asof predates its current view")

    @classmethod
    def _from_incremental_state(
        cls,
        *,
        _capability: object,
        current_records: tuple[FoundationRecord, ...],
        record_count: int,
        component_fingerprint: str,
        asof: pd.Timestamp | None,
        latest_records_by_key: dict[
            tuple[FoundationObjectType, str], FoundationRecord
        ],
        current_record_ids: set[str],
        swing_geometry_views: dict[str, _SwingGeometryView],
        swing_assignment_incumbents: dict[str, FoundationRecord],
        liquidity_interaction_ids_by_level: dict[str, frozenset[str]],
        current_view_hash_cursor: _FoundationCurrentViewHashCursor,
        foundation_version: str,
        registry_identity: str,
    ) -> "FoundationProjection":
        """Freeze an internally validated write-set without rescanning it.

        This is not a public deserialization path.  Each added record has
        already passed identity, revision, clock, and cross-link validation;
        the maps and SHA cursor are derived only from the committed owner plus
        that bounded write-set.  Public construction and every persistence
        boundary continue through ``__post_init__`` and perform a full check.
        """

        if (
            _capability is not _FOUNDATION_INCREMENTAL_CAPABILITY
            or type(current_records) is not tuple
            or type(record_count) is not int
            or record_count < len(current_records)
            or foundation_version != FOUNDATION_VERSION
            or registry_identity != FOUNDATION_CANONICAL_IDENTITY
            or len(component_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in component_fingerprint
            )
            or type(latest_records_by_key) is not dict
            or type(current_record_ids) is not set
            or type(swing_geometry_views) is not dict
            or type(swing_assignment_incumbents) is not dict
            or type(liquidity_interaction_ids_by_level) is not dict
            or type(current_view_hash_cursor)
            is not _FoundationCurrentViewHashCursor
            or current_view_hash_cursor.record_count != len(current_records)
            or current_view_hash_cursor.foundation_version
            != foundation_version
            or current_view_hash_cursor.registry_identity != registry_identity
            or len(latest_records_by_key) != len(current_records)
            or len(current_record_ids) != len(current_records)
            or (record_count == 0) != (not current_records)
            or (current_records and asof is None)
        ):
            raise ValueError("incremental foundation projection is invalid")
        frozen_asof = (
            None
            if asof is None
            else aware_timestamp(asof, name="foundation_projection.asof")
        )
        projection = object.__new__(cls)
        values = {
            "current_records": current_records,
            "record_count": record_count,
            "component_fingerprint": component_fingerprint,
            "current_view_fingerprint": (
                current_view_hash_cursor.fingerprint
            ),
            "asof": frozen_asof,
            "foundation_version": foundation_version,
            "registry_identity": registry_identity,
            "schema_version": FOUNDATION_PROJECTION_STATE_SCHEMA_VERSION,
        }
        for name, value in values.items():
            object.__setattr__(projection, name, value)
        object.__setattr__(
            projection,
            "_latest_records_by_key_cache",
            MappingProxyType(latest_records_by_key),
        )
        object.__setattr__(
            projection,
            "_current_record_ids_cache",
            frozenset(current_record_ids),
        )
        object.__setattr__(
            projection,
            "_swing_geometry_views_cache",
            MappingProxyType(swing_geometry_views),
        )
        object.__setattr__(
            projection,
            "_swing_assignment_incumbents_cache",
            MappingProxyType(swing_assignment_incumbents),
        )
        object.__setattr__(
            projection,
            "_liquidity_interaction_ids_by_level_cache",
            MappingProxyType(liquidity_interaction_ids_by_level),
        )
        object.__setattr__(
            projection,
            "_current_view_hash_cursor_cache",
            current_view_hash_cursor,
        )
        return projection

    def __getstate__(self) -> Mapping[str, Any]:
        """Serialize only current view, count, rolling hash, and identities."""

        canonical = FoundationProjectionReducer.validate_complete(self)
        return {
            "schema_version": canonical.schema_version,
            "current_records": canonical.current_records,
            "record_count": canonical.record_count,
            "component_fingerprint": canonical.component_fingerprint,
            "current_view_fingerprint": canonical.current_view_fingerprint,
            "asof": canonical.asof,
            "foundation_version": canonical.foundation_version,
            "registry_identity": canonical.registry_identity,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema_version",
            "current_records",
            "record_count",
            "component_fingerprint",
            "current_view_fingerprint",
            "asof",
            "foundation_version",
            "registry_identity",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != expected
            or state.get("schema_version") != FOUNDATION_PROJECTION_STATE_SCHEMA_VERSION
        ):
            raise ValueError("foundation projection checkpoint schema changed")
        for name in expected:
            object.__setattr__(self, name, state[name])
        self.__post_init__()
        _validate_projection_current_graph(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> "FoundationProjection":
        memo[id(self)] = self
        return self

    @property
    def latest_records(self) -> tuple[FoundationRecord, ...]:
        return self.current_records

    @property
    def active_records(self) -> tuple[FoundationRecord, ...]:
        return tuple(
            record
            for record in self.latest_records
            if record.status is FoundationRecordStatus.ACTIVE
        )

    @property
    def terminal_records(self) -> tuple[FoundationRecord, ...]:
        return tuple(
            record
            for record in self.latest_records
            if record.status is FoundationRecordStatus.TERMINAL
        )

    @property
    def active_dol_candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            record.object_id
            for record in self.latest_records
            if record.object_type is FoundationObjectType.LIQUIDITY_LEVEL
            and record.status is FoundationRecordStatus.ACTIVE
            and record.payload.get("lifecycle")
            in {
                LiquidityLevelLifecycle.ACTIVE.value,
                LiquidityLevelLifecycle.REARMED.value,
            }
        )

    def current_record_for(
        self,
        object_type: FoundationObjectType,
        object_id: str,
    ) -> FoundationRecord | None:
        kind = FoundationObjectType(object_type)
        return self._latest_records_by_key_cache.get((kind, object_id))

    def transport_payload(self) -> Mapping[str, Any]:
        """Return the compact snapshot/replay transport contract."""

        canonical = FoundationProjectionReducer.validate_complete(self)
        return FrozenDict(
            {
                "schema_version": canonical.schema_version,
                "current_records": to_primitive(canonical.current_records),
                "record_count": canonical.record_count,
                "component_fingerprint": canonical.component_fingerprint,
                "current_view_fingerprint": (
                    canonical.current_view_fingerprint
                ),
                "asof": to_primitive(canonical.asof),
                "foundation_version": canonical.foundation_version,
                "registry_identity": canonical.registry_identity,
            }
        )

    def identity_payload(self) -> Mapping[str, Any]:
        """Return the constant-size identity used by hot snapshot hashing.

        The fields were derived from validated writes.  Record-byte integrity
        is deliberately rechecked by transport, checkpoint, pickle, and cold
        replay boundaries rather than on every snapshot fingerprint read.
        """

        return FrozenDict(
            {
                "schema_version": self.schema_version,
                "record_count": self.record_count,
                "component_fingerprint": self.component_fingerprint,
                "current_view_fingerprint": self.current_view_fingerprint,
                "asof": to_primitive(self.asof),
                "foundation_version": self.foundation_version,
                "registry_identity": self.registry_identity,
            }
        )


def _projection_digest(projection: FoundationProjection) -> str:
    _require_foundation_record_identity_integrity(projection.current_records)
    return _canonical_digest(
        {
            "schema_version": projection.schema_version,
            "foundation_version": projection.foundation_version,
            "registry_identity": projection.registry_identity,
            "asof": projection.asof.isoformat() if projection.asof is not None else None,
            "current_record_ids": tuple(
                record.record_id for record in projection.current_records
            ),
            "record_count": projection.record_count,
            "component_fingerprint": projection.component_fingerprint,
            "current_view_fingerprint": projection.current_view_fingerprint,
        }
    )


@dataclass(frozen=True)
class FoundationProjectionCheckpoint:
    projection: FoundationProjection
    schema_version: int = FOUNDATION_PROJECTION_CHECKPOINT_SCHEMA_VERSION
    checkpoint_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.projection, FoundationProjection)
            or self.schema_version != FOUNDATION_PROJECTION_CHECKPOINT_SCHEMA_VERSION
        ):
            raise TypeError("checkpoint requires a FoundationProjection")
        canonical = FoundationProjectionReducer.validate_complete(
            self.projection
        )
        object.__setattr__(self, "projection", canonical)
        object.__setattr__(
            self,
            "checkpoint_id",
            f"foundation-checkpoint:{_projection_digest(canonical)}",
        )

    def __getstate__(self) -> Mapping[str, Any]:
        canonical = FoundationProjectionReducer.validate_complete(
            self.projection
        )
        expected = f"foundation-checkpoint:{_projection_digest(canonical)}"
        if self.checkpoint_id != expected:
            raise ValueError("foundation checkpoint integrity mismatch")
        return {
            "projection": canonical,
            "schema_version": self.schema_version,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        expected = {"projection", "schema_version"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("foundation projection checkpoint schema changed")
        for name in expected:
            object.__setattr__(self, name, state[name])
        self.__post_init__()


@lru_cache(maxsize=8192)
def _record_from_immutable_dto(
    dto: object,
    source_event_ids: tuple[str, ...] | None,
) -> FoundationRecord:
    """Memoize deterministic DTO envelopes without entering any identity."""

    return FoundationRecord.from_dto(dto, source_event_ids=source_event_ids)


class FoundationProjectionReducer:
    """Cold pure replay operations over already-authoritative DTO records."""

    @staticmethod
    def initial_projection() -> FoundationProjection:
        return FoundationProjection()

    @staticmethod
    def record_from_dto(
        dto: object,
        *,
        source_event_ids: Sequence[str] | None = None,
    ) -> FoundationRecord:
        sources = (
            None if source_event_ids is None else tuple(source_event_ids)
        )
        try:
            hash((dto, sources))
        except TypeError:
            return FoundationRecord.from_dto(
                dto,
                source_event_ids=sources,
            )
        return _record_from_immutable_dto(dto, sources)

    @staticmethod
    def reduce(
        projection: FoundationProjection,
        record: FoundationRecord,
    ) -> FoundationProjection:
        if not isinstance(projection, FoundationProjection):
            raise TypeError("projection must be FoundationProjection")
        if not isinstance(record, FoundationRecord):
            raise TypeError("record must be FoundationRecord")
        owner = FoundationProjectionOwner._from_cold_prefix(
            projection,
            _capability=_FOUNDATION_COLD_PREFIX_CAPABILITY,
        )
        transaction = owner.stage()
        if not transaction.append(record):
            return projection
        frozen = transaction.freeze()
        # Pure replay permits an owner-first batch prefix to be temporarily
        # incomplete.  ``replay`` performs the full completeness check after
        # the complete suffix has been reduced.  The owner is private and
        # temporary, so no public commit authority is needed here.
        return frozen

    @classmethod
    def replay(
        cls,
        records: Sequence[FoundationRecord],
        *,
        initial: FoundationProjection | None = None,
    ) -> FoundationProjection:
        projection = (
            cls.initial_projection()
            if initial is None
            else cls.validate_complete(initial)
        )
        seen: dict[str, FoundationRecord] = {}
        for record in records:
            prior = seen.get(record.record_id)
            if prior is not None:
                if prior != record:
                    raise ValueError("foundation record_id collision")
                continue
            seen[record.record_id] = record
            projection = cls.reduce(projection, record)
        return cls.validate_complete(projection)

    @staticmethod
    def validate_complete(
        projection: FoundationProjection,
    ) -> FoundationProjection:
        """Validate graph completeness after an atomic owner-first batch."""

        if not isinstance(projection, FoundationProjection):
            raise TypeError("projection must be FoundationProjection")
        canonical = FoundationProjection(
            current_records=projection.current_records,
            record_count=projection.record_count,
            component_fingerprint=projection.component_fingerprint,
            current_view_fingerprint=projection.current_view_fingerprint,
            asof=projection.asof,
            foundation_version=projection.foundation_version,
            registry_identity=projection.registry_identity,
            schema_version=projection.schema_version,
        )
        _validate_projection_current_graph(canonical)
        return canonical

    @staticmethod
    def checkpoint(
        projection: FoundationProjection,
    ) -> FoundationProjectionCheckpoint:
        return FoundationProjectionCheckpoint(projection)

    @staticmethod
    def restore(
        checkpoint: FoundationProjectionCheckpoint,
    ) -> FoundationProjection:
        if not isinstance(checkpoint, FoundationProjectionCheckpoint):
            raise TypeError("restore requires FoundationProjectionCheckpoint")
        if checkpoint.schema_version != FOUNDATION_PROJECTION_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("foundation checkpoint schema changed")
        canonical = FoundationProjectionReducer.validate_complete(
            checkpoint.projection
        )
        expected = f"foundation-checkpoint:{_projection_digest(canonical)}"
        if checkpoint.checkpoint_id != expected:
            raise ValueError("foundation checkpoint integrity mismatch")
        return canonical


class FoundationProjectionOwner:
    """Single-writer mutable owner; snapshots never expose its containers."""

    def __init__(
        self,
        projection: FoundationProjection | None = None,
    ) -> None:
        admitted = projection or FoundationProjectionReducer.initial_projection()
        current = FoundationProjectionReducer.validate_complete(admitted)
        self._initialize(current)

    def _initialize(self, current: FoundationProjection) -> None:
        if not isinstance(current, FoundationProjection):
            raise TypeError("foundation hot owner requires a compact projection")
        self._latest = dict(current._latest_records_by_key_cache)
        self._current_record_ids = set(current._current_record_ids_cache)
        self._record_count = current.record_count
        self._component_fingerprint = current.component_fingerprint
        self._asof = current.asof
        self._geometry_views = dict(current._swing_geometry_views_cache)
        self._assignment_incumbents = dict(
            current._swing_assignment_incumbents_cache
        )
        self._interaction_ids_by_level = {
            level_id: set(interaction_ids)
            for level_id, interaction_ids
            in current._liquidity_interaction_ids_by_level_cache.items()
        }
        self._projection_cache = current
        self._generation = 0

    @classmethod
    def _from_cold_prefix(
        cls,
        projection: FoundationProjection,
        *,
        _capability: object,
    ) -> "FoundationProjectionOwner":
        """Build the private pure-replay owner for an incomplete prefix."""

        if (
            _capability is not _FOUNDATION_COLD_PREFIX_CAPABILITY
            or type(projection) is not FoundationProjection
        ):
            raise ValueError("foundation cold-prefix authority is invalid")
        owner = object.__new__(cls)
        owner._initialize(projection)
        return owner

    @staticmethod
    def _canonical_projection(
        projection: FoundationProjection,
    ) -> FoundationProjection:
        if not isinstance(projection, FoundationProjection):
            raise ValueError("foundation projection owner state is invalid")
        return FoundationProjectionReducer.validate_complete(projection)

    def _require_internal_integrity(self) -> FoundationProjection:
        canonical = self._canonical_projection(self._projection_cache)
        rebuilt = type(self)._from_cold_prefix(
            canonical,
            _capability=_FOUNDATION_COLD_PREFIX_CAPABILITY,
        )
        if (
            type(self._generation) is not int
            or self._generation < 0
            or self._latest != rebuilt._latest
            or self._current_record_ids != rebuilt._current_record_ids
            or self._record_count != rebuilt._record_count
            or self._component_fingerprint != rebuilt._component_fingerprint
            or self._asof != rebuilt._asof
            or self._geometry_views != rebuilt._geometry_views
            or self._assignment_incumbents != rebuilt._assignment_incumbents
            or self._interaction_ids_by_level
            != rebuilt._interaction_ids_by_level
        ):
            raise ValueError("foundation projection owner internals differ")
        return canonical

    def __getstate__(self) -> Mapping[str, Any]:
        return {
            "schema_version": FOUNDATION_PROJECTION_OWNER_STATE_SCHEMA_VERSION,
            "projection": self._require_internal_integrity(),
            "generation": self._generation,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        expected = {"schema_version", "projection", "generation"}
        if (
            not isinstance(state, Mapping)
            or set(state) != expected
            or state.get("schema_version")
            != FOUNDATION_PROJECTION_OWNER_STATE_SCHEMA_VERSION
            or type(state.get("generation")) is not int
            or state["generation"] < 0
        ):
            raise ValueError("foundation projection owner pickle schema changed")
        canonical = self._canonical_projection(state["projection"])
        self._initialize(canonical)
        self._generation = state["generation"]

    @property
    def record_count(self) -> int:
        return self._record_count

    @property
    def component_fingerprint(self) -> str:
        return self._component_fingerprint

    @property
    def generation(self) -> int:
        return self._generation

    def contains(self, record_id: str) -> bool:
        return record_id in self._current_record_ids

    def freeze(self) -> FoundationProjection:
        return self._projection_cache

    def stage(self) -> "FoundationProjectionTransaction":
        return FoundationProjectionTransaction(self)


class FoundationProjectionTransaction:
    """Ordinary bounded write-set over one committed hot owner."""

    def __init__(self, owner: FoundationProjectionOwner) -> None:
        if not isinstance(owner, FoundationProjectionOwner):
            raise TypeError("foundation projection transaction requires its owner")
        self._owner = owner
        self._base_generation = owner.generation
        self._records: list[FoundationRecord] = []
        self._pending_ids: set[str] = set()
        self._latest_writes: dict[
            tuple[FoundationObjectType, str], FoundationRecord
        ] = {}
        self._geometry_writes: dict[str, _SwingGeometryView] = {}
        self._assignment_writes: dict[str, FoundationRecord] = {}
        self._record_count = owner.record_count
        self._component_fingerprint = owner.component_fingerprint
        self._asof = owner._asof
        self._projection_cache: FoundationProjection | None = owner.freeze()
        self._preflight_projection: FoundationProjection | None = None
        self._preflight_binding: tuple[object, ...] | None = None
        self._closed = False

    def _require_fresh(self) -> None:
        if self._closed:
            raise ValueError("foundation projection transaction is closed")
        if self._owner.generation != self._base_generation:
            raise ValueError("foundation projection transaction is stale")

    def contains(self, record_id: str) -> bool:
        self._require_fresh()
        return record_id in self._pending_ids or self._owner.contains(record_id)

    def _require_write_set_identity(self) -> None:
        _require_foundation_record_identity_integrity(tuple(self._records))

    @staticmethod
    def _projection_binding(
        projection: FoundationProjection,
    ) -> tuple[object, ...]:
        return (
            projection.current_records,
            projection._latest_records_by_key_cache,
            projection._current_record_ids_cache,
            projection._swing_geometry_views_cache,
            projection._swing_assignment_incumbents_cache,
            projection._liquidity_interaction_ids_by_level_cache,
            projection._current_view_hash_cursor_cache,
            projection.record_count,
            projection.component_fingerprint,
            projection.current_view_fingerprint,
            projection.asof,
            projection.foundation_version,
            projection.registry_identity,
            projection.schema_version,
        )

    def _bind_preflight(self, projection: FoundationProjection) -> None:
        self._preflight_projection = projection
        self._preflight_binding = self._projection_binding(projection)

    def _require_bound_preflight(
        self,
        projection: FoundationProjection,
    ) -> None:
        current = self._projection_binding(projection)
        admitted = self._preflight_binding
        if (
            projection is not self._preflight_projection
            or admitted is None
            or any(
                current[index] is not admitted[index]
                for index in range(7)
            )
            or current[7:] != admitted[7:]
        ):
            raise ValueError(
                "foundation prevalidated projection is not bound to this transaction"
            )

    def _latest_view(self) -> Mapping[tuple[FoundationObjectType, str], FoundationRecord]:
        return ChainMap(self._latest_writes, self._owner._latest)

    def append(self, record: FoundationRecord) -> bool:
        self._require_fresh()
        if not isinstance(record, FoundationRecord):
            raise TypeError("foundation projection transaction requires a record")
        _require_foundation_record_identity_integrity((record,))
        if self.contains(record.record_id):
            return False
        if self._asof is not None and record.known_at < self._asof:
            raise ValueError("foundation records must be appended in knowledge order")
        latest = self._latest_view()
        key = (record.object_type, record.object_id)
        previous = latest.get(key)
        if previous is not None:
            _validate_object_revision(previous, record)
        _validate_record_cross_links(
            latest,
            record,
            swing_geometry_views=ChainMap(
                self._geometry_writes,
                self._owner._geometry_views,
            ),
            swing_assignment_incumbents=ChainMap(
                self._assignment_writes,
                self._owner._assignment_incumbents,
            ),
        )
        self._records.append(record)
        self._pending_ids.add(record.record_id)
        self._latest_writes[key] = record
        if record.object_type is FoundationObjectType.SWING_GEOMETRY_NODE:
            self._geometry_writes[record.object_id] = _swing_geometry_view(
                record
            )
        elif record.object_type is FoundationObjectType.SWING_GEOMETRY_ASSIGNMENT:
            child_id = str(record.payload.get("child_swing_id"))
            incumbent = self._assignment_writes.get(child_id)
            if incumbent is None:
                incumbent = self._owner._assignment_incumbents.get(child_id)
            if incumbent is None or (record.known_at, record.object_id) > (
                incumbent.known_at,
                incumbent.object_id,
            ):
                self._assignment_writes[child_id] = record
        self._record_count += 1
        self._component_fingerprint = _extend_foundation_component_fingerprint(
            self._component_fingerprint,
            record.record_id,
        )
        self._asof = record.known_at
        self._projection_cache = None
        self._preflight_projection = None
        self._preflight_binding = None
        return True

    def freeze(self) -> FoundationProjection:
        self._require_fresh()
        self._require_write_set_identity()
        if self._projection_cache is None:
            latest = dict(self._owner._latest)
            current_record_ids = set(self._owner._current_record_ids)
            current_records = self._owner.freeze().current_records
            view_cursor = (
                self._owner.freeze()._current_view_hash_cursor_cache
            )
            interaction_ids_by_level = {
                level_id: set(interaction_ids)
                for level_id, interaction_ids
                in self._owner._interaction_ids_by_level.items()
            }
            for record in self._records:
                key = (record.object_type, record.object_id)
                previous = latest.get(key)
                previous_was_last = bool(latest) and next(
                    reversed(latest)
                ) == key
                latest.pop(key, None)
                latest[key] = record
                if (
                    previous is not None
                    and previous.object_type
                    is FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION
                ):
                    previous_level_id = str(
                        previous.payload.get("level_id")
                    )
                    previous_ids = interaction_ids_by_level.get(
                        previous_level_id
                    )
                    if previous_ids is not None:
                        previous_ids.discard(previous.object_id)
                        if not previous_ids:
                            interaction_ids_by_level.pop(
                                previous_level_id,
                                None,
                            )
                if (
                    record.object_type
                    is FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION
                ):
                    interaction_ids_by_level.setdefault(
                        str(record.payload.get("level_id")),
                        set(),
                    ).add(record.object_id)
                if previous is None:
                    current_records = (*current_records, record)
                    view_cursor = view_cursor.append(record.record_id)
                elif previous_was_last:
                    current_records = (*current_records[:-1], record)
                    view_cursor = view_cursor.replace_last(record.record_id)
                else:
                    current_records = tuple(latest.values())
                    view_cursor = _FoundationCurrentViewHashCursor.rebuild(
                        tuple(
                            item.record_id for item in current_records
                        ),
                        _capability=_FOUNDATION_INCREMENTAL_CAPABILITY,
                        foundation_version=(
                            self._owner.freeze().foundation_version
                        ),
                        registry_identity=(
                            self._owner.freeze().registry_identity
                        ),
                    )
                if previous is not None:
                    current_record_ids.discard(previous.record_id)
                current_record_ids.add(record.record_id)
            geometry_views = dict(self._owner._geometry_views)
            geometry_views.update(self._geometry_writes)
            assignment_incumbents = dict(
                self._owner._assignment_incumbents
            )
            assignment_incumbents.update(self._assignment_writes)
            self._projection_cache = FoundationProjection._from_incremental_state(
                _capability=_FOUNDATION_INCREMENTAL_CAPABILITY,
                current_records=current_records,
                record_count=self._record_count,
                component_fingerprint=self._component_fingerprint,
                asof=self._asof,
                latest_records_by_key=latest,
                current_record_ids=current_record_ids,
                swing_geometry_views=geometry_views,
                swing_assignment_incumbents=assignment_incumbents,
                liquidity_interaction_ids_by_level={
                    level_id: frozenset(interaction_ids)
                    for level_id, interaction_ids
                    in interaction_ids_by_level.items()
                },
                current_view_hash_cursor=view_cursor,
                foundation_version=self._owner.freeze().foundation_version,
                registry_identity=self._owner.freeze().registry_identity,
            )
        return self._projection_cache

    def delta(self) -> "FoundationRecordDelta":
        self._require_fresh()
        self._require_write_set_identity()
        if self._preflight_projection is not None:
            self._require_bound_preflight(self._preflight_projection)
        start_count = self._owner.record_count
        return FoundationRecordDelta(
            start_count=start_count,
            end_count=self._record_count,
            start_fingerprint=self._owner.component_fingerprint,
            end_fingerprint=self._component_fingerprint,
            records=tuple(self._records),
        )

    def validate_complete(self) -> None:
        self._require_fresh()
        self._require_write_set_identity()
        self._validate_completeness()

    def _validate_completeness(self) -> None:
        latest = self._latest_view()
        affected_level_ids: set[str] = set()
        for record in self._records:
            if record.object_type is FoundationObjectType.LIQUIDITY_LEVEL:
                affected_level_ids.add(record.object_id)
            elif (
                record.object_type
                is FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION
            ):
                affected_level_ids.add(str(record.payload.get("level_id")))
                _validate_liquidity_interaction_registration(latest, record)
                previous = self._owner._latest.get(
                    (record.object_type, record.object_id)
                )
                if previous is not None:
                    affected_level_ids.add(
                        str(previous.payload.get("level_id"))
                    )
        for level_id in affected_level_ids:
            level = latest.get(
                (FoundationObjectType.LIQUIDITY_LEVEL, level_id)
            )
            if level is None:
                raise ValueError(
                    "finalized liquidity interaction lacks its registered level"
                )
            _validate_liquidity_level_interaction_graph(
                latest,
                level,
                tail_only=True,
            )

    def preflight_commit(self) -> FoundationProjection:
        """Validate every fallible condition before either authority mutates."""

        frozen = self.freeze()
        self._validate_completeness()
        self._bind_preflight(frozen)
        return frozen

    def commit(
        self,
        *,
        prevalidated: FoundationProjection | None = None,
    ) -> FoundationProjection:
        self._require_fresh()
        if prevalidated is None:
            frozen = self.preflight_commit()
        else:
            self._require_write_set_identity()
            frozen = prevalidated
            self._require_bound_preflight(frozen)
        return self._commit_prevalidated(frozen)

    def _commit_prevalidated(
        self,
        frozen: FoundationProjection,
    ) -> FoundationProjection:
        """Apply an already checked write-set without another fallible read.

        The adapter calls this only after projection, lifecycle, and cold-ledger
        preflight have all succeeded.  Keeping the mutation tail validation-free
        prevents a one-sided cold-ledger commit.
        """

        self._require_fresh()
        self._require_bound_preflight(frozen)

        for record in self._records:
            key = (record.object_type, record.object_id)
            prior = self._owner._latest.pop(key, None)
            if prior is not None:
                self._owner._current_record_ids.discard(prior.record_id)
                if (
                    prior.object_type
                    is FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION
                ):
                    previous_level_id = str(
                        prior.payload.get("level_id")
                    )
                    previous_ids = self._owner._interaction_ids_by_level.get(
                        previous_level_id
                    )
                    if previous_ids is not None:
                        previous_ids.discard(prior.object_id)
                        if not previous_ids:
                            self._owner._interaction_ids_by_level.pop(
                                previous_level_id,
                                None,
                            )
            self._owner._latest[key] = record
            self._owner._current_record_ids.add(record.record_id)
            if (
                record.object_type
                is FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION
            ):
                self._owner._interaction_ids_by_level.setdefault(
                    str(record.payload.get("level_id")),
                    set(),
                ).add(record.object_id)
        self._owner._geometry_views.update(self._geometry_writes)
        self._owner._assignment_incumbents.update(
            self._assignment_writes
        )
        self._owner._record_count = self._record_count
        self._owner._component_fingerprint = self._component_fingerprint
        self._owner._asof = self._asof
        self._owner._projection_cache = frozen
        self._owner._generation += 1
        self._closed = True
        return frozen


@dataclass(frozen=True)
class FoundationRecordDelta:
    """One bounded hot-to-cold transport suffix; never a history container."""

    start_count: int
    end_count: int
    start_fingerprint: str
    end_fingerprint: str
    records: tuple[FoundationRecord, ...] = ()
    schema_version: int = FOUNDATION_RECORD_DELTA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        records = tuple(self.records)
        object.__setattr__(self, "records", records)
        _require_foundation_record_identity_integrity(records)
        fingerprint = self.start_fingerprint
        for record in records:
            fingerprint = _extend_foundation_component_fingerprint(
                fingerprint,
                record.record_id,
            )
        if (
            self.schema_version != FOUNDATION_RECORD_DELTA_SCHEMA_VERSION
            or type(self.start_count) is not int
            or type(self.end_count) is not int
            or self.start_count < 0
            or self.end_count != self.start_count + len(records)
            or fingerprint != self.end_fingerprint
        ):
            raise ValueError("foundation record delta cursor is invalid")

    def __getstate__(self) -> Mapping[str, Any]:
        return {
            "start_count": self.start_count,
            "end_count": self.end_count,
            "start_fingerprint": self.start_fingerprint,
            "end_fingerprint": self.end_fingerprint,
            "records": self.records,
            "schema_version": self.schema_version,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        expected = {
            "start_count",
            "end_count",
            "start_fingerprint",
            "end_fingerprint",
            "records",
            "schema_version",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("foundation record delta schema changed")
        for name in expected:
            object.__setattr__(self, name, state[name])
        self.__post_init__()


@dataclass(frozen=True)
class FoundationRecordLedgerCheckpoint:
    """Explicit cold-history materialization for checkpoint/replay only."""

    records: tuple[FoundationRecord, ...]
    record_count: int
    component_fingerprint: str
    schema_version: int = FOUNDATION_RECORD_LEDGER_SCHEMA_VERSION
    checkpoint_id: str = field(init=False)

    def __post_init__(self) -> None:
        records = tuple(self.records)
        object.__setattr__(self, "records", records)
        replayed = FoundationProjectionReducer.replay(records)
        if (
            self.schema_version != FOUNDATION_RECORD_LEDGER_SCHEMA_VERSION
            or self.record_count != len(records)
            or replayed.record_count != self.record_count
            or replayed.component_fingerprint != self.component_fingerprint
        ):
            raise ValueError("foundation cold-ledger checkpoint is invalid")
        object.__setattr__(
            self,
            "checkpoint_id",
            "foundation-record-ledger:"
            + _canonical_digest(
                {
                    "schema_version": self.schema_version,
                    "record_count": self.record_count,
                    "component_fingerprint": self.component_fingerprint,
                    "record_ids": tuple(record.record_id for record in records),
                }
            ),
        )

    def __getstate__(self) -> Mapping[str, Any]:
        return {
            "records": self.records,
            "record_count": self.record_count,
            "component_fingerprint": self.component_fingerprint,
            "schema_version": self.schema_version,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        expected = {
            "records",
            "record_count",
            "component_fingerprint",
            "schema_version",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("foundation cold-ledger checkpoint schema changed")
        for name in expected:
            object.__setattr__(self, name, state[name])
        self.__post_init__()


class FoundationRecordLedger:
    """Single append-only cold owner of immutable Foundation revisions."""

    def __init__(self) -> None:
        self._records: list[FoundationRecord] = []
        self._records_by_id: dict[str, str] = {}
        self._component_fingerprint = _foundation_component_fingerprint_seed(
            foundation_version=FOUNDATION_VERSION,
            registry_identity=FOUNDATION_CANONICAL_IDENTITY,
        )

    def _require_derived_index_integrity(self) -> None:
        expected = {
            record.record_id: _foundation_record_fingerprint(record)
            for record in self._records
        }
        if self._records_by_id != expected:
            raise ValueError("foundation cold-ledger identity index differs")

    def __getstate__(self) -> Mapping[str, Any]:
        checkpoint = self.checkpoint()
        return {
            "schema_version": FOUNDATION_RECORD_LEDGER_SCHEMA_VERSION,
            "records": checkpoint.records,
            "record_count": checkpoint.record_count,
            "component_fingerprint": checkpoint.component_fingerprint,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema_version",
            "records",
            "record_count",
            "component_fingerprint",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != expected
            or state.get("schema_version")
            != FOUNDATION_RECORD_LEDGER_SCHEMA_VERSION
        ):
            raise ValueError("foundation cold-ledger pickle schema changed")
        checkpoint = FoundationRecordLedgerCheckpoint(
            records=state["records"],
            record_count=state["record_count"],
            component_fingerprint=state["component_fingerprint"],
            schema_version=state["schema_version"],
        )
        restored = type(self).restore(checkpoint)
        self.__dict__.update(restored.__dict__)

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def component_fingerprint(self) -> str:
        return self._component_fingerprint

    def contains(self, record_id: str) -> bool:
        return record_id in self._records_by_id

    def preview_append(
        self,
        records: Sequence[FoundationRecord],
    ) -> FoundationRecordDelta:
        """Validate a suffix and compute its cursor without mutating history."""

        batch = tuple(records)
        start_count = self.record_count
        start_fingerprint = self.component_fingerprint
        staged: dict[str, str] = {}
        last_clock = self._records[-1].known_at if self._records else None
        for record in batch:
            if not isinstance(record, FoundationRecord):
                raise TypeError("foundation cold ledger accepts only records")
            record_fingerprint = _foundation_record_fingerprint(record)
            previous = self._records_by_id.get(record.record_id) or staged.get(record.record_id)
            if previous is not None:
                if previous != record_fingerprint:
                    raise ValueError("foundation cold ledger record identity conflicts")
                raise ValueError("foundation cold ledger suffix repeats a record")
            if last_clock is not None and record.known_at < last_clock:
                raise ValueError("foundation cold ledger moved backwards in time")
            staged[record.record_id] = record_fingerprint
            last_clock = record.known_at
        fingerprint = start_fingerprint
        for record in batch:
            fingerprint = _extend_foundation_component_fingerprint(
                fingerprint,
                record.record_id,
            )
        return FoundationRecordDelta(
            start_count=start_count,
            end_count=start_count + len(batch),
            start_fingerprint=start_fingerprint,
            end_fingerprint=fingerprint,
            records=batch,
        )

    def commit_prevalidated(
        self,
        delta: FoundationRecordDelta,
    ) -> FoundationRecordDelta:
        """Append a previously validated suffix at its exact cold cursor."""

        if not isinstance(delta, FoundationRecordDelta):
            raise TypeError("foundation cold ledger commit requires a delta")
        if (
            delta.start_count != self.record_count
            or delta.start_fingerprint != self.component_fingerprint
        ):
            raise ValueError("foundation cold ledger prevalidated cursor is stale")
        self._records.extend(delta.records)
        self._records_by_id.update(
            (record.record_id, _foundation_record_fingerprint(record))
            for record in delta.records
        )
        self._component_fingerprint = delta.end_fingerprint
        return delta

    def append(self, records: Sequence[FoundationRecord]) -> FoundationRecordDelta:
        return self.commit_prevalidated(self.preview_append(records))

    def materialize(self) -> tuple[FoundationRecord, ...]:
        """Materialize history only at an explicit cold-reader boundary."""

        return tuple(self._records)

    def checkpoint(self) -> FoundationRecordLedgerCheckpoint:
        self._require_derived_index_integrity()
        return FoundationRecordLedgerCheckpoint(
            records=self.materialize(),
            record_count=self.record_count,
            component_fingerprint=self.component_fingerprint,
        )

    @classmethod
    def restore(
        cls,
        checkpoint: FoundationRecordLedgerCheckpoint,
    ) -> "FoundationRecordLedger":
        if not isinstance(checkpoint, FoundationRecordLedgerCheckpoint):
            raise TypeError("foundation cold-ledger restore requires its checkpoint")
        ledger = cls()
        ledger.append(checkpoint.records)
        if (
            ledger.record_count != checkpoint.record_count
            or ledger.component_fingerprint != checkpoint.component_fingerprint
        ):
            raise ValueError("foundation cold-ledger checkpoint differs on replay")
        return ledger


__all__ = [
    "FOUNDATION_PROJECTION_CHECKPOINT_SCHEMA_VERSION",
    "FOUNDATION_PROJECTION_STATE_SCHEMA_VERSION",
    "FOUNDATION_RECORD_DELTA_SCHEMA_VERSION",
    "FOUNDATION_RECORD_LEDGER_SCHEMA_VERSION",
    "FoundationObjectType",
    "FoundationProjection",
    "FoundationProjectionCheckpoint",
    "FoundationProjectionOwner",
    "FoundationProjectionReducer",
    "FoundationRecord",
    "FoundationRecordDelta",
    "FoundationRecordLedger",
    "FoundationRecordLedgerCheckpoint",
    "FoundationRecordStatus",
]
