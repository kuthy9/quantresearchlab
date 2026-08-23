"""Version-bound append-only projection for semantic-foundation DTOs.

This module is deliberately a projection envelope, not a detector, event
store, or second semantic reducer.  It accepts only the frozen foundation-v2
DTO types produced by the existing lifecycle, geometry, and zone modules and
retains every immutable revision for deterministic replay.
"""

from __future__ import annotations

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
    if current.object_type is FoundationObjectType.LIQUIDITY_LEVEL:
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


def _validate_record_cross_links(
    prior: Mapping[tuple[FoundationObjectType, str], FoundationRecord],
    record: FoundationRecord,
    *,
    swing_geometry_views: Mapping[str, "_SwingGeometryView"] | None = None,
    swing_assignment_incumbents: Mapping[str, FoundationRecord] | None = None,
) -> None:
    """Validate mandatory prior-object links at the append boundary.

    Same-clock references are valid only when the owner record was already
    appended.  This deliberately forbids replay from borrowing a future
    object merely because it appears later in the submitted history.
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
                payload.get("lifecycle")
                in {
                    StructureTransitionLifecycle.STARTED.value,
                    StructureTransitionLifecycle.FAILED.value,
                }
                and incumbent.payload.get("lifecycle")
                != StructureGenerationLifecycle.CONFIRMED.value
            ):
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
                require_live_confirmed=True,
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

        parent_cache: dict[str, _SwingGeometryView | None] = {}

        def canonical_parent(
            node: _SwingGeometryView,
        ) -> _SwingGeometryView | None:
            if node.object_id in parent_cache:
                return parent_cache[node.object_id]
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
            parent_cache[node.object_id] = best
            return best

        expected_parent = canonical_parent(child_view)
        expected_parent_id = (
            None if expected_parent is None else expected_parent.object_id
        )
        expected_depth = 0
        cursor = child_view
        seen: set[str] = set()
        while (parent := canonical_parent(cursor)) is not None:
            if parent.object_id in seen:
                raise ValueError("Swing geometry contains a parent cycle")
            seen.add(parent.object_id)
            expected_depth += 1
            cursor = parent
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
            prior,
            target_type,
            payload.get("object_id"),
            role="first-retest target",
        )
        if target_type in {
            FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE,
            FoundationObjectType.STRUCTURAL_RANGE,
            FoundationObjectType.LIQUIDITY_LEVEL,
        } and target.status is not FoundationRecordStatus.ACTIVE:
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


def _validate_projection_completeness(
    latest: Mapping[tuple[FoundationObjectType, str], FoundationRecord],
) -> None:
    """Reject a finalized graph with owner-first batch prefixes left orphaned."""

    for (object_type, _), interaction in latest.items():
        if object_type is not FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION:
            continue
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
        ):
            raise ValueError(
                "finalized liquidity interaction lacks its registered level"
            )


@dataclass(frozen=True)
class FoundationProjection:
    """Append-only record history plus deterministic current-state views."""

    records: tuple[FoundationRecord, ...] = ()
    asof: pd.Timestamp | None = None
    foundation_version: str = FOUNDATION_VERSION
    registry_identity: str = FOUNDATION_CANONICAL_IDENTITY
    _latest_records_cache: tuple[FoundationRecord, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        records = tuple(self.records)
        object.__setattr__(self, "records", records)
        if (
            self.foundation_version != FOUNDATION_VERSION
            or self.registry_identity != FOUNDATION_CANONICAL_IDENTITY
            or any(not isinstance(record, FoundationRecord) for record in records)
            or any(
                record.foundation_version != self.foundation_version
                or record.registry_identity != self.registry_identity
                for record in records
            )
            or len({record.record_id for record in records}) != len(records)
            or any(
                right.known_at < left.known_at
                for left, right in zip(records, records[1:])
            )
        ):
            raise ValueError("foundation projection history is invalid")
        latest: dict[tuple[FoundationObjectType, str], FoundationRecord] = {}
        records_by_key: dict[
            tuple[FoundationObjectType, str],
            list[FoundationRecord],
        ] = {}
        geometry_views: dict[str, _SwingGeometryView] = {}
        assignment_incumbents: dict[str, FoundationRecord] = {}
        component_fingerprint = _foundation_component_fingerprint_seed(
            foundation_version=self.foundation_version,
            registry_identity=self.registry_identity,
        )
        for record in records:
            component_fingerprint = _extend_foundation_component_fingerprint(
                component_fingerprint,
                record.record_id,
            )
            key = (record.object_type, record.object_id)
            records_by_key.setdefault(key, []).append(record)
            previous = latest.get(key)
            if previous is not None:
                _validate_object_revision(previous, record)
                del latest[key]
            _validate_record_cross_links(
                latest,
                record,
                swing_geometry_views=geometry_views,
                swing_assignment_incumbents=assignment_incumbents,
            )
            latest[key] = record
            if record.object_type is FoundationObjectType.SWING_GEOMETRY_NODE:
                geometry_views[record.object_id] = _swing_geometry_view(record)
            elif (
                record.object_type
                is FoundationObjectType.SWING_GEOMETRY_ASSIGNMENT
            ):
                child_id = str(record.payload.get("child_swing_id"))
                incumbent = assignment_incumbents.get(child_id)
                if incumbent is None or (
                    record.known_at,
                    record.object_id,
                ) > (incumbent.known_at, incumbent.object_id):
                    assignment_incumbents[child_id] = record
        _validate_projection_completeness(latest)
        object.__setattr__(self, "_latest_records_cache", tuple(latest.values()))
        object.__setattr__(
            self,
            "_latest_records_by_key_cache",
            MappingProxyType(latest),
        )
        object.__setattr__(
            self,
            "_record_ids_cache",
            frozenset(record.record_id for record in records),
        )
        immutable_records_by_key = {
            key: tuple(history)
            for key, history in records_by_key.items()
        }
        object.__setattr__(
            self,
            "_records_by_key_cache",
            MappingProxyType(immutable_records_by_key),
        )
        object.__setattr__(
            self,
            "_first_records_by_key_cache",
            MappingProxyType(
                {
                    key: history[0]
                    for key, history in immutable_records_by_key.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "_component_fingerprint_cache",
            component_fingerprint,
        )
        object.__setattr__(
            self,
            "_swing_geometry_views_cache",
            MappingProxyType(geometry_views),
        )
        object.__setattr__(
            self,
            "_swing_assignment_incumbents_cache",
            MappingProxyType(assignment_incumbents),
        )
        expected_asof = records[-1].known_at if records else None
        if self.asof is None:
            object.__setattr__(self, "asof", expected_asof)
        else:
            asof = aware_timestamp(self.asof, name="foundation_projection.asof")
            object.__setattr__(self, "asof", asof)
            if asof != expected_asof:
                raise ValueError("foundation projection asof must equal final record clock")

    def __getstate__(self) -> Mapping[str, Any]:
        """Serialize canonical fields only; all indexes are derived caches."""

        return {
            "records": self.records,
            "asof": self.asof,
            "foundation_version": self.foundation_version,
            "registry_identity": self.registry_identity,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Rebuild ignored indexes and revalidate an unpickled history."""

        if not isinstance(state, Mapping):
            raise ValueError("foundation projection checkpoint state is invalid")
        for name in (
            "records",
            "asof",
            "foundation_version",
            "registry_identity",
        ):
            if name not in state:
                raise ValueError(
                    "foundation projection checkpoint state is incomplete"
                )
            object.__setattr__(self, name, state[name])
        self.__post_init__()

    def __deepcopy__(self, memo: dict[int, Any]) -> "FoundationProjection":
        """Reuse this value in adapter transactions because it is immutable."""

        memo[id(self)] = self
        return self

    @classmethod
    def _from_validated_append(
        cls,
        previous: "FoundationProjection",
        record: FoundationRecord,
    ) -> "FoundationProjection":
        """Append one already delta-validated record without replaying history.

        The public constructor remains the fail-closed boundary for arbitrary
        histories. ``FoundationProjectionReducer.reduce`` validates the one
        new edge before entering this internal path, so rechecking every prior
        immutable edge here would turn streaming projection into quadratic
        work without adding an integrity guarantee.
        """

        projection = object.__new__(cls)
        object.__setattr__(
            projection,
            "records",
            (*previous.records, record),
        )
        object.__setattr__(projection, "asof", record.known_at)
        object.__setattr__(
            projection,
            "foundation_version",
            previous.foundation_version,
        )
        object.__setattr__(
            projection,
            "registry_identity",
            previous.registry_identity,
        )
        key = (record.object_type, record.object_id)
        latest = dict(previous._latest_records_by_key_cache)
        latest.pop(key, None)
        latest[key] = record
        object.__setattr__(projection, "_latest_records_cache", tuple(latest.values()))
        object.__setattr__(
            projection,
            "_latest_records_by_key_cache",
            MappingProxyType(latest),
        )
        object.__setattr__(
            projection,
            "_record_ids_cache",
            previous._record_ids_cache | {record.record_id},
        )
        records_by_key = dict(previous._records_by_key_cache)
        records_by_key[key] = (
            *records_by_key.get(key, ()),
            record,
        )
        object.__setattr__(
            projection,
            "_records_by_key_cache",
            MappingProxyType(records_by_key),
        )
        first_records = previous._first_records_by_key_cache
        if key not in first_records:
            first_records_update = dict(first_records)
            first_records_update[key] = record
            first_records = MappingProxyType(first_records_update)
        object.__setattr__(
            projection,
            "_first_records_by_key_cache",
            first_records,
        )
        object.__setattr__(
            projection,
            "_component_fingerprint_cache",
            _extend_foundation_component_fingerprint(
                previous._component_fingerprint_cache,
                record.record_id,
            ),
        )
        geometry_views = previous._swing_geometry_views_cache
        if record.object_type is FoundationObjectType.SWING_GEOMETRY_NODE:
            geometry_views_update = dict(geometry_views)
            geometry_views_update[record.object_id] = _swing_geometry_view(record)
            geometry_views = MappingProxyType(geometry_views_update)
        object.__setattr__(
            projection,
            "_swing_geometry_views_cache",
            geometry_views,
        )
        assignment_incumbents = previous._swing_assignment_incumbents_cache
        if record.object_type is FoundationObjectType.SWING_GEOMETRY_ASSIGNMENT:
            child_id = str(record.payload.get("child_swing_id"))
            incumbent = assignment_incumbents.get(child_id)
            if incumbent is None or (
                record.known_at,
                record.object_id,
            ) > (incumbent.known_at, incumbent.object_id):
                assignment_update = dict(assignment_incumbents)
                assignment_update[child_id] = record
                assignment_incumbents = MappingProxyType(assignment_update)
        object.__setattr__(
            projection,
            "_swing_assignment_incumbents_cache",
            assignment_incumbents,
        )
        return projection

    @property
    def latest_records(self) -> tuple[FoundationRecord, ...]:
        return self._latest_records_cache

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
    def active_history(self) -> tuple[FoundationRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.status is FoundationRecordStatus.ACTIVE
        )

    @property
    def terminal_history(self) -> tuple[FoundationRecord, ...]:
        return tuple(
            record
            for record in self.records
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

    def records_for(
        self,
        object_type: FoundationObjectType,
        object_id: str,
    ) -> tuple[FoundationRecord, ...]:
        kind = FoundationObjectType(object_type)
        return self._records_by_key_cache.get((kind, object_id), ())

    def first_record_for(
        self,
        object_type: FoundationObjectType,
        object_id: str,
    ) -> FoundationRecord | None:
        """Return one object's immutable first revision in constant time."""

        kind = FoundationObjectType(object_type)
        return self._first_records_by_key_cache.get(
            (kind, object_id)
        )

    @property
    def component_fingerprint(self) -> str:
        """Versioned append-chain digest for downstream hot-path identity."""

        return self._component_fingerprint_cache


def _projection_digest(projection: FoundationProjection) -> str:
    return _canonical_digest(
        {
            "foundation_version": projection.foundation_version,
            "registry_identity": projection.registry_identity,
            "asof": projection.asof.isoformat() if projection.asof is not None else None,
            "record_ids": tuple(record.record_id for record in projection.records),
        }
    )


@dataclass(frozen=True)
class FoundationProjectionCheckpoint:
    projection: FoundationProjection
    checkpoint_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.projection, FoundationProjection):
            raise TypeError("checkpoint requires a FoundationProjection")
        object.__setattr__(
            self,
            "checkpoint_id",
            f"foundation-checkpoint:{_projection_digest(self.projection)}",
        )


@lru_cache(maxsize=8192)
def _record_from_immutable_dto(
    dto: object,
    source_event_ids: tuple[str, ...] | None,
) -> FoundationRecord:
    """Memoize deterministic DTO envelopes without entering any identity."""

    return FoundationRecord.from_dto(dto, source_event_ids=source_event_ids)


class FoundationProjectionReducer:
    """Pure append/replay operations over already-authoritative DTO records."""

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
        if record.record_id in projection._record_ids_cache:
            existing = next(
                item
                for item in reversed(projection.records)
                if item.record_id == record.record_id
            )
            if existing != record:
                raise ValueError("foundation record_id collision")
            return projection
        if projection.asof is not None and record.known_at < projection.asof:
            raise ValueError("foundation records must be appended in knowledge order")
        previous = projection._latest_records_by_key_cache.get(
            (record.object_type, record.object_id)
        )
        if previous is not None:
            _validate_object_revision(previous, record)
        _validate_record_cross_links(
            projection._latest_records_by_key_cache,
            record,
            swing_geometry_views=projection._swing_geometry_views_cache,
            swing_assignment_incumbents=(
                projection._swing_assignment_incumbents_cache
            ),
        )
        return FoundationProjection._from_validated_append(projection, record)

    @classmethod
    def replay(
        cls,
        records: Sequence[FoundationRecord],
        *,
        initial: FoundationProjection | None = None,
    ) -> FoundationProjection:
        projection = initial or cls.initial_projection()
        for record in records:
            projection = cls.reduce(projection, record)
        return cls.validate_complete(projection)

    @staticmethod
    def validate_complete(
        projection: FoundationProjection,
    ) -> FoundationProjection:
        """Validate graph completeness after an atomic owner-first batch."""

        if not isinstance(projection, FoundationProjection):
            raise TypeError("projection must be FoundationProjection")
        _validate_projection_completeness(
            projection._latest_records_by_key_cache
        )
        return projection

    @staticmethod
    def checkpoint(
        projection: FoundationProjection,
    ) -> FoundationProjectionCheckpoint:
        FoundationProjectionReducer.validate_complete(projection)
        return FoundationProjectionCheckpoint(projection)

    @staticmethod
    def restore(
        checkpoint: FoundationProjectionCheckpoint,
    ) -> FoundationProjection:
        if not isinstance(checkpoint, FoundationProjectionCheckpoint):
            raise TypeError("restore requires FoundationProjectionCheckpoint")
        expected = f"foundation-checkpoint:{_projection_digest(checkpoint.projection)}"
        if checkpoint.checkpoint_id != expected:
            raise ValueError("foundation checkpoint integrity mismatch")
        return checkpoint.projection


__all__ = [
    "FoundationObjectType",
    "FoundationProjection",
    "FoundationProjectionCheckpoint",
    "FoundationProjectionReducer",
    "FoundationRecord",
    "FoundationRecordStatus",
]
