"""Temporal market scene graph and explainable scale focus.

The graph is an in-memory semantic index over the already-causal primitive
reducers.  It does not detect trades and it never stores every candle.  A
node revision may close a lifecycle or resolve ambiguity, while every prior
revision normally remains available for decision-time reconstruction.  An
explicit calibration-only runtime mode may discard cold revisions after all
current-bar consumers have run; historical access then fails closed at the
reported retention floor.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, fields as dataclass_fields, replace
from enum import Enum
import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence
import weakref

import pandas as pd

from brain.core.brain_entry_sequence import brain_path_terminal_role
from eyes.core.displacement import DisplacementLifecycle

if TYPE_CHECKING:
    from brain.core.brain_entry_sequence import BrainInteractionView, BrainObservationView

from .model import (
    BALANCE_CLAIM_CONFIRMED,
    AuthorityLayer,
    BalanceContext,
    BOSLifecycle,
    DealingRangeLifecycle,
    DeliveryObstruction,
    Direction,
    DirectionalObstructionView,
    EntryLocationLifecycle,
    EventKind,
    FairValueGapLifecycle,
    GlobalConflictEvidence,
    GlobalConflictRole,
    GlobalMarketContext,
    LiquidityInventoryLifecycle,
    LiquidityPoolLifecycle,
    ManipulationLifecycle,
    MarketEpisodeState,
    MarketMode,
    MarketObservation,
    MarketEvent,
    NEUTRAL_MARKET_STATE_SCHEMA_VERSION,
    NeutralMarketState,
    OpenMarketThesis,
    OpenMarketThesisClaimRelation,
    OrderBlockLifecycle,
    PathSequenceLifecycle,
    Playbook,
    PlaybookPhase,
    ScaleRelation,
    ScaleRelationState,
    StructureLifecycle,
    SupportResistanceLifecycle,
    SwingLifecycle,
    ReacceptanceLifecycle,
    ThesisEvidenceState,
    Timeframe,
    aware_timestamp,
    to_primitive,
)
# The scale registry is a standalone contract owned by
# :mod:`shares.core.scale_registry`.  These names are re-exported here so a
# historical pickle that recorded ``shares.core.scene_graph.ScaleSpec`` resolves
# to the same class object.  New code imports them from `scale_registry`.
from .scale_registry import (  # noqa: F401
    ScaleRole,
    ScaleSpec,
    StructuralScale,
    parse_scale_specs,
    scale_registry_id,
)


class EvidenceStatus(str, Enum):
    CONFIRMED = "confirmed"
    FORMING = "forming"
    AMBIGUOUS = "ambiguous"
    CONFLICTING = "conflicting"
    UNKNOWN = "unknown"
    NOT_OBSERVED = "not_observed"
    INVALIDATED = "invalidated"


class LiquidityRole(str, Enum):
    CONTEXT_DRAW = "context_draw"
    INTERMEDIATE_LIQUIDITY = "intermediate_liquidity"
    PRIMARY_DELIVERABLE_TARGET = "primary_deliverable_target"
    TERMINAL_DRAW = "terminal_draw"


_IDENTITY_CACHE_MISS = object()
_CURRENT_DOL_INVENTORY_CACHE: dict[
    int,
    tuple[
        weakref.ReferenceType[MarketObservation],
        tuple[object, ...],
    ],
] = {}


def _identity_cache_get(
    cache: Mapping[int, tuple[weakref.ReferenceType[MarketObservation], object]],
    observation: MarketObservation,
) -> object:
    entry = cache.get(id(observation))
    if entry is None or entry[0]() is not observation:
        return _IDENTITY_CACHE_MISS
    return entry[1]


def _identity_cache_set(
    cache: dict[int, tuple[weakref.ReferenceType[MarketObservation], Any]],
    observation: MarketObservation,
    value: Any,
) -> Any:
    key = id(observation)

    def discard(reference: weakref.ReferenceType[MarketObservation]) -> None:
        incumbent = cache.get(key)
        if incumbent is not None and incumbent[0] is reference:
            cache.pop(key, None)

    try:
        reference = weakref.ref(observation, discard)
    except TypeError:
        # Compatibility observations used by adapters and research fixtures
        # may be immutable in practice without supporting weak references.
        # Bypass the optimization instead of retaining them strongly.
        return value
    cache[key] = (reference, value)
    return value


def current_dol_inventory(
    observation: MarketObservation,
) -> tuple[object, ...]:
    """Return the single exact DOL/target-map inventory consumer view."""

    cached = _identity_cache_get(
        _CURRENT_DOL_INVENTORY_CACHE,
        observation,
    )
    if cached is not _IDENTITY_CACHE_MISS:
        return cached
    result = tuple(
        item
        for item in observation.liquidity_inventory
        if item.lifecycle is LiquidityInventoryLifecycle.VISIBLE
        and item.confirmed_at <= observation.asof
    )
    return _identity_cache_set(
        _CURRENT_DOL_INVENTORY_CACHE,
        observation,
        result,
    )


def dol_level_terminal_sources(
    observation: MarketObservation,
    candidate_id: str,
) -> tuple[str, ...]:
    """Return exact current terminal ancestry for one DOL identity."""

    if not candidate_id:
        return ()
    legacy = next(
        (
            item
            for item in observation.liquidity_inventory
            if item.item_id == candidate_id
            and item.lifecycle is LiquidityInventoryLifecycle.CONSUMED
        ),
        None,
    )
    return () if legacy is None else tuple(legacy.source_ids)


class SceneEdgeKind(str, Enum):
    ANCHORS = "ANCHORS"
    LOCATED_AT = "LOCATED_AT"
    BREAKS = "BREAKS"
    CREATES = "CREATES"
    SWEEPS = "SWEEPS"
    RETURNS_TO = "RETURNS_TO"
    CONFIRMS = "CONFIRMS"
    CONTAINED_BY = "CONTAINED_BY"
    ALIGNS_WITH = "ALIGNS_WITH"
    OPPOSES = "OPPOSES"
    BLOCKS_PATH_TO = "BLOCKS_PATH_TO"
    SOURCED_FROM = "SOURCED_FROM"
    RESOLVES = "RESOLVES"
    PROMOTED_FROM = "PROMOTED_FROM"
    RESPONDS_TO = "RESPONDS_TO"
    PRECEDES = "PRECEDES"


SCENE_EDGE_LIFECYCLE_VOCAB = frozenset(
    {"active", "closed", "invalidated"}
)


_CAUSAL_PATH_RELATIONS = frozenset(
    {
        SceneEdgeKind.BREAKS,
        SceneEdgeKind.CREATES,
        SceneEdgeKind.SWEEPS,
        SceneEdgeKind.RETURNS_TO,
        SceneEdgeKind.CONFIRMS,
        SceneEdgeKind.SOURCED_FROM,
        SceneEdgeKind.RESOLVES,
    }
)
_ACTION_CONNECTIVITY_RELATIONS = frozenset(
    {
        SceneEdgeKind.ANCHORS,
        SceneEdgeKind.LOCATED_AT,
        *_CAUSAL_PATH_RELATIONS,
        SceneEdgeKind.CONTAINED_BY,
        SceneEdgeKind.ALIGNS_WITH,
        SceneEdgeKind.OPPOSES,
        SceneEdgeKind.PROMOTED_FROM,
    }
)


_HOT_STATE_KINDS = frozenset(
    {
        "swing",
        "swing_projection",
        "structure",
        "bos",
        "support_resistance",
        "liquidity_pool",
        "liquidity",
        "displacement",
        "fvg",
        "order_block",
        "range",
        "manipulation",
        "entry_location",
        "reacceptance",
        "micro_bos",
        "path_sequence",
        "candle_structure",
    }
)

_SCENE_EVENT_KIND_TO_NODE_KIND: Mapping[EventKind, str] = {
    EventKind.SWING_STATE: "swing",
    EventKind.STRUCTURE_STATE: "structure",
    EventKind.BOS_POST_BREAK_STATE: "bos_post_break",
    EventKind.STRUCTURE_BREAK: "bos",
    EventKind.STRUCTURE_BREAK_FAILED: "bos",
    EventKind.SUPPORT_RESISTANCE_STATE: "support_resistance",
    EventKind.LIQUIDITY_POOL_STATE: "liquidity_pool",
    EventKind.LIQUIDITY_RETIRED: "liquidity_retirement",
    EventKind.FVG_STATE: "fvg",
    EventKind.ORDER_BLOCK_STATE: "order_block",
    EventKind.DEALING_RANGE_STATE: "range",
    EventKind.MANIPULATION_STATE: "manipulation",
    EventKind.ENTRY_PATH_STATE: "path_sequence",
    EventKind.ENTRY_PATH_STEP: "path_step",
}

# This is the single vocabulary authority for Scene endpoint artifacts.  It is
# derived from the actual EventKind adapter and the few native graph-only node
# kinds, so artifact readers do not maintain a second guessed list.
SCENE_NODE_KIND_VOCAB = frozenset(
    {
        *(item.value for item in EventKind),
        *_SCENE_EVENT_KIND_TO_NODE_KIND.values(),
        *_HOT_STATE_KINDS,
        "manipulation_candidate",
        "resolution",
    }
)
SCENE_NODE_ROLE_VOCAB = frozenset(
    {
        *SCENE_NODE_KIND_VOCAB,
        *(item.value for item in LiquidityRole),
    }
)
SCENE_NODE_LIFECYCLE_VOCAB = frozenset(
    {
        *(
            item.value
            for lifecycle in (
                SwingLifecycle,
                StructureLifecycle,
                BOSLifecycle,
                SupportResistanceLifecycle,
                LiquidityPoolLifecycle,
                LiquidityInventoryLifecycle,
                FairValueGapLifecycle,
                OrderBlockLifecycle,
                DealingRangeLifecycle,
                ManipulationLifecycle,
                EntryLocationLifecycle,
                ReacceptanceLifecycle,
                PathSequenceLifecycle,
                DisplacementLifecycle,
            )
            for item in lifecycle
        ),
        "ambiguous",
        "completed",
        "conflicting",
        "observed",
        "terminal",
        "unknown",
    }
)


@dataclass(frozen=True)
class SceneNode:
    node_id: str
    kind: str
    timeframe: str
    structural_scale: StructuralScale
    direction: Direction | None
    formed_at: pd.Timestamp
    confirmed_at: pd.Timestamp | None
    observed_at: pd.Timestamp
    lifecycle: str
    price_bounds: tuple[float, float] | None
    invalidation_rule: str | None
    ambiguity_state: EvidenceStatus
    source_ids: tuple[str, ...] = ()
    entity_id: str | None = None
    liquidity_role: LiquidityRole | None = None
    resolution_reason: str | None = None
    semantic_attributes: tuple[tuple[str, str], ...] = ()
    descriptive_metrics: tuple[tuple[str, float], ...] = ()
    market_epoch_id: str = "epoch:0"
    revision_id: str = ""

    def __post_init__(self) -> None:
        for name in ("formed_at", "observed_at", "confirmed_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, aware_timestamp(value, name=f"scene_node.{name}"))
        object.__setattr__(self, "structural_scale", StructuralScale(self.structural_scale))
        object.__setattr__(self, "ambiguity_state", EvidenceStatus(self.ambiguity_state))
        if self.liquidity_role is not None:
            object.__setattr__(self, "liquidity_role", LiquidityRole(self.liquidity_role))
        object.__setattr__(self, "source_ids", tuple(dict.fromkeys(self.source_ids)))
        semantic_attributes = tuple(self.semantic_attributes)
        descriptive_metrics = tuple(
            (str(name), float(value))
            for name, value in self.descriptive_metrics
        )
        object.__setattr__(
            self, "semantic_attributes", semantic_attributes
        )
        object.__setattr__(
            self, "descriptive_metrics", descriptive_metrics
        )
        if (
            not self.node_id
            or not self.kind
            or not self.timeframe
            or not self.lifecycle
            or not self.market_epoch_id
            or self.observed_at < self.formed_at
            or (self.confirmed_at is not None and not self.formed_at <= self.confirmed_at <= self.observed_at)
            or any(not isinstance(value, str) or not value for value in self.source_ids)
            or len({name for name, _ in semantic_attributes})
            != len(semantic_attributes)
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(value, str)
                or not value
                for name, value in semantic_attributes
            )
            or len({name for name, _ in descriptive_metrics})
            != len(descriptive_metrics)
            or any(
                not name or not math.isfinite(value)
                for name, value in descriptive_metrics
            )
            or (
                self.entity_id is not None
                and (
                    not isinstance(self.entity_id, str)
                    or not self.entity_id
                )
            )
        ):
            raise ValueError("invalid scene node identity or clock")
        if self.price_bounds is not None:
            lower, upper = self.price_bounds
            if not (
                math.isfinite(float(lower))
                and math.isfinite(float(upper))
                and 0.0 < float(lower) <= float(upper)
            ):
                raise ValueError("invalid scene node price bounds")


@dataclass(frozen=True)
class SceneEdge:
    edge_id: str
    source_node_id: str
    relation: SceneEdgeKind
    target_node_id: str
    observed_at: pd.Timestamp
    source_ids: tuple[str, ...]
    first_observed_at: pd.Timestamp | None = None
    lifecycle: str = "active"
    resolution_reason: str | None = None
    revision_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", SceneEdgeKind(self.relation))
        object.__setattr__(self, "observed_at", aware_timestamp(self.observed_at, name="scene_edge.observed_at"))
        first_observed_at = self.first_observed_at or self.observed_at
        object.__setattr__(
            self,
            "first_observed_at",
            aware_timestamp(
                first_observed_at,
                name="scene_edge.first_observed_at",
            ),
        )
        object.__setattr__(self, "source_ids", tuple(dict.fromkeys(self.source_ids)))
        if (
            not self.edge_id
            or not self.source_node_id
            or not self.target_node_id
            or self.source_node_id == self.target_node_id
            or self.lifecycle not in SCENE_EDGE_LIFECYCLE_VOCAB
            or (self.lifecycle == "active") == bool(self.resolution_reason)
            or self.observed_at < self.first_observed_at
            or (
                self.lifecycle == "active"
                and self.observed_at != self.first_observed_at
            )
        ):
            raise ValueError("invalid scene edge")


@dataclass(frozen=True)
class SceneGraphDelta:
    asof: pd.Timestamp
    revision_id: str
    added_node_ids: tuple[str, ...] = ()
    revised_node_ids: tuple[str, ...] = ()
    added_edge_ids: tuple[str, ...] = ()
    revised_edge_ids: tuple[str, ...] = ()
    resolution_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof", aware_timestamp(self.asof, name="scene_delta.asof"))
        for name in (
            "added_node_ids",
            "revised_node_ids",
            "added_edge_ids",
            "revised_edge_ids",
            "resolution_event_ids",
        ):
            values = tuple(getattr(self, name))
            # Multiple existing edges can be revised while adapting one
            # completed clock.  Their identity set has no causal sequence,
            # and the upstream traversal can originate from hash-backed
            # indexes.  Canonicalize it here so a process restart with a
            # different PYTHONHASHSEED cannot change Observation parity.
            if name == "revised_edge_ids":
                values = tuple(sorted(values))
            object.__setattr__(self, name, values)
            if len(values) != len(set(values)):
                raise ValueError("scene delta contains duplicate identities")

    @property
    def changed(self) -> bool:
        return any(
            (
                self.added_node_ids,
                self.revised_node_ids,
                self.added_edge_ids,
                self.revised_edge_ids,
                self.resolution_event_ids,
            )
        )


@dataclass(frozen=True)
class FocusState:
    asof: pd.Timestamp
    primary_timeframes: tuple[str, ...]
    reason_codes: tuple[str, ...]
    question: str
    resolution_status: EvidenceStatus
    hypothesis_id: str | None = None
    switched: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof", aware_timestamp(self.asof, name="focus.asof"))
        object.__setattr__(self, "resolution_status", EvidenceStatus(self.resolution_status))
        for name in (
            "primary_timeframes",
            "reason_codes",
        ):
            values = tuple(dict.fromkeys(getattr(self, name)))
            object.__setattr__(self, name, values)
        if (
            not self.primary_timeframes
            or not self.reason_codes
            or not self.question
            or type(self.switched) is not bool
            or self.hypothesis_id == ""
        ):
            raise ValueError("invalid focus state")
@dataclass(frozen=True)
class FocusedObservation:
    asof: pd.Timestamp
    scene_revision_id: str
    focus_state: FocusState
    nodes: tuple[SceneNode, ...]
    edges: tuple[SceneEdge, ...]
    graph_paths: tuple[tuple[str, ...], ...]
    unresolved_ambiguities: tuple[str, ...]
    cross_scale_conflicts: tuple[str, ...]
    missing_evidence: Mapping[str, EvidenceStatus]

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof", aware_timestamp(self.asof, name="focused_observation.asof"))
        if self.focus_state.asof != self.asof or any(node.observed_at > self.asof for node in self.nodes):
            raise ValueError("focused observation contains a future or mismatched focus")
        if any(edge.observed_at > self.asof for edge in self.edges):
            raise ValueError("focused observation contains a future edge")


@dataclass(frozen=True)
class HypothesisState:
    hypothesis_id: str
    playbook: Playbook
    direction: Direction
    context_root_ids: tuple[str, ...]
    context_timeframe: str
    setup_timeframe: str
    trigger_timeframe: str
    sequence_stage: str
    next_expected_event: str | None
    supporting_graph_paths: tuple[tuple[str, ...], ...]
    material_conflict_ids: tuple[str, ...]
    missing_evidence: Mapping[str, EvidenceStatus]
    ambiguous_evidence: Mapping[str, EvidenceStatus]
    context_draw_id: str | None
    primary_target_id: str | None
    invalidation_id: str | None
    evidence_revision_id: str | None = None
    source_market_thesis_ids: tuple[str, ...] = ()
    playbook_match_strength: float = 0.0
    market_thesis_action_bound: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "context_root_ids", tuple(dict.fromkeys(self.context_root_ids)))
        object.__setattr__(
            self,
            "material_conflict_ids",
            tuple(dict.fromkeys(self.material_conflict_ids)),
        )
        object.__setattr__(
            self,
            "source_market_thesis_ids",
            tuple(dict.fromkeys(self.source_market_thesis_ids)),
        )
        for mapping_name in ("missing_evidence", "ambiguous_evidence"):
            normalized = {
                str(name): EvidenceStatus(value)
                for name, value in dict(getattr(self, mapping_name)).items()
            }
            object.__setattr__(self, mapping_name, normalized)
        if (
            not self.hypothesis_id
            or not self.context_timeframe
            or not self.setup_timeframe
            or not self.trigger_timeframe
            or not self.sequence_stage
            or len(self.context_root_ids) != len(set(self.context_root_ids))
            or any(not value for value in self.material_conflict_ids)
            or any(not value for value in self.source_market_thesis_ids)
            or not math.isfinite(float(self.playbook_match_strength))
            or not 0.0 <= float(self.playbook_match_strength) <= 1.0
            or type(self.market_thesis_action_bound) is not bool
            or (
                self.market_thesis_action_bound
                and not self.source_market_thesis_ids
            )
        ):
            raise ValueError("invalid competing hypothesis state")


def _value(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value)


def _clock(item: Any, *names: str, default: pd.Timestamp) -> pd.Timestamp:
    for name in names:
        value = getattr(item, name, None)
        if value is not None:
            return aware_timestamp(value, name=f"scene_source.{name}")
    return default


def _semantic_clock(item: Any, *, default: pd.Timestamp) -> pd.Timestamp:
    """Latest explicit evidence clock carried by a frozen primitive state."""

    scalar_names = (
        "ended_at",
        "terminal_at",
        "resolved_at",
        "retired_at",
        "reaccepted_at",
        "accepted_outside_at",
        "accepted_at",
        "rejected_at",
        "reentry_candidate_at",
        "reentry_failed_at",
        "departure_confirmed_at",
        "first_entered_at",
        "partial_at",
        "first_test_at",
        "rejected_at",
        "left_at",
        "reclaimed_at",
        "held_at",
        "closed_at",
        "censored_at",
        "invalidated_at",
        "mitigated_at",
        "failed_at",
        "broken_at",
        "formation_failed_at",
        "consumed_at",
        "targeted_at",
        "swept_at",
        "tested_at",
        "active_at",
        "balance_confirmed_at",
        "state_started_at",
        "last_updated_at",
        "metadata_observed_at",
        "confirmed_at",
        "formed_at",
        "pivot_end",
        "observed_at",
    )
    clocks = [
        aware_timestamp(value, name=f"scene_source.{name}")
        for name in scalar_names
        for value in (getattr(item, name, None),)
        if value is not None
    ]
    for name in ("touch_times", "source_candle_starts"):
        clocks.extend(
            aware_timestamp(value, name=f"scene_source.{name}")
            for value in (getattr(item, name, ()) or ())
            if isinstance(value, (pd.Timestamp, str))
        )
    steps = getattr(item, "steps", ()) or ()
    clocks.extend(
        aware_timestamp(step.observed_at, name="scene_source.steps")
        for step in steps
        if getattr(step, "observed_at", None) is not None
    )
    return max(clocks, default=default)


def _semantic_event_clocks(item: Any) -> tuple[tuple[str, Any], ...]:
    """Return only evidence clocks, never per-bar age/heartbeat clocks."""

    scalar_names = (
        "ended_at",
        "terminal_at",
        "resolved_at",
        "retired_at",
        "reaccepted_at",
        "accepted_outside_at",
        "accepted_at",
        "rejected_at",
        "reentry_candidate_at",
        "reentry_failed_at",
        "departure_confirmed_at",
        "first_entered_at",
        "partial_at",
        "first_test_at",
        "left_at",
        "reclaimed_at",
        "held_at",
        "closed_at",
        "censored_at",
        "invalidated_at",
        "mitigated_at",
        "failed_at",
        "broken_at",
        "formation_failed_at",
        "consumed_at",
        "targeted_at",
        "swept_at",
        "tested_at",
        "active_at",
        "balance_confirmed_at",
        "state_started_at",
        "confirmed_at",
        "formed_at",
        "pivot_end",
        "observed_at",
    )
    output: list[tuple[str, Any]] = []
    for name in scalar_names:
        value = getattr(item, name, None)
        if value is None:
            continue
        if isinstance(value, (pd.Timestamp, str)):
            value = aware_timestamp(value, name=f"scene_signature.{name}")
        output.append((name, value))
    for name in ("touch_times", "source_candle_starts"):
        values = tuple(getattr(item, name, ()) or ())
        if values:
            output.append(
                (
                    name,
                    tuple(
                        aware_timestamp(
                            value,
                            name=f"scene_signature.{name}",
                        )
                        if isinstance(value, (pd.Timestamp, str))
                        else value
                        for value in values
                    ),
                )
            )
    steps = tuple(getattr(item, "steps", ()) or ())
    if steps:
        output.append(
            (
                "step_clocks",
                tuple(
                    aware_timestamp(
                        step.observed_at,
                        name="scene_signature.steps",
                    )
                    for step in steps
                    if getattr(step, "observed_at", None) is not None
                ),
            )
        )
    return tuple(output)


def _invalidation_rule(kind: str, item: Any) -> str | None:
    explicit = getattr(item, "invalidation_rule", None)
    if explicit:
        return str(explicit)
    return {
        "swing": "close_beyond_confirmed_swing",
        "structure": "protected_swing_broken",
        "bos": "bos_confirmation_failed",
        "support_resistance": "close_beyond_frozen_zone",
        "liquidity_pool": "pool_consumed_or_accepted_outside",
        "fvg": "frozen_gap_invalidated",
        "order_block": "frozen_order_block_failed",
        "range": "close_breaks_frozen_range",
        "manipulation": "outside_acceptance_or_source_invalidated",
        "entry_location": "price_left_frozen_entry_zone",
        "reacceptance": "reacceptance_hold_failed",
        "micro_bos": "micro_structure_reference_failed",
        "path_sequence": "frozen_sequence_invalidated",
        "liquidity": "liquidity_consumed",
    }.get(kind)


def _entity_id(item: Any) -> tuple[str, str] | None:
    candidates = (
        ("swing", "swing_id"),
        ("structure", "structure_id"),
        ("support_resistance", "zone_id"),
        ("liquidity_pool", "pool_id"),
        ("fvg", "fvg_id"),
        ("order_block", "order_block_id"),
        ("range", "range_id"),
        ("manipulation", "manipulation_id"),
        ("entry_location", "location_id"),
        ("reacceptance", "reacceptance_id"),
        ("micro_bos", "reference_id"),
        ("path_sequence", "sequence_id"),
        ("liquidity", "item_id"),
        # ``MicroBOSReference`` contains both ``reference_id`` and the
        # upstream ``bos_id``.  Keep the generic BOS identity last so source
        # references cannot impersonate/revise the upstream BOS node.
        ("bos", "bos_id"),
    )
    for kind, name in candidates:
        identity = getattr(item, name, None)
        if identity:
            return kind, str(identity)
    return None


_LIGHTWEIGHT_STATE_FIELDS: Mapping[str, tuple[str, ...]] = {
    "swing": (
        "lifecycle", "side", "price", "pivot_start", "pivot_end",
        "confirmed_at", "relation", "prior_same_side_id", "broken_at",
        "failure_reason",
    ),
    "structure": (
        "lifecycle", "direction", "formed_at", "confirmed_at",
        "broken_at", "formation_failed_at", "latest_high_id",
        "latest_low_id", "protected_swing_id", "protected_price",
        "high_run", "low_run", "sequence_count", "failure_reason",
    ),
    "bos": (
        "lifecycle", "direction", "scope", "target_swing_id",
        "source_structure_id", "target_price", "pending_at", "resolved_at",
        "attempt_clocks", "failure_reason", "break_bar_id",
        "source_displacement_id", "mss_qualified", "post_break_state",
        "accepted_at", "rejected_at",
    ),
    "support_resistance": (
        "lifecycle", "side", "lower_bound", "upper_bound", "formed_at",
        "confirmed_at", "tested_at", "broken_at", "reaccepted_at",
        "retired_at", "member_swing_ids", "source_ids", "range_id",
        "source_zone_id", "touch_times", "total_touch_count",
        "transition_reason",
    ),
    "liquidity_pool": (
        "lifecycle", "side", "lower_bound", "upper_bound", "formed_at",
        "confirmed_at", "member_swing_ids", "touch_times",
        "total_touch_count", "swept_at", "sweep_extreme",
        "close_outside_on_sweep", "resolved_at", "resolution_reason",
    ),
    "fvg": (
        "lifecycle", "direction", "qualification", "source_displacement_id",
        "source_active_transition_id", "source_candle_ids",
        "source_candle_starts", "lower_bound", "upper_bound",
        "invalidation_price", "formed_at", "confirmed_at", "partial_at",
        "mitigated_at", "invalidated_at", "transition_reason",
    ),
    "order_block": (
        "lifecycle", "direction", "source_displacement_id",
        "source_active_transition_id", "source_bos_id",
        "source_bos_target_swing_id", "source_bos_structure_id",
        "source_bos_scope", "source_bos_break_bar_id", "anchor_candle_ids",
        "lower_bound", "upper_bound", "invalidation_price", "formed_at",
        "confirmed_at", "first_test_at", "mitigated_at", "failed_at",
        "transition_reason",
    ),
    "range": (
        "lifecycle", "lower_source_zone_id", "upper_source_zone_id",
        "lower_source_member_swing_ids", "upper_source_member_swing_ids",
        "formed_at", "balance_confirmed_at", "broken_at", "lower_bound",
        "upper_bound", "value_price", "transition_reason",
        "lower_touch_count", "upper_touch_count", "midpoint_crossings",
        "inside_close_fraction", "compression_ratio",
    ),
    "manipulation": (
        "lifecycle", "direction", "source_inventory_item_id",
        "source_kind", "source_range_id", "source_pool_id",
        "crossed_source_ids", "source_lower_bound", "source_upper_bound",
        "swept_at", "sweep_extreme", "reentry_candidate_at",
        "reentry_failed_at", "reaccepted_at", "accepted_outside_at",
        "censored_at", "outside_run", "inside_hold_bars",
        "transition_reason",
    ),
    "entry_location": (
        "lifecycle", "direction", "source_zone_kind", "source_zone_id",
        "source_displacement_id", "source_bos_id", "lower_bound",
        "upper_bound", "failure_boundary", "formed_at",
        "departure_confirmed_at", "first_entered_at", "rejected_at",
        "left_at", "transition_reason",
    ),
    "reacceptance": (
        "lifecycle", "context_id", "source_entity_id", "direction",
        "reference_price", "failure_boundary", "formed_at", "left_at",
        "reclaimed_at", "held_at", "failed_at", "censored_at",
        "hold_real_1m_bars", "transition_reason",
    ),
    "micro_bos": (
        "context_id", "expected_direction", "anchor_at", "bos_id",
        "bos_direction", "target_swing_id", "scope", "pending_at",
        "resolved_at", "relation", "outcome", "qualified",
    ),
    "path_sequence": (
        "lifecycle", "context_id", "direction", "formed_at", "steps",
        "ended_at", "transition_reason",
    ),
    "liquidity": (
        "lifecycle", "side", "kind", "price", "lower_bound",
        "upper_bound", "formed_at", "confirmed_at", "targeted_at",
        "consumed_at", "source_ids", "range_id", "source_zone_id",
        "transition_reason",
    ),
}


def _lightweight_state_token(
    kind: str,
    item: Any,
) -> tuple[Any, ...]:
    """Return direct semantic references without flattening retained state."""

    return tuple(
        getattr(item, name, None)
        for name in _LIGHTWEIGHT_STATE_FIELDS.get(
            kind,
            ("lifecycle", "formed_at", "confirmed_at", "state_started_at"),
        )
    )


def _source_ids(item: Any) -> tuple[str, ...]:
    names = (
        "source_ids",
        "coincident_source_ids",
        "crossed_source_ids",
        "member_swing_ids",
        "lower_source_member_swing_ids",
        "upper_source_member_swing_ids",
        "source_candle_starts",
        "source_candle_ids",
        "anchor_candle_ids",
    )
    output: list[str] = []
    for name in names:
        values = getattr(item, name, ()) or ()
        output.extend(str(value) for value in values if value is not None)
    scalar_names = (
        "prior_same_side_id",
        "latest_high_id",
        "latest_low_id",
        "protected_swing_id",
        "target_swing_id",
        "break_bar_id",
        "source_structure_id",
        "lower_source_zone_id",
        "upper_source_zone_id",
        "source_displacement_id",
        "source_active_transition_id",
        "source_displacement_seed_candle_id",
        "source_bos_id",
        "source_bos_target_swing_id",
        "source_bos_structure_id",
        "source_bos_break_bar_id",
        "source_zone_id",
        "source_pool_id",
        "source_range_id",
        "source_inventory_item_id",
        "source_id",
        "context_id",
        "source_entity_id",
        "source_event_id",
    )
    output.extend(
        str(value)
        for name in scalar_names
        for value in (getattr(item, name, None),)
        if value is not None
    )
    return tuple(dict.fromkeys(output))


def _field(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _semantic_attributes(item: Any) -> tuple[tuple[str, str], ...]:
    fields = {
        "qualification": "qualification",
        "scope": "scope",
        "source_bos_scope": "source_bos_scope",
        "source_structure_id": "source_structure_id",
        "post_break_state": "post_break_state",
        "mss_qualified": "mss_qualified",
        "source_bos_mss_qualified": "source_bos_mss_qualified",
        "source_kind": "source_kind",
        "source_timeframe": "source_timeframe",
        "structural_rank": "structural_rank",
        "is_protected_swing": "is_protected_swing",
        "protected_swing_id": "protected_swing_id",
        "lower_source_zone_id": "lower_source_zone_id",
        "upper_source_zone_id": "upper_source_zone_id",
        "zone_role": "zone_role",
        "resolved_side": "resolved_side",
        "close_outside_on_sweep": "close_outside_on_sweep",
        "deadline_elapsed": "deadline_elapsed",
        "outside_run_side": "outside_run_side",
        "reentry_candidate_at": "reentry_candidate_at",
        "reentry_failed_at": "reentry_failed_at",
        "deadline_at": "deadline_at",
        "censored_at": "censored_at",
        "context_kind": "context_kind",
        # Frozen schema-v2 MicroBOSReference compatibility only.  Canonical
        # MicroBreakFact records intentionally have neither field, so their
        # physical break cannot acquire Brain qualification in the Eye graph.
        "outcome": "outcome",
        "qualified": "qualified",
    }
    output: list[tuple[str, str]] = []
    for output_name, source_name in fields.items():
        value = _field(item, source_name)
        if value is None:
            continue
        if isinstance(value, Enum):
            value = value.value
        elif isinstance(value, bool):
            value = "true" if value else "false"
        else:
            value = str(value)
        if value:
            output.append((output_name, value))
    inventory_kind = _field(item, "kind")
    if inventory_kind in {
        "swing",
        "equal_highs",
        "equal_lows",
        "previous_session_high",
        "previous_session_low",
        "previous_day_high",
        "previous_day_low",
        "previous_week_high",
        "previous_week_low",
        "range_boundary",
    }:
        output.append(("inventory_kind", str(inventory_kind)))
    return tuple(sorted(output))


def _descriptive_metrics(item: Any) -> tuple[tuple[str, float], ...]:
    names = (
        "visibility_strength",
        "reaction_quality",
        "depletion_risk",
        "mean_overlap_ratio",
        "max_overlap_ratio",
        "mean_directional_clv",
        "width_atr",
        "max_fill_fraction",
        "break_distance_atr",
        "body_lower_bound",
        "body_upper_bound",
        "penetration_atr",
        "inside_hold_bars",
        "outside_run",
        "outside_completed_bars",
        "reentry_candidate_price",
        "lower_touch_count",
        "upper_touch_count",
        "midpoint_crossings",
        "inside_close_fraction",
        "compression_ratio",
    )
    output = []
    for name in names:
        value = _field(item, name)
        if value is None or isinstance(value, bool):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            output.append((name, numeric))
    return tuple(output)


def _bounds(item: Any) -> tuple[float, float] | None:
    lower = getattr(item, "lower_bound", None)
    upper = getattr(item, "upper_bound", None)
    if lower is None or upper is None:
        source_lower = getattr(item, "source_lower_bound", None)
        source_upper = getattr(item, "source_upper_bound", None)
        sweep_extreme = getattr(item, "sweep_extreme", None)
        if source_lower is not None and source_upper is not None:
            values = [float(source_lower), float(source_upper)]
            if sweep_extreme is not None:
                values.append(float(sweep_extreme))
            lower, upper = min(values), max(values)
    if lower is None or upper is None:
        price = getattr(item, "price", None)
        if price is None:
            price = getattr(item, "target_price", None)
        if price is None:
            price = getattr(item, "protected_price", None)
        if price is None:
            return None
        lower = upper = price
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in (lower, upper)):
        return None
    return float(lower), float(upper)


def _direction(item: Any) -> Direction | None:
    value = getattr(item, "direction", None)
    if value is None:
        value = getattr(item, "expected_direction", None)
    if isinstance(value, Direction):
        return value
    side = getattr(item, "side", None)
    if side == "above":
        return Direction.LONG
    if side == "below":
        return Direction.SHORT
    return None


def _scale(kind: str, timeframe: str, item: Any) -> StructuralScale:
    if kind == "swing":
        # Every raw confirmed swing is internal.  Intermediate/external
        # structure is represented by additional source-connected projection
        # nodes, never by relabelling this original fact.
        return StructuralScale.INTERNAL
    if kind == "bos" and _value(getattr(item, "scope", "")) == "local":
        return StructuralScale.INTERNAL
    if kind in {"liquidity", "support_resistance"} and (
        str(getattr(item, "structural_rank", "")) == "external"
        or (
            kind == "liquidity"
            and str(getattr(item, "kind", "")).startswith("previous_")
        )
    ):
        return StructuralScale.EXTERNAL
    if timeframe in {Timeframe.H4.value, Timeframe.H1.value}:
        return StructuralScale.EXTERNAL
    if timeframe == Timeframe.M15.value:
        return StructuralScale.INTERMEDIATE
    return StructuralScale.INTERNAL


def _status(kind: str, lifecycle: str) -> EvidenceStatus:
    forming = {"forming", "started", "pending", "approaching"}
    invalid = {
        "broken",
        "failed",
        "invalidated",
        "formation_failed",
        "accepted_outside",
        "left",
        "censored",
        "consumed",
        "retired",
        "exhausted",
    }
    if lifecycle == "ambiguous":
        return EvidenceStatus.AMBIGUOUS
    if lifecycle == "unknown":
        return EvidenceStatus.UNKNOWN
    if lifecycle == "conflicting":
        return EvidenceStatus.CONFLICTING
    if lifecycle in invalid:
        return EvidenceStatus.INVALIDATED
    if kind == "manipulation" and lifecycle == "swept":
        return EvidenceStatus.FORMING
    if lifecycle in forming:
        return EvidenceStatus.FORMING
    return EvidenceStatus.CONFIRMED


def _is_terminal(kind: str, lifecycle: str) -> bool:
    return lifecycle in {
        "swing": {"broken", "formation_failed"},
        "structure": {"broken", "formation_failed"},
        "bos": {"failed"},
        "support_resistance": {"reaccepted", "retired"},
        "liquidity_pool": {"accepted", "rejected"},
        "liquidity": {"consumed", "retired"},
        "fvg": {"mitigated", "invalidated"},
        "order_block": {"mitigated", "failed"},
        "range": {"broken"},
        "manipulation": {"reaccepted", "accepted_outside", "censored"},
        # ``rejected`` records a reaction at the frozen zone; Group 5 can
        # subsequently observe that the same location was left.  Only LEFT is
        # terminal for the location identity.
        "entry_location": {"left"},
        "reacceptance": {"held", "failed", "censored"},
        "path_sequence": {"closed", "censored"},
        "displacement": {"exhausted", "censored"},
        "swing_projection": {"broken", "censored"},
    }.get(kind, {"invalidated", "failed", "censored", "completed"})


def _cross_scale_conflict_role(
    edge: SceneEdge,
    node_by_id: Mapping[str, SceneNode],
    *,
    ready_timeframes: set[str] | frozenset[str] | None = None,
) -> GlobalConflictRole | None:
    """Classify an OPPOSES fact without promoting local delivery to authority."""

    if edge.relation is not SceneEdgeKind.OPPOSES or edge.lifecycle != "active":
        return None
    source = node_by_id.get(edge.source_node_id)
    target = node_by_id.get(edge.target_node_id)
    if source is None or target is None:
        return None
    if ready_timeframes is not None and (
        source.timeframe not in ready_timeframes
        or target.timeframe not in ready_timeframes
    ):
        # Keep the raw OPPOSES fact in the graph, but do not give an
        # under-warmed scale authority to contradict a completed semantic
        # context.  Its interpretation remains explicitly UNKNOWN below.
        return None
    if (
        source.timeframe == target.timeframe
        or source.kind not in {"structure", "bos", "displacement"}
        or target.kind != "structure"
        or _is_terminal(source.kind, source.lifecycle)
        or _is_terminal(target.kind, target.lifecycle)
        or source.direction is None
        or target.direction is None
        or source.direction is target.direction
    ):
        return None
    if (
        source.ambiguity_state is not EvidenceStatus.CONFIRMED
        or target.ambiguity_state is not EvidenceStatus.CONFIRMED
    ):
        return None
    # A displacement or an opposite structure without accepted BOS evidence
    # is a real local opposition fact, not an authority transfer.
    if source.kind == "displacement":
        return GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY
    attributes = dict(source.semantic_attributes)
    if source.kind != "bos":
        return GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY
    if attributes.get("post_break_state") != "accepted":
        return GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY
    if (
        source.timeframe == Timeframe.H1.value
        and source.structural_scale in {
        StructuralScale.INTERMEDIATE,
        StructuralScale.EXTERNAL,
        }
    ):
        return GlobalConflictRole.AUTHORITY_TRANSITION_CANDIDATE
    return GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY


class TemporalMarketSceneGraph:
    """Bounded-query, append-only semantic graph over completed-bar facts."""

    _PERSISTENT_RELATIONS = {
        SceneEdgeKind.ANCHORS,
        SceneEdgeKind.LOCATED_AT,
        SceneEdgeKind.BREAKS,
        SceneEdgeKind.CREATES,
        SceneEdgeKind.SWEEPS,
        SceneEdgeKind.RETURNS_TO,
        SceneEdgeKind.CONFIRMS,
        SceneEdgeKind.SOURCED_FROM,
        SceneEdgeKind.RESOLVES,
        SceneEdgeKind.PROMOTED_FROM,
        SceneEdgeKind.RESPONDS_TO,
        SceneEdgeKind.PRECEDES,
    }

    def __init__(self) -> None:
        self._nodes: dict[str, SceneNode] = {}
        self._node_revisions: dict[str, list[SceneNode]] = defaultdict(list)
        self._edges: dict[str, SceneEdge] = {}
        self._edge_revisions: dict[str, list[SceneEdge]] = defaultdict(list)
        # Path obstruction is a current-price view, not an append-only
        # causal fact.  Rebuild these edges on every observation and keep
        # them out of revision/history storage.
        self._current_path_block_edges: dict[str, SceneEdge] = {}
        self._outgoing: dict[str, set[str]] = defaultdict(set)
        self._incoming: dict[str, set[str]] = defaultdict(set)
        self._source_index: dict[str, set[str]] = defaultdict(set)
        self._entity_index: dict[str, set[str]] = defaultdict(set)
        self._revision = 0
        self._revision_id = "scene:empty"
        self._last_asof: pd.Timestamp | None = None
        self._current_added_nodes: list[str] = []
        self._current_revised_nodes: list[str] = []
        self._current_added_edges: list[str] = []
        self._current_revised_edges: list[str] = []
        self._current_resolutions: list[str] = []
        self._last_price: float | None = None
        self._market_epoch_no = 0
        self._market_epoch_id = "epoch:0"
        self._epoch_boundaries: list[tuple[pd.Timestamp, str]] = []
        self._seen_market_event_ids: set[str] = set()
        self._seen_displacement_transition_ids: set[str] = set()
        self._state_catalog: dict[tuple[str, str, str], Any] = {}
        self._state_semantic_signatures: dict[str, tuple[Any, ...]] = {}
        self._state_lightweight_tokens: dict[str, tuple[Any, ...]] = {}
        self._current_epoch_node_ids: set[str] = set()
        self._current_epoch_node_ids_by_kind: dict[str, set[str]] = (
            defaultdict(set)
        )
        self._current_epoch_node_ids_by_timeframe: dict[
            str,
            set[str],
        ] = defaultdict(set)
        self._current_epoch_ambiguous_node_ids: set[str] = set()
        self._current_epoch_active_opposes_edge_ids: set[str] = set()
        self._current_global_material_conflict_ids: set[str] = set()
        self._visible_liquidity_node_ids_by_side: dict[str, set[str]] = {
            "above": set(),
            "below": set(),
        }
        self._recent_terminal_context_node_ids_by_kind: dict[
            str,
            deque[str],
        ] = defaultdict(deque)
        # Current-kind/context reads are repeated several times while one
        # immutable graph revision is interpreted by GlobalMarketContext and
        # the Brain.  Cache only those bounded hot views; every semantic graph
        # mutation advances ``_revision`` and clears the cache below.
        self._current_kind_nodes_cache: dict[
            tuple[int, str, str, bool],
            tuple[SceneNode, ...],
        ] = {}
        # Neighbor traversals dominate the current Brain interpretation hot
        # path.  Cache only the current graph view for the duration of one
        # fully derived update; historical ``asof`` queries remain uncached.
        # Permanent revisions and the current-price BLOCKS_PATH_TO refresh
        # both invalidate this derived view below.
        self._current_neighbors_cache: dict[
            str,
            tuple[tuple[str, SceneEdge], ...],
        ] = {}
        # ``None`` means the graph still owns its complete in-process
        # history.  Calibration-only replay may explicitly compact cold
        # history after every consumer has read the current Engine snapshot.
        # Historical queries older than this floor must then fail closed;
        # returning an incomplete reconstruction would be worse than raising.
        self._history_retention_floor: pd.Timestamp | None = None

    @property
    def revision_id(self) -> str:
        return self._revision_id

    @property
    def last_asof(self) -> pd.Timestamp | None:
        return self._last_asof

    @property
    def history_retention_floor(self) -> pd.Timestamp | None:
        return self._history_retention_floor

    @property
    def nodes(self) -> tuple[SceneNode, ...]:
        return tuple(sorted(self._nodes.values(), key=lambda item: (item.observed_at, item.node_id)))

    @property
    def edges(self) -> tuple[SceneEdge, ...]:
        return tuple(
            sorted(
                (
                    *self._edges.values(),
                    *self._current_path_block_edges.values(),
                ),
                key=lambda item: (item.observed_at, item.edge_id),
            )
        )

    def nodes_asof(self, asof: pd.Timestamp) -> tuple[SceneNode, ...]:
        clock = aware_timestamp(asof, name="scene_graph.nodes_asof")
        self._require_retained_history(clock)
        if self._last_asof is not None and clock >= self._last_asof:
            # Dict insertion order is deterministic because updates are
            # causally ordered before admission.  Current-time queries do not
            # need the public presentation sort, and this path is called many
            # times inside one minute while relations are derived.
            return tuple(self._nodes.values())
        values = []
        for revisions in self._node_revisions.values():
            eligible = [item for item in revisions if item.observed_at <= clock]
            if eligible:
                values.append(eligible[-1])
        return tuple(
            sorted(values, key=lambda item: (item.observed_at, item.node_id))
        )

    def edges_asof(self, asof: pd.Timestamp) -> tuple[SceneEdge, ...]:
        clock = aware_timestamp(asof, name="scene_graph.edges_asof")
        self._require_retained_history(clock)
        if self._last_asof is not None and clock >= self._last_asof:
            return (
                *self._edges.values(),
                *self._current_path_block_edges.values(),
            )
        values = []
        for revisions in self._edge_revisions.values():
            eligible = [item for item in revisions if item.observed_at <= clock]
            if eligible:
                values.append(eligible[-1])
        return tuple(
            sorted(values, key=lambda item: (item.observed_at, item.edge_id))
        )

    def node_history(self, node_id: str) -> tuple[SceneNode, ...]:
        if self._history_retention_floor is not None:
            raise ValueError(
                "complete node history is unavailable after runtime compaction"
            )
        return tuple(self._node_revisions.get(node_id, ()))

    def edge_history(self, edge_id: str) -> tuple[SceneEdge, ...]:
        if self._history_retention_floor is not None:
            raise ValueError(
                "complete edge history is unavailable after runtime compaction"
            )
        return tuple(self._edge_revisions.get(edge_id, ()))

    def _next_revision(self) -> str:
        self._revision += 1
        self._revision_id = f"scene:r{self._revision:012d}"
        # Old checkpoints predate this derived cache, hence ``getattr``.
        # The cache is never authoritative and must not survive a revision.
        getattr(self, "_current_kind_nodes_cache", {}).clear()
        getattr(self, "_current_neighbors_cache", {}).clear()
        return self._revision_id

    def _require_retained_history(self, asof: pd.Timestamp) -> None:
        floor = self._history_retention_floor
        if floor is not None and asof < floor:
            raise ValueError(
                "scene graph history predates the retained runtime window"
            )

    def _epoch_asof(self, asof: pd.Timestamp) -> str:
        self._require_retained_history(
            aware_timestamp(asof, name="scene_graph.epoch_asof")
        )
        epoch = "epoch:0"
        for boundary, candidate in self._epoch_boundaries:
            if boundary > asof:
                break
            epoch = candidate
        return epoch

    def _current_kind_nodes(self, kind: str) -> tuple[SceneNode, ...]:
        """Return current-epoch nodes of one kind without scanning history."""

        cache = getattr(self, "_current_kind_nodes_cache", None)
        if cache is None:
            cache = {}
            self._current_kind_nodes_cache = cache
        key = (self._revision, self._market_epoch_id, str(kind), False)
        cached = cache.get(key)
        if cached is not None:
            return cached
        values = tuple(
            sorted(
                (
                    self._nodes[node_id]
                    for node_id in self._current_epoch_node_ids_by_kind.get(
                        kind,
                        (),
                    )
                    if node_id in self._nodes
                    and self._nodes[node_id].market_epoch_id
                    == self._market_epoch_id
                ),
                key=lambda value: (value.observed_at, value.node_id),
            )
        )
        cache[key] = values
        return values

    def _current_nonterminal_kind_nodes(
        self,
        kind: str,
    ) -> tuple[SceneNode, ...]:
        """Return one cached nonterminal current-kind interpretation view."""

        cache = getattr(self, "_current_kind_nodes_cache", None)
        if cache is None:
            cache = {}
            self._current_kind_nodes_cache = cache
        key = (self._revision, self._market_epoch_id, str(kind), True)
        cached = cache.get(key)
        if cached is not None:
            return cached
        values = tuple(
            node
            for node in self._current_kind_nodes(kind)
            if node.market_epoch_id == self._market_epoch_id
            and not _is_terminal(node.kind, node.lifecycle)
        )
        cache[key] = values
        return values

    def _current_kind_nodes_asof(
        self,
        kind: str,
        asof: pd.Timestamp,
    ) -> tuple[SceneNode, ...]:
        """Resolve indexed current-epoch identities at an evidence clock."""

        clock = aware_timestamp(asof, name="scene_kind_index.asof")
        values: list[SceneNode] = []
        for node_id in self._current_epoch_node_ids_by_kind.get(kind, ()):
            revision = next(
                (
                    value
                    for value in reversed(
                        self._node_revisions.get(node_id, ())
                    )
                    if value.observed_at <= clock
                    and value.market_epoch_id == self._market_epoch_id
                ),
                None,
            )
            if revision is not None:
                values.append(revision)
        return tuple(
            sorted(
                values,
                key=lambda value: (value.observed_at, value.node_id),
            )
        )

    def _recent_terminal_context_nodes(
        self,
        kind: str,
        *,
        asof: pd.Timestamp,
        maximum_age: pd.Timedelta,
    ) -> tuple[SceneNode, ...]:
        """Return a bounded necessary-context view outside hot state."""

        clock = aware_timestamp(asof, name="scene_terminal_context.asof")
        identities = self._recent_terminal_context_node_ids_by_kind[kind]
        while identities:
            candidate = self._nodes.get(identities[0])
            if (
                candidate is not None
                and candidate.market_epoch_id == self._market_epoch_id
                and clock - candidate.observed_at <= maximum_age
            ):
                break
            identities.popleft()
        return tuple(
            self._nodes[node_id]
            for node_id in identities
            if node_id in self._nodes
            and self._nodes[node_id].observed_at <= clock
            and clock - self._nodes[node_id].observed_at <= maximum_age
        )

    def _reindex_current_node(
        self,
        node: SceneNode,
        *,
        prior: SceneNode | None,
    ) -> None:
        """Maintain the hot current-epoch view beside the cold revision ledger."""

        if prior is not None and prior.market_epoch_id == self._market_epoch_id:
            self._current_epoch_node_ids.discard(prior.node_id)
            self._current_epoch_node_ids_by_kind[prior.kind].discard(
                prior.node_id
            )
            self._current_epoch_node_ids_by_timeframe[
                prior.timeframe
            ].discard(prior.node_id)
            self._current_epoch_ambiguous_node_ids.discard(prior.node_id)
            if prior.kind == "liquidity":
                for values in self._visible_liquidity_node_ids_by_side.values():
                    values.discard(prior.node_id)
        if node.market_epoch_id != self._market_epoch_id:
            return
        if node.ambiguity_state is EvidenceStatus.AMBIGUOUS:
            # Ambiguous candidates are a live focus concern even when their
            # adapter kind is intentionally excluded from the hot graph view.
            self._current_epoch_ambiguous_node_ids.add(node.node_id)
        if node.kind not in _HOT_STATE_KINDS:
            return
        if _is_terminal(node.kind, node.lifecycle):
            # Terminal facts remain in the append-only node/revision ledger
            # and source indexes, but are not part of the hot current-state
            # indexes used by cross-scale and path computations.
            if (
                node.kind == "manipulation"
                and node.ambiguity_state
                not in {EvidenceStatus.INVALIDATED, EvidenceStatus.UNKNOWN}
                and node.node_id
                not in self._recent_terminal_context_node_ids_by_kind[
                    node.kind
                ]
            ):
                # A reaccepted manipulation remains necessary context for a
                # bounded subsequent reverse displacement.  Keep it outside
                # the generic hot-state indexes and expire it at lookup.
                self._recent_terminal_context_node_ids_by_kind[
                    node.kind
                ].append(node.node_id)
            return
        self._current_epoch_node_ids.add(node.node_id)
        self._current_epoch_node_ids_by_kind[node.kind].add(node.node_id)
        self._current_epoch_node_ids_by_timeframe[node.timeframe].add(
            node.node_id
        )
        if (
            node.kind == "liquidity"
            and node.lifecycle
            in {
                LiquidityInventoryLifecycle.VISIBLE.value,
                LiquidityInventoryLifecycle.TARGETED.value,
            }
            and node.price_bounds is not None
            and node.direction is not None
        ):
            side = "above" if node.direction is Direction.LONG else "below"
            self._visible_liquidity_node_ids_by_side[side].add(node.node_id)

    def _clear_current_epoch_indexes(self) -> None:
        getattr(self, "_current_kind_nodes_cache", {}).clear()
        getattr(self, "_current_neighbors_cache", {}).clear()
        self._current_epoch_node_ids = set()
        self._current_epoch_node_ids_by_kind = defaultdict(set)
        self._current_epoch_node_ids_by_timeframe = defaultdict(set)
        self._current_epoch_ambiguous_node_ids = set()
        self._current_epoch_active_opposes_edge_ids = set()
        self._current_global_material_conflict_ids = set()
        self._visible_liquidity_node_ids_by_side = {
            "above": set(),
            "below": set(),
        }
        self._state_semantic_signatures = {}
        self._state_lightweight_tokens = {}
        self._recent_terminal_context_node_ids_by_kind = defaultdict(deque)

    def _node_id_for_source(
        self,
        source_id: str,
        *,
        kind: str | None = None,
        asof: pd.Timestamp | None = None,
    ) -> str | None:
        source = str(source_id)
        entity_candidates = set(self._entity_index.get(source, set()))
        candidates = entity_candidates or set(
            self._source_index.get(source, set())
        )
        if source in self._node_revisions:
            candidates.add(source)
        clock = None if asof is None else aware_timestamp(asof, name="scene source lookup")
        if clock is not None:
            self._require_retained_history(clock)
        revisions: dict[str, SceneNode] = {}
        for candidate in candidates:
            eligible = [
                value
                for value in self._node_revisions.get(candidate, ())
                if clock is None or value.observed_at <= clock
            ]
            if eligible:
                revisions[candidate] = eligible[-1]
        if source in revisions and (
            kind is None or revisions[source].kind == kind
        ):
            return source
        if kind is not None:
            revisions = {
                identity: value
                for identity, value in revisions.items()
                if value.kind == kind
            }
        elif any(value.kind != "swing_projection" for value in revisions.values()):
            revisions = {
                identity: value
                for identity, value in revisions.items()
                if value.kind != "swing_projection"
            }
        if not revisions:
            return None
        return max(
            revisions,
            key=lambda value: (revisions[value].observed_at, value == source, value),
        )

    def add_node(self, node: SceneNode) -> SceneNode:
        prior = self._nodes.get(node.node_id)
        resolution_pair: tuple[SceneNode, SceneNode] | None = None
        if prior is not None:
            if (
                prior.kind != node.kind
                or prior.timeframe != node.timeframe
                or (
                    prior.structural_scale is not node.structural_scale
                    and not (
                        node.kind
                        in {"liquidity", "support_resistance"}
                        and not _is_terminal(
                            prior.kind,
                            prior.lifecycle,
                        )
                    )
                )
                or prior.entity_id != node.entity_id
                or (
                    prior.direction is not None
                    and prior.direction is not node.direction
                )
                or prior.formed_at != node.formed_at
                or (
                    prior.price_bounds is not None
                    and prior.price_bounds != node.price_bounds
                )
                or (
                    prior.invalidation_rule is not None
                    and prior.invalidation_rule != node.invalidation_rule
                )
                or (
                    prior.liquidity_role is not None
                    and prior.liquidity_role != node.liquidity_role
                )
                or prior.market_epoch_id != node.market_epoch_id
                or prior.observed_at > node.observed_at
                or (prior.confirmed_at is not None and node.confirmed_at != prior.confirmed_at)
                or (
                    _is_terminal(prior.kind, prior.lifecycle)
                    and node.lifecycle != prior.lifecycle
                )
                or not set(prior.source_ids).issubset(node.source_ids)
            ):
                raise ValueError(f"scene node attempted to rewrite frozen fact: {node.node_id}")
            # ``observed_at`` is the revision clock, not evidence by itself.
            # A retained primitive can be projected again when an upstream
            # source receives a heartbeat; do not turn that clock advance into
            # a new semantic graph revision when every frozen fact is equal.
            if replace(
                prior,
                observed_at=node.observed_at,
                revision_id="",
            ) == replace(node, revision_id=""):
                return prior
            if replace(prior, revision_id="") == replace(node, revision_id=""):
                return prior
            prior_status = prior.ambiguity_state
            revision_id = self._next_revision()
            node = replace(node, revision_id=revision_id)
            self._current_revised_nodes.append(node.node_id)
            if prior_status in {EvidenceStatus.AMBIGUOUS, EvidenceStatus.UNKNOWN, EvidenceStatus.CONFLICTING} and node.ambiguity_state not in {EvidenceStatus.AMBIGUOUS, EvidenceStatus.UNKNOWN, EvidenceStatus.CONFLICTING}:
                resolution_id = f"resolution:{node.node_id}:{revision_id}"
                self._current_resolutions.append(resolution_id)
                resolution_pair = (prior, node)
        else:
            revision_id = self._next_revision()
            node = replace(node, revision_id=revision_id)
            self._current_added_nodes.append(node.node_id)
        self._nodes[node.node_id] = node
        self._node_revisions[node.node_id].append(node)
        self._reindex_current_node(node, prior=prior)
        for source_id in (node.node_id, *node.source_ids):
            self._source_index[source_id].add(node.node_id)
        if node.entity_id is not None:
            self._entity_index[node.entity_id].add(node.node_id)
        if resolution_pair is not None:
            self._append_resolution_fact(*resolution_pair)
        if _is_terminal(node.kind, node.lifecycle):
            self._close_edges_for_node(
                node.node_id,
                observed_at=node.observed_at,
                reason=node.resolution_reason or f"endpoint_{node.lifecycle}",
            )
        return node

    def _add_frame_state_node(self, node: SceneNode) -> SceneNode:
        """Admit a native-frame snapshot without reopening a terminal fact."""

        prior = self._nodes.get(node.node_id)
        if (
            prior is not None
            and _is_terminal(prior.kind, prior.lifecycle)
            and not _is_terminal(node.kind, node.lifecycle)
        ):
            # Entity IDs are generation-specific.  Later touches/age updates
            # on the frozen formation view cannot reopen a pool, zone or
            # structure after its terminal ledger fact.
            return prior
        return self.add_node(node)

    def _append_resolution_fact(
        self,
        prior: SceneNode,
        resolved: SceneNode,
    ) -> None:
        resolution_id = (
            f"resolution:{resolved.node_id}:{resolved.revision_id}"
        )
        fact = self.add_node(
            SceneNode(
                node_id=resolution_id,
                kind="resolution",
                timeframe=resolved.timeframe,
                structural_scale=resolved.structural_scale,
                direction=resolved.direction,
                formed_at=resolved.observed_at,
                confirmed_at=resolved.observed_at,
                observed_at=resolved.observed_at,
                lifecycle="confirmed",
                price_bounds=resolved.price_bounds,
                invalidation_rule=None,
                ambiguity_state=EvidenceStatus.CONFIRMED,
                source_ids=(
                    resolved.node_id,
                    prior.revision_id,
                    resolved.revision_id,
                ),
                resolution_reason=(
                    resolved.resolution_reason
                    or f"{prior.ambiguity_state.value}_resolved"
                ),
                market_epoch_id=resolved.market_epoch_id,
            )
        )
        raw = f"{fact.node_id}|{SceneEdgeKind.RESOLVES.value}|{resolved.node_id}"
        self.add_edge(
            SceneEdge(
                edge_id=f"edge:{hashlib.sha256(raw.encode()).hexdigest()[:24]}",
                source_node_id=fact.node_id,
                relation=SceneEdgeKind.RESOLVES,
                target_node_id=resolved.node_id,
                observed_at=resolved.observed_at,
                source_ids=(prior.revision_id, resolved.revision_id),
            )
        )

    def _close_edges_for_node(
        self,
        node_id: str,
        *,
        observed_at: pd.Timestamp,
        reason: str,
    ) -> None:
        for edge_id in tuple(
            self._outgoing.get(node_id, set())
            | self._incoming.get(node_id, set())
        ):
            edge = self._edges[edge_id]
            if (
                edge.lifecycle != "active"
                or edge.relation in self._PERSISTENT_RELATIONS
            ):
                continue
            self.add_edge(
                replace(
                    edge,
                    observed_at=observed_at,
                    lifecycle="closed",
                    resolution_reason=reason,
                    revision_id="",
                )
            )

    def add_edge(self, edge: SceneEdge) -> SceneEdge:
        if edge.relation is SceneEdgeKind.BLOCKS_PATH_TO:
            raise ValueError(
                "BLOCKS_PATH_TO is a current-only derived relation"
            )
        if edge.source_node_id not in self._nodes or edge.target_node_id not in self._nodes:
            raise ValueError("scene edge endpoint is not present")
        prior = self._edges.get(edge.edge_id)
        if prior is not None:
            if (
                prior.source_node_id != edge.source_node_id
                or prior.target_node_id != edge.target_node_id
                or prior.relation is not edge.relation
                or prior.observed_at > edge.observed_at
                or prior.first_observed_at != edge.first_observed_at
                or not set(prior.source_ids).issubset(edge.source_ids)
                or (prior.lifecycle != "active" and edge.lifecycle != prior.lifecycle)
            ):
                raise ValueError(f"scene edge attempted to rewrite frozen fact: {edge.edge_id}")
            if replace(prior, revision_id="") == replace(edge, revision_id=""):
                return prior
            self._current_epoch_active_opposes_edge_ids.discard(
                prior.edge_id
            )
            revision_id = self._next_revision()
            edge = replace(edge, revision_id=revision_id)
            self._current_revised_edges.append(edge.edge_id)
        else:
            revision_id = self._next_revision()
            edge = replace(edge, revision_id=revision_id)
            self._current_added_edges.append(edge.edge_id)
        self._edges[edge.edge_id] = edge
        self._edge_revisions[edge.edge_id].append(edge)
        self._outgoing[edge.source_node_id].add(edge.edge_id)
        self._incoming[edge.target_node_id].add(edge.edge_id)
        source = self._nodes[edge.source_node_id]
        target = self._nodes[edge.target_node_id]
        if (
            edge.relation is SceneEdgeKind.OPPOSES
            and edge.lifecycle == "active"
            and source.market_epoch_id == self._market_epoch_id
            and target.market_epoch_id == self._market_epoch_id
        ):
            self._current_epoch_active_opposes_edge_ids.add(edge.edge_id)
        return edge

    def _add_relation(
        self,
        source_id: str,
        relation: SceneEdgeKind,
        target_id: str,
        observed_at: pd.Timestamp,
        source_ids: Sequence[str],
    ) -> None:
        clock = aware_timestamp(observed_at, name="scene relation clock")
        source_node = self._node_id_for_source(source_id, asof=clock)
        target_node = self._node_id_for_source(target_id, asof=clock)
        if source_node is None or target_node is None or source_node == target_node:
            return
        endpoint_states = [
            next(
                value
                for value in reversed(self._node_revisions[node_id])
                if value.observed_at <= clock
            )
            for node_id in (source_node, target_node)
        ]
        if endpoint_states[0].market_epoch_id != endpoint_states[1].market_epoch_id:
            return
        raw = f"{source_node}|{relation.value}|{target_node}"
        edge_id = f"edge:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"
        prior = self._edges.get(edge_id)
        if prior is not None:
            # The relation is a first-observed causal fact. Re-deriving it on
            # later minutes must not move its evidence clock forward or
            # reopen a relation whose endpoint lifecycle already closed it.
            return
        self.add_edge(
            SceneEdge(
                edge_id=edge_id,
                source_node_id=source_node,
                relation=relation,
                target_node_id=target_node,
                observed_at=clock,
                source_ids=tuple(dict.fromkeys(source_ids)),
            )
        )

    def _state_semantic_signature(
        self,
        item: Any,
        *,
        asof: pd.Timestamp,
    ) -> tuple[Any, ...] | None:
        """Build a cheap semantic tuple before adapting a retained snapshot.

        Age, freshness and metadata/last-updated heartbeats are deliberately
        absent.  They remain query-time descriptions and cannot make the hot
        graph re-project an otherwise identical causal entity.
        """

        identity = _entity_id(item)
        if identity is None:
            return None
        kind, entity_id = identity
        lifecycle = _value(getattr(item, "lifecycle", "observed"))
        if kind == "manipulation" and bool(
            getattr(item, "deadline_elapsed", False)
        ):
            lifecycle = "censored"
        timeframe = _value(
            getattr(item, "timeframe", Timeframe.M1)
        )
        formed_at = _clock(
            item,
            "formed_at",
            "pending_at",
            "started_at",
            "pivot_end",
            "confirmed_at",
            "observed_at",
            default=asof,
        )
        confirmed_at = getattr(item, "confirmed_at", None)
        if confirmed_at is not None:
            confirmed_at = aware_timestamp(
                confirmed_at,
                name="scene_signature.confirmed_at",
            )
        if confirmed_at is None and lifecycle not in {
            "forming",
            "started",
            "pending",
            "approaching",
        }:
            confirmed_at = _clock(
                item,
                "resolved_at",
                "active_at",
                "balance_confirmed_at",
                "state_started_at",
                "observed_at",
                default=asof,
            )
        resolution_reason = (
            getattr(item, "transition_reason", None)
            or getattr(item, "lifecycle_reason", None)
            or getattr(item, "resolution_reason", None)
        )
        range_balance_evidence = (
            (
                getattr(item, "lower_touch_count", None),
                getattr(item, "upper_touch_count", None),
                getattr(item, "midpoint_crossings", None),
                getattr(item, "inside_close_fraction", None),
                getattr(item, "compression_ratio", None),
            )
            if kind == "range"
            else ()
        )
        # This is an adaptation trigger, not a serialized copy of the node.
        # In particular, structural rank/visibility and descriptive quality
        # can be refreshed by a retained tracker snapshot without a new
        # market event.  Treating those derived metadata values as graph
        # changes both defeats the hot-path cache and can try to relabel a
        # terminal fact (for example, a reaccepted S/R zone whose source swing
        # is only later promoted to protected/external).  The graph revision
        # contract is narrower: identity/lifecycle, frozen geometry and
        # invalidation, causal sources, and evidence clocks.
        return (
            "scene-state-signature-v1",
            self._market_epoch_id,
            kind,
            timeframe,
            entity_id,
            lifecycle,
            formed_at,
            confirmed_at,
            _direction(item),
            _bounds(item),
            _invalidation_rule(kind, item),
            _source_ids(item),
            resolution_reason,
            range_balance_evidence,
            _semantic_event_clocks(item),
        )

    def _adapt_snapshot_state(
        self,
        item: Any,
        *,
        asof: pd.Timestamp,
        frame_state: bool,
    ) -> SceneNode | None:
        """Adapt only a new or semantically changed retained state."""

        identity = _entity_id(item)
        if identity is None:
            return None
        kind, entity_id = identity
        timeframe = _value(
            getattr(item, "timeframe", Timeframe.M1)
        )
        node_id = (
            f"{self._market_epoch_id}:{kind}:{timeframe}:{entity_id}"
        )
        signature = self._state_semantic_signature(item, asof=asof)
        if (
            signature is not None
            and self._state_semantic_signatures.get(node_id) == signature
            and node_id in self._nodes
        ):
            return self._nodes[node_id]
        node = self._adapt_state(item, asof=asof)
        if node is None:
            return None
        stored = (
            self._add_frame_state_node(node)
            if frame_state
            else self.add_node(node)
        )
        if signature is not None:
            self._state_semantic_signatures[node_id] = signature
        return stored

    def _adapt_state(self, item: Any, *, asof: pd.Timestamp) -> SceneNode | None:
        identity = _entity_id(item)
        if identity is None:
            return None
        kind, entity_id = identity
        lifecycle = _value(getattr(item, "lifecycle", "observed"))
        if kind == "manipulation" and bool(
            getattr(item, "deadline_elapsed", False)
        ):
            lifecycle = "censored"
        timeframe_value = getattr(item, "timeframe", Timeframe.M1)
        timeframe = _value(timeframe_value)
        formed_at = _clock(
            item,
            "formed_at",
            "pending_at",
            "started_at",
            "pivot_end",
            "confirmed_at",
            "observed_at",
            default=asof,
        )
        confirmed_at = getattr(item, "confirmed_at", None)
        if confirmed_at is not None:
            confirmed_at = aware_timestamp(
                confirmed_at,
                name="scene_source.confirmed_at",
            )
        if confirmed_at is None and lifecycle not in {"forming", "started", "pending", "approaching"}:
            confirmed_at = _clock(
                item,
                "resolved_at",
                "active_at",
                "balance_confirmed_at",
                "state_started_at",
                "observed_at",
                default=asof,
            )
        observed_at = _semantic_clock(item, default=formed_at)
        if any(
            value > asof
            for value in (formed_at, confirmed_at, observed_at)
            if value is not None
        ):
            raise ValueError("scene primitive contains a future evidence clock")
        node_id = f"{self._market_epoch_id}:{kind}:{timeframe}:{entity_id}"
        prior = self._nodes.get(node_id)
        sources = tuple(
            dict.fromkeys((entity_id, *_source_ids(item)))
        )
        source_clocks = [
            next(
                value.observed_at
                for value in reversed(self._node_revisions[source_node])
                if value.observed_at <= asof
            )
            for source_id in sources
            for source_node in (
                self._node_id_for_source(source_id, asof=asof),
            )
            if source_node is not None
        ]
        if source_clocks:
            observed_at = max(observed_at, *source_clocks)
        bounds = _bounds(item)
        invalidation_rule = _invalidation_rule(kind, item)
        status = _status(kind, lifecycle)
        if kind == "micro_bos" and getattr(item, "outcome", None) == "ambiguous_same_clock":
            status = EvidenceStatus.AMBIGUOUS
        if prior is not None:
            formed_at = prior.formed_at
            bounds = prior.price_bounds or bounds
            invalidation_rule = prior.invalidation_rule or invalidation_rule
            sources = tuple(dict.fromkeys((*prior.source_ids, *sources)))
            if prior.confirmed_at is not None:
                confirmed_at = prior.confirmed_at
        resolution_reason = (
            getattr(item, "transition_reason", None)
            or getattr(item, "lifecycle_reason", None)
            or getattr(item, "resolution_reason", None)
        )
        if (
            resolution_reason is None
            and prior is not None
            and prior.resolution_reason is not None
        ):
            resolution_reason = prior.resolution_reason
        direction = _direction(item)
        if prior is not None and prior.direction is not None:
            # A typed state snapshot can omit direction even though its
            # earlier ledger event carried it.  Omission is not a semantic
            # revision, so retain the first observed direction; an actual
            # opposite value is still rejected by ``add_node``.
            direction = prior.direction if direction is None else direction
        # Liquidity role is hypothesis/route-specific. The same H1 level may
        # be a context draw, blocker, primary target or terminal draw.
        role = None
        return SceneNode(
            node_id=node_id,
            kind=kind,
            timeframe=timeframe,
            structural_scale=_scale(kind, timeframe, item),
            direction=direction,
            formed_at=formed_at,
            confirmed_at=confirmed_at,
            observed_at=max(formed_at, observed_at),
            lifecycle=lifecycle,
            price_bounds=bounds,
            invalidation_rule=invalidation_rule,
            ambiguity_state=status,
            source_ids=sources,
            entity_id=entity_id,
            liquidity_role=role,
            resolution_reason=resolution_reason,
            semantic_attributes=_semantic_attributes(item),
            descriptive_metrics=_descriptive_metrics(item),
            market_epoch_id=self._market_epoch_id,
        )

    def _adapt_displacement_transition(self, transition: Any) -> None:
        """Append one already-causal displacement lifecycle transition."""

        timeframe = Timeframe.M5.value
        node_id = (
            f"{self._market_epoch_id}:displacement:"
            f"{timeframe}:{transition.entity_id}"
        )
        prior = self._nodes.get(node_id)
        metrics = dict(transition.state_metrics)
        origin = metrics.get("origin_price")
        net = metrics.get("net_points")
        bounds = None
        if origin is not None and net is not None:
            endpoint = float(origin) + transition.direction.sign * float(net)
            bounds = tuple(sorted((float(origin), endpoint)))
        self.add_node(
            SceneNode(
                node_id=node_id,
                kind="displacement",
                timeframe=timeframe,
                structural_scale=StructuralScale.INTERMEDIATE,
                direction=transition.direction,
                formed_at=(
                    prior.formed_at
                    if prior is not None
                    else (transition.started_at or transition.observed_at)
                ),
                confirmed_at=(
                    prior.confirmed_at
                    if prior is not None
                    else transition.active_at
                ),
                observed_at=transition.observed_at,
                lifecycle=transition.lifecycle,
                price_bounds=prior.price_bounds if prior is not None else bounds,
                invalidation_rule=(
                    prior.invalidation_rule
                    if prior is not None
                    else "frozen_episode_origin_or_qualified_opposite_displacement"
                ),
                ambiguity_state=_status("displacement", transition.lifecycle),
                source_ids=tuple(
                    dict.fromkeys(
                        (
                            transition.entity_id,
                            *transition.admitted_candle_ids,
                        )
                    )
                ),
                entity_id=transition.entity_id,
                resolution_reason=transition.reason,
                descriptive_metrics=tuple(
                    (name, float(value))
                    for name, value in transition.state_metrics
                    if name
                    in {
                        "mean_overlap_ratio",
                        "max_overlap_ratio",
                        "mean_directional_clv",
                    }
                ),
                market_epoch_id=self._market_epoch_id,
            )
        )

    def _adapt_current_displacement(self, displacement: Any) -> None:
        """Revise descriptive ACTIVE metrics without inventing a transition."""

        if displacement.current_entity_id is None:
            return
        node_id = (
            f"{self._market_epoch_id}:displacement:"
            f"{Timeframe.M5.value}:{displacement.current_entity_id}"
        )
        prior = self._nodes.get(node_id)
        metrics = dict(displacement.current_metrics or ())
        origin = metrics.get("origin_price")
        net = metrics.get("net_points")
        bounds = None
        if origin is not None and net is not None:
            endpoint = float(origin) + (
                displacement.current_direction.sign * float(net)
            )
            bounds = tuple(sorted((float(origin), endpoint)))
        self.add_node(
            SceneNode(
                node_id=node_id,
                kind="displacement",
                timeframe=Timeframe.M5.value,
                structural_scale=StructuralScale.INTERMEDIATE,
                direction=displacement.current_direction,
                formed_at=(
                    prior.formed_at
                    if prior is not None
                    else displacement.current_started_at
                ),
                confirmed_at=(
                    prior.confirmed_at
                    if prior is not None and prior.confirmed_at is not None
                    else displacement.current_active_at
                ),
                observed_at=displacement.current_state_observed_at,
                lifecycle=displacement.lifecycle,
                price_bounds=(
                    prior.price_bounds if prior is not None else bounds
                ),
                invalidation_rule=(
                    prior.invalidation_rule
                    if prior is not None
                    else "frozen_episode_origin_or_qualified_opposite_displacement"
                ),
                ambiguity_state=_status(
                    "displacement", displacement.lifecycle
                ),
                source_ids=(
                    prior.source_ids
                    if prior is not None
                    else (displacement.current_entity_id,)
                ),
                entity_id=displacement.current_entity_id,
                resolution_reason=(
                    None if prior is None else prior.resolution_reason
                ),
                descriptive_metrics=tuple(
                    (name, float(value))
                    for name, value in (displacement.current_metrics or ())
                    if name
                    in {
                        "mean_overlap_ratio",
                        "max_overlap_ratio",
                        "mean_directional_clv",
                    }
                ),
                market_epoch_id=self._market_epoch_id,
            )
        )
    @staticmethod
    def _event_kind(event: MarketEvent) -> str:
        return _SCENE_EVENT_KIND_TO_NODE_KIND.get(
            event.kind,
            event.kind.value,
        )

    @staticmethod
    def _event_entity_id(event: MarketEvent) -> str:
        """Map typed ledger namespaces back to the primitive identity."""

        identity = event.entity_id or event.event_id
        if (
            event.kind is EventKind.LIQUIDITY_POOL_STATE
            and identity.startswith("pool:")
        ):
            return identity[len("pool:") :]
        return identity

    def _adapt_market_event(
        self,
        event: MarketEvent,
        *,
        enrichment: Any | None = None,
    ) -> None:
        if (
            event.kind
            in {
                EventKind.SUPPORT_RESISTANCE_STATE,
                EventKind.LIQUIDITY_POOL_STATE,
                EventKind.MANIPULATION_STATE,
            }
            and event.details.get("state_revision") is True
        ):
            # Authoritative typed state revises the existing graph node.
            # EventMemory keeps this temporal delta without manufacturing a
            # second node under the generic event id.
            return
        kind = self._event_kind(event)
        entity_id = self._event_entity_id(event)
        timeframe = event.timeframe.value
        node_id = f"{self._market_epoch_id}:{kind}:{timeframe}:{entity_id}"
        prior = self._nodes.get(node_id)
        formed_at = event.formed_at
        if formed_at is None and enrichment is not None:
            formed_at = _clock(
                enrichment,
                "formed_at",
                "pending_at",
                "started_at",
                "pivot_end",
                default=event.observed_at,
            )
        formed_at = formed_at or event.observed_at
        confirmed_at = event.confirmed_at
        lifecycle = event.lifecycle or (
            "confirmed"
            if event.kind is EventKind.STRUCTURE_BREAK
            else "failed"
            if event.kind is EventKind.STRUCTURE_BREAK_FAILED
            else "invalidated"
            if event.ended_at is not None
            else "observed"
        )
        complex_zone_kinds = {
            "support_resistance",
            "liquidity_pool",
            "fvg",
            "order_block",
            "range",
            "manipulation",
            "entry_location",
            "liquidity",
        }
        bounds = _bounds(enrichment) if enrichment is not None else None
        if bounds is None and kind not in complex_zone_kinds and event.price is not None:
            bounds = (float(event.price), float(event.price))
        sources = tuple(
            dict.fromkeys((entity_id, *event.source_ids))
        )
        if prior is not None:
            formed_at = prior.formed_at
            confirmed_at = prior.confirmed_at or confirmed_at
            bounds = prior.price_bounds or bounds
            sources = tuple(dict.fromkeys((*prior.source_ids, *sources)))
            if lifecycle == "observed" and prior.lifecycle != "observed":
                lifecycle = prior.lifecycle
        resolution_reason = event.transition_reason
        if (
            resolution_reason is None
            and prior is not None
            and prior.resolution_reason is not None
        ):
            resolution_reason = prior.resolution_reason
        status = _status(kind, lifecycle)
        if "ambiguous" in lifecycle or "ambiguous" in (event.transition_reason or ""):
            status = EvidenceStatus.AMBIGUOUS
        semantic_attributes = dict(
            _semantic_attributes(event.details)
        )
        if event.kind is EventKind.ENTRY_PATH_STEP:
            step_kind = event.details.get("kind")
            if step_kind:
                semantic_attributes["path_step_kind"] = str(
                    step_kind
                )
        self.add_node(
            SceneNode(
                node_id=node_id,
                kind=kind,
                timeframe=timeframe,
                structural_scale=_scale(
                    kind,
                    timeframe,
                    enrichment if enrichment is not None else event,
                ),
                direction=(
                    event.direction
                    or (
                        _direction(enrichment)
                        if enrichment is not None
                        else _direction(event)
                    )
                ),
                formed_at=formed_at,
                confirmed_at=confirmed_at,
                observed_at=event.observed_at,
                lifecycle=lifecycle,
                price_bounds=bounds,
                invalidation_rule=_invalidation_rule(
                    kind,
                    enrichment if enrichment is not None else event,
                ),
                ambiguity_state=status,
                source_ids=sources,
                entity_id=entity_id,
                resolution_reason=resolution_reason,
                semantic_attributes=tuple(
                    sorted(semantic_attributes.items())
                ),
                descriptive_metrics=_descriptive_metrics(event.details),
                market_epoch_id=self._market_epoch_id,
            )
        )

    @staticmethod
    def _state_records(
        observation: MarketObservation,
    ) -> tuple[tuple[Any, bool], ...]:
        frame_items = tuple(
            (item, True)
            for frame in observation.frames.values()
            for group in (
                frame.swings,
                frame.structures,
                frame.structure_breaks,
                frame.support_resistance,
                frame.liquidity_pools,
                frame.fair_value_gaps,
                frame.order_blocks,
                frame.dealing_ranges,
            )
            for item in group
        )
        interaction = observation.interaction_update
        interaction_groups = (
            ()
            if interaction is None
            else (
                interaction.zone_interactions,
                interaction.reacceptance_interactions,
                interaction.micro_break_facts,
                interaction.interaction_paths,
            )
        )
        top_level = tuple(
            (item, False)
            for group in (
                observation.liquidity_inventory,
                observation.liquidity_pool_states,
                observation.manipulations,
                *interaction_groups,
            )
            for item in group
        )
        return (*frame_items, *top_level)

    @staticmethod
    def _state_items(observation: MarketObservation) -> tuple[Any, ...]:
        return tuple(
            item
            for item, _ in TemporalMarketSceneGraph._state_records(
                observation
            )
        )

    def _unseen_ledger_events(
        self,
        observation: MarketObservation,
        *,
        bootstrap: bool,
        minimum_observed_at: pd.Timestamp | None = None,
    ) -> tuple[MarketEvent, ...]:
        """Read unseen recent events plus only each retained timeline tail."""

        candidates: dict[str, MarketEvent] = {}
        retention_floor = self._history_retention_floor

        def retained(event: MarketEvent) -> bool:
            return (
                retention_floor is None
                or event.observed_at >= retention_floor
            )

        if bootstrap:
            for timeline in observation.retained_entity_timelines.values():
                for event in timeline:
                    if (
                        retained(event)
                        and event.event_id
                        not in self._seen_market_event_ids
                        and (
                            minimum_observed_at is None
                            or event.observed_at >= minimum_observed_at
                        )
                    ):
                        candidates[event.event_id] = event
        else:
            for timeline in observation.retained_entity_timelines.values():
                unseen_tail: list[MarketEvent] = []
                for event in reversed(timeline):
                    if not retained(event):
                        break
                    if event.event_id in self._seen_market_event_ids:
                        break
                    if (
                        minimum_observed_at is None
                        or event.observed_at >= minimum_observed_at
                    ):
                        unseen_tail.append(event)
                for event in reversed(unseen_tail):
                    candidates[event.event_id] = event
        for event in observation.recent_events:
            if (
                retained(event)
                and
                event.event_id not in self._seen_market_event_ids
                and (
                    minimum_observed_at is None
                    or event.observed_at >= minimum_observed_at
                )
            ):
                candidates[event.event_id] = event
        return tuple(
            sorted(
                candidates.values(),
                key=lambda event: (
                    event.observed_at,
                    event.sequence_no,
                    event.event_id,
                ),
            )
        )

    @staticmethod
    def _event_changed_entity_ids(
        events: Sequence[MarketEvent],
    ) -> set[str]:
        changed: set[str] = set()
        for event in events:
            entity_id = TemporalMarketSceneGraph._event_entity_id(event)
            changed.add(entity_id)
            changed.update(str(value) for value in event.source_ids)
        changed.update(
            value[len("pool:") :]
            for value in tuple(changed)
            if value.startswith("pool:")
        )
        return changed

    def _adapt_observation_nodes(
        self,
        observation: MarketObservation,
        *,
        bootstrap: bool,
        boundary_transition_reasons: frozenset[str] = frozenset(),
    ) -> None:
        state_records = self._state_records(observation)
        ledger_events = self._unseen_ledger_events(
            observation,
            bootstrap=bootstrap and not boundary_transition_reasons,
            minimum_observed_at=(
                observation.asof
                if boundary_transition_reasons
                else None
            ),
        )
        if boundary_transition_reasons:
            # Boundary terminal events are marked seen before this method is
            # entered, so they are intentionally absent from ``ledger_events``.
            # Read only each retained timeline's current-clock tail; scanning
            # every historical event on every ordinary bar would turn cold
            # history back into a hot-path cost.
            terminal_snapshot_events: dict[str, MarketEvent] = {
                event.event_id: event
                for event in observation.recent_events
                if event.observed_at == observation.asof
            }
            for timeline in observation.retained_entity_timelines.values():
                for event in reversed(timeline):
                    if event.observed_at < observation.asof:
                        break
                    if event.observed_at == observation.asof:
                        terminal_snapshot_events[event.event_id] = event
            snapshot_guard_events = tuple(
                terminal_snapshot_events.values()
            )
        else:
            # On ordinary bars every newly censored event is already present
            # in the unseen ledger assembled above.  Reusing it avoids a
            # second traversal of all retained entity histories.
            snapshot_guard_events = ledger_events
        boundary_state_keys = {
            (
                self._event_kind(event),
                event.timeframe.value,
                self._event_entity_id(event),
            )
            for event in snapshot_guard_events
            if event.observed_at == observation.asof
            and (
                event.transition_reason in boundary_transition_reasons
                or event.lifecycle == "censored"
            )
        }
        # If the public 64-event view contains only this clock, it may have
        # truncated earlier same-bar generic revisions.  This is rare, but a
        # full lightweight-token fallback preserves same-bar completeness.
        recent_capacity_gap = bool(
            len(observation.recent_events) >= 64
            and observation.recent_events
            and observation.recent_events[0].observed_at
            == observation.asof
        )
        full_token_pass = bootstrap or recent_capacity_gap
        changed_entity_ids = self._event_changed_entity_ids(ledger_events)
        changed_keys = {
            (
                self._event_kind(event),
                event.timeframe.value,
                self._event_entity_id(event),
            )
            for event in ledger_events
        }
        self._state_catalog = {}
        selected_records: list[tuple[Any, bool, str, tuple[Any, ...]]] = []
        for item, frame_state in state_records:
            identity = _entity_id(item)
            if identity is None:
                continue
            kind, entity_id = identity
            timeframe = _value(getattr(item, "timeframe", Timeframe.M1))
            node_id = (
                f"{self._market_epoch_id}:{kind}:{timeframe}:{entity_id}"
            )
            if (kind, timeframe, entity_id) in boundary_state_keys:
                continue
            prior = self._nodes.get(node_id)
            lifecycle = _value(
                getattr(item, "lifecycle", "observed")
            )
            if (
                prior is None
                and self._history_retention_floor is not None
                and _is_terminal(kind, lifecycle)
                and _semantic_clock(item, default=observation.asof)
                < self._history_retention_floor
            ):
                # The observer may retain a compact terminal snapshot longer
                # than the runtime graph's explicitly bounded history.  Once
                # compacted, that stale snapshot cannot reopen the cold fact.
                continue
            if prior is not None and _is_terminal(
                prior.kind,
                prior.lifecycle,
            ):
                continue
            token = _lightweight_state_token(kind, item)
            changed = (
                full_token_pass
                or prior is None
                or (kind, timeframe, entity_id) in changed_keys
                or entity_id in changed_entity_ids
                or self._state_lightweight_tokens.get(node_id) != token
            )
            self._state_lightweight_tokens[node_id] = token
            if not changed:
                continue
            self._state_catalog[(kind, timeframe, entity_id)] = item
            selected_records.append((item, frame_state, node_id, token))
        displacement_transitions = ()
        if observation.displacement is not None:
            displacement_transitions = (
                observation.displacement.recent_transitions
                if self._last_asof is None
                else observation.displacement.transitions_this_update
            )
        ordered_updates = [
            (
                event.observed_at,
                1,
                event.sequence_no,
                event.event_id,
                "event",
                event,
            )
            for event in ledger_events
            if event.event_id not in self._seen_market_event_ids
        ] + [
            (
                transition.observed_at,
                0,
                transition.ordinal,
                transition.transition_id,
                "displacement",
                transition,
            )
            for transition in displacement_transitions
            if transition.transition_id
            not in self._seen_displacement_transition_ids
        ]
        for _, _, _, identity, source_kind, payload in sorted(ordered_updates):
            if source_kind == "event":
                kind = self._event_kind(payload)
                entity_id = self._event_entity_id(payload)
                enrichment = self._state_catalog.get(
                    (kind, payload.timeframe.value, entity_id)
                )
                self._adapt_market_event(payload, enrichment=enrichment)
                self._seen_market_event_ids.add(identity)
            else:
                self._adapt_displacement_transition(payload)
                self._seen_displacement_transition_ids.add(identity)
        if observation.displacement is not None:
            self._adapt_current_displacement(observation.displacement)
        selected_structures: list[Any] = []
        for item, frame_state, _, _ in selected_records:
            if (
                frame_state
                and getattr(item, "lifecycle", None)
                is SwingLifecycle.FORMING
            ):
                # A forming pivot is not yet a semantic structure node.
                continue
            self._adapt_snapshot_state(
                item,
                asof=observation.asof,
                frame_state=frame_state,
            )
            identity = _entity_id(item)
            if identity is not None and identity[0] == "structure":
                selected_structures.append(item)
        # Structural scale is an append-only projection over the raw swing.
        # A swing can therefore remain visible as internal while also owning
        # intermediate/external roles in the same chart.
        for structure in selected_structures:
            if structure.lifecycle in {
                StructureLifecycle.BROKEN,
                StructureLifecycle.FORMATION_FAILED,
            }:
                terminal_clock = _semantic_clock(
                    structure,
                    default=observation.asof,
                )
                for projection in self._current_kind_nodes(
                    "swing_projection"
                ):
                    if (
                        projection.kind == "swing_projection"
                        and structure.structure_id
                        in projection.source_ids
                        and not _is_terminal(
                            projection.kind,
                            projection.lifecycle,
                        )
                    ):
                        self.add_node(
                            replace(
                                projection,
                                observed_at=terminal_clock,
                                lifecycle="broken",
                                ambiguity_state=EvidenceStatus.INVALIDATED,
                                resolution_reason=(
                                    structure.failure_reason
                                    or "source_structure_terminal"
                                ),
                                revision_id="",
                            )
                        )
                continue
            if (
                structure.lifecycle is not StructureLifecycle.CONFIRMED
                or structure.confirmed_at is None
            ):
                continue
            projections = (
                (structure.latest_high_id, StructuralScale.INTERMEDIATE),
                (structure.latest_low_id, StructuralScale.INTERMEDIATE),
                (structure.protected_swing_id, StructuralScale.EXTERNAL),
            )
            for swing_id, structural_scale in projections:
                source_clock = structure.confirmed_at
                if (
                    self._history_retention_floor is not None
                    and source_clock < self._history_retention_floor
                ):
                    projection_prefix = (
                        f"{self._market_epoch_id}:swing_projection:"
                        f"{structure.timeframe.value}:{swing_id}:"
                    )
                    if any(
                        node_id.startswith(projection_prefix)
                        and structure.structure_id
                        in self._nodes[node_id].source_ids
                        for node_id in self._current_epoch_node_ids
                    ):
                        # The old structure/projection is already current and
                        # unchanged.  Its historical source revision is below
                        # the retention floor, so there is nothing to derive
                        # this minute.  A genuinely new structure is always
                        # confirmed at/after the current floor and follows the
                        # normal causal as-of path below.
                        continue
                    # A compacted runtime cannot prove a previously unseen
                    # projection whose source clock predates retained history.
                    # Fail closed rather than bind it to a newer swing state.
                    continue
                raw_node_id = (
                    None
                    if swing_id is None
                    else self._node_id_for_source(
                        swing_id,
                        kind="swing",
                        asof=source_clock,
                    )
                )
                if raw_node_id is None:
                    continue
                raw_node = next(
                    value
                    for value in reversed(
                        self._node_revisions[raw_node_id]
                    )
                    if value.observed_at <= source_clock
                )
                projection_id = (
                    f"{self._market_epoch_id}:swing_projection:"
                    f"{structure.timeframe.value}:{swing_id}:"
                    f"{structural_scale.value}:"
                    f"{structure.structure_id}"
                )
                projection = self.add_node(
                    SceneNode(
                        node_id=projection_id,
                        kind="swing_projection",
                        timeframe=structure.timeframe.value,
                        structural_scale=structural_scale,
                        direction=structure.direction,
                        formed_at=raw_node.formed_at,
                        confirmed_at=structure.confirmed_at,
                        observed_at=structure.confirmed_at,
                        lifecycle="confirmed",
                        price_bounds=raw_node.price_bounds,
                        invalidation_rule=(
                            "projection_closes_with_source_structure"
                        ),
                        ambiguity_state=EvidenceStatus.CONFIRMED,
                        source_ids=(swing_id, structure.structure_id),
                        market_epoch_id=self._market_epoch_id,
                    )
                )
                self._add_relation(
                    projection.node_id,
                    SceneEdgeKind.PROMOTED_FROM,
                    raw_node.node_id,
                    structure.confirmed_at,
                    (projection.node_id, raw_node.node_id),
                )
        for item_id in observation.group4_ambiguous_sweep_item_ids:
            node_id = (
                f"{self._market_epoch_id}:ambiguous_sweep:"
                f"{Timeframe.M1.value}:{item_id}:"
                f"{observation.asof.isoformat()}"
            )
            self.add_node(
                SceneNode(
                    node_id=node_id,
                    kind="manipulation_candidate",
                    timeframe=Timeframe.M1.value,
                    structural_scale=StructuralScale.INTERNAL,
                    direction=None,
                    formed_at=observation.asof,
                    confirmed_at=None,
                    observed_at=observation.asof,
                    lifecycle="ambiguous",
                    price_bounds=None,
                    invalidation_rule=None,
                    ambiguity_state=EvidenceStatus.AMBIGUOUS,
                    source_ids=(item_id,),
                    market_epoch_id=self._market_epoch_id,
                )
            )

    def _current_nodes_sharing_sources(
        self,
        node: SceneNode,
        *,
        kinds: frozenset[str],
    ) -> tuple[SceneNode, ...]:
        candidate_ids: set[str] = set()
        for source_id in node.source_ids:
            candidate_ids.update(self._source_index.get(source_id, ()))
            candidate_ids.update(self._entity_index.get(source_id, ()))
        changed_this_update = set(
            (*self._current_added_nodes, *self._current_revised_nodes)
        )
        return tuple(
            sorted(
                (
                    self._nodes[node_id]
                    for node_id in candidate_ids
                    if (
                        node_id in self._current_epoch_node_ids
                        or node_id in changed_this_update
                    )
                    and node_id != node.node_id
                    and self._nodes[node_id].kind in kinds
                    and self._nodes[node_id].timeframe == node.timeframe
                    and not set(node.source_ids).isdisjoint(
                        self._nodes[node_id].source_ids
                    )
                ),
                key=lambda value: (value.observed_at, value.node_id),
            )
        )

    @staticmethod
    def _is_locatable_liquidity(node: SceneNode) -> bool:
        if node.kind == "liquidity_pool":
            return True
        if node.kind != "liquidity":
            return False
        inventory_kind = dict(node.semantic_attributes).get(
            "inventory_kind"
        )
        return inventory_kind in {
            "previous_session_high",
            "previous_session_low",
            "previous_day_high",
            "previous_day_low",
            "previous_week_high",
            "previous_week_low",
            "range_boundary",
        }

    def _locate_liquidity_at_zones(
        self,
        node: SceneNode,
        *,
        relation_clock: pd.Timestamp,
    ) -> None:
        if node.kind == "support_resistance":
            liquidity_nodes = self._current_nodes_sharing_sources(
                node,
                kinds=frozenset({"liquidity", "liquidity_pool"}),
            )
            pairs = (
                (liquidity, node)
                for liquidity in liquidity_nodes
                if self._is_locatable_liquidity(liquidity)
            )
        elif self._is_locatable_liquidity(node):
            zones = self._current_nodes_sharing_sources(
                node,
                kinds=frozenset({"support_resistance"}),
            )
            pairs = ((node, zone) for zone in zones)
        else:
            return
        for liquidity, zone in pairs:
            observed_at = max(
                liquidity.observed_at,
                zone.observed_at,
                relation_clock,
            )
            self._add_relation(
                liquidity.node_id,
                SceneEdgeKind.LOCATED_AT,
                zone.node_id,
                observed_at,
                (
                    liquidity.entity_id or liquidity.node_id,
                    zone.entity_id or zone.node_id,
                ),
            )

    def _derive_source_relations(self, observation: MarketObservation) -> None:
        revised_node_ids = set(self._current_revised_nodes)

        def relation_clock(node: SceneNode) -> pd.Timestamp:
            # A new node carries its physical evidence clock.  A revision can
            # disclose a new source only on this completed observation; using
            # the frozen formation clock would backfill a relation into
            # compacted history that was not observable then.
            return (
                observation.asof
                if node.node_id in revised_node_ids
                else node.observed_at
            )

        for node_id in dict.fromkeys(
            (*self._current_added_nodes, *self._current_revised_nodes)
        ):
            node = self._nodes[node_id]
            clock = relation_clock(node)
            for source_id in node.source_ids:
                source_node = self._node_id_for_source(
                    source_id,
                    asof=clock,
                )
                if source_node is None or source_node == node.node_id:
                    continue
                source_kind = self._nodes[source_node].kind
                relation = SceneEdgeKind.SOURCED_FROM
                if (
                    node.kind == "liquidity_retirement"
                    and source_kind == "liquidity"
                ):
                    self._add_relation(
                        node.node_id,
                        SceneEdgeKind.RESOLVES,
                        source_node,
                        clock,
                        (node.node_id, source_id),
                    )
                    source_state = self._nodes[source_node]
                    if not _is_terminal(
                        source_state.kind, source_state.lifecycle
                    ):
                        self.add_node(
                            replace(
                                source_state,
                                observed_at=clock,
                                lifecycle="retired",
                                ambiguity_state=(
                                    EvidenceStatus.INVALIDATED
                                ),
                                resolution_reason=(
                                    "reference_period_replaced"
                                ),
                                revision_id="",
                            )
                        )
                    continue
                if node.kind == "support_resistance":
                    if source_kind == "swing":
                        self._add_relation(
                            source_node,
                            SceneEdgeKind.ANCHORS,
                            node.node_id,
                            clock,
                            (source_id, node.node_id),
                        )
                    elif (
                        dict(node.semantic_attributes).get("source_kind")
                        == "range_boundary"
                        and source_kind == "support_resistance"
                    ):
                        self._add_relation(
                            node.node_id,
                            SceneEdgeKind.SOURCED_FROM,
                            source_node,
                            clock,
                            (node.node_id, source_id),
                        )
                    continue
                if node.kind == "bos" and source_kind == "swing":
                    relation = SceneEdgeKind.BREAKS
                elif node.kind in {"fvg", "order_block"} and source_kind == "displacement":
                    relation = SceneEdgeKind.CREATES
                    source_node, node_id_for_edge = source_node, node.node_id
                    self._add_relation(
                        self._nodes[source_node].node_id,
                        relation,
                        node_id_for_edge,
                        clock,
                        (source_id, node.node_id),
                    )
                    continue
                elif node.kind == "order_block" and source_kind == "bos":
                    self._add_relation(
                        source_node,
                        SceneEdgeKind.CREATES,
                        node.node_id,
                        clock,
                        (source_id, node.node_id),
                    )
                    continue
                elif node.kind == "manipulation" and source_kind in {"liquidity", "liquidity_pool", "range"}:
                    relation = SceneEdgeKind.SWEEPS
                elif node.kind == "entry_location" and source_kind in {"fvg", "order_block"}:
                    relation = SceneEdgeKind.RETURNS_TO
                elif node.kind == "micro_bos" and source_kind in {"bos", "swing", "path_sequence"}:
                    relation = SceneEdgeKind.CONFIRMS
                self._add_relation(
                    node.node_id,
                    relation,
                    source_id,
                    clock,
                    (node.node_id, source_id),
                )

            if node.kind == "bos" and node.lifecycle == "confirmed":
                displacement_nodes = tuple(
                    candidate
                    for source_id in node.source_ids
                    for candidate in (
                        self._node_id_for_source(
                            source_id,
                            kind="displacement",
                            asof=clock,
                        ),
                    )
                    if candidate is not None
                )
                swing_nodes = tuple(
                    candidate
                    for source_id in node.source_ids
                    for candidate in (
                        self._node_id_for_source(
                            source_id,
                            kind="swing",
                            asof=clock,
                        ),
                    )
                    if candidate is not None
                )
                for displacement_node in displacement_nodes:
                    for swing_node in swing_nodes:
                        self._add_relation(
                            displacement_node,
                            SceneEdgeKind.BREAKS,
                            swing_node,
                            clock,
                            (node.entity_id or node.node_id,),
                        )

            self._locate_liquidity_at_zones(
                node,
                relation_clock=clock,
            )

        lower_nodes = [
            self._nodes[node_id]
            for node_id in dict.fromkeys(
                (*self._current_added_nodes, *self._current_revised_nodes)
            )
            if self._nodes[node_id].kind
            in {"swing", "structure", "bos", "displacement"}
            and not _is_terminal(
                self._nodes[node_id].kind,
                self._nodes[node_id].lifecycle,
            )
        ]
        rank = {
            Timeframe.M1.value: 0,
            Timeframe.M5.value: 1,
            Timeframe.M15.value: 2,
            Timeframe.H1.value: 3,
            Timeframe.H4.value: 4,
        }
        for node in lower_nodes:
            clock = relation_clock(node)
            structures = [
                value
                for value in self._current_kind_nodes_asof(
                    "structure",
                    clock,
                )
                if value.market_epoch_id == node.market_epoch_id
                and value.lifecycle == StructureLifecycle.CONFIRMED.value
                and value.node_id != node.node_id
            ]
            parents = [
                parent
                for parent in structures
                if rank.get(parent.timeframe, -1) > rank.get(node.timeframe, -1)
                and (parent.confirmed_at or parent.formed_at) <= node.observed_at
            ]
            if not parents:
                continue
            nearest_rank = min(rank.get(value.timeframe, 99) for value in parents)
            parent = max(
                (
                    value
                    for value in parents
                    if rank.get(value.timeframe, 99) == nearest_rank
                ),
                key=lambda value: (value.observed_at, value.node_id),
            )
            relation = (
                SceneEdgeKind.CONTAINED_BY
                if node.direction is None
                else SceneEdgeKind.ALIGNS_WITH
                if node.direction is parent.direction
                else SceneEdgeKind.OPPOSES
            )
            self._add_relation(
                node.node_id,
                relation,
                parent.node_id,
                clock,
                (node.node_id, parent.node_id),
            )

        for displacement in (
            node for node in lower_nodes if node.kind == "displacement"
        ):
            clock = relation_clock(displacement)
            manipulation_candidates = (
                *self._current_kind_nodes_asof(
                    "manipulation",
                    clock,
                ),
                *self._recent_terminal_context_nodes(
                    "manipulation",
                    asof=clock,
                    maximum_age=pd.Timedelta(minutes=60),
                ),
            )
            manipulations = [
                node
                for node in dict.fromkeys(manipulation_candidates)
                if node.market_epoch_id == displacement.market_epoch_id
                and node.ambiguity_state
                not in {EvidenceStatus.INVALIDATED, EvidenceStatus.UNKNOWN}
            ]
            eligible = [
                manipulation
                for manipulation in manipulations
                if manipulation.observed_at <= displacement.formed_at
                and displacement.formed_at - manipulation.observed_at
                <= pd.Timedelta(minutes=60)
                and manipulation.direction is not displacement.direction
            ]
            if eligible:
                source = max(
                    eligible,
                    key=lambda node: (node.observed_at, node.node_id),
                )
                self._add_relation(
                    source.node_id,
                    SceneEdgeKind.PRECEDES,
                    displacement.node_id,
                    clock,
                    (source.node_id, displacement.node_id),
                )

        self._refresh_current_path_block_edges(observation)

    def _refresh_current_path_block_edges(
        self,
        observation: MarketObservation,
    ) -> None:
        """Derive only the current adjacent-liquidity obstruction view."""

        price = float(observation.price)
        current: dict[str, SceneEdge] = {}
        for side in ("above", "below"):
            candidates = [
                self._nodes[node_id]
                for node_id in self._visible_liquidity_node_ids_by_side[
                    side
                ]
                if node_id in self._nodes
                and (
                    (
                        side == "above"
                        and self._nodes[node_id].price_bounds[0] > price
                    )
                    or (
                        side == "below"
                        and self._nodes[node_id].price_bounds[1] < price
                    )
                )
            ]
            ordered = sorted(
                candidates,
                key=lambda node: (
                    abs(sum(node.price_bounds) / 2.0 - price),
                    node.node_id,
                ),
            )
            for near, far in zip(ordered[:-1], ordered[1:]):
                raw = (
                    f"{near.node_id}|"
                    f"{SceneEdgeKind.BLOCKS_PATH_TO.value}|{far.node_id}"
                )
                edge_id = (
                    f"edge:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"
                )
                current[edge_id] = SceneEdge(
                    edge_id=edge_id,
                    source_node_id=near.node_id,
                    relation=SceneEdgeKind.BLOCKS_PATH_TO,
                    target_node_id=far.node_id,
                    observed_at=observation.asof,
                    source_ids=(near.node_id, far.node_id),
                )
        self._current_path_block_edges = current
        # BLOCKS_PATH_TO is deliberately outside permanent graph revision
        # history, so its own refresh must invalidate current neighbors.
        getattr(self, "_current_neighbors_cache", {}).clear()

    def _close_epoch(self, *, observed_at: pd.Timestamp, reason: str) -> None:
        """Censor open semantic state before a hard history boundary."""

        epoch_node_ids = tuple(self._current_epoch_node_ids)
        for node_id in epoch_node_ids:
            node = self._nodes[node_id]
            if (
                node.market_epoch_id != self._market_epoch_id
                or _is_terminal(node.kind, node.lifecycle)
                or node.kind == "resolution"
            ):
                continue
            self.add_node(
                replace(
                    node,
                    observed_at=observed_at,
                    lifecycle="censored",
                    ambiguity_state=EvidenceStatus.INVALIDATED,
                    resolution_reason=reason,
                    revision_id="",
                )
            )
        current_edge_ids = {
            edge_id
            for node_id in epoch_node_ids
            for edge_id in (
                *self._outgoing.get(node_id, ()),
                *self._incoming.get(node_id, ()),
            )
        }
        for edge_id in tuple(current_edge_ids):
            edge = self._edges[edge_id]
            if (
                edge.lifecycle != "active"
                or edge.relation in self._PERSISTENT_RELATIONS
            ):
                continue
            source = self._nodes[edge.source_node_id]
            if source.market_epoch_id != self._market_epoch_id:
                continue
            self.add_edge(
                replace(
                    edge,
                    observed_at=observed_at,
                    lifecycle="closed",
                    resolution_reason=reason,
                    revision_id="",
                )
            )

    def _mark_boundary_updates_seen(
        self,
        observation: MarketObservation,
        reasons: set[str],
    ) -> None:
        transition_reasons = set(reasons)
        transition_reasons.update(
            {
                "contract_change_reset"
                if reason == "contract_change_history_reset"
                else "data_gap_reset"
                if reason == "data_gap_history_reset"
                else reason
                for reason in reasons
            }
        )
        for event in (
            event
            for timeline in observation.retained_entity_timelines.values()
            for event in timeline
        ):
            if (
                event.observed_at == observation.asof
                and (
                    event.transition_reason in transition_reasons
                    or event.lifecycle == "censored"
                )
            ):
                self._seen_market_event_ids.add(event.event_id)
        self._seen_market_event_ids.update(
            event.event_id
            for event in observation.recent_events
            if event.observed_at == observation.asof
            and (
                event.transition_reason in transition_reasons
                or event.lifecycle == "censored"
            )
        )
        if observation.displacement is not None:
            self._seen_displacement_transition_ids.update(
                transition.transition_id
                for transition in observation.displacement.transitions_this_update
                if transition.lifecycle == "censored"
                and transition.reason in transition_reasons
            )

    def update(self, observation: MarketObservation) -> SceneGraphDelta:
        asof = aware_timestamp(observation.asof, name="scene_graph.asof")
        if self._last_asof is not None and asof <= self._last_asof:
            raise ValueError("scene graph updates must be strictly increasing")
        self._current_added_nodes = []
        self._current_revised_nodes = []
        self._current_added_edges = []
        self._current_revised_edges = []
        self._current_resolutions = []
        self._current_path_block_edges = {}
        # The current-price obstruction view changed even if this update adds
        # no permanent semantic revision.
        getattr(self, "_current_neighbors_cache", {}).clear()
        self._last_price = float(observation.price)
        boundary_reasons = {
            "contract_change_history_reset",
            "data_gap_history_reset",
        }.intersection(observation.anomalies)
        boundary_reset = bool(boundary_reasons)
        boundary_transition_reasons = frozenset(
            {
                *boundary_reasons,
                *(
                    "contract_change_reset"
                    if reason == "contract_change_history_reset"
                    else "data_gap_reset"
                    if reason == "data_gap_history_reset"
                    else reason
                    for reason in boundary_reasons
                ),
            }
        )
        if boundary_reset:
            reason = sorted(boundary_reasons)[0]
            self._close_epoch(observed_at=asof, reason=reason)
            self._mark_boundary_updates_seen(observation, boundary_reasons)
            self._market_epoch_no += 1
            self._market_epoch_id = f"epoch:{self._market_epoch_no}"
            self._epoch_boundaries.append((asof, self._market_epoch_id))
            self._clear_current_epoch_indexes()
        self._adapt_observation_nodes(
            observation,
            bootstrap=self._last_asof is None or boundary_reset,
            boundary_transition_reasons=boundary_transition_reasons,
        )
        self._derive_source_relations(observation)
        self._last_asof = asof
        if not self._revision:
            self._next_revision()
        return SceneGraphDelta(
            asof=asof,
            revision_id=self._revision_id,
            added_node_ids=tuple(dict.fromkeys(self._current_added_nodes)),
            revised_node_ids=tuple(dict.fromkeys(self._current_revised_nodes)),
            added_edge_ids=tuple(dict.fromkeys(self._current_added_edges)),
            revised_edge_ids=tuple(dict.fromkeys(self._current_revised_edges)),
            resolution_event_ids=tuple(dict.fromkeys(self._current_resolutions)),
        )

    def _neighbors(
        self,
        node_id: str,
        *,
        asof: pd.Timestamp | None = None,
    ) -> Iterable[tuple[str, SceneEdge]]:
        current_view = (
            asof is None
            or (
                self._last_asof is not None
                and aware_timestamp(asof, name="scene neighbor asof")
                >= self._last_asof
            )
        )
        edge_by_id = (
            self._edges
            if current_view
            else {edge.edge_id: edge for edge in self.edges_asof(asof)}
        )
        if current_view:
            cache = getattr(self, "_current_neighbors_cache", None)
            if cache is None:
                cache = {}
                self._current_neighbors_cache = cache
            cached = cache.get(node_id)
            if cached is not None:
                yield from cached
                return
            values: list[tuple[str, SceneEdge]] = []
            # Adjacency is stored in sets, whose iteration order changes with
            # the interpreter's hash seed and with the insertion history a
            # pickle round trip rebuilds.  A caller that reads the first
            # neighbour -- the path search does -- would otherwise get a
            # different answer from the same event log in a different process.
            for edge_id in sorted(self._outgoing.get(node_id, ())):
                edge = edge_by_id.get(edge_id)
                if edge is not None and edge.lifecycle == "active":
                    values.append((edge.target_node_id, edge))
            for edge_id in sorted(self._incoming.get(node_id, ())):
                edge = edge_by_id.get(edge_id)
                if edge is not None and edge.lifecycle == "active":
                    values.append((edge.source_node_id, edge))
            for edge in sorted(
                self._current_path_block_edges.values(),
                key=lambda item: item.edge_id,
            ):
                if edge.source_node_id == node_id:
                    values.append((edge.target_node_id, edge))
                elif edge.target_node_id == node_id:
                    values.append((edge.source_node_id, edge))
            cached = tuple(values)
            cache[node_id] = cached
            yield from cached
            return
        for edge_id in sorted(self._outgoing.get(node_id, ())):
            edge = edge_by_id.get(edge_id)
            if edge is None:
                continue
            if edge.lifecycle == "active":
                yield edge.target_node_id, edge
        for edge_id in sorted(self._incoming.get(node_id, ())):
            edge = edge_by_id.get(edge_id)
            if edge is None:
                continue
            if edge.lifecycle == "active":
                yield edge.source_node_id, edge

    def current_candle_structure_nodes(
        self,
        observation: MarketObservation,
        *,
        timeframes: Sequence[Timeframe | str] | None = None,
        include_non_real: bool = False,
    ) -> tuple[SceneNode, ...]:
        """Project latest candle geometry without persisting graph history.

        Candle geometry is deliberately opt-in: default scene queries, Focus
        selection and Brain input continue to consume semantic structure only.
        """

        if self._last_asof is None or observation.asof != self._last_asof:
            raise ValueError(
                "candle structure query must use the current graph observation"
            )
        requested = tuple(
            dict.fromkeys(
                _value(value)
                for value in (
                    observation.active_timeframes
                    if timeframes is None
                    else timeframes
                )
            )
        )
        available = {
            timeframe.value: timeframe
            for timeframe in observation.active_timeframes
        }
        unknown = set(requested).difference(available)
        if unknown:
            raise ValueError(
                f"candle structure query contains inactive timeframes: "
                f"{sorted(unknown)}"
            )
        nodes: list[SceneNode] = []
        for timeframe_name in requested:
            state = observation.frame(available[timeframe_name]).candle_structure
            if state is None or (not include_non_real and not state.real_completed):
                continue
            entity_id = (
                f"candle_structure:{timeframe_name}:"
                f"{state.start.isoformat()}:{state.observed_at.isoformat()}"
            )
            attributes = [
                ("body_class", state.body_class),
                ("range_class", state.range_class),
                ("dominant_wick", state.dominant_wick),
                ("close_class", state.close_class),
                ("real_completed", str(state.real_completed).lower()),
                ("zero_range", str(state.zero_range).lower()),
            ]
            if state.anomalies:
                attributes.append(("anomalies", ",".join(state.anomalies)))
            direction = (
                Direction.LONG
                if state.direction > 0
                else Direction.SHORT
                if state.direction < 0
                else None
            )
            nodes.append(
                SceneNode(
                    node_id=f"view:{self._market_epoch_id}:{entity_id}",
                    kind="candle_structure",
                    timeframe=timeframe_name,
                    structural_scale=StructuralScale.INTERNAL,
                    direction=direction,
                    formed_at=state.start,
                    confirmed_at=(
                        state.observed_at if state.real_completed else None
                    ),
                    observed_at=state.observed_at,
                    lifecycle="observed",
                    price_bounds=None,
                    invalidation_rule=None,
                    ambiguity_state=(
                        EvidenceStatus.CONFIRMED
                        if state.real_completed
                        else EvidenceStatus.UNKNOWN
                    ),
                    source_ids=(entity_id,),
                    entity_id=entity_id,
                    semantic_attributes=tuple(attributes),
                    descriptive_metrics=(
                        ("range_points", state.range_points),
                        ("body_points", state.body_points),
                        ("upper_wick_points", state.upper_wick_points),
                        ("lower_wick_points", state.lower_wick_points),
                        ("body_ratio", state.body_ratio),
                        ("upper_wick_ratio", state.upper_wick_ratio),
                        ("lower_wick_ratio", state.lower_wick_ratio),
                        ("close_location", state.close_location),
                    ),
                    market_epoch_id=self._market_epoch_id,
                )
            )
        return tuple(nodes)

    def find_path(
        self,
        source_ids: Sequence[str],
        target_ids: Sequence[str],
        *,
        max_depth: int = 8,
        asof: pd.Timestamp | None = None,
        allowed_relations: Iterable[SceneEdgeKind | str] | None = None,
    ) -> tuple[str, ...]:
        relation_allowlist = (
            None
            if allowed_relations is None
            else frozenset(SceneEdgeKind(value) for value in allowed_relations)
        )
        starts = tuple(
            dict.fromkeys(
                node_id
                for source_id in source_ids
                for node_id in (
                    self._node_id_for_source(source_id, asof=asof),
                )
                if node_id is not None
            )
        )
        targets = {
            node_id
            for target_id in target_ids
            for node_id in (self._node_id_for_source(target_id, asof=asof),)
            if node_id is not None
        }
        queue = deque((node_id, (node_id,)) for node_id in starts)
        seen = set(starts)
        while queue:
            node_id, path = queue.popleft()
            if node_id in targets:
                return path
            if len(path) > max_depth:
                continue
            for neighbor, edge in self._neighbors(node_id, asof=asof):
                if (
                    relation_allowlist is not None
                    and edge.relation not in relation_allowlist
                ):
                    continue
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append((neighbor, (*path, edge.relation.value, neighbor)))
        return ()

    def find_causal_path(
        self,
        source_ids: Sequence[str],
        target_ids: Sequence[str],
        *,
        max_depth: int = 8,
        asof: pd.Timestamp | None = None,
    ) -> tuple[str, ...]:
        """Resolve only registered semantic/provenance connectivity.

        Diagnostic temporal association such as ``PRECEDES`` remains visible
        through :meth:`find_path`, but it cannot satisfy a causal/action gate.
        """

        return self.find_path(
            source_ids,
            target_ids,
            max_depth=max_depth,
            asof=asof,
            allowed_relations=_CAUSAL_PATH_RELATIONS,
        )

    def find_action_path(
        self,
        source_ids: Sequence[str],
        target_ids: Sequence[str],
        *,
        max_depth: int = 8,
        asof: pd.Timestamp | None = None,
    ) -> tuple[str, ...]:
        """Resolve admitted semantic connectivity for an action gate.

        This admits explicit scale/composition relations in addition to source
        provenance.  Temporal ``PRECEDES`` and dynamic path-obstruction edges
        remain diagnostic only and cannot manufacture setup connectivity.
        """

        return self.find_path(
            source_ids,
            target_ids,
            max_depth=max_depth,
            asof=asof,
            allowed_relations=_ACTION_CONNECTIVITY_RELATIONS,
        )

    def has_direct_relation(
        self,
        source_ids: Sequence[str],
        relation: SceneEdgeKind | str,
        target_ids: Sequence[str],
        *,
        source_kind: str | None = None,
        target_kind: str | None = None,
        asof: pd.Timestamp | None = None,
    ) -> bool:
        """Check one exact directed semantic edge between source identities."""

        expected = SceneEdgeKind(relation)
        source_nodes = {
            node_id
            for source_id in source_ids
            for node_id in (
                self._node_id_for_source(
                    source_id,
                    kind=source_kind,
                    asof=asof,
                ),
            )
            if node_id is not None
        }
        target_nodes = {
            node_id
            for target_id in target_ids
            for node_id in (
                self._node_id_for_source(
                    target_id,
                    kind=target_kind,
                    asof=asof,
                ),
            )
            if node_id is not None
        }
        if not source_nodes or not target_nodes:
            return False
        current_view = bool(
            asof is None
            or (
                self._last_asof is not None
                and aware_timestamp(
                    asof,
                    name="scene direct-relation asof",
                )
                >= self._last_asof
            )
        )
        edge_by_id = (
            self._edges
            if current_view
            else {
                edge.edge_id: edge for edge in self.edges_asof(asof)
            }
        )
        return any(
            edge is not None
            and edge.lifecycle == "active"
            and edge.relation is expected
            and edge.target_node_id in target_nodes
            for source_node_id in source_nodes
            for edge_id in self._outgoing.get(source_node_id, ())
            for edge in (edge_by_id.get(edge_id),)
        )

    def query(
        self,
        focus_state: FocusState,
        *,
        context_ids: Sequence[str] = (),
        completed_only: bool = True,
        maximum_nodes: int = 256,
        ready_timeframes: Sequence[str] | None = None,
    ) -> FocusedObservation:
        if self._last_asof is None or focus_state.asof > self._last_asof:
            raise ValueError("focus query is ahead of the scene graph")
        if type(maximum_nodes) is not int or maximum_nodes < 1:
            raise ValueError("maximum_nodes must be a positive integer")
        focus_frames = set(focus_state.primary_timeframes)
        ready_frames = (
            None
            if ready_timeframes is None
            else frozenset(str(value) for value in ready_timeframes)
        )
        root_nodes = tuple(
            dict.fromkeys(
            node_id
            for source_id in context_ids
            for node_id in (
                self._node_id_for_source(source_id, asof=focus_state.asof),
            )
            if node_id is not None
            )
        )
        query_epoch = self._epoch_asof(focus_state.asof)
        current_view = focus_state.asof == self._last_asof
        if current_view:
            # The live Brain view is bounded to hot state, this minute's
            # semantic delta and the small amount of terminal context still
            # required by a reversal sequence.  Historical reconstruction
            # remains available through nodes_asof() below.
            candidate_ids = {
                node_id
                for timeframe in focus_frames
                for node_id in self._current_epoch_node_ids_by_timeframe.get(
                    timeframe,
                    (),
                )
            }
            candidate_ids.update(
                node_id
                for node_id in (
                    *self._current_added_nodes,
                    *self._current_revised_nodes,
                )
                if node_id in self._nodes
                and self._nodes[node_id].timeframe in focus_frames
                and self._nodes[node_id].market_epoch_id == query_epoch
            )
            candidate_ids.update(
                node.node_id
                for node in self._recent_terminal_context_nodes(
                    "manipulation",
                    asof=focus_state.asof,
                    maximum_age=pd.Timedelta(minutes=60),
                )
                if node.timeframe in focus_frames
            )
            asof_nodes = tuple(
                self._nodes[node_id]
                for node_id in candidate_ids
                if node_id in self._nodes
            )
            node_by_id = self._nodes
        else:
            asof_nodes = self.nodes_asof(focus_state.asof)
            node_by_id = {node.node_id: node for node in asof_nodes}
        root_nodes = tuple(
            node_id
            for node_id in root_nodes
            if node_id in node_by_id
            and node_by_id[node_id].market_epoch_id == query_epoch
        )
        if len(root_nodes) > maximum_nodes:
            raise ValueError("context roots exceed the bounded scene query")
        priority_ids = list(root_nodes)
        priority_set = set(root_nodes)
        frontier = deque((node_id, 0) for node_id in root_nodes)
        while frontier and len(priority_ids) < maximum_nodes:
            node_id, depth = frontier.popleft()
            if depth >= 3:
                continue
            neighbors = sorted(
                self._neighbors(node_id, asof=focus_state.asof),
                key=lambda item: (
                    item[1].observed_at,
                    item[1].edge_id,
                    item[0],
                ),
            )
            for neighbor, _ in neighbors:
                if (
                    neighbor in priority_set
                    or neighbor not in node_by_id
                    or node_by_id[neighbor].market_epoch_id != query_epoch
                ):
                    continue
                priority_set.add(neighbor)
                priority_ids.append(neighbor)
                frontier.append((neighbor, depth + 1))
                if len(priority_ids) >= maximum_nodes:
                    break
        recent_focus_ids = [
            node.node_id
            for node in sorted(
                asof_nodes,
                key=lambda item: (item.observed_at, item.node_id),
                reverse=True,
            )
            if node.timeframe in focus_frames
            and node.market_epoch_id == query_epoch
            and (not completed_only or node.observed_at <= focus_state.asof)
            and node.node_id not in priority_set
        ]
        selected = {
            *priority_ids,
            *recent_focus_ids[: maximum_nodes - len(priority_ids)],
        }
        nodes = tuple(
            sorted(
                (
                    node_by_id[node_id]
                    for node_id in selected
                    if node_id in node_by_id
                ),
                key=lambda item: (item.observed_at, item.node_id),
            )
        )
        node_ids = {node.node_id for node in nodes}
        if current_view:
            incident_edge_ids = {
                edge_id
                for node_id in node_ids
                for edge_id in (
                    *self._outgoing.get(node_id, ()),
                    *self._incoming.get(node_id, ()),
                )
            }
            asof_edges = (
                *(
                    self._edges[edge_id]
                    for edge_id in incident_edge_ids
                    if edge_id in self._edges
                ),
                *(
                    edge
                    for edge in self._current_path_block_edges.values()
                    if edge.source_node_id in node_ids
                    or edge.target_node_id in node_ids
                ),
            )
        else:
            asof_edges = self.edges_asof(focus_state.asof)
        edges = tuple(
            sorted(
                (
                    edge
                    for edge in asof_edges
                    if edge.source_node_id in node_ids
                    and edge.target_node_id in node_ids
                    and edge.observed_at <= focus_state.asof
                ),
                key=lambda item: (item.observed_at, item.edge_id),
            )
        )
        ambiguities = tuple(
            node.node_id
            for node in nodes
            if node.ambiguity_state in {EvidenceStatus.AMBIGUOUS, EvidenceStatus.UNKNOWN}
        )
        # GlobalMarketContext is the sole authority for conflict
        # classification.  A focused view only projects the already-derived
        # IDs that are present in this bounded query; it never reinterprets an
        # OPPOSES edge on its own.
        authoritative_conflict_ids = (
            self._current_global_material_conflict_ids
            if current_view
            else set()
        )
        conflicts = tuple(
            edge.edge_id
            for edge in edges
            if edge.edge_id in authoritative_conflict_ids
            and (
                ready_frames is None
                or (
                    node_by_id[edge.source_node_id].timeframe
                    in ready_frames
                    and node_by_id[edge.target_node_id].timeframe
                    in ready_frames
                )
            )
        )
        missing = {
            f"{timeframe}.semantic_history": EvidenceStatus.UNKNOWN
            for timeframe in focus_frames
            if (
                ready_frames is not None
                and timeframe not in ready_frames
            )
            or not any(node.timeframe == timeframe for node in nodes)
        }
        if Timeframe.M15.value in focus_frames:
            missing.update(
                {
                    "15m.displacement": EvidenceStatus.UNKNOWN,
                    "15m.fvg": EvidenceStatus.UNKNOWN,
                    "15m.order_block": EvidenceStatus.UNKNOWN,
                }
            )
        paths = tuple(
            path
            for source_id, target_id in zip(context_ids[:-1], context_ids[1:])
            for path in (
                self.find_path(
                    (source_id,),
                    (target_id,),
                    asof=focus_state.asof,
                ),
            )
            if path
        )
        return FocusedObservation(
            asof=focus_state.asof,
            scene_revision_id=max(
                (
                    value.revision_id
                    for value in (*nodes, *edges)
                    if value.revision_id
                ),
                default="scene:empty",
            ),
            focus_state=focus_state,
            nodes=nodes,
            edges=edges,
            graph_paths=paths,
            unresolved_ambiguities=ambiguities,
            cross_scale_conflicts=conflicts,
            missing_evidence=missing,
        )

    def compact_runtime_history(
        self,
        observation: MarketObservation,
        *,
        protected_source_ids: Sequence[str] = (),
    ) -> Mapping[str, Any]:
        """Drop cold graph history for an explicitly bounded runtime replay.

        This is deliberately not called by :meth:`update`.  The calibration
        runner invokes it only after the complete Engine snapshot and every
        enabled recorder/diagnostic have consumed the current bar.  The live
        graph, current reducer state, open root/position identities, current
        delta and sixty minutes of terminal context remain available.

        The revision counter is never renumbered.  Once history is removed,
        as-of queries before :attr:`history_retention_floor` raise instead of
        silently returning a partial graph.
        """

        asof = aware_timestamp(
            observation.asof,
            name="scene_graph.compaction_asof",
        )
        if self._last_asof is None or asof != self._last_asof:
            raise ValueError(
                "scene graph compaction requires the latest Engine observation"
            )
        requested_floor = asof - pd.Timedelta(minutes=60)
        floor = max(
            (
                value
                for value in (
                    self._history_retention_floor,
                    requested_floor,
                )
                if value is not None
            )
        )
        before = {
            "nodes": len(self._nodes),
            "node_revisions": sum(
                len(values) for values in self._node_revisions.values()
            ),
            "edges": len(self._edges),
            "edge_revisions": sum(
                len(values) for values in self._edge_revisions.values()
            ),
            "seen_events": len(self._seen_market_event_ids),
            "seen_displacement_transitions": len(
                self._seen_displacement_transition_ids
            ),
        }

        keep_node_ids = {
            node_id
            for node_id, node in self._nodes.items()
            if node.observed_at >= floor
        }
        keep_node_ids.update(
            node_id
            for node_id in (
                *self._current_added_nodes,
                *self._current_revised_nodes,
                *self._current_resolutions,
            )
            if node_id in self._nodes
        )
        # These indexes are the authoritative long-lived current scene, not
        # a cache that can be reconstructed from one Observation snapshot.
        # In particular, external draws and unresolved ambiguity may remain
        # semantically current long after their reducer stopped emitting a
        # transition.  Dropping them causes the next bar to re-admit facts,
        # drift the revision counter and change Brain context.
        keep_node_ids.update(
            node_id
            for node_id in (
                *self._current_epoch_node_ids,
                *self._current_epoch_ambiguous_node_ids,
            )
            if node_id in self._nodes
        )

        # Every state still materialized by the current Observation must keep
        # its graph identity, including terminal snapshots.  Some reducers
        # intentionally expose terminal states until their own bounded
        # cooling/retention policy expires; deleting those nodes here makes
        # the unchanged snapshot look new on the next bar and drifts both the
        # graph revision and transition delta.  The Observer, not this storage
        # compactor, owns that semantic retention decision.
        for item, _ in self._state_records(observation):
            identity = _entity_id(item)
            if identity is None:
                continue
            kind, entity_id = identity
            lifecycle = _value(getattr(item, "lifecycle", "observed"))
            if kind == "manipulation" and bool(
                getattr(item, "deadline_elapsed", False)
            ):
                lifecycle = "censored"
            timeframe = _value(
                getattr(item, "timeframe", Timeframe.M1)
            )
            node_id = (
                f"{self._market_epoch_id}:{kind}:{timeframe}:{entity_id}"
            )
            if node_id in self._nodes:
                keep_node_ids.add(node_id)
        if (
            observation.displacement is not None
            and observation.displacement.current_entity_id is not None
        ):
            displacement_node_id = (
                f"{self._market_epoch_id}:displacement:"
                f"{Timeframe.M5.value}:"
                f"{observation.displacement.current_entity_id}"
            )
            if displacement_node_id in self._nodes:
                keep_node_ids.add(displacement_node_id)

        protected_node_ids: set[str] = set()
        protected_edge_ids: set[str] = set()
        for source_id in tuple(
            dict.fromkeys(
                str(value)
                for value in protected_source_ids
                if isinstance(value, str) and value
            )
        ):
            if source_id in self._edges:
                protected_edge_ids.add(source_id)
                edge = self._edges[source_id]
                protected_node_ids.update(
                    (edge.source_node_id, edge.target_node_id)
                )
            node_id = self._node_id_for_source(source_id, asof=asof)
            if node_id is not None:
                protected_node_ids.add(node_id)
        for edge_id in (
            *self._current_added_edges,
            *self._current_revised_edges,
            *self._current_epoch_active_opposes_edge_ids,
            *self._current_global_material_conflict_ids,
        ):
            edge = self._edges.get(edge_id)
            if edge is None:
                continue
            protected_edge_ids.add(edge_id)
            protected_node_ids.update(
                (edge.source_node_id, edge.target_node_id)
            )
        keep_node_ids.update(protected_node_ids)

        # Preserve the bounded graph neighborhood used by root binding and
        # connected-sequence checks.  This starts only from explicitly
        # protected roots, not every historical hot node.
        frontier = deque((node_id, 0) for node_id in protected_node_ids)
        traversed = set(protected_node_ids)
        while frontier:
            node_id, depth = frontier.popleft()
            if depth >= 8:
                continue
            for edge_id in (
                *self._outgoing.get(node_id, ()),
                *self._incoming.get(node_id, ()),
            ):
                edge = self._edges.get(edge_id)
                if edge is None or edge.lifecycle != "active":
                    continue
                protected_edge_ids.add(edge_id)
                other = (
                    edge.target_node_id
                    if edge.source_node_id == node_id
                    else edge.source_node_id
                )
                if other in traversed:
                    continue
                traversed.add(other)
                keep_node_ids.add(other)
                frontier.append((other, depth + 1))

        # Frozen source identities are often more precise than a persistent
        # edge (for example an active FVG sourced from an exhausted episode).
        # Retain one causally selected source node per dependency recursively.
        dependency_frontier = deque(keep_node_ids)
        dependency_seen = set(keep_node_ids)
        while dependency_frontier:
            node_id = dependency_frontier.popleft()
            node = self._nodes.get(node_id)
            if node is None:
                continue
            for source_id in node.source_ids:
                source_node_id = self._node_id_for_source(
                    source_id,
                    asof=asof,
                )
                if (
                    source_node_id is None
                    or source_node_id in dependency_seen
                ):
                    continue
                source_node = self._nodes[source_node_id]
                if source_node.market_epoch_id != node.market_epoch_id:
                    continue
                dependency_seen.add(source_node_id)
                keep_node_ids.add(source_node_id)
                dependency_frontier.append(source_node_id)

        def retained_revisions(values: Sequence[Any]) -> list[Any]:
            before_floor = [
                value for value in values if value.observed_at < floor
            ]
            within_window = [
                value for value in values if value.observed_at >= floor
            ]
            return [
                *((before_floor[-1],) if before_floor else ()),
                *within_window,
            ]

        retained_node_revisions: dict[str, list[SceneNode]] = {}
        for node_id in sorted(keep_node_ids):
            values = retained_revisions(
                self._node_revisions.get(node_id, ())
            )
            if values:
                retained_node_revisions[node_id] = values

        keep_edge_ids = {
            edge_id
            for edge_id, edge in self._edges.items()
            if edge.source_node_id in retained_node_revisions
            and edge.target_node_id in retained_node_revisions
        }
        keep_edge_ids.update(
            edge_id
            for edge_id in protected_edge_ids
            if edge_id in self._edges
            and self._edges[edge_id].source_node_id
            in retained_node_revisions
            and self._edges[edge_id].target_node_id
            in retained_node_revisions
        )
        retained_edge_revisions: dict[str, list[SceneEdge]] = {}
        for edge_id in sorted(keep_edge_ids):
            values = retained_revisions(
                self._edge_revisions.get(edge_id, ())
            )
            if values:
                retained_edge_revisions[edge_id] = values

        self._node_revisions = defaultdict(
            list,
            retained_node_revisions,
        )
        self._nodes = {
            node_id: values[-1]
            for node_id, values in retained_node_revisions.items()
        }
        self._edge_revisions = defaultdict(
            list,
            retained_edge_revisions,
        )
        self._edges = {
            edge_id: values[-1]
            for edge_id, values in retained_edge_revisions.items()
        }
        self._current_path_block_edges = {
            edge_id: edge
            for edge_id, edge in self._current_path_block_edges.items()
            if edge.source_node_id in self._nodes
            and edge.target_node_id in self._nodes
        }

        retained_signature_ids = set(self._nodes)
        retained_signatures = {
            key: value
            for key, value in self._state_semantic_signatures.items()
            if key in retained_signature_ids
        }
        retained_tokens = {
            key: value
            for key, value in self._state_lightweight_tokens.items()
            if key in retained_signature_ids
        }
        prior_material_conflict_ids = set(
            self._current_global_material_conflict_ids
        )
        self._outgoing = defaultdict(set)
        self._incoming = defaultdict(set)
        self._source_index = defaultdict(set)
        self._entity_index = defaultdict(set)
        self._clear_current_epoch_indexes()
        for node in sorted(
            self._nodes.values(),
            key=lambda value: (value.observed_at, value.node_id),
        ):
            for source_id in (node.node_id, *node.source_ids):
                self._source_index[source_id].add(node.node_id)
            if node.entity_id is not None:
                self._entity_index[node.entity_id].add(node.node_id)
            self._reindex_current_node(node, prior=None)
        for edge in self._edges.values():
            self._outgoing[edge.source_node_id].add(edge.edge_id)
            self._incoming[edge.target_node_id].add(edge.edge_id)
            if (
                edge.relation is SceneEdgeKind.OPPOSES
                and edge.lifecycle == "active"
                and self._nodes[edge.source_node_id].market_epoch_id
                == self._market_epoch_id
                and self._nodes[edge.target_node_id].market_epoch_id
                == self._market_epoch_id
            ):
                self._current_epoch_active_opposes_edge_ids.add(
                    edge.edge_id
                )
        self._current_global_material_conflict_ids = (
            prior_material_conflict_ids.intersection(self._edges)
        )
        self._state_semantic_signatures = retained_signatures
        self._state_lightweight_tokens = retained_tokens
        self._state_catalog = {}

        observed_events = {
            event.event_id: event
            for event in observation.recent_events
        }
        for timeline in observation.retained_entity_timelines.values():
            for event in timeline:
                observed_events[event.event_id] = event
        self._seen_market_event_ids = {
            event_id
            for event_id, event in observed_events.items()
            if event.observed_at >= floor
            and event_id in self._seen_market_event_ids
        }
        observed_transitions = (
            ()
            if observation.displacement is None
            else observation.displacement.recent_transitions
        )
        self._seen_displacement_transition_ids = {
            transition.transition_id
            for transition in observed_transitions
            if transition.observed_at >= floor
            and transition.transition_id
            in self._seen_displacement_transition_ids
        }

        earlier_boundaries = [
            value for value in self._epoch_boundaries if value[0] < floor
        ]
        self._epoch_boundaries = [
            *((earlier_boundaries[-1],) if earlier_boundaries else ()),
            *(
                value
                for value in self._epoch_boundaries
                if value[0] >= floor
            ),
        ]
        self._history_retention_floor = floor
        after = {
            "nodes": len(self._nodes),
            "node_revisions": sum(
                len(values) for values in self._node_revisions.values()
            ),
            "edges": len(self._edges),
            "edge_revisions": sum(
                len(values) for values in self._edge_revisions.values()
            ),
            "seen_events": len(self._seen_market_event_ids),
            "seen_displacement_transitions": len(
                self._seen_displacement_transition_ids
            ),
        }
        return {
            "asof": asof,
            "history_retention_floor": floor,
            "revision_id": self._revision_id,
            "before": before,
            "after": after,
        }

    def write_parquet(self, destination: str | Path) -> Mapping[str, Path]:
        """Materialize node/edge revision ledgers without introducing a DB."""

        if self._history_retention_floor is not None:
            raise ValueError(
                "complete scene graph export is unavailable after runtime compaction"
            )

        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        node_path = root / "scene_nodes.parquet"
        edge_path = root / "scene_edges.parquet"
        node_rows = [to_primitive(item) for values in self._node_revisions.values() for item in values]
        edge_rows = [to_primitive(item) for values in self._edge_revisions.values() for item in values]
        pd.DataFrame(node_rows).to_parquet(node_path, index=False)
        pd.DataFrame(edge_rows).to_parquet(edge_path, index=False)
        return {"nodes": node_path, "edges": edge_path}


_GLOBAL_SCALE_ORDER: tuple[Timeframe, ...] = (
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M15,
    Timeframe.M5,
    Timeframe.M1,
)
_GLOBAL_SCALE_RANK: Mapping[str, int] = {
    timeframe.value: len(_GLOBAL_SCALE_ORDER) - index
    for index, timeframe in enumerate(_GLOBAL_SCALE_ORDER)
}
_GLOBAL_CONTEXT_NODE_KINDS = frozenset(
    {
        "structure",
        "bos",
        "displacement",
        "range",
        "liquidity",
        "support_resistance",
        "fvg",
        "order_block",
        "manipulation",
        "manipulation_candidate",
        "micro_bos",
    }
)


def _context_identity(node: SceneNode) -> str:
    return node.entity_id or node.node_id


def _current_context_nodes(
    graph: TemporalMarketSceneGraph,
    kind: str,
) -> tuple[SceneNode, ...]:
    """Read one bounded hot-state index, never the historical ledger."""

    return graph._current_nonterminal_kind_nodes(kind)


def _global_authority(
    graph: TemporalMarketSceneGraph,
    *,
    ready_timeframes: frozenset[str],
) -> tuple[AuthorityLayer, ...]:
    """Build a bounded H4/H1/M15 authority stack from current indexes."""

    authority_frames = (Timeframe.H4, Timeframe.H1, Timeframe.M15)
    structures = tuple(
        node
        for node in _current_context_nodes(graph, "structure")
        if node.lifecycle == StructureLifecycle.CONFIRMED.value
        and node.ambiguity_state is EvidenceStatus.CONFIRMED
        and node.direction is not None
        and node.timeframe in ready_timeframes
        and node.timeframe in {value.value for value in authority_frames}
    )
    accepted_boses = tuple(
        node
        for node in _current_context_nodes(graph, "bos")
        if node.lifecycle == "confirmed"
        and node.ambiguity_state is EvidenceStatus.CONFIRMED
        and node.direction is not None
        and node.timeframe in ready_timeframes
        and node.timeframe in {value.value for value in authority_frames}
        and node.structural_scale
        in {StructuralScale.INTERMEDIATE, StructuralScale.EXTERNAL}
        and dict(node.semantic_attributes).get("post_break_state")
        == "accepted"
    )

    def connected(source: SceneNode, target: SceneNode) -> bool:
        target_ids = {
            target.node_id,
            _context_identity(target),
            *target.source_ids,
        }
        if not set(source.source_ids).isdisjoint(target_ids):
            return True
        for edge_id in (
            graph._outgoing.get(source.node_id, set())
            | graph._incoming.get(source.node_id, set())
        ):
            edge = graph._edges.get(edge_id)
            if edge is None or edge.lifecycle != "active":
                continue
            peer_id = (
                edge.target_node_id
                if edge.source_node_id == source.node_id
                else edge.source_node_id
            )
            peer = graph._nodes.get(peer_id)
            if peer is not None and not target_ids.isdisjoint(
                {peer.node_id, _context_identity(peer), *peer.source_ids}
            ):
                return True
        return False

    layers: list[tuple[AuthorityLayer, SceneNode]] = []
    dominant_node: SceneNode | None = None
    h1_challenger_node: SceneNode | None = None
    for timeframe in authority_frames:
        frame_structures = tuple(
            node for node in structures if node.timeframe == timeframe.value
        )
        structure_directions = {node.direction for node in frame_structures}
        structure_owner = (
            None
            if len(structure_directions) != 1
            else max(
                frame_structures,
                key=lambda value: (
                    value.confirmed_at or value.formed_at,
                    value.observed_at,
                    value.node_id,
                ),
                default=None,
            )
        )
        frame_boses = tuple(
            node for node in accepted_boses if node.timeframe == timeframe.value
        )
        challenger = None
        comparison_node = (
            h1_challenger_node
            if timeframe is Timeframe.M15
            and h1_challenger_node is not None
            else dominant_node
        )
        if comparison_node is not None and timeframe is Timeframe.H1:
            challenger = max(
                (
                    node
                    for node in frame_boses
                    if node.direction is not comparison_node.direction
                    and connected(node, comparison_node)
                ),
                key=lambda value: (
                    value.confirmed_at or value.formed_at,
                    value.observed_at,
                    value.node_id,
                ),
                default=None,
            )
        m15_bridge = None
        if timeframe is Timeframe.M15 and h1_challenger_node is not None:
            m15_bridge = max(
                (
                    node
                    for node in frame_boses
                    if node.direction is h1_challenger_node.direction
                    and connected(node, h1_challenger_node)
                ),
                key=lambda value: (
                    value.confirmed_at or value.formed_at,
                    value.observed_at,
                    value.node_id,
                ),
                default=None,
            )
        # A newly accepted M15 bridge must not be hidden by an older M15
        # structure snapshot.  It can confirm the H1 challenger, but never
        # replaces the H4 incumbent by itself.
        owner = challenger or m15_bridge or structure_owner
        if owner is None and dominant_node is None:
            owner = max(
                frame_boses,
                key=lambda value: (
                    value.confirmed_at or value.formed_at,
                    value.observed_at,
                    value.node_id,
                ),
                default=None,
            )
        if owner is None:
            continue
        attributes = dict(owner.semantic_attributes)
        acceptance = (
            attributes.get("post_break_state")
            if owner.kind == "bos"
            else "confirmed"
        ) or "unknown"
        status = "intact"
        if (
            timeframe is Timeframe.M15
            and h1_challenger_node is not None
            and owner.direction is h1_challenger_node.direction
            and connected(owner, h1_challenger_node)
        ):
            status = "challenging"
        elif (
            dominant_node is not None
            and owner.direction is not dominant_node.direction
        ):
            if challenger is not None:
                status = "challenging"
        protected = attributes.get("protected_swing_id")
        source_ids = tuple(
            dict.fromkeys((_context_identity(owner), *owner.source_ids))
        )
        layer = AuthorityLayer(
            timeframe=timeframe,
            direction=owner.direction,
            structure_id=(
                attributes.get("source_structure_id")
                or _context_identity(owner)
            ),
            confirmed_at=(
                owner.observed_at
                if owner.kind == "bos" and acceptance == "accepted"
                else owner.confirmed_at or owner.formed_at
            ),
            protected_level_id=protected,
            structural_scope=owner.structural_scale.value,
            acceptance_state=acceptance,
            status=status,
            source_ids=source_ids,
        )
        layers.append((layer, owner))
        if dominant_node is None and status == "intact":
            dominant_node = owner
        if timeframe is Timeframe.H1 and status == "challenging":
            h1_challenger_node = owner
    return tuple(layer for layer, _ in layers)


def _global_material_conflicts(
    graph: TemporalMarketSceneGraph,
    *,
    ready_timeframes: frozenset[str],
) -> tuple[GlobalConflictEvidence, ...]:
    output: list[GlobalConflictEvidence] = []
    for edge_id in sorted(graph._current_epoch_active_opposes_edge_ids):
        edge = graph._edges.get(edge_id)
        if edge is None:
            continue
        role = _cross_scale_conflict_role(
            edge,
            graph._nodes,
            ready_timeframes=ready_timeframes,
        )
        if role is None:
            continue
        source = graph._nodes[edge.source_node_id]
        target = graph._nodes[edge.target_node_id]
        if (
            source.market_epoch_id != graph._market_epoch_id
            or target.market_epoch_id != graph._market_epoch_id
            or source.direction is None
        ):
            continue
        affected = tuple(
            dict.fromkeys(
                f"{playbook.value}:{direction.value}"
                for direction in (source.direction, target.direction)
                for playbook in Playbook
                )
        )
        output.append(
            GlobalConflictEvidence(
                conflict_id=edge.edge_id,
                event_id=_context_identity(source),
                observed_at=(
                    max(edge.observed_at, source.observed_at)
                    if role
                    is GlobalConflictRole.AUTHORITY_TRANSITION_CANDIDATE
                    else edge.observed_at
                ),
                source_node_id=source.node_id,
                target_node_id=target.node_id,
                source_timeframe=Timeframe(source.timeframe),
                target_timeframe=Timeframe(target.timeframe),
                source_direction=source.direction,
                target_direction=target.direction,
                structural_scale=source.structural_scale.value,
                role=role,
                reason=role.value,
                affected_hypothesis_ids=affected,
            )
        )
    return tuple(
        sorted(output, key=lambda value: (value.observed_at, value.conflict_id))
    )


def _global_terminal_authority_conflicts(
    graph: TemporalMarketSceneGraph,
    delta: SceneGraphDelta,
    *,
    ready_timeframes: frozenset[str],
) -> tuple[GlobalConflictEvidence, ...]:
    """Recover authority invalidation from OPPOSES edges closed this update.

    Terminal node admission correctly removes an OPPOSES edge from the live
    index.  The closed edge revision still carries the exact causal endpoints,
    so consume only this minute's revised-edge delta instead of reopening it or
    scanning graph history.
    """

    output: list[GlobalConflictEvidence] = []
    changed_node_ids = set(
        (*delta.added_node_ids, *delta.revised_node_ids)
    )
    for edge_id in dict.fromkeys(delta.revised_edge_ids):
        edge = graph._edges.get(edge_id)
        if (
            edge is None
            or edge.relation is not SceneEdgeKind.OPPOSES
            or edge.lifecycle != "closed"
            or edge.observed_at != delta.asof
        ):
            continue
        revisions = graph._edge_revisions.get(edge_id, ())
        if len(revisions) < 2 or revisions[-2].lifecycle != "active":
            continue
        source = graph._nodes.get(edge.source_node_id)
        target = graph._nodes.get(edge.target_node_id)
        if (
            source is None
            or target is None
            or target.node_id not in changed_node_ids
            or source.market_epoch_id != graph._market_epoch_id
            or target.market_epoch_id != graph._market_epoch_id
            or source.timeframe not in ready_timeframes
            or target.timeframe not in ready_timeframes
            or source.kind not in {"structure", "bos", "displacement"}
            or target.kind != "structure"
            or _is_terminal(source.kind, source.lifecycle)
            or source.ambiguity_state is not EvidenceStatus.CONFIRMED
            or target.structural_scale is not StructuralScale.EXTERNAL
            or not _is_terminal(target.kind, target.lifecycle)
            or source.direction is None
            or target.direction is None
            or source.direction is target.direction
        ):
            continue
        affected = tuple(
            dict.fromkeys(
                f"{playbook.value}:{direction.value}"
                for direction in (source.direction, target.direction)
                for playbook in Playbook
            )
        )
        output.append(
            GlobalConflictEvidence(
                conflict_id=edge.edge_id,
                event_id=_context_identity(source),
                observed_at=edge.observed_at,
                source_node_id=source.node_id,
                target_node_id=target.node_id,
                source_timeframe=Timeframe(source.timeframe),
                target_timeframe=Timeframe(target.timeframe),
                source_direction=source.direction,
                target_direction=target.direction,
                structural_scale=source.structural_scale.value,
                role=GlobalConflictRole.AUTHORITY_INVALIDATION,
                reason=GlobalConflictRole.AUTHORITY_INVALIDATION.value,
                affected_hypothesis_ids=affected,
            )
        )
    return tuple(
        sorted(output, key=lambda value: (value.observed_at, value.conflict_id))
    )


def _global_external_draws(
    graph: TemporalMarketSceneGraph,
    observation: MarketObservation,
) -> Mapping[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {}
    for side in ("above", "below"):
        candidates = tuple(
            graph._nodes[node_id]
            for node_id in graph._visible_liquidity_node_ids_by_side[side]
            if node_id in graph._nodes
            and graph._nodes[node_id].market_epoch_id
            == graph._market_epoch_id
            and graph._nodes[node_id].structural_scale
            is StructuralScale.EXTERNAL
            and graph._nodes[node_id].price_bounds is not None
        )
        ordered = sorted(
            candidates,
            key=lambda node: (
                abs(sum(node.price_bounds) / 2.0 - observation.price),
                node.node_id,
            ),
        )
        output[side] = tuple(
            dict.fromkeys(_context_identity(node) for node in ordered)
        )
    return output


def _global_dislocations_by_scale(
    graph: TemporalMarketSceneGraph,
) -> Mapping[str, tuple[str, ...]]:
    output: dict[str, tuple[str, ...]] = {
        timeframe.value: () for timeframe in Timeframe
    }
    grouped: dict[str, list[str]] = defaultdict(list)
    for node in _current_context_nodes(graph, "displacement"):
        if (
            node.lifecycle == "active"
            and node.ambiguity_state is EvidenceStatus.CONFIRMED
            and node.timeframe in output
        ):
            grouped[node.timeframe].append(_context_identity(node))
    for timeframe, identities in grouped.items():
        output[timeframe] = tuple(dict.fromkeys(sorted(identities)))
    return output


def _current_entity_node(
    graph: TemporalMarketSceneGraph,
    identity: str,
) -> SceneNode | None:
    """Resolve one identity from the bounded current epoch only."""

    candidates = tuple(
        graph._nodes[node_id]
        for node_id in graph._entity_index.get(identity, set())
        if node_id in graph._nodes
        and graph._nodes[node_id].market_epoch_id == graph._market_epoch_id
    )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda node: (node.observed_at, node.node_id),
    )


def _global_balance_context(
    observation: MarketObservation,
    graph: TemporalMarketSceneGraph,
) -> BalanceContext | None:
    """Summarize current Group4 state without granting FAVR authority."""

    ranges = tuple(
        state
        for timeframe in (Timeframe.H4, Timeframe.H1, Timeframe.M15)
        if timeframe in observation.active_timeframes
        for state in observation.frame(timeframe).dealing_ranges
        if state.lifecycle.value in {"active", "broken"}
    )
    if not ranges:
        range_nodes = tuple(
            node
            for node in _current_context_nodes(graph, "range")
            if node.timeframe in {
                Timeframe.H4.value,
                Timeframe.H1.value,
                Timeframe.M15.value,
            }
            and node.lifecycle == "active"
            and node.ambiguity_state
            in {EvidenceStatus.CONFIRMED, EvidenceStatus.FORMING}
        )
        if not range_nodes:
            return None
        node = max(
            range_nodes,
            key=lambda value: (
                value.resolution_reason == BALANCE_CLAIM_CONFIRMED,
                value.confirmed_at or value.formed_at,
                value.node_id,
            ),
        )
        identity = _context_identity(node)
        attributes = dict(node.semantic_attributes)
        metrics = dict(node.descriptive_metrics)
        bilateral = bool(
            attributes.get("lower_source_zone_id")
            and attributes.get("upper_source_zone_id")
            and metrics.get("lower_touch_count", 0.0) > 0.0
            and metrics.get("upper_touch_count", 0.0) > 0.0
        )
        internal_crossing = metrics.get("midpoint_crossings", 0.0) > 0.0
        status = (
            "authoritative"
            if node.resolution_reason == BALANCE_CLAIM_CONFIRMED
            else "descriptive"
            if bilateral
            and internal_crossing
            and metrics.get("inside_close_fraction", 0.0) > 0.0
            else "candidate"
        )
        return BalanceContext(
            context_id=identity,
            timeframe=Timeframe(node.timeframe),
            status=status,
            source_ids=tuple(
                dict.fromkeys((identity, node.node_id, *node.source_ids))
            ),
            bilateral_boundaries=(
                True if status == "authoritative" else bilateral
            ),
            internal_crossing=(
                True if status == "authoritative" else internal_crossing
            ),
            accepted_external_break=False,
            value_authoritative=status == "authoritative",
        )
    state = max(
        ranges,
        key=lambda value: (
            value.balance_confirmed_at is not None,
            value.balance_confirmed_at or value.formed_at,
            value.range_id,
        ),
    )
    bilateral = bool(
        state.lower_source_zone_id
        and state.upper_source_zone_id
        and state.lower_touch_count > 0
        and state.upper_touch_count > 0
    )
    internal_crossing = state.midpoint_crossings > 0
    accepted_break = (
        state.lifecycle.value == "broken"
        and state.transition_reason == "close_beyond_frozen_range"
    )
    if state.balance_confirmed_at is not None:
        status = "authoritative"
    elif (
        bilateral
        and internal_crossing
        and state.inside_close_fraction > 0.0
        and not accepted_break
    ):
        status = "descriptive"
    else:
        status = "candidate"
    return BalanceContext(
        context_id=state.range_id,
        timeframe=state.timeframe,
        status=status,
        source_ids=tuple(
            dict.fromkeys(
                (
                    state.range_id,
                    state.lower_source_zone_id,
                    state.upper_source_zone_id,
                    *state.lower_source_member_swing_ids,
                    *state.upper_source_member_swing_ids,
                )
            )
        ),
        bilateral_boundaries=bilateral,
        internal_crossing=internal_crossing,
        accepted_external_break=accepted_break,
        value_authoritative=status == "authoritative",
    )


def _global_obstruction_views(
    graph: TemporalMarketSceneGraph,
    observation: MarketObservation,
    draws: Mapping[str, tuple[str, ...]],
    authority_stack: Sequence[AuthorityLayer] = (),
) -> Mapping[str, DirectionalObstructionView]:
    """Build clustered obstruction facts; Brain applies plan-specific gates.

    ``BLOCKS_PATH_TO`` is only a current-price ordering relation.  It must not
    promote every nearer liquidity item to structural authority.  Hard facts
    are limited to current protected authority, mature-range boundaries and
    accepted structural breaks (plus zones causally owned by those breaks).
    """

    candidates: dict[str, DeliveryObstruction] = {}

    timeframe_rank = {
        Timeframe.H4: 5,
        Timeframe.H1: 4,
        Timeframe.M15: 3,
        Timeframe.M5: 2,
        Timeframe.M1: 1,
    }
    scope_rank = {"internal": 1, "intermediate": 2, "external": 3}

    authority_source_ids = {
        identity
        for layer in authority_stack
        for identity in (
            layer.structure_id,
            layer.protected_level_id,
            *layer.source_ids,
        )
        if identity is not None
    }
    mature_range_source_ids = {
        identity
        for node in _current_context_nodes(graph, "range")
        if node.resolution_reason == BALANCE_CLAIM_CONFIRMED
        for identity in (
            _context_identity(node),
            node.node_id,
            *node.source_ids,
        )
    }

    def accepted_structural_bos(node: SceneNode) -> bool:
        attributes = dict(node.semantic_attributes)
        scope = attributes.get("scope")
        return bool(
            attributes.get("post_break_state") == "accepted"
            and (
                scope == "continuation"
                or (
                    scope == "opposed"
                    and attributes.get("mss_qualified") == "true"
                )
            )
        )

    accepted_bos_nodes = tuple(
        node
        for node in _current_context_nodes(graph, "bos")
        if (
            node.lifecycle == "confirmed"
            and node.direction is not None
            and node.timeframe
            in {
                Timeframe.H4.value,
                Timeframe.H1.value,
                Timeframe.M15.value,
            }
            and node.structural_scale
            in {StructuralScale.INTERMEDIATE, StructuralScale.EXTERNAL}
            and accepted_structural_bos(node)
            and any(
                _node_connected_to_layer(graph, node, layer)
                for layer in authority_stack
            )
        )
    )
    # A completed accepted BOS remains a fact, but older same-role breaks do
    # not all remain independent barriers.  Keep the latest current identity
    # for each structural role.
    latest_accepted_bos: dict[tuple[str, Direction, StructuralScale], SceneNode] = {}
    for node in accepted_bos_nodes:
        key = (node.timeframe, node.direction, node.structural_scale)
        incumbent = latest_accepted_bos.get(key)
        if incumbent is None or (node.observed_at, node.node_id) > (
            incumbent.observed_at,
            incumbent.node_id,
        ):
            latest_accepted_bos[key] = node
    accepted_bos_nodes = tuple(latest_accepted_bos.values())
    def add_candidate(node: SceneNode, *, hard: bool, source_kind: str) -> None:
        if node.price_bounds is None or node.timeframe not in {
            timeframe.value for timeframe in Timeframe
        }:
            return
        lower, upper = node.price_bounds
        side = (
            "above"
            if (lower + upper) / 2.0 >= observation.price
            else "below"
        )
        attributes = dict(node.semantic_attributes)
        identity = _context_identity(node)
        candidates[identity] = DeliveryObstruction(
            obstruction_id=identity,
            timeframe=Timeframe(node.timeframe),
            direction=node.direction,
            side=side,
            lower_bound=lower,
            upper_bound=upper,
            hard=hard,
            source_kind=source_kind,
            source_ids=tuple(
                dict.fromkeys((identity, node.node_id, *node.source_ids))
            ),
            structural_scope=node.structural_scale.value,
            acceptance_state=(
                attributes.get("post_break_state")
                or ("confirmed" if node.confirmed_at is not None else "pending")
            ),
        )

    # Protection is a current structural fact, not a liquidity lifecycle.
    # A wick may consume the corresponding draw without closing the confirmed
    # structure, and the liquidity node may also retain pre-promotion metadata.
    # Resolve the frozen raw swing directly from each intact authority layer so
    # neither condition can silently remove the structural path boundary.
    for layer in authority_stack:
        if (
            layer.status != "intact"
            or layer.timeframe
            not in {Timeframe.H4, Timeframe.H1, Timeframe.M15}
            or layer.protected_level_id is None
        ):
            continue
        swing_node_id = graph._node_id_for_source(
            layer.protected_level_id,
            kind="swing",
        )
        swing = (
            None
            if swing_node_id is None
            else graph._nodes.get(swing_node_id)
        )
        if (
            swing is None
            or swing.market_epoch_id != graph._market_epoch_id
            or swing.timeframe != layer.timeframe.value
            or swing.lifecycle != SwingLifecycle.CONFIRMED.value
            or swing.ambiguity_state is not EvidenceStatus.CONFIRMED
            or swing.price_bounds is None
        ):
            continue
        lower, upper = swing.price_bounds
        identity = (
            "authority_protected_boundary:"
            f"{layer.timeframe.value}:{layer.structure_id}:"
            f"{layer.protected_level_id}"
        )
        candidates[identity] = DeliveryObstruction(
            obstruction_id=identity,
            timeframe=layer.timeframe,
            direction=layer.direction,
            side=(
                "above"
                if (lower + upper) / 2.0 >= observation.price
                else "below"
            ),
            lower_bound=lower,
            upper_bound=upper,
            hard=True,
            source_kind="authority_protected_boundary",
            source_ids=tuple(
                dict.fromkeys(
                    (
                        identity,
                        layer.structure_id,
                        layer.protected_level_id,
                        *layer.source_ids,
                        swing.node_id,
                        *swing.source_ids,
                    )
                )
            ),
            structural_scope="external",
            acceptance_state="confirmed",
        )

    for node in _current_context_nodes(graph, "liquidity"):
        attributes = dict(node.semantic_attributes)
        inventory_kind = attributes.get("inventory_kind", "liquidity")
        protected = attributes.get("is_protected_swing") == "true"
        identities = {
            _context_identity(node),
            node.node_id,
            *node.source_ids,
        }
        hard = (
            protected
            and node.timeframe
            in {
                Timeframe.H4.value,
                Timeframe.H1.value,
                Timeframe.M15.value,
            }
            and not identities.isdisjoint(authority_source_ids)
        ) or (
            inventory_kind == "range_boundary"
            and not identities.isdisjoint(mature_range_source_ids)
        )
        if hard:
            add_candidate(node, hard=True, source_kind=inventory_kind)
    for node in accepted_bos_nodes:
        add_candidate(node, hard=True, source_kind="accepted_bos")

    def zone_has_accepted_consequence(node: SceneNode) -> bool:
        """Resolve the zone's causal owners through graph identities.

        FVG/OB source identities are deliberately stored on nodes/edges; they
        are not duplicated into presentation metadata.  A zone becomes a hard
        candidate only when one of those frozen owners also belongs to a
        current, authority-connected accepted non-local BOS.
        """

        zone_owner_ids = {
            node.node_id,
            _context_identity(node),
            *node.source_ids,
        }
        for edge_id in graph._incoming.get(node.node_id, set()):
            edge = graph._edges.get(edge_id)
            if (
                edge is None
                or edge.lifecycle != "active"
                or edge.relation is not SceneEdgeKind.CREATES
            ):
                continue
            owner = graph._nodes.get(edge.source_node_id)
            if owner is None or owner.kind not in {"bos", "displacement"}:
                continue
            zone_owner_ids.update(
                (owner.node_id, _context_identity(owner), *owner.source_ids)
            )
        return any(
            not zone_owner_ids.isdisjoint(
                {
                    bos.node_id,
                    _context_identity(bos),
                    *bos.source_ids,
                }
            )
            for bos in accepted_bos_nodes
        )

    for kind in ("fvg", "order_block"):
        for node in _current_context_nodes(graph, kind):
            attributes = dict(node.semantic_attributes)
            qualified = (
                kind == "order_block"
                or (
                    attributes.get("qualification")
                    == "displacement_linked"
                )
            )
            if qualified:
                add_candidate(
                    node,
                    hard=zone_has_accepted_consequence(node),
                    source_kind=kind,
                )
    for node in _current_context_nodes(graph, "support_resistance"):
        if node.lifecycle == "tested":
            add_candidate(
                node,
                hard=False,
                source_kind="support_resistance",
            )

    def priority(item: DeliveryObstruction) -> tuple[int, int, int, int, str]:
        return (
            int(item.hard),
            timeframe_rank[item.timeframe],
            scope_rank[item.structural_scope],
            int(item.acceptance_state in {"accepted", "confirmed"}),
            item.obstruction_id,
        )

    price_epsilon = max(1e-9, abs(float(observation.price)) * 1e-10)

    def consolidate(
        values: Sequence[DeliveryObstruction],
    ) -> tuple[DeliveryObstruction, ...]:
        """Collapse co-located/nested facts into one conservative zone."""

        clusters: list[list[DeliveryObstruction]] = []
        cluster_upper_bounds: list[float] = []
        for item in sorted(
            values,
            key=lambda value: (
                value.lower_bound,
                value.upper_bound,
                value.obstruction_id,
            ),
        ):
            # Values are ordered by lower bound.  Comparing with the current
            # cluster's union upper edge gives transitive interval/tick
            # clustering without an all-pairs scan.
            cluster = clusters[-1] if clusters else None
            if (
                cluster is None
                or item.lower_bound
                > cluster_upper_bounds[-1] + price_epsilon
            ):
                clusters.append([item])
                cluster_upper_bounds.append(item.upper_bound)
            else:
                cluster.append(item)
                cluster_upper_bounds[-1] = max(
                    cluster_upper_bounds[-1],
                    item.upper_bound,
                )
        output: list[DeliveryObstruction] = []
        for cluster in clusters:
            representative = max(cluster, key=priority)
            sources = tuple(
                dict.fromkeys(
                    identity
                    for item in sorted(cluster, key=priority, reverse=True)
                    for identity in (item.obstruction_id, *item.source_ids)
                )
            )
            output.append(
                replace(
                    representative,
                    lower_bound=min(item.lower_bound for item in cluster),
                    upper_bound=max(item.upper_bound for item in cluster),
                    hard=any(item.hard for item in cluster),
                    source_ids=sources,
                )
            )
        return tuple(output)

    output: dict[str, DirectionalObstructionView] = {}
    for direction, side in (
        (Direction.LONG, "above"),
        (Direction.SHORT, "below"),
    ):
        draw_pairs = tuple(
            (identity, _current_entity_node(graph, identity))
            for identity in draws[side]
        )
        nearest_pair = next(
            (
                (identity, node)
                for identity, node in draw_pairs
                if node is not None and node.price_bounds is not None
            ),
            None,
        )
        direction_candidates = consolidate(
            sorted(
                (
                    item
                    for item in candidates.values()
                    if (
                        item.source_kind
                        not in {"accepted_bos", "fvg", "order_block"}
                        or item.direction is None
                        or item.direction is not direction
                    )
                ),
                key=lambda item: (
                    abs(item.contact_price(direction) - observation.price),
                    item.obstruction_id,
                ),
            )
        )
        ranked_candidates = tuple(
            sorted(
                direction_candidates,
                key=lambda item: (
                    abs(
                        item.contact_price(direction)
                        - observation.price
                    ),
                    item.obstruction_id,
                ),
            )
        )
        output[direction.value] = DirectionalObstructionView(
            direction=direction,
            nearest_draw_id=(None if nearest_pair is None else nearest_pair[0]),
            nearest_draw_price=(
                None
                if nearest_pair is None
                else (
                    nearest_pair[1].price_bounds[0]
                    if direction is Direction.LONG
                    else nearest_pair[1].price_bounds[1]
                )
            ),
            hard_barriers=tuple(item for item in ranked_candidates if item.hard),
            soft_frictions=tuple(item for item in ranked_candidates if not item.hard),
        )
    return output


def _refresh_obstruction_draws(
    graph: TemporalMarketSceneGraph,
    views: Mapping[str, DirectionalObstructionView],
    draws: Mapping[str, tuple[str, ...]],
) -> Mapping[str, DirectionalObstructionView]:
    """Refresh price-ranked draw heads without rebuilding obstruction facts."""

    output: dict[str, DirectionalObstructionView] = {}
    for direction, side in (
        (Direction.LONG, "above"),
        (Direction.SHORT, "below"),
    ):
        nearest: tuple[str, SceneNode] | None = None
        for identity in draws[side]:
            node = _current_entity_node(graph, identity)
            if node is not None and node.price_bounds is not None:
                nearest = (identity, node)
                break
        prior = views[direction.value]
        output[direction.value] = replace(
            prior,
            nearest_draw_id=None if nearest is None else nearest[0],
            nearest_draw_price=(
                None
                if nearest is None
                else (
                    nearest[1].price_bounds[0]
                    if direction is Direction.LONG
                    else nearest[1].price_bounds[1]
                )
            ),
        )
    return output


def _dominant_authority_layer(
    stack: Sequence[AuthorityLayer],
) -> AuthorityLayer | None:
    return next((layer for layer in stack if layer.status == "intact"), None)


def _timeframe_age_bars(
    since: pd.Timestamp | None,
    asof: pd.Timestamp,
    timeframe: Timeframe,
) -> int:
    """Query-time descriptive age; it never creates a graph revision."""

    if since is None:
        return 0
    seconds = {
        Timeframe.H4: 4 * 60 * 60,
        Timeframe.H1: 60 * 60,
        Timeframe.M15: 15 * 60,
        Timeframe.M5: 5 * 60,
        Timeframe.M1: 60,
    }[timeframe]
    return max(0, int((asof - since).total_seconds() // seconds))


def _relation_evidence_since(
    node: SceneNode,
    *,
    conflict_observed_at: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Use acceptance, not pre-acceptance BOS confirmation, as evidence time."""

    attributes = dict(node.semantic_attributes)
    if node.kind == "bos" and attributes.get("post_break_state") == "accepted":
        return max(
            value
            for value in (node.observed_at, conflict_observed_at)
            if value is not None
        )
    return node.confirmed_at or node.formed_at


def _node_connected_to_layer(
    graph: TemporalMarketSceneGraph,
    node: SceneNode,
    layer: AuthorityLayer,
) -> bool:
    layer_ids = {
        layer.structure_id,
        *layer.source_ids,
        *((layer.protected_level_id,) if layer.protected_level_id else ()),
    }
    if not set((node.node_id, _context_identity(node), *node.source_ids)).isdisjoint(
        layer_ids
    ):
        return True
    for edge_id in (
        graph._outgoing.get(node.node_id, set())
        | graph._incoming.get(node.node_id, set())
    ):
        edge = graph._edges.get(edge_id)
        if edge is None or edge.lifecycle != "active":
            continue
        peer_id = (
            edge.target_node_id
            if edge.source_node_id == node.node_id
            else edge.source_node_id
        )
        peer = graph._nodes.get(peer_id)
        if peer is not None and not layer_ids.isdisjoint(
            {peer.node_id, _context_identity(peer), *peer.source_ids}
        ):
            return True
    return False


def _global_scale_relations(
    graph: TemporalMarketSceneGraph,
    *,
    observation: MarketObservation,
    ready_timeframes: frozenset[str],
    authority_stack: tuple[AuthorityLayer, ...],
    conflicts: tuple[GlobalConflictEvidence, ...],
) -> tuple[Mapping[str, ScaleRelationState], tuple[str, ...], tuple[str, ...]]:
    active = {timeframe.value for timeframe in observation.active_timeframes}
    dominant = _dominant_authority_layer(authority_stack)
    authority_direction = None if dominant is None else dominant.direction
    authority_ids = (
        set()
        if dominant is None
        else {
            dominant.structure_id,
            *dominant.source_ids,
            *((dominant.protected_level_id,) if dominant.protected_level_id else ()),
        }
    )
    current_candidates = tuple(
        node
        for kind in ("bos", "structure", "displacement")
        for node in _current_context_nodes(graph, kind)
        if node.ambiguity_state
        in {EvidenceStatus.CONFIRMED, EvidenceStatus.AMBIGUOUS}
        and node.direction is not None
    )
    details: dict[str, ScaleRelationState] = {}
    unknown: list[str] = []
    ambiguous: list[str] = []
    for timeframe in _GLOBAL_SCALE_ORDER:
        key = timeframe.value
        layer = next(
            (value for value in authority_stack if value.timeframe is timeframe),
            None,
        )
        frame_candidates = tuple(
            node for node in current_candidates if node.timeframe == key
        )
        ambiguous_nodes = tuple(
            node
            for node in frame_candidates
            if node.ambiguity_state is EvidenceStatus.AMBIGUOUS
        )
        priority = {"bos": 3, "structure": 2, "displacement": 1}
        current_clock_candidates = tuple(
            node
            for node in frame_candidates
            if node.observed_at == observation.asof
        )
        current_priority = max(
            (priority.get(node.kind, 0) for node in current_clock_candidates),
            default=0,
        )
        same_clock_top = tuple(
            node
            for node in current_clock_candidates
            if priority.get(node.kind, 0) == current_priority
        )
        same_clock_ambiguous = len(
            {node.direction for node in same_clock_top}
        ) > 1
        if ambiguous_nodes or same_clock_ambiguous:
            evidence = tuple(
                dict.fromkeys(_context_identity(node) for node in frame_candidates)
            )
            ambiguous.extend(evidence)
            details[key] = ScaleRelationState(
                timeframe=timeframe,
                relation=ScaleRelation.UNKNOWN,
                direction=None,
                authority_layer_id=(None if dominant is None else dominant.structure_id),
                evidence_ids=evidence,
                evidence_kind=None,
                structural_scope=None,
                acceptance_state=None,
                since=min(
                    (node.confirmed_at or node.formed_at for node in frame_candidates),
                    default=None,
                ),
                age_bars=0,
                graph_connected=False,
                ambiguous=True,
            )
            continue
        if key not in active or key not in ready_timeframes:
            unknown.append(f"scale:{key}:not_ready")
            details[key] = ScaleRelationState(
                timeframe=timeframe,
                relation=ScaleRelation.UNKNOWN,
                direction=None,
                authority_layer_id=(None if dominant is None else dominant.structure_id),
                evidence_ids=(),
                evidence_kind=None,
                structural_scope=None,
                acceptance_state=None,
                since=None,
                age_bars=0,
                graph_connected=False,
                ambiguous=False,
            )
            continue
        if dominant is None or authority_direction is None:
            unknown.append(f"scale:{key}:authority_unresolved")
            details[key] = ScaleRelationState(
                timeframe=timeframe,
                relation=ScaleRelation.UNKNOWN,
                direction=(None if layer is None else layer.direction),
                authority_layer_id=None,
                evidence_ids=(() if layer is None else (layer.structure_id,)),
                evidence_kind=(None if layer is None else "structure"),
                structural_scope=(None if layer is None else layer.structural_scope),
                acceptance_state=(None if layer is None else layer.acceptance_state),
                since=(None if layer is None else layer.confirmed_at),
                age_bars=(
                    0
                    if layer is None
                    else _timeframe_age_bars(layer.confirmed_at, observation.asof, timeframe)
                ),
                graph_connected=False,
                ambiguous=False,
            )
            continue
        if key == dominant.timeframe.value:
            details[key] = ScaleRelationState(
                timeframe=timeframe,
                relation=ScaleRelation.ALIGNED,
                direction=dominant.direction,
                authority_layer_id=dominant.structure_id,
                evidence_ids=(dominant.structure_id,),
                evidence_kind="structure",
                structural_scope=dominant.structural_scope,
                acceptance_state=dominant.acceptance_state,
                since=dominant.confirmed_at,
                age_bars=_timeframe_age_bars(
                    dominant.confirmed_at, observation.asof, timeframe
                ),
                graph_connected=True,
                ambiguous=False,
            )
            continue
        relevant_conflicts = tuple(
            conflict
            for conflict in conflicts
            if conflict.source_timeframe is timeframe
            and not authority_ids.isdisjoint(
                {
                    conflict.target_node_id,
                    _context_identity(graph._nodes[conflict.target_node_id]),
                    *graph._nodes[conflict.target_node_id].source_ids,
                }
            )
        )
        if relevant_conflicts:
            conflict = max(
                relevant_conflicts,
                key=lambda value: (
                    {
                        GlobalConflictRole.AUTHORITY_INVALIDATION: 3,
                        GlobalConflictRole.AUTHORITY_TRANSITION_CANDIDATE: 2,
                        GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY: 1,
                    }[value.role],
                    value.observed_at,
                    value.conflict_id,
                ),
            )
            source = graph._nodes[conflict.source_node_id]
            relation = (
                ScaleRelation.MATERIAL_OPPOSITION
                if conflict.role
                in {
                    GlobalConflictRole.AUTHORITY_TRANSITION_CANDIDATE,
                    GlobalConflictRole.AUTHORITY_INVALIDATION,
                }
                else ScaleRelation.NORMAL_PULLBACK
            )
            attributes = dict(source.semantic_attributes)
            since = _relation_evidence_since(
                source,
                conflict_observed_at=conflict.observed_at,
            )
            details[key] = ScaleRelationState(
                timeframe=timeframe,
                relation=relation,
                direction=source.direction,
                authority_layer_id=dominant.structure_id,
                evidence_ids=tuple(
                    dict.fromkeys(
                        (_context_identity(source), conflict.conflict_id)
                    )
                ),
                evidence_kind=source.kind,
                structural_scope=source.structural_scale.value,
                acceptance_state=(
                    attributes.get("post_break_state")
                    or ("confirmed" if source.confirmed_at is not None else "pending")
                ),
                since=since,
                age_bars=_timeframe_age_bars(
                    since,
                    observation.asof,
                    timeframe,
                ),
                graph_connected=True,
                ambiguous=False,
            )
            continue
        candidate = max(
            frame_candidates,
            key=lambda node: (
                priority.get(node.kind, 0),
                node.confirmed_at or node.formed_at,
                node.node_id,
            ),
            default=None,
        )
        if candidate is None:
            unknown.append(f"scale:{key}:no_structural_evidence")
            details[key] = ScaleRelationState(
                timeframe=timeframe,
                relation=ScaleRelation.UNKNOWN,
                direction=(None if layer is None else layer.direction),
                authority_layer_id=dominant.structure_id,
                evidence_ids=(() if layer is None else (layer.structure_id,)),
                evidence_kind=(None if layer is None else "structure"),
                structural_scope=(None if layer is None else layer.structural_scope),
                acceptance_state=(None if layer is None else layer.acceptance_state),
                since=(None if layer is None else layer.confirmed_at),
                age_bars=(
                    0
                    if layer is None
                    else _timeframe_age_bars(layer.confirmed_at, observation.asof, timeframe)
                ),
                graph_connected=False,
                ambiguous=False,
            )
        else:
            challenger_layer = next(
                (
                    value
                    for value in authority_stack
                    if value.status == "challenging"
                    and value.timeframe is Timeframe.H1
                    and candidate.direction is value.direction
                    and _node_connected_to_layer(graph, candidate, value)
                ),
                None,
            )
            if challenger_layer is not None and timeframe is Timeframe.M15:
                attributes = dict(candidate.semantic_attributes)
                since = _relation_evidence_since(candidate)
                details[key] = ScaleRelationState(
                    timeframe=timeframe,
                    relation=ScaleRelation.MATERIAL_OPPOSITION,
                    direction=candidate.direction,
                    authority_layer_id=challenger_layer.structure_id,
                    evidence_ids=(_context_identity(candidate),),
                    evidence_kind=candidate.kind,
                    structural_scope=candidate.structural_scale.value,
                    acceptance_state=(
                        attributes.get("post_break_state")
                        or (
                            "confirmed"
                            if candidate.confirmed_at is not None
                            else "pending"
                        )
                    ),
                    since=since,
                    age_bars=_timeframe_age_bars(
                        since,
                        observation.asof,
                        timeframe,
                    ),
                    graph_connected=True,
                    ambiguous=False,
                )
                continue
            connected_layer = (
                dominant
                if _node_connected_to_layer(graph, candidate, dominant)
                else None
            )
            connected = connected_layer is not None
            attributes = dict(candidate.semantic_attributes)
            since = _relation_evidence_since(candidate)
            acceptance = attributes.get("post_break_state") or (
                "confirmed" if candidate.confirmed_at is not None else "pending"
            )
            if not connected:
                relation = ScaleRelation.UNKNOWN
                unknown.append(f"scale:{key}:graph_disconnected")
            elif candidate.direction is authority_direction:
                relation = ScaleRelation.ALIGNED
            else:
                relation = ScaleRelation.NORMAL_PULLBACK
            details[key] = ScaleRelationState(
                timeframe=timeframe,
                relation=relation,
                direction=candidate.direction,
                authority_layer_id=dominant.structure_id,
                evidence_ids=(_context_identity(candidate),),
                evidence_kind=candidate.kind,
                structural_scope=candidate.structural_scale.value,
                acceptance_state=acceptance,
                since=since,
                age_bars=_timeframe_age_bars(
                    since,
                    observation.asof,
                    timeframe,
                ),
                graph_connected=connected,
                ambiguous=False,
            )
    return (
        details,
        tuple(dict.fromkeys(unknown)),
        tuple(dict.fromkeys(ambiguous)),
    )


def _high_salience_episode_ids(
    nodes: Sequence[SceneNode],
    *,
    market_epoch_id: str,
) -> tuple[str, ...]:
    """Return only roots that may require playbook-level explanation."""

    output: list[str] = []
    for node in nodes:
        if node.market_epoch_id != market_epoch_id:
            continue
        attributes = dict(node.semantic_attributes)
        manipulation_root = (
            node.kind == "manipulation"
            and node.lifecycle in {"swept", "reaccepted"}
            and node.ambiguity_state
            not in {EvidenceStatus.INVALIDATED, EvidenceStatus.UNKNOWN}
        )
        typed_path_pulse = (
            node.kind == "path_sequence"
            and node.lifecycle == "active"
            and attributes.get("context_kind")
            in {"pool_reversal", "zone_return"}
            and node.ambiguity_state
            not in {EvidenceStatus.INVALIDATED, EvidenceStatus.UNKNOWN}
        )
        directional_displacement = (
            node.kind == "displacement"
            and node.timeframe
            in {
                Timeframe.H4.value,
                Timeframe.H1.value,
                Timeframe.M15.value,
                Timeframe.M5.value,
            }
            and node.lifecycle in {"started", "active"}
            and node.direction is not None
            and node.ambiguity_state
            not in {EvidenceStatus.INVALIDATED, EvidenceStatus.UNKNOWN}
        )
        if manipulation_root or typed_path_pulse or directional_displacement:
            output.append(_context_identity(node))
    return tuple(dict.fromkeys(output))


def _persistent_explanation_candidate(node: SceneNode) -> bool:
    """Whether a prior candidate is still a live structured episode root."""

    attributes = dict(node.semantic_attributes)
    return bool(
        (
            node.kind == "manipulation"
            and node.lifecycle == "swept"
            and node.ambiguity_state
            not in {EvidenceStatus.INVALIDATED, EvidenceStatus.UNKNOWN}
        )
        or (
            node.kind == "path_sequence"
            and node.lifecycle == "active"
            and attributes.get("context_kind")
            in {"pool_reversal", "zone_return"}
            and node.ambiguity_state
            not in {EvidenceStatus.INVALIDATED, EvidenceStatus.UNKNOWN}
        )
        or (
            node.kind == "displacement"
            and node.timeframe
            in {
                Timeframe.H4.value,
                Timeframe.H1.value,
                Timeframe.M15.value,
                Timeframe.M5.value,
            }
            and node.lifecycle in {"started", "active"}
            and node.direction is not None
            and node.ambiguity_state
            not in {EvidenceStatus.INVALIDATED, EvidenceStatus.UNKNOWN}
        )
    )


def _accepted_h1_continuation_root_ids(
    nodes: Sequence[SceneNode],
    *,
    market_epoch_id: str,
) -> tuple[str, ...]:
    """Emit an H1 continuation-BOS pulse for outcome-blind funnel counting."""

    return tuple(
        dict.fromkeys(
            _context_identity(node)
            for node in nodes
            if node.market_epoch_id == market_epoch_id
            and node.kind == "bos"
            and node.timeframe == Timeframe.H1.value
            and node.lifecycle == "confirmed"
            and node.ambiguity_state is EvidenceStatus.CONFIRMED
            and node.structural_scale
            in {StructuralScale.INTERMEDIATE, StructuralScale.EXTERNAL}
            and dict(node.semantic_attributes).get("scope") == "continuation"
            and dict(node.semantic_attributes).get("post_break_state")
            == "accepted"
        )
    )


def _open_thesis_mechanism(node: SceneNode) -> str:
    attributes = dict(node.semantic_attributes)
    if node.kind == "displacement":
        return "directional_displacement"
    if node.kind == "manipulation":
        source_kind = attributes.get("source_kind", "")
        return (
            "range_failed_auction"
            if "range" in source_kind
            else "liquidity_sweep"
        )
    if node.kind == "path_sequence":
        return {
            "pool_reversal": "liquidity_sweep_reversal",
            "zone_return": "frozen_zone_return",
        }.get(attributes.get("context_kind", ""), "structured_path")
    if node.kind == "bos":
        return "structural_break"
    return "structured_episode"


def _open_thesis_authority_relation(
    node: SceneNode,
    source_timeframe: Timeframe,
    context: GlobalMarketContext,
    *,
    direction: Direction | None = None,
) -> str:
    effective_direction = node.direction if direction is None else direction
    if effective_direction is None:
        return "unknown"
    relation = context.scale_relation_details.get(source_timeframe.value)
    if (
        relation is not None
        and relation.direction is effective_direction
    ):
        if relation.ambiguous:
            return "ambiguous"
        if not relation.graph_connected:
            return "unknown"
        if relation.relation is ScaleRelation.MATERIAL_OPPOSITION:
            return "challenges_incumbent"
        if relation.relation is ScaleRelation.NORMAL_PULLBACK:
            return "local_countertrend"
        if relation.relation is ScaleRelation.ALIGNED:
            return "aligned"
    authority_direction = context.authority_direction
    if authority_direction is None:
        return "unbound"
    if effective_direction is authority_direction:
        return "aligned"
    if node.kind == "manipulation":
        return "local_countertrend"
    if node.timeframe in {Timeframe.M5.value, Timeframe.M1.value}:
        return "local_countertrend"
    return "challenges_incumbent"


def _open_thesis_direction(
    root: SceneNode,
    candidates: Sequence[SceneNode],
) -> Direction | None:
    """Resolve the thesis direction, not the source sweep direction."""

    candidate_direction = next(
        (
            candidate.direction
            for candidate in sorted(
                candidates,
                key=lambda value: (
                    0 if value.kind == "path_sequence" else 1,
                    value.formed_at,
                    value.node_id,
                ),
            )
            if candidate.kind in {"path_sequence", "displacement"}
            and candidate.direction is not None
        ),
        None,
    )
    if candidate_direction is not None:
        return candidate_direction
    if root.kind == "manipulation" and root.direction is not None:
        return (
            Direction.SHORT
            if root.direction is Direction.LONG
            else Direction.LONG
        )
    return root.direction


_OPEN_THESIS_CAUSAL_RELATIONS = _CAUSAL_PATH_RELATIONS
_OPEN_THESIS_LEAF_KINDS = frozenset(
    {
        "structure",
        "swing",
        "swing_projection",
        "support_resistance",
        "liquidity",
        "liquidity_pool",
        "range",
    }
)


def _open_thesis_closure(
    graph: TemporalMarketSceneGraph,
    root: SceneNode,
    *,
    max_depth: int = 4,
) -> tuple[SceneNode, ...]:
    """Resolve one bounded current-epoch closure without scanning history."""

    visited = {root.node_id}
    frontier = deque(((root.node_id, 0),))
    while frontier and len(visited) < 64:
        node_id, depth = frontier.popleft()
        current = graph._nodes.get(node_id)
        if (
            depth >= max_depth
            or (
                depth > 0
                and current is not None
                and current.kind in _OPEN_THESIS_LEAF_KINDS
            )
        ):
            continue
        neighbors = sorted(
            graph._neighbors(node_id),
            key=lambda value: (
                value[1].relation.value,
                value[0],
                value[1].edge_id,
            ),
        )
        for neighbor_id, edge in neighbors:
            if (
                edge.relation not in _OPEN_THESIS_CAUSAL_RELATIONS
                or neighbor_id in visited
            ):
                continue
            neighbor = graph._nodes.get(neighbor_id)
            if (
                neighbor is None
                or neighbor.market_epoch_id != graph._market_epoch_id
            ):
                continue
            visited.add(neighbor_id)
            frontier.append((neighbor_id, depth + 1))
            if len(visited) >= 64:
                break
    return tuple(
        sorted(
            (graph._nodes[node_id] for node_id in visited),
            key=lambda value: (value.formed_at, value.node_id),
        )
    )


def _canonical_open_thesis_root_id(
    candidate: SceneNode,
    observation: MarketObservation,
    graph: TemporalMarketSceneGraph,
) -> str:
    """Collapse later path/zone roots into their initiating episode root."""

    if candidate.kind == "displacement":
        return _context_identity(candidate)
    if candidate.kind != "path_sequence":
        return _context_identity(candidate)
    interaction = observation.interaction_update
    paths = () if interaction is None else interaction.interaction_paths
    locations = () if interaction is None else interaction.zone_interactions
    path = next(
        (
            item
            for item in paths
            if item.sequence_id == _context_identity(candidate)
        ),
        None,
    )
    if path is None:
        return _context_identity(candidate)
    if path.context_kind == "pool_reversal":
        return path.context_id
    location = next(
        (
            item
            for item in locations
            if item.location_id == path.context_id
        ),
        None,
    )
    if location is None:
        return _context_identity(candidate)
    displacement = _current_entity_node(
        graph,
        location.source_displacement_id,
    )
    if displacement is None:
        return location.source_displacement_id
    return location.source_displacement_id


def build_open_market_theses(
    previous: Sequence[OpenMarketThesis],
    observation: MarketObservation,
    delta: SceneGraphDelta,
    graph: TemporalMarketSceneGraph,
    context: GlobalMarketContext,
    *,
    brain_interaction: BrainInteractionView | None = None,
) -> tuple[OpenMarketThesis, ...]:
    """Bind active graph roots into playbook-neutral, identity-only theses.

    Evidence identities are latched for the lifetime of the same root.  Draws,
    blockers and conflicts remain current views.  Nothing here chooses a
    playbook, score or action.
    """

    if (
        delta.asof != observation.asof
        or context.updated_at != observation.asof
        or delta.revision_id != context.scene_revision_id
        or context.market_epoch_id != graph._market_epoch_id
    ):
        raise ValueError("open thesis clocks or market epoch disagree")
    prior_by_root = {
        thesis.root_id: thesis
        for thesis in previous
        if thesis.market_epoch_id == context.market_epoch_id
    }
    output: list[OpenMarketThesis] = []
    mechanism_kinds = {
        "bos",
        "displacement",
        "fvg",
        "order_block",
        "range",
        "manipulation",
        "path_sequence",
    }
    candidate_groups: dict[str, list[SceneNode]] = defaultdict(list)
    successful_paths = {
        path.sequence_id: path
        for path in (
            ()
            if brain_interaction is None
            else brain_interaction.path_sequences
        )
        if (
            brain_path_terminal_role(path) == "successful"
            and path.ended_at == observation.asof
        )
    }
    candidate_ids = tuple(
        dict.fromkeys(
            (
                *context.candidate_structured_episode_ids,
                *successful_paths,
            )
        )
    )
    for root_id in candidate_ids:
        candidate = _current_entity_node(graph, root_id)
        if candidate is None or (
            _is_terminal(candidate.kind, candidate.lifecycle)
            and not (
                candidate.kind == "manipulation"
                and candidate.lifecycle == "reaccepted"
            )
            and root_id not in successful_paths
        ):
            continue
        canonical_id = _canonical_open_thesis_root_id(
            candidate,
            observation,
            graph,
        )
        candidate_groups[canonical_id].append(candidate)
    visible_inventory = tuple(
        sorted(
            current_dol_inventory(observation),
            key=lambda item: (
                abs(float(item.price) - float(observation.price)),
                item.item_id,
            ),
        )
    )
    visible_draws_by_side = {
        side: tuple(
            item.item_id
            for item in visible_inventory
            if item.side == side
        )
        for side in ("above", "below")
    }
    visible_all_draws = tuple(item.item_id for item in visible_inventory)
    for root_id, candidates in sorted(candidate_groups.items()):
        root = _current_entity_node(graph, root_id) or min(
            candidates,
            key=lambda value: (value.formed_at, value.node_id),
        )
        closure_by_id: dict[str, SceneNode] = {root.node_id: root}
        for candidate in sorted(candidates, key=lambda value: value.node_id):
            for node in _open_thesis_closure(graph, candidate):
                closure_by_id[node.node_id] = node
        closure = tuple(
            sorted(
                closure_by_id.values(),
                key=lambda value: (value.formed_at, value.node_id),
            )
        )
        closure_node_ids = {node.node_id for node in closure}
        identities = tuple(
            dict.fromkeys(
                _context_identity(node) for node in closure
            )
        )
        prior = prior_by_root.get(root_id)
        mechanism_ids = tuple(
            dict.fromkeys(
                (
                    root_id,
                    *(
                        ()
                        if prior is None
                        else prior.mechanism_event_ids
                    ),
                    *(
                        _context_identity(node)
                        for node in closure
                        if node.kind in mechanism_kinds
                    ),
                )
            )
        )
        location_ids = tuple(
            dict.fromkeys(
                (
                    *(() if prior is None else prior.entry_location_ids),
                    *(
                        _context_identity(node)
                        for node in closure
                        if node.kind == "entry_location"
                    ),
                )
            )
        )
        trigger_ids = tuple(
            dict.fromkeys(
                (
                    *(() if prior is None else prior.trigger_event_ids),
                    *(
                        reference.reference_id
                        for candidate in candidates
                        if candidate.kind == "path_sequence"
                        and _context_identity(candidate) in successful_paths
                        for reference in (
                            ()
                            if brain_interaction is None
                            else brain_interaction.micro_bos_references
                        )
                        if reference.qualified
                        and reference.context_kind
                        == successful_paths[
                            _context_identity(candidate)
                        ].context_kind
                        and reference.context_id
                        == successful_paths[
                            _context_identity(candidate)
                        ].context_id
                    ),
                )
            )
        )
        direction = _open_thesis_direction(root, candidates)
        draw_sides = (
            ("above",)
            if direction is Direction.LONG
            else ("below",)
            if direction is Direction.SHORT
            else ("above", "below")
        )
        visible_directional_draws = (
            visible_draws_by_side["above"]
            if direction is Direction.LONG
            else visible_draws_by_side["below"]
            if direction is Direction.SHORT
            else visible_all_draws
        )
        visible_directional_draw_ids = frozenset(
            visible_directional_draws
        )
        connected_draws = tuple(
            identity
            for node in closure
            if node.kind == "liquidity"
            and node.ambiguity_state is not EvidenceStatus.INVALIDATED
            and (
                identity := _context_identity(node)
            ) in visible_directional_draw_ids
        )
        nearest_external_draws = tuple(
            values[0]
            for side in draw_sides
            if (values := context.external_draw_candidates.get(side, ()))
        )
        draw_ids = tuple(
            dict.fromkeys(
                (
                    *connected_draws,
                    *visible_directional_draws,
                    *nearest_external_draws,
                )
            )
        )
        view = (
            None
            if direction is None
            else context.obstruction_views[direction.value]
        )
        obstruction_ids = (
            ()
            if view is None
            else tuple(
                item.obstruction_id
                for item in (
                    *view.hard_barriers[:4],
                    *view.soft_frictions[:4],
                )
            )
        )
        conflict_ids = tuple(
            conflict.conflict_id
            for conflict in context.material_conflicts
            if (
                conflict.source_node_id in closure_node_ids
                or conflict.target_node_id in closure_node_ids
            )
        )
        unknown = tuple(
            identity
            for identity in context.unknown_evidence
            if identity in identities
        )
        ambiguous = tuple(
            dict.fromkeys(
                (
                    *(
                        _context_identity(node)
                        for node in closure
                        if node.ambiguity_state
                        in {
                            EvidenceStatus.AMBIGUOUS,
                            EvidenceStatus.CONFLICTING,
                        }
                    ),
                    *(
                        identity
                        for identity in context.ambiguous_evidence
                        if identity in identities
                    ),
                )
            )
        )
        attributes = dict(root.semantic_attributes)
        raw_source_timeframe = attributes.get(
            "source_timeframe",
            root.timeframe,
        )
        try:
            source_timeframe = Timeframe(raw_source_timeframe)
        except ValueError:
            source_timeframe = Timeframe(root.timeframe)
        raw_identity = (
            f"{context.market_epoch_id}|{root_id}|"
            f"{None if direction is None else direction.value}"
        )
        supporting_ids = tuple(
            dict.fromkeys(
                (
                    *mechanism_ids,
                    *location_ids,
                    *trigger_ids,
                )
            )
        )
        opposing_ids = tuple(dict.fromkeys(conflict_ids))
        lifecycle = (
            "weakening"
            if opposing_ids
            else "active"
            if direction is not None and draw_ids
            else "forming"
        )
        semantic_revision = (
            root_id,
            None if direction is None else direction.value,
            source_timeframe.value,
            root.structural_scale.value,
            _open_thesis_mechanism(root),
            _open_thesis_authority_relation(
                root,
                source_timeframe,
                context,
                direction=direction,
            ),
            lifecycle,
            supporting_ids,
            opposing_ids,
            obstruction_ids,
            unknown,
            ambiguous,
        )
        revision_id = (
            "thesis-evidence:"
            + hashlib.sha256(repr(semantic_revision).encode()).hexdigest()[:24]
        )
        prior_supporting = (
            () if prior is None else prior.supporting_event_ids
        )
        prior_opposing = () if prior is None else prior.opposing_event_ids
        changed_at = (
            prior.updated_at
            if prior is not None
            and prior.evidence_revision_id == revision_id
            else observation.asof
        )
        evidence_state = ThesisEvidenceState(
            lifecycle=lifecycle,
            revision_id=revision_id,
            changed_at=changed_at,
            supporting_event_ids=supporting_ids,
            opposing_event_ids=opposing_ids,
            new_supporting_event_ids=tuple(
                value
                for value in supporting_ids
                if value not in prior_supporting
            ),
            new_opposing_event_ids=tuple(
                value for value in opposing_ids if value not in prior_opposing
            ),
        )
        output.append(
            OpenMarketThesis(
                thesis_id=(
                    "market-thesis:"
                    f"{hashlib.sha256(raw_identity.encode()).hexdigest()[:24]}"
                ),
                root_id=root_id,
                market_epoch_id=context.market_epoch_id,
                formed_at=(
                    root.formed_at if prior is None else prior.formed_at
                ),
                updated_at=changed_at,
                direction=direction,
                source_timeframe=source_timeframe,
                structural_scale=root.structural_scale.value,
                mechanism=_open_thesis_mechanism(root),
                authority_relation=_open_thesis_authority_relation(
                    root,
                    source_timeframe,
                    context,
                    direction=direction,
                ),
                authority_source_ids=context.authority_source_ids,
                mechanism_event_ids=mechanism_ids,
                draw_candidate_ids=draw_ids,
                entry_location_ids=location_ids,
                trigger_event_ids=trigger_ids,
                obstruction_ids=obstruction_ids,
                conflict_ids=conflict_ids,
                unknown_evidence=unknown,
                ambiguous_evidence=ambiguous,
                evidence_state=evidence_state,
            )
        )
    live_root_ids = set(candidate_groups)
    for prior in previous:
        if (
            prior.market_epoch_id != context.market_epoch_id
            or prior.root_id in live_root_ids
            or prior.lifecycle == "invalidated"
        ):
            continue
        terminal_node = _current_entity_node(graph, prior.root_id)
        source_invalidated = prior.root_id in context.invalidated_source_ids
        node_invalidated = bool(
            terminal_node is not None
            and terminal_node.lifecycle
            in {"invalidated", "failed", "broken", "accepted_outside"}
        )
        if not source_invalidated and not node_invalidated:
            continue
        terminal_identity = (
            prior.root_id
            if terminal_node is None
            else _context_identity(terminal_node)
        )
        opposing_ids = tuple(
            dict.fromkeys(
                (*prior.opposing_event_ids, terminal_identity)
            )
        )
        revision_payload = (
            prior.evidence_revision_id,
            "invalidated",
            terminal_identity,
        )
        evidence_state = ThesisEvidenceState(
            lifecycle="invalidated",
            revision_id=(
                "thesis-evidence:"
                + hashlib.sha256(repr(revision_payload).encode()).hexdigest()[:24]
            ),
            changed_at=observation.asof,
            supporting_event_ids=prior.supporting_event_ids,
            opposing_event_ids=opposing_ids,
            new_supporting_event_ids=(),
            new_opposing_event_ids=(
                ()
                if terminal_identity in prior.opposing_event_ids
                else (terminal_identity,)
            ),
        )
        output.append(
            replace(
                prior,
                updated_at=observation.asof,
                evidence_state=evidence_state,
            )
        )
    return tuple(
        sorted(output, key=lambda value: (value.formed_at, value.thesis_id))
    )


def market_episode_id(
    market_epoch_id: str,
    entry_location_id: str,
    entry_path_id: str,
    direction: Direction,
) -> str:
    """Return the physical-only identity of one neutral MarketEpisode."""

    direction = Direction(direction)
    if not market_epoch_id or not entry_location_id or not entry_path_id:
        raise ValueError("market episode physical identity is incomplete")
    raw_identity = (
        f"{market_epoch_id}|{entry_location_id}|{entry_path_id}|"
        f"{direction.value}"
    )
    return (
        "market-episode:"
        f"{hashlib.sha256(raw_identity.encode()).hexdigest()[:24]}"
    )


def _neutral_claim_relations(
    context: GlobalMarketContext,
    *,
    entry_location_id: str,
    direction: Direction,
) -> tuple[OpenMarketThesisClaimRelation, ...]:
    claims = tuple(
        OpenMarketThesisClaimRelation(
            thesis_id=thesis.thesis_id,
            root_id=thesis.root_id,
            relation=(
                "unknown"
                if thesis.direction is None
                else "aligned"
                if thesis.direction is direction
                else "opposed"
            ),
            thesis_direction=thesis.direction,
            mechanism=thesis.mechanism,
            authority_relation=thesis.authority_relation,
            evidence_revision_id=thesis.evidence_revision_id,
            lifecycle=thesis.lifecycle,
        )
        for thesis in context.open_market_theses
        if entry_location_id in thesis.entry_location_ids
    )
    return tuple(
        sorted(
            claims,
            key=lambda claim: (
                claim.thesis_id,
                claim.root_id,
                claim.relation,
            ),
        )
    )


def _merge_neutral_claim_relations(
    previous: MarketEpisodeState | None,
    current: tuple[OpenMarketThesisClaimRelation, ...],
) -> tuple[
    tuple[OpenMarketThesisClaimRelation, ...],
    tuple[str, ...],
]:
    prior_by_id = {
        claim.thesis_id: claim
        for claim in (() if previous is None else previous.claims)
    }
    merged = dict(prior_by_id)
    for claim in current:
        prior = prior_by_id.get(claim.thesis_id)
        if prior is not None and (
            prior.root_id != claim.root_id
            or prior.relation != claim.relation
            or prior.thesis_direction is not claim.thesis_direction
        ):
            raise ValueError("neutral market thesis claim identity changed")
        merged[claim.thesis_id] = claim
    claims = tuple(
        sorted(
            merged.values(),
            key=lambda claim: (
                claim.thesis_id,
                claim.root_id,
                claim.relation,
            ),
        )
    )
    active_claim_ids = tuple(
        sorted(
            claim.thesis_id
            for claim in current
            if claim.lifecycle != "invalidated"
        )
    )
    return claims, active_claim_ids


def _neutral_physical_pair_is_complete(
    observation: MarketObservation,
    location: Any,
    path: Any,
) -> bool:
    if (
        path.context_kind != "zone_return"
        or path.context_id != location.location_id
        or location.symbol != observation.symbol
        or path.symbol != observation.symbol
        or location.instrument_id != observation.instrument_id
        or path.instrument_id != observation.instrument_id
        or path.symbol != location.symbol
        or path.instrument_id != location.instrument_id
        or path.direction is not location.direction
        or path.formed_at != location.formed_at
        or not path.steps
    ):
        return False
    origin = path.steps[0]
    return bool(
        origin.kind == "zone_visible"
        and origin.observed_at == location.formed_at
        and origin.direction is location.direction
        and origin.source_entity_id == location.source_zone_id
        and origin.source_event_id == location.source_zone_id
    )


def _neutral_path_milestones(
    location: Any,
    path: Any,
) -> tuple[Any | None, Any | None]:
    pullbacks = tuple(
        step
        for step in path.steps
        if step.kind == "first_pullback"
        and step.source_entity_id == location.location_id
    )
    if len(pullbacks) > 1:
        raise ValueError("neutral market episode has repeated first pullback")
    pullback = None if not pullbacks else pullbacks[0]
    if pullback is None:
        return None, None
    triggers = tuple(
        step
        for step in path.steps
        if step.kind in {"micro_break_observed", "micro_bos_confirmed"}
        and step.observed_at > pullback.observed_at
        and step.source_event_id is not None
    )
    return pullback, None if not triggers else triggers[0]


def _with_neutral_episode_update_clock(
    previous: MarketEpisodeState | None,
    candidate: MarketEpisodeState,
) -> tuple[MarketEpisodeState, bool]:
    if previous is None:
        return candidate, True
    if all(
        field.name == "updated_at"
        or getattr(candidate, field.name) == getattr(previous, field.name)
        for field in dataclass_fields(MarketEpisodeState)
    ):
        return previous, False
    return candidate, True


def _build_unique_market_episode(
    previous: MarketEpisodeState | None,
    observation: BrainObservationView,
    context: GlobalMarketContext,
    location: Any,
    physical_path: Any,
    interpreted_path: Any,
) -> tuple[MarketEpisodeState, bool]:
    identity = market_episode_id(
        context.market_epoch_id,
        location.location_id,
        physical_path.sequence_id,
        location.direction,
    )
    if previous is not None and (
        previous.episode_id != identity
        or previous.market_epoch_id != context.market_epoch_id
        or previous.symbol != location.symbol
        or previous.instrument_id != location.instrument_id
        or previous.direction is not location.direction
        or previous.entry_location_id != location.location_id
        or previous.entry_path_id != physical_path.sequence_id
        or previous.source_zone_id != location.source_zone_id
        or previous.source_displacement_id
        != location.source_displacement_id
        or previous.entry_location_protocol_hash != location.protocol_hash
        or previous.source_zone_detector_protocol_hash
        != location.source_zone_detector_protocol_hash
        or previous.source_zone_kind != location.source_zone_kind
        or previous.source_zone_protocol_hash
        != location.source_zone_protocol_hash
        or previous.source_bos_id != location.source_bos_id
        or previous.lower_bound != location.lower_bound
        or previous.upper_bound != location.upper_bound
        or previous.midpoint != location.midpoint
        or previous.near_edge != location.near_edge
        or previous.far_edge != location.far_edge
        or previous.failure_boundary != location.failure_boundary
        or previous.formed_at != location.formed_at
    ):
        raise ValueError("neutral market episode physical custody changed")
    pullback, trigger = _neutral_path_milestones(
        location,
        interpreted_path,
    )
    if previous is not None and (
        (
            previous.first_pullback_step_id,
            previous.first_pullback_at,
        )
        != (
            None if pullback is None else pullback.step_id,
            None if pullback is None else pullback.observed_at,
        )
        and previous.first_pullback_at is not None
    ):
        raise ValueError("neutral market episode pullback custody changed")
    if previous is not None and (
        (
            previous.trigger_step_id,
            previous.trigger_event_id,
            previous.trigger_at,
        )
        != (
            None if trigger is None else trigger.step_id,
            None if trigger is None else trigger.source_event_id,
            None if trigger is None else trigger.observed_at,
        )
        and previous.trigger_at is not None
    ):
        raise ValueError("neutral market episode trigger custody changed")
    terminal_role = brain_path_terminal_role(interpreted_path)
    successful_pulse_at = (
        interpreted_path.ended_at
        if terminal_role == "successful"
        else None
    )
    successful_pulse_reason = (
        interpreted_path.transition_reason
        if successful_pulse_at is not None
        else None
    )
    local_terminal = terminal_role == "local_terminal"
    if (
        interpreted_path.lifecycle is PathSequenceLifecycle.CLOSED
        and terminal_role == "unknown"
    ):
        raise ValueError(
            "neutral market episode encountered an unknown closed-path reason"
        )
    terminal_at = (
        interpreted_path.ended_at
        if terminal_role in {"local_terminal", "censored"}
        else None
    )
    terminal_reason = (
        interpreted_path.transition_reason
        if terminal_at is not None
        else None
    )
    if previous is not None and previous.terminal_at is not None:
        if (
            terminal_at != previous.terminal_at
            or terminal_reason != previous.terminal_reason
        ):
            raise ValueError("neutral market episode terminal custody changed")
        return previous, False
    if previous is not None:
        if previous.successful_pulse_at is not None:
            successful_pulse_at = previous.successful_pulse_at
            successful_pulse_reason = previous.successful_pulse_reason
    current_claims = _neutral_claim_relations(
        context,
        entry_location_id=location.location_id,
        direction=location.direction,
    )
    claims, active_claim_ids = _merge_neutral_claim_relations(
        previous,
        current_claims,
    )
    lifecycle = (
        "terminal"
        if terminal_at is not None
        else "triggered"
        if trigger is not None
        else "pullback"
        if pullback is not None
        else "registered"
    )
    candidate = MarketEpisodeState(
        episode_id=identity,
        market_epoch_id=context.market_epoch_id,
        symbol=location.symbol,
        instrument_id=location.instrument_id,
        direction=location.direction,
        entry_location_id=location.location_id,
        entry_path_id=physical_path.sequence_id,
        source_zone_id=location.source_zone_id,
        source_displacement_id=location.source_displacement_id,
        entry_location_protocol_hash=location.protocol_hash,
        source_zone_detector_protocol_hash=(
            location.source_zone_detector_protocol_hash
        ),
        source_zone_kind=location.source_zone_kind,
        source_zone_protocol_hash=location.source_zone_protocol_hash,
        source_bos_id=location.source_bos_id,
        lower_bound=location.lower_bound,
        upper_bound=location.upper_bound,
        midpoint=location.midpoint,
        near_edge=location.near_edge,
        far_edge=location.far_edge,
        failure_boundary=location.failure_boundary,
        formed_at=location.formed_at,
        updated_at=observation.asof,
        binding_status="unique",
        claims=claims,
        active_claim_ids=active_claim_ids,
        claim_status=(
            "unbound"
            if not active_claim_ids
            else "unique"
            if len(active_claim_ids) == 1
            else "ambiguous"
        ),
        first_pullback_step_id=(
            None if pullback is None else pullback.step_id
        ),
        first_pullback_at=(
            None if pullback is None else pullback.observed_at
        ),
        trigger_step_id=None if trigger is None else trigger.step_id,
        trigger_event_id=(
            None if trigger is None else trigger.source_event_id
        ),
        trigger_at=None if trigger is None else trigger.observed_at,
        successful_pulse_at=successful_pulse_at,
        successful_pulse_reason=successful_pulse_reason,
        lifecycle=lifecycle,
        terminal_at=terminal_at,
        terminal_reason=terminal_reason,
    )
    return _with_neutral_episode_update_clock(previous, candidate)


def _rebind_neutral_episode(
    previous: MarketEpisodeState,
    observation: MarketObservation,
    context: GlobalMarketContext,
    *,
    binding_status: str,
) -> tuple[MarketEpisodeState, bool]:
    if previous.terminal_at is not None:
        return previous, False
    current_claims = _neutral_claim_relations(
        context,
        entry_location_id=previous.entry_location_id,
        direction=previous.direction,
    )
    claims, active_claim_ids = _merge_neutral_claim_relations(
        previous,
        current_claims,
    )
    candidate = replace(
        previous,
        updated_at=observation.asof,
        binding_status=binding_status,
        claims=claims,
        active_claim_ids=active_claim_ids,
        claim_status=(
            "unbound"
            if not active_claim_ids
            else "unique"
            if len(active_claim_ids) == 1
            else "ambiguous"
        ),
    )
    return _with_neutral_episode_update_clock(previous, candidate)


def _neutral_boundary_terminal(
    previous: MarketEpisodeState,
    observation: MarketObservation,
) -> MarketEpisodeState:
    interaction = observation.interaction_update
    boundary_by_path = {
        path.sequence_id: path
        for path in (
            ()
            if interaction is None or interaction.boundary_reason is None
            else interaction.interaction_path_transitions
        )
    }
    boundary = boundary_by_path.get(previous.entry_path_id)
    if boundary is not None:
        reason = boundary.transition_reason
    elif "contract_change_history_reset" in observation.anomalies:
        reason = "contract_change_reset"
    elif "data_gap_history_reset" in observation.anomalies:
        reason = "data_gap_reset"
    else:
        reason = "market_epoch_reset"
    return replace(
        previous,
        updated_at=observation.asof,
        binding_status="unbound",
        lifecycle="terminal",
        terminal_at=observation.asof,
        terminal_reason=reason,
    )


def build_neutral_market_state(
    previous: NeutralMarketState | None,
    observation: BrainObservationView,
    context: GlobalMarketContext,
) -> NeutralMarketState:
    """Reduce physical pairs with one explicit Brain interpretation view.

    Physical identity and custody always come from ``interaction_update``.
    Aligned/opposed outcomes come only from the already-constructed Brain
    view; this reducer never invokes the interpreter itself.
    """

    if (
        context.updated_at != observation.asof
        or context.scene_revision_id != observation.scene_revision_id
    ):
        raise ValueError("neutral market inputs disagree on clock or revision")
    previous_by_id = (
        {}
        if previous is None
        else previous.market_episode_by_id
    )
    previous_by_location: dict[str, list[MarketEpisodeState]] = defaultdict(list)
    for episode in previous_by_id.values():
        previous_by_location[episode.entry_location_id].append(episode)
    transitions: dict[str, MarketEpisodeState] = {}
    if (
        previous is not None
        and previous.market_epoch_id != context.market_epoch_id
    ):
        for episode in previous.market_episodes:
            if episode.terminal_at is not None:
                continue
            terminal = _neutral_boundary_terminal(episode, observation)
            transitions[terminal.episode_id] = terminal
        previous_by_id = {}
        previous_by_location.clear()

    interaction = observation.interaction_update
    physical_paths = (
        () if interaction is None else interaction.interaction_paths
    )
    physical_locations = (
        () if interaction is None else interaction.zone_interactions
    )
    interpreted_paths = observation.interaction.path_sequences
    interpreted_by_id = {
        path.sequence_id: path for path in interpreted_paths
    }
    if (
        len(interpreted_by_id) != len(interpreted_paths)
        or set(interpreted_by_id)
        != {path.sequence_id for path in physical_paths}
    ):
        raise ValueError(
            "neutral market Brain view differs from physical paths"
        )
    paths_by_location: dict[str, list[Any]] = defaultdict(list)
    for path in physical_paths:
        if path.context_kind == "zone_return":
            paths_by_location[path.context_id].append(path)

    episodes: dict[str, MarketEpisodeState] = {}
    unbound: list[str] = []
    ambiguous: list[tuple[str, tuple[str, ...]]] = []
    rejected: list[tuple[str, str]] = []
    retirements: dict[str, str] = {}
    for location in sorted(
        physical_locations,
        key=lambda item: (item.formed_at, item.location_id),
    ):
        paths = tuple(paths_by_location.get(location.location_id, ()))
        priors = tuple(previous_by_location.get(location.location_id, ()))
        if not paths:
            unbound.append(location.location_id)
            for prior in priors:
                rebound, changed = _rebind_neutral_episode(
                    prior,
                    observation,
                    context,
                    binding_status="unbound",
                )
                episodes[rebound.episode_id] = rebound
                if changed:
                    transitions[rebound.episode_id] = rebound
            continue
        if len(paths) != 1:
            path_ids = tuple(sorted(path.sequence_id for path in paths))
            ambiguous.append((location.location_id, path_ids))
            for prior in priors:
                rebound, changed = _rebind_neutral_episode(
                    prior,
                    observation,
                    context,
                    binding_status="ambiguous",
                )
                episodes[rebound.episode_id] = rebound
                if changed:
                    transitions[rebound.episode_id] = rebound
            continue
        path = paths[0]
        if not _neutral_physical_pair_is_complete(
            observation,
            location,
            path,
        ):
            rejected.append((location.location_id, path.sequence_id))
            for prior in priors:
                rebound, changed = _rebind_neutral_episode(
                    prior,
                    observation,
                    context,
                    binding_status="unbound",
                )
                episodes[rebound.episode_id] = rebound
                if changed:
                    transitions[rebound.episode_id] = rebound
            continue
        identity = market_episode_id(
            context.market_epoch_id,
            location.location_id,
            path.sequence_id,
            location.direction,
        )
        episode, changed = _build_unique_market_episode(
            previous_by_id.get(identity),
            observation,
            context,
            location,
            path,
            interpreted_by_id[path.sequence_id],
        )
        episodes[identity] = episode
        if changed:
            transitions[identity] = episode

    for episode_id, prior in previous_by_id.items():
        if episode_id in episodes:
            continue
        if prior.successful_pulse_at is not None:
            retirements[episode_id] = "upstream_compacted_after_success"
        elif prior.terminal_at is not None:
            retirements[episode_id] = "upstream_compacted_after_terminal"
        else:
            raise ValueError(
                "unresolved neutral market episode disappeared upstream"
            )

    return NeutralMarketState(
        schema_version=NEUTRAL_MARKET_STATE_SCHEMA_VERSION,
        asof=observation.asof,
        market_epoch_id=context.market_epoch_id,
        scene_revision_id=context.scene_revision_id,
        global_context=context,
        market_episodes=tuple(
            sorted(
                episodes.values(),
                key=lambda episode: (episode.formed_at, episode.episode_id),
            )
        ),
        episode_transitions_this_update=tuple(
            sorted(
                transitions.values(),
                key=lambda episode: (episode.updated_at, episode.episode_id),
            )
        ),
        unbound_entry_location_ids=tuple(sorted(unbound)),
        ambiguous_entry_location_path_ids=tuple(sorted(ambiguous)),
        rejected_entry_location_path_ids=tuple(sorted(rejected)),
        retired_episode_ids_this_update=tuple(sorted(retirements)),
        retirement_reasons_this_update=tuple(sorted(retirements.items())),
    )


def update_global_market_context(
    previous: GlobalMarketContext | None,
    observation: MarketObservation,
    delta: SceneGraphDelta,
    graph: TemporalMarketSceneGraph,
) -> GlobalMarketContext:
    """Incrementally summarize the current graph without scanning history.

    Ordinary updates use ``SceneGraphDelta`` plus the graph's bounded current
    indexes.  The same indexed bootstrap is used once at startup or after a
    contract/data epoch boundary.  Historical node/edge ledgers are never
    queried by this reducer.
    """

    if (
        delta.asof != observation.asof
        or graph.last_asof != observation.asof
        or delta.revision_id != graph.revision_id
        or (
            previous is not None
            and previous.updated_at >= observation.asof
        )
    ):
        raise ValueError("global context input clocks or revisions disagree")
    boundary = bool(
        previous is not None
        and previous.market_epoch_id != graph._market_epoch_id
    )
    changed_node_ids = tuple(
        dict.fromkeys((*delta.added_node_ids, *delta.revised_node_ids))
    )
    changed_nodes = tuple(
        graph._nodes[node_id]
        for node_id in changed_node_ids
        if node_id in graph._nodes
    )
    changed_edges = tuple(
        graph._edges[edge_id]
        for edge_id in dict.fromkeys(
            (*delta.added_edge_ids, *delta.revised_edge_ids)
        )
        if edge_id in graph._edges
    )
    ready_timeframes = frozenset(
        timeframe.value
        for timeframe in observation.active_timeframes
        if observation.frame(timeframe).ready
    )
    current_not_ready = {
        timeframe.value
        for timeframe in _GLOBAL_SCALE_ORDER
        if timeframe.value not in ready_timeframes
    }
    previous_not_ready = (
        set()
        if previous is None
        else {
            item.removeprefix("scale:").removesuffix(":not_ready")
            for item in previous.unknown_evidence
            if item.startswith("scale:") and item.endswith(":not_ready")
        }
    )
    readiness_changed = bool(
        previous is not None
        and current_not_ready != previous_not_ready
    )
    balance_context = (
        previous.balance_context
        if previous is not None
        and not boundary
        and not any(node.kind == "range" for node in changed_nodes)
        else _global_balance_context(observation, graph)
    )
    balance_changed = bool(
        previous is not None
        and previous.balance_context != balance_context
    )
    changed_candidates = _high_salience_episode_ids(
        changed_nodes,
        market_epoch_id=graph._market_epoch_id,
    )
    continuation_root_pulses = _accepted_h1_continuation_root_ids(
        changed_nodes,
        market_epoch_id=graph._market_epoch_id,
    )
    live_candidates = set(
        ()
        if previous is None or boundary
        else previous.candidate_structured_episode_ids
    )
    # Terminal roots are exposed for exactly the update that resolves them so
    # Brain can classify a reaccepted manipulation.  They are not carried as
    # live candidates on subsequent clocks.
    live_candidates = {
        identity
        for identity in live_candidates
        if (
            (node := _current_entity_node(graph, identity)) is not None
            and _persistent_explanation_candidate(node)
        )
    }
    for node in changed_nodes:
        identity = _context_identity(node)
        if _is_terminal(node.kind, node.lifecycle):
            live_candidates.discard(identity)
    live_candidates.update(changed_candidates)
    live_candidates.update(continuation_root_pulses)
    candidates = tuple(sorted(live_candidates))
    live_unexplained = tuple(
        identity
        for identity in (
            ()
            if previous is None or boundary
            else previous.unexplained_structured_episode_ids
        )
        if (
            (node := _current_entity_node(graph, identity)) is not None
            and not _is_terminal(node.kind, node.lifecycle)
        )
    )
    invalidated: list[str] = []
    if boundary and previous is not None:
        invalidated.extend(previous.authority_source_ids)
    elif previous is not None:
        prior_sources = set(previous.authority_source_ids)
        for node in changed_nodes:
            identities = {
                node.node_id,
                _context_identity(node),
                *node.source_ids,
            }
            if (
                not prior_sources.isdisjoint(identities)
                and _is_terminal(node.kind, node.lifecycle)
            ):
                invalidated.extend(
                    sorted(prior_sources.intersection(identities))
                )
    relevant_node_delta = any(
        node.kind in _GLOBAL_CONTEXT_NODE_KINDS
        for node in changed_nodes
    )
    relevant_edge_delta = any(
        edge.relation is SceneEdgeKind.OPPOSES for edge in changed_edges
    )
    rebuild_static = bool(
        previous is None
        or boundary
        or readiness_changed
        or balance_changed
        or relevant_node_delta
        or relevant_edge_delta
    )
    if not rebuild_static:
        # Price can reorder current path obstruction without changing a
        # semantic graph revision.  Every other field is a prior causal fact
        # and is copied without touching any current-kind or history scan.
        retained_mode = previous.market_mode
        if (
            retained_mode is MarketMode.TRANSITION
            and not any(
                conflict.role
                is not GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY
                for conflict in previous.material_conflicts
            )
        ):
            retained_mode = (
                MarketMode.DIRECTIONAL
                if previous.authority_direction is not None
                else MarketMode.UNCERTAIN
            )
        graph._current_global_material_conflict_ids = {
            conflict.conflict_id
            for conflict in previous.material_conflicts
            if conflict.role
            is not GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY
        }
        retained_stack = tuple(
            layer
            for layer in previous.authority_stack
            if layer.status != "invalidated"
        )
        aged_relations = {
            key: replace(
                state,
                age_bars=_timeframe_age_bars(
                    state.since,
                    observation.asof,
                    state.timeframe,
                ),
            )
            for key, state in previous.scale_relation_details.items()
        }
        draws = _global_external_draws(graph, observation)
        return GlobalMarketContext(
            updated_at=observation.asof,
            scene_revision_id=delta.revision_id,
            market_epoch_id=previous.market_epoch_id,
            authority_stack=retained_stack,
            market_mode=retained_mode,
            scale_relation_details=aged_relations,
            external_draw_candidates=draws,
            obstruction_views=_refresh_obstruction_draws(
                graph,
                previous.obstruction_views,
                draws,
            ),
            material_conflicts=previous.material_conflicts,
            unknown_evidence=previous.unknown_evidence,
            ambiguous_evidence=previous.ambiguous_evidence,
            dislocations_by_scale=previous.dislocations_by_scale,
            balance_context=balance_context,
            invalidated_source_ids=tuple(dict.fromkeys(invalidated)),
            candidate_structured_episode_ids=candidates,
            unexplained_structured_episode_ids=live_unexplained,
        )
    authority_stack = _global_authority(
        graph,
        ready_timeframes=ready_timeframes,
    )
    active_layer_timeframes = {layer.timeframe for layer in authority_stack}
    if previous is not None and invalidated:
        invalidated_set = set(invalidated)
        authority_stack = tuple(
            (
                *authority_stack,
                *(
                    replace(layer, status="invalidated")
                    for layer in previous.authority_stack
                    if layer.timeframe not in active_layer_timeframes
                    and not invalidated_set.isdisjoint(
                        {
                            layer.structure_id,
                            *layer.source_ids,
                            *((layer.protected_level_id,) if layer.protected_level_id else ()),
                        }
                    )
                ),
            )
        )
    dominant = _dominant_authority_layer(authority_stack)
    authority_timeframe = None if dominant is None else dominant.timeframe
    authority_direction = None if dominant is None else dominant.direction
    authority_source_ids = (
        ()
        if dominant is None
        else tuple(
            dict.fromkeys(
                (
                    dominant.structure_id,
                    *dominant.source_ids,
                    *((dominant.protected_level_id,) if dominant.protected_level_id else ()),
                )
            )
        )
    )
    live_conflicts = _global_material_conflicts(
        graph,
        ready_timeframes=ready_timeframes,
    )
    terminal_conflicts = _global_terminal_authority_conflicts(
        graph,
        delta,
        ready_timeframes=ready_timeframes,
    )
    conflicts = tuple(
        sorted(
            {
                conflict.conflict_id: conflict
                for conflict in (*live_conflicts, *terminal_conflicts)
            }.values(),
            key=lambda value: (value.observed_at, value.conflict_id),
        )
    )
    relation_details, unknown, relation_ambiguities = _global_scale_relations(
        graph,
        observation=observation,
        ready_timeframes=ready_timeframes,
        authority_stack=authority_stack,
        conflicts=conflicts,
    )
    ambiguous_ids = tuple(
        dict.fromkeys(
            (
                *(
                    graph._nodes[node_id].entity_id or node_id
                    for node_id in sorted(
                        graph._current_epoch_ambiguous_node_ids
                    )
                    if node_id in graph._nodes
                    and graph._nodes[node_id].market_epoch_id
                    == graph._market_epoch_id
                ),
                *relation_ambiguities,
            )
        )
    )
    authority_changed = bool(
        previous is not None
        and not boundary
        and (
            previous.authority_timeframe != authority_timeframe
            or previous.authority_direction != authority_direction
        )
    )
    authority_conflicts = tuple(
        conflict
        for conflict in conflicts
        if conflict.role
        is not GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY
    )
    authority_scale_ambiguous = any(
        relation_details[timeframe.value].ambiguous
        for timeframe in (Timeframe.H4, Timeframe.H1, Timeframe.M15)
    )
    if authority_scale_ambiguous:
        market_mode = MarketMode.UNCERTAIN
    elif authority_conflicts or authority_changed or (invalidated and not boundary):
        market_mode = MarketMode.TRANSITION
    elif (
        authority_direction is None
        and balance_context is not None
        and balance_context.status in {"descriptive", "authoritative"}
        and not balance_context.accepted_external_break
    ):
        market_mode = MarketMode.BALANCED
    elif authority_direction is None:
        market_mode = MarketMode.UNCERTAIN
    else:
        market_mode = MarketMode.DIRECTIONAL
    dislocations = _global_dislocations_by_scale(graph)
    unexplained = live_unexplained
    if authority_timeframe is None:
        unknown = tuple((*unknown, "authority:unresolved"))
    graph._current_global_material_conflict_ids = {
        conflict.conflict_id for conflict in authority_conflicts
    }
    draws = _global_external_draws(graph, observation)
    return GlobalMarketContext(
        updated_at=observation.asof,
        scene_revision_id=delta.revision_id,
        market_epoch_id=graph._market_epoch_id,
        authority_stack=authority_stack,
        market_mode=market_mode,
        scale_relation_details=relation_details,
        external_draw_candidates=draws,
        obstruction_views=_global_obstruction_views(
            graph,
            observation,
            draws,
            authority_stack,
        ),
        material_conflicts=conflicts,
        unknown_evidence=tuple(dict.fromkeys(unknown)),
        ambiguous_evidence=ambiguous_ids,
        dislocations_by_scale=dislocations,
        balance_context=balance_context,
        invalidated_source_ids=tuple(dict.fromkeys(invalidated)),
        candidate_structured_episode_ids=candidates,
        unexplained_structured_episode_ids=unexplained,
    )


_PHASE_FOCUS: Mapping[PlaybookPhase, tuple[str, ...]] = {
    PlaybookPhase.INACTIVE: (Timeframe.H4.value, Timeframe.H1.value, Timeframe.M15.value),
    PlaybookPhase.FORMING: (Timeframe.H4.value, Timeframe.H1.value, Timeframe.M15.value),
    PlaybookPhase.ARMED: (Timeframe.H1.value, Timeframe.M15.value, Timeframe.M5.value),
    PlaybookPhase.WAITING_LOCATION: (Timeframe.M15.value, Timeframe.M5.value),
    PlaybookPhase.WAITING_TRIGGER: (Timeframe.M5.value, Timeframe.M1.value),
    PlaybookPhase.EXECUTABLE: (Timeframe.M5.value, Timeframe.M1.value),
    PlaybookPhase.ENTERED: (Timeframe.M5.value, Timeframe.M1.value),
    PlaybookPhase.DELIVERING: (Timeframe.H1.value, Timeframe.M15.value, Timeframe.M5.value),
    PlaybookPhase.WEAKENING: (Timeframe.H1.value, Timeframe.M5.value, Timeframe.M1.value),
    PlaybookPhase.COMPLETED: (Timeframe.H1.value, Timeframe.M15.value),
    PlaybookPhase.INVALIDATED: (Timeframe.H1.value, Timeframe.M15.value),
}


def _focus_entry_episode_source_ids(
    hypothesis: Any,
) -> tuple[str, ...] | None:
    """Return one internally consistent child-Episode identity closure.

    Focus is descriptive, but binding it to a sibling's path would still make
    its question and conflict diagnostics misleading.  Treat a partially
    populated or internally inconsistent Episode as unbound instead of
    combining whichever setup/location/path/trigger identifiers happen to be
    present.  Runtime ``HypothesisBelief`` already enforces this contract;
    the defensive checks also keep restored/duck-typed diagnostic inputs
    fail-closed.
    """

    setup_id = getattr(hypothesis, "setup_context_id", None)
    episode_id = getattr(hypothesis, "episode_id", None)
    location_id = getattr(hypothesis, "entry_location_id", None)
    path_id = getattr(hypothesis, "entry_path_id", None)
    plan = getattr(hypothesis, "plan", None)
    trigger = getattr(hypothesis, "selected_trigger", None)
    sequence = getattr(hypothesis, "sequence", None)
    plan_setup_id = None if plan is None else getattr(plan, "setup_id", None)
    plan_location_id = (
        None if plan is None else getattr(plan, "entry_location_id", None)
    )
    plan_path_id = None if plan is None else getattr(plan, "entry_path_id", None)
    trigger_setup_id = (
        None if trigger is None else getattr(trigger, "setup_id", None)
    )
    trigger_location_id = (
        None if trigger is None else getattr(trigger, "entry_location_id", None)
    )
    trigger_path_id = (
        None if trigger is None else getattr(trigger, "entry_path_id", None)
    )
    sequence_setup_id = (
        None if sequence is None else getattr(sequence, "setup_id", None)
    )
    child_declared = any(
        value is not None
        for value in (
            episode_id,
            location_id,
            path_id,
            plan_setup_id,
            plan_location_id,
            plan_path_id,
            trigger_setup_id,
            trigger_location_id,
            trigger_path_id,
        )
    )
    if not child_declared:
        return ()
    if (
        episode_id is None
        or setup_id != episode_id
        or (
            sequence_setup_id is not None
            and sequence_setup_id != episode_id
        )
        or (location_id is None) != (path_id is None)
        or (
            path_id is not None
            and (location_id is None or episode_id is None)
        )
        or (
            plan is not None
            and any(
                value is not None
                for value in (plan_setup_id, plan_location_id, plan_path_id)
            )
            and (
                plan_setup_id != episode_id
                or plan_location_id != location_id
                or plan_path_id != path_id
            )
        )
        or (
            trigger is not None
            and (
                trigger_setup_id != episode_id
                or trigger_location_id != location_id
                or trigger_path_id != path_id
            )
        )
    ):
        return None
    metadata = getattr(hypothesis, "context_metadata", {}) or {}
    zone_id = (
        metadata.get("lsr_entry_zone_id")
        if isinstance(metadata, Mapping)
        else None
    )
    if zone_id in {None, "", "none", "unknown"}:
        zone_id = None
    values = [setup_id, episode_id, location_id, path_id, zone_id]
    if trigger is not None:
        values.extend(
            (
                getattr(trigger, "trigger_id", None),
                getattr(trigger, "source_entity_id", None),
                getattr(trigger, "source_event_id", None),
            )
        )
    return tuple(dict.fromkeys(value for value in values if value))


def _focus_hypothesis_source_ids(hypothesis: Any) -> frozenset[str]:
    values: list[str | None] = [
        getattr(hypothesis, "required_root_id", None),
        getattr(hypothesis, "market_thesis_root_id", None),
        getattr(hypothesis, "context_id", None),
        getattr(hypothesis, "context_thesis_id", None),
        getattr(hypothesis, "parent_context_thesis_id", None),
        getattr(hypothesis, "initiating_event_id", None),
    ]
    metadata = getattr(hypothesis, "context_metadata", {}) or {}
    if isinstance(metadata, Mapping):
        values.extend(
            metadata.get(name)
            for name in (
                "lsr_manipulation_id",
                "lsr_pool_path_id",
                "lsr_displacement_id",
            )
            if metadata.get(name) not in {None, "", "none", "unknown"}
        )
    episode_ids = _focus_entry_episode_source_ids(hypothesis)
    if episode_ids is not None:
        values.extend(episode_ids)
    elif getattr(hypothesis, "episode_id", None) is not None:
        # Do not let a malformed child borrow sequence/route identities from a
        # sibling.  Context-level identities above remain available.
        return frozenset(value for value in values if value)
    invalidation = getattr(hypothesis, "invalidation", None)
    values.append(
        None
        if invalidation is None
        else getattr(invalidation, "source_level_id", None)
    )
    sequence = getattr(hypothesis, "sequence", None)
    if sequence is not None:
        values.extend(
            source_id
            for step in getattr(sequence, "steps", ())
            for source_id in getattr(step, "source_ids", ())
        )
    route = getattr(hypothesis, "liquidity_route", None)
    if route is not None:
        values.extend(
            (
                getattr(route, "context_draw_id", None),
                getattr(route, "primary_deliverable_target_id", None),
                getattr(route, "terminal_draw_id", None),
                *getattr(route, "intermediate_liquidity_ids", ()),
                *getattr(route, "path_blocker_ids", ()),
                *getattr(route, "source_path_ids", ()),
            )
        )
    return frozenset(value for value in values if value)


def _focus_identity_is_bound(
    identity: str,
    bound_ids: frozenset[str],
) -> bool:
    return any(
        identity == bound_id
        or identity.endswith(f":{bound_id}")
        or bound_id.endswith(f":{identity}")
        for bound_id in bound_ids
    )


def _focus_conflict_is_bound(
    conflict: GlobalConflictEvidence,
    hypothesis: Any,
) -> bool:
    playbook = getattr(hypothesis, "playbook", None)
    direction = getattr(hypothesis, "direction", None)
    summary_key = (
        None
        if playbook is None or direction is None
        else (
            f"{getattr(playbook, 'value', playbook)}:"
            f"{getattr(direction, 'value', direction)}"
        )
    )
    hypothesis_keys = {
        value
        for value in (
            getattr(hypothesis, "key", None),
            getattr(hypothesis, "candidate_id", None),
            summary_key,
        )
        if value
    }
    if hypothesis_keys.isdisjoint(conflict.affected_hypothesis_ids):
        return False
    bound_ids = _focus_hypothesis_source_ids(hypothesis)
    return any(
        _focus_identity_is_bound(conflict_id, bound_ids)
        for conflict_id in (
            conflict.event_id,
            conflict.source_node_id,
            conflict.target_node_id,
        )
    )


def select_focus(
    previous_belief: Any | None,
    observation: MarketObservation,
    scene_delta: SceneGraphDelta | None,
    scene_graph: TemporalMarketSceneGraph | None,
    global_context: GlobalMarketContext | None = None,
    preferred_candidate: Any | None = None,
) -> FocusState:
    """Select one bounded question around the current root candidate.

    Focus is a query hint, never an entry gate.  Its semantic selection is
    retained until the current root, candidate, phase/terminal state, or a
    related conflict/ambiguity actually changes.  Unrelated graph revisions
    never change Focus.
    """

    if global_context is not None and (
        global_context.updated_at != observation.asof
        or (
            scene_graph is not None
            and global_context.scene_revision_id != scene_graph.revision_id
        )
    ):
        raise ValueError("focus received a stale global market context")
    active_frames = tuple(
        timeframe.value for timeframe in observation.active_timeframes
    )
    active_set = set(active_frames)

    def enabled(values: Sequence[str]) -> tuple[str, ...]:
        filtered = tuple(
            dict.fromkeys(value for value in values if value in active_set)
        )
        if filtered:
            return filtered
        if Timeframe.M1.value in active_set:
            return (Timeframe.M1.value,)
        return (active_frames[0],)

    prior_focus = (
        None
        if previous_belief is None
        else getattr(previous_belief, "focus_state", None)
    )
    ranked = [] if previous_belief is None else previous_belief.ranked()
    dominant = preferred_candidate or (ranked[0] if ranked else None)
    candidate_id = (
        None
        if dominant is None
        else getattr(dominant, "candidate_id", None) or dominant.key
    )
    prior_resolver = (
        None
        if previous_belief is None
        else getattr(previous_belief, "resolve_hypothesis", None)
    )
    prior_candidate = (
        None
        if previous_belief is None or prior_focus is None
        else prior_resolver(prior_focus.hypothesis_id)
        if callable(prior_resolver)
        else ranked[0]
        if ranked
        else None
    )
    phase = PlaybookPhase.FORMING if dominant is None else dominant.phase
    candidate_changed = bool(
        prior_focus is not None
        and prior_focus.hypothesis_id != candidate_id
    )
    phase_changed = bool(
        prior_candidate is not None
        and dominant is not None
        and prior_focus is not None
        and prior_focus.hypothesis_id == candidate_id
        and prior_candidate.phase is not dominant.phase
    )

    def related_conflicts(
        context: GlobalMarketContext | None,
        hypothesis: Any | None,
    ) -> tuple[str, ...]:
        if context is None or hypothesis is None:
            return ()
        return tuple(
            sorted(
                conflict.conflict_id
                for conflict in context.material_conflicts
                if conflict.role
                is not GlobalConflictRole.LOCAL_COUNTERTREND_DELIVERY
                and _focus_conflict_is_bound(conflict, hypothesis)
            )
        )

    def related_invalidated_sources(
        context: GlobalMarketContext | None,
        hypothesis: Any | None,
    ) -> tuple[str, ...]:
        if context is None or hypothesis is None:
            return ()
        bound_ids = _focus_hypothesis_source_ids(hypothesis)
        return tuple(
            sorted(
                source_id
                for source_id in getattr(
                    context,
                    "invalidated_source_ids",
                    (),
                )
                if _focus_identity_is_bound(source_id, bound_ids)
            )
        )

    conflict_ids = related_conflicts(global_context, dominant)
    previous_conflicts = related_conflicts(
        None
        if previous_belief is None
        else getattr(previous_belief, "global_context", None),
        prior_candidate,
    )
    invalidated_source_ids = related_invalidated_sources(
        global_context,
        dominant,
    )
    previous_invalidated_source_ids = related_invalidated_sources(
        None
        if previous_belief is None
        else getattr(previous_belief, "global_context", None),
        prior_candidate,
    )
    bound_thesis = next(
        (
            thesis
            for thesis in (
                ()
                if global_context is None
                else global_context.open_market_theses
            )
            if dominant is not None
            and thesis.root_id
            == getattr(dominant, "required_root_id", None)
        ),
        None,
    )
    current_ambiguity = tuple(
        sorted(
            () if bound_thesis is None else bound_thesis.ambiguous_evidence
        )
    )
    if (
        not current_ambiguity
        and dominant is None
        and global_context is None
        and scene_graph is not None
    ):
        if scene_graph.last_asof == observation.asof:
            current_ambiguity = tuple(
                sorted(scene_graph._current_epoch_ambiguous_node_ids)
            )
        else:
            current_ambiguity = tuple(
                node.node_id
                for node in scene_graph.nodes_asof(observation.asof)
                if node.ambiguity_state
                in {EvidenceStatus.AMBIGUOUS, EvidenceStatus.CONFLICTING}
            )
    prior_thesis = next(
        (
            thesis
            for thesis in (
                ()
                if previous_belief is None
                or getattr(previous_belief, "global_context", None) is None
                else previous_belief.global_context.open_market_theses
            )
            if prior_candidate is not None
            and thesis.root_id
            == getattr(prior_candidate, "required_root_id", None)
        ),
        None,
    )
    prior_ambiguity = tuple(
        sorted(
            () if prior_thesis is None else prior_thesis.ambiguous_evidence
        )
    )

    def root_revision(
        thesis: Any | None,
        hypothesis: Any | None,
    ) -> tuple[str, str, str] | tuple[str, str] | None:
        # LSR's frozen Context Thesis outlives the transient open Scene-Graph
        # root projection.  Its evidence revision is also deliberately shared
        # by sibling zone Episodes.  Use that stable Brain-owned revision so a
        # sibling path appearing/disappearing cannot spuriously reselect the
        # current child's Focus.  Candidate identity and phase changes retain
        # their independent reselection authority above.
        if (
            hypothesis is not None
            and getattr(hypothesis, "playbook", None)
            is Playbook.LIQUIDITY_SWEEP_REVERSAL
        ):
            context_thesis_id = getattr(
                hypothesis,
                "context_thesis_id",
                None,
            )
            evidence_revision_id = getattr(
                hypothesis,
                "evidence_revision_id",
                None,
            )
            if context_thesis_id and evidence_revision_id:
                return (
                    "lsr_context",
                    context_thesis_id,
                    evidence_revision_id,
                )
        if thesis is None:
            return None
        return (thesis.root_id, thesis.evidence_revision_id)

    current_root_revision = root_revision(bound_thesis, dominant)
    prior_root_revision = root_revision(prior_thesis, prior_candidate)
    root_revision_changed = current_root_revision != prior_root_revision
    context_changed = bool(
        conflict_ids != previous_conflicts
        or current_ambiguity != prior_ambiguity
        or invalidated_source_ids != previous_invalidated_source_ids
    )
    reselect = bool(
        prior_focus is None
        or candidate_changed
        or phase_changed
        or root_revision_changed
        or context_changed
    )
    if prior_focus is not None and not reselect:
        return replace(prior_focus, asof=observation.asof, switched=False)

    desired = enabled(_PHASE_FOCUS[phase])
    reasons: list[str] = [
        "initial_context" if prior_focus is None else "focus_reselected"
    ]
    if candidate_changed:
        reasons.append("root_candidate_changed")
    if phase_changed:
        reasons.append("candidate_phase_changed")
    if root_revision_changed:
        reasons.append("root_market_thesis_revised")
    if global_context is not None and global_context.authority_timeframe:
        desired = enabled(
            (global_context.authority_timeframe.value, *desired)
        )
        reasons.append("global_authority_context")
    if bound_thesis is not None:
        desired = enabled((bound_thesis.source_timeframe.value, *desired))
        reasons.append("root_market_thesis")

    question = {
        PlaybookPhase.WAITING_LOCATION: (
            "has price reached the frozen setup zone without chasing"
        ),
        PlaybookPhase.WAITING_TRIGGER: (
            "has one qualified root-bound trigger confirmed"
        ),
        PlaybookPhase.EXECUTABLE: (
            "does the frozen plan retain delivery space"
        ),
        PlaybookPhase.ENTERED: (
            "is the frozen thesis delivering, weakening or invalidated"
        ),
        PlaybookPhase.DELIVERING: (
            "is the frozen thesis still delivering"
        ),
        PlaybookPhase.WEAKENING: (
            "which current fact weakened the frozen thesis"
        ),
        PlaybookPhase.INVALIDATED: (
            "which frozen source invalidated the thesis"
        ),
    }.get(phase, "what exact causal event is required next")
    if dominant is None:
        question = "which registered mechanism explains the active market root"
    if conflict_ids:
        reasons.append("root_related_cross_scale_conflict")
        desired = enabled(
            (
                Timeframe.H4.value,
                Timeframe.H1.value,
                Timeframe.M15.value,
                Timeframe.M5.value,
            )
        )
        question = "which scale owns the root-related structural conflict"
    if current_ambiguity:
        reasons.append("root_related_ambiguity")
    if dominant is not None and global_context is not None:
        invalidated_ids = set(invalidated_source_ids)
        route = getattr(dominant, "liquidity_route", None)
        draw_ids = {
            value
            for value in (
                None if route is None else route.context_draw_id,
                None if route is None else route.primary_deliverable_target_id,
                None if route is None else route.terminal_draw_id,
            )
            if value is not None
        }
        if invalidated_ids:
            reasons.append(
                "frozen_draw_consumed"
                if draw_ids.intersection(invalidated_ids)
                else "active_thesis_source_invalidated"
            )
            question = "which frozen source closed the active thesis"
        if route is not None and set(route.path_blocker_ids).intersection(
            global_context.path_blocker_ids
        ):
            reasons.append("delivery_path_blocked")
            question = "does the frozen route retain enough delivery space"
    if dominant is not None and dominant.plan is not None:
        plan = dominant.plan
        if (
            plan.entry_zone_lower is not None
            and plan.entry_zone_upper is not None
            and plan.entry_zone_lower
            <= observation.price
            <= plan.entry_zone_upper
        ):
            reasons.append("price_in_frozen_zone")
            desired = enabled((Timeframe.M5.value, Timeframe.M1.value))
            question = "has one qualified root-bound trigger confirmed"

    resolution = (
        EvidenceStatus.INVALIDATED
        if phase is PlaybookPhase.INVALIDATED
        else EvidenceStatus.CONFLICTING
        if conflict_ids
        else EvidenceStatus.AMBIGUOUS
        if current_ambiguity
        else EvidenceStatus.CONFIRMED
        if phase
        in {
            PlaybookPhase.EXECUTABLE,
            PlaybookPhase.ENTERED,
            PlaybookPhase.DELIVERING,
        }
        else EvidenceStatus.FORMING
    )
    return FocusState(
        asof=observation.asof,
        primary_timeframes=tuple(desired),
        reason_codes=tuple(dict.fromkeys(reasons)),
        question=question,
        resolution_status=resolution,
        hypothesis_id=candidate_id,
        switched=prior_focus is not None,
    )


def build_hypothesis_states(
    hypotheses: Mapping[str, Any],
    graph: TemporalMarketSceneGraph | None,
    focused: FocusedObservation | None,
) -> Mapping[str, HypothesisState]:
    """Project read-only graph context for root-specific Brain candidates."""

    def relevant_kinds(step_id: str) -> set[str]:
        token = step_id.lower()
        kinds: set[str] = set()
        mappings = (
            (("structure", "swing", "direction"), {"structure", "swing", "swing_projection"}),
            (("draw", "liquidity", "pool", "sweep"), {"liquidity", "liquidity_pool", "manipulation"}),
            (("bos", "mss"), {"bos", "micro_bos"}),
            (("displacement", "impulse"), {"displacement"}),
            (("zone", "fvg", "order_block", "ob"), {"fvg", "order_block"}),
            (("pullback", "location", "return"), {"entry_location"}),
            (("trigger", "reaccept"), {"reacceptance", "micro_bos", "path_sequence"}),
            (("range", "auction", "value"), {"range", "manipulation"}),
        )
        for needles, values in mappings:
            if any(needle in token for needle in needles):
                kinds.update(values)
        return kinds

    def relevant_timeframe(step_id: str) -> str | None:
        token = step_id.lower()
        return next(
            (
                value
                for prefix, value in (
                    ("h4_", Timeframe.H4.value),
                    ("h1_", Timeframe.H1.value),
                    ("m15_", Timeframe.M15.value),
                    ("m5_", Timeframe.M5.value),
                    ("m1_", Timeframe.M1.value),
                )
                if token.startswith(prefix)
            ),
            None,
        )

    output: dict[str, HypothesisState] = {}
    for slot_key, belief in hypotheses.items():
        sequence = belief.sequence
        steps = () if sequence is None else sequence.steps
        satisfied = tuple(step for step in steps if step.satisfied)
        episode_source_ids = _focus_entry_episode_source_ids(belief)
        roots = tuple(
            dict.fromkeys(
                value
                for value in (
                    getattr(belief, "required_root_id", None),
                    belief.context_id,
                    belief.initiating_event_id,
                    *(
                        ()
                        if episode_source_ids is None
                        else episode_source_ids
                    ),
                    *(
                        ()
                        if episode_source_ids is None
                        else tuple(
                            source
                            for step in satisfied
                            for source in step.source_ids
                        )
                    ),
                )
                if value is not None
            )
        )
        asof = None if focused is None else focused.asof
        closure_ids: set[str] = set()
        if graph is not None and asof is not None:
            root_node_ids = {
                node_id
                for source_id in roots
                for node_id in (
                    graph._node_id_for_source(source_id, asof=asof),
                )
                if node_id is not None
            }
            closure_ids.update(root_node_ids)
            frontier = deque((node_id, 0) for node_id in root_node_ids)
            while frontier:
                node_id, depth = frontier.popleft()
                if depth >= 4:
                    continue
                for neighbor, _ in graph._neighbors(node_id, asof=asof):
                    if neighbor in closure_ids:
                        continue
                    closure_ids.add(neighbor)
                    frontier.append((neighbor, depth + 1))
        focused_nodes = {
            node.node_id: node
            for node in (() if focused is None else focused.nodes)
        }
        scoped_nodes = {
            node_id: node
            for node_id, node in focused_nodes.items()
            if node_id in closure_ids
        }
        missing: dict[str, EvidenceStatus] = {}
        ambiguous: dict[str, EvidenceStatus] = {}
        for step in steps:
            if step.satisfied:
                continue
            kinds = relevant_kinds(step.step_id)
            step_frame = relevant_timeframe(step.step_id)
            status = EvidenceStatus.NOT_OBSERVED
            relevant_missing = (
                ()
                if focused is None
                else tuple(
                    key
                    for key in focused.missing_evidence
                    if (
                        step_frame is None
                        or not key.split(".", 1)[0]
                        in {item.value for item in Timeframe}
                        or key.startswith(f"{step_frame}.")
                    )
                    and (
                        key.endswith(".semantic_history")
                        or not kinds
                        or any(kind in key.lower() for kind in kinds)
                    )
                )
            )
            scoped_statuses = {
                node.ambiguity_state
                for node in scoped_nodes.values()
                if (not kinds or node.kind in kinds)
            }
            if EvidenceStatus.CONFLICTING in scoped_statuses:
                ambiguous[step.step_id] = EvidenceStatus.CONFLICTING
            elif EvidenceStatus.AMBIGUOUS in scoped_statuses:
                ambiguous[step.step_id] = EvidenceStatus.AMBIGUOUS
            elif (
                EvidenceStatus.UNKNOWN in scoped_statuses
                or relevant_missing
            ):
                missing[step.step_id] = EvidenceStatus.UNKNOWN
            else:
                missing[step.step_id] = status
        paths: list[tuple[str, ...]] = []
        if graph is not None and asof is not None:
            for left, right in zip(satisfied[:-1], satisfied[1:]):
                path = graph.find_path(
                    left.source_ids,
                    right.source_ids,
                    asof=asof,
                )
                if path:
                    paths.append(path)
        material_conflict_ids: list[str] = []
        if focused is not None:
            focused_conflict_ids = set(focused.cross_scale_conflicts)
            material_conflict_ids.extend(
                edge.edge_id
                for edge in focused.edges
                if edge.edge_id in focused_conflict_ids
                and (
                    edge.source_node_id in closure_ids
                    or edge.target_node_id in closure_ids
                )
            )
        hypothesis_id = getattr(belief, "candidate_id", None) or slot_key
        next_event = next((step.step_id for step in steps if not step.satisfied), None)
        fallback_target_id = (
            None
            if belief.plan is None or not belief.plan.targets
            else belief.plan.targets[0].level_id
        )
        route = getattr(belief, "liquidity_route", None) or (
            None
            if belief.plan is None
            else getattr(belief.plan, "liquidity_route", None)
        )
        output[hypothesis_id] = HypothesisState(
            hypothesis_id=hypothesis_id,
            playbook=belief.playbook,
            direction=belief.direction,
            context_root_ids=roots[:8],
            context_timeframe=(
                Timeframe.H4.value
                if belief.playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
                else Timeframe.H1.value
            ),
            setup_timeframe=Timeframe.M5.value,
            trigger_timeframe=Timeframe.M1.value,
            sequence_stage=belief.phase.value,
            next_expected_event=next_event,
            supporting_graph_paths=tuple(paths),
            material_conflict_ids=tuple(material_conflict_ids),
            missing_evidence=missing,
            ambiguous_evidence=ambiguous,
            context_draw_id=(
                (None if route is None else route.context_draw_id)
                or getattr(belief, "context_draw_id", None)
            ),
            primary_target_id=(
                (
                    None
                    if route is None
                    else route.primary_deliverable_target_id
                )
                or fallback_target_id
            ),
            invalidation_id=(
                None if belief.invalidation is None else belief.invalidation.source_level_id
            ),
            evidence_revision_id=belief.evidence_revision_id,
            source_market_thesis_ids=getattr(
                belief,
                "market_thesis_ids",
                (),
            ),
            playbook_match_strength=getattr(
                belief,
                "playbook_match_strength",
                0.0,
            ),
            market_thesis_action_bound=(
                getattr(
                    belief,
                    "market_thesis_action_bound",
                    False,
                )
            ),
        )
    return output


__all__ = [
    "EvidenceStatus",
    "FocusState",
    "FocusedObservation",
    "HypothesisState",
    "LiquidityRole",
    "ScaleRole",
    "ScaleSpec",
    "SCENE_EDGE_LIFECYCLE_VOCAB",
    "SCENE_NODE_KIND_VOCAB",
    "SCENE_NODE_LIFECYCLE_VOCAB",
    "SCENE_NODE_ROLE_VOCAB",
    "SceneEdge",
    "SceneEdgeKind",
    "SceneGraphDelta",
    "SceneNode",
    "StructuralScale",
    "TemporalMarketSceneGraph",
    "build_hypothesis_states",
    "build_neutral_market_state",
    "build_open_market_theses",
    "current_dol_inventory",
    "dol_level_terminal_sources",
    "market_episode_id",
    "parse_scale_specs",
    "scale_registry_id",
    "select_focus",
    "update_global_market_context",
]
