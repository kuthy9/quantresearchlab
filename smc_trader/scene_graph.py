"""Append-only temporal market scene graph and explainable scale focus.

The graph is an in-memory semantic index over the already-causal primitive
reducers.  It does not detect trades and it never stores every candle.  A
node revision may close a lifecycle or resolve ambiguity, while every prior
revision remains available for decision-time reconstruction.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .model import (
    CORE_TIMEFRAMES,
    Direction,
    EventKind,
    LiquidityInventoryLifecycle,
    MarketObservation,
    MarketEvent,
    Playbook,
    PlaybookPhase,
    StructureLifecycle,
    SwingLifecycle,
    Timeframe,
    aware_timestamp,
    content_hash,
    to_primitive,
)


class ScaleRole(str, Enum):
    MACRO = "macro"
    CONTEXT = "context"
    CONTEXT_BRIDGE = "context_bridge"
    SETUP = "setup"
    TRIGGER = "trigger"


class StructuralScale(str, Enum):
    INTERNAL = "internal"
    INTERMEDIATE = "intermediate"
    EXTERNAL = "external"


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


_TIMEFRAME_MINUTES: Mapping[Timeframe, int] = {
    Timeframe.H4: 240,
    Timeframe.H1: 60,
    Timeframe.M15: 15,
    Timeframe.M5: 5,
    Timeframe.M1: 1,
}


@dataclass(frozen=True)
class ScaleSpec:
    """One explicitly registered observation scale.

    ``1D`` is accepted as a parked declaration so the contract can describe
    the intended macro scale without silently enabling an unimplemented daily
    aggregation rule.
    """

    timeframe: Timeframe | str
    role: ScaleRole | str
    session_anchor: str = "exchange_session"
    completed_bar_only: bool = True
    history_limit: int = 1024
    enabled: bool = True
    structural_scales: tuple[StructuralScale, ...] = (
        StructuralScale.INTERNAL,
        StructuralScale.INTERMEDIATE,
        StructuralScale.EXTERNAL,
    )

    def __post_init__(self) -> None:
        timeframe = (
            self.timeframe.value
            if isinstance(self.timeframe, Timeframe)
            else str(self.timeframe)
        )
        role = self.role if isinstance(self.role, ScaleRole) else ScaleRole(self.role)
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self,
            "structural_scales",
            tuple(StructuralScale(value) for value in self.structural_scales),
        )
        if (
            timeframe not in {item.value for item in Timeframe} | {"1D"}
            or self.session_anchor != "exchange_session"
            or type(self.completed_bar_only) is not bool
            or type(self.enabled) is not bool
            or type(self.history_limit) is not int
            or self.history_limit < 8
            or not self.structural_scales
            or len(self.structural_scales) != len(set(self.structural_scales))
        ):
            raise ValueError("invalid scale specification")
        if self.enabled and not self.completed_bar_only:
            raise ValueError("enabled market scales must use completed bars only")
        if timeframe == "1D" and self.enabled:
            raise ValueError("1D aggregation is registered but remains parked")

    @property
    def key(self) -> str:
        return self.timeframe.value if isinstance(self.timeframe, Timeframe) else str(self.timeframe)

    @property
    def native_timeframe(self) -> Timeframe | None:
        if isinstance(self.timeframe, Timeframe):
            return self.timeframe
        return next((item for item in Timeframe if item.value == self.timeframe), None)

    @property
    def minutes(self) -> int | None:
        timeframe = self.native_timeframe
        return None if timeframe is None else _TIMEFRAME_MINUTES[timeframe]


def parse_scale_specs(
    payload: Sequence[Mapping[str, Any]] | None,
    *,
    history_limit: int = 1024,
) -> tuple[ScaleSpec, ...]:
    if not payload:
        raise ValueError("scale registry must be supplied explicitly")
    specs = tuple(
        ScaleSpec(
            timeframe=str(item.get("timeframe", "")),
            role=str(item.get("role", "")),
            session_anchor=str(item.get("session_anchor", "exchange_session")),
            completed_bar_only=item.get("completed_bar_only", True),
            history_limit=int(item.get("history_limit", history_limit)),
            enabled=item.get("enabled", True),
            structural_scales=tuple(
                item.get(
                    "structural_scales",
                    tuple(value.value for value in StructuralScale),
                )
            ),
        )
        for item in payload
    )
    keys = tuple(item.key for item in specs)
    if len(keys) != len(set(keys)):
        raise ValueError("scale registry contains duplicate timeframes")
    enabled = {item.native_timeframe for item in specs if item.enabled}
    if not set(CORE_TIMEFRAMES).issubset(enabled):
        raise ValueError("scale registry must retain the causal core frames")
    return specs


def scale_registry_id(scale_specs: Sequence[ScaleSpec]) -> str:
    """Hash the complete ordered scale contract, not only its display role."""

    specs = tuple(scale_specs)
    if not specs:
        raise ValueError("scale registry cannot be empty")
    payload = {
        "registry_version": "scale-registry-v1",
        "scales": [to_primitive(item) for item in specs],
    }
    return f"scale:{content_hash(payload)[:24]}"


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
            or self.lifecycle not in {"active", "closed", "invalidated"}
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
    supplemental_timeframes: tuple[str, ...]
    reason_codes: tuple[str, ...]
    trigger_event_ids: tuple[str, ...]
    question: str
    resolution_status: EvidenceStatus
    switched: bool
    switched_at: pd.Timestamp | None
    prior_timeframes: tuple[str, ...] = ()
    hypothesis_id: str | None = None
    phase_at_selection: str | None = None
    supplemental_query_used: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "asof", aware_timestamp(self.asof, name="focus.asof"))
        if self.switched_at is not None:
            object.__setattr__(self, "switched_at", aware_timestamp(self.switched_at, name="focus.switched_at"))
        object.__setattr__(self, "resolution_status", EvidenceStatus(self.resolution_status))
        for name in (
            "primary_timeframes",
            "supplemental_timeframes",
            "reason_codes",
            "trigger_event_ids",
            "prior_timeframes",
        ):
            values = tuple(dict.fromkeys(getattr(self, name)))
            object.__setattr__(self, name, values)
        if (
            not self.primary_timeframes
            or not self.reason_codes
            or not self.question
            or self.switched != bool(self.switched_at)
            or self.supplemental_query_used and not self.supplemental_timeframes
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
    contradicting_graph_paths: tuple[tuple[str, ...], ...]
    missing_evidence: Mapping[str, EvidenceStatus]
    ambiguous_evidence: Mapping[str, EvidenceStatus]
    context_draw_id: str | None
    primary_target_id: str | None
    invalidation_id: str | None
    evidence_revision_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "context_root_ids", tuple(dict.fromkeys(self.context_root_ids)))
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
        "mature_at",
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
        "mature_at",
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
        "formed_at", "mature_at", "broken_at", "lower_bound",
        "upper_bound", "value_price", "transition_reason",
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
        "post_break_state": "post_break_state",
        "mss_qualified": "mss_qualified",
        "source_bos_mss_qualified": "source_bos_mss_qualified",
        "source_kind": "source_kind",
        "structural_rank": "structural_rank",
        "is_protected_swing": "is_protected_swing",
        "zone_role": "zone_role",
        "resolved_side": "resolved_side",
        "close_outside_on_sweep": "close_outside_on_sweep",
        "deadline_elapsed": "deadline_elapsed",
        "outside_run_side": "outside_run_side",
        "reentry_candidate_at": "reentry_candidate_at",
        "reentry_failed_at": "reentry_failed_at",
        "deadline_at": "deadline_at",
        "censored_at": "censored_at",
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


def _is_material_cross_scale_conflict(
    edge: SceneEdge,
    node_by_id: Mapping[str, SceneNode],
    *,
    ready_timeframes: set[str] | frozenset[str] | None = None,
) -> bool:
    """Distinguish structural conflict from an ordinary internal pullback."""

    if edge.relation is not SceneEdgeKind.OPPOSES or edge.lifecycle != "active":
        return False
    source = node_by_id.get(edge.source_node_id)
    target = node_by_id.get(edge.target_node_id)
    if source is None or target is None:
        return False
    if ready_timeframes is not None and (
        source.timeframe not in ready_timeframes
        or target.timeframe not in ready_timeframes
    ):
        # Keep the raw OPPOSES fact in the graph, but do not give an
        # under-warmed scale authority to contradict a completed semantic
        # context.  Its interpretation remains explicitly UNKNOWN below.
        return False
    if (
        source.timeframe == target.timeframe
        or source.kind not in {"structure", "bos", "displacement"}
        or target.kind != "structure"
        or source.ambiguity_state is not EvidenceStatus.CONFIRMED
        or target.ambiguity_state is not EvidenceStatus.CONFIRMED
        or _is_terminal(source.kind, source.lifecycle)
        or _is_terminal(target.kind, target.lifecycle)
        or source.direction is None
        or target.direction is None
        or source.direction is target.direction
    ):
        return False
    # A qualified displacement can oppose a higher structure from an internal
    # setup scale.  Bare internal swings/BOS/structures describe pullbacks and
    # stay in the graph without globally putting Focus into CONFLICTING.
    return (
        source.kind == "displacement"
        or source.structural_scale
        in {StructuralScale.INTERMEDIATE, StructuralScale.EXTERNAL}
    )


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
        self._visible_liquidity_node_ids_by_side: dict[str, set[str]] = {
            "above": set(),
            "below": set(),
        }
        self._recent_terminal_context_node_ids_by_kind: dict[
            str,
            deque[str],
        ] = defaultdict(deque)

    @property
    def revision_id(self) -> str:
        return self._revision_id

    @property
    def last_asof(self) -> pd.Timestamp | None:
        return self._last_asof

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
        return tuple(self._node_revisions.get(node_id, ()))

    def edge_history(self, edge_id: str) -> tuple[SceneEdge, ...]:
        return tuple(self._edge_revisions.get(edge_id, ()))

    def _next_revision(self) -> str:
        self._revision += 1
        self._revision_id = f"scene:r{self._revision:012d}"
        return self._revision_id

    def _epoch_asof(self, asof: pd.Timestamp) -> str:
        epoch = "epoch:0"
        for boundary, candidate in self._epoch_boundaries:
            if boundary > asof:
                break
            epoch = candidate
        return epoch

    def _current_kind_nodes(self, kind: str) -> tuple[SceneNode, ...]:
        """Return current-epoch nodes of one kind without scanning history."""

        return tuple(
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
        self._current_epoch_node_ids = set()
        self._current_epoch_node_ids_by_kind = defaultdict(set)
        self._current_epoch_node_ids_by_timeframe = defaultdict(set)
        self._current_epoch_ambiguous_node_ids = set()
        self._current_epoch_active_opposes_edge_ids = set()
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
                "mature_at",
                "state_started_at",
                "observed_at",
                default=asof,
            )
        resolution_reason = (
            getattr(item, "transition_reason", None)
            or getattr(item, "lifecycle_reason", None)
            or getattr(item, "resolution_reason", None)
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
                "mature_at",
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
        return {
            EventKind.SWING_STATE: "swing",
            EventKind.STRUCTURE_STATE: "structure",
            EventKind.BOS_STATE: "bos",
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
        }.get(event.kind, event.kind.value)

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
        top_level = tuple(
            (item, False)
            for group in (
                observation.liquidity_inventory,
                observation.liquidity_pool_states,
                observation.manipulations,
                observation.entry_locations,
                observation.qualified_reacceptances,
                observation.micro_bos_references,
                observation.path_sequences,
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
        if bootstrap:
            for timeline in observation.retained_entity_timelines.values():
                for event in timeline:
                    if (
                        event.event_id not in self._seen_market_event_ids
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
                raw_node_id = (
                    None
                    if swing_id is None
                    else self._node_id_for_source(
                        swing_id,
                        kind="swing",
                        asof=structure.confirmed_at,
                    )
                )
                if raw_node_id is None:
                    continue
                raw_node = next(
                    value
                    for value in reversed(
                        self._node_revisions[raw_node_id]
                    )
                    if value.observed_at <= structure.confirmed_at
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

    def _locate_liquidity_at_zones(self, node: SceneNode) -> None:
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
            observed_at = max(liquidity.observed_at, zone.observed_at)
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
        for node_id in (*self._current_added_nodes, *self._current_revised_nodes):
            node = self._nodes[node_id]
            for source_id in node.source_ids:
                source_node = self._node_id_for_source(
                    source_id,
                    asof=node.observed_at,
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
                        node.observed_at,
                        (node.node_id, source_id),
                    )
                    source_state = self._nodes[source_node]
                    if not _is_terminal(
                        source_state.kind, source_state.lifecycle
                    ):
                        self.add_node(
                            replace(
                                source_state,
                                observed_at=node.observed_at,
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
                            node.observed_at,
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
                            node.observed_at,
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
                        node.observed_at,
                        (source_id, node.node_id),
                    )
                    continue
                elif node.kind == "order_block" and source_kind == "bos":
                    self._add_relation(
                        source_node,
                        SceneEdgeKind.CREATES,
                        node.node_id,
                        node.observed_at,
                        (source_id, node.node_id),
                    )
                    continue
                elif node.kind == "manipulation" and source_kind in {"liquidity", "liquidity_pool", "range"}:
                    relation = SceneEdgeKind.SWEEPS
                elif node.kind == "entry_location" and source_kind in {"fvg", "order_block"}:
                    relation = SceneEdgeKind.RETURNS_TO
                elif node.kind == "micro_bos" and source_kind in {"bos", "swing", "path_sequence"}:
                    relation = SceneEdgeKind.CONFIRMS
                self._add_relation(node.node_id, relation, source_id, node.observed_at, (node.node_id, source_id))

            if node.kind == "bos" and node.lifecycle == "confirmed":
                displacement_nodes = tuple(
                    candidate
                    for source_id in node.source_ids
                    for candidate in (
                        self._node_id_for_source(
                            source_id,
                            kind="displacement",
                            asof=node.observed_at,
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
                            asof=node.observed_at,
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
                            node.observed_at,
                            (node.entity_id or node.node_id,),
                        )

            self._locate_liquidity_at_zones(node)

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
            structures = [
                value
                for value in self._current_kind_nodes_asof(
                    "structure",
                    node.observed_at,
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
            self._add_relation(node.node_id, relation, parent.node_id, node.observed_at, (node.node_id, parent.node_id))

        for displacement in (
            node for node in lower_nodes if node.kind == "displacement"
        ):
            manipulation_candidates = (
                *self._current_kind_nodes_asof(
                    "manipulation",
                    displacement.observed_at,
                ),
                *self._recent_terminal_context_nodes(
                    "manipulation",
                    asof=displacement.observed_at,
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
                    displacement.observed_at,
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
        for edge_id in self._outgoing.get(node_id, ()):
            edge = edge_by_id.get(edge_id)
            if edge is None:
                continue
            if edge.lifecycle == "active":
                yield edge.target_node_id, edge
        for edge_id in self._incoming.get(node_id, ()):
            edge = edge_by_id.get(edge_id)
            if edge is None:
                continue
            if edge.lifecycle == "active":
                yield edge.source_node_id, edge
        if current_view:
            for edge in self._current_path_block_edges.values():
                if edge.source_node_id == node_id:
                    yield edge.target_node_id, edge
                elif edge.target_node_id == node_id:
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
    ) -> tuple[str, ...]:
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
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append((neighbor, (*path, edge.relation.value, neighbor)))
        return ()

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
        focus_frames = set(focus_state.primary_timeframes) | set(focus_state.supplemental_timeframes)
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
        conflicts = tuple(
            edge.edge_id
            for edge in edges
            if _is_material_cross_scale_conflict(
                edge,
                node_by_id,
                ready_timeframes=ready_frames,
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

    def write_parquet(self, destination: str | Path) -> Mapping[str, Path]:
        """Materialize node/edge revision ledgers without introducing a DB."""

        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        node_path = root / "scene_nodes.parquet"
        edge_path = root / "scene_edges.parquet"
        node_rows = [to_primitive(item) for values in self._node_revisions.values() for item in values]
        edge_rows = [to_primitive(item) for values in self._edge_revisions.values() for item in values]
        pd.DataFrame(node_rows).to_parquet(node_path, index=False)
        pd.DataFrame(edge_rows).to_parquet(edge_path, index=False)
        return {"nodes": node_path, "edges": edge_path}


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


def select_focus(
    previous_belief: Any | None,
    observation: MarketObservation,
    scene_delta: SceneGraphDelta | None,
    scene_graph: TemporalMarketSceneGraph | None,
) -> FocusState:
    """Choose what to read using bounded, inspectable rules only."""

    prior_focus = None if previous_belief is None else getattr(previous_belief, "focus_state", None)
    ranked = [] if previous_belief is None else previous_belief.ranked()
    dominant = ranked[0] if ranked else None
    phase = PlaybookPhase.FORMING if dominant is None else dominant.phase
    active_frames = tuple(
        timeframe.value for timeframe in observation.active_timeframes
    )
    active_set = set(active_frames)
    ready_frames = frozenset(
        timeframe.value
        for timeframe in observation.active_timeframes
        if observation.frame(timeframe).ready
    )

    def enabled(values: Sequence[str]) -> tuple[str, ...]:
        filtered = tuple(
            dict.fromkeys(value for value in values if value in active_set)
        )
        if filtered:
            return filtered
        if Timeframe.M1.value in active_set:
            return (Timeframe.M1.value,)
        return (active_frames[0],)

    desired = enabled(_PHASE_FOCUS[phase])
    reasons: list[str] = ["initial_context" if prior_focus is None else "next_expected_evidence"]
    triggers: list[str] = []
    question = "establish higher-timeframe structure, bridge context and active draw"
    allowed_switch = prior_focus is None
    if prior_focus is not None and prior_focus.resolution_status in {
        EvidenceStatus.CONFIRMED,
        EvidenceStatus.INVALIDATED,
    }:
        reasons.append("focus_question_resolved_or_invalidated")
        allowed_switch = True
    if dominant is not None:
        question = {
            PlaybookPhase.WAITING_LOCATION: "has price reached the frozen setup zone without chasing",
            PlaybookPhase.WAITING_TRIGGER: "has a strict rejection, reacceptance or micro BOS confirmed",
            PlaybookPhase.EXECUTABLE: "does trigger and remaining delivery remain valid",
            PlaybookPhase.ENTERED: "is the frozen thesis delivering, weakening or invalidated",
        }.get(phase, "what exact causal event is required next")
        if prior_focus is not None and prior_focus.phase_at_selection != phase.value:
            reasons.append("playbook_stage_changed")
            allowed_switch = True
    if scene_delta is not None and scene_delta.changed:
        key_nodes = [
            scene_graph._nodes[node_id]
            for node_id in (*scene_delta.added_node_ids, *scene_delta.revised_node_ids)
            if scene_graph is not None and node_id in scene_graph._nodes
            and scene_graph._nodes[node_id].kind in {"bos", "displacement", "manipulation", "entry_location", "micro_bos"}
        ]
        if key_nodes:
            reasons.append("new_key_event")
            triggers.extend(node.node_id for node in key_nodes)
            allowed_switch = True
            event_frames = tuple(dict.fromkeys(node.timeframe for node in key_nodes))
            desired = enabled((*desired, *event_frames))
    current_scene_view = bool(
        scene_graph is not None
        and scene_graph.last_asof == observation.asof
    )
    current_epoch = (
        None
        if scene_graph is None
        else (
            scene_graph._market_epoch_id
            if current_scene_view
            else scene_graph._epoch_asof(observation.asof)
        )
    )
    conflicts: tuple[SceneEdge, ...] = ()
    if scene_graph is not None:
        current_nodes = scene_graph._nodes
        candidate_edges = (
            tuple(
                sorted(
                    (
                        scene_graph._edges[edge_id]
                        for edge_id in scene_graph._current_epoch_active_opposes_edge_ids
                        if edge_id in scene_graph._edges
                    ),
                    key=lambda edge: (edge.observed_at, edge.edge_id),
                )
            )
            if current_scene_view
            else scene_graph.edges_asof(observation.asof)
        )
        opposing = tuple(
            edge
            for edge in candidate_edges
            if _is_material_cross_scale_conflict(
                edge,
                current_nodes,
                ready_timeframes=ready_frames,
            )
            and scene_graph._nodes[edge.source_node_id].market_epoch_id
            == current_epoch
        )
        context_ids: tuple[str, ...] = ()
        if previous_belief is not None:
            dominant_context_id = getattr(
                previous_belief,
                "dominant_hypothesis_id",
                None,
            )
            dominant_context = getattr(
                previous_belief,
                "context_hypotheses",
                {},
            ).get(dominant_context_id)
            if dominant_context is not None:
                context_ids = tuple(dominant_context.context_root_ids)
            elif dominant is not None:
                context_ids = tuple(
                    value
                    for value in (
                        dominant.context_id,
                        dominant.setup_context_id,
                        dominant.initiating_event_id,
                    )
                    if value is not None
                )
        closure = {
            node_id
            for source_id in context_ids
            for node_id in (
                scene_graph._node_id_for_source(
                    source_id,
                    asof=observation.asof,
                ),
            )
            if node_id is not None
        }
        frontier = deque((node_id, 0) for node_id in closure)
        while frontier:
            node_id, depth = frontier.popleft()
            if depth >= 4:
                continue
            for neighbor, _ in scene_graph._neighbors(
                node_id,
                asof=observation.asof,
            ):
                if neighbor not in closure:
                    closure.add(neighbor)
                    frontier.append((neighbor, depth + 1))
        if closure:
            conflicts = tuple(
                edge
                for edge in opposing
                if edge.source_node_id in closure
                and edge.target_node_id in closure
            )
        elif dominant is None and scene_delta is not None:
            changed_edges = set(
                (*scene_delta.added_edge_ids, *scene_delta.revised_edge_ids)
            )
            conflicts = tuple(
                edge for edge in opposing if edge.edge_id in changed_edges
            )
    if conflicts:
        reasons.append("cross_scale_conflict")
        triggers.extend(edge.edge_id for edge in conflicts[-4:])
        desired = enabled(
            (
                Timeframe.H4.value,
                Timeframe.H1.value,
                Timeframe.M15.value,
                Timeframe.M5.value,
            )
        )
        question = "which scale owns the active structural conflict"
        allowed_switch = True
    if dominant is not None and dominant.plan is not None:
        plan = dominant.plan
        if plan.entry_zone_lower is not None and plan.entry_zone_upper is not None and plan.entry_zone_lower <= observation.price <= plan.entry_zone_upper:
            reasons.append("price_entered_frozen_zone")
            desired = enabled((Timeframe.M5.value, Timeframe.M1.value))
            question = "has the frozen-zone trigger actually confirmed"
            allowed_switch = True
        if observation.execution.minutes_to_deadline <= 10:
            reasons.append("deadline_near")
            desired = enabled((*desired, Timeframe.M1.value))
            allowed_switch = True
    prior_frames = () if prior_focus is None else prior_focus.primary_timeframes
    switched = bool(prior_focus is not None and tuple(desired) != tuple(prior_frames) and allowed_switch)
    if prior_focus is not None and not switched and not allowed_switch:
        desired = enabled(prior_focus.primary_timeframes)
        reasons = ["focus_retained_no_switch_event"]
        question = prior_focus.question
    resolution = (
        EvidenceStatus.CONFLICTING
        if conflicts
        else EvidenceStatus.AMBIGUOUS
        if scene_graph is not None
        and (
            bool(scene_graph._current_epoch_ambiguous_node_ids)
            if current_scene_view
            else any(
                node.market_epoch_id == current_epoch
                and node.ambiguity_state is EvidenceStatus.AMBIGUOUS
                for node in scene_graph.nodes_asof(observation.asof)
            )
        )
        else EvidenceStatus.FORMING
    )
    return FocusState(
        asof=observation.asof,
        primary_timeframes=tuple(desired),
        supplemental_timeframes=(),
        reason_codes=tuple(dict.fromkeys(reasons)),
        trigger_event_ids=tuple(dict.fromkeys(triggers)),
        question=question,
        resolution_status=resolution,
        switched=switched,
        switched_at=observation.asof if switched else None,
        prior_timeframes=tuple(prior_frames),
        hypothesis_id=(
            None
            if dominant is None
            else (
                getattr(previous_belief, "dominant_hypothesis_id", None)
                or dominant.key
            )
        ),
        phase_at_selection=phase.value,
    )


def supplement_focus_once(
    focus: FocusState,
    prior_hypothesis: Any | None,
    current_hypothesis: Any | None,
    active_timeframes: Sequence[Timeframe] | None = None,
) -> FocusState:
    if (
        focus.supplemental_query_used
        or prior_hypothesis is None
        or current_hypothesis is None
        or prior_hypothesis.phase is current_hypothesis.phase
    ):
        return focus
    allowed = (
        set(active_timeframes)
        if active_timeframes is not None
        else set(CORE_TIMEFRAMES)
        | {
            Timeframe(value)
            for value in (
                *focus.primary_timeframes,
                *focus.prior_timeframes,
            )
            if value in {item.value for item in Timeframe}
        }
    )
    requested = tuple(
        value
        for value in _PHASE_FOCUS[current_hypothesis.phase]
        if Timeframe(value) in allowed
    )
    supplemental = tuple(value for value in requested if value not in focus.primary_timeframes)
    if not supplemental:
        return focus
    return replace(
        focus,
        supplemental_timeframes=supplemental,
        reason_codes=tuple(dict.fromkeys((*focus.reason_codes, "playbook_stage_changed_after_query"))),
        question="resolve the evidence required by the newly reached stage",
        switched=True,
        switched_at=focus.asof,
        hypothesis_id=(
            focus.hypothesis_id
            if prior_hypothesis.key == current_hypothesis.key
            and focus.hypothesis_id is not None
            else current_hypothesis.key
        ),
        phase_at_selection=current_hypothesis.phase.value,
        supplemental_query_used=True,
    )


def build_hypothesis_states(
    hypotheses: Mapping[str, Any],
    graph: TemporalMarketSceneGraph | None,
    focused: FocusedObservation | None,
) -> Mapping[str, HypothesisState]:
    """Project dominant six-slot beliefs into at most two contexts per slot."""

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
        roots = tuple(
            dict.fromkeys(
                value
                for value in (
                    belief.context_id,
                    belief.setup_context_id,
                    belief.initiating_event_id,
                    *(source for step in satisfied for source in step.source_ids),
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
        contradicting_paths: list[tuple[str, ...]] = []
        if focused is not None:
            focused_conflict_ids = set(focused.cross_scale_conflicts)
            contradicting_paths.extend(
                (
                    edge.source_node_id,
                    edge.relation.value,
                    edge.target_node_id,
                )
                for edge in focused.edges
                if edge.edge_id in focused_conflict_ids
                and edge.source_node_id in closure_ids
                and edge.target_node_id in closure_ids
            )
        context_root = roots[0] if roots else slot_key
        raw = f"{slot_key}|{context_root}"
        hypothesis_id = f"hyp:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"
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
            contradicting_graph_paths=tuple(contradicting_paths),
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
        )
        if graph is None or asof is None or not closure_ids:
            continue
        candidate_kinds = {
            Playbook.DISPLACEMENT_FIRST_PULLBACK: {"displacement", "fvg", "order_block"},
            Playbook.LIQUIDITY_SWEEP_REVERSAL: {"liquidity_pool", "manipulation"},
            Playbook.FAILED_AUCTION_VALUE_RETURN: {"range", "manipulation"},
        }[belief.playbook]
        alternatives = [
            node
            for node in reversed(graph.nodes_asof(asof))
            if node.kind in candidate_kinds
            and node.ambiguity_state not in {EvidenceStatus.INVALIDATED, EvidenceStatus.UNKNOWN}
            and (node.direction is None or node.direction is belief.direction)
            and node.node_id not in roots
            and node.node_id in closure_ids
            and node.node_id not in {
                graph._node_id_for_source(value, asof=asof)
                for value in roots
            }
        ]
        if alternatives:
            alternate = alternatives[0]
            alternate_raw = f"{slot_key}|{alternate.node_id}"
            alternate_id = f"hyp:{hashlib.sha256(alternate_raw.encode()).hexdigest()[:24]}"
            output[alternate_id] = replace(
                output[hypothesis_id],
                hypothesis_id=alternate_id,
                context_root_ids=(alternate.node_id,),
                sequence_stage=PlaybookPhase.FORMING.value,
                next_expected_event="context_specific_sequence_evaluation",
                supporting_graph_paths=(),
                missing_evidence={
                    "context_specific_sequence_evaluation": EvidenceStatus.UNKNOWN
                },
                ambiguous_evidence=(
                    {"context_interpretation": EvidenceStatus.AMBIGUOUS}
                    if alternate.ambiguity_state is EvidenceStatus.AMBIGUOUS
                    else {}
                ),
                context_draw_id=None,
                primary_target_id=None,
                invalidation_id=None,
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
    "SceneEdge",
    "SceneEdgeKind",
    "SceneGraphDelta",
    "SceneNode",
    "StructuralScale",
    "TemporalMarketSceneGraph",
    "build_hypothesis_states",
    "parse_scale_specs",
    "scale_registry_id",
    "select_focus",
    "supplement_focus_once",
]
