"""Deterministic hierarchical projection over the existing causal Eye reducers.

This module does not detect Swing, BOS, liquidity, displacement, FVG, OB, or
dealing ranges.  It projects those already-authoritative states into the public
Timeframe/Relation/Session/Snapshot contract and emits replayable state-change
events.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import deque
import copy
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import pandas as pd

from .event_store import ImmutableEventStore, validate_canonical_event
from .foundation_registry import (
    FOUNDATION_CANONICAL_IDENTITY,
    FOUNDATION_VERSION,
)
from .market_clock import (
    expected_trading_minutes,
    next_registered_native_completion,
    registered_native_bar_bounds,
    validate_completed_bar_header,
    validate_registered_native_bar_root,
)
from .model import (
    BOSLifecycle,
    BOSScope,
    Candle,
    DealingRangeState,
    DealingRangeLifecycle,
    Direction,
    DisplacementObservation,
    EventKind,
    EventOrigin,
    FairValueGapLifecycle,
    FrameObservation,
    FrozenDict,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    MarketEvent,
    OrderBlockLifecycle,
    SMC_SEMANTIC_VERSION,
    StructuralLegState,
    StructureLifecycle,
    SwingLifecycle,
    SwingPoint,
    SwingRank,
    SwingSide,
    Timeframe,
    aware_timestamp,
    candle_identity,
    clamp,
    price_to_ticks,
    to_primitive,
)

if TYPE_CHECKING:
    from .semantic_foundation import FoundationProjection, FoundationRecord


MARKET_SNAPSHOT_SCHEMA_VERSION = 2


class DeliveryPhase(str, Enum):
    BALANCE = "balance"
    EXPANSION = "expansion"
    RETRACEMENT = "retracement"
    REVERSAL_ATTEMPT = "reversal_attempt"
    TRANSITION = "transition"


class RelationRole(str, Enum):
    ALIGNED_EXPANSION = "aligned_expansion"
    PARENT_RETRACEMENT = "parent_retracement"
    REVERSAL_ATTEMPT = "reversal_attempt"
    PARENT_TRANSITION = "parent_transition"
    BALANCE_INSIDE_PARENT = "balance_inside_parent"
    UNRESOLVED = "unresolved"


class MarketSnapshotAuthority(str, Enum):
    """The sole state source used to build one public market snapshot."""

    ATOMIC_EVENT_REDUCER = "atomic_event_reducer"
    FRAME_PROJECTION = "frame_projection"
    LEGACY_UNSPECIFIED = "legacy_unspecified"


class LiquidityRangeRole(str, Enum):
    """Deterministic candidate membership in one active dealing range."""

    IRL = "irl"
    ERL = "erl"
    UNRESOLVED = "unresolved"


_TIMEFRAME_AUTHORITY_ORDER = (
    Timeframe.H4,
    Timeframe.H1,
    Timeframe.M15,
    Timeframe.M5,
    Timeframe.M1,
)


_SWING_RANK_DEPTH = {
    SwingRank.UNRESOLVED: -1,
    SwingRank.MICRO: 0,
    SwingRank.INTERNAL: 1,
    SwingRank.STRUCTURAL: 2,
    SwingRank.EXTERNAL: 3,
}


@dataclass(frozen=True)
class SwingRankAssignment:
    """One immutable, causally clocked role assignment for a swing."""

    rank: SwingRank
    assigned_at: pd.Timestamp
    assignment_event_id: str
    source_kind: str
    source_event_ids: tuple[str, ...] = ()
    context_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "rank", SwingRank(self.rank))
        object.__setattr__(
            self,
            "assigned_at",
            aware_timestamp(self.assigned_at, name="swing_rank.assigned_at"),
        )
        object.__setattr__(self, "source_event_ids", tuple(self.source_event_ids))
        object.__setattr__(
            self, "context_event_ids", tuple(self.context_event_ids)
        )
        if (
            self.rank is SwingRank.UNRESOLVED
            or not self.assignment_event_id
            or not self.source_kind
            or any(not value for value in self.source_event_ids)
            or any(not value for value in self.context_event_ids)
        ):
            raise ValueError("swing-rank assignment is invalid")


@dataclass(frozen=True)
class SwingHierarchyView:
    """Append-only causal role history for one immutable confirmed swing."""

    swing_id: str
    timeframe: Timeframe
    semantic_rank: SwingRank
    nesting_depth: int
    assignments: tuple[SwingRankAssignment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "semantic_rank", SwingRank(self.semantic_rank))
        assignments = tuple(self.assignments)
        object.__setattr__(self, "assignments", assignments)
        if (
            not self.swing_id
            or not assignments
            or assignments[0].rank is not SwingRank.MICRO
            or type(self.nesting_depth) is not int
            or self.nesting_depth < 0
            or len(
                {item.assignment_event_id for item in assignments}
            ) != len(assignments)
            or any(
                right.assigned_at < left.assigned_at
                for left, right in zip(assignments, assignments[1:])
            )
        ):
            raise ValueError("swing hierarchy view is invalid")
        maximum = max(assignments, key=lambda item: _SWING_RANK_DEPTH[item.rank])
        if (
            self.semantic_rank is not maximum.rank
            or self.nesting_depth != _SWING_RANK_DEPTH[maximum.rank]
        ):
            raise ValueError("swing hierarchy summary disagrees with assignments")

    @property
    def role_depth(self) -> int:
        """Return causal semantic-role depth, never geometric nesting depth."""

        return self.nesting_depth


@dataclass(frozen=True)
class SwingGeometryNode:
    """One confirmed Swing's outcome-blind definitional bar envelope."""

    swing_id: str
    timeframe: Timeframe
    symbol: str
    instrument_id: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    lower_bound: float
    upper_bound: float
    known_at: pd.Timestamp
    source_candle_ids: tuple[str, ...]
    foundation_version: str = FOUNDATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        for name in ("window_start", "window_end", "known_at"):
            object.__setattr__(
                self,
                name,
                aware_timestamp(getattr(self, name), name=f"swing_geometry.{name}"),
            )
        source_ids = tuple(self.source_candle_ids)
        object.__setattr__(self, "source_candle_ids", source_ids)
        if (
            not self.swing_id
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or self.window_start >= self.window_end
            or self.known_at != self.window_end
            or not math.isfinite(float(self.lower_bound))
            or not math.isfinite(float(self.upper_bound))
            or not 0.0 < self.lower_bound < self.upper_bound
            or not source_ids
            or len(source_ids) != len(set(source_ids))
            or any(not isinstance(value, str) or not value for value in source_ids)
            or self.foundation_version != FOUNDATION_VERSION
        ):
            raise ValueError("swing geometry node is invalid")

    @property
    def duration_seconds(self) -> int:
        return int((self.window_end - self.window_start).total_seconds())

    @property
    def price_span(self) -> float:
        return float(self.upper_bound - self.lower_bound)


@dataclass(frozen=True)
class SwingGeometryAssignment:
    """One append-only parent assignment in the geometric nesting tree."""

    assignment_id: str
    child_swing_id: str
    parent_swing_id: str | None
    geometric_depth: int
    assigned_at: pd.Timestamp
    supersedes_assignment_id: str | None = None
    foundation_version: str = FOUNDATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "assigned_at",
            aware_timestamp(
                self.assigned_at,
                name="swing_geometry_assignment.assigned_at",
            ),
        )
        if (
            not self.assignment_id
            or not self.child_swing_id
            or self.parent_swing_id == self.child_swing_id
            or type(self.geometric_depth) is not int
            or self.geometric_depth < 0
            or ((self.parent_swing_id is None) != (self.geometric_depth == 0))
            or self.supersedes_assignment_id == self.assignment_id
            or self.foundation_version != FOUNDATION_VERSION
        ):
            raise ValueError("swing geometry assignment is invalid")


@dataclass(frozen=True)
class LiquidityClusterState:
    """One pure spatial generation over immutable candidate-level identities."""

    cluster_id: str
    side: str
    member_level_ids: tuple[str, ...]
    member_prices: tuple[float, ...]
    member_source_ids: tuple[str, ...]
    lower_price: float
    upper_price: float
    tick_size: float
    started_at: pd.Timestamp
    known_at: pd.Timestamp
    updated_at: pd.Timestamp
    terminated_at: pd.Timestamp | None = None
    termination_reason: str | None = None
    supersedes_cluster_ids: tuple[str, ...] = ()
    foundation_version: str = FOUNDATION_VERSION

    def __post_init__(self) -> None:
        for name in ("started_at", "known_at", "updated_at", "terminated_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"liquidity_cluster.{name}"),
                )
        member_ids = tuple(self.member_level_ids)
        prices = tuple(float(value) for value in self.member_prices)
        source_ids = tuple(self.member_source_ids)
        supersedes = tuple(self.supersedes_cluster_ids)
        object.__setattr__(self, "member_level_ids", member_ids)
        object.__setattr__(self, "member_prices", prices)
        object.__setattr__(self, "member_source_ids", source_ids)
        object.__setattr__(self, "supersedes_cluster_ids", supersedes)
        if (
            not self.cluster_id
            or self.side not in {"above", "below"}
            or len(member_ids) < 2
            or len(member_ids) != len(prices)
            or len(member_ids) != len(set(member_ids))
            or any(not isinstance(value, str) or not value for value in member_ids)
            or not source_ids
            or len(source_ids) != len(set(source_ids))
            or any(not isinstance(value, str) or not value for value in source_ids)
            or not all(math.isfinite(value) and value > 0.0 for value in prices)
            or prices != tuple(sorted(prices))
            or not math.isfinite(float(self.tick_size))
            or self.tick_size <= 0.0
            or not math.isclose(self.lower_price, prices[0])
            or not math.isclose(self.upper_price, prices[-1])
            or self.upper_price - self.lower_price > self.tick_size + 1e-12
            or self.started_at > self.known_at
            or self.known_at > self.updated_at
            or ((self.terminated_at is None) != (self.termination_reason is None))
            or (
                self.terminated_at is not None
                and (
                    self.terminated_at < self.updated_at
                    or self.termination_reason
                    not in {
                        "superseded",
                        "contract_reset",
                        "data_reset",
                        "semantic_reset",
                    }
                )
            )
            or len(supersedes) != len(set(supersedes))
            or self.cluster_id in supersedes
            or self.foundation_version != FOUNDATION_VERSION
        ):
            raise ValueError("liquidity cluster generation is invalid")

    @property
    def generation_id(self) -> str:
        return self.cluster_id


@dataclass(frozen=True)
class LiquidityClusterSupersession:
    superseded_cluster_id: str
    replacement_cluster_ids: tuple[str, ...]
    known_at: pd.Timestamp
    foundation_version: str = FOUNDATION_VERSION

    def __post_init__(self) -> None:
        replacements = tuple(self.replacement_cluster_ids)
        object.__setattr__(self, "replacement_cluster_ids", replacements)
        object.__setattr__(
            self,
            "known_at",
            aware_timestamp(self.known_at, name="liquidity_cluster_supersession.known_at"),
        )
        if (
            not self.superseded_cluster_id
            or len(replacements) != len(set(replacements))
            or self.superseded_cluster_id in replacements
            or self.foundation_version != FOUNDATION_VERSION
        ):
            raise ValueError("liquidity cluster supersession is invalid")


@dataclass(frozen=True)
class LiquidityClusterUpdate:
    active: tuple[LiquidityClusterState, ...]
    started: tuple[LiquidityClusterState, ...]
    terminated: tuple[LiquidityClusterState, ...]
    supersessions: tuple[LiquidityClusterSupersession, ...]

    def __post_init__(self) -> None:
        for name in ("active", "started", "terminated", "supersessions"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        active_ids = {item.cluster_id for item in self.active}
        started_ids = {item.cluster_id for item in self.started}
        terminated_ids = {item.cluster_id for item in self.terminated}
        if (
            len(active_ids) != len(self.active)
            or len(started_ids) != len(self.started)
            or len(terminated_ids) != len(self.terminated)
            or not started_ids.issubset(active_ids)
            or not active_ids.isdisjoint(terminated_ids)
            or any(item.terminated_at is not None for item in self.active)
            or any(item.terminated_at is None for item in self.terminated)
        ):
            raise ValueError("liquidity cluster update is inconsistent")


@dataclass(frozen=True)
class StructuralRangeState:
    """Frozen structural geometry bound to one explicit structure generation."""

    range_id: str
    structure_generation_id: str
    timeframe: Timeframe
    symbol: str
    instrument_id: int
    direction: Direction
    lower_swing_id: str
    upper_swing_id: str
    lower_bound: float
    upper_bound: float
    started_at: pd.Timestamp
    known_at: pd.Timestamp
    updated_at: pd.Timestamp
    terminated_at: pd.Timestamp | None = None
    termination_reason: str | None = None
    supersedes_range_id: str | None = None
    foundation_version: str = FOUNDATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "direction", Direction(self.direction))
        for name in ("started_at", "known_at", "updated_at", "terminated_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"structural_range.{name}"),
                )
        if (
            not self.range_id
            or not self.structure_generation_id
            or not self.symbol
            or type(self.instrument_id) is not int
            or self.instrument_id < 0
            or not self.lower_swing_id
            or not self.upper_swing_id
            or self.lower_swing_id == self.upper_swing_id
            or not math.isfinite(float(self.lower_bound))
            or not math.isfinite(float(self.upper_bound))
            or not 0.0 < self.lower_bound < self.upper_bound
            or self.started_at > self.known_at
            or self.known_at > self.updated_at
            or ((self.terminated_at is None) != (self.termination_reason is None))
            or (
                self.terminated_at is not None
                and self.terminated_at < self.updated_at
            )
            or self.supersedes_range_id == self.range_id
            or self.foundation_version != FOUNDATION_VERSION
        ):
            raise ValueError("structural range state is invalid")

    @property
    def generation_id(self) -> str:
        return self.range_id


# The Group-4 implementation already is the registered Mature Balance Range.
# Rebinding the semantic name must not fork or copy that detector/state model.
BalanceRangeState = DealingRangeState


@dataclass(frozen=True)
class DualRangeLocation:
    structural_range_id: str | None
    balance_range_id: str | None
    x_structural_range: float | None
    x_balance_range: float | None

    def __post_init__(self) -> None:
        pairs = (
            (self.structural_range_id, self.x_structural_range),
            (self.balance_range_id, self.x_balance_range),
        )
        if any(
            (identity is None) != (location is None)
            or (
                location is not None
                and not math.isfinite(float(location))
            )
            for identity, location in pairs
        ):
            raise ValueError("dual range location is invalid")


@dataclass(frozen=True)
class PriceZoneView:
    zone_id: str
    lower: float
    upper: float
    lifecycle: str
    direction: Direction | None = None

    def __post_init__(self) -> None:
        if (
            not self.zone_id
            or not self.lifecycle
            or not math.isfinite(float(self.lower))
            or not math.isfinite(float(self.upper))
            or not 0.0 < self.lower <= self.upper
        ):
            raise ValueError("market-state zone view is invalid")


@dataclass(frozen=True)
class DOLCandidateView:
    candidate_id: str
    timeframe: Timeframe
    side: str
    price: float
    source_kind: str
    rank: str = "internal"
    strength: float = 0.0
    lifecycle: str = "visible"
    age_bars: int = 0
    distance_atr: float | None = None
    source_event_id: str | None = None
    path_obstacle_ids: tuple[str, ...] = ()
    range_role: LiquidityRangeRole = LiquidityRangeRole.UNRESOLVED
    normalized_location_in_range: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(
            self, "path_obstacle_ids", tuple(self.path_obstacle_ids)
        )
        object.__setattr__(
            self, "range_role", LiquidityRangeRole(self.range_role)
        )
        if (
            not self.candidate_id
            or self.side not in {"above", "below"}
            or not math.isfinite(float(self.price))
            or self.price <= 0.0
            or not self.source_kind
            or not self.rank
            or not math.isfinite(float(self.strength))
            or not 0.0 <= float(self.strength) <= 1.0
            or not self.lifecycle
            or type(self.age_bars) is not int
            or self.age_bars < 0
            or (
                self.distance_atr is not None
                and not math.isfinite(float(self.distance_atr))
            )
            or (
                self.normalized_location_in_range is not None
                and not math.isfinite(
                    float(self.normalized_location_in_range)
                )
            )
            or (
                (self.range_role is LiquidityRangeRole.UNRESOLVED)
                != (self.normalized_location_in_range is None)
            )
        ):
            raise ValueError("DOL candidate view is invalid")


@dataclass(frozen=True)
class TimeframeStructureState:
    external_direction: Direction | None
    internal_direction: Direction | None
    protected_low: float | None
    protected_high: float | None
    protected_low_id: str | None
    protected_high_id: str | None
    protected_swing_intact: bool | None
    last_bos: float | None
    last_bos_direction: Direction | None
    last_mss: float | None
    last_mss_direction: Direction | None
    # The swing ID identifies the price-structure node; the assignment event
    # ID identifies the current protection generation.  Both are required to
    # invalidate canonical protection without confusing a stale re-assignment
    # of the same swing for the live one.
    protected_low_event_id: str | None = None
    protected_high_event_id: str | None = None


@dataclass(frozen=True)
class TimeframeDeliveryState:
    phase: DeliveryPhase
    active_leg_direction: Direction | None
    displacement_score: float | None
    displacement_features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", DeliveryPhase(self.phase))
        features = FrozenDict(
            {str(key): float(value) for key, value in self.displacement_features.items()}
        )
        object.__setattr__(self, "displacement_features", features)
        if (
            self.displacement_score is not None
            and not 0.0 <= float(self.displacement_score) <= 1.0
        ) or any(not math.isfinite(value) for value in features.values()):
            raise ValueError("timeframe delivery features are invalid")


@dataclass(frozen=True)
class TimeframeRangeState:
    range_id: str | None
    low: float | None
    high: float | None
    normalized_location: float | None
    location_label: str | None
    lifecycle: str | None
    source_ids: tuple[str, ...] = ()
    range_kind: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        empty = self.range_id is None
        if not empty and self.range_kind is None:
            object.__setattr__(
                self, "range_kind", "active_dealing_range"
            )
        values = (self.low, self.high, self.normalized_location, self.location_label, self.lifecycle)
        if empty != all(value is None for value in values):
            raise ValueError("timeframe range empty-state contract is invalid")
        if empty and self.range_kind is not None:
            raise ValueError("empty timeframe range cannot have a kind")
        if not empty and (
            not self.range_id
            or self.low is None
            or self.high is None
            or not 0.0 < self.low < self.high
            or self.normalized_location is None
            or not math.isfinite(float(self.normalized_location))
            or self.location_label not in {"discount", "equilibrium", "premium"}
            or not self.lifecycle
            or not self.source_ids
            or self.range_kind not in {
                "active_dealing_range",
                "descriptive_swing_envelope",
            }
        ):
            raise ValueError("timeframe range state is invalid")


@dataclass(frozen=True)
class TimeframeLiquidityState:
    unswept_bsl: tuple[float, ...]
    unswept_ssl: tuple[float, ...]
    candidate_bsl_ids: tuple[str, ...]
    candidate_ssl_ids: tuple[str, ...]
    recently_swept_ids: tuple[str, ...] = ()
    candidates: tuple[DOLCandidateView, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "unswept_bsl",
            "unswept_ssl",
            "candidate_bsl_ids",
            "candidate_ssl_ids",
            "recently_swept_ids",
            "candidates",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (
            len(self.unswept_bsl) != len(self.candidate_bsl_ids)
            or len(self.unswept_ssl) != len(self.candidate_ssl_ids)
            or any(price <= 0.0 for price in (*self.unswept_bsl, *self.unswept_ssl))
        ):
            raise ValueError("timeframe liquidity inventory is invalid")


@dataclass(frozen=True)
class TimeframeZoneState:
    active_fvg: tuple[PriceZoneView, ...] = ()
    active_ob: tuple[PriceZoneView, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_fvg", tuple(self.active_fvg))
        object.__setattr__(self, "active_ob", tuple(self.active_ob))


@dataclass(frozen=True)
class TimeframeQualityState:
    data_complete: bool
    semantic_version: str
    known_at: pd.Timestamp
    anomaly_tags: tuple[str, ...] = ()
    source_cutoff: pd.Timestamp | None = None
    last_transition_known_at: pd.Timestamp | None = None
    atr: float | None = None
    semantic_registry_identity: str | None = None
    last_event_id: str | None = None
    last_sequence_no: int | None = None
    events_applied: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "known_at",
            aware_timestamp(self.known_at, name="timeframe_quality.known_at"),
        )
        object.__setattr__(self, "anomaly_tags", tuple(self.anomaly_tags))
        for name in ("source_cutoff", "last_transition_known_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"timeframe_quality.{name}"),
                )
        if type(self.data_complete) is not bool or not self.semantic_version:
            raise ValueError("timeframe quality state is invalid")
        if (
            self.source_cutoff is not None
            and self.source_cutoff > self.known_at
        ) or (
            self.last_transition_known_at is not None
            and self.last_transition_known_at > self.known_at
        ) or (
            self.atr is not None
            and (
                not math.isfinite(float(self.atr))
                or self.atr < 0.0
            )
        ) or (
            self.last_sequence_no is not None
            and (
                type(self.last_sequence_no) is not int
                or self.last_sequence_no < 0
            )
        ) or type(self.events_applied) is not int or self.events_applied < 0:
            raise ValueError("timeframe quality clocks or metrics are invalid")


@dataclass(frozen=True)
class TimeframeState:
    timeframe: Timeframe
    structure: TimeframeStructureState
    delivery: TimeframeDeliveryState
    range: TimeframeRangeState
    liquidity: TimeframeLiquidityState
    zones: TimeframeZoneState
    quality: TimeframeQualityState
    structural_legs: tuple[StructuralLegState, ...] = ()
    swing_hierarchy: tuple[SwingHierarchyView, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "structural_legs", tuple(self.structural_legs))
        object.__setattr__(self, "swing_hierarchy", tuple(self.swing_hierarchy))
        if any(leg.timeframe is not self.timeframe for leg in self.structural_legs):
            raise ValueError("timeframe state contains a foreign structural leg")
        if any(
            candidate.timeframe is not self.timeframe
            for candidate in self.liquidity.candidates
        ):
            raise ValueError("timeframe state contains a foreign DOL candidate")
        if any(
            swing.timeframe is not self.timeframe
            for swing in self.swing_hierarchy
        ) or len({swing.swing_id for swing in self.swing_hierarchy}) != len(
            self.swing_hierarchy
        ) or any(
            assignment.assigned_at > self.quality.known_at
            for swing in self.swing_hierarchy
            for assignment in swing.assignments
        ):
            raise ValueError("timeframe state contains an invalid swing hierarchy")

    @property
    def label(self) -> str:
        direction = (
            self.structure.internal_direction.value
            if self.structure.internal_direction is not None
            else "unresolved"
        )
        return f"{self.timeframe.value} {direction} {self.delivery.phase.value}"


@dataclass(frozen=True)
class RelationState:
    relation_id: str
    parent_tf: Timeframe
    child_tf: Timeframe
    role: RelationRole
    parent_direction: Direction | None
    child_direction: Direction | None
    parent_protected_swing_intact: bool | None
    parent_state_invalidated: bool
    reversal_warning: bool
    child_location_in_parent_range: float | None
    parent_invalidation_distance_atr: float | None
    aligned_target_candidate_ids: tuple[str, ...] = ()
    child_mss_against_parent: bool = False
    parent_protected_swing_under_attack: bool = False
    known_at: pd.Timestamp | None = None
    parent_source_cutoff: pd.Timestamp | None = None
    child_source_cutoff: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_tf", Timeframe(self.parent_tf))
        object.__setattr__(self, "child_tf", Timeframe(self.child_tf))
        object.__setattr__(self, "role", RelationRole(self.role))
        object.__setattr__(
            self,
            "aligned_target_candidate_ids",
            tuple(self.aligned_target_candidate_ids),
        )
        for name in (
            "known_at",
            "parent_source_cutoff",
            "child_source_cutoff",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            object.__setattr__(
                self,
                name,
                aware_timestamp(value, name=f"relation.{name}"),
            )
        if (
            not self.relation_id
            or self.parent_tf is self.child_tf
            or type(self.parent_state_invalidated) is not bool
            or type(self.reversal_warning) is not bool
            or type(self.child_mss_against_parent) is not bool
            or type(self.parent_protected_swing_under_attack) is not bool
            or (
                self.child_location_in_parent_range is not None
                and not math.isfinite(
                    float(self.child_location_in_parent_range)
                )
            )
            or (
                self.parent_invalidation_distance_atr is not None
                and not math.isfinite(self.parent_invalidation_distance_atr)
            )
        ):
            raise ValueError("cross-timeframe relation state is invalid")
        if self.known_at is not None and any(
            value is not None and value > self.known_at
            for value in (
                self.parent_source_cutoff,
                self.child_source_cutoff,
            )
        ):
            raise ValueError("relation source cutoff exceeds its as-of clock")


@dataclass(frozen=True)
class SessionState:
    session_id: str
    name: str
    phase: str
    session_open: float
    session_high: float
    session_low: float
    opening_range_high: float | None
    opening_range_low: float | None
    prior_day_high: float | None
    prior_day_low: float | None
    overnight_high: float | None
    overnight_low: float | None
    elapsed_minutes: int
    realized_volatility: float
    relative_volume: float | None
    known_at: pd.Timestamp
    data_complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "known_at",
            aware_timestamp(self.known_at, name="session.known_at"),
        )
        prices = (
            self.session_open,
            self.session_high,
            self.session_low,
            self.opening_range_high,
            self.opening_range_low,
            self.prior_day_high,
            self.prior_day_low,
            self.overnight_high,
            self.overnight_low,
        )
        if (
            not self.session_id
            or not self.name
            or not self.phase
            or any(
                value is not None
                and (not math.isfinite(float(value)) or float(value) <= 0.0)
                for value in prices
            )
            or self.session_low > self.session_high
            or type(self.elapsed_minutes) is not int
            or self.elapsed_minutes < 1
            or not math.isfinite(float(self.realized_volatility))
            or self.realized_volatility < 0.0
            or (
                self.relative_volume is not None
                and (
                    not math.isfinite(float(self.relative_volume))
                    or self.relative_volume < 0.0
                )
            )
            or type(self.data_complete) is not bool
        ):
            raise ValueError("session state is invalid")


@dataclass(frozen=True)
class MarketSnapshot:
    asof: pd.Timestamp
    symbol: str
    instrument_id: int
    price: float
    semantic_version: str
    semantic_registry_identity: str
    timeframe_states: Mapping[Timeframe, TimeframeState]
    relations: Mapping[str, RelationState]
    session: SessionState
    events_this_update: tuple[MarketEvent, ...]
    labels: tuple[str, ...]
    authority: MarketSnapshotAuthority = (
        MarketSnapshotAuthority.LEGACY_UNSPECIFIED
    )
    schema_version: int = MARKET_SNAPSHOT_SCHEMA_VERSION
    foundation: "FoundationProjection | None" = None
    foundation_range_locations: Mapping[
        Timeframe, DualRangeLocation
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "asof",
            aware_timestamp(self.asof, name="market_snapshot.asof"),
        )
        states = FrozenDict(self.timeframe_states)
        relations = FrozenDict(self.relations)
        object.__setattr__(self, "timeframe_states", states)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "events_this_update", tuple(self.events_this_update))
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(
            self,
            "authority",
            MarketSnapshotAuthority(self.authority),
        )
        foundation_locations = FrozenDict(self.foundation_range_locations)
        object.__setattr__(
            self,
            "foundation_range_locations",
            foundation_locations,
        )
        foundation_invalid = False
        if self.foundation is not None:
            # Local import avoids semantic_foundation -> market_state import
            # inversion while retaining a strict runtime type contract.
            from .semantic_foundation import FoundationProjection

            foundation_invalid = (
                not isinstance(self.foundation, FoundationProjection)
                or self.foundation.asof is None
                or self.foundation.asof > self.asof
                or self.foundation.foundation_version != FOUNDATION_VERSION
                or self.foundation.registry_identity
                != FOUNDATION_CANONICAL_IDENTITY
            )
        if (
            not self.symbol
            or self.schema_version != MARKET_SNAPSHOT_SCHEMA_VERSION
            or self.instrument_id < 0
            or not math.isfinite(float(self.price))
            or not self.semantic_version
            or not self.semantic_registry_identity
            or set(states) != {state.timeframe for state in states.values()}
            or any(relation.relation_id != key for key, relation in relations.items())
            or self.session.known_at != self.asof
            or any(event.known_at > self.asof for event in self.events_this_update)
            or foundation_invalid
            or any(
                not isinstance(timeframe, Timeframe)
                or timeframe not in states
                or not isinstance(location, DualRangeLocation)
                for timeframe, location in foundation_locations.items()
            )
            or (self.foundation is None and bool(foundation_locations))
        ):
            raise ValueError("market snapshot identity or causal clock is invalid")

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["schema_version"] = MARKET_SNAPSHOT_SCHEMA_VERSION
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        expected = {
            "asof",
            "symbol",
            "instrument_id",
            "price",
            "semantic_version",
            "semantic_registry_identity",
            "timeframe_states",
            "relations",
            "session",
            "events_this_update",
            "labels",
            "authority",
            "schema_version",
            "foundation",
            "foundation_range_locations",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != expected
            or state.get("schema_version") != MARKET_SNAPSHOT_SCHEMA_VERSION
        ):
            raise ValueError("market snapshot pickle schema changed")
        for name, value in state.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    @property
    def fingerprint(self) -> str:
        snapshot_fields = dict(self.__dict__)
        snapshot_fields.pop("foundation", None)
        primitive = dict(to_primitive(snapshot_fields))
        if self.foundation is None:
            # Preserve historical snapshot identities when no v2 projection
            # transport is present.
            primitive.pop("foundation", None)
            primitive.pop("foundation_range_locations", None)
        else:
            primitive["foundation"] = dict(
                self.foundation.identity_payload()
            )
        payload = json.dumps(
            primitive,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def replay_payload(self) -> Mapping[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "timeframes": {
                timeframe.value: to_primitive(state)
                for timeframe, state in self.timeframe_states.items()
            },
            "relations": {
                key: to_primitive(state)
                for key, state in self.relations.items()
            },
            "session": to_primitive(self.session),
            "authority": self.authority.value,
        }
        if self.foundation is not None:
            payload["foundation"] = dict(
                self.foundation.transport_payload()
            )
            payload["foundation_range_locations"] = {
                timeframe.value: to_primitive(location)
                for timeframe, location
                in self.foundation_range_locations.items()
            }
        return FrozenDict(payload)


@dataclass(frozen=True)
class HierarchicalReplayState:
    timeframe_states: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    relations: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    session: Mapping[str, Any] | None = None
    known_at: pd.Timestamp | None = None


_TIMEFRAME_REDUCER_KINDS = frozenset(
    {
        EventKind.BAR_COMPLETED,
        EventKind.SWING_CONFIRMED,
        EventKind.STRUCTURAL_LEG_CREATED,
        EventKind.LIQUIDITY_LEVEL_CREATED,
        EventKind.LEVEL_TOUCHED,
        EventKind.LEVEL_PENETRATED,
        EventKind.SWEEP_CONFIRMED,
        EventKind.ACCEPTANCE_CONFIRMED,
        EventKind.DISPLACEMENT_OBSERVED,
        EventKind.FVG_CREATED,
        EventKind.FVG_PARTIALLY_FILLED,
        EventKind.FVG_MIDPOINT_TOUCHED,
        EventKind.FVG_FULLY_FILLED,
        EventKind.FVG_INVALIDATED,
        EventKind.FVG_EXPIRED,
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        EventKind.QUALIFIED_BOS,
        EventKind.PROTECTED_SWING_ASSIGNED,
        EventKind.MSS_CORE_CONFIRMED,
        EventKind.DEALING_RANGE_CREATED,
        EventKind.DEALING_RANGE_ACTIVATED,
        EventKind.DEALING_RANGE_EXTENDED,
        EventKind.DEALING_RANGE_INVALIDATED,
        EventKind.DEALING_RANGE_REPLACED,
        EventKind.ORIGIN_ZONE_CREATED,
        EventKind.ORIGIN_ZONE_TOUCHED,
        EventKind.ORIGIN_ZONE_MITIGATED,
        EventKind.ORIGIN_ZONE_INVALIDATED,
    }
)

_STRUCTURE_CHANGE_KINDS = frozenset(
    {
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        EventKind.QUALIFIED_BOS,
        EventKind.PROTECTED_SWING_ASSIGNED,
        EventKind.MSS_CORE_CONFIRMED,
        EventKind.ACCEPTANCE_CONFIRMED,
    }
)


def _empty_timeframe_state(
    event: MarketEvent,
    *,
    semantic_registry_identity: str | None,
) -> TimeframeState:
    return TimeframeState(
        timeframe=event.timeframe,
        structure=TimeframeStructureState(
            external_direction=None,
            internal_direction=None,
            protected_low=None,
            protected_high=None,
            protected_low_id=None,
            protected_high_id=None,
            protected_swing_intact=None,
            last_bos=None,
            last_bos_direction=None,
            last_mss=None,
            last_mss_direction=None,
        ),
        delivery=TimeframeDeliveryState(
            phase=DeliveryPhase.TRANSITION,
            active_leg_direction=None,
            displacement_score=None,
        ),
        range=TimeframeRangeState(
            None, None, None, None, None, None
        ),
        liquidity=TimeframeLiquidityState((), (), (), ()),
        zones=TimeframeZoneState(),
        quality=TimeframeQualityState(
            data_complete=False,
            semantic_version=event.semantic_version,
            known_at=event.known_at,
            semantic_registry_identity=semantic_registry_identity,
        ),
    )


def _range_at_price(
    state: TimeframeRangeState,
    price: float,
) -> TimeframeRangeState:
    if state.range_id is None or state.low is None or state.high is None:
        return state
    location = (float(price) - state.low) / (state.high - state.low)
    label = (
        "discount"
        if location < 0.5
        else "premium"
        if location > 0.5
        else "equilibrium"
    )
    return replace(
        state,
        normalized_location=location,
        location_label=label,
    )


def _delivery_phase(
    structure: TimeframeStructureState,
    range_state: TimeframeRangeState,
    active_leg: Direction | None,
) -> DeliveryPhase:
    external = structure.external_direction
    internal = structure.internal_direction
    if external is None:
        return (
            DeliveryPhase.BALANCE
            if range_state.range_id is not None
            and range_state.range_kind == "active_dealing_range"
            and range_state.lifecycle not in {"broken", "invalidated"}
            else DeliveryPhase.TRANSITION
        )
    if (
        internal is not None
        and internal is not external
        and structure.last_mss_direction is internal
    ):
        return DeliveryPhase.REVERSAL_ATTEMPT
    if active_leg is external:
        return DeliveryPhase.EXPANSION
    if active_leg is not None and structure.protected_swing_intact:
        return DeliveryPhase.RETRACEMENT
    return DeliveryPhase.TRANSITION


def _live_protected_assignment_direction(
    structure: TimeframeStructureState,
) -> Direction | None:
    """Return one valid live assignment direction, failing closed if corrupt."""

    if structure.protected_swing_intact is not True:
        return None
    long_live = bool(
        structure.protected_low is not None
        and structure.protected_low_id
        and structure.protected_low_event_id
        and structure.protected_high is None
        and structure.protected_high_id is None
        and structure.protected_high_event_id is None
    )
    short_live = bool(
        structure.protected_high is not None
        and structure.protected_high_id
        and structure.protected_high_event_id
        and structure.protected_low is None
        and structure.protected_low_id is None
        and structure.protected_low_event_id is None
    )
    if long_live == short_live:
        raise ValueError(
            "intact protected swing requires one complete unique assignment"
        )
    direction = Direction.LONG if long_live else Direction.SHORT
    if structure.external_direction is not direction:
        raise ValueError(
            "intact protected swing disagrees with external direction"
        )
    return direction


def _liquidity_state(
    candidates: Iterable[DOLCandidateView],
    recently_swept_ids: Iterable[str],
) -> TimeframeLiquidityState:
    values = tuple(
        sorted(candidates, key=lambda item: (item.price, item.candidate_id))
    )
    above = tuple(item for item in values if item.side == "above")
    below = tuple(item for item in values if item.side == "below")
    return TimeframeLiquidityState(
        unswept_bsl=tuple(item.price for item in above),
        unswept_ssl=tuple(item.price for item in below),
        candidate_bsl_ids=tuple(item.candidate_id for item in above),
        candidate_ssl_ids=tuple(item.candidate_id for item in below),
        recently_swept_ids=tuple(dict.fromkeys(recently_swept_ids)),
        candidates=values,
    )


def _candidate_range_membership(
    candidate: DOLCandidateView,
    range_state: TimeframeRangeState,
) -> DOLCandidateView:
    """Classify a same-timeframe candidate against one active range."""

    active = bool(
        range_state.range_id is not None
        and range_state.range_kind == "active_dealing_range"
        # ``mature`` is the compatibility projection spelling of the
        # canonical atomic lifecycle ``active``.
        and range_state.lifecycle in {"active", "mature"}
        and range_state.low is not None
        and range_state.high is not None
    )
    if not active:
        return replace(
            candidate,
            range_role=LiquidityRangeRole.UNRESOLVED,
            normalized_location_in_range=None,
        )
    low = float(range_state.low)
    high = float(range_state.high)
    location = (float(candidate.price) - low) / (high - low)
    return replace(
        candidate,
        range_role=(
            LiquidityRangeRole.IRL
            if low < float(candidate.price) < high
            else LiquidityRangeRole.ERL
        ),
        normalized_location_in_range=location,
    )


def _liquidity_for_range(
    liquidity: TimeframeLiquidityState,
    range_state: TimeframeRangeState,
) -> TimeframeLiquidityState:
    """Apply the shared atomic/compatibility IRL/ERL projection."""

    return _liquidity_state(
        (
            _candidate_range_membership(candidate, range_state)
            for candidate in liquidity.candidates
        ),
        liquidity.recently_swept_ids,
    )


def _swing_hierarchy_targets(
    event: MarketEvent,
) -> tuple[SwingRank, tuple[str, ...]] | None:
    evidence = event.evidence
    if event.kind is EventKind.SWING_CONFIRMED:
        raw_id = evidence.get("source_entity_id")
        if raw_id is None and event.source_entity_ids:
            raw_id = event.source_entity_ids[0]
        targets = () if raw_id is None else (str(raw_id),)
        rank = SwingRank.MICRO
    elif event.kind is EventKind.STRUCTURAL_LEG_CREATED:
        targets = tuple(
            str(value)
            for value in (
                evidence.get("start_swing_id"),
                evidence.get("end_swing_id"),
            )
            if value is not None
        )
        rank = SwingRank.INTERNAL
    elif event.kind is EventKind.STRUCTURE_DIRECTION_CONFIRMED:
        targets = tuple(
            str(value)
            for value in (
                evidence.get("source_high_id"),
                evidence.get("source_low_id"),
            )
            if value is not None
        )
        rank = SwingRank.STRUCTURAL
    elif event.kind is EventKind.PROTECTED_SWING_ASSIGNED:
        raw_id = evidence.get("protected_swing_id")
        targets = () if raw_id is None else (str(raw_id),)
        rank = SwingRank.EXTERNAL
    else:
        return None
    if event.origin is EventOrigin.SEMANTIC_ATOMIC and not targets:
        raise ValueError(
            f"{event.kind.value} lacks its swing hierarchy identity"
        )
    return rank, tuple(dict.fromkeys(targets))


def _swing_hierarchy_transition(
    hierarchy: tuple[SwingHierarchyView, ...],
    event: MarketEvent,
) -> tuple[SwingHierarchyView, ...]:
    transition = _swing_hierarchy_targets(event)
    if transition is None:
        return hierarchy
    rank, targets = transition
    by_id = {item.swing_id: item for item in hierarchy}
    assignment = SwingRankAssignment(
        rank=rank,
        assigned_at=event.known_at,
        assignment_event_id=event.event_id,
        source_kind=event.kind.value,
        source_event_ids=event.source_event_ids,
        context_event_ids=event.context_event_ids,
    )
    for swing_id in targets:
        current = by_id.get(swing_id)
        if current is None:
            if rank is not SwingRank.MICRO:
                if event.origin is EventOrigin.SEMANTIC_ATOMIC:
                    raise ValueError(
                        f"{event.kind.value} references an unknown confirmed "
                        f"swing: {swing_id}"
                    )
                continue
            by_id[swing_id] = SwingHierarchyView(
                swing_id=swing_id,
                timeframe=event.timeframe,
                semantic_rank=rank,
                nesting_depth=_SWING_RANK_DEPTH[rank],
                assignments=(assignment,),
            )
            continue
        if any(
            item.assignment_event_id == event.event_id
            for item in current.assignments
        ):
            continue
        assignments = (*current.assignments, assignment)
        maximum = max(
            assignments, key=lambda item: _SWING_RANK_DEPTH[item.rank]
        ).rank
        by_id[swing_id] = replace(
            current,
            semantic_rank=maximum,
            nesting_depth=_SWING_RANK_DEPTH[maximum],
            assignments=assignments,
        )
    return tuple(sorted(by_id.values(), key=lambda item: item.swing_id))


def _liquidity_for_swing_hierarchy(
    liquidity: TimeframeLiquidityState,
    hierarchy: tuple[SwingHierarchyView, ...],
) -> TimeframeLiquidityState:
    ranks = {item.swing_id: item.semantic_rank.value for item in hierarchy}
    if not ranks:
        return liquidity
    return _liquidity_state(
        (
            replace(
                candidate,
                rank=ranks.get(
                    candidate.candidate_id.removeprefix("swing:"),
                    candidate.rank,
                ),
            )
            if candidate.source_kind == "confirmed_swing"
            else candidate
            for candidate in liquidity.candidates
        ),
        liquidity.recently_swept_ids,
    )


def _zone_transition(
    zones: tuple[PriceZoneView, ...],
    event: MarketEvent,
    *,
    identity_key: str,
    lifecycle: str,
    terminal: bool,
) -> tuple[PriceZoneView, ...]:
    identity = str(event.evidence.get(identity_key, ""))
    current = tuple(item for item in zones if item.zone_id == identity)
    retained = tuple(item for item in zones if item.zone_id != identity)
    if terminal:
        if (
            identity_key == "origin_zone_id"
            and (not identity or len(current) != 1)
        ):
            raise ValueError(
                f"{event.kind.value} references an unknown active zone"
            )
        return retained
    if not identity or event.zone is None:
        raise ValueError(f"{event.kind.value} lacks frozen zone geometry")
    return tuple(
        sorted(
            (
                *retained,
                PriceZoneView(
                    identity,
                    float(event.zone[0]),
                    float(event.zone[1]),
                    lifecycle,
                    event.direction,
                ),
            ),
            key=lambda item: item.zone_id,
        )
    )


def _leg_from_event(event: MarketEvent) -> StructuralLegState:
    evidence = event.evidence
    return StructuralLegState(
        leg_id=str(evidence["leg_id"]),
        timeframe=event.timeframe,
        direction=event.direction,
        start_swing_id=str(evidence["start_swing_id"]),
        end_swing_id=str(evidence["end_swing_id"]),
        start_event_time=pd.Timestamp(evidence["start_event_time"]),
        end_event_time=pd.Timestamp(evidence["end_event_time"]),
        known_at=event.known_at,
        start_price=float(evidence["start_price"]),
        end_price=float(evidence["end_price"]),
        start_close=float(evidence["start_close"]),
        end_close=float(evidence["end_close"]),
        amplitude_points=float(evidence["amplitude_points"]),
        amplitude_atr=float(evidence["amplitude_atr"]),
        duration_bars=int(evidence["duration_bars"]),
        duration_minutes=int(evidence["duration_minutes"]),
        efficiency=float(evidence["efficiency"]),
        max_retracement_points=float(
            evidence["max_retracement_points"]
        ),
        max_retracement_atr=float(evidence["max_retracement_atr"]),
        rank=SwingRank(str(evidence.get("rank", "internal"))),
        source_swing_ids=(
            str(evidence["start_swing_id"]),
            str(evidence["end_swing_id"]),
        ),
    )


def reduce_timeframe_state(
    state: TimeframeState | None,
    event: MarketEvent,
    *,
    semantic_registry_identity: str | None = None,
    _m1_owner_fanout: bool = False,
) -> TimeframeState | None:
    """Purely reduce one timeframe from normalized and semantic events.

    Projection events and legacy lifecycle transport are intentionally
    ignored.  Consequently this reducer can be replayed after all
    ``*_STATE_CHANGED`` events have been removed from the journal.
    """

    if event.kind not in _TIMEFRAME_REDUCER_KINDS:
        return state
    if type(_m1_owner_fanout) is not bool:
        raise ValueError("M1 owner fanout authority must be boolean")
    clock_only_heartbeat = False
    native_htf_clock_only = False
    if event.kind is EventKind.BAR_COMPLETED:
        real_completed, clock_only = validate_completed_bar_header(
            event.evidence
        )
        if not real_completed:
            clock_only_heartbeat = bool(
                event.timeframe is Timeframe.M1
                or _m1_owner_fanout
            )
            native_htf_clock_only = not clock_only_heartbeat
    if native_htf_clock_only and state is None:
        # A native HTF clock root cannot create semantic owner state.
        return None
    if state is None:
        state = _empty_timeframe_state(
            event,
            semantic_registry_identity=semantic_registry_identity,
        )
    if event.timeframe is not state.timeframe:
        raise ValueError("timeframe reducer received a foreign event")
    if event.semantic_version != state.quality.semantic_version:
        raise ValueError("timeframe reducer cannot mix semantic versions")
    prior_order = (
        state.quality.known_at,
        -1
        if state.quality.last_sequence_no is None
        else state.quality.last_sequence_no,
        state.quality.last_event_id or "",
    )
    current_order = (event.known_at, event.sequence_no, event.event_id)
    if current_order <= prior_order:
        raise ValueError("timeframe reducer received out-of-order knowledge")
    if native_htf_clock_only:
        # Native HTF clock roots are audit/provenance only, after the direct
        # reducer's foreign/version/order contracts have been enforced.
        return state
    if clock_only_heartbeat:
        return replace(
            state,
            quality=replace(
                state.quality,
                known_at=event.known_at,
                semantic_registry_identity=(
                    state.quality.semantic_registry_identity
                    or semantic_registry_identity
                ),
                last_event_id=event.event_id,
                last_sequence_no=event.sequence_no,
                events_applied=state.quality.events_applied + 1,
            ),
        )

    structure = state.structure
    live_protected_direction = (
        _live_protected_assignment_direction(structure)
        if event.kind in _STRUCTURE_CHANGE_KINDS
        else None
    )
    delivery = state.delivery
    range_state = state.range
    liquidity = state.liquidity
    zones = state.zones
    legs = state.structural_legs
    hierarchy = _swing_hierarchy_transition(state.swing_hierarchy, event)
    source_cutoff = state.quality.source_cutoff
    data_complete = state.quality.data_complete
    atr = state.quality.atr

    if event.kind is EventKind.BAR_COMPLETED:
        close = float(event.evidence.get("close", event.price))
        price_update_only = _m1_owner_fanout
        if not price_update_only:
            raw_atr = float(event.evidence.get("atr", 0.0))
            atr = raw_atr if raw_atr > 0.0 else atr
            source_cutoff = event.known_at
            data_complete = bool(
                event.evidence.get("data_complete", False)
            )
        range_state = _range_at_price(range_state, close)
        candidates = tuple(
            replace(
                item,
                age_bars=item.age_bars + 1,
                distance_atr=(
                    None
                    if atr is None or atr <= 0.0
                    else (
                        (item.price - close) / atr
                        if item.side == "above"
                        else (close - item.price) / atr
                    )
                ),
            )
            for item in liquidity.candidates
        )
        liquidity = _liquidity_state(
            candidates, liquidity.recently_swept_ids
        )
    elif event.kind is EventKind.STRUCTURE_DIRECTION_CONFIRMED:
        live_direction = live_protected_direction
        if live_direction is None:
            structure = replace(
                structure,
                external_direction=event.direction,
                internal_direction=event.direction,
                protected_swing_intact=(
                    structure.protected_swing_intact
                    if structure.protected_swing_intact is False
                    else None
                ),
            )
        elif event.direction is live_direction:
            # A same-direction generation may reinforce the regime, but only
            # the exact registered Acceptance may terminalize its protection.
            structure = replace(
                structure,
                external_direction=event.direction,
                internal_direction=event.direction,
            )
        # An opposite generation remains an immutable Eye fact (and may still
        # add hierarchy provenance), but it cannot rewrite the live regime or
        # stand in for an MSS Core transition.
    elif event.kind is EventKind.STRUCTURAL_LEG_CREATED:
        leg = _leg_from_event(event)
        legs = (*legs, leg)[-8:]
        delivery = replace(delivery, active_leg_direction=leg.direction)
        if structure.internal_direction is None:
            structure = replace(structure, internal_direction=leg.direction)
    elif event.kind is EventKind.QUALIFIED_BOS:
        live_direction = live_protected_direction
        if live_direction is not None and event.direction is not live_direction:
            raise ValueError(
                "qualified BOS opposes the live protected external regime"
            )
        structure = replace(
            structure,
            external_direction=event.direction,
            internal_direction=event.direction,
            last_bos=event.price,
            last_bos_direction=event.direction,
        )
    elif event.kind is EventKind.PROTECTED_SWING_ASSIGNED:
        live_direction = live_protected_direction
        if live_direction is not None and event.direction is not live_direction:
            raise ValueError(
                "protected swing assignment opposes the live protected "
                "external regime"
            )
        if event.price is None:
            raise ValueError("protected swing assignment lacks a price")
        protected_price = float(event.price)
        if (
            live_direction is Direction.LONG
            and structure.protected_low is not None
            and protected_price < float(structure.protected_low)
        ) or (
            live_direction is Direction.SHORT
            and structure.protected_high is not None
            and protected_price > float(structure.protected_high)
        ):
            raise ValueError(
                "protected swing assignment cannot loosen live protection"
            )
        # Equality is non-loosening and may name a new assignment generation;
        # exact retries remain governed by the owner's event-id contract.
        protected_id = str(event.evidence["protected_swing_id"])
        if event.direction is Direction.LONG:
            structure = replace(
                structure,
                protected_low=protected_price,
                protected_low_id=protected_id,
                protected_high=None,
                protected_high_id=None,
                protected_swing_intact=True,
                protected_low_event_id=event.event_id,
                protected_high_event_id=None,
            )
        else:
            structure = replace(
                structure,
                protected_high=protected_price,
                protected_high_id=protected_id,
                protected_low=None,
                protected_low_id=None,
                protected_swing_intact=True,
                protected_high_event_id=event.event_id,
                protected_low_event_id=None,
            )
    elif event.kind is EventKind.MSS_CORE_CONFIRMED:
        structure = replace(
            structure,
            internal_direction=event.direction,
            last_mss=event.price,
            last_mss_direction=event.direction,
        )
    elif event.kind is EventKind.ACCEPTANCE_CONFIRMED:
        protected_id = (
            structure.protected_low_id
            if live_protected_direction is Direction.LONG
            else structure.protected_high_id
            if live_protected_direction is Direction.SHORT
            else None
        )
        opposite = (
            Direction.SHORT
            if live_protected_direction is Direction.LONG
            else Direction.LONG
            if live_protected_direction is Direction.SHORT
            else None
        )
        accepted_protected_id = event.evidence.get("protected_swing_id")
        protected_assignment_event_id = event.evidence.get(
            "protected_swing_event_id"
        )
        current_assignment_event_id = (
            structure.protected_low_event_id
            if live_protected_direction is Direction.LONG
            else structure.protected_high_event_id
            if live_protected_direction is Direction.SHORT
            else None
        )
        # Price equality is not an identity relation.  Only the canonical
        # event graph may invalidate protection, and it must point to the live
        # assignment generation as well as the stable swing identity.
        accepted_exact_protection = bool(
            event.origin is EventOrigin.SEMANTIC_ATOMIC
            and structure.protected_swing_intact is True
            and live_protected_direction is not None
            and structure.external_direction is live_protected_direction
            and protected_id is not None
            and accepted_protected_id == protected_id
            and isinstance(protected_assignment_event_id, str)
            and protected_assignment_event_id
            == current_assignment_event_id
            and protected_assignment_event_id in event.context_event_ids
        )
        if event.direction is opposite and accepted_exact_protection:
            structure = replace(
                structure,
                external_direction=None,
                protected_swing_intact=False,
            )
        level_id = str(event.evidence.get("level_id", ""))
        liquidity = _liquidity_state(
            tuple(
                item
                for item in liquidity.candidates
                if item.candidate_id != level_id
            ),
            liquidity.recently_swept_ids,
        )
    elif event.kind is EventKind.DISPLACEMENT_OBSERVED:
        features = {
            str(key): float(value)
            for key, value in event.evidence.get(
                "state_metrics", {}
            ).items()
        }
        components = (
            clamp(features.get("efficiency", event.strength)),
            clamp(features.get("relative_atr", 0.0) / 2.0),
            clamp(features.get("speed_atr_per_bar", 0.0)),
            clamp(features.get("mean_body_fraction", 0.0)),
            clamp(features.get("body_continuity", 0.0)),
            clamp(features.get("mean_directional_clv", 0.0)),
        )
        delivery = replace(
            delivery,
            displacement_score=sum(components) / len(components),
            displacement_features=features,
        )
    elif event.kind is EventKind.LIQUIDITY_LEVEL_CREATED:
        if event.price is None or event.side not in {"above", "below"}:
            raise ValueError("candidate level event lacks side or price")
        candidate_id = str(event.evidence["level_id"])
        source_kind = str(
            event.evidence.get("source_kind", "registered_level")
        )
        replaces_level_id = event.evidence.get("replaces_level_id")
        replaces_level_event_id = event.evidence.get(
            "replaces_level_event_id"
        )
        if (replaces_level_id is None) != (replaces_level_event_id is None):
            raise ValueError("candidate replacement identity is invalid")
        if replaces_level_id is not None:
            if (
                not isinstance(replaces_level_id, str)
                or not replaces_level_id
                or replaces_level_id == candidate_id
                or not isinstance(replaces_level_event_id, str)
                or not replaces_level_event_id
                or replaces_level_event_id not in event.context_event_ids
            ):
                raise ValueError("candidate replacement identity is invalid")
            replaced = tuple(
                item
                for item in liquidity.candidates
                if item.candidate_id == replaces_level_id
            )
            if replaced and (
                replaced[0].source_event_id != replaces_level_event_id
                or replaced[0].source_kind != source_kind
            ):
                raise ValueError(
                    "candidate replacement provenance does not match the "
                    "visible same-kind candidate"
                )
        retained = tuple(
            item
            for item in liquidity.candidates
            if item.candidate_id != candidate_id
            and item.candidate_id != replaces_level_id
        )
        liquidity = _liquidity_state(
            (
                *retained,
                DOLCandidateView(
                    candidate_id=candidate_id,
                    timeframe=event.timeframe,
                    side=event.side,
                    price=float(event.price),
                    source_kind=source_kind,
                    rank=str(
                        event.evidence.get(
                            "rank",
                            event.evidence.get(
                                "structural_rank",
                                event.evidence.get("semantic_rank", "internal"),
                            ),
                        )
                    ),
                    strength=float(event.strength),
                    source_event_id=event.event_id,
                ),
            ),
            liquidity.recently_swept_ids,
        )
    elif event.kind in {
        EventKind.LEVEL_TOUCHED,
        EventKind.LEVEL_PENETRATED,
    }:
        level_id = str(event.evidence.get("level_id", ""))
        lifecycle = (
            "touched"
            if event.kind is EventKind.LEVEL_TOUCHED
            else "penetrated"
        )
        liquidity = _liquidity_state(
            tuple(
                replace(item, lifecycle=lifecycle)
                if item.candidate_id == level_id
                else item
                for item in liquidity.candidates
            ),
            liquidity.recently_swept_ids,
        )
    elif event.kind is EventKind.SWEEP_CONFIRMED:
        level_id = str(event.evidence.get("level_id", ""))
        retained = tuple(
            item
            for item in liquidity.candidates
            if item.candidate_id != level_id
        )
        swept = liquidity.recently_swept_ids
        if level_id:
            swept = (*swept, level_id)[-32:]
        liquidity = _liquidity_state(retained, swept)
    elif event.kind in {
        EventKind.FVG_CREATED,
        EventKind.FVG_PARTIALLY_FILLED,
        EventKind.FVG_MIDPOINT_TOUCHED,
        EventKind.FVG_FULLY_FILLED,
        EventKind.FVG_INVALIDATED,
        EventKind.FVG_EXPIRED,
    }:
        terminal = event.kind in {
            EventKind.FVG_FULLY_FILLED,
            EventKind.FVG_INVALIDATED,
            EventKind.FVG_EXPIRED,
        }
        lifecycle = {
            EventKind.FVG_CREATED: "open",
            EventKind.FVG_PARTIALLY_FILLED: "partial",
            EventKind.FVG_MIDPOINT_TOUCHED: "midpoint_touched",
            EventKind.FVG_FULLY_FILLED: "fully_filled",
            EventKind.FVG_INVALIDATED: "invalidated",
            EventKind.FVG_EXPIRED: "expired",
        }[event.kind]
        zones = replace(
            zones,
            active_fvg=_zone_transition(
                zones.active_fvg,
                event,
                identity_key="fvg_id",
                lifecycle=lifecycle,
                terminal=terminal,
            ),
        )
    elif event.kind in {
        EventKind.ORIGIN_ZONE_CREATED,
        EventKind.ORIGIN_ZONE_TOUCHED,
        EventKind.ORIGIN_ZONE_MITIGATED,
        EventKind.ORIGIN_ZONE_INVALIDATED,
    }:
        terminal = event.kind in {
            EventKind.ORIGIN_ZONE_MITIGATED,
            EventKind.ORIGIN_ZONE_INVALIDATED,
        }
        zones = replace(
            zones,
            active_ob=_zone_transition(
                zones.active_ob,
                event,
                identity_key="origin_zone_id",
                lifecycle=event.kind.value,
                terminal=terminal,
            ),
        )
    elif event.kind in {
        EventKind.DEALING_RANGE_CREATED,
        EventKind.DEALING_RANGE_ACTIVATED,
        EventKind.DEALING_RANGE_EXTENDED,
        EventKind.DEALING_RANGE_REPLACED,
    }:
        if event.zone is None:
            raise ValueError("dealing range event lacks frozen bounds")
        range_id = str(
            event.evidence.get(
                "range_id",
                event.evidence.get("replacement_range_id", ""),
            )
        )
        low, high = (float(event.zone[0]), float(event.zone[1]))
        price = float(event.price or (low + high) / 2.0)
        range_state = TimeframeRangeState(
            range_id=range_id,
            low=low,
            high=high,
            normalized_location=(price - low) / (high - low),
            location_label="equilibrium",
            lifecycle={
                EventKind.DEALING_RANGE_CREATED: "created",
                EventKind.DEALING_RANGE_ACTIVATED: "active",
                EventKind.DEALING_RANGE_EXTENDED: "extended",
                EventKind.DEALING_RANGE_REPLACED: "replaced",
            }[event.kind],
            source_ids=event.source_event_ids or (event.event_id,),
            range_kind="active_dealing_range",
        )
    elif event.kind is EventKind.DEALING_RANGE_INVALIDATED:
        if range_state.range_id == event.evidence.get("range_id"):
            range_state = replace(range_state, lifecycle="invalidated")

    liquidity = _liquidity_for_swing_hierarchy(liquidity, hierarchy)
    liquidity = _liquidity_for_range(liquidity, range_state)
    delivery = replace(
        delivery,
        phase=_delivery_phase(
            structure,
            range_state,
            delivery.active_leg_direction,
        ),
    )
    quality = replace(
        state.quality,
        data_complete=data_complete,
        known_at=event.known_at,
        source_cutoff=source_cutoff,
        last_transition_known_at=(
            state.quality.last_transition_known_at
            if event.kind is EventKind.BAR_COMPLETED
            else event.known_at
        ),
        atr=atr,
        semantic_registry_identity=(
            state.quality.semantic_registry_identity
            or semantic_registry_identity
        ),
        last_event_id=event.event_id,
        last_sequence_no=event.sequence_no,
        events_applied=state.quality.events_applied + 1,
    )
    return TimeframeState(
        timeframe=state.timeframe,
        structure=structure,
        delivery=delivery,
        range=range_state,
        liquidity=liquidity,
        zones=zones,
        quality=quality,
        structural_legs=legs,
        swing_hierarchy=hierarchy,
    )


class TimeframeEventReducer:
    """Small deterministic owner of current per-timeframe event state."""

    def __init__(
        self,
        *,
        semantic_registry_identity: str,
        semantic_version: str = SMC_SEMANTIC_VERSION,
        expected_timeframes: Iterable[Timeframe] | None = None,
    ) -> None:
        if not semantic_registry_identity:
            raise ValueError("timeframe reducer requires definition identity")
        if not semantic_version:
            raise ValueError("timeframe reducer requires semantic version")
        self.semantic_registry_identity = semantic_registry_identity
        self.semantic_version = semantic_version
        self.states: dict[Timeframe, TimeframeState] = {}
        self._last_order_key: tuple[pd.Timestamp, int, str] | None = None
        self._event_ids: set[str] = set()
        self._events_by_id: dict[str, MarketEvent] = {}
        self._normalized_bar_event_ids: dict[
            tuple[Timeframe, pd.Timestamp], str
        ] = {}
        self._terminal_crossing_event_ids: dict[str, str] = {}
        self._latest_protected_assignment_event_ids: dict[str, str] = {}
        self._latest_protected_assignment_event_ids_by_timeframe: dict[
            Timeframe, str
        ] = {}
        self._latest_real_m1_event_id: str | None = None
        self._unresolved_forward_reference_ids: set[str] = set()
        self._expected_timeframes: tuple[Timeframe, ...] | None = None
        if expected_timeframes is not None:
            self.bind_timeframes(expected_timeframes)

    @property
    def expected_timeframes(self) -> tuple[Timeframe, ...] | None:
        return self._expected_timeframes

    def bind_timeframes(
        self,
        timeframes: Iterable[Timeframe],
    ) -> tuple[Timeframe, ...]:
        """Bind one immutable owner registry for deterministic M1 fan-out."""

        if isinstance(timeframes, (str, bytes)):
            raise ValueError("timeframe reducer registry must be a sequence")
        supplied = tuple(Timeframe(value) for value in timeframes)
        if not supplied or len(supplied) != len(set(supplied)):
            raise ValueError("timeframe reducer registry is empty or duplicated")
        supplied_set = set(supplied)
        normalized = tuple(
            timeframe
            for timeframe in _TIMEFRAME_AUTHORITY_ORDER
            if timeframe in supplied_set
        )
        unexpected_states = tuple(
            timeframe
            for timeframe in self.states
            if timeframe not in normalized
        )
        if unexpected_states:
            raise ValueError(
                "timeframe reducer already contains state outside its "
                "registry: "
                + ", ".join(
                    timeframe.value for timeframe in unexpected_states
                )
            )
        if (
            self._expected_timeframes is not None
            and normalized != self._expected_timeframes
        ):
            raise ValueError(
                "timeframe reducer registry changed within one stream"
            )
        if self._expected_timeframes is None:
            self._expected_timeframes = normalized
        return self._expected_timeframes

    def reset(self) -> None:
        self._ensure_provenance_indexes()
        self.states.clear()
        self._last_order_key = None
        self._event_ids.clear()
        self._events_by_id.clear()
        self._normalized_bar_event_ids.clear()
        self._terminal_crossing_event_ids.clear()
        self._latest_protected_assignment_event_ids.clear()
        self._latest_protected_assignment_event_ids_by_timeframe.clear()
        self._latest_real_m1_event_id = None
        self._unresolved_forward_reference_ids.clear()

    def __setstate__(self, state: Mapping[str, object]) -> None:
        """Restore only after revalidating the current-epoch event history."""

        self.__dict__.update(state)
        if not isinstance(getattr(self, "_events_by_id", None), dict):
            raise ValueError("timeframe reducer pickle history is invalid")
        for name in (
            "_normalized_bar_event_ids",
            "_terminal_crossing_event_ids",
            "_latest_protected_assignment_event_ids",
            "_latest_protected_assignment_event_ids_by_timeframe",
            "_latest_real_m1_event_id",
            "_unresolved_forward_reference_ids",
        ):
            self.__dict__.pop(name, None)
        self._ensure_provenance_indexes()

    def _ensure_provenance_indexes(self) -> None:
        """Rebuild derived indexes once for legacy reducer checkpoints."""

        if all(
            name in self.__dict__
            for name in (
                "_normalized_bar_event_ids",
                "_terminal_crossing_event_ids",
                "_latest_protected_assignment_event_ids",
                "_latest_protected_assignment_event_ids_by_timeframe",
                "_latest_real_m1_event_id",
                "_unresolved_forward_reference_ids",
            )
        ):
            return
        self._normalized_bar_event_ids = {}
        self._terminal_crossing_event_ids = {}
        self._latest_protected_assignment_event_ids = {}
        self._latest_protected_assignment_event_ids_by_timeframe = {}
        self._latest_real_m1_event_id = None
        self._unresolved_forward_reference_ids = set()
        available_events: dict[str, MarketEvent] = {}
        for prior in self._events_by_id.values():
            ImmutableEventStore._normalized_bar_identity(
                prior,
                bar_event_ids=self._normalized_bar_event_ids,
            )
            ImmutableEventStore._terminal_crossing_identity(
                prior,
                terminal_event_ids=self._terminal_crossing_event_ids,
            )
            validate_canonical_event(
                prior,
                available_events=available_events,
                normalized_bar_event_ids=self._normalized_bar_event_ids,
                latest_protected_assignment_event_ids=(
                    self._latest_protected_assignment_event_ids
                ),
                latest_protected_assignment_event_ids_by_timeframe=(
                    self._latest_protected_assignment_event_ids_by_timeframe
                ),
                unresolved_forward_reference_ids=(
                    self._unresolved_forward_reference_ids
                ),
            )
            available_events[prior.event_id] = prior
            self._advance_provenance_indexes(
                prior,
                available_event_ids=available_events,
            )

    def _advance_provenance_indexes(
        self,
        event: MarketEvent,
        *,
        available_event_ids: Iterable[str] | None = None,
    ) -> None:
        """Advance the same small authority indexes as ImmutableEventStore."""

        bar_reservation = ImmutableEventStore._normalized_bar_identity(
            event,
            bar_event_ids=self._normalized_bar_event_ids,
        )
        if bar_reservation is not None:
            bar_key, bar_event_id = bar_reservation
            self._normalized_bar_event_ids[bar_key] = bar_event_id
        terminal_reservation = ImmutableEventStore._terminal_crossing_identity(
            event,
            terminal_event_ids=self._terminal_crossing_event_ids,
        )
        if terminal_reservation is not None:
            generation_id, terminal_event_id = terminal_reservation
            self._terminal_crossing_event_ids[
                generation_id
            ] = terminal_event_id
        protected_reservation = (
            ImmutableEventStore._protected_assignment_identity(event)
        )
        if protected_reservation is not None:
            protected_id, assignment_event_id = protected_reservation
            self._latest_protected_assignment_event_ids[
                protected_id
            ] = assignment_event_id
            self._latest_protected_assignment_event_ids_by_timeframe[
                event.timeframe
            ] = assignment_event_id
        if (
            event.kind is EventKind.BAR_COMPLETED
            and event.timeframe is Timeframe.M1
            and event.origin is EventOrigin.NORMALIZED_DATA
            and event.evidence.get("real_completed") is True
            and event.evidence.get("clock_only") is False
        ):
            self._latest_real_m1_event_id = event.event_id
        available = (
            self._events_by_id.keys()
            if available_event_ids is None
            else available_event_ids
        )
        ImmutableEventStore._advance_unresolved_forward_references(
            event,
            available_event_ids=available,
            unresolved_reference_ids=(
                self._unresolved_forward_reference_ids
            ),
        )

    @property
    def latest_real_m1_event(self) -> MarketEvent | None:
        """Return the current epoch's last exact-real normalized M1 root."""

        self._ensure_provenance_indexes()
        if self._latest_real_m1_event_id is None:
            return None
        event = self._events_by_id.get(self._latest_real_m1_event_id)
        if (
            event is None
            or event.kind is not EventKind.BAR_COMPLETED
            or event.timeframe is not Timeframe.M1
            or event.origin is not EventOrigin.NORMALIZED_DATA
            or event.evidence.get("real_completed") is not True
            or event.evidence.get("clock_only") is not False
        ):
            raise ValueError("timeframe reducer latest real M1 index is invalid")
        return event

    def apply(self, event: MarketEvent) -> TimeframeState | None:
        if event.kind is EventKind.FOUNDATION_STATE_CHANGED:
            # Foundation records have their own projection reducer.  They
            # must not perturb timeframe order, provenance, or state indexes.
            return None
        self._ensure_provenance_indexes()
        if event.semantic_version != self.semantic_version:
            raise ValueError("timeframe reducer cannot mix semantic versions")
        order_key = (event.known_at, event.sequence_no, event.event_id)
        if self._last_order_key is not None and order_key <= self._last_order_key:
            raise ValueError("timeframe reducer stream is out of canonical order")
        if event.event_id in self._event_ids:
            raise ValueError("timeframe reducer stream repeats an event id")
        # Reserve read-only before any reducer state mutation. The audit store
        # normally enforces this first, while direct reducer/checkpoint users
        # must retain the same failure atomicity independently.
        ImmutableEventStore._normalized_bar_identity(
            event,
            bar_event_ids=self._normalized_bar_event_ids,
        )
        ImmutableEventStore._terminal_crossing_identity(
            event,
            terminal_event_ids=self._terminal_crossing_event_ids,
        )
        validate_canonical_event(
            event,
            available_events=self._events_by_id,
            normalized_bar_event_ids=self._normalized_bar_event_ids,
            latest_protected_assignment_event_ids=(
                self._latest_protected_assignment_event_ids
            ),
            latest_protected_assignment_event_ids_by_timeframe=(
                self._latest_protected_assignment_event_ids_by_timeframe
            ),
            unresolved_forward_reference_ids=(
                self._unresolved_forward_reference_ids
            ),
        )
        source_timeframe_value = event.evidence.get("source_timeframe")
        source_timeframe = (
            None
            if source_timeframe_value is None
            else Timeframe(str(source_timeframe_value))
        )
        staged_expected_timeframes = self._expected_timeframes
        if event.kind is EventKind.BAR_COMPLETED and event.timeframe is Timeframe.M1:
            declared_timeframes = event.evidence.get("active_timeframes")
            if declared_timeframes is not None:
                if isinstance(declared_timeframes, (str, bytes)):
                    raise ValueError(
                        "timeframe reducer registry must be a sequence"
                    )
                supplied = tuple(
                    Timeframe(value) for value in declared_timeframes
                )
                if not supplied or len(supplied) != len(set(supplied)):
                    raise ValueError(
                        "timeframe reducer registry is empty or duplicated"
                    )
                supplied_set = set(supplied)
                normalized = tuple(
                    timeframe
                    for timeframe in _TIMEFRAME_AUTHORITY_ORDER
                    if timeframe in supplied_set
                )
                unexpected_states = tuple(
                    timeframe
                    for timeframe in self.states
                    if timeframe not in normalized
                )
                if unexpected_states:
                    raise ValueError(
                        "timeframe reducer already contains state outside "
                        "its registry: "
                        + ", ".join(
                            timeframe.value
                            for timeframe in unexpected_states
                        )
                    )
                if (
                    staged_expected_timeframes is not None
                    and normalized != staged_expected_timeframes
                ):
                    raise ValueError(
                        "timeframe reducer registry changed within one stream"
                    )
                staged_expected_timeframes = (
                    normalized
                    if staged_expected_timeframes is None
                    else staged_expected_timeframes
                )
        if (
            staged_expected_timeframes is not None
            and event.kind is not EventKind.MARKET_EPOCH_RESET
            and event.timeframe not in staged_expected_timeframes
        ):
            raise ValueError(
                "timeframe reducer event is outside its bound registry: "
                f"{event.timeframe.value}"
            )
        if event.kind is EventKind.MARKET_EPOCH_RESET:
            self.states = {}
            self._events_by_id.clear()
            self._normalized_bar_event_ids.clear()
            self._terminal_crossing_event_ids.clear()
            self._latest_protected_assignment_event_ids.clear()
            self._latest_protected_assignment_event_ids_by_timeframe.clear()
            self._latest_real_m1_event_id = None
            self._unresolved_forward_reference_ids.clear()
            self._events_by_id[event.event_id] = event
            self._advance_provenance_indexes(event)
            self._expected_timeframes = staged_expected_timeframes
            self._last_order_key = order_key
            self._event_ids.add(event.event_id)
            return None
        staged_states = dict(self.states)
        prior = staged_states.get(event.timeframe)
        current = reduce_timeframe_state(
            prior,
            event,
            semantic_registry_identity=self.semantic_registry_identity,
        )
        if current is not None:
            staged_states[event.timeframe] = current
        if (
            event.kind is EventKind.BAR_COMPLETED
            and event.timeframe is Timeframe.M1
        ):
            owner_timeframes = (
                tuple(staged_states)
                if staged_expected_timeframes is None
                else staged_expected_timeframes
            )
            for owner_timeframe in owner_timeframes:
                if owner_timeframe is Timeframe.M1:
                    continue
                owner_event = replace(
                    event,
                    timeframe=owner_timeframe,
                )
                owner_state = reduce_timeframe_state(
                    staged_states.get(owner_timeframe),
                    owner_event,
                    semantic_registry_identity=(
                        self.semantic_registry_identity
                    ),
                    _m1_owner_fanout=True,
                )
                if owner_state is None:
                    raise RuntimeError(
                        "current-price event was ignored by owner reducer"
                    )
                staged_states[owner_timeframe] = owner_state
        if (
            event.kind
            in {
                EventKind.LEVEL_TOUCHED,
                EventKind.LEVEL_PENETRATED,
                EventKind.SWEEP_CONFIRMED,
                EventKind.ACCEPTANCE_CONFIRMED,
            }
            and source_timeframe is not None
            and source_timeframe is not event.timeframe
            and source_timeframe in staged_states
        ):
            owner_event = replace(event, timeframe=source_timeframe)
            owner_state = reduce_timeframe_state(
                staged_states[source_timeframe],
                owner_event,
                semantic_registry_identity=(
                    self.semantic_registry_identity
                ),
            )
            if owner_state is None:
                raise RuntimeError(
                    "candidate-owner event was ignored by its reducer"
                )
            staged_states[source_timeframe] = owner_state
        self.states = staged_states
        self._expected_timeframes = staged_expected_timeframes
        self._last_order_key = order_key
        self._event_ids.add(event.event_id)
        self._events_by_id[event.event_id] = event
        self._advance_provenance_indexes(event)
        return current

    def replay(self, events: Iterable[MarketEvent]) -> Mapping[Timeframe, TimeframeState]:
        self.reset()
        for event in events:
            self.apply(event)
        return FrozenDict(self.states)


def reduce_hierarchical_state(
    state: HierarchicalReplayState,
    event: MarketEvent,
) -> HierarchicalReplayState:
    """Pure reducer for public state-change events in the audit stream."""

    if event.kind is EventKind.TIMEFRAME_STATE_CHANGED:
        values = dict(state.timeframe_states)
        values[str(event.details["state_id"])] = event.details["projection_state"]
        return HierarchicalReplayState(
            timeframe_states=FrozenDict(values),
            relations=state.relations,
            session=state.session,
            known_at=event.known_at,
        )
    if event.kind is EventKind.RELATION_STATE_CHANGED:
        values = dict(state.relations)
        values[str(event.details["state_id"])] = event.details["projection_state"]
        return HierarchicalReplayState(
            timeframe_states=state.timeframe_states,
            relations=FrozenDict(values),
            session=state.session,
            known_at=event.known_at,
        )
    if event.kind is EventKind.SESSION_STATE_CHANGED:
        return HierarchicalReplayState(
            timeframe_states=state.timeframe_states,
            relations=state.relations,
            session=event.details["projection_state"],
            known_at=event.known_at,
        )
    return state


def _foundation_identity(*parts: object) -> str:
    payload = json.dumps(
        tuple(str(value) for value in parts),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _leg_tick_size(
    path: Sequence[Candle],
    start: SwingPoint,
    end: SwingPoint,
    explicit: float | None,
) -> float:
    if explicit is not None:
        tick_size = float(explicit)
    else:
        candle_grids = {
            float(candle.price_tick_size)
            for candle in path
            if candle.price_tick_size is not None
        }
        if len(candle_grids) > 1:
            raise ValueError("structural leg path mixes price grids")
        if candle_grids:
            tick_size = candle_grids.pop()
        else:
            candidates = tuple(
                float(swing.price) / int(swing.price_ticks)
                for swing in (start, end)
                if int(swing.price_ticks) > 0
            )
            if not candidates or not all(
                math.isclose(
                    value,
                    candidates[0],
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                for value in candidates
            ):
                raise ValueError("structural leg cannot infer one tick grid")
            tick_size = candidates[0]
    if not math.isfinite(tick_size) or tick_size <= 0.0:
        raise ValueError("structural leg tick size is invalid")
    return tick_size


_NATIVE_TIMEFRAME_MINUTES: Mapping[Timeframe, int] = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
}

_NATIVE_TIMEFRAME_ANCHOR_MINUTES: Mapping[Timeframe, int] = {
    Timeframe.M1: 0,
    Timeframe.M5: 0,
    Timeframe.M15: 0,
    Timeframe.H1: 0,
    Timeframe.H4: 18 * 60,
}


def _require_contiguous_native_candles(
    candles: Sequence[Candle],
    *,
    timeframe: Timeframe,
    object_name: str,
) -> None:
    """Fail closed when a v2 definitional path skips a native BAR."""

    expected_minutes = _NATIVE_TIMEFRAME_MINUTES[Timeframe(timeframe)]
    if not candles:
        raise ValueError(f"{object_name} lacks native-duration real BARs")
    for candle in candles:
        try:
            registered_start, registered_end = registered_native_bar_bounds(
                candle.start,
                timeframe_minutes=expected_minutes,
                anchor_minute=_NATIVE_TIMEFRAME_ANCHOR_MINUTES[timeframe],
            )
        except ValueError as error:
            raise ValueError(
                f"{object_name} lacks native-duration real BARs"
            ) from error
        registered_minutes = expected_trading_minutes(
            registered_start,
            registered_end,
        )
        if (
            candle.timeframe is not timeframe
            or not candle.real_completed
            or candle.synthetic_minutes != 0
            or candle.start != registered_start
            or candle.end != registered_end
            or candle.expected_minutes != registered_minutes
            or candle.observed_minutes != registered_minutes
            or candle.real_minutes != registered_minutes
        ):
            raise ValueError(f"{object_name} lacks native-duration real BARs")
    try:
        contiguous = all(
            next_registered_native_completion(
                left.end,
                timeframe_minutes=expected_minutes,
                anchor_minute=_NATIVE_TIMEFRAME_ANCHOR_MINUTES[timeframe],
            )
            == right.end
            for left, right in zip(candles, candles[1:])
        )
    except ValueError as error:
        raise ValueError(
            f"{object_name} native BAR path is not contiguous"
        ) from error
    if not contiguous:
        raise ValueError(f"{object_name} native BAR path is not contiguous")


def build_swing_geometry_nodes(
    swings: Sequence[SwingPoint],
    candles: Sequence[Candle],
    *,
    tick_size: float,
) -> tuple[SwingGeometryNode, ...]:
    """Freeze each Swing's real completed pivot/confirmation window."""

    values: list[SwingGeometryNode] = []
    seen: set[str] = set()
    for swing in sorted(
        (
            item
            for item in swings
            if item.lifecycle in {SwingLifecycle.CONFIRMED, SwingLifecycle.BROKEN}
            and item.confirmed_at is not None
        ),
        key=lambda item: (item.confirmed_at, item.swing_id),
    ):
        if swing.swing_id in seen:
            raise ValueError("swing geometry input repeats a Swing identity")
        seen.add(swing.swing_id)
        native = tuple(
            sorted(
                (
                    candle
                    for candle in candles
                    if candle.real_completed
                    and candle.timeframe is swing.timeframe
                    and candle.symbol == swing.symbol
                    and candle.instrument_id == swing.instrument_id
                    and candle.end <= swing.confirmed_at
                ),
                key=lambda item: (item.start, item.end),
            )
        )
        pivot_indices = tuple(
            index
            for index, candle in enumerate(native)
            if candle.start == swing.pivot_start
        )
        if len(pivot_indices) != 1:
            raise ValueError("swing geometry lacks one exact pivot candle")
        pivot_index = pivot_indices[0]
        span = int(swing.confirmation_delay_bars)
        left_index = pivot_index - span
        right_index = pivot_index + span
        if (
            left_index < 0
            or right_index >= len(native)
            or native[right_index].end != swing.confirmed_at
        ):
            raise ValueError("swing geometry lacks its definitional bar window")
        window = native[left_index : right_index + 1]
        _require_contiguous_native_candles(
            window,
            timeframe=swing.timeframe,
            object_name="swing geometry",
        )
        values.append(
            SwingGeometryNode(
                swing_id=swing.swing_id,
                timeframe=swing.timeframe,
                symbol=swing.symbol,
                instrument_id=swing.instrument_id,
                window_start=window[0].start,
                window_end=window[-1].end,
                lower_bound=min(float(candle.low) for candle in window),
                upper_bound=max(float(candle.high) for candle in window),
                known_at=swing.confirmed_at,
                source_candle_ids=tuple(
                    candle_identity(candle, tick_size=tick_size)
                    for candle in window
                ),
            )
        )
    return tuple(values)


def _geometric_parent(
    child: SwingGeometryNode,
    nodes: Sequence[SwingGeometryNode],
) -> SwingGeometryNode | None:
    candidates = tuple(
        parent
        for parent in nodes
        if parent.swing_id != child.swing_id
        and parent.symbol == child.symbol
        and parent.instrument_id == child.instrument_id
        and parent.window_start <= child.window_start
        and parent.window_end >= child.window_end
        and parent.lower_bound <= child.lower_bound
        and parent.upper_bound >= child.upper_bound
        and parent.duration_seconds > child.duration_seconds
    )
    return (
        None
        if not candidates
        else min(
            candidates,
            key=lambda item: (
                item.duration_seconds,
                item.price_span,
                item.swing_id,
            ),
        )
    )


def update_swing_geometry_assignments(
    nodes: Sequence[SwingGeometryNode],
    prior_assignments: Sequence[SwingGeometryAssignment] = (),
    *,
    known_at: pd.Timestamp,
) -> tuple[SwingGeometryAssignment, ...]:
    """Append parent changes without rewriting an earlier geometric view."""

    clock = aware_timestamp(known_at, name="swing_geometry.known_at")
    visible = tuple(node for node in nodes if node.known_at <= clock)
    by_id = {node.swing_id: node for node in visible}
    if len(by_id) != len(visible):
        raise ValueError("swing geometry repeats a node identity")
    parent_by_child = {
        node.swing_id: (
            None
            if (parent := _geometric_parent(node, visible)) is None
            else parent.swing_id
        )
        for node in visible
    }

    depth_cache: dict[str, int] = {}

    def depth(swing_id: str) -> int:
        cached = depth_cache.get(swing_id)
        if cached is not None:
            return cached
        parent_id = parent_by_child[swing_id]
        value = 0 if parent_id is None else depth(parent_id) + 1
        depth_cache[swing_id] = value
        return value

    history = tuple(prior_assignments)
    if len({item.assignment_id for item in history}) != len(history):
        raise ValueError("swing geometry repeats an assignment identity")
    latest: dict[str, SwingGeometryAssignment] = {}
    for assignment in history:
        incumbent = latest.get(assignment.child_swing_id)
        if incumbent is None or (
            assignment.assigned_at,
            assignment.assignment_id,
        ) > (incumbent.assigned_at, incumbent.assignment_id):
            latest[assignment.child_swing_id] = assignment
    appended: list[SwingGeometryAssignment] = []
    for child_id in sorted(by_id):
        parent_id = parent_by_child[child_id]
        geometric_depth = depth(child_id)
        incumbent = latest.get(child_id)
        if (
            incumbent is not None
            and incumbent.parent_swing_id == parent_id
            and incumbent.geometric_depth == geometric_depth
        ):
            continue
        assignment_id = _foundation_identity(
            FOUNDATION_VERSION,
            "swing_geometry_assignment",
            child_id,
            parent_id or "root",
            geometric_depth,
            clock.isoformat(),
        )
        appended.append(
            SwingGeometryAssignment(
                assignment_id=assignment_id,
                child_swing_id=child_id,
                parent_swing_id=parent_id,
                geometric_depth=geometric_depth,
                assigned_at=clock,
                supersedes_assignment_id=(
                    None if incumbent is None else incumbent.assignment_id
                ),
            )
        )
    return (*history, *appended)


def update_liquidity_clusters(
    levels: Sequence[LiquidityInventoryItem],
    prior_active: Sequence[LiquidityClusterState] = (),
    *,
    tick_size: float,
    known_at: pd.Timestamp,
) -> LiquidityClusterUpdate:
    """Apply same-side, point-price, one-tick complete-link clustering."""

    tick = float(tick_size)
    clock = aware_timestamp(known_at, name="liquidity_cluster.known_at")
    if not math.isfinite(tick) or tick <= 0.0:
        raise ValueError("liquidity cluster tick size is invalid")
    visible = tuple(
        sorted(
            (
                item
                for item in levels
                if item.lifecycle is LiquidityInventoryLifecycle.VISIBLE
                and item.confirmed_at <= clock
            ),
            key=lambda item: (item.side, float(item.price), item.item_id),
        )
    )
    if len({item.item_id for item in visible}) != len(visible):
        raise ValueError("liquidity cluster input repeats a level identity")
    groups: list[tuple[LiquidityInventoryItem, ...]] = []
    for side in ("above", "below"):
        side_levels = tuple(item for item in visible if item.side == side)
        current: list[LiquidityInventoryItem] = []
        for item in side_levels:
            if not current or float(item.price) - float(current[0].price) <= tick + 1e-12:
                current.append(item)
                continue
            if len(current) >= 2:
                groups.append(tuple(current))
            current = [item]
        if len(current) >= 2:
            groups.append(tuple(current))

    prior = tuple(prior_active)
    if (
        len({item.cluster_id for item in prior}) != len(prior)
        or any(item.terminated_at is not None for item in prior)
        or any(item.updated_at > clock for item in prior)
        or any(
            not math.isclose(
                item.tick_size,
                tick,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for item in prior
        )
    ):
        raise ValueError("prior liquidity cluster workset is invalid")

    def signature(
        side: str,
        member_ids: tuple[str, ...],
        prices: tuple[float, ...],
    ) -> tuple[str, tuple[str, ...], tuple[float, ...]]:
        return side, member_ids, prices

    prior_by_signature = {
        signature(item.side, item.member_level_ids, item.member_prices): item
        for item in prior
    }
    active: list[LiquidityClusterState] = []
    started: list[LiquidityClusterState] = []
    reused_prior_ids: set[str] = set()
    pending_specs: list[
        tuple[
            tuple[LiquidityInventoryItem, ...],
            tuple[str, ...],
            tuple[float, ...],
            tuple[str, ...],
        ]
    ] = []
    for group in groups:
        member_ids = tuple(item.item_id for item in group)
        prices = tuple(float(item.price) for item in group)
        source_ids = tuple(
            dict.fromkeys(
                source_id
                for item in group
                for source_id in item.source_ids
            )
        )
        existing = prior_by_signature.get(
            signature(group[0].side, member_ids, prices)
        )
        if existing is not None:
            # Unchanged membership/geometry is the same cluster generation,
            # not a per-clock semantic revision.
            active.append(existing)
            reused_prior_ids.add(existing.cluster_id)
        else:
            pending_specs.append((group, member_ids, prices, source_ids))

    for group, member_ids, prices, source_ids in pending_specs:
        member_set = set(member_ids)
        supersedes = tuple(
            sorted(
                item.cluster_id
                for item in prior
                if not member_set.isdisjoint(item.member_level_ids)
            )
        )
        cluster_id = _foundation_identity(
            FOUNDATION_VERSION,
            "liquidity_cluster",
            group[0].side,
            *member_ids,
            clock.isoformat(),
        )
        state = LiquidityClusterState(
            cluster_id=cluster_id,
            side=group[0].side,
            member_level_ids=member_ids,
            member_prices=prices,
            member_source_ids=source_ids,
            lower_price=prices[0],
            upper_price=prices[-1],
            tick_size=tick,
            started_at=max(item.confirmed_at for item in group),
            known_at=clock,
            updated_at=clock,
            supersedes_cluster_ids=supersedes,
        )
        active.append(state)
        started.append(state)

    active = sorted(active, key=lambda item: (item.side, item.lower_price, item.cluster_id))
    current_members = {
        item.cluster_id: set(item.member_level_ids) for item in active
    }
    terminated: list[LiquidityClusterState] = []
    supersessions: list[LiquidityClusterSupersession] = []
    for old in prior:
        if old.cluster_id in reused_prior_ids:
            continue
        replacements = tuple(
            item.cluster_id
            for item in active
            if not set(old.member_level_ids).isdisjoint(current_members[item.cluster_id])
        )
        terminated.append(
            replace(
                old,
                updated_at=clock,
                terminated_at=clock,
                termination_reason="superseded",
            )
        )
        supersessions.append(
            LiquidityClusterSupersession(
                superseded_cluster_id=old.cluster_id,
                replacement_cluster_ids=replacements,
                known_at=clock,
            )
        )
    return LiquidityClusterUpdate(
        active=tuple(active),
        started=tuple(started),
        terminated=tuple(terminated),
        supersessions=tuple(supersessions),
    )


def build_structural_range(
    structure_generation_id: str,
    direction: Direction,
    lower_swing: SwingPoint,
    upper_swing: SwingPoint,
    *,
    known_at: pd.Timestamp,
    supersedes_range_id: str | None = None,
) -> StructuralRangeState:
    """Bind two already-qualified opposite structural Swings into geometry."""

    clock = aware_timestamp(known_at, name="structural_range.known_at")
    if (
        not structure_generation_id
        or lower_swing.side is not SwingSide.LOW
        or upper_swing.side is not SwingSide.HIGH
        or lower_swing.timeframe is not upper_swing.timeframe
        or lower_swing.symbol != upper_swing.symbol
        or lower_swing.instrument_id != upper_swing.instrument_id
        or lower_swing.confirmed_at is None
        or upper_swing.confirmed_at is None
        or max(lower_swing.confirmed_at, upper_swing.confirmed_at) > clock
        or lower_swing.price >= upper_swing.price
    ):
        raise ValueError("structural range source Swings are invalid")
    started_at = max(lower_swing.pivot_start, upper_swing.pivot_start)
    range_id = _foundation_identity(
        FOUNDATION_VERSION,
        "structural_range",
        structure_generation_id,
        lower_swing.swing_id,
        upper_swing.swing_id,
        clock.isoformat(),
    )
    return StructuralRangeState(
        range_id=range_id,
        structure_generation_id=structure_generation_id,
        timeframe=lower_swing.timeframe,
        symbol=lower_swing.symbol,
        instrument_id=lower_swing.instrument_id,
        direction=direction,
        lower_swing_id=lower_swing.swing_id,
        upper_swing_id=upper_swing.swing_id,
        lower_bound=float(lower_swing.price),
        upper_bound=float(upper_swing.price),
        started_at=started_at,
        known_at=clock,
        updated_at=clock,
        supersedes_range_id=supersedes_range_id,
    )


def terminate_structural_range(
    state: StructuralRangeState,
    *,
    terminated_at: pd.Timestamp,
    reason: str,
) -> StructuralRangeState:
    clock = aware_timestamp(terminated_at, name="structural_range.terminated_at")
    if state.terminated_at is not None:
        if state.terminated_at == clock and state.termination_reason == reason:
            return state
        raise ValueError("terminal structural range is immutable")
    if clock < state.updated_at or not reason:
        raise ValueError("structural range termination is invalid")
    return replace(
        state,
        updated_at=clock,
        terminated_at=clock,
        termination_reason=reason,
    )


def dual_range_location(
    price: float,
    *,
    structural_range: StructuralRangeState | None,
    balance_range: BalanceRangeState | None,
) -> DualRangeLocation:
    value = float(price)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("range location price is invalid")
    structural_active = bool(
        structural_range is not None and structural_range.terminated_at is None
    )
    balance_active = bool(
        balance_range is not None
        and balance_range.lifecycle is not DealingRangeLifecycle.BROKEN
    )
    return DualRangeLocation(
        structural_range_id=(
            structural_range.range_id if structural_active else None
        ),
        balance_range_id=(balance_range.range_id if balance_active else None),
        x_structural_range=(
            None
            if not structural_active
            else (value - structural_range.lower_bound)
            / (structural_range.upper_bound - structural_range.lower_bound)
        ),
        x_balance_range=(
            None
            if not balance_active
            else (value - balance_range.lower_bound)
            / (balance_range.upper_bound - balance_range.lower_bound)
        ),
    )


def foundation_dual_range_locations(
    projection: "FoundationProjection | None",
    *,
    price: float,
    timeframes: Iterable[Timeframe],
) -> Mapping[Timeframe, DualRangeLocation]:
    """Derive independent structural/balance locations from v2 records."""

    if projection is None:
        return FrozenDict()
    from .semantic_foundation import (
        FoundationObjectType,
        FoundationProjection,
        FoundationRecordStatus,
    )

    if not isinstance(projection, FoundationProjection):
        raise TypeError("foundation range locations require FoundationProjection")
    value = float(price)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("foundation range location price is invalid")
    structural: dict[Timeframe, object] = {}
    balance: dict[Timeframe, object] = {}
    for record in projection.latest_records:
        if record.status is not FoundationRecordStatus.ACTIVE:
            continue
        if record.object_type not in {
            FoundationObjectType.STRUCTURAL_RANGE,
            FoundationObjectType.BALANCE_RANGE,
        }:
            continue
        timeframe = Timeframe(record.payload["timeframe"])
        target = (
            structural
            if record.object_type is FoundationObjectType.STRUCTURAL_RANGE
            else balance
        )
        incumbent = target.get(timeframe)
        if incumbent is None or (
            record.known_at,
            record.object_id,
        ) > (incumbent.known_at, incumbent.object_id):
            target[timeframe] = record
    locations: dict[Timeframe, DualRangeLocation] = {}
    for timeframe in tuple(Timeframe(item) for item in timeframes):
        structural_record = structural.get(timeframe)
        balance_record = balance.get(timeframe)

        def normalized(record) -> float | None:
            if record is None:
                return None
            lower = float(record.payload["lower_bound"])
            upper = float(record.payload["upper_bound"])
            if not 0.0 < lower < upper:
                raise ValueError("foundation range record has invalid geometry")
            return (value - lower) / (upper - lower)

        locations[timeframe] = DualRangeLocation(
            structural_range_id=(
                None if structural_record is None else structural_record.object_id
            ),
            balance_range_id=(
                None if balance_record is None else balance_record.object_id
            ),
            x_structural_range=normalized(structural_record),
            x_balance_range=normalized(balance_record),
        )
    return FrozenDict(locations)


def foundation_dol_timeframe_states(
    projection: "FoundationProjection | None",
    *,
    states: Mapping[Timeframe, TimeframeState],
    price: float,
    candidate_templates: Mapping[str, DOLCandidateView] | None = None,
    real_bar_ordinals: Mapping[Timeframe, int] | None = None,
) -> Mapping[Timeframe, TimeframeState]:
    """Overlay canonical active/rearmed liquidity on the public DOL view.

    The v1.2 inventory remains the compatibility source.  This additive view
    replaces a matching source identity with its foundation level identity,
    suppresses explicitly disarmed/retired identities, and can materialize a
    rearmed level after the compatibility inventory froze its first crossing.
    It never emits a new market-semantic event.
    """

    frozen_states = FrozenDict(states)
    if projection is None:
        return frozen_states
    from .semantic_foundation import FoundationObjectType, FoundationProjection
    from .semantic_lifecycle import LiquidityLevelLifecycle

    if not isinstance(projection, FoundationProjection):
        raise TypeError("foundation DOL view requires FoundationProjection")
    current_price = float(price)
    if not math.isfinite(current_price) or current_price <= 0.0:
        raise ValueError("foundation DOL view price is invalid")
    level_records = tuple(
        record
        for record in projection.latest_records
        if record.object_type is FoundationObjectType.LIQUIDITY_LEVEL
    )
    by_source = {
        str(record.payload["source_identity"]): record
        for record in level_records
    }
    interaction_records = {
        record.object_id: record
        for record in projection.latest_records
        if record.object_type
        is FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION
    }
    ordinals = (
        None
        if real_bar_ordinals is None
        else {
            Timeframe(timeframe): ordinal
            for timeframe, ordinal in real_bar_ordinals.items()
        }
    )
    if ordinals is not None and any(
        type(ordinal) is not int or ordinal < 0
        for ordinal in ordinals.values()
    ):
        raise ValueError("foundation DOL real-bar ordinal is invalid")

    def generation_age(record, fallback: int) -> int:
        if ordinals is None:
            return fallback
        generation_id = record.payload.get("active_generation_id")
        interaction = interaction_records.get(generation_id)
        if interaction is None:
            raise ValueError("foundation DOL level lacks its active interaction")
        timeframe = Timeframe(interaction.payload["interaction_timeframe"])
        current = ordinals.get(timeframe)
        armed = interaction.payload.get("armed_real_bar_ordinal")
        if current is None or type(armed) is not int or current < armed:
            raise ValueError("foundation DOL interaction age is unavailable")
        return current - armed
    if len(by_source) != len(level_records):
        raise ValueError("foundation DOL source identity is ambiguous")
    active_lifecycles = {
        LiquidityLevelLifecycle.ACTIVE.value,
        LiquidityLevelLifecycle.REARMED.value,
    }
    non_materializable_pool_sources = {
        "formed_liquidity_pool",
        "formed_pool",
        "equal_highs",
        "equal_lows",
    }
    active_ids = frozenset(projection.active_dol_candidate_ids)
    templates = FrozenDict(candidate_templates or {})
    if any(
        not isinstance(template, DOLCandidateView)
        or source_identity != template.candidate_id
        or template.rank not in {"internal", "external"}
        for source_identity, template in templates.items()
    ):
        raise ValueError("foundation DOL candidate template is invalid")
    projected: dict[Timeframe, TimeframeState] = {}
    for timeframe, state in frozen_states.items():
        candidates: list[DOLCandidateView] = []
        represented: set[str] = set()
        for candidate in state.liquidity.candidates:
            record = by_source.get(candidate.candidate_id)
            if record is None:
                candidates.append(candidate)
                continue
            lifecycle = str(record.payload["lifecycle"])
            if lifecycle not in active_lifecycles:
                continue
            template = templates.get(candidate.candidate_id)
            tick_size = float(record.payload["tick_size"])
            level_price = int(record.payload["price_ticks"]) * tick_size
            side = str(record.payload["side"])
            atr = state.quality.atr
            candidates.append(
                replace(
                    candidate,
                    candidate_id=record.object_id,
                    # A formed-pool source event retains its exact arithmetic
                    # midpoint even when that midpoint is off-grid.  The
                    # canonical lifecycle freezes the registered near-side
                    # tradable anchor in ``price_ticks``; public DOL state must
                    # therefore publish that anchor, not leak the descriptive
                    # source midpoint back into execution-facing geometry.
                    price=level_price,
                    rank=(
                        template.rank
                        if template is not None
                        else "external"
                        if candidate.rank == "external"
                        else "internal"
                    ),
                    lifecycle=LiquidityInventoryLifecycle.VISIBLE.value,
                    age_bars=generation_age(record, candidate.age_bars),
                    distance_atr=(
                        None
                        if atr is None or atr <= 0.0
                        else (
                            (level_price - current_price) / atr
                            if side == "above"
                            else (current_price - level_price) / atr
                        )
                    ),
                    source_event_id=record.source_event_ids[0],
                )
            )
            represented.add(record.object_id)
        for record in level_records:
            if (
                record.object_id in represented
                or record.object_id not in active_ids
                or Timeframe(record.payload["source_timeframe"]) is not timeframe
                or str(record.payload["source_kind"])
                in non_materializable_pool_sources
            ):
                continue
            tick_size = float(record.payload["tick_size"])
            level_price = int(record.payload["price_ticks"]) * tick_size
            atr = state.quality.atr
            side = str(record.payload["side"])
            source_identity = str(record.payload["source_identity"])
            template = templates.get(source_identity)
            candidates.append(
                DOLCandidateView(
                    candidate_id=record.object_id,
                    timeframe=timeframe,
                    side=side,
                    price=level_price,
                    source_kind=str(record.payload["source_kind"]),
                    rank=("internal" if template is None else template.rank),
                    strength=(0.0 if template is None else template.strength),
                    lifecycle=LiquidityInventoryLifecycle.VISIBLE.value,
                    age_bars=generation_age(record, 0),
                    distance_atr=(
                        None
                        if atr is None or atr <= 0.0
                        else (
                            (level_price - current_price) / atr
                            if side == "above"
                            else (current_price - level_price) / atr
                        )
                    ),
                    source_event_id=record.source_event_ids[0],
                )
            )
            represented.add(record.object_id)
        liquidity = _liquidity_state(
            candidates,
            state.liquidity.recently_swept_ids,
        )
        liquidity = _liquidity_for_range(liquidity, state.range)
        projected[timeframe] = replace(state, liquidity=liquidity)
    return FrozenDict(projected)


def foundation_dol_candidate_template(
    candidate: DOLCandidateView,
) -> DOLCandidateView:
    """Freeze the compatibility rank/strength used by a future rearm view.

    Swing geometric/semantic roles include ``micro`` and ``structural``;
    Brain's preregistered DOL rank admits only ``internal``/``external``.
    Preserve an explicit external creation rank and conservatively map every
    other creation spelling to internal.  Later hierarchy changes do not
    rewrite this frozen compatibility template.
    """

    if not isinstance(candidate, DOLCandidateView):
        raise TypeError("foundation DOL template requires DOLCandidateView")
    return replace(
        candidate,
        rank="external" if candidate.rank == "external" else "internal",
    )


def foundation_dol_protected_candidate_template(
    candidate: DOLCandidateView,
    *,
    protected_swing_id: str,
) -> DOLCandidateView:
    """Apply one exact protected-role promotion to a frozen DOL template.

    Rank is the only compatibility attribute changed by a later protected
    assignment.  Geometry, strength, source identity, and creation-time
    metadata remain frozen, while the model invariant that every protected
    swing is externally ranked stays intact.  Stream publication and atomic
    replay both call this helper from the authoritative
    ``PROTECTED_SWING_ASSIGNED`` event rather than inferring a promotion from
    timeframe or future usefulness.
    """

    if not isinstance(candidate, DOLCandidateView):
        raise TypeError("protected DOL template requires DOLCandidateView")
    if not isinstance(protected_swing_id, str) or not protected_swing_id:
        raise ValueError("protected DOL template lacks a swing identity")
    source_identity = f"swing:{protected_swing_id}"
    if candidate.candidate_id != source_identity:
        raise ValueError("protected DOL template source identity disagrees")
    return replace(candidate, rank="external")


def build_structural_legs(
    timeframe: Timeframe,
    swings: Sequence[SwingPoint],
    candles: Sequence[Candle],
    *,
    atr: float | None = None,
    atr_period: int = 14,
    tick_size: float | None = None,
    frozen_start_atr_by_swing_id: Mapping[str, float] | None = None,
    frozen_start_atr_source_candle_ids_by_swing_id: Mapping[
        str,
        Sequence[str],
    ]
    | None = None,
    protected_swing_ids: Sequence[str] = (),
    structural_swing_ids: Sequence[str] = (),
) -> tuple[StructuralLegState, ...]:
    """Project complete opposite-swing legs without future-role backfill.

    Foundation-v2 paths end at the target pivot bar.  ATR is calculated from
    fourteen real native bars completed strictly before the origin pivot.  The
    explicit per-origin mapping is the causal cold-prefix escape hatch; ``atr``
    remains only as a historical-call compatibility fallback and callers must
    treat it as an already-frozen start ATR.

    ``protected_swing_ids`` and ``structural_swing_ids`` remain accepted for
    checkpoint/caller compatibility only.  Those roles are learned after a
    leg is created and belong to ``SwingHierarchyView`` assignments, never to
    the historical leg fact.
    """

    confirmed = sorted(
        (
            swing
            for swing in swings
            if swing.lifecycle in {SwingLifecycle.CONFIRMED, SwingLifecycle.BROKEN}
            and swing.confirmed_at is not None
        ),
        key=lambda item: (item.pivot_start, item.confirmed_at, item.swing_id),
    )
    contracts = {
        (item.symbol, item.instrument_id)
        for item in confirmed
    }
    if len(contracts) > 1:
        raise ValueError("structural legs require one contract history")
    contract = next(iter(contracts), None)
    native_candles = tuple(
        sorted(
            (
                candle
                for candle in candles
                if candle.real_completed and candle.timeframe is timeframe
                and (
                    contract is None
                    or (candle.symbol, candle.instrument_id) == contract
                )
            ),
            key=lambda item: (item.start, item.end),
        )
    )
    candle_by_start = {candle.start: candle for candle in native_candles}
    if len(candle_by_start) != len(native_candles):
        raise ValueError("structural leg history repeats a native bar start")
    native_starts = tuple(candle.start for candle in native_candles)
    # ATR ancestry is ordered by completion, not start.  A malformed or
    # warm-up candle may have an unusually long duration, so ``end`` is not
    # guaranteed monotone in the start-ordered path array.
    native_by_end = tuple(
        sorted(native_candles, key=lambda candle: (candle.end, candle.start))
    )
    native_ends = tuple(candle.end for candle in native_by_end)
    del protected_swing_ids, structural_swing_ids
    frozen_atr = dict(frozen_start_atr_by_swing_id or {})
    frozen_atr_sources = {
        key: tuple(value)
        for key, value in (
            frozen_start_atr_source_candle_ids_by_swing_id or {}
        ).items()
    }
    output: list[StructuralLegState] = []
    anchor: SwingPoint | None = None
    for current in confirmed:
        if anchor is None or current.side is anchor.side:
            anchor = current
            continue
        if current.pivot_start <= anchor.pivot_start:
            anchor = current
            continue
        path_start = bisect_left(native_starts, anchor.pivot_start)
        path_end = bisect_right(native_starts, current.pivot_start)
        path = native_candles[path_start:path_end]
        if (
            len(path) < 2
            or anchor.pivot_start not in candle_by_start
            or current.pivot_start not in candle_by_start
        ):
            anchor = current
            continue
        if math.isclose(
            float(current.price),
            float(anchor.price),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            # Opposite equal-price pivots describe balance, not a directional
            # structural leg with positive amplitude.
            anchor = current
            continue
        direction = (
            Direction.LONG
            if current.price > anchor.price
            else Direction.SHORT
        )
        closes = tuple(float(candle.close) for candle in path)
        close_travel = sum(
            abs(right - left) for left, right in zip(closes, closes[1:])
        )
        close_efficiency = clamp(
            abs(closes[-1] - closes[0]) / max(close_travel, 1e-12)
        )
        running = closes[0]
        max_retracement = 0.0
        for close in closes[1:]:
            if direction is Direction.LONG:
                running = max(running, close)
                max_retracement = max(max_retracement, running - close)
            else:
                running = min(running, close)
                max_retracement = max(max_retracement, close - running)
        amplitude = abs(float(current.price) - float(anchor.price))
        directional_extremes = (
            (float(anchor.price), *(float(candle.high) for candle in path))
            if direction is Direction.LONG
            else (float(anchor.price), *(float(candle.low) for candle in path))
        )
        extreme_travel = sum(
            abs(right - left)
            for left, right in zip(
                directional_extremes,
                directional_extremes[1:],
            )
        )
        extreme_efficiency = clamp(amplitude / max(extreme_travel, 1e-12))
        close_mae = (
            max(0.0, closes[0] - min(closes))
            if direction is Direction.LONG
            else max(0.0, max(closes) - closes[0])
        )
        wick_mae = (
            max(0.0, float(anchor.price) - min(float(item.low) for item in path))
            if direction is Direction.LONG
            else max(0.0, max(float(item.high) for item in path) - float(anchor.price))
        )
        prior_count = bisect_right(native_ends, anchor.pivot_start)
        prior_candles = tuple(
            candle
            for candle in native_by_end[:prior_count]
            if candle.end <= anchor.pivot_start
        )
        if len(prior_candles) < atr_period:
            atr_result = None
        else:
            source_start = len(prior_candles) - atr_period
            atr_source_candles = prior_candles[source_start:]
            true_ranges: list[float] = []
            for index, candle in enumerate(
                atr_source_candles,
                start=source_start,
            ):
                value = float(candle.high - candle.low)
                if index:
                    prior_close = float(prior_candles[index - 1].close)
                    value = max(
                        value,
                        abs(float(candle.high) - prior_close),
                        abs(float(candle.low) - prior_close),
                    )
                true_ranges.append(max(0.0, value))
            atr_value = sum(true_ranges) / atr_period
            atr_result = (
                (atr_value, atr_source_candles)
                if math.isfinite(atr_value) and atr_value > 0.0
                else None
            )
        grid = _leg_tick_size(path, anchor, current, tick_size)
        foundation_atr_sources: tuple[str, ...] = ()
        foundation_metrics = atr_result is not None
        if atr_result is not None:
            atr0, atr_source_candles = atr_result
            foundation_atr_sources = tuple(
                candle_identity(candle, tick_size=grid)
                for candle in atr_source_candles
            )
        else:
            origin_frozen_atr = frozen_atr.get(anchor.swing_id)
            origin_frozen_sources = frozen_atr_sources.get(anchor.swing_id)
            if (
                origin_frozen_atr is not None
                and origin_frozen_sources is not None
            ):
                if (
                    not math.isfinite(float(origin_frozen_atr))
                    or float(origin_frozen_atr) <= 0.0
                    or len(origin_frozen_sources) != atr_period
                    or len(origin_frozen_sources)
                    != len(set(origin_frozen_sources))
                    or any(
                        not isinstance(value, str) or not value
                        for value in origin_frozen_sources
                    )
                ):
                    raise ValueError(
                        "structural leg frozen ATR source binding is invalid"
                    )
                atr0 = float(origin_frozen_atr)
                foundation_atr_sources = origin_frozen_sources
                foundation_metrics = True
            else:
                explicit_atr = (
                    origin_frozen_atr
                    if origin_frozen_atr is not None
                    else atr
                )
                if (
                    explicit_atr is None
                    or not math.isfinite(float(explicit_atr))
                    or float(explicit_atr) <= 0.0
                ):
                    raise ValueError(
                        "structural leg lacks fourteen strict-prior bars or an "
                        "explicit frozen start ATR"
                    )
                atr0 = float(explicit_atr)
        if (
            anchor.swing_id in frozen_atr_sources
            and anchor.swing_id not in frozen_atr
        ):
            raise ValueError(
                "structural leg frozen ATR ancestry lacks its frozen value"
            )
        if atr_period != 14 and foundation_metrics:
            raise ValueError(
                "foundation structural legs require the registered 14-bar ATR"
            )
        if foundation_metrics and len(foundation_atr_sources) != 14:
            raise ValueError(
                "foundation structural leg lacks exact 14-bar ATR ancestry"
            )
        if foundation_metrics:
            _require_contiguous_native_candles(
                path,
                timeframe=timeframe,
                object_name="foundation structural leg",
            )
        anchor_ticks = price_to_ticks(
            anchor.price,
            grid,
            name="structural_leg.start_price",
        )
        current_ticks = price_to_ticks(
            current.price,
            grid,
            name="structural_leg.end_price",
        )
        if (
            anchor_ticks != int(anchor.price_ticks)
            or current_ticks != int(current.price_ticks)
        ):
            raise ValueError("structural leg Swing ticks disagree with its grid")
        amplitude_ticks = abs(current_ticks - anchor_ticks)
        if amplitude_ticks <= 0:
            raise ValueError(
                "structural leg lacks positive integer-tick amplitude"
            )
        source_ids = (anchor.swing_id, current.swing_id)
        rank = SwingRank.MICRO if len(path) <= 2 else SwingRank.INTERNAL
        raw_id = "|".join(
            (
                SMC_SEMANTIC_VERSION,
                "structural_leg",
                timeframe.value,
                anchor.swing_id,
                current.swing_id,
            )
        )
        foundation_values: dict[str, object] = {}
        if foundation_metrics:
            foundation_values = {
                "amplitude_ticks": amplitude_ticks,
                "atr_at_leg_start": atr0,
                "duration_seconds": int(
                    (current.pivot_start - anchor.pivot_start).total_seconds()
                ),
                "close_efficiency": close_efficiency,
                "extreme_path_efficiency": extreme_efficiency,
                "close_mae_points": close_mae,
                "close_mae_atr": close_mae / atr0,
                "wick_mae_points": wick_mae,
                "wick_mae_atr": wick_mae / atr0,
                "path_candle_ids": tuple(
                    candle_identity(candle, tick_size=grid) for candle in path
                ),
                "atr_source_candle_ids": foundation_atr_sources,
                "foundation_version": FOUNDATION_VERSION,
            }
        output.append(
            StructuralLegState(
                leg_id=hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24],
                timeframe=timeframe,
                direction=direction,
                start_swing_id=anchor.swing_id,
                end_swing_id=current.swing_id,
                start_event_time=anchor.pivot_start,
                end_event_time=current.pivot_start,
                known_at=current.confirmed_at,
                start_price=float(anchor.price),
                end_price=float(current.price),
                start_close=closes[0],
                end_close=closes[-1],
                amplitude_points=amplitude,
                amplitude_atr=amplitude / atr0,
                duration_bars=len(path),
                duration_minutes=int(
                    (current.pivot_start - anchor.pivot_start).total_seconds()
                    // 60
                ),
                efficiency=close_efficiency,
                max_retracement_points=max_retracement,
                max_retracement_atr=max_retracement / atr0,
                rank=rank,
                source_swing_ids=source_ids,
                **foundation_values,
            )
        )
        anchor = current
    return tuple(output[-128:])


class SessionStateReducer:
    """Cross-timeframe session context reduced from completed 1m candles."""

    def __init__(self) -> None:
        self._session_id: str | None = None
        self._open: float | None = None
        self._high: float | None = None
        self._low: float | None = None
        self._prior_high: float | None = None
        self._prior_low: float | None = None
        self._overnight_high: float | None = None
        self._overnight_low: float | None = None
        self._opening_high: float | None = None
        self._opening_low: float | None = None
        self._elapsed = 0
        self._sum_squared_log_returns = 0.0
        self._last_close: float | None = None
        self._last_relative_volume: float | None = None
        self._recent_volume: deque[float] = deque(maxlen=20)
        self._coverage_complete = False

    @staticmethod
    def _key(candle: Candle) -> str:
        local = candle.start.tz_convert("America/New_York")
        date = local.normalize() + (
            pd.Timedelta(days=1) if local.hour >= 18 else pd.Timedelta(0)
        )
        return date.date().isoformat()

    @staticmethod
    def _name_phase(clock: pd.Timestamp) -> tuple[str, str]:
        return session_name_phase(clock)

    def require_update(self, candle: Candle) -> None:
        """Validate one session clock without mutating reducer state."""

        if candle.timeframe is not Timeframe.M1 or not candle.complete:
            raise ValueError("session reducer requires a completed 1m candle")
        if candle.real_completed:
            return
        if self._last_close is None or self._session_id is None:
            raise ValueError(
                "clock-only session candle requires a prior real 1m candle"
            )
        if self._key(candle) != self._session_id:
            raise ValueError(
                "clock-only session candle cannot establish a new session"
            )

    def _state(
        self,
        *,
        candle: Candle,
        name: str,
        phase: str,
    ) -> SessionState:
        if (
            self._session_id is None
            or self._open is None
            or self._high is None
            or self._low is None
        ):  # pragma: no cover - guarded by require_update/real initialization
            raise RuntimeError("session reducer state is not initialized")
        return SessionState(
            session_id=self._session_id,
            name=name,
            phase=phase,
            session_open=float(self._open),
            session_high=float(self._high),
            session_low=float(self._low),
            opening_range_high=self._opening_high,
            opening_range_low=self._opening_low,
            prior_day_high=self._prior_high,
            prior_day_low=self._prior_low,
            overnight_high=self._overnight_high,
            overnight_low=self._overnight_low,
            elapsed_minutes=self._elapsed,
            realized_volatility=math.sqrt(
                self._sum_squared_log_returns
            ),
            relative_volume=self._last_relative_volume,
            known_at=candle.end,
            data_complete=self._coverage_complete,
        )

    def update(self, candle: Candle) -> SessionState:
        self.require_update(candle)
        local_start = candle.start.tz_convert("America/New_York")
        key = self._key(candle)
        if not candle.real_completed:
            self._elapsed += 1
            name, phase = session_name_phase(local_start)
            return self._state(candle=candle, name=name, phase=phase)
        if key != self._session_id:
            if self._session_id is not None:
                self._prior_high = self._high
                self._prior_low = self._low
            self._session_id = key
            self._open = float(candle.open)
            self._high = float(candle.high)
            self._low = float(candle.low)
            self._overnight_high = None
            self._overnight_low = None
            self._opening_high = None
            self._opening_low = None
            self._elapsed = 0
            self._sum_squared_log_returns = 0.0
            self._last_close = None
            self._coverage_complete = bool(
                local_start.hour == 18 and local_start.minute == 0
            )
        self._elapsed += 1
        self._high = max(float(self._high), float(candle.high))
        self._low = min(float(self._low), float(candle.low))
        minute = local_start.hour * 60 + local_start.minute
        if minute >= 18 * 60 or minute < 9 * 60 + 30:
            self._overnight_high = (
                float(candle.high)
                if self._overnight_high is None
                else max(self._overnight_high, float(candle.high))
            )
            self._overnight_low = (
                float(candle.low)
                if self._overnight_low is None
                else min(self._overnight_low, float(candle.low))
            )
        if 9 * 60 + 30 <= minute < 10 * 60:
            self._opening_high = (
                float(candle.high)
                if self._opening_high is None
                else max(self._opening_high, float(candle.high))
            )
            self._opening_low = (
                float(candle.low)
                if self._opening_low is None
                else min(self._opening_low, float(candle.low))
            )
        if self._last_close is not None and candle.close > 0.0:
            value = math.log(float(candle.close) / self._last_close)
            self._sum_squared_log_returns += value * value
        self._last_close = float(candle.close)
        baseline = (
            sum(self._recent_volume) / len(self._recent_volume)
            if self._recent_volume
            else None
        )
        relative_volume = (
            None
            if baseline is None or baseline <= 0.0
            else float(candle.volume) / baseline
        )
        self._last_relative_volume = relative_volume
        self._recent_volume.append(float(candle.volume))
        name, phase = session_name_phase(local_start)
        return self._state(candle=candle, name=name, phase=phase)


def session_name_phase(clock: pd.Timestamp) -> tuple[str, str]:
    """Return the registered exchange-session labels for any aware clock."""

    local = aware_timestamp(clock, name="session.label_clock").tz_convert(
        "America/New_York"
    )
    minute = local.hour * 60 + local.minute
    if minute >= 18 * 60 or minute < 8 * 60 + 30:
        return "overnight", "overnight_delivery"
    if minute < 9 * 60 + 30:
        return "pre_market", "pre_open"
    if minute < 10 * 60:
        return "new_york_am", "opening_expansion"
    if minute < 12 * 60:
        return "new_york_am", "morning_delivery"
    if minute < 13 * 60 + 30:
        return "midday", "midday_balance"
    if minute < 16 * 60:
        return "new_york_pm", "afternoon_delivery"
    return "new_york_close", "closing_rotation"


def _m1_candle_from_event(event: MarketEvent) -> Candle:
    """Reconstruct one normalized completed M1 bar for causal replay."""

    if (
        event.kind is not EventKind.BAR_COMPLETED
        or event.timeframe is not Timeframe.M1
        or event.origin is not EventOrigin.NORMALIZED_DATA
    ):
        raise ValueError(
            "session replay requires a normalized completed M1 BAR event"
        )
    evidence = event.evidence
    required = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
        "instrument_id",
        "data_complete",
        "real_completed",
        "clock_only",
    }
    if not required.issubset(evidence):
        raise ValueError("normalized 1m event lacks replayable candle fields")
    real_completed = evidence["real_completed"]
    clock_only = evidence["clock_only"]
    if (
        type(real_completed) is not bool
        or type(clock_only) is not bool
        or clock_only is not (not real_completed)
    ):
        raise ValueError("normalized 1m event has invalid complementary flags")
    return Candle(
        timeframe=Timeframe.M1,
        start=event.known_at - pd.Timedelta(1, unit="min"),
        end=event.known_at,
        open=float(evidence["open"]),
        high=float(evidence["high"]),
        low=float(evidence["low"]),
        close=float(evidence["close"]),
        volume=float(evidence["volume"]),
        symbol=str(evidence["symbol"]),
        instrument_id=int(evidence["instrument_id"]),
        observed_minutes=1,
        expected_minutes=1,
        complete=True,
        real_minutes=1 if real_completed else 0,
        synthetic_minutes=0 if real_completed else 1,
    )



def _projection_event(
    kind: EventKind,
    *,
    asof: pd.Timestamp,
    timeframe: Timeframe,
    state_id: str,
    state: Any,
    source_event_ids: Sequence[str],
    primitive_state: Mapping[str, Any] | None = None,
) -> MarketEvent:
    primitive = (
        to_primitive(state)
        if primitive_state is None
        else primitive_state
    )
    payload = json.dumps(primitive, sort_keys=True, separators=(",", ":"))
    projection_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    raw = "|".join(
        (SMC_SEMANTIC_VERSION, kind.value, asof.isoformat(), state_id, payload)
    )
    return MarketEvent(
        event_id=hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24],
        kind=kind,
        observed_at=asof,
        timeframe=timeframe,
        side=None,
        price=None,
        strength=0.0,
        source_ids=tuple(source_event_ids),
        details={
            "canonical_semantic": True,
            "projection_only": True,
            "state_id": state_id,
            "projection_sha256": projection_sha256,
            "projection_state": primitive,
        },
        event_time=asof,
        known_at=asof,
        source_event_ids=tuple(source_event_ids),
        origin=EventOrigin.STATE_PROJECTION,
    )


def foundation_record_projection_event(
    record: "FoundationRecord",
    *,
    timeframe: Timeframe,
    published_at: pd.Timestamp | None = None,
    sequence_no: int = 0,
) -> MarketEvent:
    """Build a legacy read-only replay envelope for historical journals.

    Production publication stopped at snapshot schema v2.  This non-exported
    helper remains only so decoder/tamper tests can construct the old wire
    shape and prove cold replay compatibility.
    """

    # Local import is required because semantic_foundation owns the DTO but
    # imports this module's geometry types.
    from .semantic_foundation import FoundationRecord

    if not isinstance(record, FoundationRecord):
        raise TypeError("foundation projection transport requires FoundationRecord")
    publication_clock = aware_timestamp(
        record.known_at if published_at is None else published_at,
        name="foundation projection published_at",
    )
    if publication_clock < record.known_at:
        raise ValueError("foundation projection cannot publish before record knowledge")
    event = _projection_event(
        EventKind.FOUNDATION_STATE_CHANGED,
        asof=publication_clock,
        timeframe=Timeframe(timeframe),
        state_id=record.record_id,
        state=record,
        source_event_ids=record.source_event_ids,
        primitive_state=to_primitive(record),
    )
    evidence = {
        **dict(event.evidence),
        "canonical_semantic": False,
        "technical_projection": "foundation_record",
    }
    return replace(
        event,
        details=evidence,
        evidence=evidence,
        sequence_no=sequence_no,
        event_time=record.known_at,
    )


def foundation_record_from_projection_event(
    event: MarketEvent,
) -> "FoundationRecord":
    """Strictly decode and re-bind one foundation projection transport."""

    from .semantic_foundation import (
        FoundationObjectType,
        FoundationRecord,
        FoundationRecordStatus,
    )

    if not isinstance(event, MarketEvent):
        raise TypeError("foundation projection decoder requires MarketEvent")
    if (
        event.kind is not EventKind.FOUNDATION_STATE_CHANGED
        or event.origin is not EventOrigin.STATE_PROJECTION
        or event.semantic_version != SMC_SEMANTIC_VERSION
    ):
        raise ValueError("event is not registered foundation projection transport")
    primitive = event.evidence.get("projection_state")
    expected_keys = {
        "object_type",
        "object_id",
        "status",
        "known_at",
        "payload",
        "source_event_ids",
        "foundation_version",
        "registry_identity",
        "record_id",
    }
    if not isinstance(primitive, Mapping) or set(primitive) != expected_keys:
        raise ValueError("foundation projection payload is incomplete")
    try:
        record = FoundationRecord(
            object_type=FoundationObjectType(primitive["object_type"]),
            object_id=primitive["object_id"],
            status=FoundationRecordStatus(primitive["status"]),
            known_at=primitive["known_at"],
            payload=primitive["payload"],
            source_event_ids=primitive["source_event_ids"],
            foundation_version=primitive["foundation_version"],
            registry_identity=primitive["registry_identity"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("foundation projection payload is invalid") from error
    if record.record_id != primitive["record_id"]:
        raise ValueError("foundation projection record_id does not bind its payload")
    if record.known_at > event.known_at:
        raise ValueError("foundation projection publishes future record knowledge")
    expected = foundation_record_projection_event(
        record,
        timeframe=event.timeframe,
        published_at=event.known_at,
        sequence_no=event.sequence_no,
    )
    if event != expected:
        raise ValueError("foundation projection event does not exactly bind its record")
    return record


def _validate_foundation_authoritative_sources(
    record: "FoundationRecord",
    authoritative_events: Mapping[str, MarketEvent],
    prior_projection: "FoundationProjection | None" = None,
) -> None:
    """Bind critical foundation DTO fields to exact authoritative event kinds."""

    from .semantic_foundation import FoundationObjectType

    payload = record.payload

    def prior_record(
        object_type: FoundationObjectType,
        object_id: object,
        *,
        role: str,
    ) -> "FoundationRecord":
        if not isinstance(object_id, str) or not object_id or prior_projection is None:
            raise ValueError(f"foundation {role} prior object is missing")
        candidates = tuple(
            item
            for item in prior_projection.latest_records
            if item.object_type is object_type and item.object_id == object_id
        )
        if len(candidates) != 1:
            raise ValueError(f"foundation {role} prior object is absent or ambiguous")
        return candidates[0]

    def source(
        event_id: object,
        expected_kinds: frozenset[EventKind],
        *,
        role: str,
        timeframe: object | None = None,
    ) -> MarketEvent:
        if not isinstance(event_id, str) or not event_id:
            raise ValueError(f"foundation {role} identity is missing")
        event = authoritative_events.get(event_id)
        if event is None or event.kind not in expected_kinds:
            raise ValueError(
                f"foundation {role} has an incompatible authoritative event kind"
            )
        if event.kind is EventKind.BAR_COMPLETED and (
            event.origin is not EventOrigin.NORMALIZED_DATA
            or event.evidence.get("real_completed") is not True
            or event.evidence.get("clock_only") is not False
        ):
            raise ValueError(f"foundation {role} requires an exact real BAR")
        if timeframe is not None and event.timeframe.value != timeframe:
            raise ValueError(f"foundation {role} crosses timeframe scope")
        if event.known_at > record.known_at:
            raise ValueError(f"foundation {role} occurs after record knowledge")
        return event

    def protected_acceptance_for_owner(
        owner_generation_id: object,
        *,
        role: str,
    ) -> str:
        owner = prior_record(
            FoundationObjectType.STRUCTURE_GENERATION,
            owner_generation_id,
            role=f"{role} Structure owner",
        )
        acceptance_event_id = owner.payload.get(
            "protected_acceptance_event_id"
        )
        if (
            owner.payload.get("lifecycle") != "terminated"
            or owner.payload.get("termination_reason")
            != "protected_break_accepted"
            or not isinstance(acceptance_event_id, str)
            or not acceptance_event_id
        ):
            raise ValueError(
                f"foundation {role} does not bind a protected-break owner"
            )
        source(
            acceptance_event_id,
            frozenset({EventKind.ACCEPTANCE_CONFIRMED}),
            role=f"{role} protected Acceptance",
            timeframe=owner.payload.get("timeframe"),
        )
        return acceptance_event_id

    if record.object_type is FoundationObjectType.BASE_ORIGIN_CORE:
        displacement = source(
            payload.get("source_displacement_event_id"),
            frozenset({EventKind.DISPLACEMENT_OBSERVED}),
            role="Base Origin displacement",
            timeframe=payload.get("timeframe"),
        )
        if (
            displacement.evidence.get("displacement_id")
            != payload.get("source_displacement_id")
            or displacement.direction is None
            or displacement.direction.value != payload.get("direction")
            or displacement.known_at != record.known_at
        ):
            raise ValueError("foundation Base Origin displacement identity is incompatible")
        anchor_bars: list[MarketEvent] = []
        for candle_id, bar_event_id in zip(
            payload.get("anchor_candle_ids", ()),
            payload.get("anchor_bar_event_ids", ()),
            strict=True,
        ):
            bar = source(
                bar_event_id,
                frozenset({EventKind.BAR_COMPLETED}),
                role="Base Origin anchor BAR",
                timeframe=payload.get("timeframe"),
            )
            if (
                bar.evidence.get("detector_candle_id") != candle_id
                or bar.evidence.get("symbol") != payload.get("symbol")
                or bar.evidence.get("instrument_id") != payload.get("instrument_id")
                or bar.evidence.get("real_completed") is not True
                or bar.evidence.get("clock_only") is not False
            ):
                raise ValueError("foundation Base Origin BAR identity is incompatible")
            anchor_bars.append(bar)
        anchor_clocks = tuple(
            aware_timestamp(value, name="base_origin.anchor_completed_at")
            for value in payload.get("anchor_completed_at", ())
        )
        interval = pd.Timedelta(
            minutes=_NATIVE_TIMEFRAME_MINUTES[
                Timeframe(payload.get("timeframe"))
            ]
        )
        tick_size = float(payload.get("tick_size"))
        bar_values = tuple(
            (
                float(bar.evidence.get("open")),
                float(bar.evidence.get("high")),
                float(bar.evidence.get("low")),
                float(bar.evidence.get("close")),
            )
            for bar in anchor_bars
        )
        expected_lower = min(values[2] for values in bar_values)
        expected_upper = max(values[1] for values in bar_values)
        expected_body_lower = min(
            min(values[0], values[3]) for values in bar_values
        )
        expected_body_upper = max(
            max(values[0], values[3]) for values in bar_values
        )
        if (
            not anchor_bars
            or record.source_event_ids
            != (displacement.event_id, *tuple(payload.get("anchor_bar_event_ids", ())))
            or tuple(bar.known_at for bar in anchor_bars) != anchor_clocks
            or any(
                right - left != interval
                for left, right in zip(anchor_clocks, anchor_clocks[1:])
            )
            or aware_timestamp(payload.get("formed_at"), name="base_origin.formed_at")
            != anchor_bars[-1].known_at
            or not math.isclose(float(payload.get("lower_bound")), expected_lower)
            or not math.isclose(float(payload.get("upper_bound")), expected_upper)
            or not math.isclose(
                float(payload.get("body_lower_bound")), expected_body_lower
            )
            or not math.isclose(
                float(payload.get("body_upper_bound")), expected_body_upper
            )
        ):
            raise ValueError(
                "foundation Base Origin ancestry, clocks, or BAR geometry conflicts"
            )
        for index, values in enumerate(bar_values):
            for role, value in zip(
                ("open", "high", "low", "close"), values, strict=True
            ):
                price_to_ticks(
                    value,
                    tick_size,
                    name=f"Base Origin anchor {index} {role}",
                )
        return

    if record.object_type is FoundationObjectType.QUALIFIED_ORDER_BLOCK:
        displacement = source(
            payload.get("source_displacement_event_id"),
            frozenset({EventKind.DISPLACEMENT_OBSERVED}),
            role="Qualified OB displacement",
            timeframe=payload.get("timeframe"),
        )
        if displacement.evidence.get("displacement_id") != payload.get(
            "source_displacement_id"
        ) or displacement.direction is None or displacement.direction.value != payload.get(
            "direction"
        ):
            raise ValueError("foundation Qualified OB displacement identity is incompatible")
        compatible_kind = payload.get("compatible_structure_kind")
        expected_kind = {
            "qualified_bos": EventKind.QUALIFIED_BOS,
            "mss_core_confirmed": EventKind.MSS_CORE_CONFIRMED,
        }.get(compatible_kind)
        if expected_kind is None:
            raise ValueError("foundation Qualified OB structure kind is unregistered")
        compatible = source(
            payload.get("compatible_structure_event_id"),
            frozenset({expected_kind}),
            role="Qualified OB structure qualification",
            timeframe=payload.get("timeframe"),
        )
        if (
            compatible.known_at != record.known_at
            or compatible.direction is None
            or compatible.direction.value != payload.get("direction")
        ):
            raise ValueError("foundation Qualified OB qualification clock is incompatible")
        return

    if record.object_type is FoundationObjectType.LIQUIDITY_LEVEL:
        creation_events = tuple(
            authoritative_events[event_id]
            for event_id in record.source_event_ids
            if event_id in authoritative_events
            and authoritative_events[event_id].kind
            is EventKind.LIQUIDITY_LEVEL_CREATED
        )
        if len(creation_events) != 1:
            raise ValueError("foundation Liquidity Level lacks one exact creation fact")
        creation = creation_events[0]
        tick_size = float(payload.get("tick_size"))
        lower_ticks = payload.get("lower_bound_ticks")
        upper_ticks = payload.get("upper_bound_ticks")
        zone_lower = None if creation.zone is None else float(creation.zone[0])
        zone_upper = None if creation.zone is None else float(creation.zone[1])
        price_anchor_rule = payload.get("price_anchor_rule")
        if price_anchor_rule == "exact_event_price":
            source_price_ticks = (
                None
                if creation.price is None
                else price_to_ticks(
                    float(creation.price),
                    tick_size,
                    name="liquidity level source price",
                )
            )
            price_anchor_valid = source_price_ticks == payload.get("price_ticks")
        else:
            midpoint = (
                None
                if creation.zone is None
                else (zone_lower + zone_upper) / 2.0
            )
            price_anchor_valid = bool(
                creation.price is not None
                and midpoint is not None
                and math.isclose(float(creation.price), midpoint)
                and payload.get("price_ticks")
                == (lower_ticks if payload.get("side") == "above" else upper_ticks)
            )
        bound_geometry_valid = bool(
            creation.zone is not None
            and type(lower_ticks) is int
            and type(upper_ticks) is int
            and lower_ticks * tick_size >= zone_lower - 1e-12
            and (lower_ticks - 1) * tick_size < zone_lower - 1e-12
            and upper_ticks * tick_size <= zone_upper + 1e-12
            and (upper_ticks + 1) * tick_size > zone_upper + 1e-12
        )
        if (
            creation.timeframe.value != payload.get("source_timeframe")
            or creation.evidence.get("level_id") != payload.get("source_identity")
            or creation.side != payload.get("side")
            or not price_anchor_valid
            or not bound_geometry_valid
            or creation.known_at
            != aware_timestamp(payload.get("created_at"), name="level.created_at")
        ):
            raise ValueError("foundation Liquidity Level creation is incompatible")
        return

    if record.object_type is FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION:
        interaction_tf = payload.get("interaction_timeframe")
        level_creation = tuple(
            authoritative_events[event_id]
            for event_id in record.source_event_ids
            if event_id in authoritative_events
            and authoritative_events[event_id].kind
            is EventKind.LIQUIDITY_LEVEL_CREATED
        )
        generation_number = payload.get("generation_number")
        if generation_number == 1 and len(level_creation) != 1:
            raise ValueError("foundation Liquidity Interaction lacks its level creation")
        if generation_number > 1:
            level = prior_record(
                FoundationObjectType.LIQUIDITY_LEVEL,
                payload.get("level_id"),
                role="rearmed Liquidity Interaction level",
            )
            if (
                len(level_creation) != 1
                or level_creation[0].event_id not in level.source_event_ids
                or level_creation[0].evidence.get("level_id")
                != level.payload.get("source_identity")
            ):
                raise ValueError(
                    "rearmed Liquidity Interaction does not retain its exact "
                    "same-level creation"
                )
        creation = level_creation[0]
        level_source_identity = creation.evidence.get("level_id")
        if (
            not isinstance(level_source_identity, str)
            or not level_source_identity
            or creation.zone is None
            or len(creation.zone) != 2
        ):
            raise ValueError("foundation Liquidity Interaction level is incompatible")
        zone_lower, zone_upper = (float(creation.zone[0]), float(creation.zone[1]))
        prior_levels = (
            ()
            if prior_projection is None
            else tuple(
                item
                for item in prior_projection.latest_records
                if item.object_type is FoundationObjectType.LIQUIDITY_LEVEL
                and item.object_id == payload.get("level_id")
            )
        )
        if len(prior_levels) > 1:
            raise ValueError("foundation Liquidity Interaction level is ambiguous")
        if prior_levels:
            prior_level = prior_levels[0]
            if (
                prior_level.payload.get("source_identity")
                != level_source_identity
                or creation.event_id not in prior_level.source_event_ids
            ):
                raise ValueError(
                    "foundation Liquidity Interaction level identity conflicts"
                )
            level_tick_size = float(prior_level.payload.get("tick_size"))
            zone_lower = (
                int(prior_level.payload.get("lower_bound_ticks"))
                * level_tick_size
            )
            zone_upper = (
                int(prior_level.payload.get("upper_bound_ticks"))
                * level_tick_size
            )
        touch_bars = tuple(
            source(
                event_id,
                frozenset({EventKind.BAR_COMPLETED}),
                role="Liquidity Interaction touch BAR",
                timeframe=interaction_tf,
            )
            for event_id in payload.get("touch_bar_ids", ())
        )
        touch_facts: tuple[MarketEvent, ...] = ()
        if payload.get("first_touch_at") is not None:
            touch_facts = tuple(
                authoritative_events[event_id]
                for event_id in record.source_event_ids
                if event_id in authoritative_events
                and authoritative_events[event_id].kind is EventKind.LEVEL_TOUCHED
            )
            if (
                not touch_facts
                or min(event.known_at for event in touch_facts)
                != aware_timestamp(
                    payload.get("first_touch_at"),
                    name="interaction.first_touch_at",
                )
                or any(
                    event.evidence.get("level_id") != level_source_identity
                    for event in touch_facts
                )
                or any(
                    len(event.source_event_ids) != 2
                    or event.source_event_ids[0] != creation.event_id
                    or not any(
                        event.source_event_ids[1] == bar.event_id
                        and bar.known_at == event.known_at
                        for bar in touch_bars
                    )
                    or event.side != creation.side
                    or event.zone != creation.zone
                    or event.price is None
                    or not zone_lower <= float(event.price) <= zone_upper
                    or not isinstance(event.evidence.get("source_kind"), str)
                    or not event.evidence.get("source_kind")
                    for event in touch_facts
                )
            ):
                raise ValueError("foundation Liquidity Interaction touch fact is absent")
        if touch_bars:
            if min(event.known_at for event in touch_bars) != aware_timestamp(
                payload.get("first_touch_at"),
                name="interaction.first_touch_at",
            ):
                raise ValueError(
                    "foundation Liquidity Interaction touch BAR clock conflicts"
                )
            touch_fact_by_bar_id = {
                event.source_event_ids[1]: event
                for event in touch_facts
                if len(event.source_event_ids) == 2
            }
            for bar in touch_bars:
                touch_fact = touch_fact_by_bar_id.get(bar.event_id)
                if touch_fact is None:
                    raise ValueError(
                        "foundation Liquidity Interaction touch BAR lacks its fact"
                    )
                # A swing-derived S/R or pool membership touch is learned at
                # the later swing confirmation clock.  The atomic fact
                # transports that confirmation BAR, not the earlier pivot
                # window containing the geometric contact.  Keep this
                # compatibility seam explicit while direct and boundary
                # touches stay bound to their exact BAR/zone intersection.
                deferred_swing_geometry = bool(
                    touch_fact.evidence.get("source_kind")
                    in {"structural_swing", "formed_liquidity_pool"}
                    and touch_fact.evidence.get("touch_reason") is None
                    and type(touch_fact.evidence.get("touch_ordinal")) is int
                    and touch_fact.evidence.get("touch_ordinal") > 1
                )
                high = float(bar.evidence.get("high"))
                low = float(bar.evidence.get("low"))
                boundary_crossing = touch_fact.evidence.get("touch_reason") in {
                    "boundary_crossing",
                    "raw_swing_price_crossing",
                    "external_h1_close_crossing",
                }
                geometry_matches = (
                    high >= zone_lower
                    if boundary_crossing and creation.side == "above"
                    else low <= zone_upper
                    if boundary_crossing and creation.side == "below"
                    else high >= zone_lower and low <= zone_upper
                )
                if not deferred_swing_geometry and not geometry_matches:
                    raise ValueError(
                        "foundation Liquidity Interaction touch BAR geometry conflicts"
                    )
        penetration_facts: tuple[MarketEvent, ...] = ()
        if payload.get("first_penetration_at") is not None:
            penetration_facts = tuple(
                authoritative_events[event_id]
                for event_id in record.source_event_ids
                if event_id in authoritative_events
                and authoritative_events[event_id].kind
                is EventKind.LEVEL_PENETRATED
            )
            first_penetration_at = aware_timestamp(
                payload.get("first_penetration_at"),
                name="interaction.first_penetration_at",
            )
            if (
                not penetration_facts
                or min(event.known_at for event in penetration_facts)
                != first_penetration_at
                or any(
                    event.evidence.get("level_id") != level_source_identity
                    or aware_timestamp(
                        event.evidence.get("crossed_at"),
                        name="interaction.penetration.crossed_at",
                    )
                    != event.known_at
                    for event in penetration_facts
                )
            ):
                raise ValueError("foundation Liquidity Interaction penetration is absent")
        constituents = tuple(payload.get("constituents", ()))
        if constituents:
            level = prior_record(
                FoundationObjectType.LIQUIDITY_LEVEL,
                payload.get("level_id"),
                role="Liquidity Interaction level",
            )
            tick_size = float(level.payload.get("tick_size"))
            for constituent in constituents:
                bar = source(
                    constituent.get("bar_event_id"),
                    frozenset({EventKind.BAR_COMPLETED}),
                    role="Liquidity Interaction constituent BAR",
                    timeframe=interaction_tf,
                )
                if (
                    bar.known_at
                    != aware_timestamp(
                        constituent.get("known_at"),
                        name="interaction.constituent.known_at",
                    )
                    or price_to_ticks(
                        float(bar.evidence.get("high")),
                        tick_size,
                        name="interaction constituent high",
                    )
                    != constituent.get("high_ticks")
                    or price_to_ticks(
                        float(bar.evidence.get("low")),
                        tick_size,
                        name="interaction constituent low",
                    )
                    != constituent.get("low_ticks")
                    or price_to_ticks(
                        float(bar.evidence.get("close")),
                        tick_size,
                        name="interaction constituent close",
                    )
                    != constituent.get("close_ticks")
                ):
                    raise ValueError(
                        "foundation Liquidity Interaction BAR geometry conflicts"
                    )
            penetration_bar_ids = frozenset(
                constituent.get("bar_event_id")
                for constituent in constituents
                if constituent.get("role") == "penetration"
            )
            if penetration_facts and any(
                penetration_bar_ids.isdisjoint(event.source_event_ids)
                for event in penetration_facts
            ):
                raise ValueError(
                    "foundation Liquidity Interaction penetration ancestry conflicts"
                )
        terminal_state = payload.get("terminal_state")
        expected_terminal_kind = {
            "sweep": EventKind.SWEEP_CONFIRMED,
            "acceptance": EventKind.ACCEPTANCE_CONFIRMED,
        }.get(terminal_state)
        if expected_terminal_kind is not None:
            terminal = source(
                payload.get("terminal_event_id"),
                frozenset({expected_terminal_kind}),
                role="Liquidity Interaction terminal",
            )
            if (
                terminal.known_at != record.known_at
                or terminal.evidence.get("level_id") != level_source_identity
                or aware_timestamp(
                    terminal.evidence.get("crossed_at"),
                    name="interaction.terminal.crossed_at",
                )
                != aware_timestamp(
                    payload.get("first_penetration_at"),
                    name="interaction.first_penetration_at",
                )
                or aware_timestamp(
                    terminal.evidence.get("resolved_at"),
                    name="interaction.terminal.resolved_at",
                )
                != aware_timestamp(
                    payload.get("terminal_at"),
                    name="interaction.terminal_at",
                )
                or not penetration_facts
                or all(
                    event.event_id not in terminal.source_event_ids
                    for event in penetration_facts
                )
            ):
                raise ValueError("foundation Liquidity Interaction terminal clock conflicts")
        return

    if record.object_type is FoundationObjectType.STRUCTURE_GENERATION:
        scope = payload.get("scope")
        direction = payload.get("direction")
        timeframe = payload.get("timeframe")
        origin_kind = (
            EventKind.STRUCTURE_DIRECTION_CONFIRMED
            if scope == "external"
            else EventKind.MSS_CORE_CONFIRMED
        )
        origin = source(
            payload.get("origin_event_id"),
            frozenset({origin_kind}),
            role="Structure Generation origin",
            timeframe=timeframe,
        )
        if (
            origin.direction is None
            or origin.direction.value != direction
            or origin.known_at
            != aware_timestamp(payload.get("started_at"), name="structure.started_at")
        ):
            raise ValueError("foundation Structure Generation origin is incompatible")
        origin_swing_id = payload.get("origin_swing_id")
        if scope == "external":
            origin_candidates = tuple(
                value
                for key in (
                    "candidate_protected_swing_id",
                    "source_low_id" if direction == "long" else "source_high_id",
                )
                if isinstance((value := origin.evidence.get(key)), str) and value
            )
            if origin_swing_id not in origin_candidates:
                raise ValueError("foundation external Structure origin Swing is incompatible")
        elif origin_swing_id not in origin.source_entity_ids:
            raise ValueError("foundation internal Structure origin identity is incompatible")
        confirmation_event_id = payload.get("confirmation_event_id")
        if confirmation_event_id is not None:
            confirmation = source(
                confirmation_event_id,
                frozenset({EventKind.STRUCTURE_DIRECTION_CONFIRMED}),
                role="Structure Generation confirmation",
                timeframe=timeframe,
            )
            if (
                confirmation.direction is None
                or confirmation.direction.value != direction
                or confirmation.known_at
                != aware_timestamp(
                    payload.get("confirmed_at"),
                    name="structure.confirmed_at",
                )
                or (scope == "external" and confirmation.event_id != origin.event_id)
            ):
                raise ValueError(
                    "foundation Structure Generation confirmation is incompatible"
                )
        for bos_event_id in payload.get("bos_event_ids", ()):
            bos = source(
                bos_event_id,
                frozenset({EventKind.QUALIFIED_BOS}),
                role="Structure Generation BOS evidence",
                timeframe=timeframe,
            )
            if bos.direction is None or bos.direction.value != direction:
                raise ValueError("foundation Structure Generation BOS direction conflicts")
        for mss_event_id in payload.get("mss_event_ids", ()):
            mss = source(
                mss_event_id,
                frozenset({EventKind.MSS_CORE_CONFIRMED}),
                role="Structure Generation MSS evidence",
                timeframe=timeframe,
            )
            if (
                scope == "internal"
                and (mss.direction is None or mss.direction.value != direction)
            ):
                raise ValueError("foundation Structure Generation MSS direction conflicts")
        assignment_event_id = payload.get("protected_swing_assignment_event_id")
        if assignment_event_id is not None:
            assignment = source(
                assignment_event_id,
                frozenset({EventKind.PROTECTED_SWING_ASSIGNED}),
                role="Structure Generation protected Swing assignment",
                timeframe=timeframe,
            )
            if (
                assignment.evidence.get("protected_swing_id")
                != payload.get("protected_swing_id")
                or assignment.direction is None
                or assignment.direction.value != direction
            ):
                raise ValueError("foundation Structure protected Swing is incompatible")
        acceptance_event_id = payload.get("protected_acceptance_event_id")
        if acceptance_event_id is not None:
            acceptance = source(
                acceptance_event_id,
                frozenset({EventKind.ACCEPTANCE_CONFIRMED}),
                role="Structure Generation protected Acceptance",
            )
            expected = "short" if direction == "long" else "long"
            if (
                acceptance.direction is None
                or acceptance.direction.value != expected
                or acceptance.evidence.get("protected_swing_id")
                != payload.get("protected_swing_id")
                or acceptance.known_at
                != aware_timestamp(
                    payload.get("terminated_at"),
                    name="structure.terminated_at",
                )
            ):
                raise ValueError("foundation Structure Acceptance is incompatible")
        return

    if record.object_type is FoundationObjectType.STRUCTURE_TRANSITION:
        timeframe = payload.get("timeframe")
        challenger = payload.get("challenger_direction")
        incumbent = payload.get("incumbent_direction")
        mss_ids = tuple(payload.get("mss_event_ids", ()))
        for mss_event_id in mss_ids:
            mss = source(
                mss_event_id,
                frozenset({EventKind.MSS_CORE_CONFIRMED}),
                role="Structure Transition MSS evidence",
                timeframe=timeframe,
            )
            if mss.direction is None or mss.direction.value != challenger:
                raise ValueError("foundation Structure Transition MSS direction conflicts")
        if mss_ids and authoritative_events[mss_ids[0]].known_at != aware_timestamp(
            payload.get("started_at"), name="transition.started_at"
        ):
            raise ValueError("foundation Structure Transition start clock is incompatible")
        acceptance_event_id = payload.get("protected_acceptance_event_id")
        if acceptance_event_id is not None:
            acceptance = source(
                acceptance_event_id,
                frozenset({EventKind.ACCEPTANCE_CONFIRMED}),
                role="Structure Transition protected Acceptance",
            )
            if acceptance.direction is None or acceptance.direction.value != challenger:
                raise ValueError("foundation Structure Transition Acceptance conflicts")
        opposite_event_id = payload.get("opposite_confirmation_event_id")
        if opposite_event_id is not None:
            opposite = source(
                opposite_event_id,
                frozenset({EventKind.STRUCTURE_DIRECTION_CONFIRMED}),
                role="Structure Transition opposite confirmation",
                timeframe=timeframe,
            )
            if opposite.direction is None or opposite.direction.value != challenger:
                raise ValueError("foundation Structure Transition confirmation conflicts")
        resumption_event_id = payload.get("resumption_event_id")
        if resumption_event_id is not None:
            resumption = source(
                resumption_event_id,
                frozenset({EventKind.STRUCTURE_DIRECTION_CONFIRMED}),
                role="Structure Transition resumption",
                timeframe=timeframe,
            )
            if resumption.direction is None or resumption.direction.value != incumbent:
                raise ValueError("foundation Structure Transition resumption conflicts")
        return

    if record.object_type is FoundationObjectType.RELATION_GENERATION:
        allowed = {
            EventKind.STRUCTURE_DIRECTION_CONFIRMED,
            EventKind.BAR_COMPLETED,
            EventKind.MARKET_EPOCH_RESET,
        }
        acceptance_sources = tuple(
            event_id
            for event_id in record.source_event_ids
            if authoritative_events.get(event_id) is not None
            and authoritative_events[event_id].kind
            is EventKind.ACCEPTANCE_CONFIRMED
        )
        if acceptance_sources:
            reason = payload.get("termination_reason")
            if reason == "parent_invalidated":
                owner_generation_id = payload.get(
                    "parent_structure_generation_id"
                )
            elif reason == "child_realigned":
                owner_generation_id = payload.get(
                    "child_structure_generation_id"
                )
            else:
                raise ValueError(
                    "foundation Relation Acceptance lacks owner termination semantics"
                )
            expected_acceptance = protected_acceptance_for_owner(
                owner_generation_id,
                role="Relation Generation",
            )
            if acceptance_sources != (expected_acceptance,):
                raise ValueError(
                    "foundation Relation Generation cites the wrong Acceptance"
                )
            allowed.add(EventKind.ACCEPTANCE_CONFIRMED)
        for event_id in record.source_event_ids:
            source(
                event_id,
                frozenset(allowed),
                role="Relation Generation source",
            )
        return

    if record.object_type is FoundationObjectType.DELIVERY_PHASE_GENERATION:
        origin = source(
            payload.get("origin_event_id"),
            frozenset({EventKind.BAR_COMPLETED}),
            role="Delivery Phase origin BAR",
            timeframe=payload.get("timeframe"),
        )
        if origin.known_at != aware_timestamp(
            payload.get("entered_at"), name="delivery.entered_at"
        ):
            raise ValueError("foundation Delivery Phase origin clock is incompatible")
        allowed = {
            EventKind.STRUCTURE_DIRECTION_CONFIRMED,
            EventKind.BAR_COMPLETED,
            EventKind.MARKET_EPOCH_RESET,
        }
        acceptance_sources = tuple(
            event_id
            for event_id in record.source_event_ids
            if authoritative_events.get(event_id) is not None
            and authoritative_events[event_id].kind
            is EventKind.ACCEPTANCE_CONFIRMED
        )
        if acceptance_sources:
            if payload.get("termination_reason") != "parent_structure_terminated":
                raise ValueError(
                    "foundation Delivery Acceptance lacks owner termination semantics"
                )
            expected_acceptance = protected_acceptance_for_owner(
                payload.get("parent_structure_generation_id"),
                role="Delivery Phase",
            )
            if acceptance_sources != (expected_acceptance,):
                raise ValueError(
                    "foundation Delivery Phase cites the wrong Acceptance"
                )
            allowed.add(EventKind.ACCEPTANCE_CONFIRMED)
        for event_id in record.source_event_ids:
            source(event_id, frozenset(allowed), role="Delivery Phase source")

        # The lifecycle counter resets only at an explicit market-epoch
        # boundary.  Reconstruct that exact native BAR prefix so serialized
        # payloads cannot rewrite ordinal/age while retaining valid sources.
        ordered_events = tuple(authoritative_events.values())
        origin_index = next(
            (
                index
                for index, event in enumerate(ordered_events)
                if event.event_id == origin.event_id
            ),
            None,
        )
        if (
            origin_index is None
            or origin.origin is not EventOrigin.NORMALIZED_DATA
            or origin.evidence.get("real_completed") is not True
            or origin.evidence.get("clock_only") is not False
        ):
            raise ValueError("foundation Delivery origin is not one real native BAR")
        previous_resets = tuple(
            index
            for index, event in enumerate(ordered_events[:origin_index])
            if event.kind is EventKind.MARKET_EPOCH_RESET
        )
        epoch_start = (previous_resets[-1] + 1) if previous_resets else 0
        following_resets = tuple(
            (index, event)
            for index, event in enumerate(
                ordered_events[origin_index + 1 :],
                start=origin_index + 1,
            )
            if event.kind is EventKind.MARKET_EPOCH_RESET
            and event.known_at <= record.known_at
        )
        reset_reasons = {"contract_reset", "data_reset", "semantic_reset"}
        if following_resets:
            reset_index, reset_event = following_resets[0]
            if (
                payload.get("termination_reason") not in reset_reasons
                or reset_event.event_id not in record.source_event_ids
                or reset_event.known_at != record.known_at
                or len(following_resets) != 1
            ):
                raise ValueError("foundation Delivery crosses an unbound epoch reset")
            epoch_stop = reset_index
        else:
            epoch_stop = len(ordered_events)
        origin_symbol = origin.evidence.get("symbol")
        origin_instrument = origin.evidence.get("instrument_id")
        native_bars = tuple(
            event
            for event in ordered_events[epoch_start:epoch_stop]
            if event.kind is EventKind.BAR_COMPLETED
            and event.origin is EventOrigin.NORMALIZED_DATA
            and event.timeframe.value == payload.get("timeframe")
            and event.evidence.get("real_completed") is True
            and event.evidence.get("clock_only") is False
            and event.evidence.get("symbol") == origin_symbol
            and event.evidence.get("instrument_id") == origin_instrument
            and event.known_at
            <= aware_timestamp(
                payload.get("last_updated_at"),
                name="delivery.last_updated_at",
            )
        )
        native_ids = tuple(event.event_id for event in native_bars)
        if origin.event_id not in native_ids:
            raise ValueError("foundation Delivery origin is outside its native epoch")
        entered_ordinal = native_ids.index(origin.event_id) + 1
        expected_duration_bars = len(native_bars) - entered_ordinal
        if (
            payload.get("entered_real_bar_ordinal") != entered_ordinal
            or payload.get("duration_bars") != expected_duration_bars
        ):
            raise ValueError("foundation Delivery native ordinal or duration conflicts")

        prior_delivery = None
        if prior_projection is not None:
            prior_candidates = tuple(
                item
                for item in prior_projection.latest_records
                if item.object_type
                is FoundationObjectType.DELIVERY_PHASE_GENERATION
                and item.object_id == record.object_id
            )
            if len(prior_candidates) > 1:
                raise ValueError("foundation Delivery prior revision is ambiguous")
            prior_delivery = prior_candidates[0] if prior_candidates else None
        price_observation = (
            payload.get("lifecycle") == "active"
            or payload.get("termination_reason") == "phase_changed"
        )
        if price_observation:
            cited_native_bars = tuple(
                event
                for event in native_bars
                if event.event_id in record.source_event_ids
            )
            if (
                not cited_native_bars
                or cited_native_bars[-1].event_id != native_bars[-1].event_id
                or cited_native_bars[-1].known_at
                != aware_timestamp(
                    payload.get("last_updated_at"),
                    name="delivery.last_updated_at",
                )
            ):
                raise ValueError(
                    "foundation Delivery lacks its latest exact native BAR"
                )
            origin_close = origin.evidence.get("close")
            current_close = cited_native_bars[-1].evidence.get("close")
            if (
                isinstance(origin_close, bool)
                or not isinstance(origin_close, (int, float))
                or not math.isfinite(float(origin_close))
                or float(origin_close) <= 0.0
                or isinstance(current_close, bool)
                or not isinstance(current_close, (int, float))
                or not math.isfinite(float(current_close))
            ):
                raise ValueError("foundation Delivery BAR price is incompatible")
            implied_tick_size = float(origin_close) / payload.get(
                "origin_price_ticks"
            )
            if (
                not math.isfinite(implied_tick_size)
                or implied_tick_size <= 0.0
                or not math.isclose(
                    payload.get("current_price_ticks") * implied_tick_size,
                    float(current_close),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    "foundation Delivery current price conflicts with its native BAR"
                )
        elif (
            prior_delivery is None
            or payload.get("current_price_ticks")
            != prior_delivery.payload.get("current_price_ticks")
        ):
            # Administrative/owner termination has no price field in its
            # normalized transition.  Its current price must therefore remain
            # the already source-bound value from the last active revision.
            raise ValueError(
                "foundation Delivery terminal price changed without observation"
            )
        return

    if record.object_type is FoundationObjectType.BALANCE_RANGE:
        lifecycle = payload.get("lifecycle")
        expected_kind = {
            "forming": EventKind.DEALING_RANGE_CREATED,
            "mature": EventKind.DEALING_RANGE_ACTIVATED,
            "broken": EventKind.DEALING_RANGE_INVALIDATED,
        }.get(lifecycle)
        if expected_kind is None:
            raise ValueError("foundation BalanceRange lifecycle is unregistered")
        lifecycle_sources = tuple(
            authoritative_events[event_id]
            for event_id in record.source_event_ids
            if event_id in authoritative_events
            and authoritative_events[event_id].kind is expected_kind
        )
        if len(lifecycle_sources) != 1:
            raise ValueError("foundation BalanceRange lacks one exact lifecycle fact")
        lifecycle_event = lifecycle_sources[0]
        source_member_ids = (
            *tuple(payload.get("lower_source_member_swing_ids", ())),
            *tuple(payload.get("upper_source_member_swing_ids", ())),
        )
        expected_entities = (
            payload.get("range_id"),
            payload.get("lower_source_zone_id"),
            payload.get("upper_source_zone_id"),
            *source_member_ids,
        )
        event_clock = aware_timestamp(
            payload.get("state_started_at"),
            name="balance_range.state_started_at",
        )
        last_updated_at = aware_timestamp(
            payload.get("last_updated_at"),
            name="balance_range.last_updated_at",
        )
        expected_event_time = aware_timestamp(
            payload.get("formed_at") if lifecycle == "forming" else payload.get("state_started_at"),
            name="balance_range.event_time",
        )
        if (
            lifecycle_event.timeframe is not Timeframe.H1
            or lifecycle_event.known_at != event_clock
            or lifecycle_event.event_time != expected_event_time
            or lifecycle_event.evidence.get("range_id") != payload.get("range_id")
            or lifecycle_event.evidence.get("lifecycle") != lifecycle
            or lifecycle_event.evidence.get("lower_bound")
            != payload.get("lower_bound")
            or lifecycle_event.evidence.get("upper_bound")
            != payload.get("upper_bound")
            or lifecycle_event.evidence.get("lower_source_zone_id")
            != payload.get("lower_source_zone_id")
            or lifecycle_event.evidence.get("upper_source_zone_id")
            != payload.get("upper_source_zone_id")
            or tuple(lifecycle_event.evidence.get("source_member_swing_ids", ()))
            != source_member_ids
            or lifecycle_event.zone
            != (payload.get("lower_bound"), payload.get("upper_bound"))
            or lifecycle_event.source_entity_ids != expected_entities
        ):
            raise ValueError("foundation BalanceRange lifecycle fact is incompatible")
        bars = tuple(
            authoritative_events[event_id]
            for event_id in record.source_event_ids
            if event_id in authoritative_events
            and authoritative_events[event_id].kind is EventKind.BAR_COMPLETED
        )
        if len(bars) > 1 or any(
            event_id not in {lifecycle_event.event_id, *(bar.event_id for bar in bars)}
            for event_id in record.source_event_ids
        ):
            raise ValueError("foundation BalanceRange ancestry is incompatible")
        if bars:
            bar = bars[0]
            if (
                record.source_event_ids != (lifecycle_event.event_id, bar.event_id)
                or bar.timeframe is not Timeframe.M1
                or bar.known_at != last_updated_at
                or bar.origin is not EventOrigin.NORMALIZED_DATA
                or bar.evidence.get("real_completed") is not True
                or bar.evidence.get("clock_only") is not False
                or bar.evidence.get("symbol") != payload.get("symbol")
                or bar.evidence.get("instrument_id")
                != payload.get("instrument_id")
            ):
                raise ValueError("foundation BalanceRange latest BAR is incompatible")
        elif (
            record.source_event_ids != (lifecycle_event.event_id,)
            or last_updated_at != lifecycle_event.known_at
        ):
            raise ValueError("foundation BalanceRange lacks its latest exact BAR")
        return

    if record.object_type is FoundationObjectType.STRUCTURAL_LEG:
        leg_events = tuple(
            authoritative_events[event_id]
            for event_id in record.source_event_ids
            if event_id in authoritative_events
            and authoritative_events[event_id].kind
            is EventKind.STRUCTURAL_LEG_CREATED
        )
        if len(leg_events) != 1:
            raise ValueError("foundation Structural Leg lacks one exact atomic fact")
        leg = leg_events[0]
        expected_sources = (
            leg.event_id,
            *leg.source_event_ids,
            *leg.context_event_ids,
        )
        if (
            record.source_event_ids != expected_sources
            or leg.evidence.get("leg_id") != record.object_id
            or leg.timeframe.value != payload.get("timeframe")
            or leg.direction is None
            or leg.direction.value != payload.get("direction")
            or leg.known_at != record.known_at
            or tuple(leg.source_entity_ids[1:])
            != tuple(payload.get("source_swing_ids", ()))
        ):
            raise ValueError("foundation Structural Leg atomic identity is incompatible")
        bound_fields = (
            "leg_id",
            "start_swing_id",
            "end_swing_id",
            "start_event_time",
            "end_event_time",
            "start_price",
            "end_price",
            "start_close",
            "end_close",
            "amplitude_points",
            "amplitude_atr",
            "duration_bars",
            "duration_minutes",
            "efficiency",
            "max_retracement_points",
            "max_retracement_atr",
            "rank",
            "foundation_version",
            "amplitude_ticks",
            "atr_at_leg_start",
            "atr_source_candle_ids",
            "duration_seconds",
            "close_efficiency",
            "extreme_path_efficiency",
            "close_mae_points",
            "close_mae_atr",
            "wick_mae_points",
            "wick_mae_atr",
            "path_candle_ids",
        )
        if any(leg.evidence.get(name) != payload.get(name) for name in bound_fields):
            raise ValueError("foundation Structural Leg metrics changed from atomic fact")
        return

    if record.object_type is FoundationObjectType.SWING_GEOMETRY_NODE:
        swing_events = tuple(
            authoritative_events[event_id]
            for event_id in record.source_event_ids
            if event_id in authoritative_events
            and authoritative_events[event_id].kind is EventKind.SWING_CONFIRMED
        )
        if len(swing_events) != 1:
            raise ValueError("foundation Swing Geometry lacks one exact Swing fact")
        swing = swing_events[0]
        if record.source_event_ids != (swing.event_id, *swing.source_event_ids):
            raise ValueError("foundation Swing Geometry ancestry is not exact")
        bars = tuple(authoritative_events[event_id] for event_id in swing.source_event_ids)
        timeframe = payload.get("timeframe")
        symbol = payload.get("symbol")
        instrument_id = payload.get("instrument_id")
        if (
            swing.evidence.get("source_entity_id") != record.object_id
            or swing.timeframe.value != timeframe
            or swing.known_at != record.known_at
            or any(
                bar.kind is not EventKind.BAR_COMPLETED
                or bar.origin is not EventOrigin.NORMALIZED_DATA
                or bar.timeframe is not swing.timeframe
                or bar.evidence.get("real_completed") is not True
                or bar.evidence.get("clock_only") is not False
                or bar.evidence.get("symbol") != symbol
                or bar.evidence.get("instrument_id") != instrument_id
                for bar in bars
            )
        ):
            raise ValueError("foundation Swing Geometry source scope is incompatible")
        interval = pd.Timedelta(
            minutes=_NATIVE_TIMEFRAME_MINUTES[swing.timeframe]
        )
        expected_candle_ids = tuple(
            bar.evidence.get("detector_candle_id") for bar in bars
        )
        expected_lower = min(float(bar.evidence["low"]) for bar in bars)
        expected_upper = max(float(bar.evidence["high"]) for bar in bars)
        if (
            tuple(payload.get("source_candle_ids", ())) != expected_candle_ids
            or aware_timestamp(payload.get("window_start"), name="geometry.window_start")
            != bars[0].known_at - interval
            or aware_timestamp(payload.get("window_end"), name="geometry.window_end")
            != bars[-1].known_at
            or not math.isclose(float(payload.get("lower_bound")), expected_lower)
            or not math.isclose(float(payload.get("upper_bound")), expected_upper)
        ):
            raise ValueError("foundation Swing Geometry differs from its BAR envelope")
        return

    if record.object_type is FoundationObjectType.STRUCTURAL_RANGE:
        # Terminal revisions inherit immutable geometry from their already
        # validated active revision.  At creation, bind both named node IDs to
        # the exact authoritative low/high Swing pivots; a generic geometry
        # envelope alone cannot prove either range boundary price.
        if payload.get("terminated_at") is not None:
            prior_record(
                FoundationObjectType.STRUCTURAL_RANGE,
                record.object_id,
                role="terminal StructuralRange active revision",
            )
            return
        owner = prior_record(
            FoundationObjectType.STRUCTURE_GENERATION,
            payload.get("structure_generation_id"),
            role="StructuralRange owner",
        )
        confirmation_event_id = owner.payload.get("confirmation_event_id")
        confirmation = source(
            confirmation_event_id,
            frozenset({EventKind.STRUCTURE_DIRECTION_CONFIRMED}),
            role="StructuralRange owner confirmation",
            timeframe=payload.get("timeframe"),
        )
        lower = source(
            next(
                (
                    event_id
                    for event_id in record.source_event_ids
                    if (
                        candidate := authoritative_events.get(event_id)
                    ) is not None
                    and candidate.kind is EventKind.SWING_CONFIRMED
                    and candidate.source_entity_ids
                    == (payload.get("lower_swing_id"),)
                ),
                None,
            ),
            frozenset({EventKind.SWING_CONFIRMED}),
            role="StructuralRange lower Swing",
            timeframe=payload.get("timeframe"),
        )
        upper = source(
            next(
                (
                    event_id
                    for event_id in record.source_event_ids
                    if (
                        candidate := authoritative_events.get(event_id)
                    ) is not None
                    and candidate.kind is EventKind.SWING_CONFIRMED
                    and candidate.source_entity_ids
                    == (payload.get("upper_swing_id"),)
                ),
                None,
            ),
            frozenset({EventKind.SWING_CONFIRMED}),
            role="StructuralRange upper Swing",
            timeframe=payload.get("timeframe"),
        )
        if (
            record.source_event_ids
            != (confirmation.event_id, lower.event_id, upper.event_id)
            or confirmation.direction is None
            or confirmation.direction.value != payload.get("direction")
            or lower.side != "below"
            or lower.evidence.get("side") != "low"
            or upper.side != "above"
            or upper.evidence.get("side") != "high"
            or lower.price is None
            or upper.price is None
            or not math.isclose(
                float(lower.price),
                float(payload.get("lower_bound")),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(upper.price),
                float(payload.get("upper_bound")),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "foundation StructuralRange boundaries conflict with exact Swing pivots"
            )
        return

    if record.object_type is FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE:
        creation = source(
            payload.get("source_creation_event_id"),
            frozenset({EventKind.FVG_CREATED}),
            role="FVG creation",
            timeframe=payload.get("timeframe"),
        )
        if (
            creation.evidence.get("fvg_id") != payload.get("fvg_id")
            or creation.known_at
            != aware_timestamp(payload.get("known_at"), name="fvg.known_at")
            or creation.event_time
            != aware_timestamp(payload.get("created_at"), name="fvg.created_at")
        ):
            raise ValueError("foundation FVG creation identity or clock is incompatible")
        creation_bars = tuple(
            source(
                event_id,
                frozenset({EventKind.BAR_COMPLETED}),
                role="FVG creation BAR",
                timeframe=payload.get("timeframe"),
            )
            for event_id in creation.source_event_ids
        )
        if (
            len(creation_bars) != 3
            or any(
                bar.origin is not EventOrigin.NORMALIZED_DATA
                or bar.evidence.get("real_completed") is not True
                or bar.evidence.get("clock_only") is not False
                or bar.evidence.get("symbol") != payload.get("symbol")
                or bar.evidence.get("instrument_id")
                != payload.get("instrument_id")
                for bar in creation_bars
            )
        ):
            raise ValueError("foundation FVG creation BAR scope is incompatible")
        for context_event_id in payload.get("context_source_event_ids", ()):
            source(
                context_event_id,
                frozenset(
                    {
                        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
                        EventKind.SWING_CONFIRMED,
                    }
                ),
                role="FVG structural context",
                timeframe=payload.get("timeframe"),
            )
        terminal_reason = payload.get("terminal_reason")
        if terminal_reason is not None:
            terminal_sources = tuple(payload.get("terminal_source_event_ids", ()))
            terminal_event_id = payload.get("terminal_event_id")
            if (
                record.source_event_ids != terminal_sources
                or terminal_event_id not in terminal_sources
                or terminal_sources[
                    : 1 + len(tuple(payload.get("context_source_event_ids", ())))
                ]
                != (
                    payload.get("source_creation_event_id"),
                    *tuple(payload.get("context_source_event_ids", ())),
                )
            ):
                raise ValueError("foundation FVG terminal ancestry is incompatible")
            cause_kinds = {
                "close_through_far_edge": frozenset({EventKind.BAR_COMPLETED}),
                "data_gap": frozenset(
                    {EventKind.MARKET_EPOCH_RESET, EventKind.DISPLACEMENT_OBSERVED}
                ),
                "parent_structure_terminated": frozenset(
                    {
                        EventKind.ACCEPTANCE_CONFIRMED,
                        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
                        EventKind.MARKET_EPOCH_RESET,
                    }
                ),
                "structural_range_replaced": frozenset(
                    {
                        EventKind.ACCEPTANCE_CONFIRMED,
                        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
                        EventKind.MARKET_EPOCH_RESET,
                    }
                ),
                "contract_rollover": frozenset({EventKind.MARKET_EPOCH_RESET}),
                "semantic_reset": frozenset({EventKind.MARKET_EPOCH_RESET}),
            }.get(terminal_reason)
            if cause_kinds is None:
                raise ValueError("foundation FVG terminal reason is unregistered")
            terminal_kind = (
                frozenset({EventKind.FVG_INVALIDATED})
                if terminal_reason == "close_through_far_edge"
                else cause_kinds
            )
            terminal = source(
                terminal_event_id,
                terminal_kind,
                role="FVG terminal fact",
            )
            if terminal.known_at != record.known_at:
                raise ValueError("foundation FVG terminal clock is incompatible")
            cause_ids = tuple(
                event_id
                for event_id in terminal_sources[
                    1 + len(tuple(payload.get("context_source_event_ids", ()))) :
                ]
                if event_id != terminal_event_id
            )
            for event_id in cause_ids:
                source(event_id, cause_kinds, role="FVG terminal cause")
        return

    if record.object_type is FoundationObjectType.ZONE_FIRST_RETEST:
        object_kind = payload.get("object_kind")
        creation_kind = {
            "fvg": EventKind.FVG_CREATED,
            "qualified_order_block": EventKind.ORIGIN_ZONE_CREATED,
            "base_origin_core": EventKind.DISPLACEMENT_OBSERVED,
            "structural_range": EventKind.STRUCTURE_DIRECTION_CONFIRMED,
            "liquidity_zone": EventKind.LIQUIDITY_LEVEL_CREATED,
        }.get(object_kind)
        if creation_kind is None:
            raise ValueError("foundation first-retest object kind is unregistered")
        creation = source(
            payload.get("creation_event_id"),
            frozenset({creation_kind}),
            role="first-retest creation",
            timeframe=payload.get("timeframe"),
        )
        departure = source(
            payload.get("departure_source_event_id"),
            frozenset({EventKind.DISPLACEMENT_OBSERVED, EventKind.BAR_COMPLETED}),
            role="first-retest departure",
            timeframe=payload.get("timeframe"),
        )
        interaction_bar = source(
            payload.get("source_bar_event_id"),
            frozenset({EventKind.BAR_COMPLETED}),
            role="first-retest interaction BAR",
            timeframe=payload.get("timeframe"),
        )
        if departure.known_at >= record.known_at or interaction_bar.known_at != record.known_at:
            raise ValueError("foundation first-retest source clocks are incompatible")
        if record.source_event_ids != (
            payload.get("creation_event_id"),
            payload.get("departure_source_event_id"),
            payload.get("source_bar_event_id"),
        ):
            raise ValueError("foundation first-retest ancestry is not exact")
        target_type = {
            "fvg": FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE,
            "qualified_order_block": FoundationObjectType.QUALIFIED_ORDER_BLOCK,
            "base_origin_core": FoundationObjectType.BASE_ORIGIN_CORE,
            "structural_range": FoundationObjectType.STRUCTURAL_RANGE,
            "liquidity_zone": FoundationObjectType.LIQUIDITY_LEVEL,
        }[object_kind]
        target = prior_record(
            target_type,
            payload.get("object_id"),
            role="first-retest target",
        )
        target_known_field = {
            "fvg": "known_at",
            "qualified_order_block": "known_at",
            "base_origin_core": "known_at",
            "structural_range": "known_at",
            "liquidity_zone": "created_at",
        }[object_kind]
        target_known_at = aware_timestamp(
            target.payload.get(target_known_field),
            name="first_retest.target_known_at",
        )
        for scope_field in ("symbol", "instrument_id", "timeframe"):
            target_value = target.payload.get(scope_field)
            if target_value is not None and target_value != payload.get(scope_field):
                raise ValueError("foundation first-retest target scope is incompatible")
        if object_kind == "fvg":
            if creation.evidence.get("fvg_id") != target.object_id:
                raise ValueError("foundation first-retest FVG identity is incompatible")
            zone = creation.zone
            target_direction = (
                None if creation.direction is None else creation.direction.value
            )
            expected_displacement_id = creation.evidence.get(
                "source_displacement_id"
            )
        elif object_kind == "qualified_order_block":
            core = prior_record(
                FoundationObjectType.BASE_ORIGIN_CORE,
                target.payload.get("base_origin_core_id"),
                role="first-retest Qualified OB Base Origin Core",
            )
            matching_fields = (
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
            )
            if (
                any(
                    target.payload.get(field) != core.payload.get(field)
                    for field in matching_fields
                )
                or creation.direction is None
                or creation.direction.value != target.payload.get("direction")
                or creation.zone
                != (
                    target.payload.get("lower_bound"),
                    target.payload.get("upper_bound"),
                )
                or creation.evidence.get("source_displacement_id")
                != target.payload.get("source_displacement_id")
                or target.payload.get("source_displacement_event_id")
                not in creation.source_event_ids
            ):
                raise ValueError(
                    "foundation first-retest Qualified OB creation is incompatible"
                )
            zone = creation.zone
            target_direction = target.payload.get("direction")
            expected_displacement_id = target.payload.get(
                "source_displacement_id"
            )
        else:
            zone = (
                target.payload.get("lower_bound"),
                target.payload.get("upper_bound"),
            )
            target_direction = target.payload.get("direction")
            expected_displacement_id = target.payload.get(
                "source_displacement_id"
            )
        if (
            departure.kind is EventKind.DISPLACEMENT_OBSERVED
            and departure.evidence.get("displacement_id")
            != expected_displacement_id
        ):
            raise ValueError(
                "foundation first-retest departure displacement is incompatible"
            )
        if (
            zone is None
            or len(zone) != 2
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                for value in zone
            )
            or target_direction != payload.get("direction")
            or interaction_bar.evidence.get("symbol") != payload.get("symbol")
            or interaction_bar.evidence.get("instrument_id")
            != payload.get("instrument_id")
        ):
            raise ValueError("foundation first-retest target scope is incompatible")
        lower, upper = (float(zone[0]), float(zone[1]))
        bar_open = float(interaction_bar.evidence.get("open"))
        bar_high = float(interaction_bar.evidence.get("high"))
        bar_low = float(interaction_bar.evidence.get("low"))
        expected_entry_side = (
            "gap_opened_inside"
            if lower <= bar_open <= upper
            else "from_above"
            if bar_open > upper
            else "from_below"
        )
        expected_fill = (
            (upper - max(bar_low, lower)) / (upper - lower)
            if target_direction == "long"
            else (min(bar_high, upper) - lower) / (upper - lower)
        )
        native_bars = tuple(
            sorted(
                (
                    event
                    for event in authoritative_events.values()
                    if event.kind is EventKind.BAR_COMPLETED
                    and event.origin is EventOrigin.NORMALIZED_DATA
                    and event.timeframe.value == payload.get("timeframe")
                    and event.evidence.get("real_completed") is True
                    and event.evidence.get("clock_only") is False
                    and event.evidence.get("symbol") == payload.get("symbol")
                    and event.evidence.get("instrument_id")
                    == payload.get("instrument_id")
                    and target_known_at < event.known_at <= record.known_at
                ),
                key=lambda event: (event.known_at, event.event_id),
            )
        )
        native_timeframe = Timeframe(payload.get("timeframe"))
        expected_native_clocks_list: list[pd.Timestamp] = []
        prior_native_clock = target_known_at
        try:
            for _ in native_bars:
                prior_native_clock = next_registered_native_completion(
                    prior_native_clock,
                    timeframe_minutes=_NATIVE_TIMEFRAME_MINUTES[native_timeframe],
                    anchor_minute=_NATIVE_TIMEFRAME_ANCHOR_MINUTES[
                        native_timeframe
                    ],
                )
                expected_native_clocks_list.append(prior_native_clock)
        except ValueError as error:
            raise ValueError(
                "foundation first-retest causal metrics are incompatible"
            ) from error
        expected_native_clocks = tuple(expected_native_clocks_list)
        if (
            not native_bars
            or tuple(event.known_at for event in native_bars)
            != expected_native_clocks
            or native_bars[-1].event_id != interaction_bar.event_id
            or payload.get("age_bars") != len(native_bars)
            or payload.get("age_seconds")
            != int((record.known_at - target_known_at).total_seconds())
            or payload.get("entry_side") != expected_entry_side
            or not math.isclose(
                float(payload.get("fill_fraction")),
                min(1.0, max(0.0, expected_fill)),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or (
                isinstance(interaction_bar.evidence.get("session_name"), str)
                and payload.get("session")
                != interaction_bar.evidence.get("session_name")
            )
        ):
            raise ValueError("foundation first-retest causal metrics are incompatible")
        return

    if record.object_type is FoundationObjectType.BOUNDARY_ATTACK:
        from .semantic_lifecycle import canonical_semantic_id

        swing = source(
            payload.get("target_swing_event_id"),
            frozenset({EventKind.SWING_CONFIRMED}),
            role="Boundary Attack target Swing",
            timeframe=payload.get("timeframe"),
        )
        bar = source(
            payload.get("bar_event_id"),
            frozenset({EventKind.BAR_COMPLETED}),
            role="Boundary Attack BAR",
            timeframe=payload.get("timeframe"),
        )
        direction = payload.get("direction")
        boundary_ticks = payload.get("boundary_ticks")
        extreme_ticks = payload.get("extreme_ticks")
        close_ticks = payload.get("close_ticks")
        swing_price = swing.price
        bar_high = bar.evidence.get("high")
        bar_low = bar.evidence.get("low")
        bar_close = bar.evidence.get("close")
        if (
            bar.known_at != record.known_at
            or type(boundary_ticks) is not int
            or type(extreme_ticks) is not int
            or type(close_ticks) is not int
            or not isinstance(swing_price, (int, float))
            or isinstance(swing_price, bool)
            or not math.isfinite(float(swing_price))
            or float(swing_price) <= 0.0
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (bar_high, bar_low, bar_close)
            )
        ):
            raise ValueError("foundation Boundary Attack sources are incompatible")
        implied_tick_size = float(swing_price) / boundary_ticks
        long_geometry = (
            direction == "long"
            and swing.side == "above"
            and swing.evidence.get("side") == "high"
            and float(bar_high) > float(swing_price)
            and float(bar_close) <= float(swing_price)
            and math.isclose(
                extreme_ticks * implied_tick_size,
                float(bar_high),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        )
        short_geometry = (
            direction == "short"
            and swing.side == "below"
            and swing.evidence.get("side") == "low"
            and float(bar_low) < float(swing_price)
            and float(bar_close) >= float(swing_price)
            and math.isclose(
                extreme_ticks * implied_tick_size,
                float(bar_low),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        )
        if (
            not math.isfinite(implied_tick_size)
            or implied_tick_size <= 0.0
            or not math.isclose(
                close_ticks * implied_tick_size,
                float(bar_close),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not (long_geometry or short_geometry)
        ):
            raise ValueError("foundation Boundary Attack event geometry is incompatible")
        if record.source_event_ids != (
            payload.get("target_swing_event_id"),
            payload.get("bar_event_id"),
        ):
            raise ValueError("foundation Boundary Attack ancestry is not exact")
        expected_identity = canonical_semantic_id(
            "boundary-attack",
            payload.get("timeframe"),
            payload.get("bos_generation_id"),
            payload.get("bar_event_id"),
            payload.get("direction"),
            payload.get("boundary_ticks"),
        )
        prior_attacks = (
            ()
            if prior_projection is None
            else tuple(
                item
                for item in prior_projection.latest_records
                if item.object_type is FoundationObjectType.BOUNDARY_ATTACK
                and item.payload.get("bos_generation_id")
                == payload.get("bos_generation_id")
            )
        )
        if any(
            item.payload.get("timeframe") != payload.get("timeframe")
            or item.payload.get("direction") != payload.get("direction")
            or item.payload.get("target_swing_event_id")
            != payload.get("target_swing_event_id")
            or item.payload.get("boundary_ticks")
            != payload.get("boundary_ticks")
            or item.payload.get("bar_event_id") == payload.get("bar_event_id")
            for item in prior_attacks
        ):
            raise ValueError("foundation Boundary Attack generation is inconsistent")
        if (
            record.object_id != expected_identity
            or payload.get("attempt_ordinal") != len(prior_attacks) + 1
        ):
            raise ValueError("foundation Boundary Attack identity or ordinal conflicts")


class RelationResolver:
    """Resolve only deterministic parent/child facts from independent states."""

    def __init__(
        self,
        *,
        edges: Sequence[tuple[Timeframe, Timeframe]],
    ) -> None:
        normalized = tuple(
            (Timeframe(parent), Timeframe(child))
            for parent, child in edges
        )
        if (
            not normalized
            or len(normalized) != len(set(normalized))
            or any(parent is child for parent, child in normalized)
        ):
            raise ValueError("relation resolver edge registry is invalid")
        self.edges = normalized

    def resolve(
        self,
        states: Mapping[Timeframe, TimeframeState],
        *,
        price: float,
        asof: pd.Timestamp,
    ) -> Mapping[str, RelationState]:
        relations = {
            f"{parent.value}__{child.value}": (
                MarketSnapshotPublisher._relation(
                    states[parent],
                    states[child],
                    price=price,
                    asof=asof,
                )
            )
            for parent, child in self.edges
            if parent in states and child in states
        }
        return FrozenDict(relations)


class MarketSnapshotPublisher:
    """Project and event-source the public hierarchical market state."""

    _ORDER = (Timeframe.H4, Timeframe.H1, Timeframe.M15, Timeframe.M5, Timeframe.M1)
    _RELATION_EDGES = (
        (Timeframe.H4, Timeframe.H1),
        (Timeframe.H1, Timeframe.M15),
        (Timeframe.H1, Timeframe.M5),
        (Timeframe.M15, Timeframe.M5),
        (Timeframe.M5, Timeframe.M1),
    )
    _STATE_SCHEMA_VERSION = 2

    def __init__(
        self,
        *,
        semantic_registry_identity: str,
        atomic_authority: bool = False,
    ) -> None:
        if not semantic_registry_identity:
            raise ValueError("snapshot publisher requires semantic registry identity")
        if type(atomic_authority) is not bool:
            raise ValueError("snapshot publisher authority flag must be boolean")
        self.semantic_registry_identity = semantic_registry_identity
        self.atomic_authority = atomic_authority
        self._event_reducer = TimeframeEventReducer(
            semantic_registry_identity=semantic_registry_identity
        )
        self._relation_resolver = RelationResolver(
            edges=self._RELATION_EDGES
        )
        self._session = SessionStateReducer()
        self._last_projection_payloads: dict[str, Mapping[str, Any]] = {}
        self._last_projection_event_ids: dict[str, str] = {}
        self._formal_structures: dict[
            Timeframe,
            TimeframeStructureState,
        ] = {}
        self._last_real_m1_price: float | None = None
        self._last_real_m1_event_id: str | None = None
        self._boundary_reset_pending = False
        self._publisher_state_schema_version = self._STATE_SCHEMA_VERSION

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_publisher_state_schema_version"] = self._STATE_SCHEMA_VERSION
        return state

    def __setstate__(self, state: Mapping[str, object]) -> None:
        if (
            not isinstance(state, Mapping)
            or state.get("_publisher_state_schema_version")
            != self._STATE_SCHEMA_VERSION
        ):
            raise ValueError("market snapshot publisher checkpoint schema changed")
        self.__dict__.update(state)
        self._require_checkpoint_state()

    def _require_checkpoint_state(self) -> None:
        """Bind cached session/price state to authoritative M1 roots."""

        if (
            not {
                "semantic_registry_identity",
                "atomic_authority",
                "_event_reducer",
                "_session",
                "_last_real_m1_price",
                "_last_real_m1_event_id",
                "_boundary_reset_pending",
            }.issubset(self.__dict__)
            or not isinstance(self._event_reducer, TimeframeEventReducer)
            or not isinstance(self._session, SessionStateReducer)
            or self.semantic_registry_identity
            != self._event_reducer.semantic_registry_identity
            or type(self.atomic_authority) is not bool
            or type(self._boundary_reset_pending) is not bool
            or (
                self._last_real_m1_event_id is not None
                and not isinstance(self._last_real_m1_event_id, str)
            )
            or (
                self._last_real_m1_price is not None
                and (
                    isinstance(self._last_real_m1_price, bool)
                    or not isinstance(self._last_real_m1_price, (int, float))
                    or not math.isfinite(float(self._last_real_m1_price))
                )
            )
        ):
            raise ValueError("market snapshot publisher checkpoint is invalid")
        if not self.atomic_authority:
            return
        expected_session = SessionStateReducer()
        if not self._boundary_reset_pending:
            for event in self._event_reducer._events_by_id.values():
                if (
                    event.kind is EventKind.BAR_COMPLETED
                    and event.timeframe is Timeframe.M1
                ):
                    expected_session.update(_m1_candle_from_event(event))
            latest_real = self._event_reducer.latest_real_m1_event
            expected_price = (
                None
                if latest_real is None
                else float(_m1_candle_from_event(latest_real).close)
            )
            expected_event_id = (
                None if latest_real is None else latest_real.event_id
            )
        else:
            expected_price = None
            expected_event_id = None
        if (
            self._session.__dict__ != expected_session.__dict__
            or self._last_real_m1_price != expected_price
            or self._last_real_m1_event_id != expected_event_id
        ):
            raise ValueError(
                "market snapshot publisher checkpoint differs from M1 replay"
            )

    def on_boundary(self) -> None:
        """Clear projections and require the matching atomic reset event."""

        if self.atomic_authority:
            if self._boundary_reset_pending:
                raise RuntimeError(
                    "atomic market-state boundary reset is already pending"
                )
            self._boundary_reset_pending = True
        self._clear_epoch_projections()

    def _clear_epoch_projections(self) -> None:
        self._session = SessionStateReducer()
        self._last_projection_payloads.clear()
        self._last_projection_event_ids.clear()
        self._formal_structures.clear()
        self._last_real_m1_price = None
        self._last_real_m1_event_id = None
        # Do not clear the authoritative event DAG out of band.  A boundary
        # update may first publish terminal facts for the prior epoch; the
        # immutable MARKET_EPOCH_RESET event then resets the reducer in
        # canonical stream order before any new-epoch bar is consumed.

    def _require_real_m1_anchor_consistency(self) -> None:
        """Keep the effective price, session close, and root index exact."""

        session_close = self._session._last_close
        if (
            (self._last_real_m1_price is None) != (session_close is None)
            or (
                self._last_real_m1_price is not None
                and float(self._last_real_m1_price) != float(session_close)
            )
            or (
                self._last_real_m1_price is None
                and self._last_real_m1_event_id is not None
            )
            or (
                not self.atomic_authority
                and self._last_real_m1_event_id is not None
            )
        ):
            raise RuntimeError("market snapshot real M1 anchors diverged")
        if not self.atomic_authority:
            return
        if self._last_real_m1_event_id is None:
            if (
                not self._boundary_reset_pending
                and self._event_reducer.latest_real_m1_event is not None
            ):
                raise RuntimeError("market snapshot real M1 anchors diverged")
            return
        latest_real = self._event_reducer.latest_real_m1_event
        if (
            latest_real is None
            or latest_real.event_id != self._last_real_m1_event_id
            or float(_m1_candle_from_event(latest_real).close)
            != float(self._last_real_m1_price)
        ):
            raise RuntimeError("market snapshot real M1 anchors diverged")

    def _preflight_atomic_m1_batch(
        self,
        events: Sequence[MarketEvent],
        *,
        current_m1_root: MarketEvent,
        asof: pd.Timestamp,
    ) -> None:
        """Reject unanchored clocks before any publisher owner mutates."""

        asof = aware_timestamp(asof, name="market_snapshot.asof")
        if any(event.known_at > asof for event in events):
            raise RuntimeError(
                "atomic batch contains an event after snapshot asof"
            )
        for event in events:
            if event.kind is not EventKind.BAR_COMPLETED:
                continue
            validate_completed_bar_header(event.evidence)
            if event.origin is EventOrigin.NORMALIZED_DATA:
                validate_registered_native_bar_root(
                    timeframe=event.timeframe,
                    event_time=event.event_time,
                    known_at=event.known_at,
                    evidence=event.evidence,
                )
        current_positions = tuple(
            index
            for index, event in enumerate(events)
            if event.event_id == current_m1_root.event_id
        )
        if len(current_positions) != 1:
            raise RuntimeError(
                "atomic M1 preflight cannot identify the current root"
            )
        reset_positions = tuple(
            index
            for index, event in enumerate(events)
            if event.kind is EventKind.MARKET_EPOCH_RESET
        )
        if reset_positions and reset_positions[-1] >= current_positions[0]:
            raise RuntimeError(
                "atomic batch MARKET_EPOCH_RESET must strictly precede "
                "the current normalized M1 BAR"
            )
        staged_session = copy.deepcopy(self._session)
        staged_real_price = self._last_real_m1_price
        staged_state: SessionState | None = None
        for event in events:
            if event.kind is EventKind.MARKET_EPOCH_RESET:
                staged_session = SessionStateReducer()
                staged_real_price = None
                staged_state = None
                continue
            if (
                event.kind is not EventKind.BAR_COMPLETED
                or event.timeframe is not Timeframe.M1
            ):
                continue
            candle = _m1_candle_from_event(event)
            if candle.real_completed:
                staged_real_price = float(candle.close)
            elif staged_real_price is None:
                raise RuntimeError(
                    "atomic clock-only M1 root requires a prior real "
                    "M1 price in the current epoch"
                )
            staged_state = staged_session.update(candle)
        if staged_real_price is None:
            raise RuntimeError(
                "atomic M1 batch lacks a real price anchor in its "
                "current epoch"
            )
        if staged_state is None or staged_state.known_at != asof:
            raise RuntimeError(
                "atomic session preflight did not reach snapshot asof"
            )
        if (
            staged_session._last_close is None
            or float(staged_session._last_close) != float(staged_real_price)
        ):
            raise RuntimeError(
                "atomic M1 preflight price and session anchors diverged"
            )

    @staticmethod
    def _require_current_m1_root(
        *,
        asof: pd.Timestamp,
        symbol: str,
        instrument_id: int,
        price: float,
        completed_1m: Candle,
        events: Sequence[MarketEvent],
    ) -> MarketEvent:
        asof = aware_timestamp(asof, name="market_snapshot.asof")
        candidates = tuple(
            event
            for event in events
            if event.kind is EventKind.BAR_COMPLETED
            and event.timeframe is Timeframe.M1
            and event.known_at == asof
        )
        if len(candidates) != 1:
            raise RuntimeError(
                "atomic market-state authority requires exactly one "
                "current normalized M1 BAR event"
            )
        event = candidates[0]
        if event.origin is not EventOrigin.NORMALIZED_DATA:
            raise RuntimeError(
                "atomic current M1 BAR must have normalized_data authority"
            )
        replayed = _m1_candle_from_event(event)
        expected_identity = (
            asof,
            str(symbol),
            int(instrument_id),
            float(price),
            float(completed_1m.open),
            float(completed_1m.high),
            float(completed_1m.low),
            float(completed_1m.close),
            float(completed_1m.volume),
            bool(completed_1m.real_completed),
        )
        actual_identity = (
            replayed.end,
            replayed.symbol,
            replayed.instrument_id,
            float(replayed.close),
            float(replayed.open),
            float(replayed.high),
            float(replayed.low),
            float(replayed.close),
            float(replayed.volume),
            bool(replayed.real_completed),
        )
        if (
            completed_1m.timeframe is not Timeframe.M1
            or not completed_1m.complete
            or completed_1m.end != asof
            or completed_1m.symbol != symbol
            or completed_1m.instrument_id != instrument_id
            or float(completed_1m.close) != float(price)
            or (
                event.evidence.get("real_completed")
                is not bool(completed_1m.real_completed)
            )
            or event.price is None
            or float(event.price) != float(replayed.close)
            or actual_identity != expected_identity
        ):
            raise RuntimeError(
                "atomic current M1 BAR differs from snapshot input"
            )
        return event

    @staticmethod
    def _structure(frame: FrameObservation) -> TimeframeStructureState:
        active = sorted(
            (
                item
                for item in frame.structures
                if item.lifecycle is StructureLifecycle.CONFIRMED
                and item.confirmed_at is not None
            ),
            key=lambda item: (item.confirmed_at, item.structure_id or ""),
        )
        external = active[-1] if active else None
        confirmed_breaks = sorted(
            (
                item
                for item in frame.structure_breaks
                if item.lifecycle is BOSLifecycle.CONFIRMED
                and item.resolved_at is not None
            ),
            key=lambda item: (item.resolved_at, item.bos_id),
        )
        last_bos = next(
            (item for item in reversed(confirmed_breaks) if item.scope is BOSScope.CONTINUATION),
            None,
        )
        last_mss = next(
            (item for item in reversed(confirmed_breaks) if item.scope is BOSScope.OPPOSED),
            None,
        )
        latest_break = confirmed_breaks[-1] if confirmed_breaks else None
        latest_leg = frame.structural_legs[-1] if frame.structural_legs else None
        internal = (
            latest_break.direction
            if latest_break is not None
            else latest_leg.direction
            if latest_leg is not None
            else external.direction
            if external is not None
            else None
        )
        origin_swing = None
        if external is not None:
            active_bos = next(
                (
                    item
                    for item in reversed(confirmed_breaks)
                    if (
                        item.scope is BOSScope.CONTINUATION
                        and item.source_structure_id
                        == external.structure_id
                    )
                ),
                None,
            )
            if active_bos is not None:
                origin_leg = next(
                    (
                        leg
                        for leg in reversed(frame.structural_legs)
                        if (
                            leg.end_swing_id
                            == active_bos.target_swing_id
                            and leg.direction is active_bos.direction
                        )
                    ),
                    None,
                )
                if origin_leg is not None:
                    origin_swing = next(
                        (
                            swing
                            for swing in frame.swings
                            if swing.swing_id
                            == origin_leg.start_swing_id
                        ),
                        None,
                    )
        protected_low = (
            float(origin_swing.price)
            if (
                external is not None
                and external.direction is Direction.LONG
                and origin_swing is not None
            )
            else None
        )
        protected_high = (
            float(origin_swing.price)
            if (
                external is not None
                and external.direction is Direction.SHORT
                and origin_swing is not None
            )
            else None
        )
        return TimeframeStructureState(
            external_direction=None if external is None else external.direction,
            internal_direction=internal,
            protected_low=protected_low,
            protected_high=protected_high,
            protected_low_id=(
                origin_swing.swing_id
                if protected_low is not None and origin_swing is not None
                else None
            ),
            protected_high_id=(
                origin_swing.swing_id
                if protected_high is not None and origin_swing is not None
                else None
            ),
            protected_swing_intact=(
                True if origin_swing is not None else None
            ),
            last_bos=None if last_bos is None else float(last_bos.target_price),
            last_bos_direction=None if last_bos is None else last_bos.direction,
            last_mss=None if last_mss is None else float(last_mss.target_price),
            last_mss_direction=None if last_mss is None else last_mss.direction,
        )

    def _formalize_structure(
        self,
        timeframe: Timeframe,
        candidate: TimeframeStructureState,
        semantic_events: Sequence[MarketEvent],
    ) -> TimeframeStructureState:
        """Apply the registered acceptance standard to public structure."""

        prior = self._formal_structures.get(timeframe)
        assignment_event = next(
            (
                event
                for event in reversed(tuple(semantic_events))
                if event.kind is EventKind.PROTECTED_SWING_ASSIGNED
                and event.origin is EventOrigin.SEMANTIC_ATOMIC
                and event.timeframe is timeframe
                and event.evidence.get("protected_swing_id")
                in {
                    candidate.protected_low_id,
                    candidate.protected_high_id,
                }
            ),
            None,
        )
        if assignment_event is not None:
            if assignment_event.direction is Direction.LONG:
                candidate = replace(
                    candidate,
                    protected_low_event_id=assignment_event.event_id,
                    protected_high_event_id=None,
                )
            elif assignment_event.direction is Direction.SHORT:
                candidate = replace(
                    candidate,
                    protected_high_event_id=assignment_event.event_id,
                    protected_low_event_id=None,
                )
        elif prior is not None:
            candidate = replace(
                candidate,
                protected_low_event_id=(
                    prior.protected_low_event_id
                    if candidate.protected_low_id == prior.protected_low_id
                    else None
                ),
                protected_high_event_id=(
                    prior.protected_high_event_id
                    if candidate.protected_high_id == prior.protected_high_id
                    else None
                ),
            )
        if (
            prior is not None
            and prior.external_direction is None
            and prior.protected_swing_intact is False
            and candidate.external_direction is None
        ):
            # Preserve the identity of the formally invalidated protection
            # through the transition. It is replaced only when this same
            # timeframe establishes a new external structure.
            formal = replace(
                candidate,
                protected_low=prior.protected_low,
                protected_high=prior.protected_high,
                protected_low_id=prior.protected_low_id,
                protected_high_id=prior.protected_high_id,
                protected_swing_intact=False,
                protected_low_event_id=prior.protected_low_event_id,
                protected_high_event_id=prior.protected_high_event_id,
            )
        elif prior is None or prior.external_direction is None:
            formal = candidate
        else:
            protected_price = (
                prior.protected_low
                if prior.protected_low is not None
                else prior.protected_high
            )
            protected_id = (
                prior.protected_low_id
                if prior.protected_low_id is not None
                else prior.protected_high_id
            )
            protected_assignment_id = (
                prior.protected_low_event_id
                if prior.protected_low_id is not None
                else prior.protected_high_event_id
            )
            opposite = (
                Direction.SHORT
                if prior.external_direction is Direction.LONG
                else Direction.LONG
            )

            def accepts_exact_protection(event: MarketEvent) -> bool:
                if (
                    event.kind is not EventKind.ACCEPTANCE_CONFIRMED
                    or event.timeframe is not timeframe
                    or event.direction is not opposite
                ):
                    return False
                accepted_protected_id = event.evidence.get(
                    "protected_swing_id"
                )
                protected_assignment_event_id = event.evidence.get(
                    "protected_swing_event_id"
                )
                return bool(
                    event.origin is EventOrigin.SEMANTIC_ATOMIC
                    and protected_id is not None
                    and accepted_protected_id == protected_id
                    and isinstance(protected_assignment_event_id, str)
                    and protected_assignment_event_id
                    == protected_assignment_id
                    and protected_assignment_event_id
                    in event.context_event_ids
                )

            accepted = any(
                accepts_exact_protection(event)
                for event in semantic_events
            )
            assigned = assignment_event is not None
            direction_changed = (
                candidate.external_direction
                is not prior.external_direction
            )
            requires_acceptance = bool(
                protected_price is not None
                and prior.protected_swing_intact is True
            )
            protection_changed = (
                candidate.protected_low_id != prior.protected_low_id
                or candidate.protected_high_id
                != prior.protected_high_id
            )
            if (
                accepted
                and candidate.external_direction
                in {None, prior.external_direction}
            ):
                formal = replace(
                    candidate,
                    protected_low=prior.protected_low,
                    protected_high=prior.protected_high,
                    protected_low_id=prior.protected_low_id,
                    protected_high_id=prior.protected_high_id,
                    protected_swing_intact=False,
                    protected_low_event_id=prior.protected_low_event_id,
                    protected_high_event_id=prior.protected_high_event_id,
                )
            elif direction_changed and requires_acceptance and not accepted:
                formal = TimeframeStructureState(
                    external_direction=prior.external_direction,
                    internal_direction=candidate.internal_direction,
                    protected_low=prior.protected_low,
                    protected_high=prior.protected_high,
                    protected_low_id=prior.protected_low_id,
                    protected_high_id=prior.protected_high_id,
                    protected_swing_intact=(
                        prior.protected_swing_intact
                    ),
                    last_bos=candidate.last_bos,
                    last_bos_direction=candidate.last_bos_direction,
                    last_mss=candidate.last_mss,
                    last_mss_direction=candidate.last_mss_direction,
                    protected_low_event_id=prior.protected_low_event_id,
                    protected_high_event_id=prior.protected_high_event_id,
                )
            elif protection_changed and not assigned and not accepted:
                formal = replace(
                    candidate,
                    protected_low=prior.protected_low,
                    protected_high=prior.protected_high,
                    protected_low_id=prior.protected_low_id,
                    protected_high_id=prior.protected_high_id,
                    protected_swing_intact=prior.protected_swing_intact,
                    protected_low_event_id=prior.protected_low_event_id,
                    protected_high_event_id=prior.protected_high_event_id,
                )
            else:
                formal = candidate
        self._formal_structures[timeframe] = formal
        return formal

    @staticmethod
    def _displacement(
        timeframe: Timeframe,
        displacement: DisplacementObservation | None,
    ) -> tuple[float | None, Mapping[str, float]]:
        if timeframe is not Timeframe.M5 or displacement is None or displacement.current_metrics is None:
            return None, {}
        features = {name: float(value) for name, value in displacement.current_metrics}
        components = (
            clamp(features.get("efficiency", 0.0)),
            clamp(features.get("relative_atr", 0.0) / 2.0),
            clamp(features.get("speed_atr_per_bar", 0.0)),
            clamp(features.get("mean_body_fraction", 0.0)),
            clamp(features.get("body_continuity", 0.0)),
            clamp(features.get("mean_directional_clv", 0.0)),
        )
        return sum(components) / len(components), features

    @staticmethod
    def _range(frame: FrameObservation, price: float) -> TimeframeRangeState:
        ranges = sorted(
            (
                item
                for item in frame.dealing_ranges
                if item.lifecycle is not DealingRangeLifecycle.BROKEN
            ),
            key=lambda item: (item.state_started_at, item.range_id),
        )
        if ranges:
            current = ranges[-1]
            low = float(current.lower_bound)
            high = float(current.upper_bound)
            range_id = current.range_id
            lifecycle = current.lifecycle.value
            source_ids = (
                current.lower_source_zone_id,
                current.upper_source_zone_id,
            )
            range_kind = "active_dealing_range"
        else:
            highs = sorted(
                (
                    swing
                    for swing in frame.swings
                    if swing.side is SwingSide.HIGH
                    and swing.lifecycle in {SwingLifecycle.CONFIRMED, SwingLifecycle.BROKEN}
                    and swing.confirmed_at is not None
                ),
                key=lambda item: (item.pivot_start, item.swing_id),
            )
            lows = sorted(
                (
                    swing
                    for swing in frame.swings
                    if swing.side is SwingSide.LOW
                    and swing.lifecycle in {SwingLifecycle.CONFIRMED, SwingLifecycle.BROKEN}
                    and swing.confirmed_at is not None
                ),
                key=lambda item: (item.pivot_start, item.swing_id),
            )
            if not highs or not lows or lows[-1].price >= highs[-1].price:
                return TimeframeRangeState(None, None, None, None, None, None)
            low = float(lows[-1].price)
            high = float(highs[-1].price)
            source_ids = (lows[-1].swing_id, highs[-1].swing_id)
            range_id = "swing_range:" + hashlib.sha256("|".join(source_ids).encode("utf-8")).hexdigest()[:16]
            lifecycle = "descriptive"
            range_kind = "descriptive_swing_envelope"
        location = (float(price) - low) / (high - low)
        label = "discount" if location < 0.5 else "premium" if location > 0.5 else "equilibrium"
        return TimeframeRangeState(
            range_id=range_id,
            low=low,
            high=high,
            normalized_location=location,
            location_label=label,
            lifecycle=lifecycle,
            source_ids=source_ids,
            range_kind=range_kind,
        )

    @staticmethod
    def _phase(
        structure: TimeframeStructureState,
        range_state: TimeframeRangeState,
        legs: Sequence[StructuralLegState],
    ) -> DeliveryPhase:
        active_leg = (
            legs[-1].direction
            if legs
            else structure.internal_direction
        )
        return _delivery_phase(structure, range_state, active_leg)

    def _timeframe_state(
        self,
        frame: FrameObservation,
        *,
        asof: pd.Timestamp,
        price: float,
        inventory: Sequence[LiquidityInventoryItem],
        displacement: DisplacementObservation | None,
        events: Sequence[MarketEvent],
        anomalies: Sequence[str],
    ) -> TimeframeState:
        structure = self._structure(frame)
        range_state = self._range(frame, price)
        score, features = self._displacement(frame.timeframe, displacement)
        candidates = sorted(
            (
                item
                for item in inventory
                if item.timeframe is frame.timeframe
                and item.lifecycle is not LiquidityInventoryLifecycle.CONSUMED
            ),
            key=lambda item: (item.price, item.item_id),
        )
        candidate_views = tuple(
            DOLCandidateView(
                candidate_id=item.item_id,
                timeframe=item.timeframe,
                side=item.side,
                price=float(item.price),
                source_kind=item.kind,
                rank=item.structural_rank,
                strength=float(item.strength),
                lifecycle=item.lifecycle.value,
                age_bars=item.age_bars,
                distance_atr=(
                    None
                    if frame.metrics.get("atr", 0.0) <= 0.0
                    else (
                        (float(item.price) - float(price))
                        / float(frame.metrics["atr"])
                        if item.side == "above"
                        else (float(price) - float(item.price))
                        / float(frame.metrics["atr"])
                    )
                ),
            )
            for item in candidates
        )
        swept_ids = tuple(
            dict.fromkeys(
                source_id
                for event in events
                if event.kind is EventKind.SWEEP_CONFIRMED
                and event.timeframe is frame.timeframe
                for source_id in event.source_ids
            )
        )
        liquidity_state = _liquidity_for_range(
            _liquidity_state(candidate_views, swept_ids),
            range_state,
        )
        fvgs = tuple(
            PriceZoneView(
                item.fvg_id,
                float(item.lower_bound),
                float(item.upper_bound),
                item.lifecycle.value,
                item.direction,
            )
            for item in frame.fair_value_gaps
            if item.lifecycle in {FairValueGapLifecycle.OPEN, FairValueGapLifecycle.PARTIAL}
        )
        obs = tuple(
            PriceZoneView(
                item.order_block_id,
                float(item.lower_bound),
                float(item.upper_bound),
                item.lifecycle.value,
                item.direction,
            )
            for item in frame.order_blocks
            if item.lifecycle in {OrderBlockLifecycle.CREATED, OrderBlockLifecycle.UNTESTED}
        )
        return TimeframeState(
            timeframe=frame.timeframe,
            structure=structure,
            delivery=TimeframeDeliveryState(
                phase=self._phase(structure, range_state, frame.structural_legs),
                active_leg_direction=(
                    frame.structural_legs[-1].direction
                    if frame.structural_legs
                    else structure.internal_direction
                ),
                displacement_score=score,
                displacement_features=features,
            ),
            range=range_state,
            liquidity=liquidity_state,
            zones=TimeframeZoneState(active_fvg=fvgs, active_ob=obs),
            quality=TimeframeQualityState(
                data_complete=bool(frame.ready),
                semantic_version=SMC_SEMANTIC_VERSION,
                known_at=asof,
                anomaly_tags=tuple(anomalies),
                source_cutoff=frame.cutoff,
                last_transition_known_at=frame.cutoff,
                atr=(
                    None
                    if frame.metrics.get("atr", 0.0) <= 0.0
                    else float(frame.metrics["atr"])
                ),
                semantic_registry_identity=(
                    self.semantic_registry_identity
                ),
            ),
            # The immutable event stream owns complete leg history. The
            # current public state keeps only a bounded recent working set.
            structural_legs=frame.structural_legs[-8:],
        )

    @staticmethod
    def _relation(
        parent: TimeframeState,
        child: TimeframeState,
        *,
        price: float,
        asof: pd.Timestamp | None = None,
    ) -> RelationState:
        parent_direction = parent.structure.external_direction
        child_direction = child.structure.internal_direction
        child_mss_against = bool(
            parent_direction is not None
            and child_direction is not None
            and child_direction is not parent_direction
            and child.structure.last_mss_direction is not None
            and child.structure.last_mss_direction is child_direction
        )
        atr = parent.quality.atr
        protected = parent.structure.protected_low or parent.structure.protected_high
        distance = None
        if (
            protected is not None
            and parent_direction is not None
            and atr is not None
            and atr > 0.0
        ):
            distance = (
                (price - protected) / atr
                if parent_direction is Direction.LONG
                else (protected - price) / atr
            )
        under_attack = bool(
            protected is not None
            and parent_direction is not None
            and (
                float(price) <= float(protected)
                if parent_direction is Direction.LONG
                else float(price) >= float(protected)
            )
        )
        invalidated = bool(
            parent.structure.protected_swing_intact is False
        )
        if parent_direction is None:
            role = RelationRole.PARENT_TRANSITION if parent.delivery.phase is DeliveryPhase.TRANSITION else RelationRole.UNRESOLVED
        elif child_direction is None:
            role = RelationRole.BALANCE_INSIDE_PARENT if child.delivery.phase is DeliveryPhase.BALANCE else RelationRole.UNRESOLVED
        elif invalidated:
            role = RelationRole.PARENT_TRANSITION
        elif child_direction is parent_direction:
            role = (
                RelationRole.BALANCE_INSIDE_PARENT
                if child.delivery.phase is DeliveryPhase.BALANCE
                else RelationRole.ALIGNED_EXPANSION
            )
        elif child_mss_against and under_attack:
            role = RelationRole.REVERSAL_ATTEMPT
        else:
            role = RelationRole.PARENT_RETRACEMENT
        targets = (
            parent.liquidity.candidate_bsl_ids
            if parent_direction is Direction.LONG
            else parent.liquidity.candidate_ssl_ids
            if parent_direction is Direction.SHORT
            else ()
        )
        parent_location = None
        if parent.range.low is not None and parent.range.high is not None:
            parent_location = (
                float(price) - parent.range.low
            ) / (parent.range.high - parent.range.low)
        relation_asof = (
            max(parent.quality.known_at, child.quality.known_at)
            if asof is None
            else aware_timestamp(asof, name="relation.asof")
        )
        return RelationState(
            relation_id=f"{parent.timeframe.value}__{child.timeframe.value}",
            parent_tf=parent.timeframe,
            child_tf=child.timeframe,
            role=role,
            parent_direction=parent_direction,
            child_direction=child_direction,
            parent_protected_swing_intact=parent.structure.protected_swing_intact,
            parent_state_invalidated=invalidated,
            reversal_warning=bool(child_mss_against or under_attack),
            child_location_in_parent_range=parent_location,
            parent_invalidation_distance_atr=distance,
            aligned_target_candidate_ids=targets,
            child_mss_against_parent=child_mss_against,
            parent_protected_swing_under_attack=under_attack,
            known_at=relation_asof,
            parent_source_cutoff=parent.quality.source_cutoff,
            child_source_cutoff=child.quality.source_cutoff,
        )

    def publish(
        self,
        *,
        asof: pd.Timestamp,
        symbol: str,
        instrument_id: int,
        price: float,
        completed_1m: Candle,
        frames: Mapping[Timeframe, FrameObservation],
        inventory: Sequence[LiquidityInventoryItem],
        displacement: DisplacementObservation | None,
        semantic_events: Sequence[MarketEvent],
        anomalies: Sequence[str],
        emit_projection_events: bool = True,
    ) -> tuple[MarketSnapshot, tuple[MarketEvent, ...]]:
        if type(emit_projection_events) is not bool:
            raise ValueError("projection-event emission flag must be boolean")
        if not frames or any(
            key is not frame.timeframe
            for key, frame in frames.items()
        ):
            raise ValueError(
                "market snapshot frame registry is empty or mis-keyed"
            )
        self._require_real_m1_anchor_consistency()
        ordered_events = tuple(
            sorted(
                semantic_events,
                key=lambda item: (
                    item.known_at,
                    item.sequence_no,
                    item.event_id,
                ),
            )
        )
        has_epoch_reset = any(
            event.kind is EventKind.MARKET_EPOCH_RESET
            for event in ordered_events
        )
        if self.atomic_authority:
            if self._boundary_reset_pending and not has_epoch_reset:
                raise RuntimeError(
                    "atomic market-state boundary requires a "
                    "MARKET_EPOCH_RESET event"
                )
            current_m1_root = self._require_current_m1_root(
                asof=asof,
                symbol=symbol,
                instrument_id=instrument_id,
                price=price,
                completed_1m=completed_1m,
                events=ordered_events,
            )
            self._preflight_atomic_m1_batch(
                ordered_events,
                current_m1_root=current_m1_root,
                asof=asof,
            )
            # This registry is immutable for the lifetime of one publisher.
            # It lets the first normalized M1 event initialize empty owner
            # states deterministically while preserving source_cutoff=None
            # until that owner's first native completed bar arrives.
            expected_timeframes = self._event_reducer.bind_timeframes(frames)
        elif has_epoch_reset:
            self._clear_epoch_projections()

        session: SessionState | None = None
        for event in ordered_events:
            if (
                self.atomic_authority
                and event.kind is EventKind.MARKET_EPOCH_RESET
            ):
                self._clear_epoch_projections()
            self._event_reducer.apply(event)
            if (
                self.atomic_authority
                and event.kind is EventKind.MARKET_EPOCH_RESET
            ):
                self._require_real_m1_anchor_consistency()
            if (
                self.atomic_authority
                and event.kind is EventKind.BAR_COMPLETED
                and event.timeframe is Timeframe.M1
            ):
                m1_candle = _m1_candle_from_event(event)
                if m1_candle.real_completed:
                    self._last_real_m1_price = float(m1_candle.close)
                    self._last_real_m1_event_id = event.event_id
                elif self._last_real_m1_price is None:
                    raise RuntimeError(
                        "atomic clock-only M1 root requires a prior real "
                        "M1 price in the current epoch"
                    )
                session = self._session.update(m1_candle)
                self._require_real_m1_anchor_consistency()
        if self.atomic_authority:
            if session is None or session.known_at != asof:
                raise RuntimeError(
                    "atomic session state did not reach snapshot asof"
                )
        else:
            if completed_1m.real_completed:
                self._last_real_m1_price = float(completed_1m.close)
                self._last_real_m1_event_id = None
            elif self._last_real_m1_price is None:
                raise RuntimeError(
                    "clock-only M1 candle requires a prior real M1 price "
                    "in the current epoch"
                )
            session = self._session.update(completed_1m)
            self._require_real_m1_anchor_consistency()
        if self._last_real_m1_price is None:  # pragma: no cover - guarded above
            raise RuntimeError("market snapshot lacks a real M1 price")
        effective_price = self._last_real_m1_price
        event_states = dict(self._event_reducer.states)
        if self.atomic_authority:
            missing = tuple(
                timeframe
                for timeframe in frames
                if timeframe not in event_states
            )
            if missing:
                missing_text = ", ".join(
                    timeframe.value for timeframe in missing
                )
                raise RuntimeError(
                    "atomic market-state authority lacks timeframe state: "
                    f"{missing_text}; frame projection fallback is forbidden"
                )
            unexpected = tuple(
                timeframe
                for timeframe in event_states
                if timeframe not in frames
            )
            if unexpected:
                raise RuntimeError(
                    "atomic market-state authority contains unregistered "
                    "timeframe state: "
                    + ", ".join(
                        timeframe.value for timeframe in unexpected
                    )
                )
            states = {
                timeframe: event_states[timeframe]
                for timeframe in expected_timeframes
            }
            authority = MarketSnapshotAuthority.ATOMIC_EVENT_REDUCER
        else:
            projected_states = {
                timeframe: self._timeframe_state(
                    frame,
                    asof=asof,
                    price=effective_price,
                    inventory=inventory,
                    displacement=displacement,
                    events=semantic_events,
                    anomalies=anomalies,
                )
                for timeframe, frame in frames.items()
            }
            for timeframe, state in tuple(projected_states.items()):
                formal_structure = self._formalize_structure(
                    timeframe,
                    state.structure,
                    semantic_events,
                )
                if formal_structure != state.structure:
                    projected_states[timeframe] = replace(
                        state,
                        structure=formal_structure,
                        delivery=replace(
                            state.delivery,
                            phase=self._phase(
                                formal_structure,
                                state.range,
                                state.structural_legs,
                            ),
                        ),
                    )
            states = projected_states
            authority = MarketSnapshotAuthority.FRAME_PROJECTION
        relations = self._relation_resolver.resolve(
            states,
            price=effective_price,
            asof=asof,
        )
        base_ids_by_timeframe = {
            timeframe: tuple(
                dict.fromkeys(
                    event.event_id
                    for event in ordered_events
                    if (
                        event.timeframe is timeframe
                        or event.kind is EventKind.MARKET_EPOCH_RESET
                        or (
                            event.kind is EventKind.BAR_COMPLETED
                            and event.timeframe is Timeframe.M1
                        )
                        or event.evidence.get("source_timeframe")
                        == timeframe.value
                    )
                )
            )
            for timeframe in states
        }
        projection_events: list[MarketEvent] = []
        if emit_projection_events:
            timeframe_event_ids: dict[Timeframe, str] = {}
            for timeframe in self._ORDER:
                state = states.get(timeframe)
                if state is None:
                    continue
                state_key = f"timeframe:{timeframe.value}"
                primitive = to_primitive(state)
                if self._last_projection_payloads.get(state_key) != primitive:
                    event = _projection_event(
                        EventKind.TIMEFRAME_STATE_CHANGED,
                        asof=asof,
                        timeframe=timeframe,
                        state_id=timeframe.value,
                        state=state,
                        source_event_ids=base_ids_by_timeframe[timeframe],
                        primitive_state=primitive,
                    )
                    projection_events.append(event)
                    self._last_projection_payloads[state_key] = primitive
                    self._last_projection_event_ids[state_key] = event.event_id
                timeframe_event_ids[timeframe] = self._last_projection_event_ids[
                    state_key
                ]
            session_key = "session"
            session_primitive = to_primitive(session)
            if self._last_projection_payloads.get(session_key) != session_primitive:
                session_event = _projection_event(
                    EventKind.SESSION_STATE_CHANGED,
                    asof=asof,
                    timeframe=Timeframe.M1,
                    state_id=session.session_id,
                    state=session,
                    source_event_ids=(timeframe_event_ids.get(Timeframe.M1),)
                    if Timeframe.M1 in timeframe_event_ids
                    else (),
                    primitive_state=session_primitive,
                )
                projection_events.append(session_event)
                self._last_projection_payloads[session_key] = session_primitive
                self._last_projection_event_ids[session_key] = (
                    session_event.event_id
                )
            for relation in relations.values():
                relation_key = f"relation:{relation.relation_id}"
                primitive = to_primitive(relation)
                if self._last_projection_payloads.get(relation_key) != primitive:
                    event = _projection_event(
                        EventKind.RELATION_STATE_CHANGED,
                        asof=asof,
                        timeframe=relation.child_tf,
                        state_id=relation.relation_id,
                        state=relation,
                        source_event_ids=(
                            timeframe_event_ids[relation.parent_tf],
                            timeframe_event_ids[relation.child_tf],
                        ),
                        primitive_state=primitive,
                    )
                    projection_events.append(event)
                    self._last_projection_payloads[relation_key] = primitive
                    self._last_projection_event_ids[relation_key] = event.event_id
        labels = tuple(state.label for state in states.values()) + tuple(
            f"{relation.relation_id} {relation.role.value}"
            for relation in relations.values()
        )
        snapshot = MarketSnapshot(
            asof=asof,
            symbol=symbol,
            instrument_id=instrument_id,
            price=effective_price,
            semantic_version=SMC_SEMANTIC_VERSION,
            semantic_registry_identity=self.semantic_registry_identity,
            timeframe_states=states,
            relations=relations,
            session=session,
            events_this_update=tuple((*ordered_events, *projection_events)),
            labels=labels,
            authority=authority,
        )
        if self.atomic_authority and has_epoch_reset:
            self._boundary_reset_pending = False
        return snapshot, tuple(projection_events)


def replay_atomic_market_snapshot(
    events: Iterable[MarketEvent],
    *,
    semantic_registry_identity: str,
    semantic_version: str = SMC_SEMANTIC_VERSION,
    expected_timeframes: Iterable[Timeframe] | None = None,
    foundation_records: Iterable["FoundationRecord"] = (),
) -> MarketSnapshot:
    """Rebuild the latest hierarchy from normalized and semantic events only.

    Legacy projection events are accepted but ignored by the timeframe
    reducer. Foundation projection transport is decoded by its own pure
    reducer and attached without becoming timeframe evidence. A registered
    epoch-reset event clears base timeframe/session state; foundation history
    remains append-only until explicit terminal/archive records close it.
    """

    from .semantic_foundation import (
        FoundationProjectionReducer,
        FoundationRecord,
    )

    reducer = TimeframeEventReducer(
        semantic_registry_identity=semantic_registry_identity,
        semantic_version=semantic_version,
        expected_timeframes=expected_timeframes,
    )
    session_reducer = SessionStateReducer()
    session: SessionState | None = None
    latest_bar: Candle | None = None
    latest_real_bar: Candle | None = None
    retained_events: list[MarketEvent] = []
    epoch_scale_registry_id: str | None = None
    epoch_contract: tuple[str, int] | None = None
    foundation = None
    foundation_candidate_templates: dict[str, DOLCandidateView] = {}
    foundation_real_bar_ordinals: dict[Timeframe, int] = {}
    last_input_order_key: tuple[pd.Timestamp, int, str] | None = None
    input_event_ids: set[str] = set()
    authoritative_events: dict[str, MarketEvent] = {}
    legacy_foundation_records_by_id: dict[str, FoundationRecord] = {}
    for event in events:
        if not isinstance(event, MarketEvent):
            raise TypeError("atomic replay accepts only MarketEvent values")
        order_key = (event.known_at, event.sequence_no, event.event_id)
        if last_input_order_key is not None and order_key <= last_input_order_key:
            raise ValueError("atomic replay event stream is out of canonical order")
        if event.event_id in input_event_ids:
            raise ValueError("atomic replay event stream repeats an event id")
        last_input_order_key = order_key
        input_event_ids.add(event.event_id)
        if event.kind is EventKind.FOUNDATION_STATE_CHANGED:
            record = foundation_record_from_projection_event(event)
            prior_record = legacy_foundation_records_by_id.get(record.record_id)
            if prior_record is not None:
                if prior_record != record:
                    raise ValueError("foundation record_id collision")
                retained_events.append(event)
                continue
            legacy_foundation_records_by_id[record.record_id] = record
            missing_sources = tuple(
                source_id
                for source_id in record.source_event_ids
                if source_id not in authoritative_events
            )
            if missing_sources:
                raise ValueError(
                    "foundation projection references unavailable earlier "
                    "source events: " + ", ".join(missing_sources)
                )
            future_sources = tuple(
                source_id
                for source_id in record.source_event_ids
                if authoritative_events[source_id].known_at > record.known_at
            )
            if future_sources:
                raise ValueError(
                    "foundation projection sources occur after embedded record "
                    "knowledge: " + ", ".join(future_sources)
                )
            _validate_foundation_authoritative_sources(
                record,
                authoritative_events,
                foundation,
            )
            foundation = FoundationProjectionReducer.reduce(
                FoundationProjectionReducer.initial_projection()
                if foundation is None
                else foundation,
                record,
            )
            retained_events.append(event)
            continue
        reducer.apply(event)
        if event.origin in {
            EventOrigin.NORMALIZED_DATA,
            EventOrigin.SEMANTIC_ATOMIC,
        }:
            authoritative_events[event.event_id] = event
        if (
            event.kind is EventKind.BAR_COMPLETED
            and event.origin is EventOrigin.NORMALIZED_DATA
            and event.evidence.get("real_completed") is True
            and event.evidence.get("clock_only") is False
        ):
            foundation_real_bar_ordinals[event.timeframe] = (
                foundation_real_bar_ordinals.get(event.timeframe, 0) + 1
            )
        if event.kind is EventKind.MARKET_EPOCH_RESET:
            session_reducer = SessionStateReducer()
            session = None
            latest_bar = None
            latest_real_bar = None
            retained_events.clear()
            epoch_scale_registry_id = None
            epoch_contract = None
            foundation_candidate_templates.clear()
            foundation_real_bar_ordinals.clear()
            continue
        if event.kind is EventKind.LIQUIDITY_LEVEL_CREATED:
            source_identity = event.evidence.get("level_id")
            if not isinstance(source_identity, str) or not source_identity:
                raise ValueError(
                    "atomic replay liquidity level lacks its source identity"
                )
            templates = tuple(
                candidate
                for state in reducer.states.values()
                for candidate in state.liquidity.candidates
                if candidate.candidate_id == source_identity
            )
            if len(templates) != 1:
                raise ValueError(
                    "atomic replay liquidity template is absent or ambiguous"
                )
            foundation_candidate_templates[source_identity] = (
                foundation_dol_candidate_template(templates[0])
            )
        if event.kind is EventKind.PROTECTED_SWING_ASSIGNED:
            protected_swing_id = event.evidence.get("protected_swing_id")
            if (
                not isinstance(protected_swing_id, str)
                or not protected_swing_id
            ):
                raise ValueError(
                    "atomic replay protected assignment lacks its swing identity"
                )
            source_identity = f"swing:{protected_swing_id}"
            template = foundation_candidate_templates.get(source_identity)
            if template is not None:
                foundation_candidate_templates[source_identity] = (
                    foundation_dol_protected_candidate_template(
                        template,
                        protected_swing_id=protected_swing_id,
                    )
                )
        retained_events.append(event)
        if (
            event.kind is EventKind.BAR_COMPLETED
            and event.origin is EventOrigin.NORMALIZED_DATA
        ):
            scale_registry_id = event.evidence.get("scale_registry_id")
            symbol = event.evidence.get("symbol")
            instrument_id = event.evidence.get("instrument_id")
            if (
                not isinstance(scale_registry_id, str)
                or not scale_registry_id
                or not isinstance(symbol, str)
                or not symbol
                or isinstance(instrument_id, bool)
                or not isinstance(instrument_id, int)
                or instrument_id < 0
            ):
                raise ValueError(
                    "atomic replay normalized BAR lacks stream identity"
                )
            contract = (symbol, instrument_id)
            if (
                epoch_scale_registry_id is not None
                and scale_registry_id != epoch_scale_registry_id
            ):
                raise ValueError(
                    "atomic replay scale registry changed without "
                    "MARKET_EPOCH_RESET"
                )
            if epoch_contract is not None and contract != epoch_contract:
                raise ValueError(
                    "atomic replay contract changed without "
                    "MARKET_EPOCH_RESET"
                )
            epoch_scale_registry_id = scale_registry_id
            epoch_contract = contract
        if (
            event.kind is not EventKind.BAR_COMPLETED
            or event.timeframe is not Timeframe.M1
        ):
            continue
        latest_bar = _m1_candle_from_event(event)
        if latest_bar.real_completed:
            latest_real_bar = latest_bar
        elif latest_real_bar is None:
            raise ValueError(
                "atomic replay clock-only M1 root requires a prior real "
                "M1 price in the current epoch"
            )
        session = session_reducer.update(latest_bar)
    if (
        session is None
        or latest_bar is None
        or latest_real_bar is None
        or not reducer.states
    ):
        raise ValueError(
            "atomic replay requires a current epoch with a completed 1m bar"
        )
    if any(event.known_at > session.known_at for event in retained_events):
        raise ValueError(
            "atomic replay ends after its latest completed 1m clock"
        )
    if reducer.expected_timeframes is not None:
        missing = tuple(
            timeframe
            for timeframe in reducer.expected_timeframes
            if timeframe not in reducer.states
        )
        if missing:
            raise ValueError(
                "atomic replay lacks bound timeframe state: "
                + ", ".join(timeframe.value for timeframe in missing)
            )
        unexpected = tuple(
            timeframe
            for timeframe in reducer.states
            if timeframe not in reducer.expected_timeframes
        )
        if unexpected:
            raise ValueError(
                "atomic replay contains unregistered timeframe state: "
                + ", ".join(timeframe.value for timeframe in unexpected)
            )
    cold_records = tuple(foundation_records)
    if cold_records:
        cold_projection = FoundationProjectionReducer.initial_projection()
        cold_records_by_id: dict[str, FoundationRecord] = {}
        for record in cold_records:
            if not isinstance(record, FoundationRecord):
                raise TypeError("atomic replay cold foundation ledger is invalid")
            prior_record = cold_records_by_id.get(record.record_id)
            if prior_record is not None:
                if prior_record != record:
                    raise ValueError("foundation record_id collision")
                continue
            cold_records_by_id[record.record_id] = record
            missing_sources = tuple(
                source_id
                for source_id in record.source_event_ids
                if source_id not in authoritative_events
            )
            if missing_sources:
                raise ValueError(
                    "cold foundation record references unavailable source events: "
                    + ", ".join(missing_sources)
                )
            future_sources = tuple(
                source_id
                for source_id in record.source_event_ids
                if authoritative_events[source_id].known_at > record.known_at
            )
            if future_sources:
                raise ValueError(
                    "cold foundation record sources occur after embedded "
                    "record knowledge: " + ", ".join(future_sources)
                )
            _validate_foundation_authoritative_sources(
                record,
                authoritative_events,
                cold_projection,
            )
            cold_projection = FoundationProjectionReducer.reduce(
                cold_projection,
                record,
            )
        cold_projection = FoundationProjectionReducer.validate_complete(
            cold_projection
        )
        if foundation is not None and foundation != cold_projection:
            raise ValueError(
                "legacy foundation transport differs from cold ledger replay"
            )
        foundation = cold_projection
    elif foundation is not None:
        foundation = FoundationProjectionReducer.validate_complete(foundation)
    states = foundation_dol_timeframe_states(
        foundation,
        states=FrozenDict(reducer.states),
        price=float(latest_real_bar.close),
        candidate_templates=foundation_candidate_templates,
        real_bar_ordinals=foundation_real_bar_ordinals,
    )
    relations = RelationResolver(
        edges=MarketSnapshotPublisher._RELATION_EDGES
    ).resolve(
        states,
        price=float(latest_real_bar.close),
        asof=session.known_at,
    )
    labels = tuple(state.label for state in states.values()) + tuple(
        f"{relation.relation_id} {relation.role.value}"
        for relation in relations.values()
    )
    return MarketSnapshot(
        asof=session.known_at,
        symbol=latest_bar.symbol,
        instrument_id=latest_bar.instrument_id,
        price=float(latest_real_bar.close),
        semantic_version=semantic_version,
        semantic_registry_identity=semantic_registry_identity,
        timeframe_states=states,
        relations=relations,
        session=session,
        events_this_update=(),
        labels=labels,
        authority=MarketSnapshotAuthority.ATOMIC_EVENT_REDUCER,
        foundation=foundation,
        foundation_range_locations=foundation_dual_range_locations(
            foundation,
            price=float(latest_real_bar.close),
            timeframes=states,
        ),
    )


__all__ = [
    "BalanceRangeState",
    "DOLCandidateView",
    "DeliveryPhase",
    "DualRangeLocation",
    "HierarchicalReplayState",
    "LiquidityClusterState",
    "LiquidityClusterSupersession",
    "LiquidityClusterUpdate",
    "LiquidityRangeRole",
    "MarketSnapshot",
    "MarketSnapshotAuthority",
    "MarketSnapshotPublisher",
    "MARKET_SNAPSHOT_SCHEMA_VERSION",
    "PriceZoneView",
    "RelationRole",
    "RelationResolver",
    "RelationState",
    "SessionState",
    "SessionStateReducer",
    "StructuralRangeState",
    "SwingGeometryAssignment",
    "SwingGeometryNode",
    "SwingHierarchyView",
    "SwingRankAssignment",
    "TimeframeEventReducer",
    "TimeframeDeliveryState",
    "TimeframeLiquidityState",
    "TimeframeQualityState",
    "TimeframeRangeState",
    "TimeframeState",
    "TimeframeStructureState",
    "TimeframeZoneState",
    "build_structural_range",
    "build_structural_legs",
    "build_swing_geometry_nodes",
    "dual_range_location",
    "foundation_record_from_projection_event",
    "foundation_dual_range_locations",
    "foundation_dol_candidate_template",
    "foundation_dol_protected_candidate_template",
    "foundation_dol_timeframe_states",
    "reduce_hierarchical_state",
    "reduce_timeframe_state",
    "replay_atomic_market_snapshot",
    "session_name_phase",
    "terminate_structural_range",
    "update_liquidity_clusters",
    "update_swing_geometry_assignments",
]
