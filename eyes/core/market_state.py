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

from .event_store import EventStore
from .foundation_registry import FOUNDATION_VERSION
from shares.core.market_clock import (
    expected_trading_minutes,
    next_registered_native_completion,
    registered_native_bar_bounds,
    validate_completed_bar_header,
    validate_registered_native_bar_root,
)
from contract.market import (
    Candle,
    Direction,
    FrozenDict,
    SMC_SEMANTIC_VERSION,
    Timeframe,
    aware_timestamp,
    candle_coverage,
    candle_identity,
    clamp,
    price_to_ticks,
    to_primitive,
)
from contract.eye import (
    BOSLifecycle,
    BOSScope,
    DealingRangeLifecycle,
    DealingRangeState,
    DisplacementObservation,
    EventKind,
    EventOrigin,
    FairValueGapLifecycle,
    FrameObservation,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    MarketEvent,
    OrderBlockLifecycle,
    StructuralLegState,
    StructureLifecycle,
    SwingLifecycle,
    SwingPoint,
    SwingRank,
    SwingSide,
)

if TYPE_CHECKING:
    from .semantic_foundation import FoundationRecord


MARKET_SNAPSHOT_SCHEMA_VERSION = 8
TIMEFRAME_EVENT_REDUCER_SCHEMA_VERSION = 3


class DeliveryPhase(str, Enum):
    BALANCE = "balance"
    EXPANSION = "expansion"
    RETRACEMENT = "retracement"
    REVERSAL_ATTEMPT = "reversal_attempt"
    TRANSITION = "transition"


class StructureScope(str, Enum):
    """Which structural claim a generation belongs to."""

    INTERNAL = "internal"
    EXTERNAL = "external"


@dataclass(frozen=True)
class StructureGenerationState:
    """One continuous structural claim on one timeframe and scope.

    A generation opens when a direction is confirmed, absorbs every qualified
    BOS and MSS core observed while its protected swing holds, and ends when
    that protection is accepted through or the direction reverses.  Consumers
    reference ``generation_id`` so that a phase, a relation or a DOL candidate
    spanning several breaks still names one parent.
    """

    generation_id: str
    timeframe: Timeframe
    scope: StructureScope
    direction: Direction
    started_at: pd.Timestamp
    confirmed_at: pd.Timestamp | None = None
    origin_event_id: str | None = None
    protected_swing_id: str | None = None
    bos_event_ids: tuple[str, ...] = ()
    mss_event_ids: tuple[str, ...] = ()
    terminated_at: pd.Timestamp | None = None
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "scope", StructureScope(self.scope))
        object.__setattr__(self, "direction", Direction(self.direction))
        for name in ("bos_event_ids", "mss_event_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("started_at", "confirmed_at", "terminated_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"structure_generation.{name}"),
                )
        if (
            not self.generation_id
            or (self.terminated_at is None) != (self.termination_reason is None)
            or (
                self.terminated_at is not None
                and self.terminated_at < self.started_at
            )
            or len(set(self.bos_event_ids)) != len(self.bos_event_ids)
            or len(set(self.mss_event_ids)) != len(self.mss_event_ids)
        ):
            raise ValueError("structure generation state is invalid")

    @property
    def is_active(self) -> bool:
        return self.terminated_at is None


@dataclass(frozen=True)
class DeliveryPhaseOccupancy:
    """One continuous stay in a delivery phase on one timeframe."""

    phase: DeliveryPhase
    entered_at: pd.Timestamp
    entered_bar_ordinal: int
    previous_phase: DeliveryPhase | None
    origin_event_id: str | None
    parent_structure_generation_id: str | None
    inputs: tuple[object, ...]
    observation_count: int = 1


@dataclass(frozen=True)
class DeliveryPhaseTransition:
    """What the Eye publishes about a phase occupancy on one bar."""

    kind: EventKind
    timeframe: Timeframe
    phase: DeliveryPhase
    known_at: pd.Timestamp
    price: float
    entered_at: pd.Timestamp
    age_bars: int
    observation_count: int
    previous_phase: DeliveryPhase | None
    next_phase: DeliveryPhase | None
    origin_event_id: str | None
    parent_structure_generation_id: str | None
    structure_regime: Direction | None
    active_leg_direction: Direction | None
    protected_swing_intact: bool | None
    range_available: bool
    source_event_ids: tuple[str, ...]


class RelationRole(str, Enum):
    ALIGNED_EXPANSION = "aligned_expansion"
    PARENT_RETRACEMENT = "parent_retracement"
    REVERSAL_ATTEMPT = "reversal_attempt"
    PARENT_TRANSITION = "parent_transition"
    BALANCE_INSIDE_PARENT = "balance_inside_parent"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class RelationGenerationState:
    """One continuous occupancy of a cross-timeframe relation.

    ``RelationState`` is recomputed on every completed minute, so a parent
    retracement that holds for fifty bars yields fifty identical rows.  Treating
    those as independent observations is pseudo-replication: it counts one
    market fact fifty times.  A generation is the experimental unit instead --
    it opens when a role is established under a named pair of structural
    claims, counts the observations it spanned, and closes when the role or
    either structural claim changes.
    """

    generation_id: str
    relation_id: str
    parent_tf: Timeframe
    child_tf: Timeframe
    role: RelationRole
    entered_at: pd.Timestamp
    updated_at: pd.Timestamp
    observation_count: int = 1
    parent_structure_generation_id: str | None = None
    child_structure_generation_id: str | None = None
    terminated_at: pd.Timestamp | None = None
    termination_reason: str | None = None
    next_role: RelationRole | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent_tf", Timeframe(self.parent_tf))
        object.__setattr__(self, "child_tf", Timeframe(self.child_tf))
        object.__setattr__(self, "role", RelationRole(self.role))
        if self.next_role is not None:
            object.__setattr__(self, "next_role", RelationRole(self.next_role))
        for name in ("entered_at", "updated_at", "terminated_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"relation_generation.{name}"),
                )
        if (
            not self.generation_id
            or not self.relation_id
            or self.parent_tf is self.child_tf
            or type(self.observation_count) is not int
            or self.observation_count < 1
            or self.updated_at < self.entered_at
            or (self.terminated_at is None) != (self.termination_reason is None)
            or (
                self.terminated_at is not None
                and self.terminated_at < self.entered_at
            )
            # A successor role is evidence of a role change and of nothing
            # else; a claim that ended because its structure ended has no
            # successor to name.
            or (
                self.next_role is not None
                and (
                    self.termination_reason != "role_changed"
                    or self.next_role is self.role
                )
            )
            or (
                self.termination_reason == "role_changed"
                and self.next_role is None
            )
        ):
            raise ValueError("relation generation state is invalid")

    @property
    def is_active(self) -> bool:
        return self.terminated_at is None

    @property
    def signature(self) -> tuple[str | None, str | None, str]:
        """What must hold for this occupancy to still be the same one."""

        return (
            self.parent_structure_generation_id,
            self.child_structure_generation_id,
            self.role.value,
        )


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

# How many confirmed Swings one timeframe keeps in reduced state.  The
# immutable event stream owns the complete role history; the current public
# state keeps only a bounded recent working set, exactly as ``structural_legs``
# does.  Two reads can still reach an older Swing, and both were measured over
# 2022-02: a later rank assignment reached at most 4 Swings back on its own
# timeframe, and the geometry tree's retroactive adoption at most 447 -- bounded
# by the span of the largest enabled timeframe's confirmation window rather than
# by elapsed bars.  The window carries a wide margin over both, so eviction only
# ever drops Swings that nothing reads.
SWING_HIERARCHY_HOT_RETENTION = 2048


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
    # The definitional bar window this Swing was confirmed from, frozen at
    # confirmation.  Geometric nesting is decided from these four numbers and
    # nothing else -- never from the role the Swing happens to be playing.
    window_start: pd.Timestamp | None = None
    window_end: pd.Timestamp | None = None
    window_low: float | None = None
    window_high: float | None = None
    geometric_parent_id: str | None = None
    geometric_depth: int = 0
    child_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(self, "semantic_rank", SwingRank(self.semantic_rank))
        object.__setattr__(self, "child_ids", tuple(self.child_ids))
        for name in ("window_start", "window_end"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"swing_hierarchy.{name}"),
                )
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
        window = (
            self.window_start,
            self.window_end,
            self.window_low,
            self.window_high,
        )
        if any(value is None for value in window) != all(
            value is None for value in window
        ):
            raise ValueError("swing hierarchy window is partially known")
        if (
            type(self.geometric_depth) is not int
            or self.geometric_depth < 0
            # A root has no parent and a child is never its own parent.
            or (self.geometric_parent_id is None) != (self.geometric_depth == 0)
            or self.geometric_parent_id == self.swing_id
            or self.swing_id in self.child_ids
            or len(set(self.child_ids)) != len(self.child_ids)
        ):
            raise ValueError("swing geometric nesting is invalid")
        if self.window_start is not None and (
            self.window_start >= self.window_end
            or not 0.0 < float(self.window_low) <= float(self.window_high)
        ):
            raise ValueError("swing hierarchy window is invalid")
        if self.window_start is None and self.geometric_parent_id is not None:
            raise ValueError("swing without a window cannot be nested")

    @property
    def role_depth(self) -> int:
        """Return causal semantic-role depth, never geometric nesting depth."""

        return self.nesting_depth

    @property
    def lower_bound(self) -> float | None:
        return self.window_low

    @property
    def upper_bound(self) -> float | None:
        return self.window_high

    @property
    def duration_seconds(self) -> int:
        return int((self.window_end - self.window_start).total_seconds())

    @property
    def price_span(self) -> float:
        return float(self.window_high - self.window_low)


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
    # Densified no-trade minutes inside the definitional window; see
    # ``StructuralLegState.synthetic_path_minutes``.
    synthetic_window_minutes: int = 0
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


# A level in one of these states still exists and still has an identity, but
# it is not currently resting liquidity: it was taken, and has not yet been
# offered again.
_DISARMED_LEVEL_LIFECYCLES = frozenset({"disarmed", "rearmable"})


@dataclass(frozen=True)
class DOLCandidateView:
    candidate_id: str
    timeframe: Timeframe
    side: str
    price: float
    source_kind: str
    lower_bound: float
    upper_bound: float
    formed_at: pd.Timestamp
    confirmed_at: pd.Timestamp
    rank: str = "internal"
    strength: float = 0.0
    lifecycle: str = "visible"
    age_bars: int = 0
    distance_atr: float | None = None
    source_event_id: str | None = None
    owner_id: str | None = None
    path_obstacle_ids: tuple[str, ...] = ()
    range_role: LiquidityRangeRole = LiquidityRangeRole.UNRESOLVED
    normalized_location_in_range: float | None = None
    # A level outlives the sweep that takes it.  The ordinal counts how many
    # times this same identity has been armed, so a level tested four times is
    # four generations of one level rather than four unrelated levels.
    generation_ordinal: int = 1
    disarmed_at: pd.Timestamp | None = None
    rearm_departure: float | None = None
    rearm_departure_bar_event_id: str | None = None

    @property
    def generation_id(self) -> str:
        return f"{self.candidate_id}_generation_{self.generation_ordinal:04d}"

    @property
    def is_armed(self) -> bool:
        """Whether this level is currently offered as resting liquidity."""

        return self.lifecycle not in _DISARMED_LEVEL_LIFECYCLES

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        if self.disarmed_at is not None:
            object.__setattr__(
                self,
                "disarmed_at",
                aware_timestamp(
                    self.disarmed_at, name="dol_candidate.disarmed_at"
                ),
            )
        object.__setattr__(
            self,
            "formed_at",
            aware_timestamp(self.formed_at, name="dol_candidate.formed_at"),
        )
        object.__setattr__(
            self,
            "confirmed_at",
            aware_timestamp(
                self.confirmed_at,
                name="dol_candidate.confirmed_at",
            ),
        )
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
            or not math.isfinite(float(self.lower_bound))
            or not math.isfinite(float(self.upper_bound))
            or not 0.0 < self.lower_bound <= self.price <= self.upper_bound
            or self.formed_at > self.confirmed_at
            or not self.source_kind
            or not self.rank
            or not math.isfinite(float(self.strength))
            or not 0.0 <= float(self.strength) <= 1.0
            or not self.lifecycle
            or (
                self.owner_id is not None
                and (
                    not isinstance(self.owner_id, str)
                    or not self.owner_id
                )
            )
            or type(self.age_bars) is not int
            or self.age_bars < 0
            or type(self.generation_ordinal) is not int
            or self.generation_ordinal < 1
            # A later generation only exists because an earlier one was swept.
            or (self.generation_ordinal > 1 and self.disarmed_at is None)
            or (self.rearm_departure is None)
            != (self.rearm_departure_bar_event_id is None)
            or (
                self.rearm_departure is not None
                and (
                    not math.isfinite(float(self.rearm_departure))
                    or float(self.rearm_departure) <= 0.0
                )
            )
            or (
                self.lifecycle in _DISARMED_LEVEL_LIFECYCLES
                and self.disarmed_at is None
            )
            or (
                self.lifecycle == "rearmable"
                and self.rearm_departure is None
            )
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
    event_count: int
    event_prefix_fingerprint: str
    # Keyed by generation_id so a phase, a relation or a DOL candidate can name
    # one continuous structural claim instead of its latest break.  A snapshot
    # rebuilt from the event log alone carries none.
    structure_generations: Mapping[str, StructureGenerationState] = field(
        default_factory=dict
    )
    # Keyed by generation_id.  One row per continuous relation occupancy, so a
    # study can use the episode as its experimental unit instead of the bar.
    relation_generations: Mapping[str, RelationGenerationState] = field(
        default_factory=dict
    )
    authority: MarketSnapshotAuthority = (
        MarketSnapshotAuthority.LEGACY_UNSPECIFIED
    )
    schema_version: int = MARKET_SNAPSHOT_SCHEMA_VERSION

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
        object.__setattr__(
            self,
            "structure_generations",
            FrozenDict(self.structure_generations),
        )
        object.__setattr__(
            self,
            "relation_generations",
            FrozenDict(self.relation_generations),
        )
        object.__setattr__(self, "events_this_update", tuple(self.events_this_update))
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(
            self,
            "authority",
            MarketSnapshotAuthority(self.authority),
        )
        if (
            not self.symbol
            or self.schema_version != MARKET_SNAPSHOT_SCHEMA_VERSION
            or self.instrument_id < 0
            or not math.isfinite(float(self.price))
            or not self.semantic_version
            or not self.semantic_registry_identity
            or type(self.event_count) is not int
            or self.event_count < 0
            or not isinstance(self.event_prefix_fingerprint, str)
            or len(self.event_prefix_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.event_prefix_fingerprint
            )
            or set(states) != {state.timeframe for state in states.values()}
            or any(relation.relation_id != key for key, relation in relations.items())
            or self.session.known_at != self.asof
            or any(event.known_at > self.asof for event in self.events_this_update)
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
            "event_count",
            "event_prefix_fingerprint",
            "structure_generations",
            "relation_generations",
            "authority",
            "schema_version",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != expected
            or state.get("schema_version") != MARKET_SNAPSHOT_SCHEMA_VERSION
        ):
            raise ValueError("market snapshot pickle schema changed")
        candidate = object.__new__(type(self))
        for name, value in state.items():
            object.__setattr__(candidate, name, value)
        candidate.__post_init__()
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)

    @property
    def fingerprint(self) -> str:
        primitive = dict(to_primitive(self.__dict__))
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
            "event_count": self.event_count,
            "event_prefix_fingerprint": self.event_prefix_fingerprint,
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
        EventKind.FVG_FIRST_RETEST,
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
        EventKind.BALANCE_RANGE_OBSERVED,
        EventKind.BALANCE_RANGE_MATURED,
        EventKind.ORIGIN_ZONE_CREATED,
        EventKind.BASE_ORIGIN_CORE_CREATED,
        EventKind.QUALIFIED_ORIGIN_ZONE_CREATED,
        EventKind.ORIGIN_ZONE_TOUCHED,
        EventKind.ORIGIN_ZONE_MITIGATED,
        EventKind.ORIGIN_ZONE_INVALIDATED,
        EventKind.DELIVERY_PHASE_ENTERED,
        EventKind.DELIVERY_PHASE_UPDATED,
        EventKind.DELIVERY_PHASE_EXITED,
    }
)

# The reducer accepts only the atomic kinds emitted by the v1.3 semantic
# registry plus the journal-level epoch reset.  Reserved compatibility kinds
# remain readable in EventStore but cannot acquire atomic authority here.
# FVG_FIRST_RETEST is admitted as an authoritative fact but has no branch
# below: it records what was true at the first re-entry and never revises the
# compact current view, which the fill observation on the same bar owns.
# BASE_ORIGIN_CORE_CREATED is admitted the same way: the compact view tracks
# the qualified origin zone, while the bare geometry stays readable as a fact
# the Brain interprets.  The retired ORIGIN_ZONE_CREATED keeps no authority.
_CANONICAL_ATOMIC_KINDS = frozenset(
    (
        _TIMEFRAME_REDUCER_KINDS
        - {
            EventKind.BAR_COMPLETED,
            EventKind.FVG_EXPIRED,
            EventKind.ORIGIN_ZONE_CREATED,
            EventKind.ORIGIN_ZONE_TOUCHED,
            EventKind.DEALING_RANGE_EXTENDED,
            EventKind.DEALING_RANGE_ACTIVATED,
        }
    )
    | {
        EventKind.RAW_BOUNDARY_BREAK,
        EventKind.MARKET_EPOCH_RESET,
        # Target outcomes are facts about an inventory item, published on
        # its own timeframe; the compact view already tracks the item, so
        # they are admitted without a state branch, like RAW_BOUNDARY_BREAK.
        EventKind.LEVEL_REACHED,
        EventKind.LEVEL_INVALIDATED,
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
    # ``candidates`` keeps every level the timeframe knows about, including the
    # ones a sweep disarmed; only armed levels are published as inventory.
    above = tuple(
        item for item in values if item.side == "above" and item.is_armed
    )
    below = tuple(
        item for item in values if item.side == "below" and item.is_armed
    )
    return TimeframeLiquidityState(
        unswept_bsl=tuple(item.price for item in above),
        unswept_ssl=tuple(item.price for item in below),
        candidate_bsl_ids=tuple(item.candidate_id for item in above),
        candidate_ssl_ids=tuple(item.candidate_id for item in below),
        recently_swept_ids=tuple(dict.fromkeys(recently_swept_ids)),
        candidates=values,
    )


def _advance_level_rearm(
    item: DOLCandidateView,
    *,
    close: float,
    bar_event_id: str,
) -> DOLCandidateView:
    """Carry a disarmed level towards being offered again.

    Two observable facts, no tunable threshold.  The level is *left behind*
    when a close is strictly outside its own band on the side it was offered
    from -- a swept high is left behind by closing below it -- and it is
    *offered again* on the first close that comes back inside that reach.  The
    recorded departure is the extreme, so a still-receding close extends the
    distance price must return from rather than arming the level early.
    """

    if item.lifecycle not in _DISARMED_LEVEL_LIFECYCLES:
        return item
    departure = (
        item.lower_bound - close
        if item.side == "above"
        else close - item.upper_bound
    )
    if item.lifecycle == "disarmed":
        if departure <= 0.0:
            return item
        return replace(
            item,
            lifecycle="rearmable",
            rearm_departure=float(departure),
            rearm_departure_bar_event_id=bar_event_id,
        )
    reach = float(item.rearm_departure)
    if departure > reach:
        return replace(
            item,
            rearm_departure=float(departure),
            rearm_departure_bar_event_id=bar_event_id,
        )
    if departure == reach:
        return item
    return replace(
        item,
        lifecycle="rearmed",
        generation_ordinal=item.generation_ordinal + 1,
    )


# A structural range locates price whenever it exists and price has not closed
# outside it.  The historical spellings are kept so journals written before the
# lifecycle lost its balance-derived states still reduce.
_LOCATING_RANGE_LIFECYCLES = frozenset(
    {"created", "forming", "extended", "replaced", "active", "mature"}
)


def _candidate_range_membership(
    candidate: DOLCandidateView,
    range_state: TimeframeRangeState,
) -> DOLCandidateView:
    """Classify a same-timeframe candidate against one active range."""

    active = bool(
        range_state.range_id is not None
        and range_state.range_kind == "active_dealing_range"
        # Location is the interval's own arithmetic, so a registered
        # structural range answers it from creation.  ``active``/``mature``
        # additionally assert the separate balance claim; requiring them here
        # made a two-sided-test statistic a precondition for premium/discount.
        and range_state.lifecycle in _LOCATING_RANGE_LIFECYCLES
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


def _swing_confirmation_window(
    event: MarketEvent,
) -> dict[str, object]:
    """Read the frozen definitional window off a confirmation event."""

    evidence = event.evidence
    if any(
        key not in evidence
        for key in ("window_start", "window_end", "window_low", "window_high")
    ):
        # Historical journals predate the frozen window; such a swing simply
        # has no geometry rather than an invented one.
        return {}
    return {
        "window_start": aware_timestamp(
            pd.Timestamp(evidence["window_start"]),
            name="swing_hierarchy.window_start",
        ),
        "window_end": aware_timestamp(
            pd.Timestamp(evidence["window_end"]),
            name="swing_hierarchy.window_end",
        ),
        "window_low": float(evidence["window_low"]),
        "window_high": float(evidence["window_high"]),
    }


class SwingGeometryTree:
    """The cross-timeframe geometric nesting tree, maintained incrementally.

    Re-deriving the whole tree on every bar is quadratic in a population that
    only grows: three days of real data spent 70% of the replay inside it.  The
    work is instead done once per confirmed Swing, and it is bounded by the new
    Swing's own window rather than by all history -- the only nodes a new
    window can adopt are the ones that fit inside it.

    Adoption has to be retroactive.  A Swing's enclosing higher-timeframe
    window is confirmed at or after the Swing it contains, so an append-only
    assignment would leave almost everything a root.
    """

    _JOURNALLED = ("_nodes", "_parent", "_children", "_depth", "_index", "_starts")

    __slots__ = (
        "_nodes",
        "_parent",
        "_children",
        "_depth",
        "_index",
        "_starts",
        "_undo",
    )

    def __init__(self) -> None:
        self._nodes: dict[str, SwingHierarchyView] = {}
        self._parent: dict[str, str | None] = {}
        self._children: dict[str, list[str]] = {}
        self._depth: dict[str, int] = {}
        # Per timeframe, swing ids ordered by window_start.  Windows on one
        # timeframe all share a length, so this orders window_end too.
        self._index: dict[Timeframe, list[str]] = {}
        # The same order as ``_index``, holding the window starts themselves so
        # the bisect that bounds every search never rebuilds them from nodes.
        self._starts: dict[Timeframe, list[pd.Timestamp]] = {}
        self._undo: list[tuple[str, object, bool, object]] | None = None

    def clear(self) -> None:
        self._record_whole()
        self._nodes.clear()
        self._parent.clear()
        self._children.clear()
        self._depth.clear()
        self._index.clear()
        self._starts.clear()

    def begin(self) -> None:
        """Start journalling so one publish attempt can be undone.

        A bar changes a bounded number of entries, so recording what it is
        about to overwrite costs what the bar changed.  Copying the containers
        instead costs the whole Swing population on every bar, including the
        overwhelming majority of bars that confirm nothing.
        """

        self._undo = []

    def commit(self) -> None:
        self._undo = None

    def rollback(self) -> None:
        if self._undo is None:
            return
        for name, key, present, value in reversed(self._undo):
            container = getattr(self, name)
            if key is None:
                setattr(self, name, value)
            elif present:
                container[key] = value
            else:
                container.pop(key, None)
        self._undo = None

    def _record(self, name: str, key: object) -> None:
        if self._undo is None:
            return
        container = getattr(self, name)
        present = key in container
        value = container[key] if present else None
        self._undo.append(
            (name, key, present, list(value) if type(value) is list else value)
        )

    def _record_whole(self) -> None:
        """An epoch reset replaces everything, so nothing narrower undoes it."""

        if self._undo is None:
            return
        for name in self._JOURNALLED:
            container = getattr(self, name)
            self._undo.append(
                (
                    name,
                    None,
                    True,
                    {
                        key: list(value) if type(value) is list else value
                        for key, value in container.items()
                    },
                )
            )

    def view_of(self, swing: SwingHierarchyView) -> SwingHierarchyView:
        """Return ``swing`` carrying its settled place in the tree."""

        parent_id = self._parent.get(swing.swing_id)
        depth = self._depth.get(swing.swing_id, 0)
        children = tuple(sorted(self._children.get(swing.swing_id, ())))
        if (
            swing.geometric_parent_id == parent_id
            and swing.geometric_depth == depth
            and swing.child_ids == children
        ):
            return swing
        return replace(
            swing,
            geometric_parent_id=parent_id,
            geometric_depth=depth,
            child_ids=children,
        )

    def _tightest(self, left: str | None, right: str) -> str:
        if left is None:
            return right
        return min(
            (left, right),
            key=lambda name: _tightest_parent_key(self._nodes[name]),
        )

    def _find_parent(self, child: SwingHierarchyView) -> str | None:
        best: str | None = None
        for timeframe, order in self._index.items():
            if not order:
                continue
            starts = self._starts[timeframe]
            cursor = bisect_right(starts, child.window_start) - 1
            while cursor >= 0:
                candidate = self._nodes[order[cursor]]
                # Equal-length windows on one timeframe end in the same order
                # they start, so once a candidate ends too early none earlier
                # can reach this child.
                if candidate.window_end < child.window_end:
                    break
                if candidate.swing_id != child.swing_id and _window_contains(
                    candidate, child
                ):
                    best = self._tightest(best, candidate.swing_id)
                cursor -= 1
        return best

    def _adoptable(self, parent: SwingHierarchyView) -> list[str]:
        """Ids whose window fits inside ``parent``: a contiguous slice."""

        adopted: list[str] = []
        for timeframe, order in self._index.items():
            if not order:
                continue
            starts = self._starts[timeframe]
            cursor = bisect_left(starts, parent.window_start)
            while cursor < len(order):
                candidate = self._nodes[order[cursor]]
                if candidate.window_end > parent.window_end:
                    break
                if candidate.swing_id != parent.swing_id and _window_contains(
                    parent, candidate
                ):
                    adopted.append(candidate.swing_id)
                cursor += 1
        return adopted

    def _reparent(self, child_id: str, parent_id: str | None) -> None:
        incumbent = self._parent.get(child_id)
        if incumbent == parent_id:
            return
        if incumbent is not None:
            siblings = self._children.get(incumbent)
            if siblings is not None and child_id in siblings:
                self._record("_children", incumbent)
                siblings.remove(child_id)
        self._record("_parent", child_id)
        self._parent[child_id] = parent_id
        if parent_id is not None:
            self._record("_children", parent_id)
            self._children.setdefault(parent_id, []).append(child_id)

    def _resettle_depths(self, roots: Iterable[str]) -> set[str]:
        """Repair depths below ``roots``; the subtree bounds the work."""

        touched: set[str] = set()
        pending = list(roots)
        while pending:
            swing_id = pending.pop()
            parent_id = self._parent.get(swing_id)
            depth = 0 if parent_id is None else self._depth.get(parent_id, 0) + 1
            if self._depth.get(swing_id) == depth and swing_id in touched:
                continue
            self._record("_depth", swing_id)
            self._depth[swing_id] = depth
            touched.add(swing_id)
            pending.extend(self._children.get(swing_id, ()))
        return touched

    def admit(self, swing: SwingHierarchyView) -> set[str]:
        """Place one newly confirmed Swing; return every id whose place moved."""

        if swing.window_start is None or swing.swing_id in self._nodes:
            return set()
        self._record("_nodes", swing.swing_id)
        self._nodes[swing.swing_id] = swing
        self._record("_index", swing.timeframe)
        self._record("_starts", swing.timeframe)
        order = self._index.setdefault(swing.timeframe, [])
        starts = self._starts.setdefault(swing.timeframe, [])
        position = bisect_right(starts, swing.window_start)
        order.insert(position, swing.swing_id)
        starts.insert(position, swing.window_start)
        self._record("_children", swing.swing_id)
        self._children.setdefault(swing.swing_id, [])

        moved = {swing.swing_id}
        self._reparent(swing.swing_id, self._find_parent(swing))
        for candidate_id in self._adoptable(swing):
            incumbent = self._parent.get(candidate_id)
            if incumbent is not None and (
                _tightest_parent_key(self._nodes[incumbent])
                <= _tightest_parent_key(swing)
            ):
                continue
            self._reparent(candidate_id, swing.swing_id)
            moved.add(candidate_id)
            if incumbent is not None:
                moved.add(incumbent)
        return moved | self._resettle_depths(moved)


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
                **_swing_confirmation_window(event),
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
    if len(by_id) > SWING_HIERARCHY_HOT_RETENTION:
        # Confirmation order, not identifier order: a Swing leaves the working
        # set because it is old, and its identifier says nothing about when it
        # was confirmed.
        by_id = {
            item.swing_id: item
            for item in sorted(
                by_id.values(),
                key=lambda item: (item.assignments[0].assigned_at, item.swing_id),
            )[-SWING_HIERARCHY_HOT_RETENTION:]
        }
    return tuple(sorted(by_id.values(), key=lambda item: item.swing_id))


def _project_candidate_views(
    liquidity: TimeframeLiquidityState,
    hierarchy: tuple[SwingHierarchyView, ...],
    range_state: TimeframeRangeState,
) -> TimeframeLiquidityState:
    """Refresh both snapshot-derived candidate projections in one pass.

    Semantic rank and IRL/ERL membership are read off the state, never folded
    into it by an event: the registry binds them as "snapshot-derived only ...
    no canonical membership event is claimed".  Running them at the tail of
    every reduced event walked the whole candidate collection once per event
    and re-sorted it twice, and that collection grows with market history, so
    the cost of one bar grew for the life of the run.

    Both projections read ``price`` and write neither it nor ``lifecycle``, so
    the armed inventory ``_liquidity_state`` derives -- and the order it
    derives it in -- is the same whether they run once or ten times.
    """

    ranks = {item.swing_id: item.semantic_rank.value for item in hierarchy}
    return _liquidity_state(
        (
            _candidate_range_membership(
                replace(
                    candidate,
                    rank=ranks.get(
                        candidate.candidate_id.removeprefix("swing:"),
                        candidate.rank,
                    ),
                )
                if ranks and candidate.source_kind == "confirmed_swing"
                else candidate,
                range_state,
            )
            for candidate in liquidity.candidates
        ),
        liquidity.recently_swept_ids,
    )


def _settled_candidate_state(state: TimeframeState) -> TimeframeState:
    """Return ``state`` carrying the projections a reader sees.

    Every path that materializes a published hierarchy goes through here --
    the publisher and the cold replay from the atomic log alike -- so a
    checkpoint-restored view and a log-rebuilt one cannot disagree about a
    field the reducer no longer folds in.
    """

    liquidity = _project_candidate_views(
        state.liquidity,
        state.swing_hierarchy,
        state.range,
    )
    if liquidity == state.liquidity:
        return state
    return replace(state, liquidity=liquidity)


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
        path_class=SwingRank(str(evidence["path_class"])),
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
            _advance_level_rearm(
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
                ),
                close=close,
                bar_event_id=event.event_id,
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
        lower_bound, upper_bound = (
            (float(event.price), float(event.price))
            if event.zone is None
            else (float(event.zone[0]), float(event.zone[1]))
        )
        formed_at = aware_timestamp(
            event.evidence.get("source_formed_at", event.event_time),
            name="liquidity_level_created.source_formed_at",
        )
        confirmed_at = aware_timestamp(
            event.evidence.get("source_confirmed_at", event.known_at),
            name="liquidity_level_created.source_confirmed_at",
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
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    formed_at=formed_at,
                    confirmed_at=confirmed_at,
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
                    owner_id=(
                        str(range_id)
                        if source_kind == "mature_range_boundary"
                        and isinstance(
                            range_id := event.evidence.get("range_id"),
                            str,
                        )
                        and range_id
                        else None
                    ),
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
                # A disarmed level is not re-armed by being touched; only a
                # recorded departure and return can offer it again.
                replace(item, lifecycle=lifecycle)
                if item.candidate_id == level_id and item.is_armed
                else item
                for item in liquidity.candidates
            ),
            liquidity.recently_swept_ids,
        )
    elif event.kind is EventKind.SWEEP_CONFIRMED:
        level_id = str(event.evidence.get("level_id", ""))
        # The sweep closes this generation of the level; the level itself
        # survives, disarmed, so a later re-approach is recognisably the same
        # level rather than a brand-new identity at the same price.
        retained = tuple(
            replace(
                item,
                lifecycle="disarmed",
                disarmed_at=event.known_at,
                rearm_departure=None,
                rearm_departure_bar_event_id=None,
            )
            if item.candidate_id == level_id
            else item
            for item in liquidity.candidates
        )
        swept = liquidity.recently_swept_ids
        if level_id:
            swept = tuple(
                dict.fromkeys((*swept, level_id))
            )[-32:]
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
        EventKind.QUALIFIED_ORIGIN_ZONE_CREATED,
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
        EventKind.BALANCE_RANGE_MATURED,
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
                # The compact view calls a range active once its balance claim
                # matures; the structural interval itself never "activates".
                EventKind.BALANCE_RANGE_MATURED: "active",
                EventKind.DEALING_RANGE_EXTENDED: "extended",
                EventKind.DEALING_RANGE_REPLACED: "replaced",
            }[event.kind],
            source_ids=event.source_event_ids or (event.event_id,),
            range_kind="active_dealing_range",
        )
    elif event.kind is EventKind.DEALING_RANGE_INVALIDATED:
        if range_state.range_id == event.evidence.get("range_id"):
            range_state = replace(range_state, lifecycle="invalidated")
        invalidated_range_id = event.evidence.get("range_id")
        if isinstance(invalidated_range_id, str) and invalidated_range_id:
            liquidity = _liquidity_state(
                tuple(
                    candidate
                    for candidate in liquidity.candidates
                    if candidate.owner_id != invalidated_range_id
                ),
                liquidity.recently_swept_ids,
            )

    # The candidate projections are deliberately absent here: they are read
    # from the published state, so ``_settle_candidate_views`` runs them once
    # per bar in ``_publish_committed_suffix`` instead of once per event.
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


# The delivery-phase lifecycle is derived from the snapshot the publisher has
# just built, so it can only be appended after that publication.  It is a
# physical fact, not projection transport, and it is consumed in the same
# update so the reducer cursor still covers every committed physical event.
_DELIVERY_PHASE_TAIL_KINDS = frozenset(
    {
        EventKind.DELIVERY_PHASE_ENTERED,
        EventKind.DELIVERY_PHASE_UPDATED,
        EventKind.DELIVERY_PHASE_EXITED,
    }
)

_CURRENT_STATE_PROJECTION_KINDS = frozenset(
    {
        EventKind.TIMEFRAME_STATE_CHANGED,
        EventKind.RELATION_STATE_CHANGED,
        EventKind.SESSION_STATE_CHANGED,
    }
)


class TimeframeEventReducer:
    """Current-state reducer consuming the one authoritative EventStore."""

    def __init__(
        self,
        *,
        event_store: EventStore,
        semantic_registry_identity: str,
        semantic_version: str = SMC_SEMANTIC_VERSION,
        expected_timeframes: Iterable[Timeframe] | None = None,
    ) -> None:
        if not isinstance(event_store, EventStore):
            raise TypeError("timeframe reducer requires an EventStore")
        if not semantic_registry_identity:
            raise ValueError("timeframe reducer requires definition identity")
        if not semantic_version or event_store.semantic_version != semantic_version:
            raise ValueError("timeframe reducer semantic version differs from store")
        self.event_store = event_store
        self.semantic_registry_identity = semantic_registry_identity
        self.semantic_version = semantic_version
        self.states: dict[Timeframe, TimeframeState] = {}
        self._cursor = 0
        self._epoch_cursor = 0
        self._last_order_key: tuple[pd.Timestamp, int, str] | None = None
        self._latest_real_m1_event_id: str | None = None
        self._expected_timeframes: tuple[Timeframe, ...] | None = None
        self._store_prefix_fingerprint = event_store.fingerprint()
        self._reducer_schema_version = TIMEFRAME_EVENT_REDUCER_SCHEMA_VERSION
        if expected_timeframes is not None:
            self.bind_timeframes(expected_timeframes)

    @property
    def expected_timeframes(self) -> tuple[Timeframe, ...] | None:
        return self._expected_timeframes

    @property
    def cursor(self) -> int:
        return self._cursor

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
                "timeframe reducer already contains state outside its registry: "
                + ", ".join(item.value for item in unexpected_states)
            )
        if self._expected_timeframes is not None and normalized != self._expected_timeframes:
            raise ValueError("timeframe reducer registry changed within one stream")
        if self._expected_timeframes is None:
            self._expected_timeframes = normalized
        return self._expected_timeframes

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_store_prefix_fingerprint"] = (
            self.event_store.prefix_fingerprint(self._cursor)
        )
        state["_reducer_schema_version"] = TIMEFRAME_EVENT_REDUCER_SCHEMA_VERSION
        return state

    def __setstate__(self, state: Mapping[str, object]) -> None:
        expected = {
            "event_store",
            "semantic_registry_identity",
            "semantic_version",
            "states",
            "_cursor",
            "_epoch_cursor",
            "_last_order_key",
            "_latest_real_m1_event_id",
            "_expected_timeframes",
            "_store_prefix_fingerprint",
            "_reducer_schema_version",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != expected
            or state.get("_reducer_schema_version")
            != TIMEFRAME_EVENT_REDUCER_SCHEMA_VERSION
        ):
            raise ValueError("timeframe reducer checkpoint schema changed")
        candidate = object.__new__(type(self))
        candidate.__dict__ = dict(state)
        candidate._require_checkpoint_state()
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)

    def _require_checkpoint_state(self) -> None:
        if (
            not isinstance(self.event_store, EventStore)
            or self.event_store.semantic_version != self.semantic_version
            or type(self._cursor) is not int
            or not 0 <= self._cursor <= len(self.event_store)
            or type(self._epoch_cursor) is not int
            or not 0 <= self._epoch_cursor <= self._cursor
            or self._store_prefix_fingerprint
            != self.event_store.prefix_fingerprint(self._cursor)
            or not isinstance(self.states, dict)
        ):
            raise ValueError("timeframe reducer checkpoint is not store-bound")
        _ = self.latest_real_m1_event
        # Checkpoint restore is the explicit cold-validation boundary.  The
        # hot reducer never scans history; restore proves that compact state
        # is exactly reproducible from the committed prefix.
        prefix_store = EventStore.from_events(
            self.event_store.events_since(0)[: self._cursor],
            semantic_version=self.semantic_version,
            definition_identity=(
                self.event_store.definition_identity
                or self.event_store.semantic_definition_identity
            ),
        )
        verifier = TimeframeEventReducer(
            event_store=prefix_store,
            semantic_registry_identity=self.semantic_registry_identity,
            semantic_version=self.semantic_version,
            expected_timeframes=self._expected_timeframes,
        )
        verifier.consume_available()
        if (
            verifier.states != self.states
            or verifier._epoch_cursor != self._epoch_cursor
            or verifier._last_order_key != self._last_order_key
            or verifier._latest_real_m1_event_id
            != self._latest_real_m1_event_id
        ):
            raise ValueError("timeframe reducer compact checkpoint failed replay")

    @property
    def latest_real_m1_event(self) -> MarketEvent | None:
        if self._latest_real_m1_event_id is None:
            return None
        event = self.event_store.get(self._latest_real_m1_event_id)
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

    def consume_available(
        self,
        *,
        projection_only: bool = False,
        delivery_phase_only: bool = False,
    ) -> tuple[MarketEvent, ...]:
        """Consume one committed suffix atomically without scanning history."""

        if type(projection_only) is not bool:
            raise TypeError("projection-only suffix flag must be boolean")
        if type(delivery_phase_only) is not bool:
            raise TypeError("delivery-phase suffix flag must be boolean")
        if projection_only and delivery_phase_only:
            raise ValueError("a suffix is either projection or delivery phase")
        suffix = self.event_store.events_since(self._cursor)
        if not suffix:
            self._store_prefix_fingerprint = self.event_store.fingerprint()
            return ()
        # Bypass pickle/copy hooks: checkpoint validation intentionally cold
        # replays a prefix, whereas the hot transaction must clone only this
        # bounded state overlay.
        staged = object.__new__(type(self))
        staged.__dict__ = dict(self.__dict__)
        staged.states = dict(self.states)
        for offset, event in enumerate(suffix, start=self._cursor):
            if projection_only and not (
                event.origin is EventOrigin.STATE_PROJECTION
                and event.kind in _CURRENT_STATE_PROJECTION_KINDS
            ):
                raise ValueError(
                    "timeframe reducer projection tail contains a physical fact"
                )
            if delivery_phase_only and not (
                event.origin is EventOrigin.SEMANTIC_ATOMIC
                and event.kind in _DELIVERY_PHASE_TAIL_KINDS
            ):
                raise ValueError(
                    "timeframe reducer delivery tail contains a foreign event"
                )
            staged._consume_committed(event, store_index=offset)
        staged._cursor = self._cursor + len(suffix)
        staged._store_prefix_fingerprint = self.event_store.fingerprint()
        self.states = staged.states
        self._cursor = staged._cursor
        self._epoch_cursor = staged._epoch_cursor
        self._last_order_key = staged._last_order_key
        self._latest_real_m1_event_id = staged._latest_real_m1_event_id
        self._expected_timeframes = staged._expected_timeframes
        self._store_prefix_fingerprint = staged._store_prefix_fingerprint
        return suffix

    def _consume_committed(self, event: MarketEvent, *, store_index: int) -> None:
        if self.event_store.get(event.event_id) is not event:
            raise ValueError("timeframe reducer accepts only committed store objects")
        if (
            self.event_store.recompute_event_digest(event)
            != self.event_store.event_digest(event.event_id)
        ):
            raise ValueError("timeframe reducer rejected tampered store bytes")
        if event.semantic_version != self.semantic_version:
            raise ValueError("timeframe reducer cannot mix semantic versions")
        order_key = (event.known_at, event.sequence_no, event.event_id)
        if self._last_order_key is not None and order_key <= self._last_order_key:
            raise ValueError("timeframe reducer stream is out of canonical order")
        self._last_order_key = order_key

        if event.origin is EventOrigin.STATE_PROJECTION:
            if (
                event.kind not in _CURRENT_STATE_PROJECTION_KINDS
                and event.kind is not EventKind.FOUNDATION_STATE_CHANGED
            ):
                raise ValueError("unknown state projection transport kind")
            return
        if event.kind is EventKind.FOUNDATION_STATE_CHANGED:
            # This retired transport is interpreted only by its dedicated
            # cold decoder.  It never changes the current MarketState view.
            return
        if event.kind in _DELIVERY_PHASE_TAIL_KINDS:
            # The delivery-phase lifecycle is derived from the state this
            # reducer publishes.  Reducing it back in would be circular, and
            # counting it would make a clock-only minute look like it carried
            # market events.  The cursor still advances past it, so the
            # checkpoint contract keeps covering every committed fact.
            return
        if event.origin is EventOrigin.LEGACY_TRANSPORT:
            if (
                event.kind in _CANONICAL_ATOMIC_KINDS
                or event.kind in {
                    EventKind.BAR_COMPLETED,
                }
            ):
                raise ValueError(
                    "canonical current-state kind has noncanonical origin"
                )
            return
        if event.origin is EventOrigin.NORMALIZED_DATA:
            if event.kind is not EventKind.BAR_COMPLETED:
                raise ValueError(
                    "normalized-data origin is reserved for BAR roots"
                )
        elif event.origin is EventOrigin.SEMANTIC_ATOMIC:
            if event.kind not in _CANONICAL_ATOMIC_KINDS:
                raise ValueError(
                    "semantic-atomic current-state kind is not registry-emitted"
                )
        else:  # pragma: no cover - EventOrigin is closed.
            raise ValueError("current-state event origin is unsupported")

        if (
            event.kind is EventKind.BAR_COMPLETED
            and event.origin is not EventOrigin.NORMALIZED_DATA
        ):
            raise ValueError("BAR root has non-normalized origin")
        if (
            event.kind is not EventKind.BAR_COMPLETED
            and event.kind is not EventKind.MARKET_EPOCH_RESET
            and event.origin is not EventOrigin.SEMANTIC_ATOMIC
        ):
            raise ValueError("current semantic fact has noncanonical origin")
        if (
            event.kind is EventKind.MARKET_EPOCH_RESET
            and event.origin is not EventOrigin.SEMANTIC_ATOMIC
        ):
            raise ValueError("market epoch reset has noncanonical origin")

        staged_expected_timeframes = self._expected_timeframes
        if event.kind is EventKind.BAR_COMPLETED and event.timeframe is Timeframe.M1:
            declared_timeframes = event.evidence.get("active_timeframes")
            if declared_timeframes is not None:
                if isinstance(declared_timeframes, (str, bytes)):
                    raise ValueError("timeframe reducer registry must be a sequence")
                supplied = tuple(Timeframe(value) for value in declared_timeframes)
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
                        + ", ".join(item.value for item in unexpected_states)
                    )
                if staged_expected_timeframes is not None and normalized != staged_expected_timeframes:
                    raise ValueError("timeframe reducer registry changed within one stream")
                staged_expected_timeframes = normalized
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
            self.states.clear()
            self._latest_real_m1_event_id = None
            self._epoch_cursor = store_index + 1
            self._expected_timeframes = staged_expected_timeframes
            return

        source_timeframe_value = event.evidence.get("source_timeframe")
        source_timeframe = (
            None if source_timeframe_value is None else Timeframe(str(source_timeframe_value))
        )
        current = reduce_timeframe_state(
            self.states.get(event.timeframe),
            event,
            semantic_registry_identity=self.semantic_registry_identity,
        )
        if current is not None:
            self.states[event.timeframe] = current
        if event.kind is EventKind.BAR_COMPLETED and event.timeframe is Timeframe.M1:
            owner_timeframes = (
                tuple(self.states)
                if staged_expected_timeframes is None
                else staged_expected_timeframes
            )
            for owner_timeframe in owner_timeframes:
                if owner_timeframe is Timeframe.M1:
                    continue
                owner_state = reduce_timeframe_state(
                    self.states.get(owner_timeframe),
                    replace(event, timeframe=owner_timeframe),
                    semantic_registry_identity=self.semantic_registry_identity,
                    _m1_owner_fanout=True,
                )
                if owner_state is None:
                    raise RuntimeError("current-price event was ignored by owner reducer")
                self.states[owner_timeframe] = owner_state
        if (
            event.kind in {
                EventKind.LEVEL_TOUCHED,
                EventKind.LEVEL_PENETRATED,
                EventKind.SWEEP_CONFIRMED,
                EventKind.ACCEPTANCE_CONFIRMED,
            }
            and source_timeframe is not None
            and source_timeframe is not event.timeframe
            and source_timeframe in self.states
        ):
            owner_state = reduce_timeframe_state(
                self.states[source_timeframe],
                replace(event, timeframe=source_timeframe),
                semantic_registry_identity=self.semantic_registry_identity,
            )
            if owner_state is None:
                raise RuntimeError("candidate-owner event was ignored by its reducer")
            self.states[source_timeframe] = owner_state
        if (
            event.kind is EventKind.BAR_COMPLETED
            and event.timeframe is Timeframe.M1
            and event.origin is EventOrigin.NORMALIZED_DATA
            and event.evidence.get("real_completed") is True
            and event.evidence.get("clock_only") is False
        ):
            self._latest_real_m1_event_id = event.event_id
        self._expected_timeframes = staged_expected_timeframes


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
        raise ValueError(f"{object_name} lacks native-duration BARs")
    for candle in candles:
        try:
            registered_start, registered_end = registered_native_bar_bounds(
                candle.start,
                timeframe_minutes=expected_minutes,
                anchor_minute=_NATIVE_TIMEFRAME_ANCHOR_MINUTES[timeframe],
            )
        except ValueError as error:
            raise ValueError(
                f"{object_name} lacks native-duration BARs"
            ) from error
        registered_minutes = expected_trading_minutes(
            registered_start,
            registered_end,
        )
        if (
            candle.timeframe is not timeframe
            or not candle_coverage(candle).admits_definitional_path
            or candle.start != registered_start
            or candle.end != registered_end
            or candle.expected_minutes != registered_minutes
            or candle.observed_minutes != registered_minutes
        ):
            raise ValueError(f"{object_name} lacks native-duration BARs")
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
                    if candle_coverage(candle).admits_definitional_path
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
                synthetic_window_minutes=sum(
                    int(candle.synthetic_minutes) for candle in window
                ),
            )
        )
    return tuple(values)


def _window_contains(parent, child) -> bool:
    """The whole geometric rule: containment in time and in price.

    Nothing about direction, rank, role or outcome enters here.  A strictly
    longer window is required so that two coincident windows can never adopt
    each other and no cycle is representable.
    """

    return (
        parent.window_start <= child.window_start
        and parent.window_end >= child.window_end
        and parent.lower_bound <= child.lower_bound
        and parent.upper_bound >= child.upper_bound
        and parent.duration_seconds > child.duration_seconds
    )


def _tightest_parent_key(node) -> tuple[int, float, str]:
    """The nearest enclosing window wins; ties break on identity."""

    return (node.duration_seconds, node.price_span, node.swing_id)


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
        and _window_contains(parent, child)
    )
    return None if not candidates else min(candidates, key=_tightest_parent_key)


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
        if (
            (incumbent is None)
            != (assignment.supersedes_assignment_id is None)
            or (
                incumbent is not None
                and (
                    assignment.supersedes_assignment_id
                    != incumbent.assignment_id
                    or assignment.assigned_at < incumbent.assigned_at
                )
            )
        ):
            raise ValueError("swing assignment history is not append-only")
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
                if candle_coverage(candle).admits_definitional_path
                and candle.timeframe is timeframe
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
    # ATR ancestry admits real bars only, and every leg asks for the same
    # prefix of them.  Applying the shared coverage rule once here turns a
    # per-leg scan of the whole history into a bisect on this array.
    atr_by_end = tuple(
        candle
        for candle in native_by_end
        if candle_coverage(candle).admits_atr_window
    )
    atr_ends = tuple(candle.end for candle in atr_by_end)
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
        prior_candles = atr_by_end[
            : bisect_right(atr_ends, anchor.pivot_start)
        ]
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
        path_class = SwingRank.MICRO if len(path) <= 2 else SwingRank.INTERNAL
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
                "synthetic_path_minutes": sum(
                    int(candle.synthetic_minutes) for candle in path
                ),
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
                path_class=path_class,
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
    expected = _projection_event(
        EventKind.FOUNDATION_STATE_CHANGED,
        asof=event.known_at,
        timeframe=event.timeframe,
        state_id=record.record_id,
        state=record,
        source_event_ids=record.source_event_ids,
        primitive_state=primitive,
    )
    expected_evidence = {
        **dict(expected.evidence),
        "canonical_semantic": False,
        "technical_projection": "foundation_record",
    }
    expected = replace(
        expected,
        details=expected_evidence,
        evidence=expected_evidence,
        sequence_no=event.sequence_no,
        event_time=record.known_at,
    )
    if event != expected:
        raise ValueError("foundation projection event does not exactly bind its record")
    return record


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
    _STATE_SCHEMA_VERSION = 8
    _PICKLE_FIELDS = frozenset(
        {
            "semantic_registry_identity",
            "atomic_authority",
            "_event_reducer",
            "_relation_resolver",
            "_session",
            "_last_projection_payloads",
            "_last_projection_event_ids",
            "_formal_structures",
            "_last_real_m1_price",
            "_last_real_m1_event_id",
            "_boundary_reset_pending",
            "_delivery_occupancies",
            "_structure_generations",
            "_structure_generation_ordinals",
            "_relation_generations",
            "_relation_generation_ordinals",
            "_swing_geometry",
            "_settled_hierarchies",
            "_retired_relation_generations",
            "_real_m1_bar_ordinal",
            "_publisher_state_schema_version",
        }
    )

    def __init__(
        self,
        *,
        event_store: EventStore,
        semantic_registry_identity: str,
        atomic_authority: bool = False,
    ) -> None:
        if not semantic_registry_identity:
            raise ValueError("snapshot publisher requires semantic registry identity")
        if type(atomic_authority) is not bool:
            raise ValueError("snapshot publisher authority flag must be boolean")
        if not isinstance(event_store, EventStore):
            raise TypeError("snapshot publisher requires the shared EventStore")
        self.semantic_registry_identity = semantic_registry_identity
        self.atomic_authority = atomic_authority
        self._event_reducer = TimeframeEventReducer(
            event_store=event_store,
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
        self._delivery_occupancies: dict[
            Timeframe,
            DeliveryPhaseOccupancy,
        ] = {}
        self._structure_generations: dict[
            tuple[Timeframe, StructureScope],
            StructureGenerationState,
        ] = {}
        self._structure_generation_ordinals: dict[
            tuple[Timeframe, StructureScope],
            int,
        ] = {}
        self._relation_generations: dict[str, RelationGenerationState] = {}
        self._relation_generation_ordinals: dict[str, int] = {}
        self._swing_geometry = SwingGeometryTree()
        self._settled_hierarchies: dict[
            Timeframe, tuple[SwingHierarchyView, ...]
        ] = {}
        # Closed on the current bar only, so a terminated occupancy is
        # published exactly once alongside the successor that replaced it.
        self._retired_relation_generations: tuple[
            RelationGenerationState,
            ...,
        ] = ()
        self._real_m1_bar_ordinal = 0
        self._publisher_state_schema_version = self._STATE_SCHEMA_VERSION

    @property
    def event_store(self) -> EventStore:
        return self._event_reducer.event_store

    def _consume_committed_projection_tail(self) -> tuple[MarketEvent, ...]:
        """Advance past projection transport committed after publication."""

        return self._event_reducer.consume_available(projection_only=True)

    def _consume_committed_delivery_phase_tail(self) -> tuple[MarketEvent, ...]:
        """Advance past the phase lifecycle committed after publication."""

        return self._event_reducer.consume_available(delivery_phase_only=True)

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_publisher_state_schema_version"] = self._STATE_SCHEMA_VERSION
        if set(state) != self._PICKLE_FIELDS:
            raise ValueError("market snapshot publisher pickle state is not exact")
        return state

    def __setstate__(self, state: Mapping[str, object]) -> None:
        if (
            not isinstance(state, Mapping)
            or set(state) != self._PICKLE_FIELDS
            or state.get("_publisher_state_schema_version")
            != self._STATE_SCHEMA_VERSION
        ):
            raise ValueError("market snapshot publisher checkpoint schema changed")
        candidate = object.__new__(type(self))
        candidate.__dict__ = dict(state)
        candidate._require_checkpoint_state()
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)

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
        self._event_reducer._require_checkpoint_state()
        unconsumed_suffix = self.event_store.events_since(
            self._event_reducer.cursor
        )
        if any(
            event.origin is not EventOrigin.STATE_PROJECTION
            or event.kind not in _CURRENT_STATE_PROJECTION_KINDS
            for event in unconsumed_suffix
        ):
            raise ValueError(
                "market snapshot publisher checkpoint has a physical suffix"
            )
        if not self.atomic_authority:
            return
        expected_session = SessionStateReducer()
        if not self._boundary_reset_pending:
            consumed_prefix = self.event_store.events_since(0)[
                self._event_reducer._epoch_cursor :
                self._event_reducer.cursor
            ]
            for event in consumed_prefix:
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

    def _advance_delivery_phases(
        self,
        states: Mapping[Timeframe, TimeframeState],
        *,
        asof: pd.Timestamp,
        price: float,
        ordered_events: Sequence[MarketEvent],
        base_ids_by_timeframe: Mapping[Timeframe, tuple[str, ...]],
    ) -> tuple[DeliveryPhaseTransition, ...]:
        """Turn the per-bar phase label into an entered/updated/exited stay.

        Regime and phase stay two independent dimensions here: the regime is
        carried alongside the phase as evidence and never decides it.  An
        update is published only when a registered phase input moved while the
        phase itself did not, so a quiet bar publishes nothing.
        """

        self._advance_structure_generations(
            states,
            asof=asof,
            ordered_events=ordered_events,
        )
        transitions: list[DeliveryPhaseTransition] = []
        for timeframe in self._ORDER:
            state = states.get(timeframe)
            if state is None:
                continue
            phase = state.delivery.phase
            regime = state.structure.external_direction
            active_leg = state.delivery.active_leg_direction
            protected = state.structure.protected_swing_intact
            range_available = bool(
                state.range.range_id is not None
                and state.range.lifecycle
                not in {"broken", "invalidated"}
            )
            inputs = (
                regime,
                active_leg,
                protected,
                range_available,
            )
            source_ids = base_ids_by_timeframe.get(timeframe, ())
            external = self._structure_generations.get(
                (timeframe, StructureScope.EXTERNAL)
            )
            parent = (
                None
                if external is None or not external.is_active
                else external.generation_id
            )
            current = self._delivery_occupancies.get(timeframe)
            if current is not None and current.phase is phase:
                if current.inputs == inputs:
                    continue
                updated = replace(
                    current,
                    inputs=inputs,
                    observation_count=current.observation_count + 1,
                )
                self._delivery_occupancies[timeframe] = updated
                transitions.append(
                    DeliveryPhaseTransition(
                        kind=EventKind.DELIVERY_PHASE_UPDATED,
                        timeframe=timeframe,
                        phase=phase,
                        known_at=asof,
                        price=price,
                        entered_at=updated.entered_at,
                        age_bars=(
                            self._real_m1_bar_ordinal
                            - updated.entered_bar_ordinal
                        ),
                        observation_count=updated.observation_count,
                        previous_phase=updated.previous_phase,
                        next_phase=None,
                        origin_event_id=updated.origin_event_id,
                        parent_structure_generation_id=(
                            updated.parent_structure_generation_id
                        ),
                        structure_regime=regime,
                        active_leg_direction=active_leg,
                        protected_swing_intact=protected,
                        range_available=range_available,
                        source_event_ids=source_ids,
                    )
                )
                continue
            if current is not None:
                transitions.append(
                    DeliveryPhaseTransition(
                        kind=EventKind.DELIVERY_PHASE_EXITED,
                        timeframe=timeframe,
                        phase=current.phase,
                        known_at=asof,
                        price=price,
                        entered_at=current.entered_at,
                        age_bars=(
                            self._real_m1_bar_ordinal
                            - current.entered_bar_ordinal
                        ),
                        observation_count=current.observation_count,
                        previous_phase=current.previous_phase,
                        next_phase=phase,
                        origin_event_id=current.origin_event_id,
                        parent_structure_generation_id=(
                            current.parent_structure_generation_id
                        ),
                        structure_regime=regime,
                        active_leg_direction=active_leg,
                        protected_swing_intact=protected,
                        range_available=range_available,
                        source_event_ids=source_ids,
                    )
                )
            origin = self._delivery_origin_event_id(
                timeframe,
                ordered_events=ordered_events,
                source_ids=source_ids,
            )
            entered = DeliveryPhaseOccupancy(
                phase=phase,
                entered_at=asof,
                entered_bar_ordinal=self._real_m1_bar_ordinal,
                previous_phase=None if current is None else current.phase,
                origin_event_id=origin,
                parent_structure_generation_id=parent,
                inputs=inputs,
            )
            self._delivery_occupancies[timeframe] = entered
            transitions.append(
                DeliveryPhaseTransition(
                    kind=EventKind.DELIVERY_PHASE_ENTERED,
                    timeframe=timeframe,
                    phase=phase,
                    known_at=asof,
                    price=price,
                    entered_at=asof,
                    age_bars=0,
                    observation_count=1,
                    previous_phase=entered.previous_phase,
                    next_phase=None,
                    origin_event_id=origin,
                    parent_structure_generation_id=parent,
                    structure_regime=regime,
                    active_leg_direction=active_leg,
                    protected_swing_intact=protected,
                    range_available=range_available,
                    source_event_ids=source_ids,
                )
            )
        return tuple(transitions)

    def _advance_structure_generations(
        self,
        states: Mapping[Timeframe, TimeframeState],
        *,
        asof: pd.Timestamp,
        ordered_events: Sequence[MarketEvent],
    ) -> None:
        """Open, extend and close one structural claim per timeframe and scope.

        A generation is not a break.  Every qualified BOS and MSS core observed
        while a claim holds is absorbed into it, so a phase or a relation that
        spans four breaks still names one parent.  The claim ends when its
        protected swing is accepted through or the direction reverses; it does
        not silently restart under the same identity.
        """

        breaks_by_timeframe: dict[Timeframe, list[MarketEvent]] = {}
        for event in ordered_events:
            if event.kind in _STRUCTURE_CHANGE_KINDS:
                breaks_by_timeframe.setdefault(event.timeframe, []).append(event)

        for timeframe in self._ORDER:
            state = states.get(timeframe)
            if state is None:
                continue
            events = breaks_by_timeframe.get(timeframe, ())
            for scope in (StructureScope.EXTERNAL, StructureScope.INTERNAL):
                direction = (
                    state.structure.external_direction
                    if scope is StructureScope.EXTERNAL
                    else state.structure.internal_direction
                )
                # Only the external claim owns a protected swing; an internal
                # claim ends on direction change alone.
                protection_failed = (
                    scope is StructureScope.EXTERNAL
                    and state.structure.protected_swing_intact is False
                )
                self._advance_one_structure_generation(
                    timeframe,
                    scope,
                    direction=direction,
                    protection_failed=protection_failed,
                    protected_swing_id=(
                        state.structure.protected_low_id
                        if direction is Direction.LONG
                        else state.structure.protected_high_id
                    ),
                    asof=asof,
                    events=events,
                )

    def _advance_one_structure_generation(
        self,
        timeframe: Timeframe,
        scope: StructureScope,
        *,
        direction: Direction | None,
        protection_failed: bool,
        protected_swing_id: str | None,
        asof: pd.Timestamp,
        events: Sequence[MarketEvent],
    ) -> None:
        key = (timeframe, scope)
        current = self._structure_generations.get(key)
        if current is not None and current.is_active:
            # Accepting through the protected swing also clears the external
            # direction, so both conditions fire on the same bar.  Name the
            # cause rather than the effect it erased.
            if protection_failed:
                self._structure_generations[key] = replace(
                    current,
                    terminated_at=asof,
                    termination_reason="protected_swing_accepted_through",
                )
                current = None
            elif direction is not current.direction:
                self._structure_generations[key] = replace(
                    current,
                    terminated_at=asof,
                    termination_reason="direction_reversed",
                )
                current = None
        elif current is not None:
            current = None

        if direction is None or protection_failed:
            return

        if current is None:
            origin = next(
                (event for event in events if event.timeframe is timeframe),
                None,
            )
            if origin is None:
                # A structural claim with no originating event on this bar is
                # not well founded.  The direction is already visible in the
                # timeframe state; wait for the break that establishes it
                # rather than minting an identity with no ancestry.
                return
            ordinal = self._structure_generation_ordinals.get(key, 0) + 1
            self._structure_generation_ordinals[key] = ordinal
            self._structure_generations[key] = StructureGenerationState(
                generation_id=(
                    f"{timeframe.value}_{scope.value}_generation_{ordinal:04d}"
                ),
                timeframe=timeframe,
                scope=scope,
                direction=direction,
                started_at=asof,
                confirmed_at=asof,
                origin_event_id=origin.event_id,
                protected_swing_id=protected_swing_id,
            )
            current = self._structure_generations[key]

        bos = tuple(
            event.event_id
            for event in events
            if event.kind is EventKind.QUALIFIED_BOS
            and event.event_id not in current.bos_event_ids
        )
        mss = tuple(
            event.event_id
            for event in events
            if event.kind is EventKind.MSS_CORE_CONFIRMED
            and event.event_id not in current.mss_event_ids
        )
        if bos or mss or protected_swing_id != current.protected_swing_id:
            self._structure_generations[key] = replace(
                current,
                bos_event_ids=(*current.bos_event_ids, *bos),
                mss_event_ids=(*current.mss_event_ids, *mss),
                protected_swing_id=(
                    protected_swing_id or current.protected_swing_id
                ),
            )

    def _snapshot_projection_payloads(self) -> dict[str, Mapping[str, Any]]:
        """A rollback point for the projection payloads, without a deep copy.

        Each payload is a primitive built fresh by ``to_primitive`` and is only
        ever compared or replaced wholesale -- never edited in place -- so the
        mapping is the only thing that has to be copied.
        """

        return dict(self._last_projection_payloads)

    def _settle_candidate_views(
        self,
        states: Mapping[Timeframe, TimeframeState],
    ) -> Mapping[Timeframe, TimeframeState]:
        """Refresh the snapshot-derived candidate fields once for this bar.

        The reducer folded these back on every event it applied, which walked
        a collection that grows with market history once per event.  They are
        projections, so one pass per published timeframe answers the same
        question.

        The result is persisted the way the settled Swing views are: a rank
        assigned while its Swing was still in the hot hierarchy must survive
        that Swing's eviction, and the projection falls back to the stored
        rank exactly when the hierarchy can no longer supply one.
        """

        updated: dict[Timeframe, TimeframeState] = {}
        for timeframe, state in states.items():
            projected = _settled_candidate_state(state)
            updated[timeframe] = projected
            if projected is not state:
                self._event_reducer.states[timeframe] = projected
        return updated

    def _settle_swing_geometry(
        self,
        states: Mapping[Timeframe, TimeframeState],
    ) -> Mapping[Timeframe, TimeframeState]:
        """Give every confirmed Swing its place in one cross-timeframe tree.

        The tree cannot live inside a single timeframe: every Swing on a
        timeframe is confirmed from a window of the same length, so none can
        enclose another.  Only newly confirmed Swings do any work, and a quiet
        bar does none at all.
        """

        moved: set[str] = set()
        for timeframe, state in states.items():
            # The reducer returns the very same tuple when an event confirms
            # nothing, so identity is an exact "nothing was confirmed here"
            # test.  Length is not: the hot set is bounded, so once it is full
            # its length stops changing while Swings keep arriving, and a
            # length test would silently stop admitting all of them.
            hierarchy = state.swing_hierarchy
            if hierarchy is self._settled_hierarchies.get(timeframe):
                continue
            self._settled_hierarchies[timeframe] = hierarchy
            for swing in hierarchy:
                moved |= self._swing_geometry.admit(swing)
        if not moved:
            return states
        updated: dict[Timeframe, TimeframeState] = {}
        for timeframe, state in states.items():
            hierarchy = tuple(
                self._swing_geometry.view_of(swing)
                for swing in state.swing_hierarchy
            )
            if hierarchy == state.swing_hierarchy:
                updated[timeframe] = state
                continue
            settled = replace(state, swing_hierarchy=hierarchy)
            updated[timeframe] = settled
            # Persist the settled views so the reducer carries them forward and
            # a bar that confirms nothing re-derives nothing.
            self._event_reducer.states[timeframe] = settled
            self._settled_hierarchies[timeframe] = settled.swing_hierarchy
        return updated

    def _active_structure_generation_id(
        self,
        timeframe: Timeframe,
        scope: StructureScope,
    ) -> str | None:
        generation = self._structure_generations.get((timeframe, scope))
        if generation is None or not generation.is_active:
            return None
        return generation.generation_id

    def _advance_relation_generations(
        self,
        relations: Mapping[str, RelationState],
        *,
        asof: pd.Timestamp,
    ) -> None:
        """Turn the per-bar relation row into one countable occupancy.

        Must run after ``_advance_structure_generations``: an occupancy is
        defined by the structural claims it sits between, so a relation whose
        role text is unchanged but whose parent claim was replaced is a new
        occupancy, not a continuation of the old one.
        """

        retired: list[RelationGenerationState] = []
        for relation_id, relation in relations.items():
            # A relation compares the parent's external claim with the child's
            # internal one; those are exactly the two generations it spans.
            parent_generation = self._active_structure_generation_id(
                relation.parent_tf, StructureScope.EXTERNAL
            )
            child_generation = self._active_structure_generation_id(
                relation.child_tf, StructureScope.INTERNAL
            )
            signature = (parent_generation, child_generation, relation.role.value)
            current = self._relation_generations.get(relation_id)
            if current is not None:
                if current.signature == signature:
                    self._relation_generations[relation_id] = replace(
                        current,
                        updated_at=asof,
                        observation_count=current.observation_count + 1,
                    )
                    continue
                if current.role is not relation.role:
                    reason: str = "role_changed"
                    successor: RelationRole | None = relation.role
                elif current.parent_structure_generation_id != parent_generation:
                    reason, successor = "parent_structure_terminated", None
                else:
                    reason, successor = "child_structure_terminated", None
                retired.append(
                    replace(
                        current,
                        terminated_at=asof,
                        termination_reason=reason,
                        next_role=successor,
                    )
                )
            ordinal = self._relation_generation_ordinals.get(relation_id, 0) + 1
            self._relation_generation_ordinals[relation_id] = ordinal
            self._relation_generations[relation_id] = RelationGenerationState(
                generation_id=f"{relation_id}_generation_{ordinal:04d}",
                relation_id=relation_id,
                parent_tf=relation.parent_tf,
                child_tf=relation.child_tf,
                role=relation.role,
                entered_at=asof,
                updated_at=asof,
                parent_structure_generation_id=parent_generation,
                child_structure_generation_id=child_generation,
            )
        self._retired_relation_generations = tuple(retired)

    @staticmethod
    def _delivery_origin_event_id(
        timeframe: Timeframe,
        *,
        ordered_events: Sequence[MarketEvent],
        source_ids: tuple[str, ...],
    ) -> str | None:
        """The most specific same-bar cause available for this entry.

        A structure change on this timeframe is the strongest claim the Eye can
        make; otherwise the entry cites the last fact it saw for the timeframe
        on this bar, and nothing at all when there was none.
        """

        for event in reversed(ordered_events):
            if (
                event.timeframe is timeframe
                and event.kind in _STRUCTURE_CHANGE_KINDS
            ):
                return event.event_id
        return source_ids[-1] if source_ids else None

    def _clear_epoch_projections(self) -> None:
        self._session = SessionStateReducer()
        self._last_projection_payloads.clear()
        self._last_projection_event_ids.clear()
        self._formal_structures.clear()
        self._last_real_m1_price = None
        self._last_real_m1_event_id = None
        self._delivery_occupancies.clear()
        self._structure_generations.clear()
        self._relation_generations.clear()
        self._retired_relation_generations = ()
        self._swing_geometry.clear()
        self._settled_hierarchies.clear()
        # Ordinals keep counting across an epoch reset so a generation identity
        # is never reused for a different structural claim.
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
                lower_bound=float(item.lower_bound),
                upper_bound=float(item.upper_bound),
                formed_at=item.formed_at,
                confirmed_at=item.confirmed_at,
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
        anomalies: Sequence[str],
        emit_projection_events: bool = True,
    ) -> tuple[
        MarketSnapshot,
        tuple[MarketEvent, ...],
        tuple[DeliveryPhaseTransition, ...],
    ]:
        """Publish one suffix with bounded hot-state failure atomicity."""

        reducer_state = {
            "states": dict(self._event_reducer.states),
            "_cursor": self._event_reducer._cursor,
            "_epoch_cursor": self._event_reducer._epoch_cursor,
            "_last_order_key": self._event_reducer._last_order_key,
            "_latest_real_m1_event_id": (
                self._event_reducer._latest_real_m1_event_id
            ),
            "_expected_timeframes": self._event_reducer._expected_timeframes,
            "_store_prefix_fingerprint": (
                self._event_reducer._store_prefix_fingerprint
            ),
        }
        publisher_state = {
            "_session": copy.deepcopy(self._session),
            "_last_projection_payloads": self._snapshot_projection_payloads(),
            "_last_projection_event_ids": dict(
                self._last_projection_event_ids
            ),
            "_formal_structures": dict(self._formal_structures),
            "_last_real_m1_price": self._last_real_m1_price,
            "_last_real_m1_event_id": self._last_real_m1_event_id,
            "_boundary_reset_pending": self._boundary_reset_pending,
            "_delivery_occupancies": dict(self._delivery_occupancies),
            "_structure_generations": dict(self._structure_generations),
            "_structure_generation_ordinals": dict(
                self._structure_generation_ordinals
            ),
            "_relation_generations": dict(self._relation_generations),
            "_settled_hierarchies": dict(self._settled_hierarchies),
            "_relation_generation_ordinals": dict(
                self._relation_generation_ordinals
            ),
            "_retired_relation_generations": self._retired_relation_generations,
            "_real_m1_bar_ordinal": self._real_m1_bar_ordinal,
        }
        self._swing_geometry.begin()
        try:
            published = self._publish_committed_suffix(
                asof=asof,
                symbol=symbol,
                instrument_id=instrument_id,
                price=price,
                completed_1m=completed_1m,
                frames=frames,
                inventory=inventory,
                displacement=displacement,
                anomalies=anomalies,
                emit_projection_events=emit_projection_events,
            )
        except BaseException:
            for name, value in reducer_state.items():
                setattr(self._event_reducer, name, value)
            for name, value in publisher_state.items():
                setattr(self, name, value)
            self._swing_geometry.rollback()
            raise
        self._swing_geometry.commit()
        return published

    def _publish_committed_suffix(
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
        anomalies: Sequence[str],
        emit_projection_events: bool,
    ) -> tuple[
        MarketSnapshot,
        tuple[MarketEvent, ...],
        tuple[DeliveryPhaseTransition, ...],
    ]:
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
        expected_timeframes = (
            self._event_reducer.bind_timeframes(frames)
            if self.atomic_authority
            else None
        )
        committed_suffix = self._event_reducer.consume_available()
        ordered_events = tuple(
            event
            for event in committed_suffix
            if (
                event.origin is not EventOrigin.STATE_PROJECTION
                and event.kind is not EventKind.FOUNDATION_STATE_CHANGED
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
        elif has_epoch_reset:
            self._clear_epoch_projections()

        session: SessionState | None = None
        for event in ordered_events:
            if (
                self.atomic_authority
                and event.kind is EventKind.MARKET_EPOCH_RESET
            ):
                self._clear_epoch_projections()
            if (
                self.atomic_authority
                and event.kind is EventKind.BAR_COMPLETED
                and event.timeframe is Timeframe.M1
            ):
                m1_candle = _m1_candle_from_event(event)
                if m1_candle.real_completed:
                    self._last_real_m1_price = float(m1_candle.close)
                    self._last_real_m1_event_id = event.event_id
                    self._real_m1_bar_ordinal += 1
                elif self._last_real_m1_price is None:
                    raise RuntimeError(
                        "atomic clock-only M1 root requires a prior real "
                        "M1 price in the current epoch"
                    )
                session = self._session.update(m1_candle)
        if self.atomic_authority:
            if session is None or session.known_at != asof:
                raise RuntimeError(
                    "atomic session state did not reach snapshot asof"
                )
            self._require_real_m1_anchor_consistency()
        else:
            if completed_1m.real_completed:
                self._last_real_m1_price = float(completed_1m.close)
                self._last_real_m1_event_id = None
                self._real_m1_bar_ordinal += 1
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
            states = self._settle_candidate_views(
                {
                    timeframe: event_states[timeframe]
                    for timeframe in expected_timeframes
                }
            )
            authority = MarketSnapshotAuthority.ATOMIC_EVENT_REDUCER
        else:
            projected_states = {
                timeframe: self._timeframe_state(
                    frame,
                    asof=asof,
                    price=effective_price,
                    inventory=inventory,
                    displacement=displacement,
                    events=ordered_events,
                    anomalies=anomalies,
                )
                for timeframe, frame in frames.items()
            }
            for timeframe, state in tuple(projected_states.items()):
                formal_structure = self._formalize_structure(
                    timeframe,
                    state.structure,
                    ordered_events,
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
        states = self._settle_swing_geometry(states)
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
        delivery_transitions = self._advance_delivery_phases(
            states,
            asof=asof,
            price=effective_price,
            ordered_events=ordered_events,
            base_ids_by_timeframe=base_ids_by_timeframe,
        )
        self._advance_relation_generations(relations, asof=asof)
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
            structure_generations={
                generation.generation_id: generation
                for generation in self._structure_generations.values()
            },
            relation_generations={
                generation.generation_id: generation
                for generation in (
                    *self._retired_relation_generations,
                    *self._relation_generations.values(),
                )
            },
            session=session,
            events_this_update=tuple((*ordered_events, *projection_events)),
            labels=labels,
            event_count=self._event_reducer.cursor,
            event_prefix_fingerprint=self.event_store.fingerprint(),
            authority=authority,
        )
        if self.atomic_authority and has_epoch_reset:
            self._boundary_reset_pending = False
        return snapshot, tuple(projection_events), delivery_transitions


def replay_atomic_market_snapshot(
    events: Iterable[MarketEvent],
    *,
    semantic_registry_identity: str,
    semantic_version: str = SMC_SEMANTIC_VERSION,
    expected_timeframes: Iterable[Timeframe] | None = None,
) -> MarketSnapshot:
    """Rebuild the current hierarchy from the canonical atomic event log.

    Legacy state-projection transports remain admissible EventStore input but
    never become current-state evidence. Foundation revisions are validated by
    their cold ledger, not decoded into a second hot snapshot subsystem.
    """

    replay_definition_identity = (
        semantic_registry_identity
        if len(semantic_registry_identity) == 64
        and all(
            character in "0123456789abcdef"
            for character in semantic_registry_identity
        )
        else None
    )
    replay_store = EventStore(
        semantic_version=semantic_version,
        definition_identity=replay_definition_identity,
    )
    reducer = TimeframeEventReducer(
        event_store=replay_store,
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
    last_input_order_key: tuple[pd.Timestamp, int, str] | None = None
    input_event_ids: set[str] = set()

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
        replay_store.append(event)
        reducer.consume_available()
        if event.kind is EventKind.MARKET_EPOCH_RESET:
            session_reducer = SessionStateReducer()
            session = None
            latest_bar = None
            latest_real_bar = None
            retained_events.clear()
            epoch_scale_registry_id = None
            epoch_contract = None
            continue
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
                    "atomic replay contract changed without MARKET_EPOCH_RESET"
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
        raise ValueError("atomic replay ends after its latest completed 1m clock")
    if reducer.expected_timeframes is not None:
        missing = tuple(
            timeframe
            for timeframe in reducer.expected_timeframes
            if timeframe not in reducer.states
        )
        unexpected = tuple(
            timeframe
            for timeframe in reducer.states
            if timeframe not in reducer.expected_timeframes
        )
        if missing:
            raise ValueError(
                "atomic replay lacks bound timeframe state: "
                + ", ".join(timeframe.value for timeframe in missing)
            )
        if unexpected:
            raise ValueError(
                "atomic replay contains unregistered timeframe state: "
                + ", ".join(timeframe.value for timeframe in unexpected)
            )
    states = FrozenDict(
        {
            timeframe: _settled_candidate_state(state)
            for timeframe, state in reducer.states.items()
        }
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
        event_count=reducer.cursor,
        event_prefix_fingerprint=replay_store.fingerprint(),
        authority=MarketSnapshotAuthority.ATOMIC_EVENT_REDUCER,
    )


__all__ = [
    "BalanceRangeState",
    "DOLCandidateView",
    "DeliveryPhase",
    "DeliveryPhaseOccupancy",
    "StructureGenerationState",
    "StructureScope",
    "DeliveryPhaseTransition",
    "HierarchicalReplayState",
    "LiquidityClusterState",
    "LiquidityClusterSupersession",
    "LiquidityRangeRole",
    "MarketSnapshot",
    "MarketSnapshotAuthority",
    "MarketSnapshotPublisher",
    "MARKET_SNAPSHOT_SCHEMA_VERSION",
    "PriceZoneView",
    "RelationGenerationState",
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
    "SWING_HIERARCHY_HOT_RETENTION",
    "build_swing_geometry_nodes",
    "foundation_record_from_projection_event",
    "reduce_hierarchical_state",
    "reduce_timeframe_state",
    "replay_atomic_market_snapshot",
    "session_name_phase",
    "terminate_structural_range",
    "update_swing_geometry_assignments",
]
