"""Deterministic hierarchical projection over the existing causal Eye reducers.

This module does not detect Swing, BOS, liquidity, displacement, FVG, OB, or
dealing ranges.  It projects those already-authoritative states into the public
Timeframe/Relation/Session/Snapshot contract and emits replayable state-change
events.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .event_store import ImmutableEventStore, validate_canonical_event
from .model import (
    BOSLifecycle,
    BOSScope,
    Candle,
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
    clamp,
    to_primitive,
)


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
        if (
            not self.symbol
            or self.instrument_id < 0
            or not math.isfinite(float(self.price))
            or not self.semantic_version
            or not self.semantic_registry_identity
            or set(states) != {state.timeframe for state in states.values()}
            or any(relation.relation_id != key for key, relation in relations.items())
            or self.session.known_at != self.asof
            or any(event.known_at > self.asof for event in self.events_this_update)
        ):
            raise ValueError("market snapshot identity or causal clock is invalid")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            to_primitive(self),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def replay_payload(self) -> Mapping[str, Any]:
        return FrozenDict(
            {
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
        )


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
) -> TimeframeState | None:
    """Purely reduce one timeframe from normalized and semantic events.

    Projection events and legacy lifecycle transport are intentionally
    ignored.  Consequently this reducer can be replayed after all
    ``*_STATE_CHANGED`` events have been removed from the journal.
    """

    if event.kind not in _TIMEFRAME_REDUCER_KINDS:
        return state
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
        price_update_only = bool(
            event.evidence.get("owner_price_update_only", False)
        )
        real_completed = event.evidence.get("real_completed", True)
        if type(real_completed) is not bool:
            raise ValueError("completed BAR real-data flag must be boolean")
        clock_only = not real_completed
        if not price_update_only and not clock_only:
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
                age_bars=item.age_bars + (0 if clock_only else 1),
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
                "_unresolved_forward_reference_ids",
            )
        ):
            return
        self._normalized_bar_event_ids = {}
        self._terminal_crossing_event_ids = {}
        self._latest_protected_assignment_event_ids = {}
        self._latest_protected_assignment_event_ids_by_timeframe = {}
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

    def apply(self, event: MarketEvent) -> TimeframeState | None:
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
        if event.kind is EventKind.BAR_COMPLETED and event.timeframe is Timeframe.M1:
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
                    evidence={
                        **dict(event.evidence),
                        "owner_price_update_only": True,
                        "price_source_timeframe": Timeframe.M1.value,
                    },
                )
                owner_state = reduce_timeframe_state(
                    staged_states.get(owner_timeframe),
                    owner_event,
                    semantic_registry_identity=(
                        self.semantic_registry_identity
                    ),
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


def build_structural_legs(
    timeframe: Timeframe,
    swings: Sequence[SwingPoint],
    candles: Sequence[Candle],
    *,
    atr: float,
    protected_swing_ids: Sequence[str] = (),
    structural_swing_ids: Sequence[str] = (),
) -> tuple[StructuralLegState, ...]:
    """Project complete opposite-swing legs without future-role backfill.

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
    candle_by_start = {candle.start: candle for candle in candles if candle.real_completed}
    del protected_swing_ids, structural_swing_ids
    output: list[StructuralLegState] = []
    anchor: SwingPoint | None = None
    for current in confirmed:
        if anchor is None or current.side is anchor.side:
            anchor = current
            continue
        if current.pivot_start <= anchor.pivot_start:
            anchor = current
            continue
        path = tuple(
            candle
            for candle in candles
            if candle.real_completed
            and anchor.pivot_start <= candle.start <= current.pivot_start
        )
        if len(path) < 2 or anchor.pivot_start not in candle_by_start or current.pivot_start not in candle_by_start:
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
        travel = sum(abs(right - left) for left, right in zip(closes, closes[1:]))
        efficiency = clamp(abs(closes[-1] - closes[0]) / max(travel, 1e-12))
        running = closes[0]
        max_retracement = 0.0
        for close in closes[1:]:
            if direction is Direction.LONG:
                running = max(running, close)
                max_retracement = max(max_retracement, running - close)
            else:
                running = min(running, close)
                max_retracement = max(max_retracement, close - running)
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
        amplitude = abs(float(current.price) - float(anchor.price))
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
                amplitude_atr=amplitude / max(float(atr), 1e-12),
                duration_bars=len(path),
                duration_minutes=int((current.pivot_start - anchor.pivot_start).total_seconds() // 60),
                efficiency=efficiency,
                max_retracement_points=max_retracement,
                max_retracement_atr=max_retracement / max(float(atr), 1e-12),
                rank=rank,
                source_swing_ids=source_ids,
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

    def update(self, candle: Candle) -> SessionState:
        if candle.timeframe is not Timeframe.M1 or not candle.complete:
            raise ValueError("session reducer requires a completed 1m candle")
        local_start = candle.start.tz_convert("America/New_York")
        key = self._key(candle)
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
        self._recent_volume.append(float(candle.volume))
        name, phase = session_name_phase(local_start)
        return SessionState(
            session_id=key,
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
            realized_volatility=math.sqrt(self._sum_squared_log_returns),
            relative_volume=relative_volume,
            known_at=candle.end,
            data_complete=self._coverage_complete,
        )


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
    }
    if not required.issubset(evidence):
        raise ValueError("normalized 1m event lacks replayable candle fields")
    real_completed = evidence["real_completed"]
    if type(real_completed) is not bool:
        raise ValueError("normalized 1m event has an invalid real-data flag")
    return Candle(
        timeframe=Timeframe.M1,
        start=event.known_at - pd.Timedelta(minutes=1),
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
        self._boundary_reset_pending = False

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
        # Do not clear the authoritative event DAG out of band.  A boundary
        # update may first publish terminal facts for the prior epoch; the
        # immutable MARKET_EPOCH_RESET event then resets the reducer in
        # canonical stream order before any new-epoch bar is consumed.

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
            # This registry is immutable for the lifetime of one publisher.
            # It lets the first normalized M1 event initialize empty owner
            # states deterministically while preserving source_cutoff=None
            # until that owner's first native completed bar arrives.
            expected_timeframes = self._event_reducer.bind_timeframes(frames)
            self._require_current_m1_root(
                asof=asof,
                symbol=symbol,
                instrument_id=instrument_id,
                price=price,
                completed_1m=completed_1m,
                events=ordered_events,
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
            self._event_reducer.apply(event)
            if (
                self.atomic_authority
                and event.kind is EventKind.BAR_COMPLETED
                and event.timeframe is Timeframe.M1
            ):
                session = self._session.update(
                    _m1_candle_from_event(event)
                )
        if self.atomic_authority:
            if session is None or session.known_at != asof:
                raise RuntimeError(
                    "atomic session state did not reach snapshot asof"
                )
        else:
            session = self._session.update(completed_1m)
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
                    price=price,
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
            price=price,
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
            price=price,
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
) -> MarketSnapshot:
    """Rebuild the latest hierarchy from normalized and semantic events only.

    Projection events are accepted but ignored by the timeframe reducer. A
    registered epoch-reset event clears contract-local timeframe and Session
    state, making the final snapshot reproducible across rolls and hard gaps.
    """

    reducer = TimeframeEventReducer(
        semantic_registry_identity=semantic_registry_identity,
        semantic_version=semantic_version,
        expected_timeframes=expected_timeframes,
    )
    session_reducer = SessionStateReducer()
    session: SessionState | None = None
    latest_bar: Candle | None = None
    retained_events: list[MarketEvent] = []
    epoch_scale_registry_id: str | None = None
    epoch_contract: tuple[str, int] | None = None
    for event in events:
        reducer.apply(event)
        if event.kind is EventKind.MARKET_EPOCH_RESET:
            session_reducer = SessionStateReducer()
            session = None
            latest_bar = None
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
        session = session_reducer.update(latest_bar)
    if session is None or latest_bar is None or not reducer.states:
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
    states = FrozenDict(reducer.states)
    relations = RelationResolver(
        edges=MarketSnapshotPublisher._RELATION_EDGES
    ).resolve(
        states,
        price=float(latest_bar.close),
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
        price=float(latest_bar.close),
        semantic_version=semantic_version,
        semantic_registry_identity=semantic_registry_identity,
        timeframe_states=states,
        relations=relations,
        session=session,
        events_this_update=(),
        labels=labels,
        authority=MarketSnapshotAuthority.ATOMIC_EVENT_REDUCER,
    )


__all__ = [
    "DOLCandidateView",
    "DeliveryPhase",
    "HierarchicalReplayState",
    "LiquidityRangeRole",
    "MarketSnapshot",
    "MarketSnapshotAuthority",
    "MarketSnapshotPublisher",
    "PriceZoneView",
    "RelationRole",
    "RelationResolver",
    "RelationState",
    "SessionState",
    "SessionStateReducer",
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
    "build_structural_legs",
    "reduce_hierarchical_state",
    "reduce_timeframe_state",
    "replay_atomic_market_snapshot",
    "session_name_phase",
]
