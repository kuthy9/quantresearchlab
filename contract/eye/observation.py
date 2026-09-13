"""Eye observation: the published per-clock transport contract.

``MarketObservation`` is the Eye's sole output and the Brain's sole input.
It carries the authoritative snapshot, the frames, the semantic event delta
and every typed transition observed on this completed clock."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any, ClassVar, TYPE_CHECKING
import pandas as pd

from contract.market.primitives import CORE_TIMEFRAMES, Direction, FrozenDict, SMC_SEMANTIC_VERSION, Timeframe, _exact_dataclass_pickle_state, _restore_exact_dataclass_pickle_state, _strict_payload_equal, aware_timestamp, clamp
from contract.execution.reality import ExecutionObservation
from contract.eye.vocabulary import DealingRangeLifecycle, EventKind, EventOrigin, FairValueGapLifecycle, LiquidityInventoryLifecycle, MARKET_OBSERVATION_SCHEMA_VERSION, ManipulationLifecycle, ManipulationSourceDispositionKind, OrderBlockLifecycle, PathSequenceLifecycle
from contract.eye.entities import BreakOfStructureState, CandleStructureState, DealingRangeState, FairValueGapState, LiquidityInventoryItem, LiquidityPoolState, ManipulationSourceDisposition, ManipulationState, OrderBlockFunnelSnapshot, OrderBlockState, RangeFormationFunnelSnapshot, StructuralLegState, StructureSequenceState, SupportResistanceState, SwingPoint
from contract.eye.interaction import InteractionUpdate

if TYPE_CHECKING:
    from eyes.core.market_state import MarketSnapshot


@dataclass(frozen=True)
class MarketEvent:
    event_id: str
    kind: EventKind
    observed_at: pd.Timestamp
    timeframe: Timeframe
    side: str | None
    price: float | None
    strength: float
    source_ids: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    entity_id: str | None = None
    lifecycle: str | None = None
    formed_at: pd.Timestamp | None = None
    confirmed_at: pd.Timestamp | None = None
    ended_at: pd.Timestamp | None = None
    direction: Direction | None = None
    transition_reason: str | None = None
    sequence_no: int = 0
    event_time: pd.Timestamp | None = None
    known_at: pd.Timestamp | None = None
    semantic_version: str = SMC_SEMANTIC_VERSION
    evidence: Mapping[str, Any] = field(default_factory=dict)
    zone: tuple[float, float] | None = None
    # ``source_ids`` remains the legacy compatibility surface.  New semantic
    # producers must classify provenance explicitly so event ancestry cannot
    # be confused with raw-data, entity, or contextual evidence identities.
    # These fields intentionally live at the end of the dataclass to preserve
    # every existing positional constructor call.
    source_event_ids: tuple[str, ...] = ()
    source_data_ids: tuple[str, ...] = ()
    source_entity_ids: tuple[str, ...] = ()
    context_event_ids: tuple[str, ...] = ()
    origin: EventOrigin = EventOrigin.LEGACY_TRANSPORT

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observed_at", aware_timestamp(self.observed_at, name="event.observed_at")
        )
        event_time = self.event_time or self.observed_at
        known_at = self.known_at or self.observed_at
        object.__setattr__(
            self,
            "event_time",
            aware_timestamp(event_time, name="event.event_time"),
        )
        object.__setattr__(
            self,
            "known_at",
            aware_timestamp(known_at, name="event.known_at"),
        )
        if self.known_at != self.observed_at:
            raise ValueError(
                "event observed_at is the legacy alias of known_at and must match"
            )
        if self.event_time > self.known_at:
            raise ValueError("event cannot be known before its market event time")
        if (
            not isinstance(self.semantic_version, str)
            or not self.semantic_version.strip()
        ):
            raise ValueError("event semantic_version is required")
        try:
            origin = EventOrigin(self.origin)
        except (TypeError, ValueError) as error:
            raise ValueError("event origin is invalid") from error
        object.__setattr__(self, "origin", origin)
        raw_details = self.details
        raw_evidence = self.evidence
        details = FrozenDict(raw_details)
        # ``details`` is the legacy name of the canonical evidence mapping.
        # Producers commonly supply both aliases with exactly the same
        # immutable payload.  Share that one frozen value instead of retaining
        # and serializing two complete copies on every historical event.
        evidence = (
            details
            if (
                not raw_evidence
                or raw_evidence is raw_details
                or _strict_payload_equal(raw_evidence, raw_details)
            )
            else FrozenDict(raw_evidence)
        )
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "evidence", evidence)
        source_namespaces: dict[str, tuple[str, ...]] = {}
        for name in (
            "source_ids",
            "source_event_ids",
            "source_data_ids",
            "source_entity_ids",
            "context_event_ids",
        ):
            raw_values = getattr(self, name)
            if isinstance(raw_values, str):
                raise ValueError(
                    f"event {name} identities must be a sequence"
                )
            try:
                values = tuple(raw_values)
            except TypeError as error:
                raise ValueError(
                    f"event {name} identities must be a sequence"
                ) from error
            if any(
                not isinstance(value, str) or not value.strip()
                for value in values
            ):
                raise ValueError(
                    f"event {name} identities must be non-empty text"
                )
            if len(values) != len(set(values)):
                if name == "source_ids":
                    # Older lifecycle transports legitimately repeated one
                    # entity through several structural roles.  Preserve
                    # that API while exposing a stable de-duplicated alias.
                    values = tuple(dict.fromkeys(values))
                else:
                    raise ValueError(
                        f"event {name} identities must be unique"
                    )
            source_namespaces[name] = values

        legacy_source_ids = source_namespaces["source_ids"]
        explicit_event_ids = source_namespaces["source_event_ids"]
        classified_event_ancestry = origin in {
            EventOrigin.SEMANTIC_ATOMIC,
            EventOrigin.STATE_PROJECTION,
        }
        if classified_event_ancestry:
            if (
                legacy_source_ids
                and explicit_event_ids
                and legacy_source_ids != explicit_event_ids
            ):
                raise ValueError(
                    "canonical event source_ids must equal "
                    "source_event_ids"
                )
            # This is the only compatibility normalization that is
            # unambiguous: a canonical producer supplied just one of the two
            # event-ancestry spellings.  Mixed raw/entity identities must use
            # their explicit namespaces instead.
            canonical_event_ids = explicit_event_ids or legacy_source_ids
            legacy_source_ids = canonical_event_ids
            explicit_event_ids = canonical_event_ids
        elif not explicit_event_ids:
            # Preserve the former ``source_event_ids`` alias for old,
            # non-canonical lifecycle fixtures and consumers.
            explicit_event_ids = legacy_source_ids

        source_namespaces["source_ids"] = legacy_source_ids
        source_namespaces["source_event_ids"] = explicit_event_ids
        for name, values in source_namespaces.items():
            object.__setattr__(self, name, values)

        explicit_namespaces = {
            name: source_namespaces[name]
            for name in (
                "source_event_ids",
                "source_data_ids",
                "source_entity_ids",
                "context_event_ids",
            )
        }
        owners: dict[str, str] = {}
        for name, values in explicit_namespaces.items():
            for value in values:
                previous = owners.setdefault(value, name)
                if previous != name:
                    raise ValueError(
                        "event provenance identities must be mutually "
                        f"exclusive across namespaces: {value!r} appears "
                        f"in {previous} and {name}"
                    )
        zone = self.zone
        if zone is None:
            lower = details.get("lower_bound")
            upper = details.get("upper_bound")
            if lower is not None and upper is not None:
                zone = (float(lower), float(upper))
        if zone is not None:
            if (
                not isinstance(zone, (tuple, list))
                or len(zone) != 2
                or not all(math.isfinite(float(value)) for value in zone)
                or float(zone[0]) > float(zone[1])
            ):
                raise ValueError("event zone is invalid")
            zone = (float(zone[0]), float(zone[1]))
        object.__setattr__(self, "zone", zone)
        object.__setattr__(self, "strength", clamp(self.strength))
        for name in ("formed_at", "confirmed_at", "ended_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    aware_timestamp(value, name=f"event.{name}"),
                )
        if (self.entity_id is None) != (self.lifecycle is None):
            raise ValueError(
                "typed market event requires both entity_id and lifecycle"
            )
        if self.entity_id == "" or self.lifecycle == "":
            raise ValueError("typed market event identity cannot be empty")
        if type(self.sequence_no) is not int or self.sequence_no < 0:
            raise ValueError("market event sequence must be non-negative")
        if self.formed_at is not None and self.formed_at > self.observed_at:
            raise ValueError("event formation cannot be in the future")
        if self.confirmed_at is not None and (
            self.formed_at is None
            or self.confirmed_at < self.formed_at
            or self.confirmed_at > self.observed_at
        ):
            raise ValueError("event confirmation clock is invalid")
        if self.ended_at is not None and self.ended_at != self.observed_at:
            raise ValueError(
                "terminal event transition must end at its observation clock"
            )
        if self.ended_at is not None and not self.transition_reason:
            raise ValueError("terminal event requires a transition reason")
        if self.transition_reason == "":
            raise ValueError("event transition reason cannot be empty")

    @property
    def semantic_type(self) -> str:
        """Canonical semantic name used by the v1 event contract."""

        return self.kind.value

    @property
    def is_canonical_semantic(self) -> bool:
        """Whether this is an authoritative preregistered semantic fact."""

        return self.origin is EventOrigin.SEMANTIC_ATOMIC

    @property
    def is_projection(self) -> bool:
        """Whether this event is a rebuildable state projection."""

        return self.origin is EventOrigin.STATE_PROJECTION


_TYPED_EVENT_ENTITY_PREFIXES: Mapping[EventKind, str] = {
    EventKind.SWING_STATE: "swing",
    EventKind.STRUCTURE_STATE: "structure",
    EventKind.STRUCTURE_BREAK: "bos",
    EventKind.STRUCTURE_BREAK_FAILED: "bos",
    EventKind.SUPPORT_RESISTANCE_STATE: "zone",
    EventKind.LIQUIDITY_POOL_STATE: "pool",
    EventKind.FVG_STATE: "fvg",
    EventKind.ORDER_BLOCK_STATE: "order_block",
    EventKind.DEALING_RANGE_STATE: "range",
    EventKind.MANIPULATION_STATE: "manipulation",
    EventKind.ENTRY_PATH_STATE: "entry_path",
}


def typed_event_entity_key(event: MarketEvent) -> str | None:
    """Return the collision-safe key for a registered typed lifecycle event."""

    if event.entity_id is None:
        return None
    prefix = _TYPED_EVENT_ENTITY_PREFIXES.get(event.kind)
    if prefix is None:
        raise ValueError(
            "typed market event kind has no registered entity timeline"
        )
    if prefix == "pool":
        expected_prefix = "pool:"
        if not event.entity_id.startswith(expected_prefix):
            raise ValueError(
                "liquidity-pool event entity id must use the pool namespace"
            )
        suffix = event.entity_id[len(expected_prefix) :]
    else:
        suffix = event.entity_id
    if not suffix:
        raise ValueError("typed market event timeline identity is empty")
    return f"{prefix}:{suffix}"


@dataclass(frozen=True)
class FrameObservation:
    timeframe: Timeframe
    cutoff: pd.Timestamp
    bars: int
    metrics: Mapping[str, float]
    ready: bool = False
    swings: tuple[SwingPoint, ...] = ()
    structures: tuple[StructureSequenceState, ...] = ()
    structure_breaks: tuple[BreakOfStructureState, ...] = ()
    candle_structure: CandleStructureState | None = None
    support_resistance: tuple[SupportResistanceState, ...] = ()
    liquidity_pools: tuple[LiquidityPoolState, ...] = ()
    fair_value_gaps: tuple[FairValueGapState, ...] = ()
    order_blocks: tuple[OrderBlockState, ...] = ()
    dealing_ranges: tuple[DealingRangeState, ...] = ()
    structural_legs: tuple[StructuralLegState, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoff", aware_timestamp(self.cutoff, name="frame.cutoff"))
        for name, value in self.metrics.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{self.timeframe.value}.{name} is non-finite")
        for item in self.swings:
            if item.timeframe is not self.timeframe:
                raise ValueError("frame contains a swing from another timeframe")
        for item in self.structures:
            if item.timeframe is not self.timeframe:
                raise ValueError("frame contains a structure from another timeframe")
        for item in self.structure_breaks:
            if item.timeframe is not self.timeframe:
                raise ValueError("frame contains a BOS from another timeframe")
        if (
            self.candle_structure is not None
            and self.candle_structure.timeframe is not self.timeframe
        ):
            raise ValueError(
                "frame contains candle structure from another timeframe"
            )
        for item in self.support_resistance:
            if item.timeframe is not self.timeframe:
                raise ValueError(
                    "frame contains support/resistance from another timeframe"
                )
        for item in self.liquidity_pools:
            if item.timeframe is not self.timeframe:
                raise ValueError(
                    "frame contains a liquidity pool from another timeframe"
                )
        for item in self.fair_value_gaps:
            if item.timeframe is not self.timeframe:
                raise ValueError("frame contains an FVG from another timeframe")
        for item in self.order_blocks:
            if item.timeframe is not self.timeframe:
                raise ValueError(
                    "frame contains an order block from another timeframe"
                )
        for item in self.dealing_ranges:
            if item.timeframe is not self.timeframe:
                raise ValueError(
                    "frame contains a dealing range from another timeframe"
                )
        for item in self.structural_legs:
            if item.timeframe is not self.timeframe:
                raise ValueError(
                    "frame contains a structural leg from another timeframe"
                )
            if item.known_at > self.cutoff:
                raise ValueError(
                    "frame contains a future-known structural leg"
                )
        for swing in self.swings:
            if (
                (
                    swing.observed_at is not None
                    and swing.observed_at > self.cutoff
                )
                or (
                    swing.confirmed_at is not None
                    and swing.confirmed_at > self.cutoff
                )
                or (
                    swing.broken_at is not None
                    and swing.broken_at > self.cutoff
                )
            ):
                raise ValueError(
                    "frame contains a future-known swing state"
                )
        for structure in self.structures:
            if (
                (
                    structure.formed_at is not None
                    and structure.formed_at > self.cutoff
                )
                or (
                    structure.confirmed_at is not None
                    and structure.confirmed_at > self.cutoff
                )
                or (
                    structure.broken_at is not None
                    and structure.broken_at > self.cutoff
                )
                or (
                    structure.formation_failed_at is not None
                    and structure.formation_failed_at > self.cutoff
                )
            ):
                raise ValueError(
                    "frame contains a future-known structure state"
                )
        for item in self.structure_breaks:
            if (
                (
                    item.pending_at is not None
                    and item.pending_at > self.cutoff
                )
                or (
                    item.resolved_at is not None
                    and item.resolved_at > self.cutoff
                )
                or (
                    item.last_attempt_at is not None
                    and item.last_attempt_at > self.cutoff
                )
            ):
                raise ValueError("frame contains a future-known BOS state")
        if (
            self.candle_structure is not None
            and self.candle_structure.observed_at > self.cutoff
        ):
            raise ValueError("frame contains future candle structure")
        for item in self.support_resistance:
            if any(
                value is not None and value > self.cutoff
                for value in (
                    item.confirmed_at,
                    item.tested_at,
                    item.broken_at,
                    item.reaccepted_at,
                    item.retired_at,
                    *item.touch_times,
                )
            ):
                raise ValueError(
                    "frame contains future support/resistance state"
                )
        for item in self.liquidity_pools:
            if any(
                value is not None and value > self.cutoff
                for value in (
                    item.confirmed_at,
                    item.swept_at,
                    item.resolved_at,
                    *item.touch_times,
                )
            ):
                raise ValueError("frame contains future liquidity pool state")
        for item in self.fair_value_gaps:
            if any(
                value is not None and value > self.cutoff
                for value in (
                    *item.source_candle_starts,
                    item.source_displacement_started_at,
                    item.source_displacement_active_at,
                    item.formed_at,
                    item.confirmed_at,
                    item.state_started_at,
                    item.last_updated_at,
                    item.partial_at,
                    item.mitigated_at,
                    item.invalidated_at,
                )
            ):
                raise ValueError("frame contains future FVG state")
        for item in self.order_blocks:
            if any(
                value is not None and value > self.cutoff
                for value in (
                    item.source_displacement_started_at,
                    item.source_displacement_active_at,
                    item.source_bos_resolved_at,
                    item.anchor_start,
                    item.anchor_end,
                    item.formed_at,
                    item.confirmed_at,
                    item.state_started_at,
                    item.last_updated_at,
                    item.first_test_at,
                    item.mitigated_at,
                    item.failed_at,
                )
            ):
                raise ValueError("frame contains future order-block state")
        for item in self.dealing_ranges:
            if any(
                value is not None and value > self.cutoff
                for value in (
                    item.lower_source_confirmed_at,
                    item.upper_source_confirmed_at,
                    item.lower_source_tested_at,
                    item.upper_source_tested_at,
                    item.formed_at,
                    item.balance_confirmed_at,
                    item.broken_at,
                    item.state_started_at,
                    item.last_updated_at,
                )
            ):
                raise ValueError("frame contains future dealing-range state")
        range_ids = tuple(item.range_id for item in self.dealing_ranges)
        if (
            len(range_ids) != len(set(range_ids))
            or sum(
                item.lifecycle
                in {
                    DealingRangeLifecycle.ACTIVE,
                    DealingRangeLifecycle.ACTIVE,
                }
                for item in self.dealing_ranges
            )
            > 1
        ):
            raise ValueError(
                "frame dealing-range identities or live capacity are invalid"
            )


@dataclass(frozen=True)
class DisplacementTransitionObservation:
    """A descriptive lifecycle transition kept outside legacy event memory."""

    transition_id: str
    entity_id: str
    lifecycle: str
    reason: str | None
    observed_at: pd.Timestamp
    direction: Direction
    ordinal: int = 0
    started_at: pd.Timestamp | None = None
    active_at: pd.Timestamp | None = None
    state_started_at: pd.Timestamp | None = None
    terminal_at: pd.Timestamp | None = None
    prefix_last_admitted_at: pd.Timestamp | None = None
    terminal_evidence_candle_id: str | None = None
    admitted_candle_ids: tuple[str, ...] = ()
    state_metrics: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observed_at",
            aware_timestamp(self.observed_at, name="displacement transition"),
        )
        object.__setattr__(self, "state_metrics", tuple(self.state_metrics))
        object.__setattr__(
            self,
            "admitted_candle_ids",
            tuple(self.admitted_candle_ids),
        )
        terminal = self.lifecycle in {"exhausted", "censored"}
        if (
            not self.transition_id
            or not self.entity_id
            or self.lifecycle not in {"started", "active", "exhausted", "censored"}
            or not isinstance(self.direction, Direction)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
            or terminal != bool(self.reason)
            or not self.admitted_candle_ids
            or any(not value for value in self.admitted_candle_ids)
        ):
            raise ValueError("invalid displacement transition observation")
        clocks = (
            self.started_at,
            self.active_at,
            self.state_started_at,
            self.terminal_at,
            self.prefix_last_admitted_at,
        )
        if any(
            aware_timestamp(value, name="displacement transition clock")
            > self.observed_at
            for value in clocks
            if value is not None
        ):
            raise ValueError("displacement transition contains the future")
        if terminal != bool(self.terminal_at):
            raise ValueError("displacement terminal clock is inconsistent")
        if any(
            not name or not math.isfinite(float(value))
            for name, value in self.state_metrics
        ):
            raise ValueError("displacement transition metrics are invalid")


@dataclass(frozen=True)
class DisplacementObservation:
    """The current displacement state plus an isolated transition history."""

    asof: pd.Timestamp
    lifecycle: str
    current_entity_id: str | None
    current_direction: Direction | None
    current_state_observed_at: pd.Timestamp | None
    current_started_at: pd.Timestamp | None
    current_active_at: pd.Timestamp | None
    current_last_admitted_at: pd.Timestamp | None
    current_metrics: tuple[tuple[str, float], ...] | None
    latest_transition: DisplacementTransitionObservation | None
    recent_transitions: tuple[DisplacementTransitionObservation, ...] = ()
    transitions_this_update: tuple[
        DisplacementTransitionObservation, ...
    ] = ()
    reader_anomalies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "asof",
            aware_timestamp(self.asof, name="displacement observation"),
        )
        object.__setattr__(
            self, "recent_transitions", tuple(self.recent_transitions)
        )
        object.__setattr__(
            self,
            "transitions_this_update",
            tuple(self.transitions_this_update),
        )
        object.__setattr__(self, "reader_anomalies", tuple(self.reader_anomalies))
        if self.current_metrics is not None:
            object.__setattr__(
                self,
                "current_metrics",
                tuple(tuple(item) for item in self.current_metrics),
            )
        if (
            self.lifecycle not in {"idle", "started", "active"}
            or len(self.recent_transitions) > 64
        ):
            raise ValueError("invalid current displacement lifecycle or history")
        transitions_are_suffix = (
            not self.transitions_this_update
            or (
                len(self.transitions_this_update)
                <= len(self.recent_transitions)
                and self.recent_transitions[
                    -len(self.transitions_this_update) :
                ]
                == self.transitions_this_update
            )
        )
        if not transitions_are_suffix:
            raise ValueError("current displacement transitions are not a history suffix")
        if tuple(
            item.ordinal for item in self.transitions_this_update
        ) != tuple(range(len(self.transitions_this_update))):
            raise ValueError("current displacement transition order is invalid")
        if any(
            right.observed_at < left.observed_at
            for left, right in zip(
                self.recent_transitions[:-1],
                self.recent_transitions[1:],
            )
        ):
            raise ValueError("displacement transition history is out of order")
        expected_latest = (
            self.recent_transitions[-1] if self.recent_transitions else None
        )
        if self.latest_transition != expected_latest:
            raise ValueError("latest displacement transition does not match history")
        current_fields = (
            self.current_entity_id,
            self.current_direction,
            self.current_state_observed_at,
            self.current_started_at,
            self.current_active_at,
            self.current_last_admitted_at,
            self.current_metrics,
        )
        if self.lifecycle == "idle":
            if any(value is not None for value in current_fields):
                raise ValueError("idle displacement cannot expose current state")
        elif (
            not self.current_entity_id
            or not isinstance(self.current_direction, Direction)
            or self.current_state_observed_at is None
            or self.current_started_at is None
            or self.current_last_admitted_at is None
            or self.current_metrics is None
        ):
            raise ValueError("open displacement state is incomplete")
        if (
            (self.lifecycle == "started" and self.current_active_at is not None)
            or (self.lifecycle == "active" and self.current_active_at is None)
        ):
            raise ValueError("current displacement active clock is inconsistent")
        clocks = tuple(
            item.observed_at for item in self.recent_transitions
        ) + (
            self.current_state_observed_at,
            self.current_started_at,
            self.current_active_at,
            self.current_last_admitted_at,
        )
        if any(
            aware_timestamp(value, name="displacement clock") > self.asof
            for value in clocks
            if value is not None
        ):
            raise ValueError("displacement observation contains the future")
        if self.current_metrics is not None and any(
            not name or not math.isfinite(float(value))
            for name, value in self.current_metrics
        ):
            raise ValueError("current displacement metrics are invalid")


@dataclass(frozen=True)
class MarketObservation:
    market_snapshot: "MarketSnapshot | None"
    frames: Mapping[Timeframe, FrameObservation]
    recent_events: tuple[MarketEvent, ...]
    event_durations_minutes: Mapping[str, int]
    execution: ExecutionObservation
    _snapshot_free_identity: tuple[
        pd.Timestamp,
        str,
        int,
        float,
        tuple[MarketEvent, ...],
    ] | None = field(
        default=None,
        repr=False,
        compare=False,
        metadata={"primitive": False},
    )
    anomalies: tuple[str, ...] = ()
    displacement: DisplacementObservation | None = None
    liquidity_inventory: tuple[LiquidityInventoryItem, ...] = ()
    liquidity_pool_states: tuple[LiquidityPoolState, ...] = ()
    event_ages_minutes: Mapping[str, int] = field(default_factory=dict)
    retained_entity_timelines: Mapping[
        str,
        tuple[MarketEvent, ...],
    ] = field(default_factory=dict)
    incomplete_entity_timeline_keys: tuple[str, ...] = ()
    # One transport-level semantic delta for lightweight causal consumers.
    # Reducer snapshots remain the authoritative current state; these tuples
    # contain only baseline entities or entities whose lifecycle/evidence
    # changed on this completed update.  Age/freshness heartbeats are omitted.
    typed_transition_delta_available: bool = False
    liquidity_inventory_transitions_this_update: tuple[
        LiquidityInventoryItem,
        ...,
    ] = ()
    liquidity_pool_transitions_this_update: tuple[
        LiquidityPoolState,
        ...,
    ] = ()
    group3_fvg_transitions_this_update: tuple[
        FairValueGapState,
        ...,
    ] = ()
    group3_order_block_transitions_this_update: tuple[
        OrderBlockState,
        ...,
    ] = ()
    group4_range_transitions_this_update: tuple[
        DealingRangeState,
        ...,
    ] = ()
    group4_manipulation_transitions_this_update: tuple[
        ManipulationState,
        ...,
    ] = ()
    group3_boundary_fvg_transitions: tuple[
        FairValueGapState,
        ...,
    ] = ()
    group3_boundary_order_block_transitions: tuple[
        OrderBlockState,
        ...,
    ] = ()
    group3_order_block_funnel: tuple[
        OrderBlockFunnelSnapshot,
        ...,
    ] = ()
    manipulations: tuple[ManipulationState, ...] = ()
    group4_boundary_range_transitions: tuple[
        DealingRangeState,
        ...,
    ] = ()
    group4_boundary_manipulation_transitions: tuple[
        ManipulationState,
        ...,
    ] = ()
    group4_ambiguous_sweep_item_ids: tuple[str, ...] = ()
    group4_atr_unready_sweep_item_ids: tuple[str, ...] = ()
    group4_source_dispositions: tuple[
        ManipulationSourceDisposition,
        ...,
    ] = ()
    group4_range_funnel: tuple[
        RangeFormationFunnelSnapshot,
        ...,
    ] = ()
    interaction_update: InteractionUpdate | None = None
    active_timeframes: tuple[Timeframe, ...] = ()
    # The scales whose events form the main event clock; the rest of the
    # active scales are microstructure.  Empty publishes every scale.
    published_timeframes: tuple[Timeframe, ...] = ()
    scale_registry_id: str = ""
    scene_revision_id: str | None = None
    scene_added_node_ids: tuple[str, ...] = ()
    scene_revised_node_ids: tuple[str, ...] = ()
    scene_added_edge_ids: tuple[str, ...] = ()
    scene_revised_edge_ids: tuple[str, ...] = ()
    scene_resolution_event_ids: tuple[str, ...] = ()

    schema_version: ClassVar[int] = MARKET_OBSERVATION_SCHEMA_VERSION

    @property
    def asof(self) -> pd.Timestamp:
        if self.market_snapshot is not None:
            return self.market_snapshot.asof
        assert self._snapshot_free_identity is not None
        return self._snapshot_free_identity[0]

    @property
    def symbol(self) -> str:
        if self.market_snapshot is not None:
            return self.market_snapshot.symbol
        assert self._snapshot_free_identity is not None
        return self._snapshot_free_identity[1]

    @property
    def instrument_id(self) -> int:
        if self.market_snapshot is not None:
            return self.market_snapshot.instrument_id
        assert self._snapshot_free_identity is not None
        return self._snapshot_free_identity[2]

    @property
    def price(self) -> float:
        if self.market_snapshot is not None:
            return self.market_snapshot.price
        assert self._snapshot_free_identity is not None
        return self._snapshot_free_identity[3]

    @property
    def events_this_update(self) -> tuple[MarketEvent, ...]:
        """Every event of this update, on every scale."""

        if self.market_snapshot is not None:
            return self.market_snapshot.events_this_update
        assert self._snapshot_free_identity is not None
        return self._snapshot_free_identity[4]

    @property
    def semantic_events_this_update(self) -> tuple[MarketEvent, ...]:
        """The main event clock: this update's events on the published scales.

        An empty ``published_timeframes`` publishes every scale, which is the
        pre-schema-6 behaviour.
        """

        if not self.published_timeframes:
            return self.events_this_update
        published = frozenset(self.published_timeframes)
        return tuple(
            event
            for event in self.events_this_update
            if event.timeframe in published
        )

    @property
    def microstructure_events_this_update(self) -> tuple[MarketEvent, ...]:
        """This update's events on the scales kept out of the main clock.

        The 1m tape by default: the channel a trigger reads, and never
        silently dropped -- the two channels partition ``events_this_update``.
        """

        if not self.published_timeframes:
            return ()
        published = frozenset(self.published_timeframes)
        return tuple(
            event
            for event in self.events_this_update
            if event.timeframe not in published
        )

    def __post_init__(self) -> None:
        identity = self._snapshot_free_identity
        if self.market_snapshot is None:
            if (
                not isinstance(identity, tuple)
                or len(identity) != 5
                or not isinstance(identity[1], str)
                or not identity[1]
                or type(identity[2]) is not int
                or identity[2] < 0
                or not math.isfinite(float(identity[3]))
                or not isinstance(identity[4], tuple)
            ):
                raise ValueError(
                    "snapshot-free observation identity is invalid"
                )
            object.__setattr__(
                self,
                "_snapshot_free_identity",
                (
                    aware_timestamp(identity[0], name="observation.asof"),
                    identity[1],
                    identity[2],
                    float(identity[3]),
                    tuple(identity[4]),
                ),
            )
        else:
            # Local import preserves the model/market_state module boundary:
            # ``market_state`` owns the concrete snapshot and imports these
            # shared contracts in turn.  Published observations must carry
            # that exact immutable authority object, never a duck-typed shim
            # whose aliases could disagree or evade snapshot validation.
            from eyes.core.market_state import MarketSnapshot

            if type(self.market_snapshot) is not MarketSnapshot:
                raise TypeError(
                    "published observation requires an exact MarketSnapshot"
                )
            if identity is not None:
                raise ValueError(
                    "published observation cannot duplicate snapshot identity"
                )
        if any(
            not isinstance(event, MarketEvent)
            or event.known_at > self.asof
            for event in self.semantic_events_this_update
        ):
            raise ValueError(
                "observation semantic event delta contains future or invalid data"
            )
        active_timeframes = tuple(self.active_timeframes)
        if not active_timeframes:
            raise ValueError("observation requires an explicit scale registry")
        object.__setattr__(self, "active_timeframes", active_timeframes)
        for name in (
            "scene_added_node_ids",
            "scene_revised_node_ids",
            "scene_added_edge_ids",
            "scene_revised_edge_ids",
            "scene_resolution_event_ids",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

        for name in (
            "liquidity_inventory_transitions_this_update",
            "liquidity_pool_transitions_this_update",
            "group3_fvg_transitions_this_update",
            "group3_order_block_transitions_this_update",
            "group4_range_transitions_this_update",
            "group4_manipulation_transitions_this_update",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(
            self,
            "group3_boundary_fvg_transitions",
            tuple(self.group3_boundary_fvg_transitions),
        )
        object.__setattr__(
            self,
            "group3_boundary_order_block_transitions",
            tuple(self.group3_boundary_order_block_transitions),
        )
        object.__setattr__(
            self,
            "group3_order_block_funnel",
            tuple(self.group3_order_block_funnel),
        )
        object.__setattr__(
            self,
            "manipulations",
            tuple(self.manipulations),
        )
        object.__setattr__(
            self,
            "group4_boundary_range_transitions",
            tuple(self.group4_boundary_range_transitions),
        )
        object.__setattr__(
            self,
            "group4_boundary_manipulation_transitions",
            tuple(self.group4_boundary_manipulation_transitions),
        )
        object.__setattr__(
            self,
            "group4_ambiguous_sweep_item_ids",
            tuple(self.group4_ambiguous_sweep_item_ids),
        )
        object.__setattr__(
            self,
            "group4_atr_unready_sweep_item_ids",
            tuple(self.group4_atr_unready_sweep_item_ids),
        )
        object.__setattr__(
            self,
            "group4_source_dispositions",
            tuple(self.group4_source_dispositions),
        )
        object.__setattr__(
            self,
            "group4_range_funnel",
            tuple(self.group4_range_funnel),
        )
        object.__setattr__(self, "anomalies", tuple(self.anomalies))
        if type(self.typed_transition_delta_available) is not bool:
            raise ValueError(
                "typed transition delta availability must be boolean"
            )
        typed_transition_collections = (
            self.liquidity_inventory_transitions_this_update,
            self.liquidity_pool_transitions_this_update,
            self.group3_fvg_transitions_this_update,
            self.group3_order_block_transitions_this_update,
            self.group4_range_transitions_this_update,
            self.group4_manipulation_transitions_this_update,
        )
        if (
            not self.typed_transition_delta_available
            and any(typed_transition_collections)
        ):
            raise ValueError(
                "typed transition delta payload requires availability"
            )
        typed_transition_contracts = (
            (
                self.liquidity_inventory_transitions_this_update,
                LiquidityInventoryItem,
            ),
            (
                self.liquidity_pool_transitions_this_update,
                LiquidityPoolState,
            ),
            (
                self.group3_fvg_transitions_this_update,
                FairValueGapState,
            ),
            (
                self.group3_order_block_transitions_this_update,
                OrderBlockState,
            ),
            (
                self.group4_range_transitions_this_update,
                DealingRangeState,
            ),
            (
                self.group4_manipulation_transitions_this_update,
                ManipulationState,
            ),
        )
        if any(
            not isinstance(item, expected_type)
            for collection, expected_type in typed_transition_contracts
            for item in collection
        ):
            raise TypeError("typed transition delta contains an invalid state")
        if not self.symbol or int(self.instrument_id) < 0:
            raise ValueError("observation contract identity is invalid")
        if not math.isfinite(float(self.price)):
            raise ValueError("observation price is invalid")
        if (
            any(
                not isinstance(value, str) or not value
                for value in self.anomalies
            )
            or len(self.anomalies) != len(set(self.anomalies))
        ):
            raise ValueError("observation anomalies are invalid")
        if (
            set(self.frames) != set(active_timeframes)
            or not set(CORE_TIMEFRAMES).issubset(active_timeframes)
            or len(active_timeframes) != len(set(active_timeframes))
            or not self.scale_registry_id
        ):
            raise ValueError(
                "observation frames must match its enabled scale registry"
            )
        if self.scene_revision_id is not None and not self.scene_revision_id:
            raise ValueError("scene revision identity cannot be empty")
        if self.scene_revision_id is None and any(
            (
                self.scene_added_node_ids,
                self.scene_revised_node_ids,
                self.scene_added_edge_ids,
                self.scene_revised_edge_ids,
                self.scene_resolution_event_ids,
            )
        ):
            raise ValueError("scene delta identities require a scene revision")
        if any(
            len(values) != len(set(values))
            or any(not isinstance(value, str) or not value for value in values)
            for values in (
                self.scene_added_node_ids,
                self.scene_revised_node_ids,
                self.scene_added_edge_ids,
                self.scene_revised_edge_ids,
                self.scene_resolution_event_ids,
            )
        ):
            raise ValueError("scene delta identities are invalid")
        for timeframe, frame in self.frames.items():
            if frame.timeframe is not timeframe:
                raise ValueError("frame mapping key does not match frame timeframe")
            if frame.cutoff > self.asof:
                raise ValueError("observation includes an uncompleted future frame")
            if any(
                (
                    state.symbol,
                    state.instrument_id,
                )
                != (self.symbol, self.instrument_id)
                for state in (
                    *frame.fair_value_gaps,
                    *frame.order_blocks,
                    *frame.dealing_ranges,
                )
            ):
                raise ValueError(
                    "observation contains a cross-contract typed entity"
                )
        if any(event.observed_at > self.asof for event in self.recent_events):
            raise ValueError("observation contains a future market event")
        if (
            any(
                not isinstance(item, OrderBlockFunnelSnapshot)
                or item.observed_at > self.asof
                for item in self.group3_order_block_funnel
            )
            or len(
                {
                    item.observed_at
                    for item in self.group3_order_block_funnel
                }
            )
            != len(self.group3_order_block_funnel)
        ):
            raise ValueError(
                "observation contains an invalid Group 3 OB funnel"
            )
        boundary_anomaly_by_reason = {
            "data_gap_reset": "data_gap_history_reset",
            "contract_change_reset": "contract_change_history_reset",
            "data_anomaly": "data_anomaly",
            "tick_size_mismatch": "tick_size_mismatch",
        }
        fvg_boundary_ids = tuple(
            state.fvg_id
            for state in self.group3_boundary_fvg_transitions
        )
        order_block_boundary_ids = tuple(
            state.order_block_id
            for state in self.group3_boundary_order_block_transitions
        )
        range_boundary_ids = tuple(
            state.range_id
            for state in self.group4_boundary_range_transitions
        )
        manipulation_boundary_ids = tuple(
            state.manipulation_id
            for state in self.group4_boundary_manipulation_transitions
        )
        if (
            len(fvg_boundary_ids) != len(set(fvg_boundary_ids))
            or len(order_block_boundary_ids)
            != len(set(order_block_boundary_ids))
            or len(range_boundary_ids) != len(set(range_boundary_ids))
            or len(manipulation_boundary_ids)
            != len(set(manipulation_boundary_ids))
        ):
            raise ValueError(
                "observation contains duplicate typed boundary transitions"
            )
        if any(
            state.lifecycle is not FairValueGapLifecycle.INVALIDATED
            or state.invalidated_at != self.asof
            or state.transition_reason
            not in boundary_anomaly_by_reason
            or boundary_anomaly_by_reason[state.transition_reason]
            not in self.anomalies
            for state in self.group3_boundary_fvg_transitions
        ):
            raise ValueError(
                "observation contains an invalid boundary FVG transition"
            )
        if any(
            state.lifecycle is not OrderBlockLifecycle.FAILED
            or state.failed_at != self.asof
            or state.transition_reason
            not in boundary_anomaly_by_reason
            or boundary_anomaly_by_reason[state.transition_reason]
            not in self.anomalies
            for state in self.group3_boundary_order_block_transitions
        ):
            raise ValueError(
                "observation contains an invalid boundary OB transition"
            )
        if any(
            state.lifecycle is not DealingRangeLifecycle.BROKEN
            or state.broken_at != self.asof
            or state.transition_reason
            not in boundary_anomaly_by_reason
            or boundary_anomaly_by_reason[state.transition_reason]
            not in self.anomalies
            for state in self.group4_boundary_range_transitions
        ):
            raise ValueError(
                "observation contains an invalid boundary range transition"
            )
        if any(
            state.lifecycle is not ManipulationLifecycle.SWEPT
            or state.censored_at != self.asof
            or state.last_updated_at != self.asof
            or state.deadline_elapsed
            or state.transition_reason
            not in boundary_anomaly_by_reason
            or boundary_anomaly_by_reason[state.transition_reason]
            not in self.anomalies
            for state in self.group4_boundary_manipulation_transitions
        ):
            raise ValueError(
                "observation contains an invalid boundary manipulation transition"
            )
        boundary_reasons = {
            state.transition_reason
            for state in (
                *self.group3_boundary_fvg_transitions,
                *self.group3_boundary_order_block_transitions,
                *self.group4_boundary_range_transitions,
                *self.group4_boundary_manipulation_transitions,
            )
        }
        if (
            self.interaction_update is not None
            and self.interaction_update.boundary_reason is not None
        ):
            boundary_reasons.add(self.interaction_update.boundary_reason)
        if len(boundary_reasons) > 1:
            raise ValueError(
                "observation mixes typed boundary transition reasons"
            )
        if any(value < 0 for value in self.event_durations_minutes.values()):
            raise ValueError("event durations cannot be negative")
        if any(value < 0 for value in self.event_ages_minutes.values()):
            raise ValueError("event ages cannot be negative")
        timeline_event_ids: set[str] = set()
        for entity_key, timeline in self.retained_entity_timelines.items():
            if (
                not isinstance(entity_key, str)
                or not entity_key
                or not isinstance(timeline, tuple)
                or not timeline
            ):
                raise ValueError(
                    "retained entity timeline identity or history is invalid"
                )
            prior_order: tuple[pd.Timestamp, int, str] | None = None
            lifecycles: set[str] = set()
            for event in timeline:
                if typed_event_entity_key(event) != entity_key:
                    raise ValueError(
                        "retained entity timeline key disagrees with its event"
                    )
                order = (
                    event.observed_at,
                    event.sequence_no,
                    event.event_id,
                )
                if prior_order is not None and order <= prior_order:
                    raise ValueError(
                        "retained entity timeline is not causally ordered"
                    )
                if event.observed_at > self.asof:
                    raise ValueError(
                        "retained entity timeline contains a future event"
                    )
                if event.event_id in timeline_event_ids:
                    raise ValueError(
                        "retained entity timelines contain a duplicate event"
                    )
                if event.lifecycle in lifecycles:
                    raise ValueError(
                        "retained entity timeline repeats a lifecycle"
                    )
                if (
                    event.event_id not in self.event_durations_minutes
                    or event.event_id not in self.event_ages_minutes
                ):
                    raise ValueError(
                        "retained entity timeline lacks duration or age"
                    )
                timeline_event_ids.add(event.event_id)
                lifecycles.add(event.lifecycle)
                prior_order = order
        if (
            tuple(sorted(set(self.incomplete_entity_timeline_keys)))
            != self.incomplete_entity_timeline_keys
            or not set(self.incomplete_entity_timeline_keys).issubset(
                self.retained_entity_timelines
            )
        ):
            raise ValueError(
                "incomplete entity timeline keys are invalid"
            )
        if (
            self.displacement is not None
            and self.displacement.asof != self.asof
        ):
            raise ValueError("displacement and market observation clocks differ")
        manipulation_ids = tuple(
            item.manipulation_id for item in self.manipulations
        )
        if (
            len(manipulation_ids) != len(set(manipulation_ids))
            or sum(
                item.lifecycle is ManipulationLifecycle.SWEPT
                and item.censored_at is None
                for item in self.manipulations
            )
            > 1
        ):
            raise ValueError(
                "observation manipulation identities or live capacity are invalid"
            )
        if any(
            (item.symbol, item.instrument_id)
            != (self.symbol, self.instrument_id)
            or item.last_updated_at > self.frames[Timeframe.M1].cutoff
            or (
                item.censored_at is not None
                and not item.deadline_elapsed
            )
            for item in self.manipulations
        ):
            raise ValueError(
                "ordinary manipulation contract, clock or boundary state is invalid"
            )
        inventory_ids = [
            item.item_id for item in self.liquidity_inventory
        ]
        if len(inventory_ids) != len(set(inventory_ids)):
            raise ValueError(
                "observation contains duplicate liquidity inventory ids"
            )
        if any(
            not isinstance(item, ManipulationSourceDisposition)
            for item in self.group4_source_dispositions
        ):
            raise TypeError(
                "observation Group 4 source disposition is not typed"
            )
        if (
            any(
                not isinstance(item, RangeFormationFunnelSnapshot)
                or item.observed_at > self.asof
                for item in self.group4_range_funnel
            )
            or len(
                {item.observed_at for item in self.group4_range_funnel}
            )
            != len(self.group4_range_funnel)
        ):
            raise ValueError(
                "observation Group 4 range funnel is invalid"
            )
        disposition_ids = tuple(
            item.source_inventory_item_id
            for item in self.group4_source_dispositions
        )
        if (
            any(
                item.observed_at > self.asof
                for item in self.group4_source_dispositions
            )
            or len(disposition_ids) != len(set(disposition_ids))
        ):
            raise ValueError(
                "observation Group 4 source dispositions are invalid"
            )
        disposition_ambiguous_ids = {
            item.source_inventory_item_id
            for item in self.group4_source_dispositions
            if item.disposition
            is ManipulationSourceDispositionKind.AMBIGUOUS_DUAL_SIDE
        }
        disposition_atr_unready_ids = {
            item.source_inventory_item_id
            for item in self.group4_source_dispositions
            if item.disposition
            is ManipulationSourceDispositionKind.ATR_UNREADY
        }
        if (
            len(self.group4_ambiguous_sweep_item_ids)
            != len(set(self.group4_ambiguous_sweep_item_ids))
            or len(self.group4_atr_unready_sweep_item_ids)
            != len(set(self.group4_atr_unready_sweep_item_ids))
            or set(self.group4_ambiguous_sweep_item_ids)
            & set(self.group4_atr_unready_sweep_item_ids)
            or any(
                not value or value not in set(inventory_ids)
                for value in (
                    *self.group4_ambiguous_sweep_item_ids,
                    *self.group4_atr_unready_sweep_item_ids,
                )
            )
            or (
                self.group4_source_dispositions
                and (
                    disposition_ambiguous_ids
                    != set(self.group4_ambiguous_sweep_item_ids)
                    or disposition_atr_unready_ids
                    != set(self.group4_atr_unready_sweep_item_ids)
                )
            )
        ):
            raise ValueError(
                "observation inventory or Group 4 ambiguity is invalid"
            )
        for item in self.liquidity_inventory:
            if item.lifecycle is LiquidityInventoryLifecycle.TARGETED:
                raise ValueError(
                    "the descriptive eye cannot target liquidity"
                )
            source_cutoff = self.frames[item.timeframe].cutoff
            if (
                item.formed_at > source_cutoff
                or item.confirmed_at > source_cutoff
            ):
                raise ValueError(
                    "liquidity inventory postdates its source frame cutoff"
                )
            if any(
                value is not None and value > self.asof
                for value in (
                    item.confirmed_at,
                    item.targeted_at,
                    item.consumed_at,
                )
            ):
                raise ValueError(
                    "observation contains future liquidity inventory state"
                )
            if (
                item.consumed_at is not None
                and item.consumed_at > self.frames[Timeframe.M1].cutoff
            ):
                raise ValueError(
                    "liquidity consumption postdates the completed 1m cutoff"
                )
        pool_ids = [item.pool_id for item in self.liquidity_pool_states]
        if len(pool_ids) != len(set(pool_ids)):
            raise ValueError(
                "observation contains duplicate liquidity pool ids"
            )
        inventory_by_id = {
            item.item_id: item for item in self.liquidity_inventory
        }
        swing_ids_by_timeframe = {
            timeframe: {swing.swing_id for swing in frame.swings}
            for timeframe, frame in self.frames.items()
        }
        range_ids = {
            item.range_id
            for item in self.frames[Timeframe.H1].dealing_ranges
        }
        for item in self.liquidity_inventory:
            if item.kind == "range_boundary" and (
                item.timeframe is not Timeframe.H1
                or len(set(item.source_ids) & range_ids) != 1
            ):
                raise ValueError(
                    "range-boundary inventory lacks one retained range identity"
                )
            if item.kind == "swing" and (
                len(item.source_ids) != 1
                or item.source_ids[0]
                not in swing_ids_by_timeframe[item.timeframe]
            ):
                raise ValueError(
                    "swing inventory lacks one retained swing identity"
                )
        frozen_source_pool_id_list = [
            pool.pool_id
            for frame in self.frames.values()
            for pool in frame.liquidity_pools
        ]
        if (
            len(frozen_source_pool_id_list)
            != len(set(frozen_source_pool_id_list))
            or set(frozen_source_pool_id_list) != set(pool_ids)
        ):
            raise ValueError(
                "frozen pool sources and authoritative pool set disagree"
            )
        inventory_pool_ids = {
            item.item_id
            for item in self.liquidity_inventory
            if item.kind in {"equal_highs", "equal_lows"}
        }
        if inventory_pool_ids != {
            f"pool:{pool_id}" for pool_id in pool_ids
        }:
            raise ValueError(
                "pool inventory and authoritative pool set disagree"
            )
        for item in self.manipulations:
            if item.source_inventory_item_id in inventory_by_id:
                continue
            if (
                (
                    item.lifecycle is ManipulationLifecycle.SWEPT
                    and item.censored_at is None
                )
                or item.last_updated_at == self.asof
            ):
                continue
            else:
                raise ValueError(
                    "manipulation lacks its retained inventory identity"
                )
        if (
            self.interaction_update is not None
            and not isinstance(self.interaction_update, InteractionUpdate)
        ):
            raise TypeError("interaction update has an invalid contract")
        if self.interaction_update is not None:
            interaction = self.interaction_update
            clocked_interactions = (
                *interaction.zone_interactions,
                *interaction.reacceptance_interactions,
                *interaction.interaction_paths,
            )
            if any(
                item.symbol != self.symbol
                or item.instrument_id != self.instrument_id
                or item.last_updated_at > self.frames[Timeframe.M1].cutoff
                or item.last_updated_at > self.asof
                for item in clocked_interactions
            ):
                raise ValueError(
                    "interaction fact contract or completed clock is invalid"
                )
            if any(
                fact.resolved_at > self.frames[Timeframe.M1].cutoff
                or fact.resolved_at > self.asof
                for fact in interaction.micro_break_facts
            ):
                raise ValueError("interaction micro-break fact is in the future")
            location_ids = tuple(
                item.location_id for item in interaction.zone_interactions
            )
            reacceptance_ids = tuple(
                item.reacceptance_id
                for item in interaction.reacceptance_interactions
            )
            reference_ids = tuple(
                item.reference_id for item in interaction.micro_break_facts
            )
            sequence_ids = tuple(
                item.sequence_id for item in interaction.interaction_paths
            )
            if any(
                len(values) != len(set(values))
                for values in (
                    location_ids,
                    reacceptance_ids,
                    reference_ids,
                    sequence_ids,
                )
            ):
                raise ValueError("interaction observation identities repeat")
            path_contexts = {
                (state.context_kind, state.context_id)
                for state in interaction.interaction_paths
            }
            if len(path_contexts) != len(interaction.interaction_paths):
                raise ValueError(
                    "retained interaction path contexts must be unique"
                )
            if any(
                ("zone_return", state.location_id) not in path_contexts
                for state in interaction.zone_interactions
            ):
                raise ValueError(
                    "zone interaction lacks its retained physical path"
                )
            if any(
                (
                    (
                        "zone_return"
                        if state.context_kind == "entry_zone"
                        else "pool_reversal"
                    ),
                    state.context_id,
                )
                not in path_contexts
                for state in interaction.reacceptance_interactions
            ):
                raise ValueError(
                    "reacceptance interaction lacks its physical path"
                )
            if interaction.boundary_reason is not None:
                if (
                    boundary_anomaly_by_reason[interaction.boundary_reason]
                    not in self.anomalies
                ):
                    raise ValueError(
                        "interaction boundary lacks its observation anomaly"
                    )
                boundary_paths = interaction.interaction_path_transitions
                boundary_reacceptances = (
                    interaction.reacceptance_interaction_transitions
                )
                if any(
                    state.lifecycle is not PathSequenceLifecycle.CENSORED
                    or state.ended_at != self.asof
                    or state.last_updated_at != self.asof
                    or state.transition_reason
                    != interaction.boundary_reason
                    or boundary_anomaly_by_reason[state.transition_reason]
                    not in self.anomalies
                    or (
                        state.transition_reason == "contract_change_reset"
                        and (state.symbol, state.instrument_id)
                        == (self.symbol, self.instrument_id)
                    )
                    or (
                        state.transition_reason != "contract_change_reset"
                        and (state.symbol, state.instrument_id)
                        != (self.symbol, self.instrument_id)
                    )
                    for state in boundary_paths
                ):
                    raise ValueError(
                        "interaction boundary path transition is invalid"
                    )
                if any(
                    state.censored_at != self.asof
                    or state.last_updated_at != self.asof
                    for state in boundary_reacceptances
                ):
                    raise ValueError(
                        "interaction boundary reacceptance clock is invalid"
                    )
                boundary_path_contexts = {
                    (state.context_kind, state.context_id)
                    for state in boundary_paths
                }
                if any(
                    (
                        (
                            "zone_return"
                            if state.context_kind == "entry_zone"
                            else "pool_reversal"
                        ),
                        state.context_id,
                    )
                    not in boundary_path_contexts
                    for state in boundary_reacceptances
                ):
                    raise ValueError(
                        "boundary reacceptance lacks its censored path context"
                    )

    def _with_scene_delta(
        self,
        *,
        scene_revision_id: str | None,
        scene_added_node_ids: tuple[str, ...] = (),
        scene_revised_node_ids: tuple[str, ...] = (),
        scene_added_edge_ids: tuple[str, ...] = (),
        scene_revised_edge_ids: tuple[str, ...] = (),
        scene_resolution_event_ids: tuple[str, ...] = (),
    ) -> "MarketObservation":
        """Attach the graph delta without revalidating the immutable Eye payload.

        ``MarketObservation`` has already completed its full causal and typed
        contract validation before the Scene Graph consumes it.  Graph
        projection changes only these six transport fields, so rebuilding the
        dataclass through :func:`dataclasses.replace` needlessly repeats the
        complete (and deliberately strict) validation of frames, inventories,
        timelines, and Group 3-5 state.

        This private clone path preserves every validated object reference and
        validates exactly the newly attached revision/delta identities.  It is
        intentionally not a general-purpose unchecked replacement API.
        """
        delta_names = (
            "scene_added_node_ids",
            "scene_revised_node_ids",
            "scene_added_edge_ids",
            "scene_revised_edge_ids",
            "scene_resolution_event_ids",
        )
        delta_values = tuple(
            tuple(value)
            for value in (
                scene_added_node_ids,
                scene_revised_node_ids,
                scene_added_edge_ids,
                scene_revised_edge_ids,
                scene_resolution_event_ids,
            )
        )
        if scene_revision_id is not None and not scene_revision_id:
            raise ValueError("scene revision identity cannot be empty")
        if scene_revision_id is None and any(delta_values):
            raise ValueError("scene delta identities require a scene revision")
        if any(
            len(values) != len(set(values))
            or any(not isinstance(value, str) or not value for value in values)
            for values in delta_values
        ):
            raise ValueError("scene delta identities are invalid")

        clone = object.__new__(type(self))
        object.__setattr__(clone, "__dict__", self.__dict__.copy())
        object.__setattr__(clone, "scene_revision_id", scene_revision_id)
        for name, values in zip(delta_names, delta_values, strict=True):
            object.__setattr__(clone, name, values)
        return clone

    def __getstate__(self) -> Mapping[str, Any]:
        if self.market_snapshot is not None:
            from eyes.core.market_state import MarketSnapshot

            if type(self.market_snapshot) is not MarketSnapshot:
                raise TypeError(
                    "published observation requires an exact MarketSnapshot"
                )
        return _exact_dataclass_pickle_state(
            self,
            schema_version=MARKET_OBSERVATION_SCHEMA_VERSION,
            label="MarketObservation",
        )

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        _restore_exact_dataclass_pickle_state(
            self,
            state,
            schema_version=MARKET_OBSERVATION_SCHEMA_VERSION,
            label="MarketObservation",
        )

    def frame(self, timeframe: Timeframe) -> FrameObservation:
        return self.frames[timeframe]


__all__ = [
    "DisplacementObservation",
    "DisplacementTransitionObservation",
    "FrameObservation",
    "MarketEvent",
    "MarketObservation",
    "typed_event_entity_key",
]
