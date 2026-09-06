"""Single owner of canonical semantic-event emission for the Trading Eye.

Detectors own registered market state; they never see the event stream. Every
canonical fact, however, must cite the exact prior events it descends from — a
Range-Auction manipulation names the swing, level and penetration events that
Structure and Liquidity produced, and every event names its normalized BAR root.

That cross-detector ancestry index and the emitter that writes it are one
responsibility, and this class is its only owner. It holds the shared Eye inputs
(`config`, `semantic_registry`, `audit_store`, `memory`, the active timeframes,
the scale-registry identity and the structure protocol) under their original
names so the emission logic is unchanged by having a home.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace
import hashlib
import json
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .causal import ReaderUpdate
from .event_memory import EventMemory
from .event_store import EventStore
from .interaction import InteractionUpdate
from .market_state import DeliveryPhaseTransition, session_name_phase
from .model import (
    BOS_CONFIRMATION_REASON,
    BOSLifecycle,
    BOSScope,
    Candle,
    DealingRangeLifecycle,
    Direction,
    EventKind,
    EventOrigin,
    FairValueGapLifecycle,
    FairValueGapState,
    FrameObservation,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityPoolLifecycle,
    ManipulationLifecycle,
    MarketEvent,
    MarketObservation,
    OrderBlockLifecycle,
    PathSequenceLifecycle,
    PathSequenceState,
    StructureLifecycle,
    SupportResistanceLifecycle,
    SupportResistanceState,
    SwingLifecycle,
    SwingRelation,
    Timeframe,
    SMC_SEMANTIC_VERSION,
    candle_identity,
    clamp,
    to_primitive,
)
from .range_auction import (
    BALANCE_CLAIM_ABANDONED,
    CausalRangeAuctionTracker,
    RangeAuctionUpdate,
)
from .scale_registry import _TIMEFRAME_MINUTES
from .structure import StructureConfig
from .zone import ZoneUpdate


def _event(
    kind: EventKind,
    observed_at: pd.Timestamp,
    timeframe: Timeframe,
    side: str | None,
    price: float | None,
    strength: float,
    source_ids: Iterable[str] = (),
    details: Mapping[str, object] | None = None,
    *,
    entity_id: str | None = None,
    lifecycle: str | None = None,
    formed_at: pd.Timestamp | None = None,
    confirmed_at: pd.Timestamp | None = None,
    ended_at: pd.Timestamp | None = None,
    direction: Direction | None = None,
    transition_reason: str | None = None,
    event_time: pd.Timestamp | None = None,
    known_at: pd.Timestamp | None = None,
    semantic_version: str = SMC_SEMANTIC_VERSION,
    evidence: Mapping[str, object] | None = None,
    zone: tuple[float, float] | None = None,
    source_event_ids: Iterable[str] = (),
    source_data_ids: Iterable[str] = (),
    source_entity_ids: Iterable[str] = (),
    context_event_ids: Iterable[str] = (),
    origin: EventOrigin = EventOrigin.LEGACY_TRANSPORT,
) -> MarketEvent:
    source_ids = tuple(
        str(value) for value in source_ids if value is not None
    )
    explicit_event_ids = tuple(
        str(value) for value in source_event_ids if value is not None
    )
    data_ids = tuple(
        str(value) for value in source_data_ids if value is not None
    )
    entity_ids = tuple(
        str(value) for value in source_entity_ids if value is not None
    )
    context_ids = tuple(
        str(value) for value in context_event_ids if value is not None
    )
    raw = (
        f"{semantic_version}|{kind.value}|{observed_at.isoformat()}|"
        f"{timeframe.value}|{side}|"
        f"{price}|{'|'.join(source_ids)}|{'|'.join(explicit_event_ids)}|"
        f"{'|'.join(data_ids)}|{'|'.join(entity_ids)}|"
        f"{'|'.join(context_ids)}|{EventOrigin(origin).value}|"
        f"{entity_id}|{lifecycle}"
    )
    return MarketEvent(
        event_id=hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24],
        kind=kind,
        observed_at=observed_at,
        timeframe=timeframe,
        side=side,
        price=price,
        strength=clamp(strength),
        source_ids=tuple(source_ids),
        details={} if details is None else dict(details),
        entity_id=entity_id,
        lifecycle=lifecycle,
        formed_at=formed_at,
        confirmed_at=confirmed_at,
        ended_at=ended_at,
        direction=direction,
        transition_reason=transition_reason,
        event_time=event_time or observed_at,
        known_at=known_at or observed_at,
        semantic_version=semantic_version,
        evidence={} if evidence is None else dict(evidence),
        zone=zone,
        source_event_ids=explicit_event_ids,
        source_data_ids=data_ids,
        source_entity_ids=entity_ids,
        context_event_ids=context_ids,
        origin=origin,
    )


class SemanticEventEmitter:
    """Canonical event emitter plus the cross-detector ancestry index."""

    def __init__(
        self,
        *,
        config,
        semantic_registry,
        audit_store: EventStore,
        memory: EventMemory,
        active_timeframes: tuple[Timeframe, ...],
        scale_registry_id: str,
        structure_config: StructureConfig,
    ) -> None:
        self.config = config
        self.semantic_registry = semantic_registry
        self.audit_store = audit_store
        self.memory = memory
        self._active_timeframes = active_timeframes
        self._scale_registry_id = scale_registry_id
        self._structure_config = structure_config
        self._known_level_ids: set[tuple[str, SwingLifecycle]] = set()
        self._known_level_order: deque[
            tuple[str, SwingLifecycle]
        ] = deque(
            maxlen=max(2048, self.config.memory_events * 4)
        )
        self._known_structural_leg_ids: set[str] = set()
        self._known_structural_leg_order: deque[str] = deque(
            maxlen=max(2048, self.config.memory_events * 4)
        )
        self._confirmed_swing_event_ids: dict[str, str] = {}
        self._structural_leg_event_ids: dict[str, str] = {}
        self._structure_direction_event_ids: dict[str, str] = {}
        self._latest_structure_direction_event_ids: dict[
            Timeframe,
            str,
        ] = {}
        self._bar_event_ids_by_candle_id: dict[str, str] = {}
        self._bar_close_by_candle_id: dict[str, float] = {}
        self._bar_close_by_event_id: dict[str, float] = {}
        # (low, high) of each real BAR, so a confirmed Swing can freeze
        # the price envelope of its own definitional window.
        self._bar_range_by_event_id: dict[str, tuple[float, float]] = {}
        self._bar_event_ids_by_timeframe: dict[
            Timeframe,
            list[tuple[pd.Timestamp, str]],
        ] = {
            timeframe: [] for timeframe in self._active_timeframes
        }
        # Keep the inclusive normalized M1 clock/root index above for session
        # replay and cold-prefix initialization.  Semantic detectors consume
        # only real-completed bars, matching StructureTracker and the other
        # primitive reducers that treat synthetic no-trade minutes as clock
        # advancement only.
        self._real_bar_event_ids_by_timeframe: dict[
            Timeframe,
            list[tuple[pd.Timestamp, str]],
        ] = {
            timeframe: [] for timeframe in self._active_timeframes
        }
        self._level_touch_event_ids: dict[
            tuple[str, pd.Timestamp],
            str,
        ] = {}
        self._candidate_level_event_ids: dict[str, str] = {}
        self._known_level_touch_ids: set[str] = set()
        # Source reducers expose complete touch histories for live zones.
        # Evicting these occurrence keys causes old touches to be rediscovered
        # on every later frame, so retain the compact IDs for the contract
        # epoch and clear them only at a hard boundary.
        self._known_level_touch_order: deque[str] = deque()
        self._penetration_event_ids: dict[
            tuple[str, Timeframe, pd.Timestamp],
            str,
        ] = {}
        self._raw_break_event_ids: dict[str, str] = {}
        self._displacement_event_ids: dict[str, str] = {}
        self._protected_swing_event_ids: dict[str, str] = {}
        self._terminal_crossing_events: dict[str, MarketEvent] = {}
        self._fvg_created_event_ids: dict[str, str] = {}
        self._fvg_first_retest_event_ids: dict[str, str] = {}
        self._fvg_terminal_event_ids: dict[str, str] = {}
        self._base_origin_core_event_ids: dict[str, str] = {}
        self._origin_zone_created_event_ids: dict[str, str] = {}
        self._range_created_event_ids: dict[str, str] = {}
        self._range_active_event_ids: dict[str, str] = {}
        self._balance_range_observed_event_ids: dict[str, str] = {}
        self._range_terminal_event_ids: dict[str, str] = {}
        self._range_boundary_level_ids: dict[tuple[str, str], str] = {}
        self._last_invalidated_range_event_id: str | None = None
        self._known_displacement_transition_ids: set[str] = set()
        self._known_displacement_transition_order: deque[str] = deque(
            maxlen=max(512, self.config.memory_events * 2)
        )
        self._known_structure_events: set[
            tuple[str, BOSLifecycle]
        ] = set()
        self._known_structure_event_order: deque[
            tuple[str, BOSLifecycle]
        ] = deque(maxlen=max(512, self.config.memory_events * 2))
        self._known_sequence_events: set[
            tuple[str, StructureLifecycle]
        ] = set()
        self._known_sequence_event_order: deque[
            tuple[str, StructureLifecycle]
        ] = deque(maxlen=max(512, self.config.memory_events * 2))
        self._liquidity_entity_revisions: dict[
            str,
            tuple[object, ...],
        ] = {}

    def rebind_memory(self, memory: EventMemory) -> None:
        """Follow the observer onto the memory it rebuilt for a new epoch."""

        self.memory = memory

    def reset_contract_state(self) -> None:
        """Clear every ancestry index carried over from the prior contract."""

        self._known_level_ids.clear()
        self._known_level_order.clear()
        self._known_structural_leg_ids.clear()
        self._known_structural_leg_order.clear()
        self._confirmed_swing_event_ids.clear()
        self._structural_leg_event_ids.clear()
        self._structure_direction_event_ids.clear()
        self._latest_structure_direction_event_ids.clear()
        self._bar_event_ids_by_candle_id.clear()
        self._bar_close_by_candle_id.clear()
        self._bar_close_by_event_id.clear()
        self._bar_range_by_event_id.clear()
        for bar_events in self._bar_event_ids_by_timeframe.values():
            bar_events.clear()
        for bar_events in self._real_bar_event_ids_by_timeframe.values():
            bar_events.clear()
        self._level_touch_event_ids.clear()
        self._candidate_level_event_ids.clear()
        self._known_level_touch_ids.clear()
        self._known_level_touch_order.clear()
        self._penetration_event_ids.clear()
        self._raw_break_event_ids.clear()
        self._displacement_event_ids.clear()
        self._protected_swing_event_ids.clear()
        self._terminal_crossing_events.clear()
        self._fvg_created_event_ids.clear()
        self._fvg_first_retest_event_ids.clear()
        self._fvg_terminal_event_ids.clear()
        self._base_origin_core_event_ids.clear()
        self._origin_zone_created_event_ids.clear()
        self._range_created_event_ids.clear()
        self._range_active_event_ids.clear()
        self._balance_range_observed_event_ids.clear()
        self._range_terminal_event_ids.clear()
        self._range_boundary_level_ids.clear()
        self._last_invalidated_range_event_id = None
        self._known_displacement_transition_ids.clear()
        self._known_displacement_transition_order.clear()
        self._known_structure_events.clear()
        self._known_structure_event_order.clear()
        self._known_sequence_events.clear()
        self._known_sequence_event_order.clear()
        self._liquidity_entity_revisions.clear()

    def emit_reference_period_retirements(
        self,
        retired: Sequence[LiquidityInventoryItem],
        *,
        observed_at: pd.Timestamp,
        replacement_period: str,
    ) -> None:
        """Emit the retirement terminals a replaced reference period leaves."""

        for item in retired:
            self.memory.append(
                _event(
                    EventKind.LIQUIDITY_RETIRED,
                    observed_at,
                    Timeframe.M1,
                    item.side,
                    item.price,
                    0.0,
                    (item.item_id,),
                    {
                        "source_kind": item.kind,
                        "replacement_period": replacement_period,
                    },
                    transition_reason=(
                        "reference_period_replaced"
                    ),
                )
            )

    def emit_boundary_structure_break_failed(
        self,
        failed_bos,
        observed_at: pd.Timestamp,
        reset_anomalies: Sequence[str],
    ) -> None:
        """Emit the boundary failed-BOS terminals for a contract reset."""

        for item in (
            () if self.config.range_auction_projection_only else failed_bos
        ):
            key = (item.bos_id, item.lifecycle)
            self._remember_bounded(
                key,
                known=self._known_structure_events,
                order=self._known_structure_event_order,
            )
            self.memory.append(
                _event(
                    EventKind.STRUCTURE_BREAK_FAILED,
                    item.resolved_at or observed_at,
                    item.timeframe,
                    (
                        "above"
                        if item.direction is Direction.LONG
                        else "below"
                    ),
                    item.target_price,
                    0.0,
                    tuple(
                        value
                        for value in (
                            item.target_swing_id,
                            item.source_structure_id,
                        )
                        if value is not None
                    ),
                    {
                        "bos_id": item.bos_id,
                        "scope": item.scope.value,
                        "pending_at": item.pending_at.isoformat(),
                        "resolved_at": (
                            None
                            if item.resolved_at is None
                            else item.resolved_at.isoformat()
                        ),
                        "failure_reason": item.failure_reason,
                        "direction": item.direction.value,
                        "lifecycle": item.lifecycle.value,
                        "target_swing_id": item.target_swing_id,
                        "source_structure_id": item.source_structure_id,
                        "target_ticks": item.target_ticks,
                        "age_bars": item.age_bars,
                        "attempt_count": item.attempt_count,
                        "attempt_clocks": tuple(
                            value.isoformat()
                            for value in item.attempt_clocks
                        ),
                        "last_attempt_at": (
                            None
                            if item.last_attempt_at is None
                            else item.last_attempt_at.isoformat()
                        ),
                        "reset_anomalies": tuple(reset_anomalies),
                    },
                    entity_id=item.bos_id,
                    lifecycle=item.lifecycle.value,
                    formed_at=item.pending_at,
                    ended_at=item.resolved_at or observed_at,
                    direction=item.direction,
                    transition_reason=item.failure_reason,
                )
            )

    @staticmethod
    def _remember_bounded(
        value,
        *,
        known: set,
        order: deque,
    ) -> bool:
        if value in known:
            return False
        if len(order) == order.maxlen and order:
            known.discard(order[0])
        order.append(value)
        known.add(value)
        return True

    def _append_semantic_atomic(
        self,
        kind: EventKind,
        known_at: pd.Timestamp,
        timeframe: Timeframe,
        side: str | None,
        price: float | None,
        strength: float,
        source_event_ids: Iterable[str] = (),
        evidence: Mapping[str, object] | None = None,
        *,
        direction: Direction | None = None,
        event_time: pd.Timestamp | None = None,
        zone: tuple[float, float] | None = None,
        source_data_ids: Iterable[str] = (),
        source_entity_ids: Iterable[str] = (),
        context_event_ids: Iterable[str] = (),
        sequence_floor: int | None = None,
    ) -> MarketEvent:
        registry = getattr(self, "semantic_registry", None)
        if registry is None:
            raise ValueError(
                "canonical semantic emitter requires a loaded registry"
            )
        # Epoch reset is journal/control infrastructure rather than an SMC
        # semantic concept.  It intentionally remains outside the semantic
        # registry while sharing the immutable append path.
        if (
            kind is not EventKind.MARKET_EPOCH_RESET
            and kind not in registry.canonical_emitted_event_kinds
        ):
            binding = registry.event_binding_by_kind.get(kind)
            status = "unregistered" if binding is None else binding.status
            raise ValueError(
                "canonical semantic emitter rejects non-emitted event kind: "
                f"{kind.value} ({status})"
            )
        source_event_ids = tuple(
            dict.fromkeys(
                str(value) for value in source_event_ids if value is not None
            )
        )
        source_data_ids = tuple(
            dict.fromkeys(
                str(value) for value in source_data_ids if value is not None
            )
        )
        source_entity_ids = tuple(
            dict.fromkeys(
                str(value) for value in source_entity_ids if value is not None
            )
        )
        context_event_ids = tuple(
            dict.fromkeys(
                str(value) for value in context_event_ids if value is not None
            )
        )
        session_name, session_phase = session_name_phase(known_at)
        payload = {
            "canonical_semantic": True,
            "projection_only": False,
            "session_name": session_name,
            "session_phase": session_phase,
            **({} if evidence is None else dict(evidence)),
        }
        event = _event(
            kind,
            known_at,
            timeframe,
            side,
            price,
            strength,
            source_event_ids,
            payload,
            direction=direction,
            event_time=event_time or known_at,
            known_at=known_at,
            evidence=payload,
            zone=zone,
            source_event_ids=source_event_ids,
            source_data_ids=source_data_ids,
            source_entity_ids=source_entity_ids,
            context_event_ids=context_event_ids,
            origin=EventOrigin.SEMANTIC_ATOMIC,
        )
        identity_payload = {
            "semantic_version": event.semantic_version,
            "semantic_type": event.kind.value,
            "event_time": event.event_time,
            "known_at": event.known_at,
            "timeframe": event.timeframe.value,
            "side": event.side,
            "price": event.price,
            "direction": (
                None if event.direction is None else event.direction.value
            ),
            "source_event_ids": event.source_event_ids,
            "source_data_ids": event.source_data_ids,
            "source_entity_ids": event.source_entity_ids,
            "context_event_ids": event.context_event_ids,
            "origin": event.origin.value,
            "evidence": event.evidence,
            "zone": event.zone,
        }
        event = replace(
            event,
            event_id=hashlib.sha256(
                json.dumps(
                    to_primitive(identity_payload),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:24],
        )
        existing = self.audit_store.get(event.event_id)
        if existing is not None:
            # Event identity intentionally excludes the convenience
            # ``strength`` score: a long-lived zone can revise that score
            # after the touch occurred. If a bounded hot-memory key is later
            # evicted and the same occurrence is rediscovered, retain the
            # first-known frozen score rather than rewriting history or
            # creating a second semantic occurrence.
            retry = replace(
                event,
                sequence_no=existing.sequence_no,
                strength=existing.strength,
            )
            if retry != existing:
                raise ValueError(
                    "canonical semantic event id conflicts with audit history: "
                    f"{event.event_id} ({event.kind.value})"
                )
            return existing
        # Return the memory-assigned copy.  ``append`` stamps the transport
        # ordinal, so returning the pre-append object hands callers an event
        # whose ``sequence_no`` disagrees with the one the audit store commits.
        return self.memory.append(
            event,
            include_in_recent=False,
            sequence_floor=sequence_floor,
        )

    def _normalized_crossing_level_id(self, level_id: str) -> str:
        value = str(level_id)
        if value.startswith("swing:"):
            return value
        if (
            value in self._confirmed_swing_event_ids
            and f"swing:{value}" in self._candidate_level_event_ids
        ):
            return f"swing:{value}"
        return value

    def _crossing_generation_id(
        self,
        *,
        level_id: str,
        timeframe: Timeframe,
        crossed_at: pd.Timestamp,
    ) -> str:
        normalized_level_id = self._normalized_crossing_level_id(level_id)
        payload = (
            f"{self.semantic_registry.semantic_version}|crossing-v1|"
            f"{timeframe.value}|{normalized_level_id}|"
            f"{pd.Timestamp(crossed_at).isoformat()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _penetration_key(
        self,
        *,
        level_id: str,
        timeframe: Timeframe,
        crossed_at: pd.Timestamp,
    ) -> tuple[str, Timeframe, pd.Timestamp]:
        """Return the unique lookup key for one crossing generation."""

        return (
            self._normalized_crossing_level_id(level_id),
            timeframe,
            pd.Timestamp(crossed_at),
        )

    def _live_protected_assignments(
        self,
        timeframe: Timeframe,
        *,
        asof: pd.Timestamp,
    ) -> tuple[tuple[str, str, MarketEvent], ...]:
        """Resolve the exact pending-or-committed protection for one TF.

        ``_protected_swing_event_ids`` is the producer-side custody index for
        the assignment that the persistent market-state reducer considers
        live.  Reading through ``audit_event_including_pending`` is important:
        an assignment and a later break may be published at the same observer
        clock before the atomic audit batch is flushed.
        """

        clock = pd.Timestamp(asof)
        if clock.tzinfo is None:
            raise ValueError("protected-swing custody clock must be aware")
        assignments: list[tuple[str, str, MarketEvent]] = []
        for swing_id, event_id in tuple(
            self._protected_swing_event_ids.items()
        ):
            event = self.memory.audit_event_including_pending(event_id)
            if (
                event is None
                or event.kind is not EventKind.PROTECTED_SWING_ASSIGNED
                or not event.is_canonical_semantic
                or event.evidence.get("protected_swing_id") != swing_id
                or swing_id not in event.source_entity_ids
                or event.direction not in {Direction.LONG, Direction.SHORT}
                or event.price is None
                or event.known_at > clock
            ):
                raise ValueError(
                    "live protected-swing custody references an invalid "
                    f"assignment: {swing_id} -> {event_id}"
                )
            if event.timeframe is timeframe:
                assignments.append((swing_id, event_id, event))
        if len(assignments) > 1:
            raise ValueError(
                "one timeframe cannot retain multiple live protected-swing "
                "assignments"
            )
        return tuple(assignments)

    def _replace_live_protected_assignment(
        self,
        event: MarketEvent,
        prior_assignments: tuple[tuple[str, str, MarketEvent], ...],
    ) -> None:
        """Publish a successfully appended assignment into producer custody."""

        protected_swing_id = event.evidence.get("protected_swing_id")
        if (
            event.kind is not EventKind.PROTECTED_SWING_ASSIGNED
            or not event.is_canonical_semantic
            or not isinstance(protected_swing_id, str)
            or not protected_swing_id
            or protected_swing_id not in event.source_entity_ids
        ):
            raise ValueError(
                "protected-swing custody requires a valid assignment"
            )
        for swing_id, event_id, prior in prior_assignments:
            if prior.timeframe is not event.timeframe:
                raise ValueError(
                    "protected-swing replacement crossed timeframe custody"
                )
            if self._protected_swing_event_ids.get(swing_id) == event_id:
                self._protected_swing_event_ids.pop(swing_id)
        self._protected_swing_event_ids[protected_swing_id] = event.event_id

    def _crossing_bar_roles(
        self,
        kind: EventKind,
        source_ids: tuple[str, ...],
    ) -> dict[str, object]:
        """Name the bars that decided a crossing, in their causal roles.

        The roles are read back off the terminal's own ancestry rather than
        asked of each call site, so every crossing terminal reports them the
        same way and none can drift.
        """

        penetration_bar_id: str | None = None
        resolution_bar_id: str | None = None
        for event_id in source_ids:
            event = self.memory.audit_event_including_pending(event_id)
            if event is None:
                continue
            if event.kind is EventKind.BAR_COMPLETED:
                resolution_bar_id = event.event_id
            elif event.kind is EventKind.LEVEL_PENETRATED:
                for parent_id in event.source_event_ids:
                    parent = self.memory.audit_event_including_pending(
                        parent_id
                    )
                    if parent is not None and (
                        parent.kind is EventKind.BAR_COMPLETED
                    ):
                        penetration_bar_id = parent.event_id
        held = kind is EventKind.ACCEPTANCE_CONFIRMED
        constituents = tuple(
            dict.fromkeys(
                value
                for value in (penetration_bar_id, resolution_bar_id)
                if value is not None
            )
        )
        return {
            "constituent_bar_ids": constituents,
            "penetration_bar_id": penetration_bar_id,
            # One verdict, one resolving role: the bar either came back inside
            # or it held outside.
            "reentry_bar_id": None if held else resolution_bar_id,
            "hold_bar_id": resolution_bar_id if held else None,
            "outside_close_ids": (
                (resolution_bar_id,)
                if held and resolution_bar_id is not None
                else ()
            ),
        }

    def _append_crossing_resolution(
        self,
        kind: EventKind,
        resolved_at: pd.Timestamp,
        timeframe: Timeframe,
        side: str,
        price: float,
        strength: float,
        source_event_ids: Iterable[str],
        evidence: Mapping[str, object],
        *,
        direction: Direction,
        crossed_at: pd.Timestamp,
        known_at: pd.Timestamp,
        zone: tuple[float, float] | None = None,
        context_event_ids: Iterable[str] = (),
    ) -> MarketEvent:
        """Append exactly one terminal result for one crossing generation.

        ``resolved_at`` is the market clock at which the crossing was decided
        and is preserved in evidence; ``known_at`` is the observation clock at
        which the Eye could first derive the terminal.  They differ whenever a
        reducer only reaches its terminal verdict on a later observation, and
        stamping the market clock as ``known_at`` would backdate the fact
        behind ancestry it has to cite.
        """

        if pd.Timestamp(known_at) < pd.Timestamp(resolved_at):
            raise ValueError(
                "crossing terminal cannot be known before it resolved"
            )
        if kind not in {
            EventKind.SWEEP_CONFIRMED,
            EventKind.ACCEPTANCE_CONFIRMED,
        }:
            raise ValueError("crossing terminal kind must be sweep or acceptance")
        level_id = str(evidence.get("level_id", ""))
        if not level_id:
            raise ValueError("crossing terminal requires a level_id")
        generation_id = self._crossing_generation_id(
            level_id=level_id,
            timeframe=timeframe,
            crossed_at=crossed_at,
        )
        normalized_level_id = self._normalized_crossing_level_id(level_id)
        protected_swing_id = (
            normalized_level_id.removeprefix("swing:")
            if normalized_level_id.startswith("swing:")
            else normalized_level_id
        )
        protected_event_id = self._protected_swing_event_ids.get(
            protected_swing_id
        )
        protected_assignment = (
            self.memory.audit_event_including_pending(protected_event_id)
            if protected_event_id is not None
            else None
        )
        if (
            protected_assignment is not None
            and protected_assignment.known_at > pd.Timestamp(known_at)
        ):
            raise ValueError(
                "protected assignment cannot be known after crossing "
                "resolution"
            )
        source_ids = tuple(source_event_ids)
        context_ids = tuple(context_event_ids)
        if (
            protected_event_id is not None
            and protected_event_id not in context_ids
        ):
            context_ids = (*context_ids, protected_event_id)
        prior = self._terminal_crossing_events.get(generation_id)
        if prior is not None:
            if (
                prior.kind is not kind
                or prior.direction is not direction
                or prior.side != side
                or prior.timeframe is not timeframe
                # A generation keeps the clock it was first knowable at;
                # re-deriving it on a later observation is idempotent, but
                # never allowed to backdate the terminal.
                or prior.known_at > pd.Timestamp(known_at)
                # The market clock that decided the crossing is part of the
                # generation's frozen result and must not move.
                or prior.evidence.get("resolved_at")
                != pd.Timestamp(resolved_at).isoformat()
                or prior.event_time != pd.Timestamp(crossed_at)
                or prior.evidence.get("crossing_generation_id")
                != generation_id
                or prior.evidence.get("crossed_at")
                != pd.Timestamp(crossed_at).isoformat()
            ):
                raise ValueError(
                    "one crossing generation produced conflicting terminal "
                    f"resolutions: {generation_id}"
                )
            return prior
        payload = {
            **self._crossing_bar_roles(kind, source_ids),
            **dict(evidence),
            "crossing_generation_id": generation_id,
            "crossed_at": pd.Timestamp(crossed_at).isoformat(),
            "resolved_at": pd.Timestamp(resolved_at).isoformat(),
            **(
                {
                    "protected_swing_id": protected_swing_id,
                    "protected_swing_event_id": protected_event_id,
                }
                if protected_event_id is not None
                else {}
            ),
        }
        event = self._append_semantic_atomic(
            kind,
            known_at,
            timeframe,
            side,
            price,
            strength,
            source_ids,
            payload,
            direction=direction,
            event_time=crossed_at,
            zone=zone,
            source_entity_ids=(normalized_level_id,),
            context_event_ids=context_ids,
        )
        self._terminal_crossing_events[generation_id] = event
        opposite_assignment_direction = (
            Direction.SHORT
            if (
                protected_assignment is not None
                and protected_assignment.direction is Direction.LONG
            )
            else Direction.LONG
            if (
                protected_assignment is not None
                and protected_assignment.direction is Direction.SHORT
            )
            else None
        )
        if (
            kind is EventKind.ACCEPTANCE_CONFIRMED
            and protected_assignment is not None
            and protected_assignment.kind
            is EventKind.PROTECTED_SWING_ASSIGNED
            and protected_assignment.is_canonical_semantic
            and (
                protected_assignment.timeframe is timeframe
                or (
                    timeframe is Timeframe.M1
                    and event.evidence.get("source_timeframe")
                    == protected_assignment.timeframe.value
                )
            )
            and protected_assignment.evidence.get("protected_swing_id")
            == protected_swing_id
            and protected_swing_id
            in protected_assignment.source_entity_ids
            and event.kind is EventKind.ACCEPTANCE_CONFIRMED
            and event.direction is opposite_assignment_direction
            and event.evidence.get("protected_swing_id")
            == protected_swing_id
            and event.evidence.get("protected_swing_event_id")
            == protected_event_id
            and protected_event_id in event.context_event_ids
            and self._protected_swing_event_ids.get(protected_swing_id)
            == protected_event_id
        ):
            # Acceptance terminalizes only the exact live assignment that it
            # names.  Comparing the captured event ID before popping prevents
            # a stale crossing from deleting a newer assignment of the same
            # swing; failed appends never reach this mutation.
            self._protected_swing_event_ids.pop(protected_swing_id)
        return event

    def _resolve_swing_crossing_if_due(
        self,
        swing,
        *,
        timeframe: Timeframe,
        asof: pd.Timestamp,
    ) -> MarketEvent | None:
        """Resolve a confirmed-swing price crossing on the next TF close."""

        if swing.broken_at is None:
            return None
        level_id = f"swing:{swing.swing_id}"
        penetration_event_id = self._penetration_event_ids.get(
            self._penetration_key(
                level_id=level_id,
                timeframe=timeframe,
                crossed_at=swing.broken_at,
            )
        )
        if penetration_event_id is None:
            return None
        generation_id = self._crossing_generation_id(
            level_id=level_id,
            timeframe=timeframe,
            crossed_at=swing.broken_at,
        )
        prior = self._terminal_crossing_events.get(generation_id)
        if prior is not None:
            return prior
        next_bar = next(
            (
                (clock, event_id)
                for clock, event_id in self._real_bar_event_ids_by_timeframe[
                    timeframe
                ]
                if swing.broken_at < clock <= asof
            ),
            None,
        )
        if next_bar is None:
            return None
        resolved_at, resolution_bar_event_id = next_bar
        resolved_close = self._bar_close_by_event_id.get(
            resolution_bar_event_id
        )
        if resolved_close is None:
            raise ValueError(
                "swing crossing resolution lacks its completed close"
            )
        crossed_above = swing.side.value == "high"
        accepted_outside = (
            resolved_close > swing.price
            if crossed_above
            else resolved_close < swing.price
        )
        crossing_direction = (
            Direction.LONG if crossed_above else Direction.SHORT
        )
        reaction_direction = (
            crossing_direction
            if accepted_outside
            else (
                Direction.SHORT
                if crossing_direction is Direction.LONG
                else Direction.LONG
            )
        )
        return self._append_crossing_resolution(
            (
                EventKind.ACCEPTANCE_CONFIRMED
                if accepted_outside
                else EventKind.SWEEP_CONFIRMED
            ),
            resolved_at,
            timeframe,
            "above" if crossed_above else "below",
            swing.price,
            clamp(swing.magnitude_atr),
            (penetration_event_id, resolution_bar_event_id),
            {
                "level_id": level_id,
                "source_kind": "confirmed_swing",
                "target_swing_id": swing.swing_id,
                "resolution_bars": 1,
                "resolved_close": float(resolved_close),
                "resolution": (
                    "later_close_held_outside_confirmed_swing_price"
                    if accepted_outside
                    else "later_close_returned_inside_confirmed_swing_price"
                ),
            },
            direction=reaction_direction,
            crossed_at=swing.broken_at,
            known_at=resolved_at,
            zone=(swing.price, swing.price),
        )

    def _append_completed_bar_event(
        self,
        candle: Candle,
        *,
        atr: float,
        data_complete: bool,
    ) -> MarketEvent:
        """Append one normalized data fact consumed by event reducers.

        This is deliberately not an SMC interpretation.  Its external data
        identity is carried in evidence so derived semantic events can later
        distinguish event lineage from raw candle/entity provenance.
        """

        data_id = hashlib.sha256(
            json.dumps(
                to_primitive(
                    {
                        "data_type": "completed_ohlcv_bar",
                        "timeframe": candle.timeframe,
                        "start": candle.start,
                        "end": candle.end,
                        "open": float(candle.open),
                        "high": float(candle.high),
                        "low": float(candle.low),
                        "close": float(candle.close),
                        "volume": float(candle.volume),
                        "symbol": candle.symbol,
                        "instrument_id": int(candle.instrument_id),
                        "complete": bool(candle.complete),
                        "observed_minutes": candle.observed_minutes,
                        "expected_minutes": candle.expected_minutes,
                        "real_minutes": candle.real_minutes,
                        "synthetic_minutes": candle.synthetic_minutes,
                    }
                ),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        try:
            detector_candle_id = candle_identity(
                candle,
                tick_size=self.config.tick_size,
            )
        except ValueError:
            # Normalization must remain able to audit off-grid vendor/test
            # bars even though registered semantic detectors reject them.
            # Such a bar can be replayed by its exact data digest, but it can
            # never masquerade as a tick-grid detector candle identity.
            detector_candle_id = data_id
        session_name, session_phase = session_name_phase(candle.end)
        bar_evidence = {
            "event_category": "normalized_data",
            "source_data_ids": (data_id,),
            "detector_candle_id": detector_candle_id,
            # The enabled owner registry is part of the normalized M1 replay
            # contract.  One completed M1 fact can therefore initialize empty
            # higher-timeframe state without inventing a completed HTF bar or
            # consulting the rich FrameObservation projection.
            "active_timeframes": (
                tuple(
                    timeframe.value
                    for timeframe in self._active_timeframes
                )
                if candle.timeframe is Timeframe.M1
                else ()
            ),
            "scale_registry_id": self._scale_registry_id,
            "open": float(candle.open),
            "high": float(candle.high),
            "low": float(candle.low),
            "close": float(candle.close),
            "volume": float(candle.volume),
            "atr": float(atr),
            "data_complete": bool(data_complete),
            "real_completed": bool(candle.real_completed),
            "clock_only": not candle.real_completed,
            "symbol": candle.symbol,
            "instrument_id": int(candle.instrument_id),
            "session_name": session_name,
            "session_phase": session_phase,
        }
        if not candle.real_completed:
            # Clock-only roots must expose the exact coverage defect they
            # transport.  Preserve historical real-root evidence/identities.
            bar_evidence.update(
                {
                    "complete": bool(candle.complete),
                    "start": candle.start,
                    "observed_minutes": int(candle.observed_minutes),
                    "expected_minutes": int(candle.expected_minutes),
                    "real_minutes": int(candle.real_minutes),
                    "synthetic_minutes": int(candle.synthetic_minutes),
                }
            )
        event = _event(
            EventKind.BAR_COMPLETED,
            candle.end,
            candle.timeframe,
            None,
            float(candle.close),
            0.0,
            (),
            bar_evidence,
            event_time=candle.end,
            known_at=candle.end,
            evidence=bar_evidence,
            source_data_ids=(data_id,),
            source_entity_ids=(
                f"scale_registry:{self._scale_registry_id}",
            ),
            origin=EventOrigin.NORMALIZED_DATA,
        )
        existing = self.audit_store.get(event.event_id)
        if existing is not None:
            self._bar_event_ids_by_candle_id[detector_candle_id] = (
                existing.event_id
            )
            self._bar_close_by_candle_id[detector_candle_id] = float(
                candle.close
            )
            self._bar_range_by_event_id[existing.event_id] = (
                float(candle.low),
                float(candle.high),
            )
            self._bar_close_by_event_id[existing.event_id] = float(
                candle.close
            )
            if not any(
                event_id == existing.event_id
                for _, event_id in self._bar_event_ids_by_timeframe[
                    candle.timeframe
                ]
            ):
                self._bar_event_ids_by_timeframe[candle.timeframe].append(
                    (candle.end, existing.event_id)
                )
            if candle.real_completed and not any(
                event_id == existing.event_id
                for _, event_id in self._real_bar_event_ids_by_timeframe[
                    candle.timeframe
                ]
            ):
                self._real_bar_event_ids_by_timeframe[
                    candle.timeframe
                ].append((candle.end, existing.event_id))
            return existing
        self.memory.append(event, include_in_recent=False)
        self._bar_event_ids_by_candle_id[detector_candle_id] = event.event_id
        self._bar_close_by_candle_id[detector_candle_id] = float(candle.close)
        self._bar_close_by_event_id[event.event_id] = float(candle.close)
        self._bar_range_by_event_id[event.event_id] = (
            float(candle.low),
            float(candle.high),
        )
        self._bar_event_ids_by_timeframe[candle.timeframe].append(
            (candle.end, event.event_id)
        )
        if candle.real_completed:
            self._real_bar_event_ids_by_timeframe[
                candle.timeframe
            ].append((candle.end, event.event_id))
        return event

    def _append_available_bar_events(
        self,
        update: ReaderUpdate,
        histories: Mapping[Timeframe, Sequence[Candle]],
        frames: Mapping[Timeframe, FrameObservation],
    ) -> None:
        """Publish normalized bars before semantics that consume them.

        On the first attached snapshot the observer may receive a retained
        causal prefix rather than one bar.  We publish that prefix once and
        calculate each bar's ATR only from bars available through that bar;
        using the final frame ATR for old bars would itself leak the future
        into the normalized event stream.  Later updates publish only the
        newly completed bars and may reuse the frame's current causal ATR.
        """

        candidates: list[tuple[Candle, float, bool]] = []
        coverage_start = self.memory.clock_coverage_start
        for timeframe in self._active_timeframes:
            frame = frames[timeframe]
            known_bars = self._bar_event_ids_by_timeframe[timeframe]
            if known_bars:
                for candle in update.newly_completed.get(timeframe, ()):
                    if (
                        candle.complete
                        and (
                            coverage_start is None
                            or candle.end >= coverage_start
                        )
                    ):
                        candidates.append(
                            (
                                candle,
                                float(frame.metrics.get("atr", 0.0)),
                                bool(frame.ready),
                            )
                        )
                continue

            eligible_history = tuple(
                candle
                for candle in histories[timeframe]
                if (
                    candle.complete
                    and (
                        coverage_start is None
                        or candle.end >= coverage_start
                    )
                )
            )
            true_ranges: deque[float] = deque(
                maxlen=max(1, self.config.atr_period)
            )
            prior_close: float | None = None
            real_bars_seen = 0
            for candle in eligible_history:
                if candle.real_completed:
                    real_bars_seen += 1
                    true_range = (
                        float(candle.high - candle.low)
                        if prior_close is None
                        else max(
                            float(candle.high - candle.low),
                            abs(float(candle.high) - prior_close),
                            abs(float(candle.low) - prior_close),
                        )
                    )
                    true_ranges.append(max(0.0, true_range))
                    prior_close = float(candle.close)
                positive = tuple(value for value in true_ranges if value > 0.0)
                causal_atr = (
                    float(np.mean(positive)) if positive else 0.0
                )
                candidates.append(
                    (
                        candle,
                        causal_atr,
                        real_bars_seen
                        >= self.config.minimum_bars[timeframe],
                    )
                )

        for candle, atr, data_complete in sorted(
            candidates,
            key=lambda value: (
                value[0].end,
                value[0].timeframe.value,
                value[0].start,
            ),
        ):
            self._append_completed_bar_event(
                candle,
                atr=atr,
                data_complete=data_complete,
            )

    def _bar_event_id_for_candle_id(self, candle_id: str) -> str:
        try:
            return self._bar_event_ids_by_candle_id[candle_id]
        except KeyError as error:
            raise ValueError(
                "semantic source candle has no BAR_COMPLETED event: "
                f"{candle_id}"
            ) from error

    def _bar_event_id_at(
        self,
        timeframe: Timeframe,
        known_at: pd.Timestamp,
    ) -> str:
        clock = pd.Timestamp(known_at)
        for event_clock, event_id in reversed(
            self._real_bar_event_ids_by_timeframe[timeframe]
        ):
            if event_clock == clock:
                return event_id
            if event_clock < clock:
                break
        raise ValueError(
            "semantic occurrence has no exact completed-bar source: "
            f"{timeframe.value}@{clock.isoformat()}"
        )

    def _fvg_approach_speed_atr(
        self,
        state: FairValueGapState,
        *,
        observed_at: pd.Timestamp,
        entry_bar_event_id: str | None,
    ) -> float | None:
        """ATRs per bar closed against the near edge on the way in.

        The distance still separating the last completed M5 close from the
        edge price then entered is, by construction, distance covered in one
        bar.  Normalizing by that bar's own causal ATR makes it comparable
        across regimes.  Every input is strictly prior or same-bar; a missing
        input yields ``None`` rather than an invented number.
        """

        if entry_bar_event_id is None:
            return None
        entry_bar = self.memory.audit_event_including_pending(
            entry_bar_event_id
        )
        if entry_bar is None:
            return None
        atr = entry_bar.details.get("atr")
        if not isinstance(atr, (int, float)) or not float(atr) > 0.0:
            return None
        previous_close: float | None = None
        for event_clock, event_id in reversed(
            self._real_bar_event_ids_by_timeframe[Timeframe.M5]
        ):
            if event_clock < observed_at:
                previous_close = self._bar_close_by_event_id.get(event_id)
                break
        if previous_close is None:
            return None
        near_edge = (
            state.upper_bound
            if state.direction is Direction.LONG
            else state.lower_bound
        )
        distance = (
            previous_close - near_edge
            if state.direction is Direction.LONG
            else near_edge - previous_close
        )
        return max(0.0, float(distance)) / float(atr)

    def _fvg_first_retest_evidence(
        self,
        state: FairValueGapState,
        *,
        observed_at: pd.Timestamp,
        entry_bar_event_id: str | None,
    ) -> dict[str, object]:
        """Freeze what was true about this gap the first time price re-entered.

        The zone protocol forbids MBO as a Group-3 input, so no book evidence
        is claimed here.
        """

        session_name, session_phase = session_name_phase(observed_at)
        return {
            "fvg_id": state.fvg_id,
            "fill_depth_at_entry": float(state.max_fill_fraction),
            "age_bars": int(state.age_bars),
            "age_seconds": int(
                (observed_at - state.confirmed_at).total_seconds()
            ),
            "entry_lifecycle": state.lifecycle.value,
            "entry_reason": state.transition_reason,
            "qualification": state.qualification.value,
            "session": session_name,
            "session_phase": session_phase,
            "approach_speed_atr": self._fvg_approach_speed_atr(
                state,
                observed_at=observed_at,
                entry_bar_event_id=entry_bar_event_id,
            ),
            "source_displacement_id": state.source_displacement_id,
        }

    def _clock_root_event_id_at(
        self,
        timeframe: Timeframe,
        known_at: pd.Timestamp,
    ) -> str:
        """Return one exact normalized BAR root, including clock-only M1."""

        clock = pd.Timestamp(known_at)
        matches = tuple(
            event_id
            for event_clock, event_id in self._bar_event_ids_by_timeframe[
                timeframe
            ]
            if event_clock == clock
        )
        if len(matches) != 1:
            raise ValueError(
                "semantic context requires one exact inclusive clock root: "
                f"{timeframe.value}@{clock.isoformat()}"
            )
        event = self.memory.audit_event_including_pending(matches[0])
        if (
            event is None
            or event.origin is not EventOrigin.NORMALIZED_DATA
            or event.kind is not EventKind.BAR_COMPLETED
            or event.timeframe is not timeframe
            or event.event_time != clock
            or event.known_at != clock
        ):
            raise ValueError("inclusive clock root is not an exact normalized BAR")
        return event.event_id

    def _synthetic_m1_context_event_ids_for_m5_terminal(
        self,
        known_at: pd.Timestamp,
    ) -> tuple[str, ...]:
        """Return every clock-only M1 constituent of an incomplete M5 bar."""

        clock = pd.Timestamp(known_at)
        interval_start = clock - pd.Timedelta(minutes=5)
        current_root_id = self._clock_root_event_id_at(Timeframe.M1, clock)
        current_root = self.memory.audit_event_including_pending(
            current_root_id
        )
        if current_root is None:
            raise ValueError(
                "synthetic displacement terminal lacks its current M1 root"
            )
        market_identity = (
            current_root.evidence.get("symbol"),
            current_root.evidence.get("instrument_id"),
        )
        interval_roots: list[tuple[pd.Timestamp, str]] = []
        roots: list[tuple[pd.Timestamp, str]] = []
        for event_clock, event_id in self._bar_event_ids_by_timeframe[
            Timeframe.M1
        ]:
            if not interval_start < event_clock <= clock:
                continue
            event = self.memory.audit_event_including_pending(event_id)
            if (
                event is None
                or event.origin is not EventOrigin.NORMALIZED_DATA
                or event.kind is not EventKind.BAR_COMPLETED
                or event.timeframe is not Timeframe.M1
                or event.event_time != event_clock
                or event.known_at != event_clock
            ):
                raise ValueError(
                    "synthetic displacement M5 interval has an invalid M1 root"
                )
            real_completed = event.evidence.get("real_completed")
            clock_only = event.evidence.get("clock_only")
            if (
                not isinstance(real_completed, bool)
                or not isinstance(clock_only, bool)
                or clock_only is not (not real_completed)
                or (
                    event.evidence.get("symbol"),
                    event.evidence.get("instrument_id"),
                )
                != market_identity
            ):
                raise ValueError(
                    "synthetic displacement M5 interval has inconsistent M1 "
                    "root evidence"
                )
            interval_roots.append((event_clock, event.event_id))
            if clock_only:
                roots.append((event_clock, event.event_id))
        ordered_interval = tuple(sorted(interval_roots))
        expected_clocks = tuple(
            interval_start + pd.Timedelta(minutes=offset)
            for offset in range(1, 6)
        )
        if (
            len(ordered_interval) != 5
            or tuple(item[0] for item in ordered_interval) != expected_clocks
            or len({item[1] for item in ordered_interval}) != 5
        ):
            raise ValueError(
                "synthetic displacement M5 interval lacks five contiguous "
                "unique M1 roots"
            )
        ordered = tuple(sorted(roots))
        event_ids = tuple(item[1] for item in ordered)
        if (
            not ordered
            or len(ordered) != len(set(ordered))
            or len(event_ids) != len(set(event_ids))
        ):
            raise ValueError(
                "synthetic displacement terminal lacks exact clock-only M1 "
                "constituent roots"
            )
        clocks = tuple(item[0] for item in ordered)
        if len(clocks) != len(set(clocks)):
            raise ValueError(
                "synthetic displacement M5 interval repeats an M1 root clock"
            )
        return tuple(item[1] for item in ordered)

    def _swing_window_event_ids(self, swing) -> tuple[str, ...]:
        if self._structure_config is None:
            raise ValueError("confirmed swing lacks a structure protocol")
        span = self._structure_config.span_for(swing.timeframe)
        bar_events = self._real_bar_event_ids_by_timeframe[swing.timeframe]
        pivot_index = next(
            (
                index
                for index, (clock, _) in enumerate(bar_events)
                if clock == swing.pivot_end
            ),
            None,
        )
        if (
            pivot_index is None
            or pivot_index < span
            or pivot_index + span >= len(bar_events)
        ):
            raise ValueError(
                "confirmed swing lacks its full registered bar window"
            )
        window = tuple(
            event_id
            for _, event_id in bar_events[
                pivot_index - span : pivot_index + span + 1
            ]
        )
        if bar_events[pivot_index + span][0] != swing.confirmed_at:
            raise ValueError(
                "confirmed swing bar window and known_at disagree"
            )
        return window

    def _swing_window_geometry(
        self,
        source_bar_events: tuple[str, ...],
        timeframe,
    ) -> dict[str, object]:
        """Freeze the definitional window a confirmed Swing was decided on.

        Geometric nesting asks whether one Swing's window sits inside
        another's, in time and in price.  That question can only be answered
        from the exact bars the Swing was confirmed from, so the window travels
        with the confirmation rather than being re-derived later from whatever
        bars happen to still be in memory.
        """

        native = Timeframe(timeframe)
        clocks = {
            event_id: clock
            for clock, event_id in self._real_bar_event_ids_by_timeframe[native]
        }
        ranges = tuple(
            self._bar_range_by_event_id.get(event_id)
            for event_id in source_bar_events
        )
        if (
            not source_bar_events
            or any(item is None for item in ranges)
            or any(event_id not in clocks for event_id in source_bar_events)
        ):
            raise ValueError("confirmed swing window is not readable")
        minutes = _TIMEFRAME_MINUTES[native]
        window_clocks = tuple(clocks[event_id] for event_id in source_bar_events)
        return {
            "window_start": (
                min(window_clocks) - pd.Timedelta(minutes=minutes)
            ).isoformat(),
            "window_end": max(window_clocks).isoformat(),
            "window_low": min(low for low, _ in ranges),
            "window_high": max(high for _, high in ranges),
        }

    def _append_reference_zone_admission_prefixes(
        self,
        zone: SupportResistanceState,
        *,
        final_observed_at: pd.Timestamp,
    ) -> None:
        """Preserve same-admission reference-zone lifecycle order.

        A completed previous-session/day/week source can be admitted and then
        tested or broken by the first tradable bar before a public snapshot is
        built.  This is not a general cold-start backfill: it applies only to a
        reference S/R entity's first projection when its terminal/current
        transition occurred at this exact projection clock.  Prefix events use
        only frozen geometry and source identity, stay out of the recent deque,
        and therefore cannot carry later reaction or extreme information into
        an earlier clock.
        """

        if (
            zone.source_kind == "structural_swing"
            or zone.lifecycle is SupportResistanceLifecycle.ACTIVE
        ):
            return
        final_clocks = {
            SupportResistanceLifecycle.TESTED: zone.tested_at,
            SupportResistanceLifecycle.BROKEN: zone.broken_at,
            SupportResistanceLifecycle.REACCEPTED: zone.reaccepted_at,
            SupportResistanceLifecycle.RETIRED: zone.retired_at,
        }
        final_clock = final_clocks.get(zone.lifecycle)
        if final_clock is None or final_clock != final_observed_at:
            return
        entity_key = f"zone:{zone.zone_id}"
        if self.memory.timeline(entity_key):
            return
        frozen_details = {
            "causal_prefix_recovered": True,
            "lower_bound": zone.lower_bound,
            "upper_bound": zone.upper_bound,
            "source_kind": zone.source_kind,
            "source_ids": zone.source_ids,
            "range_id": zone.range_id,
            "source_zone_id": zone.source_zone_id,
            "zone_role": zone.zone_role,
            "structural_rank": zone.structural_rank,
            "is_protected_swing": zone.is_protected_swing,
        }

        def append_prefix(
            lifecycle: SupportResistanceLifecycle,
            observed_at: pd.Timestamp,
            transition_reason: str | None = None,
        ) -> None:
            self.memory.append(
                _event(
                    EventKind.SUPPORT_RESISTANCE_STATE,
                    observed_at,
                    zone.timeframe,
                    "below" if zone.side == "support" else "above",
                    zone.anchor_price,
                    0.0,
                    zone.causal_source_ids,
                    frozen_details,
                    entity_id=zone.zone_id,
                    lifecycle=lifecycle.value,
                    formed_at=zone.formed_at,
                    confirmed_at=zone.confirmed_at,
                    transition_reason=transition_reason,
                ),
                include_in_recent=False,
                audit=False,
            )

        append_prefix(
            SupportResistanceLifecycle.ACTIVE,
            zone.confirmed_at,
        )
        prior_clock = zone.confirmed_at
        broken_prefix_clock = (
            zone.broken_at
            if (
                zone.lifecycle
                in {
                    SupportResistanceLifecycle.REACCEPTED,
                    SupportResistanceLifecycle.RETIRED,
                }
                and zone.broken_at is not None
                and zone.broken_at < final_observed_at
            )
            else None
        )
        tested_upper_bound = broken_prefix_clock or final_observed_at
        if (
            zone.tested_at is not None
            and prior_clock < zone.tested_at < tested_upper_bound
        ):
            append_prefix(
                SupportResistanceLifecycle.TESTED,
                zone.tested_at,
            )
            prior_clock = zone.tested_at
        if broken_prefix_clock is not None and prior_clock < broken_prefix_clock:
            append_prefix(
                SupportResistanceLifecycle.BROKEN,
                broken_prefix_clock,
                "close_beyond_frozen_zone",
            )

    def _record_frame_events(
        self,
        frame: FrameObservation,
        newly_completed: bool,
        *,
        event_clock: pd.Timestamp | None = None,
        prior: MarketObservation | None,
    ) -> None:
        if not newly_completed:
            return
        event_clock = frame.cutoff if event_clock is None else event_clock
        if event_clock > frame.cutoff:
            raise ValueError(
                "frame event clock cannot exceed the observation cutoff"
            )
        atomic_bar_roots_available = bool(
            self._real_bar_event_ids_by_timeframe[frame.timeframe]
        )
        for swing in frame.swings:
            key = (swing.swing_id, swing.lifecycle)
            if self._remember_bounded(
                key,
                known=self._known_level_ids,
                order=self._known_level_order,
            ):
                if self.memory.has_entity_lifecycle(
                    f"swing:{swing.swing_id}",
                    swing.lifecycle.value,
                ):
                    continue
                observed_at = (
                    swing.broken_at
                    if swing.lifecycle is SwingLifecycle.BROKEN
                    else swing.observed_at
                )
                direction = (
                    Direction.LONG
                    if swing.relation
                    in {SwingRelation.HH, SwingRelation.HL}
                    else Direction.SHORT
                    if swing.relation
                    in {SwingRelation.LH, SwingRelation.LL}
                    else None
                )
                swing_state_event = _event(
                        EventKind.SWING_STATE,
                        observed_at,
                        frame.timeframe,
                        "above" if swing.side.value == "high" else "below",
                        swing.price,
                        clamp(swing.magnitude_atr),
                        tuple(
                            value
                            for value in (
                                swing.swing_id,
                                swing.prior_same_side_id,
                            )
                            if value is not None
                        ),
                        {
                            "relation": swing.relation.value,
                            "pivot_start": swing.pivot_start.isoformat(),
                            "pivot_end": swing.pivot_end.isoformat(),
                            "confirmed_at": (
                                None
                                if swing.confirmed_at is None
                                else swing.confirmed_at.isoformat()
                            ),
                            "age_bars": swing.age_bars,
                            "delta_ticks": swing.delta_ticks,
                            "delta_points": swing.delta_points,
                            "magnitude_atr": swing.magnitude_atr,
                            "prominence_atr": swing.prominence_atr,
                            "confirmation_delay_bars": (
                                swing.confirmation_delay_bars
                            ),
                            "nesting_depth": swing.nesting_depth,
                            "semantic_rank": swing.semantic_rank.value,
                        },
                        entity_id=swing.swing_id,
                        lifecycle=swing.lifecycle.value,
                        formed_at=swing.pivot_end,
                        confirmed_at=swing.confirmed_at,
                        ended_at=(
                            observed_at
                            if swing.lifecycle
                            in {
                                SwingLifecycle.BROKEN,
                                SwingLifecycle.FORMATION_FAILED,
                            }
                            else None
                        ),
                        direction=direction,
                        transition_reason=swing.failure_reason,
                        event_time=(
                            swing.pivot_start
                            if swing.lifecycle
                            in {
                                SwingLifecycle.FORMING,
                                SwingLifecycle.CONFIRMED,
                            }
                            else observed_at
                        ),
                    )
                self.memory.append(swing_state_event)
                if (
                    swing.lifecycle
                    in {SwingLifecycle.CONFIRMED, SwingLifecycle.BROKEN}
                    and swing.confirmed_at is not None
                    and swing.swing_id
                    not in self._confirmed_swing_event_ids
                    and atomic_bar_roots_available
                ):
                    source_bar_events = self._swing_window_event_ids(
                        swing
                    )
                    canonical = self._append_semantic_atomic(
                        EventKind.SWING_CONFIRMED,
                        swing.confirmed_at,
                        frame.timeframe,
                        "above" if swing.side.value == "high" else "below",
                        swing.price,
                        clamp(swing.magnitude_atr),
                        source_bar_events,
                        {
                            "source_entity_id": swing.swing_id,
                            "side": swing.side.value,
                            "relation": swing.relation.value,
                            "pivot_start": swing.pivot_start.isoformat(),
                            "pivot_end": swing.pivot_end.isoformat(),
                            "prominence_atr": swing.prominence_atr,
                            "legacy_same_side_magnitude_atr": (
                                swing.magnitude_atr
                            ),
                            "confirmation_delay_bars": (
                                swing.confirmation_delay_bars
                            ),
                            "nesting_depth": swing.nesting_depth,
                            "semantic_rank": swing.semantic_rank.value,
                            "delta_ticks": swing.delta_ticks,
                            "confirmation_delay_minutes": int(
                                (
                                    swing.confirmed_at - swing.pivot_start
                                ).total_seconds()
                                // 60
                            ),
                            **self._swing_window_geometry(
                                source_bar_events,
                                frame.timeframe,
                            ),
                        },
                        direction=direction,
                        event_time=swing.pivot_start,
                        source_entity_ids=(swing.swing_id,),
                        context_event_ids=(swing_state_event.event_id,),
                    )
                    self._confirmed_swing_event_ids[swing.swing_id] = (
                        canonical.event_id
                    )
                    swing_level_id = f"swing:{swing.swing_id}"
                    candidate = self._append_semantic_atomic(
                        EventKind.LIQUIDITY_LEVEL_CREATED,
                        swing.confirmed_at,
                        frame.timeframe,
                        "above" if swing.side.value == "high" else "below",
                        swing.price,
                        clamp(swing.magnitude_atr),
                        (canonical.event_id,),
                        {
                            "level_id": swing_level_id,
                            "candidate_only": True,
                            "source_kind": "confirmed_swing",
                            "source_swing_id": swing.swing_id,
                            "semantic_rank": swing.semantic_rank.value,
                            "source_formed_at": swing.pivot_end.isoformat(),
                            "source_confirmed_at": (
                                swing.confirmed_at.isoformat()
                            ),
                        },
                        event_time=swing.pivot_start,
                        zone=(swing.price, swing.price),
                        source_entity_ids=(swing_level_id, swing.swing_id),
                    )
                    self._candidate_level_event_ids[swing_level_id] = (
                        candidate.event_id
                    )
                if (
                    swing.lifecycle is SwingLifecycle.BROKEN
                    and swing.broken_at is not None
                    and atomic_bar_roots_available
                ):
                    swing_level_id = f"swing:{swing.swing_id}"
                    candidate_event_id = self._candidate_level_event_ids.get(
                        swing_level_id
                    )
                    if candidate_event_id is None:
                        raise ValueError(
                            "broken swing lacks its candidate liquidity level"
                        )
                    penetration_key = self._penetration_key(
                        level_id=swing_level_id,
                        timeframe=frame.timeframe,
                        crossed_at=swing.broken_at,
                    )
                    if penetration_key not in self._penetration_event_ids:
                        break_bar_event_id = self._bar_event_id_at(
                            frame.timeframe,
                            swing.broken_at,
                        )
                        touch_event = self._append_semantic_atomic(
                            EventKind.LEVEL_TOUCHED,
                            swing.broken_at,
                            frame.timeframe,
                            (
                                "above"
                                if swing.side.value == "high"
                                else "below"
                            ),
                            swing.price,
                            clamp(swing.magnitude_atr),
                            (candidate_event_id, break_bar_event_id),
                            {
                                "level_id": swing_level_id,
                                "source_kind": "confirmed_swing",
                                "target_swing_id": swing.swing_id,
                                "touch_reason": "raw_swing_price_crossing",
                            },
                            event_time=swing.broken_at,
                            zone=(swing.price, swing.price),
                            source_entity_ids=(
                                swing_level_id,
                                swing.swing_id,
                            ),
                        )
                        crossing_generation_id = (
                            self._crossing_generation_id(
                                level_id=swing_level_id,
                                timeframe=frame.timeframe,
                                crossed_at=swing.broken_at,
                            )
                        )
                        break_bar_event = (
                            self.memory.audit_event_including_pending(
                                break_bar_event_id
                            )
                        )
                        if break_bar_event is None:
                            raise ValueError(
                                "swing penetration lacks its exact BAR"
                            )
                        penetration_price = float(
                            break_bar_event.evidence[
                                "high"
                                if swing.side.value == "high"
                                else "low"
                            ]
                        )
                        penetrated = self._append_semantic_atomic(
                            EventKind.LEVEL_PENETRATED,
                            swing.broken_at,
                            frame.timeframe,
                            (
                                "above"
                                if swing.side.value == "high"
                                else "below"
                            ),
                            penetration_price,
                            clamp(swing.magnitude_atr),
                            (
                                candidate_event_id,
                                touch_event.event_id,
                                break_bar_event_id,
                            ),
                            {
                                "level_id": swing_level_id,
                                "source_kind": "confirmed_swing",
                                "target_swing_id": swing.swing_id,
                                "penetration_standard": (
                                    "strict_close_beyond_confirmed_swing_"
                                    "price"
                                ),
                                "crossing_generation_id": (
                                    crossing_generation_id
                                ),
                                "crossed_at": (
                                    swing.broken_at.isoformat()
                                ),
                            },
                            direction=(
                                Direction.LONG
                                if swing.side.value == "high"
                                else Direction.SHORT
                            ),
                            event_time=swing.broken_at,
                            zone=(swing.price, swing.price),
                            source_entity_ids=(
                                swing_level_id,
                                swing.swing_id,
                            ),
                        )
                        self._penetration_event_ids[penetration_key] = (
                            penetrated.event_id
                        )
            if (
                swing.lifecycle is SwingLifecycle.BROKEN
                and swing.broken_at is not None
                and atomic_bar_roots_available
            ):
                # Lifecycle projection is deduplicated above, but a crossing
                # remains pending until the first *later* native-timeframe
                # close.  Revisit it on subsequent frame updates.
                self._resolve_swing_crossing_if_due(
                    swing,
                    timeframe=frame.timeframe,
                    asof=event_clock,
                )
        for leg in frame.structural_legs:
            if not self._remember_bounded(
                leg.leg_id,
                known=self._known_structural_leg_ids,
                order=self._known_structural_leg_order,
            ):
                continue
            source_event_ids = tuple(
                event_id
                for swing_id in leg.source_swing_ids
                if (
                    event_id := self._confirmed_swing_event_ids.get(
                        swing_id
                    )
                )
            )
            if len(source_event_ids) != 2:
                if not atomic_bar_roots_available:
                    continue
                raise ValueError(
                    "structural leg requires exactly two confirmed-swing "
                    "source events"
                )
            foundation_evidence: dict[str, object] = {}
            foundation_context_event_ids: tuple[str, ...] = ()
            foundation_source_data_ids: tuple[str, ...] = ()
            if leg.foundation_version is not None:
                foundation_source_data_ids = (
                    *leg.atr_source_candle_ids,
                    *leg.path_candle_ids,
                )
                foundation_context_event_ids = tuple(
                    self._bar_event_id_for_candle_id(candle_id)
                    for candle_id in foundation_source_data_ids
                )
                bound_bars = tuple(
                    self.memory.audit_event_including_pending(event_id)
                    for event_id in foundation_context_event_ids
                )
                if any(event is None for event in bound_bars):
                    raise ValueError(
                        "foundation structural leg lacks canonical BAR ancestry"
                    )
                first_bar = bound_bars[0]
                foundation_evidence = {
                    "foundation_version": leg.foundation_version,
                    "amplitude_ticks": leg.amplitude_ticks,
                    "atr_at_leg_start": leg.atr_at_leg_start,
                    "atr_source_candle_ids": leg.atr_source_candle_ids,
                    "duration_seconds": leg.duration_seconds,
                    "close_efficiency": leg.close_efficiency,
                    "extreme_path_efficiency": (
                        leg.extreme_path_efficiency
                    ),
                    "close_mae_points": leg.close_mae_points,
                    "close_mae_atr": leg.close_mae_atr,
                    "wick_mae_points": leg.wick_mae_points,
                    "wick_mae_atr": leg.wick_mae_atr,
                    "path_candle_ids": leg.path_candle_ids,
                    "tick_size": self.config.tick_size,
                    "symbol": first_bar.evidence.get("symbol"),
                    "instrument_id": first_bar.evidence.get(
                        "instrument_id"
                    ),
                }
            canonical_leg = self._append_semantic_atomic(
                EventKind.STRUCTURAL_LEG_CREATED,
                leg.known_at,
                frame.timeframe,
                "above" if leg.direction is Direction.LONG else "below",
                leg.end_price,
                clamp(leg.efficiency),
                source_event_ids,
                {
                    "leg_id": leg.leg_id,
                    "start_swing_id": leg.start_swing_id,
                    "end_swing_id": leg.end_swing_id,
                    "start_event_time": leg.start_event_time.isoformat(),
                    "end_event_time": leg.end_event_time.isoformat(),
                    "start_price": leg.start_price,
                    "end_price": leg.end_price,
                    "start_close": leg.start_close,
                    "end_close": leg.end_close,
                    "amplitude_points": leg.amplitude_points,
                    "amplitude_atr": leg.amplitude_atr,
                    "duration_bars": leg.duration_bars,
                    "duration_minutes": leg.duration_minutes,
                    "efficiency": leg.efficiency,
                    "max_retracement_points": (
                        leg.max_retracement_points
                    ),
                    "max_retracement_atr": leg.max_retracement_atr,
                    "path_class": leg.path_class.value,
                    **foundation_evidence,
                },
                direction=leg.direction,
                event_time=leg.end_event_time,
                source_entity_ids=(leg.leg_id, *leg.source_swing_ids),
                source_data_ids=foundation_source_data_ids,
                context_event_ids=foundation_context_event_ids,
            )
            self._structural_leg_event_ids[leg.leg_id] = (
                canonical_leg.event_id
            )
        for state in frame.structures:
            if (
                state.structure_id is None
                or state.lifecycle is StructureLifecycle.INACTIVE
            ):
                continue
            key = (state.structure_id, state.lifecycle)
            if not self._remember_bounded(
                key,
                known=self._known_sequence_events,
                order=self._known_sequence_event_order,
            ):
                continue
            if self.memory.has_entity_lifecycle(
                f"structure:{state.structure_id}",
                state.lifecycle.value,
            ):
                continue
            observed_at = (
                state.broken_at
                if state.lifecycle is StructureLifecycle.BROKEN
                else state.formation_failed_at
                if state.lifecycle
                is StructureLifecycle.FORMATION_FAILED
                else state.confirmed_at
                if state.lifecycle is StructureLifecycle.CONFIRMED
                else state.formed_at
            )
            structure_state_event = _event(
                    EventKind.STRUCTURE_STATE,
                    observed_at,
                    frame.timeframe,
                    (
                        "above"
                        if state.direction is Direction.LONG
                        else "below"
                    ),
                    state.protected_price,
                    clamp(state.cumulative_magnitude_atr),
                    tuple(
                        value
                        for value in (
                            state.latest_high_id,
                            state.latest_low_id,
                            state.protected_swing_id,
                        )
                        if value is not None
                    ),
                    {
                        "high_run": state.high_run,
                        "low_run": state.low_run,
                        "sequence_count": state.sequence_count,
                        "age_bars": state.age_bars,
                    },
                    entity_id=state.structure_id,
                    lifecycle=state.lifecycle.value,
                    formed_at=state.formed_at,
                    confirmed_at=state.confirmed_at,
                    ended_at=(
                        state.broken_at
                        if state.lifecycle is StructureLifecycle.BROKEN
                        else state.formation_failed_at
                        if state.lifecycle
                        is StructureLifecycle.FORMATION_FAILED
                        else None
                    ),
                    direction=state.direction,
                    transition_reason=state.failure_reason,
                )
            self.memory.append(structure_state_event)
            if (
                state.lifecycle is StructureLifecycle.CONFIRMED
                and state.confirmed_at is not None
                and atomic_bar_roots_available
            ):
                # ``latest_*`` belongs to the current snapshot and may point
                # to a swing confirmed after this structure generation.  A
                # retrospective first attach must reconstruct parents from
                # facts that were actually knowable at ``confirmed_at``.
                causal_swings = tuple(
                    max(
                        (
                            swing
                            for swing in frame.swings
                            if (
                                swing.side.value == side
                                and swing.confirmed_at is not None
                                and swing.confirmed_at <= state.confirmed_at
                                and swing.swing_id
                                in self._confirmed_swing_event_ids
                            )
                        ),
                        key=lambda swing: (
                            swing.confirmed_at,
                            swing.pivot_start,
                            swing.swing_id,
                        ),
                    )
                    for side in ("high", "low")
                )
                source_swing_events = tuple(
                    self._confirmed_swing_event_ids[swing.swing_id]
                    for swing in causal_swings
                )
                if len(source_swing_events) != 2:
                    raise ValueError(
                        "confirmed structure requires its exact high and low "
                        "swing events"
                    )
                structure_direction_event = self._append_semantic_atomic(
                    EventKind.STRUCTURE_DIRECTION_CONFIRMED,
                    state.confirmed_at,
                    frame.timeframe,
                    (
                        "above"
                        if state.direction is Direction.LONG
                        else "below"
                    ),
                    state.protected_price,
                    clamp(state.cumulative_magnitude_atr),
                    source_swing_events,
                    {
                        "structure_id": state.structure_id,
                        "direction": state.direction.value,
                        "source_high_id": causal_swings[0].swing_id,
                        "source_low_id": causal_swings[1].swing_id,
                        "sequence_count": state.sequence_count,
                        "candidate_protected_swing_id": (
                            state.protected_swing_id
                        ),
                    },
                    direction=state.direction,
                    event_time=state.formed_at,
                    source_entity_ids=(
                        state.structure_id,
                        causal_swings[0].swing_id,
                        causal_swings[1].swing_id,
                    ),
                    context_event_ids=(structure_state_event.event_id,),
                )
                self._structure_direction_event_ids[state.structure_id] = (
                    structure_direction_event.event_id
                )
                self._latest_structure_direction_event_ids[
                    frame.timeframe
                ] = structure_direction_event.event_id
        for zone in frame.support_resistance:
            revision = (
                zone.lifecycle.value,
                zone.total_touch_count,
                zone.member_swing_ids,
                zone.source_ids,
                zone.range_id,
                zone.source_zone_id,
                zone.source_kind,
                zone.structural_rank,
                zone.is_protected_swing,
                zone.zone_role,
                zone.reaction_quality,
                zone.depletion_risk,
            )
            prior_revision = self._liquidity_entity_revisions.get(
                zone.zone_id
            )
            if prior_revision == revision:
                continue
            lifecycle_revision = bool(
                prior_revision is None
                or prior_revision[0] != zone.lifecycle.value
            )
            observed_at = (
                zone.retired_at
                or zone.reaccepted_at
                or zone.broken_at
                or (
                    zone.touch_times[-1]
                    if zone.lifecycle
                    is SupportResistanceLifecycle.TESTED
                    else None
                )
                or zone.confirmed_at
            )
            if prior_revision is None and prior is not None:
                # Completed reference-period levels are admitted only when
                # the next period is observed. Their source geometry was
                # complete earlier, but the semantic level first becomes
                # available at this admission clock.
                observed_at = max(observed_at, event_clock)
            if not lifecycle_revision:
                observed_at = max(
                    observed_at,
                    zone.metadata_observed_at,
                )
            if prior_revision is None:
                self._append_reference_zone_admission_prefixes(
                    zone,
                    final_observed_at=observed_at,
                )
            self._liquidity_entity_revisions[zone.zone_id] = revision
            zone_state_event = _event(
                    EventKind.SUPPORT_RESISTANCE_STATE,
                    observed_at,
                    frame.timeframe,
                    (
                        "below"
                        if zone.side == "support"
                        else "above"
                    ),
                    zone.anchor_price,
                    zone.strength,
                    (
                        *(() if lifecycle_revision else (zone.zone_id,)),
                        *zone.causal_source_ids,
                    ),
                    {
                        "state_revision": not lifecycle_revision,
                        "lower_bound": zone.lower_bound,
                        "upper_bound": zone.upper_bound,
                        "touch_times": tuple(
                            value.isoformat()
                            for value in zone.touch_times
                        ),
                        "reaction_magnitudes_atr": (
                            zone.reaction_magnitudes_atr
                        ),
                        "touch_count": zone.touch_count,
                        "age_bars": zone.age_bars,
                        "source_kind": zone.source_kind,
                        "source_ids": zone.source_ids,
                        "range_id": zone.range_id,
                        "source_zone_id": zone.source_zone_id,
                        "zone_role": zone.zone_role,
                        "structural_rank": zone.structural_rank,
                        "is_protected_swing": zone.is_protected_swing,
                        "visibility_strength": zone.visibility_strength,
                        "reaction_quality": zone.reaction_quality,
                        "freshness": zone.freshness,
                        "depletion_risk": zone.depletion_risk,
                    },
                    entity_id=(
                        zone.zone_id if lifecycle_revision else None
                    ),
                    lifecycle=(
                        zone.lifecycle.value if lifecycle_revision else None
                    ),
                    formed_at=zone.formed_at,
                    confirmed_at=zone.confirmed_at,
                    ended_at=(
                        observed_at
                        if lifecycle_revision and zone.lifecycle
                        in {
                            SupportResistanceLifecycle.REACCEPTED,
                            SupportResistanceLifecycle.RETIRED,
                        }
                        else None
                    ),
                    transition_reason=(
                        zone.transition_reason
                        if lifecycle_revision
                        else "support_resistance_evidence_revised"
                    ),
                )
            self.memory.append(zone_state_event)
        for pool in frame.liquidity_pools:
            if pool.lifecycle is not LiquidityPoolLifecycle.FORMED:
                continue
            entity_id = f"pool:{pool.pool_id}"
            revision = (
                pool.lifecycle.value,
                pool.touch_count,
                pool.member_swing_ids,
            )
            prior_revision = self._liquidity_entity_revisions.get(
                entity_id
            )
            if prior_revision == revision:
                continue
            self._liquidity_entity_revisions[entity_id] = revision
            lifecycle_revision = bool(
                prior_revision is None
                or prior_revision[0] != pool.lifecycle.value
            )
            pool_observed_at = (
                max(pool.confirmed_at, event_clock)
                if prior_revision is None and prior is not None
                else (
                    pool.confirmed_at
                    if lifecycle_revision
                    else pool.touch_times[-1]
                )
            )
            pool_state_event = _event(
                    EventKind.LIQUIDITY_POOL_STATE,
                    pool_observed_at,
                    frame.timeframe,
                    pool.side,
                    pool.midpoint,
                    pool.strength,
                    (
                        *(() if lifecycle_revision else (entity_id,)),
                        *pool.member_swing_ids,
                    ),
                    {
                        "state_revision": not lifecycle_revision,
                        "lower_bound": pool.lower_bound,
                        "upper_bound": pool.upper_bound,
                        "touch_times": tuple(
                            value.isoformat()
                            for value in pool.touch_times
                        ),
                        "touch_count": pool.touch_count,
                        "age_bars": pool.age_bars,
                    },
                    entity_id=entity_id if lifecycle_revision else None,
                    lifecycle=(
                        pool.lifecycle.value if lifecycle_revision else None
                    ),
                    formed_at=pool.formed_at,
                    confirmed_at=pool.confirmed_at,
                    transition_reason=(
                        None
                        if lifecycle_revision
                        else "liquidity_pool_membership_revised"
                    ),
                )
            self.memory.append(pool_state_event)
            if entity_id not in self._candidate_level_event_ids:
                source_swing_events = tuple(
                    event_id
                    for source_id in pool.member_swing_ids
                    if (
                        event_id
                        := self._confirmed_swing_event_ids.get(source_id)
                    )
                )
                created = self._append_semantic_atomic(
                    EventKind.LIQUIDITY_LEVEL_CREATED,
                    pool_observed_at,
                    frame.timeframe,
                    pool.side,
                    pool.midpoint,
                    pool.strength,
                    source_swing_events,
                    {
                        "level_id": entity_id,
                        "candidate_only": True,
                        "source_kind": "formed_liquidity_pool",
                        "member_swing_ids": pool.member_swing_ids,
                        "source_formed_at": pool.formed_at.isoformat(),
                        "source_confirmed_at": (
                            pool.confirmed_at.isoformat()
                        ),
                    },
                    event_time=pool.formed_at,
                    zone=(pool.lower_bound, pool.upper_bound),
                    source_entity_ids=(entity_id, *pool.member_swing_ids),
                    context_event_ids=(pool_state_event.event_id,),
                )
                self._candidate_level_event_ids[entity_id] = (
                    created.event_id
                )
            candidate_event_id = self._candidate_level_event_ids[
                entity_id
            ]
            for touch_ordinal, touch_at in enumerate(
                pool.touch_times,
                start=1,
            ):
                if touch_at <= pool.confirmed_at:
                    continue
                touch_identity = (
                    f"{entity_id}|{pd.Timestamp(touch_at).isoformat()}"
                )
                if not self._remember_bounded(
                    touch_identity,
                    known=self._known_level_touch_ids,
                    order=self._known_level_touch_order,
                ):
                    continue
                if not atomic_bar_roots_available:
                    continue
                bar_event_id = self._bar_event_id_at(
                    frame.timeframe,
                    touch_at,
                )
                touch_event = self._append_semantic_atomic(
                    EventKind.LEVEL_TOUCHED,
                    touch_at,
                    frame.timeframe,
                    pool.side,
                    pool.midpoint,
                    pool.strength,
                    (candidate_event_id, bar_event_id),
                    {
                        "level_id": entity_id,
                        "touch_ordinal": touch_ordinal,
                        "source_kind": "formed_liquidity_pool",
                    },
                    event_time=touch_at,
                    zone=(pool.lower_bound, pool.upper_bound),
                )
                self._level_touch_event_ids[
                    (entity_id, pd.Timestamp(touch_at))
                ] = touch_event.event_id
        for item in frame.structure_breaks:
            key = (item.bos_id, item.lifecycle)
            is_new_lifecycle = self._remember_bounded(
                key,
                known=self._known_structure_events,
                order=self._known_structure_event_order,
            )
            if (
                is_new_lifecycle
                # A pending BOS carries no fact the structure-break kinds do
                # not already carry, so the timeline starts at its terminal.
                and item.lifecycle is not BOSLifecycle.PENDING
                and not self.memory.has_entity_lifecycle(
                    f"bos:{item.bos_id}", item.lifecycle.value
                )
            ):
                bos_state_event = _event(
                    (
                        EventKind.STRUCTURE_BREAK
                        if item.lifecycle is BOSLifecycle.CONFIRMED
                        else EventKind.STRUCTURE_BREAK_FAILED
                    ),
                    item.resolved_at or item.pending_at,
                    frame.timeframe,
                    "above" if item.direction is Direction.LONG else "below",
                    item.target_price,
                    item.strength,
                    tuple(
                        value
                        for value in (
                            item.target_swing_id,
                            item.source_structure_id,
                            item.source_displacement_id,
                            item.break_bar_id,
                        )
                        if value is not None
                    ),
                    {
                        "bos_id": item.bos_id,
                        "scope": item.scope.value,
                        "source_structure_id": item.source_structure_id,
                        "pending_at": item.pending_at.isoformat(),
                        "resolved_at": (
                            None
                            if item.resolved_at is None
                            else item.resolved_at.isoformat()
                        ),
                        "failure_reason": item.failure_reason,
                        "strength": item.strength,
                        "break_bar_id": item.break_bar_id,
                        "break_distance_atr": item.break_distance_atr,
                        "source_displacement_id": (
                            item.source_displacement_id
                        ),
                        "mss_qualified": item.mss_qualified,
                        "post_break_state": (
                            "pending"
                            if item.lifecycle is BOSLifecycle.CONFIRMED
                            else None
                        ),
                        "accepted_at": None,
                        "rejected_at": None,
                    },
                    entity_id=item.bos_id,
                    lifecycle=item.lifecycle.value,
                    formed_at=item.pending_at,
                    confirmed_at=(
                        item.resolved_at
                        if item.lifecycle is BOSLifecycle.CONFIRMED
                        else None
                    ),
                    ended_at=(
                        item.resolved_at
                        if item.lifecycle is BOSLifecycle.FAILED
                        else None
                    ),
                    direction=item.direction,
                    transition_reason=(
                        BOS_CONFIRMATION_REASON
                        if item.lifecycle is BOSLifecycle.CONFIRMED
                        else item.failure_reason
                    ),
                )
                self.memory.append(bos_state_event)
                if (
                    item.lifecycle is BOSLifecycle.CONFIRMED
                    and item.resolved_at is not None
                    and atomic_bar_roots_available
                ):
                    target_event_id = (
                        self._confirmed_swing_event_ids.get(
                            item.target_swing_id
                        )
                    )
                    if target_event_id is None or item.break_bar_id is None:
                        raise ValueError(
                            "raw boundary break lacks its target swing or "
                            "break-bar identity"
                        )
                    break_bar_event_id = self._bar_event_id_for_candle_id(
                        item.break_bar_id
                    )
                    break_close = self._bar_close_by_candle_id[
                        item.break_bar_id
                    ]
                    raw_break = self._append_semantic_atomic(
                        EventKind.RAW_BOUNDARY_BREAK,
                        item.resolved_at,
                        frame.timeframe,
                        (
                            "above"
                            if item.direction is Direction.LONG
                            else "below"
                        ),
                        item.target_price,
                        item.strength,
                        (target_event_id, break_bar_event_id),
                        {
                            "bos_id": item.bos_id,
                            "target_swing_id": item.target_swing_id,
                            "scope": item.scope.value,
                            "break_bar_id": item.break_bar_id,
                            "break_distance_atr": item.break_distance_atr,
                            "break_close": break_close,
                            "break_buffer_ticks": 0,
                            "comparison": "strict_close_beyond",
                            "break_standard": (
                                "close_beyond_confirmed_boundary"
                            ),
                            "source_displacement_id": (
                                item.source_displacement_id
                            ),
                        },
                        direction=item.direction,
                        event_time=item.resolved_at,
                        source_data_ids=(item.break_bar_id,),
                        source_entity_ids=(
                            item.bos_id,
                            item.target_swing_id,
                            *((
                                item.source_structure_id,
                            ) if item.source_structure_id else ()),
                            *((
                                item.source_displacement_id,
                            ) if item.source_displacement_id else ()),
                        ),
                        context_event_ids=(
                            bos_state_event.event_id,
                            *((
                                self._displacement_event_ids[
                                    item.source_displacement_id
                                ],
                            ) if (
                                item.source_displacement_id
                                in self._displacement_event_ids
                            ) else ()),
                        ),
                    )
                    self._raw_break_event_ids[item.bos_id] = (
                        raw_break.event_id
                    )
                    live_protected_assignments = ()
                    continuation_opposes_live_protection = False
                    if item.scope is BOSScope.CONTINUATION:
                        live_protected_assignments = (
                            self._live_protected_assignments(
                                frame.timeframe,
                                asof=item.resolved_at,
                            )
                        )
                        live_protected_event = (
                            live_protected_assignments[0][2]
                            if live_protected_assignments
                            else None
                        )
                        continuation_opposes_live_protection = bool(
                            live_protected_event is not None
                            and live_protected_event.direction
                            is not item.direction
                        )
                    if (
                        item.scope is BOSScope.CONTINUATION
                        and not continuation_opposes_live_protection
                    ):
                        structure_direction_event_id = (
                            self._structure_direction_event_ids.get(
                                item.source_structure_id
                            )
                            if item.source_structure_id is not None
                            else None
                        )
                        if structure_direction_event_id is None:
                            raise ValueError(
                                "qualified BOS lacks the exact prior "
                                "structure-direction event"
                            )
                        qualified_bos = self._append_semantic_atomic(
                            EventKind.QUALIFIED_BOS,
                            item.resolved_at,
                            frame.timeframe,
                            (
                                "above"
                                if item.direction is Direction.LONG
                                else "below"
                            ),
                            item.target_price,
                            item.strength,
                            (
                                raw_break.event_id,
                                structure_direction_event_id,
                            ),
                            {
                                "bos_id": item.bos_id,
                                "scope": item.scope.value,
                                "qualification": (
                                    "aligned_with_confirmed_structure"
                                ),
                                "displacement_context_present": bool(
                                    item.source_displacement_id
                                ),
                            },
                            direction=item.direction,
                            event_time=item.resolved_at,
                            source_entity_ids=(
                                item.bos_id,
                                item.source_structure_id,
                            ),
                        )
                        origin_leg = next(
                            (
                                leg
                                for leg in reversed(frame.structural_legs)
                                if (
                                    leg.end_swing_id
                                    == item.target_swing_id
                                    and leg.direction is item.direction
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
                            if origin_swing is not None:
                                protected_source = (
                                    self._confirmed_swing_event_ids.get(
                                        origin_swing.swing_id
                                    )
                                )
                                origin_leg_event_id = (
                                    self._structural_leg_event_ids.get(
                                        origin_leg.leg_id
                                    )
                                )
                                if (
                                    protected_source is None
                                    or origin_leg_event_id is None
                                ):
                                    raise ValueError(
                                        "protected swing lacks its exact "
                                        "swing or structural-leg event"
                                    )
                                live_protected_event = (
                                    live_protected_assignments[0][2]
                                    if live_protected_assignments
                                    else None
                                )
                                protection_is_monotonic = bool(
                                    live_protected_event is None
                                    or (
                                        item.direction is Direction.LONG
                                        and origin_swing.price
                                        >= float(live_protected_event.price)
                                    )
                                    or (
                                        item.direction is Direction.SHORT
                                        and origin_swing.price
                                        <= float(live_protected_event.price)
                                    )
                                )
                                if protection_is_monotonic:
                                    protected_event = (
                                        self._append_semantic_atomic(
                                            EventKind.PROTECTED_SWING_ASSIGNED,
                                            item.resolved_at,
                                            frame.timeframe,
                                            (
                                                "below"
                                                if item.direction
                                                is Direction.LONG
                                                else "above"
                                            ),
                                            origin_swing.price,
                                            clamp(origin_leg.efficiency),
                                            (
                                                qualified_bos.event_id,
                                                origin_leg_event_id,
                                                protected_source,
                                            ),
                                            {
                                                "bos_id": item.bos_id,
                                                "structure_id": (
                                                    item.source_structure_id
                                                ),
                                                "origin_leg_id": (
                                                    origin_leg.leg_id
                                                ),
                                                "protected_swing_id": (
                                                    origin_swing.swing_id
                                                ),
                                                "break_standard": (
                                                    "later_acceptance_beyond"
                                                ),
                                            },
                                            direction=item.direction,
                                            event_time=(
                                                origin_swing.pivot_start
                                            ),
                                            source_entity_ids=(
                                                item.bos_id,
                                                item.source_structure_id,
                                                origin_leg.leg_id,
                                                origin_swing.swing_id,
                                            ),
                                        )
                                    )
                                    self._replace_live_protected_assignment(
                                        protected_event,
                                        live_protected_assignments,
                                    )
                    elif item.scope is BOSScope.OPPOSED:
                        structure_direction_event_id = (
                            self._structure_direction_event_ids.get(
                                item.source_structure_id
                            )
                            if item.source_structure_id is not None
                            else None
                        )
                        if structure_direction_event_id is None:
                            raise ValueError(
                                "MSS Core lacks the exact prior "
                                "structure-direction event"
                            )
                        mss_core = self._append_semantic_atomic(
                            EventKind.MSS_CORE_CONFIRMED,
                            item.resolved_at,
                            frame.timeframe,
                            (
                                "above"
                                if item.direction is Direction.LONG
                                else "below"
                            ),
                            item.target_price,
                            item.strength,
                            (
                                raw_break.event_id,
                                structure_direction_event_id,
                            ),
                            {
                                "bos_id": item.bos_id,
                                "scope": item.scope.value,
                                "core_definition": (
                                    "first_opposed_confirmed_boundary_break"
                                ),
                                "prior_sweep": None,
                                "displacement_context_present": bool(
                                    item.source_displacement_id
                                ),
                                "legacy_mss_qualified_context": (
                                    item.mss_qualified
                                ),
                            },
                            direction=item.direction,
                            event_time=item.resolved_at,
                            source_entity_ids=(
                                item.bos_id,
                                item.source_structure_id,
                            ),
                        )
            post_break_at = item.accepted_at or item.rejected_at
            if (
                item.lifecycle is BOSLifecycle.CONFIRMED
                and post_break_at is not None
                and item.post_break_state is not None
            ):
                self.memory.append(
                    post_break_event := _event(
                        EventKind.BOS_POST_BREAK_STATE,
                        post_break_at,
                        frame.timeframe,
                        (
                            "above"
                            if item.direction is Direction.LONG
                            else "below"
                        ),
                        item.target_price,
                        item.strength,
                        tuple(
                            value
                            for value in (
                                item.bos_id,
                                item.target_swing_id,
                                item.break_bar_id,
                                item.source_displacement_id,
                            )
                            if value is not None
                        ),
                        {
                            "bos_id": item.bos_id,
                            "scope": item.scope.value,
                            "post_break_state": (
                                item.post_break_state.value
                            ),
                            "accepted_at": (
                                None
                                if item.accepted_at is None
                                else item.accepted_at.isoformat()
                            ),
                            "rejected_at": (
                                None
                                if item.rejected_at is None
                                else item.rejected_at.isoformat()
                            ),
                        },
                        direction=item.direction,
                        transition_reason=(
                            f"post_break_{item.post_break_state.value}"
                        ),
                    )
                )
                raw_event_id = self._raw_break_event_ids.get(item.bos_id)
                if raw_event_id is not None:
                    crossing_level_id = f"swing:{item.target_swing_id}"
                    if (
                        crossing_level_id
                        not in self._candidate_level_event_ids
                    ):
                        raise ValueError(
                            "BOS post-break resolution lacks its raw-swing "
                            "candidate-level identity"
                        )
                    penetration_event_id = self._penetration_event_ids.get(
                        self._penetration_key(
                            level_id=crossing_level_id,
                            timeframe=frame.timeframe,
                            crossed_at=item.resolved_at,
                        )
                    )
                    if penetration_event_id is None:
                        raise ValueError(
                            "BOS post-break resolution lacks its canonical "
                            "penetration event"
                        )
                    accepted = item.accepted_at is not None
                    resolution_bar_event_id = self._bar_event_id_at(
                        frame.timeframe,
                        post_break_at,
                    )
                    reaction_direction = (
                        item.direction
                        if accepted
                        else (
                            Direction.SHORT
                            if item.direction is Direction.LONG
                            else Direction.LONG
                        )
                    )
                    self._append_crossing_resolution(
                        (
                            EventKind.ACCEPTANCE_CONFIRMED
                            if accepted
                            else EventKind.SWEEP_CONFIRMED
                        ),
                        post_break_at,
                        frame.timeframe,
                        (
                            "above"
                            if item.direction is Direction.LONG
                            else "below"
                        ),
                        item.target_price,
                        item.strength,
                        (
                            penetration_event_id,
                            resolution_bar_event_id,
                        ),
                        {
                            "bos_id": item.bos_id,
                            "level_id": crossing_level_id,
                            "target_swing_id": item.target_swing_id,
                            "resolution_bars": 1,
                            "resolution": (
                                "held_outside"
                                if accepted
                                else "returned_inside"
                            ),
                        },
                        direction=reaction_direction,
                        crossed_at=item.resolved_at,
                        known_at=event_clock,
                        context_event_ids=(
                            raw_event_id,
                            post_break_event.event_id,
                        ),
                    )

    def _record_base_origin_core(self, core) -> None:
        """Publish frozen impulse geometry at the moment it is locked.

        The core is emitted whether or not a break ever qualifies it, so the
        population the Eye can count is not silently restricted to the cores
        that went on to work.
        """

        if core.base_origin_core_id in self._base_origin_core_event_ids:
            return
        anchor_bar_events = tuple(
            self._bar_event_id_for_candle_id(candle_id)
            for candle_id in core.anchor_candle_ids
        )
        event = self._append_semantic_atomic(
            EventKind.BASE_ORIGIN_CORE_CREATED,
            core.observed_at,
            Timeframe.M5,
            "below" if core.direction is Direction.LONG else "above",
            core.midpoint,
            0.0,
            anchor_bar_events,
            {
                "base_origin_core_id": core.base_origin_core_id,
                "geometry": "frozen_group3_order_block_range",
                "anchor_candle_ids": core.anchor_candle_ids,
                "locating_impulse_id": core.source_displacement_id,
            },
            direction=core.direction,
            event_time=core.anchor_end,
            zone=(core.lower_bound, core.upper_bound),
            source_data_ids=core.anchor_candle_ids,
            source_entity_ids=(core.base_origin_core_id,),
        )
        self._base_origin_core_event_ids[core.base_origin_core_id] = (
            event.event_id
        )

    def _record_group3_events(
        self,
        update: ZoneUpdate,
    ) -> None:
        for core in update.base_origin_cores:
            self._record_base_origin_core(core)
        for state in update.fvg_transitions:
            midpoint_revision = bool(
                state.lifecycle is FairValueGapLifecycle.PARTIAL
                and state.transition_reason == "midpoint_touched"
            )
            observed_at = (
                state.midpoint_touched_at
                if midpoint_revision
                else state.state_started_at
            )
            if observed_at is None:
                raise ValueError("FVG transition lacks its causal clock")
            terminal = state.lifecycle in {
                FairValueGapLifecycle.MITIGATED,
                FairValueGapLifecycle.INVALIDATED,
                FairValueGapLifecycle.EXPIRED,
            }
            fvg_state_event = _event(
                    EventKind.FVG_STATE,
                    observed_at,
                    Timeframe.M5,
                    (
                        "below"
                        if state.direction is Direction.LONG
                        else "above"
                    ),
                    state.midpoint,
                    state.strength,
                    (
                        state.source_displacement_id,
                        state.source_active_transition_id,
                        *state.source_candle_ids,
                    ),
                    {
                        "state_revision": midpoint_revision,
                        "protocol_hash": state.protocol_hash,
                        "qualification": state.qualification.value,
                        "lower_bound": state.lower_bound,
                        "upper_bound": state.upper_bound,
                        "midpoint": state.midpoint,
                        "invalidation_price": (
                            state.invalidation_price
                        ),
                        "width_ticks": state.width_ticks,
                        "width_atr": state.width_atr,
                        "formation_atr": state.formation_atr,
                        "age_bars": state.age_bars,
                        "max_fill_fraction": (
                            state.max_fill_fraction
                        ),
                        "midpoint_touched_at": (
                            None
                            if state.midpoint_touched_at is None
                            else state.midpoint_touched_at.isoformat()
                        ),
                        "source_displacement_protocol_hash": (
                            state.source_displacement_protocol_hash
                        ),
                        "source_displacement_started_at": (
                            None
                            if state.source_displacement_started_at is None
                            else state.source_displacement_started_at.isoformat()
                        ),
                        "source_displacement_active_at": (
                            None
                            if state.source_displacement_active_at is None
                            else state.source_displacement_active_at.isoformat()
                        ),
                        "source_displacement_prefix_commitment": (
                            state.source_displacement_prefix_commitment
                        ),
                        "source_candle_starts": tuple(
                            value.isoformat()
                            for value in state.source_candle_starts
                        ),
                    },
                    entity_id=(None if midpoint_revision else state.fvg_id),
                    lifecycle=(
                        None
                        if midpoint_revision
                        else state.lifecycle.value
                    ),
                    formed_at=state.formed_at,
                    confirmed_at=state.confirmed_at,
                    ended_at=observed_at if terminal else None,
                    direction=state.direction,
                    transition_reason=state.transition_reason,
                )
            self.memory.append(
                fvg_state_event,
                include_in_recent=False,
            )
            if state.lifecycle is FairValueGapLifecycle.OPEN:
                source_bar_events = tuple(
                    self._bar_event_id_for_candle_id(candle_id)
                    for candle_id in state.source_candle_ids
                )
                if len(source_bar_events) != 3:
                    raise ValueError(
                        "FVG creation requires exactly three completed-bar "
                        "source events"
                    )
                displacement_context_id = self._displacement_event_ids.get(
                    state.source_active_transition_id
                ) or self._displacement_event_ids.get(
                    state.source_displacement_id
                )
                created = self._append_semantic_atomic(
                    EventKind.FVG_CREATED,
                    state.confirmed_at,
                    Timeframe.M5,
                    (
                        "below"
                        if state.direction is Direction.LONG
                        else "above"
                    ),
                    state.midpoint,
                    state.strength,
                    source_bar_events,
                    {
                        "fvg_id": state.fvg_id,
                        "qualification": state.qualification.value,
                        "width_ticks": state.width_ticks,
                        "width_atr": state.width_atr,
                        "source_displacement_id": (
                            state.source_displacement_id
                        ),
                        "source_candle_ids": state.source_candle_ids,
                    },
                    direction=state.direction,
                    event_time=state.formed_at,
                    zone=(state.lower_bound, state.upper_bound),
                    source_data_ids=state.source_candle_ids,
                    source_entity_ids=(
                        state.fvg_id,
                        state.source_displacement_id,
                    ),
                    context_event_ids=(
                        fvg_state_event.event_id,
                        *((
                            displacement_context_id,
                        ) if displacement_context_id else ()),
                    ),
                )
                self._fvg_created_event_ids[state.fvg_id] = (
                    created.event_id
                )
            elif state.lifecycle in {
                FairValueGapLifecycle.PARTIAL,
                FairValueGapLifecycle.MITIGATED,
                FairValueGapLifecycle.INVALIDATED,
                FairValueGapLifecycle.EXPIRED,
            }:
                created_event_id = self._fvg_created_event_ids.get(
                    state.fvg_id
                )
                if created_event_id is None:
                    raise ValueError(
                        "FVG lifecycle transition lacks its creation event"
                    )
                lifecycle_kind = {
                    FairValueGapLifecycle.PARTIAL: (
                        EventKind.FVG_MIDPOINT_TOUCHED
                        if midpoint_revision
                        else EventKind.FVG_PARTIALLY_FILLED
                    ),
                    FairValueGapLifecycle.MITIGATED: (
                        EventKind.FVG_FULLY_FILLED
                    ),
                    FairValueGapLifecycle.INVALIDATED: (
                        EventKind.FVG_INVALIDATED
                    ),
                    FairValueGapLifecycle.EXPIRED: EventKind.FVG_EXPIRED,
                }[state.lifecycle]
                try:
                    transition_bar_event_id = self._bar_event_id_at(
                        Timeframe.M5,
                        observed_at,
                    )
                except ValueError:
                    transition_bar_event_id = None
                source_event_ids = (
                    created_event_id,
                    *((
                        transition_bar_event_id,
                    ) if transition_bar_event_id else ()),
                )
                evidence = {
                    "fvg_id": state.fvg_id,
                    "lifecycle": state.lifecycle.value,
                    "max_fill_fraction": state.max_fill_fraction,
                    "midpoint_touched": bool(
                        state.midpoint_touched_at is not None
                    ),
                    "midpoint_touched_at": (
                        None
                        if state.midpoint_touched_at is None
                        else state.midpoint_touched_at.isoformat()
                    ),
                    "fully_filled": bool(
                        state.lifecycle
                        is FairValueGapLifecycle.MITIGATED
                    ),
                    "transition_reason": state.transition_reason,
                }
                if (
                    state.fvg_id not in self._fvg_first_retest_event_ids
                    and state.lifecycle
                    is not FairValueGapLifecycle.EXPIRED
                ):
                    # Partial, mitigated and invalidated all require price to
                    # have entered the frozen gap on this bar; only the
                    # reserved age-based expiry does not.  The first of them
                    # is therefore the first re-entry, and it is published
                    # before the revisable fill observation it shares a bar
                    # with.
                    retest = self._append_semantic_atomic(
                        EventKind.FVG_FIRST_RETEST,
                        observed_at,
                        Timeframe.M5,
                        (
                            "below"
                            if state.direction is Direction.LONG
                            else "above"
                        ),
                        state.midpoint,
                        state.strength,
                        source_event_ids,
                        self._fvg_first_retest_evidence(
                            state,
                            observed_at=observed_at,
                            entry_bar_event_id=transition_bar_event_id,
                        ),
                        direction=state.direction,
                        event_time=observed_at,
                        zone=(state.lower_bound, state.upper_bound),
                        source_entity_ids=(state.fvg_id,),
                        context_event_ids=(fvg_state_event.event_id,),
                    )
                    self._fvg_first_retest_event_ids[state.fvg_id] = (
                        retest.event_id
                    )
                lifecycle_event = self._append_semantic_atomic(
                    lifecycle_kind,
                    observed_at,
                    Timeframe.M5,
                    (
                        "below"
                        if state.direction is Direction.LONG
                        else "above"
                    ),
                    state.midpoint,
                    state.strength,
                    source_event_ids,
                    evidence,
                    direction=state.direction,
                    event_time=observed_at,
                    zone=(state.lower_bound, state.upper_bound),
                    source_entity_ids=(state.fvg_id,),
                    context_event_ids=(fvg_state_event.event_id,),
                )
                if state.lifecycle in {
                    FairValueGapLifecycle.INVALIDATED,
                    FairValueGapLifecycle.EXPIRED,
                }:
                    self._fvg_terminal_event_ids[state.fvg_id] = (
                        lifecycle_event.event_id
                    )
        for state in update.order_block_transitions:
            observed_at = state.state_started_at
            terminal = state.lifecycle in {
                OrderBlockLifecycle.MITIGATED,
                OrderBlockLifecycle.FAILED,
            }
            order_block_state_event = _event(
                    EventKind.ORDER_BLOCK_STATE,
                    observed_at,
                    Timeframe.M5,
                    (
                        "below"
                        if state.direction is Direction.LONG
                        else "above"
                    ),
                    state.midpoint,
                    state.strength,
                    (
                        state.source_displacement_id,
                        state.source_active_transition_id,
                        state.source_bos_id,
                        state.anchor_candle_id,
                    ),
                    {
                        "protocol_hash": state.protocol_hash,
                        "lower_bound": state.lower_bound,
                        "upper_bound": state.upper_bound,
                        "midpoint": state.midpoint,
                        "invalidation_price": (
                            state.invalidation_price
                        ),
                        "anchor_open": state.anchor_open,
                        "anchor_close": state.anchor_close,
                        "anchor_candle_ids": state.anchor_candle_ids,
                        "body_lower_bound": state.body_lower_bound,
                        "body_upper_bound": state.body_upper_bound,
                        "width_ticks": state.width_ticks,
                        "width_atr": state.width_atr,
                        "age_bars": state.age_bars,
                        "source_displacement_protocol_hash": (
                            state.source_displacement_protocol_hash
                        ),
                        "source_displacement_seed_candle_id": (
                            state.source_displacement_seed_candle_id
                        ),
                        "source_displacement_started_at": (
                            state.source_displacement_started_at.isoformat()
                        ),
                        "source_displacement_active_at": (
                            state.source_displacement_active_at.isoformat()
                        ),
                        "source_displacement_prefix_commitment": (
                            state.source_displacement_prefix_commitment
                        ),
                        "source_bos_protocol_hash": (
                            state.source_bos_protocol_hash
                        ),
                        "source_bos_target_swing_id": (
                            state.source_bos_target_swing_id
                        ),
                        "source_bos_structure_id": (
                            state.source_bos_structure_id
                        ),
                        "source_bos_scope": (
                            state.source_bos_scope.value
                        ),
                        "source_bos_resolved_at": (
                            state.source_bos_resolved_at.isoformat()
                        ),
                        "source_bos_pending_at": (
                            state.source_bos_pending_at.isoformat()
                        ),
                        "source_bos_break_bar_id": (
                            state.source_bos_break_bar_id
                        ),
                        "source_bos_mss_qualified": (
                            state.source_bos_mss_qualified
                        ),
                        "anchor_start": state.anchor_start.isoformat(),
                        "anchor_end": state.anchor_end.isoformat(),
                    },
                    entity_id=state.order_block_id,
                    lifecycle=state.lifecycle.value,
                    formed_at=state.formed_at,
                    confirmed_at=state.confirmed_at,
                    ended_at=observed_at if terminal else None,
                    direction=state.direction,
                    transition_reason=state.transition_reason,
                )
            self.memory.append(
                order_block_state_event,
                include_in_recent=False,
            )
            created_event_id = self._origin_zone_created_event_ids.get(
                state.order_block_id
            )
            if state.lifecycle in {
                OrderBlockLifecycle.CREATED,
                OrderBlockLifecycle.UNTESTED,
            } and created_event_id is None:
                displacement_event_id = self._displacement_event_ids.get(
                    state.source_active_transition_id
                ) or self._displacement_event_ids.get(
                    state.source_displacement_id
                )
                raw_break_event_id = self._raw_break_event_ids.get(
                    state.source_bos_id
                )
                if (
                    displacement_event_id is None
                    or raw_break_event_id is None
                ):
                    raise ValueError(
                        "origin zone lacks its exact displacement or raw "
                        "boundary-break event"
                    )
                # The core is not minted here: the impulse published it when
                # it locked the candles, before this break existed.  A
                # qualification that cannot find its core is a causality bug,
                # never a reason to backdate one.
                core_event_id = self._base_origin_core_event_ids.get(
                    state.base_origin_core_id or ""
                )
                if core_event_id is None:
                    raise ValueError(
                        "qualified origin zone lacks the base origin core its "
                        "impulse published"
                    )
                created = self._append_semantic_atomic(
                    EventKind.QUALIFIED_ORIGIN_ZONE_CREATED,
                    state.confirmed_at,
                    Timeframe.M5,
                    (
                        "below"
                        if state.direction is Direction.LONG
                        else "above"
                    ),
                    state.midpoint,
                    state.strength,
                    (
                        core_event_id,
                        displacement_event_id,
                        raw_break_event_id,
                    ),
                    {
                        "origin_zone_id": state.order_block_id,
                        "base_origin_core_id": state.base_origin_core_id,
                        "base_origin_core_event_id": core_event_id,
                        "source_displacement_id": (
                            state.source_displacement_id
                        ),
                        "source_bos_id": state.source_bos_id,
                        "anchor_candle_ids": state.anchor_candle_ids,
                        "lifecycle": state.lifecycle.value,
                    },
                    direction=state.direction,
                    event_time=state.formed_at,
                    zone=(state.lower_bound, state.upper_bound),
                    source_entity_ids=(
                        state.order_block_id,
                        state.source_displacement_id,
                        state.source_bos_id,
                    ),
                    context_event_ids=(order_block_state_event.event_id,),
                )
                self._origin_zone_created_event_ids[
                    state.order_block_id
                ] = created.event_id
            elif state.lifecycle in {
                OrderBlockLifecycle.MITIGATED,
                OrderBlockLifecycle.FAILED,
            }:
                if created_event_id is None:
                    raise ValueError(
                        "origin-zone terminal transition lacks creation event"
                    )
                try:
                    transition_bar_event_id = self._bar_event_id_at(
                        Timeframe.M5,
                        observed_at,
                    )
                except ValueError:
                    transition_bar_event_id = None
                source_events = (
                    created_event_id,
                    *((
                        transition_bar_event_id,
                    ) if transition_bar_event_id else ()),
                )
                self._append_semantic_atomic(
                    (
                        EventKind.ORIGIN_ZONE_MITIGATED
                        if state.lifecycle
                        is OrderBlockLifecycle.MITIGATED
                        else EventKind.ORIGIN_ZONE_INVALIDATED
                    ),
                    observed_at,
                    Timeframe.M5,
                    (
                        "below"
                        if state.direction is Direction.LONG
                        else "above"
                    ),
                    state.midpoint,
                    state.strength,
                    source_events,
                    {
                        "origin_zone_id": state.order_block_id,
                        "lifecycle": state.lifecycle.value,
                        "transition_reason": state.transition_reason,
                    },
                    direction=state.direction,
                    event_time=observed_at,
                    zone=(state.lower_bound, state.upper_bound),
                    source_entity_ids=(state.order_block_id,),
                    context_event_ids=(order_block_state_event.event_id,),
                )

    def _record_displacement_events(
        self,
        displacement,
    ) -> None:
        if displacement is None:
            return
        for transition in displacement.transitions_this_update:
            if not self._remember_bounded(
                transition.transition_id,
                known=self._known_displacement_transition_ids,
                order=self._known_displacement_transition_order,
            ):
                continue
            metrics = {
                str(name): float(value)
                for name, value in transition.state_metrics
            }
            source_bar_events = tuple(
                self._bar_event_id_for_candle_id(candle_id)
                for candle_id in transition.admitted_candle_ids
            )
            source_bar_facts = tuple(
                self.memory.audit_event_including_pending(event_id)
                for event_id in source_bar_events
            )
            if not source_bar_facts or any(
                event is None
                or event.origin is not EventOrigin.NORMALIZED_DATA
                or event.kind is not EventKind.BAR_COMPLETED
                or event.timeframe is not Timeframe.M5
                or event.evidence.get("real_completed") is not True
                or event.evidence.get("clock_only") is not False
                or not isinstance(
                    event.evidence.get("detector_candle_id"), str
                )
                or not event.evidence.get("detector_candle_id")
                for event in source_bar_facts
            ):
                raise ValueError(
                    "displacement semantic source BAR detector lineage is invalid"
                )
            source_bar_detector_ids = tuple(
                str(event.evidence["detector_candle_id"])
                for event in source_bar_facts
                if event is not None
            )
            if (
                len(source_bar_detector_ids)
                != len(set(source_bar_detector_ids))
                or len(transition.admitted_candle_ids)
                != len(set(transition.admitted_candle_ids))
                or set(source_bar_detector_ids)
                != set(transition.admitted_candle_ids)
            ):
                raise ValueError(
                    "displacement admitted detector candle lineage is invalid"
                )
            synthetic_context_event_ids: tuple[str, ...] = ()
            if (
                transition.lifecycle == "censored"
                and transition.reason == "synthetic_interruption"
            ):
                synthetic_context_event_ids = (
                    self._synthetic_m1_context_event_ids_for_m5_terminal(
                        transition.observed_at,
                    )
                )
            displacement_event = self._append_semantic_atomic(
                EventKind.DISPLACEMENT_OBSERVED,
                transition.observed_at,
                Timeframe.M5,
                (
                    "above"
                    if transition.direction is Direction.LONG
                    else "below"
                ),
                None,
                clamp(metrics.get("efficiency", 0.0)),
                source_bar_events,
                {
                    "transition_id": transition.transition_id,
                    "displacement_id": transition.entity_id,
                    "lifecycle": transition.lifecycle,
                    "terminal_reason": transition.reason,
                    "state_metrics": metrics,
                    "admitted_candle_ids": (
                        transition.admitted_candle_ids
                    ),
                    "prefix_last_admitted_at": (
                        None
                        if transition.prefix_last_admitted_at is None
                        else transition.prefix_last_admitted_at.isoformat()
                    ),
                },
                direction=transition.direction,
                event_time=(
                    transition.started_at or transition.observed_at
                ),
                source_data_ids=transition.admitted_candle_ids,
                source_entity_ids=(transition.entity_id,),
                context_event_ids=synthetic_context_event_ids,
            )
            self._displacement_event_ids[transition.transition_id] = (
                displacement_event.event_id
            )
            self._displacement_event_ids[transition.entity_id] = (
                displacement_event.event_id
            )

    def _record_group4_events(
        self,
        update: RangeAuctionUpdate,
        *,
        include_ranges: bool = True,
        include_resolutions: bool = True,
        include_creations: bool = True,
        prior: MarketObservation | None,
    ) -> None:
        if any(
            type(value) is not bool
            for value in (
                include_ranges,
                include_resolutions,
                include_creations,
            )
        ):
            raise TypeError("Group 4 event phase flags must be boolean")
        boundary_reason = update.boundary_reason
        if boundary_reason is not None and any(
            state.lifecycle is not DealingRangeLifecycle.BROKEN
            or state.broken_at != state.state_started_at
            or state.transition_reason != boundary_reason
            for state in update.range_transitions
        ):
            raise ValueError(
                "Group 4 boundary range transition is not an exact terminal"
            )
        if (
            boundary_reason is None
            and include_creations
            and prior is not None
        ):
            prior_manipulations = {
                state.manipulation_id: state
                for state in prior.manipulations
            }
            revision_fields = (
                "reentry_candidate_at",
                "reentry_candidate_price",
                "inside_hold_bars",
                "reentry_failed_at",
                "outside_run",
                "outside_run_side",
            )
            for state in update.manipulations:
                prior_state = prior_manipulations.get(
                    state.manipulation_id
                )
                if (
                    prior_state is None
                    or state.lifecycle
                    is not ManipulationLifecycle.SWEPT
                    or state.deadline_elapsed
                    or all(
                        getattr(state, name)
                        == getattr(prior_state, name)
                        for name in revision_fields
                    )
                ):
                    continue
                if (
                    state.reentry_candidate_at is not None
                    and state.reentry_candidate_at
                    != prior_state.reentry_candidate_at
                ):
                    revision_reason = "reentry_candidate_started"
                elif (
                    state.reentry_failed_at is not None
                    and state.reentry_failed_at
                    != prior_state.reentry_failed_at
                ):
                    revision_reason = "reentry_candidate_failed"
                else:
                    revision_reason = "outside_acceptance_progressed"
                self.memory.append(
                    _event(
                        EventKind.MANIPULATION_STATE,
                        state.last_updated_at,
                        Timeframe.M1,
                        state.side,
                        (
                            state.reentry_candidate_price
                            if state.reentry_candidate_price is not None
                            else state.sweep_extreme
                        ),
                        state.strength,
                        (
                            state.manipulation_id,
                            state.source_inventory_item_id,
                            *state.crossed_source_ids,
                        ),
                        {
                            "state_revision": True,
                            "revision_reason": revision_reason,
                            "protocol_hash": state.protocol_hash,
                            "source_kind": state.source_kind,
                            "reentry_candidate_at": (
                                None
                                if state.reentry_candidate_at is None
                                else state.reentry_candidate_at.isoformat()
                            ),
                            "reentry_candidate_price": (
                                state.reentry_candidate_price
                            ),
                            "inside_hold_bars": state.inside_hold_bars,
                            "reentry_failed_at": (
                                None
                                if state.reentry_failed_at is None
                                else state.reentry_failed_at.isoformat()
                            ),
                            "outside_run": state.outside_run,
                            "outside_run_side": state.outside_run_side,
                            "age_1m_bars": state.age_1m_bars,
                        },
                        transition_reason=revision_reason,
                    )
                )
        for state in (
            update.range_transitions
            if include_ranges
            else ()
        ):
            terminal = (
                state.lifecycle is DealingRangeLifecycle.BROKEN
            )
            range_state_event = _event(
                    EventKind.DEALING_RANGE_STATE,
                    state.state_started_at,
                    Timeframe.H1,
                    None,
                    state.midpoint,
                    state.strength,
                    (
                        state.lower_source_zone_id,
                        state.upper_source_zone_id,
                        *state.lower_source_member_swing_ids,
                        *state.upper_source_member_swing_ids,
                    ),
                    {
                        "protocol_hash": state.protocol_hash,
                        "lower_bound": state.lower_bound,
                        "upper_bound": state.upper_bound,
                        "value_price": state.value_price,
                        "candidate_real_h1_bars": (
                            state.candidate_real_h1_bars
                        ),
                        "lower_touch_count": state.lower_touch_count,
                        "upper_touch_count": state.upper_touch_count,
                        "midpoint_crossings": (
                            state.midpoint_crossings
                        ),
                        "inside_close_fraction": (
                            state.inside_close_fraction
                        ),
                        "compression_ratio": (
                            state.compression_ratio
                        ),
                        "age_h1_bars": state.age_h1_bars,
                    },
                    entity_id=state.range_id,
                    lifecycle=state.lifecycle.value,
                    formed_at=state.formed_at,
                    confirmed_at=state.balance_confirmed_at,
                    ended_at=state.broken_at if terminal else None,
                    transition_reason=state.transition_reason,
                )
            self.memory.append(
                range_state_event,
                include_in_recent=False,
            )
            if boundary_reason is not None:
                # Contract/data-gap resets censor the previous market epoch;
                # they are not a market-observed H1 acceptance.  Preserve the
                # lifecycle timeline transport, while the MARKET_EPOCH_RESET
                # event owns the authoritative causal transition.
                continue
            # A Structural Range has two location facts and no third: it is
            # created, and it is invalidated when price closes outside it.
            # Balance is published separately as BALANCE_RANGE_OBSERVED and is
            # never a transition of the interval.
            range_kind = {
                DealingRangeLifecycle.ACTIVE: (
                    EventKind.DEALING_RANGE_CREATED
                ),
                DealingRangeLifecycle.BROKEN: (
                    EventKind.DEALING_RANGE_INVALIDATED
                ),
            }[state.lifecycle]
            created_event_id = self._range_created_event_ids.get(
                state.range_id
            )
            active_event_id = self._range_active_event_ids.get(
                state.range_id
            )
            anchor_event_ids = tuple(
                dict.fromkeys(
                    event_id
                    for event_id in (
                        self._candidate_level_event_ids.get(
                            state.lower_source_zone_id
                        ),
                        self._candidate_level_event_ids.get(
                            state.upper_source_zone_id
                        ),
                        *(
                            self._confirmed_swing_event_ids.get(swing_id)
                            for swing_id in (
                                *state.lower_source_member_swing_ids,
                                *state.upper_source_member_swing_ids,
                            )
                        ),
                    )
                    if event_id is not None
                )
            )
            try:
                transition_bar_event_id = self._bar_event_id_at(
                    Timeframe.H1,
                    state.state_started_at,
                )
            except ValueError:
                transition_bar_event_id = None
            if (
                state.lifecycle is not DealingRangeLifecycle.ACTIVE
                and created_event_id is None
                and transition_bar_event_id is None
            ):
                # Private compatibility callers can project an isolated range
                # lifecycle state without its normalized history.  Keep only
                # the legacy lifecycle transport in that case; an authoritative
                # semantic transition may never invent missing ancestry.
                continue
            external_acceptance_event_id: str | None = None
            if (
                state.lifecycle is DealingRangeLifecycle.BROKEN
                and state.transition_reason == "close_beyond_frozen_range"
                and active_event_id is not None
            ):
                if transition_bar_event_id is None:
                    raise ValueError(
                        "active dealing-range acceptance lacks its exact "
                        "completed H1 BAR event"
                    )
                accepted_close = self._bar_close_by_event_id.get(
                    transition_bar_event_id
                )
                if accepted_close is None:
                    raise ValueError(
                        "active dealing-range acceptance lacks its frozen "
                        "completed H1 close"
                    )
                if accepted_close < state.lower_bound:
                    accepted_side = "below"
                    accepted_price = float(state.lower_bound)
                    accepted_direction = Direction.SHORT
                elif accepted_close > state.upper_bound:
                    accepted_side = "above"
                    accepted_price = float(state.upper_bound)
                    accepted_direction = Direction.LONG
                else:
                    raise ValueError(
                        "close-beyond range transition does not close "
                        "strictly outside its frozen bounds"
                    )
                level_id = self._range_boundary_level_ids.get(
                    (state.range_id, accepted_side)
                )
                if level_id is None:
                    raise ValueError(
                        "active dealing-range acceptance lacks its frozen "
                        "candidate boundary identity"
                    )
                candidate_event_id = self._candidate_level_event_ids.get(
                    level_id
                )
                if candidate_event_id is None:
                    raise ValueError(
                        "active dealing-range boundary lacks its candidate "
                        "creation event"
                    )
                crossing_generation_id = self._crossing_generation_id(
                    level_id=level_id,
                    timeframe=Timeframe.H1,
                    crossed_at=state.state_started_at,
                )
                touch_event = self._append_semantic_atomic(
                    EventKind.LEVEL_TOUCHED,
                    state.state_started_at,
                    Timeframe.H1,
                    accepted_side,
                    accepted_price,
                    state.strength,
                    (candidate_event_id, transition_bar_event_id),
                    {
                        "level_id": level_id,
                        "range_id": state.range_id,
                        "source_kind": "mature_range_boundary",
                        "touch_reason": "external_h1_close_crossing",
                    },
                    event_time=state.state_started_at,
                    zone=(accepted_price, accepted_price),
                    source_entity_ids=(level_id, state.range_id),
                )
                penetrated_event = self._append_semantic_atomic(
                    EventKind.LEVEL_PENETRATED,
                    state.state_started_at,
                    Timeframe.H1,
                    accepted_side,
                    accepted_price,
                    state.strength,
                    (
                        candidate_event_id,
                        touch_event.event_id,
                        transition_bar_event_id,
                    ),
                    {
                        "level_id": level_id,
                        "range_id": state.range_id,
                        "source_kind": "mature_range_boundary",
                        "penetration_standard": (
                            "first_completed_h1_close_strictly_outside_"
                            "frozen_range"
                        ),
                        "crossing_generation_id": crossing_generation_id,
                        "crossed_at": state.state_started_at.isoformat(),
                        "accepted_close": float(accepted_close),
                    },
                    direction=accepted_direction,
                    event_time=state.state_started_at,
                    zone=(accepted_price, accepted_price),
                    source_entity_ids=(level_id, state.range_id),
                )
                accepted_event = self._append_crossing_resolution(
                    EventKind.ACCEPTANCE_CONFIRMED,
                    state.state_started_at,
                    Timeframe.H1,
                    accepted_side,
                    accepted_price,
                    state.strength,
                    (
                        penetrated_event.event_id,
                        transition_bar_event_id,
                    ),
                    {
                        "level_id": level_id,
                        "range_id": state.range_id,
                        "source_kind": "mature_range_boundary",
                        "acceptance_bars": 1,
                        "accepted_close": float(accepted_close),
                        "resolution_standard": (
                            "first_completed_h1_close_strictly_outside_"
                            "frozen_range"
                        ),
                    },
                    direction=accepted_direction,
                    crossed_at=state.state_started_at,
                    zone=(accepted_price, accepted_price),
                    known_at=state.state_started_at,
                )
                external_acceptance_event_id = accepted_event.event_id
            if state.lifecycle is DealingRangeLifecycle.ACTIVE:
                range_sources = anchor_event_ids
            else:
                if created_event_id is None:
                    raise ValueError(
                        "dealing-range transition lacks its creation event"
                    )
                range_sources = (
                    created_event_id,
                    *((active_event_id,) if active_event_id else ()),
                    *((
                        transition_bar_event_id,
                    ) if transition_bar_event_id else ()),
                    *((
                        external_acceptance_event_id,
                    ) if external_acceptance_event_id else ()),
                )
            semantic_transition_reason = state.transition_reason
            if (
                state.lifecycle is DealingRangeLifecycle.BROKEN
                and state.transition_reason == "close_beyond_frozen_range"
                and active_event_id is None
            ):
                # A forming candidate can fail before it ever becomes the
                # active range.  It has no activated boundary inventory and
                # therefore cannot manufacture the active-range Acceptance
                # ancestry used by a mature range invalidation.
                semantic_transition_reason = (
                    "close_beyond_frozen_range_before_activation"
                )
            range_event = self._append_semantic_atomic(
                range_kind,
                state.state_started_at,
                Timeframe.H1,
                None,
                state.midpoint,
                state.strength,
                range_sources,
                {
                    "range_id": state.range_id,
                    "lifecycle": state.lifecycle.value,
                    "lower_bound": state.lower_bound,
                    "upper_bound": state.upper_bound,
                    "normalized_location_unclamped": None,
                    "lower_source_zone_id": (
                        state.lower_source_zone_id
                    ),
                    "upper_source_zone_id": (
                        state.upper_source_zone_id
                    ),
                    "source_member_swing_ids": (
                        *state.lower_source_member_swing_ids,
                        *state.upper_source_member_swing_ids,
                    ),
                    "transition_reason": semantic_transition_reason,
                },
                event_time=(
                    state.formed_at
                    if state.lifecycle is DealingRangeLifecycle.ACTIVE
                    else state.state_started_at
                ),
                zone=(state.lower_bound, state.upper_bound),
                source_entity_ids=(
                    state.range_id,
                    state.lower_source_zone_id,
                    state.upper_source_zone_id,
                    *state.lower_source_member_swing_ids,
                    *state.upper_source_member_swing_ids,
                ),
                context_event_ids=(range_state_event.event_id,),
            )
            self._record_balance_range_observation(
                state,
                observed_at=state.state_started_at,
                anchor_event_id=(
                    range_event.event_id
                    if state.lifecycle is DealingRangeLifecycle.ACTIVE
                    else created_event_id or range_event.event_id
                ),
                bar_event_id=transition_bar_event_id,
                context_event_ids=(range_state_event.event_id,),
            )
            if state.lifecycle is DealingRangeLifecycle.ACTIVE:
                self._range_created_event_ids[state.range_id] = (
                    range_event.event_id
                )
                if self._last_invalidated_range_event_id is not None:
                    self._append_semantic_atomic(
                        EventKind.DEALING_RANGE_REPLACED,
                        state.state_started_at,
                        Timeframe.H1,
                        None,
                        state.midpoint,
                        state.strength,
                        (
                            self._last_invalidated_range_event_id,
                            range_event.event_id,
                        ),
                        {
                            "replacement_range_id": state.range_id,
                            "replacement_standard": (
                                "new_registered_range_after_invalidation"
                            ),
                        },
                        event_time=state.formed_at,
                        zone=(state.lower_bound, state.upper_bound),
                    )
                    self._last_invalidated_range_event_id = None
            if state.lifecycle is DealingRangeLifecycle.BROKEN:
                self._range_terminal_event_ids[state.range_id] = (
                    range_event.event_id
                )
                self._last_invalidated_range_event_id = (
                    range_event.event_id
                )

        # A hard reset has already replaced EventMemory and imported only
        # transitionable old-epoch prefixes.  Join a range's BROKEN transition
        # to that exact identity at the boundary clock, but do not reinterpret
        # a boundary-censored manipulation (whose reducer lifecycle is still
        # SWEPT) as a new creation in the fresh epoch.
        if boundary_reason is not None:
            return

        if include_ranges:
            # Boundary promotion first: the levels a settled claim mints
            # are ancestry the observation may cite.
            self._record_confirmed_range_boundaries(update)
            self._record_live_balance_range_observations(update)

        # Publish the physical mature-boundary crossing independently of
        # whether Group 4 has enough prior ATR to classify a manipulation.
        # The tracker conserves every raw crossing in source_dispositions and
        # retains the consumed boundary geometry in this bounded update.
        if include_creations:
            for disposition in update.source_dispositions:
                boundary_items = tuple(
                    item
                    for item in update.range_boundary_inventory
                    if (
                        item.item_id
                        == disposition.source_inventory_item_id
                        and item.kind == "range_boundary"
                    )
                )
                if not boundary_items:
                    continue
                if len(boundary_items) != 1:
                    raise ValueError(
                        "Group 4 physical range-boundary identity repeats"
                    )
                item = boundary_items[0]
                crossed_at = disposition.observed_at
                if (
                    item.lifecycle
                    is not LiquidityInventoryLifecycle.CONSUMED
                    or item.consumed_at != crossed_at
                ):
                    raise ValueError(
                        "Group 4 range-boundary disposition lacks its exact "
                        "physical consumption clock"
                    )
                penetration_key = self._penetration_key(
                    level_id=item.item_id,
                    timeframe=Timeframe.M1,
                    crossed_at=crossed_at,
                )
                if penetration_key in self._penetration_event_ids:
                    continue
                candidate_event_id = self._candidate_level_event_ids.get(
                    item.item_id
                )
                if candidate_event_id is None:
                    raise ValueError(
                        "Group 4 range-boundary crossing lacks its "
                        "first-visible candidate"
                    )
                candidate_event = self.memory.audit_event_including_pending(
                    candidate_event_id
                )
                range_id = (
                    None
                    if candidate_event is None
                    else candidate_event.evidence.get("range_id")
                )
                if not isinstance(range_id, str) or not range_id:
                    raise ValueError(
                        "Group 4 range-boundary candidate lacks its owner"
                    )
                bar_event_id = self._bar_event_id_at(
                    Timeframe.M1,
                    crossed_at,
                )
                bar_event = self.memory.audit_event_including_pending(
                    bar_event_id
                )
                if bar_event is None:
                    raise ValueError(
                        "Group 4 range-boundary crossing lacks its exact BAR"
                    )
                boundary_price = (
                    item.upper_bound
                    if item.side == "above"
                    else item.lower_bound
                )
                crossing_price = float(
                    bar_event.evidence[
                        "high" if item.side == "above" else "low"
                    ]
                )
                touch_event = self._append_semantic_atomic(
                    EventKind.LEVEL_TOUCHED,
                    crossed_at,
                    Timeframe.M1,
                    item.side,
                    boundary_price,
                    item.strength,
                    (candidate_event_id, bar_event_id),
                    {
                        "level_id": item.item_id,
                        "range_id": range_id,
                        "source_timeframe": item.timeframe.value,
                        "source_kind": "mature_range_boundary",
                        "touch_reason": "registered_m1_boundary_crossing",
                    },
                    event_time=crossed_at,
                    zone=(item.lower_bound, item.upper_bound),
                    source_entity_ids=(item.item_id, range_id),
                )
                self._level_touch_event_ids[
                    (item.item_id, pd.Timestamp(crossed_at))
                ] = touch_event.event_id
                crossing_generation_id = self._crossing_generation_id(
                    level_id=item.item_id,
                    timeframe=Timeframe.M1,
                    crossed_at=crossed_at,
                )
                penetration_event = self._append_semantic_atomic(
                    EventKind.LEVEL_PENETRATED,
                    crossed_at,
                    Timeframe.M1,
                    item.side,
                    crossing_price,
                    item.strength,
                    (
                        candidate_event_id,
                        touch_event.event_id,
                        bar_event_id,
                    ),
                    {
                        "level_id": item.item_id,
                        "range_id": range_id,
                        "source_timeframe": item.timeframe.value,
                        "source_kind": "mature_range_boundary",
                        "frozen_lower_bound": item.lower_bound,
                        "frozen_upper_bound": item.upper_bound,
                        "crossing_generation_id": crossing_generation_id,
                        "crossed_at": crossed_at.isoformat(),
                        "penetration_standard": (
                            "registered_m1_wick_beyond_mature_range_boundary"
                        ),
                    },
                    direction=(
                        Direction.LONG
                        if item.side == "above"
                        else Direction.SHORT
                    ),
                    event_time=crossed_at,
                    zone=(item.lower_bound, item.upper_bound),
                    source_entity_ids=(item.item_id, range_id),
                )
                self._penetration_event_ids[penetration_key] = (
                    penetration_event.event_id
                )

        for state in update.manipulation_transitions:
            terminal = state.lifecycle in {
                ManipulationLifecycle.REACCEPTED,
                ManipulationLifecycle.ACCEPTED_OUTSIDE,
            } or state.deadline_elapsed
            if (
                (terminal and not include_resolutions)
                or (not terminal and not include_creations)
            ):
                continue
            manipulation_state_event = _event(
                    EventKind.MANIPULATION_STATE,
                    (
                        state.censored_at
                        if state.deadline_elapsed
                        else state.state_started_at
                    ),
                    Timeframe.M1,
                    state.side,
                    (
                        state.reentry_price
                        if state.reentry_price is not None
                        else state.sweep_extreme
                    ),
                    state.strength,
                    (
                        state.source_inventory_item_id,
                        *state.crossed_source_ids,
                    ),
                    {
                        "protocol_hash": state.protocol_hash,
                        "source_kind": state.source_kind,
                        "source_timeframe": (
                            state.source_timeframe.value
                        ),
                        "source_lower_bound": (
                            state.source_lower_bound
                        ),
                        "source_upper_bound": (
                            state.source_upper_bound
                        ),
                        "sweep_extreme": state.sweep_extreme,
                        "close_outside_on_sweep": (
                            state.close_outside_on_sweep
                        ),
                        "outside_completed_bars": (
                            state.outside_completed_bars
                        ),
                        "outside_run": state.outside_run,
                        "outside_run_side": state.outside_run_side,
                        "reentry_candidate_at": (
                            None
                            if state.reentry_candidate_at is None
                            else state.reentry_candidate_at.isoformat()
                        ),
                        "reentry_candidate_price": (
                            state.reentry_candidate_price
                        ),
                        "inside_hold_bars": state.inside_hold_bars,
                        "reentry_failed_at": (
                            None
                            if state.reentry_failed_at is None
                            else state.reentry_failed_at.isoformat()
                        ),
                        "deadline_at": (
                            None
                            if state.deadline_at is None
                            else state.deadline_at.isoformat()
                        ),
                        "deadline_elapsed": state.deadline_elapsed,
                        "penetration_atr": state.penetration_atr,
                        "age_1m_bars": state.age_1m_bars,
                        "resolved_side": state.resolved_side,
                    },
                    entity_id=state.manipulation_id,
                    lifecycle=(
                        "censored"
                        if state.deadline_elapsed
                        else state.lifecycle.value
                    ),
                    formed_at=state.formed_at,
                    confirmed_at=state.confirmed_at,
                    ended_at=(
                        state.censored_at
                        if state.deadline_elapsed
                        else state.resolved_at if terminal else None
                    ),
                    transition_reason=state.transition_reason,
                )
            self.memory.append(
                manipulation_state_event,
                include_in_recent=False,
                sequence_floor=(
                    None
                    if terminal
                    else EventMemory._GROUP4_CREATION_SEQUENCE_FLOOR
                ),
            )
            if terminal and not state.deadline_elapsed:
                accepted = (
                    state.lifecycle
                    is ManipulationLifecycle.ACCEPTED_OUTSIDE
                )
                penetration_event_id = self._penetration_event_ids.get(
                    self._penetration_key(
                        level_id=state.source_inventory_item_id,
                        timeframe=Timeframe.M1,
                        crossed_at=state.formed_at,
                    )
                )
                if penetration_event_id is None:
                    if not self._real_bar_event_ids_by_timeframe[Timeframe.M1]:
                        # Legacy private lifecycle fixtures do not register
                        # normalized roots and therefore cannot publish an
                        # authoritative terminal semantic.
                        continue
                    raise ValueError(
                        "Group 4 terminal resolution lacks its canonical "
                        "penetration event"
                    )
                resolution_bar_event_id = self._bar_event_id_at(
                    Timeframe.M1,
                    state.resolved_at,
                )
                self._append_crossing_resolution(
                    (
                        EventKind.ACCEPTANCE_CONFIRMED
                        if accepted
                        else EventKind.SWEEP_CONFIRMED
                    ),
                    state.resolved_at,
                    Timeframe.M1,
                    state.side,
                    (
                        state.reentry_price
                        if state.reentry_price is not None
                        else state.sweep_extreme
                    ),
                    state.strength,
                    (penetration_event_id, resolution_bar_event_id),
                    {
                        "level_id": state.source_inventory_item_id,
                        "manipulation_id": state.manipulation_id,
                        "source_kind": state.source_kind,
                        "source_timeframe": state.source_timeframe.value,
                        "penetration_atr": state.penetration_atr,
                        "outside_completed_bars": (
                            state.outside_completed_bars
                        ),
                        "outside_run": state.outside_run,
                        "inside_hold_bars": state.inside_hold_bars,
                        "resolution": (
                            "registered_outside_acceptance"
                            if accepted
                            else "registered_reacceptance"
                        ),
                    },
                    direction=(
                        (
                            Direction.LONG
                            if state.side == "above"
                            else Direction.SHORT
                        )
                        if accepted
                        else (
                            Direction.SHORT
                            if state.side == "above"
                            else Direction.LONG
                        )
                    ),
                    crossed_at=state.formed_at,
                    zone=(
                        state.source_lower_bound,
                        state.source_upper_bound,
                    ),
                    context_event_ids=(manipulation_state_event.event_id,),
                    known_at=state.resolved_at,
                )

    def _record_live_balance_range_observations(
        self,
        update: RangeAuctionUpdate,
    ) -> None:
        """Test the balance claim on every completed H1 bar.

        Group 4 recomputes each candidate's boundary touches, midpoint
        crossings, inside-close fraction and compression on every completed H1
        bar, but only a lifecycle change ever reached the emitter.  A candidate
        that simply keeps forming has no next transition, so the registered
        two-sided test could be met for an entire month with no clock on which
        the Eye was allowed to say so.  The claim is still published exactly
        once per range -- on the first bar that meets the frozen standard.
        """

        for state in update.dealing_ranges:
            if (
                state.lifecycle is DealingRangeLifecycle.BROKEN
                or state.transition_reason == BALANCE_CLAIM_ABANDONED
                or state.range_id in self._balance_range_observed_event_ids
            ):
                continue
            anchor_event_id = self._range_created_event_ids.get(
                state.range_id
            )
            if anchor_event_id is None:
                # A compatibility caller can project an isolated range without
                # its normalized history.  An authoritative semantic fact may
                # never invent the ancestry it lacks.
                continue
            try:
                bar_event_id = self._bar_event_id_at(
                    Timeframe.H1,
                    state.last_updated_at,
                )
            except ValueError:
                bar_event_id = None
            self._record_balance_range_observation(
                state,
                observed_at=state.last_updated_at,
                anchor_event_id=anchor_event_id,
                bar_event_id=bar_event_id,
            )

    def _record_confirmed_range_boundaries(
        self,
        update: RangeAuctionUpdate,
    ) -> None:
        """Publish the boundary levels a settled balance claim promotes.

        Settling the claim is not a transition of the Structural Range -- the
        interval is the same location it was on the previous bar -- so this
        runs over the live population rather than over transitions.  Provenance
        is the range's own creation event: a boundary level descends from the
        interval that froze it, never from the claim that promoted it.
        """

        for state in update.dealing_ranges:
            if (
                state.balance_confirmed_at is None
                or state.range_id in self._range_active_event_ids
            ):
                continue
            created_event_id = self._range_created_event_ids.get(
                state.range_id
            )
            if created_event_id is None:
                continue
            boundary_items = tuple(
                item
                for item in update.range_boundary_inventory
                if item.kind == "range_boundary"
                and state.range_id in item.source_ids
            )
            if {item.side for item in boundary_items} != {"above", "below"}:
                # A private projector fixture can settle a claim without a
                # normalized inventory behind it.  That is not an
                # authoritative DAG, so do not invent identities for it.
                continue
            self._range_active_event_ids[state.range_id] = created_event_id
            for item in boundary_items:
                self._range_boundary_level_ids[
                    (state.range_id, item.side)
                ] = item.item_id
                if item.item_id in self._candidate_level_event_ids:
                    continue
                candidate = self._append_semantic_atomic(
                    EventKind.LIQUIDITY_LEVEL_CREATED,
                    state.balance_confirmed_at,
                    Timeframe.H1,
                    item.side,
                    item.price,
                    item.strength,
                    (created_event_id,),
                    {
                        "level_id": item.item_id,
                        "range_id": state.range_id,
                        "candidate_only": True,
                        "source_kind": "mature_range_boundary",
                        "source_ids": item.source_ids,
                        "source_formed_at": item.formed_at.isoformat(),
                        "source_confirmed_at": (
                            item.confirmed_at.isoformat()
                        ),
                    },
                    event_time=item.confirmed_at,
                    zone=(item.lower_bound, item.upper_bound),
                    source_entity_ids=(
                        item.item_id,
                        state.range_id,
                        *item.source_ids,
                    ),
                )
                self._candidate_level_event_ids[item.item_id] = (
                    candidate.event_id
                )

    def _record_balance_range_observation(
        self,
        state: DealingRangeState,
        *,
        observed_at: pd.Timestamp,
        anchor_event_id: str,
        bar_event_id: str | None,
        context_event_ids: tuple[str, ...] = (),
    ) -> None:
        """Publish the balance claim over an existing structural range, once.

        Failing to balance never invalidates the location, so this is a
        separate append-only fact rather than a lifecycle of the range itself.
        ``observed_at`` is the completed H1 bar at which the registered
        two-sided test was first met, which is a live bar for a candidate that
        keeps forming and the transition clock for one that changes lifecycle
        on the same bar.  Under ``balance_range_v1.2`` the test is price
        interacting with each frozen boundary, counted in distinct visits, not
        the source zone's structural touch count.
        """

        minimum_tests = int(
            self.semantic_registry.parameters.parameters[
                "balance_range_bilateral_price_tests_each"
            ]["value"]
        )
        if (
            state.range_id in self._balance_range_observed_event_ids
            or state.balance_lower_test_generations < minimum_tests
            or state.balance_upper_test_generations < minimum_tests
        ):
            return
        observed = self._append_semantic_atomic(
            EventKind.BALANCE_RANGE_OBSERVED,
            observed_at,
            Timeframe.H1,
            None,
            state.midpoint,
            state.strength,
            (
                anchor_event_id,
                *((bar_event_id,) if bar_event_id else ()),
            ),
            {
                "range_id": state.range_id,
                "structural_range_event_id": anchor_event_id,
                # The balance evidence, and beside it the structural touch
                # count it is now decoupled from: the two are different claims
                # and a consumer must be able to tell them apart.
                "balance_lower_test_generations": int(
                    state.balance_lower_test_generations
                ),
                "balance_upper_test_generations": int(
                    state.balance_upper_test_generations
                ),
                "balance_lower_test_kinds": tuple(
                    state.balance_lower_test_kinds
                ),
                "balance_upper_test_kinds": tuple(
                    state.balance_upper_test_kinds
                ),
                "lower_touch_count": int(state.lower_touch_count),
                "upper_touch_count": int(state.upper_touch_count),
                "midpoint_crossings": int(state.midpoint_crossings),
                "inside_close_fraction": float(
                    state.inside_close_fraction
                ),
                "compression_ratio": float(state.compression_ratio),
                "candidate_real_h1_bars": int(
                    state.candidate_real_h1_bars
                ),
                "age_h1_bars": int(state.age_h1_bars),
                "observed_at_lifecycle": state.lifecycle.value,
                "bilateral_price_tests_each_standard": minimum_tests,
            },
            event_time=observed_at,
            zone=(state.lower_bound, state.upper_bound),
            source_entity_ids=(state.range_id,),
            context_event_ids=context_event_ids,
        )
        self._balance_range_observed_event_ids[state.range_id] = (
            observed.event_id
        )

    def _record_delivery_phase_events(
        self,
        transitions: Sequence[DeliveryPhaseTransition],
    ) -> None:
        """Publish one bar's delivery-phase occupancy changes.

        The phase itself is decided by the snapshot publisher; this only turns
        the occupancy it already tracks into immutable facts.  ``structure_regime``
        travels as evidence beside the phase and never as its cause: the two
        remain independent dimensions.
        """

        for transition in transitions:
            evidence = {
                "phase": transition.phase.value,
                "previous_phase": (
                    None
                    if transition.previous_phase is None
                    else transition.previous_phase.value
                ),
                "next_phase": (
                    None
                    if transition.next_phase is None
                    else transition.next_phase.value
                ),
                "entered_at": transition.entered_at.isoformat(),
                "age_bars": int(transition.age_bars),
                "observation_count": int(transition.observation_count),
                "origin_event": transition.origin_event_id,
                "parent_structure_generation": (
                    transition.parent_structure_generation_id
                ),
                "structure_regime": (
                    None
                    if transition.structure_regime is None
                    else transition.structure_regime.value
                ),
                "active_leg_direction": (
                    None
                    if transition.active_leg_direction is None
                    else transition.active_leg_direction.value
                ),
                "protected_swing_intact": transition.protected_swing_intact,
                "range_available": bool(transition.range_available),
            }
            self._append_semantic_atomic(
                transition.kind,
                transition.known_at,
                transition.timeframe,
                None,
                float(transition.price),
                0.0,
                transition.source_event_ids,
                evidence,
                event_time=transition.known_at,
                sequence_floor=EventMemory._DELIVERY_PHASE_SEQUENCE_FLOOR,
            )

    def _record_interaction_events(self, update: InteractionUpdate) -> None:
        path_transitions = update.interaction_path_transitions
        milestone_transitions = update.milestone_transitions

        active = tuple(
            state
            for state in path_transitions
            if state.lifecycle is PathSequenceLifecycle.ACTIVE
        )
        terminal = tuple(
            state
            for state in path_transitions
            if state.lifecycle is not PathSequenceLifecycle.ACTIVE
        )

        def append_path(state: PathSequenceState) -> None:
            self.memory.append(
                _event(
                    EventKind.ENTRY_PATH_STATE,
                    state.state_started_at,
                    Timeframe.M1,
                    (
                        "above"
                        if state.direction is Direction.LONG
                        else "below"
                    ),
                    None,
                    0.0,
                    tuple(step.step_id for step in state.steps),
                    {
                        "protocol_hash": state.protocol_hash,
                        "context_kind": state.context_kind,
                        "context_id": state.context_id,
                        "age_real_1m_bars": state.age_real_1m_bars,
                        "state_duration_real_1m_bars": (
                            state.state_duration_real_1m_bars
                        ),
                        "step_count": len(state.steps),
                    },
                    entity_id=state.sequence_id,
                    lifecycle=state.lifecycle.value,
                    formed_at=state.formed_at,
                    confirmed_at=state.formed_at,
                    ended_at=state.ended_at,
                    direction=state.direction,
                    transition_reason=state.transition_reason,
                ),
                include_in_recent=False,
            )

        for state in sorted(
            active,
            key=lambda item: (item.formed_at, item.sequence_id),
        ):
            append_path(state)
        for sequence_id, step in milestone_transitions:
            source_ids = (
                sequence_id,
                step.step_id,
                step.source_entity_id,
                *(
                    ()
                    if step.source_event_id is None
                    else (step.source_event_id,)
                ),
                *step.predecessor_step_ids,
            )
            self.memory.append(
                _event(
                    EventKind.ENTRY_PATH_STEP,
                    step.observed_at,
                    Timeframe.M1,
                    (
                        "above"
                        if step.direction is Direction.LONG
                        else "below"
                    ),
                    None,
                    step.strength,
                    source_ids,
                    {
                        "sequence_id": sequence_id,
                        "step_id": step.step_id,
                        "kind": step.kind,
                        "source_event_id": step.source_event_id,
                        "source_entity_id": step.source_entity_id,
                        "predecessor_step_ids": (
                            step.predecessor_step_ids
                        ),
                        "same_clock_relation": (
                            step.same_clock_relation
                        ),
                        "reason": step.reason,
                    },
                    direction=step.direction,
                    transition_reason=step.reason,
                )
            )
        for state in sorted(
            terminal,
            key=lambda item: (
                item.ended_at,
                item.sequence_id,
            ),
        ):
            append_path(state)

    @staticmethod
    def _pool_close_outside(
        item: LiquidityInventoryItem,
        candle: Candle,
    ) -> bool:
        return (
            candle.close > item.upper_bound
            if item.side == "above"
            else candle.close < item.lower_bound
        )

    def _append_inventory_crossing_event(
        self,
        item: LiquidityInventoryItem,
        candle: Candle,
        *,
        atr: float,
        defer_resolution: bool = False,
    ) -> None:
        if self.config.range_auction_projection_only:
            return
        semantic_source_kind = (
            "confirmed_swing" if item.kind == "swing" else item.kind
        )
        outside = self._pool_close_outside(item, candle)
        extreme = candle.high if item.side == "above" else candle.low
        distance = (
            extreme - item.upper_bound
            if item.side == "above"
            else item.lower_bound - extreme
        )
        try:
            bar_event_id = self._bar_event_id_at(Timeframe.M1, candle.end)
        except ValueError:
            # Direct compatibility projection callers may intentionally omit
            # the normalized event stream.  Preserve their legacy lifecycle
            # transport, but never manufacture a canonical semantic fact
            # without a BAR_COMPLETED root.
            self.memory.append(
                _event(
                    (
                        EventKind.LIQUIDITY_CONSUMED
                        if outside
                        else EventKind.LIQUIDITY_SWEEP
                    ),
                    candle.end,
                    Timeframe.M1,
                    item.side,
                    extreme,
                    clamp(distance / max(atr, self.config.tick_size)),
                    (item.item_id,),
                    {
                        "source_timeframe": item.timeframe.value,
                        "source_kind": item.kind,
                        "close_accepted_outside": outside,
                        "frozen_lower_bound": item.lower_bound,
                        "frozen_upper_bound": item.upper_bound,
                    },
                )
            )
            return
        candidate_event_id = self._candidate_level_event_ids.get(
            item.item_id
        )
        if candidate_event_id is None:
            raise ValueError(
                "inventory crossing lacks its first-visible canonical "
                "liquidity-level admission"
            )
        touch_identity = f"{item.item_id}|{candle.end.isoformat()}"
        touch_event_id = self._level_touch_event_ids.get(
            (item.item_id, pd.Timestamp(candle.end))
        )
        if self._remember_bounded(
            touch_identity,
            known=self._known_level_touch_ids,
            order=self._known_level_touch_order,
        ):
            touch_event = self._append_semantic_atomic(
                EventKind.LEVEL_TOUCHED,
                candle.end,
                Timeframe.M1,
                item.side,
                item.price,
                item.strength,
                (candidate_event_id, bar_event_id),
                {
                    "level_id": item.item_id,
                    "source_timeframe": item.timeframe.value,
                    "source_kind": semantic_source_kind,
                    "source_inventory_kind": item.kind,
                },
                event_time=candle.end,
                zone=(item.lower_bound, item.upper_bound),
            )
            touch_event_id = touch_event.event_id
            self._level_touch_event_ids[
                (item.item_id, pd.Timestamp(candle.end))
            ] = touch_event_id
        if touch_event_id is None:
            raise ValueError(
                "level penetration lacks its exact touch event"
            )
        crossing_generation_id = self._crossing_generation_id(
            level_id=item.item_id,
            timeframe=Timeframe.M1,
            crossed_at=candle.end,
        )
        penetration = self._append_semantic_atomic(
            EventKind.LEVEL_PENETRATED,
            candle.end,
            Timeframe.M1,
            item.side,
            extreme,
            clamp(distance / max(atr, self.config.tick_size)),
            (candidate_event_id, touch_event_id, bar_event_id),
            {
                "level_id": item.item_id,
                "source_timeframe": item.timeframe.value,
                "source_kind": semantic_source_kind,
                "source_inventory_kind": item.kind,
                "penetration_points": max(0.0, distance),
                "close_accepted_outside": outside,
                "frozen_lower_bound": item.lower_bound,
                "frozen_upper_bound": item.upper_bound,
                "penetration_standard": (
                    "intrabar_trade_beyond_frozen_candidate_level"
                ),
                "strict_close_beyond_confirmed_swing_price": bool(
                    item.kind == "swing" and outside
                ),
                "crossing_generation_id": crossing_generation_id,
                "crossed_at": candle.end.isoformat(),
            },
            direction=(
                Direction.LONG
                if item.side == "above"
                else Direction.SHORT
            ),
            event_time=candle.end,
            zone=(item.lower_bound, item.upper_bound),
        )
        self._penetration_event_ids[
            self._penetration_key(
                level_id=item.item_id,
                timeframe=Timeframe.M1,
                crossed_at=candle.end,
            )
        ] = penetration.event_id
        self.memory.append(
            _event(
                (
                    EventKind.LIQUIDITY_CONSUMED
                    if outside
                    else EventKind.LIQUIDITY_SWEEP
                ),
                candle.end,
                Timeframe.M1,
                item.side,
                extreme,
                clamp(distance / max(atr, self.config.tick_size)),
                (item.item_id,),
                {
                    "source_timeframe": item.timeframe.value,
                    "source_kind": item.kind,
                    "close_accepted_outside": outside,
                    "frozen_lower_bound": item.lower_bound,
                    "frozen_upper_bound": item.upper_bound,
                },
            )
        )
        if not outside and not defer_resolution:
            self._append_crossing_resolution(
                EventKind.SWEEP_CONFIRMED,
                candle.end,
                Timeframe.M1,
                item.side,
                extreme,
                clamp(distance / max(atr, self.config.tick_size)),
                (penetration.event_id, bar_event_id),
                {
                    "level_id": item.item_id,
                    "resolution_bars": 0,
                    "resolution": "same_bar_close_returned_inside",
                    "penetration_points": max(0.0, distance),
                },
                direction=(
                    Direction.SHORT
                    if item.side == "above"
                    else Direction.LONG
                ),
                crossed_at=candle.end,
                zone=(item.lower_bound, item.upper_bound),
                known_at=candle.end,
            )

    def _append_projected_pool_sweep_events(
        self,
        item: LiquidityInventoryItem,
        candle: Candle,
        *,
        atr: float,
    ) -> None:
        if self.config.range_auction_projection_only:
            return
        self._append_inventory_crossing_event(
            item,
            candle,
            atr=atr,
            defer_resolution=True,
        )
        outside = self._pool_close_outside(item, candle)
        extreme = candle.high if item.side == "above" else candle.low
        self.memory.append(
            _event(
                EventKind.LIQUIDITY_POOL_STATE,
                candle.end,
                item.timeframe,
                item.side,
                extreme,
                item.strength,
                item.source_ids,
                {
                    "source_timeframe": item.timeframe.value,
                    "pool_item_id": item.item_id,
                    "close_outside_on_sweep": outside,
                },
                entity_id=item.item_id,
                lifecycle=LiquidityPoolLifecycle.SWEPT.value,
                formed_at=item.formed_at,
                confirmed_at=item.confirmed_at,
            )
        )

    def _append_projected_pool_resolution_event(
        self,
        item: LiquidityInventoryItem,
        candle: Candle,
        *,
        crossed_at: pd.Timestamp | None = None,
        range_auction_tracker: CausalRangeAuctionTracker | None,
    ) -> None:
        if self.config.range_auction_projection_only:
            return
        outside = self._pool_close_outside(item, candle)
        resolution_state_event = _event(
                EventKind.LIQUIDITY_POOL_STATE,
                candle.end,
                item.timeframe,
                item.side,
                candle.close,
                item.strength,
                item.source_ids,
                {
                    "source_timeframe": item.timeframe.value,
                    "pool_item_id": item.item_id,
                },
                entity_id=item.item_id,
                lifecycle=(
                    LiquidityPoolLifecycle.ACCEPTED.value
                    if outside
                    else LiquidityPoolLifecycle.REJECTED.value
                ),
                formed_at=item.formed_at,
                confirmed_at=item.confirmed_at,
                ended_at=candle.end,
                transition_reason=(
                    "close_held_outside"
                    if outside
                    else "close_returned_inside"
                ),
            )
        self.memory.append(resolution_state_event)
        try:
            resolution_bar_event_id = self._bar_event_id_at(
                Timeframe.M1,
                candle.end,
            )
        except ValueError:
            return
        penetration_event_id = (
            None
            if crossed_at is None
            else self._penetration_event_ids.get(
                self._penetration_key(
                    level_id=item.item_id,
                    timeframe=Timeframe.M1,
                    crossed_at=crossed_at,
                )
            )
        )
        if range_auction_tracker is not None:
            # A source admitted by registered Group 4 is resolved solely by
            # that manipulation protocol.  Rejected/unadmitted pool sources
            # still need the generic crossing terminal; otherwise a published
            # penetration would remain permanently unresolved.
            group4_claims_source = any(
                state.source_inventory_item_id == item.item_id
                for state in range_auction_tracker.snapshot().manipulations
            )
            if group4_claims_source:
                return
        if penetration_event_id is None:
            raise ValueError(
                "projected pool resolution lacks its canonical penetration"
            )
        if crossed_at is None:
            raise ValueError(
                "canonical projected pool resolution requires its original "
                "crossing clock"
            )
        self._append_crossing_resolution(
            (
                EventKind.ACCEPTANCE_CONFIRMED
                if outside
                else EventKind.SWEEP_CONFIRMED
            ),
            candle.end,
            Timeframe.M1,
            item.side,
            candle.close,
            item.strength,
            (penetration_event_id, resolution_bar_event_id),
            {
                "level_id": item.item_id,
                "source_timeframe": item.timeframe.value,
                "resolution_bars": 1,
                "resolution": (
                    "later_close_held_outside"
                    if outside
                    else "later_close_returned_inside"
                ),
            },
            direction=(
                (
                    Direction.LONG
                    if item.side == "above"
                    else Direction.SHORT
                )
                if outside
                else (
                    Direction.SHORT
                    if item.side == "above"
                    else Direction.LONG
                )
            ),
            crossed_at=crossed_at,
            zone=(item.lower_bound, item.upper_bound),
            context_event_ids=(resolution_state_event.event_id,),
            known_at=candle.end,
        )

    def _append_level_resolution_event(
        self,
        item: LiquidityInventoryItem,
        candle: Candle,
        *,
        crossed_at: pd.Timestamp,
    ) -> None:
        """Resolve one non-pool penetration on the next completed real bar."""

        outside = self._pool_close_outside(item, candle)
        try:
            resolution_bar_event_id = self._bar_event_id_at(
                Timeframe.M1,
                candle.end,
            )
        except ValueError:
            return
        penetration_event_id = self._penetration_event_ids.get(
            self._penetration_key(
                level_id=item.item_id,
                timeframe=Timeframe.M1,
                crossed_at=crossed_at,
            )
        )
        if penetration_event_id is None:
            raise ValueError(
                "level resolution lacks its exact crossing penetration"
            )
        self._append_crossing_resolution(
            (
                EventKind.ACCEPTANCE_CONFIRMED
                if outside
                else EventKind.SWEEP_CONFIRMED
            ),
            candle.end,
            Timeframe.M1,
            item.side,
            float(candle.close),
            item.strength,
            (penetration_event_id, resolution_bar_event_id),
            {
                "level_id": item.item_id,
                "source_timeframe": item.timeframe.value,
                "source_kind": item.kind,
                "resolution_bars": 1,
                "resolution": (
                    "later_close_held_outside"
                    if outside
                    else "later_close_returned_inside"
                ),
                "crossed_at": crossed_at.isoformat(),
            },
            direction=(
                (
                    Direction.LONG
                    if item.side == "above"
                    else Direction.SHORT
                )
                if outside
                else (
                    Direction.SHORT
                    if item.side == "above"
                    else Direction.LONG
                )
            ),
            crossed_at=crossed_at,
            zone=(item.lower_bound, item.upper_bound),
            known_at=candle.end,
        )


__all__ = ["SemanticEventEmitter"]
