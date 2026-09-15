"""Bounded causal event memory for the Trading Eye.

This is the Eye's short-term working set: a bounded window of recently
appended :class:`~contract.eye.observation.MarketEvent` objects plus the derived
indexes the observer needs to reason about the current clock — latest event per
entity, per-entity lifecycle timelines and their registered legal transitions,
closed durations, same-clock sequence counts, and synthetic-minute runs.

It is deliberately **not** an event authority.  Complete history belongs to
:class:`~eyes.core.event_store.EventStore`; this module forwards every
admitted event there and keeps only the bounded view the observer reads back.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import replace
from typing import Iterable, Mapping

import pandas as pd

from .event_store import EventStore, event_order_key
from contract.market import (
    Candle,
    Timeframe,
)
from contract.eye import (
    BOSLifecycle,
    DealingRangeLifecycle,
    FairValueGapLifecycle,
    LiquidityPoolLifecycle,
    ManipulationLifecycle,
    MarketEvent,
    OrderBlockLifecycle,
    PathSequenceLifecycle,
    StructureLifecycle,
    SupportResistanceLifecycle,
    SwingLifecycle,
    typed_event_entity_key,
)


class EventMemory:
    _GROUP4_CREATION_SEQUENCE_FLOOR = 1_000_000
    # The delivery-phase lifecycle is derived from the published
    # snapshot, so it is appended after every other fact and after the
    # projection transport of the same clock.  Its floor keeps canonical
    # order equal to that append order.
    _DELIVERY_PHASE_SEQUENCE_FLOOR = 3_000_000
    # Candidate retirements are derived from the published snapshot too and
    # appended after the delivery-phase lifecycle of the same clock.
    _CANDIDATE_RETIREMENT_SEQUENCE_FLOOR = 4_000_000
    _TIMELINE_TRANSITIONS: Mapping[
        str,
        Mapping[str, frozenset[str]],
    ] = {
        "swing": {
            SwingLifecycle.FORMING.value: frozenset(
                {
                    SwingLifecycle.CONFIRMED.value,
                    SwingLifecycle.FORMATION_FAILED.value,
                }
            ),
            SwingLifecycle.CONFIRMED.value: frozenset(
                {SwingLifecycle.BROKEN.value}
            ),
            SwingLifecycle.BROKEN.value: frozenset(),
            SwingLifecycle.FORMATION_FAILED.value: frozenset(),
        },
        "structure": {
            StructureLifecycle.FORMING.value: frozenset(
                {
                    StructureLifecycle.CONFIRMED.value,
                    StructureLifecycle.FORMATION_FAILED.value,
                }
            ),
            StructureLifecycle.CONFIRMED.value: frozenset(
                {StructureLifecycle.BROKEN.value}
            ),
            StructureLifecycle.BROKEN.value: frozenset(),
            StructureLifecycle.FORMATION_FAILED.value: frozenset(),
        },
        "bos": {
            BOSLifecycle.CONFIRMED.value: frozenset(),
            BOSLifecycle.FAILED.value: frozenset(),
        },
        "zone": {
            SupportResistanceLifecycle.ACTIVE.value: frozenset(
                {
                    SupportResistanceLifecycle.TESTED.value,
                    SupportResistanceLifecycle.BROKEN.value,
                    SupportResistanceLifecycle.RETIRED.value,
                }
            ),
            SupportResistanceLifecycle.TESTED.value: frozenset(
                {
                    SupportResistanceLifecycle.BROKEN.value,
                    SupportResistanceLifecycle.RETIRED.value,
                }
            ),
            SupportResistanceLifecycle.BROKEN.value: frozenset(
                {
                    SupportResistanceLifecycle.REACCEPTED.value,
                    SupportResistanceLifecycle.RETIRED.value,
                }
            ),
            SupportResistanceLifecycle.REACCEPTED.value: frozenset(),
            SupportResistanceLifecycle.RETIRED.value: frozenset(),
        },
        "pool": {
            LiquidityPoolLifecycle.FORMED.value: frozenset(
                {LiquidityPoolLifecycle.SWEPT.value}
            ),
            LiquidityPoolLifecycle.SWEPT.value: frozenset(
                {
                    LiquidityPoolLifecycle.ACCEPTED.value,
                    LiquidityPoolLifecycle.REJECTED.value,
                }
            ),
            LiquidityPoolLifecycle.ACCEPTED.value: frozenset(),
            LiquidityPoolLifecycle.REJECTED.value: frozenset(),
        },
        "fvg": {
            FairValueGapLifecycle.OPEN.value: frozenset(
                {
                    FairValueGapLifecycle.PARTIAL.value,
                    FairValueGapLifecycle.MITIGATED.value,
                    FairValueGapLifecycle.INVALIDATED.value,
                    FairValueGapLifecycle.EXPIRED.value,
                }
            ),
            FairValueGapLifecycle.PARTIAL.value: frozenset(
                {
                    FairValueGapLifecycle.MITIGATED.value,
                    FairValueGapLifecycle.INVALIDATED.value,
                    FairValueGapLifecycle.EXPIRED.value,
                }
            ),
            FairValueGapLifecycle.MITIGATED.value: frozenset(),
            FairValueGapLifecycle.INVALIDATED.value: frozenset(),
            FairValueGapLifecycle.EXPIRED.value: frozenset(),
        },
        "order_block": {
            OrderBlockLifecycle.CREATED.value: frozenset(
                {
                    OrderBlockLifecycle.UNTESTED.value,
                    OrderBlockLifecycle.MITIGATED.value,
                    OrderBlockLifecycle.FAILED.value,
                }
            ),
            OrderBlockLifecycle.UNTESTED.value: frozenset(
                {
                    OrderBlockLifecycle.MITIGATED.value,
                    OrderBlockLifecycle.FAILED.value,
                }
            ),
            OrderBlockLifecycle.MITIGATED.value: frozenset(),
            OrderBlockLifecycle.FAILED.value: frozenset(),
        },
        "range": {
            DealingRangeLifecycle.ACTIVE.value: frozenset(
                {
                    DealingRangeLifecycle.BROKEN.value,
                }
            ),
            DealingRangeLifecycle.BROKEN.value: frozenset(),
        },
        "manipulation": {
            ManipulationLifecycle.SWEPT.value: frozenset(
                {
                    ManipulationLifecycle.REACCEPTED.value,
                    ManipulationLifecycle.ACCEPTED_OUTSIDE.value,
                    "censored",
                }
            ),
            ManipulationLifecycle.REACCEPTED.value: frozenset(),
            ManipulationLifecycle.ACCEPTED_OUTSIDE.value: frozenset(),
            "censored": frozenset(),
        },
        "entry_path": {
            PathSequenceLifecycle.ACTIVE.value: frozenset(
                {
                    PathSequenceLifecycle.CLOSED.value,
                    PathSequenceLifecycle.CENSORED.value,
                }
            ),
            PathSequenceLifecycle.CLOSED.value: frozenset(),
            PathSequenceLifecycle.CENSORED.value: frozenset(),
        },
    }
    _TIMELINE_LIMITS: Mapping[str, int] = {
        "swing": 3,
        "structure": 3,
        "bos": 1,
        "zone": 5,
        "pool": 3,
        "fvg": 4,
        "order_block": 4,
        "range": 3,
        "manipulation": 2,
        "entry_path": 2,
    }
    _COMPLETE_INITIAL_LIFECYCLES: Mapping[str, frozenset[str]] = {
        "swing": frozenset({SwingLifecycle.FORMING.value}),
        "structure": frozenset({StructureLifecycle.FORMING.value}),
        "bos": frozenset(
            {BOSLifecycle.CONFIRMED.value, BOSLifecycle.FAILED.value}
        ),
        "zone": frozenset(
            {SupportResistanceLifecycle.ACTIVE.value}
        ),
        "pool": frozenset({LiquidityPoolLifecycle.FORMED.value}),
        "fvg": frozenset({FairValueGapLifecycle.OPEN.value}),
        "order_block": frozenset(
            {OrderBlockLifecycle.CREATED.value}
        ),
        "range": frozenset({DealingRangeLifecycle.ACTIVE.value}),
        "manipulation": frozenset(
            {ManipulationLifecycle.SWEPT.value}
        ),
        "entry_path": frozenset(
            {PathSequenceLifecycle.ACTIVE.value}
        ),
    }

    def __init__(
        self,
        maximum_events: int,
        *,
        audit_store: EventStore | None = None,
    ) -> None:
        if type(maximum_events) is not int or maximum_events < 1:
            raise ValueError(
                "event memory maximum_events must be a positive integer"
            )
        self._events: deque[MarketEvent] = deque(maxlen=maximum_events)
        self._ids: set[str] = set()
        self._latest_by_entity: dict[str, MarketEvent] = {}
        self._closed_durations: dict[str, int] = {}
        self._sequence_counts: dict[pd.Timestamp, int] = {}
        self._synthetic_run_starts_ns: list[int] = []
        self._synthetic_runs: list[tuple[int, int, int]] = []
        self._last_minute_end: pd.Timestamp | None = None
        self._last_minute_real_completed: bool | None = None
        self._clock_coverage_start: pd.Timestamp | None = None
        self._entity_timelines: dict[str, list[MarketEvent]] = {}
        self._retained_entity_keys: set[str] = set()
        self._incomplete_entity_keys: set[str] = set()
        self._boundary_cooling_entity_keys: set[str] = set()
        # Sequence-clock retention is synchronized once after the completed
        # observation has projected every same-clock event.  Keeping only a
        # pending bit here avoids rescanning every retained lifecycle timeline
        # after each individual append when the clock table is above its
        # bounded cleanup threshold.
        self._sequence_counts_prune_pending = False
        self._semantic_version: str | None = None
        self._audit_store = audit_store
        self._audit_pending: list[MarketEvent] = []

    @property
    def last_minute_end(self) -> pd.Timestamp | None:
        return self._last_minute_end

    @property
    def clock_coverage_start(self) -> pd.Timestamp | None:
        return self._clock_coverage_start

    @property
    def semantic_version(self) -> str | None:
        return self._semantic_version

    @staticmethod
    def _required_event_clock(event: MarketEvent) -> pd.Timestamp:
        age_origin = (
            event.formed_at
            or event.confirmed_at
            or event.event_time
        )
        return min(age_origin, event.known_at)

    def set_clock_coverage_start(
        self,
        value: pd.Timestamp,
    ) -> None:
        start = pd.Timestamp(value)
        if start.tzinfo is None:
            raise ValueError(
                "event memory clock coverage must be timezone aware"
            )
        if (
            self._clock_coverage_start is not None
            and start != self._clock_coverage_start
        ):
            raise ValueError(
                "event memory clock coverage cannot be rewritten"
            )
        retained_events = {
            event.event_id: event
            for timeline in self._entity_timelines.values()
            for event in timeline
        }
        retained_events.update(
            (event.event_id, event)
            for event in self._events
        )
        if any(
            self._required_event_clock(event) < start
            for event in retained_events.values()
        ):
            raise ValueError(
                "event origin predates retained 1m clock coverage"
            )
        self._clock_coverage_start = start

    def observe_minute(self, candle: Candle) -> None:
        """Advance market time while excluding synthetic minutes from age."""

        if (
            candle.timeframe is not Timeframe.M1
            or not candle.complete
        ):
            raise ValueError(
                "event memory clock requires a completed 1m candle"
            )
        if (
            self._last_minute_end is not None
            and candle.end < self._last_minute_end
        ):
            raise ValueError(
                "event memory received an out-of-order minute"
            )
        if candle.end == self._last_minute_end:
            if (
                self._last_minute_real_completed
                is not bool(candle.real_completed)
            ):
                raise ValueError(
                    "event memory minute provenance changed on retry"
                )
            return
        self._last_minute_end = candle.end
        self._last_minute_real_completed = bool(candle.real_completed)
        if candle.real_completed:
            return
        minute_ns = int(pd.Timedelta(minutes=1).value)
        end_ns = int(candle.end.value)
        if (
            self._synthetic_runs
            and end_ns == self._synthetic_runs[-1][1] + minute_ns
        ):
            start_ns, _, cumulative = self._synthetic_runs[-1]
            self._synthetic_runs[-1] = (
                start_ns,
                end_ns,
                cumulative + 1,
            )
            return
        prior_total = (
            0
            if not self._synthetic_runs
            else self._synthetic_runs[-1][2]
        )
        self._synthetic_run_starts_ns.append(end_ns)
        self._synthetic_runs.append(
            (end_ns, end_ns, prior_total + 1)
        )

    def _synthetic_count_through(self, timestamp: pd.Timestamp) -> int:
        if not self._synthetic_runs:
            return 0
        value = int(timestamp.value)
        index = bisect_right(
            self._synthetic_run_starts_ns,
            value,
        ) - 1
        if index < 0:
            return 0
        start_ns, end_ns, cumulative = self._synthetic_runs[index]
        prior_total = (
            0
            if index == 0
            else self._synthetic_runs[index - 1][2]
        )
        if value >= end_ns:
            return cumulative
        minute_ns = int(pd.Timedelta(minutes=1).value)
        return prior_total + max(
            0,
            int((value - start_ns) // minute_ns) + 1,
        )

    def _elapsed_minutes(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> int:
        wall_minutes = max(
            0,
            int((end - start).total_seconds() // 60),
        )
        synthetic_minutes = (
            self._synthetic_count_through(end)
            - self._synthetic_count_through(start)
        )
        return max(0, wall_minutes - synthetic_minutes)

    @staticmethod
    def _normalized_event(event: MarketEvent) -> MarketEvent:
        return replace(event, sequence_no=0)

    def _existing_event(
        self,
        event_id: str,
        entity_key: str | None,
    ) -> MarketEvent | None:
        if entity_key is not None:
            existing = next(
                (
                    event
                    for event in self._entity_timelines.get(
                        entity_key,
                        (),
                    )
                    if event.event_id == event_id
                ),
                None,
            )
            if existing is not None:
                return existing
        if event_id not in self._ids:
            return None
        return next(
            (
                event
                for event in self._events
                if event.event_id == event_id
            ),
            None,
        )

    @classmethod
    def _timeline_namespace(cls, entity_key: str) -> str:
        namespace, separator, identity = entity_key.partition(":")
        if (
            separator != ":"
            or not identity
            or namespace not in cls._TIMELINE_TRANSITIONS
        ):
            raise ValueError(
                "retained entity key has no registered namespace"
            )
        return namespace

    def _validate_timeline_append(
        self,
        entity_key: str,
        event: MarketEvent,
    ) -> None:
        namespace = self._timeline_namespace(entity_key)
        transitions = self._TIMELINE_TRANSITIONS[namespace]
        if event.lifecycle not in transitions:
            raise ValueError(
                "typed market event has an unregistered lifecycle"
            )
        timeline = self._entity_timelines.get(entity_key, ())
        if any(
            previous.lifecycle == event.lifecycle
            for previous in timeline
        ):
            raise ValueError(
                "typed entity lifecycle cannot be recorded twice"
            )
        if not timeline:
            return
        previous = timeline[-1]
        if event.observed_at <= previous.observed_at:
            raise ValueError(
                "typed entity lifecycle event is out of order"
            )
        if event.lifecycle not in transitions[previous.lifecycle]:
            raise ValueError(
                "typed entity lifecycle transition is not registered"
            )
        if len(timeline) >= self._TIMELINE_LIMITS[namespace]:
            raise ValueError(
                "typed entity lifecycle exceeds its bounded timeline"
            )

    @classmethod
    def _timeline_starts_incomplete(
        cls,
        entity_key: str,
        event: MarketEvent,
    ) -> bool:
        namespace = cls._timeline_namespace(entity_key)
        if event.lifecycle in cls._COMPLETE_INITIAL_LIFECYCLES[
            namespace
        ]:
            return False
        return not (
            namespace == "structure"
            and event.lifecycle
            == StructureLifecycle.CONFIRMED.value
            and event.formed_at == event.observed_at
        )

    def _retain_recent_event(self, event: MarketEvent) -> None:
        """Retain one event in the bounded hot view without duplicating it."""

        if event.event_id in self._ids:
            return
        if len(self._events) == self._events.maxlen and self._events:
            removed = self._events[0]
            self._ids.discard(removed.event_id)
            if removed.entity_id is None:
                self._closed_durations.pop(
                    removed.event_id,
                    None,
                )
            if removed.entity_id is not None:
                latest_entity = self._latest_by_entity.get(
                    removed.entity_id
                )
                if (
                    latest_entity is not None
                    and latest_entity.event_id == removed.event_id
                ):
                    self._latest_by_entity.pop(
                        removed.entity_id,
                        None,
                    )
        self._events.append(event)
        self._ids.add(event.event_id)

    def append(
        self,
        event: MarketEvent,
        *,
        include_in_recent: bool = True,
        sequence_floor: int | None = None,
        audit: bool = True,
    ) -> MarketEvent:
        if type(include_in_recent) is not bool:
            raise ValueError(
                "event-memory recent inclusion flag must be boolean"
            )
        if (
            self._semantic_version is not None
            and event.semantic_version != self._semantic_version
        ):
            raise ValueError(
                "event memory cannot mix semantic versions"
            )
        if (
            sequence_floor is not None
            and (
                type(sequence_floor) is not int
                or sequence_floor < 0
            )
        ):
            raise ValueError(
                "event-memory sequence floor must be a non-negative integer"
            )
        if (
            self._clock_coverage_start is not None
            and self._required_event_clock(event)
            < self._clock_coverage_start
        ):
            raise ValueError(
                "event origin predates retained 1m clock coverage"
            )
        entity_key = typed_event_entity_key(event)
        existing = self._existing_event(
            event.event_id,
            entity_key,
        )
        if existing is not None:
            if (
                self._normalized_event(existing)
                != self._normalized_event(event)
            ):
                raise ValueError(
                    "market event id conflicts with retained payload"
                )
            return existing
        if entity_key is not None:
            self._validate_timeline_append(entity_key, event)
        if self._semantic_version is None:
            self._semantic_version = event.semantic_version
        starts_incomplete = bool(
            entity_key is not None
            and entity_key not in self._entity_timelines
            and self._timeline_starts_incomplete(
                entity_key,
                event,
            )
        )
        sequence_no = self._sequence_counts.get(event.observed_at, 0)
        self._sequence_counts[event.observed_at] = sequence_no + 1
        if sequence_floor is not None:
            sequence_no += sequence_floor
        event = replace(event, sequence_no=sequence_no)
        if audit and self._audit_store is not None:
            self._audit_pending.append(event)
        if include_in_recent:
            self._retain_recent_event(event)
        if entity_key is not None:
            self._entity_timelines.setdefault(entity_key, []).append(
                event
            )
            if starts_incomplete:
                self._incomplete_entity_keys.add(entity_key)
        if event.entity_id is not None:
            previous = self._latest_by_entity.get(event.entity_id)
            if previous is not None:
                self._closed_durations[previous.event_id] = max(
                    0,
                    self._elapsed_minutes(
                        previous.observed_at,
                        event.observed_at,
                    ),
                )
            if event.ended_at is None:
                self._latest_by_entity[event.entity_id] = event
            else:
                self._latest_by_entity.pop(event.entity_id, None)
                self._closed_durations[event.event_id] = 0
        if len(self._sequence_counts) > self._events.maxlen * 2:
            self._sequence_counts_prune_pending = True
        return event

    def flush_audit(self) -> int:
        """Atomically append this update's events in canonical availability order."""

        if self._audit_store is None or not self._audit_pending:
            self._audit_pending.clear()
            return 0
        ordered = tuple(sorted(self._audit_pending, key=event_order_key))
        appended = self._audit_store.append_batch(ordered)
        self._audit_pending.clear()
        return appended

    def audit_event_including_pending(self, event_id: str) -> MarketEvent | None:
        """Resolve an exact audit event without flushing the current update."""

        pending = tuple(
            event for event in self._audit_pending if event.event_id == event_id
        )
        if len(pending) > 1:
            raise ValueError("pending audit event identity is duplicated")
        committed = (
            None if self._audit_store is None else self._audit_store.get(event_id)
        )
        if pending and committed is not None and pending[0] != committed:
            raise ValueError("pending audit event conflicts with committed identity")
        return pending[0] if pending else committed

    def transfer_pending_from(self, prior: "EventMemory") -> int:
        """Move uncommitted events while preserving audit and causal timelines.

        Typed live prefixes are already retained solely for a terminal join;
        they deliberately do not re-enter the new epoch's bounded recent view.
        """

        if not isinstance(prior, EventMemory) or prior is self:
            raise TypeError("pending event-memory source is invalid")
        if self._audit_store is not prior._audit_store:
            raise ValueError("pending events cannot change audit store")
        if self._audit_pending:
            raise ValueError("pending events require an empty destination")
        pending = tuple(prior._audit_pending)
        for event in pending:
            entity_key = typed_event_entity_key(event)
            existing = self._existing_event(event.event_id, entity_key)
            if existing is not None:
                if (
                    self._normalized_event(existing)
                    != self._normalized_event(event)
                ):
                    raise ValueError(
                        "pending event conflicts with retained boundary prefix"
                    )
                transferred = existing
            else:
                transferred = self.append(
                    event,
                    include_in_recent=event.event_id in prior._ids,
                    audit=False,
                )
            self._audit_pending.append(transferred)
        prior._audit_pending.clear()
        return len(pending)

    def sync_retained_entity_timelines(
        self,
        entity_keys: Iterable[str],
        *,
        asof: pd.Timestamp,
    ) -> None:
        """Retain complete histories only for current typed snapshot entities."""

        clock = pd.Timestamp(asof)
        if clock.tzinfo is None:
            raise ValueError(
                "retained entity timeline cutoff must be timezone aware"
            )
        retained = set(entity_keys)
        for key in retained:
            self._timeline_namespace(key)
        missing = retained - set(self._entity_timelines)
        if missing:
            raise ValueError(
                "retained typed entity lacks a lifecycle timeline"
            )
        for timeline in self._entity_timelines.values():
            # ``append`` enforces a strictly increasing lifecycle clock, so
            # the tail is the maximum observation time for this entity.
            if timeline and timeline[-1].observed_at > clock:
                raise ValueError(
                    "retained entity timeline contains the future"
                )
        if getattr(self, "_sequence_counts_prune_pending", False):
            # Match the former post-append retention boundary exactly: the
            # final event of this completed update could still see every
            # pre-synchronization timeline.  Timeline membership is narrowed
            # only after the clock table has been pruned against that same
            # view, preserving checkpoint state as well as same-clock counts.
            retained_clocks = {
                event.observed_at
                for event in self._events
            } | {
                event.observed_at
                for timeline in self._entity_timelines.values()
                for event in timeline
            }
            self._sequence_counts = {
                event_clock: count
                for event_clock, count in self._sequence_counts.items()
                if event_clock in retained_clocks
            }
            self._sequence_counts_prune_pending = False
        next_timelines = {
            # Retention changes dictionary membership, never lifecycle list
            # ownership.  Keeping the internal list avoids copying every
            # retained history on each completed minute; public readers still
            # receive immutable tuples from ``entity_timelines``/``timeline``.
            key: self._entity_timelines[key]
            for key in sorted(retained)
        }
        self._entity_timelines = next_timelines
        self._retained_entity_keys = retained
        self._incomplete_entity_keys.intersection_update(retained)
        self._boundary_cooling_entity_keys.intersection_update(retained)
        retained_event_ids = {
            event.event_id
            for event in self._events
        } | {
            event.event_id
            for timeline in next_timelines.values()
            for event in timeline
        }
        self._closed_durations = {
            event_id: duration
            for event_id, duration in self._closed_durations.items()
            if event_id in retained_event_ids
        }
        self._latest_by_entity = {
            entity_id: event
            for entity_id, event in self._latest_by_entity.items()
            if event.event_id in retained_event_ids
        }

    def entity_timelines(
        self,
    ) -> Mapping[str, tuple[MarketEvent, ...]]:
        return {
            key: tuple(self._entity_timelines[key])
            for key in sorted(self._retained_entity_keys)
        }

    def timeline(self, entity_key: str) -> tuple[MarketEvent, ...]:
        if entity_key not in self._retained_entity_keys:
            return ()
        return tuple(self._entity_timelines[entity_key])

    def incomplete_entity_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                self._incomplete_entity_keys
                & self._retained_entity_keys
            )
        )

    def live_entity_keys(self) -> tuple[str, ...]:
        """Return retained lifecycle prefixes which can still transition.

        Reducer snapshots are bounded projections and may transiently omit an
        entity before its later terminal transition is emitted.  The prefix is
        therefore part of the hot causal state until its latest lifecycle has
        no registered successor.  Terminal timelines remain eligible for the
        normal snapshot-driven cooling performed by
        :meth:`sync_retained_entity_timelines`.
        """

        output: list[str] = []
        for entity_key, timeline in self._entity_timelines.items():
            if (
                not timeline
                or entity_key in self._boundary_cooling_entity_keys
            ):
                continue
            namespace = self._timeline_namespace(entity_key)
            if self._TIMELINE_TRANSITIONS[namespace][timeline[-1].lifecycle]:
                output.append(entity_key)
        return tuple(sorted(output))

    def retain_live_prefixes_from(
        self,
        prior: "EventMemory",
        *,
        asof: pd.Timestamp,
    ) -> tuple[str, ...]:
        """Carry only transitionable prefixes across one hard boundary.

        Boundary reducers emit their terminal transitions after the Observer
        has reset contract-local state.  Keeping the old recent deque or every
        terminal timeline would turn EventMemory into an unbounded audit
        archive; copying only live prefixes gives those same-clock terminal
        events their causal history.  The imported keys are excluded from the
        ordinary live-retention union, so a key not exposed by the new typed
        snapshot cools on the next synchronization.
        """

        if not isinstance(prior, EventMemory) or prior is self:
            raise TypeError("boundary EventMemory source is invalid")
        clock = pd.Timestamp(asof)
        if clock.tzinfo is None:
            raise ValueError("boundary prefix cutoff must be timezone aware")
        if self._entity_timelines or self._retained_entity_keys:
            raise ValueError("boundary prefixes require an empty EventMemory")
        keys = prior.live_entity_keys()
        timelines = {
            key: list(prior._entity_timelines[key])
            for key in keys
        }
        versions = {
            event.semantic_version
            for timeline in timelines.values()
            for event in timeline
        }
        if len(versions) > 1 or (
            versions
            and prior.semantic_version not in versions
        ):
            raise ValueError("boundary prefix mixes semantic versions")
        self._semantic_version = prior.semantic_version
        if any(
            event.observed_at > clock
            for timeline in timelines.values()
            for event in timeline
        ):
            raise ValueError("boundary prefix contains a future event")
        if (
            self._clock_coverage_start is not None
            and any(
                self._required_event_clock(event)
                < self._clock_coverage_start
                for timeline in timelines.values()
                for event in timeline
            )
        ):
            raise ValueError(
                "boundary prefix predates retained 1m clock coverage"
            )
        self._entity_timelines = timelines
        self._retained_entity_keys = set(keys)
        self._incomplete_entity_keys = (
            prior._incomplete_entity_keys & set(keys)
        )
        self._boundary_cooling_entity_keys = set(keys)
        retained_event_ids = {
            event.event_id
            for timeline in timelines.values()
            for event in timeline
        }
        self._closed_durations = {
            event_id: duration
            for event_id, duration in prior._closed_durations.items()
            if event_id in retained_event_ids
        }
        self._latest_by_entity = {
            entity_id: event
            for entity_id, event in prior._latest_by_entity.items()
            if event.event_id in retained_event_ids
        }
        retained_clocks = {
            event.observed_at
            for timeline in timelines.values()
            for event in timeline
        }
        self._sequence_counts = {
            event_clock: count
            for event_clock, count in prior._sequence_counts.items()
            if event_clock in retained_clocks
        }
        self._sequence_counts_prune_pending = False
        return keys

    def has_entity_lifecycle(
        self,
        entity_key: str,
        lifecycle: str,
    ) -> bool:
        return any(
            event.lifecycle == lifecycle
            for event in self._entity_timelines.get(entity_key, ())
        )

    def recent(self, limit: int = 64) -> tuple[MarketEvent, ...]:
        return tuple(self._events)[-int(limit) :]

    def temporal_metrics(
        self,
        asof: pd.Timestamp,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Materialize event duration and age in one causal traversal."""

        # A retained lifecycle commonly contributes the same formation clock
        # to several events, while the current ``asof`` is shared by every
        # active duration and age.  Resolve each timestamp onto the existing
        # real-minute clock once per materialization instead of repeatedly
        # constructing Timedelta objects and bisecting synthetic runs.
        #
        # Keep the sub-minute remainder alongside the minute coordinate.  The
        # remainder correction makes this exactly equivalent to
        # ``floor((end - start) / one_minute)`` even for non-aligned aware
        # timestamps; synthetic minutes retain the original (start, end]
        # inclusion convention from ``_synthetic_count_through``.
        minute_ns = 60_000_000_000
        clock_coordinates: dict[int, tuple[int, int, int]] = {}

        def coordinate(
            timestamp: pd.Timestamp,
        ) -> tuple[int, int, int]:
            timestamp_ns = int(timestamp.value)
            cached = clock_coordinates.get(timestamp_ns)
            if cached is not None:
                return cached
            minute, remainder = divmod(timestamp_ns, minute_ns)
            value = (
                minute,
                remainder,
                self._synthetic_count_through(timestamp),
            )
            clock_coordinates[timestamp_ns] = value
            return value

        def elapsed_minutes(
            start: pd.Timestamp,
            end: pd.Timestamp,
        ) -> int:
            start_minute, start_remainder, start_synthetic = coordinate(
                start
            )
            end_minute, end_remainder, end_synthetic = coordinate(end)
            wall_minutes = max(
                0,
                end_minute
                - start_minute
                - int(end_remainder < start_remainder),
            )
            return max(
                0,
                wall_minutes - (end_synthetic - start_synthetic),
            )

        durations: dict[str, int] = {}
        ages: dict[str, int] = {}
        active = {
            event.event_id
            for event in self._latest_by_entity.values()
        }
        events = {
            event.event_id: event
            for event in self._events
        }
        for event in events.values():
            if event.event_id in self._closed_durations:
                durations[event.event_id] = self._closed_durations[
                    event.event_id
                ]
            elif event.event_id in active:
                durations[event.event_id] = max(
                    0,
                    elapsed_minutes(
                        event.formed_at or event.observed_at,
                        asof,
                    ),
                )
            else:
                durations[event.event_id] = 0
            origin = (
                event.formed_at
                or event.confirmed_at
                or event.observed_at
            )
            ages[event.event_id] = max(
                0,
                elapsed_minutes(origin, asof),
            )
        for timeline in self._entity_timelines.values():
            for index, event in enumerate(timeline):
                if index + 1 < len(timeline):
                    end = timeline[index + 1].observed_at
                    durations[event.event_id] = max(
                        0,
                        elapsed_minutes(event.observed_at, end),
                    )
                elif event.ended_at is not None:
                    durations[event.event_id] = 0
                else:
                    durations[event.event_id] = max(
                        0,
                        elapsed_minutes(event.observed_at, asof),
                    )
                origin = (
                    event.formed_at
                    or event.confirmed_at
                    or event.event_time
                )
                ages[event.event_id] = max(
                    0,
                    elapsed_minutes(origin, asof),
                )
        return durations, ages



__all__ = ["EventMemory"]
