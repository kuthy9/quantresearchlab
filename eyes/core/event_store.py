"""Append-only semantic event storage and deterministic causal replay.

The in-memory store is the hot replay authority.  The journal helpers provide
its deliberately small durable counterpart: immutable, bounded Parquet shards
plus a hash-bound manifest.  They reuse :mod:`shares.core.artifact_stream` and
do not introduce a service, database, or second event model.
"""
from __future__ import annotations

from collections import ChainMap, Counter
from dataclasses import dataclass, fields, replace
import hashlib
from itertools import islice
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Generic, Iterable, Mapping, TypeVar

import pandas as pd

from shares.core.artifact_stream import (
    atomic_bytes,
    canonical_json,
    new_stream_state,
    sha256_file,
    verify_stream_shards,
    write_stream_manifest,
    write_stream_shard,
)
from .foundation_registry import FOUNDATION_VERSION
from shares.core.market_clock import validate_registered_native_bar_root
from contract.market import (
    Direction,
    FrozenDict,
    SMC_SEMANTIC_VERSION,
    Timeframe,
    aware_timestamp,
    bar_evidence_coverage,
    price_to_ticks,
    to_primitive,
)
from contract.eye import (
    EventKind,
    EventOrigin,
    MarketEvent,
)
from .semantics import (
    SemanticDefinitionIdentity,
    SemanticRegistryError,
)


StateT = TypeVar("StateT")
Reducer = Callable[[StateT, MarketEvent], StateT]

_SHA256_PATTERN = frozenset("0123456789abcdef")
_EVENT_JOURNAL_FORMAT_VERSION = 1
_EVENT_JOURNAL_ARTIFACT = "smc_semantic_event_journal"
_EVENT_JOURNAL_STREAM = "events"
_EVENT_JOURNAL_FIELD_TYPES: Mapping[str, str] = {
    "ordinal": "int64",
    "event_id": "large_string",
    "known_at": "timestamp_utc",
    "sequence_no": "int64",
    "event_digest": "large_string",
    "event_json": "large_string",
}

# These source-event multisets are the authoritative Phase 2/3 derivation
# contracts.  Context events may explain a fact but can never substitute for
# one of these definitional parents.
_EXACT_AUTHORITATIVE_SOURCE_KINDS: Mapping[
    EventKind,
    tuple[EventKind, ...],
] = {
    EventKind.LEVEL_TOUCHED: (
        EventKind.LIQUIDITY_LEVEL_CREATED,
        EventKind.BAR_COMPLETED,
    ),
    EventKind.LEVEL_PENETRATED: (
        EventKind.LIQUIDITY_LEVEL_CREATED,
        EventKind.LEVEL_TOUCHED,
        EventKind.BAR_COMPLETED,
    ),
    EventKind.SWEEP_CONFIRMED: (
        EventKind.LEVEL_PENETRATED,
        EventKind.BAR_COMPLETED,
    ),
    EventKind.ACCEPTANCE_CONFIRMED: (
        EventKind.LEVEL_PENETRATED,
        EventKind.BAR_COMPLETED,
    ),
    EventKind.STRUCTURAL_LEG_CREATED: (
        EventKind.SWING_CONFIRMED,
        EventKind.SWING_CONFIRMED,
    ),
    EventKind.RAW_BOUNDARY_BREAK: (
        EventKind.SWING_CONFIRMED,
        EventKind.BAR_COMPLETED,
    ),
    EventKind.STRUCTURE_DIRECTION_CONFIRMED: (
        EventKind.SWING_CONFIRMED,
        EventKind.SWING_CONFIRMED,
    ),
    EventKind.QUALIFIED_BOS: (
        EventKind.RAW_BOUNDARY_BREAK,
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
    ),
    EventKind.PROTECTED_SWING_ASSIGNED: (
        EventKind.QUALIFIED_BOS,
        EventKind.STRUCTURAL_LEG_CREATED,
        EventKind.SWING_CONFIRMED,
    ),
    EventKind.MSS_CORE_CONFIRMED: (
        EventKind.RAW_BOUNDARY_BREAK,
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
    ),
    EventKind.FVG_CREATED: (
        EventKind.BAR_COMPLETED,
        EventKind.BAR_COMPLETED,
        EventKind.BAR_COMPLETED,
    ),
}

# A base origin core is geometry: it cites the candles it froze and nothing
# else.  Qualification is a separate, append-only fact that binds that core to
# the displacement and structure break which qualified it, so the core stays
# readable on its own and is never backdated.
_BASE_ORIGIN_CORE_SOURCE_KINDS = frozenset(
    {
        EventKind.BAR_COMPLETED,
    }
)

_ORIGIN_ZONE_SOURCE_KINDS = frozenset(
    {
        EventKind.BASE_ORIGIN_CORE_CREATED,
        EventKind.DISPLACEMENT_OBSERVED,
        EventKind.RAW_BOUNDARY_BREAK,
    }
)

_ORIGIN_ZONE_TERMINAL_SOURCE_KINDS = (
    EventKind.QUALIFIED_ORIGIN_ZONE_CREATED,
    EventKind.BAR_COMPLETED,
)

_RANGE_INVALIDATION_SOURCE_KINDS = (
    EventKind.DEALING_RANGE_CREATED,
    EventKind.BALANCE_RANGE_MATURED,
    EventKind.BAR_COMPLETED,
    EventKind.ACCEPTANCE_CONFIRMED,
)

_FORMING_RANGE_INVALIDATION_SOURCE_KINDS = (
    EventKind.DEALING_RANGE_CREATED,
    EventKind.BAR_COMPLETED,
)

_SWING_WINDOW_SPANS: Mapping[Timeframe, int] = {
    Timeframe.H4: 2,
    Timeframe.H1: 2,
    Timeframe.M15: 2,
    Timeframe.M5: 2,
    Timeframe.M1: 1,
}

_CANDIDATE_LIQUIDITY_SOURCE_KINDS = frozenset(
    {
        "confirmed_swing",
        "structural_swing",
        "formed_liquidity_pool",
        "equal_highs",
        "equal_lows",
        "previous_session",
        "previous_day",
        "previous_week",
        "previous_session_high",
        "previous_session_low",
        "previous_day_high",
        "previous_day_low",
        "previous_week_high",
        "previous_week_low",
        "mature_range_boundary",
        "range_boundary",
    }
)

_CANDIDATE_SWING_PARENT_SOURCE_KINDS = frozenset(
    {
        "confirmed_swing",
        "structural_swing",
        "formed_liquidity_pool",
        "equal_highs",
        "equal_lows",
    }
)

_CANDIDATE_REFERENCE_PARENT_SOURCE_KINDS = frozenset(
    {
        "previous_session_high",
        "previous_session_low",
        "previous_day_high",
        "previous_day_low",
        "previous_week_high",
        "previous_week_low",
    }
)

# Source-free reference-zone candidates exist in persisted v1.2 journals.
# Keep that replay seam pinned to the exact historical semantic version;
# current sourceful candidates take the stricter point-parent branch below.
_LEGACY_SOURCE_FREE_REFERENCE_ZONE_SEMANTIC_VERSIONS = frozenset(
    {"smc_semantics_v1.2"}
)

_DISPLACEMENT_LIFECYCLES = frozenset(
    {"started", "active", "exhausted", "censored"}
)

_DISPLACEMENT_REQUIRED_METRICS = frozenset(
    {
        "age_minutes_at_last_admitted",
        "atr0",
        "efficiency",
        "favorable_extreme",
        "mean_body_fraction",
        "mean_overlap_ratio",
        "max_overlap_ratio",
        "mean_directional_clv",
        "min_directional_clv",
        "nested_seed_observed",
        "net_points",
        "net_ticks",
        "origin_price",
        "real_episode_bar_count",
        "relative_atr",
        "speed_atr_per_bar",
        "travel_points",
        "volume_ready",
        "protection_price",
        "last_favorable_close",
        "interruption_run",
        "total_interruption_bars",
        "directional_bar_count",
        "neutral_bar_count",
        "opposite_bar_count",
        "directional_body_points",
        "opposite_body_points",
        "body_continuity",
        "activation_gate_count",
        "activation_weakest_ratio",
        "activation_episode_bar_count_ratio",
        "activation_relative_atr_ratio",
        "activation_efficiency_ratio",
        "activation_speed_ratio",
        "activation_mean_body_fraction_ratio",
        "activation_body_continuity_ratio",
    }
)


def _require_sha256(value: Any, *, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        character not in _SHA256_PATTERN for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 identity")
    return normalized


def _definition_binding(
    value: SemanticDefinitionIdentity | Mapping[str, Any] | str | None,
    *,
    semantic_version: str,
) -> tuple[SemanticDefinitionIdentity | None, str | None]:
    if value is None:
        return None, None
    try:
        identity = (
            SemanticDefinitionIdentity.from_metadata(value)
            if isinstance(value, Mapping)
            else value
        )
    except SemanticRegistryError as error:
        raise ValueError("semantic definition identity is invalid") from error
    if isinstance(identity, SemanticDefinitionIdentity):
        if identity.semantic_version != semantic_version:
            raise ValueError(
                "semantic definition identity and event store versions differ"
            )
        return identity, identity.identity
    if isinstance(identity, str):
        return None, _require_sha256(
            identity,
            name="semantic definition identity",
        )
    raise TypeError(
        "definition_identity must be SemanticDefinitionIdentity, metadata, "
        "a SHA-256 digest, or None"
    )


def event_order_key(event: MarketEvent) -> tuple[pd.Timestamp, int, str]:
    """Canonical availability order; event_time never controls availability."""

    return event.known_at, event.sequence_no, event.event_id


def _event_digest(event: MarketEvent) -> str:
    # ``sequence_no`` is an EventMemory transport ordinal, not semantic
    # identity. A bounded-memory replay may rediscover the same immutable
    # event after its hot prefix cools and assign a different local ordinal;
    # the append-only audit store must treat that as an idempotent retry.
    digest_event = replace(event, sequence_no=0)
    if event.is_projection:
        projection_sha256 = _require_sha256(
            event.details.get("projection_sha256"),
            name="event projection hash",
        )
        if "projection_state" not in event.details:
            raise ValueError("state projection payload is missing")
        projection_payload = json.dumps(
            to_primitive(event.details["projection_state"]),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if hashlib.sha256(projection_payload).hexdigest() != projection_sha256:
            raise ValueError("state projection hash does not bind its payload")
        # The checked SHA-256 is a content-addressed commitment to the full
        # projection state.  Bind that commitment (plus every other detail)
        # into the event digest instead of serializing the often-large state a
        # second time.  This keeps integrity identical while avoiding two
        # recursive primitive conversions for every technical projection.
        digest_event = replace(
            digest_event,
            details=FrozenDict(
                {
                    key: value
                    for key, value in event.details.items()
                    if key != "projection_state"
                }
                | {"projection_state_sha256": projection_sha256}
            ),
        )
    # Bind every immutable field for both semantic facts and projections.
    # ``sequence_no`` alone is local hot-memory transport and intentionally
    # remains outside identity.
    payload = json.dumps(
        to_primitive(digest_event),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _market_event_schema_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            [item.name for item in fields(MarketEvent)],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


_MARKET_EVENT_FIELD_NAMES = frozenset(
    item.name for item in fields(MarketEvent)
)


def _require_exact_market_event(event: MarketEvent) -> None:
    if (
        type(event) is not MarketEvent
        or set(event.__dict__) != _MARKET_EVENT_FIELD_NAMES
    ):
        raise ValueError("MarketEvent runtime field-set is not exact")


def _event_json(event: MarketEvent) -> str:
    _require_exact_market_event(event)
    return json.dumps(
        to_primitive(event),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _event_from_json(payload: str) -> MarketEvent:
    try:
        raw = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("event journal row contains invalid event JSON") from error
    if not isinstance(raw, Mapping):
        raise ValueError("event journal event payload must be an object")
    expected_fields = {item.name for item in fields(MarketEvent)}
    missing = expected_fields - set(raw)
    extra = set(raw) - expected_fields
    if missing or extra:
        raise ValueError(
            "event journal payload schema changed: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    values = dict(raw)
    try:
        values["kind"] = EventKind(values["kind"])
        values["timeframe"] = Timeframe(values["timeframe"])
        values["direction"] = (
            None
            if values["direction"] is None
            else Direction(values["direction"])
        )
        for name in (
            "source_ids",
            "source_event_ids",
            "source_data_ids",
            "source_entity_ids",
            "context_event_ids",
        ):
            values[name] = tuple(values[name])
        values["zone"] = (
            None if values["zone"] is None else tuple(values["zone"])
        )
        for name in (
            "observed_at",
            "formed_at",
            "confirmed_at",
            "ended_at",
            "event_time",
            "known_at",
        ):
            if values[name] is not None:
                values[name] = pd.Timestamp(values[name])
        return MarketEvent(**values)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("event journal row cannot reconstruct MarketEvent") from error


def _fingerprint_from_digests(
    *,
    semantic_version: str,
    definition_identity: str | None,
    digests: Iterable[str],
) -> str:
    digest = hashlib.sha256()
    # Preserve the original fingerprint for legacy, unbound stores. New
    # definition-bound stores have an explicit domain separator so a semantic
    # version label cannot stand in for the exact preregistered definition.
    if definition_identity is None:
        digest.update(semantic_version.encode("utf-8"))
    else:
        digest.update(b"smc-event-store-definition-bound-v1\0")
        digest.update(semantic_version.encode("utf-8"))
        digest.update(b"\0")
        digest.update(definition_identity.encode("ascii"))
    for event_digest in digests:
        digest.update(event_digest.encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ReplayResult(Generic[StateT]):
    state: StateT
    events_applied: int
    last_known_at: pd.Timestamp | None
    event_fingerprint: str


class EventStore:
    """An append-only in-memory audit log of immutable ``MarketEvent`` values.

    The store exposes no update or delete operation. Re-appending an identical
    event is idempotent; reusing an ID for different content fails closed.
    Events from different semantic versions can never share one store.
    """

    def __init__(
        self,
        *,
        semantic_version: str = SMC_SEMANTIC_VERSION,
        definition_identity: (
            SemanticDefinitionIdentity | Mapping[str, Any] | str | None
        ) = None,
    ) -> None:
        if not isinstance(semantic_version, str) or not semantic_version:
            raise ValueError("event store semantic_version is required")
        self.semantic_version = semantic_version
        (
            self._definition_identity,
            self._definition_identity_digest,
        ) = _definition_binding(
            definition_identity,
            semantic_version=semantic_version,
        )
        self._events: list[MarketEvent] = []
        self._by_id: dict[str, MarketEvent] = {}
        self._digests: dict[str, str] = {}
        self._terminal_crossing_event_ids: dict[str, str] = {}
        self._normalized_bar_event_ids: dict[
            tuple[Timeframe, pd.Timestamp], str
        ] = {}
        self._latest_protected_assignment_event_ids: dict[str, str] = {}
        self._latest_protected_assignment_event_ids_by_timeframe: dict[
            Timeframe, str
        ] = {}
        self._unresolved_forward_reference_ids: set[str] = set()
        self._rebuild_fingerprint_cache()

    def _rebuild_fingerprint_cache(self) -> None:
        """Rebuild the exact full-prefix SHA-256 from canonical event digests."""

        digest = hashlib.sha256()
        if self.semantic_definition_identity is None:
            digest.update(self.semantic_version.encode("utf-8"))
        else:
            digest.update(b"smc-event-store-definition-bound-v1\0")
            digest.update(self.semantic_version.encode("utf-8"))
            digest.update(b"\0")
            digest.update(self.semantic_definition_identity.encode("ascii"))
        for event in self._events:
            digest.update(self._digests[event.event_id].encode("ascii"))
        self._full_prefix_fingerprint_hasher = digest

    @property
    def definition_identity(self) -> SemanticDefinitionIdentity | None:
        """Structured identity when full definition metadata was supplied."""

        return getattr(self, "_definition_identity", None)

    @property
    def semantic_definition_identity(self) -> str | None:
        """Content digest bound into fingerprints and checkpoint metadata."""

        return getattr(self, "_definition_identity_digest", None)

    def require_definition_identity(
        self,
        expected: SemanticDefinitionIdentity | Mapping[str, Any] | str,
    ) -> None:
        _, expected_digest = _definition_binding(
            expected,
            semantic_version=self.semantic_version,
        )
        actual = self.semantic_definition_identity
        if actual is None:
            raise ValueError("event store is not bound to a semantic definition")
        if actual != expected_digest:
            raise ValueError(
                "event store semantic definition identity drifted: "
                f"expected={expected_digest}, actual={actual}"
            )

    def __len__(self) -> int:
        return len(self._events)

    def __eq__(self, other: object) -> bool:
        return bool(
            isinstance(other, EventStore)
            and self.semantic_version == other.semantic_version
            and self.semantic_definition_identity
            == other.semantic_definition_identity
            and self._events == other._events
        )

    def _require_committed_integrity(self) -> None:
        if (
            len(self._events) != len(self._digests)
            or len(self._events) != len(self._by_id)
        ):
            raise ValueError("event store contains mutated committed evidence")
        for event in self._events:
            _require_exact_market_event(event)
            if (
                self._by_id.get(event.event_id) is not event
                or self._digests.get(event.event_id) != _event_digest(event)
            ):
                raise ValueError(
                    "event store contains mutated committed evidence"
                )

    def __getstate__(self) -> dict[str, Any]:
        # Carry canonical evidence exactly once.  All maps and the prefix
        # hasher are derived and are rebuilt/revalidated by ``__setstate__``.
        self._require_committed_integrity()
        return {
            "semantic_version": self.semantic_version,
            "_definition_identity": self._definition_identity,
            "_definition_identity_digest": (
                self._definition_identity_digest
            ),
            "_events": self._events,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Revalidate serialized history and rebuild every derived authority."""

        semantic_version = state.get("semantic_version")
        events = state.get("_events")
        if (
            not isinstance(semantic_version, str)
            or not semantic_version
            or not isinstance(events, (tuple, list))
            or any(not isinstance(event, MarketEvent) for event in events)
        ):
            raise ValueError("event store pickle state is invalid")
        definition_identity = state.get("_definition_identity")
        if definition_identity is None:
            definition_identity = state.get("_definition_identity_digest")
        rebuilt = EventStore(
            semantic_version=semantic_version,
            definition_identity=definition_identity,
        )
        rebuilt.append_batch(events)
        self.__dict__.clear()
        self.__dict__.update(rebuilt.__dict__)

    def append(self, event: MarketEvent) -> bool:
        """Append once, returning false only for an exact idempotent retry."""

        _require_exact_market_event(event)
        if event.semantic_version != self.semantic_version:
            raise ValueError("event store cannot mix semantic versions")
        digest = _event_digest(event)
        previous = self._by_id.get(event.event_id)
        if previous is not None:
            if self._digests[event.event_id] != digest:
                raise ValueError("event id conflicts with immutable history")
            return False
        bar_reservation = self._normalized_bar_identity(
            event,
            bar_event_ids=self._normalized_bar_event_ids,
        )
        self._validate_canonical_provenance(
            event,
            available_events=self._by_id,
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
        terminal_reservation = self._terminal_crossing_identity(
            event,
            terminal_event_ids=self._terminal_crossing_event_ids,
        )
        if self._events and event_order_key(event) < event_order_key(self._events[-1]):
            raise ValueError("event store append is out of known_at order")
        self._events.append(event)
        self._by_id[event.event_id] = event
        self._digests[event.event_id] = digest
        self._full_prefix_fingerprint_hasher.update(digest.encode("ascii"))
        if bar_reservation is not None:
            key, event_id = bar_reservation
            self._normalized_bar_event_ids[key] = event_id
        if terminal_reservation is not None:
            generation_id, event_id = terminal_reservation
            self._terminal_crossing_event_ids[generation_id] = event_id
        protected_reservation = self._protected_assignment_identity(event)
        if protected_reservation is not None:
            protected_id, event_id = protected_reservation
            self._latest_protected_assignment_event_ids[
                protected_id
            ] = event_id
            self._latest_protected_assignment_event_ids_by_timeframe[
                event.timeframe
            ] = event_id
        self._advance_unresolved_forward_references(
            event,
            available_event_ids=self._by_id.keys(),
            unresolved_reference_ids=self._unresolved_forward_reference_ids,
        )
        return True

    def append_batch(self, events: Iterable[MarketEvent]) -> int:
        """Validate a complete batch before committing any of it."""

        incoming = tuple(events)
        staged: list[tuple[MarketEvent, str]] = []
        staged_digests: dict[str, str] = {}
        # Do not clone the complete immutable history for every completed-bar
        # batch.  A small write overlay provides the same all-or-nothing
        # validation view while keeping append cost proportional to this batch
        # instead of the lifetime event count.
        staged_events: dict[str, MarketEvent] = {}
        available_events: Mapping[str, MarketEvent] = ChainMap(
            staged_events,
            self._by_id,
        )
        staged_terminal_event_ids: dict[str, str] = {}
        terminal_event_ids: Mapping[str, str] = ChainMap(
            staged_terminal_event_ids,
            self._terminal_crossing_event_ids,
        )
        staged_bar_event_ids: dict[
            tuple[Timeframe, pd.Timestamp], str
        ] = {}
        normalized_bar_event_ids: Mapping[
            tuple[Timeframe, pd.Timestamp], str
        ] = ChainMap(
            staged_bar_event_ids,
            self._normalized_bar_event_ids,
        )
        staged_protected_assignment_event_ids: dict[str, str] = {}
        latest_protected_assignment_event_ids: Mapping[str, str] = ChainMap(
            staged_protected_assignment_event_ids,
            self._latest_protected_assignment_event_ids,
        )
        staged_protected_assignment_event_ids_by_timeframe: dict[
            Timeframe, str
        ] = {}
        latest_protected_assignment_event_ids_by_timeframe: Mapping[
            Timeframe, str
        ] = ChainMap(
            staged_protected_assignment_event_ids_by_timeframe,
            self._latest_protected_assignment_event_ids_by_timeframe,
        )
        staged_unresolved_forward_reference_ids = set(
            self._unresolved_forward_reference_ids
        )
        last_key = (
            event_order_key(self._events[-1])
            if self._events
            else None
        )
        appended = 0
        for event in incoming:
            _require_exact_market_event(event)
            if event.semantic_version != self.semantic_version:
                raise ValueError("event store cannot mix semantic versions")
            digest = _event_digest(event)
            prior_digest = self._digests.get(event.event_id)
            if prior_digest is None:
                prior_digest = staged_digests.get(event.event_id)
            if prior_digest is not None:
                if prior_digest != digest:
                    raise ValueError(
                        "event id conflicts with immutable history: "
                        f"{event.event_id}"
                    )
                continue
            bar_reservation = self._normalized_bar_identity(
                event,
                bar_event_ids=normalized_bar_event_ids,
            )
            self._validate_canonical_provenance(
                event,
                available_events=available_events,
                normalized_bar_event_ids=normalized_bar_event_ids,
                latest_protected_assignment_event_ids=(
                    latest_protected_assignment_event_ids
                ),
                latest_protected_assignment_event_ids_by_timeframe=(
                    latest_protected_assignment_event_ids_by_timeframe
                ),
                unresolved_forward_reference_ids=(
                    staged_unresolved_forward_reference_ids
                ),
            )
            terminal_reservation = self._terminal_crossing_identity(
                event,
                terminal_event_ids=terminal_event_ids,
            )
            if terminal_reservation is not None:
                generation_id, event_id = terminal_reservation
                staged_terminal_event_ids[generation_id] = event_id
            key = event_order_key(event)
            if last_key is not None and key < last_key:
                raise ValueError(
                    "event store append is out of known_at order: "
                    f"{event.event_id} at {event.known_at.isoformat()} "
                    f"after {last_key[0].isoformat()}"
                )
            staged.append((event, digest))
            staged_digests[event.event_id] = digest
            staged_events[event.event_id] = event
            if bar_reservation is not None:
                bar_key, bar_event_id = bar_reservation
                staged_bar_event_ids[bar_key] = bar_event_id
            protected_reservation = self._protected_assignment_identity(event)
            if protected_reservation is not None:
                protected_id, assignment_event_id = protected_reservation
                staged_protected_assignment_event_ids[
                    protected_id
                ] = assignment_event_id
                staged_protected_assignment_event_ids_by_timeframe[
                    event.timeframe
                ] = assignment_event_id
            self._advance_unresolved_forward_references(
                event,
                available_event_ids=available_events.keys(),
                unresolved_reference_ids=(
                    staged_unresolved_forward_reference_ids
                ),
            )
            last_key = key
            appended += 1
        for event, digest in staged:
            self._events.append(event)
            self._by_id[event.event_id] = event
            self._digests[event.event_id] = digest
            self._full_prefix_fingerprint_hasher.update(
                digest.encode("ascii")
            )
        self._terminal_crossing_event_ids.update(staged_terminal_event_ids)
        self._normalized_bar_event_ids.update(staged_bar_event_ids)
        self._latest_protected_assignment_event_ids.update(
            staged_protected_assignment_event_ids
        )
        self._latest_protected_assignment_event_ids_by_timeframe.update(
            staged_protected_assignment_event_ids_by_timeframe
        )
        self._unresolved_forward_reference_ids = (
            staged_unresolved_forward_reference_ids
        )
        return appended

    @staticmethod
    def _normalized_bar_identity(
        event: MarketEvent,
        *,
        bar_event_ids: Mapping[tuple[Timeframe, pd.Timestamp], str],
    ) -> tuple[tuple[Timeframe, pd.Timestamp], str] | None:
        """Reserve one normalized BAR root per timeframe/availability clock."""

        if (
            event.kind is not EventKind.BAR_COMPLETED
            or event.origin is not EventOrigin.NORMALIZED_DATA
        ):
            return None
        validate_registered_native_bar_root(
            timeframe=event.timeframe,
            event_time=event.event_time,
            known_at=event.known_at,
            evidence=event.evidence,
        )
        key = (event.timeframe, event.known_at)
        previous_event_id = bar_event_ids.get(key)
        if previous_event_id is not None and previous_event_id != event.event_id:
            raise ValueError(
                "normalized BAR root clock already has an immutable event: "
                f"{event.timeframe.value}@{event.known_at.isoformat()} "
                f"({previous_event_id})"
            )
        return key, event.event_id

    @staticmethod
    def _protected_assignment_identity(
        event: MarketEvent,
    ) -> tuple[str, str] | None:
        if (
            not event.is_canonical_semantic
            or event.is_projection
            or event.kind is not EventKind.PROTECTED_SWING_ASSIGNED
        ):
            return None
        protected_id = event.evidence.get("protected_swing_id")
        if not isinstance(protected_id, str) or not protected_id.strip():
            # The authoritative validator raises the detailed contract error.
            return None
        return protected_id, event.event_id

    @staticmethod
    def _advance_unresolved_forward_references(
        event: MarketEvent,
        *,
        available_event_ids: Iterable[str],
        unresolved_reference_ids: set[str],
    ) -> None:
        """Maintain the only condition under which a DAG walk is necessary."""

        available = available_event_ids
        unresolved_reference_ids.discard(event.event_id)
        for reference_id in (
            *event.source_event_ids,
            *event.context_event_ids,
        ):
            if reference_id not in available:
                unresolved_reference_ids.add(reference_id)

    @staticmethod
    def _terminal_crossing_identity(
        event: MarketEvent,
        *,
        terminal_event_ids: Mapping[str, str],
    ) -> tuple[str, str] | None:
        """Reserve one immutable terminal identity per crossing generation."""

        if (
            not event.is_canonical_semantic
            or event.is_projection
            or event.kind
            not in {
                EventKind.SWEEP_CONFIRMED,
                EventKind.ACCEPTANCE_CONFIRMED,
            }
        ):
            return None
        generation_id = event.evidence.get("crossing_generation_id")
        if not isinstance(generation_id, str) or not generation_id.strip():
            raise ValueError(
                "authoritative crossing terminal requires a non-empty "
                "crossing_generation_id"
            )
        previous_event_id = terminal_event_ids.get(generation_id)
        if previous_event_id is not None and previous_event_id != event.event_id:
            raise ValueError(
                "crossing_generation_id already has an immutable terminal "
                f"event: {generation_id} ({previous_event_id})"
            )
        return generation_id, event.event_id

    @staticmethod
    def _validate_canonical_provenance(
        event: MarketEvent,
        *,
        available_events: Mapping[str, MarketEvent],
        normalized_bar_event_ids: (
            Mapping[tuple[Timeframe, pd.Timestamp], str] | None
        ) = None,
        latest_protected_assignment_event_ids: Mapping[str, str] | None = None,
        latest_protected_assignment_event_ids_by_timeframe: (
            Mapping[Timeframe, str] | None
        ) = None,
        unresolved_forward_reference_ids: set[str] | None = None,
    ) -> None:
        """Validate the closed causal DAG for an authoritative semantic fact.

        Projection events are reproducible transport and deliberately do not
        become causal graph authorities.  ``LEGACY_TRANSPORT`` lifecycle
        fixtures retain their historical permissive behavior.
        """

        if not event.is_canonical_semantic or event.is_projection:
            return
        if normalized_bar_event_ids is None:
            derived_bar_event_ids: dict[
                tuple[Timeframe, pd.Timestamp], str
            ] = {}
            for candidate in available_events.values():
                reservation = EventStore._normalized_bar_identity(
                    candidate,
                    bar_event_ids=derived_bar_event_ids,
                )
                if reservation is not None:
                    key, event_id = reservation
                    derived_bar_event_ids[key] = event_id
            normalized_bar_event_ids = derived_bar_event_ids
        if latest_protected_assignment_event_ids is None:
            latest_assignments: dict[str, MarketEvent] = {}
            for candidate in available_events.values():
                reservation = (
                    EventStore._protected_assignment_identity(
                        candidate
                    )
                )
                if reservation is None:
                    continue
                protected_id, _ = reservation
                previous = latest_assignments.get(protected_id)
                if previous is None or event_order_key(previous) < event_order_key(
                    candidate
                ):
                    latest_assignments[protected_id] = candidate
            latest_protected_assignment_event_ids = {
                protected_id: assignment.event_id
                for protected_id, assignment in latest_assignments.items()
            }
        if latest_protected_assignment_event_ids_by_timeframe is None:
            latest_by_timeframe: dict[Timeframe, MarketEvent] = {}
            for candidate in available_events.values():
                if (
                    EventStore._protected_assignment_identity(
                        candidate
                    )
                    is None
                ):
                    continue
                previous = latest_by_timeframe.get(candidate.timeframe)
                if previous is None or event_order_key(previous) < event_order_key(
                    candidate
                ):
                    latest_by_timeframe[candidate.timeframe] = candidate
            latest_protected_assignment_event_ids_by_timeframe = {
                timeframe: assignment.event_id
                for timeframe, assignment in latest_by_timeframe.items()
            }
        if unresolved_forward_reference_ids is None:
            needs_cycle_check = any(
                event.event_id
                in {
                    *candidate.source_event_ids,
                    *candidate.context_event_ids,
                }
                for candidate in available_events.values()
            )
        else:
            needs_cycle_check = (
                event.event_id in unresolved_forward_reference_ids
            )
        reference_ids = (*event.source_event_ids, *event.context_event_ids)
        if event.event_id in reference_ids:
            raise ValueError(
                "canonical event provenance cannot self-reference: "
                f"{event.event_id}"
            )
        for reference_id in reference_ids:
            parent = available_events.get(reference_id)
            if parent is None:
                raise ValueError(
                    "canonical event provenance references an unavailable "
                    f"earlier event: {reference_id}"
                )
            if parent.semantic_version != event.semantic_version:
                raise ValueError(
                    "canonical event provenance cannot cross semantic "
                    f"versions: {reference_id}"
                )
            if parent.known_at > event.known_at:
                raise ValueError(
                    "canonical event provenance is not causally ordered by "
                    f"known_at: {reference_id}"
                )
            if needs_cycle_check and EventStore._provenance_reaches(
                reference_id,
                event.event_id,
                available_events=available_events,
            ):
                raise ValueError(
                    "canonical event provenance contains a cycle: "
                    f"{reference_id} -> {event.event_id}"
                )
        EventStore._validate_authoritative_parent_contract(
            event,
            available_events=available_events,
            normalized_bar_event_ids=normalized_bar_event_ids,
            latest_protected_assignment_event_ids=(
                latest_protected_assignment_event_ids
            ),
            latest_protected_assignment_event_ids_by_timeframe=(
                latest_protected_assignment_event_ids_by_timeframe
            ),
        )

    @staticmethod
    def _validate_authoritative_parent_contract(
        event: MarketEvent,
        *,
        available_events: Mapping[str, MarketEvent],
        normalized_bar_event_ids: Mapping[
            tuple[Timeframe, pd.Timestamp], str
        ],
        latest_protected_assignment_event_ids: Mapping[str, str],
        latest_protected_assignment_event_ids_by_timeframe: Mapping[
            Timeframe, str
        ],
    ) -> None:
        """Fail closed when a derived fact lacks its registered parent kinds."""

        source_parents = tuple(
            available_events[event_id]
            for event_id in event.source_event_ids
        )
        exact_kinds = _EXACT_AUTHORITATIVE_SOURCE_KINDS.get(event.kind)
        if exact_kinds is not None:
            EventStore._require_source_kind_multiset(
                event,
                source_parents,
                exact_kinds,
            )
            EventStore._require_authoritative_parent_origins(
                event,
                source_parents,
            )

        if event.kind is EventKind.SWING_CONFIRMED:
            EventStore._validate_confirmed_swing_contract(
                event,
                source_parents=source_parents,
            )

        if event.kind is EventKind.STRUCTURAL_LEG_CREATED:
            EventStore._validate_structural_leg_contract(
                event,
                source_parents=source_parents,
                available_events=available_events,
            )

        if event.kind is EventKind.LIQUIDITY_LEVEL_CREATED:
            EventStore._validate_candidate_liquidity_contract(
                event,
                source_parents=source_parents,
                available_events=available_events,
            )

        if event.kind is EventKind.DISPLACEMENT_OBSERVED:
            EventStore._validate_displacement_contract(
                event,
                source_parents=source_parents,
                available_events=available_events,
                normalized_bar_event_ids=normalized_bar_event_ids,
            )

        if event.kind is EventKind.RAW_BOUNDARY_BREAK:
            EventStore._validate_raw_break_contract(
                event,
                source_parents=source_parents,
            )

        if event.kind is EventKind.STRUCTURE_DIRECTION_CONFIRMED:
            EventStore._validate_structure_direction_contract(
                event,
                source_parents=source_parents,
            )

        if event.kind in {
            EventKind.QUALIFIED_BOS,
            EventKind.MSS_CORE_CONFIRMED,
        }:
            EventStore._validate_bos_relation_contract(
                event,
                source_parents=source_parents,
            )

        if event.kind is EventKind.PROTECTED_SWING_ASSIGNED:
            EventStore._validate_protected_swing_contract(
                event,
                source_parents=source_parents,
            )

        if event.kind is EventKind.LEVEL_TOUCHED:
            EventStore._validate_level_touch_contract(
                event,
                source_parents=source_parents,
            )

        if event.kind is EventKind.LEVEL_PENETRATED:
            EventStore._validate_penetration_contract(
                event,
                source_parents=source_parents,
                available_events=available_events,
            )

        if event.kind in {
            EventKind.SWEEP_CONFIRMED,
            EventKind.ACCEPTANCE_CONFIRMED,
        }:
            EventStore._validate_crossing_terminal_contract(
                event,
                source_parents=source_parents,
                available_events=available_events,
            )
            if event.kind is EventKind.ACCEPTANCE_CONFIRMED:
                EventStore._validate_protected_acceptance_context(
                    event,
                    available_events=available_events,
                    latest_protected_assignment_event_ids=(
                        latest_protected_assignment_event_ids
                    ),
                    latest_protected_assignment_event_ids_by_timeframe=(
                        latest_protected_assignment_event_ids_by_timeframe
                    ),
                )

        if event.kind is EventKind.BASE_ORIGIN_CORE_CREATED:
            actual_counts = Counter(parent.kind for parent in source_parents)
            if (
                actual_counts[EventKind.BAR_COMPLETED] < 1
                or any(
                    kind not in _BASE_ORIGIN_CORE_SOURCE_KINDS
                    for kind in actual_counts
                )
            ):
                raise ValueError(
                    "authoritative parent contract failed for "
                    f"{event.kind.value}: a base origin core cites only the "
                    "completed candles whose geometry it froze"
                )
            EventStore._require_authoritative_parent_origins(
                event,
                source_parents,
            )

        if event.kind is EventKind.QUALIFIED_ORIGIN_ZONE_CREATED:
            actual_counts = Counter(parent.kind for parent in source_parents)
            missing = tuple(
                kind
                for kind in sorted(
                    _ORIGIN_ZONE_SOURCE_KINDS,
                    key=lambda value: value.value,
                )
                if actual_counts[kind] < 1
            )
            unexpected = tuple(
                kind
                for kind in sorted(
                    actual_counts,
                    key=lambda value: value.value,
                )
                if kind not in _ORIGIN_ZONE_SOURCE_KINDS
            )
            if missing or unexpected:
                raise ValueError(
                    "authoritative parent contract failed for "
                    f"{event.kind.value}: requires at least one "
                    "base_origin_core_created, displacement_observed, and "
                    "raw_boundary_break source; "
                    f"missing={[kind.value for kind in missing]}, "
                    f"unexpected={[kind.value for kind in unexpected]}"
                )
            EventStore._require_authoritative_parent_origins(
                event,
                source_parents,
            )

        if event.kind in {
            EventKind.ORIGIN_ZONE_MITIGATED,
            EventKind.ORIGIN_ZONE_INVALIDATED,
        }:
            EventStore._validate_origin_zone_terminal_contract(
                event,
                source_parents=source_parents,
                available_events=available_events,
            )

        if event.kind is not EventKind.DEALING_RANGE_INVALIDATED:
            return
        detail_reason = event.details.get("transition_reason")
        if (
            event.transition_reason is not None
            and detail_reason is not None
            and event.transition_reason != detail_reason
        ):
            raise ValueError(
                "authoritative dealing-range transition reasons conflict"
            )
        transition_reason = event.transition_reason or detail_reason
        if transition_reason == "close_beyond_frozen_range":
            EventStore._validate_external_range_invalidation(
                event,
                source_parents=source_parents,
            )
        elif transition_reason == (
            "close_beyond_frozen_range_before_activation"
        ):
            EventStore._validate_forming_range_invalidation(
                event,
                source_parents=source_parents,
            )

    @staticmethod
    def _require_source_kind_multiset(
        event: MarketEvent,
        source_parents: tuple[MarketEvent, ...],
        expected_kinds: tuple[EventKind, ...],
    ) -> None:
        actual = Counter(parent.kind for parent in source_parents)
        expected = Counter(expected_kinds)
        if actual == expected:
            return
        actual_text = {
            kind.value: count
            for kind, count in sorted(
                actual.items(),
                key=lambda item: item[0].value,
            )
        }
        expected_text = {
            kind.value: count
            for kind, count in sorted(
                expected.items(),
                key=lambda item: item[0].value,
            )
        }
        raise ValueError(
            "authoritative parent contract failed for "
            f"{event.kind.value}: expected={expected_text}, "
            f"actual={actual_text}"
        )

    @staticmethod
    def _require_authoritative_parent_origins(
        event: MarketEvent,
        source_parents: tuple[MarketEvent, ...],
    ) -> None:
        for parent in source_parents:
            expected_origin = (
                EventOrigin.NORMALIZED_DATA
                if parent.kind is EventKind.BAR_COMPLETED
                else EventOrigin.SEMANTIC_ATOMIC
            )
            if parent.origin is not expected_origin:
                raise ValueError(
                    "authoritative parent contract failed for "
                    f"{event.kind.value}: parent {parent.event_id} "
                    f"({parent.kind.value}) must be "
                    f"{expected_origin.value}, got {parent.origin.value}"
                )
            if parent.kind is EventKind.BAR_COMPLETED:
                EventStore._require_real_normalized_bar(
                    parent,
                    contract=f"{event.kind.value} parent",
                )

    @staticmethod
    def _required_authoritative_text(event: MarketEvent, name: str) -> str:
        value = event.evidence.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "authoritative contract failed for "
                f"{event.kind.value}: evidence.{name} must be non-empty text"
            )
        return value

    @staticmethod
    def _finite_authoritative_number(
        event: MarketEvent,
        name: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> float:
        value = (event.evidence if payload is None else payload).get(name)
        if isinstance(value, bool):
            raise ValueError(
                "authoritative contract failed for "
                f"{event.kind.value}: {name} must be finite numeric"
            )
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "authoritative contract failed for "
                f"{event.kind.value}: {name} must be finite numeric"
            ) from error
        if not math.isfinite(result):
            raise ValueError(
                "authoritative contract failed for "
                f"{event.kind.value}: {name} must be finite numeric"
            )
        return result

    @staticmethod
    def _authoritative_clock(event: MarketEvent, name: str) -> pd.Timestamp:
        value = event.evidence.get(name)
        if not isinstance(value, (str, pd.Timestamp)):
            raise ValueError(
                "authoritative contract failed for "
                f"{event.kind.value}: evidence.{name} must be an aware clock"
            )
        try:
            return aware_timestamp(
                value,
                name=f"{event.kind.value}.evidence.{name}",
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "authoritative contract failed for "
                f"{event.kind.value}: evidence.{name} must be an aware clock"
            ) from error

    @staticmethod
    def _bar_number(bar: MarketEvent, name: str) -> float:
        value = bar.evidence.get(name)
        if isinstance(value, bool):
            raise ValueError(
                "authoritative BAR evidence requires finite " f"{name}"
            )
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "authoritative BAR evidence requires finite " f"{name}"
            ) from error
        if not math.isfinite(result):
            raise ValueError(
                "authoritative BAR evidence requires finite " f"{name}"
            )
        return result

    @staticmethod
    def _require_real_normalized_bar(
        bar: MarketEvent,
        *,
        timeframe: Timeframe | None = None,
        contract: str,
        allow_densified: bool = False,
    ) -> None:
        coverage = bar_evidence_coverage(bar.evidence)
        real = (
            coverage.admits_definitional_path
            if allow_densified
            else coverage.admits_atr_window
        )
        if (
            bar.kind is not EventKind.BAR_COMPLETED
            or bar.origin is not EventOrigin.NORMALIZED_DATA
            or (timeframe is not None and bar.timeframe is not timeframe)
            or bar.event_time != bar.known_at
            or not real
        ):
            raise ValueError(
                f"authoritative {contract} requires an exact real normalized "
                "BAR root"
            )

    @staticmethod
    def _validate_confirmed_swing_contract(
        event: MarketEvent,
        *,
        source_parents: tuple[MarketEvent, ...],
    ) -> None:
        span = _SWING_WINDOW_SPANS.get(event.timeframe)
        if span is None:
            raise ValueError("authoritative confirmed swing timeframe is invalid")
        EventStore._require_source_kind_multiset(
            event,
            source_parents,
            (EventKind.BAR_COMPLETED,) * (2 * span + 1),
        )
        EventStore._require_authoritative_parent_origins(
            event,
            source_parents,
        )
        if any(
            parent.timeframe is not event.timeframe
            or parent.evidence.get("real_completed") is not True
            or parent.evidence.get("clock_only") is not False
            for parent in source_parents
        ):
            raise ValueError(
                "authoritative confirmed swing requires a frozen window of "
                "real normalized BARs on its own timeframe"
            )
        clocks = tuple(parent.known_at for parent in source_parents)
        if clocks != tuple(sorted(clocks)) or len(clocks) != len(set(clocks)):
            raise ValueError(
                "authoritative confirmed swing BAR window must be ordered "
                "and clock-unique"
            )
        market_scopes = {
            (
                parent.evidence.get("symbol"),
                parent.evidence.get("instrument_id"),
            )
            for parent in source_parents
        }
        if (
            len(market_scopes) != 1
            or any(
                not isinstance(symbol, str)
                or not symbol.strip()
                or type(instrument_id) is not int
                or instrument_id < 0
                for symbol, instrument_id in market_scopes
            )
        ):
            raise ValueError(
                "authoritative confirmed swing BAR window has inconsistent "
                "market identity"
            )
        swing_id = EventStore._required_authoritative_text(
            event,
            "source_entity_id",
        )
        swing_side = EventStore._required_authoritative_text(
            event,
            "side",
        )
        if (
            swing_side not in {"high", "low"}
            or event.side != ("above" if swing_side == "high" else "below")
            or event.source_entity_ids != (swing_id,)
        ):
            raise ValueError(
                "authoritative confirmed swing side/entity identity conflicts"
            )
        pivot_start = EventStore._authoritative_clock(
            event,
            "pivot_start",
        )
        pivot_end = EventStore._authoritative_clock(
            event,
            "pivot_end",
        )
        pivot = source_parents[span]
        if (
            event.event_time != pivot_start
            or pivot.known_at != pivot_end
            or source_parents[-1].known_at != event.known_at
        ):
            raise ValueError(
                "authoritative confirmed swing pivot/window clocks conflict"
            )
        delay_bars = event.evidence.get("confirmation_delay_bars")
        delay_minutes = event.evidence.get("confirmation_delay_minutes")
        expected_delay_minutes = int(
            (event.known_at - event.event_time).total_seconds() // 60
        )
        if (
            type(delay_bars) is not int
            or delay_bars != span
            or type(delay_minutes) is not int
            or delay_minutes != expected_delay_minutes
        ):
            raise ValueError(
                "authoritative confirmed swing confirmation delay conflicts "
                "with its frozen BAR window"
            )
        highs = tuple(
            EventStore._bar_number(parent, "high")
            for parent in source_parents
        )
        lows = tuple(
            EventStore._bar_number(parent, "low")
            for parent in source_parents
        )
        pivot_price = highs[span] if swing_side == "high" else lows[span]
        comparison_values = highs if swing_side == "high" else lows
        left = comparison_values[:span]
        right = comparison_values[span + 1 :]
        is_extreme = (
            all(pivot_price > value for value in (*left, *right))
            if swing_side == "high"
            else all(pivot_price < value for value in (*left, *right))
        )
        if (
            not is_extreme
            or event.price is None
            or not math.isclose(float(event.price), pivot_price)
        ):
            raise ValueError(
                "authoritative confirmed swing price is not the strict "
                "two-sided pivot of its frozen BAR window"
            )
        prominence_atr = EventStore._finite_authoritative_number(
            event,
            "prominence_atr",
        )
        if prominence_atr < 0.0:
            raise ValueError(
                "authoritative confirmed swing prominence must be finite and "
                "non-negative"
            )
        if (
            type(event.evidence.get("nesting_depth")) is not int
            or int(event.evidence["nesting_depth"]) < 0
            or type(event.evidence.get("delta_ticks")) is not int
        ):
            raise ValueError(
                "authoritative confirmed swing nesting/delta fields are invalid"
            )
        EventStore._required_authoritative_text(event, "relation")
        EventStore._required_authoritative_text(event, "semantic_rank")
        magnitude = EventStore._finite_authoritative_number(
            event,
            "legacy_same_side_magnitude_atr",
        )
        if magnitude < 0.0:
            raise ValueError(
                "authoritative confirmed swing magnitude must be non-negative"
            )

    @staticmethod
    def _validate_structural_leg_contract(
        event: MarketEvent,
        *,
        source_parents: tuple[MarketEvent, ...],
        available_events: Mapping[str, MarketEvent],
    ) -> None:
        start, end = source_parents
        leg_id = EventStore._required_authoritative_text(event, "leg_id")
        start_id = EventStore._required_authoritative_text(
            event, "start_swing_id"
        )
        end_id = EventStore._required_authoritative_text(
            event, "end_swing_id"
        )
        start_parent_id = EventStore._required_authoritative_text(
            start, "source_entity_id"
        )
        end_parent_id = EventStore._required_authoritative_text(
            end, "source_entity_id"
        )
        start_clock = EventStore._authoritative_clock(
            event, "start_event_time"
        )
        end_clock = EventStore._authoritative_clock(
            event, "end_event_time"
        )
        start_price = EventStore._finite_authoritative_number(
            event, "start_price"
        )
        end_price = EventStore._finite_authoritative_number(
            event, "end_price"
        )
        if (
            start_id != start_parent_id
            or end_id != end_parent_id
            or start_id == end_id
            or event.source_entity_ids != (leg_id, start_id, end_id)
            or event.timeframe is not start.timeframe
            or event.timeframe is not end.timeframe
            or start.event_time != start_clock
            or end.event_time != end_clock
            or start.price is None
            or end.price is None
            or not math.isclose(float(start.price), start_price)
            or not math.isclose(float(end.price), end_price)
            or event.event_time != end_clock
            or event.price is None
            or not math.isclose(float(event.price), end_price)
        ):
            raise ValueError(
                "authoritative structural leg endpoint parent cross-link failed"
            )
        expected_direction = (
            Direction.LONG if end_price > start_price else Direction.SHORT
        )
        if (
            end_price == start_price
            or event.direction is not expected_direction
            or event.side
            != ("above" if expected_direction is Direction.LONG else "below")
            or start.side not in {"above", "below"}
            or end.side not in {"above", "below"}
            or start.side == end.side
        ):
            raise ValueError(
                "authoritative structural leg direction/endpoint sides conflict"
            )

        foundation_fields = frozenset(
            {
                "amplitude_ticks",
                "atr_at_leg_start",
                "atr_source_candle_ids",
                "close_efficiency",
                "close_mae_atr",
                "close_mae_points",
                "duration_seconds",
                "extreme_path_efficiency",
                "foundation_version",
                "instrument_id",
                "path_candle_ids",
                "symbol",
                "tick_size",
                "wick_mae_atr",
                "wick_mae_points",
            }
        )
        foundation_version = event.evidence.get("foundation_version")
        if foundation_version is None:
            if foundation_fields.intersection(event.evidence):
                raise ValueError(
                    "foundation structural leg fields require an explicit "
                    "foundation version"
                )
            return
        if foundation_version != FOUNDATION_VERSION:
            raise ValueError(
                "foundation structural leg version is unregistered"
            )
        missing = foundation_fields - set(event.evidence)
        if missing:
            raise ValueError(
                "foundation structural leg evidence is incomplete: "
                f"{sorted(missing)}"
            )

        def declared_id_tuple(name: str) -> tuple[str, ...]:
            raw = event.evidence.get(name)
            if isinstance(raw, str):
                values: tuple[object, ...] = ()
            else:
                try:
                    values = tuple(raw)
                except TypeError:
                    values = ()
            if (
                not values
                or len(values) != len(set(values))
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in values
                )
            ):
                raise ValueError(
                    f"foundation structural leg {name} is invalid"
                )
            return tuple(str(value) for value in values)

        duration_bars = event.evidence.get("duration_bars")
        duration_seconds = event.evidence.get("duration_seconds")
        amplitude_ticks = event.evidence.get("amplitude_ticks")
        instrument_id = event.evidence.get("instrument_id")
        symbol = event.evidence.get("symbol")
        if (
            type(duration_bars) is not int
            or duration_bars < 2
            or type(duration_seconds) is not int
            or duration_seconds <= 0
            or type(amplitude_ticks) is not int
            or amplitude_ticks <= 0
            or type(instrument_id) is not int
            or instrument_id < 0
            or not isinstance(symbol, str)
            or not symbol.strip()
        ):
            raise ValueError(
                "foundation structural leg integer or market identity is invalid"
            )
        atr_candle_ids = declared_id_tuple("atr_source_candle_ids")
        path_candle_ids = declared_id_tuple("path_candle_ids")
        if len(atr_candle_ids) != 14 or len(path_candle_ids) != duration_bars:
            raise ValueError(
                "foundation structural leg ATR/path ancestry count conflicts"
            )
        if len(event.context_event_ids) != 14 + duration_bars:
            raise ValueError(
                "foundation structural leg lacks exact BAR event context"
            )
        atr_bars = tuple(
            available_events[event_id]
            for event_id in event.context_event_ids[:14]
        )
        path_bars = tuple(
            available_events[event_id]
            for event_id in event.context_event_ids[14:]
        )
        all_bound_bars = (*atr_bars, *path_bars)
        for bar in atr_bars:
            EventStore._require_real_normalized_bar(
                bar,
                timeframe=event.timeframe,
                contract="foundation structural leg",
            )
        for bar in path_bars:
            EventStore._require_real_normalized_bar(
                bar,
                timeframe=event.timeframe,
                contract="foundation structural leg",
                allow_densified=True,
            )
        if any(
            bar.evidence.get("symbol") != symbol
            or bar.evidence.get("instrument_id") != instrument_id
            for bar in all_bound_bars
        ):
            raise ValueError(
                "foundation structural leg BAR context crosses market identity"
            )

        atr_clocks = tuple(bar.known_at for bar in atr_bars)
        path_clocks = tuple(bar.known_at for bar in path_bars)
        if (
            atr_clocks != tuple(sorted(atr_clocks))
            or path_clocks != tuple(sorted(path_clocks))
            or len(set((*atr_clocks, *path_clocks)))
            != len(all_bound_bars)
        ):
            raise ValueError(
                "foundation structural leg BAR ancestry must be ordered and unique"
            )

        eligible_bars = tuple(
            sorted(
                (
                    candidate
                    for candidate in available_events.values()
                    if candidate.kind is EventKind.BAR_COMPLETED
                    and candidate.origin is EventOrigin.NORMALIZED_DATA
                    and candidate.semantic_version == event.semantic_version
                    and candidate.timeframe is event.timeframe
                    and candidate.event_time == candidate.known_at
                    and bar_evidence_coverage(
                        candidate.evidence
                    ).admits_definitional_path
                    and candidate.evidence.get("symbol") == symbol
                    and candidate.evidence.get("instrument_id") == instrument_id
                ),
                key=lambda candidate: (
                    candidate.known_at,
                    candidate.event_id,
                ),
            )
        )
        # The registered session calendar truncates and restarts timeframe
        # buckets around the daily maintenance break, so a swing's pivot BAR is
        # the next real bar after its pivot clock, not one arithmetic stride
        # later.  Deriving both endpoint clocks from the eligible sequence keeps
        # this contract in agreement with the producer, which builds the path
        # from the candles it actually observed.
        start_pivot_terminal = next(
            (bar.known_at for bar in eligible_bars if bar.known_at > start_clock),
            None,
        )
        path_terminal = next(
            (bar.known_at for bar in eligible_bars if bar.known_at > end_clock),
            None,
        )
        if start_pivot_terminal is None or path_terminal is None:
            raise ValueError(
                "foundation structural leg endpoint lacks the completed BAR "
                "that closes its pivot"
            )
        # The path may cross a densified no-trade bar, but ATR may not: the
        # producer selects its fourteen strict-prior bars from real ones only,
        # and this contract has to reach past the same bars it does.
        eligible_prior = tuple(
            bar
            for bar in eligible_bars
            if bar.known_at <= start_clock
            and bar_evidence_coverage(bar.evidence).admits_atr_window
        )
        expected_atr_bars = eligible_prior[-14:]
        expected_path_bars = tuple(
            bar
            for bar in eligible_bars
            if start_clock < bar.known_at <= path_terminal
        )
        if (
            len(expected_atr_bars) != 14
            or tuple(bar.event_id for bar in atr_bars)
            != tuple(bar.event_id for bar in expected_atr_bars)
            or tuple(bar.event_id for bar in path_bars)
            != tuple(bar.event_id for bar in expected_path_bars)
        ):
            raise ValueError(
                "foundation structural leg does not bind the exact strict-prior "
                "ATR or native path BARs"
            )

        detector_ids = tuple(
            bar.evidence.get("detector_candle_id") for bar in all_bound_bars
        )
        declared_detector_ids = (*atr_candle_ids, *path_candle_ids)
        if (
            any(
                not isinstance(value, str) or not value.strip()
                for value in detector_ids
            )
            or detector_ids != declared_detector_ids
            or event.source_data_ids != declared_detector_ids
        ):
            raise ValueError(
                "foundation structural leg self-reported candle ancestry does "
                "not bind its normalized BAR parents"
            )

        span = _SWING_WINDOW_SPANS.get(event.timeframe)
        if span is None:
            raise ValueError(
                "foundation structural leg timeframe is unregistered"
            )
        endpoint_pivot_bars: list[MarketEvent] = []
        for swing in (start, end):
            if len(swing.source_event_ids) != 2 * span + 1:
                raise ValueError(
                    "foundation structural leg endpoint Swing lacks its "
                    "definitional BAR window"
                )
            pivot_bar = available_events.get(swing.source_event_ids[span])
            if pivot_bar is None:
                raise ValueError(
                    "foundation structural leg endpoint Swing pivot is unavailable"
                )
            EventStore._require_real_normalized_bar(
                pivot_bar,
                timeframe=event.timeframe,
                contract="foundation structural leg endpoint Swing",
            )
            endpoint_pivot_bars.append(pivot_bar)
        start_pivot_bar, end_pivot_bar = endpoint_pivot_bars
        if (
            start_pivot_bar.event_id != path_bars[0].event_id
            or end_pivot_bar.event_id != path_bars[-1].event_id
            or start_pivot_bar.known_at
            != EventStore._authoritative_clock(start, "pivot_end")
            or end_pivot_bar.known_at
            != EventStore._authoritative_clock(end, "pivot_end")
            or start_pivot_bar.known_at != start_pivot_terminal
            or end_pivot_bar.known_at != path_terminal
            or event.known_at != end.known_at
        ):
            raise ValueError(
                "foundation structural leg path endpoints do not bind its "
                "confirmed Swing pivots and clocks"
            )

        tick_size = EventStore._finite_authoritative_number(
            event,
            "tick_size",
        )
        if tick_size <= 0.0:
            raise ValueError(
                "foundation structural leg tick size must be positive"
            )
        try:
            expected_amplitude_ticks = abs(
                price_to_ticks(
                    end_price,
                    tick_size,
                    name="structural_leg.end_price",
                )
                - price_to_ticks(
                    start_price,
                    tick_size,
                    name="structural_leg.start_price",
                )
            )
        except ValueError as error:
            raise ValueError(
                "foundation structural leg endpoint price is off-grid"
            ) from error

        start_close = EventStore._finite_authoritative_number(
            event,
            "start_close",
        )
        end_close = EventStore._finite_authoritative_number(
            event,
            "end_close",
        )
        amplitude = EventStore._finite_authoritative_number(
            event,
            "amplitude_points",
        )
        atr0 = EventStore._finite_authoritative_number(
            event,
            "atr_at_leg_start",
        )
        amplitude_atr = EventStore._finite_authoritative_number(
            event,
            "amplitude_atr",
        )
        duration_minutes = event.evidence.get("duration_minutes")
        close_efficiency = EventStore._finite_authoritative_number(
            event,
            "close_efficiency",
        )
        efficiency = EventStore._finite_authoritative_number(
            event,
            "efficiency",
        )
        extreme_efficiency = EventStore._finite_authoritative_number(
            event,
            "extreme_path_efficiency",
        )
        max_retracement = EventStore._finite_authoritative_number(
            event,
            "max_retracement_points",
        )
        max_retracement_atr = (
            EventStore._finite_authoritative_number(
                event,
                "max_retracement_atr",
            )
        )
        close_mae = EventStore._finite_authoritative_number(
            event,
            "close_mae_points",
        )
        close_mae_atr = EventStore._finite_authoritative_number(
            event,
            "close_mae_atr",
        )
        wick_mae = EventStore._finite_authoritative_number(
            event,
            "wick_mae_points",
        )
        wick_mae_atr = EventStore._finite_authoritative_number(
            event,
            "wick_mae_atr",
        )
        if (
            atr0 <= 0.0
            or amplitude <= 0.0
            or type(duration_minutes) is not int
            or duration_minutes <= 0
            or any(
                value < 0.0
                for value in (
                    max_retracement,
                    max_retracement_atr,
                    close_mae,
                    close_mae_atr,
                    wick_mae,
                    wick_mae_atr,
                )
            )
            or any(
                not 0.0 <= value <= 1.0
                for value in (
                    close_efficiency,
                    efficiency,
                    extreme_efficiency,
                )
            )
        ):
            raise ValueError(
                "foundation structural leg metric domains are invalid"
            )

        true_ranges: list[float] = []
        for index, bar in enumerate(eligible_prior):
            high = EventStore._bar_number(bar, "high")
            low = EventStore._bar_number(bar, "low")
            true_range = high - low
            if index:
                prior_close = EventStore._bar_number(
                    eligible_prior[index - 1],
                    "close",
                )
                true_range = max(
                    true_range,
                    abs(high - prior_close),
                    abs(low - prior_close),
                )
            true_ranges.append(max(0.0, true_range))
        expected_atr = sum(true_ranges[-14:]) / 14.0

        closes = tuple(
            EventStore._bar_number(bar, "close")
            for bar in path_bars
        )
        close_travel = sum(
            abs(right - left)
            for left, right in zip(closes, closes[1:])
        )
        expected_close_efficiency = min(
            1.0,
            max(
                0.0,
                abs(closes[-1] - closes[0])
                / max(close_travel, 1e-12),
            ),
        )
        running = closes[0]
        expected_retracement = 0.0
        for close in closes[1:]:
            if expected_direction is Direction.LONG:
                running = max(running, close)
                expected_retracement = max(
                    expected_retracement,
                    running - close,
                )
            else:
                running = min(running, close)
                expected_retracement = max(
                    expected_retracement,
                    close - running,
                )
        directional_extremes = (
            (
                start_price,
                *(
                    EventStore._bar_number(bar, "high")
                    for bar in path_bars
                ),
            )
            if expected_direction is Direction.LONG
            else (
                start_price,
                *(
                    EventStore._bar_number(bar, "low")
                    for bar in path_bars
                ),
            )
        )
        extreme_travel = sum(
            abs(right - left)
            for left, right in zip(
                directional_extremes,
                directional_extremes[1:],
            )
        )
        expected_extreme_efficiency = min(
            1.0,
            max(0.0, amplitude / max(extreme_travel, 1e-12)),
        )
        expected_close_mae = (
            max(0.0, closes[0] - min(closes))
            if expected_direction is Direction.LONG
            else max(0.0, max(closes) - closes[0])
        )
        expected_wick_mae = (
            max(
                0.0,
                start_price
                - min(
                    EventStore._bar_number(bar, "low")
                    for bar in path_bars
                ),
            )
            if expected_direction is Direction.LONG
            else max(
                0.0,
                max(
                    EventStore._bar_number(bar, "high")
                    for bar in path_bars
                )
                - start_price,
            )
        )

        expected_seconds = int((end_clock - start_clock).total_seconds())
        expected_values = (
            (start_close, closes[0]),
            (end_close, closes[-1]),
            (amplitude, abs(end_price - start_price)),
            (amplitude, amplitude_ticks * tick_size),
            (atr0, expected_atr),
            (amplitude_atr, amplitude / atr0),
            (close_efficiency, expected_close_efficiency),
            (efficiency, expected_close_efficiency),
            (extreme_efficiency, expected_extreme_efficiency),
            (max_retracement, expected_retracement),
            (max_retracement_atr, expected_retracement / atr0),
            (close_mae, expected_close_mae),
            (close_mae_atr, expected_close_mae / atr0),
            (wick_mae, expected_wick_mae),
            (wick_mae_atr, expected_wick_mae / atr0),
        )
        if (
            amplitude_ticks != expected_amplitude_ticks
            or duration_bars != len(path_bars)
            or duration_seconds != expected_seconds
            or duration_minutes != expected_seconds // 60
            or not math.isclose(event.strength, expected_close_efficiency)
            or any(
                not math.isclose(
                    actual,
                    expected,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                for actual, expected in expected_values
            )
        ):
            raise ValueError(
                "foundation structural leg metrics conflict with bound BARs"
            )

    @staticmethod
    def _validate_candidate_liquidity_contract(
        event: MarketEvent,
        *,
        source_parents: tuple[MarketEvent, ...],
        available_events: Mapping[str, MarketEvent],
    ) -> None:
        level_id = EventStore._required_authoritative_text(
            event, "level_id"
        )
        source_kind = EventStore._required_authoritative_text(
            event, "source_kind"
        )
        if (
            event.evidence.get("candidate_only") is not True
            or source_kind not in _CANDIDATE_LIQUIDITY_SOURCE_KINDS
            or not event.source_entity_ids
            or event.source_entity_ids[0] != level_id
            or event.side not in {"above", "below"}
            or event.price is None
            or not math.isfinite(float(event.price))
            or event.zone is None
            or not all(math.isfinite(float(value)) for value in event.zone)
            or event.zone[0] > float(event.price)
            or float(event.price) > event.zone[1]
        ):
            raise ValueError(
                "authoritative candidate liquidity identity, taxonomy, or "
                "frozen geometry is invalid"
            )
        EventStore._require_authoritative_parent_origins(
            event,
            source_parents,
        )
        parent_kinds = tuple(parent.kind for parent in source_parents)
        if source_kind == "confirmed_swing":
            EventStore._require_source_kind_multiset(
                event,
                source_parents,
                (EventKind.SWING_CONFIRMED,),
            )
            swing = source_parents[0]
            swing_id = EventStore._required_authoritative_text(
                swing,
                "source_entity_id",
            )
            if (
                event.source_entity_ids != (level_id, swing_id)
                or event.evidence.get("source_swing_id") != swing_id
                or event.timeframe is not swing.timeframe
                or event.side != swing.side
                or event.price != swing.price
                or event.event_time != swing.event_time
                or event.known_at != swing.known_at
                or event.zone != (float(event.price), float(event.price))
            ):
                raise ValueError(
                    "authoritative confirmed-swing candidate does not bind "
                    "its exact Swing parent"
                )
        elif source_kind in _CANDIDATE_SWING_PARENT_SOURCE_KINDS:
            minimum = 2 if source_kind in {
                "formed_liquidity_pool",
                "equal_highs",
                "equal_lows",
            } else 1
            if (
                len(source_parents) < minimum
                or any(
                    kind is not EventKind.SWING_CONFIRMED
                    for kind in parent_kinds
                )
            ):
                raise ValueError(
                    "authoritative swing-derived candidate lacks its exact "
                    "confirmed Swing parents"
                )
            swing_ids = tuple(
                EventStore._required_authoritative_text(
                    parent,
                    "source_entity_id",
                )
                for parent in source_parents
            )
            declared_ids = event.evidence.get(
                "member_swing_ids"
                if source_kind == "formed_liquidity_pool"
                else "source_ids"
            )
            if isinstance(declared_ids, str):
                declared_ids = ()
            try:
                declared_ids = tuple(declared_ids)
            except TypeError:
                declared_ids = ()
            if (
                event.source_entity_ids != (level_id, *swing_ids)
                or declared_ids not in {(), swing_ids}
                or any(
                    parent.timeframe is not event.timeframe
                    for parent in source_parents
                )
            ):
                raise ValueError(
                    "authoritative swing-derived candidate entity lineage "
                    "conflicts with its Swing parents"
                )
        elif source_kind in _CANDIDATE_REFERENCE_PARENT_SOURCE_KINDS:
            if len(source_parents) != 2 or any(
                kind is not EventKind.BAR_COMPLETED for kind in parent_kinds
            ):
                raise ValueError(
                    "authoritative completed-period candidate requires its "
                    "exact extreme/admission BAR roots"
                )
            for parent in source_parents:
                EventStore._require_real_normalized_bar(
                    parent,
                    timeframe=Timeframe.M1,
                    contract="completed-period candidate",
                )
            extreme, admission = source_parents
            extreme_at = EventStore._authoritative_clock(
                event, "reference_extreme_at"
            )
            admitted_at = EventStore._authoritative_clock(
                event, "reference_admitted_at"
            )
            period_started_at = EventStore._authoritative_clock(
                event, "reference_period_started_at"
            )
            period_last_at = EventStore._authoritative_clock(
                event, "reference_period_last_completed_at"
            )
            source_confirmed_at = EventStore._authoritative_clock(
                event, "source_confirmed_at"
            )
            raw_source_ids = event.evidence.get("source_ids")
            if isinstance(raw_source_ids, str):
                source_ids: tuple[object, ...] = ()
            else:
                try:
                    source_ids = tuple(raw_source_ids)
                except TypeError:
                    source_ids = ()
            high_reference = source_kind.endswith("_high")
            expected_side = "above" if high_reference else "below"
            expected_price = EventStore._bar_number(
                extreme,
                "high" if high_reference else "low",
            )
            if (
                event.timeframe is not Timeframe.M1
                or extreme.known_at != extreme_at
                or admission.known_at != admitted_at
                or event.event_time != extreme_at
                or event.known_at != admitted_at
                or source_confirmed_at != period_last_at
                or not (
                    period_started_at
                    <= extreme_at
                    <= period_last_at
                    < admitted_at
                )
                or extreme.evidence.get("symbol")
                != admission.evidence.get("symbol")
                or extreme.evidence.get("instrument_id")
                != admission.evidence.get("instrument_id")
                or event.side != expected_side
                or float(event.price) != expected_price
                or event.zone != (expected_price, expected_price)
                or event.evidence.get("reference_extreme_tie_rule")
                != "first_completed_m1_at_extreme"
                or len(source_ids) != 1
                or not isinstance(source_ids[0], str)
                or not source_ids[0].strip()
                or event.source_entity_ids != (level_id, source_ids[0])
            ):
                raise ValueError(
                    "authoritative completed-period candidate does not bind "
                    "its exact reference clocks, BAR geometry, or identity"
                )
        elif source_kind in {
            "previous_session",
            "previous_day",
            "previous_week",
        }:
            raw_source_ids = event.evidence.get("source_ids")
            if isinstance(raw_source_ids, str):
                source_ids = ()
            else:
                try:
                    source_ids = tuple(raw_source_ids)
                except TypeError:
                    source_ids = ()
            contexts = tuple(
                available_events[event_id]
                for event_id in event.context_event_ids
            )
            context = contexts[0] if len(contexts) == 1 else None
            compatibility_invalid = (
                event.timeframe is not Timeframe.M1
                or len(source_ids) != 1
                or not isinstance(source_ids[0], str)
                or not source_ids[0].startswith(
                    f"reference_source:{source_kind.removeprefix('previous_')}:"
                )
                or event.source_entity_ids != (level_id, source_ids[0])
                or context is None
                or context.kind is not EventKind.SUPPORT_RESISTANCE_STATE
                or context.origin is not EventOrigin.LEGACY_TRANSPORT
                or context.entity_id != level_id
                or context.timeframe is not event.timeframe
                or context.side != event.side
                or context.price != event.price
                or context.zone != event.zone
                or context.formed_at != event.event_time
                or context.known_at != event.known_at
                or context.evidence.get("source_kind") != source_kind
                or tuple(context.evidence.get("source_ids", ())) != source_ids
            )
            if not source_parents:
                if (
                    event.semantic_version
                    not in _LEGACY_SOURCE_FREE_REFERENCE_ZONE_SEMANTIC_VERSIONS
                    or compatibility_invalid
                ):
                    raise ValueError(
                        "authoritative legacy reference-zone candidate must "
                        "bind its exact source-free compatibility state "
                        "projection"
                    )
                return
            point = source_parents[0] if len(source_parents) == 1 else None
            point_level_id = (
                None if point is None else point.evidence.get("level_id")
            )
            expected_point_kind = (
                f"{source_kind}_"
                f"{'high' if event.side == 'above' else 'low'}"
            )
            expected_point_level_id = (
                None
                if len(source_ids) != 1 or not isinstance(source_ids[0], str)
                else source_ids[0].replace(
                    "reference_source:",
                    "reference:",
                    1,
                )
            )
            if (
                compatibility_invalid
                or point is None
                or point.kind is not EventKind.LIQUIDITY_LEVEL_CREATED
                or point.origin is not EventOrigin.SEMANTIC_ATOMIC
                or point.timeframe is not Timeframe.M1
                or point.side != event.side
                or point.price is None
                or float(point.price) != float(event.price)
                or point.zone != (float(event.price), float(event.price))
                or point.known_at > event.known_at
                or point.evidence.get("candidate_only") is not True
                or point.evidence.get("source_kind") != expected_point_kind
                or tuple(point.evidence.get("source_ids", ())) != source_ids
                or not isinstance(point_level_id, str)
                or point_level_id != expected_point_level_id
                or point.source_entity_ids
                != (point_level_id, source_ids[0])
            ):
                raise ValueError(
                    "authoritative reference-zone candidate must bind its "
                    "exact completed-period point parent and compatibility "
                    "state projection"
                )
        elif source_kind == "mature_range_boundary":
            # A boundary level descends from the interval that froze it. It
            # used to cite BALANCE_RANGE_MATURED because promotion happened at
            # a maturity transition; the range no longer has one, so the
            # parent is the creation event the boundaries belong to.
            if parent_kinds != (EventKind.DEALING_RANGE_CREATED,):
                raise ValueError(
                    "authoritative mature-range candidate requires its exact "
                    "created structural-range parent"
                )
            range_id = EventStore._required_authoritative_text(
                source_parents[0],
                "range_id",
            )
            if (
                event.evidence.get("range_id") != range_id
                or len(event.source_entity_ids) < 2
                or event.source_entity_ids[:2] != (level_id, range_id)
            ):
                raise ValueError(
                    "authoritative mature-range candidate does not bind its "
                    "range identity"
                )
        elif source_kind == "range_boundary":
            if source_parents:
                if len(source_parents) != 1 or parent_kinds[0] not in {
                    EventKind.BAR_COMPLETED,
                    EventKind.DEALING_RANGE_CREATED,
                }:
                    raise ValueError(
                        "authoritative range-boundary candidate parent is "
                        "invalid"
                    )
                parent = source_parents[0]
                if parent.kind is EventKind.BAR_COMPLETED:
                    EventStore._require_real_normalized_bar(
                        parent,
                        timeframe=event.timeframe,
                        contract="range-boundary candidate",
                    )
                    expected_price = EventStore._bar_number(
                        parent,
                        "high" if event.side == "above" else "low",
                    )
                    if (
                        event.event_time != parent.event_time
                        or event.known_at != parent.known_at
                        or float(event.price) != expected_price
                        or event.zone != (expected_price, expected_price)
                    ):
                        raise ValueError(
                            "authoritative BAR-derived range boundary does "
                            "not bind its exact extreme geometry and clock"
                        )
                else:
                    range_id = EventStore._required_authoritative_text(
                        parent,
                        "range_id",
                    )
                    expected_price = (
                        float(parent.zone[1])
                        if parent.zone is not None and event.side == "above"
                        else (
                            float(parent.zone[0])
                            if parent.zone is not None
                            else None
                        )
                    )
                    if (
                        event.evidence.get("range_id") != range_id
                        or len(event.source_entity_ids) < 2
                        or event.source_entity_ids[:2] != (level_id, range_id)
                        or event.timeframe is not parent.timeframe
                        or event.event_time != parent.event_time
                        or event.known_at != parent.known_at
                        or expected_price is None
                        or float(event.price) != expected_price
                        or event.zone != (expected_price, expected_price)
                    ):
                        raise ValueError(
                            "authoritative range-boundary candidate does not "
                            "bind its activated range identity"
                        )
            else:
                contexts = tuple(
                    available_events[event_id]
                    for event_id in event.context_event_ids
                )
                context = contexts[0] if len(contexts) == 1 else None
                raw_source_ids = event.evidence.get("source_ids", ())
                if isinstance(raw_source_ids, str):
                    source_ids: tuple[object, ...] = ()
                else:
                    try:
                        source_ids = tuple(raw_source_ids)
                    except TypeError:
                        source_ids = ()
                typed_context_shape = bool(
                    context is not None
                    and context.entity_id == level_id
                    and event.source_entity_ids
                    == (level_id, *context.source_ids)
                )
                initial_context_shape = bool(
                    context is not None
                    and context.entity_id is None
                    and context.source_ids
                    and context.source_ids[0] == level_id
                    and event.source_entity_ids == context.source_ids
                )
                if (
                    context is None
                    or context.kind is not EventKind.SUPPORT_RESISTANCE_STATE
                    or context.origin is not EventOrigin.LEGACY_TRANSPORT
                    or context.timeframe is not event.timeframe
                    or context.side != event.side
                    or context.price != event.price
                    or context.zone != event.zone
                    or context.formed_at != event.event_time
                    or context.known_at != event.known_at
                    or context.evidence.get("source_kind") != source_kind
                    or tuple(context.evidence.get("source_ids", ()))
                    != source_ids
                    or not (typed_context_shape or initial_context_shape)
                ):
                    raise ValueError(
                        "authoritative source-free range-boundary candidate "
                        "must bind its exact compatibility state projection"
                    )

    @staticmethod
    def _validate_level_touch_contract(
        event: MarketEvent,
        *,
        source_parents: tuple[MarketEvent, ...],
    ) -> None:
        parent_by_kind = {parent.kind: parent for parent in source_parents}
        candidate = parent_by_kind[EventKind.LIQUIDITY_LEVEL_CREATED]
        bar = parent_by_kind[EventKind.BAR_COMPLETED]
        EventStore._require_real_normalized_bar(
            bar,
            timeframe=event.timeframe,
            contract="level touch",
        )
        level_id = EventStore._required_crossing_text(event, "level_id")
        candidate_level_id = EventStore._required_crossing_text(
            candidate,
            "level_id",
        )
        candidate_timeframe_bound = (
            candidate.timeframe is event.timeframe
            or (
                event.timeframe is Timeframe.M1
                and event.evidence.get("source_timeframe")
                == candidate.timeframe.value
            )
        )
        if (
            level_id != candidate_level_id
            or not candidate_timeframe_bound
            or event.event_time != bar.event_time
            or event.known_at < bar.known_at
            or event.side != candidate.side
        ):
            raise ValueError(
                "authoritative level touch does not bind its candidate and "
                "exact real crossing BAR"
            )

    @staticmethod
    def _validate_displacement_contract(
        event: MarketEvent,
        *,
        source_parents: tuple[MarketEvent, ...],
        available_events: Mapping[str, MarketEvent],
        normalized_bar_event_ids: Mapping[
            tuple[Timeframe, pd.Timestamp], str
        ],
    ) -> None:
        if not source_parents:
            raise ValueError(
                "authoritative displacement requires real M5 BAR parents"
            )
        EventStore._require_source_kind_multiset(
            event,
            source_parents,
            (EventKind.BAR_COMPLETED,) * len(source_parents),
        )
        EventStore._require_authoritative_parent_origins(
            event,
            source_parents,
        )
        if (
            event.timeframe is not Timeframe.M5
            or event.direction not in {Direction.LONG, Direction.SHORT}
            or event.side
            != ("above" if event.direction is Direction.LONG else "below")
            or any(
                parent.timeframe is not Timeframe.M5
                or parent.evidence.get("real_completed") is not True
                or parent.evidence.get("clock_only") is not False
                for parent in source_parents
            )
        ):
            raise ValueError(
                "authoritative displacement requires real normalized M5 BARs"
            )
        bar_clocks = tuple(parent.known_at for parent in source_parents)
        if bar_clocks != tuple(sorted(bar_clocks)):
            raise ValueError(
                "authoritative displacement BAR lineage is not producer-ordered"
            )
        detector_ids = tuple(
            parent.evidence.get("detector_candle_id")
            for parent in source_parents
        )
        admitted_ids = event.evidence.get("admitted_candle_ids")
        if isinstance(admitted_ids, str):
            admitted_ids = ()
        try:
            admitted_ids = tuple(admitted_ids)
        except TypeError:
            admitted_ids = ()
        if (
            any(
                not isinstance(value, str) or not value.strip()
                for value in detector_ids
            )
            or detector_ids != admitted_ids
            or event.source_data_ids != admitted_ids
        ):
            raise ValueError(
                "authoritative displacement admitted detector BAR identities "
                "conflict"
            )
        displacement_id = EventStore._required_authoritative_text(
            event, "displacement_id"
        )
        EventStore._required_authoritative_text(event, "transition_id")
        lifecycle = EventStore._required_authoritative_text(
            event, "lifecycle"
        )
        terminal_reason = event.evidence.get("terminal_reason")
        if (
            lifecycle not in _DISPLACEMENT_LIFECYCLES
            or event.source_entity_ids != (displacement_id,)
            or (
                lifecycle in {"started", "active"}
                and terminal_reason is not None
            )
            or (
                lifecycle in {"exhausted", "censored"}
                and (
                    not isinstance(terminal_reason, str)
                    or not terminal_reason.strip()
                )
            )
        ):
            raise ValueError(
                "authoritative displacement lifecycle/entity identity is invalid"
            )
        prefix_clock = EventStore._authoritative_clock(
            event, "prefix_last_admitted_at"
        )
        if prefix_clock != source_parents[-1].known_at or event.known_at < prefix_clock:
            raise ValueError(
                "authoritative displacement prefix clock conflicts with its "
                "last admitted BAR"
            )
        metrics = event.evidence.get("state_metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(
                "authoritative displacement state_metrics must be a mapping"
            )
        missing_metrics = _DISPLACEMENT_REQUIRED_METRICS - set(metrics)
        if missing_metrics:
            raise ValueError(
                "authoritative displacement state_metrics are incomplete: "
                f"{sorted(missing_metrics)}"
            )
        numeric_metrics = {
            name: EventStore._finite_authoritative_number(
                event,
                name,
                payload=metrics,
            )
            for name in _DISPLACEMENT_REQUIRED_METRICS
        }
        if (
            numeric_metrics["atr0"] <= 0.0
            or numeric_metrics["real_episode_bar_count"]
            != len(source_parents)
            or any(
                numeric_metrics[name] < 0.0
                for name in {
                    "age_minutes_at_last_admitted",
                    "real_episode_bar_count",
                    "interruption_run",
                    "total_interruption_bars",
                    "directional_bar_count",
                    "neutral_bar_count",
                    "opposite_bar_count",
                    "activation_gate_count",
                }
            )
        ):
            raise ValueError(
                "authoritative displacement metric domains or BAR count conflict"
            )
        synthetic_terminal = (
            lifecycle == "censored"
            and terminal_reason == "synthetic_interruption"
        )
        if not synthetic_terminal:
            if event.context_event_ids:
                raise ValueError(
                    "authoritative displacement context is reserved for an "
                    "exact synthetic-interruption terminal"
                )
            return
        interval_start = event.known_at - pd.Timedelta(5, unit="min")
        if source_parents[-1].known_at != interval_start:
            raise ValueError(
                "authoritative synthetic displacement terminal must retain "
                "the immediately preceding real M5 BAR as its last source"
            )
        expected_clocks = tuple(
            interval_start + pd.Timedelta(offset, unit="min")
            for offset in range(1, 6)
        )
        interval_roots: list[MarketEvent] = []
        for clock in expected_clocks:
            root_id = normalized_bar_event_ids.get((Timeframe.M1, clock))
            root = (
                None if root_id is None else available_events.get(root_id)
            )
            if (
                root is None
                or root.kind is not EventKind.BAR_COMPLETED
                or root.origin is not EventOrigin.NORMALIZED_DATA
                or root.timeframe is not Timeframe.M1
                or root.event_time != clock
                or root.known_at != clock
            ):
                raise ValueError(
                    "authoritative synthetic displacement terminal lacks "
                    "five contiguous unique M1 BAR roots"
                )
            interval_roots.append(root)
        if len({root.event_id for root in interval_roots}) != 5:
            raise ValueError(
                "authoritative synthetic displacement terminal repeats an "
                "M1 BAR root"
            )
        market_identities = {
            (
                root.evidence.get("symbol"),
                root.evidence.get("instrument_id"),
            )
            for root in (*source_parents, *interval_roots)
        }
        if len(market_identities) != 1:
            raise ValueError(
                "authoritative synthetic displacement terminal crosses "
                "market identities"
            )
        for root in interval_roots:
            real_completed = root.evidence.get("real_completed")
            clock_only = root.evidence.get("clock_only")
            if (
                not isinstance(real_completed, bool)
                or not isinstance(clock_only, bool)
                or clock_only is not (not real_completed)
            ):
                raise ValueError(
                    "authoritative synthetic displacement M1 BAR flags are "
                    "inconsistent"
                )
        expected_context = tuple(
            root.event_id
            for root in interval_roots
            if root.evidence["clock_only"] is True
        )
        if not expected_context or event.context_event_ids != expected_context:
            raise ValueError(
                "authoritative synthetic displacement context must be the "
                "exact clock-only M1 subset of its open-closed M5 interval"
            )

    @staticmethod
    def _validate_raw_break_contract(
        event: MarketEvent,
        *,
        source_parents: tuple[MarketEvent, ...],
    ) -> None:
        parent_by_kind = {parent.kind: parent for parent in source_parents}
        swing = parent_by_kind[EventKind.SWING_CONFIRMED]
        bar = parent_by_kind[EventKind.BAR_COMPLETED]
        EventStore._require_real_normalized_bar(
            bar,
            timeframe=event.timeframe,
            contract="raw boundary break",
        )
        bos_id = EventStore._required_authoritative_text(
            event,
            "bos_id",
        )
        target_swing_id = EventStore._required_authoritative_text(
            event,
            "target_swing_id",
        )
        swing_id = EventStore._required_authoritative_text(
            swing,
            "source_entity_id",
        )
        scope = EventStore._required_authoritative_text(
            event,
            "scope",
        )
        break_bar_id = EventStore._required_authoritative_text(
            event,
            "break_bar_id",
        )
        break_close = EventStore._finite_authoritative_number(
            event,
            "break_close",
        )
        bar_close = EventStore._bar_number(bar, "close")
        if (
            event.direction not in {Direction.LONG, Direction.SHORT}
            or scope not in {"continuation", "opposed", "local"}
            or target_swing_id != swing_id
            or event.timeframe is not swing.timeframe
            or event.event_time != bar.event_time
            or event.known_at != bar.known_at
            or event.side
            != ("above" if event.direction is Direction.LONG else "below")
            or swing.side != event.side
            or event.price != swing.price
            or bar.evidence.get("detector_candle_id") != break_bar_id
            or event.source_data_ids != (break_bar_id,)
            or len(event.source_entity_ids) < 2
            or event.source_entity_ids[:2] != (bos_id, target_swing_id)
            or not math.isclose(break_close, bar_close)
            or event.evidence.get("break_buffer_ticks") != 0
            or event.evidence.get("comparison") != "strict_close_beyond"
            or event.evidence.get("break_standard")
            != "close_beyond_confirmed_boundary"
            or (
                event.direction is Direction.LONG
                and not bar_close > float(swing.price)
            )
            or (
                event.direction is Direction.SHORT
                and not bar_close < float(swing.price)
            )
        ):
            raise ValueError(
                "authoritative raw boundary break direction, clock, strict "
                "close, or Swing/BAR entity cross-link failed"
            )

    @staticmethod
    def _validate_structure_direction_contract(
        event: MarketEvent,
        *,
        source_parents: tuple[MarketEvent, ...],
    ) -> None:
        high, low = source_parents
        high_id = EventStore._required_authoritative_text(
            high,
            "source_entity_id",
        )
        low_id = EventStore._required_authoritative_text(
            low,
            "source_entity_id",
        )
        structure_id = EventStore._required_authoritative_text(
            event,
            "structure_id",
        )
        protected_id = EventStore._required_authoritative_text(
            event,
            "candidate_protected_swing_id",
        )
        expected_protected = (
            low if event.direction is Direction.LONG else high
        )
        if (
            high.evidence.get("side") != "high"
            or low.evidence.get("side") != "low"
            or high.side != "above"
            or low.side != "below"
            or event.direction not in {Direction.LONG, Direction.SHORT}
            or event.evidence.get("direction") != event.direction.value
            or event.side
            != ("above" if event.direction is Direction.LONG else "below")
            or event.timeframe is not high.timeframe
            or event.timeframe is not low.timeframe
            or event.evidence.get("source_high_id") != high_id
            or event.evidence.get("source_low_id") != low_id
            or event.source_entity_ids != (structure_id, high_id, low_id)
            or protected_id
            != expected_protected.evidence.get("source_entity_id")
            or event.price != expected_protected.price
            or type(event.evidence.get("sequence_count")) is not int
            or int(event.evidence["sequence_count"]) < 1
        ):
            raise ValueError(
                "authoritative structure direction does not bind its exact "
                "high/low Swing formation chain"
            )

    @staticmethod
    def _validate_bos_relation_contract(
        event: MarketEvent,
        *,
        source_parents: tuple[MarketEvent, ...],
    ) -> None:
        parent_by_kind = {parent.kind: parent for parent in source_parents}
        raw = parent_by_kind[EventKind.RAW_BOUNDARY_BREAK]
        structure = parent_by_kind[EventKind.STRUCTURE_DIRECTION_CONFIRMED]
        bos_id = EventStore._required_authoritative_text(event, "bos_id")
        raw_bos_id = EventStore._required_authoritative_text(raw, "bos_id")
        structure_id = EventStore._required_authoritative_text(
            structure, "structure_id"
        )
        scope = EventStore._required_authoritative_text(event, "scope")
        raw_scope = EventStore._required_authoritative_text(raw, "scope")
        expected_scope = (
            "continuation"
            if event.kind is EventKind.QUALIFIED_BOS
            else "opposed"
        )
        direction_relation = (
            structure.direction is event.direction
            if event.kind is EventKind.QUALIFIED_BOS
            else structure.direction is not event.direction
        )
        required_definition_field = (
            event.evidence.get("qualification")
            == "aligned_with_confirmed_structure"
            if event.kind is EventKind.QUALIFIED_BOS
            else event.evidence.get("core_definition")
            == "first_opposed_confirmed_boundary_break"
        )
        if (
            event.direction not in {Direction.LONG, Direction.SHORT}
            or raw.direction is not event.direction
            or structure.direction not in {Direction.LONG, Direction.SHORT}
            or not direction_relation
            or event.timeframe is not raw.timeframe
            or event.timeframe is not structure.timeframe
            or event.event_time != raw.event_time
            or event.known_at != raw.known_at
            or event.side != raw.side
            or event.price != raw.price
            or scope != expected_scope
            or raw_scope != expected_scope
            or bos_id != raw_bos_id
            or event.source_entity_ids != (bos_id, structure_id)
            or not raw.source_entity_ids
            or raw.source_entity_ids[0] != bos_id
            or structure_id not in raw.source_entity_ids
            or not structure.source_entity_ids
            or structure.source_entity_ids[0] != structure_id
            or structure.evidence.get("direction") != structure.direction.value
            or not required_definition_field
        ):
            raise ValueError(
                "authoritative BOS/MSS direction, timeframe, clock, or entity "
                "cross-link failed"
            )

    @staticmethod
    def _validate_protected_swing_contract(
        event: MarketEvent,
        *,
        source_parents: tuple[MarketEvent, ...],
    ) -> None:
        parent_by_kind = {parent.kind: parent for parent in source_parents}
        qualified = parent_by_kind[EventKind.QUALIFIED_BOS]
        leg = parent_by_kind[EventKind.STRUCTURAL_LEG_CREATED]
        swing = parent_by_kind[EventKind.SWING_CONFIRMED]
        bos_id = EventStore._required_authoritative_text(event, "bos_id")
        structure_id = EventStore._required_authoritative_text(
            event, "structure_id"
        )
        leg_id = EventStore._required_authoritative_text(
            event, "origin_leg_id"
        )
        protected_id = EventStore._required_authoritative_text(
            event, "protected_swing_id"
        )
        if (
            event.direction not in {Direction.LONG, Direction.SHORT}
            or qualified.direction is not event.direction
            or leg.direction is not event.direction
            or event.timeframe is not qualified.timeframe
            or event.timeframe is not leg.timeframe
            or event.timeframe is not swing.timeframe
            or event.known_at != qualified.known_at
            or event.event_time != swing.event_time
            or event.price != swing.price
            or event.side
            != ("below" if event.direction is Direction.LONG else "above")
            or qualified.evidence.get("bos_id") != bos_id
            or qualified.source_entity_ids != (bos_id, structure_id)
            or leg.evidence.get("leg_id") != leg_id
            or leg.evidence.get("start_swing_id") != protected_id
            or swing.evidence.get("source_entity_id") != protected_id
            or event.source_entity_ids
            != (bos_id, structure_id, leg_id, protected_id)
            or qualified.evidence.get("scope") != "continuation"
            or swing.side
            != ("below" if event.direction is Direction.LONG else "above")
            or event.evidence.get("break_standard")
            != "later_acceptance_beyond"
        ):
            raise ValueError(
                "authoritative protected-swing direction, timeframe, endpoint, "
                "or entity cross-link failed"
            )

    @staticmethod
    def _validate_protected_acceptance_context(
        event: MarketEvent,
        *,
        available_events: Mapping[str, MarketEvent],
        latest_protected_assignment_event_ids: Mapping[str, str],
        latest_protected_assignment_event_ids_by_timeframe: Mapping[
            Timeframe, str
        ],
    ) -> None:
        has_id = "protected_swing_id" in event.evidence
        has_event_id = "protected_swing_event_id" in event.evidence
        if not has_id and not has_event_id:
            return
        if has_id != has_event_id:
            raise ValueError(
                "authoritative protected acceptance context is incomplete"
            )
        protected_id = EventStore._required_authoritative_text(
            event, "protected_swing_id"
        )
        assignment_id = EventStore._required_authoritative_text(
            event, "protected_swing_event_id"
        )
        level_id = EventStore._required_authoritative_text(
            event, "level_id"
        )
        assignment = available_events.get(assignment_id)
        if (
            level_id != f"swing:{protected_id}"
            or assignment_id not in event.context_event_ids
            or assignment is None
            or assignment.kind is not EventKind.PROTECTED_SWING_ASSIGNED
            or not assignment.is_canonical_semantic
            or assignment.evidence.get("protected_swing_id") != protected_id
            or protected_id not in assignment.source_entity_ids
            or (
                assignment.timeframe is not event.timeframe
                and (
                    event.timeframe is not Timeframe.M1
                    or event.evidence.get("source_timeframe")
                    != assignment.timeframe.value
                )
            )
            or assignment.direction not in {Direction.LONG, Direction.SHORT}
            or event.direction
            is not (
                Direction.SHORT
                if assignment.direction is Direction.LONG
                else Direction.LONG
            )
        ):
            raise ValueError(
                "authoritative acceptance does not point to its protected-"
                "swing assignment context"
            )
        if assignment is None:
            raise ValueError(
                "authoritative acceptance lacks its protected assignment"
            )
        if (
            latest_protected_assignment_event_ids_by_timeframe.get(
                assignment.timeframe
            )
            != assignment_id
        ):
            raise ValueError(
                "authoritative acceptance does not reference the latest "
                "protected-swing assignment for its owner timeframe"
            )

    @staticmethod
    def _required_crossing_text(
        event: MarketEvent,
        name: str,
    ) -> str:
        value = event.evidence.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "authoritative crossing contract failed for "
                f"{event.kind.value}: evidence.{name} must be non-empty text"
            )
        return value

    @staticmethod
    def _required_crossing_clock(
        event: MarketEvent,
        name: str,
    ) -> pd.Timestamp:
        raw_value = event.evidence.get(name)
        if not isinstance(raw_value, (str, pd.Timestamp)):
            raise ValueError(
                "authoritative crossing contract failed for "
                f"{event.kind.value}: evidence.{name} must be an aware clock"
            )
        try:
            clock = aware_timestamp(
                raw_value,
                name=f"{event.kind.value}.evidence.{name}",
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "authoritative crossing contract failed for "
                f"{event.kind.value}: evidence.{name} must be an aware clock"
            ) from error
        if pd.isna(clock):
            raise ValueError(
                "authoritative crossing contract failed for "
                f"{event.kind.value}: evidence.{name} must be an aware clock"
            )
        return clock

    @staticmethod
    def _expected_crossing_direction(side: str) -> Direction:
        if side == "above":
            return Direction.LONG
        if side == "below":
            return Direction.SHORT
        raise ValueError(
            "authoritative crossing contract failed: side must be above or below"
        )

    @staticmethod
    def _expected_crossing_generation_id(
        event: MarketEvent,
        *,
        level_id: str,
        crossed_at: pd.Timestamp,
    ) -> str:
        payload = (
            f"{event.semantic_version}|crossing-v1|"
            f"{event.timeframe.value}|{level_id}|{crossed_at.isoformat()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _authoritative_source_parents(
        event: MarketEvent,
        *,
        available_events: Mapping[str, MarketEvent],
    ) -> tuple[MarketEvent, ...]:
        parents: list[MarketEvent] = []
        for event_id in event.source_event_ids:
            parent = available_events.get(event_id)
            if parent is None:
                raise ValueError(
                    "authoritative crossing contract failed for "
                    f"{event.kind.value}: unavailable transitive source "
                    f"{event_id}"
                )
            parents.append(parent)
        return tuple(parents)

    @staticmethod
    def _validate_penetration_contract(
        event: MarketEvent,
        *,
        source_parents: tuple[MarketEvent, ...],
        available_events: Mapping[str, MarketEvent],
    ) -> None:
        """Bind one penetration to its candidate, touch, and crossing bar."""

        parent_by_kind = {parent.kind: parent for parent in source_parents}
        candidate = parent_by_kind[EventKind.LIQUIDITY_LEVEL_CREATED]
        touch = parent_by_kind[EventKind.LEVEL_TOUCHED]
        crossing_bar = parent_by_kind[EventKind.BAR_COMPLETED]
        touch_parents = EventStore._authoritative_source_parents(
            touch,
            available_events=available_events,
        )
        EventStore._require_source_kind_multiset(
            touch,
            touch_parents,
            (
                EventKind.LIQUIDITY_LEVEL_CREATED,
                EventKind.BAR_COMPLETED,
            ),
        )
        EventStore._require_authoritative_parent_origins(
            touch,
            touch_parents,
        )
        EventStore._validate_level_touch_contract(
            touch,
            source_parents=touch_parents,
        )
        touch_by_kind = {parent.kind: parent for parent in touch_parents}
        if (
            touch_by_kind[EventKind.LIQUIDITY_LEVEL_CREATED].event_id
            != candidate.event_id
            or touch_by_kind[EventKind.BAR_COMPLETED].event_id
            != crossing_bar.event_id
        ):
            raise ValueError(
                "authoritative crossing contract failed for "
                "level_penetrated: touch does not bind the same candidate "
                "and crossing bar"
            )

        level_id = EventStore._required_crossing_text(
            event,
            "level_id",
        )
        candidate_level_id = EventStore._required_crossing_text(
            candidate,
            "level_id",
        )
        touch_level_id = EventStore._required_crossing_text(
            touch,
            "level_id",
        )
        if candidate_level_id != level_id or touch_level_id != level_id:
            raise ValueError(
                "authoritative crossing contract failed for "
                "level_penetrated: candidate, touch, and penetration "
                "level_id values differ"
            )

        crossed_at = EventStore._required_crossing_clock(
            event,
            "crossed_at",
        )
        generation_id = EventStore._required_crossing_text(
            event,
            "crossing_generation_id",
        )
        expected_generation_id = (
            EventStore._expected_crossing_generation_id(
                event,
                level_id=level_id,
                crossed_at=crossed_at,
            )
        )
        if generation_id != expected_generation_id:
            raise ValueError(
                "authoritative crossing contract failed for "
                "level_penetrated: crossing_generation_id does not bind "
                "semantic version, level, timeframe, and crossed_at"
            )

        if (
            event.event_time != crossed_at
            or touch.event_time != crossed_at
            or crossing_bar.event_time != crossed_at
            or crossing_bar.known_at != crossed_at
        ):
            raise ValueError(
                "authoritative crossing contract failed for "
                "level_penetrated: crossing clocks differ"
            )
        if (
            event.timeframe is not touch.timeframe
            or event.timeframe is not crossing_bar.timeframe
            or (
                candidate.timeframe is not event.timeframe
                and (
                    event.timeframe is not Timeframe.M1
                    or event.evidence.get("source_timeframe")
                    != candidate.timeframe.value
                )
            )
        ):
            raise ValueError(
                "authoritative crossing contract failed for "
                "level_penetrated: touch, penetration, and crossing bar "
                "timeframes differ"
            )
        if (
            event.side not in {"above", "below"}
            or touch.side != event.side
            or candidate.side != event.side
        ):
            raise ValueError(
                "authoritative crossing contract failed for "
                "level_penetrated: candidate, touch, and penetration sides "
                "differ"
            )
        expected_direction = (
            EventStore._expected_crossing_direction(event.side)
        )
        if event.direction is not expected_direction:
            raise ValueError(
                "authoritative crossing contract failed for "
                "level_penetrated: direction does not match crossing side"
            )
        EventStore._require_real_normalized_bar(
            crossing_bar,
            timeframe=event.timeframe,
            contract="level penetration",
        )

    @staticmethod
    def _validate_crossing_terminal_contract(
        event: MarketEvent,
        *,
        source_parents: tuple[MarketEvent, ...],
        available_events: Mapping[str, MarketEvent],
    ) -> None:
        """Bind a sweep/acceptance to one closed penetration generation."""

        parent_by_kind = {parent.kind: parent for parent in source_parents}
        penetration = parent_by_kind[EventKind.LEVEL_PENETRATED]
        resolution_bar = parent_by_kind[EventKind.BAR_COMPLETED]
        penetration_parents = (
            EventStore._authoritative_source_parents(
                penetration,
                available_events=available_events,
            )
        )
        EventStore._require_source_kind_multiset(
            penetration,
            penetration_parents,
            _EXACT_AUTHORITATIVE_SOURCE_KINDS[
                EventKind.LEVEL_PENETRATED
            ],
        )
        EventStore._require_authoritative_parent_origins(
            penetration,
            penetration_parents,
        )
        EventStore._validate_penetration_contract(
            penetration,
            source_parents=penetration_parents,
            available_events=available_events,
        )
        penetration_by_kind = {
            parent.kind: parent for parent in penetration_parents
        }
        candidate = penetration_by_kind[EventKind.LIQUIDITY_LEVEL_CREATED]

        level_id = EventStore._required_crossing_text(
            event,
            "level_id",
        )
        penetration_level_id = EventStore._required_crossing_text(
            penetration,
            "level_id",
        )
        generation_id = EventStore._required_crossing_text(
            event,
            "crossing_generation_id",
        )
        penetration_generation_id = (
            EventStore._required_crossing_text(
                penetration,
                "crossing_generation_id",
            )
        )
        crossed_at = EventStore._required_crossing_clock(
            event,
            "crossed_at",
        )
        penetration_crossed_at = (
            EventStore._required_crossing_clock(
                penetration,
                "crossed_at",
            )
        )
        resolved_at = EventStore._required_crossing_clock(
            event,
            "resolved_at",
        )
        if (
            level_id != penetration_level_id
            or generation_id != penetration_generation_id
            or crossed_at != penetration_crossed_at
        ):
            raise ValueError(
                "authoritative crossing contract failed for "
                f"{event.kind.value}: terminal and penetration generation "
                "identity differ"
            )
        expected_generation_id = (
            EventStore._expected_crossing_generation_id(
                event,
                level_id=level_id,
                crossed_at=crossed_at,
            )
        )
        if generation_id != expected_generation_id:
            raise ValueError(
                "authoritative crossing contract failed for "
                f"{event.kind.value}: crossing_generation_id does not bind "
                "semantic version, level, timeframe, and crossed_at"
            )
        if resolved_at < crossed_at:
            raise ValueError(
                "authoritative crossing contract failed for "
                f"{event.kind.value}: resolved_at predates crossed_at"
            )
        if (
            event.event_time != crossed_at
            or event.known_at != resolved_at
            or resolution_bar.event_time != resolved_at
            or resolution_bar.known_at != resolved_at
        ):
            raise ValueError(
                "authoritative crossing contract failed for "
                f"{event.kind.value}: terminal and resolution-bar clocks "
                "differ"
            )
        if (
            event.timeframe is not penetration.timeframe
            or event.timeframe is not resolution_bar.timeframe
        ):
            raise ValueError(
                "authoritative crossing contract failed for "
                f"{event.kind.value}: penetration, terminal, and resolution "
                "bar timeframes differ"
            )
        if (
            event.side not in {"above", "below"}
            or event.side != penetration.side
        ):
            raise ValueError(
                "authoritative crossing contract failed for "
                f"{event.kind.value}: terminal and penetration sides differ"
            )
        crossing_direction = (
            EventStore._expected_crossing_direction(event.side)
        )
        expected_direction = (
            crossing_direction
            if event.kind is EventKind.ACCEPTANCE_CONFIRMED
            else (
                Direction.SHORT
                if crossing_direction is Direction.LONG
                else Direction.LONG
            )
        )
        if event.direction is not expected_direction:
            raise ValueError(
                "authoritative crossing contract failed for "
                f"{event.kind.value}: direction does not match kind and side"
            )
        EventStore._require_real_normalized_bar(
            resolution_bar,
            timeframe=event.timeframe,
            contract=f"{event.kind.value} resolution",
        )
        if event.source_entity_ids != (level_id,):
            raise ValueError(
                "authoritative crossing terminal does not bind its level "
                "entity identity"
            )
        if candidate.zone is None:
            raise ValueError(
                "authoritative crossing terminal candidate lacks frozen "
                "geometry"
            )
        resolved_close = EventStore._bar_number(
            resolution_bar,
            "close",
        )
        lower, upper = candidate.zone
        accepted_outside = (
            resolved_close > upper
            if event.side == "above"
            else resolved_close < lower
        )
        if accepted_outside != (
            event.kind is EventKind.ACCEPTANCE_CONFIRMED
        ):
            raise ValueError(
                "authoritative crossing terminal kind conflicts with the "
                "exact resolution BAR close and frozen candidate geometry"
            )

    @staticmethod
    def _origin_zone_bar_scope(bar: MarketEvent) -> tuple[str, int]:
        symbol = bar.evidence.get("symbol")
        instrument_id = bar.evidence.get("instrument_id")
        if (
            not isinstance(symbol, str)
            or not symbol.strip()
            or type(instrument_id) is not int
            or instrument_id < 0
        ):
            raise ValueError(
                "authoritative origin-zone terminal contract requires "
                "explicit BAR symbol and instrument_id"
            )
        return symbol, instrument_id

    @staticmethod
    def _origin_zone_bar_price(
        bar: MarketEvent,
        name: str,
    ) -> float:
        value = bar.evidence.get(name)
        if isinstance(value, bool):
            raise ValueError(
                "authoritative origin-zone terminal contract requires "
                f"finite BAR {name}"
            )
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "authoritative origin-zone terminal contract requires "
                f"finite BAR {name}"
            ) from error
        if not math.isfinite(result):
            raise ValueError(
                "authoritative origin-zone terminal contract requires "
                f"finite BAR {name}"
            )
        return result

    @staticmethod
    def _validate_origin_zone_terminal_contract(
        event: MarketEvent,
        *,
        source_parents: tuple[MarketEvent, ...],
        available_events: Mapping[str, MarketEvent],
    ) -> None:
        """Bind one terminal Origin Zone transition to its frozen zone/bar."""

        EventStore._require_source_kind_multiset(
            event,
            source_parents,
            _ORIGIN_ZONE_TERMINAL_SOURCE_KINDS,
        )
        EventStore._require_authoritative_parent_origins(
            event,
            source_parents,
        )
        parent_by_kind = {parent.kind: parent for parent in source_parents}
        created = parent_by_kind[EventKind.QUALIFIED_ORIGIN_ZONE_CREATED]
        transition_bar = parent_by_kind[EventKind.BAR_COMPLETED]

        created_parents = EventStore._authoritative_source_parents(
            created,
            available_events=available_events,
        )
        actual_created_kinds = Counter(
            parent.kind for parent in created_parents
        )
        if (
            any(
                actual_created_kinds[kind] < 1
                for kind in _ORIGIN_ZONE_SOURCE_KINDS
            )
            or any(
                kind not in _ORIGIN_ZONE_SOURCE_KINDS
                for kind in actual_created_kinds
            )
        ):
            raise ValueError(
                "authoritative origin-zone terminal contract references "
                "an invalid QUALIFIED_ORIGIN_ZONE_CREATED parent"
            )
        EventStore._require_authoritative_parent_origins(
            created,
            created_parents,
        )

        zone_id = event.evidence.get("origin_zone_id")
        created_zone_id = created.evidence.get("origin_zone_id")
        if (
            not isinstance(zone_id, str)
            or not zone_id.strip()
            or zone_id != created_zone_id
            or event.source_entity_ids != (zone_id,)
            or not created.source_entity_ids
            or created.source_entity_ids[0] != zone_id
        ):
            raise ValueError(
                "authoritative origin-zone terminal contract has a "
                "mismatched zone identity"
            )

        if (
            event.timeframe is not Timeframe.M5
            or created.timeframe is not Timeframe.M5
            or transition_bar.timeframe is not Timeframe.M5
            or any(
                parent.timeframe is not Timeframe.M5
                for parent in created_parents
            )
            or event.direction is None
            or event.direction is not created.direction
            or event.side != created.side
            or event.side
            != (
                "below"
                if event.direction is Direction.LONG
                else "above"
            )
        ):
            raise ValueError(
                "authoritative origin-zone terminal contract has a "
                "mismatched timeframe, direction, or side"
            )

        if (
            created.known_at >= event.known_at
            or event.event_time != event.known_at
            or transition_bar.event_time != event.known_at
            or transition_bar.known_at != event.known_at
        ):
            raise ValueError(
                "authoritative origin-zone terminal contract has "
                "inconsistent creation or transition clocks"
            )
        if (
            transition_bar.evidence.get("real_completed") is not True
            or transition_bar.evidence.get("clock_only") is not False
        ):
            raise ValueError(
                "authoritative origin-zone terminal contract requires a "
                "real completed transition BAR"
            )

        transition_scope = EventStore._origin_zone_bar_scope(
            transition_bar
        )
        # The anchor candles now hang off the base origin core rather than
        # off the qualification, so the scope check follows that one hop.
        core = next(
            (
                parent
                for parent in created_parents
                if parent.kind is EventKind.BASE_ORIGIN_CORE_CREATED
            ),
            None,
        )
        anchor_bars = (
            ()
            if core is None
            else tuple(
                parent
                for parent in EventStore._authoritative_source_parents(
                    core,
                    available_events=available_events,
                )
                if parent.kind is EventKind.BAR_COMPLETED
            )
        )
        if (
            not anchor_bars
            or any(
                EventStore._origin_zone_bar_scope(anchor)
                != transition_scope
                for anchor in anchor_bars
            )
        ):
            raise ValueError(
                "authoritative origin-zone terminal contract crosses "
                "symbol or instrument scope"
            )

        if (
            created.zone is None
            or event.zone is None
            or event.zone != created.zone
            or created.price is None
            or event.price is None
            or not math.isclose(
                float(event.price),
                float(created.price),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(event.strength),
                float(created.strength),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "authoritative origin-zone terminal contract changed its "
                "frozen geometry"
            )
        lower, upper = event.zone
        midpoint = (lower + upper) / 2.0
        if (
            lower >= upper
            or not math.isclose(
                float(event.price),
                midpoint,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "authoritative origin-zone terminal contract has invalid "
                "frozen bounds or midpoint"
            )

        low = EventStore._origin_zone_bar_price(
            transition_bar,
            "low",
        )
        high = EventStore._origin_zone_bar_price(
            transition_bar,
            "high",
        )
        close = EventStore._origin_zone_bar_price(
            transition_bar,
            "close",
        )
        if low > high or not low <= close <= high:
            raise ValueError(
                "authoritative origin-zone terminal contract received an "
                "invalid transition BAR range"
            )
        intersects = low <= upper and high >= lower
        failed = (
            close < lower
            if event.direction is Direction.LONG
            else close > upper
        )
        expected_lifecycle, expected_reason = (
            ("mitigated", "zone_intersected")
            if event.kind is EventKind.ORIGIN_ZONE_MITIGATED
            else ("failed", "close_through_distal_edge")
        )
        if (
            event.evidence.get("lifecycle") != expected_lifecycle
            or event.evidence.get("transition_reason") != expected_reason
            or (
                event.kind is EventKind.ORIGIN_ZONE_MITIGATED
                and (not intersects or failed)
            )
            or (
                event.kind is EventKind.ORIGIN_ZONE_INVALIDATED
                and not failed
            )
        ):
            raise ValueError(
                "authoritative origin-zone terminal contract conflicts "
                "with its BAR geometry or lifecycle"
            )

    @staticmethod
    def _validate_external_range_invalidation(
        event: MarketEvent,
        *,
        source_parents: tuple[MarketEvent, ...],
    ) -> None:
        EventStore._require_source_kind_multiset(
            event,
            source_parents,
            _RANGE_INVALIDATION_SOURCE_KINDS,
        )
        EventStore._require_authoritative_parent_origins(
            event,
            source_parents,
        )
        parent_by_kind = {parent.kind: parent for parent in source_parents}
        range_id = EventStore._validate_range_break_geometry(
            event,
            bar=parent_by_kind[EventKind.BAR_COMPLETED],
        )
        for kind in (
            EventKind.DEALING_RANGE_CREATED,
            EventKind.BALANCE_RANGE_MATURED,
            EventKind.ACCEPTANCE_CONFIRMED,
        ):
            parent = parent_by_kind[kind]
            if parent.timeframe is not Timeframe.H1:
                raise ValueError(
                    "close-beyond dealing-range parent must be H1: "
                    f"{parent.event_id} ({kind.value})"
                )
            if parent.details.get("range_id") != range_id:
                raise ValueError(
                    "close-beyond dealing-range parent range_id differs: "
                    f"{parent.event_id} ({kind.value})"
                )

    @staticmethod
    def _validate_forming_range_invalidation(
        event: MarketEvent,
        *,
        source_parents: tuple[MarketEvent, ...],
    ) -> None:
        EventStore._require_source_kind_multiset(
            event,
            source_parents,
            _FORMING_RANGE_INVALIDATION_SOURCE_KINDS,
        )
        EventStore._require_authoritative_parent_origins(
            event,
            source_parents,
        )
        parent_by_kind = {parent.kind: parent for parent in source_parents}
        range_id = EventStore._validate_range_break_geometry(
            event,
            bar=parent_by_kind[EventKind.BAR_COMPLETED],
        )
        created = parent_by_kind[EventKind.DEALING_RANGE_CREATED]
        if created.timeframe is not Timeframe.H1:
            raise ValueError(
                "close-beyond dealing-range parent must be H1: "
                f"{created.event_id} ({created.kind.value})"
            )
        if created.details.get("range_id") != range_id:
            raise ValueError(
                "close-beyond dealing-range parent range_id differs: "
                f"{created.event_id} ({created.kind.value})"
            )

    @staticmethod
    def _validate_range_break_geometry(
        event: MarketEvent,
        *,
        bar: MarketEvent,
    ) -> str:
        if event.timeframe is not Timeframe.H1:
            raise ValueError(
                "close-beyond dealing-range invalidation must be H1"
            )
        range_id = event.details.get("range_id")
        if not isinstance(range_id, str) or not range_id.strip():
            raise ValueError(
                "close-beyond dealing-range invalidation requires range_id"
            )
        if bar.timeframe is not Timeframe.H1:
            raise ValueError(
                "close-beyond dealing-range invalidation requires an H1 BAR"
            )
        close_value = bar.details.get("close")
        if isinstance(close_value, bool):
            raise ValueError(
                "close-beyond dealing-range H1 BAR close is invalid"
            )
        try:
            close = float(close_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "close-beyond dealing-range H1 BAR close is invalid"
            ) from error
        if not math.isfinite(close):
            raise ValueError(
                "close-beyond dealing-range H1 BAR close is invalid"
            )
        if event.zone is None:
            raise ValueError(
                "close-beyond dealing-range invalidation requires a frozen "
                "range zone"
            )
        lower, upper = event.zone
        if not (close < lower or close > upper):
            raise ValueError(
                "close-beyond dealing-range H1 BAR close must be strictly "
                "outside the frozen range zone"
            )
        return range_id

    @staticmethod
    def _provenance_reaches(
        start_event_id: str,
        target_event_id: str,
        *,
        available_events: Mapping[str, MarketEvent],
    ) -> bool:
        """Return whether stored event ancestry reaches ``target_event_id``."""

        pending = [start_event_id]
        visited: set[str] = set()
        while pending:
            event_id = pending.pop()
            if event_id in visited:
                continue
            visited.add(event_id)
            parent = available_events.get(event_id)
            if parent is None:
                continue
            for reference_id in (
                *parent.source_event_ids,
                *parent.context_event_ids,
            ):
                if reference_id == target_event_id:
                    return True
                if reference_id not in visited:
                    pending.append(reference_id)
        return False

    def events_since(self, index: int) -> tuple[MarketEvent, ...]:
        """Return the append-only suffix beginning at ``index``."""

        if type(index) is not int or not 0 <= index <= len(self._events):
            raise ValueError("event store suffix index is out of range")
        return tuple(self._events[index:])

    def get(self, event_id: str) -> MarketEvent | None:
        return self._by_id.get(event_id)

    def normalized_bar_at(
        self,
        timeframe: Timeframe,
        known_at: pd.Timestamp,
    ) -> MarketEvent | None:
        """Return the exact normalized BAR registered at one native clock."""

        native_timeframe = Timeframe(timeframe)
        clock = aware_timestamp(known_at, name="normalized_bar_at.known_at")
        event_id = self._normalized_bar_event_ids.get((native_timeframe, clock))
        if event_id is None:
            return None
        event = self._by_id.get(event_id)
        if (
            event is None
            or event.kind is not EventKind.BAR_COMPLETED
            or event.origin is not EventOrigin.NORMALIZED_DATA
            or event.timeframe is not native_timeframe
            or event.known_at != clock
        ):
            raise ValueError("normalized BAR index differs from EventStore history")
        return event

    def event_digest(self, event_id: str) -> str:
        """Return the exact immutable digest for one committed event."""

        try:
            return self._digests[event_id]
        except KeyError as error:
            raise KeyError(f"event store has no event: {event_id}") from error

    def recompute_event_digest(self, event: MarketEvent) -> str:
        """Digest supplied bytes for comparison with the same audit identity."""

        if not isinstance(event, MarketEvent):
            raise TypeError("event digest requires a MarketEvent")
        if event.semantic_version != self.semantic_version:
            raise ValueError("event digest semantic version differs")
        return _event_digest(event)

    def events(
        self,
        *,
        known_at: pd.Timestamp | None = None,
    ) -> tuple[MarketEvent, ...]:
        if known_at is None:
            return tuple(self._events)
        cutoff = aware_timestamp(known_at, name="event_store.known_at")
        return tuple(event for event in self._events if event.known_at <= cutoff)

    def fingerprint(self, *, known_at: pd.Timestamp | None = None) -> str:
        if known_at is None:
            return self._full_prefix_fingerprint_hasher.copy().hexdigest()
        return _fingerprint_from_digests(
            semantic_version=self.semantic_version,
            definition_identity=self.semantic_definition_identity,
            digests=(
                self._digests[event.event_id]
                for event in self.events(known_at=known_at)
            ),
        )

    def prefix_fingerprint(self, event_count: int) -> str:
        """Return a checkpoint-only commitment to an exact stored prefix.

        Normal hot consumers use the cached full ``fingerprint`` after they
        consume the current suffix.  This bounded restore/checkpoint helper
        exists for the valid case where projection transport was appended
        after the last physical-state publication.
        """

        if type(event_count) is not int or not 0 <= event_count <= len(self):
            raise ValueError("event store prefix count is out of range")
        if event_count == len(self):
            return self.fingerprint()
        return _fingerprint_from_digests(
            semantic_version=self.semantic_version,
            definition_identity=self.semantic_definition_identity,
            digests=(
                self._digests[event.event_id]
                for event in islice(self._events, event_count)
            ),
        )

    def metadata(self) -> dict[str, Any]:
        """Stable identity metadata for manifests and observational reports."""

        return {
            "schema_version": 1,
            "semantic_version": self.semantic_version,
            "semantic_definition_identity": self.semantic_definition_identity,
            "semantic_definition": (
                None
                if self.definition_identity is None
                else self.definition_identity.to_metadata()
            ),
        }

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Content-bound metadata for an external replay checkpoint."""

        self._require_committed_integrity()
        last_known_at = self._events[-1].known_at if self._events else None
        return {
            **self.metadata(),
            "checkpoint_schema_version": 1,
            "event_count": len(self),
            "last_known_at": (
                None if last_known_at is None else last_known_at.isoformat()
            ),
            "event_fingerprint": self.fingerprint(),
        }

    def replay(
        self,
        initial_state: StateT,
        reducer: Reducer[StateT],
        *,
        known_at: pd.Timestamp | None = None,
    ) -> ReplayResult[StateT]:
        if not callable(reducer):
            raise TypeError("event replay reducer must be callable")
        available = self.events(known_at=known_at)
        state = initial_state
        for event in available:
            state = reducer(state, event)
        return ReplayResult(
            state=state,
            events_applied=len(available),
            last_known_at=(available[-1].known_at if available else None),
            event_fingerprint=self.fingerprint(known_at=known_at),
        )

    @classmethod
    def from_events(
        cls,
        events: Iterable[MarketEvent],
        *,
        semantic_version: str = SMC_SEMANTIC_VERSION,
        definition_identity: (
            SemanticDefinitionIdentity | Mapping[str, Any] | str | None
        ) = None,
    ) -> "EventStore":
        store = cls(
            semantic_version=semantic_version,
            definition_identity=definition_identity,
        )
        store.append_batch(events)
        return store

    @classmethod
    def from_checkpoint(
        cls,
        events: Iterable[MarketEvent],
        checkpoint_metadata: Mapping[str, Any],
        *,
        expected_definition_identity: (
            SemanticDefinitionIdentity | Mapping[str, Any] | str | None
        ) = None,
    ) -> "EventStore":
        """Restore events only when all external checkpoint bindings agree."""

        expected_fields = {
            "schema_version",
            "semantic_version",
            "semantic_definition_identity",
            "semantic_definition",
            "checkpoint_schema_version",
            "event_count",
            "last_known_at",
            "event_fingerprint",
        }
        if (
            not isinstance(checkpoint_metadata, Mapping)
            or set(checkpoint_metadata) != expected_fields
            or checkpoint_metadata.get("schema_version") != 1
            or checkpoint_metadata.get("checkpoint_schema_version") != 1
        ):
            raise ValueError("event store checkpoint metadata is invalid")
        semantic_version = checkpoint_metadata.get("semantic_version")
        if not isinstance(semantic_version, str) or not semantic_version:
            raise ValueError("event store checkpoint semantic version is invalid")
        definition_metadata = checkpoint_metadata.get("semantic_definition")
        recorded_digest = checkpoint_metadata.get(
            "semantic_definition_identity"
        )
        definition_binding: SemanticDefinitionIdentity | str | None
        if definition_metadata is not None:
            try:
                definition_binding = SemanticDefinitionIdentity.from_metadata(
                    definition_metadata
                )
            except SemanticRegistryError as error:
                raise ValueError(
                    "event store checkpoint definition metadata is invalid"
                ) from error
            if definition_binding.identity != recorded_digest:
                raise ValueError(
                    "event store checkpoint definition digest is invalid"
                )
        else:
            definition_binding = (
                None
                if recorded_digest is None
                else _require_sha256(
                    recorded_digest,
                    name="checkpoint semantic definition identity",
                )
            )
        store = cls.from_events(
            events,
            semantic_version=semantic_version,
            definition_identity=definition_binding,
        )
        if expected_definition_identity is not None:
            store.require_definition_identity(expected_definition_identity)
        if (
            checkpoint_metadata.get("event_count") != len(store)
            or checkpoint_metadata.get("event_fingerprint")
            != store.fingerprint()
            or checkpoint_metadata.get("last_known_at")
            != (
                None
                if not store._events
                else store._events[-1].known_at.isoformat()
            )
        ):
            raise ValueError("event store checkpoint content binding is invalid")
        return store


def validate_canonical_event(
    event: MarketEvent,
    *,
    available_events: Mapping[str, MarketEvent],
    normalized_bar_event_ids: (
        Mapping[tuple[Timeframe, pd.Timestamp], str] | None
    ) = None,
    latest_protected_assignment_event_ids: Mapping[str, str] | None = None,
    latest_protected_assignment_event_ids_by_timeframe: (
        Mapping[Timeframe, str] | None
    ) = None,
    unresolved_forward_reference_ids: set[str] | None = None,
) -> None:
    """Validate one reducer input against the store's authority contract."""

    _require_exact_market_event(event)
    EventStore._validate_canonical_provenance(
        event,
        available_events=available_events,
        normalized_bar_event_ids=normalized_bar_event_ids,
        latest_protected_assignment_event_ids=(
            latest_protected_assignment_event_ids
        ),
        latest_protected_assignment_event_ids_by_timeframe=(
            latest_protected_assignment_event_ids_by_timeframe
        ),
        unresolved_forward_reference_ids=unresolved_forward_reference_ids,
    )


@dataclass(frozen=True)
class EventJournalManifest:
    path: Path
    manifest_sha256: str
    semantic_version: str
    semantic_definition_identity: str
    rows: int
    logical_fingerprint: str
    shard_sha256: tuple[str, ...]


@dataclass(frozen=True)
class EventJournalReadResult:
    store: EventStore
    manifest: EventJournalManifest

    @property
    def logical_fingerprint(self) -> str:
        return self.manifest.logical_fingerprint


def _journal_paths(
    destination: str | Path,
    *,
    stream_name: str,
) -> tuple[Path, Path, Path]:
    if (
        not isinstance(stream_name, str)
        or not stream_name
        or "/" in stream_name
        or stream_name in {".", ".."}
    ):
        raise ValueError("invalid event journal stream name")
    root = Path(destination)
    manifest = root / f"{stream_name}.manifest.json"
    sidecar = root / f"{stream_name}.manifest.sha256"
    shard_root = root / stream_name
    return manifest, sidecar, shard_root


def _journal_binding(
    store: EventStore,
    *,
    events: tuple[MarketEvent, ...],
) -> tuple[dict[str, Any], str]:
    definition_digest = store.semantic_definition_identity
    definition = store.definition_identity
    if definition_digest is None or definition is None:
        raise ValueError(
            "event journal requires a complete semantic definition identity"
        )
    logical_fingerprint = _fingerprint_from_digests(
        semantic_version=store.semantic_version,
        definition_identity=definition_digest,
        digests=(store._digests[event.event_id] for event in events),
    )
    binding = {
        "journal_format_version": _EVENT_JOURNAL_FORMAT_VERSION,
        "semantic_version": store.semantic_version,
        "semantic_definition_identity": definition_digest,
        "semantic_definition": definition.to_metadata(),
        "event_order": "known_at,sequence_no,event_id",
        "market_event_schema_sha256": _market_event_schema_sha256(),
        "logical_fingerprint": logical_fingerprint,
        "first_event_id": None if not events else events[0].event_id,
        "last_event_id": None if not events else events[-1].event_id,
    }
    return binding, logical_fingerprint


def write_event_journal(
    destination: str | Path,
    store: EventStore,
    *,
    maximum_rows_per_shard: int = 100_000,
    stream_name: str = _EVENT_JOURNAL_STREAM,
) -> EventJournalManifest:
    """Write one immutable, definition-bound Parquet event journal.

    The commit marker is the manifest, written only after every bounded shard
    is durable and verified. Existing shard or manifest paths are never
    overwritten; a failed partial attempt must be inspected or written to a
    new destination.
    """

    if not isinstance(store, EventStore):
        raise TypeError("event journal writer requires EventStore")
    if type(maximum_rows_per_shard) is not int or maximum_rows_per_shard < 1:
        raise ValueError("maximum_rows_per_shard must be positive")
    manifest_path, sidecar_path, shard_root = _journal_paths(
        destination,
        stream_name=stream_name,
    )
    if manifest_path.exists() or sidecar_path.exists() or shard_root.exists():
        raise FileExistsError(
            "event journal target already exists and is immutable"
        )

    events = store.events()
    binding, logical_fingerprint = _journal_binding(store, events=events)
    stream_state = new_stream_state(_EVENT_JOURNAL_FIELD_TYPES)
    buffer: list[dict[str, Any]] = []
    for ordinal, event in enumerate(events):
        buffer.append(
            {
                "ordinal": ordinal,
                "event_id": event.event_id,
                "known_at": event.known_at.tz_convert("UTC"),
                "sequence_no": event.sequence_no,
                "event_digest": store._digests[event.event_id],
                "event_json": _event_json(event),
            }
        )
        if len(buffer) >= maximum_rows_per_shard:
            write_stream_shard(
                destination,
                stream_name,
                buffer,
                stream_state,
                key_column="ordinal",
                field_types=_EVENT_JOURNAL_FIELD_TYPES,
            )
    if buffer:
        write_stream_shard(
            destination,
            stream_name,
            buffer,
            stream_state,
            key_column="ordinal",
            field_types=_EVENT_JOURNAL_FIELD_TYPES,
        )

    # A concurrent append must not silently produce a prefix journal bearing
    # metadata from a later store state.
    if len(store) != len(events) or store.fingerprint() != logical_fingerprint:
        raise RuntimeError("event store changed while its journal was written")
    manifest_path = write_stream_manifest(
        destination,
        stream_name,
        stream_state,
        artifact=_EVENT_JOURNAL_ARTIFACT,
        bindings=binding,
    )
    manifest_sha256 = sha256_file(manifest_path)
    atomic_bytes(sidecar_path, f"{manifest_sha256}\n".encode("ascii"))
    return EventJournalManifest(
        path=manifest_path,
        manifest_sha256=manifest_sha256,
        semantic_version=store.semantic_version,
        semantic_definition_identity=str(
            store.semantic_definition_identity
        ),
        rows=len(events),
        logical_fingerprint=logical_fingerprint,
        shard_sha256=tuple(
            str(shard["sha256"])
            for shard in stream_state["committed_shards"]
        ),
    )


def _read_journal_manifest_payload(
    destination: str | Path,
    *,
    stream_name: str,
) -> tuple[Path, str, Mapping[str, Any]]:
    manifest_path, sidecar_path, _ = _journal_paths(
        destination,
        stream_name=stream_name,
    )
    if (
        manifest_path.is_symlink()
        or sidecar_path.is_symlink()
        or not manifest_path.is_file()
        or not sidecar_path.is_file()
    ):
        raise ValueError("event journal manifest or hash sidecar is missing")
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise ValueError("event journal manifest cannot be read") from error
    actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        recorded_sha256 = sidecar_path.read_text(encoding="ascii").strip()
    except OSError as error:
        raise ValueError("event journal manifest hash cannot be read") from error
    if actual_sha256 != recorded_sha256:
        raise ValueError("event journal manifest hash is invalid")
    try:
        payload = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("event journal manifest is invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("event journal manifest must be an object")
    if canonical_json(payload) != manifest_bytes:
        raise ValueError("event journal manifest is not canonical JSON")
    return manifest_path, actual_sha256, payload


def _validate_journal_manifest(
    destination: str | Path,
    payload: Mapping[str, Any],
    *,
    stream_name: str,
    expected_definition_identity: (
        SemanticDefinitionIdentity | Mapping[str, Any] | str | None
    ),
) -> tuple[Mapping[str, Any], SemanticDefinitionIdentity, str]:
    required = {
        "format_version",
        "artifact",
        "status",
        "stream",
        "rows",
        "shards",
        "bindings",
        "schema_fingerprint",
        "field_types",
    }
    if set(payload) != required:
        raise ValueError("event journal manifest fields differ from format v1")
    if (
        payload.get("format_version") != 1
        or payload.get("artifact") != _EVENT_JOURNAL_ARTIFACT
        or payload.get("status") != "complete"
        or payload.get("stream") != stream_name
        or payload.get("field_types") != dict(_EVENT_JOURNAL_FIELD_TYPES)
    ):
        raise ValueError("event journal manifest contract is invalid")
    bindings = payload.get("bindings")
    expected_binding_fields = {
        "journal_format_version",
        "semantic_version",
        "semantic_definition_identity",
        "semantic_definition",
        "event_order",
        "market_event_schema_sha256",
        "logical_fingerprint",
        "first_event_id",
        "last_event_id",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != expected_binding_fields:
        raise ValueError("event journal bindings are invalid")
    if (
        bindings.get("journal_format_version")
        != _EVENT_JOURNAL_FORMAT_VERSION
        or bindings.get("event_order") != "known_at,sequence_no,event_id"
        or bindings.get("market_event_schema_sha256")
        != _market_event_schema_sha256()
    ):
        raise ValueError("event journal runtime contract changed")
    semantic_version = bindings.get("semantic_version")
    if not isinstance(semantic_version, str) or not semantic_version:
        raise ValueError("event journal semantic version is invalid")
    recorded_digest = _require_sha256(
        bindings.get("semantic_definition_identity"),
        name="event journal semantic definition identity",
    )
    definition_metadata = bindings.get("semantic_definition")
    try:
        definition_binding = SemanticDefinitionIdentity.from_metadata(
            definition_metadata
        )
    except (SemanticRegistryError, TypeError) as error:
        raise ValueError(
            "event journal semantic definition metadata is invalid"
        ) from error
    if (
        definition_binding.semantic_version != semantic_version
        or definition_binding.identity != recorded_digest
    ):
        raise ValueError("event journal semantic definition binding is invalid")
    if expected_definition_identity is not None:
        _, expected_digest = _definition_binding(
            expected_definition_identity,
            semantic_version=semantic_version,
        )
        if expected_digest != recorded_digest:
            raise ValueError(
                "event journal semantic definition identity drifted"
            )

    shards = payload.get("shards")
    if (
        type(payload.get("rows")) is not int
        or int(payload["rows"]) < 0
        or not isinstance(shards, list)
    ):
        raise ValueError("event journal shard registry is invalid")
    shard_root = Path(destination) / stream_name
    if shard_root.is_symlink():
        raise ValueError("event journal shard root cannot be a symlink")
    for shard in shards:
        if (
            not isinstance(shard, Mapping)
            or set(shard)
            != {"index", "path", "rows", "first_key", "last_key", "sha256"}
            or type(shard.get("index")) is not int
            or type(shard.get("rows")) is not int
            or int(shard["rows"]) < 1
        ):
            raise ValueError("event journal shard metadata is invalid")
        _require_sha256(shard.get("sha256"), name="event journal shard hash")
        relative = PurePosixPath(str(shard.get("path", "")))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != stream_name
        ):
            raise ValueError("event journal shard path escapes its stream")
    stream_state = {
        "rows": payload.get("rows"),
        "next_shard_index": len(shards),
        "committed_shards": shards,
        "schema_fingerprint": payload.get("schema_fingerprint"),
        "field_types": payload.get("field_types"),
    }
    verify_stream_shards(destination, stream_state)
    return bindings, definition_binding, recorded_digest


def read_event_journal(
    destination: str | Path,
    *,
    expected_definition_identity: (
        SemanticDefinitionIdentity | Mapping[str, Any] | str | None
    ) = None,
    stream_name: str = _EVENT_JOURNAL_STREAM,
) -> EventJournalReadResult:
    """Verify and reconstruct one immutable Parquet event journal."""

    manifest_path, manifest_sha256, payload = _read_journal_manifest_payload(
        destination,
        stream_name=stream_name,
    )
    bindings, definition_binding, recorded_definition_digest = (
        _validate_journal_manifest(
            destination,
            payload,
            stream_name=stream_name,
            expected_definition_identity=expected_definition_identity,
        )
    )
    expected_ordinal = 0
    events: list[MarketEvent] = []
    shard_hashes: list[str] = []
    for shard in payload["shards"]:
        path = Path(destination) / str(shard["path"])
        frame = pd.read_parquet(path)
        if tuple(frame.columns) != tuple(_EVENT_JOURNAL_FIELD_TYPES):
            raise ValueError("event journal Parquet columns are invalid")
        if (
            frame.empty
            or str(frame["ordinal"].iloc[0]) != str(shard["first_key"])
            or str(frame["ordinal"].iloc[-1]) != str(shard["last_key"])
        ):
            raise ValueError("event journal shard key bounds are invalid")
        shard_hashes.append(str(shard["sha256"]))
        for row in frame.itertuples(index=False):
            if int(row.ordinal) != expected_ordinal:
                raise ValueError("event journal ordinals are not contiguous")
            event = _event_from_json(str(row.event_json))
            if (
                event.event_id != row.event_id
                or event.sequence_no != int(row.sequence_no)
                or event.known_at.tz_convert("UTC") != pd.Timestamp(row.known_at)
                or _event_digest(event) != row.event_digest
            ):
                raise ValueError("event journal row binding is invalid")
            events.append(event)
            expected_ordinal += 1
    if expected_ordinal != int(payload["rows"]):
        raise ValueError("event journal row count is invalid")
    if bindings["first_event_id"] != (
        None if not events else events[0].event_id
    ) or bindings["last_event_id"] != (
        None if not events else events[-1].event_id
    ):
        raise ValueError("event journal endpoint binding is invalid")

    store = EventStore.from_events(
        events,
        semantic_version=str(bindings["semantic_version"]),
        definition_identity=definition_binding,
    )
    logical_fingerprint = _require_sha256(
        bindings["logical_fingerprint"],
        name="event journal logical fingerprint",
    )
    if (
        store.semantic_definition_identity != recorded_definition_digest
        or store.fingerprint() != logical_fingerprint
    ):
        raise ValueError("event journal logical fingerprint is invalid")
    manifest = EventJournalManifest(
        path=manifest_path,
        manifest_sha256=manifest_sha256,
        semantic_version=store.semantic_version,
        semantic_definition_identity=recorded_definition_digest,
        rows=len(store),
        logical_fingerprint=logical_fingerprint,
        shard_sha256=tuple(shard_hashes),
    )
    return EventJournalReadResult(store=store, manifest=manifest)


# Historical pickle compatibility only.  Checkpoints written before the
# public class rename resolve this module global while every current producer
# and type annotation uses ``EventStore``.  Keep the alias out of ``__all__``
# and out of the package root so no new runtime can select the legacy name.
ImmutableEventStore = EventStore


__all__ = [
    "EventJournalManifest",
    "EventJournalReadResult",
    "EventStore",
    "Reducer",
    "ReplayResult",
    "event_order_key",
    "read_event_journal",
    "write_event_journal",
]
