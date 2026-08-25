"""Causal adapter from authoritative v1.2 facts to foundation-v2 reducers.

The adapter does not detect market semantics and does not own an event store.
It validates the already-published v1.2 facts, builds the explicit normalized
inputs required by :mod:`semantic_lifecycle`, and projects only the resulting
immutable DTO revisions through :mod:`semantic_foundation`.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .foundation_registry import FOUNDATION_VERSION
from .market_clock import (
    next_registered_native_completion,
    validate_registered_native_bar_root,
)
from .market_state import DeliveryPhase, RelationState
from .model import (
    Direction,
    EventKind,
    EventOrigin,
    MarketEvent,
    SMC_SEMANTIC_VERSION,
    Timeframe,
    aware_timestamp,
    content_hash,
    price_to_ticks,
    ticks_to_price,
)
from .semantic_foundation import (
    FoundationProjection,
    FoundationProjectionCheckpoint,
    FoundationProjectionReducer,
    FoundationRecord,
)
from .semantic_lifecycle import (
    GenerationLifecycle,
    LifecycleCheckpoint,
    LiquidityInteractionLifecycle,
    LiquidityInteractionTerminal,
    LiquidityLevelLifecycle,
    NormalizedLifecycleTransition,
    NormalizedTransitionKind,
    REARMABLE_LIQUIDITY_SOURCE_KINDS,
    SemanticLifecycleReducer,
    SemanticLifecycleState,
    StructureGenerationLifecycle,
    StructureScope,
    StructureTransitionLifecycle,
    _expected_formation_role_ledger,
    canonical_semantic_id,
)


_MAPPED_ATOMIC_KINDS = frozenset(
    {
        EventKind.MARKET_EPOCH_RESET,
        EventKind.LIQUIDITY_LEVEL_CREATED,
        EventKind.LEVEL_TOUCHED,
        EventKind.LEVEL_PENETRATED,
        EventKind.SWEEP_CONFIRMED,
        EventKind.ACCEPTANCE_CONFIRMED,
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        EventKind.QUALIFIED_BOS,
        EventKind.PROTECTED_SWING_ASSIGNED,
        EventKind.MSS_CORE_CONFIRMED,
    }
)

_RESET_REASON_MAP = {
    "contract_change_reset": "contract_reset",
    "contract_reset": "contract_reset",
    "data_gap_reset": "data_reset",
    "data_reset": "data_reset",
    "semantic_reset": "semantic_reset",
}

_RELATION_TERMINATION_REASONS = frozenset(
    {
        "child_realigned",
        "parent_invalidated",
        "parent_rollover",
        "relation_reclassified",
        "semantic_reset",
    }
)
_DELIVERY_TERMINATION_REASONS = frozenset(
    {
        "phase_changed",
        "parent_structure_terminated",
        "contract_reset",
        "data_reset",
        "semantic_reset",
    }
)
_DTO_SOURCE_ORIGINS = frozenset(
    {
        EventOrigin.NORMALIZED_DATA,
        EventOrigin.SEMANTIC_ATOMIC,
    }
)
_RETIREMENT_SOURCE_ORIGINS = frozenset(
    {EventOrigin.NORMALIZED_DATA, EventOrigin.SEMANTIC_ATOMIC}
)
_LEVEL_RETIREMENT_REASONS = frozenset(
    {
        "reference_rollover",
        "supersession",
        "contract_reset",
        "source_retired",
        "source_range_terminated",
        "structure_generation_terminated",
    }
)
_TIMEFRAME_INTERVAL = {
    Timeframe.M1: pd.Timedelta(1, unit="min"),
    Timeframe.M5: pd.Timedelta(5, unit="min"),
    Timeframe.M15: pd.Timedelta(15, unit="min"),
    Timeframe.H1: pd.Timedelta(1, unit="h"),
    Timeframe.H4: pd.Timedelta(4, unit="h"),
}
FOUNDATION_ADAPTER_CHECKPOINT_SCHEMA_VERSION = 2
FOUNDATION_ADAPTER_STATE_SCHEMA_VERSION = 2


def _unique_ids(values: Iterable[str], *, name: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(values))
    if not result or any(
        not isinstance(value, str) or not value.strip() for value in result
    ):
        raise ValueError(f"{name} requires non-empty event identities")
    return result


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"v1.2 event evidence requires {key}")
    return value


def _tradable_zone_bound_ticks(
    value: object,
    tick_size: float,
    *,
    lower: bool,
) -> int:
    """Return the inward tradable tick envelope for one continuous bound.

    Zone bounds may be ATR-derived and therefore off-grid.  Decimal ratios
    avoid binary-float/banker rounding: lower bounds use ceiling and upper
    bounds use floor so canonical v2 never publishes an untradeable price.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("liquidity zone bound must be finite numeric")
    try:
        raw = Decimal(str(value))
        tick = Decimal(str(tick_size))
        if not raw.is_finite() or raw <= 0 or not tick.is_finite() or tick <= 0:
            raise ValueError("liquidity zone bound must be positive and finite")
        raw_numerator, raw_denominator = raw.as_integer_ratio()
        tick_numerator, tick_denominator = tick.as_integer_ratio()
        numerator = raw_numerator * tick_denominator
        denominator = raw_denominator * tick_numerator
        quotient, remainder = divmod(numerator, denominator)
    except (InvalidOperation, OverflowError, ValueError, ZeroDivisionError) as error:
        raise ValueError("liquidity zone bound ratio is invalid") from error
    return quotient + int(lower and remainder != 0)


_FORMED_POOL_SOURCE_KINDS = frozenset(
    {"formed_liquidity_pool", "formed_pool", "equal_highs", "equal_lows"}
)


def _exact_numeric_fraction(value: object, *, name: str) -> Fraction:
    """Convert a finite positive numeric spelling to an exact rational."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite numeric")
    parsed = Decimal(str(value))
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{name} must be positive and finite")
    numerator, denominator = parsed.as_integer_ratio()
    return Fraction(numerator, denominator)


@dataclass(frozen=True)
class _RealBarFact:
    event_id: str
    timeframe: Timeframe
    known_at: pd.Timestamp
    high_ticks: int
    low_ticks: int
    close_ticks: int

    @property
    def bar_event_id(self) -> str:
        """Expose the lifecycle constituent identity without copying BAR DTOs."""

        return self.event_id

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframe", Timeframe(self.timeframe))
        object.__setattr__(
            self,
            "known_at",
            aware_timestamp(self.known_at, name="foundation_adapter.bar.known_at"),
        )
        if (
            not self.event_id
            or self.low_ticks > self.close_ticks
            or self.close_ticks > self.high_ticks
        ):
            raise ValueError("foundation adapter BAR fact is invalid")


@dataclass(frozen=True)
class _CrossingBinding:
    source_level_id: str
    foundation_level_id: str
    generation_id: str
    crossed_at: pd.Timestamp
    penetration_event_id: str
    penetration_bar_event_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "crossed_at",
            aware_timestamp(
                self.crossed_at,
                name="foundation_adapter.crossing.crossed_at",
            ),
        )
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.source_level_id,
                self.foundation_level_id,
                self.generation_id,
                self.penetration_event_id,
                self.penetration_bar_event_id,
            )
        ):
            raise ValueError("foundation crossing binding is invalid")


@dataclass(frozen=True)
class FoundationAdapterUpdate:
    input_fact_id: str
    records: tuple[FoundationRecord, ...]
    lifecycle: SemanticLifecycleState
    projection: FoundationProjection
    ignored: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        if (
            not self.input_fact_id
            or not isinstance(self.lifecycle, SemanticLifecycleState)
            or not isinstance(self.projection, FoundationProjection)
            or any(not isinstance(item, FoundationRecord) for item in self.records)
        ):
            raise ValueError("foundation adapter update is invalid")


def _checkpoint_payload(checkpoint: "FoundationAdapterCheckpoint") -> Mapping[str, Any]:
    return {
        "schema_version": checkpoint.schema_version,
        "tick_size": checkpoint.tick_size,
        "lifecycle_digest": checkpoint.lifecycle_checkpoint.state_digest,
        "projection_checkpoint_id": (
            checkpoint.projection_checkpoint.checkpoint_id
        ),
        "last_order": checkpoint.last_order,
        "seen_event_fingerprints": checkpoint.seen_event_fingerprints,
        "seen_event_metadata": checkpoint.seen_event_metadata,
        "real_bars": checkpoint.real_bars,
        "crossings": checkpoint.crossings,
        "structure_bindings": checkpoint.structure_bindings,
        "semantic_version": checkpoint.semantic_version,
    }


@dataclass(frozen=True)
class FoundationAdapterCheckpoint:
    tick_size: float
    lifecycle_checkpoint: LifecycleCheckpoint
    projection_checkpoint: FoundationProjectionCheckpoint
    last_order: tuple[pd.Timestamp, int, str] | None
    seen_event_fingerprints: tuple[tuple[str, str], ...]
    seen_event_metadata: tuple[tuple[str, pd.Timestamp, EventOrigin], ...]
    real_bars: tuple[_RealBarFact, ...]
    crossings: tuple[_CrossingBinding, ...]
    structure_bindings: tuple[tuple[str, str], ...]
    schema_version: int = FOUNDATION_ADAPTER_CHECKPOINT_SCHEMA_VERSION
    semantic_version: str = FOUNDATION_VERSION
    checkpoint_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            "schema_version" not in vars(self)
            or self.schema_version
            != FOUNDATION_ADAPTER_CHECKPOINT_SCHEMA_VERSION
            or isinstance(self.tick_size, bool)
            or not math.isfinite(float(self.tick_size))
            or self.tick_size <= 0.0
            or not isinstance(self.lifecycle_checkpoint, LifecycleCheckpoint)
            or not isinstance(
                self.projection_checkpoint, FoundationProjectionCheckpoint
            )
            or self.semantic_version != FOUNDATION_VERSION
        ):
            raise ValueError("foundation adapter checkpoint is invalid")
        SemanticLifecycleReducer.restore(self.lifecycle_checkpoint)
        object.__setattr__(
            self,
            "seen_event_fingerprints",
            tuple(self.seen_event_fingerprints),
        )
        metadata = tuple(self.seen_event_metadata)
        if (
            any(
                not isinstance(event_id, str)
                or not event_id
                or not isinstance(known_at, pd.Timestamp)
                or known_at.tz is None
                or not isinstance(origin, EventOrigin)
                for event_id, known_at, origin in metadata
            )
            or {event_id for event_id, _, _ in metadata}
            != {event_id for event_id, _ in self.seen_event_fingerprints}
        ):
            raise ValueError("foundation adapter checkpoint event metadata is invalid")
        object.__setattr__(self, "seen_event_metadata", metadata)
        metadata_by_id = {
            event_id: (known_at, origin)
            for event_id, known_at, origin in metadata
        }
        if any(
            source_id not in metadata_by_id
            or metadata_by_id[source_id][1] not in _DTO_SOURCE_ORIGINS
            for record in self.projection_checkpoint.projection.records
            for source_id in record.source_event_ids
        ):
            raise ValueError(
                "foundation adapter checkpoint contains non-authoritative/unknown record ancestry"
            )
        object.__setattr__(self, "real_bars", tuple(self.real_bars))
        object.__setattr__(self, "crossings", tuple(self.crossings))
        object.__setattr__(
            self, "structure_bindings", tuple(self.structure_bindings)
        )
        object.__setattr__(
            self,
            "checkpoint_digest",
            content_hash(_checkpoint_payload(self)),
        )


class _AppendOnlyOverlay(MutableMapping[str, Any]):
    """Small transactional write overlay over one committed dictionary."""

    def __init__(self, base: dict[str, Any]) -> None:
        if not isinstance(base, dict):
            raise TypeError("foundation overlay base must be committed")
        self._base = base
        self._writes: dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key in self._writes:
            return self._writes[key]
        return self._base[key]

    def __setitem__(self, key: str, value: Any) -> None:
        current = self.get(key)
        if current is not None and current != value:
            raise ValueError("foundation append-only index conflicts")
        self._writes[key] = value

    def __delitem__(self, key: str) -> None:
        raise TypeError("foundation append-only index cannot delete")

    def __iter__(self):
        yield from self._base
        yield from (
            key for key in self._writes if key not in self._base
        )

    def __len__(self) -> int:
        return len(self._base) + sum(
            key not in self._base for key in self._writes
        )

    def commit(self) -> dict[str, Any]:
        self._base.update(self._writes)
        return self._base


class CanonicalFoundationAdapter:
    """Incrementally normalize v1.2 facts into the frozen v2 projection."""

    _LIFECYCLE_COLLECTION_ORDER = (
        "structure_generations",
        "interactions",
        "levels",
        "structure_transitions",
        "relation_generations",
        "delivery_generations",
        "boundary_attacks",
    )

    def __init__(self, *, tick_size: float) -> None:
        if (
            isinstance(tick_size, bool)
            or not math.isfinite(float(tick_size))
            or float(tick_size) <= 0.0
        ):
            raise ValueError("foundation adapter tick_size must be positive")
        self.tick_size = float(tick_size)
        self.lifecycle = SemanticLifecycleReducer.initial_state()
        self.projection = FoundationProjectionReducer.initial_projection()
        self._last_order: tuple[pd.Timestamp, int, str] | None = None
        self._seen_event_fingerprints: dict[str, str] = {}
        self._seen_event_metadata: dict[
            str, tuple[pd.Timestamp, EventOrigin]
        ] = {}
        self._real_bars: list[_RealBarFact] = []
        self._crossings: dict[tuple[str, pd.Timestamp], _CrossingBinding] = {}
        self._structure_bindings: dict[str, str] = {}
        self._state_schema_version = FOUNDATION_ADAPTER_STATE_SCHEMA_VERSION
        self._staged_transaction_open = False
        self._rebuild_derived_indexes()

    def _rebuild_derived_indexes(self) -> None:
        """Rebuild non-authoritative lookup/delta indexes from frozen state."""

        self._lifecycle_indexes = {
            name: {
                self._object_key(item): item
                for item in getattr(self.lifecycle, name)
            }
            for name in self._LIFECYCLE_COLLECTION_ORDER
        }
        self._real_bar_by_id = {
            item.event_id: item for item in self._real_bars
        }
        if len(self._real_bar_by_id) != len(self._real_bars):
            raise ValueError("foundation real BAR identities are duplicated")

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_state_schema_version"] = FOUNDATION_ADAPTER_STATE_SCHEMA_VERSION
        state.pop("_lifecycle_indexes", None)
        state.pop("_real_bar_by_id", None)
        state.pop("_staged_transaction_open", None)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        if (
            not isinstance(state, Mapping)
            or state.get("_state_schema_version")
            != FOUNDATION_ADAPTER_STATE_SCHEMA_VERSION
        ):
            raise ValueError("foundation adapter pickle state schema changed")
        self.__dict__.update(state)
        self._staged_transaction_open = False
        self._rebuild_derived_indexes()

    def _transaction_candidate(self) -> "CanonicalFoundationAdapter":
        """Fork the mutable adapter shell while sharing immutable state.

        Lifecycle/projection DTOs and the facts stored in the local containers
        are frozen values.  Copying their complete append-only histories on
        every clock made transactional staging quadratic.  A shallow
        container copy preserves rollback isolation: all in-place mutations
        target the candidate's own dicts/lists, while reducers replace the
        shared immutable state values.
        """

        candidate = object.__new__(type(self))
        candidate.tick_size = self.tick_size
        candidate.lifecycle = self.lifecycle
        candidate.projection = self.projection
        candidate._last_order = self._last_order
        candidate._seen_event_fingerprints = _AppendOnlyOverlay(
            self._seen_event_fingerprints
        )
        candidate._seen_event_metadata = _AppendOnlyOverlay(
            self._seen_event_metadata
        )
        candidate._real_bars = list(self._real_bars)
        candidate._crossings = dict(self._crossings)
        candidate._structure_bindings = dict(self._structure_bindings)
        candidate._state_schema_version = FOUNDATION_ADAPTER_STATE_SCHEMA_VERSION
        # Inner lifecycle indexes are immutable snapshots.  A changed reducer
        # collection installs a fresh index, so an outer-only copy preserves
        # transaction rollback without rescanning/copying every collection.
        candidate._lifecycle_indexes = dict(self._lifecycle_indexes)
        candidate._real_bar_by_id = dict(self._real_bar_by_id)
        candidate._staged_transaction_open = False
        return candidate

    def _mutation_candidate(
        self,
    ) -> tuple["CanonicalFoundationAdapter", bool]:
        if self._staged_transaction_open:
            return self, False
        return self._transaction_candidate(), True

    def seal_staged_candidate(self) -> None:
        """Close an Observer-owned transaction before it becomes current."""

        if not self._staged_transaction_open:
            raise ValueError("foundation staged transaction is not open")
        FoundationProjectionReducer.validate_complete(self.projection)
        self._staged_transaction_open = False

    def _commit_candidate(self, candidate: "CanonicalFoundationAdapter") -> None:
        if not isinstance(candidate, type(self)):
            raise TypeError("foundation transaction candidate type mismatch")
        if candidate is self:
            raise ValueError("foundation transaction cannot commit itself")
        candidate._commit_append_only_overlays()
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)

    def _commit_append_only_overlays(self) -> None:
        if isinstance(self._seen_event_fingerprints, _AppendOnlyOverlay):
            self._seen_event_fingerprints = (
                self._seen_event_fingerprints.commit()
            )
        if isinstance(self._seen_event_metadata, _AppendOnlyOverlay):
            self._seen_event_metadata = self._seen_event_metadata.commit()

    def commit_staged_candidate(self) -> None:
        """Publish append-only index writes after the wider audit commits."""

        if self._staged_transaction_open:
            raise ValueError("foundation staged transaction is still open")
        if not isinstance(
            self._seen_event_fingerprints,
            _AppendOnlyOverlay,
        ) or not isinstance(self._seen_event_metadata, _AppendOnlyOverlay):
            raise ValueError("foundation staged transaction is already committed")
        self._commit_append_only_overlays()

    def contains_projection_record_id(self, record_id: str) -> bool:
        """Query the projection's derived immutable identity index."""

        if not isinstance(record_id, str) or not record_id:
            raise ValueError("foundation record identity must be non-empty")
        return record_id in self.projection._record_ids_cache

    @staticmethod
    def _event_order(event: MarketEvent) -> tuple[pd.Timestamp, int, str]:
        return event.known_at, event.sequence_no, event.event_id

    def _known_input_ids(self, *additional: str) -> frozenset[str]:
        return frozenset((*self._seen_event_fingerprints, *additional))

    @property
    def known_input_event_ids(self) -> frozenset[str]:
        """Return the immutable identities already accepted as v1.2 inputs."""

        return frozenset(self._seen_event_fingerprints)

    def is_known_input_event_id(self, event_id: str) -> bool:
        """Return whether one exact v1.2 input identity was accepted."""

        if not isinstance(event_id, str) or not event_id:
            raise ValueError("foundation input event identity must be non-empty")
        return event_id in self._seen_event_fingerprints

    def _validated_seen_sources(
        self,
        values: Sequence[str],
        *,
        name: str,
        known_at: pd.Timestamp | None = None,
        require_dto_authority: bool = False,
    ) -> tuple[str, ...]:
        sources = tuple(values)
        if (
            not sources
            or any(not isinstance(value, str) or not value.strip() for value in sources)
            or len(sources) != len(set(sources))
        ):
            raise ValueError(f"{name} requires unique non-empty source event ids")
        missing = tuple(
            identity
            for identity in sources
            if identity not in self._seen_event_metadata
        )
        if missing:
            raise ValueError(f"{name} cites unseen source events: {missing}")
        if known_at is not None:
            clock = aware_timestamp(known_at, name=f"{name}.known_at")
            future = tuple(
                identity
                for identity in sources
                if self._seen_event_metadata[identity][0] > clock
            )
            if future:
                raise ValueError(
                    f"{name} predates exact source-event knowledge: {future}"
                )
        if require_dto_authority:
            non_authoritative = tuple(
                identity
                for identity in sources
                if self._seen_event_metadata[identity][1] not in _DTO_SOURCE_ORIGINS
            )
            if non_authoritative:
                raise ValueError(
                    f"{name} cites non-authoritative DTO sources: "
                    f"{non_authoritative}"
                )
        return sources

    def _require_authoritative_event(self, event: MarketEvent) -> None:
        if event.semantic_version != SMC_SEMANTIC_VERSION:
            raise ValueError("foundation adapter accepts only semantic v1.2 input")
        if event.kind is EventKind.BAR_COMPLETED:
            if event.origin is not EventOrigin.NORMALIZED_DATA:
                raise ValueError("BAR_COMPLETED must be authoritative normalized data")
            return
        if event.kind is EventKind.LIQUIDITY_RETIRED:
            if event.origin not in {
                EventOrigin.LEGACY_TRANSPORT,
                EventOrigin.SEMANTIC_ATOMIC,
            }:
                raise ValueError("liquidity retirement origin is not authoritative")
            return
        if event.kind in _MAPPED_ATOMIC_KINDS and (
            event.origin is not EventOrigin.SEMANTIC_ATOMIC
        ):
            raise ValueError(
                f"{event.kind.value} must be an authoritative semantic fact"
            )

    def _require_prior_sources(
        self,
        event: MarketEvent,
        *,
        allow_empty: bool = False,
    ) -> tuple[str, ...]:
        sources = tuple(event.source_event_ids)
        if not sources and not allow_empty:
            raise ValueError(f"{event.kind.value} lacks source_event_ids")
        unknown = tuple(
            identity
            for identity in sources
            if identity not in self._seen_event_fingerprints
        )
        if unknown:
            raise ValueError(
                f"{event.kind.value} references unseen source events: {unknown}"
            )
        return sources

    @staticmethod
    def _fact_id(
        input_id: str,
        kind: NormalizedTransitionKind,
        suffix: str = "",
    ) -> str:
        return canonical_semantic_id(
            "foundation-adapter-fact", input_id, kind.value, suffix
        )

    def _transition(
        self,
        event: MarketEvent,
        kind: NormalizedTransitionKind,
        *,
        timeframe: Timeframe | None = None,
        payload: Mapping[str, Any],
        source_event_ids: Sequence[str] | None = None,
        suffix: str = "",
    ) -> NormalizedLifecycleTransition:
        sources = _unique_ids(
            source_event_ids
            if source_event_ids is not None
            else (event.event_id, *event.source_event_ids),
            name="normalized adapter ancestry",
        )
        return NormalizedLifecycleTransition(
            fact_id=self._fact_id(event.event_id, kind, suffix),
            kind=kind,
            known_at=event.known_at,
            timeframe=event.timeframe if timeframe is None else timeframe,
            source_event_ids=sources,
            payload=payload,
            sequence_no=event.sequence_no,
        )

    @staticmethod
    def _state_objects(state: SemanticLifecycleState) -> tuple[object, ...]:
        return (
            *state.levels,
            *state.interactions,
            *state.structure_generations,
            *state.structure_transitions,
            *state.relation_generations,
            *state.delivery_generations,
            *state.boundary_attacks,
        )

    @staticmethod
    def _object_key(value: object) -> tuple[type[object], str]:
        for attribute in (
            "level_id",
            "generation_id",
            "structure_generation_id",
            "structure_transition_id",
            "relation_generation_id",
            "delivery_generation_id",
            "boundary_attack_id",
        ):
            identity = getattr(value, attribute, None)
            if isinstance(identity, str) and identity:
                return type(value), identity
        raise TypeError(f"unsupported lifecycle DTO: {type(value).__name__}")

    def _project_delta(
        self,
        before: SemanticLifecycleState,
        after: SemanticLifecycleState,
        *,
        additional_known_ids: Sequence[str] = (),
    ) -> tuple[FoundationRecord, ...]:
        changed: list[object] = []
        state_collections = (
            # Structure owners precede same-clock Relation/Delivery/level
            # revisions, including confirmation-time level ownership.
            (
                "structure_generations",
                before.structure_generations,
                after.structure_generations,
            ),
            # A level revision points at its live interaction generation.
            # Publish the same-clock owner first so incremental replay never
            # needs to borrow a future record from the batch.
            ("interactions", before.interactions, after.interactions),
            ("levels", before.levels, after.levels),
            (
                "structure_transitions",
                before.structure_transitions,
                after.structure_transitions,
            ),
            (
                "relation_generations",
                before.relation_generations,
                after.relation_generations,
            ),
            (
                "delivery_generations",
                before.delivery_generations,
                after.delivery_generations,
            ),
            (
                "boundary_attacks",
                before.boundary_attacks,
                after.boundary_attacks,
            ),
        )
        for name, old_items, new_items in state_collections:
            if old_items is new_items:
                continue
            old = self._lifecycle_indexes[name]
            new = {self._object_key(item): item for item in new_items}
            changed.extend(
                item
                for key, item in new.items()
                if old.get(key) != item
            )
            self._lifecycle_indexes[name] = new
        # Do not materialize the complete accepted-event key set for every
        # lifecycle transition.  The two exact authorities are already
        # immutable for this projection step: the adapter's accepted-event
        # index plus the small same-batch allowance.
        additional_allowed = frozenset(additional_known_ids)
        projection = self.projection
        records: list[FoundationRecord] = []
        for item in changed:
            record = FoundationProjectionReducer.record_from_dto(item)
            missing = tuple(
                identity
                for identity in record.source_event_ids
                if identity not in self._seen_event_fingerprints
                and identity not in additional_allowed
            )
            if missing:
                raise ValueError(
                    "foundation DTO ancestry is not an input event or real BAR: "
                    f"{missing}"
                )
            non_authoritative = tuple(
                identity
                for identity in record.source_event_ids
                if self._seen_event_metadata[identity][1]
                not in _DTO_SOURCE_ORIGINS
            )
            if non_authoritative:
                raise ValueError(
                    "foundation record cannot cite non-authoritative or "
                    "legacy transport ancestry: "
                    f"{non_authoritative}"
                )
            updated = FoundationProjectionReducer.reduce(projection, record)
            if updated is not projection:
                records.append(record)
                projection = updated
        if not self._staged_transaction_open:
            FoundationProjectionReducer.validate_complete(projection)
        self.projection = projection
        return tuple(records)

    def _apply(
        self,
        transitions: Sequence[NormalizedLifecycleTransition],
        *,
        additional_known_ids: Sequence[str] = (),
    ) -> tuple[FoundationRecord, ...]:
        before = self.lifecycle
        state = before
        for transition in transitions:
            state = SemanticLifecycleReducer.reduce(state, transition)
        records = self._project_delta(
            before,
            state,
            additional_known_ids=additional_known_ids,
        )
        self.lifecycle = state
        return records

    def _bar_by_id(self, event_id: str) -> _RealBarFact | None:
        return self._real_bar_by_id.get(event_id)

    def _source_bar(self, event: MarketEvent) -> _RealBarFact:
        candidates = tuple(
            item
            for identity in event.source_event_ids
            if (item := self._bar_by_id(identity)) is not None
            and item.timeframe is Timeframe.M1
            and item.known_at == event.known_at
        )
        if len(candidates) != 1:
            raise ValueError(
                f"{event.kind.value} requires one exact current real M1 BAR source"
            )
        return candidates[0]

    def _level(self, source_level_id: str):
        candidates = tuple(
            item
            for item in self.lifecycle.levels
            if item.level_id == source_level_id
            or item.source_identity == source_level_id
            or (
                not source_level_id.startswith("swing:")
                and item.source_identity == f"swing:{source_level_id}"
            )
        )
        if len(candidates) != 1:
            raise ValueError(
                f"unknown or ambiguous v1.2 liquidity level: {source_level_id}"
            )
        return candidates[0]

    def _active_external(self, timeframe: Timeframe):
        active = tuple(
            item
            for item in self.lifecycle.structure_generations
            if item.timeframe is timeframe
            and item.scope is StructureScope.EXTERNAL
            and item.lifecycle is not StructureGenerationLifecycle.TERMINATED
        )
        if len(active) > 1:
            raise ValueError("multiple active external generations are impossible")
        return active[0] if active else None

    def _active_internal(self, timeframe: Timeframe):
        active = tuple(
            item
            for item in self.lifecycle.structure_generations
            if item.timeframe is timeframe
            and item.scope is StructureScope.INTERNAL
            and item.lifecycle is not StructureGenerationLifecycle.TERMINATED
        )
        if len(active) > 1:
            raise ValueError("multiple active internal generations are impossible")
        return active[0] if active else None

    def _started_transition(self, timeframe: Timeframe):
        active = tuple(
            item
            for item in self.lifecycle.structure_transitions
            if item.timeframe is timeframe
            and item.scope is StructureScope.EXTERNAL
            and item.lifecycle is StructureTransitionLifecycle.STARTED
        )
        if len(active) > 1:
            raise ValueError("multiple active external transitions are impossible")
        return active[0] if active else None

    def _consume_bar(self, event: MarketEvent) -> tuple[FoundationRecord, ...]:
        real_completed, _ = validate_registered_native_bar_root(
            timeframe=event.timeframe,
            event_time=event.event_time,
            known_at=event.known_at,
            evidence=event.evidence,
        )
        transition_payload = {
            "bar_event_id": event.event_id,
            "real_completed": real_completed,
        }
        if not real_completed:
            transition_payload["clock_only"] = True
        transition = self._transition(
            event,
            NormalizedTransitionKind.REAL_BAR_COMPLETED,
            payload=transition_payload,
            source_event_ids=(event.event_id,),
        )
        records = list(
            self._apply((transition,), additional_known_ids=(event.event_id,))
        )
        if not real_completed:
            return tuple(records)
        high = event.evidence.get("high")
        low = event.evidence.get("low")
        close = event.evidence.get("close", event.price)
        if any(value is None for value in (high, low, close)):
            raise ValueError("real BAR lacks frozen high/low/close")
        bar = _RealBarFact(
            event_id=event.event_id,
            timeframe=event.timeframe,
            known_at=event.known_at,
            high_ticks=price_to_ticks(high, self.tick_size, name="BAR high"),
            low_ticks=price_to_ticks(low, self.tick_size, name="BAR low"),
            close_ticks=price_to_ticks(close, self.tick_size, name="BAR close"),
        )
        if self._bar_by_id(bar.event_id) is not None:
            raise ValueError("real BAR identity repeats with a new input fact")
        self._real_bars.append(bar)
        self._real_bar_by_id[bar.event_id] = bar
        # Same-source registered levels rearm from the first strictly later real
        # interaction-TF BAR whose completed close has departed at least one
        # tick in the registered direction.  This is a pure lifecycle rule:
        # formed/equal pools and unknown source kinds remain DISARMED until a
        # new source identity is published.
        eligible_level_ids = tuple(
            item.level_id
            for item in self.lifecycle.levels
            if item.lifecycle is LiquidityLevelLifecycle.DISARMED
            and item.source_kind in REARMABLE_LIQUIDITY_SOURCE_KINDS
            and item.last_terminal_generation_id is not None
            and (
                prior := self.lifecycle.interaction(
                    item.last_terminal_generation_id
                )
            ).interaction_timeframe
            is bar.timeframe
            and prior.terminal_state is LiquidityInteractionTerminal.SWEEP
            and prior.terminal_at is not None
            and bar.known_at > prior.terminal_at
            and (
                bar.close_ticks < item.lower_bound_ticks
                if item.side == "above"
                else bar.close_ticks > item.upper_bound_ticks
            )
        )
        for level_id in eligible_level_ids:
            update = self._observe_rearm_inplace(
                source_level_id=level_id,
                departure_bar_event_id=bar.event_id,
                departure_price=ticks_to_price(bar.close_ticks, self.tick_size),
                additional_known_ids=(event.event_id,),
            )
            records.extend(update.records)
        return tuple(records)

    def _consume_reset(self, event: MarketEvent) -> tuple[FoundationRecord, ...]:
        raw_reason = _required_text(event.evidence, "reason")
        reason = _RESET_REASON_MAP.get(raw_reason)
        if reason is None:
            raise ValueError("MARKET_EPOCH_RESET reason is not registered")
        transition = self._transition(
            event,
            NormalizedTransitionKind.RESET,
            timeframe=None,
            payload={"reason": reason},
            source_event_ids=(event.event_id,),
        )
        records = self._apply((transition,), additional_known_ids=(event.event_id,))
        self._real_bars.clear()
        self._real_bar_by_id.clear()
        self._crossings.clear()
        return records

    def _consume_level_created(
        self, event: MarketEvent
    ) -> tuple[FoundationRecord, ...]:
        self._require_prior_sources(event)
        source_identity = _required_text(event.evidence, "level_id")
        source_kind = _required_text(event.evidence, "source_kind")
        if event.side not in {"above", "below"} or event.price is None:
            raise ValueError("liquidity level lacks side or price")
        if event.zone is None:
            price_ticks = price_to_ticks(
                event.price, self.tick_size, name="liquidity level price"
            )
            lower_bound_ticks = upper_bound_ticks = price_ticks
            price_anchor_rule = "exact_event_price"
        else:
            lower_bound_ticks = _tradable_zone_bound_ticks(
                event.zone[0], self.tick_size, lower=True
            )
            upper_bound_ticks = _tradable_zone_bound_ticks(
                event.zone[1], self.tick_size, lower=False
            )
            try:
                price_ticks = price_to_ticks(
                    event.price,
                    self.tick_size,
                    name="liquidity level price",
                )
                price_anchor_rule = "exact_event_price"
            except ValueError:
                midpoint = (
                    _exact_numeric_fraction(
                        event.zone[0], name="liquidity zone lower"
                    )
                    + _exact_numeric_fraction(
                        event.zone[1], name="liquidity zone upper"
                    )
                ) / 2
                if (
                    source_kind not in _FORMED_POOL_SOURCE_KINDS
                    or _exact_numeric_fraction(
                        event.price, name="liquidity pool midpoint"
                    )
                    != midpoint
                ):
                    raise ValueError(
                        "only an exact formed-pool midpoint may be nontradable"
                    )
                price_ticks = (
                    lower_bound_ticks
                    if event.side == "above"
                    else upper_bound_ticks
                )
                price_anchor_rule = (
                    "near_side_tradable_zone_boundary_for_nontradable_midpoint"
                )
        if not lower_bound_ticks <= price_ticks <= upper_bound_ticks:
            raise ValueError(
                "liquidity zone has no valid tradable tick envelope around its price"
            )
        payload: dict[str, Any] = {
            "source_kind": source_kind,
            "source_identity": source_identity,
            "side": event.side,
            "price_ticks": price_ticks,
            "lower_bound_ticks": lower_bound_ticks,
            "upper_bound_ticks": upper_bound_ticks,
            "tick_size": self.tick_size,
            "price_anchor_rule": price_anchor_rule,
            "interaction_timeframe": Timeframe.M1.value,
        }
        replaced_id = event.evidence.get("replaces_level_id")
        replacement_event_id = event.evidence.get("replaces_level_event_id")
        if (replaced_id is None) != (replacement_event_id is None):
            raise ValueError("liquidity replacement provenance is incomplete")
        if replaced_id is not None:
            if (
                not isinstance(replaced_id, str)
                or not isinstance(replacement_event_id, str)
                or replacement_event_id not in event.context_event_ids
                or replacement_event_id not in self._seen_event_fingerprints
            ):
                raise ValueError("liquidity replacement provenance is invalid")
            prior = self._level(replaced_id)
            if prior.source_event_ids[0] != replacement_event_id:
                raise ValueError("replacement does not cite the prior level event")
            if prior.lifecycle not in {
                LiquidityLevelLifecycle.RETIRED,
                LiquidityLevelLifecycle.ARCHIVED,
            }:
                payload["supersedes_level_id"] = prior.level_id
        transition = self._transition(
            event,
            NormalizedTransitionKind.LIQUIDITY_LEVEL_CREATED,
            payload=payload,
        )
        return self._apply((transition,), additional_known_ids=(event.event_id,))

    def _consume_touch(self, event: MarketEvent) -> tuple[FoundationRecord, ...]:
        if event.timeframe is not Timeframe.M1:
            return ()
        self._require_prior_sources(event)
        source_level_id = _required_text(event.evidence, "level_id")
        level = self._level(source_level_id)
        bar = self._source_bar(event)
        if level.active_generation_id is None:
            return ()
        transition = self._transition(
            event,
            NormalizedTransitionKind.LIQUIDITY_TOUCHED,
            payload={"level_id": level.level_id, "bar_event_id": bar.event_id},
        )
        return self._apply((transition,), additional_known_ids=(event.event_id,))

    def _consume_penetration(
        self, event: MarketEvent
    ) -> tuple[FoundationRecord, ...]:
        if event.timeframe is not Timeframe.M1:
            return ()
        self._require_prior_sources(event)
        source_level_id = _required_text(event.evidence, "level_id")
        level = self._level(source_level_id)
        bar = self._source_bar(event)
        if level.active_generation_id is None:
            return ()
        crossed_at = aware_timestamp(
            _required_text(event.evidence, "crossed_at"),
            name="penetration.crossed_at",
        )
        if crossed_at != event.known_at or event.price is None:
            raise ValueError("penetration crossing clock or price is invalid")
        event_price_ticks = price_to_ticks(
            event.price, self.tick_size, name="penetration price"
        )
        expected_extreme = (
            bar.high_ticks if level.side == "above" else bar.low_ticks
        )
        penetration_ticks = (
            max(0, bar.high_ticks - level.upper_bound_ticks)
            if level.side == "above"
            else max(0, level.lower_bound_ticks - bar.low_ticks)
        )
        if event_price_ticks != expected_extreme:
            raise ValueError("penetration event price is not the exact BAR extreme")
        if penetration_ticks < 1:
            raise ValueError("penetration must cross the level by at least one tick")
        interaction = self.lifecycle.interaction(level.active_generation_id)
        transition = self._transition(
            event,
            NormalizedTransitionKind.LIQUIDITY_PENETRATED,
            payload={
                "level_id": level.level_id,
                "bar_event_id": bar.event_id,
                "penetration_ticks": penetration_ticks,
                "high_ticks": bar.high_ticks,
                "low_ticks": bar.low_ticks,
                "close_ticks": bar.close_ticks,
            },
        )
        records = self._apply((transition,), additional_known_ids=(event.event_id,))
        key = (source_level_id, crossed_at)
        binding = _CrossingBinding(
            source_level_id=source_level_id,
            foundation_level_id=level.level_id,
            generation_id=interaction.generation_id,
            crossed_at=crossed_at,
            penetration_event_id=event.event_id,
            penetration_bar_event_id=bar.event_id,
        )
        prior = self._crossings.get(key)
        if prior is not None and prior != binding:
            raise ValueError("one crossing clock has conflicting penetration")
        self._crossings[key] = binding
        return records

    def _formation_bars(
        self,
        crossed_at: pd.Timestamp,
        resolved_at: pd.Timestamp,
        timeframe: Timeframe,
    ) -> tuple[_RealBarFact, ...]:
        interaction_tf = Timeframe(timeframe)
        bars = tuple(
            item
            for item in self._real_bars
            if item.timeframe is interaction_tf
            and crossed_at <= item.known_at <= resolved_at
        )
        if (
            not bars
            or bars[0].known_at != crossed_at
            or bars[-1].known_at != resolved_at
            or any(
                right.known_at
                != next_registered_native_completion(
                    left.known_at,
                    timeframe_minutes=int(
                        _TIMEFRAME_INTERVAL[interaction_tf]
                        / pd.Timedelta(1, unit="min")
                    ),
                    anchor_minute=(
                        18 * 60 if interaction_tf is Timeframe.H4 else 0
                    ),
                )
                for left, right in zip(bars, bars[1:])
            )
        ):
            raise ValueError(
                "crossing formation lacks a continuous real M1 BAR ancestry"
            )
        return bars

    @staticmethod
    def _formation_ledger(
        bars: Sequence[_RealBarFact],
        *,
        level: object,
        terminal_kind: NormalizedTransitionKind,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        tuple[pd.Timestamp, ...],
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        int,
        pd.Timestamp | None,
        pd.Timestamp | None,
    ]:
        ordered = tuple(bars)
        side = getattr(level, "side")
        lower = getattr(level, "lower_bound_ticks")
        upper = getattr(level, "upper_bound_ticks")
        outside = tuple(
            item.close_ticks > upper
            if side == "above"
            else item.close_ticks < lower
            for item in ordered
        )
        terminal_state = (
            LiquidityInteractionTerminal.ACCEPTANCE
            if terminal_kind
            is NormalizedTransitionKind.LIQUIDITY_ACCEPTANCE_TERMINAL
            else LiquidityInteractionTerminal.SWEEP
        )
        role_ledger = _expected_formation_role_ledger(
            ordered,
            terminal_state=terminal_state,
            side=side,
            lower_bound_ticks=lower,
            upper_bound_ticks=upper,
        )
        bars_by_id = {item.event_id: item for item in ordered}
        ledger = tuple(
            (bars_by_id[bar_event_id], role)
            for bar_event_id, role in role_ledger
        )
        penetrations = tuple(
            max(0, item.high_ticks - upper)
            if side == "above"
            else max(0, lower - item.low_ticks)
            for item in ordered
        )
        entries = tuple(ledger)
        return (
            tuple(item.event_id for item, _ in entries),
            tuple(role for _, role in entries),
            tuple(item.known_at for item, _ in entries),
            tuple(item.high_ticks for item, _ in entries),
            tuple(item.low_ticks for item, _ in entries),
            tuple(item.close_ticks for item, _ in entries),
            max(penetrations),
            next(
                (item.known_at for item, value in zip(ordered, outside) if not value),
                None,
            ),
            next(
                (item.known_at for item, value in zip(ordered, outside) if value),
                None,
            ),
        )

    def _consume_terminal(self, event: MarketEvent) -> tuple[FoundationRecord, ...]:
        if event.timeframe is not Timeframe.M1:
            return ()
        self._require_prior_sources(event)
        source_level_id = _required_text(event.evidence, "level_id")
        level = self._level(source_level_id)
        crossed_at = aware_timestamp(
            _required_text(event.evidence, "crossed_at"),
            name="terminal.crossed_at",
        )
        resolved_at = aware_timestamp(
            _required_text(event.evidence, "resolved_at"),
            name="terminal.resolved_at",
        )
        if resolved_at != event.known_at or crossed_at > resolved_at:
            raise ValueError("terminal crossing clocks are inconsistent")
        binding = self._crossings.get((source_level_id, crossed_at))
        if binding is None and level.active_generation_id is None:
            # The v1.2 tracker may immediately open another crossing after a
            # same-bar Sweep.  Canonical same-level rearm requires a strictly
            # later real BAR that first departs the level by one tick.  Its
            # preceding penetration was therefore intentionally not bound;
            # the later legacy terminal remains atomic history but cannot
            # terminalize the already immutable canonical generation.
            return ()
        if binding is None and level.active_generation_id is not None:
            active_interaction = self.lifecycle.interaction(
                level.active_generation_id
            )
            if (
                active_interaction.generation_number > 1
                and crossed_at < active_interaction.armed_at
            ):
                # A detector crossing observed while the level was DISARMED
                # has no canonical generation binding.  The resolution BAR
                # may itself be the registered departure that rearms a new
                # generation before the delayed legacy terminal is consumed.
                # The old terminal still predates Generation N's armed clock,
                # so it cannot be borrowed by that new immutable generation.
                return ()
        if (
            binding is not None
            and binding.foundation_level_id == level.level_id
            and binding.penetration_event_id in event.source_event_ids
            and level.active_generation_id is None
        ):
            bound_interaction = self.lifecycle.interaction(binding.generation_id)
            if (
                bound_interaction.lifecycle
                is LiquidityInteractionLifecycle.TERMINAL
                and bound_interaction.terminal_state
                in {
                    LiquidityInteractionTerminal.EXPIRED,
                    LiquidityInteractionTerminal.CENSORED,
                }
                and bound_interaction.terminal_at is not None
                and bound_interaction.terminal_at <= event.known_at
            ):
                # An owner/level/reset lifecycle fact can administratively
                # terminalize the exact bound interaction before its legacy
                # detector publishes the price outcome (including later in
                # the same atomic clock).  Preserve the registered canonical
                # terminal; Sweep/Acceptance terminals remain competing and
                # are deliberately rejected by the check below.
                return ()
        if (
            binding is None
            or binding.foundation_level_id != level.level_id
            or binding.penetration_event_id not in event.source_event_ids
            or level.active_generation_id != binding.generation_id
        ):
            raise ValueError("terminal lacks its exact active penetration generation")
        interaction = self.lifecycle.interaction(binding.generation_id)
        bars = self._formation_bars(
            crossed_at,
            resolved_at,
            interaction.interaction_timeframe,
        )
        if binding.penetration_bar_event_id != bars[0].event_id:
            raise ValueError("terminal BAR ancestry does not start at penetration")
        terminal_bar = self._source_bar(event)
        if terminal_bar.event_id != bars[-1].event_id:
            raise ValueError("terminal does not cite its exact resolution BAR")
        terminal_kind = (
            NormalizedTransitionKind.LIQUIDITY_SWEEP_TERMINAL
            if event.kind is EventKind.SWEEP_CONFIRMED
            else NormalizedTransitionKind.LIQUIDITY_ACCEPTANCE_TERMINAL
        )
        if terminal_kind is NormalizedTransitionKind.LIQUIDITY_ACCEPTANCE_TERMINAL:
            outside = tuple(
                item.close_ticks > level.upper_bound_ticks
                if level.side == "above"
                else item.close_ticks < level.lower_bound_ticks
                for item in bars
            )
            outside_completed_bars = sum(outside)
            outside_suffix_bars = 0
            for value in reversed(outside):
                if not value:
                    break
                outside_suffix_bars += 1
            expected_counts = {
                "outside_completed_bars": outside_completed_bars,
                "outside_run": outside_suffix_bars,
            }
            for evidence_field, expected in expected_counts.items():
                asserted = event.evidence.get(evidence_field)
                if asserted is not None and (
                    type(asserted) is not int
                    or asserted < 1
                    or asserted != expected
                ):
                    raise ValueError(
                        "Acceptance outside counters conflict with exact BAR ancestry"
                    )
        (
            bar_ids,
            roles,
            clocks,
            highs,
            lows,
            closes,
            maximum,
            first_inside,
            first_outside,
        ) = self._formation_ledger(
            bars,
            level=level,
            terminal_kind=terminal_kind,
        )
        sources = _unique_ids(
            (
                event.event_id,
                *event.source_event_ids,
                *(item.event_id for item in bars),
            ),
            name="terminal input and BAR ancestry",
        )
        transitions = [
            self._transition(
                event,
                terminal_kind,
                payload={
                    "level_id": level.level_id,
                    "interaction_generation_id": interaction.generation_id,
                    "terminal_event_id": event.event_id,
                    "constituent_bar_ids": bar_ids,
                    "constituent_bar_roles": roles,
                    "constituent_bar_known_at": clocks,
                    "constituent_bar_high_ticks": highs,
                    "constituent_bar_low_ticks": lows,
                    "constituent_bar_close_ticks": closes,
                    "max_penetration_ticks": maximum,
                    "first_inside_close_at": first_inside,
                    "first_outside_close_at": first_outside,
                },
                source_event_ids=sources,
            )
        ]
        if event.kind is EventKind.ACCEPTANCE_CONFIRMED:
            transitions.extend(self._protected_acceptance_transitions(event, level))
        return self._apply(tuple(transitions), additional_known_ids=(event.event_id,))

    def _protected_acceptance_transitions(
        self,
        event: MarketEvent,
        level,
    ) -> tuple[NormalizedLifecycleTransition, ...]:
        protected_swing_id = event.evidence.get("protected_swing_id")
        assignment_event_id = event.evidence.get("protected_swing_event_id")
        if protected_swing_id is None and assignment_event_id is None:
            return ()
        if (
            not isinstance(protected_swing_id, str)
            or not protected_swing_id
            or not isinstance(assignment_event_id, str)
            or not assignment_event_id
            or assignment_event_id not in event.context_event_ids
            or assignment_event_id not in self._seen_event_fingerprints
            or level.source_identity
            not in {protected_swing_id, f"swing:{protected_swing_id}"}
        ):
            raise ValueError("Acceptance does not bind exact protected swing identity")
        source_tf_raw = event.evidence.get(
            "source_timeframe", level.source_timeframe.value
        )
        source_tf = Timeframe(source_tf_raw)
        owners = tuple(
            generation
            for generation in self.lifecycle.structure_generations
            if generation.protected_swing_assignment_event_id
            == assignment_event_id
        )
        if len(owners) > 1:
            raise ValueError("protected Swing assignment has ambiguous generation owners")
        if not owners:
            # The referenced v1.2 assignment belonged to a tracker-only
            # Structure identity and was never canonicalized.  The liquidity
            # Acceptance remains valid, but it has no Structure lifecycle
            # authority.
            return ()
        generation = owners[0]
        if (
            generation.timeframe is not source_tf
            or generation.protected_swing_id != protected_swing_id
            or generation.protected_swing_assignment_event_id
            != assignment_event_id
            or event.direction
            is not (
                Direction.SHORT
                if generation.direction is Direction.LONG
                else Direction.LONG
            )
        ):
            raise ValueError("Acceptance does not invalidate exact live protection")
        if (
            generation.scope is not StructureScope.EXTERNAL
            or generation.lifecycle is StructureGenerationLifecycle.TERMINATED
        ):
            # An internal challenger (including one already failed/censored)
            # cannot donate its protection to the incumbent EXTERNAL regime.
            return ()
        active_external = self._active_external(source_tf)
        if (
            generation.lifecycle is not StructureGenerationLifecycle.CONFIRMED
            or active_external is None
            or active_external.generation_id != generation.generation_id
        ):
            raise ValueError("Acceptance does not invalidate exact live protection")
        transitions: list[NormalizedLifecycleTransition] = [
            self._transition(
                event,
                NormalizedTransitionKind.STRUCTURE_GENERATION_TERMINATED,
                timeframe=source_tf,
                payload={
                    "structure_generation_id": generation.generation_id,
                    "reason": "protected_break_accepted",
                    "protected_acceptance_event_id": event.event_id,
                },
                source_event_ids=(event.event_id,),
                suffix=generation.generation_id,
            )
        ]
        transition = self._started_transition(source_tf)
        if (
            transition is not None
            and transition.incumbent_structure_generation_id
            == generation.generation_id
        ):
            transitions.append(
                self._transition(
                    event,
                    NormalizedTransitionKind.STRUCTURE_TRANSITION_EVIDENCE,
                    timeframe=source_tf,
                    payload={
                        "structure_transition_id": (
                            transition.structure_transition_id
                        ),
                        "protected_acceptance_event_id": event.event_id,
                    },
                    source_event_ids=(event.event_id,),
                    suffix=transition.structure_transition_id,
                )
            )
        return tuple(transitions)

    def _structure_origin_swing_id(self, event: MarketEvent) -> str:
        protected = event.evidence.get("candidate_protected_swing_id")
        if isinstance(protected, str) and protected:
            return protected
        key = (
            "source_low_id"
            if event.direction is Direction.LONG
            else "source_high_id"
        )
        return _required_text(event.evidence, key)

    def _event_is_strictly_after_seen_clock(
        self,
        event: MarketEvent,
        *,
        prior_event_id: str,
        prior_known_at: pd.Timestamp,
    ) -> bool:
        """Use monotonic input order for a same-clock causal successor."""

        if (
            prior_event_id not in self._seen_event_fingerprints
            or self._seen_event_metadata.get(prior_event_id)
            != (prior_known_at, EventOrigin.SEMANTIC_ATOMIC)
        ):
            return False
        if event.known_at > prior_known_at:
            return True
        return bool(
            event.known_at == prior_known_at
            and self._last_order is not None
            and self._last_order[0] == prior_known_at
            and event.sequence_no > self._last_order[1]
        )

    def _mss_origin_identity(self, event: MarketEvent) -> str:
        """Resolve the exact swing/structure identity carried by an MSS fact."""

        bos_id = _required_text(event.evidence, "bos_id")
        if bos_id not in event.source_entity_ids:
            raise ValueError("MSS bos_id is not exact source-entity provenance")
        candidates = tuple(
            identity
            for identity in event.source_entity_ids
            if identity != bos_id
        )
        explicit = tuple(
            value
            for key in (
                "origin_swing_id",
                "target_swing_id",
                "protected_swing_id",
                "source_structure_id",
            )
            if isinstance((value := event.evidence.get(key)), str)
            and value
        )
        if explicit:
            if len(set(explicit)) != 1 or explicit[0] not in candidates:
                raise ValueError("MSS origin identity conflicts with source provenance")
            return explicit[0]
        if len(candidates) != 1:
            raise ValueError(
                "MSS requires one verifiable source swing/structure identity"
            )
        return candidates[0]

    @staticmethod
    def _structure_evidence_identity(event: MarketEvent) -> str:
        """Return the exact v1.2 Structure identity cited by BOS/assignment."""

        explicit = event.evidence.get("structure_id")
        if isinstance(explicit, str) and explicit:
            return explicit
        if event.kind is EventKind.QUALIFIED_BOS:
            bos_id = _required_text(event.evidence, "bos_id")
            candidates = tuple(
                identity
                for identity in event.source_entity_ids
                if identity != bos_id
            )
            if len(candidates) == 1:
                return candidates[0]
        raise ValueError(
            f"{event.kind.value} lacks one exact source Structure identity"
        )

    def _terminate_internal_transition(
        self,
        event: MarketEvent,
        generation: object,
        *,
        reason: str = "scope_rollover",
    ) -> NormalizedLifecycleTransition:
        generation_id = getattr(generation, "generation_id", None)
        if not isinstance(generation_id, str) or not generation_id:
            raise TypeError("internal structure termination requires a generation")
        if reason not in {"scope_rollover", "superseded"}:
            raise ValueError("internal structure termination reason is invalid")
        return self._transition(
            event,
            NormalizedTransitionKind.STRUCTURE_GENERATION_TERMINATED,
            payload={
                "structure_generation_id": generation_id,
                "reason": reason,
            },
            source_event_ids=(event.event_id,),
            suffix=f"internal:{generation_id}",
        )

    def _new_internal_structure_transitions(
        self,
        event: MarketEvent,
        *,
        state: SemanticLifecycleState,
    ) -> tuple[NormalizedLifecycleTransition, ...]:
        if event.direction not in {Direction.LONG, Direction.SHORT}:
            raise ValueError("MSS internal generation lacks direction")
        origin_identity = self._mss_origin_identity(event)
        sources = self._require_prior_sources(event)
        bos_id = _required_text(event.evidence, "bos_id")
        started = self._transition(
            event,
            NormalizedTransitionKind.STRUCTURE_GENERATION_STARTED,
            payload={
                "scope": StructureScope.INTERNAL.value,
                "direction": event.direction.value,
                "origin_event_id": event.event_id,
                "origin_swing_id": origin_identity,
            },
            source_event_ids=(event.event_id, *sources),
            suffix=f"internal:{bos_id}",
        )
        provisional = SemanticLifecycleReducer.reduce(state, started)
        generation = provisional.structure_generations[-1]
        evidence = self._transition(
            event,
            NormalizedTransitionKind.STRUCTURE_GENERATION_EVIDENCE,
            payload={
                "structure_generation_id": generation.generation_id,
                "evidence_kind": "mss",
                "evidence_event_id": event.event_id,
            },
            source_event_ids=(event.event_id,),
            suffix=f"internal-mss:{generation.generation_id}",
        )
        return started, evidence

    def _internal_confirmation_transition(
        self,
        event: MarketEvent,
        generation: object,
    ) -> NormalizedLifecycleTransition:
        generation_id = getattr(generation, "generation_id", None)
        direction = getattr(generation, "direction", None)
        lifecycle = getattr(generation, "lifecycle", None)
        started_at = getattr(generation, "started_at", None)
        if (
            not isinstance(generation_id, str)
            or not generation_id
            or direction is not event.direction
            or lifecycle is not StructureGenerationLifecycle.FORMING
            or not isinstance(started_at, pd.Timestamp)
            or event.known_at <= started_at
        ):
            raise ValueError(
                "structure direction does not independently confirm the exact challenger"
            )
        return self._transition(
            event,
            NormalizedTransitionKind.STRUCTURE_GENERATION_CONFIRMED,
            payload={
                "structure_generation_id": generation_id,
                "confirmation_event_id": event.event_id,
            },
            source_event_ids=(event.event_id, *event.source_event_ids),
            suffix=f"internal-confirmation:{generation_id}",
        )

    def _new_structure_transitions(
        self,
        event: MarketEvent,
    ) -> tuple[NormalizedLifecycleTransition, ...]:
        structure_id = _required_text(event.evidence, "structure_id")
        if event.direction not in {Direction.LONG, Direction.SHORT}:
            raise ValueError("structure direction fact lacks direction")
        sources = self._require_prior_sources(event)
        started = self._transition(
            event,
            NormalizedTransitionKind.STRUCTURE_GENERATION_STARTED,
            payload={
                "scope": StructureScope.EXTERNAL.value,
                "direction": event.direction.value,
                "origin_event_id": event.event_id,
                "origin_swing_id": self._structure_origin_swing_id(event),
            },
            source_event_ids=(event.event_id, *sources),
            suffix=structure_id,
        )
        provisional = SemanticLifecycleReducer.reduce(self.lifecycle, started)
        generation = provisional.structure_generations[-1]
        confirmed = self._transition(
            event,
            NormalizedTransitionKind.STRUCTURE_GENERATION_CONFIRMED,
            payload={
                "structure_generation_id": generation.generation_id,
                "confirmation_event_id": event.event_id,
            },
            source_event_ids=(event.event_id,),
            suffix=generation.generation_id,
        )
        return started, confirmed

    def _consume_structure_direction(
        self, event: MarketEvent
    ) -> tuple[FoundationRecord, ...]:
        structure_id = _required_text(event.evidence, "structure_id")
        if event.direction not in {Direction.LONG, Direction.SHORT}:
            raise ValueError("structure direction fact lacks direction")
        self._require_prior_sources(event)
        incumbent = self._active_external(event.timeframe)
        internal = self._active_internal(event.timeframe)
        transition = self._started_transition(event.timeframe)
        if incumbent is not None:
            if event.direction is incumbent.direction:
                self._structure_bindings[structure_id] = incumbent.generation_id
                transitions: list[NormalizedLifecycleTransition] = []
                if internal is not None:
                    transitions.append(
                        self._terminate_internal_transition(event, internal)
                    )
                if transition is None and not transitions:
                    return ()
                if transition is not None:
                    transitions.append(
                        self._transition(
                            event,
                            NormalizedTransitionKind.STRUCTURE_DIRECTION_RESUMED,
                            payload={
                                "structure_transition_id": (
                                    transition.structure_transition_id
                                ),
                                "resumed_structure_generation_id": (
                                    incumbent.generation_id
                                ),
                                "resumption_event_id": event.event_id,
                            },
                            source_event_ids=(event.event_id,),
                        )
                    )
                return self._apply(
                    tuple(transitions),
                    additional_known_ids=(event.event_id,),
                )
            # A later independent opposite structure fact may mature the
            # exact INTERNAL challenger, but it cannot replace the protected
            # EXTERNAL incumbent.  Exact protected Acceptance remains the
            # only release of that external scope.
            if internal is None or internal.direction is not event.direction:
                return ()
            self._structure_bindings[structure_id] = internal.generation_id
            if internal.lifecycle is StructureGenerationLifecycle.FORMING:
                confirmed = self._internal_confirmation_transition(
                    event, internal
                )
                return self._apply(
                    (confirmed,), additional_known_ids=(event.event_id,)
                )
            if internal.lifecycle is StructureGenerationLifecycle.CONFIRMED:
                return ()
            raise ValueError("opposite structure references a terminal challenger")

        prefix_records: list[FoundationRecord] = []
        if (
            internal is not None
            and internal.direction is event.direction
            and internal.lifecycle is StructureGenerationLifecycle.FORMING
        ):
            confirmed = self._internal_confirmation_transition(event, internal)
            prefix_records.extend(
                self._apply(
                    (confirmed,), additional_known_ids=(event.event_id,)
                )
            )
            internal = self.lifecycle.structure(internal.generation_id)
        transitions = []
        if internal is not None:
            accepted_incumbent = (
                None
                if transition is None
                else self.lifecycle.structure(
                    transition.incumbent_structure_generation_id
                )
            )
            promoted_challenger = (
                transition is not None
                and transition.protected_acceptance_event_id is not None
                and event.direction is transition.challenger_direction
                and internal.direction is event.direction
                and internal.scope is StructureScope.INTERNAL
                and internal.lifecycle is StructureGenerationLifecycle.CONFIRMED
                and internal.timeframe is event.timeframe
                and internal.started_at == transition.started_at
                and internal.mss_event_ids == transition.mss_event_ids
                and accepted_incumbent is not None
                and accepted_incumbent.scope is StructureScope.EXTERNAL
                and accepted_incumbent.timeframe is event.timeframe
                and accepted_incumbent.direction
                is transition.incumbent_direction
                and accepted_incumbent.lifecycle
                is StructureGenerationLifecycle.TERMINATED
                and accepted_incumbent.termination_reason
                == "protected_break_accepted"
                and accepted_incumbent.protected_acceptance_event_id
                == transition.protected_acceptance_event_id
                and accepted_incumbent.terminated_at is not None
                and transition.updated_at == accepted_incumbent.terminated_at
                and self._seen_event_metadata.get(
                    transition.protected_acceptance_event_id
                )
                == (
                    accepted_incumbent.terminated_at,
                    EventOrigin.SEMANTIC_ATOMIC,
                )
                and self._event_is_strictly_after_seen_clock(
                    event,
                    prior_event_id=(
                        transition.protected_acceptance_event_id
                    ),
                    prior_known_at=accepted_incumbent.terminated_at,
                )
            )
            transitions.append(
                self._terminate_internal_transition(
                    event,
                    internal,
                    reason=(
                        "superseded"
                        if promoted_challenger
                        else "scope_rollover"
                    ),
                )
            )
        transitions.extend(self._new_structure_transitions(event))
        provisional = self.lifecycle
        for fact in transitions:
            provisional = SemanticLifecycleReducer.reduce(provisional, fact)
        new_generation = provisional.structure_generations[-1]
        self._structure_bindings[structure_id] = new_generation.generation_id
        if transition is not None:
            if event.direction is transition.challenger_direction:
                if (
                    transition.protected_acceptance_event_id is None
                    or not promoted_challenger
                ):
                    raise ValueError(
                        "opposite structure confirmation lacks exact later "
                        "protected Acceptance promotion"
                    )
                transitions.append(
                    self._transition(
                        event,
                        NormalizedTransitionKind.STRUCTURE_TRANSITION_CONFIRMED,
                        payload={
                            "structure_transition_id": transition.structure_transition_id,
                            "protected_acceptance_event_id": (
                                transition.protected_acceptance_event_id
                            ),
                            "opposite_structure_generation_id": (
                                new_generation.generation_id
                            ),
                            "opposite_confirmation_event_id": event.event_id,
                        },
                        source_event_ids=(
                            transition.protected_acceptance_event_id,
                            event.event_id,
                        ),
                    )
                )
            elif (
                event.direction is transition.incumbent_direction
                and transition.protected_acceptance_event_id is None
            ):
                # Before Acceptance, exact original-direction evidence fails
                # the challenger.  After Acceptance the old incumbent is
                # already terminal: rolling over the exact FORMING internal
                # challenger censors that transition, while this fact starts
                # a distinct external generation instead of rewriting FAILED.
                transitions.append(
                    self._transition(
                        event,
                        NormalizedTransitionKind.STRUCTURE_DIRECTION_RESUMED,
                        payload={
                            "structure_transition_id": transition.structure_transition_id,
                            "resumed_structure_generation_id": (
                                new_generation.generation_id
                            ),
                            "resumption_event_id": event.event_id,
                        },
                        source_event_ids=(event.event_id,),
                    )
                )
        return (
            *prefix_records,
            *self._apply(
                tuple(transitions), additional_known_ids=(event.event_id,)
            ),
        )

    def _is_exact_unbound_transition_evidence(
        self,
        event: MarketEvent,
        *,
        evidence_kind: str,
        structure_identity: str,
        external: object | None,
    ) -> bool:
        """Recognize one sourceful v1.2 tracker-only continuation seam.

        This predicate deliberately carries no new checkpointed tracker state.
        It can therefore accept only the immediately sourced Q-BOS/assignment
        pair while the exact canonical MSS transition is still current.
        ImmutableEventStore validates the cited parent event kinds before the
        adapter sees production input; the adapter rechecks every state,
        timing, origin, ordering, and entity condition available locally.
        """

        if (
            external is not None
            or event.direction not in {Direction.LONG, Direction.SHORT}
            or event.context_event_ids
        ):
            return False
        internal = self._active_internal(event.timeframe)
        transition = self._started_transition(event.timeframe)
        if (
            internal is None
            or transition is None
            or internal.scope is not StructureScope.INTERNAL
            or internal.lifecycle is not StructureGenerationLifecycle.FORMING
            or internal.timeframe is not event.timeframe
            or internal.direction is not event.direction
            or transition.scope is not StructureScope.EXTERNAL
            or transition.timeframe is not event.timeframe
            or transition.challenger_direction is not event.direction
            or transition.incumbent_direction is event.direction
            or transition.started_at != internal.started_at
            or internal.updated_at != internal.started_at
            or not internal.origin_event_id
            or internal.mss_event_ids != (internal.origin_event_id,)
            or transition.mss_event_ids != internal.mss_event_ids
            or transition.protected_acceptance_event_id is None
        ):
            return False

        incumbent = self.lifecycle.structure(
            transition.incumbent_structure_generation_id
        )
        acceptance_id = transition.protected_acceptance_event_id
        acceptance_metadata = self._seen_event_metadata.get(acceptance_id)
        if (
            incumbent.scope is not StructureScope.EXTERNAL
            or incumbent.timeframe is not event.timeframe
            or incumbent.direction is not transition.incumbent_direction
            or incumbent.lifecycle is not StructureGenerationLifecycle.TERMINATED
            or incumbent.termination_reason != "protected_break_accepted"
            or incumbent.protected_acceptance_event_id != acceptance_id
            or incumbent.terminated_at is None
            or incumbent.terminated_at != transition.updated_at
            or acceptance_metadata
            != (incumbent.terminated_at, EventOrigin.SEMANTIC_ATOMIC)
            or not (
                internal.started_at
                < incumbent.terminated_at
                < event.known_at
            )
        ):
            return False

        sources = tuple(event.source_event_ids)
        if (
            self._last_order is None
            or self._last_order[0] != event.known_at
            or self._last_order[1] >= event.sequence_no
            or not sources
            or len(sources) != len(set(sources))
            or self._last_order[2] != sources[0]
        ):
            return False

        if evidence_kind == "bos":
            bos_id = event.evidence.get("bos_id")
            if (
                event.kind is not EventKind.QUALIFIED_BOS
                or not isinstance(bos_id, str)
                or not bos_id
                or event.evidence.get("scope") != "continuation"
                or event.evidence.get("qualification")
                != "aligned_with_confirmed_structure"
                or tuple(event.source_entity_ids)
                != (bos_id, structure_identity)
                or len(sources) != 2
                or self._seen_event_metadata.get(sources[0])
                != (event.known_at, EventOrigin.SEMANTIC_ATOMIC)
                or self._seen_event_metadata.get(sources[1])
                != (internal.started_at, EventOrigin.SEMANTIC_ATOMIC)
            ):
                return False
            return True

        if evidence_kind == "protected_swing_assignment":
            bos_id = event.evidence.get("bos_id")
            origin_leg_id = event.evidence.get("origin_leg_id")
            protected_swing_id = event.evidence.get("protected_swing_id")
            if (
                event.kind is not EventKind.PROTECTED_SWING_ASSIGNED
                or not all(
                    isinstance(value, str) and bool(value)
                    for value in (bos_id, origin_leg_id, protected_swing_id)
                )
                or event.evidence.get("structure_id") != structure_identity
                or event.evidence.get("break_standard")
                != "later_acceptance_beyond"
                or tuple(event.source_entity_ids)
                != (
                    bos_id,
                    structure_identity,
                    origin_leg_id,
                    protected_swing_id,
                )
                or len(sources) != 3
                or self._seen_event_metadata.get(sources[0])
                != (event.known_at, EventOrigin.SEMANTIC_ATOMIC)
                or any(
                    self._seen_event_metadata.get(source_id) is None
                    or self._seen_event_metadata[source_id][0] > event.known_at
                    or self._seen_event_metadata[source_id][1]
                    is not EventOrigin.SEMANTIC_ATOMIC
                    for source_id in sources[1:]
                )
            ):
                return False
            return True

        return False

    def _consume_structure_evidence(
        self,
        event: MarketEvent,
        evidence_kind: str,
    ) -> tuple[FoundationRecord, ...]:
        self._require_prior_sources(event)
        structure_identity = self._structure_evidence_identity(event)
        generation_id = self._structure_bindings.get(structure_identity)
        external = self._active_external(event.timeframe)
        if generation_id is None and self._is_exact_unbound_transition_evidence(
            event,
            evidence_kind=evidence_kind,
            structure_identity=structure_identity,
            external=external,
        ):
            # A v1.2 StructureDirection can remain tracker-only while an
            # independently sourced MSS starts the canonical challenger.  If
            # protected Acceptance has since released that exact incumbent,
            # the tracker's strictly later continuation/Q-BOS chain still has
            # no canonical Structure identity to mutate.  The event-store
            # parent contracts establish the RAW/Q-BOS/leg/Swing kinds; the
            # checks below additionally require their exact current ordering,
            # entity cross-links, and the unchanged canonical transition.
            return ()
        if generation_id is None and (
            external is not None
            and external.lifecycle is StructureGenerationLifecycle.CONFIRMED
            and event.direction is not external.direction
        ):
            # v1.2's structure tracker can publish a continuation/Q-BOS for
            # an opposite geometric StructureDirection that the canonical
            # lifecycle deliberately did not promote over the incumbent.
            # Such a fact remains authoritative atomic history, but it is not
            # evidence for this EXTERNAL generation and must not silently
            # rewrite it (or manufacture a transition without MSS).
            return ()
        if generation_id is None:
            raise ValueError(
                "structure evidence does not bind a canonical Structure identity"
            )
        generation = self.lifecycle.structure(generation_id)
        if (
            generation.scope is StructureScope.INTERNAL
            and generation.lifecycle is StructureGenerationLifecycle.TERMINATED
        ):
            # The legacy tracker can keep citing a prior Structure identity
            # after the canonical INTERNAL challenger was explicitly rolled
            # over and a new challenger generation started.  This exact-bound
            # evidence remains atomic history, but it must neither mutate the
            # immutable terminal owner nor be rebound by direction to the new
            # INTERNAL generation.
            return ()
        if (
            generation.timeframe is not event.timeframe
            or generation.lifecycle is not StructureGenerationLifecycle.CONFIRMED
            or event.direction is not generation.direction
        ):
            raise ValueError("structure evidence does not bind live external generation")
        payload: dict[str, Any] = {
            "structure_generation_id": generation.generation_id,
            "evidence_kind": evidence_kind,
            "evidence_event_id": event.event_id,
        }
        if evidence_kind == "protected_swing_assignment":
            payload["protected_swing_id"] = _required_text(
                event.evidence, "protected_swing_id"
            )
        fact = self._transition(
            event,
            NormalizedTransitionKind.STRUCTURE_GENERATION_EVIDENCE,
            payload=payload,
            source_event_ids=(event.event_id, *event.source_event_ids),
        )
        transitions = [fact]
        if evidence_kind == "bos" and generation.scope is StructureScope.EXTERNAL:
            internal = self._active_internal(event.timeframe)
            candidate = self._started_transition(event.timeframe)
            if (
                internal is not None
                and internal.lifecycle
                in {
                    StructureGenerationLifecycle.FORMING,
                    StructureGenerationLifecycle.CONFIRMED,
                }
                and candidate is not None
                and candidate.incumbent_structure_generation_id
                == generation.generation_id
                and internal.direction is candidate.challenger_direction
                and event.known_at > candidate.started_at
            ):
                transitions.append(
                    self._terminate_internal_transition(event, internal)
                )
                transitions.append(
                    self._transition(
                        event,
                        NormalizedTransitionKind.STRUCTURE_DIRECTION_RESUMED,
                        payload={
                            "structure_transition_id": (
                                candidate.structure_transition_id
                            ),
                            "resumed_structure_generation_id": (
                                generation.generation_id
                            ),
                            "resumption_event_id": event.event_id,
                        },
                        source_event_ids=(event.event_id,),
                    )
                )
        return self._apply(
            tuple(transitions),
            additional_known_ids=(event.event_id,),
        )

    def _is_exact_post_acceptance_internal_counter_mss(
        self,
        event: MarketEvent,
        *,
        source_generation_id: str,
    ) -> bool:
        """Recognize counter-MSS evidence with no remaining external owner."""

        if (
            event.direction not in {Direction.LONG, Direction.SHORT}
            or self._active_external(event.timeframe) is not None
        ):
            return False
        internal = self._active_internal(event.timeframe)
        transition = self._started_transition(event.timeframe)
        if (
            internal is None
            or transition is None
            or internal.generation_id != source_generation_id
            or internal.scope is not StructureScope.INTERNAL
            or internal.lifecycle is not StructureGenerationLifecycle.CONFIRMED
            or internal.timeframe is not event.timeframe
            or internal.direction is not transition.challenger_direction
            or event.direction is not transition.incumbent_direction
            or event.direction is internal.direction
            or transition.scope is not StructureScope.EXTERNAL
            or transition.timeframe is not event.timeframe
            or transition.started_at != internal.started_at
            or transition.mss_event_ids != internal.mss_event_ids
            or internal.origin_event_id not in internal.mss_event_ids
            or transition.protected_acceptance_event_id is None
            or internal.confirmed_at is None
            or internal.confirmation_event_id is None
        ):
            return False

        acceptance_id = transition.protected_acceptance_event_id
        incumbent = self.lifecycle.structure(
            transition.incumbent_structure_generation_id
        )
        sources = tuple(event.source_event_ids)
        if (
            incumbent.scope is not StructureScope.EXTERNAL
            or incumbent.timeframe is not event.timeframe
            or incumbent.direction is not transition.incumbent_direction
            or incumbent.lifecycle is not StructureGenerationLifecycle.TERMINATED
            or incumbent.termination_reason != "protected_break_accepted"
            or incumbent.protected_acceptance_event_id != acceptance_id
            or incumbent.terminated_at is None
            or transition.updated_at != incumbent.terminated_at
            or self._seen_event_metadata.get(acceptance_id)
            != (incumbent.terminated_at, EventOrigin.SEMANTIC_ATOMIC)
            or not (
                transition.started_at
                < internal.confirmed_at
                < incumbent.terminated_at
                < event.known_at
            )
            or len(sources) != 2
            or sources[1] != internal.confirmation_event_id
            or self._seen_event_metadata.get(sources[0])
            != (event.known_at, EventOrigin.SEMANTIC_ATOMIC)
            or self._seen_event_metadata.get(sources[1])
            != (internal.confirmed_at, EventOrigin.SEMANTIC_ATOMIC)
        ):
            return False
        return True

    def _consume_mss(self, event: MarketEvent) -> tuple[FoundationRecord, ...]:
        self._require_prior_sources(event)
        incumbent = self._active_external(event.timeframe)
        origin_identity = self._mss_origin_identity(event)
        source_generation_id = self._structure_bindings.get(origin_identity)
        if (
            source_generation_id is not None
            and self._is_exact_post_acceptance_internal_counter_mss(
                event,
                source_generation_id=source_generation_id,
            )
        ):
            # Acceptance has already terminalized the old external owner, but
            # this MSS only counters the exact confirmed INTERNAL challenger.
            # MSS cannot fail, confirm, or replace a regime; retain its atomic
            # history and await registered Structure/Q-BOS terminal authority.
            return ()
        if (
            source_generation_id is None
            and incumbent is not None
            and incumbent.lifecycle is StructureGenerationLifecycle.CONFIRMED
            and event.direction is incumbent.direction
        ):
            # An MSS against a tracker-only opposite Structure does not oppose
            # the canonical incumbent and cannot confirm its resumption.
            return ()
        if source_generation_id is None and incumbent is None:
            terminated_external = tuple(
                generation
                for generation in self.lifecycle.structure_generations
                if generation.timeframe is event.timeframe
                and generation.scope is StructureScope.EXTERNAL
                and generation.lifecycle
                is StructureGenerationLifecycle.TERMINATED
                and generation.terminated_at is not None
            )
            if terminated_external:
                latest_terminal_at = max(
                    generation.terminated_at
                    for generation in terminated_external
                    if generation.terminated_at is not None
                )
                latest = tuple(
                    generation
                    for generation in terminated_external
                    if generation.terminated_at == latest_terminal_at
                )
                if (
                    len(latest) == 1
                    and latest[0].termination_reason
                    == "protected_break_accepted"
                    and event.direction in {Direction.LONG, Direction.SHORT}
                    and event.direction is latest[0].direction
                    and event.known_at > latest_terminal_at
                ):
                    # The exact source Structure remained tracker-only while
                    # this incumbent was live.  Once protected Acceptance has
                    # terminalized the latest canonical regime, its later
                    # same-direction MSS still has no generation authority.
                    return ()
        if source_generation_id is not None:
            source_generation = self.lifecycle.structure(source_generation_id)
            if (
                source_generation.scope is StructureScope.EXTERNAL
                and source_generation.lifecycle
                is StructureGenerationLifecycle.TERMINATED
                and source_generation.termination_reason
                == "protected_break_accepted"
                and source_generation.terminated_at is not None
                and event.known_at > source_generation.terminated_at
                and event.direction in {Direction.LONG, Direction.SHORT}
                and event.direction is not source_generation.direction
            ):
                # A tracker can publish the MSS derived from an exact opposed
                # boundary after Acceptance has already terminalized that
                # protected EXTERNAL generation.  The late atomic fact cannot
                # retroactively start a transition before its Acceptance.
                return ()
            if (
                source_generation.scope is StructureScope.INTERNAL
                and source_generation.lifecycle
                is StructureGenerationLifecycle.TERMINATED
            ):
                # The legacy tracker may emit a later MSS against an exact
                # Structure identity whose canonical INTERNAL challenger was
                # already rolled over.  Preserve that MSS in atomic history,
                # but never rebind it by direction to the current challenger
                # (or use it to mutate/fail the incumbent transition).
                return ()
            if (
                incumbent is not None
                and incumbent.lifecycle is StructureGenerationLifecycle.CONFIRMED
                and source_generation.scope is StructureScope.INTERNAL
                and source_generation.lifecycle
                in {
                    StructureGenerationLifecycle.FORMING,
                    StructureGenerationLifecycle.CONFIRMED,
                }
                and source_generation.direction is not event.direction
                and event.direction is incumbent.direction
            ):
                # This is counter-evidence against the exact INTERNAL
                # challenger.  MSS remains transition evidence only; retain
                # the started transition and wait for a later registered
                # incumbent StructureDirection/Q-BOS before marking FAILED.
                return ()
        if (
            incumbent is None
            or incumbent.lifecycle is not StructureGenerationLifecycle.CONFIRMED
            or event.direction not in {Direction.LONG, Direction.SHORT}
            or event.direction is incumbent.direction
            or source_generation_id != incumbent.generation_id
        ):
            raise ValueError("MSS does not oppose one confirmed external generation")
        transitions: list[NormalizedLifecycleTransition] = []
        working = self.lifecycle
        internal = self._active_internal(event.timeframe)
        if (
            internal is not None
            and internal.direction is event.direction
            and internal.origin_swing_id == origin_identity
        ):
            transitions.append(
                self._transition(
                    event,
                    NormalizedTransitionKind.STRUCTURE_GENERATION_EVIDENCE,
                    payload={
                        "structure_generation_id": internal.generation_id,
                        "evidence_kind": "mss",
                        "evidence_event_id": event.event_id,
                    },
                    source_event_ids=(event.event_id, *event.source_event_ids),
                    suffix=f"internal-mss:{internal.generation_id}",
                )
            )
        else:
            if internal is not None:
                terminal = self._terminate_internal_transition(
                    event,
                    internal,
                    reason="superseded",
                )
                transitions.append(terminal)
                working = SemanticLifecycleReducer.reduce(working, terminal)
            transitions.extend(
                self._new_internal_structure_transitions(event, state=working)
            )
        transitions.append(
            self._transition(
                event,
                NormalizedTransitionKind.MSS_TRANSITION_STARTED,
                payload={
                    "incumbent_structure_generation_id": incumbent.generation_id,
                    "challenger_direction": event.direction.value,
                    "mss_event_id": event.event_id,
                },
                source_event_ids=(event.event_id, *event.source_event_ids),
            )
        )
        return self._apply(
            tuple(transitions),
            additional_known_ids=(event.event_id,),
        )

    def _consume_inplace(self, event: MarketEvent) -> FoundationAdapterUpdate:
        if not isinstance(event, MarketEvent):
            raise TypeError("foundation adapter accepts MarketEvent input")
        fingerprint = content_hash(event)
        previous = self._seen_event_fingerprints.get(event.event_id)
        if previous is not None:
            if previous != fingerprint:
                raise ValueError("MarketEvent identity conflicts with adapter history")
            return FoundationAdapterUpdate(
                event.event_id, (), self.lifecycle, self.projection, ignored=True
            )
        order = self._event_order(event)
        if self._last_order is not None and order <= self._last_order:
            raise ValueError("foundation adapter input is out of knowledge order")
        self._require_authoritative_event(event)
        # Register causal metadata inside the transactional candidate before
        # projection, so every emitted record can fail closed on legacy
        # ancestry.  A rejected handler discards this candidate wholesale.
        self._seen_event_metadata[event.event_id] = (
            event.known_at,
            event.origin,
        )

        if event.kind is EventKind.BAR_COMPLETED:
            records = self._consume_bar(event)
        elif event.kind is EventKind.MARKET_EPOCH_RESET:
            self._require_prior_sources(event, allow_empty=True)
            records = self._consume_reset(event)
        elif event.kind is EventKind.LIQUIDITY_LEVEL_CREATED:
            records = self._consume_level_created(event)
        elif event.kind is EventKind.LEVEL_TOUCHED:
            records = self._consume_touch(event)
        elif event.kind is EventKind.LEVEL_PENETRATED:
            records = self._consume_penetration(event)
        elif event.kind in {
            EventKind.SWEEP_CONFIRMED,
            EventKind.ACCEPTANCE_CONFIRMED,
        }:
            records = self._consume_terminal(event)
        elif event.kind is EventKind.LIQUIDITY_RETIRED:
            # v1.2 publishes this lifecycle transport as LEGACY_TRANSPORT.
            # Keep its input order/identity visible, but never let it become
            # canonical foundation ancestry.  ``retire_level`` requires an
            # exact current normalized/atomic source fact instead.
            records = ()
        elif event.kind is EventKind.STRUCTURE_DIRECTION_CONFIRMED:
            records = self._consume_structure_direction(event)
        elif event.kind is EventKind.QUALIFIED_BOS:
            records = self._consume_structure_evidence(event, "bos")
        elif event.kind is EventKind.PROTECTED_SWING_ASSIGNED:
            records = self._consume_structure_evidence(
                event, "protected_swing_assignment"
            )
        elif event.kind is EventKind.MSS_CORE_CONFIRMED:
            records = self._consume_mss(event)
        else:
            records = ()

        self._seen_event_fingerprints[event.event_id] = fingerprint
        self._last_order = order
        return FoundationAdapterUpdate(
            event.event_id,
            records,
            self.lifecycle,
            self.projection,
            ignored=not records and event.kind is not EventKind.BAR_COMPLETED,
        )

    def consume(self, event: MarketEvent) -> FoundationAdapterUpdate:
        """Atomically consume one knowledge-ordered v1.2 fact."""

        candidate, owned = self._mutation_candidate()
        update = candidate._consume_inplace(event)
        if owned:
            self._commit_candidate(candidate)
        return update

    def stage_batch(
        self,
        events: Sequence[MarketEvent],
    ) -> tuple[
        "CanonicalFoundationAdapter",
        tuple[FoundationAdapterUpdate, ...],
    ]:
        """Return an uncommitted atomic candidate and its batch updates.

        The production Observer can add downstream DTOs to this candidate and
        retain it only after the wider event-store audit succeeds.  That keeps
        rollback semantics without checkpointing and rebuilding the complete
        append-only adapter history on every clock.
        """

        if isinstance(events, (str, bytes)):
            raise TypeError("foundation adapter batch requires MarketEvent inputs")
        batch = tuple(events)
        if self._staged_transaction_open:
            updates = tuple(
                self._consume_inplace(event) for event in batch
            )
            return self, updates
        candidate = self._transaction_candidate()
        candidate._staged_transaction_open = True
        updates = tuple(candidate._consume_inplace(event) for event in batch)
        return candidate, updates

    def consume_batch(
        self,
        events: Sequence[MarketEvent],
    ) -> tuple[FoundationAdapterUpdate, ...]:
        """Atomically consume a knowledge-ordered fact batch with one copy.

        This is only a transaction boundary around ``consume`` semantics; it
        does not introduce a second buffer, detector, or event store.  Any
        rejected fact discards all earlier candidate updates in the batch.
        """

        if self._staged_transaction_open:
            raise ValueError("nested foundation batch transaction is invalid")
        candidate, updates = self.stage_batch(events)
        candidate.seal_staged_candidate()
        self._commit_candidate(candidate)
        return updates

    @staticmethod
    def _rearmable_source_kind(source_kind: str) -> bool:
        return source_kind in REARMABLE_LIQUIDITY_SOURCE_KINDS

    def _observe_rearm_inplace(
        self,
        *,
        source_level_id: str,
        departure_bar_event_id: str,
        departure_price: float,
        additional_known_ids: Sequence[str] = (),
    ) -> FoundationAdapterUpdate:
        level = self._level(source_level_id)
        if not self._rearmable_source_kind(level.source_kind):
            raise ValueError(
                "same-source rearm is not registered for this exact source kind"
            )
        if level.lifecycle is not LiquidityLevelLifecycle.DISARMED:
            raise ValueError("rearm requires one swept disarmed level")
        bar = self._bar_by_id(departure_bar_event_id)
        prior_generation_id = level.last_terminal_generation_id
        if prior_generation_id is None:
            raise ValueError("rearm level lacks its prior terminal generation")
        prior = self.lifecycle.interaction(prior_generation_id)
        latest_bar = next(
            (
                item
                for item in reversed(self._real_bars)
                if item.timeframe is prior.interaction_timeframe
            ),
            None,
        )
        if (
            bar is None
            or bar.timeframe is not prior.interaction_timeframe
            or bar != latest_bar
        ):
            raise ValueError(
                "rearm departure must cite the latest real interaction-TF BAR"
            )
        departure_price_ticks = price_to_ticks(
            departure_price,
            self.tick_size,
            name="rearm departure price",
        )
        if departure_price_ticks != bar.close_ticks:
            raise ValueError("rearm departure price must equal the frozen BAR close")
        rearm_id = canonical_semantic_id(
            "foundation-adapter-rearm",
            level.level_id,
            prior_generation_id,
            bar.event_id,
            departure_price_ticks,
        )
        rearmable_fact = NormalizedLifecycleTransition(
            fact_id=canonical_semantic_id(rearm_id, "rearmable"),
            kind=NormalizedTransitionKind.LIQUIDITY_LEVEL_REARMABLE,
            known_at=bar.known_at,
            timeframe=prior.interaction_timeframe,
            source_event_ids=(bar.event_id,),
            payload={
                "level_id": level.level_id,
                "prior_generation_id": prior_generation_id,
                "departure_price_ticks": departure_price_ticks,
                "departure_bar_event_id": bar.event_id,
            },
        )
        rearmable_records = self._apply(
            (rearmable_fact,), additional_known_ids=additional_known_ids
        )
        rearmed_fact = NormalizedLifecycleTransition(
            fact_id=canonical_semantic_id(rearm_id, "rearmed"),
            kind=NormalizedTransitionKind.LIQUIDITY_LEVEL_REARMED,
            known_at=bar.known_at,
            timeframe=prior.interaction_timeframe,
            # Generation N retains the immutable source-level creation and
            # adds the exact departure BAR that armed this new interaction.
            # That distinguishes same-level rearm from borrowing a newly
            # published spatially similar level.
            source_event_ids=(level.source_event_ids[0], bar.event_id),
            payload={
                "level_id": level.level_id,
                "prior_generation_id": prior_generation_id,
                "rearmable_fact_id": rearmable_fact.fact_id,
                "departure_bar_event_id": bar.event_id,
            },
        )
        rearmed_records = self._apply(
            (rearmed_fact,), additional_known_ids=additional_known_ids
        )
        return FoundationAdapterUpdate(
            rearmed_fact.fact_id,
            (*rearmable_records, *rearmed_records),
            self.lifecycle,
            self.projection,
        )

    def observe_rearm_departure(
        self,
        *,
        source_level_id: str,
        departure_bar_event_id: str,
        departure_price: float,
    ) -> FoundationAdapterUpdate:
        """Apply the sole explicit same-source rearm fact."""

        candidate, owned = self._mutation_candidate()
        update = candidate._observe_rearm_inplace(
            source_level_id=source_level_id,
            departure_bar_event_id=departure_bar_event_id,
            departure_price=departure_price,
        )
        if owned:
            self._commit_candidate(candidate)
        return update

    def _retire_level_inplace(
        self,
        *,
        source_level_id: str,
        reason: str,
        source_event_ids: Sequence[str],
        known_at: pd.Timestamp,
        timeframe: Timeframe,
    ) -> FoundationAdapterUpdate:
        if reason not in _LEVEL_RETIREMENT_REASONS:
            raise ValueError("liquidity retirement reason is not preregistered")
        level = self._level(source_level_id)
        if level.lifecycle in {
            LiquidityLevelLifecycle.RETIRED,
            LiquidityLevelLifecycle.ARCHIVED,
        }:
            raise ValueError("liquidity retirement cannot rewrite terminal state")
        source_tf = Timeframe(timeframe)
        if source_tf is not level.source_timeframe:
            raise ValueError("liquidity retirement timeframe conflicts with source")
        clock = aware_timestamp(known_at, name="liquidity retirement.known_at")
        sources = self._validated_seen_sources(
            source_event_ids,
            name="liquidity retirement",
            known_at=clock,
        )
        invalid_origins = tuple(
            identity
            for identity in sources
            if self._seen_event_metadata[identity][1]
            not in _RETIREMENT_SOURCE_ORIGINS
        )
        if invalid_origins:
            raise ValueError(
                "liquidity retirement requires normalized/atomic source events: "
                f"{invalid_origins}"
            )
        fact = NormalizedLifecycleTransition(
            fact_id=canonical_semantic_id(
                "foundation-adapter-level-retirement",
                level.level_id,
                reason,
                clock,
                *sources,
            ),
            kind=NormalizedTransitionKind.LIQUIDITY_LEVEL_RETIRED,
            known_at=clock,
            timeframe=source_tf,
            source_event_ids=sources,
            payload={"level_id": level.level_id, "reason": reason},
        )
        records = self._apply((fact,))
        return FoundationAdapterUpdate(
            fact.fact_id, records, self.lifecycle, self.projection
        )

    def retire_level(
        self,
        *,
        source_level_id: str,
        reason: str,
        source_event_ids: Sequence[str],
        known_at: pd.Timestamp,
        timeframe: Timeframe,
    ) -> FoundationAdapterUpdate:
        """Retire a level from exact normalized/atomic causal ancestry."""

        candidate, owned = self._mutation_candidate()
        update = candidate._retire_level_inplace(
            source_level_id=source_level_id,
            reason=reason,
            source_event_ids=source_event_ids,
            known_at=known_at,
            timeframe=timeframe,
        )
        if owned:
            self._commit_candidate(candidate)
        return update

    def _observe_boundary_inplace(
        self,
        fact: NormalizedLifecycleTransition,
    ) -> FoundationAdapterUpdate:
        if (
            not isinstance(fact, NormalizedLifecycleTransition)
            or fact.kind is not NormalizedTransitionKind.BOUNDARY_ATTACK_OBSERVED
        ):
            raise TypeError("adapter accepts only an explicit boundary-attack fact")
        missing = tuple(
            identity
            for identity in fact.source_event_ids
            if identity not in self._seen_event_fingerprints
        )
        if missing:
            raise ValueError(f"boundary attack cites unseen inputs: {missing}")
        bar_event_id = fact.payload.get("bar_event_id")
        bar = (
            None
            if not isinstance(bar_event_id, str)
            else self._bar_by_id(bar_event_id)
        )
        if (
            bar is None
            or bar_event_id not in fact.source_event_ids
            or bar.timeframe is not fact.timeframe
            or bar.known_at != fact.known_at
        ):
            raise ValueError(
                "boundary attack requires its exact same-clock real BAR source"
            )
        records = self._apply((fact,))
        return FoundationAdapterUpdate(
            fact.fact_id, records, self.lifecycle, self.projection
        )

    def observe_boundary_attack(
        self,
        fact: NormalizedLifecycleTransition,
    ) -> FoundationAdapterUpdate:
        """Append one pre-normalized wick-only structural attack fact."""

        candidate, owned = self._mutation_candidate()
        update = candidate._observe_boundary_inplace(fact)
        if owned:
            self._commit_candidate(candidate)
        return update

    def _append_dto_inplace(
        self,
        dto: object,
        *,
        source_event_ids: Sequence[str] | None = None,
    ) -> FoundationAdapterUpdate:
        record = FoundationProjectionReducer.record_from_dto(
            dto,
            source_event_ids=source_event_ids,
        )
        self._validated_seen_sources(
            record.source_event_ids,
            name="foundation DTO",
            known_at=record.known_at,
            require_dto_authority=True,
        )
        projection = FoundationProjectionReducer.reduce(self.projection, record)
        ignored = projection is self.projection
        if not self._staged_transaction_open:
            FoundationProjectionReducer.validate_complete(projection)
        self.projection = projection
        return FoundationAdapterUpdate(
            record.record_id,
            () if ignored else (record,),
            self.lifecycle,
            self.projection,
            ignored=ignored,
        )

    def append_dto(
        self,
        dto: object,
        *,
        source_event_ids: Sequence[str] | None = None,
    ) -> FoundationAdapterUpdate:
        """Atomically append one supported DTO with exact seen-event ancestry."""

        candidate, owned = self._mutation_candidate()
        update = candidate._append_dto_inplace(
            dto,
            source_event_ids=source_event_ids,
        )
        if owned:
            self._commit_candidate(candidate)
        return update

    def _observe_relation_inplace(
        self,
        relation: RelationState,
        *,
        parent_structure_generation_id: str,
        child_structure_generation_id: str,
        source_event_ids: Sequence[str],
    ) -> FoundationAdapterUpdate:
        if not isinstance(relation, RelationState):
            raise TypeError("relation observation requires RelationState")
        if relation.known_at is None:
            raise ValueError("relation observation requires known_at")
        sources = self._validated_seen_sources(
            source_event_ids,
            name="relation observation",
            known_at=relation.known_at,
        )
        fact = NormalizedLifecycleTransition.from_relation_state(
            relation,
            parent_structure_generation_id=parent_structure_generation_id,
            child_structure_generation_id=child_structure_generation_id,
            source_event_ids=sources,
        )
        records = self._apply((fact,))
        return FoundationAdapterUpdate(
            fact.fact_id,
            records,
            self.lifecycle,
            self.projection,
            ignored=not records,
        )

    def observe_relation(
        self,
        relation: RelationState,
        *,
        parent_structure_generation_id: str,
        child_structure_generation_id: str,
        source_event_ids: Sequence[str],
    ) -> FoundationAdapterUpdate:
        """Project one authoritative cross-timeframe relation observation."""

        candidate, owned = self._mutation_candidate()
        update = candidate._observe_relation_inplace(
            relation,
            parent_structure_generation_id=parent_structure_generation_id,
            child_structure_generation_id=child_structure_generation_id,
            source_event_ids=source_event_ids,
        )
        if owned:
            self._commit_candidate(candidate)
        return update

    def _observe_delivery_phase_inplace(
        self,
        phase: DeliveryPhase,
        *,
        timeframe: Timeframe,
        known_at: pd.Timestamp,
        parent_structure_generation_id: str,
        origin_event_id: str,
        source_event_ids: Sequence[str],
        current_price_ticks: int,
        extension_ticks: float | None = None,
        retracement_ticks: float | None = None,
    ) -> FoundationAdapterUpdate:
        try:
            phase = DeliveryPhase(phase)
        except (TypeError, ValueError) as error:
            raise ValueError("delivery observation phase is not registered") from error
        clock = aware_timestamp(known_at, name="delivery observation.known_at")
        sources = self._validated_seen_sources(
            source_event_ids,
            name="delivery observation",
            known_at=clock,
        )
        fact = NormalizedLifecycleTransition.from_delivery_phase(
            phase,
            timeframe=Timeframe(timeframe),
            known_at=clock,
            parent_structure_generation_id=parent_structure_generation_id,
            origin_event_id=origin_event_id,
            source_event_ids=sources,
            current_price_ticks=current_price_ticks,
            extension_ticks=extension_ticks,
            retracement_ticks=retracement_ticks,
        )
        records = self._apply((fact,))
        return FoundationAdapterUpdate(
            fact.fact_id,
            records,
            self.lifecycle,
            self.projection,
            ignored=not records,
        )

    def observe_delivery_phase(
        self,
        phase: DeliveryPhase,
        *,
        timeframe: Timeframe,
        known_at: pd.Timestamp,
        parent_structure_generation_id: str,
        origin_event_id: str,
        source_event_ids: Sequence[str],
        current_price_ticks: int,
        extension_ticks: float | None = None,
        retracement_ticks: float | None = None,
    ) -> FoundationAdapterUpdate:
        """Project one explicit delivery-phase observation."""

        candidate, owned = self._mutation_candidate()
        update = candidate._observe_delivery_phase_inplace(
            phase,
            timeframe=timeframe,
            known_at=known_at,
            parent_structure_generation_id=parent_structure_generation_id,
            origin_event_id=origin_event_id,
            source_event_ids=source_event_ids,
            current_price_ticks=current_price_ticks,
            extension_ticks=extension_ticks,
            retracement_ticks=retracement_ticks,
        )
        if owned:
            self._commit_candidate(candidate)
        return update

    def _terminate_relation_inplace(
        self,
        *,
        relation_generation_id: str,
        known_at: pd.Timestamp,
        reason: str,
        source_event_ids: Sequence[str],
    ) -> FoundationAdapterUpdate:
        if reason not in _RELATION_TERMINATION_REASONS:
            raise ValueError("relation termination reason is not preregistered")
        matches = tuple(
            item
            for item in self.lifecycle.relation_generations
            if item.generation_id == relation_generation_id
        )
        if len(matches) != 1 or matches[0].lifecycle is not GenerationLifecycle.ACTIVE:
            raise ValueError("relation termination requires exact active generation")
        generation = matches[0]
        clock = aware_timestamp(known_at, name="relation termination.known_at")
        sources = self._validated_seen_sources(
            source_event_ids,
            name="relation termination",
            known_at=clock,
        )
        fact = NormalizedLifecycleTransition(
            fact_id=canonical_semantic_id(
                "foundation-adapter-relation-terminal",
                relation_generation_id,
                reason,
                clock,
                *sources,
            ),
            kind=NormalizedTransitionKind.RELATION_TERMINATED,
            known_at=clock,
            timeframe=generation.child_tf,
            source_event_ids=sources,
            payload={
                "relation_generation_id": relation_generation_id,
                "reason": reason,
            },
        )
        records = self._apply((fact,))
        return FoundationAdapterUpdate(
            fact.fact_id, records, self.lifecycle, self.projection
        )

    def terminate_relation(
        self,
        *,
        relation_generation_id: str,
        known_at: pd.Timestamp,
        reason: str,
        source_event_ids: Sequence[str],
    ) -> FoundationAdapterUpdate:
        """Explicitly terminate one relation generation for a frozen reason."""

        candidate, owned = self._mutation_candidate()
        update = candidate._terminate_relation_inplace(
            relation_generation_id=relation_generation_id,
            known_at=known_at,
            reason=reason,
            source_event_ids=source_event_ids,
        )
        if owned:
            self._commit_candidate(candidate)
        return update

    def _terminate_delivery_inplace(
        self,
        *,
        delivery_generation_id: str,
        known_at: pd.Timestamp,
        reason: str,
        source_event_ids: Sequence[str],
    ) -> FoundationAdapterUpdate:
        if reason not in _DELIVERY_TERMINATION_REASONS:
            raise ValueError("delivery termination reason is not preregistered")
        matches = tuple(
            item
            for item in self.lifecycle.delivery_generations
            if item.generation_id == delivery_generation_id
        )
        if len(matches) != 1 or matches[0].lifecycle is not GenerationLifecycle.ACTIVE:
            raise ValueError("delivery termination requires exact active generation")
        generation = matches[0]
        clock = aware_timestamp(known_at, name="delivery termination.known_at")
        sources = self._validated_seen_sources(
            source_event_ids,
            name="delivery termination",
            known_at=clock,
        )
        fact = NormalizedLifecycleTransition(
            fact_id=canonical_semantic_id(
                "foundation-adapter-delivery-terminal",
                delivery_generation_id,
                reason,
                clock,
                *sources,
            ),
            kind=NormalizedTransitionKind.DELIVERY_PHASE_TERMINATED,
            known_at=clock,
            timeframe=generation.timeframe,
            source_event_ids=sources,
            payload={
                "delivery_generation_id": delivery_generation_id,
                "reason": reason,
            },
        )
        records = self._apply((fact,))
        return FoundationAdapterUpdate(
            fact.fact_id, records, self.lifecycle, self.projection
        )

    def terminate_delivery(
        self,
        *,
        delivery_generation_id: str,
        known_at: pd.Timestamp,
        reason: str,
        source_event_ids: Sequence[str],
    ) -> FoundationAdapterUpdate:
        """Explicitly terminate one delivery generation for a frozen reason."""

        candidate, owned = self._mutation_candidate()
        update = candidate._terminate_delivery_inplace(
            delivery_generation_id=delivery_generation_id,
            known_at=known_at,
            reason=reason,
            source_event_ids=source_event_ids,
        )
        if owned:
            self._commit_candidate(candidate)
        return update

    def checkpoint(self) -> FoundationAdapterCheckpoint:
        return FoundationAdapterCheckpoint(
            tick_size=self.tick_size,
            lifecycle_checkpoint=SemanticLifecycleReducer.checkpoint(
                self.lifecycle
            ),
            projection_checkpoint=FoundationProjectionReducer.checkpoint(
                self.projection
            ),
            last_order=self._last_order,
            seen_event_fingerprints=tuple(
                sorted(self._seen_event_fingerprints.items())
            ),
            seen_event_metadata=tuple(
                (event_id, known_at, origin)
                for event_id, (known_at, origin) in sorted(
                    self._seen_event_metadata.items()
                )
            ),
            real_bars=tuple(self._real_bars),
            crossings=tuple(
                sorted(
                    self._crossings.values(),
                    key=lambda item: (
                        item.crossed_at,
                        item.source_level_id,
                        item.penetration_event_id,
                    ),
                )
            ),
            structure_bindings=tuple(sorted(self._structure_bindings.items())),
        )

    @classmethod
    def restore(
        cls,
        checkpoint: FoundationAdapterCheckpoint,
    ) -> "CanonicalFoundationAdapter":
        if not isinstance(checkpoint, FoundationAdapterCheckpoint):
            raise TypeError("adapter restore requires FoundationAdapterCheckpoint")
        if (
            "schema_version" not in vars(checkpoint)
            or checkpoint.schema_version
            != FOUNDATION_ADAPTER_CHECKPOINT_SCHEMA_VERSION
            or checkpoint.checkpoint_digest
            != content_hash(_checkpoint_payload(checkpoint))
        ):
            raise ValueError("foundation adapter checkpoint integrity mismatch")
        adapter = cls(tick_size=checkpoint.tick_size)
        adapter.lifecycle = SemanticLifecycleReducer.restore(
            checkpoint.lifecycle_checkpoint
        )
        adapter.projection = FoundationProjectionReducer.restore(
            checkpoint.projection_checkpoint
        )
        adapter._last_order = checkpoint.last_order
        adapter._seen_event_fingerprints = dict(
            checkpoint.seen_event_fingerprints
        )
        adapter._seen_event_metadata = {
            event_id: (known_at, origin)
            for event_id, known_at, origin in checkpoint.seen_event_metadata
        }
        adapter._real_bars = list(checkpoint.real_bars)
        adapter._crossings = {
            (item.source_level_id, item.crossed_at): item
            for item in checkpoint.crossings
        }
        adapter._structure_bindings = dict(checkpoint.structure_bindings)
        adapter._rebuild_derived_indexes()
        return adapter

    @classmethod
    def replay(
        cls,
        events: Sequence[MarketEvent],
        *,
        tick_size: float,
    ) -> "CanonicalFoundationAdapter":
        adapter = cls(tick_size=tick_size)
        for event in events:
            adapter.consume(event)
        return adapter


__all__ = [
    "CanonicalFoundationAdapter",
    "FoundationAdapterCheckpoint",
    "FoundationAdapterUpdate",
]
