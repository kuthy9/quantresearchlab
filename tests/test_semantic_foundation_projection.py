from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, dataclass
import pickle

import pandas as pd
import pytest

from smc_trader.foundation_registry import (
    FOUNDATION_CANONICAL_IDENTITY,
    FOUNDATION_VERSION,
)
from smc_trader.market_state import (
    SwingGeometryNode,
    update_swing_geometry_assignments,
)
from smc_trader.model import FrozenDict, Timeframe, to_primitive
from smc_trader.semantic_foundation import (
    FoundationObjectType,
    FoundationProjection,
    FoundationProjectionReducer,
    FoundationRecord,
    FoundationRecordStatus,
)
from smc_trader.semantic_lifecycle import (
    NormalizedLifecycleTransition,
    NormalizedTransitionKind,
    SemanticLifecycleReducer,
)
from smc_trader.semantic_zones import FVGStructuralLifecycle


TZ = "America/New_York"


def _clock(minutes: int) -> pd.Timestamp:
    return pd.Timestamp("2024-06-03 09:30", tz=TZ) + pd.Timedelta(
        minutes, unit="m"
    )


def _level_history():
    created = NormalizedLifecycleTransition(
        fact_id="level-created",
        kind=NormalizedTransitionKind.LIQUIDITY_LEVEL_CREATED,
        known_at=_clock(0),
        timeframe=Timeframe.H1,
        source_event_ids=("source-level-event",),
        payload={
            "source_kind": "confirmed_swing",
            "source_identity": "v1.2-source-level",
            "side": "above",
            "price_ticks": 400,
            "tick_size": 0.25,
            "interaction_timeframe": Timeframe.M1.value,
        },
    )
    state = SemanticLifecycleReducer.reduce(
        SemanticLifecycleReducer.initial_state(), created
    )
    active = state.levels[0]
    active_interaction = state.interactions[0]
    retired_fact = NormalizedLifecycleTransition(
        fact_id="level-retired",
        kind=NormalizedTransitionKind.LIQUIDITY_LEVEL_RETIRED,
        known_at=_clock(1),
        timeframe=Timeframe.H1,
        source_event_ids=("reference-rollover-event",),
        payload={
            "level_id": active.level_id,
            "reason": "reference_rollover",
        },
    )
    state = SemanticLifecycleReducer.reduce(state, retired_fact)
    return (
        active,
        state.level(active.level_id),
        active_interaction,
        state.interaction(active.active_generation_id),
    )


def _swing_node() -> SwingGeometryNode:
    return SwingGeometryNode(
        swing_id="swing-geometry-1",
        timeframe=Timeframe.M5,
        symbol="NQ",
        instrument_id=1,
        window_start=_clock(5),
        window_end=_clock(10),
        lower_bound=100.0,
        upper_bound=105.0,
        known_at=_clock(10),
        source_candle_ids=("candle-1", "candle-2"),
    )


def _active_fvg() -> FVGStructuralLifecycle:
    return FVGStructuralLifecycle(
        fvg_id="fvg-1",
        source_creation_event_id="fvg-created-event",
        symbol="NQ",
        instrument_id=1,
        timeframe=Timeframe.M5,
        created_at=_clock(15),
        known_at=_clock(15),
    )


def test_record_is_deeply_immutable_version_bound_and_deterministic() -> None:
    active, _, _, _ = _level_history()
    first = FoundationRecord.from_dto(active)
    second = FoundationRecord.from_dto(active)

    assert first == second
    assert first.record_id == second.record_id
    assert first.record_id.startswith("foundation-record:")
    assert first.foundation_version == FOUNDATION_VERSION
    assert first.registry_identity == FOUNDATION_CANONICAL_IDENTITY
    assert isinstance(first.payload, FrozenDict)
    assert first.source_event_ids == active.source_event_ids

    with pytest.raises(FrozenInstanceError):
        first.status = FoundationRecordStatus.TERMINAL  # type: ignore[misc]
    with pytest.raises(TypeError, match="immutable"):
        first.payload["lifecycle"] = "retired"

    nested_payload = dict(first.payload)
    nested_payload["audit"] = {"roles": ["source", "confirmation"]}
    with pytest.raises(ValueError, match="canonical DTO"):
        FoundationRecord(
            object_type=first.object_type,
            object_id=first.object_id,
            status=first.status,
            known_at=first.known_at,
            payload=nested_payload,
            source_event_ids=first.source_event_ids,
        )


@pytest.mark.parametrize(
    "bad_value,expected",
    [
        (pd.Timestamp("2024-06-03", tz="UTC"), TypeError),
        (float("nan"), ValueError),
        ({1: "non-string-key"}, TypeError),
    ],
)
def test_direct_record_payload_rejects_non_primitives(
    bad_value: object,
    expected: type[Exception],
) -> None:
    active, _, _, _ = _level_history()
    valid = FoundationRecord.from_dto(active)
    payload = dict(valid.payload)
    payload["bad"] = bad_value
    with pytest.raises(expected):
        FoundationRecord(
            object_type=valid.object_type,
            object_id=valid.object_id,
            status=valid.status,
            known_at=valid.known_at,
            payload=payload,
            source_event_ids=valid.source_event_ids,
        )


def test_exact_dto_ancestry_is_derived_and_mismatch_fails_closed() -> None:
    active, _, _, _ = _level_history()
    derived = FoundationRecord.from_dto(active)
    assert derived.source_event_ids == ("source-level-event",)

    with pytest.raises(ValueError, match="exact DTO ancestry"):
        FoundationRecord.from_dto(
            active,
            source_event_ids=("different-event",),
        )


def test_geometry_requires_explicit_event_ancestry_and_sources_bind_record_id() -> None:
    node = _swing_node()
    with pytest.raises(ValueError, match="source_event_ids are required"):
        FoundationRecord.from_dto(node)

    first = FoundationRecord.from_dto(
        node,
        source_event_ids=("confirmed-swing-event",),
    )
    second = FoundationRecord.from_dto(
        node,
        source_event_ids=("alternate-authoritative-event",),
    )
    assert first.object_type is FoundationObjectType.SWING_GEOMETRY_NODE
    assert first.status is FoundationRecordStatus.FACT
    assert first.record_id != second.record_id


def test_zone_dto_uses_creation_event_as_exact_active_ancestry() -> None:
    fvg = _active_fvg()
    record = FoundationRecord.from_dto(fvg)

    assert record.object_type is FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE
    assert record.status is FoundationRecordStatus.ACTIVE
    assert record.source_event_ids == (fvg.source_creation_event_id,)
    assert record.known_at == fvg.last_updated_at


def test_serialized_records_rerun_level_interaction_and_fvg_dto_invariants() -> None:
    active, _, active_interaction, retired_interaction = _level_history()
    level_record = FoundationRecord.from_dto(active)
    level_payload = dict(level_record.payload)
    level_payload["active_generation_id"] = None
    with pytest.raises(ValueError, match="active generation"):
        FoundationRecord(
            object_type=level_record.object_type,
            object_id=level_record.object_id,
            status=level_record.status,
            known_at=level_record.known_at,
            payload=level_payload,
            source_event_ids=level_record.source_event_ids,
        )

    interaction_record = FoundationRecord.from_dto(active_interaction)
    interaction_payload = dict(interaction_record.payload)
    interaction_payload.update(
        {
            "generation_number": 2,
            "previous_generation_id": "prior-generation",
            "rearm_fact_id": None,
        }
    )
    with pytest.raises(ValueError, match="ancestry"):
        FoundationRecord(
            object_type=interaction_record.object_type,
            object_id=interaction_record.object_id,
            status=interaction_record.status,
            known_at=interaction_record.known_at,
            payload=interaction_payload,
            source_event_ids=interaction_record.source_event_ids,
        )

    terminal_record = FoundationRecord.from_dto(retired_interaction)
    terminal_payload = dict(terminal_record.payload)
    terminal_payload.update(
        {
            "first_penetration_at": terminal_payload["updated_at"],
            "max_penetration_ticks": 1,
        }
    )
    with pytest.raises(ValueError, match="path clocks|penetration stage"):
        FoundationRecord(
            object_type=terminal_record.object_type,
            object_id=terminal_record.object_id,
            status=terminal_record.status,
            known_at=terminal_record.known_at,
            payload=terminal_payload,
            source_event_ids=terminal_record.source_event_ids,
        )

    fvg_record = FoundationRecord.from_dto(_active_fvg())
    fvg_payload = dict(fvg_record.payload)
    fvg_payload["last_updated_at"] = _clock(20).isoformat()
    with pytest.raises(ValueError, match="continuous age"):
        FoundationRecord(
            object_type=fvg_record.object_type,
            object_id=fvg_record.object_id,
            status=fvg_record.status,
            known_at=_clock(20),
            payload=fvg_payload,
            source_event_ids=fvg_record.source_event_ids,
        )


def test_append_only_replay_checkpoint_and_retirement_filter_are_deterministic() -> None:
    active, retired, active_interaction, retired_interaction = _level_history()
    active_interaction_record = FoundationRecord.from_dto(active_interaction)
    active_record = FoundationRecord.from_dto(active)
    retired_interaction_record = FoundationRecord.from_dto(retired_interaction)
    retired_record = FoundationRecord.from_dto(retired)

    full = FoundationProjectionReducer.replay(
        (
            active_interaction_record,
            active_record,
            retired_interaction_record,
            retired_record,
        )
    )
    prefix = FoundationProjectionReducer.replay(
        (active_interaction_record, active_record)
    )
    checkpoint = FoundationProjectionReducer.checkpoint(prefix)
    restored = FoundationProjectionReducer.restore(checkpoint)
    resumed = FoundationProjectionReducer.replay(
        (retired_interaction_record, retired_record), initial=restored
    )

    assert resumed == full
    assert len(full.records) == 4
    assert full.records_for(
        FoundationObjectType.LIQUIDITY_LEVEL, active.level_id
    ) == (active_record, retired_record)
    assert full.latest_records == (retired_interaction_record, retired_record)
    assert full.active_records == ()
    assert full.terminal_records == (retired_interaction_record, retired_record)
    assert full.active_history == (active_interaction_record, active_record)
    assert full.terminal_history == (retired_interaction_record, retired_record)
    assert full.active_dol_candidate_ids == ()
    assert prefix.active_dol_candidate_ids == (active.level_id,)
    assert full.asof == retired.updated_at
    assert FoundationProjectionReducer.reduce(prefix, active_record) is prefix


def test_same_clock_canonical_revisions_keep_order_but_lifecycle_cannot_rewind() -> None:
    active, _, active_interaction, _ = _level_history()
    active_interaction_record = FoundationRecord.from_dto(active_interaction)
    active_record = FoundationRecord.from_dto(active)
    disarmed_payload = dict(active_record.payload)
    disarmed_payload.update(
        {
            "lifecycle": "disarmed",
            "active_generation_id": None,
            "last_terminal_generation_id": active.active_generation_id,
        }
    )
    disarmed = FoundationRecord(
        object_type=active_record.object_type,
        object_id=active_record.object_id,
        status=FoundationRecordStatus.INACTIVE,
        known_at=active_record.known_at,
        payload=disarmed_payload,
        source_event_ids=active_record.source_event_ids,
    )
    projection = FoundationProjectionReducer.replay(
        (active_interaction_record, active_record, disarmed)
    )
    assert projection.records == (
        active_interaction_record,
        active_record,
        disarmed,
    )

    rewound_payload = dict(active_record.payload)
    rewound_payload["source_event_ids"] = (
        *active_record.source_event_ids,
        "rewind-event",
    )
    rewound = FoundationRecord(
        object_type=active_record.object_type,
        object_id=active_record.object_id,
        status=active_record.status,
        known_at=active_record.known_at,
        payload=rewound_payload,
        source_event_ids=(
            *active_record.source_event_ids,
            "rewind-event",
        ),
    )
    with pytest.raises(ValueError, match="not preregistered"):
        FoundationProjectionReducer.reduce(projection, rewound)


def test_checkpoint_integrity_and_terminal_history_cannot_be_rewritten() -> None:
    active, retired, active_interaction, retired_interaction = _level_history()
    active_interaction_record = FoundationRecord.from_dto(active_interaction)
    active_record = FoundationRecord.from_dto(active)
    retired_interaction_record = FoundationRecord.from_dto(retired_interaction)
    retired_record = FoundationRecord.from_dto(retired)
    projection = FoundationProjectionReducer.replay(
        (
            active_interaction_record,
            active_record,
            retired_interaction_record,
            retired_record,
        )
    )
    checkpoint = FoundationProjectionReducer.checkpoint(projection)
    object.__setattr__(checkpoint, "checkpoint_id", "foundation-checkpoint:corrupt")
    with pytest.raises(ValueError, match="integrity"):
        FoundationProjectionReducer.restore(checkpoint)

    rewritten_payload = dict(active_record.payload)
    rewritten_payload["updated_at"] = _clock(2).isoformat()
    later_active_rewrite = FoundationRecord(
        object_type=active_record.object_type,
        object_id=active_record.object_id,
        status=active_record.status,
        known_at=_clock(2),
        payload=rewritten_payload,
        source_event_ids=active_record.source_event_ids,
    )
    with pytest.raises(ValueError, match="terminal object is immutable"):
        FoundationProjectionReducer.reduce(projection, later_active_rewrite)


@dataclass(frozen=True)
class _UnknownFoundationLikeDTO:
    semantic_version: str = FOUNDATION_VERSION


def test_unknown_types_fail_closed_without_generic_dataclass_admission() -> None:
    with pytest.raises(TypeError, match="unsupported semantic-foundation DTO"):
        FoundationRecord.from_dto(
            _UnknownFoundationLikeDTO(),
            source_event_ids=("event-1",),
        )

    active, _, _, _ = _level_history()
    valid = FoundationRecord.from_dto(active)
    with pytest.raises(ValueError):
        FoundationRecord(
            object_type="unknown_foundation_type",  # type: ignore[arg-type]
            object_id=valid.object_id,
            status=valid.status,
            known_at=valid.known_at,
            payload=valid.payload,
            source_event_ids=valid.source_event_ids,
        )


def test_projection_indexes_are_derived_pickle_safe_and_history_equivalent() -> None:
    child = _swing_node()
    parent = SwingGeometryNode(
        swing_id="swing-geometry-parent",
        timeframe=Timeframe.H1,
        symbol=child.symbol,
        instrument_id=child.instrument_id,
        window_start=_clock(0),
        window_end=_clock(15),
        lower_bound=90.0,
        upper_bound=110.0,
        known_at=_clock(15),
        source_candle_ids=("parent-candle-1", "parent-candle-2"),
    )
    first_history = update_swing_geometry_assignments(
        (child, parent),
        known_at=child.known_at,
    )
    complete_history = update_swing_geometry_assignments(
        (child, parent),
        first_history,
        known_at=parent.known_at,
    )
    records = (
        FoundationRecord.from_dto(
            child,
            source_event_ids=("child-swing-event",),
        ),
        FoundationRecord.from_dto(
            first_history[0],
            source_event_ids=("child-swing-event",),
        ),
        FoundationRecord.from_dto(
            parent,
            source_event_ids=("parent-swing-event",),
        ),
        *(
            FoundationRecord.from_dto(
                assignment,
                source_event_ids=(
                    "child-swing-event",
                    "parent-swing-event",
                ),
            )
            for assignment in complete_history[1:]
        ),
    )

    incremental = FoundationProjectionReducer.replay(records)
    arbitrary_history = FoundationProjection(records=records)
    assert arbitrary_history == incremental
    assert arbitrary_history.latest_records == incremental.latest_records
    assert copy.deepcopy(incremental) is incremental

    primitive = to_primitive(incremental)
    assert "_record_ids_cache" not in primitive
    assert "_latest_records_by_key_cache" not in primitive
    assert "_swing_geometry_views_cache" not in primitive
    assert "_swing_assignment_incumbents_cache" not in primitive
    with pytest.raises(TypeError):
        incremental._swing_geometry_views_cache["invented"] = object()

    checkpoint = FoundationProjectionReducer.checkpoint(incremental)
    restored_checkpoint = pickle.loads(pickle.dumps(checkpoint))
    restored = FoundationProjectionReducer.restore(restored_checkpoint)
    assert restored == incremental
    assert restored.latest_records == incremental.latest_records
    assert restored_checkpoint.checkpoint_id == checkpoint.checkpoint_id
    assert restored._record_ids_cache == incremental._record_ids_cache
