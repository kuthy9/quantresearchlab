from __future__ import annotations

import copy
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest

from smc_trader.causal import CausalMarketReader
from smc_trader.market_state import (
    SwingGeometryNode,
    dual_range_location,
    foundation_record_from_projection_event,
    foundation_record_projection_event,
    replay_atomic_market_snapshot,
)
from smc_trader.event_store import event_order_key
from smc_trader.model import (
    Bar,
    Direction,
    EventKind,
    EventOrigin,
    SwingSide,
    Timeframe,
)
from smc_trader.observation import CausalObserver
from smc_trader.semantic_foundation import (
    FoundationObjectType,
    FoundationProjectionReducer,
    FoundationRecord,
    FoundationRecordStatus,
)
from smc_trader.semantic_lifecycle import (
    GenerationLifecycle,
    LiquidityLevelLifecycle,
    StructureGenerationLifecycle,
    canonical_semantic_id,
)

from .helpers import MODEL_SCALE_SPECS
from .test_foundation_adapter import _bar as _adapter_bar
from .test_foundation_adapter import _atomic as _adapter_atomic
from .test_foundation_adapter import _clock as _adapter_clock
from .test_foundation_adapter import _cross_level
from .test_foundation_adapter import _seed_level
from .test_foundation_adapter import CanonicalFoundationAdapter
from .test_observer import _all_typed_observer_config, _tick_aligned_bars
from .test_semantic_foundation_geometry import _balance_range, _swing


def _eye() -> tuple[CausalMarketReader, CausalObserver]:
    return (
        CausalMarketReader(scale_specs=MODEL_SCALE_SPECS),
        CausalObserver(
            _all_typed_observer_config(
                materialize_event_view=False,
                eye_authority_mode=True,
            )
        ),
    )


def _observe(observer: CausalObserver, reader: CausalMarketReader, bars):
    observation = None
    for bar in bars:
        observation = observer.observe(reader.on_bar(bar))
    assert observation is not None
    assert observation.market_snapshot is not None
    assert observation.market_snapshot.foundation is not None
    return observation


def test_eye_foundation_clock_survives_real_clock_only_real_sequence() -> None:
    reader, observer = _eye()
    start = pd.Timestamp("2024-06-06 23:08", tz="America/New_York")
    observations = []
    for index in range(3):
        observations.append(
            observer.observe(
                reader.on_bar(
                    Bar(
                        start=start + pd.Timedelta(minutes=index),
                        open=19_000.0,
                        high=19_000.25,
                        low=18_999.75,
                        close=19_000.0,
                        volume=0.0 if index == 1 else 10.0,
                        symbol="NQM4",
                        instrument_id=13743,
                        synthetic_no_trade=index == 1,
                    )
                )
            )
        )

    adapter = observer._foundation_adapter
    assert adapter is not None
    registered = next(
        item
        for item in adapter.lifecycle.registered_bar_clocks
        if item.timeframe is Timeframe.M1
    )
    real = next(
        item
        for item in adapter.lifecycle.real_bar_clocks
        if item.timeframe is Timeframe.M1
    )
    clock_only_roots = tuple(
        event
        for event in observations[1].semantic_events_this_update
        if event.kind is EventKind.BAR_COMPLETED
        and event.timeframe is Timeframe.M1
        and event.evidence.get("real_completed") is False
        and event.evidence.get("clock_only") is True
    )
    assert len(clock_only_roots) == 1
    assert clock_only_roots[0].known_at == pd.Timestamp(
        "2024-06-06 23:10", tz="America/New_York"
    )
    assert not any(
        event.kind is EventKind.FOUNDATION_STATE_CHANGED
        for event in observations[1].semantic_events_this_update
    )
    assert registered.count == 3
    assert registered.last_completed_at == pd.Timestamp(
        "2024-06-06 23:11", tz="America/New_York"
    )
    assert real.count == 2
    assert real.last_completed_at == registered.last_completed_at
    assert adapter.lifecycle.epoch == 0


def test_foundation_noop_clocks_skip_geometry_and_cluster_rebuilds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimized_reader, optimized = _eye()
    eager_reader, eager = _eye()
    calls = Counter()
    eager_calls = Counter()
    optimized_geometry = optimized._foundation_geometry_plans
    optimized_clusters = optimized._foundation_update_clusters
    eager_geometry = eager._foundation_geometry_plans
    eager_clusters = eager._foundation_update_clusters

    def count_optimized_geometry(**kwargs):
        calls["geometry"] += 1
        return optimized_geometry(**kwargs)

    def count_optimized_clusters(**kwargs):
        calls["clusters"] += 1
        return optimized_clusters(**kwargs)

    def count_eager_geometry(**kwargs):
        eager_calls["geometry"] += 1
        return eager_geometry(**kwargs)

    def count_eager_clusters(**kwargs):
        eager_calls["clusters"] += 1
        return eager_clusters(**kwargs)

    monkeypatch.setattr(
        optimized,
        "_foundation_geometry_plans",
        count_optimized_geometry,
    )
    monkeypatch.setattr(
        optimized,
        "_foundation_update_clusters",
        count_optimized_clusters,
    )
    monkeypatch.setattr(eager, "_foundation_geometry_plans", count_eager_geometry)
    monkeypatch.setattr(eager, "_foundation_update_clusters", count_eager_clusters)
    monkeypatch.setattr(
        eager,
        "_foundation_geometry_invalidated",
        lambda _events: True,
    )
    monkeypatch.setattr(
        eager,
        "_foundation_cluster_membership_invalidated",
        lambda _records, _events: True,
    )

    optimized_observation = eager_observation = None
    bars = _tick_aligned_bars(3)
    for bar in bars:
        optimized_observation = optimized.observe(
            optimized_reader.on_bar(bar)
        )
        eager_observation = eager.observe(eager_reader.on_bar(bar))

    assert optimized_observation is not None
    assert eager_observation is not None
    assert calls == Counter()
    assert eager_calls == Counter({"geometry": 3, "clusters": 3})
    assert optimized.audit_store.events() == eager.audit_store.events()
    assert optimized._foundation_adapter is not None
    assert eager._foundation_adapter is not None
    assert (
        optimized._foundation_adapter.checkpoint()
        == eager._foundation_adapter.checkpoint()
    )
    assert optimized_observation.market_snapshot is not None
    assert eager_observation.market_snapshot is not None
    assert (
        optimized_observation.market_snapshot.fingerprint
        == eager_observation.market_snapshot.fingerprint
    )


def test_eye_foundation_m5_clock_recovers_after_contaminated_bucket() -> None:
    reader, observer = _eye()
    start = pd.Timestamp("2024-06-06 23:00", tz="America/New_York")
    observations = []
    for index in range(15):
        observations.append(
            observer.observe(
                reader.on_bar(
                    Bar(
                        start=start + pd.Timedelta(index, unit="min"),
                        open=19_000.0,
                        high=19_000.25,
                        low=18_999.75,
                        close=19_000.0,
                        volume=0.0 if index == 9 else 10.0,
                        symbol="NQM4",
                        instrument_id=13743,
                        synthetic_no_trade=index == 9,
                    )
                )
            )
        )

    adapter = observer._foundation_adapter
    assert adapter is not None
    registered = next(
        item
        for item in adapter.lifecycle.registered_bar_clocks
        if item.timeframe is Timeframe.M5
    )
    real = next(
        item
        for item in adapter.lifecycle.real_bar_clocks
        if item.timeframe is Timeframe.M5
    )
    contaminated = tuple(
        event
        for event in observations[9].semantic_events_this_update
        if event.kind is EventKind.BAR_COMPLETED
        and event.timeframe is Timeframe.M5
    )
    assert len(contaminated) == 1
    assert contaminated[0].evidence["real_completed"] is False
    assert contaminated[0].evidence["clock_only"] is True
    assert contaminated[0].evidence["real_minutes"] == 4
    assert contaminated[0].evidence["synthetic_minutes"] == 1
    assert registered.count == 3
    assert registered.last_completed_at == pd.Timestamp(
        "2024-06-06 23:15", tz="America/New_York"
    )
    assert real.count == 2
    assert real.last_completed_at == registered.last_completed_at


def test_eye_publishes_clock_only_roots_for_every_contaminated_timeframe() -> None:
    reader, observer = _eye()
    start = pd.Timestamp("2024-06-06 22:00", tz="America/New_York")
    update = None
    for index in range(240):
        update = reader.on_bar(
            Bar(
                start=start + pd.Timedelta(index, unit="min"),
                open=19_000.0,
                high=19_000.25,
                low=18_999.75,
                close=19_000.0,
                volume=0.0 if index == 69 else 10.0,
                symbol="NQM4",
                instrument_id=13743,
                synthetic_no_trade=index == 69,
            )
        )
    assert update is not None

    observer.observe(update)
    roots = tuple(
        event
        for event in observer.audit_store.events()
        if event.kind is EventKind.BAR_COMPLETED
        and event.evidence.get("real_completed") is False
        and event.evidence.get("clock_only") is True
    )
    by_timeframe = {event.timeframe: event for event in roots}
    assert set(by_timeframe) == {
        Timeframe.M1,
        Timeframe.M5,
        Timeframe.M15,
        Timeframe.H1,
        Timeframe.H4,
    }
    expected_coverage = {
        Timeframe.M1: (1, 0, 1),
        Timeframe.M5: (5, 4, 1),
        Timeframe.M15: (15, 14, 1),
        Timeframe.H1: (60, 59, 1),
        Timeframe.H4: (240, 239, 1),
    }
    for timeframe, (expected, real, synthetic) in expected_coverage.items():
        evidence = by_timeframe[timeframe].evidence
        assert evidence["complete"] is True
        assert evidence["observed_minutes"] == expected
        assert evidence["expected_minutes"] == expected
        assert evidence["real_minutes"] == real
        assert evidence["synthetic_minutes"] == synthetic

    clock_only_ids = {event.event_id for event in roots}
    assert not any(
        clock_only_ids.intersection(event.source_event_ids)
        for event in observer.audit_store.events()
        if event.origin is EventOrigin.SEMANTIC_ATOMIC
    )


def _descending_tail(start: pd.Timestamp, count: int) -> tuple[Bar, ...]:
    return tuple(
        Bar(
            start=start + pd.Timedelta(minutes=index),
            open=(price := 20_007.0 - index) + 0.25,
            high=price + 0.50,
            low=price - 0.50,
            close=price - 0.25,
            volume=200 + index,
            symbol="NQH5",
            instrument_id=1,
        )
        for index in range(count)
    )


def _m1_structure_bars() -> tuple[Bar, ...]:
    pairs = (
        (10, 8),
        (11, 9),
        (13, 10),
        (12, 9),
        (11, 8),
        (12, 9),
        (14, 10),
        (13, 10),
        (12, 9),
        (13, 10),
        (15, 11),
    )
    start = pd.Timestamp("2025-01-05 18:00", tz="America/New_York")
    values: list[Bar] = []
    for index, (high_units, low_units) in enumerate(pairs):
        high = 20_000.0 + high_units * 0.25
        low = 20_000.0 + low_units * 0.25
        close = low + 0.25
        values.append(
            Bar(
                start=start + pd.Timedelta(minutes=index),
                open=close,
                high=high,
                low=low,
                close=close,
                volume=100 + index,
                symbol="NQH5",
                instrument_id=1,
            )
        )
    return tuple(values)


def _m5_parent_m1_child_mss_bars() -> tuple[Bar, ...]:
    coarse = (
        (10, 8),
        (11, 9),
        (13, 10),
        (12, 9),
        (11, 8),
        (12, 9),
        (14, 10),
        (13, 10),
        (12, 9),
        (13, 10),
        (15, 11),
        (14, 10),
        (16, 12),
    )
    fine = (
        (12, 10),
        (13, 11),
        (15, 12),
        (14, 11),
        (13, 10),
        (14, 11),
        (16, 12),
        (15, 12),
        (14, 11),
        (15, 12),
        (17, 13),
    )
    bearish = ((16, 12, 12), (15, 10, 10), (14, 8, 8))
    start = pd.Timestamp("2025-01-05 18:00", tz="America/New_York")
    bars: list[Bar] = []

    def append(high_units: int, low_units: int, close_units: int | None = None):
        index = len(bars)
        high = 20_000.0 + high_units * 0.25
        low = 20_000.0 + low_units * 0.25
        close = low + 0.25 if close_units is None else 20_000.0 + close_units * 0.25
        bars.append(
            Bar(
                start=start + pd.Timedelta(minutes=index),
                open=close,
                high=high,
                low=low,
                close=close,
                volume=100 + index,
                symbol="NQH5",
                instrument_id=1,
            )
        )

    for high_units, low_units in coarse:
        for _ in range(5):
            append(high_units, low_units)
    for high_units, low_units in fine:
        append(high_units, low_units)
    for high_units, low_units, close_units in bearish:
        append(high_units, low_units, close_units)
    return tuple(bars)


def _assert_atomic_replay_parity(observer: CausalObserver, observation) -> None:
    expected = observation.market_snapshot
    replayed = replay_atomic_market_snapshot(
        observer.audit_store.events(),
        semantic_registry_identity=observer.semantic_registry.identity,
        foundation_records=observer.materialize_foundation_history(),
    )
    assert replayed.foundation == expected.foundation
    assert replayed.timeframe_states == expected.timeframe_states
    assert replayed.relations == expected.relations
    assert replayed.session == expected.session
    assert replayed.foundation_range_locations == expected.foundation_range_locations


def _legacy_foundation_transport_stream(
    observer: CausalObserver,
) -> tuple:
    """Materialize the retained v1 transport only for legacy decoder tests."""

    events = observer.audit_store.events()
    records = observer.materialize_foundation_history()
    if not records:
        return events
    published_at = observer.last_market_snapshot.asof
    start_sequence = 1 + max(
        (
            event.sequence_no
            for event in events
            if event.known_at == published_at
        ),
        default=0,
    )
    transports = tuple(
        foundation_record_projection_event(
            record,
            timeframe=Timeframe.M1,
            published_at=published_at,
            sequence_no=start_sequence + index,
        )
        for index, record in enumerate(records)
    )
    return tuple(sorted((*events, *transports), key=event_order_key))


_JUNE_2024_CAUSAL_FRONT = (
    Path(__file__).resolve().parents[1]
    / "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"
)


@pytest.mark.skipif(
    not _JUNE_2024_CAUSAL_FRONT.exists(),
    reason="local materialized 2024-06 causal front is unavailable",
)
def test_2024_06_tracker_only_evidence_stays_noncanonical_and_replayable() -> None:
    frame = pd.read_parquet(
        _JUNE_2024_CAUSAL_FRONT,
        filters=[
            ("ts", ">=", pd.Timestamp("2024-06-01", tz="America/New_York")),
            ("ts", "<", pd.Timestamp("2024-07-01", tz="America/New_York")),
        ],
    ).iloc[:495]
    bars = tuple(
        Bar(
            start=clock,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
            symbol=str(row.symbol),
            instrument_id=int(row.instrument_id),
        )
        for clock, row in frame.iterrows()
    )
    reader, observer = _eye()
    observation = _observe(observer, reader, bars)
    events = observer.audit_store.events()
    opposite_bos = next(
        event
        for event in events
        if event.kind is EventKind.QUALIFIED_BOS
        and event.event_id == "22dd71de76fd07356015e14e"
    )
    opposite_assignment = next(
        event
        for event in events
        if event.kind is EventKind.PROTECTED_SWING_ASSIGNED
        and event.event_id == "531d843b8ab0b17bf0980d25"
    )
    structures = tuple(
        record
        for record in observation.market_snapshot.foundation.latest_records
        if record.object_type is FoundationObjectType.STRUCTURE_GENERATION
    )
    initial_external = tuple(
        record
        for record in structures
        if record.payload["scope"] == "external"
        and record.payload["direction"] == Direction.LONG.value
        and record.payload["started_at"]
        == "2024-06-02T18:28:00-04:00"
    )

    assert opposite_bos.direction is Direction.SHORT
    assert opposite_assignment.direction is Direction.SHORT
    assert len(initial_external) == 1
    assert opposite_bos.event_id not in initial_external[0].payload["bos_event_ids"]
    assert (
        initial_external[0].payload["protected_swing_assignment_event_id"]
        != opposite_assignment.event_id
    )
    legacy_relation = observation.market_snapshot.relations["5m__1m"]
    assert legacy_relation.parent_direction is Direction.SHORT
    active_relations = tuple(
        generation
        for generation in observer._foundation_adapter.lifecycle.relation_generations
        if generation.parent_tf is Timeframe.M5
        and generation.child_tf is Timeframe.M1
        and generation.lifecycle is GenerationLifecycle.ACTIVE
    )
    assert len(active_relations) == 1
    canonical_parent = observer._foundation_adapter.lifecycle.structure(
        active_relations[0].parent_structure_generation_id
    )
    assert canonical_parent.scope.value == "external"
    assert canonical_parent.direction is Direction.LONG
    assert active_relations[0].last_updated_at < observation.market_snapshot.asof
    _assert_atomic_replay_parity(observer, observation)

    events = _legacy_foundation_transport_stream(observer)
    transports = tuple(
        event
        for event in events
        if event.kind is EventKind.FOUNDATION_STATE_CHANGED
    )
    relation_transport, relation_record = next(
        (event, record)
        for event in transports
        if (
            (record := foundation_record_from_projection_event(event)).object_type
            is FoundationObjectType.RELATION_GENERATION
            and record.payload["termination_reason"] == "child_realigned"
            and any(
                source.kind is EventKind.ACCEPTANCE_CONFIRMED
                for source_id in record.source_event_ids
                if (source := next(item for item in events if item.event_id == source_id))
            )
        )
    )
    correct_acceptance_id = next(
        source_id
        for source_id in relation_record.source_event_ids
        if next(item for item in events if item.event_id == source_id).kind
        is EventKind.ACCEPTANCE_CONFIRMED
    )
    wrong_acceptance = next(
        event
        for event in events
        if event.kind is EventKind.ACCEPTANCE_CONFIRMED
        and event.event_id != correct_acceptance_id
        and event.known_at <= relation_record.known_at
    )
    forged_sources = tuple(
        wrong_acceptance.event_id
        if source_id == correct_acceptance_id
        else source_id
        for source_id in relation_record.source_event_ids
    )
    forged_payload = dict(relation_record.payload)
    forged_payload["source_event_ids"] = forged_sources
    forged_record = replace(
        relation_record,
        payload=forged_payload,
        source_event_ids=forged_sources,
    )
    forged_transport = foundation_record_projection_event(
        forged_record,
        timeframe=relation_transport.timeframe,
        published_at=relation_transport.known_at,
        sequence_no=relation_transport.sequence_no,
    )
    tampered = tuple(
        forged_transport if event.event_id == relation_transport.event_id else event
        for event in events
    )
    with pytest.raises(ValueError, match="wrong Acceptance"):
        replay_atomic_market_snapshot(
            tampered,
            semantic_registry_identity=observer.semantic_registry.identity,
        )


def test_eye_foundation_cold_ledger_is_exact_and_not_event_republished() -> None:
    reader, observer = _eye()
    observation = _observe(observer, reader, _tick_aligned_bars(61))
    projection = observation.market_snapshot.foundation
    events = observer.audit_store.events()
    cold_records = observer.materialize_foundation_history()
    assert not any(
        event.kind is EventKind.FOUNDATION_STATE_CHANGED for event in events
    )
    assert len(cold_records) == projection.record_count
    assert (
        FoundationProjectionReducer.replay(cold_records) == projection
    )
    order = {event.event_id: index for index, event in enumerate(events)}
    for record in cold_records:
        assert all(source_id in order for source_id in record.source_event_ids)
        assert all(
            events[order[source_id]].origin
            in {EventOrigin.NORMALIZED_DATA, EventOrigin.SEMANTIC_ATOMIC}
            for source_id in record.source_event_ids
        )
    adapter = observer._foundation_adapter
    assert adapter.pending_record_delta.records == ()
    assert adapter.pending_record_delta.end_count == projection.record_count
    assert not hasattr(observation, "foundation_record_delta")
    assert not any(
        event.kind is EventKind.FOUNDATION_STATE_CHANGED
        for event in observation.semantic_events_this_update
    )
    _assert_atomic_replay_parity(observer, observation)


def test_cold_replay_tracks_superseded_record_ids_without_hot_history() -> None:
    reader, observer = _eye()
    observation = _observe(observer, reader, _m5_parent_m1_child_mss_bars())
    cold_records = observer.materialize_foundation_history()
    current_ids = {
        record.record_id
        for record in observation.market_snapshot.foundation.current_records
    }
    superseded = next(
        record for record in cold_records if record.record_id not in current_ids
    )

    replayed = replay_atomic_market_snapshot(
        observer.audit_store.events(),
        semantic_registry_identity=observer.semantic_registry.identity,
        foundation_records=(*cold_records, superseded),
    )
    assert replayed.foundation == observation.market_snapshot.foundation

    conflicting = copy.copy(superseded)
    object.__setattr__(
        conflicting,
        "source_event_ids",
        (*superseded.source_event_ids, "forged-source"),
    )
    with pytest.raises(ValueError, match="record_id collision"):
        replay_atomic_market_snapshot(
            observer.audit_store.events(),
            semantic_registry_identity=observer.semantic_registry.identity,
            foundation_records=(*cold_records, conflicting),
        )


def test_atomic_replay_rejects_structure_transition_and_terminal_kind_swaps() -> None:
    reader, observer = _eye()
    observation = _observe(observer, reader, _m5_parent_m1_child_mss_bars())
    events = _legacy_foundation_transport_stream(observer)
    authoritative = {event.event_id: event for event in events}
    transports = {
        foundation_record_from_projection_event(event).record_id: event
        for event in events
        if event.kind is EventKind.FOUNDATION_STATE_CHANGED
    }
    records = observer.materialize_foundation_history()

    def replay_with(
        original_record: FoundationRecord,
        forged: FoundationRecord,
        message: str,
    ) -> None:
        original = transports[original_record.record_id]
        replacement = foundation_record_projection_event(
            forged,
            timeframe=original.timeframe,
            published_at=original.known_at,
            sequence_no=original.sequence_no,
        )
        tampered = tuple(
            sorted(
                (
                    replacement if event.event_id == original.event_id else event
                    for event in events
                ),
                key=event_order_key,
            )
        )
        with pytest.raises(ValueError, match=message):
            replay_atomic_market_snapshot(
                tampered,
                semantic_registry_identity=observer.semantic_registry.identity,
            )

    structure = next(
        record
        for record in records
        if record.object_type is FoundationObjectType.STRUCTURE_GENERATION
        and record.payload["scope"] == "external"
        and record.payload["confirmation_event_id"] is not None
    )
    structure_bar = next(
        event
        for event in events
        if event.kind is EventKind.BAR_COMPLETED
        and event.timeframe.value == structure.payload["timeframe"]
        and event.known_at == structure.known_at
    )
    structure_payload = dict(structure.payload)
    old_structure_sources = {
        structure_payload["origin_event_id"],
        structure_payload["confirmation_event_id"],
    }
    structure_sources = tuple(
        dict.fromkeys(
            (
                *(
                    identity
                    for identity in structure.source_event_ids
                    if identity not in old_structure_sources
                ),
                structure_bar.event_id,
            )
        )
    )
    structure_payload.update(
        {
            "origin_event_id": structure_bar.event_id,
            "confirmation_event_id": structure_bar.event_id,
            "source_event_ids": structure_sources,
        }
    )
    forged_structure = FoundationRecord(
        object_type=structure.object_type,
        object_id=structure.object_id,
        status=structure.status,
        known_at=structure.known_at,
        payload=structure_payload,
        source_event_ids=structure_sources,
    )
    replay_with(
        structure,
        forged_structure,
        "Structure Generation origin.*kind",
    )

    transition = next(
        record
        for record in records
        if record.object_type is FoundationObjectType.STRUCTURE_TRANSITION
        and record.payload["mss_event_ids"]
    )
    transition_bar = next(
        event
        for event in events
        if event.kind is EventKind.BAR_COMPLETED
        and event.timeframe.value == transition.payload["timeframe"]
        and event.known_at == pd.Timestamp(transition.payload["started_at"])
    )
    old_mss_ids = set(transition.payload["mss_event_ids"])
    transition_sources = tuple(
        dict.fromkeys(
            (
                *(
                    identity
                    for identity in transition.source_event_ids
                    if identity not in old_mss_ids
                ),
                transition_bar.event_id,
            )
        )
    )
    transition_payload = dict(transition.payload)
    transition_payload.update(
        {
            "mss_event_ids": (transition_bar.event_id,),
            "source_event_ids": transition_sources,
        }
    )
    forged_transition = FoundationRecord(
        object_type=transition.object_type,
        object_id=transition.object_id,
        status=transition.status,
        known_at=transition.known_at,
        payload=transition_payload,
        source_event_ids=transition_sources,
    )
    replay_with(
        transition,
        forged_transition,
        "Structure Transition MSS evidence.*kind",
    )

    interaction = next(
        record
        for record in records
        if record.object_type
        is FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION
        and record.status is FoundationRecordStatus.TERMINAL
        and record.payload["terminal_state"] in {"sweep", "acceptance"}
        and any(
            authoritative[event_id].kind is EventKind.LEVEL_PENETRATED
            for event_id in record.source_event_ids
        )
    )
    penetration_id = next(
        event_id
        for event_id in interaction.source_event_ids
        if authoritative[event_id].kind is EventKind.LEVEL_PENETRATED
    )
    interaction_payload = dict(interaction.payload)
    interaction_payload["terminal_event_id"] = penetration_id
    forged_interaction = FoundationRecord(
        object_type=interaction.object_type,
        object_id=interaction.object_id,
        status=interaction.status,
        known_at=interaction.known_at,
        payload=interaction_payload,
        source_event_ids=interaction.source_event_ids,
    )
    replay_with(
        interaction,
        forged_interaction,
        "Liquidity Interaction terminal.*kind",
    )


def test_group3_fvg_terminal_freezes_age_and_exact_terminal_ancestry() -> None:
    prefix = _tick_aligned_bars(30)
    bars = (*prefix, *_descending_tail(prefix[-1].end, 6))
    reader, observer = _eye()
    observation = _observe(observer, reader, bars)
    projection = observation.market_snapshot.foundation
    cold_records = observer.materialize_foundation_history()
    fvg_records = tuple(
        record
        for record in cold_records
        if record.object_type is FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE
    )
    terminal = tuple(
        record
        for record in fvg_records
        if record.status is FoundationRecordStatus.TERMINAL
        and record.payload["availability"] == "invalidated"
    )
    assert terminal
    events = {event.event_id: event for event in observer.audit_store.events()}
    for record in terminal:
        history = tuple(
            item
            for item in cold_records
            if item.object_type is record.object_type
            and item.object_id == record.object_id
        )
        assert len(history) == 2
        assert history[0].payload["age_bars"] == 0
        assert record.payload["age_bars"] >= 1
        assert record.payload["age_seconds"] > 0
        assert record.source_event_ids == tuple(record.payload["terminal_source_event_ids"])
        kinds = tuple(events[source_id].kind for source_id in record.source_event_ids)
        assert kinds[0] is EventKind.FVG_CREATED
        assert kinds.count(EventKind.BAR_COMPLETED) == 1
        assert kinds[-1] is EventKind.FVG_INVALIDATED
        assert all(
            events[source_id].origin
            in {EventOrigin.NORMALIZED_DATA, EventOrigin.SEMANTIC_ATOMIC}
            for source_id in record.source_event_ids
        )
    _assert_atomic_replay_parity(observer, observation)


def test_contract_reset_keeps_foundation_history_and_explicitly_closes_fvg() -> None:
    prefix = _tick_aligned_bars(30)
    reader, observer = _eye()
    before = _observe(observer, reader, prefix)
    active_ids = {
        record.object_id
        for record in before.market_snapshot.foundation.active_records
        if record.object_type is FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE
    }
    assert active_ids
    prior = prefix[-1]
    reset_bar = Bar(
        start=prior.end,
        open=prior.close,
        high=prior.close + 0.50,
        low=prior.close - 0.50,
        close=prior.close + 0.25,
        volume=500.0,
        symbol="NQM5",
        instrument_id=2,
    )
    observation = observer.observe(reader.on_bar(reset_bar))
    projection = observation.market_snapshot.foundation
    reset_events = tuple(
        event
        for event in observation.semantic_events_this_update
        if event.kind is EventKind.MARKET_EPOCH_RESET
    )
    assert len(reset_events) == 1
    reset_id = reset_events[0].event_id
    cold_records = observer.materialize_foundation_history()
    for fvg_id in active_ids:
        history = tuple(
            record
            for record in cold_records
            if record.object_type
            is FoundationObjectType.FVG_STRUCTURAL_LIFECYCLE
            and record.object_id == fvg_id
        )
        assert history[0].status is FoundationRecordStatus.ACTIVE
        assert history[-1].status is FoundationRecordStatus.TERMINAL
        assert history[-1].payload["terminal_reason"] == "contract_rollover"
        assert reset_id in history[-1].source_event_ids
    _assert_atomic_replay_parity(observer, observation)


def test_foundation_cold_ledger_failure_does_not_commit_staged_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, observer = _eye()
    bars = _tick_aligned_bars(30)
    for bar in bars[:14]:
        observer.observe(reader.on_bar(bar))
    raised = False

    def reject_ledger(_records):
        raise RuntimeError("injected foundation cold ledger failure")

    monkeypatch.setattr(
        observer._foundation_adapter._record_ledger,
        "preview_append",
        reject_ledger,
    )
    failed_update = None
    for bar in bars[14:]:
        checkpoint = observer._foundation_adapter.checkpoint()
        cold_records = observer.materialize_foundation_history()
        event_count = len(observer.audit_store)
        update = reader.on_bar(bar)
        try:
            observer.observe(update)
        except RuntimeError as error:
            assert "injected foundation cold ledger failure" in str(error)
            raised = True
            assert observer._foundation_adapter.checkpoint() == checkpoint
            assert observer.materialize_foundation_history() == cold_records
            new_events = observer.audit_store.events_since(event_count)
            assert new_events
            assert not any(
                event.kind is EventKind.FOUNDATION_STATE_CHANGED
                for event in new_events
            )
            failed_update = update
            break
    assert raised
    assert failed_update is not None
    with pytest.raises(RuntimeError, match="discard this observer"):
        observer.observe(failed_update)


def test_equal_tick_structure_has_no_range_but_balance_and_later_range_survive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, observer = _eye()
    observer._active_timeframes = (Timeframe.H1,)
    adapter = CanonicalFoundationAdapter(tick_size=0.25)
    exact_events = {}
    monkeypatch.setattr(
        observer,
        "_foundation_exact_event",
        lambda event_id: exact_events[event_id],
    )

    def publish_structure(
        minute: int,
        label: str,
        *,
        low_price: float,
        high_price: float,
    ):
        clock = _adapter_clock(minute)
        bar = _adapter_bar(minute, event_id=f"bar:{label}")
        high_event = _adapter_atomic(
            f"swing-high:{label}",
            EventKind.SWING_CONFIRMED,
            minute,
            1,
            timeframe=Timeframe.H1,
            source_event_ids=(bar.event_id,),
            evidence={"source_entity_id": f"high:{label}", "side": "high"},
            side="above",
            price=high_price,
            direction=Direction.LONG,
        )
        low_event = _adapter_atomic(
            f"swing-low:{label}",
            EventKind.SWING_CONFIRMED,
            minute,
            2,
            timeframe=Timeframe.H1,
            source_event_ids=(bar.event_id,),
            evidence={"source_entity_id": f"low:{label}", "side": "low"},
            side="below",
            price=low_price,
            direction=Direction.SHORT,
        )
        origin = _adapter_atomic(
            f"structure:{label}",
            EventKind.STRUCTURE_DIRECTION_CONFIRMED,
            minute,
            3,
            timeframe=Timeframe.H1,
            source_event_ids=(high_event.event_id, low_event.event_id),
            evidence={
                "structure_id": f"structure-source:{label}",
                "source_high_id": f"high:{label}",
                "source_low_id": f"low:{label}",
                "candidate_protected_swing_id": f"low:{label}",
            },
            direction=Direction.LONG,
        )
        events = (bar, high_event, low_event, origin)
        adapter.consume_batch(events)
        exact_events.update({event.event_id: event for event in events})
        observer._confirmed_swing_event_ids.update(
            {
                f"high:{label}": high_event.event_id,
                f"low:{label}": low_event.event_id,
            }
        )
        high = _swing(
            f"high:{label}",
            SwingSide.HIGH,
            high_price,
            clock - pd.Timedelta(121, unit="m"),
            clock,
            timeframe=Timeframe.H1,
        )
        low = _swing(
            f"low:{label}",
            SwingSide.LOW,
            low_price,
            clock - pd.Timedelta(61, unit="m"),
            clock,
            timeframe=Timeframe.H1,
        )
        for swing, source in ((high, high_event), (low, low_event)):
            adapter.append_dto(
                SwingGeometryNode(
                    swing_id=swing.swing_id,
                    timeframe=Timeframe.H1,
                    symbol=swing.symbol,
                    instrument_id=swing.instrument_id,
                    window_start=swing.pivot_start,
                    window_end=clock,
                    lower_bound=swing.price - 0.5,
                    upper_bound=swing.price + 0.5,
                    known_at=clock,
                    source_candle_ids=(f"candle:{swing.swing_id}",),
                ),
                source_event_ids=(source.event_id,),
            )
        return SimpleNamespace(swings=(low, high)), events

    revisions = {}
    equal_frame, equal_events = publish_structure(
        180,
        "equal",
        low_price=100.0,
        high_price=100.0,
    )
    ranges, terminated = observer._foundation_update_structural_ranges(
        adapter=adapter,
        ranges={},
        frames={Timeframe.H1: equal_frame},
        known_at=_adapter_clock(180),
        clock_events=equal_events,
        revisions=revisions,
    )
    assert ranges == {}
    assert terminated == ()
    assert len(
        tuple(
            record
            for record in adapter.projection.latest_records
            if record.object_type
            is FoundationObjectType.SWING_GEOMETRY_NODE
        )
    ) == 2
    assert not any(
        record.object_type is FoundationObjectType.STRUCTURAL_RANGE
        for record in adapter.materialize_foundation_history()
    )

    # The immutable equal-tick origin remains explicitly range-ineligible on
    # later clocks; it must not trip the missed-confirmation guard.
    ranges, terminated = observer._foundation_update_structural_ranges(
        adapter=adapter,
        ranges=ranges,
        frames={Timeframe.H1: equal_frame},
        known_at=_adapter_clock(181),
        clock_events=(),
        revisions=revisions,
    )
    assert ranges == {}
    assert terminated == ()
    balance = _balance_range()
    balance_only = dual_range_location(
        100.0,
        structural_range=None,
        balance_range=balance,
    )
    assert balance_only.structural_range_id is None
    assert balance_only.balance_range_id == balance.range_id
    assert balance_only.x_balance_range is not None

    reset = _adapter_atomic(
        "reset-after-equal-range",
        EventKind.MARKET_EPOCH_RESET,
        182,
        0,
        evidence={"reason": "semantic_reset"},
    )
    adapter.consume(reset)
    ranges, _ = observer._foundation_update_structural_ranges(
        adapter=adapter,
        ranges=ranges,
        frames={Timeframe.H1: equal_frame},
        known_at=_adapter_clock(182),
        clock_events=(reset,),
        revisions=revisions,
    )

    valid_frame, valid_events = publish_structure(
        183,
        "valid-after-equal",
        low_price=90.0,
        high_price=110.0,
    )
    ranges, terminated = observer._foundation_update_structural_ranges(
        adapter=adapter,
        ranges=ranges,
        frames={Timeframe.H1: valid_frame},
        known_at=_adapter_clock(183),
        clock_events=valid_events,
        revisions=revisions,
    )
    active_range = ranges[Timeframe.H1]
    assert active_range.terminated_at is None
    assert (active_range.lower_bound, active_range.upper_bound) == (90.0, 110.0)
    assert terminated == ()
    coexisting = dual_range_location(
        100.0,
        structural_range=active_range,
        balance_range=balance,
    )
    assert coexisting.structural_range_id == active_range.range_id
    assert coexisting.balance_range_id == balance.range_id

    final_reset = _adapter_atomic(
        "reset-after-valid-range",
        EventKind.MARKET_EPOCH_RESET,
        184,
        0,
        evidence={"reason": "semantic_reset"},
    )
    before_reset = (adapter.projection, adapter.lifecycle)
    with pytest.raises(ValueError, match="StructuralRange owner"):
        adapter.consume(final_reset)
    assert (adapter.projection, adapter.lifecycle) == before_reset
    candidate, _ = adapter.stage_batch((final_reset,))
    ranges, terminated = observer._foundation_update_structural_ranges(
        adapter=candidate,
        ranges=ranges,
        frames={Timeframe.H1: valid_frame},
        known_at=_adapter_clock(184),
        clock_events=(final_reset,),
        revisions=revisions,
    )
    candidate.seal_staged_candidate()
    candidate.commit_staged_candidate()
    adapter = candidate
    terminal_range = ranges[Timeframe.H1]
    assert terminal_range.termination_reason == "semantic_reset"
    assert terminated == ((active_range.range_id, final_reset.event_id),)
    after_terminal = dual_range_location(
        100.0,
        structural_range=terminal_range,
        balance_range=balance,
    )
    assert after_terminal.structural_range_id is None
    assert after_terminal.balance_range_id == balance.range_id

    inverted_frame, inverted_events = publish_structure(
        185,
        "inverted-after-equal",
        low_price=110.0,
        high_price=100.0,
    )
    with pytest.raises(ValueError, match="structural range source Swings are invalid"):
        observer._foundation_update_structural_ranges(
            adapter=adapter,
            ranges=ranges,
            frames={Timeframe.H1: inverted_frame},
            known_at=_adapter_clock(185),
            clock_events=inverted_events,
            revisions=revisions,
        )


def test_structural_range_location_and_delivery_use_independent_native_clocks() -> None:
    reader, observer = _eye()
    observation = _observe(observer, reader, _m1_structure_bars())
    projection = observation.market_snapshot.foundation
    location = observation.market_snapshot.foundation_range_locations[Timeframe.M1]
    ranges = tuple(
        record
        for record in projection.latest_records
        if record.object_type is FoundationObjectType.STRUCTURAL_RANGE
        and record.payload["timeframe"] == Timeframe.M1.value
    )
    assert len(ranges) == 1
    assert location.structural_range_id == ranges[0].object_id
    assert location.x_structural_range is not None
    assert location.balance_range_id is None
    assert location.x_balance_range is None
    deliveries = tuple(
        record
        for record in observer.materialize_foundation_history()
        if record.object_type is FoundationObjectType.DELIVERY_PHASE_GENERATION
        and record.payload["timeframe"] == Timeframe.M1.value
    )
    latest = deliveries[-1]
    assert latest.payload["observation_count"] == 2
    assert latest.payload["duration_bars"] == 1
    assert len({record.object_id for record in deliveries}) == 1
    _assert_atomic_replay_parity(observer, observation)


@pytest.mark.parametrize(
    ("object_type", "field", "mutate", "message"),
    (
        (
            FoundationObjectType.STRUCTURAL_RANGE,
            "lower_bound",
            lambda value: value + 0.25,
            "StructuralRange boundaries conflict",
        ),
        (
            FoundationObjectType.DELIVERY_PHASE_GENERATION,
            "entered_real_bar_ordinal",
            lambda value: value - 1,
            "Delivery native ordinal or duration conflicts",
        ),
        (
            FoundationObjectType.DELIVERY_PHASE_GENERATION,
            "duration_bars",
            lambda value: value + 1,
            "Delivery native ordinal or duration conflicts",
        ),
        (
            FoundationObjectType.DELIVERY_PHASE_GENERATION,
            "current_price_ticks",
            lambda value: value + 1,
            "Delivery current price conflicts",
        ),
        (
            FoundationObjectType.BOUNDARY_ATTACK,
            "attempt_ordinal",
            lambda value: value + 1,
            "Boundary Attack identity or ordinal conflicts",
        ),
    ),
)
def test_atomic_replay_rebinds_foundation_range_delivery_and_attack_payloads(
    object_type: FoundationObjectType,
    field: str,
    mutate,
    message: str,
) -> None:
    reader, observer = _eye()
    observation = _observe(observer, reader, _m1_structure_bars())
    events = _legacy_foundation_transport_stream(observer)
    records = tuple(
        record
        for record in observation.market_snapshot.foundation.latest_records
        if record.object_type is object_type
    )
    assert len(records) == 1
    record = records[0]
    payload = dict(record.payload)
    payload[field] = mutate(payload[field])
    forged = FoundationRecord(
        object_type=record.object_type,
        object_id=record.object_id,
        status=record.status,
        known_at=record.known_at,
        payload=payload,
        source_event_ids=record.source_event_ids,
    )
    original = next(
        event
        for event in events
        if event.kind is EventKind.FOUNDATION_STATE_CHANGED
        and foundation_record_from_projection_event(event).record_id
        == record.record_id
    )
    replacement = foundation_record_projection_event(
        forged,
        timeframe=original.timeframe,
        published_at=original.known_at,
        sequence_no=original.sequence_no,
    )
    tampered = tuple(
        sorted(
            (
                replacement if event.event_id == original.event_id else event
                for event in events
            ),
            key=event_order_key,
        )
    )
    with pytest.raises(ValueError, match=message):
        replay_atomic_market_snapshot(
            tampered,
            semantic_registry_identity=observer.semantic_registry.identity,
        )


def test_boundary_attack_cannot_borrow_an_ordinal_into_a_new_generation() -> None:
    reader, observer = _eye()
    observation = _observe(observer, reader, _m5_parent_m1_child_mss_bars())
    events = _legacy_foundation_transport_stream(observer)
    attacks = tuple(
        record
        for record in observation.market_snapshot.foundation.latest_records
        if record.object_type is FoundationObjectType.BOUNDARY_ATTACK
    )
    second = next(record for record in attacks if record.payload["attempt_ordinal"] == 2)
    payload = dict(second.payload)
    payload["bos_generation_id"] = "forged-boundary-generation"
    forged_id = canonical_semantic_id(
        "boundary-attack",
        payload["timeframe"],
        payload["bos_generation_id"],
        payload["bar_event_id"],
        payload["direction"],
        payload["boundary_ticks"],
    )
    payload["boundary_attack_id"] = forged_id
    forged = FoundationRecord(
        object_type=second.object_type,
        object_id=forged_id,
        status=second.status,
        known_at=second.known_at,
        payload=payload,
        source_event_ids=second.source_event_ids,
    )
    original = next(
        event
        for event in events
        if event.kind is EventKind.FOUNDATION_STATE_CHANGED
        and foundation_record_from_projection_event(event).record_id
        == second.record_id
    )
    replacement = foundation_record_projection_event(
        forged,
        timeframe=original.timeframe,
        published_at=original.known_at,
        sequence_no=original.sequence_no,
    )
    with pytest.raises(ValueError, match="Boundary Attack identity or ordinal conflicts"):
        replay_atomic_market_snapshot(
            tuple(
                sorted(
                    (
                        replacement
                        if event.event_id == original.event_id
                        else event
                        for event in events
                    ),
                    key=event_order_key,
                )
            ),
            semantic_registry_identity=observer.semantic_registry.identity,
        )


def test_relation_keeps_external_owners_while_child_internal_mss_is_evidence() -> None:
    reader, observer = _eye()
    observation = _observe(observer, reader, _m5_parent_m1_child_mss_bars())
    snapshot = observation.market_snapshot
    relation = snapshot.relations["5m__1m"]
    assert relation.parent_direction is Direction.LONG
    assert relation.child_direction is Direction.SHORT
    assert relation.child_mss_against_parent
    child_external = tuple(
        generation
        for generation in observer._foundation_adapter.lifecycle.structure_generations
        if generation.timeframe is Timeframe.M1
        and generation.scope.value == "external"
        and generation.lifecycle is StructureGenerationLifecycle.CONFIRMED
    )
    assert len(child_external) == 1
    assert child_external[0].direction is Direction.LONG
    relation_generations = tuple(
        generation
        for generation in observer._foundation_adapter.lifecycle.relation_generations
        if generation.parent_tf is Timeframe.M5
        and generation.child_tf is Timeframe.M1
    )
    assert sum(
        generation.lifecycle is GenerationLifecycle.ACTIVE
        for generation in relation_generations
    ) == 1
    assert relation_generations[-1].child_structure_generation_id == child_external[0].generation_id
    assert relation_generations[-1].role in {
        "parent_retracement",
        "reversal_attempt",
    }
    starts = Counter(generation.role for generation in relation_generations)
    assert max(starts.values()) == 1
    real_bar_ordinals = {
        item.timeframe: item.count
        for item in observer._foundation_adapter.lifecycle.real_bar_clocks
    }
    candidates = {
        candidate.candidate_id: candidate
        for state in snapshot.timeframe_states.values()
        for candidate in state.liquidity.candidates
    }
    rearmed = tuple(
        level
        for level in observer._foundation_adapter.lifecycle.levels
        if level.lifecycle is LiquidityLevelLifecycle.REARMED
    )
    assert rearmed
    assert any(
        observer._foundation_dol_templates[level.source_identity].strength > 0.0
        for level in rearmed
    )
    for level in rearmed:
        candidate = candidates[level.level_id]
        interaction = observer._foundation_adapter.lifecycle.interaction(
            level.active_generation_id
        )
        template = observer._foundation_dol_templates[level.source_identity]
        assert candidate.rank == template.rank
        assert candidate.strength == template.strength
        assert candidate.age_bars == (
            real_bar_ordinals[interaction.interaction_timeframe]
            - interaction.armed_real_bar_ordinal
        )
    current_bar = next(
        event
        for event in reversed(observer.audit_store.events())
        if event.kind is EventKind.BAR_COMPLETED
        and event.timeframe is Timeframe.M1
        and event.known_at == snapshot.asof
    )
    contradictory = replace(
        relation,
        parent_direction=Direction.SHORT,
    )
    contradictory_snapshot = replace(
        snapshot,
        relations={**snapshot.relations, relation.relation_id: contradictory},
    )
    frozen_relations = observer._foundation_adapter.lifecycle.relation_generations
    observer._foundation_relation_delivery(
        adapter=observer._foundation_adapter,
        snapshot=contradictory_snapshot,
        current_bar_event_id=current_bar.event_id,
    )
    assert (
        observer._foundation_adapter.lifecycle.relation_generations
        == frozen_relations
    )
    _assert_atomic_replay_parity(observer, observation)


def test_cache_disappearance_alone_cannot_retire_a_foundation_level() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=0.25)
    _seed_level(adapter, label="capacity-invariant")
    current_bar = _adapter_bar(1)
    adapter.consume(current_bar)
    observer = CausalObserver(
        _all_typed_observer_config(
            materialize_event_view=False,
            eye_authority_mode=True,
        )
    )
    before = adapter.projection

    observer._foundation_reference_retirement(
        adapter=adapter,
        known_at=current_bar.known_at,
        clock_events=(current_bar,),
        retirement_events=(),
    )

    assert adapter.projection == before
    level = adapter.lifecycle.levels[0]
    assert level.lifecycle is LiquidityLevelLifecycle.ACTIVE


def test_clusters_follow_canonical_disarm_rearm_and_retirement_lifecycle() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=0.25)
    _, first_event = _seed_level(
        adapter,
        label="cluster-a",
        minute=0,
        price=100.0,
    )
    _seed_level(
        adapter,
        label="cluster-b",
        minute=1,
        price=100.25,
    )
    observer = CausalObserver(
        _all_typed_observer_config(
            materialize_event_view=False,
            eye_authority_mode=True,
        )
    )
    revisions: dict[tuple[object, ...], object] = {}

    first_clusters = observer._foundation_update_clusters(
        adapter=adapter,
        known_at=_adapter_clock(1),
        authoritative_events=(),
        revisions=revisions,
    )
    assert len(first_clusters) == 1
    first_cluster = first_clusters[0]
    levels = {level.source_identity: level for level in adapter.lifecycle.levels}
    assert first_cluster.member_level_ids == (
        levels["cluster-a"].level_id,
        levels["cluster-b"].level_id,
    )
    assert first_cluster.member_source_ids == ("cluster-a", "cluster-b")

    observer._foundation_active_clusters = first_clusters
    adapter_before = (adapter.projection, adapter.lifecycle)
    with pytest.raises(ValueError, match="active liquidity cluster"):
        adapter.retire_level(
            source_level_id="cluster-a",
            reason="source_retired",
            source_event_ids=(first_event.event_id,),
            known_at=_adapter_clock(1),
            timeframe=Timeframe.H1,
        )
    assert (adapter.projection, adapter.lifecycle) == adapter_before

    candidate, _ = adapter.stage_batch(())
    _cross_level(
        candidate,
        first_event,
        crossed_minute=2,
        resolved_minute=2,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
    )
    after_sweep = observer._foundation_update_clusters(
        adapter=candidate,
        known_at=_adapter_clock(2),
        authoritative_events=(),
        revisions=revisions,
    )
    candidate.seal_staged_candidate()
    candidate.commit_staged_candidate()
    adapter = candidate
    assert after_sweep == ()
    assert adapter.lifecycle.levels[0].lifecycle is LiquidityLevelLifecycle.DISARMED

    observer._foundation_active_clusters = after_sweep
    departure = _adapter_bar(3, high=100.0, low=99.0, close=99.75)
    adapter.consume(departure)
    rearmed_level = adapter.lifecycle.levels[0]
    assert rearmed_level.lifecycle is LiquidityLevelLifecycle.REARMED
    rearmed_interaction = adapter.lifecycle.interaction(
        rearmed_level.active_generation_id
    )
    after_rearm = observer._foundation_update_clusters(
        adapter=adapter,
        known_at=departure.known_at,
        authoritative_events=(),
        revisions=revisions,
    )
    assert len(after_rearm) == 1
    assert after_rearm[0].cluster_id != first_cluster.cluster_id
    assert after_rearm[0].started_at == rearmed_interaction.armed_at
    assert rearmed_level.level_id in after_rearm[0].member_level_ids

    observer._foundation_active_clusters = after_rearm
    retirement_bar = _adapter_bar(4)
    candidate, _ = adapter.stage_batch((retirement_bar,))
    candidate.retire_level(
        source_level_id="cluster-b",
        reason="source_retired",
        source_event_ids=(retirement_bar.event_id,),
        known_at=retirement_bar.known_at,
        timeframe=Timeframe.H1,
    )
    after_retirement = observer._foundation_update_clusters(
        adapter=candidate,
        known_at=retirement_bar.known_at,
        authoritative_events=(),
        revisions=revisions,
    )
    candidate.seal_staged_candidate()
    candidate.commit_staged_candidate()
    adapter = candidate
    assert after_retirement == ()
    assert levels["cluster-b"].level_id not in {
        member_id
        for cluster in after_retirement
        for member_id in cluster.member_level_ids
    }


def test_mature_range_invalidation_retires_both_boundary_targets() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=0.25)
    _, upper = _seed_level(
        adapter,
        label="range-upper",
        minute=0,
        source_kind="mature_range_boundary",
        source_timeframe=Timeframe.H1,
        side="above",
        price=100.0,
    )
    _seed_level(
        adapter,
        label="range-lower",
        minute=1,
        source_kind="mature_range_boundary",
        source_timeframe=Timeframe.H1,
        side="below",
        price=99.0,
    )
    _, acceptance = _cross_level(
        adapter,
        upper,
        crossed_minute=2,
        resolved_minute=3,
        terminal_kind=EventKind.ACCEPTANCE_CONFIRMED,
    )
    invalidation = _adapter_atomic(
        "range-invalidated:range-foundation",
        EventKind.DEALING_RANGE_INVALIDATED,
        3,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(acceptance.event_id,),
        evidence={"range_id": "range-foundation"},
    )
    adapter.consume(invalidation)
    observer = CausalObserver(
        _all_typed_observer_config(
            materialize_event_view=False,
            eye_authority_mode=True,
        )
    )
    observer._range_boundary_level_ids = {
        ("range-foundation", "above"): "range-upper",
        ("range-foundation", "below"): "range-lower",
    }

    observer._foundation_reference_retirement(
        adapter=adapter,
        known_at=invalidation.known_at,
        clock_events=(invalidation,),
        retirement_events=(),
    )

    levels = {
        level.source_identity: level for level in adapter.lifecycle.levels
    }
    assert levels["range-upper"].retirement_reason == "acceptance"
    assert (
        levels["range-lower"].retirement_reason
        == "source_range_terminated"
    )
    assert adapter.projection.active_dol_candidate_ids == ()
    assert observer._foundation_update_clusters(
        adapter=adapter,
        known_at=invalidation.known_at,
        authoritative_events=(invalidation,),
        revisions={},
    ) == ()
    cold_records = adapter.materialize_foundation_history()
    lower_history = tuple(
        record
        for record in cold_records
        if record.object_type is FoundationObjectType.LIQUIDITY_LEVEL
        and record.object_id == levels["range-lower"].level_id
    )
    assert lower_history[-1].status is FoundationRecordStatus.TERMINAL
    assert invalidation.event_id in lower_history[-1].source_event_ids
    assert (
        FoundationProjectionReducer.replay(cold_records)
        == adapter.projection
    )
