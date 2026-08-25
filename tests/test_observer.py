from __future__ import annotations

from dataclasses import replace
import json
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from smc_trader.causal import CausalMarketReader
from smc_trader.model import (
    Bar,
    Candle,
    Direction,
    EventKind,
    FrameObservation,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityPoolLifecycle,
    LiquidityPoolState,
    ManipulationSourceDisposition,
    ManipulationSourceDispositionKind,
    MarketObservation,
    PathSequenceLifecycle,
    PathSequenceState,
    PathSequenceStep,
    SupportResistanceLifecycle,
    SupportResistanceState,
    Timeframe,
    to_primitive,
)
from smc_trader.observation import (
    CausalObserver,
    EventMemory,
    ExecutionRealityInput,
    ObserverConfig,
    _event,
    _typed_native_transitions_or_baseline,
    _typed_state_delta_from_cache,
)

from .helpers import (
    MODEL_SCALE_SPECS,
    CORE_TEST_SCALE_SPECS,
    market_observation,
    session_bars,
)


STRUCTURE_PROTOCOL = "configs/primitives_structure_liquidity.json"
DISPLACEMENT_PROTOCOL = "configs/primitives_displacement.json"
GROUP3_PROTOCOL = "configs/primitives_zones.json"
GROUP4_PROTOCOL = "configs/primitives_range.json"
GROUP5_PROTOCOL = "configs/primitives_entry.json"


def test_market_observation_rejects_duck_typed_published_snapshot() -> None:
    observation = market_observation()
    fake_snapshot = SimpleNamespace(
        asof=observation.asof,
        symbol=observation.symbol,
        instrument_id=observation.instrument_id,
        price=observation.price,
        events_this_update=(),
    )

    with pytest.raises(TypeError, match="exact MarketSnapshot"):
        replace(
            observation,
            market_snapshot=fake_snapshot,
            _snapshot_free_identity=None,
        )

    state = observation.__getstate__()
    serialized = dict(state["fields"])
    serialized["market_snapshot"] = fake_snapshot
    serialized["_snapshot_free_identity"] = None
    forged_state = {
        "schema_version": state["schema_version"],
        "fields": tuple(
            (name, serialized[name]) for name, _ in state["fields"]
        ),
    }
    with pytest.raises(TypeError, match="exact MarketSnapshot"):
        object.__new__(MarketObservation).__setstate__(forged_state)

    forged_observation = object.__new__(MarketObservation)
    forged_observation.__dict__.update(observation.__dict__)
    object.__setattr__(forged_observation, "market_snapshot", fake_snapshot)
    object.__setattr__(forged_observation, "_snapshot_free_identity", None)
    with pytest.raises(TypeError, match="exact MarketSnapshot"):
        pickle.dumps(forged_observation, protocol=pickle.HIGHEST_PROTOCOL)


def test_snapshot_free_observation_primitive_hides_transport_identity() -> None:
    primitive = to_primitive(market_observation())

    assert {
        "asof",
        "symbol",
        "instrument_id",
        "price",
        "semantic_events_this_update",
        "_snapshot_free_identity",
    }.isdisjoint(primitive)
    assert primitive["market_snapshot"] is None
    json.dumps(primitive, sort_keys=True)


def _tick_aligned_bars(count: int) -> tuple[Bar, ...]:
    start = pd.Timestamp(
        "2025-01-05 18:00",
        tz="America/New_York",
    )
    return tuple(
        Bar(
            start=start + pd.Timedelta(minutes=index),
            open=(price := 20_000.0 + 0.25 * index),
            high=price + 0.50,
            low=price - 0.25,
            close=price + 0.25,
            volume=100 + index,
            symbol="NQH5",
            instrument_id=1,
        )
        for index in range(count)
    )


def _typed_liquidity_observer() -> CausalObserver:
    return CausalObserver(
        ObserverConfig(
            structure_protocol=STRUCTURE_PROTOCOL,
            liquidity_protocol=STRUCTURE_PROTOCOL,
            scale_specs=MODEL_SCALE_SPECS,
        )
    )


def _all_typed_observer_config(**overrides: object) -> ObserverConfig:
    values: dict[str, object] = {
        "structure_protocol": STRUCTURE_PROTOCOL,
        "liquidity_protocol": STRUCTURE_PROTOCOL,
        "displacement_protocol": DISPLACEMENT_PROTOCOL,
        "zone_protocol": GROUP3_PROTOCOL,
        "range_auction_protocol": GROUP4_PROTOCOL,
        "group5_protocol": GROUP5_PROTOCOL,
        "scale_specs": MODEL_SCALE_SPECS,
        "project_scene_graph": False,
    }
    values.update(overrides)
    return ObserverConfig(**values)


def test_observer_exposes_all_requested_descriptive_primitives() -> None:
    reader = CausalMarketReader(scale_specs=CORE_TEST_SCALE_SPECS)
    update = None
    for bar in session_bars(4, first_trade_date="2025-01-13"):
        update = reader.on_bar(bar)
    assert update is not None
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
    observation = observer.observe(
        update,
        ExecutionRealityInput(
            spread_points=0.25,
            deadline=update.asof + pd.Timedelta(minutes=60),
            size_available=10,
        ),
    )
    expected = {
        Timeframe.H4: {
            "directional_displacement",
            "path_efficiency",
            "structure_age_bars",
            "range_position",
            "external_above_distance_atr",
        },
        Timeframe.H1: {
            "swing_progression",
            "acceptance_direction",
            "rejection_direction",
            "rolling_range_position",
            "up_path_obstruction_atr",
        },
        Timeframe.M5: {
            "compression",
        },
        Timeframe.M1: {
            "acceleration",
            "counter_pressure",
        },
    }
    for timeframe, columns in expected.items():
        frame = observation.frame(timeframe)
        assert frame.ready
        assert columns <= set(frame.metrics)
        assert frame.cutoff <= observation.asof


def test_missing_deadline_remains_visible_to_risk_layer() -> None:
    reader = CausalMarketReader(scale_specs=CORE_TEST_SCALE_SPECS)
    update = reader.on_bar(session_bars(1)[0])
    observation = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    ).observe(update)
    assert "deadline_missing" in observation.anomalies


def test_direct_eye_defaults_to_no_scene_graph_projection() -> None:
    reader = CausalMarketReader(scale_specs=CORE_TEST_SCALE_SPECS)
    update = reader.on_bar(session_bars(1)[0])
    observer = CausalObserver(ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS))

    observation = observer.observe(update)

    assert observer.config.project_scene_graph is False
    assert observation.scene_revision_id is None
    assert observer.last_scene_delta is None
    assert observer.scene_graph.last_asof is None


def test_scene_delta_clone_matches_validated_replace_without_revalidating_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = CausalMarketReader(scale_specs=CORE_TEST_SCALE_SPECS)
    update = reader.on_bar(session_bars(1)[0])
    observation = CausalObserver(
        ObserverConfig(
            scale_specs=CORE_TEST_SCALE_SPECS,
            project_scene_graph=False,
        )
    ).observe(update)
    payload = {
        "scene_revision_id": "scene:r000000000001",
        "scene_added_node_ids": ("node:1",),
        "scene_revised_node_ids": ("node:2",),
        "scene_added_edge_ids": ("edge:1",),
        "scene_revised_edge_ids": ("edge:2",),
        "scene_resolution_event_ids": ("event:1",),
    }
    expected = replace(observation, **payload)

    def unexpected_validation(_self: MarketObservation) -> None:
        raise AssertionError("validated Eye payload was revalidated")

    monkeypatch.setattr(MarketObservation, "__post_init__", unexpected_validation)
    actual = observation._with_scene_delta(**payload)

    assert actual == expected
    assert actual is not observation
    assert actual.frames is observation.frames
    assert actual.recent_events is observation.recent_events
    assert (
        actual.retained_entity_timelines
        is observation.retained_entity_timelines
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (
            {"scene_revision_id": ""},
            "scene revision identity cannot be empty",
        ),
        (
            {
                "scene_revision_id": None,
                "scene_added_node_ids": ("node:1",),
            },
            "scene delta identities require a scene revision",
        ),
        (
            {
                "scene_revision_id": "scene:r000000000001",
                "scene_added_node_ids": ("node:1", "node:1"),
            },
            "scene delta identities are invalid",
        ),
    ),
)
def test_scene_delta_clone_validates_only_new_transport_identity(
    payload: dict[str, object],
    message: str,
) -> None:
    reader = CausalMarketReader(scale_specs=CORE_TEST_SCALE_SPECS)
    update = reader.on_bar(session_bars(1)[0])
    observation = CausalObserver(
        ObserverConfig(
            scale_specs=CORE_TEST_SCALE_SPECS,
            project_scene_graph=False,
        )
    ).observe(update)

    with pytest.raises(ValueError, match=message):
        observation._with_scene_delta(**payload)


def test_authority_scan_projection_keeps_group4_state_identical() -> None:
    full_reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    light_reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    common = {
        "structure_protocol": STRUCTURE_PROTOCOL,
        "liquidity_protocol": STRUCTURE_PROTOCOL,
        "range_auction_protocol": GROUP4_PROTOCOL,
        "scale_specs": MODEL_SCALE_SPECS,
        "project_scene_graph": False,
    }
    full = CausalObserver(ObserverConfig(**common))
    light = CausalObserver(
        ObserverConfig(
            **common,
            materialize_event_view=False,
            range_auction_projection_only=True,
        )
    )

    saw_range_funnel = False
    for bar in session_bars(1)[:600]:
        full_observation = full.observe(full_reader.on_bar(bar))
        light_observation = light.observe(light_reader.on_bar(bar))
        full_group4_inventory = tuple(
            item
            for item in full_observation.liquidity_inventory
            if item.kind in {"equal_highs", "equal_lows", "range_boundary"}
        )
        assert tuple(
            (
                item.item_id,
                item.timeframe,
                item.side,
                item.kind,
                item.price,
                item.lower_bound,
                item.upper_bound,
                item.formed_at,
                item.confirmed_at,
                item.lifecycle,
                item.source_ids,
                item.consumed_at,
                item.lifecycle_reason,
            )
            for item in light_observation.liquidity_inventory
        ) == tuple(
            (
                item.item_id,
                item.timeframe,
                item.side,
                item.kind,
                item.price,
                item.lower_bound,
                item.upper_bound,
                item.formed_at,
                item.confirmed_at,
                item.lifecycle,
                item.source_ids,
                item.consumed_at,
                item.lifecycle_reason,
            )
            for item in full_group4_inventory
        )
        assert tuple(
            (
                item.pool_id,
                item.timeframe,
                item.side,
                item.lower_bound,
                item.upper_bound,
                item.formed_at,
                item.confirmed_at,
                item.lifecycle,
                item.member_swing_ids,
                item.swept_at,
                item.resolved_at,
            )
            for item in light_observation.liquidity_pool_states
        ) == tuple(
            (
                item.pool_id,
                item.timeframe,
                item.side,
                item.lower_bound,
                item.upper_bound,
                item.formed_at,
                item.confirmed_at,
                item.lifecycle,
                item.member_swing_ids,
                item.swept_at,
                item.resolved_at,
            )
            for item in full_observation.liquidity_pool_states
        )
        assert light_observation.manipulations == (
            full_observation.manipulations
        )
        assert light_observation.group4_boundary_range_transitions == (
            full_observation.group4_boundary_range_transitions
        )
        assert (
            light_observation.group4_boundary_manipulation_transitions
            == full_observation.group4_boundary_manipulation_transitions
        )
        assert light_observation.group4_ambiguous_sweep_item_ids == (
            full_observation.group4_ambiguous_sweep_item_ids
        )
        assert light_observation.group4_atr_unready_sweep_item_ids == (
            full_observation.group4_atr_unready_sweep_item_ids
        )
        assert light_observation.group4_range_funnel == (
            full_observation.group4_range_funnel
        )
        saw_range_funnel = (
            saw_range_funnel
            or bool(light_observation.group4_range_funnel)
        )
        assert light_observation.frame(Timeframe.H1).dealing_ranges == (
            full_observation.frame(Timeframe.H1).dealing_ranges
        )
        assert tuple(
            (
                item.zone_id,
                item.source_kind,
                item.side,
                item.lifecycle,
                item.lower_bound,
                item.upper_bound,
                item.confirmed_at,
                item.total_touch_count,
                item.member_swing_ids,
                item.source_ids,
            )
            for item in light_observation.frame(
                Timeframe.H1
            ).support_resistance
        ) == tuple(
            (
                item.zone_id,
                item.source_kind,
                item.side,
                item.lifecycle,
                item.lower_bound,
                item.upper_bound,
                item.confirmed_at,
                item.total_touch_count,
                item.member_swing_ids,
                item.source_ids,
            )
            for item in full_observation.frame(
                Timeframe.H1
            ).support_resistance
        )

    assert saw_range_funnel
    assert light.memory.last_minute_end == full.memory.last_minute_end
    assert light.memory.clock_coverage_start == full.memory.clock_coverage_start
    assert light.memory.recent() == ()
    assert light_observation.recent_events == ()
    assert light_observation.event_durations_minutes == {}
    assert light_observation.event_ages_minutes == {}
    assert light_observation.retained_entity_timelines == {}
    assert light_observation.incomplete_entity_timeline_keys == ()
    assert light_observation.active_timeframes == tuple(
        spec.native_timeframe
        for spec in MODEL_SCALE_SPECS
        if spec.enabled and spec.native_timeframe is not None
    )


def test_eye_authority_mode_preserves_all_typed_state_and_internal_memory() -> None:
    full_reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    light_reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    full = CausalObserver(_all_typed_observer_config())
    light = CausalObserver(
        _all_typed_observer_config(
            materialize_event_view=False,
            eye_authority_mode=True,
        )
    )

    for bar in _tick_aligned_bars(120):
        full_observation = full.observe(full_reader.on_bar(bar))
        light_observation = light.observe(light_reader.on_bar(bar))

        assert not full_observation.typed_transition_delta_available
        non_foundation_events = tuple(
            event
            for event in light_observation.semantic_events_this_update
            if event.kind is not EventKind.FOUNDATION_STATE_CHANGED
        )
        assert non_foundation_events == (
            full_observation.semantic_events_this_update
        )
        assert replace(
            light_observation.market_snapshot,
            foundation=None,
            foundation_range_locations={},
            events_this_update=non_foundation_events,
        ) == full_observation.market_snapshot
        assert light_observation == replace(
            full_observation,
            execution=light_observation.execution,
            anomalies=tuple(
                value
                for value in full_observation.anomalies
                if value
                not in {
                    "spread_missing_used_one_tick",
                    "deadline_missing",
                }
            ),
            recent_events=(),
            event_durations_minutes={},
            event_ages_minutes={},
            retained_entity_timelines={},
                incomplete_entity_timeline_keys=(),
                market_snapshot=light_observation.market_snapshot,
                typed_transition_delta_available=True,
            liquidity_inventory_transitions_this_update=(
                light_observation
                .liquidity_inventory_transitions_this_update
            ),
            liquidity_pool_transitions_this_update=(
                light_observation.liquidity_pool_transitions_this_update
            ),
            group3_fvg_transitions_this_update=(
                light_observation.group3_fvg_transitions_this_update
            ),
            group3_order_block_transitions_this_update=(
                light_observation
                .group3_order_block_transitions_this_update
            ),
            group4_range_transitions_this_update=(
                light_observation.group4_range_transitions_this_update
            ),
            group4_manipulation_transitions_this_update=(
                light_observation
                .group4_manipulation_transitions_this_update
            ),
            group5_entry_location_transitions_this_update=(
                light_observation
                .group5_entry_location_transitions_this_update
            ),
            group5_reacceptance_transitions_this_update=(
                light_observation
                .group5_reacceptance_transitions_this_update
            ),
            group5_micro_bos_transitions_this_update=(
                light_observation
                .group5_micro_bos_transitions_this_update
            ),
            group5_path_transitions_this_update=(
                light_observation.group5_path_transitions_this_update
            ),
            group5_step_transitions_this_update=(
                light_observation.group5_step_transitions_this_update
            ),
        )
        technical_memory_fields = {
            "_audit_store",
            "_audit_pending",
            "_sequence_counts",
        }
        assert {
            key: value
            for key, value in light.memory.__dict__.items()
            if key not in technical_memory_fields
        } == {
            key: value
            for key, value in full.memory.__dict__.items()
            if key not in technical_memory_fields
        }
        assert tuple(
            event
            for event in light.audit_store.events()
            if event.kind is not EventKind.FOUNDATION_STATE_CHANGED
        ) == full.audit_store.events()

    assert light_observation.recent_events == ()
    assert light_observation.event_durations_minutes == {}
    assert light_observation.event_ages_minutes == {}
    assert light_observation.retained_entity_timelines == {}
    assert light_observation.incomplete_entity_timeline_keys == ()
    assert light_observation.execution.source == "not_evaluated"
    assert light_observation.execution.anomalies == ()
    assert light_observation.execution.expected_round_trip_cost_points == 0.0
    assert light.memory.recent()
    assert light.memory.entity_timelines()


def test_typed_delta_transport_preserves_full_observer_semantics() -> None:
    baseline_reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    transport_reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    baseline = CausalObserver(_all_typed_observer_config())
    transport = CausalObserver(
        _all_typed_observer_config(
            typed_transition_delta_transport=True,
        )
    )
    reality = ExecutionRealityInput(
        spread_points=0.25,
        deadline=pd.Timestamp(
            "2025-01-06 17:00",
            tz="America/New_York",
        ),
    )

    for bar in _tick_aligned_bars(40):
        expected = baseline.observe(
            baseline_reader.on_bar(bar),
            reality,
        )
        actual = transport.observe(
            transport_reader.on_bar(bar),
            reality,
        )
        assert actual.typed_transition_delta_available
        assert replace(
            actual,
            typed_transition_delta_available=False,
            liquidity_inventory_transitions_this_update=(),
            liquidity_pool_transitions_this_update=(),
            group3_fvg_transitions_this_update=(),
            group3_order_block_transitions_this_update=(),
            group4_range_transitions_this_update=(),
            group4_manipulation_transitions_this_update=(),
            group5_entry_location_transitions_this_update=(),
            group5_reacceptance_transitions_this_update=(),
            group5_micro_bos_transitions_this_update=(),
            group5_path_transitions_this_update=(),
            group5_step_transitions_this_update=(),
        ) == expected


def test_normal_observer_skips_typed_delta_signature_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import smc_trader.observation as observation_module

    calls = 0
    original = observation_module._typed_semantic_signature

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        observation_module,
        "_typed_semantic_signature",
        counted,
    )
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(_all_typed_observer_config())

    for bar in _tick_aligned_bars(20):
        observation = observer.observe(reader.on_bar(bar))

    assert calls == 0
    assert not observation.typed_transition_delta_available
    assert not any(
        (
            observation.liquidity_inventory_transitions_this_update,
            observation.liquidity_pool_transitions_this_update,
            observation.group3_fvg_transitions_this_update,
            observation.group3_order_block_transitions_this_update,
            observation.group4_range_transitions_this_update,
            observation.group4_manipulation_transitions_this_update,
            observation.group5_entry_location_transitions_this_update,
            observation.group5_reacceptance_transitions_this_update,
            observation.group5_micro_bos_transitions_this_update,
            observation.group5_path_transitions_this_update,
            observation.group5_step_transitions_this_update,
        )
    )


def test_observer_keeps_hidden_live_zone_timeline_until_terminal_cooling() -> None:
    """A transiently absent snapshot must not amputate future lifecycle input."""

    observers = (
        CausalObserver(_all_typed_observer_config()),
        CausalObserver(
            _all_typed_observer_config(
                materialize_event_view=False,
                eye_authority_mode=True,
            )
        ),
    )
    active_at = pd.Timestamp(
        "2023-01-02 17:58",
        tz="America/New_York",
    )
    broken_at = active_at + pd.Timedelta(minutes=3)
    reaccepted_at = broken_at + pd.Timedelta(minutes=1)
    key = "zone:hidden-live-zone"
    active = _event(
        EventKind.SUPPORT_RESISTANCE_STATE,
        active_at,
        Timeframe.M1,
        "below",
        100.0,
        0.5,
        entity_id="hidden-live-zone",
        lifecycle=SupportResistanceLifecycle.ACTIVE.value,
        formed_at=active_at,
        confirmed_at=active_at,
    )
    broken = _event(
        EventKind.SUPPORT_RESISTANCE_STATE,
        broken_at,
        Timeframe.M1,
        "below",
        100.0,
        0.5,
        entity_id="hidden-live-zone",
        lifecycle=SupportResistanceLifecycle.BROKEN.value,
        formed_at=active_at,
        confirmed_at=active_at,
        transition_reason="close_beyond_frozen_zone",
    )
    reaccepted = _event(
        EventKind.SUPPORT_RESISTANCE_STATE,
        reaccepted_at,
        Timeframe.M1,
        "below",
        100.0,
        0.5,
        entity_id="hidden-live-zone",
        lifecycle=SupportResistanceLifecycle.REACCEPTED.value,
        formed_at=active_at,
        confirmed_at=active_at,
        ended_at=reaccepted_at,
        transition_reason="close_reentered_frozen_zone",
    )

    for observer in observers:
        observer.memory.append(active)
        observer.memory.sync_retained_entity_timelines(
            {key},
            asof=active_at,
        )

        # The reducer may temporarily omit a still-live entity from its public
        # snapshot.  Observer retention must keep its causal prefix hot because
        # a later bar can still produce BROKEN/REACCEPTED.
        observer.memory.sync_retained_entity_timelines(
            observer._retained_timeline_keys({}, ()),
            asof=active_at + pd.Timedelta(minutes=1),
        )
        observer.memory.append(broken)
        observer.memory.sync_retained_entity_timelines(
            observer._retained_timeline_keys({}, ()),
            asof=broken_at,
        )

        assert observer.memory.incomplete_entity_keys() == ()
        assert tuple(
            event.lifecycle for event in observer.memory.timeline(key)
        ) == ("active", "broken")

        observer.memory.append(reaccepted)
        observer.memory.sync_retained_entity_timelines(
            {key},
            asof=reaccepted_at,
        )
        observer.memory.sync_retained_entity_timelines(
            observer._retained_timeline_keys({}, ()),
            asof=reaccepted_at + pd.Timedelta(minutes=1),
        )
        assert key not in observer.memory.entity_timelines()

    assert observers[0].memory.__dict__ == observers[1].memory.__dict__


def test_contract_boundary_carries_only_live_prefix_for_terminal_join() -> None:
    observers = (
        CausalObserver(_all_typed_observer_config()),
        CausalObserver(
            _all_typed_observer_config(
                materialize_event_view=False,
                eye_authority_mode=True,
            )
        ),
    )
    active_at = pd.Timestamp(
        "2023-01-02 17:59",
        tz="America/New_York",
    )
    boundary_at = active_at + pd.Timedelta(minutes=2)
    key = "zone:boundary-live-zone"
    active = _event(
        EventKind.SUPPORT_RESISTANCE_STATE,
        active_at,
        Timeframe.M1,
        "below",
        100.0,
        0.5,
        entity_id="boundary-live-zone",
        lifecycle=SupportResistanceLifecycle.ACTIVE.value,
        formed_at=active_at,
        confirmed_at=active_at,
    )
    broken = _event(
        EventKind.SUPPORT_RESISTANCE_STATE,
        boundary_at,
        Timeframe.M1,
        "below",
        100.0,
        0.5,
        entity_id="boundary-live-zone",
        lifecycle=SupportResistanceLifecycle.BROKEN.value,
        formed_at=active_at,
        confirmed_at=active_at,
        transition_reason="contract_change_reset",
    )

    for observer in observers:
        prior_memory = observer.memory
        prior_memory.append(active)
        prior_memory.sync_retained_entity_timelines(
            {key},
            asof=active_at,
        )
        observer._reset_contract_state(
            reason="contract_change_reset",
            observed_at=boundary_at,
            reset_anomalies=("contract_change_history_reset",),
            boundary_symbol="NQM3",
            boundary_instrument_id=2,
        )

        assert observer.memory is not prior_memory
        assert observer.memory.recent() == ()
        assert tuple(
            event.lifecycle for event in observer.memory.timeline(key)
        ) == ("active",)

        observer.memory.append(broken)
        observer.memory.sync_retained_entity_timelines(
            {key},
            asof=boundary_at,
        )
        assert observer.memory.incomplete_entity_keys() == ()
        assert tuple(
            event.lifecycle for event in observer.memory.timeline(key)
        ) == ("active", "broken")

        observer.memory.sync_retained_entity_timelines(
            observer._retained_timeline_keys({}, ()),
            asof=boundary_at + pd.Timedelta(minutes=1),
        )
        assert key not in observer.memory.entity_timelines()

    assert observers[0].memory.__dict__ == observers[1].memory.__dict__


def test_event_memory_sync_reuses_timeline_storage_and_survives_pickle() -> None:
    observer = CausalObserver(_all_typed_observer_config())
    formed_at = pd.Timestamp(
        "2023-01-03 10:00",
        tz="America/New_York",
    )
    event = _event(
        EventKind.SUPPORT_RESISTANCE_STATE,
        formed_at,
        Timeframe.M1,
        "below",
        100.0,
        0.5,
        entity_id="sync-storage-zone",
        lifecycle=SupportResistanceLifecycle.ACTIVE.value,
        formed_at=formed_at,
        confirmed_at=formed_at,
        transition_reason="confirmed_swing_cluster",
    )
    key = "zone:sync-storage-zone"
    observer.memory.append(event)
    observer.memory.sync_retained_entity_timelines(
        {key},
        asof=formed_at,
    )
    storage = observer.memory._entity_timelines[key]

    observer.memory.sync_retained_entity_timelines(
        {key},
        asof=formed_at + pd.Timedelta(minutes=1),
    )
    assert observer.memory._entity_timelines[key] is storage
    assert observer.memory.timeline(key) == (event,)

    resumed = pickle.loads(pickle.dumps(observer.memory))
    resumed_storage = resumed._entity_timelines[key]
    resumed.sync_retained_entity_timelines(
        {key},
        asof=formed_at + pd.Timedelta(minutes=2),
    )
    assert resumed._entity_timelines[key] is resumed_storage
    assert resumed.timeline(key) == observer.memory.timeline(key)

    with pytest.raises(
        ValueError,
        match="retained entity timeline contains the future",
    ):
        resumed.sync_retained_entity_timelines(
            {key},
            asof=formed_at - pd.Timedelta(minutes=1),
        )


def test_event_memory_sequence_clock_pruning_is_once_per_sync_and_resume_safe() -> None:
    """High retained history must not change same-clock event ordering.

    The reference applies the former pruning rule after every append.  The
    production memory defers the same retention result to the completed-bar
    synchronization boundary, where all same-clock events have already been
    assigned their sequence numbers.
    """

    reference = EventMemory(4)
    deferred = EventMemory(4)
    clock_origin = pd.Timestamp(
        "2023-01-03 09:30",
        tz="America/New_York",
    )

    def legacy_prune(memory: EventMemory) -> None:
        if len(memory._sequence_counts) <= memory._events.maxlen * 2:
            return
        clocks = {
            event.observed_at for event in memory._events
        } | {
            event.observed_at
            for timeline in memory._entity_timelines.values()
            for event in timeline
        }
        memory._sequence_counts = {
            clock: count
            for clock, count in memory._sequence_counts.items()
            if clock in clocks
        }

    retained: set[str] = set()
    for index in range(96):
        clock = clock_origin + pd.Timedelta(minutes=index)
        identity = f"retained-sequence-zone-{index}"
        event = _event(
            EventKind.SUPPORT_RESISTANCE_STATE,
            clock,
            Timeframe.M1,
            "below",
            100.0 + index,
            0.5,
            entity_id=identity,
            lifecycle=SupportResistanceLifecycle.ACTIVE.value,
            formed_at=clock,
            confirmed_at=clock,
        )
        reference.append(event)
        legacy_prune(reference)
        deferred.append(event)
        retained.add(f"zone:{identity}")

    same_clock = clock_origin + pd.Timedelta(minutes=100)
    for index in range(7):
        event = _event(
            EventKind.LIQUIDITY_CONSUMED,
            same_clock,
            Timeframe.M1,
            "above",
            200.0,
            0.25,
            source_ids=(f"same-clock-{index}",),
        )
        reference.append(event)
        legacy_prune(reference)
        deferred.append(event)

    reference.sync_retained_entity_timelines(
        retained,
        asof=same_clock,
    )
    deferred.sync_retained_entity_timelines(
        retained,
        asof=same_clock,
    )
    assert deferred.recent() == reference.recent()
    assert tuple(event.sequence_no for event in deferred.recent()) == (
        3,
        4,
        5,
        6,
    )
    assert deferred.entity_timelines() == reference.entity_timelines()
    assert deferred._sequence_counts == reference._sequence_counts
    assert deferred._sequence_counts_prune_pending is False

    # A pre-change checkpoint has no pending-bit attribute.  It must resume,
    # accept the next same-clock batch and converge to the same retained state.
    resumed = pickle.loads(pickle.dumps(deferred))
    resumed.__dict__.pop("_sequence_counts_prune_pending")
    resumed = pickle.loads(pickle.dumps(resumed))
    reference = pickle.loads(pickle.dumps(reference))
    resume_clock = same_clock + pd.Timedelta(minutes=1)
    for index in range(3):
        event = _event(
            EventKind.LIQUIDITY_CONSUMED,
            resume_clock,
            Timeframe.M1,
            "below",
            199.0,
            0.25,
            source_ids=(f"resume-clock-{index}",),
        )
        reference.append(event)
        legacy_prune(reference)
        resumed.append(event)
    reference.sync_retained_entity_timelines(
        retained,
        asof=resume_clock,
    )
    resumed.sync_retained_entity_timelines(
        retained,
        asof=resume_clock,
    )
    assert resumed.__dict__ == reference.__dict__


def test_temporal_metrics_cached_clock_matches_each_legacy_event_with_synthetic_minutes_and_pickle() -> None:
    def legacy_temporal_metrics(
        memory: EventMemory,
        asof: pd.Timestamp,
    ) -> tuple[dict[str, int], dict[str, int]]:
        durations: dict[str, int] = {}
        ages: dict[str, int] = {}
        active = {
            event.event_id
            for event in memory._latest_by_entity.values()
        }
        events = {
            event.event_id: event
            for event in memory._events
        }
        for event in events.values():
            if event.event_id in memory._closed_durations:
                durations[event.event_id] = memory._closed_durations[
                    event.event_id
                ]
            elif event.event_id in active:
                durations[event.event_id] = max(
                    0,
                    memory._elapsed_minutes(
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
                memory._elapsed_minutes(origin, asof),
            )
        for timeline in memory._entity_timelines.values():
            for index, event in enumerate(timeline):
                if index + 1 < len(timeline):
                    durations[event.event_id] = max(
                        0,
                        memory._elapsed_minutes(
                            event.observed_at,
                            timeline[index + 1].observed_at,
                        ),
                    )
                elif event.ended_at is not None:
                    durations[event.event_id] = 0
                else:
                    durations[event.event_id] = max(
                        0,
                        memory._elapsed_minutes(event.observed_at, asof),
                    )
                origin = (
                    event.formed_at
                    or event.confirmed_at
                    or event.observed_at
                )
                ages[event.event_id] = max(
                    0,
                    memory._elapsed_minutes(origin, asof),
                )
        return durations, ages

    start = pd.Timestamp(
        "2024-03-08 15:59:00.250000",
        tz="America/New_York",
    )
    memory = EventMemory(32)
    lifecycle_by_minute = {
        1: SupportResistanceLifecycle.ACTIVE,
        4: SupportResistanceLifecycle.TESTED,
        7: SupportResistanceLifecycle.BROKEN,
        9: SupportResistanceLifecycle.REACCEPTED,
    }
    retained_key = "zone:clock-cache-zone"
    retained_events = []

    for minute in range(1, 11):
        end = start + pd.Timedelta(minutes=minute)
        synthetic = minute in {2, 3, 6, 8}
        memory.observe_minute(
            Candle(
                timeframe=Timeframe.M1,
                start=end - pd.Timedelta(minutes=1),
                end=end,
                open=100.0,
                high=100.25,
                low=99.75,
                close=100.0,
                volume=0.0 if synthetic else 10.0,
                symbol="NQH4",
                instrument_id=1,
                observed_minutes=1,
                expected_minutes=1,
                complete=True,
                real_minutes=0 if synthetic else 1,
                synthetic_minutes=1 if synthetic else 0,
            )
        )
        lifecycle = lifecycle_by_minute.get(minute)
        if lifecycle is not None:
            terminal = lifecycle is SupportResistanceLifecycle.REACCEPTED
            event = _event(
                EventKind.SUPPORT_RESISTANCE_STATE,
                end,
                Timeframe.M1,
                "below",
                100.0,
                0.5,
                entity_id="clock-cache-zone",
                lifecycle=lifecycle.value,
                formed_at=start + pd.Timedelta(minutes=1),
                confirmed_at=start + pd.Timedelta(minutes=1),
                ended_at=end if terminal else None,
                transition_reason=f"test_{lifecycle.value}",
            )
            memory.append(event)
            retained_events.append(event)
        memory.append(
            _event(
                EventKind.LIQUIDITY_CONSUMED,
                end,
                Timeframe.M1,
                "below",
                99.75,
                0.25,
                source_ids=(f"minute-{minute}",),
            )
        )
        memory.sync_retained_entity_timelines(
            {retained_key},
            asof=end,
        )
        for query_asof in (
            end,
            end
            + pd.Timedelta(seconds=37)
            + pd.Timedelta(microseconds=125),
        ):
            expected = legacy_temporal_metrics(memory, query_asof)
            actual = memory.temporal_metrics(query_asof)
            assert actual == expected
            assert set(actual[0]) == set(actual[1])
            assert all(
                actual[index][event.event_id]
                == expected[index][event.event_id]
                for event in (*memory.recent(), *retained_events)
                for index in (0, 1)
            )
        if minute == 5:
            memory = pickle.loads(pickle.dumps(memory))

    # A scheduled market closure advances wall time without inserting
    # synthetic bars.  Preserve that existing distinction from explicit
    # synthetic no-trade minutes as well.
    after_weekend = pd.Timestamp(
        "2024-03-11 09:30:17.125",
        tz="America/New_York",
    )
    assert memory.temporal_metrics(after_weekend) == (
        legacy_temporal_metrics(memory, after_weekend)
    )


def test_reference_zone_same_admission_prefix_is_causal_and_not_recent() -> None:
    observer = CausalObserver(_all_typed_observer_config())
    source_period_end = pd.Timestamp(
        "2025-01-06 17:00",
        tz="America/New_York",
    )
    source_period_tail = Candle(
        timeframe=Timeframe.M1,
        start=source_period_end - pd.Timedelta(minutes=1),
        end=source_period_end,
        open=100.0,
        high=100.25,
        low=99.75,
        close=100.0,
        volume=100.0,
        symbol="NQH5",
        instrument_id=1,
        observed_minutes=1,
        expected_minutes=1,
        complete=True,
        real_minutes=1,
        synthetic_minutes=0,
    )
    reference_admission = replace(
        source_period_tail,
        start=pd.Timestamp(
            "2025-01-06 18:00",
            tz="America/New_York",
        ),
        end=pd.Timestamp(
            "2025-01-06 18:01",
            tz="America/New_York",
        ),
        high=100.0,
    )
    observer._reference_coverage_start = observer._reference_period_start(
        "session",
        source_period_tail,
    )
    for candle in (source_period_tail, reference_admission):
        observer._advance_reference_periods(
            candle,
            append_retirement_events=True,
        )
        observer._append_completed_bar_event(
            candle,
            atr=1.0,
            data_complete=True,
        )
    observer._publish_reference_candidate_events()
    reference_level = next(
        item
        for item in observer._reference_inventory.values()
        if item.kind == "previous_session_high"
    )
    reference_event = observer.memory.audit_event_including_pending(
        observer._candidate_level_event_ids[reference_level.item_id]
    )
    assert reference_event is not None
    assert reference_event.known_at == reference_admission.end

    confirmed_at = reference_admission.end + pd.Timedelta(minutes=1)
    tested_at = confirmed_at + pd.Timedelta(minutes=1)
    broken_at = tested_at + pd.Timedelta(minutes=1)
    for candle in (
        replace(
            source_period_tail,
            start=confirmed_at - pd.Timedelta(minutes=1),
            end=confirmed_at,
        ),
        replace(
            source_period_tail,
            start=tested_at - pd.Timedelta(minutes=1),
            end=tested_at,
            high=100.5,
            close=100.25,
        ),
        replace(
            source_period_tail,
            start=broken_at - pd.Timedelta(minutes=1),
            end=broken_at,
            open=100.25,
            high=101.0,
            low=100.0,
            close=100.75,
        ),
    ):
        observer._append_completed_bar_event(
            candle,
            atr=1.0,
            data_complete=True,
        )
    zone = SupportResistanceState(
        zone_id="previous-session-high-composite",
        timeframe=Timeframe.M1,
        side="resistance",
        lower_bound=reference_level.price - 0.25,
        upper_bound=reference_level.price + 0.25,
        anchor_price=reference_level.price,
        formed_at=confirmed_at,
        confirmed_at=confirmed_at,
        lifecycle=SupportResistanceLifecycle.BROKEN,
        member_swing_ids=(),
        touch_times=(confirmed_at, tested_at),
        reaction_magnitudes_atr=(0.0, 0.8),
        age_bars=2,
        strength=0.9,
        tested_at=tested_at,
        broken_at=broken_at,
        transition_reason="close_beyond_frozen_zone",
        total_touch_count=2,
        source_kind="previous_session",
        structural_rank="external",
        zone_role="both",
        visibility_strength=0.9,
        reaction_quality=0.8,
        freshness=0.7,
        depletion_risk=0.6,
        metadata_observed_at=broken_at,
        source_ids=reference_level.source_ids,
    )
    observer._record_frame_events(
        FrameObservation(
            timeframe=Timeframe.M1,
            cutoff=broken_at,
            bars=1,
            metrics={},
            support_resistance=(zone,),
        ),
        newly_completed=True,
        event_clock=broken_at,
    )
    key = f"zone:{zone.zone_id}"
    observer.memory.sync_retained_entity_timelines(
        {key},
        asof=broken_at,
    )

    timeline = observer.memory.timeline(key)
    assert tuple(event.lifecycle for event in timeline) == (
        "active",
        "tested",
        "broken",
    )
    assert tuple(event.observed_at for event in timeline) == (
        confirmed_at,
        tested_at,
        broken_at,
    )
    assert observer.memory.incomplete_entity_keys() == ()
    assert tuple(event.lifecycle for event in observer.memory.recent()) == (
        "broken",
    )
    for prefix in timeline[:-1]:
        assert prefix.strength == 0.0
        assert prefix.details["causal_prefix_recovered"] is True
        assert prefix.details["lower_bound"] == zone.lower_bound
        assert prefix.details["upper_bound"] == zone.upper_bound
        assert prefix.details["source_ids"] == zone.source_ids
        assert {
            "reaction_magnitudes_atr",
            "touch_times",
            "touch_count",
            "age_bars",
            "freshness",
            "depletion_risk",
        }.isdisjoint(prefix.details)


def test_structural_cold_zone_is_not_backfilled_as_reference_admission() -> None:
    observer = CausalObserver(_all_typed_observer_config())
    confirmed_at = pd.Timestamp(
        "2023-01-02 17:59",
        tz="America/New_York",
    )
    broken_at = confirmed_at + pd.Timedelta(minutes=2)
    zone = SupportResistanceState(
        zone_id="structural-cold-zone",
        timeframe=Timeframe.M1,
        side="resistance",
        lower_bound=100.0,
        upper_bound=100.5,
        anchor_price=100.25,
        formed_at=confirmed_at,
        confirmed_at=confirmed_at,
        lifecycle=SupportResistanceLifecycle.BROKEN,
        member_swing_ids=("swing-source",),
        touch_times=(confirmed_at,),
        reaction_magnitudes_atr=(0.5,),
        age_bars=2,
        strength=0.5,
        broken_at=broken_at,
        transition_reason="close_beyond_frozen_zone",
        total_touch_count=1,
        source_kind="structural_swing",
        metadata_observed_at=broken_at,
    )
    observer._record_frame_events(
        FrameObservation(
            timeframe=Timeframe.M1,
            cutoff=broken_at,
            bars=1,
            metrics={},
            support_resistance=(zone,),
        ),
        newly_completed=True,
        event_clock=broken_at,
    )
    key = f"zone:{zone.zone_id}"
    observer.memory.sync_retained_entity_timelines(
        {key},
        asof=broken_at,
    )

    assert tuple(
        event.lifecycle for event in observer.memory.timeline(key)
    ) == ("broken",)
    assert observer.memory.incomplete_entity_keys() == (key,)


def test_eye_authority_mode_keeps_incomplete_clock_anomaly_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        _all_typed_observer_config(
            materialize_event_view=False,
            eye_authority_mode=True,
        )
    )
    monkeypatch.setattr(
        observer.memory,
        "incomplete_entity_keys",
        lambda: ("swing:incomplete-test",),
    )

    observation = observer.observe(reader.on_bar(_tick_aligned_bars(1)[0]))

    assert "clock_incomplete_entity_timeline" in observation.anomalies
    assert observation.incomplete_entity_timeline_keys == ()
    assert observation.recent_events == ()
    assert observation.event_durations_minutes == {}
    assert observation.event_ages_minutes == {}
    assert observation.retained_entity_timelines == {}


def test_eye_authority_observer_pickle_resume_matches_uninterrupted() -> None:
    baseline_reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    resumed_reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    baseline = CausalObserver(
        _all_typed_observer_config(
            materialize_event_view=False,
            eye_authority_mode=True,
        )
    )
    resumed = CausalObserver(
        _all_typed_observer_config(
            materialize_event_view=False,
            eye_authority_mode=True,
        )
    )
    bars = _tick_aligned_bars(90)

    for bar in bars[:47]:
        baseline.observe(baseline_reader.on_bar(bar))
        resumed.observe(resumed_reader.on_bar(bar))
    resumed_reader, resumed = pickle.loads(
        pickle.dumps((resumed_reader, resumed))
    )

    for bar in bars[47:]:
        baseline_observation = baseline.observe(
            baseline_reader.on_bar(bar)
        )
        resumed_observation = resumed.observe(
            resumed_reader.on_bar(bar)
        )

    assert resumed_observation == baseline_observation
    assert resumed.memory.__dict__ == baseline.memory.__dict__
    assert resumed.last_scene_delta is None
    assert resumed.scene_graph.last_asof is None


def test_lightweight_event_view_is_rejected_outside_authority_scan() -> None:
    with pytest.raises(ValueError, match="authority scanner"):
        CausalObserver(
            ObserverConfig(
                scale_specs=MODEL_SCALE_SPECS,
                materialize_event_view=False,
            )
        )


def test_eye_authority_mode_rejects_range_auction_projection_only() -> None:
    with pytest.raises(ValueError, match="Group 4 projection-only"):
        CausalObserver(
            _all_typed_observer_config(
                materialize_event_view=False,
                range_auction_projection_only=True,
                eye_authority_mode=True,
            )
        )


@pytest.mark.parametrize(
    "config",
    (
        ObserverConfig(
            scale_specs=MODEL_SCALE_SPECS,
            canonical_foundation_enabled=True,
        ),
        _all_typed_observer_config(
            range_auction_projection_only=True,
            canonical_foundation_enabled=True,
        ),
    ),
)
def test_canonical_foundation_requires_fully_typed_atomic_observer(
    config: ObserverConfig,
) -> None:
    with pytest.raises(
        ValueError,
        match="canonical-foundation projection requires all typed protocols",
    ):
        CausalObserver(config)


def test_eye_authority_mode_rejects_graph_without_event_view() -> None:
    with pytest.raises(
        ValueError,
        match="Scene Graph projection requires a materialized EventMemory",
    ):
        CausalObserver(
            _all_typed_observer_config(
                project_scene_graph=True,
                materialize_event_view=False,
                eye_authority_mode=True,
            )
        )


def test_eye_authority_mode_rejects_execution_reality_input() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        _all_typed_observer_config(
            materialize_event_view=False,
            eye_authority_mode=True,
        )
    )

    with pytest.raises(ValueError, match="does not evaluate execution"):
        observer.observe(
            reader.on_bar(_tick_aligned_bars(1)[0]),
            ExecutionRealityInput(spread_points=0.25),
        )

    assert observer._prior is None
    assert observer.memory.last_minute_end is None


def test_eye_authority_mode_can_materialize_memory_and_scene_graph() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        _all_typed_observer_config(
            project_scene_graph=True,
            materialize_event_view=True,
            eye_authority_mode=True,
        )
    )

    observation = observer.observe(reader.on_bar(_tick_aligned_bars(1)[0]))

    assert observation.execution.source == "not_evaluated"
    assert observation.execution.anomalies == ()
    assert "spread_missing_used_one_tick" not in observation.anomalies
    assert "deadline_missing" not in observation.anomalies
    assert observation.recent_events == observer.memory.recent()
    assert (
        observation.retained_entity_timelines
        == observer.memory.entity_timelines()
    )
    assert observation.scene_revision_id == observer.scene_graph.revision_id


def test_group4_disposition_can_reference_prior_inventory_only() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(_all_typed_observer_config())
    observation = observer.observe(reader.on_bar(_tick_aligned_bars(1)[0]))
    disposition = ManipulationSourceDisposition(
        source_inventory_item_id="prior-inventory-only",
        observed_at=observation.asof,
        disposition=(
            ManipulationSourceDispositionKind.REJECTED_SOURCE_MISSING_OR_STALE
        ),
    )

    projected = replace(
        observation,
        group4_source_dispositions=(disposition,),
    )

    assert projected.group4_source_dispositions == (disposition,)
    assert "prior-inventory-only" not in {
        item.item_id for item in projected.liquidity_inventory
    }


def test_invalid_execution_reality_fails_before_observer_mutation() -> None:
    reader = CausalMarketReader(scale_specs=CORE_TEST_SCALE_SPECS)
    update = reader.on_bar(session_bars(1)[0])
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
    with pytest.raises(ValueError, match="spread cannot be negative"):
        observer.observe(
            update,
            ExecutionRealityInput(spread_points=-0.25),
        )
    assert observer._prior is None
    assert observer.memory.last_minute_end is None
    assert observer.memory.clock_coverage_start is None


def test_warmed_plain_minute_reuses_unchanged_liquidity_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = _typed_liquidity_observer()
    bars = _tick_aligned_bars(17)
    for bar in bars[:-1]:
        observer.observe(reader.on_bar(bar))

    calls = {timeframe: 0 for timeframe in observer._liquidity_trackers}
    for timeframe, tracker in observer._liquidity_trackers.items():
        original = tracker.snapshot

        def counted_snapshot(
            *,
            _timeframe: Timeframe = timeframe,
            _original=original,
        ):
            calls[_timeframe] += 1
            return _original()

        monkeypatch.setattr(tracker, "snapshot", counted_snapshot)

    update = reader.on_bar(bars[-1])
    assert {
        timeframe
        for timeframe, candles in update.newly_completed.items()
        if candles
    } == {Timeframe.M1}
    observer.observe(update)

    assert calls == {
        timeframe: int(timeframe is Timeframe.M1)
        for timeframe in observer._liquidity_trackers
    }


def test_observation_materializes_event_time_metrics_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = CausalMarketReader(scale_specs=CORE_TEST_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
    bars = session_bars(2)
    observer.observe(reader.on_bar(bars[0]))
    original = observer.memory.temporal_metrics
    calls = 0

    def counted(asof: pd.Timestamp):
        nonlocal calls
        calls += 1
        return original(asof)

    monkeypatch.setattr(observer.memory, "temporal_metrics", counted)
    observer.observe(reader.on_bar(bars[1]))

    assert calls == 1


def test_projected_pool_state_overrides_preprojection_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        _all_typed_observer_config(
            materialize_event_view=False,
            eye_authority_mode=True,
        )
    )
    bars = _tick_aligned_bars(6)
    for bar in bars[:-2]:
        prior = observer.observe(reader.on_bar(bar))

    formed_at = prior.asof - pd.Timedelta(minutes=2)
    confirmed_at = prior.asof - pd.Timedelta(minutes=1)
    formed = LiquidityPoolState(
        pool_id="cache-projection-pool",
        timeframe=Timeframe.M1,
        side="above",
        lower_bound=20_000.75,
        upper_bound=20_001.00,
        midpoint=20_000.875,
        formed_at=formed_at,
        confirmed_at=confirmed_at,
        lifecycle=LiquidityPoolLifecycle.FORMED,
        member_swing_ids=("cache-swing-a", "cache-swing-b"),
        touch_times=(formed_at, confirmed_at),
        age_bars=0,
        strength=0.5,
        total_touch_count=2,
    )
    visible = LiquidityInventoryItem(
        item_id=f"pool:{formed.pool_id}",
        timeframe=Timeframe.M1,
        side="above",
        kind="equal_highs",
        price=formed.upper_bound,
        lower_bound=formed.lower_bound,
        upper_bound=formed.upper_bound,
        formed_at=formed.formed_at,
        confirmed_at=formed.confirmed_at,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=formed.member_swing_ids,
        age_bars=0,
        strength=formed.strength,
    )
    m1_tracker = observer._liquidity_trackers[Timeframe.M1]
    original_snapshot = m1_tracker.snapshot

    # This test isolates post-projection cache behavior and injects a pool
    # directly, without its two canonical Swing parents.  Mark its candidate
    # as already recorded so the fixture does not manufacture an invalid
    # authoritative liquidity event; provenance is covered independently.
    observer._candidate_level_event_ids[visible.item_id] = (
        "projection-cache-existing-candidate"
    )

    def formation_snapshot():
        zones, _, _ = original_snapshot()
        return zones, (formed,), (visible,)

    monkeypatch.setattr(m1_tracker, "snapshot", formation_snapshot)

    update = reader.on_bar(bars[-2])
    swept = replace(
        formed,
        lifecycle=LiquidityPoolLifecycle.SWEPT,
        swept_at=update.asof,
        sweep_extreme=formed.upper_bound + 0.25,
        close_outside_on_sweep=False,
    )
    consumed = replace(
        visible,
        lifecycle=LiquidityInventoryLifecycle.CONSUMED,
        consumed_at=update.asof,
        lifecycle_reason="pool_swept",
    )

    def project_inventory(
        _update,
        _frames,
        _base_inventory,
        *,
        projected_pool_states,
    ):
        projected_pool_states[swept.pool_id] = swept
        return (consumed,)

    monkeypatch.setattr(observer, "_project_inventory", project_inventory)
    observation = observer.observe(update)

    assert observation.frame(Timeframe.M1).liquidity_pools == (formed,)
    assert observation.liquidity_pool_states == (swept,)
    assert observation.liquidity_inventory == (consumed,)
    assert observation.typed_transition_delta_available
    assert observation.liquidity_pool_transitions_this_update == (swept,)
    assert observation.liquidity_inventory_transitions_this_update == (
        consumed,
    )

    # Repeating the same lifecycle and frozen geometry on the next completed
    # minute is not another semantic transition.
    unchanged = observer.observe(reader.on_bar(bars[-1]))
    assert unchanged.liquidity_pool_states == (swept,)
    assert unchanged.liquidity_inventory == (consumed,)
    assert unchanged.liquidity_pool_transitions_this_update == ()
    assert unchanged.liquidity_inventory_transitions_this_update == ()


def test_group5_semantic_delta_keeps_steps_but_ignores_age_heartbeat() -> None:
    formed_at = pd.Timestamp(
        "2025-01-06 09:30",
        tz="America/New_York",
    )
    first_step = PathSequenceStep(
        step_id="step-zone-visible",
        kind="zone_visible",
        observed_at=formed_at,
        source_event_id="fvg-source",
        source_entity_id="fvg-source",
        predecessor_step_ids=(),
        same_clock_relation="origin",
        direction=Direction.LONG,
        strength=0.5,
        reason="typed_entry_zone_registered",
    )
    baseline = PathSequenceState(
        sequence_id="path-zone-return",
        protocol_hash="protocol",
        symbol="NQH5",
        instrument_id=1,
        context_kind="zone_return",
        context_id="location-1",
        direction=Direction.LONG,
        lifecycle=PathSequenceLifecycle.ACTIVE,
        formed_at=formed_at,
        state_started_at=formed_at,
        last_updated_at=formed_at,
        age_real_1m_bars=0,
        state_duration_real_1m_bars=0,
        steps=(first_step,),
    )
    signatures: dict[str, tuple[tuple[str, object], ...]] = {}
    assert _typed_state_delta_from_cache(
        candidates=(baseline,),
        signatures=signatures,
        identity_field="sequence_id",
    ) == (baseline,)

    next_at = formed_at + pd.Timedelta(minutes=1)
    heartbeat = replace(
        baseline,
        last_updated_at=next_at,
        age_real_1m_bars=1,
        state_duration_real_1m_bars=1,
    )
    assert _typed_state_delta_from_cache(
        candidates=(heartbeat,),
        signatures=signatures,
        identity_field="sequence_id",
    ) == ()

    trigger = PathSequenceStep(
        step_id="step-micro-bos",
        kind="micro_bos_confirmed",
        observed_at=next_at,
        source_event_id="bos-1",
        source_entity_id="bos-reference-1",
        predecessor_step_ids=(first_step.step_id,),
        same_clock_relation="strictly_after",
        direction=Direction.LONG,
        strength=0.75,
        reason="aligned_micro_bos",
    )
    appended = replace(
        heartbeat,
        steps=(first_step, trigger),
    )
    assert _typed_state_delta_from_cache(
        candidates=(appended,),
        signatures=signatures,
        identity_field="sequence_id",
    ) == (appended,)


def test_typed_cold_start_baseline_never_masks_boundary_terminals() -> None:
    assert _typed_native_transitions_or_baseline(
        current=("open",),
        transitions=(),
        first_observation=True,
        boundary_reason=None,
    ) == ("open",)
    assert _typed_native_transitions_or_baseline(
        current=(),
        transitions=("hard-boundary-terminal",),
        first_observation=True,
        boundary_reason="data_gap_reset",
    ) == ("hard-boundary-terminal",)
