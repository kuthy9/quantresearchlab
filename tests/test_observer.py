from __future__ import annotations

from dataclasses import replace
import pickle

import pandas as pd
import pytest

from smc_trader.causal import CausalMarketReader
from smc_trader.model import (
    Bar,
    EventKind,
    FrameObservation,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityPoolLifecycle,
    LiquidityPoolState,
    ManipulationSourceDisposition,
    ManipulationSourceDispositionKind,
    SupportResistanceLifecycle,
    SupportResistanceState,
    Timeframe,
)
from smc_trader.observation import (
    CausalObserver,
    ExecutionRealityInput,
    ObserverConfig,
    _event,
)

from .helpers import MODEL_SCALE_SPECS, CORE_TEST_SCALE_SPECS, session_bars


STRUCTURE_PROTOCOL = "configs/primitives_structure_liquidity.json"
DISPLACEMENT_PROTOCOL = "configs/primitives_displacement.json"
GROUP3_PROTOCOL = "configs/primitives_zones.json"
GROUP4_PROTOCOL = "configs/primitives_range.json"
GROUP5_PROTOCOL = "configs/primitives_entry.json"


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
        "group3_protocol": GROUP3_PROTOCOL,
        "group4_protocol": GROUP4_PROTOCOL,
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


def test_lightweight_observer_can_skip_only_scene_graph_projection() -> None:
    reader = CausalMarketReader(scale_specs=CORE_TEST_SCALE_SPECS)
    update = reader.on_bar(session_bars(1)[0])
    observer = CausalObserver(
        ObserverConfig(
            scale_specs=CORE_TEST_SCALE_SPECS,
            project_scene_graph=False,
        )
    )

    observation = observer.observe(update)

    assert observation.scene_revision_id is None
    assert observer.last_scene_delta is None
    assert observer.scene_graph.last_asof is None


def test_authority_scan_projection_keeps_group4_state_identical() -> None:
    full_reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    light_reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    common = {
        "structure_protocol": STRUCTURE_PROTOCOL,
        "liquidity_protocol": STRUCTURE_PROTOCOL,
        "group4_protocol": GROUP4_PROTOCOL,
        "scale_specs": MODEL_SCALE_SPECS,
        "project_scene_graph": False,
    }
    full = CausalObserver(ObserverConfig(**common))
    light = CausalObserver(
        ObserverConfig(
            **common,
            materialize_event_view=False,
            group4_projection_only=True,
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
        )
        assert light.memory.__dict__ == full.memory.__dict__

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


def test_reference_zone_same_admission_prefix_is_causal_and_not_recent() -> None:
    observer = CausalObserver(_all_typed_observer_config())
    confirmed_at = pd.Timestamp(
        "2022-12-30 17:00",
        tz="America/New_York",
    )
    tested_at = pd.Timestamp(
        "2023-01-02 18:00",
        tz="America/New_York",
    )
    broken_at = tested_at + pd.Timedelta(minutes=1)
    zone = SupportResistanceState(
        zone_id="previous-session-high-composite",
        timeframe=Timeframe.M1,
        side="resistance",
        lower_bound=100.0,
        upper_bound=100.5,
        anchor_price=100.25,
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
        source_ids=("previous-session:2022-12-30:high",),
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


def test_eye_authority_mode_rejects_group4_projection_only() -> None:
    with pytest.raises(ValueError, match="Group 4 projection-only"):
        CausalObserver(
            _all_typed_observer_config(
                materialize_event_view=False,
                group4_projection_only=True,
                eye_authority_mode=True,
            )
        )


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
    observer = _typed_liquidity_observer()
    bars = _tick_aligned_bars(5)
    for bar in bars[:-1]:
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

    def formation_snapshot():
        zones, _, _ = original_snapshot()
        return zones, (formed,), (visible,)

    monkeypatch.setattr(m1_tracker, "snapshot", formation_snapshot)

    update = reader.on_bar(bars[-1])
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
