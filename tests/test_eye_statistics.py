from __future__ import annotations

import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from smc_trader.eye_statistics import EyeAuthorityStatistics
from smc_trader.model import Timeframe


def _candle(end: pd.Timestamp, *, real: bool = True) -> SimpleNamespace:
    return SimpleNamespace(end=end, real_completed=real)


def _frame(
    asof: pd.Timestamp,
    *,
    timeframe: Timeframe = Timeframe.M1,
    ready: bool = True,
    ranges: tuple[object, ...] = (),
    fvgs: tuple[object, ...] = (),
    order_blocks: tuple[object, ...] = (),
    structure_breaks: tuple[object, ...] = (),
) -> SimpleNamespace:
    candle = SimpleNamespace(
        real_completed=True,
        body_class="normal",
        range_class="normal",
        dominant_wick="none",
        close_class="middle",
    )
    return SimpleNamespace(
        timeframe=timeframe,
        cutoff=asof,
        ready=ready,
        candle_structure=candle,
        swings=(),
        structures=(),
        structure_breaks=structure_breaks,
        support_resistance=(),
        liquidity_pools=(),
        fair_value_gaps=fvgs,
        order_blocks=order_blocks,
        dealing_ranges=ranges,
    )


def _observation(
    asof: pd.Timestamp,
    *,
    inventory: tuple[object, ...] = (),
    dispositions: tuple[object, ...] = (),
    displacement: object | None = None,
    ranges: tuple[object, ...] = (),
    manipulations: tuple[object, ...] = (),
    anomalies: tuple[str, ...] = (),
    execution_anomalies: tuple[str, ...] = (),
    ob_funnel: tuple[object, ...] = (),
    range_funnel: tuple[object, ...] = (),
    paths: tuple[object, ...] = (),
    fvgs: tuple[object, ...] = (),
    order_blocks: tuple[object, ...] = (),
    entry_locations: tuple[object, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        asof=asof,
        symbol="NQ",
        instrument_id=1,
        frames={
            Timeframe.M1: _frame(
                asof,
                ranges=ranges,
                fvgs=fvgs,
                order_blocks=order_blocks,
            )
        },
        anomalies=anomalies,
        execution=SimpleNamespace(anomalies=execution_anomalies),
        displacement=displacement,
        liquidity_inventory=inventory,
        liquidity_pool_states=(),
        group3_boundary_fvg_transitions=(),
        group3_boundary_order_block_transitions=(),
        group3_order_block_funnel=ob_funnel,
        manipulations=manipulations,
        group4_boundary_range_transitions=(),
        group4_boundary_manipulation_transitions=(),
        group4_ambiguous_sweep_item_ids=(),
        group4_atr_unready_sweep_item_ids=(),
        group4_source_dispositions=dispositions,
        group4_range_funnel=range_funnel,
        group5_typed_available=True,
        entry_locations=entry_locations,
        qualified_reacceptances=(),
        micro_bos_references=(),
        path_sequences=paths,
        group5_boundary_path_transitions=(),
        group5_boundary_reacceptance_transitions=(),
    )


def _update(
    asof: pd.Timestamp,
    *,
    anomalies: tuple[str, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        newly_completed={Timeframe.M1: (_candle(asof),)},
        anomalies=anomalies,
    )


def _inventory(item_id: str, formed_at: pd.Timestamp) -> SimpleNamespace:
    return SimpleNamespace(
        item_id=item_id,
        kind="equal_highs",
        timeframe=Timeframe.H1,
        side="above",
        structural_rank="external",
        lifecycle="visible",
        confirmed_at=formed_at,
        formed_at=formed_at,
        targeted_at=None,
        consumed_at=None,
        lifecycle_reason=None,
        source_ids=("pool-1",),
    )


def test_window_baseline_dispositions_and_execution_anomaly_filter() -> None:
    warmup = pd.Timestamp("2022-12-31 23:59", tz="UTC")
    start = pd.Timestamp("2023-01-01 00:00", tz="UTC")
    end = pd.Timestamp("2023-01-01 00:03", tz="UTC")
    source = _inventory("liq-1", warmup)
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=end,
        coverage_start=warmup,
    )

    assert statistics.observe(_update(warmup), _observation(warmup, inventory=(source,)))
    disposition = SimpleNamespace(
        source_inventory_item_id="liq-1",
        observed_at=start,
        disposition="selected_primary",
    )
    current = _observation(
        start,
        inventory=(source,),
        dispositions=(disposition,),
        anomalies=("spread_missing_used_one_tick", "market_gap"),
        execution_anomalies=("spread_missing_used_one_tick",),
    )
    current.group4_ambiguous_sweep_item_ids = ("ignored-fallback",)
    assert statistics.observe(_update(start), current)
    assert statistics.observe(_update(start), current) is False

    repeated_at = start + pd.Timedelta(minutes=1)
    repeated = SimpleNamespace(
        source_inventory_item_id="liq-1",
        observed_at=repeated_at,
        disposition="attached_same_side_secondary",
    )
    assert statistics.observe(
        _update(repeated_at),
        _observation(repeated_at, dispositions=(repeated,)),
    )

    summary = statistics.finalize()
    assert summary["window"]["observations"] == 2
    assert summary["data_clock"]["observation_anomalies"] == {"market_gap": 1}
    assert all(
        row["source_kind"] == "formed_liquidity_pool"
        and row["structural_rank"] == "external"
        and row["internal_external"] == "external"
        for row in summary["group4"]["source_dispositions"]
    )
    conservation = summary["group4"]["source_disposition_conservation"]
    assert conservation["balanced"]
    assert conservation["raw_crossed_source_ids"] == 2
    assert conservation["raw_crossed_source_clock_ids"] == 2
    assert conservation["raw_crossed_unique_source_ids"] == 1
    assert conservation["sum_across_ten_dispositions"] == 2
    assert sum(
        row["source_clock_decisions"]
        for row in summary["group4"]["source_dispositions"]
    ) == 2
    assert summary["group4"]["visible_eligible_sources"] == 1
    assert summary["group4"]["visible_eligible_sources_by_strata"] == [
        {
            "source_kind": "formed_liquidity_pool",
            "source_timeframe": "1H",
            "side": "above",
            "structural_rank": "external",
            "internal_external": "external",
            "unique_sources": 1,
        }
    ]
    assert not summary["group4"]["selected_primary_to_episode_join"]["balanced"]

    with pytest.raises(ValueError, match="strictly increasing"):
        statistics.observe(
            _update(warmup + pd.Timedelta(seconds=30)),
            _observation(warmup + pd.Timedelta(seconds=30)),
        )


def test_source_disposition_rejects_conflict_at_same_source_clock() -> None:
    start = pd.Timestamp("2023-01-03 14:00", tz="UTC")
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=start + pd.Timedelta(minutes=1),
        coverage_start=start,
    )
    conflicting = (
        SimpleNamespace(
            source_inventory_item_id="liq-conflict",
            observed_at=start,
            disposition="selected_primary",
        ),
        SimpleNamespace(
            source_inventory_item_id="liq-conflict",
            observed_at=start,
            disposition="blocked_existing_live",
        ),
    )
    with pytest.raises(ValueError, match="source-clock received conflicting"):
        statistics.observe(
            _update(start),
            _observation(start, dispositions=conflicting),
        )


def test_reader_is_clock_anomaly_authority_and_primitive_deadline_is_retained() -> None:
    start = pd.Timestamp("2023-01-04 14:00", tz="UTC")
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=start + pd.Timedelta(minutes=1),
        coverage_start=start,
    )
    statistics.observe(
        _update(start, anomalies=("data_gap_reset",)),
        _observation(
            start,
            anomalies=(
                "data_gap_reset",
                "data_gap_history_reset",
                "deadline_elapsed",
            ),
        ),
    )

    summary = statistics.finalize()
    assert summary["data_clock"]["reader_anomalies"] == {"data_gap_reset": 1}
    assert summary["data_clock"]["observation_anomalies"] == {
        "data_gap_history_reset": 1,
        "deadline_elapsed": 1,
    }
    assert summary["data_clock"]["integrity"]["clock_authority"] == "reader_update"
    assert summary["data_clock"]["integrity"]["data_gap_resets"] == 1
    assert summary["data_clock"]["integrity"]["reducer_exception_anomalies"] == 0
    assert all(
        row["exceptions"] == 0
        for row in summary["data_clock"]["reducer_updates"]
    )

    execution_statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=start + pd.Timedelta(minutes=1),
        coverage_start=start,
    )
    execution_statistics.observe(
        _update(start),
        _observation(
            start,
            anomalies=("deadline_elapsed",),
            execution_anomalies=("deadline_elapsed",),
        ),
    )
    assert execution_statistics.finalize()["data_clock"][
        "observation_anomalies"
    ] == {}


def test_numeric_summary_exposes_population_stddev_and_bounded_histogram() -> None:
    start = pd.Timestamp("2023-02-01 14:00", tz="UTC")
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=start + pd.Timedelta(minutes=1),
        coverage_start=start,
    )
    low = _manipulation("m-low", "swept", start, start)
    high = _manipulation("m-high", "swept", start, start)
    low.penetration_atr = 0.25
    high.penetration_atr = 0.75
    statistics.observe(
        _update(start),
        _observation(start, manipulations=(low, high)),
    )

    row = next(
        value
        for value in statistics.finalize()["numeric_distributions"]
        if value["group"] == "group4"
        and value["primitive"] == "manipulation"
        and value["metric"] == "penetration_atr"
    )
    assert row["mean"] == pytest.approx(0.5)
    assert row["standard_deviation"] == pytest.approx(0.25)
    assert row["distribution"]["kind"] == "fixed_signed_log2_histogram"
    assert sum(row["distribution"]["counts"].values()) == row["count"] == 2


def _range_state(
    range_id: str,
    asof: pd.Timestamp,
    lifecycle: str,
    reason: str | None = None,
    *,
    formed_at: pd.Timestamp | None = None,
    formation_atr: float = 1.0,
) -> SimpleNamespace:
    origin = asof if formed_at is None else formed_at
    return SimpleNamespace(
        range_id=range_id,
        lifecycle=lifecycle,
        timeframe=Timeframe.H1,
        formed_at=origin,
        mature_at=asof if lifecycle == "mature" else None,
        broken_at=None,
        state_started_at=asof,
        last_updated_at=asof,
        transition_reason=(
            reason
            if reason is not None
            else "maturity_conditions_met"
            if lifecycle == "mature"
            else None
        ),
        candidate_real_h1_bars=8,
        lower_touch_count=2,
        upper_touch_count=2,
        midpoint_crossings=2,
        inside_close_fraction=0.9,
        width_atr_at_formation=3.0,
        compression_ratio=0.7,
        formation_atr=formation_atr,
        lower_source_zone_id="sr-low",
        upper_source_zone_id="sr-high",
        lower_bound=99.0,
        upper_bound=103.0,
        midpoint=101.0,
    )


def _transition(
    entity: str,
    lifecycle: str,
    at: pd.Timestamp,
    ordinal: int,
    *,
    reason: str | None = None,
    started_at: pd.Timestamp | None = None,
    activation_ratios: dict[str, float] | None = None,
) -> SimpleNamespace:
    origin = at if started_at is None else started_at
    return SimpleNamespace(
        entity_id=entity,
        lifecycle=lifecycle,
        observed_at=at,
        direction="long",
        reason=reason,
        ordinal=ordinal,
        started_at=origin,
        active_at=origin if lifecycle != "started" else None,
        terminal_at=at if lifecycle in {"exhausted", "censored"} else None,
        state_metrics=(
            ("total_interruption_bars", 1.0),
            *(activation_ratios or {}).items(),
        ),
        admitted_candle_ids=("c1",),
    )


def _manipulation(
    entity: str,
    lifecycle: str,
    swept_at: pd.Timestamp,
    asof: pd.Timestamp,
) -> SimpleNamespace:
    return SimpleNamespace(
        manipulation_id=entity,
        lifecycle=lifecycle,
        timeframe=Timeframe.M1,
        side="above",
        source_kind="formed_liquidity_pool",
        source_timeframe=Timeframe.H1,
        swept_at=swept_at,
        state_started_at=asof,
        reaccepted_at=asof if lifecycle == "reaccepted" else None,
        accepted_outside_at=None,
        resolved_at=asof if lifecycle == "reaccepted" else None,
        censored_at=None,
        deadline_elapsed=False,
        transition_reason=("reaccepted_after_hold" if lifecycle == "reaccepted" else None),
        source_inventory_item_id="liq-1",
        source_id="pool-1",
        source_formed_at=swept_at - pd.Timedelta(hours=2),
        source_eligible_at=swept_at - pd.Timedelta(hours=1),
        penetration_atr=0.25,
        coincident_source_ids=("liq-2",),
        crossed_source_ids=("liq-1",),
    )


def _range_funnel(
    at: pd.Timestamp,
    *,
    range_id: str,
    unmet: tuple[str, ...],
) -> SimpleNamespace:
    names = (
        "duration",
        "bilateral_touches",
        "midpoint_crossing",
        "inside_close_fraction",
        "width",
        "compression",
    )
    return SimpleNamespace(
        observed_at=at,
        pair_counts=(
            ("live_structural_pairs", 1),
            ("invalid_geometry_pairs", 0),
            ("geometry_valid_pairs", 1),
            ("close_outside_pair_pairs", 0),
            ("already_admitted_pairs", 0),
            ("cold_start_blocked_pairs", 0),
            ("same_bar_terminal_blocked_pairs", 0),
            ("live_range_blocked_pairs", 0),
            ("atr_unready_pairs", 0),
            ("eligible_pairs", 1),
            ("forming_selected", 1),
        ),
        maturity_range_id=range_id,
        maturity_gates=tuple(
            (name, 1.0, 1.0, -0.1 if name in unmet else 0.1)
            for name in names
        ),
        unmet_maturity_gates=unmet,
    )


def _path(
    sequence_id: str,
    at: pd.Timestamp,
    *,
    reason: str,
    trigger: bool,
    context_id: str,
    lifecycle: str = "closed",
    formed_at: pd.Timestamp | None = None,
    context_kind: str = "pool_reversal",
) -> SimpleNamespace:
    origin = at if formed_at is None else formed_at
    first = SimpleNamespace(
        step_id=f"{sequence_id}:0",
        kind="first_pullback",
        observed_at=origin,
        source_event_id=None,
        source_entity_id=context_id,
        predecessor_step_ids=(),
        direction="long",
        same_clock_relation="origin",
        reason="first_pullback",
    )
    steps = [first]
    if trigger:
        steps.append(
            SimpleNamespace(
                step_id=f"{sequence_id}:1",
                kind="micro_bos_confirmed",
                observed_at=at,
                source_event_id=None,
                source_entity_id=context_id,
                predecessor_step_ids=(first.step_id,),
                direction="long",
                same_clock_relation="strictly_after",
                reason="aligned",
            )
        )
    return SimpleNamespace(
        sequence_id=sequence_id,
        lifecycle=lifecycle,
        formed_at=origin,
        state_started_at=at,
        last_updated_at=at,
        ended_at=at if lifecycle in {"closed", "censored"} else None,
        direction="long",
        context_kind=context_kind,
        context_id=context_id,
        transition_reason=reason,
        steps=tuple(steps),
    )


def _identity_path(
    sequence_id: str,
    *,
    context_kind: str,
    context_id: str,
    steps: tuple[tuple[str, pd.Timestamp, str | None, str], ...],
    reason: str,
) -> SimpleNamespace:
    values = []
    for index, (kind, observed_at, source_event_id, source_entity_id) in enumerate(
        steps
    ):
        values.append(
            SimpleNamespace(
                step_id=f"{sequence_id}:{index}",
                kind=kind,
                observed_at=observed_at,
                source_event_id=source_event_id,
                source_entity_id=source_entity_id,
                predecessor_step_ids=(
                    () if index == 0 else (f"{sequence_id}:{index - 1}",)
                ),
                direction="long",
                same_clock_relation="origin" if index == 0 else "strictly_after",
                reason=kind,
            )
        )
    return SimpleNamespace(
        sequence_id=sequence_id,
        lifecycle="closed",
        formed_at=values[0].observed_at,
        state_started_at=values[-1].observed_at,
        last_updated_at=values[-1].observed_at,
        ended_at=values[-1].observed_at,
        direction="long",
        context_kind=context_kind,
        context_id=context_id,
        transition_reason=reason,
        steps=tuple(values),
    )


def _fvg(
    zone_id: str,
    at: pd.Timestamp,
    *,
    qualification: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        fvg_id=zone_id,
        lifecycle="open",
        timeframe=Timeframe.M5,
        direction="long",
        qualification=qualification,
        formed_at=at,
        confirmed_at=at,
        state_started_at=at,
        transition_reason="strict_three_bar_geometry",
        source_displacement_id=(
            f"disp:{zone_id}"
            if qualification == "displacement_linked"
            else None
        ),
        source_displacement_active_at=(
            at if qualification == "displacement_linked" else None
        ),
        lower_bound=100.0,
        upper_bound=101.0,
    )


def _order_block(zone_id: str, at: pd.Timestamp) -> SimpleNamespace:
    return SimpleNamespace(
        order_block_id=zone_id,
        lifecycle="created",
        timeframe=Timeframe.M5,
        direction="long",
        source_bos_scope="continuation",
        source_bos_mss_qualified=False,
        formed_at=at,
        confirmed_at=at,
        state_started_at=at,
        transition_reason="qualified_displacement_bos_cluster",
        source_displacement_id=f"disp:{zone_id}",
        source_displacement_active_at=at,
        source_bos_id=f"bos:{zone_id}",
        lower_bound=100.0,
        upper_bound=101.0,
    )


def _entry_location(
    location_id: str,
    zone_id: str,
    zone_kind: str,
    at: pd.Timestamp,
) -> SimpleNamespace:
    return SimpleNamespace(
        location_id=location_id,
        lifecycle="approaching",
        direction="long",
        source_zone_kind=zone_kind,
        source_zone_id=zone_id,
        source_displacement_id=f"disp:{zone_id}",
        formed_at=at,
        state_started_at=at,
        transition_reason="source_registered",
        lower_bound=100.0,
        upper_bound=101.0,
    )


def test_range_manipulation_displacement_and_ob_funnel_conservation() -> None:
    start = pd.Timestamp("2023-04-03 14:00", tz="UTC")
    end = start + pd.Timedelta(minutes=4)
    protocol = SimpleNamespace(
        minimum_candidate_real_h1_bars=8,
        minimum_boundary_touches_each=2,
        minimum_midpoint_crossings=2,
        minimum_inside_close_fraction=0.8,
        maximum_width_atr_at_formation=4.0,
        maximum_compression_ratio=0.8,
    )
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=end,
        coverage_start=start,
        group4_protocol=protocol,
    )
    first = _transition("d1", "started", start, 0)
    ob1 = SimpleNamespace(
        observed_at=start,
        stages=(("active_displacement", 0), ("compatible_bos", 0),
                ("break_bar_belongs_to_displacement", 0),
                ("reverse_anchor_cluster_found", 0),
                ("unique_eligible_bos", 0), ("ob_created", 0)),
        outcome="no_active_displacement",
    )
    statistics.observe(
        _update(start),
        _observation(
            start,
            inventory=(_inventory("liq-1", start),),
            displacement=SimpleNamespace(transitions_this_update=(first,)),
            ranges=(_range_state("r1", start, "forming"),),
            manipulations=(_manipulation("m1", "swept", start, start),),
            ob_funnel=(ob1,),
            range_funnel=(
                _range_funnel(start, range_id="r1", unmet=("compression",)),
            ),
        ),
    )

    second_at = start + pd.Timedelta(minutes=1)
    transitions = (
        _transition("d1", "active", second_at, 0, started_at=start),
        _transition(
            "d1",
            "exhausted",
            second_at,
            1,
            reason="opposite_displacement",
            started_at=start,
        ),
        _transition("d2", "started", second_at, 2),
    )
    ob2 = SimpleNamespace(
        observed_at=second_at,
        stages=(("active_displacement", 1), ("compatible_bos", 1),
                ("break_bar_belongs_to_displacement", 1),
                ("reverse_anchor_cluster_found", 1),
                ("unique_eligible_bos", 1), ("ob_created", 1)),
        outcome="created",
    )
    statistics.observe(
        _update(second_at),
        _observation(
            second_at,
            displacement=SimpleNamespace(transitions_this_update=transitions),
            ranges=(_range_state("r1", second_at, "mature"),),
            manipulations=(_manipulation("m1", "reaccepted", start, second_at),),
            ob_funnel=(ob2,),
            range_funnel=(_range_funnel(second_at, range_id="r1", unmet=()),),
        ),
    )
    summary = statistics.finalize()

    assert summary["displacement"]["same_bar_terminal_then_restart"] == 1
    assert summary["displacement"]["duration_minutes"]["count"] == 1
    assert summary["group3"]["order_block_admission"]["attempts"] == 2
    assert summary["group3"]["order_block_admission"]["conservation"]["balanced"]
    assert summary["group4"]["range_gate_observations"] == 2
    assert summary["group4"]["unmet_gate_combinations"] == {
        "compression": 1,
        "none": 1,
    }
    assert summary["group4"]["range_formation_funnel"]["conservation"]["balanced"]
    assert summary["group4"]["mature_ranges_by_month"] == {"2023-04": 1}
    manipulation = summary["group4"]["manipulation_conservation"]
    assert manipulation["swept_created"] == 1
    assert manipulation["reaccepted"] == 1
    assert manipulation["conservation"]["balanced"]
    metrics = {
        row["metric"]: row
        for row in summary["numeric_distributions"]
        if row["group"] == "group4" and row["primitive"] == "manipulation"
    }
    assert metrics["penetration_atr"]["count"] == 1
    assert metrics["source_age_from_eligible_minutes"]["mean"] == 60.0
    assert metrics["source_age_from_formed_minutes"]["mean"] == 120.0
    assert metrics["coincident_source_count"]["mean"] == 1.0
    assert metrics["penetration_atr"]["strata"]["structural_rank"] == "external"


def test_case_index_is_bounded_deterministic_and_pickle_safe() -> None:
    start = pd.Timestamp("2023-05-01 00:00", tz="UTC")
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=start + pd.Timedelta(hours=1),
        coverage_start=start,
    )
    ranges = tuple(
        _range_state(f"r-{index:02d}", start, "mature")
        for index in range(45)
    )
    observation = _observation(start, ranges=ranges)
    statistics.observe(_update(start), observation)
    summary = statistics.finalize()

    assert summary["case_selection"]["selected"] == 40
    assert summary["case_selection"]["status"] == "complete"
    assert {case["stratum"] for case in summary["case_index"]} == {
        "all_recognized_mature"
    }
    assert all("pnl" not in case and "future" not in case for case in summary["case_index"])
    assert pickle.loads(pickle.dumps(statistics)).finalize() == summary


def test_profile_case_categories_and_group5_trigger_accounting() -> None:
    start = pd.Timestamp("2023-06-01 00:00", tz="UTC")
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=start + pd.Timedelta(minutes=3),
        coverage_start=start,
    )
    mature_manipulation = _manipulation("m-range", "swept", start, start)
    mature_manipulation.source_kind = "mature_range_boundary"
    complete = _path(
        "p-complete",
        start,
        reason="pool_reversal_sequence_observed",
        trigger=True,
        context_id="m-range",
    )
    interrupted = _path(
        "p-interrupted",
        start,
        reason="location_left",
        trigger=False,
        context_id="m-missing",
    )
    statistics.observe(
        _update(start),
        _observation(
            start,
            ranges=(
                _range_state("mature", start, "mature"),
                _range_state(
                    "broken-ok",
                    start,
                    "broken",
                    reason="maturity_deadline_elapsed",
                ),
                _range_state(
                    "broken-source",
                    start,
                    "broken",
                    reason="forming_source_invalidated",
                ),
            ),
            manipulations=(mature_manipulation,),
            paths=(complete, interrupted),
            range_funnel=(
                _range_funnel(start, range_id="near", unmet=("width",)),
            ),
        ),
    )
    next_at = start + pd.Timedelta(minutes=1)
    statistics.observe(
        _update(next_at),
        _observation(
            next_at,
            range_funnel=(
                _range_funnel(
                    next_at,
                    range_id="multiple",
                    unmet=("duration", "compression"),
                ),
            ),
        ),
    )
    summary = statistics.finalize()
    retained = summary["case_selection"]["retained_by_stratum"]
    expected_nonzero = {
        "all_recognized_mature",
        "near_mature_single_gate",
        "multiple_gate_rejected",
        "forming_reasonably_broken",
        "source_identity_or_reset_failure",
        "mature_range_manipulation",
        "group5_complete_path",
        "group5_interrupted_path",
    }
    assert {name for name, count in retained.items() if count} == expected_nonzero
    assert retained["obvious_mature_looking_but_rejected"] == 0
    assert (
        summary["case_selection"]["category_status"]
        ["obvious_mature_looking_but_rejected"]
        == "requires_blind_image_classification_not_inferred"
    )
    coverage = summary["case_selection"]["category_coverage"]
    assert set(coverage) == set(summary["case_selection"]["frozen_strata"])
    assert coverage["group5_complete_path"] == {
        "retained_cases": 1,
        "covered": True,
        "status": "covered",
    }
    assert coverage["obvious_mature_looking_but_rejected"] == {
        "retained_cases": 0,
        "covered": False,
        "status": "requires_blind_image_classification_not_inferred",
    }
    assert summary["group5"]["after_first_pullback"] == {
        "with_trigger": 1,
        "without_trigger": 1,
    }
    assert summary["group5"]["denominator_status"] == "producer_exposed"
    assert summary["group5"]["path_order_errors"] == 0
    assert summary["group5"]["terminal_reasons"] == {
        "location_left": 1,
        "pool_reversal_sequence_observed": 1,
    }
    assert summary["group5"]["terminal_classification"] == {
        "complete": 1,
        "interrupted": 1,
    }
    assert summary["group5"][
        "manipulations_entering_group5_by_source_timeframe"
    ] == {"1H": 1}


def test_group5_trigger_does_not_turn_unsuccessful_terminal_into_complete() -> None:
    start = pd.Timestamp("2023-06-02 00:00", tz="UTC")
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=start + pd.Timedelta(minutes=2),
        coverage_start=start,
    )
    failed_after_trigger = _path(
        "p-trigger-then-left",
        start,
        reason="location_left",
        trigger=True,
        context_id="m-1",
    )
    censored_after_trigger = _path(
        "p-trigger-then-censored",
        start + pd.Timedelta(minutes=1),
        reason="data_gap_reset",
        trigger=True,
        context_id="m-2",
        lifecycle="censored",
        formed_at=start,
    )
    statistics.observe(
        _update(start),
        _observation(start, paths=(failed_after_trigger,)),
    )
    statistics.observe(
        _update(start + pd.Timedelta(minutes=1)),
        _observation(start + pd.Timedelta(minutes=1), paths=(censored_after_trigger,)),
    )

    summary = statistics.finalize()
    retained = summary["case_selection"]["retained_by_stratum"]
    assert retained["group5_complete_path"] == 0
    assert retained["group5_interrupted_path"] == 2
    assert summary["group5"]["terminal_classification"] == {
        "complete": 0,
        "interrupted": 2,
    }


def test_group5_identity_funnels_preserve_zone_and_manipulation_sources() -> None:
    start = pd.Timestamp("2023-06-03 14:00", tz="UTC")
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=start + pd.Timedelta(minutes=1),
        coverage_start=start,
    )
    qualified_fvg = _fvg(
        "fvg-qualified",
        start,
        qualification="displacement_linked",
    )
    raw_fvg = _fvg("fvg-raw", start, qualification="raw")
    order_block = _order_block("ob-qualified", start)
    locations = (
        _entry_location("loc-fvg", "fvg-qualified", "fvg", start),
        _entry_location("loc-ob", "ob-qualified", "order_block", start),
    )
    paths = (
        _path(
            "path-fvg",
            start,
            reason="location_left",
            trigger=False,
            context_id="loc-fvg",
            context_kind="zone_return",
        ),
        _path(
            "path-ob",
            start,
            reason="micro_bos_aligned",
            trigger=True,
            context_id="loc-ob",
            context_kind="zone_return",
        ),
        _path(
            "path-pool",
            start,
            reason="pool_reversal_sequence_observed",
            trigger=True,
            context_id="manipulation-1",
        ),
    )
    source = _inventory("liq-1", start)
    disposition = SimpleNamespace(
        source_inventory_item_id="liq-1",
        observed_at=start,
        disposition="selected_primary",
    )
    statistics.observe(
        _update(start),
        _observation(
            start,
            inventory=(source,),
            dispositions=(disposition,),
            fvgs=(qualified_fvg, raw_fvg),
            order_blocks=(order_block,),
            entry_locations=locations,
            manipulations=(
                _manipulation("manipulation-1", "swept", start, start),
            ),
            paths=paths,
        ),
    )

    summary = statistics.finalize()
    zone_rows = {
        row["source_kind"]: row
        for row in summary["group5"][
            "qualified_zone_to_terminal_identity_funnel"
        ]
    }
    assert set(zone_rows) == {"fvg", "order_block"}
    assert zone_rows["fvg"]["identity_stage_counts"] == {
        "entry_location": 1,
        "path": 1,
        "path_terminal_all": 1,
        "qualified_zone": 1,
        "terminal_interrupted": 1,
    }
    assert zone_rows["order_block"]["identity_stage_counts"] == {
        "entry_location": 1,
        "path": 1,
        "path_terminal_all": 1,
        "qualified_zone": 1,
        "terminal_after_trigger": 1,
        "terminal_complete": 1,
        "trigger": 1,
    }
    assert all(
        row["conservation"]["terminal_balanced"]
        and row["conservation"]["funnel_monotone"]
        for row in zone_rows.values()
    )
    manipulation_row = summary["group5"][
        "manipulation_to_path_identity_funnel"
    ][0]
    assert manipulation_row["source_kind"] == "formed_liquidity_pool"
    assert manipulation_row["source_timeframe"] == "1H"
    assert manipulation_row["identity_stage_counts"] == {
        "manipulation": 1,
        "path": 1,
        "path_terminal_all": 1,
        "terminal_after_trigger": 1,
        "terminal_complete": 1,
        "trigger": 1,
    }
    assert manipulation_row["conservation"]["terminal_balanced"]
    identity = summary["group5"]["exact_identity_conservation"]
    assert identity["exact_identity_conserved"]
    assert identity["exact_identity_violation_count"] == 0
    assert identity["qualified_zone_chain"]["qualified_roots_by_kind"] == {
        "fvg": 1,
        "order_block": 1,
    }
    assert identity["qualified_zone_chain"][
        "qualified_zone_to_entry_location"
    ]["valid_parent_child_links"] == 2
    assert identity["qualified_zone_chain"][
        "entry_location_to_path_sequence"
    ]["valid_parent_child_links"] == 2
    assert identity["qualified_zone_chain"]["trigger_is_path_subset"][
        "is_subset"
    ]
    assert identity["qualified_zone_chain"]["terminal_is_path_subset"][
        "is_subset"
    ]
    assert identity["manipulation_pool_path_chain"][
        "manipulation_to_pool_path"
    ]["valid_parent_child_links"] == 1
    assert summary["group4"]["selected_primary_to_episode_join"] == {
        "identity": "(swept_at_iso8601, source_inventory_item_id)",
        "selected_primary_source_clocks": 1,
        "created_episode_source_clocks": 1,
        "swept_created": 1,
        "selected_without_episode": 0,
        "episode_without_selected": 0,
        "bad_ids": {
            "selected_without_episode": [],
            "episode_without_selected": [],
        },
        "balanced": True,
    }


def test_group5_exact_identity_conservation_reports_bounded_orphans() -> None:
    start = pd.Timestamp("2023-06-04 14:00", tz="UTC")
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=start + pd.Timedelta(minutes=1),
        coverage_start=start,
    )
    locations = tuple(
        _entry_location(
            f"orphan-location-{index}",
            f"missing-zone-{index}",
            "fvg",
            start,
        )
        for index in range(10)
    )
    statistics.observe(
        _update(start),
        _observation(start, entry_locations=locations),
    )

    conservation = statistics.finalize()["group5"][
        "exact_identity_conservation"
    ]
    link = conservation["qualified_zone_chain"][
        "qualified_zone_to_entry_location"
    ]
    assert not conservation["exact_identity_conserved"]
    assert link["orphan_children"] == 10
    assert link["exact_identity_violation_count"] == 10
    assert len(link["bad_ids"]["orphan_children"]) == 8


@pytest.mark.parametrize(
    ("trigger_kind", "break_at_trigger", "expected_complete"),
    (
        ("micro_bos_confirmed", False, 1),
        ("wick_rejection", False, 0),
        ("micro_bos_confirmed", True, 0),
    ),
)
def test_favr_observation_chain_uses_exact_source_clock_and_geometry_identity(
    trigger_kind: str,
    break_at_trigger: bool,
    expected_complete: int,
) -> None:
    start = pd.Timestamp("2023-06-04 15:00", tz="UTC")
    swept_at = start + pd.Timedelta(minutes=1)
    reaccepted_at = start + pd.Timedelta(minutes=2)
    zone_at = start + pd.Timedelta(minutes=3)
    trigger_at = start + pd.Timedelta(minutes=5)
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=trigger_at + pd.Timedelta(minutes=1),
        coverage_start=start,
    )
    manipulation = _manipulation(
        "favr-manipulation",
        "reaccepted",
        swept_at,
        reaccepted_at,
    )
    manipulation.source_kind = "mature_range_boundary"
    manipulation.source_id = "mature-range-1"
    manipulation.side = "below"
    zone = _fvg(
        "favr-zone",
        zone_at,
        qualification="displacement_linked",
    )
    location = _entry_location(
        "favr-location",
        "favr-zone",
        "fvg",
        zone_at,
    )
    zone_path = _identity_path(
        "favr-zone-path",
        context_kind="zone_return",
        context_id="favr-location",
        reason="micro_bos_aligned",
        steps=(
            ("zone_visible", zone_at, "favr-zone", "favr-zone"),
            (
                "departure_confirmed",
                zone_at,
                None,
                "favr-location",
            ),
            (
                "first_pullback",
                start + pd.Timedelta(minutes=4),
                None,
                "favr-location",
            ),
            (
                trigger_kind,
                trigger_at,
                None,
                "favr-location",
            ),
        ),
    )
    mature_range = _range_state("mature-range-1", start, "mature")
    statistics.observe(
        _update(start),
        _observation(start, ranges=(mature_range,)),
    )
    final_range = mature_range
    if break_at_trigger:
        final_range = _range_state(
            "mature-range-1",
            trigger_at,
            "broken",
            reason="close_beyond_frozen_range",
            formed_at=start,
        )
        final_range.mature_at = start
        final_range.broken_at = trigger_at
    opposed_mss = SimpleNamespace(
        bos_id="favr-return-mss",
        lifecycle="confirmed",
        direction="long",
        scope="opposed",
        mss_qualified=True,
        source_displacement_id="disp:favr-zone",
        resolved_at=zone_at,
        pending_at=start,
        post_break_state="pending",
        accepted_at=None,
        rejected_at=None,
        failure_reason=None,
    )
    observation = _observation(
        trigger_at,
        entry_locations=(location,),
        manipulations=(manipulation,),
        paths=(zone_path,),
    )
    observation.frames = {
        Timeframe.M1: _frame(trigger_at),
        Timeframe.M5: _frame(
            trigger_at,
            timeframe=Timeframe.M5,
            fvgs=(zone,),
            structure_breaks=(opposed_mss,),
        ),
        Timeframe.H1: _frame(
            trigger_at,
            timeframe=Timeframe.H1,
            ranges=(final_range,),
        ),
    }
    statistics.observe(
        _update(trigger_at),
        observation,
    )

    chain = statistics.finalize()["group5"]["favr_observation_chain"]
    assert chain["status"] == "authoritative_identity_join_exposed"
    expected_stages = {
        "entry_location": 1,
        "first_pullback": 1,
        "mature_range_identity": 1,
        "opposite_displacement": 1,
        "opposite_displacement_linked_zone": 1,
        "opposed_mss": 1,
        "reaccepted_mature_range_manipulation": 1,
        "zone_return_path": 1,
    }
    if expected_complete:
        expected_stages["trigger"] = 1
    assert chain["stage_counts_by_root_manipulation"] == expected_stages
    assert chain["complete_chains"] == expected_complete
    if not expected_complete:
        assert chain["attrition"] == (
            {"range_broken_before_or_at_trigger": 1}
            if break_at_trigger
            else {"trigger_missing_after_first_pullback": 1}
        )
    assert chain["identity_violations"] == {}
    assert chain["authoritative_join_not_exposed"] == {}


def test_warmup_entities_are_left_censored_from_cohorts_cases_and_path_triggers() -> None:
    start = pd.Timestamp("2023-06-05 14:00", tz="UTC")
    warmup = start - pd.Timedelta(minutes=1)
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=start + pd.Timedelta(minutes=1),
        coverage_start=warmup,
    )
    warm_range = _range_state(
        "warm-range",
        warmup,
        "forming",
        formed_at=warmup,
    )
    warm_path = _path(
        "warm-path",
        warmup,
        reason="context_registered",
        trigger=True,
        context_id="warm-manipulation",
        lifecycle="active",
        formed_at=warmup,
    )
    statistics.observe(
        _update(warmup),
        _observation(warmup, ranges=(warm_range,), paths=(warm_path,)),
    )

    terminal_range = _range_state(
        "warm-range",
        start,
        "broken",
        reason="maturity_deadline_elapsed",
        formed_at=warmup,
    )
    terminal_path = _path(
        "warm-path",
        start,
        reason="pool_reversal_sequence_observed",
        trigger=True,
        context_id="warm-manipulation",
        lifecycle="closed",
        formed_at=warmup,
    )
    statistics.observe(
        _update(start),
        _observation(
            start,
            ranges=(terminal_range,),
            paths=(terminal_path,),
            range_funnel=(
                _range_funnel(
                    start,
                    range_id="warm-range",
                    unmet=("compression",),
                ),
            ),
        ),
    )

    summary = statistics.finalize()
    assert summary["window"]["left_boundary_censored_total"] == 2
    assert summary["window"]["left_boundary_censored_entities"] == [
        {"group": "group4", "primitive": "dealing_range", "entities": 1},
        {"group": "group5", "primitive": "path_sequence", "entities": 1},
    ]
    assert summary["group4"]["range_gate_observations"] == 0
    assert summary["group4"]["left_boundary_range_gate_evaluations_excluded"] == 1
    assert summary["group4"]["unmet_gate_combinations"] == {}
    assert summary["group5"]["after_first_pullback"] == {
        "with_trigger": 0,
        "without_trigger": 0,
    }
    assert summary["group5"]["terminal_reasons"] == {}
    assert summary["group5"]["terminal_classification"] == {
        "complete": 0,
        "interrupted": 0,
    }
    assert not any(
        row["group"] == "group4" and row["primitive"] == "dealing_range"
        for row in summary["funnels"]
    )
    assert not any(
        case["entity_id"] in {"warm-range", "warm-path"}
        for case in summary["case_index"]
    )


def test_range_atr_tertiles_are_aggregate_only_and_case_sampling_is_episode_unique() -> None:
    start = pd.Timestamp("2023-06-06 14:00", tz="UTC")
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=start + pd.Timedelta(minutes=3),
        coverage_start=start,
    )
    ranges = tuple(
        _range_state(
            f"atr-range-{index}",
            start,
            "mature",
            formation_atr=float(index),
        )
        for index in range(1, 7)
    )
    duplicate_episode = _range_state(
        "duplicate-case-range",
        start,
        "mature",
        formation_atr=7.0,
    )
    statistics.observe(
        _update(start),
        _observation(
            start,
            ranges=(*ranges, duplicate_episode),
            range_funnel=(
                _range_funnel(
                    start,
                    range_id="near-repeat",
                    unmet=("width",),
                ),
            ),
        ),
    )
    second_at = start + pd.Timedelta(minutes=1)
    statistics.observe(
        _update(second_at),
        _observation(
            second_at,
            ranges=(
                _range_state(
                    "duplicate-case-range",
                    second_at,
                    "broken",
                    reason="maturity_deadline_elapsed",
                    formed_at=start,
                    formation_atr=7.0,
                ),
            ),
            range_funnel=(
                _range_funnel(
                    second_at,
                    range_id="near-repeat",
                    unmet=("compression",),
                ),
            ),
        ),
    )

    summary = statistics.finalize()
    tertiles = summary["group4"]["mature_range_formation_atr_tertiles"]
    assert tertiles["usage"] == (
        "outcome_blind_aggregate_only_not_a_reducer_gate_or_case_selector"
    )
    assert tertiles["cohort_count"] == 7
    assert sum(row["range_entities"] for row in tertiles["by_tertile"]) == 7
    assert sum(row["ever_mature"] for row in tertiles["by_tertile"]) == 7
    assert (
        summary["case_selection"]["retained_by_stratum"]
        ["near_mature_single_gate"]
        == 1
    )
    duplicate_selected = [
        case
        for case in summary["case_index"]
        if case["entity_id"] == "duplicate-case-range"
    ]
    assert len(duplicate_selected) == 1
    assert duplicate_selected[0]["stratum"] == "all_recognized_mature"
    assert summary["group4"]["range_formation_funnel"]["counting_basis"] == (
        "evaluation_weighted"
    )
    assert summary["group4"]["range_gate_counting_basis"] == (
        "evaluation_weighted"
    )
    diagnostics = summary["group4"]["unique_range_episode_diagnostics"]
    assert diagnostics["counting_basis"] == "unique_range_episode"
    assert diagnostics["first_evaluation"]["unique_range_episodes"] == 1
    assert diagnostics["latest_evaluation"]["unique_range_episodes"] == 1
    assert diagnostics["first_evaluation"]["unmet_gate_combinations"] == {
        "width": 1
    }
    assert diagnostics["latest_evaluation"]["unmet_gate_combinations"] == {
        "compression": 1
    }
    assert summary["case_selection"]["method"] == (
        "all_recognized_mature_first_smallest_sha256_then_"
        "registered_strata_round_robin_with_global_episode_uniqueness"
    )


def test_case_selection_prioritizes_mature_and_deduplicates_range_gate_episode() -> None:
    start = pd.Timestamp("2023-06-06 16:00", tz="UTC")
    mature_at = start + pd.Timedelta(minutes=1)
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=mature_at + pd.Timedelta(minutes=1),
        coverage_start=start,
    )
    statistics.observe(
        _update(start),
        _observation(
            start,
            range_funnel=(
                _range_funnel(
                    start,
                    range_id="eventually-mature",
                    unmet=("width",),
                ),
            ),
        ),
    )
    statistics.observe(
        _update(mature_at),
        _observation(
            mature_at,
            ranges=(
                _range_state(
                    "eventually-mature",
                    mature_at,
                    "mature",
                    formed_at=start,
                ),
            ),
            paths=(
                _path(
                    "other-interrupted-path",
                    mature_at,
                    reason="location_left",
                    trigger=False,
                    context_id="missing-manipulation",
                ),
            ),
        ),
    )

    cases = statistics.finalize()["case_index"]
    assert cases[0]["entity_id"] == "eventually-mature"
    assert cases[0]["stratum"] == "all_recognized_mature"
    assert sum(
        case["entity_id"] == "eventually-mature" for case in cases
    ) == 1


def test_displacement_never_active_diagnostics_use_final_visible_gate_ratios() -> None:
    start = pd.Timestamp("2023-06-07 14:00", tz="UTC")
    second_at = start + pd.Timedelta(minutes=1)
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=second_at + pd.Timedelta(minutes=1),
        coverage_start=start,
    )
    passing = {
        "activation_episode_bar_count_ratio": 1.2,
        "activation_relative_atr_ratio": 1.2,
        "activation_efficiency_ratio": 1.2,
        "activation_speed_ratio": 1.2,
        "activation_mean_body_fraction_ratio": 1.2,
        "activation_body_continuity_ratio": 1.2,
    }
    terminal_final = {
        **passing,
        "activation_efficiency_ratio": 0.5,
        "activation_speed_ratio": 0.8,
    }
    right_censored_final = {
        **passing,
        "activation_body_continuity_ratio": 0.5,
    }
    started = (
        _transition(
            "never-active-terminal",
            "started",
            start,
            0,
            activation_ratios=passing,
        ),
        _transition(
            "never-active-open",
            "started",
            start,
            1,
            activation_ratios=passing,
        ),
    )
    statistics.observe(
        _update(start),
        _observation(
            start,
            displacement=SimpleNamespace(
                transitions_this_update=started,
                current_entity_id="never-active-open",
                current_metrics=tuple(passing.items()),
            ),
        ),
    )
    terminal = _transition(
        "never-active-terminal",
        "exhausted",
        second_at,
        0,
        reason="lost_progression",
        started_at=start,
        activation_ratios=terminal_final,
    )
    statistics.observe(
        _update(second_at),
        _observation(
            second_at,
            displacement=SimpleNamespace(
                transitions_this_update=(terminal,),
                current_entity_id="never-active-open",
                current_metrics=tuple(right_censored_final.items()),
            ),
        ),
    )

    diagnostics = statistics.finalize()["displacement"][
        "started_never_active_diagnostics"
    ]["by_status"]
    terminal_row = diagnostics["terminal"]
    assert terminal_row["episodes"] == 1
    assert terminal_row["failed_gate_counts"] == {
        "activation_efficiency_ratio": 1,
        "activation_speed_ratio": 1,
    }
    assert terminal_row["failed_gate_combinations"] == {
        "activation_efficiency_ratio+activation_speed_ratio": 1
    }
    assert terminal_row["gate_margin_ratio_minus_one"][
        "activation_efficiency_ratio"
    ]["mean"] == pytest.approx(-0.5)
    open_row = diagnostics["right_censored"]
    assert open_row["episodes"] == 1
    assert open_row["failed_gate_counts"] == {
        "activation_body_continuity_ratio": 1
    }
    assert open_row["failed_gate_combinations"] == {
        "activation_body_continuity_ratio": 1
    }


def test_unchanged_higher_timeframe_snapshot_is_not_reprocessed() -> None:
    start = pd.Timestamp("2023-07-03 14:00", tz="UTC")
    warmup = start - pd.Timedelta(minutes=1)
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=start + pd.Timedelta(minutes=2),
        coverage_start=warmup,
    )
    h1_warm = _frame(warmup, ready=True)
    h1_warm.timeframe = Timeframe.H1
    warm = _observation(warmup)
    warm.frames = {
        Timeframe.M1: warm.frames[Timeframe.M1],
        Timeframe.H1: h1_warm,
    }
    statistics.observe(_update(warmup), warm)

    calls: list[str] = []
    original = statistics._observe_group12_frame

    def recording(frame: object, asof: pd.Timestamp) -> None:
        calls.append(frame.timeframe.value)
        original(frame, asof)

    statistics._observe_group12_frame = recording  # type: ignore[method-assign]
    h1 = _frame(start, ready=True)
    h1.timeframe = Timeframe.H1
    first = _observation(start)
    first.frames = {Timeframe.M1: first.frames[Timeframe.M1], Timeframe.H1: h1}
    statistics.observe(_update(start), first)
    second_at = start + pd.Timedelta(minutes=1)
    second = _observation(second_at)
    second.frames = {Timeframe.M1: second.frames[Timeframe.M1], Timeframe.H1: h1}
    statistics.observe(_update(second_at), second)

    # The new H1 cutoff is processed once, then the repeated snapshot is skipped.
    assert calls == ["1m", "1H", "1m"]
    h1_clock = next(
        row
        for row in statistics.finalize()["data_clock"]["by_timeframe"]
        if row["timeframe"] == "1H"
    )
    assert h1_clock["ready_at_window_start"] is True


def test_inventory_snapshot_is_consumed_once_and_reused_for_disposition() -> None:
    start = pd.Timestamp("2023-07-04 14:00", tz="UTC")
    source = _inventory("single-pass-liquidity", start)

    class SinglePassInventory:
        def __init__(self) -> None:
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("inventory snapshot was scanned twice")
            yield source

    inventory = SinglePassInventory()
    observation = _observation(
        start,
        dispositions=(
            SimpleNamespace(
                source_inventory_item_id=source.item_id,
                observed_at=start,
                disposition="selected_primary",
            ),
        ),
    )
    observation.liquidity_inventory = inventory
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=start + pd.Timedelta(minutes=1),
        coverage_start=start,
    )

    assert statistics.observe(_update(start), observation)
    summary = statistics.finalize()
    assert inventory.iterations == 1
    assert summary["group4"]["visible_eligible_sources"] == 1
    assert summary["group4"]["source_disposition_conservation"]["balanced"]


def test_group5_repeated_snapshot_processes_only_appended_steps_with_summary_parity(
) -> None:
    start = pd.Timestamp("2023-07-05 14:00", tz="UTC")
    statistics = EyeAuthorityStatistics(
        start=start,
        end_exclusive=start + pd.Timedelta(minutes=4),
        coverage_start=start,
    )
    path_step_calls = 0
    original_record_entity = statistics._record_entity

    def counting_record_entity(**kwargs):
        nonlocal path_step_calls
        if kwargs.get("primitive") == "path_step":
            path_step_calls += 1
        return original_record_entity(**kwargs)

    statistics._record_entity = counting_record_entity  # type: ignore[method-assign]

    first = _identity_path(
        "incremental-path",
        context_kind="zone_return",
        context_id="location-1",
        reason="context_registered",
        steps=(("first_pullback", start, None, "location-1"),),
    )
    first.lifecycle = "active"
    first.ended_at = None
    second_at = start + pd.Timedelta(minutes=1)
    second = _identity_path(
        "incremental-path",
        context_kind="zone_return",
        context_id="location-1",
        reason="zone_rejection_observed",
        steps=(
            ("first_pullback", start, None, "location-1"),
            ("micro_bos_confirmed", second_at, None, "location-1"),
        ),
    )
    second.lifecycle = "active"
    second.ended_at = None
    terminal_at = start + pd.Timedelta(minutes=2)
    terminal = _identity_path(
        "incremental-path",
        context_kind="zone_return",
        context_id="location-1",
        reason="micro_bos_aligned",
        steps=(
            ("first_pullback", start, None, "location-1"),
            ("micro_bos_confirmed", second_at, None, "location-1"),
        ),
    )
    terminal.state_started_at = terminal_at
    terminal.last_updated_at = terminal_at
    terminal.ended_at = terminal_at

    for at, path in (
        (start, first),
        (second_at, second),
        (terminal_at, terminal),
        (start + pd.Timedelta(minutes=3), terminal),
    ):
        statistics.observe(_update(at), _observation(at, paths=(path,)))

    summary = statistics.finalize()
    assert path_step_calls == 2
    assert statistics._path_steps_by_id["incremental-path"] == (
        ("incremental-path:0", "first_pullback", start),
        ("incremental-path:1", "micro_bos_confirmed", second_at),
    )
    assert summary["group5"]["after_first_pullback"] == {
        "with_trigger": 1,
        "without_trigger": 0,
    }
    assert summary["group5"]["path_order_errors"] == 0
    assert summary["group5"]["terminal_reasons"] == {"micro_bos_aligned": 1}
    assert summary["group5"]["terminal_classification"] == {
        "complete": 1,
        "interrupted": 0,
    }
