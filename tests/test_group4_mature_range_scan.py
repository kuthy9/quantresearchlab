from types import SimpleNamespace
import json

import pandas as pd
import pytest

from scripts.scan_mature_ranges import (
    CoverageAccumulator,
    DEFAULT_CONFIG,
    _calendar_warmup_start,
    _coverage_payload,
    _validate_config,
    geometrically_valid_source_pairs,
    select_stratified_cases,
    unmet_maturity_gates,
)
from smc_trader.model import (
    DealingRangeLifecycle,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    ManipulationLifecycle,
    SupportResistanceLifecycle,
    Timeframe,
)


START = pd.Timestamp("2020-06-01T18:00:00-04:00")
END = pd.Timestamp("2020-07-01T18:00:00-04:00")


def _protocol() -> SimpleNamespace:
    return SimpleNamespace(
        minimum_candidate_real_h1_bars=8,
        minimum_boundary_touches_each=2,
        minimum_midpoint_crossings=2,
        minimum_inside_close_fraction=0.8,
        maximum_width_atr_at_formation=4.0,
        maximum_compression_ratio=0.8,
    )


def _terminal(**changes: object) -> SimpleNamespace:
    values = {
        "range_id": "range-1",
        "lifecycle": DealingRangeLifecycle.BROKEN,
        "formed_at": START + pd.Timedelta(hours=1),
        "mature_at": None,
        "broken_at": START + pd.Timedelta(hours=10),
        "transition_reason": "close_beyond_frozen_range",
        "lower_bound": 99.0,
        "upper_bound": 101.0,
        "midpoint": 100.0,
        "lower_source_zone_id": "support-1",
        "upper_source_zone_id": "resistance-1",
        "candidate_real_h1_bars": 10,
        "lower_touch_count": 2,
        "upper_touch_count": 2,
        "midpoint_crossings": 2,
        "inside_close_fraction": 0.7,
        "width_atr_at_formation": 3.0,
        "compression_ratio": 1.0,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _inventory(
    identity: str,
    timeframe: Timeframe,
    *,
    side: str,
) -> LiquidityInventoryItem:
    price = 101.0 if side == "above" else 99.0
    kind = "equal_highs" if side == "above" else "equal_lows"
    return LiquidityInventoryItem(
        item_id=identity,
        timeframe=timeframe,
        side=side,
        kind=kind,
        price=price,
        lower_bound=price,
        upper_bound=price,
        formed_at=START - pd.Timedelta(hours=2),
        confirmed_at=START - pd.Timedelta(hours=1),
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=(f"pool-{identity}",),
        age_bars=1,
        strength=0.5,
    )


def _manipulation(
    identity: str,
    lifecycle: ManipulationLifecycle,
    *,
    censored_reason: str | None = None,
) -> SimpleNamespace:
    swept_at = START + pd.Timedelta(minutes=1)
    resolved_at = swept_at + pd.Timedelta(minutes=2)
    return SimpleNamespace(
        manipulation_id=identity,
        source_inventory_item_id=f"pool:{identity}",
        source_kind="formed_liquidity_pool",
        source_timeframe=Timeframe.M5,
        side="above",
        swept_at=swept_at,
        lifecycle=lifecycle,
        reaccepted_at=(
            resolved_at
            if lifecycle is ManipulationLifecycle.REACCEPTED
            else None
        ),
        accepted_outside_at=(
            resolved_at
            if lifecycle is ManipulationLifecycle.ACCEPTED_OUTSIDE
            else None
        ),
        censored_at=(resolved_at if censored_reason else None),
        deadline_elapsed=censored_reason == "deadline_elapsed",
        transition_reason=censored_reason,
    )


def test_terminal_reason_and_unmet_gates_are_separate_and_deduplicated() -> None:
    state = _terminal()
    accumulator = CoverageAccumulator("2020-june", START, END, _protocol())
    accumulator.observe_range(state)
    accumulator.observe_range(state)

    result = accumulator.result(source_rows=100, observed_updates=80)
    funnel = result["range_funnel"]
    assert funnel["broken"] == 1
    assert funnel["terminal_reason_counts"] == {
        "close_beyond_frozen_range": 1
    }
    assert unmet_maturity_gates(state, _protocol()) == (
        "inside_close_fraction",
        "compression",
    )
    assert funnel["forming_terminal_unmet_gate_counts"] == {
        "duration": 0,
        "bilateral_touches": 0,
        "midpoint_crossing": 0,
        "inside_close_fraction": 1,
        "width": 0,
        "compression": 1,
    }


def test_geometric_source_pairs_count_only_live_nonoverlapping_enclosures() -> None:
    def zone(
        identity: str,
        side: str,
        lower: float,
        upper: float,
        lifecycle: SupportResistanceLifecycle,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            zone_id=identity,
            side=side,
            lower_bound=lower,
            upper_bound=upper,
            lifecycle=lifecycle,
        )

    zones = (
        zone("s1", "support", 98.0, 99.0, SupportResistanceLifecycle.TESTED),
        zone("s2", "support", 99.5, 100.5, SupportResistanceLifecycle.BROKEN),
        zone("r1", "resistance", 101.0, 102.0, SupportResistanceLifecycle.ACTIVE),
        zone("r2", "resistance", 98.5, 99.5, SupportResistanceLifecycle.ACTIVE),
    )
    assert geometrically_valid_source_pairs(zones, 100.0) == (("s1", "r1"),)


def test_right_censoring_counts_only_live_ranges_formed_inside_window() -> None:
    accumulator = CoverageAccumulator("2020-june", START, END, _protocol())
    accumulator.latest_ranges = {
        "inside": SimpleNamespace(
            range_id="inside",
            lifecycle=DealingRangeLifecycle.FORMING,
            formed_at=START + pd.Timedelta(hours=1),
            mature_at=None,
        ),
        "warmup": SimpleNamespace(
            range_id="warmup",
            lifecycle=DealingRangeLifecycle.FORMING,
            formed_at=START - pd.Timedelta(hours=1),
            mature_at=None,
        ),
        "terminal": SimpleNamespace(
            range_id="terminal",
            lifecycle=DealingRangeLifecycle.BROKEN,
            formed_at=START + pd.Timedelta(hours=2),
            mature_at=None,
        ),
    }
    result = accumulator.result(source_rows=10, observed_updates=5)
    assert result["range_funnel"]["forming_right_censored"] == 1
    assert result["right_censored_range_ids"]["forming"] == ["inside"]


def test_stratified_selection_is_deterministic_and_allows_empty_strata() -> None:
    base = {
        "case_class": "forming_terminal",
        "entity_kind": "dealing_range",
        "focus_clock": START + pd.Timedelta(hours=12),
        "candidate_real_h1_bars": 12,
        "transition_reason": "maturity_deadline_elapsed",
        "total_gate_shortfall": 0.1,
    }
    records = (
        {**base, "entity_id": "b", "unmet_gates": ("compression",)},
        {**base, "entity_id": "a", "unmet_gates": ("compression",)},
    )
    first = select_stratified_cases(records)
    second = select_stratified_cases(tuple(reversed(records)))
    assert first == second
    assert first["near_mature_single_gate"] is not None
    assert first["mature_recognized"] is None
    assert first["natural_mature_range_manipulation"] is None


def test_registered_2023_profile_is_exact_and_outcome_blind() -> None:
    source = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    payload = _coverage_payload(
        source,
        DEFAULT_CONFIG,
        profile="group4_natural_authority_2023_full_year",
    )

    _validate_config(payload)
    assert payload["allowed_ohlcv_role"] == "brain_validation"
    assert payload["threshold_search"] is False
    assert payload["outcome_fields_used"] is False
    assert payload["pnl_used"] is False
    assert payload["mbo_used"] is False
    assert payload["include_all_pool_source_timeframes"] is True
    assert len(payload["windows"]) == 1
    assert payload["windows"][0] == {
        "id": "2023-full-year",
        "start": "2023-01-01T00:00:00-05:00",
        "end_exclusive": "2024-01-01T00:00:00-05:00",
    }

    with pytest.raises(ValueError, match="unknown registered"):
        _coverage_payload(
            source,
            DEFAULT_CONFIG,
            profile="arbitrary-calibration-window",
        )
    invalid = dict(payload)
    invalid["registered_calibration_exception"] = None
    with pytest.raises(ValueError, match="registered outcome-blind"):
        _validate_config(invalid)
    custom_config = dict(payload)
    custom_config["validation_protocol"] = "configs/custom_splits.json"
    with pytest.raises(ValueError, match="canonical registered profile"):
        _validate_config(custom_config)
    altered_window = dict(payload)
    altered_window["windows"] = [
        {
            "id": "not-2023",
            "start": "2022-06-01T00:00:00-04:00",
            "end_exclusive": "2022-07-01T00:00:00-04:00",
        }
    ]
    with pytest.raises(ValueError, match="differs from the canonical"):
        _validate_config(altered_window)


def test_aggregate_records_self_describing_scan_scope() -> None:
    profile = {
        "profile": "registered-authority-window",
        "timezone": "America/New_York",
        "warmup_calendar_days": 7,
        "pool_source_timeframes": ["4H", "1H", "15m", "5m", "1m"],
    }
    result = CoverageAccumulator(
        "window",
        START,
        END,
        _protocol(),
        coverage_start=START - pd.Timedelta(days=7),
    ).result(source_rows=10, observed_updates=5)

    from scripts.scan_mature_ranges import aggregate_results

    aggregate = aggregate_results((result,), profile=profile)
    context = aggregate["run_context"]
    assert context["timezone"] == "America/New_York"
    assert context["warmup_calendar_days"] == 7
    assert context["pool_source_timeframes"] == [
        "4H",
        "1H",
        "15m",
        "5m",
        "1m",
    ]
    assert context["executed_protocols"] == ["group12", "group4"]
    assert context["context_identity_only_protocols"] == [
        "displacement",
        "group3",
        "group5",
    ]
    assert context["event_view_materialized"] is False
    assert context["group4_projection_only"] is True
    assert context["complete_group4_eligible_inventory"] is True
    assert aggregate["windows"][0]["warmup_start"] == (
        START - pd.Timedelta(days=7)
    )


def test_calendar_warmup_preserves_wall_clock_across_dst() -> None:
    assert _calendar_warmup_start(
        pd.Timestamp("2023-11-05T18:00:00-05:00"),
        days=7,
        timezone="America/New_York",
    ) == pd.Timestamp("2023-10-29 18:00", tz="America/New_York")


def test_all_scale_source_inputs_and_unclassified_sources_are_counted() -> None:
    accumulator = CoverageAccumulator("all-scales", START, END, _protocol())
    items = tuple(
        _inventory(
            f"source-{timeframe.value}",
            timeframe,
            side="above" if index % 2 == 0 else "below",
        )
        for index, timeframe in enumerate(
            (
                Timeframe.H4,
                Timeframe.H1,
                Timeframe.M15,
                Timeframe.M5,
                Timeframe.M1,
            )
        )
    )
    candle = SimpleNamespace(
        start=START,
        high=102.0,
        low=98.0,
    )
    accumulator.observe_source_inputs(items, candle=candle)
    inventory = {item.item_id: item for item in items}
    accumulator.observe_unclassified_sources(
        (items[0].item_id,),
        metric="ambiguous_dual_side",
        inventory=inventory,
    )
    accumulator.observe_unclassified_sources(
        (items[-1].item_id,),
        metric="atr_unready",
        inventory=inventory,
    )

    result = accumulator.result(source_rows=5, observed_updates=1)
    funnel = result["manipulation_funnel"]
    assert funnel["visible_eligible_sources"] == 5
    assert funnel["crossed_sources"] == 5
    assert funnel["ambiguous_dual_side"] == 1
    assert funnel["atr_unready"] == 1
    assert funnel["swept_created"] == 0
    assert {
        row["source_timeframe"]
        for row in result["manipulation_by_source"]
    } == {"4H", "1H", "15m", "5m", "1m"}


def test_manipulation_created_cohort_has_exactly_one_outcome() -> None:
    accumulator = CoverageAccumulator("outcomes", START, END, _protocol())
    states = (
        _manipulation("reaccepted", ManipulationLifecycle.REACCEPTED),
        _manipulation(
            "accepted",
            ManipulationLifecycle.ACCEPTED_OUTSIDE,
        ),
        _manipulation(
            "deadline",
            ManipulationLifecycle.SWEPT,
            censored_reason="deadline_elapsed",
        ),
        _manipulation(
            "boundary",
            ManipulationLifecycle.SWEPT,
            censored_reason="data_gap_reset",
        ),
        _manipulation("right", ManipulationLifecycle.SWEPT),
    )
    for state in states:
        accumulator.observe_manipulation(state)

    result = accumulator.result(source_rows=10, observed_updates=10)
    funnel = result["manipulation_funnel"]
    assert funnel["swept_created"] == 5
    assert {
        name: funnel[name]
        for name in (
            "reaccepted",
            "accepted_outside",
            "deadline_censored",
            "hard_boundary_censored",
            "right_censored",
        )
    } == {
        "reaccepted": 1,
        "accepted_outside": 1,
        "deadline_censored": 1,
        "hard_boundary_censored": 1,
        "right_censored": 1,
    }
    assert result["manipulation_conservation"]["balanced"] is True


def test_unclassified_source_cannot_enter_created_cohort() -> None:
    accumulator = CoverageAccumulator("conflict", START, END, _protocol())
    item = _inventory(
        "pool:conflict",
        Timeframe.M5,
        side="above",
    )
    accumulator.observe_unclassified_sources(
        (item.item_id,),
        metric="atr_unready",
        inventory={item.item_id: item},
    )
    accumulator.observe_manipulation(
        _manipulation("conflict", ManipulationLifecycle.REACCEPTED)
    )

    with pytest.raises(
        AssertionError,
        match="ATR-unready source created",
    ):
        accumulator.result(source_rows=10, observed_updates=10)
