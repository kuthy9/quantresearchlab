from types import SimpleNamespace

import pandas as pd

from scripts.scan_mature_ranges import (
    CoverageAccumulator,
    geometrically_valid_source_pairs,
    select_stratified_cases,
    unmet_maturity_gates,
)
from smc_trader.model import (
    DealingRangeLifecycle,
    SupportResistanceLifecycle,
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
