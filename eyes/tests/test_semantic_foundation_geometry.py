from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from eyes.core.foundation_registry import FOUNDATION_VERSION
from eyes.core.market_state import (
    BalanceRangeState,
    StructuralRangeState,
    SwingGeometryNode,
    SwingHierarchyView,
    SwingRankAssignment,
    build_structural_legs,
    build_structural_range,
    build_swing_geometry_nodes,
    terminate_structural_range,
    update_swing_geometry_assignments,
    _require_contiguous_native_candles,
)
from contract.market import (
    BarCoverage,
    Candle,
    Direction,
    Timeframe,
    bar_evidence_coverage,
    candle_coverage,
    candle_identity,
)
from contract.eye import (
    DealingRangeLifecycle,
    DealingRangeState,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    SwingLifecycle,
    SwingPoint,
    SwingRank,
    SwingRelation,
    SwingSide,
)


TZ = "America/New_York"
BASE = pd.Timestamp("2024-06-03 08:00", tz=TZ)


def _clock(minutes: int) -> pd.Timestamp:
    return BASE + pd.Timedelta(int(minutes), unit="min")


def _minutes(timeframe: Timeframe) -> int:
    return {
        Timeframe.M1: 1,
        Timeframe.M5: 5,
        Timeframe.M15: 15,
        Timeframe.H1: 60,
        Timeframe.H4: 240,
    }[timeframe]


def _candle(
    index: int,
    *,
    timeframe: Timeframe = Timeframe.M5,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.0,
) -> Candle:
    minutes = _minutes(timeframe)
    start = BASE + pd.Timedelta(int(minutes * index), unit="min")
    return Candle(
        timeframe=timeframe,
        start=start,
        end=start + pd.Timedelta(int(minutes), unit="min"),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100.0 + index,
        symbol="NQ",
        instrument_id=1,
        observed_minutes=minutes,
        expected_minutes=minutes,
        complete=True,
        price_tick_size=0.25,
    )


def _swing(
    swing_id: str,
    side: SwingSide,
    price: float,
    pivot_start: pd.Timestamp,
    confirmed_at: pd.Timestamp,
    *,
    timeframe: Timeframe = Timeframe.M5,
    confirmation_delay_bars: int = 1,
    nesting_depth: int = 0,
) -> SwingPoint:
    minutes = _minutes(timeframe)
    return SwingPoint(
        swing_id=swing_id,
        timeframe=timeframe,
        symbol="NQ",
        instrument_id=1,
        side=side,
        price=price,
        price_ticks=int(price * 4),
        pivot_start=pivot_start,
        pivot_end=pivot_start + pd.Timedelta(int(minutes), unit="min"),
        observed_at=confirmed_at,
        confirmed_at=confirmed_at,
        lifecycle=SwingLifecycle.CONFIRMED,
        relation=(
            SwingRelation.HH
            if side is SwingSide.HIGH
            else SwingRelation.HL
        ),
        confirmation_delay_bars=confirmation_delay_bars,
        nesting_depth=nesting_depth,
    )


def _level(
    item_id: str,
    side: str,
    price: float,
    source_id: str,
    *,
    confirmed_minutes: int = 10,
) -> LiquidityInventoryItem:
    confirmed_at = _clock(confirmed_minutes)
    return LiquidityInventoryItem(
        item_id=item_id,
        timeframe=Timeframe.M5,
        side=side,
        kind="swing",
        price=price,
        lower_bound=price,
        upper_bound=price,
        formed_at=confirmed_at - pd.Timedelta(5, unit="min"),
        confirmed_at=confirmed_at,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=(source_id,),
        age_bars=0,
        strength=0.5,
    )


def _balance_range() -> DealingRangeState:
    formed_at = _clock(180)
    narrowness = 0.5
    compression = 0.0
    boundary_tests = 1.0 / 3.0
    crossings = 0.0
    inside = 1.0
    return DealingRangeState(
        range_id="balance-range",
        protocol_hash="balance-protocol",
        source_group12_protocol_hash="source-protocol",
        symbol="NQ",
        instrument_id=1,
        timeframe=Timeframe.H1,
        lifecycle=DealingRangeLifecycle.ACTIVE,
        lower_source_zone_id="support-zone",
        upper_source_zone_id="resistance-zone",
        lower_source_confirmed_at=_clock(60),
        upper_source_confirmed_at=_clock(120),
        lower_source_tested_at=None,
        upper_source_tested_at=None,
        lower_source_member_swing_ids=("balance-low",),
        upper_source_member_swing_ids=("balance-high",),
        lower_source_lower_bound=95.0,
        lower_source_upper_bound=96.0,
        upper_source_lower_bound=104.0,
        upper_source_upper_bound=105.0,
        formed_at=formed_at,
        balance_confirmed_at=None,
        broken_at=None,
        state_started_at=formed_at,
        last_updated_at=formed_at,
        lower_bound=95.0,
        upper_bound=105.0,
        midpoint=100.0,
        value_price=100.0,
        formation_atr=5.0,
        width_points=10.0,
        width_atr_at_formation=2.0,
        candidate_real_h1_bars=1,
        lower_touch_count=1,
        upper_touch_count=1,
        midpoint_crossings=0,
        inside_close_fraction=inside,
        compression_ratio=1.0,
        narrowness_strength=narrowness,
        compression_strength=compression,
        boundary_test_strength=boundary_tests,
        crossing_strength=crossings,
        strength=(
            narrowness
            + compression
            + boundary_tests
            + crossings
            + inside
        )
        / 5.0,
        age_h1_bars=0,
        transition_reason="source_pair_selected",
    )


def test_structural_leg_freezes_complete_path_and_strict_prior_atr() -> None:
    warmup = tuple(_candle(index) for index in range(14))
    path = (
        _candle(14, open_=100.0, high=101.0, low=99.0, close=100.0),
        _candle(15, open_=100.0, high=102.0, low=98.5, close=99.5),
        _candle(16, open_=99.5, high=103.0, low=99.25, close=102.0),
        _candle(17, open_=102.0, high=104.0, low=101.0, close=103.0),
    )
    low = _swing(
        "leg-low",
        SwingSide.LOW,
        99.0,
        path[0].start,
        path[1].end,
    )
    high = _swing(
        "leg-high",
        SwingSide.HIGH,
        104.0,
        path[-1].start,
        path[-1].end + pd.Timedelta(5, unit="min"),
    )

    leg = build_structural_legs(
        Timeframe.M5,
        (low, high),
        (*warmup, *path),
        tick_size=0.25,
    )[0]

    assert leg.foundation_version == FOUNDATION_VERSION
    assert leg.atr_at_leg_start == pytest.approx(2.0)
    assert leg.amplitude_points == pytest.approx(5.0)
    assert leg.amplitude_ticks == 20
    assert leg.amplitude_atr == pytest.approx(2.5)
    assert leg.duration_bars == 4
    assert leg.duration_seconds == 15 * 60
    assert leg.path_candle_ids == tuple(
        candle_identity(candle, tick_size=0.25) for candle in path
    )
    assert leg.atr_source_candle_ids == tuple(
        candle_identity(candle, tick_size=0.25) for candle in warmup
    )
    assert len(set(leg.atr_source_candle_ids)) == 14
    assert leg.close_efficiency == pytest.approx(0.75)
    assert leg.efficiency == leg.close_efficiency
    assert leg.extreme_path_efficiency == pytest.approx(1.0)
    assert leg.close_mae_points == pytest.approx(0.5)
    assert leg.close_mae_atr == pytest.approx(0.25)
    assert leg.wick_mae_points == pytest.approx(0.5)
    assert leg.wick_mae_atr == pytest.approx(0.25)

    future = _candle(
        18,
        open_=103.0,
        high=150.0,
        low=50.0,
        close=125.0,
    )
    with_future = build_structural_legs(
        Timeframe.M5,
        (low, high),
        (*warmup, *path, future),
        tick_size=0.25,
    )[0]
    assert with_future == leg


def test_structural_leg_atr_excludes_a_future_ending_warmup_candle() -> None:
    ordinary_warmup = tuple(_candle(index) for index in range(15))
    path = (
        _candle(15, low=99.0, close=100.0),
        _candle(16, high=103.0, close=102.0),
    )
    future_ending = replace(
        ordinary_warmup[5],
        end=path[-1].end + pd.Timedelta(5, unit="min"),
        high=150.0,
        low=50.0,
        normalized_ohlc_ticks=None,
    )
    warmup = (
        *ordinary_warmup[:5],
        future_ending,
        *ordinary_warmup[6:],
    )
    low = _swing(
        "future-end-low",
        SwingSide.LOW,
        99.0,
        path[0].start,
        path[0].end + pd.Timedelta(5, unit="min"),
    )
    high = _swing(
        "future-end-high",
        SwingSide.HIGH,
        103.0,
        path[-1].start,
        path[-1].end + pd.Timedelta(5, unit="min"),
    )

    leg = build_structural_legs(
        Timeframe.M5,
        (low, high),
        (*warmup, *path),
        tick_size=0.25,
    )[0]
    expected_sources = (
        *ordinary_warmup[:5],
        *ordinary_warmup[6:],
    )

    assert leg.atr_at_leg_start == pytest.approx(2.0)
    assert leg.atr_source_candle_ids == tuple(
        candle_identity(candle, tick_size=0.25)
        for candle in expected_sources
    )
    assert candle_identity(future_ending, tick_size=0.25) not in (
        leg.atr_source_candle_ids
    )


def test_structural_leg_cold_prefix_fails_closed_or_uses_frozen_start_atr() -> None:
    path = (
        _candle(14, open_=100.0, high=101.0, low=99.0, close=100.0),
        _candle(15, open_=100.0, high=102.0, low=99.5, close=101.0),
    )
    low = _swing(
        "cold-low",
        SwingSide.LOW,
        99.0,
        path[0].start,
        path[0].end + pd.Timedelta(5, unit="min"),
    )
    high = _swing(
        "cold-high",
        SwingSide.HIGH,
        102.0,
        path[1].start,
        path[1].end + pd.Timedelta(5, unit="min"),
    )

    with pytest.raises(ValueError, match="strict-prior bars"):
        build_structural_legs(Timeframe.M5, (low, high), path)

    leg = build_structural_legs(
        Timeframe.M5,
        (low, high),
        path,
        frozen_start_atr_by_swing_id={low.swing_id: 1.5},
        frozen_start_atr_source_candle_ids_by_swing_id={
            low.swing_id: tuple(f"frozen-prior-{index}" for index in range(14))
        },
    )[0]
    assert leg.atr_at_leg_start == 1.5
    assert len(leg.atr_source_candle_ids) == 14

    compatibility = build_structural_legs(
        Timeframe.M5,
        (low, high),
        path,
        atr=1.5,
    )[0]
    assert compatibility.foundation_version is None
    assert compatibility.atr_at_leg_start is None
    assert compatibility.atr_source_candle_ids == ()
    assert compatibility.path_candle_ids == ()


def test_structural_leg_v2_rejects_a_missing_native_path_bar() -> None:
    warmup = tuple(_candle(index) for index in range(14))
    path = (
        _candle(14, low=99.0, close=100.0),
        _candle(16, high=103.0, close=102.0),
        _candle(17, high=104.0, close=103.0),
    )
    low = _swing(
        "gapped-leg-low",
        SwingSide.LOW,
        99.0,
        path[0].start,
        path[0].end + pd.Timedelta(5, unit="min"),
    )
    high = _swing(
        "gapped-leg-high",
        SwingSide.HIGH,
        104.0,
        path[-1].start,
        path[-1].end + pd.Timedelta(5, unit="min"),
    )

    with pytest.raises(ValueError, match="not contiguous"):
        build_structural_legs(
            Timeframe.M5,
            (low, high),
            (*warmup, *path),
            tick_size=0.25,
        )

    compatibility = build_structural_legs(
        Timeframe.M5,
        (low, high),
        path,
        atr=1.5,
        tick_size=0.25,
    )[0]
    assert compatibility.foundation_version is None


def test_swing_geometry_uses_real_definitional_window_not_role_depth() -> None:
    candles = (
        _candle(0, high=101.0, low=99.0),
        _candle(1, high=102.0, low=99.0),
        _candle(2, high=105.0, low=100.0),
        _candle(3, high=103.0, low=99.5),
        _candle(4, high=102.0, low=98.0),
    )
    swing = _swing(
        "geometry-swing",
        SwingSide.HIGH,
        105.0,
        candles[2].start,
        candles[4].end,
        confirmation_delay_bars=2,
        nesting_depth=9,
    )

    node = build_swing_geometry_nodes((swing,), candles, tick_size=0.25)[0]

    assert node.window_start == candles[0].start
    assert node.window_end == candles[-1].end
    assert node.known_at == swing.confirmed_at
    assert node.lower_bound == 98.0
    assert node.upper_bound == 105.0
    assert len(node.source_candle_ids) == 5
    assert not hasattr(node, "role_depth")
    assert not hasattr(node, "nesting_depth")


def test_swing_geometry_rejects_a_gapped_definitional_window() -> None:
    candles = (_candle(0), _candle(2, high=105.0), _candle(3))
    swing = _swing(
        "gapped-geometry",
        SwingSide.HIGH,
        105.0,
        candles[1].start,
        candles[-1].end,
    )

    with pytest.raises(ValueError, match="not contiguous"):
        build_swing_geometry_nodes((swing,), candles, tick_size=0.25)


def test_swing_geometry_accepts_each_registered_bar_across_memorial_closure(
) -> None:
    starts = tuple(
        pd.Timestamp(value, tz=TZ)
        for value in (
            "2024-05-27 12:59",
            "2024-05-27 18:00",
            "2024-05-27 18:01",
        )
    )
    candles = tuple(
        replace(
            _candle(index, timeframe=Timeframe.M1, high=105.0),
            start=start,
            end=start + pd.Timedelta(1, unit="min"),
        )
        for index, start in enumerate(starts)
    )
    swing = _swing(
        "memorial-reopen-geometry",
        SwingSide.HIGH,
        105.0,
        candles[1].start,
        candles[-1].end,
        timeframe=Timeframe.M1,
    )

    node = build_swing_geometry_nodes(
        (swing,),
        candles,
        tick_size=0.25,
    )[0]

    assert node.window_start == candles[0].start
    assert node.window_end == candles[-1].end
    assert node.source_candle_ids == tuple(
        candle_identity(candle, tick_size=0.25) for candle in candles
    )


def test_swing_geometry_accepts_registered_shortened_memorial_h4_bucket() -> None:
    bounds = (
        ("2024-05-27 02:00", "2024-05-27 06:00", 240),
        ("2024-05-27 06:00", "2024-05-27 10:00", 240),
        ("2024-05-27 10:00", "2024-05-27 13:00", 180),
        ("2024-05-27 18:00", "2024-05-27 22:00", 240),
        ("2024-05-27 22:00", "2024-05-28 02:00", 240),
    )
    candles = tuple(
        replace(
            _candle(index, timeframe=Timeframe.H4, high=105.0),
            start=pd.Timestamp(start, tz=TZ),
            end=pd.Timestamp(end, tz=TZ),
            expected_minutes=minutes,
            observed_minutes=minutes,
            real_minutes=minutes,
        )
        for index, (start, end, minutes) in enumerate(bounds)
    )
    swing = replace(
        _swing(
            "memorial-shortened-h4-geometry",
            SwingSide.HIGH,
            105.0,
            candles[2].start,
            candles[-1].end,
            timeframe=Timeframe.H4,
            confirmation_delay_bars=2,
        ),
        pivot_end=candles[2].end,
    )

    node = build_swing_geometry_nodes((swing,), candles, tick_size=0.25)[0]

    assert node.window_start == candles[0].start
    assert node.window_end == candles[-1].end
    assert node.source_candle_ids == tuple(
        candle_identity(candle, tick_size=0.25) for candle in candles
    )


@pytest.mark.parametrize(
    ("start", "end", "coverage_minutes"),
    (
        ("2024-05-27 10:00", "2024-05-27 13:00", 240),
        ("2024-06-03 10:00", "2024-06-03 13:00", 180),
        ("2024-06-03 10:00", "2024-06-03 15:00", 300),
    ),
)
def test_swing_geometry_rejects_registered_coverage_or_wall_clock_tamper(
    start: str,
    end: str,
    coverage_minutes: int,
) -> None:
    candle = replace(
        _candle(0, timeframe=Timeframe.H4, high=105.0),
        start=pd.Timestamp(start, tz=TZ),
        end=pd.Timestamp(end, tz=TZ),
        expected_minutes=coverage_minutes,
        observed_minutes=coverage_minutes,
        real_minutes=coverage_minutes,
    )

    with pytest.raises(ValueError, match="lacks native-duration"):
        _require_contiguous_native_candles(
            (candle,),
            timeframe=Timeframe.H4,
            object_name="swing geometry",
        )


@pytest.mark.parametrize(
    "starts",
    (
        (
            "2024-05-27 12:59:00",
            "2024-05-27 18:01:00",
            "2024-05-27 18:02:00",
        ),
        (
            "2024-06-03 08:00:00",
            "2024-06-03 08:00:30",
            "2024-06-03 08:01:30",
        ),
    ),
)
def test_swing_geometry_rejects_skipped_reopen_bar_or_overlap(
    starts: tuple[str, str, str],
) -> None:
    candles = tuple(
        replace(
            _candle(index, timeframe=Timeframe.M1, high=105.0),
            start=pd.Timestamp(start, tz=TZ),
            end=pd.Timestamp(start, tz=TZ) + pd.Timedelta(1, unit="min"),
        )
        for index, start in enumerate(starts)
    )
    swing = _swing(
        "invalid-registered-geometry",
        SwingSide.HIGH,
        105.0,
        candles[1].start,
        candles[-1].end,
        timeframe=Timeframe.M1,
    )

    with pytest.raises(ValueError, match="not contiguous|lacks native-duration"):
        build_swing_geometry_nodes((swing,), candles, tick_size=0.25)


def test_swing_parent_assignments_are_append_only_and_role_independent() -> None:
    child = SwingGeometryNode(
        swing_id="child",
        timeframe=Timeframe.M5,
        symbol="NQ",
        instrument_id=1,
        window_start=_clock(125),
        window_end=_clock(130),
        lower_bound=100.0,
        upper_bound=104.0,
        known_at=_clock(130),
        source_candle_ids=("child-bar",),
    )
    early_parent = SwingGeometryNode(
        swing_id="early-parent",
        timeframe=Timeframe.H1,
        symbol="NQ",
        instrument_id=1,
        window_start=_clock(60),
        window_end=_clock(135),
        lower_bound=90.0,
        upper_bound=110.0,
        known_at=_clock(135),
        source_candle_ids=("early-parent-bars",),
    )
    later_smaller_parent = SwingGeometryNode(
        swing_id="later-smaller-parent",
        timeframe=Timeframe.M15,
        symbol="NQ",
        instrument_id=1,
        window_start=_clock(120),
        window_end=_clock(140),
        lower_bound=99.0,
        upper_bound=105.0,
        known_at=_clock(140),
        source_candle_ids=("later-parent-bars",),
    )

    first = update_swing_geometry_assignments(
        (child, early_parent, later_smaller_parent),
        known_at=_clock(135),
    )
    first_child = next(item for item in first if item.child_swing_id == "child")
    assert first_child.parent_swing_id == early_parent.swing_id
    assert first_child.geometric_depth == 1

    history = update_swing_geometry_assignments(
        (child, early_parent, later_smaller_parent),
        first,
        known_at=_clock(140),
    )
    child_history = tuple(
        item for item in history if item.child_swing_id == child.swing_id
    )
    assert len(child_history) == 2
    assert child_history[0] == first_child
    assert child_history[1].parent_swing_id == later_smaller_parent.swing_id
    assert child_history[1].supersedes_assignment_id == first_child.assignment_id

    role_view = SwingHierarchyView(
        swing_id=child.swing_id,
        timeframe=Timeframe.M5,
        semantic_rank=SwingRank.EXTERNAL,
        nesting_depth=3,
        assignments=(
            SwingRankAssignment(
                rank=SwingRank.MICRO,
                assigned_at=_clock(130),
                assignment_event_id="micro-role",
                source_kind="confirmed_swing",
            ),
            SwingRankAssignment(
                rank=SwingRank.EXTERNAL,
                assigned_at=_clock(140),
                assignment_event_id="external-role",
                source_kind="protected_swing",
            ),
        ),
    )
    assert role_view.role_depth == 3
    assert child_history[-1].geometric_depth == 1


def test_structural_and_balance_ranges_remain_distinct_cold_geometries() -> None:
    low = _swing(
        "structural-low",
        SwingSide.LOW,
        90.0,
        _clock(0),
        _clock(120),
        timeframe=Timeframe.H1,
    )
    high = _swing(
        "structural-high",
        SwingSide.HIGH,
        110.0,
        _clock(60),
        _clock(180),
        timeframe=Timeframe.H1,
    )
    structural = build_structural_range(
        "h1-external-generation",
        Direction.LONG,
        low,
        high,
        known_at=_clock(180),
    )
    balance = _balance_range()

    assert isinstance(structural, StructuralRangeState)
    assert BalanceRangeState is DealingRangeState
    assert not isinstance(structural, BalanceRangeState)
    assert structural.symbol == balance.symbol == "NQ"
    terminated = terminate_structural_range(
        structural,
        terminated_at=_clock(240),
        reason="structure_generation_terminated",
    )
    assert terminated.terminated_at == _clock(240)
    assert terminated.termination_reason == "structure_generation_terminated"


def _densified_candle(
    index: int,
    *,
    timeframe: Timeframe = Timeframe.M5,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.0,
    synthetic_minutes: int = 1,
) -> Candle:
    """One native candle whose bucket contains a densified no-trade minute."""

    real = _candle(
        index,
        timeframe=timeframe,
        open_=open_,
        high=high,
        low=low,
        close=close,
    )
    return replace(
        real,
        synthetic_minutes=synthetic_minutes,
        real_minutes=real.observed_minutes - synthetic_minutes,
    )


def test_structural_leg_path_admits_a_densified_no_trade_bar() -> None:
    warmup = tuple(_candle(index) for index in range(14))
    path = (
        _candle(14, open_=100.0, high=101.0, low=99.0, close=100.0),
        _densified_candle(15, open_=100.0, high=102.0, low=98.5, close=99.5),
        _candle(16, open_=99.5, high=103.0, low=99.25, close=102.0),
        _candle(17, open_=102.0, high=104.0, low=101.0, close=103.0),
    )
    low = _swing("leg-low", SwingSide.LOW, 99.0, path[0].start, path[1].end)
    high = _swing(
        "leg-high",
        SwingSide.HIGH,
        104.0,
        path[-1].start,
        path[-1].end + pd.Timedelta(5, unit="min"),
    )

    leg = build_structural_legs(
        Timeframe.M5,
        (low, high),
        (*warmup, *path),
        tick_size=0.25,
    )[0]

    assert leg.foundation_version == FOUNDATION_VERSION
    assert leg.synthetic_path_minutes == 1
    assert leg.duration_bars == 4
    assert leg.path_candle_ids == tuple(
        candle_identity(candle, tick_size=0.25) for candle in path
    )


def test_structural_leg_atr_ancestry_still_cites_only_real_bars() -> None:
    # A densified bar sits immediately before the leg, inside the ATR window.
    warmup = (
        *(_candle(index) for index in range(14)),
        _densified_candle(14),
    )
    path = (
        _candle(15, open_=100.0, high=101.0, low=99.0, close=100.0),
        _candle(16, open_=100.0, high=102.0, low=98.5, close=99.5),
        _candle(17, open_=99.5, high=103.0, low=99.25, close=102.0),
        _candle(18, open_=102.0, high=104.0, low=101.0, close=103.0),
    )
    low = _swing("leg-low", SwingSide.LOW, 99.0, path[0].start, path[1].end)
    high = _swing(
        "leg-high",
        SwingSide.HIGH,
        104.0,
        path[-1].start,
        path[-1].end + pd.Timedelta(5, unit="min"),
    )

    leg = build_structural_legs(
        Timeframe.M5,
        (low, high),
        (*warmup, *path),
        tick_size=0.25,
    )[0]

    densified_id = candle_identity(warmup[-1], tick_size=0.25)
    assert densified_id not in leg.atr_source_candle_ids
    assert len(leg.atr_source_candle_ids) == 14
    assert leg.atr_source_candle_ids == tuple(
        candle_identity(candle, tick_size=0.25) for candle in warmup[:14]
    )


def test_swing_geometry_window_admits_a_densified_no_trade_bar() -> None:
    candles = (
        _candle(0),
        _candle(1, high=105.0),
        _densified_candle(2),
    )
    swing = _swing(
        "densified-geometry",
        SwingSide.HIGH,
        105.0,
        candles[1].start,
        candles[-1].end,
    )

    node = build_swing_geometry_nodes((swing,), candles, tick_size=0.25)[0]

    assert node.synthetic_window_minutes == 1
    assert node.source_candle_ids == tuple(
        candle_identity(candle, tick_size=0.25) for candle in candles
    )


@pytest.mark.parametrize(
    ("synthetic_minutes", "complete", "expected"),
    (
        (0, True, BarCoverage.REAL),
        (1, True, BarCoverage.DENSIFIED),
        (5, True, BarCoverage.DENSIFIED),
        (0, False, BarCoverage.INCOMPLETE),
        (1, False, BarCoverage.INCOMPLETE),
    ),
)
def test_one_rule_classifies_bar_coverage(
    synthetic_minutes: int,
    complete: bool,
    expected: BarCoverage,
) -> None:
    """Producer and contract read one admission rule, not two copies of it."""

    minutes = _minutes(Timeframe.M5)
    base = _candle(0, timeframe=Timeframe.M5)
    candle = replace(
        base,
        complete=complete,
        observed_minutes=minutes if complete else minutes - 1,
        real_minutes=(minutes if complete else minutes - 1) - synthetic_minutes,
        synthetic_minutes=synthetic_minutes,
    )

    assert candle_coverage(candle) == expected
    # the same shape, described the way a BAR event's evidence describes it
    assert bar_evidence_coverage(
        {
            "real_completed": candle.real_completed,
            "clock_only": not candle.real_completed,
            "complete": candle.complete,
            "observed_minutes": candle.observed_minutes,
            "expected_minutes": candle.expected_minutes,
            "real_minutes": candle.real_minutes,
            "synthetic_minutes": candle.synthetic_minutes,
        }
    ) == expected


def test_definitional_path_and_atr_admission_follow_the_coverage_rule() -> None:
    assert BarCoverage.REAL.admits_definitional_path is True
    assert BarCoverage.DENSIFIED.admits_definitional_path is True
    assert BarCoverage.INCOMPLETE.admits_definitional_path is False

    assert BarCoverage.REAL.admits_atr_window is True
    assert BarCoverage.DENSIFIED.admits_atr_window is False
    assert BarCoverage.INCOMPLETE.admits_atr_window is False


def test_structural_leg_path_class_names_what_the_formula_measures() -> None:
    """The leg field measures its own path length, not a swing's rank."""

    warmup = tuple(_candle(index) for index in range(14))
    two_bar = (
        _candle(14, open_=100.0, high=101.0, low=99.0, close=100.0),
        _candle(15, open_=100.0, high=104.0, low=99.5, close=103.0),
    )
    low = _swing("pc-low", SwingSide.LOW, 99.0, two_bar[0].start, two_bar[1].end)
    high = _swing(
        "pc-high",
        SwingSide.HIGH,
        104.0,
        two_bar[-1].start,
        two_bar[-1].end + pd.Timedelta(5, unit="min"),
    )

    leg = build_structural_legs(
        Timeframe.M5,
        (low, high),
        (*warmup, *two_bar),
        tick_size=0.25,
    )[0]

    assert leg.path_class is SwingRank.MICRO
    assert not hasattr(leg, "rank")


def test_leg_decoder_fails_closed_on_a_missing_path_class() -> None:
    """A leg payload without path_class is a defect, not an 'internal' leg.

    Every other field in the decoder is a strict lookup; silently defaulting
    this one would turn an unreadable payload into a plausible-looking leg.
    """

    from eyes.core.market_state import _leg_from_event
    from contract.eye import EventKind
    from eyes.core.semantic_event_emitter import _event

    warmup = tuple(_candle(index) for index in range(14))
    path = (
        _candle(14, open_=100.0, high=101.0, low=99.0, close=100.0),
        _candle(15, open_=100.0, high=102.0, low=98.5, close=99.5),
        _candle(16, open_=99.5, high=103.0, low=99.25, close=102.0),
        _candle(17, open_=102.0, high=104.0, low=101.0, close=103.0),
    )
    low = _swing("dec-low", SwingSide.LOW, 99.0, path[0].start, path[1].end)
    high = _swing(
        "dec-high",
        SwingSide.HIGH,
        104.0,
        path[-1].start,
        path[-1].end + pd.Timedelta(5, unit="min"),
    )
    leg = build_structural_legs(
        Timeframe.M5,
        (low, high),
        (*warmup, *path),
        tick_size=0.25,
    )[0]

    evidence = {
        "leg_id": leg.leg_id,
        "start_swing_id": leg.start_swing_id,
        "end_swing_id": leg.end_swing_id,
        "start_event_time": leg.start_event_time.isoformat(),
        "end_event_time": leg.end_event_time.isoformat(),
        "start_price": leg.start_price,
        "end_price": leg.end_price,
        "start_close": leg.start_close,
        "end_close": leg.end_close,
        "amplitude_points": leg.amplitude_points,
        "amplitude_atr": leg.amplitude_atr,
        "duration_bars": leg.duration_bars,
        "duration_minutes": leg.duration_minutes,
        "efficiency": leg.efficiency,
        "max_retracement_points": leg.max_retracement_points,
        "max_retracement_atr": leg.max_retracement_atr,
    }
    event = _event(
        EventKind.STRUCTURAL_LEG_CREATED,
        leg.known_at,
        Timeframe.M5,
        "above",
        leg.end_price,
        0.5,
        (),
        evidence,
        direction=Direction.LONG,
        event_time=leg.end_event_time,
    )

    with pytest.raises(KeyError, match="path_class"):
        _leg_from_event(event)
