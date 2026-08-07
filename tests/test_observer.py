from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from smc_trader.causal import CausalMarketReader
from smc_trader.model import (
    Bar,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityPoolLifecycle,
    LiquidityPoolState,
    Timeframe,
)
from smc_trader.observation import (
    CausalObserver,
    ExecutionRealityInput,
    ObserverConfig,
)

from .helpers import MODEL_SCALE_SPECS, CORE_TEST_SCALE_SPECS, session_bars


STRUCTURE_PROTOCOL = "configs/primitives_structure_liquidity.json"


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
