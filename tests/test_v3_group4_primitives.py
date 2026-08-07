from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from smc_trader.group4 import (
    CausalGroup4Tracker,
    Group4Protocol,
    Group4Update,
)
from smc_trader.model import (
    Candle,
    DealingRangeLifecycle,
    DealingRangeState,
    EventKind,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityPoolLifecycle,
    LiquidityPoolState,
    ManipulationLifecycle,
    SupportResistanceLifecycle,
    SupportResistanceState,
    Timeframe,
)
from smc_trader.observation import (
    CausalObserver,
    ObserverConfig,
    _event,
)
from smc_trader.playbooks import _visible_levels
from smc_trader.risk import _visible_level_ids

from .helpers import CORE_TEST_SCALE_SPECS, market_observation


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/primitives_range.json"
GROUP12_PROTOCOL_PATH = (
    ROOT / "configs/primitives_structure_liquidity.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


PROTOCOL_SHA = _sha256(PROTOCOL_PATH)
GROUP12_SHA = _sha256(GROUP12_PROTOCOL_PATH)
H1_BASE = pd.Timestamp("2025-01-06T00:00:00-05:00")
M1_BASE = pd.Timestamp("2025-01-07T09:30:00-05:00")


def _protocol() -> Group4Protocol:
    return Group4Protocol.from_file(PROTOCOL_PATH)


def _candle(
    timeframe: Timeframe,
    start: pd.Timestamp,
    *,
    close: float = 100.0,
    span: float = 1.0,
    high: float | None = None,
    low: float | None = None,
    synthetic: bool = False,
) -> Candle:
    minutes = {
        Timeframe.H1: 60,
        Timeframe.M1: 1,
    }[timeframe]
    upper = close + span / 2.0 if high is None else high
    lower = close - span / 2.0 if low is None else low
    return Candle(
        timeframe=timeframe,
        start=start,
        end=start + pd.Timedelta(minutes=minutes),
        open=close,
        high=upper,
        low=lower,
        close=close,
        volume=0.0 if synthetic else 100.0,
        symbol="NQH5",
        instrument_id=1,
        observed_minutes=minutes,
        expected_minutes=minutes,
        complete=True,
        real_minutes=0 if synthetic else minutes,
        synthetic_minutes=minutes if synthetic else 0,
    )


def _h1(
    index: int,
    *,
    close: float = 100.0,
    span: float = 1.0,
    synthetic: bool = False,
) -> Candle:
    return _candle(
        Timeframe.H1,
        H1_BASE + pd.Timedelta(hours=index),
        close=close,
        span=span,
        synthetic=synthetic,
    )


def _m1(
    index: int,
    *,
    base: pd.Timestamp = M1_BASE,
    close: float = 100.0,
    high: float = 100.25,
    low: float = 99.75,
    synthetic: bool = False,
) -> Candle:
    return _candle(
        Timeframe.M1,
        base + pd.Timedelta(minutes=index),
        close=close,
        high=high,
        low=low,
        synthetic=synthetic,
    )


def _zone(side: str) -> SupportResistanceState:
    confirmed_at = H1_BASE - pd.Timedelta(hours=3)
    tested_at = H1_BASE - pd.Timedelta(hours=2)
    if side == "support":
        lower_bound, upper_bound, anchor = 99.0, 99.25, 99.125
    else:
        lower_bound, upper_bound, anchor = 100.75, 101.0, 100.875
    prefix = "lo" if side == "support" else "hi"
    return SupportResistanceState(
        zone_id=f"zone:{side}",
        timeframe=Timeframe.H1,
        side=side,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        anchor_price=anchor,
        formed_at=confirmed_at - pd.Timedelta(hours=1),
        confirmed_at=confirmed_at,
        lifecycle=SupportResistanceLifecycle.TESTED,
        member_swing_ids=(f"{prefix}:1", f"{prefix}:2"),
        touch_times=(confirmed_at, tested_at),
        reaction_magnitudes_atr=(0.5, 0.6),
        age_bars=2,
        strength=0.7,
        tested_at=tested_at,
        total_touch_count=2,
    )


def _pool(
    side: str,
    *,
    confirmed_at: pd.Timestamp,
    identity: str | None = None,
) -> tuple[LiquidityPoolState, LiquidityInventoryItem]:
    pool_id = identity or f"pool-{side}"
    formed_at = confirmed_at - pd.Timedelta(minutes=1)
    if side == "above":
        lower_bound, upper_bound = 100.75, 101.0
        kind, price = "equal_highs", upper_bound
    else:
        lower_bound, upper_bound = 99.0, 99.25
        kind, price = "equal_lows", lower_bound
    members = (f"{pool_id}:1", f"{pool_id}:2")
    pool = LiquidityPoolState(
        pool_id=pool_id,
        timeframe=Timeframe.M1,
        side=side,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        midpoint=(lower_bound + upper_bound) / 2.0,
        formed_at=formed_at,
        confirmed_at=confirmed_at,
        lifecycle=LiquidityPoolLifecycle.FORMED,
        member_swing_ids=members,
        touch_times=(formed_at, confirmed_at),
        age_bars=1,
        strength=0.75,
        total_touch_count=2,
    )
    inventory = LiquidityInventoryItem(
        item_id=f"pool:{pool_id}",
        timeframe=Timeframe.M1,
        side=side,
        kind=kind,
        price=price,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        formed_at=formed_at,
        confirmed_at=confirmed_at,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=members,
        age_bars=1,
        strength=pool.strength,
    )
    return pool, inventory


def _warm_h1(
    tracker: CausalGroup4Tracker,
    count: int = 14,
) -> None:
    for index in range(count):
        output = tracker.on_completed_h1(_h1(index), ())
        assert output.dealing_ranges == ()


def test_range_candidates_ignore_nonstructural_support_resistance() -> None:
    tracker = CausalGroup4Tracker(_protocol())
    support = _zone("support")
    resistance = _zone("resistance")

    def as_previous_day(zone: SupportResistanceState) -> SupportResistanceState:
        source_id = f"reference_source:day:2025-01-05:{zone.side}"
        return replace(
            zone,
            member_swing_ids=(),
            source_kind="previous_day",
            source_ids=(source_id,),
            structural_rank="external",
        )

    assert tracker._eligible_pairs(
        _h1(0, close=100.0),
        (as_previous_day(support), as_previous_day(resistance)),
    ) == ()


def _mature_range(
    protocol: Group4Protocol,
) -> tuple[
    CausalGroup4Tracker,
    DealingRangeState,
    DealingRangeState,
]:
    tracker = CausalGroup4Tracker(protocol)
    zones = (_zone("support"), _zone("resistance"))
    _warm_h1(tracker)
    formed = tracker.on_completed_h1(
        _h1(14, close=99.75, span=1.0),
        zones,
    ).dealing_ranges[-1]
    output = None
    for index, close, span in (
        (15, 100.25, 1.0),
        (16, 99.75, 1.0),
        (17, 100.25, 1.0),
        (18, 99.75, 0.25),
        (19, 100.25, 0.25),
        (20, 99.75, 0.25),
        (21, 100.25, 0.25),
    ):
        output = tracker.on_completed_h1(
            _h1(index, close=close, span=span),
            zones,
        )
    assert output is not None
    return tracker, formed, output.dealing_ranges[-1]


def _warm_m1(
    tracker: CausalGroup4Tracker,
    *,
    base: pd.Timestamp = M1_BASE,
    count: int = 15,
    inventory: tuple[LiquidityInventoryItem, ...] = (),
    pools: tuple[LiquidityPoolState, ...] = (),
) -> None:
    for index in range(count):
        output = tracker.on_completed_update(
            _m1(index, base=base),
            prior_inventory=inventory,
            liquidity_pools=pools,
        )
        assert output.manipulations == ()


def _frozen_range_geometry(
    state: DealingRangeState,
) -> tuple[object, ...]:
    return tuple(
        getattr(state, name)
        for name in (
            "range_id",
            "protocol_hash",
            "source_group12_protocol_hash",
            "lower_source_zone_id",
            "upper_source_zone_id",
            "lower_source_lower_bound",
            "lower_source_upper_bound",
            "upper_source_lower_bound",
            "upper_source_upper_bound",
            "formed_at",
            "lower_bound",
            "upper_bound",
            "midpoint",
            "value_price",
            "formation_atr",
            "width_points",
            "width_atr_at_formation",
        )
    )


def test_group4_protocol_tracks_current_config_and_upstream_binding() -> None:
    protocol = _protocol()
    payload = json.loads(PROTOCOL_PATH.read_bytes())
    parameters = payload["engineering_parameters"]

    assert _sha256(PROTOCOL_PATH) == PROTOCOL_SHA
    assert protocol.protocol_hash == PROTOCOL_SHA
    assert protocol.source_group12_protocol_hash == GROUP12_SHA
    assert (
        payload["upstream"]["group12_protocol_sha256"]
        == GROUP12_SHA
    )
    assert (
        protocol.protocol_version,
        protocol.tick_size,
        protocol.h1_atr_period,
        protocol.m1_atr_period,
        protocol.minimum_candidate_real_h1_bars,
        protocol.maximum_forming_real_h1_bars,
        protocol.maximum_ranges,
        protocol.maximum_manipulations,
    ) == (
        payload["protocol_version"],
        payload["tick_size"],
        parameters["h1_atr_period"],
        parameters["m1_atr_period"],
        parameters["minimum_candidate_real_h1_bars"],
        parameters["maximum_forming_real_h1_bars"],
        parameters["retained_dealing_range_states"],
        parameters["retained_manipulation_states"],
    )
    assert isinstance(protocol.protocol_version, str)
    assert protocol.protocol_version
    assert isinstance(protocol.tick_size, (int, float))
    assert not isinstance(protocol.tick_size, bool)
    assert protocol.tick_size > 0.0
    assert all(
        type(item) is int and item > 0
        for item in (
            protocol.h1_atr_period,
            protocol.m1_atr_period,
            protocol.minimum_candidate_real_h1_bars,
            protocol.maximum_forming_real_h1_bars,
            protocol.maximum_ranges,
            protocol.maximum_manipulations,
        )
    )
    assert (
        protocol.minimum_candidate_real_h1_bars
        <= protocol.maximum_forming_real_h1_bars
    )
    with pytest.raises(ValueError):
        replace(protocol, tick_size=0.0)


def test_h1_range_forms_matures_breaks_and_never_rewrites_geometry() -> None:
    tracker, formed, mature = _mature_range(_protocol())

    assert formed.lifecycle is DealingRangeLifecycle.FORMING
    assert formed.transition_reason == "source_pair_selected"
    assert formed.candidate_real_h1_bars == 1
    assert (formed.lower_bound, formed.upper_bound, formed.midpoint) == (
        99.0,
        101.0,
        100.0,
    )
    assert formed.value_price == formed.midpoint
    assert formed.formation_atr == pytest.approx(1.0)
    assert formed.width_atr_at_formation == pytest.approx(2.0)

    assert mature.lifecycle is DealingRangeLifecycle.MATURE
    assert mature.transition_reason == "maturity_conditions_met"
    assert mature.candidate_real_h1_bars == 8
    assert mature.age_h1_bars == 7
    assert mature.midpoint_crossings >= 2
    assert mature.inside_close_fraction == pytest.approx(1.0)
    assert mature.compression_ratio <= 0.8
    assert _frozen_range_geometry(mature) == _frozen_range_geometry(formed)
    assert {
        (item.side, item.kind, item.price, item.lifecycle)
        for item in tracker.snapshot().range_boundary_inventory
    } == {
        (
            "below",
            "range_boundary",
            mature.lower_bound,
            LiquidityInventoryLifecycle.VISIBLE,
        ),
        (
            "above",
            "range_boundary",
            mature.upper_bound,
            LiquidityInventoryLifecycle.VISIBLE,
        ),
    }

    broken_output = tracker.on_completed_h1(
        _h1(22, close=101.25, span=0.5),
        (_zone("support"), _zone("resistance")),
    )
    broken = broken_output.dealing_ranges[-1]
    assert broken.lifecycle is DealingRangeLifecycle.BROKEN
    assert broken.broken_at == _h1(22).end
    assert broken.transition_reason == "close_beyond_frozen_range"
    assert broken.candidate_real_h1_bars == 8
    assert broken.age_h1_bars == 8
    assert _frozen_range_geometry(broken) == _frozen_range_geometry(formed)
    assert len(broken_output.range_boundary_inventory) == 2


def test_mature_range_owns_a_same_price_sweep_over_a_local_pool() -> None:
    tracker, _, mature = _mature_range(_protocol())
    pool, pool_inventory = _pool(
        "below",
        confirmed_at=M1_BASE,
        identity="coincident-lower-pool",
    )
    inventory = (
        *tracker.snapshot().range_boundary_inventory,
        pool_inventory,
    )
    _warm_m1(
        tracker,
        inventory=inventory,
        pools=(pool,),
    )

    output = tracker.on_completed_update(
        _m1(
            15,
            close=98.90,
            high=100.25,
            low=98.75,
        ),
        prior_inventory=inventory,
        liquidity_pools=(pool,),
    )

    assert len(output.manipulations) == 1
    manipulation = output.manipulations[0]
    assert manipulation.source_kind == "mature_range_boundary"
    assert manipulation.source_id == mature.range_id
    assert pool.pool_id in manipulation.coincident_source_ids


def test_forming_deadline_includes_the_terminal_h1_bar() -> None:
    tracker = CausalGroup4Tracker(_protocol())
    zones = (_zone("support"), _zone("resistance"))
    _warm_h1(tracker)
    tracker.on_completed_h1(
        _h1(14, close=99.75, span=1.0),
        zones,
    )

    output = None
    for index in range(15, 38):
        output = tracker.on_completed_h1(
            _h1(index, close=99.75, span=1.0),
            zones,
        )

    assert output is not None
    terminal = output.dealing_ranges[-1]
    assert terminal.lifecycle is DealingRangeLifecycle.BROKEN
    assert terminal.transition_reason == "maturity_deadline_elapsed"
    assert terminal.candidate_real_h1_bars == 24
    assert terminal.age_h1_bars == 23
    assert terminal.broken_at == _h1(37).end


def test_cold_existing_range_pair_waits_for_a_new_source_identity() -> None:
    tracker = CausalGroup4Tracker(_protocol())
    zones = (_zone("support"), _zone("resistance"))
    _warm_h1(tracker)
    tracker.mark_existing_source_pairs_ineligible(zones)

    blocked = tracker.on_completed_h1(
        _h1(14, close=100.0),
        zones,
    )
    assert blocked.dealing_ranges == ()
    still_blocked = tracker.on_completed_h1(
        _h1(15, close=100.0),
        zones,
    )
    assert still_blocked.dealing_ranges == ()

    fresh_support = replace(
        zones[0],
        zone_id="zone:support:fresh",
    )
    formed = tracker.on_completed_h1(
        _h1(16, close=100.0),
        (fresh_support, zones[1]),
    ).dealing_ranges[-1]
    assert formed.lifecycle is DealingRangeLifecycle.FORMING
    assert formed.lower_source_zone_id == fresh_support.zone_id


def test_pool_source_must_exist_before_bar_then_reentry_must_hold() -> None:
    tracker = CausalGroup4Tracker(_protocol())
    _warm_m1(tracker)
    gated = _m1(
        15,
        close=100.0,
        high=101.25,
        low=99.75,
    )
    pool, inventory = _pool(
        "above",
        confirmed_at=gated.end,
        identity="gated-above",
    )

    gated_output = tracker.on_completed_update(
        gated,
        prior_inventory=(inventory,),
        liquidity_pools=(pool,),
    )
    assert gated_output.manipulations == ()

    sweep = _m1(
        16,
        close=100.0,
        high=101.25,
        low=99.75,
    )
    swept_output = tracker.on_completed_update(
        sweep,
        prior_inventory=(inventory,),
        liquidity_pools=(pool,),
    )
    swept = swept_output.manipulations[-1]
    frozen_extreme = swept.sweep_extreme
    assert swept.lifecycle is ManipulationLifecycle.SWEPT
    assert swept.source_kind == "formed_liquidity_pool"
    assert swept.source_eligible_at == sweep.start
    assert swept.swept_at == sweep.end
    assert swept.transition_reason == "source_swept"
    live_observation = replace(
        market_observation(asof=sweep.end),
        manipulations=(swept,),
    )
    assert live_observation.manipulations == (swept,)

    consumed = replace(
        inventory,
        lifecycle=LiquidityInventoryLifecycle.CONSUMED,
        consumed_at=sweep.end,
        lifecycle_reason="pool_swept",
    )
    resolution = _m1(
        17,
        close=100.0,
        high=101.5,
        low=99.75,
    )
    candidate_output = tracker.on_completed_update(
        resolution,
        prior_inventory=(consumed,),
        liquidity_pools=(),
    )
    candidate = candidate_output.manipulations[-1]
    assert candidate.lifecycle is ManipulationLifecycle.SWEPT
    assert candidate.reentry_candidate_at == resolution.end
    assert candidate.inside_hold_bars == 0
    held = _m1(
        18,
        close=100.0,
        high=100.5,
        low=99.75,
    )
    resolved_output = tracker.on_completed_update(
        held,
        prior_inventory=(consumed,),
        liquidity_pools=(),
    )
    resolved = resolved_output.manipulations[-1]
    assert resolved.lifecycle is ManipulationLifecycle.REACCEPTED
    assert resolved.resolved_at == held.end
    assert resolved.reentry_price == resolution.close
    assert resolved.inside_hold_bars == 1
    assert resolved.sweep_extreme == frozen_extreme
    assert resolved.age_1m_bars == 2

    same_bar_observation = replace(
        market_observation(asof=held.end),
        manipulations=(resolved,),
    )
    assert same_bar_observation.manipulations == (resolved,)
    with pytest.raises(
        ValueError,
        match="retained inventory identity",
    ):
        replace(
            market_observation(
                asof=held.end + pd.Timedelta(minutes=1)
            ),
            manipulations=(resolved,),
        )

    synthetic_output = tracker.on_completed_update(
        _m1(19, synthetic=True),
        prior_inventory=(),
        liquidity_pools=(),
    )
    assert synthetic_output.manipulations == (resolved,)

    retained = tracker.on_completed_update(
        _m1(20),
        prior_inventory=(consumed,),
        liquidity_pools=(pool,),
    )
    assert retained.manipulations == (resolved,)

    compacted = tracker.on_completed_update(
        _m1(21),
        prior_inventory=(),
        liquidity_pools=(),
    )
    assert compacted.manipulations == ()


def test_pool_outside_acceptance_requires_two_consecutive_closes() -> None:
    tracker = CausalGroup4Tracker(_protocol())
    _warm_m1(tracker)
    pool, inventory = _pool(
        "above",
        confirmed_at=M1_BASE - pd.Timedelta(minutes=1),
        identity="outside-above",
    )
    sweep = _m1(15, close=101.25, high=101.5, low=100.0)
    first = tracker.on_completed_update(
        sweep,
        prior_inventory=(inventory,),
        liquidity_pools=(pool,),
    ).manipulations[-1]
    assert first.lifecycle is ManipulationLifecycle.SWEPT
    assert first.close_outside_on_sweep
    assert first.outside_run == 1

    consumed = replace(
        inventory,
        lifecycle=LiquidityInventoryLifecycle.CONSUMED,
        consumed_at=sweep.end,
        lifecycle_reason="pool_swept",
    )
    later = _m1(16, close=101.25, high=101.5, low=100.0)
    resolved = tracker.on_completed_update(
        later,
        prior_inventory=(consumed,),
        liquidity_pools=(),
    ).manipulations[-1]
    assert resolved.lifecycle is ManipulationLifecycle.ACCEPTED_OUTSIDE
    assert resolved.accepted_outside_at == later.end
    assert resolved.outside_run == 2
    assert resolved.resolved_side == "above"
    assert resolved.sweep_extreme == first.sweep_extreme


def test_pool_reentry_failure_resets_then_deadline_censors_unresolved() -> None:
    tracker = CausalGroup4Tracker(_protocol())
    _warm_m1(tracker)
    pool, inventory = _pool(
        "above",
        confirmed_at=M1_BASE - pd.Timedelta(minutes=1),
        identity="deadline-above",
    )
    sweep = _m1(15, close=100.5, high=101.25, low=100.0)
    swept = tracker.on_completed_update(
        sweep,
        prior_inventory=(inventory,),
        liquidity_pools=(pool,),
    ).manipulations[-1]
    assert swept.deadline_at is None
    consumed = replace(
        inventory,
        lifecycle=LiquidityInventoryLifecycle.CONSUMED,
        consumed_at=sweep.end,
        lifecycle_reason="pool_swept",
    )

    candidate = tracker.on_completed_update(
        _m1(16, close=100.5, high=100.75, low=100.0),
        prior_inventory=(consumed,),
        liquidity_pools=(),
    ).manipulations[-1]
    assert candidate.reentry_candidate_at == _m1(16).end

    failed = tracker.on_completed_update(
        _m1(17, close=101.25, high=101.5, low=100.0),
        prior_inventory=(consumed,),
        liquidity_pools=(),
    ).manipulations[-1]
    assert failed.lifecycle is ManipulationLifecycle.SWEPT
    assert failed.reentry_candidate_at is None
    assert failed.reentry_failed_at == _m1(17).end
    assert failed.outside_run == 1

    tracker.on_completed_update(
        _m1(18, close=100.5, high=100.75, low=100.0),
        prior_inventory=(consumed,),
        liquidity_pools=(),
    )
    tracker.on_completed_update(
        _m1(19, close=101.25, high=101.5, low=100.0),
        prior_inventory=(consumed,),
        liquidity_pools=(),
    )
    censored = tracker.on_completed_update(
        _m1(20, close=100.5, high=100.75, low=100.0),
        prior_inventory=(consumed,),
        liquidity_pools=(),
    ).manipulations[-1]
    assert censored.lifecycle is ManipulationLifecycle.SWEPT
    assert censored.deadline_elapsed
    assert censored.censored_at == _m1(20).end
    assert censored.deadline_at == censored.censored_at
    assert censored.age_1m_bars == 5
    assert censored.sweep_extreme == swept.sweep_extreme


def test_manipulation_candidate_and_failure_enter_event_memory() -> None:
    tracker = CausalGroup4Tracker(_protocol())
    _warm_m1(tracker)
    pool, inventory = _pool(
        "above",
        confirmed_at=M1_BASE - pd.Timedelta(minutes=1),
        identity="memory-revision",
    )
    swept_update = tracker.on_completed_update(
        _m1(15, close=100.5, high=101.25, low=100.0),
        prior_inventory=(inventory,),
        liquidity_pools=(pool,),
    )
    swept = swept_update.manipulations[-1]
    consumed = replace(
        inventory,
        lifecycle=LiquidityInventoryLifecycle.CONSUMED,
        consumed_at=_m1(15).end,
        lifecycle_reason="pool_swept",
    )
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
    observer._prior = SimpleNamespace(manipulations=(swept,))

    candidate_update = tracker.on_completed_update(
        _m1(16, close=100.5, high=100.75, low=100.0),
        prior_inventory=(consumed,),
        liquidity_pools=(),
    )
    observer._record_group4_events(
        candidate_update,
        include_ranges=False,
    )
    candidate = candidate_update.manipulations[-1]
    observer._prior = SimpleNamespace(manipulations=(candidate,))

    failed_update = tracker.on_completed_update(
        _m1(17, close=101.25, high=101.5, low=100.0),
        prior_inventory=(consumed,),
        liquidity_pools=(),
    )
    observer._record_group4_events(
        failed_update,
        include_ranges=False,
    )

    revisions = tuple(
        event
        for event in observer.memory.recent()
        if event.kind is EventKind.MANIPULATION_STATE
        and event.details.get("state_revision") is True
    )
    assert [event.transition_reason for event in revisions] == [
        "reentry_candidate_started",
        "reentry_candidate_failed",
    ]
    assert revisions[0].details["reentry_candidate_at"] == (
        _m1(16).end.isoformat()
    )
    assert revisions[1].details["reentry_failed_at"] == (
        _m1(17).end.isoformat()
    )


def test_native_swept_pool_is_passed_to_group4_as_frozen_formation() -> None:
    pool, inventory = _pool(
        "above",
        confirmed_at=M1_BASE - pd.Timedelta(minutes=1),
        identity="native-sweep-formation",
    )
    sweep_bar = _m1(15, close=101.25, high=101.5, low=100.0)
    swept = replace(
        pool,
        lifecycle=LiquidityPoolLifecycle.SWEPT,
        swept_at=sweep_bar.end,
        sweep_extreme=sweep_bar.high,
        close_outside_on_sweep=True,
    )

    formation = CausalObserver._pool_formation_source(swept)

    assert formation.lifecycle is LiquidityPoolLifecycle.FORMED
    assert formation.swept_at is None
    assert formation.sweep_extreme is None
    assert formation.close_outside_on_sweep is None
    assert CausalGroup4Tracker(_protocol())._pool_by_inventory(
        inventory,
        (formation,),
    ) == formation


def test_dual_side_sweep_is_ambiguous_and_creates_no_manipulation() -> None:
    tracker = CausalGroup4Tracker(_protocol())
    _warm_m1(tracker)
    confirmed_at = M1_BASE - pd.Timedelta(minutes=1)
    upper_pool, upper_item = _pool(
        "above",
        confirmed_at=confirmed_at,
        identity="dual-above",
    )
    lower_pool, lower_item = _pool(
        "below",
        confirmed_at=confirmed_at,
        identity="dual-below",
    )

    output = tracker.on_completed_update(
        _m1(
            15,
            close=100.0,
            high=101.25,
            low=98.75,
        ),
        prior_inventory=(upper_item, lower_item),
        liquidity_pools=(upper_pool, lower_pool),
    )

    assert output.manipulations == ()
    assert output.manipulation_transitions == ()
    assert output.ambiguous_sweep_item_ids == tuple(
        sorted((upper_item.item_id, lower_item.item_id))
    )


def test_range_inventory_sweep_and_hard_boundary_preserve_typed_terminals() -> None:
    protocol = _protocol()
    tracker, _, mature = _mature_range(protocol)
    assert mature.mature_at is not None
    base = mature.mature_at + pd.Timedelta(minutes=1)
    inventory = tracker.snapshot().range_boundary_inventory
    _warm_m1(
        tracker,
        base=base,
        inventory=inventory,
    )
    sweep = _m1(
        15,
        base=base,
        close=100.5,
        high=101.25,
        low=99.75,
    )
    swept_output = tracker.on_completed_update(
        sweep,
        prior_inventory=tracker.snapshot().range_boundary_inventory,
        liquidity_pools=(),
    )
    swept = swept_output.manipulations[-1]
    assert swept.lifecycle is ManipulationLifecycle.SWEPT
    assert swept.source_kind == "mature_range_boundary"
    assert swept.source_id == mature.range_id
    assert {
        item.side: item.lifecycle
        for item in swept_output.range_boundary_inventory
    } == {
        "above": LiquidityInventoryLifecycle.CONSUMED,
        "below": LiquidityInventoryLifecycle.VISIBLE,
    }

    boundary_at = sweep.end + pd.Timedelta(minutes=1)
    boundary = tracker.on_boundary("data_gap_reset", boundary_at)
    assert boundary.boundary_reason == "data_gap_reset"
    assert boundary.dealing_ranges == ()
    assert boundary.manipulations == ()
    assert len(boundary.range_transitions) == 1
    terminal_range = boundary.range_transitions[0]
    assert terminal_range.lifecycle is DealingRangeLifecycle.BROKEN
    assert terminal_range.broken_at == boundary_at
    assert terminal_range.transition_reason == "data_gap_reset"
    assert _frozen_range_geometry(terminal_range) == _frozen_range_geometry(
        mature
    )
    assert len(boundary.manipulation_transitions) == 1
    censored = boundary.manipulation_transitions[0]
    assert censored.lifecycle is ManipulationLifecycle.SWEPT
    assert censored.state_started_at == swept.swept_at
    assert censored.censored_at == boundary_at
    assert censored.last_updated_at == boundary_at
    assert censored.transition_reason == "data_gap_reset"
    assert tracker.on_boundary("data_gap_reset", boundary_at) is boundary


def test_exact_retry_is_cached_and_same_clock_or_older_input_fails() -> None:
    candle = _m1(1)
    tracker = CausalGroup4Tracker(_protocol())
    first = tracker.on_completed_update(
        candle,
        prior_inventory=(),
        liquidity_pools=(),
    )
    retry = tracker.on_completed_update(
        candle,
        prior_inventory=(),
        liquidity_pools=(),
    )
    assert retry is first

    conflicting = replace(candle, close=100.1, high=100.25)
    with pytest.raises(
        ValueError,
        match="duplicate or out-of-order Group 4 1m candle",
    ):
        tracker.on_completed_update(
            conflicting,
            prior_inventory=(),
            liquidity_pools=(),
        )

    boundary_tracker = CausalGroup4Tracker(_protocol())
    boundary_tracker.on_completed_update(
        candle,
        prior_inventory=(),
        liquidity_pools=(),
    )
    boundary_at = candle.end + pd.Timedelta(minutes=1)
    boundary_tracker.on_boundary("data_gap_reset", boundary_at)
    with pytest.raises(
        ValueError,
        match="boundary is duplicate or out of order",
    ):
        boundary_tracker.on_boundary(
            "data_anomaly",
            boundary_at,
        )
    recovered = tracker.on_completed_update(
        _m1(2),
        prior_inventory=(),
        liquidity_pools=(),
    )
    assert recovered.manipulations == ()

    older_tracker = CausalGroup4Tracker(_protocol())
    older_tracker.on_completed_update(
        candle,
        prior_inventory=(),
        liquidity_pools=(),
    )
    with pytest.raises(
        ValueError,
        match="duplicate or out-of-order Group 4 1m candle",
    ):
        older_tracker.on_completed_update(
            _m1(0),
            prior_inventory=(),
            liquidity_pools=(),
        )


def test_cold_sweep_without_prior_atr_is_unclassified_and_advances() -> None:
    tracker, _, mature = _mature_range(_protocol())
    assert mature.mature_at is not None
    base = mature.mature_at + pd.Timedelta(minutes=1)
    inventory = tracker.snapshot().range_boundary_inventory
    tracker.on_completed_update(
        _m1(0, base=base),
        prior_inventory=inventory,
        liquidity_pools=(),
    )
    sweep = _m1(
        1,
        base=base,
        close=100.5,
        high=101.25,
        low=99.75,
    )
    crossed_ids = {
        item.item_id
        for item in inventory
        if (
            sweep.high > item.upper_bound
            if item.side == "above"
            else sweep.low < item.lower_bound
        )
    }
    output = tracker.on_completed_update(
        sweep,
        prior_inventory=inventory,
        liquidity_pools=(),
    )

    assert output.manipulations == ()
    assert output.ambiguous_sweep_item_ids == ()
    assert set(output.atr_unready_sweep_item_ids) == crossed_ids
    assert tracker.last_m1_end == sweep.end
    retained = {
        item.item_id: item
        for item in tracker.snapshot().range_boundary_inventory
    }
    assert all(
        retained[item_id].lifecycle
        is LiquidityInventoryLifecycle.CONSUMED
        for item_id in crossed_ids
    )
    next_bar = _m1(2, base=base)
    tracker.on_completed_update(
        next_bar,
        prior_inventory=tracker.snapshot().range_boundary_inventory,
        liquidity_pools=(),
    )
    assert tracker.last_m1_end == next_bar.end


def test_h1_synthetic_bar_advances_only_the_raw_causal_cutoff() -> None:
    tracker = CausalGroup4Tracker(_protocol())
    synthetic = _h1(0, synthetic=True)
    before = tracker.snapshot()

    output = tracker.on_completed_h1(synthetic, ())

    assert output == before
    assert tracker.snapshot() == before
    assert tracker.last_h1_end is None
    assert tracker.on_completed_h1(synthetic, ()) is output
    with pytest.raises(
        ValueError,
        match="duplicate or out-of-order Group 4 H1 candle",
    ):
        tracker.on_completed_h1(_h1(0), ())
    real = _h1(1)
    tracker.on_completed_h1(real, ())
    assert tracker.last_h1_end == real.end


def test_session_aware_completed_h1_does_not_require_60_observed_minutes() -> None:
    tracker = CausalGroup4Tracker(_protocol())
    shortened = replace(
        _h1(0),
        observed_minutes=45,
        expected_minutes=45,
        real_minutes=45,
    )

    tracker.on_completed_h1(shortened, ())

    assert tracker.last_h1_end == shortened.end


def test_synthetic_completed_bar_is_a_semantic_noop() -> None:
    tracker = CausalGroup4Tracker(_protocol())
    before = tracker.snapshot()
    synthetic = _m1(0, synthetic=True)

    output = tracker.on_completed_update(
        synthetic,
        prior_inventory=(),
        liquidity_pools=(),
    )

    assert output == before
    assert tracker.snapshot() == before
    assert tracker.last_m1_end is None
    assert (
        tracker.on_completed_update(
            synthetic,
            prior_inventory=(),
            liquidity_pools=(),
        )
        is output
    )
    with pytest.raises(
        ValueError,
        match="duplicate or out-of-order Group 4 1m candle",
    ):
        tracker.on_completed_update(
            _m1(0),
            prior_inventory=(),
            liquidity_pools=(),
        )
    real = _m1(1)
    tracker.on_completed_update(
        real,
        prior_inventory=(),
        liquidity_pools=(),
    )
    assert tracker.last_m1_end == real.end


def test_cold_prefix_trailing_synthetic_bar_keeps_raw_cutoff() -> None:
    tracker = CausalGroup4Tracker(_protocol())
    output = tracker.bootstrap_completed_1m_prefix(
        (_m1(0), _m1(1, synthetic=True)),
        pool_inventory=(),
        liquidity_pools=(),
    )

    assert output.manipulations == ()
    assert tracker.last_m1_end == _m1(0).end
    with pytest.raises(
        ValueError,
        match="duplicate or out-of-order Group 4 1m candle",
    ):
        tracker.on_completed_update(
            _m1(1),
            prior_inventory=(),
            liquidity_pools=(),
        )
    later = _m1(2)
    tracker.on_completed_update(
        later,
        prior_inventory=(),
        liquidity_pools=(),
    )
    assert tracker.last_m1_end == later.end


def test_cold_prefix_reconstructs_first_sweep_and_next_bar_resolution() -> None:
    protocol = _protocol()
    pool, inventory = _pool(
        "above",
        confirmed_at=M1_BASE,
        identity="cold-above",
    )
    predecessor = _m1(-1)
    prefix = (predecessor,) + tuple(
        _m1(index)
        for index in range(15)
    ) + (
        _m1(
            15,
            close=100.0,
            high=101.25,
            low=99.75,
        ),
        _m1(
            16,
            close=100.0,
            high=100.5,
            low=99.75,
        ),
        _m1(
            17,
            close=100.0,
            high=100.5,
            low=99.75,
        ),
    )

    output = CausalGroup4Tracker(
        protocol
    ).bootstrap_completed_1m_prefix(
        prefix,
        pool_inventory=(inventory,),
        liquidity_pools=(pool,),
    )

    final = output.manipulations[-1]
    assert final.lifecycle is ManipulationLifecycle.REACCEPTED
    assert final.swept_at == _m1(15).end
    assert final.resolved_at == _m1(17).end
    assert [
        state.lifecycle
        for state in output.manipulation_transitions
    ] == [
        ManipulationLifecycle.SWEPT,
        ManipulationLifecycle.REACCEPTED,
    ]


def test_cold_prefix_starting_after_source_confirmation_fails_closed() -> None:
    pool, inventory = _pool(
        "above",
        confirmed_at=M1_BASE,
        identity="late-prefix",
    )

    with pytest.raises(
        ValueError,
        match="cold prefix lacks a real source predecessor",
    ):
        CausalGroup4Tracker(
            _protocol()
        ).bootstrap_completed_1m_prefix(
            (_m1(1),),
            pool_inventory=(inventory,),
            liquidity_pools=(pool,),
        )


def test_exact_source_bindings_fail_closed_without_partial_commit() -> None:
    pool_tracker = CausalGroup4Tracker(_protocol())
    _warm_m1(pool_tracker)
    pool, inventory = _pool(
        "above",
        confirmed_at=M1_BASE - pd.Timedelta(minutes=1),
        identity="malformed-pool",
    )
    malformed_pool_item = replace(
        inventory,
        source_ids=("wrong:1", "wrong:2"),
    )
    before_pool = pool_tracker.snapshot()
    with pytest.raises(
        ValueError,
        match="pool inventory and source geometry disagree",
    ):
        pool_tracker.on_completed_update(
            _m1(
                15,
                close=100.0,
                high=101.25,
                low=99.75,
            ),
            prior_inventory=(malformed_pool_item,),
            liquidity_pools=(pool,),
        )
    assert pool_tracker.snapshot() == before_pool

    range_tracker, _, mature = _mature_range(_protocol())
    assert mature.mature_at is not None
    base = mature.mature_at + pd.Timedelta(minutes=1)
    range_inventory = range_tracker.snapshot().range_boundary_inventory
    _warm_m1(
        range_tracker,
        base=base,
        inventory=range_inventory,
    )
    malformed_range_item = replace(
        next(item for item in range_inventory if item.side == "above"),
        strength=0.1,
    )
    before_range = range_tracker.snapshot()
    with pytest.raises(
        ValueError,
        match="range inventory differs from retained source",
    ):
        range_tracker.on_completed_update(
            _m1(
                15,
                base=base,
                close=100.5,
                high=101.25,
                low=99.75,
            ),
            prior_inventory=(malformed_range_item,),
            liquidity_pools=(),
        )
    assert range_tracker.snapshot() == before_range


def test_cold_prefix_validates_every_retained_pool_before_replay() -> None:
    _, inventory = _pool(
        "above",
        confirmed_at=M1_BASE,
        identity="missing-cold-pool",
    )
    tracker = CausalGroup4Tracker(_protocol())

    with pytest.raises(
        ValueError,
        match="cold source lacks its exact pool state",
    ):
        tracker.bootstrap_completed_1m_prefix(
            (_m1(-1), _m1(0)),
            pool_inventory=(inventory,),
            liquidity_pools=(),
        )
    assert tracker.snapshot().manipulations == ()


def test_group4_requires_the_typed_h1_structure_source() -> None:
    with pytest.raises(
        ValueError,
        match="requires the confirmed-swing tracker",
    ):
        CausalObserver(
            ObserverConfig(
                scale_specs=CORE_TEST_SCALE_SPECS,
                liquidity_protocol=(
                    "configs/primitives_structure_liquidity.json"
                ),
                group4_protocol=str(PROTOCOL_PATH),
            )
        )


def test_range_boundaries_join_the_visible_liquidity_route_inventory() -> None:
    tracker, _, mature = _mature_range(_protocol())
    assert mature.mature_at is not None
    pool, pool_item = _pool(
        "above",
        confirmed_at=mature.mature_at,
        identity="canonical-draw",
    )
    observation = SimpleNamespace(
        liquidity_inventory=(
            *tracker.snapshot().range_boundary_inventory,
            pool_item,
        ),
        asof=mature.last_updated_at,
    )

    expected_ids = {
        item.item_id for item in observation.liquidity_inventory
    }
    assert {
        level.level_id for level in _visible_levels(observation)
    } == expected_ids
    assert _visible_level_ids(observation) == expected_ids


def test_same_clock_event_sequence_matches_the_frozen_causal_order() -> None:
    pool, inventory = _pool(
        "above",
        confirmed_at=M1_BASE,
        identity="event-order",
    )
    prefix = (
        _m1(-1),
        *(_m1(index) for index in range(15)),
        _m1(
            15,
            close=100.0,
            high=101.25,
            low=99.75,
        ),
        _m1(
            16,
            close=100.0,
            high=100.5,
            low=99.75,
        ),
        _m1(
            17,
            close=100.0,
            high=100.5,
            low=99.75,
        ),
    )
    update = CausalGroup4Tracker(
        _protocol()
    ).bootstrap_completed_1m_prefix(
        prefix,
        pool_inventory=(inventory,),
        liquidity_pools=(pool,),
    )
    swept, resolved = update.manipulation_transitions
    assert resolved.resolved_at is not None
    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
    observer._record_group4_events(
        update,
        include_ranges=False,
    )
    observer.memory.append(
        _event(
            EventKind.LIQUIDITY_SWEEP,
            swept.swept_at,
            Timeframe.M1,
            swept.side,
            swept.sweep_extreme,
            swept.strength,
            (inventory.item_id,),
        )
    )
    observer.memory.append(
        _event(
            EventKind.STRUCTURE_BREAK,
            swept.swept_at,
            Timeframe.H1,
            swept.side,
            swept.sweep_extreme,
            swept.strength,
            ("h1-source",),
        )
    )
    _, _, mature = _mature_range(_protocol())
    same_clock_range = replace(
        mature,
        mature_at=swept.swept_at,
        state_started_at=swept.swept_at,
        last_updated_at=swept.swept_at,
    )
    observer._record_group4_events(
        Group4Update(
            dealing_ranges=(same_clock_range,),
            manipulations=(),
            range_boundary_inventory=(),
            range_transitions=(same_clock_range,),
        ),
        include_resolutions=False,
        include_creations=False,
    )
    observer.memory.append(
        _event(
            EventKind.LIQUIDITY_CONSUMED,
            resolved.resolved_at,
            Timeframe.M1,
            resolved.side,
            resolved.reentry_price,
            resolved.strength,
            (inventory.item_id,),
        )
    )
    observer.memory.sync_retained_entity_timelines(
        {
            f"manipulation:{swept.manipulation_id}",
            f"range:{same_clock_range.range_id}",
        },
        asof=resolved.resolved_at,
    )
    events_by_id = {
        event.event_id: event
        for event in (
            *observer.memory.recent(),
            *(
                event
                for timeline in (
                    observer.memory.entity_timelines().values()
                )
                for event in timeline
            ),
        )
    }
    ordered = sorted(
        events_by_id.values(),
        key=lambda event: (
            event.observed_at,
            event.sequence_no,
            event.event_id,
        ),
    )

    assert [
        event.kind
        for event in ordered
        if event.observed_at == swept.swept_at
    ] == [
        EventKind.LIQUIDITY_SWEEP,
        EventKind.STRUCTURE_BREAK,
        EventKind.DEALING_RANGE_STATE,
        EventKind.MANIPULATION_STATE,
    ]
    assert [
        event.kind
        for event in ordered
        if event.observed_at == resolved.resolved_at
    ] == [
        EventKind.MANIPULATION_STATE,
        EventKind.LIQUIDITY_CONSUMED,
    ]
