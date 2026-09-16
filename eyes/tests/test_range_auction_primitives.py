from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from eyes.core.causal import CausalMarketReader
from eyes.core.range_auction import (
    CausalRangeAuctionTracker,
    RangeAuctionProtocol,
    RangeAuctionUpdate,
)
from contract.market import (
    Bar,
    Candle,
    Timeframe,
)
from contract.eye import (
    BALANCE_CLAIM_ABANDONED,
    DealingRangeLifecycle,
    DealingRangeState,
    EventKind,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityPoolLifecycle,
    LiquidityPoolState,
    ManipulationLifecycle,
    ManipulationSourceDispositionKind,
    RANGE_MATURITY_GATE_NAMES,
    RANGE_PAIR_FUNNEL_COUNTS,
    SupportResistanceLifecycle,
    SupportResistanceState,
)
from eyes.core.observation import (
    CausalObserver,
    ObserverConfig,
)
from eyes.core.semantic_event_emitter import _event
from shares.core.scene_graph import current_dol_inventory
from brain.core.eye_view import visible_liquidity_ids as _visible_level_ids

from shares.tests.helpers import CORE_TEST_SCALE_SPECS, market_observation


ROOT = Path(__file__).resolve().parents[2]
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


def _protocol() -> RangeAuctionProtocol:
    return RangeAuctionProtocol.from_file(PROTOCOL_PATH)


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
    tracker: CausalRangeAuctionTracker,
    count: int = 14,
) -> None:
    for index in range(count):
        output = tracker.on_completed_h1(_h1(index), ())
        assert output.dealing_ranges == ()


def test_range_candidates_ignore_nonstructural_support_resistance() -> None:
    tracker = CausalRangeAuctionTracker(_protocol())
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
    protocol: RangeAuctionProtocol,
) -> tuple[
    CausalRangeAuctionTracker,
    DealingRangeState,
    DealingRangeState,
]:
    tracker = CausalRangeAuctionTracker(protocol)
    zones = (_zone("support"), _zone("resistance"))
    _warm_h1(tracker)
    formed = tracker.on_completed_h1(
        _h1(14, close=99.75, span=1.0),
        zones,
    ).dealing_ranges[-1]
    output = None
    # Maturity now needs two distinct price tests of *each* boundary, so the
    # candidate visits the lower band on bars 16 and 18 and the upper band on
    # bars 15 and 17.  The late bars stay narrow, which is what the compression
    # gate reads.
    for index, close, span in (
        (15, 100.25, 1.0),
        (16, 99.75, 1.0),
        (17, 100.25, 1.0),
        (18, 99.75, 1.0),
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
    tracker: CausalRangeAuctionTracker,
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


def _source_dispositions(
    output: RangeAuctionUpdate,
) -> dict[str, ManipulationSourceDispositionKind]:
    result = {
        item.source_inventory_item_id: item.disposition
        for item in output.source_dispositions
    }
    assert len(result) == len(output.source_dispositions)
    return result


def _range_funnel(output: RangeAuctionUpdate):
    assert len(output.range_funnel) == 1
    return output.range_funnel[0]


def test_range_auction_protocol_tracks_current_config_and_upstream_binding() -> None:
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

    assert formed.lifecycle is DealingRangeLifecycle.ACTIVE
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

    assert mature.lifecycle is DealingRangeLifecycle.ACTIVE
    assert mature.transition_reason == "balance_claim_confirmed"
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


def test_range_funnel_conserves_pair_selection_and_actual_gate_margins() -> None:
    tracker = CausalRangeAuctionTracker(_protocol())
    zones = (_zone("support"), _zone("resistance"))
    _warm_h1(tracker)

    formed_output = tracker.on_completed_h1(
        _h1(14, close=99.75, span=1.0),
        zones,
    )
    formed = formed_output.dealing_ranges[-1]
    funnel = _range_funnel(formed_output)
    assert tuple(name for name, _ in funnel.pair_counts) == (
        RANGE_PAIR_FUNNEL_COUNTS
    )
    assert dict(funnel.pair_counts) == {
        "live_structural_pairs": 1,
        "invalid_geometry_pairs": 0,
        "geometry_valid_pairs": 1,
        "close_outside_pair_pairs": 0,
        "already_admitted_pairs": 0,
        "cold_start_blocked_pairs": 0,
        "same_bar_terminal_blocked_pairs": 0,
        "live_range_blocked_pairs": 0,
        "atr_unready_pairs": 0,
        "eligible_pairs": 1,
        "forming_selected": 1,
    }
    assert funnel.selected_source_pair_ids == (
        zones[0].zone_id,
        zones[1].zone_id,
    )
    assert funnel.selected_range_id == formed.range_id
    assert funnel.maturity_range_id is None
    assert funnel.maturity_gates == ()

    continued_output = tracker.on_completed_h1(
        _h1(15, close=100.25, span=1.0),
        zones,
    )
    continued = continued_output.dealing_ranges[-1]
    gate = _range_funnel(continued_output)
    assert gate.selected_source_pair_ids is None
    assert dict(gate.pair_counts)["already_admitted_pairs"] == 1
    assert gate.maturity_range_id == formed.range_id
    assert tuple(name for name, *_ in gate.maturity_gates) == (
        RANGE_MATURITY_GATE_NAMES
    )
    rows = {
        name: (actual, threshold, margin)
        for name, actual, threshold, margin in gate.maturity_gates
    }
    assert rows["duration"] == pytest.approx((2.0, 8.0, -0.75))
    # This bar reached the upper band and nowhere near the lower one, so the
    # two-sided standard reads the side that has nothing: min(0, 1).
    assert rows["bilateral_price_tests"] == pytest.approx((0.0, 2.0, -1.0))
    assert continued.balance_upper_test_generations == 1
    assert continued.balance_lower_test_generations == 0
    assert gate.unmet_maturity_gates == tuple(
        name
        for name in RANGE_MATURITY_GATE_NAMES
        if rows[name][2] < 0.0
    )
    assert continued.lifecycle is DealingRangeLifecycle.ACTIVE


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


def test_the_forming_deadline_ends_the_balance_claim_not_the_interval() -> None:
    """v1.3: running out of room to balance costs the balance claim only.

    The structural interval is still an interval and still locates price, so
    it stays live and is ended only by a close outside it.
    """

    tracker = CausalRangeAuctionTracker(_protocol())
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
    # Abandoning the balance claim is not a lifecycle transition of the range
    # entity: it is still FORMING, and FORMING was already recorded when the
    # range was created.  Publishing it again makes the typed entity timeline
    # record one lifecycle twice, which real data reaches within days.
    assert output.range_transitions == ()
    abandoned = output.dealing_ranges[-1]
    assert abandoned.lifecycle is DealingRangeLifecycle.ACTIVE
    assert abandoned.transition_reason == BALANCE_CLAIM_ABANDONED
    assert abandoned.candidate_real_h1_bars == 24
    assert abandoned.age_h1_bars == 23
    assert abandoned.broken_at is None
    assert abandoned.balance_confirmed_at is None
    # The interval still knows where it is.
    assert abandoned.lower_bound < abandoned.midpoint < abandoned.upper_bound


def test_cold_existing_range_pair_waits_for_a_new_source_identity() -> None:
    tracker = CausalRangeAuctionTracker(_protocol())
    zones = (_zone("support"), _zone("resistance"))
    _warm_h1(tracker)
    tracker.mark_existing_source_pairs_ineligible(zones)

    blocked = tracker.on_completed_h1(
        _h1(14, close=100.0),
        zones,
    )
    assert blocked.dealing_ranges == ()
    blocked_funnel = _range_funnel(blocked)
    assert dict(blocked_funnel.pair_counts)[
        "cold_start_blocked_pairs"
    ] == 1
    assert dict(blocked_funnel.pair_counts)["forming_selected"] == 0
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
    assert formed.lifecycle is DealingRangeLifecycle.ACTIVE
    assert formed.lower_source_zone_id == fresh_support.zone_id


def test_range_funnel_counts_new_pair_blocked_by_existing_live_range() -> None:
    tracker = CausalRangeAuctionTracker(_protocol())
    support = _zone("support")
    resistance = _zone("resistance")
    _warm_h1(tracker)
    tracker.on_completed_h1(
        _h1(14, close=99.75, span=1.0),
        (support, resistance),
    )
    fresh_support = replace(
        support,
        zone_id="zone:support:fresh-live-block",
    )

    output = tracker.on_completed_h1(
        _h1(15, close=100.25, span=1.0),
        (support, fresh_support, resistance),
    )

    counts = dict(_range_funnel(output).pair_counts)
    assert counts["live_structural_pairs"] == 2
    assert counts["geometry_valid_pairs"] == 2
    assert counts["already_admitted_pairs"] == 1
    assert counts["live_range_blocked_pairs"] == 1
    assert counts["forming_selected"] == 0


def test_pool_source_must_exist_before_bar_then_reentry_must_hold() -> None:
    tracker = CausalRangeAuctionTracker(_protocol())
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
    tracker = CausalRangeAuctionTracker(_protocol())
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
    tracker = CausalRangeAuctionTracker(_protocol())
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
    tracker = CausalRangeAuctionTracker(_protocol())
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
    observer._emitter._record_group4_events(
        candidate_update,
        include_ranges=False,
            prior=observer._prior,
    )
    candidate = candidate_update.manipulations[-1]
    observer._prior = SimpleNamespace(manipulations=(candidate,))

    failed_update = tracker.on_completed_update(
        _m1(17, close=101.25, high=101.5, low=100.0),
        prior_inventory=(consumed,),
        liquidity_pools=(),
    )
    observer._emitter._record_group4_events(
        failed_update,
        include_ranges=False,
            prior=observer._prior,
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
    assert CausalRangeAuctionTracker(_protocol())._pool_by_inventory(
        inventory,
        (formation,),
    ) == formation


def test_dual_side_sweep_is_ambiguous_and_creates_no_manipulation() -> None:
    tracker = CausalRangeAuctionTracker(_protocol())
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
    assert _source_dispositions(output) == {
        upper_item.item_id: (
            ManipulationSourceDispositionKind.AMBIGUOUS_DUAL_SIDE
        ),
        lower_item.item_id: (
            ManipulationSourceDispositionKind.AMBIGUOUS_DUAL_SIDE
        ),
    }


def test_raw_crossed_sources_reject_missing_prior_or_stale_source() -> None:
    confirmed_at = M1_BASE - pd.Timedelta(minutes=1)
    pool, item = _pool(
        "above",
        confirmed_at=confirmed_at,
        identity="cold-prior",
    )
    cold = CausalRangeAuctionTracker(_protocol()).on_completed_update(
        _m1(0, close=100.0, high=101.25),
        prior_inventory=(item,),
        liquidity_pools=(pool,),
    )
    assert cold.manipulations == ()
    assert _source_dispositions(cold) == {
        item.item_id: (
            ManipulationSourceDispositionKind.REJECTED_PRIOR_CLOSE
        )
    }

    stale_tracker = CausalRangeAuctionTracker(_protocol())
    _warm_m1(stale_tracker)
    stale = stale_tracker.on_completed_update(
        _m1(15, close=100.0, high=101.25),
        prior_inventory=(item,),
        liquidity_pools=(),
    )
    assert stale.manipulations == ()
    assert _source_dispositions(stale) == {
        item.item_id: (
            ManipulationSourceDispositionKind.REJECTED_SOURCE_MISSING_OR_STALE
        )
    }


def test_duplicate_source_identity_is_rejected_before_partial_commit() -> None:
    confirmed_at = M1_BASE - pd.Timedelta(minutes=1)
    pool, item = _pool(
        "above",
        confirmed_at=confirmed_at,
        identity="duplicate-source",
    )
    tracker = CausalRangeAuctionTracker(_protocol())
    _warm_m1(tracker)
    last_end = tracker.last_m1_end

    with pytest.raises(
        ValueError,
        match="inventory source identity repeats",
    ):
        tracker.on_completed_update(
            _m1(15, close=100.0, high=101.25),
            prior_inventory=(item, item),
            liquidity_pools=(pool,),
        )

    assert tracker.last_m1_end == last_end


def test_same_side_sources_are_conserved_as_primary_and_secondaries() -> None:
    confirmed_at = M1_BASE - pd.Timedelta(minutes=1)
    primary_pool, primary_item = _pool(
        "above",
        confirmed_at=confirmed_at,
        identity="a-primary",
    )
    coincident_pool, coincident_item = _pool(
        "above",
        confirmed_at=confirmed_at,
        identity="b-coincident",
    )
    far_pool, far_item = _pool(
        "above",
        confirmed_at=confirmed_at,
        identity="c-far",
    )
    far_pool = replace(
        far_pool,
        lower_bound=101.25,
        upper_bound=101.5,
        midpoint=101.375,
    )
    far_item = replace(
        far_item,
        price=101.5,
        lower_bound=101.25,
        upper_bound=101.5,
    )
    tracker = CausalRangeAuctionTracker(_protocol())
    _warm_m1(tracker)
    output = tracker.on_completed_update(
        _m1(15, close=100.0, high=102.0),
        prior_inventory=(
            primary_item,
            coincident_item,
            far_item,
        ),
        liquidity_pools=(
            primary_pool,
            coincident_pool,
            far_pool,
        ),
    )

    assert len(output.manipulation_transitions) == 1
    assert _source_dispositions(output) == {
        primary_item.item_id: (
            ManipulationSourceDispositionKind.SELECTED_PRIMARY
        ),
        coincident_item.item_id: (
            ManipulationSourceDispositionKind.ATTACHED_COINCIDENT_SECONDARY
        ),
        far_item.item_id: (
            ManipulationSourceDispositionKind.ATTACHED_SAME_SIDE_SECONDARY
        ),
    }
    assert len(output.source_dispositions) == 3


def test_existing_live_and_same_bar_resolution_block_new_sources() -> None:
    confirmed_at = M1_BASE - pd.Timedelta(minutes=1)
    live_pool, live_item = _pool(
        "above",
        confirmed_at=confirmed_at,
        identity="live-source",
    )
    pending_pool, pending_item = _pool(
        "above",
        confirmed_at=confirmed_at,
        identity="pending-source",
    )
    pending_pool = replace(
        pending_pool,
        lower_bound=101.25,
        upper_bound=101.5,
        midpoint=101.375,
    )
    pending_item = replace(
        pending_item,
        price=101.5,
        lower_bound=101.25,
        upper_bound=101.5,
    )
    resolved_pool, resolved_item = _pool(
        "above",
        confirmed_at=confirmed_at,
        identity="resolved-source",
    )
    resolved_pool = replace(
        resolved_pool,
        lower_bound=101.5,
        upper_bound=101.75,
        midpoint=101.625,
    )
    resolved_item = replace(
        resolved_item,
        price=101.75,
        lower_bound=101.5,
        upper_bound=101.75,
    )
    pools = (live_pool, pending_pool, resolved_pool)
    tracker = CausalRangeAuctionTracker(_protocol())
    _warm_m1(tracker)
    created = tracker.on_completed_update(
        _m1(15, close=101.25, high=101.5),
        prior_inventory=(live_item,),
        liquidity_pools=pools,
    )
    assert _source_dispositions(created) == {
        live_item.item_id: (
            ManipulationSourceDispositionKind.SELECTED_PRIMARY
        )
    }

    still_live = tracker.on_completed_update(
        _m1(16, close=100.0, high=101.75),
        prior_inventory=(pending_item,),
        liquidity_pools=pools,
    )
    assert _source_dispositions(still_live) == {
        pending_item.item_id: (
            ManipulationSourceDispositionKind.BLOCKED_EXISTING_LIVE
        )
    }

    resolved_same_bar = tracker.on_completed_update(
        _m1(17, close=100.0, high=102.0),
        prior_inventory=(resolved_item,),
        liquidity_pools=pools,
    )
    assert len(resolved_same_bar.manipulation_transitions) == 1
    assert (
        resolved_same_bar.manipulation_transitions[0].lifecycle
        is ManipulationLifecycle.REACCEPTED
    )
    assert _source_dispositions(resolved_same_bar) == {
        resolved_item.item_id: (
            ManipulationSourceDispositionKind.BLOCKED_LIVE_RESOLVED_SAME_BAR
        )
    }


def test_same_clock_range_invalidation_has_one_source_disposition() -> None:
    tracker, _, mature = _mature_range(_protocol())
    assert mature.balance_confirmed_at is not None
    completed_h1 = _h1(22, close=102.0, span=0.25)
    base = completed_h1.end - pd.Timedelta(minutes=16)
    inventory = tracker.snapshot().range_boundary_inventory
    _warm_m1(
        tracker,
        base=base,
        inventory=inventory,
    )
    upper = next(item for item in inventory if item.side == "above")
    output = tracker.on_completed_update(
        _m1(15, base=base, close=100.0, high=101.25),
        prior_inventory=tracker.snapshot().range_boundary_inventory,
        liquidity_pools=(),
        completed_h1=completed_h1,
        h1_support_resistance=(_zone("support"), _zone("resistance")),
    )

    assert output.manipulations == ()
    assert any(
        state.range_id == mature.range_id
        and state.lifecycle is DealingRangeLifecycle.BROKEN
        for state in output.range_transitions
    )
    assert _source_dispositions(output) == {
        upper.item_id: (
            ManipulationSourceDispositionKind.REJECTED_RANGE_INVALIDATED_SAME_CLOCK
        )
    }


def test_range_inventory_sweep_and_hard_boundary_preserve_typed_terminals() -> None:
    protocol = _protocol()
    tracker, _, mature = _mature_range(protocol)
    assert mature.balance_confirmed_at is not None
    base = mature.balance_confirmed_at + pd.Timedelta(minutes=1)
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


@pytest.mark.parametrize(
    (
        "boundary_reason",
        "boundary_anomaly",
        "boundary_symbol",
        "boundary_instrument_id",
        "data_gap_before_minutes",
        "prior_offset_minutes",
    ),
    (
        pytest.param(
            "contract_change_reset",
            "contract_change_history_reset",
            "NQM5",
            2,
            0,
            2,
            id="contract-reset",
        ),
        pytest.param(
            "data_gap_reset",
            "data_gap_history_reset",
            "NQH5",
            1,
            6,
            8,
            id="data-boundary",
        ),
    ),
)
def test_observer_records_group4_range_terminal_only_at_boundary_clock(
    boundary_reason: str,
    boundary_anomaly: str,
    boundary_symbol: str,
    boundary_instrument_id: int,
    data_gap_before_minutes: int,
    prior_offset_minutes: int,
) -> None:
    tracker, formed, mature = _mature_range(_protocol())
    assert mature.balance_confirmed_at is not None
    base = mature.balance_confirmed_at + pd.Timedelta(minutes=1)
    inventory = tracker.snapshot().range_boundary_inventory
    _warm_m1(tracker, base=base, inventory=inventory)
    sweep_bar = _m1(
        15,
        base=base,
        close=100.5,
        high=101.25,
        low=99.75,
    )
    swept_update = tracker.on_completed_update(
        sweep_bar,
        prior_inventory=inventory,
        liquidity_pools=(),
    )
    swept = swept_update.manipulations[-1]

    observer = CausalObserver(
        ObserverConfig(
            structure_protocol=str(GROUP12_PROTOCOL_PATH),
            liquidity_protocol=str(GROUP12_PROTOCOL_PATH),
            range_auction_protocol=str(PROTOCOL_PATH),
            scale_specs=CORE_TEST_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    observer._range_auction_tracker = tracker
    observer.memory.set_clock_coverage_start(formed.formed_at)
    # A Structural Range has exactly one non-terminal transition -- its
    # creation. Confirming the balance claim settles that claim and promotes
    # the boundaries without moving the range's lifecycle, so replaying the
    # confirmed state as a second transition would record `active` twice.
    observer._emitter._record_group4_events(
        RangeAuctionUpdate(
            dealing_ranges=(mature,),
            manipulations=(),
            range_boundary_inventory=inventory,
            range_transitions=(formed,),
        ),
        prior=observer._prior,
    )
    sweep_bar_event = observer._emitter._append_completed_bar_event(
        sweep_bar,
        atr=1.0,
        data_complete=True,
    )
    observer._emitter._record_group4_events(
        swept_update,
        include_ranges=False,
            prior=observer._prior,
    )
    range_key = f"range:{mature.range_id}"
    manipulation_key = f"manipulation:{swept.manipulation_id}"
    observer.memory.sync_retained_entity_timelines(
        {range_key, manipulation_key},
        asof=swept.swept_at,
    )
    pending_before_boundary = {
        event.event_id for event in observer.memory._audit_pending
    }
    assert pending_before_boundary

    boundary_at = swept.swept_at + pd.Timedelta(minutes=1)
    reader = CausalMarketReader(scale_specs=CORE_TEST_SCALE_SPECS)
    reader.on_bar(
        Bar(
            start=boundary_at
            - pd.Timedelta(minutes=prior_offset_minutes),
            open=100.0,
            high=100.25,
            low=99.75,
            close=100.0,
            volume=10.0,
            symbol="NQH5",
            instrument_id=1,
        )
    )
    boundary_update = reader.on_bar(
        Bar(
            start=boundary_at - pd.Timedelta(minutes=1),
            open=100.0,
            high=100.25,
            low=99.75,
            close=100.0,
            volume=10.0,
            symbol=boundary_symbol,
            instrument_id=boundary_instrument_id,
            data_gap_before_minutes=data_gap_before_minutes,
        )
    )
    assert boundary_update.anomalies == (boundary_anomaly,)

    boundary_observation = observer.observe(boundary_update)
    assert all(
        observer.audit_store.get(event_id) is not None
        for event_id in pending_before_boundary
    )
    candidate = next(
        event
        for event in observer.audit_store.events()
        if event.kind is EventKind.LIQUIDITY_LEVEL_CREATED
        and event.evidence.get("level_id")
        == swept.source_inventory_item_id
    )
    touch = next(
        event
        for event in observer.audit_store.events()
        if event.kind is EventKind.LEVEL_TOUCHED
        and event.timeframe is Timeframe.M1
        and event.evidence.get("level_id")
        == swept.source_inventory_item_id
    )
    penetration = next(
        event
        for event in observer.audit_store.events()
        if event.kind is EventKind.LEVEL_PENETRATED
        and event.timeframe is Timeframe.M1
        and event.evidence.get("level_id")
        == swept.source_inventory_item_id
    )
    assert touch.source_event_ids == (
        candidate.event_id,
        sweep_bar_event.event_id,
    )
    assert penetration.source_event_ids == (
        candidate.event_id,
        touch.event_id,
        sweep_bar_event.event_id,
    )
    assert penetration.evidence["source_kind"] == "mature_range_boundary"

    (terminal_range,) = (
        boundary_observation.group4_boundary_range_transitions
    )
    assert terminal_range.range_id == mature.range_id
    assert terminal_range.symbol == mature.symbol
    assert terminal_range.instrument_id == mature.instrument_id
    assert terminal_range.lifecycle is DealingRangeLifecycle.BROKEN
    assert terminal_range.broken_at == boundary_at
    assert terminal_range.transition_reason == boundary_reason
    timeline = boundary_observation.retained_entity_timelines[range_key]
    # Two states, not three: a Structural Range is created and is eventually
    # left. Settling the balance claim in between is not a state it passes
    # through.
    assert tuple(event.lifecycle for event in timeline) == (
        "active",
        "broken",
    )
    terminal_event = timeline[-1]
    assert terminal_event.kind is EventKind.DEALING_RANGE_STATE
    assert terminal_event.entity_id == terminal_range.range_id
    assert terminal_event.observed_at == boundary_at
    assert terminal_event.ended_at == boundary_at
    assert terminal_event.transition_reason == boundary_reason
    assert boundary_observation.incomplete_entity_timeline_keys == ()

    (censored_manipulation,) = (
        boundary_observation.group4_boundary_manipulation_transitions
    )
    assert censored_manipulation.manipulation_id == swept.manipulation_id
    assert censored_manipulation.censored_at == boundary_at
    assert censored_manipulation.transition_reason == boundary_reason
    assert manipulation_key not in (
        boundary_observation.retained_entity_timelines
    )

    next_observation = observer.observe(
        reader.on_bar(
            Bar(
                start=boundary_at,
                open=100.0,
                high=100.25,
                low=99.75,
                close=100.0,
                volume=10.0,
                symbol=boundary_symbol,
                instrument_id=boundary_instrument_id,
            )
        )
    )
    assert next_observation.group4_boundary_range_transitions == ()
    assert range_key not in next_observation.retained_entity_timelines
    assert observer.memory.timeline(range_key) == ()
    assert all(
        state.range_id != mature.range_id
        for state in next_observation.frame(Timeframe.H1).dealing_ranges
    )


def test_exact_retry_is_cached_and_same_clock_or_older_input_fails() -> None:
    candle = _m1(1)
    tracker = CausalRangeAuctionTracker(_protocol())
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

    boundary_tracker = CausalRangeAuctionTracker(_protocol())
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

    older_tracker = CausalRangeAuctionTracker(_protocol())
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
    assert mature.balance_confirmed_at is not None
    base = mature.balance_confirmed_at + pd.Timedelta(minutes=1)
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
    assert _source_dispositions(output) == {
        item_id: ManipulationSourceDispositionKind.ATR_UNREADY
        for item_id in crossed_ids
    }
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


def test_observer_publishes_mature_boundary_crossing_without_prior_atr() -> None:
    tracker, formed, mature = _mature_range(_protocol())
    assert mature.balance_confirmed_at is not None
    base = mature.balance_confirmed_at + pd.Timedelta(minutes=1)
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
    output = tracker.on_completed_update(
        sweep,
        prior_inventory=inventory,
        liquidity_pools=(),
    )
    assert output.manipulation_transitions == ()
    assert output.atr_unready_sweep_item_ids

    observer = CausalObserver(
        ObserverConfig(
            structure_protocol=str(GROUP12_PROTOCOL_PATH),
            liquidity_protocol=str(GROUP12_PROTOCOL_PATH),
            range_auction_protocol=str(PROTOCOL_PATH),
            scale_specs=CORE_TEST_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    observer.memory.set_clock_coverage_start(formed.formed_at)
    # A Structural Range has exactly one non-terminal transition -- its
    # creation. Confirming the balance claim settles that claim and promotes
    # the boundaries without moving the range's lifecycle, so replaying the
    # confirmed state as a second transition would record `active` twice.
    observer._emitter._record_group4_events(
        RangeAuctionUpdate(
            dealing_ranges=(mature,),
            manipulations=(),
            range_boundary_inventory=inventory,
            range_transitions=(formed,),
        ),
        prior=observer._prior,
    )
    crossing_bar = observer._emitter._append_completed_bar_event(
        sweep,
        atr=1.0,
        data_complete=True,
    )
    observer._emitter._record_group4_events(output, include_ranges=False, prior=observer._prior)
    observer.memory.flush_audit()

    events = observer.audit_store.events()
    assert not any(
        event.kind is EventKind.MANIPULATION_STATE
        for event in events
    )
    for item_id in output.atr_unready_sweep_item_ids:
        candidate = next(
            event
            for event in events
            if event.kind is EventKind.LIQUIDITY_LEVEL_CREATED
            and event.evidence.get("level_id") == item_id
        )
        touch = next(
            event
            for event in events
            if event.kind is EventKind.LEVEL_TOUCHED
            and event.evidence.get("level_id") == item_id
        )
        penetration = next(
            event
            for event in events
            if event.kind is EventKind.LEVEL_PENETRATED
            and event.evidence.get("level_id") == item_id
        )
        assert touch.source_event_ids == (
            candidate.event_id,
            crossing_bar.event_id,
        )
        assert penetration.source_event_ids == (
            candidate.event_id,
            touch.event_id,
            crossing_bar.event_id,
        )
        assert touch.evidence["range_id"] == mature.range_id
        assert penetration.evidence["range_id"] == mature.range_id


def test_h1_synthetic_bar_advances_only_the_raw_causal_cutoff() -> None:
    tracker = CausalRangeAuctionTracker(_protocol())
    synthetic = _h1(0, synthetic=True)
    before = tracker.snapshot()

    output = tracker.on_completed_h1(synthetic, ())

    assert output == before
    assert tracker.snapshot() == before
    assert tracker.last_h1_end is None
    assert tracker.on_completed_h1(synthetic, ()) is output
    with pytest.raises(
        ValueError,
        match="duplicate or out-of-order Group 4 native candle",
    ):
        tracker.on_completed_h1(_h1(0), ())
    real = _h1(1)
    tracker.on_completed_h1(real, ())
    assert tracker.last_h1_end == real.end


def test_session_aware_completed_h1_does_not_require_60_observed_minutes() -> None:
    tracker = CausalRangeAuctionTracker(_protocol())
    shortened = replace(
        _h1(0),
        observed_minutes=45,
        expected_minutes=45,
        real_minutes=45,
    )

    tracker.on_completed_h1(shortened, ())

    assert tracker.last_h1_end == shortened.end


def test_synthetic_completed_bar_is_a_semantic_noop() -> None:
    tracker = CausalRangeAuctionTracker(_protocol())
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
    tracker = CausalRangeAuctionTracker(_protocol())
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

    output = CausalRangeAuctionTracker(
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
        CausalRangeAuctionTracker(
            _protocol()
        ).bootstrap_completed_1m_prefix(
            (_m1(1),),
            pool_inventory=(inventory,),
            liquidity_pools=(pool,),
        )


def test_exact_source_bindings_fail_closed_without_partial_commit() -> None:
    pool_tracker = CausalRangeAuctionTracker(_protocol())
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
    assert mature.balance_confirmed_at is not None
    base = mature.balance_confirmed_at + pd.Timedelta(minutes=1)
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
    tracker = CausalRangeAuctionTracker(_protocol())

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
                range_auction_protocol=str(PROTOCOL_PATH),
            )
        )


def test_range_boundaries_join_the_visible_liquidity_route_inventory() -> None:
    tracker, _, mature = _mature_range(_protocol())
    assert mature.balance_confirmed_at is not None
    pool, pool_item = _pool(
        "above",
        confirmed_at=mature.balance_confirmed_at,
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
        item.item_id for item in current_dol_inventory(observation)
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
    update = CausalRangeAuctionTracker(
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
    observer._emitter._record_group4_events(
        update,
        include_ranges=False,
            prior=observer._prior,
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
        balance_confirmed_at=swept.swept_at,
        state_started_at=swept.swept_at,
        last_updated_at=swept.swept_at,
    )
    observer._emitter._record_group4_events(
        RangeAuctionUpdate(
            dealing_ranges=(same_clock_range,),
            manipulations=(),
            range_boundary_inventory=(),
            range_transitions=(same_clock_range,),
        ),
        include_resolutions=False,
        include_creations=False,
            prior=observer._prior,
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
