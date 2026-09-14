"""A resolved pool or reaccepted zone is exposed once, then leaves the snapshot.

The liquidity tracker kept every terminal zone and pool until ``retained_zones``
/ ``retained_pools`` (128) forced the oldest out, so on the real tape 90 of the
1m tracker's 128 zones and 123 of its 128 pools were terminal history, rebuilt
into every snapshot and re-projected into every inventory.  The terminal
transition is delivered on the bar it happens and the fact stays in the event
log; the record itself now stays in the tracker for
``terminal_state_retention_native_bars`` completed native bars, counting the
bar that made it terminal, and is then compacted.  The capacity rule is
unchanged: it still fails closed when live state exhausts the retention.
"""
from __future__ import annotations

from contract.eye import (
    LiquidityPoolLifecycle,
    SupportResistanceLifecycle,
    SwingSide,
)
from contract.market import Timeframe
from eyes.core.liquidity import CausalLiquidityTracker, LiquidityConfig

from eyes.tests.test_v3_group12_primitives import _candle, _swing

PROTOCOL = "configs/primitives_structure_liquidity.json"


def test_the_protocol_names_the_terminal_retention() -> None:
    config = LiquidityConfig.from_file(PROTOCOL, tick_size=0.25, atr_period=14)
    assert config.terminal_state_retention_native_bars == 1


def _swept_pool_tracker() -> CausalLiquidityTracker:
    tracker = CausalLiquidityTracker(Timeframe.M1)
    first = _swing("high-a", side=SwingSide.HIGH, price=100.0, confirmed_index=0)
    second = _swing("high-b", side=SwingSide.HIGH, price=100.25, confirmed_index=1)
    tracker.on_candle(_candle(0, open_=99.5, high=101.0, low=99.0, close=99.75), (first,))
    tracker.on_candle(_candle(1, open_=99.75, high=101.0, low=99.0, close=99.75), (first, second))
    tracker.on_candle(_candle(2, open_=100.0, high=100.75, low=99.75, close=100.5), (first, second))
    tracker.on_candle(_candle(3, open_=100.5, high=100.75, low=100.25, close=100.5), (first, second))
    tracker._swings = (first, second)  # test-only handle for the next bars
    return tracker


def test_a_resolved_pool_is_exposed_on_its_bar_and_compacted_on_the_next() -> None:
    tracker = _swept_pool_tracker()
    swings = tracker._swings
    _, pools, inventory = tracker.snapshot()
    assert [pool.lifecycle for pool in pools] == [LiquidityPoolLifecycle.ACCEPTED]
    assert [item.item_id for item in inventory if item.kind == "equal_highs"] == [
        f"pool:{pools[0].pool_id}"
    ]

    tracker.on_candle(_candle(4, open_=100.5, high=100.5, low=99.75, close=100.0), swings)
    zones, pools, inventory = tracker.snapshot()
    assert pools == ()
    assert all(item.kind != "equal_highs" for item in inventory)
    # The zone the pool belonged to reaccepted on this bar: exposed now ...
    assert [zone.lifecycle for zone in zones] == [SupportResistanceLifecycle.REACCEPTED]

    tracker.on_candle(_candle(5, open_=100.0, high=100.25, low=99.75, close=100.0), swings)
    zones, _, _ = tracker.snapshot()
    # ... and compacted on the next completed native bar.
    assert zones == ()


def test_a_live_zone_is_never_compacted_by_retention() -> None:
    tracker = CausalLiquidityTracker(Timeframe.M1)
    swing = _swing("high-a", side=SwingSide.HIGH, price=100.0, confirmed_index=0)
    tracker.on_candle(_candle(0, open_=99.5, high=101.0, low=99.0, close=99.75), (swing,))
    for index in range(1, 6):
        tracker.on_candle(
            _candle(index, open_=99.5, high=99.75, low=99.0, close=99.5),
            (swing,),
        )
        zones, _, _ = tracker.snapshot()
        assert [zone.lifecycle for zone in zones] == [SupportResistanceLifecycle.ACTIVE]


def test_a_projected_resolution_is_exposed_by_the_native_bar_that_carries_it() -> None:
    """The observer resolves a pending sweep from the completed 1m bar *before*
    the native trackers see that bar, so the terminal bar of a projected
    resolution is the native bar about to be processed, not the last one."""
    import pandas as pd

    from eyes.tests.test_v3_group12_primitives import BASE

    tracker = CausalLiquidityTracker(Timeframe.M1)
    first = _swing("high-a", side=SwingSide.HIGH, price=100.0, confirmed_index=0)
    second = _swing("high-b", side=SwingSide.HIGH, price=100.25, confirmed_index=1)
    tracker.on_candle(_candle(0, open_=99.5, high=100.0, low=99.0, close=99.75), (first,))
    tracker.on_candle(_candle(1, open_=99.75, high=100.0, low=99.0, close=99.75), (first, second))
    swings = (first, second)
    (pool,) = tracker.snapshot()[1]
    assert pool.lifecycle is LiquidityPoolLifecycle.FORMED
    tracker.project_pool_sweep(
        pool.pool_id,
        observed_at=BASE + pd.Timedelta(minutes=3),
        sweep_extreme=100.75,
        close_outside_on_sweep=True,
    )
    tracker.on_candle(_candle(2, open_=100.0, high=100.75, low=99.75, close=100.5), swings)
    tracker.project_pool_resolution(
        pool.pool_id,
        observed_at=BASE + pd.Timedelta(minutes=4),
        accepted_outside=False,
    )
    # The native bar whose clock the projection carries still exposes it ...
    tracker.on_candle(_candle(3, open_=100.5, high=100.6, low=100.0, close=100.1), swings)
    (resolved,) = tracker.snapshot()[1]
    assert resolved.lifecycle is LiquidityPoolLifecycle.REJECTED
    assert resolved.resolved_at == BASE + pd.Timedelta(minutes=4)
    # ... and the bar after compacts it.
    tracker.on_candle(_candle(4, open_=100.1, high=100.2, low=99.9, close=100.0), swings)
    assert tracker.snapshot()[1] == ()
