"""The liquidity tracker's rollback is a bounded snapshot, not a deep copy.

Once a scale's support/resistance retention is at capacity (128 zones on 1m
after about a day of tape, before terminal zones were compacted after their
exposure), every new swing runs the compaction path and the tracker took a
``copy.deepcopy`` of its whole ``__dict__`` first, as insurance against a
fail-closed compaction.  That deep copy walked every retained zone,
pool and generation record and every frozen state under them -- 6 million
``deepcopy`` calls per 500 bars at bar 2,000 on the real tape, the largest
single growth owner left after the candidate set was bounded.  The rollback
now snapshots each container one level deep and copies the mutable records;
the frozen states under them are shared, because nothing mutates them.
"""
from __future__ import annotations

import copy

import pytest

from contract.eye import SwingSide
from contract.market import Timeframe
from eyes.core.liquidity import CausalLiquidityTracker, LiquidityConfig

from eyes.tests.test_v3_group12_primitives import _candle, _swing


def _tracker_at_capacity() -> tuple[CausalLiquidityTracker, tuple]:
    """Eight live zones fill the retention; the next update must snapshot.

    Every swing stays in the structure snapshot, so its zone stays live and
    is never compacted; the update under test drops the first swing (its zone
    retires on that bar) and confirms a ninth, which is the one shape that
    reaches capacity with the same-bar terminal exposure the protocol allows.
    """

    tracker = CausalLiquidityTracker(
        Timeframe.M1,
        LiquidityConfig(tick_size=0.25, cluster_tolerance_atr=0.10, retained_zones=8, retained_pools=8),
    )
    swings = []
    for index in range(8):
        # Distinct prices well apart, so every swing is its own zone.
        swings.append(
            _swing(
                f"high-{index}",
                side=SwingSide.HIGH,
                price=100.0 + 5.0 * index,
                confirmed_index=index,
            )
        )
        candle = _candle(index, open_=99.5, high=100.0 + 5.0 * index + 0.5, low=99.0, close=99.75)
        tracker.on_candle(candle, tuple(swings))
    assert len(tracker._zones) == 8
    ninth = _swing("high-8", side=SwingSide.HIGH, price=200.0, confirmed_index=8)
    return tracker, (*swings[1:], ninth)


def test_the_rollback_snapshot_does_not_deep_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker, swings = _tracker_at_capacity()

    def refuse(*_args, **_kwargs):
        raise AssertionError("rollback took a deep copy")

    monkeypatch.setattr(copy, "deepcopy", refuse)
    tracker.on_candle(_candle(8, open_=99.5, high=200.5, low=99.0, close=99.75), swings)
    assert len(tracker._zones) == 9


def test_a_failed_update_restores_the_exact_prior_state(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker, swings = _tracker_at_capacity()
    before = copy.deepcopy(tracker.__dict__)

    def fail(*_args, **_kwargs):
        raise RuntimeError("compaction refused")

    monkeypatch.setattr(type(tracker), "_compact_terminal_zones", fail)
    with pytest.raises(RuntimeError, match="compaction refused"):
        tracker.on_candle(_candle(8, open_=99.5, high=200.5, low=99.0, close=99.75), swings)
    after = tracker.__dict__
    assert set(after) == set(before)
    for name, value in before.items():
        assert after[name] == value, name
