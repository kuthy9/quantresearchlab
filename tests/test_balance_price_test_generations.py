"""Balance evidence is price interacting with a boundary, not structure recurring.

Balance asks whether price repeatedly traded into, and was rejected by, both
sides of an interval.  Until ``balance_range_v1.2`` the evidence for that came
from the boundary zone's ``total_touch_count``, which a structural_swing zone
only increments when *another confirmed swing* forms inside it -- price trading
into the boundary never counted, because ``_update_reference_contacts`` skips
structural_swing zones outright.  A second confirmed H1 swing landing in one
narrow band inside a range's 8-24 bar life is rare, so 17 of 18 zones in
2022-02 stayed at one touch for life and the two-sided standard was effectively
unreachable.

The two counters are now separate concepts and separate fields: the S/R zone
keeps its structural touch count untouched, and the balance claim reads price
test generations it computes itself.
"""
from __future__ import annotations

import pytest

from smc_trader.model import DealingRangeLifecycle, RANGE_MATURITY_GATE_NAMES
from smc_trader.range_auction import CausalRangeAuctionTracker

from .test_range_auction_primitives import (
    _h1,
    _protocol,
    _warm_h1,
    _zone,
)


# The harness range is 99.0 - 101.0 with a formation ATR of 1.0, so the
# registered band is max(1 tick, 0.25 * 1.0) = 0.25 wide and deep penetration
# begins 0.25 beyond the boundary.
UPPER = 101.0
LOWER = 99.0
BAND = 0.25


def _formed() -> tuple[CausalRangeAuctionTracker, tuple]:
    tracker = CausalRangeAuctionTracker(_protocol())
    zones = (_zone("support"), _zone("resistance"))
    _warm_h1(tracker)
    output = tracker.on_completed_h1(_h1(14, close=99.75, span=1.0), zones)
    assert output.dealing_ranges[-1].lifecycle is DealingRangeLifecycle.ACTIVE
    return tracker, zones


def _drive(tracker, zones, index, *, close, high, low):
    candle = _h1(index, close=close, span=1.0)
    candle = type(candle)(
        **{
            **{
                field: getattr(candle, field)
                for field in candle.__dataclass_fields__
            },
            "high": high,
            "low": low,
        }
    )
    return tracker.on_completed_h1(candle, zones)


def _live(output):
    return output.dealing_ranges[-1]


def test_a_bar_reaching_the_band_without_crossing_is_one_touch_only_test() -> None:
    tracker, zones = _formed()
    state = _live(
        _drive(tracker, zones, 15, close=100.0, high=UPPER - 0.05, low=99.9)
    )

    assert state.balance_upper_test_generations == 1
    assert state.balance_upper_test_kinds == ("touch_only",)
    assert state.balance_lower_test_generations == 0


def test_consecutive_bars_hugging_one_boundary_are_one_test_not_many() -> None:
    tracker, zones = _formed()
    for index in (15, 16, 17):
        output = _drive(
            tracker, zones, index, close=100.0, high=UPPER - 0.05, low=99.9
        )

    state = _live(output)
    assert state.balance_upper_test_generations == 1
    assert state.candidate_real_h1_bars == 4


def test_leaving_the_band_and_returning_opens_the_second_test() -> None:
    tracker, zones = _formed()
    _drive(tracker, zones, 15, close=100.0, high=UPPER - 0.05, low=99.9)
    # Well clear of the band: this is what closes the first test.
    _drive(tracker, zones, 16, close=100.0, high=100.2, low=99.9)
    state = _live(
        _drive(tracker, zones, 17, close=100.0, high=UPPER - 0.05, low=99.9)
    )

    assert state.balance_upper_test_generations == 2
    assert state.balance_upper_test_kinds == ("touch_only", "touch_only")


def test_a_test_is_classified_by_its_deepest_interaction() -> None:
    tracker, zones = _formed()
    _drive(tracker, zones, 15, close=100.0, high=UPPER - 0.05, low=99.9)
    _drive(tracker, zones, 16, close=100.0, high=UPPER + 0.10, low=99.9)
    state = _live(
        _drive(tracker, zones, 17, close=100.0, high=UPPER + 0.50, low=99.9)
    )

    assert state.balance_upper_test_generations == 1
    assert state.balance_upper_test_kinds == ("deep_penetration",)


def test_both_boundaries_accumulate_their_own_tests() -> None:
    tracker, zones = _formed()
    _drive(tracker, zones, 15, close=100.0, high=UPPER - 0.05, low=99.9)
    _drive(tracker, zones, 16, close=100.0, high=100.2, low=LOWER + 0.05)
    _drive(tracker, zones, 17, close=100.0, high=100.2, low=99.9)
    state = _live(
        _drive(tracker, zones, 18, close=100.0, high=UPPER - 0.05, low=LOWER + 0.05)
    )

    assert state.balance_upper_test_generations == 2
    assert state.balance_lower_test_generations == 2


def test_a_close_outside_the_range_still_breaks_it() -> None:
    tracker, zones = _formed()
    output = _drive(
        tracker, zones, 15, close=UPPER + 0.50, high=UPPER + 0.75, low=99.9
    )

    broken = output.range_transitions[-1]
    assert broken.lifecycle is DealingRangeLifecycle.BROKEN
    assert broken.transition_reason == "close_beyond_frozen_range"
    assert broken.balance_upper_test_kinds[-1] == "close_outside"


def test_price_tests_never_move_the_structural_touch_count() -> None:
    tracker, zones = _formed()
    for index in (15, 16, 17, 18):
        output = _drive(
            tracker,
            zones,
            index,
            close=100.0,
            high=UPPER - 0.05 if index % 2 else 100.2,
            low=LOWER + 0.05 if index % 2 else 99.9,
        )

    state = _live(output)
    # The zone the boundary was frozen from is untouched by price interaction:
    # its count is still the two member swings the harness zone was built with.
    assert (state.lower_touch_count, state.upper_touch_count) == (2, 2)
    assert state.balance_upper_test_generations >= 1


def test_the_maturity_gate_reads_price_tests_rather_than_structural_touches() -> None:
    assert "bilateral_price_tests" in RANGE_MATURITY_GATE_NAMES
    assert "bilateral_touches" not in RANGE_MATURITY_GATE_NAMES


def test_the_balance_sub_protocol_declares_its_own_version() -> None:
    protocol = _protocol()
    assert protocol.balance_sub_protocol_version == "balance_range_v1.2"
    assert protocol.balance_minimum_price_test_generations_each == 2
    assert protocol.balance_price_test_band_atr_fraction > 0.0
    assert protocol.balance_deep_penetration_atr_fraction > 0.0


@pytest.mark.parametrize("side", ("upper", "lower"))
def test_every_recorded_test_kind_is_registered(side: str) -> None:
    tracker, zones = _formed()
    for index in (15, 16, 17, 18, 19):
        output = _drive(
            tracker,
            zones,
            index,
            close=100.0,
            high=UPPER + 0.10 if index % 2 else 100.2,
            low=LOWER - 0.10 if index % 2 else 99.9,
        )

    state = _live(output)
    kinds = getattr(state, f"balance_{side}_test_kinds")
    assert set(kinds) <= {
        "touch_only",
        "shallow_penetration",
        "deep_penetration",
        "close_outside",
    }
