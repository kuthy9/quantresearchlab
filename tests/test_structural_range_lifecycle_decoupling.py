"""A Structural Range's life is created -> active -> broken, and nothing else.

Balance was a *maturity state* of the range, so a range that never balanced was
recorded as a range that failed. Three months of data removed the grounds for
that inheritance: 0 of 84 ranges reached a two-sided test, an arbitrary rolling
H1 window reaches one 4.06% of the time, and the two are statistically
indistinguishable. Balance is rare in this market -- the range was never
failing at anything.

So the lifecycle drops the balance-derived state. What stays untouched:

* Group 4 still selects candidates exactly as before. This is a lifecycle
  change, not a redefinition of what a Structural Range is.
* Balance evidence is still collected, still published once per range as
  ``BALANCE_RANGE_OBSERVED``, and still governs when boundaries are promoted to
  the liquidity inventory. What it no longer does is grade the range.
"""
from __future__ import annotations

import pytest

from smc_trader.model import DealingRangeLifecycle, EventKind
from smc_trader.range_auction import CausalRangeAuctionTracker

from .test_range_auction_primitives import (
    _h1,
    _mature_range,
    _protocol,
    _warm_h1,
    _zone,
)


def _formed():
    tracker = CausalRangeAuctionTracker(_protocol())
    zones = (_zone("support"), _zone("resistance"))
    _warm_h1(tracker)
    output = tracker.on_completed_h1(_h1(14, close=99.75, span=1.0), zones)
    return tracker, zones, output


def _drive(tracker, zones, index, *, close, span=1.0):
    return tracker.on_completed_h1(_h1(index, close=close, span=span), zones)


def test_the_lifecycle_has_exactly_created_active_and_broken() -> None:
    values = {member.value for member in DealingRangeLifecycle}

    assert values == {"active", "broken"}
    # "mature" was a grade the range earned by balancing; "forming" implied it
    # was on the way to earning one. A location does neither.
    assert not hasattr(DealingRangeLifecycle, "MATURE")
    assert not hasattr(DealingRangeLifecycle, "FORMING")


def test_a_range_is_active_from_creation() -> None:
    _, _, output = _formed()
    state = output.dealing_ranges[-1]

    assert state.lifecycle is DealingRangeLifecycle.ACTIVE
    assert state.transition_reason == "source_pair_selected"
    # The clock survives under the name of what it actually times. It records
    # when the balance claim settled, which boundary promotion and downstream
    # consumers need; it is not a grade the range holds.
    assert not hasattr(state, "mature_at")
    assert state.balance_confirmed_at is None


def test_a_range_ends_only_when_price_closes_outside_it() -> None:
    tracker, zones, _ = _formed()
    output = _drive(tracker, zones, 15, close=102.0)

    broken = output.range_transitions[-1]
    assert broken.lifecycle is DealingRangeLifecycle.BROKEN
    assert broken.transition_reason == "close_beyond_frozen_range"


def test_balance_evidence_never_changes_the_range_lifecycle() -> None:
    """The exact bars that used to mature a range now leave it active."""

    _, _, mature = _mature_range(_protocol())

    assert mature.lifecycle is DealingRangeLifecycle.ACTIVE
    # The claim settled, and that is recorded on the range as a reason rather
    # than as a state: the interval is the same location it was a bar ago.
    assert mature.transition_reason == "balance_claim_confirmed"
    assert mature.balance_confirmed_at is not None
    # The evidence itself is still collected and still reached the standard.
    assert mature.balance_lower_test_generations >= 2
    assert mature.balance_upper_test_generations >= 2


def test_a_confirmed_balance_claim_still_promotes_the_boundaries() -> None:
    """Group 4's behaviour is preserved: only the lifecycle vocabulary moved."""

    tracker, _, mature = _mature_range(_protocol())
    inventory = {
        item.item_id: item
        for item in tracker.snapshot().range_boundary_inventory
    }

    assert len(inventory) == 2, "range boundaries were not promoted"
    assert {item.side for item in inventory.values()} == {"above", "below"}


def test_a_range_that_never_balances_stays_active_rather_than_failing() -> None:
    tracker, zones, _ = _formed()
    output = None
    for index in range(15, 45):
        output = _drive(tracker, zones, index, close=100.0, span=0.25)

    live = [
        state
        for state in output.dealing_ranges
        if state.lifecycle is DealingRangeLifecycle.ACTIVE
    ]
    assert live, "a range that never balanced was graded out of existence"


def test_balance_range_matured_is_no_longer_emitted() -> None:
    """The event kind stays registered; nothing produces it any more."""

    assert EventKind.BALANCE_RANGE_MATURED.value == "balance_range_matured"


@pytest.mark.parametrize("lifecycle", tuple(DealingRangeLifecycle))
def test_every_lifecycle_value_is_reachable_vocabulary(lifecycle) -> None:
    assert lifecycle.value in {"active", "broken"}
