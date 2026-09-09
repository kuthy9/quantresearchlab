"""Location and balance are two claims about an interval, not one.

v1.2 folded both into the Active Dealing Range: a range was "active" only once
maturity conditions held, so an interval that located price perfectly well but
had never been two-sided tested carried no usable identity.  v1.3 keeps the
frozen structural interval (``DEALING_RANGE_CREATED``) as the location fact and
publishes the balance claim separately, so failing to balance no longer costs
the location.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from contract.eye import (
    DealingRangeLifecycle,
    EventKind,
)
from eyes.core.observation import CausalObserver, ObserverConfig
from eyes.core.range_auction import RangeAuctionUpdate
from eyes.core.semantics import SemanticRegistry

from shares.tests.helpers import CORE_TEST_SCALE_SPECS
from eyes.tests.test_semantic_foundation_geometry import _balance_range


def _observer() -> CausalObserver:
    return CausalObserver(ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS))


def _emit(observer: CausalObserver, *states) -> None:
    observer._emitter._record_group4_events(
        RangeAuctionUpdate(
            dealing_ranges=(),
            manipulations=(),
            range_boundary_inventory=(),
            range_transitions=tuple(states),
        ),
        include_resolutions=False,
        include_creations=False,
        prior=observer._prior,
    )


def _events(observer: CausalObserver):
    observer.memory.flush_audit()
    return observer.audit_store.events()


def _kinds(observer: CausalObserver) -> list[EventKind]:
    return [event.kind for event in _events(observer)]


def _two_sided_tested():
    """A range price has tested twice on each side, to the registered standard.

    Under ``balance_range_v1.2`` the evidence is price visiting each frozen
    boundary's tolerance band in distinct generations.  The structural touch
    count stays where it was: it is a different claim, and leaving it at the
    structural minimum here is what proves the two are decoupled.
    """

    base = _balance_range()
    narrowness = base.narrowness_strength
    compression = base.compression_strength
    boundary_tests = base.boundary_test_strength
    crossings = base.crossing_strength
    return replace(
        base,
        balance_lower_test_generations=2,
        balance_upper_test_generations=2,
        balance_lower_test_kinds=("touch_only", "shallow_penetration"),
        balance_upper_test_kinds=("touch_only", "touch_only"),
        # The source zones' own test clocks are deliberately left alone: a
        # tested zone is a structural fact with its own invariant (two member
        # touches), and balance no longer reads it.
        last_updated_at=base.formed_at + pd.Timedelta(hours=2),
        candidate_real_h1_bars=3,
        age_h1_bars=2,
        strength=(
            narrowness
            + compression
            + boundary_tests
            + crossings
            + base.inside_close_fraction
        )
        / 5.0,
    )


def test_a_two_sided_tested_range_publishes_its_balance_claim_once() -> None:
    observer = _observer()
    forming = _two_sided_tested()

    _emit(observer, forming)

    kinds = _kinds(observer)
    assert EventKind.DEALING_RANGE_CREATED in kinds
    observed = [
        event
        for event in _events(observer)
        if event.kind is EventKind.BALANCE_RANGE_OBSERVED
    ]
    assert len(observed) == 1
    details = observed[0].details
    assert details["range_id"] == forming.range_id
    assert details["balance_lower_test_generations"] == 2
    assert details["balance_upper_test_generations"] == 2
    assert details["balance_upper_test_kinds"] == ("touch_only", "touch_only")
    # The structural touch count travels beside the balance evidence and is
    # still the structural minimum: the two counters are separate concepts.
    assert details["lower_touch_count"] == forming.lower_touch_count
    assert details["upper_touch_count"] == forming.upper_touch_count
    assert details["midpoint_crossings"] == forming.midpoint_crossings
    assert details["inside_close_fraction"] == forming.inside_close_fraction
    assert details["compression_ratio"] == forming.compression_ratio


def test_a_range_below_the_test_standard_still_locates_but_claims_no_balance() -> None:
    observer = _observer()
    # No price test generation on either side: the range located price and
    # nothing ever came back to test where it was.
    untested = _balance_range()

    _emit(observer, untested)

    kinds = _kinds(observer)
    assert EventKind.DEALING_RANGE_CREATED in kinds
    assert EventKind.BALANCE_RANGE_OBSERVED not in kinds


def test_the_balance_claim_is_not_restated_on_a_later_transition() -> None:
    observer = _observer()
    forming = _two_sided_tested()
    _emit(observer, forming)
    later = forming.formed_at + pd.Timedelta(hours=2)
    broken = replace(
        forming,
        lifecycle=DealingRangeLifecycle.BROKEN,
        broken_at=later,
        state_started_at=later,
        last_updated_at=later,
        transition_reason="maturity_deadline_elapsed",
    )

    _emit(observer, broken)

    observed = [
        event
        for event in _events(observer)
        if event.kind is EventKind.BALANCE_RANGE_OBSERVED
    ]
    assert len(observed) == 1


def test_balance_is_its_own_fact_and_the_activation_alias_is_retired() -> None:
    registry = SemanticRegistry.from_file()

    assert (
        EventKind.DEALING_RANGE_ACTIVATED
        not in registry.canonical_emitted_event_kinds
    )
    assert (
        EventKind.BALANCE_RANGE_OBSERVED
        in registry.canonical_emitted_event_kinds
    )
    # Maturity was a state of the Structural Range, and the range no longer
    # has one. The observation survives because it is a fact about price;
    # the grade does not, because it was a fact about the interval.
    assert (
        registry.event_binding_by_kind[
            EventKind.BALANCE_RANGE_MATURED
        ].status
        == "reserved_not_emitted"
    )


def test_a_structural_range_locates_price_before_anything_balances() -> None:
    """The location claim needs no maturity: it is the interval itself.

    A registered structural interval already knows its boundaries, so it can
    say where price sits inside it -- premium, discount, IRL, ERL -- from the
    moment it is created.  Withholding that until the balance gates pass made
    a two-sided-test statistic a precondition for arithmetic.
    """

    from eyes.core.market_state import (
        DOLCandidateView,
        LiquidityRangeRole,
        TimeframeRangeState,
        _candidate_range_membership,
    )
    from contract.market import Timeframe

    created = TimeframeRangeState(
        range_id="range-1",
        low=100.0,
        high=120.0,
        normalized_location=0.5,
        location_label="equilibrium",
        lifecycle="created",
        source_ids=("event-1",),
        range_kind="active_dealing_range",
    )
    inside = DOLCandidateView(
        candidate_id="level-inside",
        timeframe=Timeframe.H1,
        side="above",
        price=110.0,
        source_kind="confirmed_swing",
        lower_bound=110.0,
        upper_bound=110.0,
        formed_at=pd.Timestamp("2025-01-06 10:00", tz="America/New_York"),
        confirmed_at=pd.Timestamp("2025-01-06 10:00", tz="America/New_York"),
    )

    resolved = _candidate_range_membership(inside, created)

    assert resolved.range_role is LiquidityRangeRole.IRL
    assert resolved.normalized_location_in_range == 0.5


def test_an_unbalanced_candidate_keeps_its_structural_interval() -> None:
    """Failing to balance ends the balance claim, never the location."""

    from eyes.core.range_auction import BALANCE_CLAIM_ABANDONED

    state = replace(
        _balance_range(),
        lifecycle=DealingRangeLifecycle.ACTIVE,
        transition_reason=BALANCE_CLAIM_ABANDONED,
        broken_at=None,
        balance_confirmed_at=None,
    )

    assert state.lifecycle is DealingRangeLifecycle.ACTIVE
    assert state.broken_at is None


def _live(observer: CausalObserver, *states) -> None:
    """One completed H1 bar that changes no range lifecycle."""

    observer._emitter._record_group4_events(
        RangeAuctionUpdate(
            dealing_ranges=tuple(states),
            manipulations=(),
            range_boundary_inventory=(),
            range_transitions=(),
        ),
        include_resolutions=False,
        include_creations=False,
        prior=observer._prior,
    )


def test_the_balance_claim_lands_on_the_bar_that_completes_the_second_test() -> None:
    """The observation clock is the H1 bar, not the range's next transition.

    Group 4 recomputes every balance statistic on each completed H1 bar but
    only surfaced a range when its lifecycle changed, so the earliest clock at
    which the Eye could say "both boundaries have now been tested" was the
    range's next transition -- which for a candidate that simply keeps forming
    never arrives.  Over two complete months the detector fired zero times.
    """

    observer = _observer()
    forming = _balance_range()
    _emit(observer, forming)
    assert EventKind.BALANCE_RANGE_OBSERVED not in _kinds(observer)

    tested = _two_sided_tested()
    _live(observer, tested)

    observed = [
        event
        for event in _events(observer)
        if event.kind is EventKind.BALANCE_RANGE_OBSERVED
    ]
    assert len(observed) == 1
    assert observed[0].known_at == tested.last_updated_at
    assert observed[0].details["range_id"] == tested.range_id
    assert observed[0].details["observed_at_lifecycle"] == (
        DealingRangeLifecycle.ACTIVE.value
    )


def test_a_live_bar_below_the_test_standard_claims_no_balance() -> None:
    observer = _observer()
    forming = _balance_range()
    _emit(observer, forming)

    # One boundary tested twice is still one-sided: the gate reads the minimum.
    _live(
        observer,
        replace(
            forming,
            upper_touch_count=2,
            upper_source_tested_at=forming.formed_at,
            last_updated_at=forming.formed_at + pd.Timedelta(hours=1),
        ),
    )

    assert EventKind.BALANCE_RANGE_OBSERVED not in _kinds(observer)


def test_the_live_balance_claim_is_published_once_and_never_restated() -> None:
    observer = _observer()
    _emit(observer, _balance_range())
    tested = _two_sided_tested()

    for hours in (2, 3, 4):
        _live(
            observer,
            replace(
                tested,
                last_updated_at=tested.formed_at + pd.Timedelta(hours=hours),
            ),
        )

    observed = [
        event
        for event in _events(observer)
        if event.kind is EventKind.BALANCE_RANGE_OBSERVED
    ]
    assert len(observed) == 1
