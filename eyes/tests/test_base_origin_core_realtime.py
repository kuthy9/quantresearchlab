"""The core of an origin zone exists before anything qualifies it.

The last opposite candles before an impulse are frozen the moment the impulse
starts.  v1.3 published that geometry only when a later BOS created the order
block, which reads the past with knowledge of the future: every core the Eye
had ever seen was, by construction, one that went on to work.  The core is now
published on the bar that locks it -- qualified or not -- and the qualified
zone is a separate, later fact that cites it.
"""
from __future__ import annotations

import pytest

from shares.core.model import Direction, OrderBlockLifecycle

from eyes.tests.test_zone_primitives import (
    _Harness,
    _displacement_pair,
    _form_order_block,
    _older_opposite_anchor,
    _opposite_anchor,
    _warm,
)


def _impulse_start(direction: Direction):
    """Replay up to the bar that starts the impulse, keeping its output."""

    harness = _Harness()
    history = _warm(
        harness,
        count=64,
        overrides={
            10: _older_opposite_anchor(direction),
            63: _opposite_anchor(direction),
        },
    )
    seed_values, active_values = _displacement_pair(direction)
    seed_candle, displacement, _, output = harness.send(seed_values)
    return harness, history[-1], seed_candle, displacement, output, active_values


@pytest.mark.parametrize("direction", (Direction.LONG, Direction.SHORT))
def test_the_impulse_publishes_its_core_with_no_break_in_sight(
    direction: Direction,
) -> None:
    _, anchor, seed_candle, _, output, _ = _impulse_start(direction)

    assert output.order_block_transitions == (), (
        "nothing qualified this impulse, so no origin zone may exist"
    )
    assert len(output.base_origin_cores) == 1
    core = output.base_origin_cores[0]
    assert core.direction is direction
    assert core.observed_at == seed_candle.end
    assert core.anchor_end == anchor.end
    assert core.lower_bound == anchor.low
    assert core.upper_bound == anchor.high
    assert core.anchor_candle_ids
    # No bar the clock has not reached, and no break in its ancestry.
    assert core.anchor_end <= core.observed_at


def test_the_core_is_published_on_the_bar_that_locks_it_and_not_again() -> None:
    harness, _, _, _, first, active_values = _impulse_start(Direction.LONG)
    assert len(first.base_origin_cores) == 1

    _, _, _, later = harness.send(active_values)

    assert later.base_origin_cores == ()


def test_a_core_outlives_an_impulse_that_never_breaks_structure() -> None:
    """The point of publishing early: unqualified cores are observable."""

    harness, _, _, _, output, active_values = _impulse_start(Direction.LONG)
    core = output.base_origin_cores[0]

    _, _, _, later = harness.send(active_values)

    assert later.order_block_transitions == ()
    assert core.base_origin_core_id in harness.group3._base_origin_cores


@pytest.mark.parametrize("direction", (Direction.LONG, Direction.SHORT))
def test_a_qualified_zone_names_the_core_the_impulse_already_published(
    direction: Direction,
) -> None:
    harness, state, _, _, _, _, output = _form_order_block(direction)

    assert state.lifecycle is OrderBlockLifecycle.CREATED
    assert output.base_origin_cores == (), (
        "the core was already published when the impulse locked it"
    )
    assert state.base_origin_core_id is not None
    assert state.base_origin_core_id in harness.group3._base_origin_cores
    core = harness.group3._base_origin_cores[state.base_origin_core_id]
    # The zone is a different object from its core, not a rename of it.
    assert state.order_block_id != core.base_origin_core_id
    assert (state.lower_bound, state.upper_bound) == (
        core.lower_bound,
        core.upper_bound,
    )
    assert core.observed_at < state.confirmed_at


def test_the_core_identity_does_not_depend_on_the_break_that_qualified_it() -> None:
    """Same impulse, same frozen candles, same core -- break or no break."""

    _, _, _, _, unqualified, _ = _impulse_start(Direction.LONG)
    _, state, _, _, _, _, _ = _form_order_block(Direction.LONG)

    assert (
        unqualified.base_origin_cores[0].base_origin_core_id
        == state.base_origin_core_id
    )
