"""A terminal FVG or order block is exposed once, then compacted.

Group 3 kept every mitigated, invalidated, expired or failed entity until
``maximum_fvg_states`` / ``maximum_order_block_states`` forced the oldest
out -- the same "retain terminal state until capacity" pattern the Group 5
context paths and the liquidity tracker replaced with an explicit exposure.
On the 2022-02 tape the 5m frame handed Group 5 285 FVG/OB sources per bar at
bar 20,000 against 9 at bar 500, nearly all of them terminal, and each
terminal state also held a live entity timeline in EventMemory.

A terminal state now stays in the tracker for
``terminal_state_retention_native_bars`` (1) completed native bars, the one
that made it terminal, then is compacted; the terminal transition was
delivered on its bar and the event log owns the fact.  Capacity still refuses
to evict a live entity.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from contract.eye import FairValueGapLifecycle
from contract.market import Direction, Timeframe
from eyes.core.causal import CausalMarketReader
from eyes.core.observation import CausalObserver, ObserverConfig
from eyes.core.zone import CausalZoneTracker, ZoneProtocol

from eyes.tests.test_facts_on_every_scale import _noisy
from eyes.tests.test_zone_primitives import _form_fvg, _fvg_by_id, _zone_protocol
from shares.tests.helpers import MODEL_SCALE_SPECS, session_bars


def test_protocol_registers_a_positive_terminal_retention() -> None:
    protocol = _zone_protocol()
    assert protocol.terminal_state_retention_native_bars == 1
    with pytest.raises(ValueError, match="frozen contract"):
        replace(protocol, terminal_state_retention_native_bars=0)


def test_a_mitigated_fvg_is_exposed_on_its_bar_and_gone_on_the_next() -> None:
    harness, created, _, _, _ = _form_fvg(Direction.LONG)
    terminal_output = harness.send((102.25, 102.5, 100.5, 100.5))[-1]
    terminal = _fvg_by_id(terminal_output, created.fvg_id)
    assert terminal.lifecycle is FairValueGapLifecycle.MITIGATED
    assert any(
        state.fvg_id == created.fvg_id
        for state in terminal_output.fair_value_gaps
    )

    next_output = harness.send((100.5, 100.75, 100.25, 100.5))[-1]
    assert all(
        state.fvg_id != created.fvg_id
        for state in next_output.fair_value_gaps
    )
    assert created.fvg_id not in harness.group3._fair_value_gaps


def test_capacity_still_refuses_to_evict_live_state() -> None:
    _, created, _, _, _ = _form_fvg(Direction.LONG)
    protocol = _zone_protocol()
    blocked = CausalZoneTracker(protocol)
    for index in range(protocol.maximum_fvg_states):
        entity_id = f"live-fvg-{index:03d}"
        blocked._fair_value_gaps[entity_id] = replace(created, fvg_id=entity_id)
        blocked._fvg_order.append(entity_id)
    with pytest.raises(RuntimeError, match="cannot admit"):
        blocked._admit_capacity(
            states=blocked._fair_value_gaps,
            order=blocked._fvg_order,
            maximum=protocol.maximum_fvg_states,
            terminal=blocked._is_fvg_terminal,
        )


def test_every_terminal_zone_a_frame_exposes_became_terminal_on_its_last_native_bar() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            structure_protocol="configs/primitives_structure_liquidity.json",
            liquidity_protocol="configs/primitives_structure_liquidity.json",
            displacement_protocol="configs/primitives_displacement.json",
            zone_protocol="configs/primitives_zones.json",
            scale_specs=MODEL_SCALE_SPECS,
            project_scene_graph=False,
        )
    )
    terminal_seen = 0
    for bar in _noisy(session_bars(2), seed=9):
        observation = observer.observe(reader.on_bar(bar))
        for timeframe, tracker in observer._zone_trackers.items():
            frame = observation.frames[timeframe]
            last_native = tracker._last_clock
            for state in frame.fair_value_gaps:
                if CausalZoneTracker._is_fvg_terminal(state):
                    terminal_seen += 1
                    terminal_at = (
                        state.mitigated_at
                        or state.invalidated_at
                        or state.expired_at
                    )
                    assert terminal_at == last_native, (timeframe, state.fvg_id)
            for state in frame.order_blocks:
                if CausalZoneTracker._is_order_block_terminal(state):
                    terminal_seen += 1
                    terminal_at = state.mitigated_at or state.failed_at
                    assert terminal_at == last_native, (timeframe, state.order_block_id)
    assert terminal_seen > 0
