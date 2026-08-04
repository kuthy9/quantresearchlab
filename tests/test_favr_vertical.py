from __future__ import annotations

from dataclasses import replace
import json

import pandas as pd

from smc_trader.decision import UtilityDecisionLayer
from smc_trader.group5 import CausalGroup5Reducer
from smc_trader.model import (
    Candle,
    DealingRangeLifecycle,
    Direction,
    EngineSnapshot,
    LiquidityInventoryLifecycle,
    ManipulationLifecycle,
    Playbook,
    PlaybookPhase,
    Timeframe,
    content_hash,
)
from smc_trader.playbook_registry import load_playbook_registry
from smc_trader.playbooks import PlaybookBrain
from smc_trader.risk import StructuralRiskEngine
from smc_trader.visualization import DecisionVisualizer, SealedVisualAudit

from .helpers import market_observation
from .test_v3_group4_primitives import (
    _m1 as _group4_m1,
    _mature_range,
    _protocol as _group4_protocol,
)
from .test_v4_typed_vertical import (
    ROOT,
    _execution,
    _fvg,
    _group5_protocol,
    _m1,
)


def _unparked_favr_brain() -> PlaybookBrain:
    registry = load_playbook_registry(
        ROOT / "configs/playbooks.json"
    )
    protocols = tuple(
        replace(
            protocol,
            status="typed_causal_implemented_test_only",
        )
        if protocol.playbook
        is Playbook.FAILED_AUCTION_VALUE_RETURN
        else protocol
        for protocol in registry.protocols
    )
    return PlaybookBrain(registry=replace(registry, protocols=protocols))


def _range_failed_auction():
    tracker, _, _ = _mature_range(_group4_protocol())
    inventory = tracker.snapshot().range_boundary_inventory
    for index in range(15):
        output = tracker.on_completed_update(
            _group4_m1(index),
            prior_inventory=inventory,
            liquidity_pools=(),
        )
        inventory = output.range_boundary_inventory

    swept = tracker.on_completed_update(
        _group4_m1(
            15,
            close=98.90,
            high=100.25,
            low=98.75,
        ),
        prior_inventory=inventory,
        liquidity_pools=(),
    )
    assert len(swept.manipulations) == 1
    assert (
        swept.manipulations[0].lifecycle
        is ManipulationLifecycle.SWEPT
    )

    reaccepted = tracker.on_completed_update(
        _group4_m1(
            16,
            close=99.50,
            high=100.00,
            low=99.25,
        ),
        prior_inventory=swept.range_boundary_inventory,
        liquidity_pools=(),
    )
    mature = next(
        state
        for state in reaccepted.dealing_ranges
        if state.lifecycle is DealingRangeLifecycle.MATURE
    )
    manipulation = reaccepted.manipulations[0]
    assert manipulation.lifecycle is ManipulationLifecycle.REACCEPTED
    assert manipulation.source_kind == "mature_range_boundary"
    return mature, manipulation, reaccepted.range_boundary_inventory


def _favr_observation():
    mature, manipulation, inventory = _range_failed_auction()
    reducer = CausalGroup5Reducer(_group5_protocol())

    formation = _m1(
        30,
        open_=100.00,
        high=100.50,
        low=100.00,
        close=100.25,
    )
    fvg = _fvg(
        formation.end,
        direction=Direction.LONG,
        identity="favr-return-fvg",
        lower=99.25,
        upper=99.75,
    )
    output = reducer.on_completed_1m(
        formation,
        fair_value_gaps=(fvg,),
        liquidity_inventory=inventory,
        m1_atr=1.0,
    )
    assert len(output.entry_locations) == 1

    first_pullback = _m1(
        31,
        open_=100.25,
        high=100.25,
        low=99.50,
        close=99.50,
    )
    reducer.on_completed_1m(
        first_pullback,
        fair_value_gaps=(fvg,),
        liquidity_inventory=inventory,
        m1_atr=1.0,
    )
    reclaim = _m1(
        32,
        open_=99.50,
        high=100.25,
        low=99.50,
        close=100.00,
    )
    reducer.on_completed_1m(
        reclaim,
        fair_value_gaps=(fvg,),
        liquidity_inventory=inventory,
        m1_atr=0.25,
    )
    hold = _m1(
        33,
        open_=100.00,
        high=100.25,
        low=99.75,
        close=100.00,
    )
    output = reducer.on_completed_1m(
        hold,
        fair_value_gaps=(fvg,),
        liquidity_inventory=inventory,
        m1_atr=0.25,
    )

    base = market_observation(asof=hold.end, price=hold.close)
    frames = dict(base.frames)
    frames[Timeframe.H1] = replace(
        frames[Timeframe.H1],
        cutoff=hold.end,
        dealing_ranges=(mature,),
    )
    frames[Timeframe.M5] = replace(
        frames[Timeframe.M5],
        cutoff=hold.end,
        fair_value_gaps=(fvg,),
    )
    frames[Timeframe.M1] = replace(
        frames[Timeframe.M1],
        cutoff=hold.end,
    )
    frames[Timeframe.H4] = replace(
        frames[Timeframe.H4],
        cutoff=hold.end,
    )
    observation = replace(
        base,
        frames=frames,
        execution=_execution(hold.end, cost=0.025),
        liquidity_inventory=inventory,
        liquidity_inventory_authoritative=True,
        manipulations=(manipulation,),
        group5_authoritative=True,
        entry_locations=output.entry_locations,
        qualified_reacceptances=output.qualified_reacceptances,
        micro_bos_references=output.micro_bos_references,
        path_sequences=output.path_sequences,
    )
    return observation, mature, manipulation


def _favr_causal_histories(observation):
    durations = {
        Timeframe.H4: 240,
        Timeframe.H1: 60,
        Timeframe.M5: 5,
        Timeframe.M1: 1,
    }
    histories = {}
    for timeframe, minutes in durations.items():
        end = observation.frame(timeframe).cutoff
        histories[timeframe] = (
            Candle(
                timeframe=timeframe,
                start=end - pd.Timedelta(minutes=minutes),
                end=end,
                open=observation.price,
                high=101.25,
                low=98.50,
                close=observation.price,
                volume=100.0,
                symbol=observation.symbol,
                instrument_id=observation.instrument_id,
                observed_minutes=minutes,
                expected_minutes=minutes,
                complete=True,
                real_minutes=minutes,
            ),
        )
    return histories


def test_favr_full_causal_chain_is_implemented_but_runtime_parked() -> None:
    observation, mature, manipulation = _favr_observation()
    belief = PlaybookBrain(
        registry=load_playbook_registry(
            ROOT / "configs/playbooks.json"
        )
    ).update(observation)
    hypothesis = belief.hypotheses[
        "failed_auction_value_return:long"
    ]

    assert hypothesis.phase is PlaybookPhase.INACTIVE
    assert hypothesis.plan is None
    assert hypothesis.sequence is not None
    assert hypothesis.sequence.complete
    assert all(hypothesis.hard_gate_results.values())
    assert hypothesis.context_id == mature.range_id
    assert hypothesis.episode_id == manipulation.manipulation_id
    assert hypothesis.draw_selection is not None
    assert (
        hypothesis.draw_selection.lifecycle
        is LiquidityInventoryLifecycle.TARGETED
    )


def test_unparked_favr_freezes_range_sweep_entry_and_opposite_draw() -> None:
    observation, mature, manipulation = _favr_observation()
    belief = _unparked_favr_brain().update(observation)
    hypothesis = belief.hypotheses[
        "failed_auction_value_return:long"
    ]

    assert hypothesis.phase is PlaybookPhase.EXECUTABLE
    assert hypothesis.plan is not None
    plan = hypothesis.plan
    context = plan.range_auction
    assert context is not None
    assert (
        context.lower_bound,
        context.upper_bound,
        context.midpoint,
        context.value_price,
    ) == (
        mature.lower_bound,
        mature.upper_bound,
        mature.midpoint,
        mature.value_price,
    )
    assert context.manipulation_id == manipulation.manipulation_id
    assert context.manipulation_extreme == manipulation.sweep_extreme
    assert context.reentry_price == manipulation.reentry_price
    assert plan.invalidation.price == manipulation.sweep_extreme
    assert plan.targets[0].price == mature.upper_bound
    assert plan.planned_entry == 99.75
    assert plan.planned_entry != observation.price
    assert plan.draw_selection is not None
    assert plan.draw_selection.source_kind == "range_boundary"
    assert (
        plan.draw_selection.selection_reason
        == "failed_auction_value_return:"
        "frozen_opposite_range_boundary"
    )

    decision = UtilityDecisionLayer().decide(observation, belief)
    assert decision.selected_action.value == "enter"
    risk = StructuralRiskEngine().review(decision, observation)
    assert risk.passed
    assert risk.final_action.value == "enter"
    assert risk.frozen_thesis is not None
    assert risk.frozen_thesis.range_auction == context
    assert risk.frozen_thesis.draw_selection == plan.draw_selection


def test_favr_missing_frozen_sources_closes_episode_instead_of_resetting() -> None:
    observation, _, _ = _favr_observation()
    brain = _unparked_favr_brain()
    first = brain.update(observation).hypotheses[
        "failed_auction_value_return:long"
    ]
    assert first.phase is PlaybookPhase.EXECUTABLE

    next_clock = observation.asof + pd.Timedelta(minutes=1)
    frames = dict(observation.frames)
    frames[Timeframe.H1] = replace(
        frames[Timeframe.H1],
        cutoff=next_clock,
        dealing_ranges=(),
    )
    for timeframe in (Timeframe.H4, Timeframe.M5, Timeframe.M1):
        frames[timeframe] = replace(
            frames[timeframe],
            cutoff=next_clock,
        )
    missing = replace(
        observation,
        asof=next_clock,
        frames=frames,
        execution=_execution(next_clock, cost=0.025),
        liquidity_inventory=(),
        manipulations=(),
    )
    terminal = brain.update(missing).hypotheses[
        "failed_auction_value_return:long"
    ]

    assert terminal.phase is PlaybookPhase.INVALIDATED
    assert terminal.terminal_at == next_clock
    assert terminal.terminal_reason == "frozen_dealing_range_missing"
    assert first.context_id in terminal.terminal_source_ids
    assert first.episode_id in terminal.terminal_source_ids


def test_favr_sealed_decision_visual_keeps_identity_and_future_hidden(
    tmp_path,
) -> None:
    observation, mature, manipulation = _favr_observation()
    belief = _unparked_favr_brain().update(observation)
    key = "failed_auction_value_return:long"
    hypothesis = belief.hypotheses[key]
    assert hypothesis.plan is not None
    plan = hypothesis.plan
    decision = UtilityDecisionLayer().decide(observation, belief)
    risk = StructuralRiskEngine().review(decision, observation)
    assert risk.passed
    snapshot = EngineSnapshot(
        observation=observation,
        belief=belief,
        decision=decision,
        risk=risk,
        snapshot_hash=content_hash(
            ("favr-visual-smoke", observation.asof, plan.setup_id)
        ),
    )

    audit = SealedVisualAudit.seal(
        DecisionVisualizer(),
        snapshot,
        _favr_causal_histories(observation),
        tmp_path / "favr-sealed",
        hypothesis_key=key,
    )

    assert audit.decision_artifact.path.is_file()
    assert audit.decision_artifact.path.stat().st_size > 0
    assert audit.decision_packet_path.is_file()
    assert not (audit.directory / "future_reveal.png").exists()
    assert audit.future_1m == []
    assert audit.decision_artifact.maximum_market_time <= observation.asof
    assert audit.permit.hypothesis_key == key
    assert audit.permit.setup_id == plan.setup_id
    assert audit.permit.entry_location_id == plan.entry_location_id
    assert audit.permit.entry_path_id == plan.entry_path_id

    packet = json.loads(
        audit.decision_packet_path.read_text(encoding="utf-8")
    )
    frozen = packet["belief_t"]["hypotheses"][key]["plan"]
    assert frozen["range_auction"]["range_id"] == mature.range_id
    assert (
        frozen["range_auction"]["manipulation_id"]
        == manipulation.manipulation_id
    )
    assert (
        frozen["invalidation"]["source_level_id"]
        == manipulation.manipulation_id
    )
    assert frozen["draw_selection"]["draw_id"] == plan.selected_draw_id
