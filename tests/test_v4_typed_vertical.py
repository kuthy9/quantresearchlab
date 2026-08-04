from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from smc_trader.calibration import (
    DimensionReliabilityMap,
    DimensionReliabilityPoint,
    TYPED_CALIBRATION_DIMENSIONS,
    TypedBrainCalibrator,
)
from smc_trader.decision_trace import (
    build_decision_trace,
    build_frozen_decision_packet,
)
from smc_trader.decision import UtilityDecisionLayer
from smc_trader.group5 import CausalGroup5Reducer, Group5Protocol
from smc_trader.model import (
    AccountState,
    BOSLifecycle,
    BOSScope,
    BreakOfStructureState,
    Candle,
    EngineSnapshot,
    Direction,
    ExecutionObservation,
    FairValueGapLifecycle,
    FairValueGapState,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityLevel,
    ManipulationLifecycle,
    ManipulationState,
    MarketObservation,
    Playbook,
    PlaybookPhase,
    PositionSnapshot,
    StructureLifecycle,
    StructureSequenceState,
    SwingLifecycle,
    SwingPoint,
    SwingSide,
    Timeframe,
    content_hash,
)
from smc_trader.playbook_registry import load_playbook_registry
from smc_trader.playbooks import PlaybookBrain
from smc_trader.risk import StructuralRiskEngine

from .helpers import market_observation


ROOT = Path(__file__).resolve().parents[1]
GROUP12_SHA = (
    "189b6af3bff631c3985fa37bcf9f5f82528296800886d9c9bd4cbe123ea4c701"
)
GROUP3_SHA = (
    "4086ed67c7fe849e175c149bca8688749ef44d6649a672736535e1fec2d18c51"
)
GROUP4_SHA = (
    "14b049facadb815c3fdc0d134275ee3f1efb3ca7c775a5f18ec4efad61663bc5"
)
DISPLACEMENT_SHA = "a" * 64
BASE = pd.Timestamp("2025-01-07T09:30:00-05:00")


def _group5_protocol() -> Group5Protocol:
    return Group5Protocol.from_file(
        ROOT / "configs/smc_primitives_v3_group5.json"
    )


def _brain() -> PlaybookBrain:
    return PlaybookBrain(
        registry=load_playbook_registry(
            ROOT / "configs/playbooks_v4.json"
        )
    )


def _mapped_brain() -> PlaybookBrain:
    registry = load_playbook_registry(
        ROOT / "configs/playbooks_v4.json"
    )
    maps = {}
    for playbook in (
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
    ):
        dimensions = {}
        for dimension in TYPED_CALIBRATION_DIMENSIONS:
            low, high = (
                (0.1, 0.9)
                if dimension == "thesis_strength"
                else (0.0, 1.0)
            )
            dimensions[dimension] = DimensionReliabilityMap(
                playbook=playbook,
                dimension=dimension,
                episodes=100,
                points=(
                    DimensionReliabilityPoint(0.0, low, 50),
                    DimensionReliabilityPoint(1.0, high, 50),
                ),
            )
        maps[playbook] = dimensions
    return PlaybookBrain(
        registry=registry,
        calibrator=TypedBrainCalibrator(
            version="typed-test-ready",
            fingerprint="f" * 64,
            registry_hash=registry.fingerprint,
            maps=maps,
            status="ready",
            primitive_protocol_hashes={},
            brain_input_contract_hash="e" * 64,
        ),
    )


def _m1(
    index: int,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> Candle:
    start = BASE + pd.Timedelta(minutes=index)
    return Candle(
        timeframe=Timeframe.M1,
        start=start,
        end=start + pd.Timedelta(minutes=1),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100.0,
        symbol="NQH5",
        instrument_id=1,
        observed_minutes=1,
        expected_minutes=1,
        complete=True,
        real_minutes=1,
        synthetic_minutes=0,
    )


def _fvg(
    confirmed_at: pd.Timestamp,
    *,
    direction: Direction,
    identity: str,
    lower: float,
    upper: float,
) -> FairValueGapState:
    starts = (
        confirmed_at - pd.Timedelta(minutes=15),
        confirmed_at - pd.Timedelta(minutes=10),
        confirmed_at - pd.Timedelta(minutes=5),
    )
    return FairValueGapState(
        fvg_id=identity,
        protocol_hash=GROUP3_SHA,
        symbol="NQH5",
        instrument_id=1,
        timeframe=Timeframe.M5,
        direction=direction,
        lifecycle=FairValueGapLifecycle.OPEN,
        source_displacement_id=f"displacement:{identity}",
        source_active_transition_id=f"active:{identity}",
        source_displacement_protocol_hash=DISPLACEMENT_SHA,
        source_displacement_started_at=starts[1],
        source_displacement_active_at=starts[2],
        source_displacement_prefix_commitment=f"prefix:{identity}",
        source_candle_ids=(
            f"{identity}:one",
            f"{identity}:two",
            f"{identity}:three",
        ),
        source_candle_starts=starts,
        lower_bound=lower,
        upper_bound=upper,
        midpoint=(lower + upper) / 2.0,
        invalidation_price=(
            lower if direction is Direction.LONG else upper
        ),
        width_points=upper - lower,
        width_ticks=round((upper - lower) / 0.25),
        width_atr=upper - lower,
        strength=0.8,
        formed_at=confirmed_at,
        confirmed_at=confirmed_at,
        state_started_at=confirmed_at,
        last_updated_at=confirmed_at,
        age_bars=0,
        max_fill_fraction=0.0,
    )


def _inventory_draw(
    *,
    identity: str,
    timeframe: Timeframe,
    side: str,
    price: float,
    confirmed_at: pd.Timestamp,
    source_id: str,
    age_bars: int = 1,
    strength: float = 0.7,
) -> LiquidityInventoryItem:
    return LiquidityInventoryItem(
        item_id=identity,
        timeframe=timeframe,
        side=side,
        kind="swing",
        price=price,
        lower_bound=price,
        upper_bound=price,
        formed_at=confirmed_at - pd.Timedelta(minutes=5),
        confirmed_at=confirmed_at,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=(source_id,),
        age_bars=age_bars,
        strength=strength,
    )


def _execution(asof: pd.Timestamp, *, cost: float) -> ExecutionObservation:
    return ExecutionObservation(
        spread_points=0.25,
        expected_slippage_points=0.0,
        expected_round_trip_cost_points=cost,
        minutes_to_deadline=60,
        fillability=0.9,
        data_age_seconds=0.0,
        size_available=10.0,
        anomalies=(),
        source="synthetic_observed_execution",
        bid=100.0,
        ask=100.25,
        bid_size=10.0,
        ask_size=10.0,
        depth_imbalance=0.0,
    )


def _h4_structure(clock: pd.Timestamp) -> StructureSequenceState:
    return StructureSequenceState(
        structure_id="h4-long-structure",
        timeframe=Timeframe.H4,
        direction=Direction.LONG,
        lifecycle=StructureLifecycle.CONFIRMED,
        formed_at=clock - pd.Timedelta(hours=1),
        confirmed_at=clock,
        broken_at=None,
        high_run=2,
        low_run=2,
        sequence_count=2,
        latest_high_id="h4-high",
        latest_low_id="h4-low",
        protected_swing_id="h4-low",
        protected_price=97.0,
        cumulative_magnitude_atr=1.5,
        age_bars=1,
    )


def _continuation_bos(clock: pd.Timestamp) -> BreakOfStructureState:
    return BreakOfStructureState(
        bos_id="h1-long-continuation",
        timeframe=Timeframe.H1,
        direction=Direction.LONG,
        lifecycle=BOSLifecycle.CONFIRMED,
        scope=BOSScope.CONTINUATION,
        target_swing_id="h1-prior-high",
        source_structure_id="h1-long-structure",
        target_price=100.5,
        target_ticks=402,
        pending_at=clock - pd.Timedelta(hours=1),
        resolved_at=clock,
        age_bars=1,
        strength=0.8,
    )


def _dfp_observation(
    *,
    asof: pd.Timestamp,
    price: float,
    output,
    fvg: FairValueGapState,
    draw: LiquidityInventoryItem,
) -> MarketObservation:
    base = market_observation(asof=asof, price=price)
    target = LiquidityLevel(
        level_id=draw.item_id,
        timeframe=Timeframe.H4,
        side=draw.side,
        price=draw.price,
        formed_at=draw.formed_at,
        confirmed_at=draw.confirmed_at,
        touches=0,
    )
    frames = dict(base.frames)
    frames[Timeframe.H4] = replace(
        frames[Timeframe.H4],
        cutoff=asof,
        liquidity=(target,),
        swings=(
            SwingPoint(
                swing_id=draw.source_ids[0],
                timeframe=Timeframe.H4,
                symbol=base.symbol,
                instrument_id=base.instrument_id,
                side=(
                    SwingSide.HIGH
                    if draw.side == "above"
                    else SwingSide.LOW
                ),
                price=draw.price,
                price_ticks=round(draw.price / 0.25),
                pivot_start=draw.formed_at - pd.Timedelta(hours=4),
                pivot_end=draw.formed_at,
                observed_at=draw.confirmed_at,
                confirmed_at=draw.confirmed_at,
                lifecycle=SwingLifecycle.CONFIRMED,
                magnitude_atr=draw.strength,
                age_bars=draw.age_bars,
            ),
        ),
        structures=(
            _h4_structure(BASE - pd.Timedelta(hours=4)),
        ),
    )
    frames[Timeframe.H1] = replace(
        frames[Timeframe.H1],
        cutoff=asof,
        structure_breaks=(
            _continuation_bos(BASE - pd.Timedelta(hours=2)),
        ),
    )
    frames[Timeframe.M5] = replace(
        frames[Timeframe.M5],
        cutoff=asof,
        fair_value_gaps=(fvg,),
    )
    frames[Timeframe.M1] = replace(
        frames[Timeframe.M1],
        cutoff=asof,
    )
    return replace(
        base,
        frames=frames,
        execution=_execution(asof, cost=0.10),
        liquidity_inventory=(draw,),
        liquidity_inventory_authoritative=True,
        group5_authoritative=True,
        entry_locations=output.entry_locations,
        qualified_reacceptances=output.qualified_reacceptances,
        micro_bos_references=output.micro_bos_references,
        path_sequences=output.path_sequences,
    )


def _dfp_fixture():
    reducer = CausalGroup5Reducer(_group5_protocol())
    formation = _m1(
        0,
        open_=101.0,
        high=101.25,
        low=100.5,
        close=101.0,
    )
    fvg = _fvg(
        formation.end,
        direction=Direction.LONG,
        identity="dfp-fvg",
        lower=99.0,
        upper=100.0,
    )
    draw = _inventory_draw(
        identity="swing:dfp-h4-draw",
        timeframe=Timeframe.H4,
        side="above",
        price=103.0,
        confirmed_at=BASE - pd.Timedelta(hours=3),
        source_id="dfp-h4-draw",
    )
    forming = reducer.on_completed_1m(
        formation,
        fair_value_gaps=(fvg,),
        liquidity_inventory=(draw,),
        m1_atr=1.0,
    )
    pullback = _m1(
        1,
        open_=100.5,
        high=100.75,
        low=99.5,
        close=100.5,
    )
    triggered = reducer.on_completed_1m(
        pullback,
        fair_value_gaps=(fvg,),
        liquidity_inventory=(draw,),
        m1_atr=1.0,
    )
    return (
        reducer,
        fvg,
        draw,
        _dfp_observation(
            asof=formation.end,
            price=formation.close,
            output=forming,
            fvg=fvg,
            draw=draw,
        ),
        _dfp_observation(
            asof=pullback.end,
            price=pullback.close,
            output=triggered,
            fvg=fvg,
            draw=draw,
        ),
    )


def _advance_observation(
    observation: MarketObservation,
    asof: pd.Timestamp,
    *,
    price: float | None = None,
    execution: ExecutionObservation | None = None,
) -> MarketObservation:
    return replace(
        observation,
        asof=asof,
        price=observation.price if price is None else price,
        frames={
            timeframe: replace(frame, cutoff=asof)
            for timeframe, frame in observation.frames.items()
        },
        execution=(
            _execution(asof, cost=0.10)
            if execution is None
            else execution
        ),
    )


def _snapshot(
    observation: MarketObservation,
    belief,
    marker: str,
) -> EngineSnapshot:
    decision = UtilityDecisionLayer().decide(observation, belief)
    return EngineSnapshot(
        observation=observation,
        belief=belief,
        decision=decision,
        risk=StructuralRiskEngine().review(decision, observation),
        snapshot_hash=marker * 64,
    )


def test_dfp_vertical_chain_uses_exact_group5_plan_and_risk_binding() -> None:
    _, _, _, forming, triggered = _dfp_fixture()
    brain = _brain()

    first = brain.update(forming)
    first_dfp = first.hypotheses[
        "displacement_first_pullback:long"
    ]
    assert first_dfp.phase is PlaybookPhase.WAITING_LOCATION
    assert first_dfp.deliverable_targets[0].level_id == "swing:dfp-h4-draw"
    assert first_dfp.plan is not None

    belief = brain.update(triggered)
    hypothesis = belief.hypotheses[
        "displacement_first_pullback:long"
    ]
    assert hypothesis.phase is PlaybookPhase.EXECUTABLE
    assert hypothesis.sequence is not None
    assert hypothesis.sequence.complete
    assert all(hypothesis.hard_gate_results.values())
    assert hypothesis.plan is not None
    assert hypothesis.plan.setup_id == hypothesis.plan.entry_path_id
    assert (
        hypothesis.plan.entry_location_id
        == hypothesis.entry_location_id
    )
    assert hypothesis.plan.selected_draw_id == "swing:dfp-h4-draw"

    decision = UtilityDecisionLayer().decide(triggered, belief)
    assert decision.selected_action.value == "enter"
    risk = StructuralRiskEngine().review(decision, triggered)
    assert risk.passed
    assert risk.final_action.value == "enter"
    assert risk.frozen_thesis is not None
    assert (
        risk.frozen_thesis.entry_path_id
        == hypothesis.plan.entry_path_id
    )

    orphaned_frames = dict(triggered.frames)
    orphaned_frames[Timeframe.M5] = replace(
        orphaned_frames[Timeframe.M5],
        fair_value_gaps=(),
    )
    orphaned_zone = replace(triggered, frames=orphaned_frames)
    orphaned_zone_risk = StructuralRiskEngine().review(
        decision,
        orphaned_zone,
    )
    assert not orphaned_zone_risk.passed
    assert orphaned_zone_risk.final_action.value == "abstain"

    missing_execution = replace(
        triggered,
        execution=replace(
            triggered.execution,
            source="unknown",
        ),
    )
    vetoed = StructuralRiskEngine().review(
        decision,
        missing_execution,
    )
    assert not vetoed.passed
    assert vetoed.final_action.value == "abstain"


def test_typed_decision_trace_and_frozen_packet_are_causal_and_complete() -> None:
    _, _, _, forming, triggered = _dfp_fixture()
    brain = _brain()
    prior_belief = brain.update(forming)
    current_belief = brain.update(triggered)
    decision_layer = UtilityDecisionLayer()
    risk_engine = StructuralRiskEngine()
    prior_decision = decision_layer.decide(forming, prior_belief)
    current_decision = decision_layer.decide(triggered, current_belief)
    prior = EngineSnapshot(
        observation=forming,
        belief=prior_belief,
        decision=prior_decision,
        risk=risk_engine.review(prior_decision, forming),
        snapshot_hash="1" * 64,
    )
    current = EngineSnapshot(
        observation=triggered,
        belief=current_belief,
        decision=current_decision,
        risk=risk_engine.review(current_decision, triggered),
        snapshot_hash="2" * 64,
    )
    source_bar = _m1(
        1,
        open_=100.5,
        high=100.75,
        low=99.5,
        close=100.5,
    )
    account = AccountState(equity=100_000.0)
    trace = build_decision_trace(
        current,
        prior,
        source_bar=source_bar,
        account_state=account,
    )

    key = "displacement_first_pullback:long"
    assert trace["future_path_included"] is False
    assert trace["prior_belief_asof"] == forming.asof.isoformat()
    assert trace["observation"]["source_bar"]["end"] == (
        triggered.asof.isoformat()
    )
    assert trace["belief_delta"][key]["phase_from"] == "waiting_location"
    assert trace["belief_delta"][key]["phase_to"] == "executable"
    assert set(trace["belief_t"][key]["qualities"]) == {
        "thesis_strength",
        "sequence_progress",
        "location_quality",
        "entry_readiness",
        "delivery_quality",
        "uncertainty",
    }
    assert set(trace["belief_t"][key]["raw_qualities"]) == {
        "thesis_strength",
        "sequence_progress",
        "location_quality",
        "entry_readiness",
        "delivery_quality",
        "uncertainty",
    }
    assert set(trace["belief_t"][key]["evidence_groups"]) == {
        "structure",
        "displacement",
        "location",
        "liquidity",
        "trigger",
        "execution",
    }
    summary = trace["belief_t"][key]
    hypothesis = current_belief.hypotheses[key]
    assert summary["context_id"] == hypothesis.context_id
    assert summary["episode_id"] == hypothesis.episode_id
    assert (
        summary["initiating_event_id"]
        == hypothesis.initiating_event_id
    )
    assert (
        summary["evidence_revision_id"]
        == hypothesis.evidence_revision_id
    )
    assert summary["terminal_at"] is None
    assert summary["terminal_reason"] is None
    assert summary["terminal_source_ids"] == []
    assert summary["eligible"] is hypothesis.eligible
    assert (
        summary["effective_probability"]
        == hypothesis.effective_probability
    )
    assert trace["belief_delta"][key]["episode_id_from"] == (
        prior_belief.hypotheses[key].episode_id
    )
    assert trace["belief_delta"][key]["episode_id_to"] == (
        hypothesis.episode_id
    )
    assert trace["belief_t"][key]["hard_gates"]
    evidence_rows = (
        trace["belief_t"][key]["supporting"]
        + trace["belief_t"][key]["contradicting"]
    )
    assert evidence_rows
    assert all(
        item["reason"]
        for item in evidence_rows
    )
    assert trace["decision"]["selected_action"] == "enter"
    assert trace["risk"]["final_action"] == "enter"
    assert {
        item["family"] for item in trace["typed_state_transitions"]
    }.intersection({"entry_location", "entry_path"})
    path_transitions = [
        item
        for item in trace["typed_state_transitions"]
        if item["family"] == "entry_path"
    ]
    assert path_transitions
    assert all(
        item["state"]["sequence_id"] == item["entity_id"]
        for item in path_transitions
    )
    assert not any(
        item.get("lifecycle") == "closed"
        and item.get("transition_reason")
        in {
            "micro_bos_aligned",
            "pool_reversal_sequence_observed",
            "qualified_reacceptance_held",
            "zone_rejection_observed",
        }
        for item in trace["events_invalidated"]
    )

    histories = {
        Timeframe.H4: (
            Candle(
                timeframe=Timeframe.H4,
                start=triggered.asof - pd.Timedelta(hours=4),
                end=triggered.asof,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=400.0,
                symbol="NQH5",
                instrument_id=1,
                observed_minutes=240,
                expected_minutes=240,
                complete=True,
            ),
        ),
        Timeframe.H1: (
            Candle(
                timeframe=Timeframe.H1,
                start=triggered.asof - pd.Timedelta(hours=1),
                end=triggered.asof,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=200.0,
                symbol="NQH5",
                instrument_id=1,
                observed_minutes=60,
                expected_minutes=60,
                complete=True,
            ),
        ),
        Timeframe.M5: (
            Candle(
                timeframe=Timeframe.M5,
                start=triggered.asof - pd.Timedelta(minutes=5),
                end=triggered.asof,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=150.0,
                symbol="NQH5",
                instrument_id=1,
                observed_minutes=5,
                expected_minutes=5,
                complete=True,
            ),
        ),
        Timeframe.M1: (source_bar,),
    }
    packet = build_frozen_decision_packet(
        current,
        histories,
        prior,
        hypothesis_key=key,
        source_bar=source_bar,
        account_state=account,
    )
    claimed_hash = packet["packet_hash"]
    payload = dict(packet)
    payload.pop("packet_hash")
    assert claimed_hash == content_hash(payload)
    assert packet["maximum_market_time"] == triggered.asof.isoformat()
    assert packet["future_path"] == {
        "included": False,
        "revealed": False,
        "storage": "physically_separate_artifact",
    }
    assert packet["decision_trace"]["decision_hash"] == current.snapshot_hash

    reset_observation = replace(
        triggered,
        anomalies=tuple(
            sorted(
                {
                    *triggered.anomalies,
                    "data_gap_history_reset",
                }
            )
        ),
    )
    reset_current = replace(current, observation=reset_observation)
    reset_trace = build_decision_trace(
        reset_current,
        prior,
        source_bar=source_bar,
        account_state=account,
    )
    assert reset_trace["brain_reset_before_update"] is True
    assert reset_trace["belief_t_minus_1"] is None
    assert reset_trace["prior_belief_asof"] is None
    assert reset_trace["discarded_pre_reset_belief_asof"] == (
        forming.asof.isoformat()
    )
    assert all(
        item["initialized"]
        for item in reset_trace["belief_delta"].values()
    )
    reset_packet = build_frozen_decision_packet(
        reset_current,
        histories,
        prior,
        hypothesis_key=key,
        source_bar=source_bar,
        account_state=account,
    )
    assert reset_packet["brain_reset_before_update"] is True
    assert reset_packet["belief_t_minus_1"] is None
    assert reset_packet["belief_discarded_before_reset"]["asof"] == (
        forming.asof.isoformat()
    )


def test_dfp_freezes_draw_and_does_not_retarget_after_setup() -> None:
    reducer = CausalGroup5Reducer(_group5_protocol())
    formation = _m1(
        0,
        open_=101.0,
        high=101.25,
        low=100.5,
        close=101.0,
    )
    fvg = _fvg(
        formation.end,
        direction=Direction.LONG,
        identity="dfp-fvg",
        lower=99.0,
        upper=100.0,
    )
    draw = _inventory_draw(
        identity="swing:dfp-h4-draw",
        timeframe=Timeframe.H4,
        side="above",
        price=103.0,
        confirmed_at=BASE - pd.Timedelta(hours=3),
        source_id="dfp-h4-draw",
    )
    forming_output = reducer.on_completed_1m(
        formation,
        fair_value_gaps=(fvg,),
        liquidity_inventory=(draw,),
        m1_atr=1.0,
    )
    forming = _dfp_observation(
        asof=formation.end,
        price=formation.close,
        output=forming_output,
        fvg=fvg,
        draw=draw,
    )
    brain = _brain()
    first = brain.update(forming)
    initial = first.hypotheses[
        "displacement_first_pullback:long"
    ]
    assert (
        initial.deliverable_targets[0].level_id
        == draw.item_id
    )

    replacement_draw = replace(
        draw,
        item_id="swing:replacement-h4-draw",
        source_ids=("replacement-h4-draw",),
        price=104.0,
        lower_bound=104.0,
        upper_bound=104.0,
    )
    no_touch = _m1(
        1,
        open_=101.0,
        high=101.5,
        low=100.5,
        close=101.25,
    )
    output = reducer.on_completed_1m(
        no_touch,
        fair_value_gaps=(fvg,),
        liquidity_inventory=(replacement_draw,),
        m1_atr=1.0,
    )
    observation = _dfp_observation(
        asof=no_touch.end,
        price=no_touch.close,
        output=output,
        fvg=fvg,
        draw=replacement_draw,
    )
    updated = brain.update(observation).hypotheses[
        "displacement_first_pullback:long"
    ]

    assert updated.phase is PlaybookPhase.INVALIDATED
    assert updated.deliverable_targets == ()
    assert updated.plan is None
    assert not updated.hard_gate_results["h4_structure_and_draw"]
    assert updated.setup_context_id == initial.setup_context_id
    assert updated.context_id == initial.context_id
    assert updated.terminal_at == observation.asof
    assert (
        updated.terminal_reason
        == "context_draw_consumed_or_missing"
    )
    assert initial.setup_context_id in updated.terminal_source_ids


def test_typed_thesis_updates_once_per_evidence_revision_and_not_per_minute() -> None:
    _, _, _, forming, _ = _dfp_fixture()
    without_h1_frames = dict(forming.frames)
    without_h1_frames[Timeframe.H1] = replace(
        without_h1_frames[Timeframe.H1],
        structure_breaks=(),
    )
    without_h1 = replace(forming, frames=without_h1_frames)
    brain = _brain()
    key = "displacement_first_pullback:long"

    initial = brain.update(without_h1).hypotheses[key]
    assert initial.thesis_strength == 0.0

    revised_at = forming.asof + pd.Timedelta(minutes=1)
    unavailable_execution = replace(
        _execution(revised_at, cost=0.10),
        source="unknown",
        anomalies=("spread_missing_used_one_tick",),
    )
    revised_observation = _advance_observation(
        forming,
        revised_at,
        execution=unavailable_execution,
    )
    revised = brain.update(revised_observation).hypotheses[key]
    assert revised.thesis_strength == pytest.approx(0.7)
    assert (
        revised.raw_quality_dimensions["thesis_strength"]
        == pytest.approx(0.7)
    )
    assert revised.evidence_revision_id != initial.evidence_revision_id
    assert 0.0 < revised.uncertainty < 1.0
    assert revised.evidence_group_scores["execution"] == 0.0

    repeated = brain.update(
        _advance_observation(
            revised_observation,
            revised_at + pd.Timedelta(minutes=1),
            execution=_execution(
                revised_at + pd.Timedelta(minutes=1),
                cost=0.10,
            ),
        )
    ).hypotheses[key]
    assert repeated.evidence_revision_id == revised.evidence_revision_id
    assert repeated.thesis_strength == revised.thesis_strength
    assert repeated.uncertainty == revised.uncertainty
    assert repeated.evidence_group_scores["execution"] > 0.0


def test_dfp_new_confirmed_swing_updates_thesis_once_within_context() -> None:
    _, _, _, forming, _ = _dfp_fixture()
    low_structure = replace(
        forming.frames[Timeframe.H4].structures[0],
        latest_high_id="h4-high-revision-1",
        cumulative_magnitude_atr=0.8,
    )
    initial_frames = dict(forming.frames)
    initial_frames[Timeframe.H4] = replace(
        initial_frames[Timeframe.H4],
        structures=(low_structure,),
    )
    initial_observation = replace(forming, frames=initial_frames)
    brain = _brain()
    key = "displacement_first_pullback:long"

    initial = brain.update(initial_observation).hypotheses[key]
    assert initial.raw_quality_dimensions["thesis_strength"] == pytest.approx(
        0.4
    )

    revised_at = forming.asof + pd.Timedelta(minutes=1)
    revised_structure = replace(
        low_structure,
        latest_high_id="h4-high-revision-2",
        cumulative_magnitude_atr=1.6,
    )
    revised_observation = _advance_observation(
        initial_observation,
        revised_at,
    )
    revised_frames = dict(revised_observation.frames)
    revised_frames[Timeframe.H4] = replace(
        revised_frames[Timeframe.H4],
        structures=(revised_structure,),
    )
    revised_observation = replace(
        revised_observation,
        frames=revised_frames,
    )
    revised = brain.update(revised_observation).hypotheses[key]

    assert revised.context_id == initial.context_id
    assert revised.evidence_revision_id != initial.evidence_revision_id
    assert revised.raw_quality_dimensions["thesis_strength"] == pytest.approx(
        0.7
    )

    repeated = brain.update(
        _advance_observation(
            revised_observation,
            revised_at + pd.Timedelta(minutes=1),
        )
    ).hypotheses[key]
    assert repeated.evidence_revision_id == revised.evidence_revision_id
    assert repeated.raw_quality_dimensions["thesis_strength"] == pytest.approx(
        0.7
    )


def test_clock_authority_affects_market_uncertainty_not_execution_group() -> None:
    _, _, _, forming, _ = _dfp_fixture()
    observation = replace(
        forming,
        anomalies=("clock_incomplete_timeline",),
    )
    hypothesis = _brain().update(observation).hypotheses[
        "displacement_first_pullback:long"
    ]

    assert hypothesis.uncertainty == 1.0
    assert hypothesis.evidence_group_scores["execution"] > 0.0


def test_typed_calibration_maps_outputs_without_feeding_back_into_raw_belief() -> None:
    _, _, _, forming, triggered = _dfp_fixture()
    brain = _mapped_brain()
    key = "displacement_first_pullback:long"

    brain.update(forming)
    calibrated = brain.update(triggered).hypotheses[key]
    raw_thesis = calibrated.raw_quality_dimensions["thesis_strength"]
    assert raw_thesis == pytest.approx(0.7)
    assert calibrated.thesis_strength == pytest.approx(
        0.1 + raw_thesis * 0.8
    )
    assert calibrated.raw_probability == raw_thesis
    assert calibrated.probability == calibrated.thesis_strength
    assert calibrated.calibration_version == "typed-test-ready"
    assert calibrated.calibration_hash == "f" * 64
    assert calibrated.phase is PlaybookPhase.EXECUTABLE

    repeated = brain.update(
        _advance_observation(
            triggered,
            triggered.asof + pd.Timedelta(minutes=1),
        )
    ).hypotheses[key]
    assert repeated.evidence_revision_id == calibrated.evidence_revision_id
    assert repeated.raw_quality_dimensions == calibrated.raw_quality_dimensions
    assert repeated.thesis_strength == calibrated.thesis_strength

    parked = brain.current.hypotheses[
        "failed_auction_value_return:long"
    ]
    assert parked.phase is PlaybookPhase.INACTIVE


def test_terminal_episode_is_immutable_and_a_newer_episode_can_rearm() -> None:
    _, _, draw, forming, triggered = _dfp_fixture()
    brain = _brain()
    key = "displacement_first_pullback:long"
    brain.update(forming)
    active_belief = brain.update(triggered)
    active = active_belief.hypotheses[key]
    assert active.phase is PlaybookPhase.EXECUTABLE
    assert active.episode_id is not None
    assert active.plan is not None

    terminal_observation = _advance_observation(
        triggered,
        triggered.asof + pd.Timedelta(minutes=1),
        price=active.plan.invalidation.price - 0.25,
    )
    terminal_belief = brain.update(terminal_observation)
    terminal = terminal_belief.hypotheses[key]
    assert terminal.phase is PlaybookPhase.INVALIDATED
    assert terminal.terminal_at == terminal_observation.asof
    assert terminal.terminal_reason == "frozen_invalidation_breached"
    assert terminal.episode_id == active.episode_id
    assert terminal.setup_context_id == active.setup_context_id
    assert active.plan.invalidation.source_level_id in (
        terminal.terminal_source_ids
    )
    assert not terminal.eligible
    assert terminal.effective_probability == 0.0

    terminal_trace = build_decision_trace(
        _snapshot(terminal_observation, terminal_belief, "t"),
        _snapshot(triggered, active_belief, "a"),
        hypothesis_key=key,
    )
    terminal_summary = terminal_trace["belief_t"][key]
    terminal_delta = terminal_trace["belief_delta"][key]
    assert terminal_summary["episode_id"] == active.episode_id
    assert terminal_summary["eligible"] is False
    assert terminal_summary["effective_probability"] == 0.0
    assert terminal_summary["terminal_at"] == (
        terminal_observation.asof.isoformat()
    )
    assert terminal_summary["terminal_reason"] == (
        "frozen_invalidation_breached"
    )
    assert terminal_delta["episode_changed"] is False
    assert terminal_delta["terminal_from"] is None
    assert terminal_delta["terminal_to"]["reason"] == (
        "frozen_invalidation_breached"
    )
    assert terminal_delta["terminal_changed"] is True

    unchanged_observation = _advance_observation(
        triggered,
        terminal_observation.asof + pd.Timedelta(minutes=5),
    )
    unchanged_belief = brain.update(unchanged_observation)
    unchanged = unchanged_belief.hypotheses[key]
    assert unchanged is terminal

    rearm_bar = _m1(
        10,
        open_=101.0,
        high=101.25,
        low=100.5,
        close=101.0,
    )
    rearm_fvg = _fvg(
        rearm_bar.end,
        direction=Direction.LONG,
        identity="dfp-fvg-rearm",
        lower=99.0,
        upper=100.0,
    )
    rearm_output = CausalGroup5Reducer(
        _group5_protocol()
    ).on_completed_1m(
        rearm_bar,
        fair_value_gaps=(rearm_fvg,),
        liquidity_inventory=(draw,),
        m1_atr=1.0,
    )
    rearmed_observation = _dfp_observation(
        asof=rearm_bar.end,
        price=rearm_bar.close,
        output=rearm_output,
        fvg=rearm_fvg,
        draw=draw,
    )
    rearmed_belief = brain.update(rearmed_observation)
    rearmed = rearmed_belief.hypotheses[key]
    assert rearmed.phase is PlaybookPhase.WAITING_LOCATION
    assert rearmed.episode_id is not None
    assert rearmed.episode_id != terminal.episode_id
    assert rearmed.context_id == terminal.context_id
    assert rearmed.terminal_at is None
    assert rearmed.eligible

    rearm_trace = build_decision_trace(
        _snapshot(rearmed_observation, rearmed_belief, "r"),
        _snapshot(unchanged_observation, unchanged_belief, "u"),
        hypothesis_key=key,
    )
    rearm_delta = rearm_trace["belief_delta"][key]
    assert rearm_delta["episode_id_from"] == terminal.episode_id
    assert rearm_delta["episode_id_to"] == rearmed.episode_id
    assert rearm_delta["episode_changed"] is True
    assert rearm_delta["terminal_from"]["reason"] == (
        "frozen_invalidation_breached"
    )
    assert rearm_delta["terminal_to"] is None
    assert rearm_delta["terminal_changed"] is True


def test_historical_entry_trigger_does_not_weaken_a_filled_dfp() -> None:
    _, fvg, draw, forming, triggered = _dfp_fixture()
    brain = _brain()
    brain.update(forming)
    entered_belief = brain.update(triggered)
    hypothesis = entered_belief.hypotheses[
        "displacement_first_pullback:long"
    ]
    assert hypothesis.plan is not None
    later = triggered.asof + pd.Timedelta(minutes=1)
    later_frames = {
        timeframe: replace(frame, cutoff=later)
        for timeframe, frame in triggered.frames.items()
    }
    later_observation = replace(
        triggered,
        asof=later,
        price=100.75,
        frames=later_frames,
        execution=_execution(later, cost=0.10),
    )
    position = PositionSnapshot(
        thesis_hash="filled-dfp",
        symbol=triggered.symbol,
        instrument_id=triggered.instrument_id,
        playbook=hypothesis.playbook,
        direction=hypothesis.direction,
        entry_price=hypothesis.plan.planned_entry,
        original_invalidation=hypothesis.plan.invalidation,
        current_stop=hypothesis.plan.invalidation.price,
        primary_target=hypothesis.plan.targets[0],
        opened_at=triggered.asof,
        deadline=hypothesis.plan.deadline,
        quantity=1,
        unrealized_R=0.75,
        elapsed_minutes=1,
        mfe_R=0.75,
        mae_R=0.0,
        setup_id=hypothesis.plan.setup_id,
        entry_location_id=hypothesis.plan.entry_location_id,
        entry_path_id=hypothesis.plan.entry_path_id,
    )

    updated = brain.update(
        later_observation,
        position=position,
    ).hypotheses["displacement_first_pullback:long"]
    assert updated.phase is PlaybookPhase.DELIVERING


def test_filled_dfp_keeps_typed_position_identity_when_sources_disappear() -> None:
    _, _, _, forming, triggered = _dfp_fixture()
    brain = _brain()
    brain.update(forming)
    entered_belief = brain.update(triggered)
    entered = entered_belief.hypotheses[
        "displacement_first_pullback:long"
    ]
    assert entered.plan is not None
    plan = entered.plan
    position = PositionSnapshot(
        thesis_hash="filled-dfp",
        symbol=triggered.symbol,
        instrument_id=triggered.instrument_id,
        playbook=entered.playbook,
        direction=entered.direction,
        entry_price=plan.planned_entry,
        original_invalidation=plan.invalidation,
        current_stop=plan.invalidation.price,
        primary_target=plan.targets[0],
        opened_at=triggered.asof,
        deadline=plan.deadline,
        quantity=1,
        unrealized_R=0.75,
        elapsed_minutes=1,
        mfe_R=0.75,
        mae_R=0.0,
        setup_id=plan.setup_id,
        entry_location_id=plan.entry_location_id,
        entry_path_id=plan.entry_path_id,
    )
    later = triggered.asof + pd.Timedelta(minutes=1)
    missing_sources = replace(
        _advance_observation(triggered, later, price=100.75),
        entry_locations=(),
        qualified_reacceptances=(),
        micro_bos_references=(),
        path_sequences=(),
    )

    belief = brain.update(
        missing_sources,
        position=position,
    )
    updated = belief.hypotheses[
        "displacement_first_pullback:long"
    ]
    assert updated.phase is PlaybookPhase.WEAKENING
    assert updated.setup_context_id == plan.setup_id
    assert updated.entry_location_id == plan.entry_location_id
    assert updated.plan == plan
    decision_layer = UtilityDecisionLayer()
    decision = decision_layer.decide(
        missing_sources,
        belief,
        AccountState(equity=100_000.0, position=position),
    )
    hold = next(
        item for item in decision.utilities if item.action.value == "hold"
    )
    # The vanished trigger legitimately reduces delivery and raises missing-
    # evidence uncertainty, while the matching frozen thesis avoids the full
    # unmatched-position fallback.
    assert hold.components["uncertainty"] == pytest.approx(
        -decision_layer.config.uncertainty_penalty * updated.uncertainty
    )
    assert hold.components["uncertainty"] > (
        -decision_layer.config.uncertainty_penalty
    )


def test_typed_position_cannot_rewrite_frozen_episode_invalidation() -> None:
    _, _, _, forming, triggered = _dfp_fixture()
    brain = _brain()
    brain.update(forming)
    belief = brain.update(triggered)
    entered = belief.hypotheses[
        "displacement_first_pullback:long"
    ]
    assert entered.plan is not None
    plan = entered.plan
    rewritten = replace(
        plan.invalidation,
        price=plan.invalidation.price - 1.0,
        source_level_id="rewritten-stop",
    )
    position = PositionSnapshot(
        thesis_hash="filled-dfp",
        symbol=triggered.symbol,
        instrument_id=triggered.instrument_id,
        playbook=entered.playbook,
        direction=entered.direction,
        entry_price=plan.planned_entry,
        original_invalidation=rewritten,
        current_stop=rewritten.price,
        primary_target=plan.targets[0],
        opened_at=triggered.asof,
        deadline=plan.deadline,
        quantity=1,
        unrealized_R=0.0,
        elapsed_minutes=1,
        setup_id=plan.setup_id,
        entry_location_id=plan.entry_location_id,
        entry_path_id=plan.entry_path_id,
    )

    with pytest.raises(
        ValueError,
        match="position invalidation differs from its frozen belief episode",
    ):
        brain.update(
            _advance_observation(
                triggered,
                triggered.asof + pd.Timedelta(minutes=1),
            ),
            position=position,
        )


def test_same_episode_plan_deadline_can_tighten_but_never_expand() -> None:
    _, _, _, forming, triggered = _dfp_fixture()
    brain = _brain()
    brain.update(forming)
    first = brain.update(triggered).hypotheses[
        "displacement_first_pullback:long"
    ]
    assert first.plan is not None
    first_deadline = first.plan.deadline

    tight_clock = triggered.asof + pd.Timedelta(minutes=1)
    tight_execution = replace(
        _execution(tight_clock, cost=0.10),
        minutes_to_deadline=5,
    )
    tightened = brain.update(
        _advance_observation(
            triggered,
            tight_clock,
            execution=tight_execution,
        )
    ).hypotheses["displacement_first_pullback:long"]
    assert tightened.plan is not None
    assert tightened.plan.deadline < first_deadline

    relaxed_clock = tight_clock + pd.Timedelta(minutes=1)
    relaxed_execution = replace(
        _execution(relaxed_clock, cost=0.10),
        minutes_to_deadline=50,
    )
    relaxed = brain.update(
        _advance_observation(
            triggered,
            relaxed_clock,
            execution=relaxed_execution,
        )
    ).hypotheses["displacement_first_pullback:long"]
    assert relaxed.plan is not None
    assert relaxed.plan.deadline == tightened.plan.deadline


def _pool_manipulation(swept_at: pd.Timestamp) -> ManipulationState:
    return ManipulationState(
        manipulation_id="manipulation:pool-above",
        protocol_hash=GROUP4_SHA,
        source_group12_protocol_hash=GROUP12_SHA,
        symbol="NQH5",
        instrument_id=1,
        timeframe=Timeframe.M1,
        lifecycle=ManipulationLifecycle.SWEPT,
        side="above",
        source_kind="formed_liquidity_pool",
        source_id="pool-above",
        source_protocol_hash=GROUP12_SHA,
        source_timeframe=Timeframe.M1,
        source_inventory_item_id="pool:pool-above",
        source_inventory_lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        coincident_source_ids=(),
        source_formed_at=swept_at - pd.Timedelta(minutes=5),
        source_eligible_at=swept_at - pd.Timedelta(minutes=1),
        source_lower_bound=100.75,
        source_upper_bound=101.0,
        formed_at=swept_at,
        confirmed_at=swept_at,
        swept_at=swept_at,
        reaccepted_at=None,
        accepted_outside_at=None,
        resolved_at=None,
        state_started_at=swept_at,
        last_updated_at=swept_at,
        sweep_extreme=101.25,
        close_outside_on_sweep=False,
        reentry_price=None,
        resolved_side=None,
        outside_completed_bars=0,
        penetration_atr=0.8,
        strength=0.8,
        age_1m_bars=0,
        transition_reason="source_swept",
        censored_at=None,
    )


def _local_bos(clock: pd.Timestamp) -> BreakOfStructureState:
    return BreakOfStructureState(
        bos_id="lsr-trigger-bos",
        timeframe=Timeframe.M1,
        direction=Direction.SHORT,
        lifecycle=BOSLifecycle.CONFIRMED,
        scope=BOSScope.LOCAL,
        target_swing_id="lsr-trigger-swing",
        source_structure_id=None,
        target_price=100.5,
        target_ticks=402,
        pending_at=clock - pd.Timedelta(minutes=1),
        resolved_at=clock,
        age_bars=1,
        strength=0.8,
    )


def _lsr_target(
    confirmed_at: pd.Timestamp,
) -> tuple[SwingPoint, LiquidityInventoryItem]:
    swing = SwingPoint(
        swing_id="lsr-target-low",
        timeframe=Timeframe.H1,
        symbol="NQH5",
        instrument_id=1,
        side=SwingSide.LOW,
        price=98.0,
        price_ticks=392,
        pivot_start=confirmed_at - pd.Timedelta(hours=2),
        pivot_end=confirmed_at - pd.Timedelta(hours=1),
        observed_at=confirmed_at,
        confirmed_at=confirmed_at,
        lifecycle=SwingLifecycle.CONFIRMED,
        magnitude_atr=0.7,
        age_bars=1,
    )
    inventory = LiquidityInventoryItem(
        item_id=f"swing:{swing.swing_id}",
        timeframe=Timeframe.H1,
        side="below",
        kind="swing",
        price=swing.price,
        lower_bound=swing.price,
        upper_bound=swing.price,
        formed_at=swing.pivot_end,
        confirmed_at=swing.confirmed_at,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=(swing.swing_id,),
        age_bars=swing.age_bars,
        strength=swing.magnitude_atr,
    )
    return swing, inventory


def _lsr_observation() -> MarketObservation:
    reducer = CausalGroup5Reducer(_group5_protocol())
    swing, target = _lsr_target(BASE - pd.Timedelta(hours=1))
    sweep_bar = _m1(
        0,
        open_=101.0,
        high=101.25,
        low=100.0,
        close=100.0,
    )
    manipulation = _pool_manipulation(sweep_bar.end)
    reducer.on_completed_1m(
        sweep_bar,
        manipulations=(manipulation,),
        liquidity_inventory=(target,),
        m1_atr=1.0,
    )
    formation = _m1(
        1,
        open_=100.5,
        high=100.75,
        low=100.25,
        close=100.5,
    )
    fvg = _fvg(
        formation.end,
        direction=Direction.SHORT,
        identity="lsr-short-fvg",
        lower=101.0,
        upper=102.0,
    )
    reducer.on_completed_1m(
        formation,
        fair_value_gaps=(fvg,),
        manipulations=(manipulation,),
        liquidity_inventory=(target,),
        m1_atr=1.0,
    )
    pullback = _m1(
        2,
        open_=100.75,
        high=101.5,
        low=100.5,
        close=101.0,
    )
    reducer.on_completed_1m(
        pullback,
        fair_value_gaps=(fvg,),
        manipulations=(manipulation,),
        liquidity_inventory=(target,),
        m1_atr=1.0,
    )
    trigger = _m1(
        3,
        open_=101.0,
        high=101.25,
        low=100.5,
        close=101.0,
    )
    trigger_bos = _local_bos(trigger.end)
    output = reducer.on_completed_1m(
        trigger,
        fair_value_gaps=(fvg,),
        manipulations=(manipulation,),
        liquidity_inventory=(target,),
        m1_bos=(trigger_bos,),
        m1_atr=1.0,
    )
    base = market_observation(
        asof=trigger.end,
        price=trigger.close,
    )
    frames = dict(base.frames)
    frames[Timeframe.H4] = replace(
        frames[Timeframe.H4],
        cutoff=trigger.end,
    )
    frames[Timeframe.H1] = replace(
        frames[Timeframe.H1],
        cutoff=trigger.end,
        swings=(swing,),
    )
    frames[Timeframe.M5] = replace(
        frames[Timeframe.M5],
        cutoff=trigger.end,
        fair_value_gaps=(fvg,),
    )
    frames[Timeframe.M1] = replace(
        frames[Timeframe.M1],
        cutoff=trigger.end,
        structure_breaks=(trigger_bos,),
    )
    return replace(
        base,
        frames=frames,
        execution=_execution(trigger.end, cost=0.025),
        liquidity_inventory=(target,),
        liquidity_inventory_authoritative=True,
        manipulations=(manipulation,),
        group5_authoritative=True,
        entry_locations=output.entry_locations,
        qualified_reacceptances=output.qualified_reacceptances,
        micro_bos_references=output.micro_bos_references,
        path_sequences=output.path_sequences,
    )


def test_lsr_requires_exact_later_micro_bos_and_original_sweep_stop() -> None:
    observation = _lsr_observation()
    belief = _brain().update(observation)
    hypothesis = belief.hypotheses[
        "liquidity_sweep_reversal:short"
    ]

    assert hypothesis.phase is PlaybookPhase.EXECUTABLE
    assert hypothesis.sequence is not None
    assert hypothesis.sequence.complete
    assert hypothesis.plan is not None
    assert hypothesis.plan.setup_id != hypothesis.plan.entry_path_id
    assert (
        hypothesis.plan.invalidation.source_level_id
        == "manipulation:pool-above"
    )
    assert hypothesis.plan.invalidation.price == 101.25
    assert hypothesis.plan.selected_draw_id == "swing:lsr-target-low"

    decision = UtilityDecisionLayer().decide(observation, belief)
    assert decision.selected_action.value == "enter"
    risk = StructuralRiskEngine().review(decision, observation)
    assert risk.passed
    assert risk.frozen_thesis is not None
    assert (
        risk.frozen_thesis.original_invalidation.source_level_id
        == "manipulation:pool-above"
    )

    orphaned_frames = dict(observation.frames)
    orphaned_frames[Timeframe.M1] = replace(
        orphaned_frames[Timeframe.M1],
        structure_breaks=(),
    )
    orphaned_micro = replace(observation, frames=orphaned_frames)
    orphaned_belief = _brain().update(orphaned_micro)
    assert (
        orphaned_belief.hypotheses[
            "liquidity_sweep_reversal:short"
        ].phase
        is PlaybookPhase.WAITING_TRIGGER
    )
    orphaned_micro_risk = StructuralRiskEngine().review(
        decision,
        orphaned_micro,
    )
    assert not orphaned_micro_risk.passed
    assert orphaned_micro_risk.final_action.value == "abstain"


def test_insufficient_remaining_path_stays_in_delivery_not_uncertainty() -> None:
    observation = _lsr_observation()
    target = replace(
        observation.liquidity_inventory[0],
        price=100.50,
        lower_bound=100.50,
        upper_bound=100.50,
    )
    frames = dict(observation.frames)
    frames[Timeframe.H1] = replace(
        frames[Timeframe.H1],
        swings=(
            replace(
                frames[Timeframe.H1].swings[0],
                price=100.50,
                price_ticks=402,
            ),
        ),
    )
    near_draw = replace(
        observation,
        frames=frames,
        liquidity_inventory=(target,),
        manipulations=(
            replace(
                observation.manipulations[0],
                sweep_extreme=102.50,
            ),
        ),
    )

    hypothesis = _brain().update(near_draw).hypotheses[
        "liquidity_sweep_reversal:short"
    ]
    remaining_evidence = next(
        item
        for item in hypothesis.contradicting
        if item.primitive == "remaining_path_consumed"
    )
    assert remaining_evidence.value == 1.0
    assert hypothesis.plan is not None
    assert (
        hypothesis.plan.remaining_path_R
        < _brain().config.minimum_remaining_path_R
    )
    assert hypothesis.uncertainty == 0.0


def _lsr_without_plan_or_trigger() -> MarketObservation:
    observation = _lsr_observation()
    frames = {
        timeframe: replace(frame, liquidity=(), swings=())
        for timeframe, frame in observation.frames.items()
    }
    frames[Timeframe.M1] = replace(
        frames[Timeframe.M1],
        structure_breaks=(),
    )
    return replace(
        observation,
        frames=frames,
        liquidity_inventory=(),
    )


def test_pre_entry_episode_freezes_invalidation_and_closes_on_breach() -> None:
    observation = _lsr_without_plan_or_trigger()
    brain = _brain()
    key = "liquidity_sweep_reversal:short"

    active_belief = brain.update(observation)
    active = active_belief.hypotheses[key]
    assert active.phase is PlaybookPhase.WAITING_TRIGGER
    assert active.plan is None
    assert active.episode_deadline == (
        observation.asof + pd.Timedelta(minutes=60)
    )
    assert active.invalidation is not None

    breached_observation = _advance_observation(
        observation,
        observation.asof + pd.Timedelta(minutes=1),
        price=active.invalidation.price + 0.25,
    )
    terminal_belief = brain.update(breached_observation)
    terminal = terminal_belief.hypotheses[key]

    assert terminal.phase is PlaybookPhase.INVALIDATED
    assert terminal.terminal_reason == "frozen_invalidation_breached"
    assert terminal.episode_id == active.episode_id
    assert terminal.episode_deadline == active.episode_deadline
    assert terminal.invalidation == active.invalidation
    assert active.invalidation.source_level_id in terminal.terminal_source_ids
    assert not terminal.eligible

    trace = build_decision_trace(
        _snapshot(breached_observation, terminal_belief, "e"),
        _snapshot(observation, active_belief, "d"),
        hypothesis_key=key,
    )
    assert trace["belief_t"][key]["episode_deadline"] == (
        active.episode_deadline.isoformat()
    )
    assert trace["belief_delta"][key]["episode_deadline_changed"] is False

    reclaimed = brain.update(
        _advance_observation(
            observation,
            breached_observation.asof + pd.Timedelta(minutes=1),
            price=active.invalidation.price - 0.25,
        )
    ).hypotheses[key]
    assert reclaimed is terminal


def test_pre_entry_episode_deadline_is_frozen_without_a_plan() -> None:
    observation = _lsr_observation()
    pool_paths = tuple(
        path
        for path in observation.path_sequences
        if path.context_kind == "pool_reversal"
    )
    pool_references = tuple(
        reference
        for reference in observation.micro_bos_references
        if reference.context_kind == "pool_reversal"
    )
    armed_observation = replace(
        observation,
        entry_locations=(),
        qualified_reacceptances=tuple(
            item
            for item in observation.qualified_reacceptances
            if item.context_kind == "pool_sweep"
        ),
        micro_bos_references=pool_references,
        path_sequences=pool_paths,
    )
    brain = _brain()
    key = "liquidity_sweep_reversal:short"

    active = brain.update(armed_observation).hypotheses[key]
    assert active.phase is PlaybookPhase.ARMED
    assert active.plan is None
    assert active.episode_deadline is not None

    deadline_observation = _advance_observation(
        armed_observation,
        active.episode_deadline,
    )
    terminal = brain.update(deadline_observation).hypotheses[key]

    assert terminal.phase is PlaybookPhase.INVALIDATED
    assert terminal.terminal_reason == "episode_deadline_elapsed"
    assert terminal.episode_deadline == active.episode_deadline
    assert not terminal.eligible


def test_entry_zone_cannot_cross_the_frozen_invalidation() -> None:
    observation = _lsr_observation()
    frames = dict(observation.frames)
    frames[Timeframe.M1] = replace(
        frames[Timeframe.M1],
        structure_breaks=(),
    )
    manipulation = replace(
        observation.manipulations[0],
        source_lower_bound=100.25,
        source_upper_bound=100.50,
        sweep_extreme=100.75,
    )
    wrong_side_zone = replace(
        observation,
        price=100.50,
        frames=frames,
        manipulations=(manipulation,),
    )

    hypothesis = _brain().update(wrong_side_zone).hypotheses[
        "liquidity_sweep_reversal:short"
    ]

    assert hypothesis.plan is None
    assert hypothesis.phase is PlaybookPhase.INVALIDATED
    assert (
        hypothesis.terminal_reason
        == "entry_zone_beyond_frozen_invalidation"
    )
    assert manipulation.manipulation_id in hypothesis.terminal_source_ids


def test_favr_is_parked_and_opposite_non_setup_remains_inactive() -> None:
    _, _, _, forming, _ = _dfp_fixture()
    belief = _brain().update(forming)

    assert (
        belief.hypotheses["displacement_first_pullback:short"].phase
        is PlaybookPhase.INACTIVE
    )
    for direction in ("long", "short"):
        favr = belief.hypotheses[
            f"{Playbook.FAILED_AUCTION_VALUE_RETURN.value}:{direction}"
        ]
        assert favr.phase is PlaybookPhase.INACTIVE
        assert favr.plan is None
        assert favr.thesis_strength == 0.0
        assert favr.uncertainty == 1.0
        assert favr.raw_quality_dimensions["uncertainty"] == 1.0
