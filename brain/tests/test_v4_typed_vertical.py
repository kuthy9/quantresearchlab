from __future__ import annotations

from dataclasses import replace as _dataclass_replace
import hashlib
import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from brain.core.brain_entry_sequence import (
    BrainObservationView,
    brain_interaction_view,
    brain_observation_view,
)
from brain.core.calibration import (
    DimensionReliabilityMap,
    DimensionReliabilityPoint,
    TYPED_CALIBRATION_DIMENSIONS,
    TypedBrainCalibrator,
)
from brain.core.brain_calibration import BrainCalibrationRecorder
from brain.core.decision import UtilityDecisionLayer
from shares.tests.legacy_group5 import CausalGroup5Reducer, Group5Protocol
from shares.core.model import (
    Action,
    ActionUtility,
    AccountState,
    AuthorityLayer,
    BOSLifecycle,
    BOSPostBreakState,
    BOSScope,
    Bar,
    BreakOfStructureState,
    Candle,
    DeliveryObstruction,
    Decision,
    Direction,
    DirectionalObstructionView,
    DisplacementObservation,
    DisplacementTransitionObservation,
    EntryEpisodeState,
    EntryLocationLifecycle,
    ExecutionObservation,
    FairValueGapLifecycle,
    FVGQualification,
    FairValueGapState,
    HypothesisBelief,
    InteractionUpdate,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityLevel,
    LiquidityPoolLifecycle,
    LiquidityPoolState,
    ManipulationLifecycle,
    ManipulationState,
    MicroBreakFact,
    MicroBOSReference,
    GlobalMarketContext,
    MarketObservation,
    MarketMode,
    PathSequenceLifecycle,
    PathSequenceStep,
    Playbook,
    PlaybookPhase,
    PositionSnapshot,
    ScaleRelation,
    ScaleRelationState,
    StructureLifecycle,
    StructureSequenceState,
    SwingLifecycle,
    SwingPoint,
    SwingSide,
    Timeframe,
)
from brain.core.playbook_registry import load_playbook_registry
from brain.core.playbooks import (
    BrainConfig,
    PlaybookBrain,
    _ContextClosure,
    _Evaluation,
    _LSRExecutionOwner,
    _SequenceSignal,
    _build_context_episode_views,
    _cascade_context_terminals,
    _context_level_terminal,
    _dfp_context_terminal_source_ids,
    _fail_closed_shared_episode_ownership,
    _freeze_lsr_context_execution_owner,
    _frozen_target_crosses_authority_barrier,
    _lsr_context_can_spawn_child,
    _lsr_candidate,
    _lsr_zone_candidate_id,
    _route_global_context,
    _stable_context_thesis_id,
    _retain_matching_prior_trigger,
    _select_countertrend_lsr_target,
    _plan_feasibility,
    _position_context_terminal_phase,
    _select_frozen_trigger,
    _select_pool_path,
    _terminal_candidate_is_new,
    _thesis_candidate_id,
    _typed_dfp,
    _typed_lsr,
)
from brain.core.risk import StructuralRiskEngine
from shares.core.scene_graph import (
    TemporalMarketSceneGraph,
    build_open_market_theses,
)

from shares.tests.helpers import (
    graph_free_action_belief,
    market_observation,
    replace_market_observation as _replace_market_observation,
)


ROOT = Path(__file__).resolve().parents[2]
GROUP12_PROTOCOL_PATH = (
    ROOT / "configs/primitives_structure_liquidity.json"
)
GROUP3_PROTOCOL_PATH = ROOT / "configs/primitives_zones.json"
GROUP4_PROTOCOL_PATH = ROOT / "configs/primitives_range.json"


def _interaction_identity(*parts: object) -> str:
    return hashlib.sha256(
        "|".join(
            value.isoformat()
            if isinstance(value, pd.Timestamp)
            else str(value)
            for value in parts
        ).encode("utf-8")
    ).hexdigest()


def _micro_break_facts(
    references: tuple[object, ...],
) -> tuple[MicroBreakFact, ...]:
    return tuple(
        MicroBreakFact(
            reference_id=reference.reference_id,
            protocol_hash=reference.protocol_hash,
            context_kind=reference.context_kind,
            context_id=reference.context_id,
            context_direction=reference.expected_direction,
            anchor_at=reference.anchor_at,
            bos_id=reference.bos_id,
            bos_direction=reference.bos_direction,
            target_swing_id=reference.target_swing_id,
            scope=reference.scope,
            pending_at=reference.pending_at,
            resolved_at=reference.resolved_at,
            relation=reference.relation,
            strength=reference.strength,
        )
        for reference in references
    )


def _physical_paths(
    paths: tuple[PathSequenceState, ...],
    references: tuple[object, ...],
) -> tuple[PathSequenceState, ...]:
    references_by_key = {
        (reference.context_kind, reference.context_id, reference.bos_id): reference
        for reference in references
    }
    legacy_micro_kinds = {
        "micro_bos_simultaneous",
        "micro_bos_confirmed",
        "micro_bos_opposed",
        "micro_bos_ambiguous",
    }
    physical_paths: list[PathSequenceState] = []
    for path in paths:
        physical_steps: list[PathSequenceStep] = []
        converted_micro = False
        for ordinal, step in enumerate(path.steps):
            reference = references_by_key.get(
                (
                    path.context_kind,
                    path.context_id,
                    step.source_event_id or "",
                )
            )
            if step.kind in legacy_micro_kinds:
                # These fixtures edit the historical Brain view.  Removing a
                # reference must also remove its interpreted step before the
                # fixture is reconstructed as a canonical Eye update.
                if reference is None:
                    continue
                converted_micro = True
                kind = "micro_break_observed"
                reason = (
                    "confirmed_m1_break_at_anchor_clock"
                    if reference.relation == "same_clock_unknown"
                    else "first_strictly_later_confirmed_m1_break"
                )
                step_id = _interaction_identity(
                    "group5-step-v1",
                    path.protocol_hash,
                    path.sequence_id,
                    ordinal,
                    kind,
                    step.observed_at,
                    step.source_entity_id,
                )
            else:
                kind = step.kind
                reason = step.reason
                step_id = step.step_id
            physical_steps.append(
                _dataclass_replace(
                    step,
                    step_id=step_id,
                    kind=kind,
                    predecessor_step_ids=(
                        ()
                        if not physical_steps
                        else (physical_steps[-1].step_id,)
                    ),
                    reason=reason,
                )
            )
        transition_reason = path.transition_reason
        interpreted_reason = transition_reason in {
            "qualified_reacceptance_held",
            "micro_bos_aligned",
            "micro_bos_opposed",
            "micro_bos_ambiguous_same_clock",
            "pool_reversal_sequence_observed",
        }
        if converted_micro and interpreted_reason:
            transition_reason = "first_strict_micro_break_observed"
        elif interpreted_reason:
            # The caller removed the interpreted break.  Restore the physical
            # pre-break active state instead of smuggling a Brain success
            # reason into InteractionUpdate.
            last_step = physical_steps[-1]
            transition_reason = {
                "first_pullback": "context_registered",
                "wick_rejection": "zone_rejection_observed",
                "reacceptance_held": "hold_completed",
            }.get(last_step.kind, "context_registered")
            physical_paths.append(
                _dataclass_replace(
                    path,
                    lifecycle=PathSequenceLifecycle.ACTIVE,
                    state_started_at=last_step.observed_at,
                    last_updated_at=last_step.observed_at,
                    state_duration_real_1m_bars=0,
                    steps=tuple(physical_steps),
                    ended_at=None,
                    transition_reason=transition_reason,
                )
            )
            continue
        physical_paths.append(
            _dataclass_replace(
                path,
                steps=tuple(physical_steps),
                transition_reason=transition_reason,
            )
        )
    return tuple(physical_paths)


def _interaction_update_from_brain_fields(
    *,
    entry_locations: tuple[object, ...],
    qualified_reacceptances: tuple[object, ...],
    micro_bos_references: tuple[object, ...],
    path_sequences: tuple[PathSequenceState, ...],
) -> InteractionUpdate:
    return InteractionUpdate(
        zone_interactions=tuple(entry_locations),
        reacceptance_interactions=tuple(qualified_reacceptances),
        micro_break_facts=_micro_break_facts(micro_bos_references),
        interaction_paths=_physical_paths(
            tuple(path_sequences),
            tuple(micro_bos_references),
        ),
    )


def _replace_brain_observation(
    observation: BrainObservationView,
    /,
    **changes: object,
) -> BrainObservationView:
    raw = observation._observation
    current = brain_interaction_view(observation)
    names = {
        "entry_locations",
        "qualified_reacceptances",
        "micro_bos_references",
        "path_sequences",
    }
    brain_changes = {
        name: changes.pop(name)
        for name in tuple(changes)
        if name in names
    }
    references = tuple(
        brain_changes.get(
            "micro_bos_references",
            current.micro_bos_references,
        )
    )
    paths = tuple(
        brain_changes.get("path_sequences", current.path_sequences)
    )
    if brain_changes:
        changes["interaction_update"] = _interaction_update_from_brain_fields(
            entry_locations=tuple(
                brain_changes.get("entry_locations", current.zone_interactions)
            ),
            qualified_reacceptances=tuple(
                brain_changes.get(
                    "qualified_reacceptances",
                    current.reacceptance_interactions,
                )
            ),
            micro_bos_references=references,
            path_sequences=paths,
        )
    return brain_observation_view(
        _replace_market_observation(raw, **changes)
    )


def replace(value: object, /, **changes: object):
    if isinstance(value, BrainObservationView):
        return _replace_brain_observation(value, **changes)
    return _dataclass_replace(value, **changes)


def replace_market_observation(
    observation: object,
    /,
    **changes: object,
):
    if isinstance(observation, BrainObservationView):
        return _replace_brain_observation(observation, **changes)
    return _replace_market_observation(observation, **changes)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


GROUP12_SHA = _sha256(GROUP12_PROTOCOL_PATH)
GROUP3_SHA = _sha256(GROUP3_PROTOCOL_PATH)
GROUP4_SHA = _sha256(GROUP4_PROTOCOL_PATH)
DISPLACEMENT_SHA = "a" * 64
BASE = pd.Timestamp("2025-01-07T09:30:00-05:00")


def _brain_global_context(
    *,
    direction: Direction,
    hard_barriers: tuple[DeliveryObstruction, ...] = (),
    invalidated_source_ids: tuple[str, ...] = (),
    ambiguous_evidence: tuple[str, ...] = (),
    unknown_evidence: tuple[str, ...] = (),
    market_mode: MarketMode = MarketMode.DIRECTIONAL,
) -> GlobalMarketContext:
    authority = AuthorityLayer(
        timeframe=Timeframe.H4,
        direction=direction,
        structure_id="h4-authority",
        confirmed_at=BASE - pd.Timedelta(hours=4),
        protected_level_id=None,
        structural_scope="external",
        acceptance_state="confirmed",
        status="intact",
        source_ids=("h4-authority",),
    )
    return GlobalMarketContext(
        updated_at=BASE,
        scene_revision_id="scene:r000000000001",
        market_epoch_id="epoch:test",
        authority_stack=(authority,),
        market_mode=market_mode,
        scale_relation_details={
            timeframe.value: ScaleRelationState(
                timeframe=timeframe,
                relation=ScaleRelation.ALIGNED,
                direction=direction,
                authority_layer_id=authority.structure_id,
                evidence_ids=(authority.structure_id,),
                evidence_kind="structure",
                structural_scope="external",
                acceptance_state="confirmed",
                since=authority.confirmed_at,
                age_bars=1,
                graph_connected=True,
                ambiguous=False,
            )
            for timeframe in Timeframe
        },
        external_draw_candidates={"above": (), "below": ()},
        obstruction_views={
            candidate_direction.value: DirectionalObstructionView(
                direction=candidate_direction,
                nearest_draw_id=None,
                nearest_draw_price=None,
                hard_barriers=(
                    hard_barriers
                    if candidate_direction is Direction.SHORT
                    else ()
                ),
                soft_frictions=(),
            )
            for candidate_direction in Direction
        },
        material_conflicts=(),
        unknown_evidence=unknown_evidence,
        ambiguous_evidence=ambiguous_evidence,
        dislocations_by_scale={
            timeframe.value: () for timeframe in Timeframe
        },
        invalidated_source_ids=invalidated_source_ids,
    )


def _group5_protocol() -> Group5Protocol:
    return Group5Protocol.from_file(
        ROOT / "configs/primitives_entry.json"
    )


def _brain() -> PlaybookBrain:
    return PlaybookBrain(
        registry=load_playbook_registry(
            ROOT / "brain/configs/playbooks.json"
        )
    )


def test_trigger_opposition_names_do_not_misclassify_ambiguity() -> None:
    registry = load_playbook_registry(ROOT / "brain/configs/playbooks.json")
    dfp = registry.for_playbook(Playbook.DISPLACEMENT_FIRST_PULLBACK)
    lsr = registry.for_playbook(Playbook.LIQUIDITY_SWEEP_REVERSAL)
    favr = registry.for_playbook(Playbook.FAILED_AUCTION_VALUE_RETURN)

    assert "trigger_opposed" in dfp.contradicting_evidence
    assert "micro_bos_opposed" in lsr.contradicting_evidence
    assert "trigger_opposed" in favr.contradicting_evidence
    assert all(
        "ambiguous" not in primitive
        for protocol in (dfp, lsr, favr)
        for primitive in protocol.contradicting_evidence
    )


def _mapped_brain() -> PlaybookBrain:
    registry = load_playbook_registry(
        ROOT / "brain/configs/playbooks.json"
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
            registry_hash=registry.fingerprint,
            maps=maps,
            status="ready",
        ),
    )


def _ready_decision_layer(brain: PlaybookBrain) -> UtilityDecisionLayer:
    return UtilityDecisionLayer(
        calibration_ready=brain.calibrator.is_ready,
        calibration_version=brain.calibrator.version,
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
        qualification=FVGQualification.DISPLACEMENT_LINKED,
        source_displacement_id=f"displacement:{identity}",
        source_active_transition_id=f"active:{identity}",
        source_displacement_protocol_hash=DISPLACEMENT_SHA,
        source_displacement_started_at=starts[1],
        source_displacement_active_at=confirmed_at,
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
        formation_atr=1.0,
        width_atr=upper - lower,
        strength=0.0,
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
        break_bar_id="h1-continuation-break-bar",
        break_distance_atr=0.5,
        post_break_state=BOSPostBreakState.ACCEPTED,
        accepted_at=clock + pd.Timedelta(hours=1),
    )


def _dfp_observation(
    *,
    asof: pd.Timestamp,
    price: float,
    output,
    fvg: FairValueGapState,
    draw: LiquidityInventoryItem,
    primary_draw: LiquidityInventoryItem | None = None,
) -> MarketObservation:
    base = market_observation(asof=asof, price=price)
    primary_draw = primary_draw or _inventory_draw(
        identity="previous-day:dfp-primary-draw",
        timeframe=Timeframe.H1,
        side=draw.side,
        price=(
            draw.price - 1.0
            if draw.side == "above"
            else draw.price + 1.0
        ),
        confirmed_at=draw.confirmed_at,
        source_id="dfp-primary-draw-source",
        strength=0.6,
    )
    primary_draw = replace(
        primary_draw,
        kind=(
            "previous_day_high"
            if draw.side == "above"
            else "previous_day_low"
        ),
    )
    frames = dict(base.frames)
    frames[Timeframe.H4] = replace(
        frames[Timeframe.H4],
        cutoff=asof,
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
    return brain_observation_view(replace(
        base,
        frames=frames,
        execution=_execution(asof, cost=0.10),
        liquidity_inventory=(primary_draw, draw),
        interaction_update=_interaction_update_from_brain_fields(
            entry_locations=output.entry_locations,
            qualified_reacceptances=output.qualified_reacceptances,
            micro_bos_references=output.micro_bos_references,
            path_sequences=output.path_sequences,
        ),
    ))


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
    return replace_market_observation(
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




def test_dfp_vertical_chain_uses_exact_group5_plan_and_risk_binding() -> None:
    _, _, _, forming, triggered = _dfp_fixture()
    brain = _mapped_brain()

    first = brain.update(forming)
    first_dfp = first.hypotheses[
        "displacement_first_pullback:long"
    ]
    assert first_dfp.phase is PlaybookPhase.WAITING_LOCATION
    assert (
        first_dfp.deliverable_targets[0].level_id
        == "previous-day:dfp-primary-draw"
    )
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
    assert (
        hypothesis.plan.selected_draw_id
        == "previous-day:dfp-primary-draw"
    )
    assert hypothesis.thesis_draw is not None
    assert hypothesis.thesis_draw.level_id == "swing:dfp-h4-draw"
    assert hypothesis.liquidity_route is not None
    assert (
        hypothesis.liquidity_route.context_draw_id
        == "swing:dfp-h4-draw"
    )
    assert (
        hypothesis.liquidity_route.terminal_draw_id
        == "swing:dfp-h4-draw"
    )

    decision = _ready_decision_layer(brain).decide(
        triggered,
        graph_free_action_belief(belief),
    )
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
    assert orphaned_zone_risk.passed
    assert orphaned_zone_risk.final_action.value == "enter"

    missing_physical_interaction = replace(
        triggered,
        interaction_update=None,
    )
    missing_physical_risk = StructuralRiskEngine().review(
        decision,
        missing_physical_interaction,
    )
    assert not missing_physical_risk.passed
    assert missing_physical_risk.final_action.value == "abstain"

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


def test_dfp_wide_structural_target_keeps_inventory_price_for_risk() -> None:
    _, _, _, forming, triggered = _dfp_fixture()
    primary = replace(
        forming.liquidity_inventory[0],
        price=102.0,
        lower_bound=101.75,
        upper_bound=102.25,
    )
    forming = replace(
        forming,
        liquidity_inventory=(primary, forming.liquidity_inventory[1]),
    )
    triggered = replace(
        triggered,
        liquidity_inventory=(primary, triggered.liquidity_inventory[1]),
    )
    brain = _mapped_brain()
    brain.update(forming)
    belief = brain.update(triggered)
    hypothesis = belief.hypotheses[
        "displacement_first_pullback:long"
    ]

    assert hypothesis.plan is not None
    assert hypothesis.plan.targets[0].price == primary.price
    assert hypothesis.liquidity_route is not None
    assert (
        hypothesis.liquidity_route.primary_target_price_basis
        == "inventory_price"
    )
    decision = _ready_decision_layer(brain).decide(
        triggered,
        graph_free_action_belief(belief),
    )
    risk = StructuralRiskEngine().review(decision, triggered)
    assert risk.passed
    assert risk.final_action.value == "enter"


def test_dfp_without_intermediate_primary_draw_reports_precise_failure() -> None:
    _, _, terminal_draw, forming, triggered = _dfp_fixture()
    forming = replace(forming, liquidity_inventory=(terminal_draw,))
    triggered = replace(triggered, liquidity_inventory=(terminal_draw,))
    brain = _mapped_brain()
    brain.update(forming)
    belief = brain.update(triggered)
    hypothesis = belief.hypotheses[
        "displacement_first_pullback:long"
    ]

    assert hypothesis.thesis_draw is not None
    assert hypothesis.thesis_draw.level_id == terminal_draw.item_id
    assert hypothesis.plan is None
    assert hypothesis.phase is not PlaybookPhase.EXECUTABLE
    assert hypothesis.plan_feasibility is not None
    assert not hypothesis.plan_feasibility.valid
    assert hypothesis.plan_feasibility.failure_reason == (
        "all_draws_at_or_beyond_context_draw"
    )
    decision = _ready_decision_layer(brain).decide(
        triggered,
        graph_free_action_belief(belief),
    )
    assert decision.selected_action.value in {"wait", "abstain"}


def test_dfp_rejects_primary_draw_beyond_confirmed_hard_barrier() -> None:
    _, _, _, _, triggered = _dfp_fixture()
    barrier = DeliveryObstruction(
        obstruction_id="h1-accepted-opposition",
        timeframe=Timeframe.H1,
        direction=None,
        side="above",
        lower_bound=101.5,
        upper_bound=101.5,
        hard=True,
        source_kind="accepted_bos",
        source_ids=("h1-accepted-opposition",),
        structural_scope="external",
        acceptance_state="accepted",
    )
    context = _brain_global_context(direction=Direction.LONG)
    views = dict(context.obstruction_views)
    views[Direction.LONG.value] = DirectionalObstructionView(
        direction=Direction.LONG,
        nearest_draw_id=None,
        nearest_draw_price=None,
        hard_barriers=(barrier,),
        soft_frictions=(),
    )
    context = replace(
        context,
        updated_at=triggered.asof,
        obstruction_views=views,
    )
    protocol = load_playbook_registry(
        ROOT / "brain/configs/playbooks.json"
    ).for_playbook(Playbook.DISPLACEMENT_FIRST_PULLBACK)
    evaluation = _typed_dfp(
        triggered,
        Direction.LONG,
        protocol,
        None,
        BrainConfig(),
        global_context=context,
    )

    assert evaluation.plan is None
    assert evaluation.target_selection_failure_reason == (
        "all_draws_beyond_barrier"
    )
    feasibility = _plan_feasibility(
        evaluation.plan,
        evaluation,
        BrainConfig(),
    )
    assert not feasibility.valid
    assert feasibility.failure_reason == "all_draws_beyond_barrier"


@pytest.mark.parametrize("self_relation", ("identity", "source", "price"))
def test_dfp_primary_draw_does_not_obstruct_itself(
    self_relation: str,
) -> None:
    _, _, _, _, triggered = _dfp_fixture()
    primary = triggered.liquidity_inventory[0]
    obstruction = DeliveryObstruction(
        obstruction_id=(
            primary.item_id
            if self_relation == "identity"
            else f"primary-obstruction:{self_relation}"
        ),
        timeframe=Timeframe.H1,
        direction=None,
        side="above",
        lower_bound=(
            primary.price if self_relation == "price" else 101.5
        ),
        upper_bound=(
            primary.price if self_relation == "price" else 101.5
        ),
        hard=True,
        source_kind="protected_swing",
        source_ids=(
            primary.source_ids
            if self_relation == "source"
            else (f"unrelated-source:{self_relation}",)
        ),
        structural_scope="external",
        acceptance_state="confirmed",
    )
    context = _brain_global_context(direction=Direction.LONG)
    views = dict(context.obstruction_views)
    views[Direction.LONG.value] = DirectionalObstructionView(
        direction=Direction.LONG,
        nearest_draw_id=primary.item_id,
        nearest_draw_price=primary.price,
        hard_barriers=(obstruction,),
        soft_frictions=(),
    )
    context = replace(
        context,
        updated_at=triggered.asof,
        obstruction_views=views,
    )
    protocol = load_playbook_registry(
        ROOT / "brain/configs/playbooks.json"
    ).for_playbook(Playbook.DISPLACEMENT_FIRST_PULLBACK)
    evaluation = _typed_dfp(
        triggered,
        Direction.LONG,
        protocol,
        None,
        BrainConfig(),
        global_context=context,
    )

    assert evaluation.plan is not None
    assert evaluation.plan.selected_draw_id == primary.item_id
    assert evaluation.target_selection_failure_reason is None
    assert not _frozen_target_crosses_authority_barrier(
        evaluation.plan,
        triggered,
        context,
        BrainConfig().tick_size,
    )


def test_dfp_frozen_target_only_weakens_for_unrelated_pre_target_barrier() -> None:
    _, _, _, _, triggered = _dfp_fixture()
    hypothesis = _brain().update(triggered).hypotheses[
        "displacement_first_pullback:long"
    ]
    assert hypothesis.plan is not None
    primary = triggered.liquidity_inventory[0]
    self_obstruction = DeliveryObstruction(
        obstruction_id="self-primary-obstruction",
        timeframe=Timeframe.H1,
        direction=None,
        side="above",
        lower_bound=101.5,
        upper_bound=101.5,
        hard=True,
        source_kind="protected_swing",
        source_ids=primary.source_ids,
        structural_scope="external",
        acceptance_state="confirmed",
    )
    unrelated = replace(
        self_obstruction,
        obstruction_id="unrelated-pre-target-obstruction",
        source_ids=("unrelated-pre-target-source",),
    )

    def context_with(
        obstruction: DeliveryObstruction,
    ) -> GlobalMarketContext:
        context = _brain_global_context(direction=Direction.LONG)
        views = dict(context.obstruction_views)
        views[Direction.LONG.value] = DirectionalObstructionView(
            direction=Direction.LONG,
            nearest_draw_id=primary.item_id,
            nearest_draw_price=primary.price,
            hard_barriers=(obstruction,),
            soft_frictions=(),
        )
        return replace(
            context,
            updated_at=triggered.asof,
            obstruction_views=views,
        )

    assert not _frozen_target_crosses_authority_barrier(
        hypothesis.plan,
        triggered,
        context_with(self_obstruction),
        BrainConfig().tick_size,
    )
    assert _frozen_target_crosses_authority_barrier(
        hypothesis.plan,
        triggered,
        context_with(unrelated),
        BrainConfig().tick_size,
    )
    assert hypothesis.plan.selected_draw_id == primary.item_id
    assert hypothesis.selected_trigger is not None


def test_dfp_armed_context_registers_thesis_before_entry_zone() -> None:
    _, _, draw, forming, _ = _dfp_fixture()
    context_only = replace(
        forming,
        entry_locations=(),
        qualified_reacceptances=(),
        micro_bos_references=(),
        path_sequences=(),
    )
    belief = _brain().update(context_only)
    hypothesis = belief.hypotheses[
        "displacement_first_pullback:long"
    ]

    assert hypothesis.phase is PlaybookPhase.ARMED
    assert hypothesis.plan is None
    assert hypothesis.thesis_draw is not None
    assert hypothesis.thesis_draw.level_id == draw.item_id

    recorder = BrainCalibrationRecorder()
    source_bar = Bar(
        start=context_only.asof - pd.Timedelta(minutes=1),
        open=context_only.price,
        high=context_only.price,
        low=context_only.price,
        close=context_only.price,
        volume=1.0,
        symbol=context_only.symbol,
        instrument_id=context_only.instrument_id,
    )
    recorder.observe(
        SimpleNamespace(
            observation=context_only,
            belief=graph_free_action_belief(belief),
        ),
        source_bar=source_bar,
    )

    thesis = [
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
        and sample.hypothesis_key == hypothesis.key
    ]
    assert len(thesis) == 1
    assert thesis[0].draw_id == draw.item_id
    assert thesis[0].calibration_unit_id == hypothesis.context_id


def test_dfp_context_requires_a_visible_h4_draw() -> None:
    _, _, h4_draw, _, triggered = _dfp_fixture()
    h1_internal = replace(
        h4_draw,
        item_id="h1-internal-not-dfp-context",
        timeframe=Timeframe.H1,
        kind="previous_day_high",
        price=h4_draw.price - 1.0,
        lower_bound=h4_draw.price - 1.0,
        upper_bound=h4_draw.price - 1.0,
        source_ids=("h1-internal-not-dfp-context-source",),
        structural_rank="internal",
        is_protected_swing=False,
    )

    without_h4_draw = replace(
        triggered,
        liquidity_inventory=(h1_internal,),
    )
    missing = _brain().update(without_h4_draw).hypotheses[
        "displacement_first_pullback:long"
    ]

    assert not missing.hard_gate_results["h4_structure_and_draw"]
    assert missing.phase is not PlaybookPhase.EXECUTABLE
    assert missing.thesis_draw is None

    with_both = replace(
        triggered,
        liquidity_inventory=(h1_internal, h4_draw),
    )
    valid = _brain().update(with_both).hypotheses[
        "displacement_first_pullback:long"
    ]

    assert valid.hard_gate_results["h4_structure_and_draw"]
    assert valid.thesis_draw is not None
    assert valid.thesis_draw.level_id == h4_draw.item_id
    assert valid.thesis_draw.timeframe is Timeframe.H4
    assert valid.plan is not None
    assert valid.plan.selected_draw_id == h1_internal.item_id


def test_dfp_optional_h1_acceptance_cannot_block_a_complete_episode() -> None:
    _, _, _, _, triggered = _dfp_fixture()
    frames = dict(triggered.frames)
    frames[Timeframe.H1] = replace(
        frames[Timeframe.H1],
        structure_breaks=(),
    )
    without_h1 = replace(triggered, frames=frames)

    hypothesis = _brain().update(without_h1).hypotheses[
        "displacement_first_pullback:long"
    ]

    assert hypothesis.phase is PlaybookPhase.EXECUTABLE
    assert hypothesis.sequence is not None
    assert tuple(step.step_id for step in hypothesis.sequence.steps) == (
        "h4_structure_and_draw",
        "m5_displacement_zone",
        "first_pullback_to_frozen_zone",
        "typed_entry_trigger",
    )
    assert all(hypothesis.hard_gate_results.values())
    assert "h1_continuation_bos" not in hypothesis.hard_gate_results
    assert hypothesis.plan is not None


def test_graph_backed_dfp_requires_and_receives_exact_open_thesis_binding() -> None:
    _, _, _, forming, triggered = _dfp_fixture()

    def with_displacement(observation: MarketObservation) -> MarketObservation:
        location = observation.entry_locations[0]
        return replace(
            observation,
            displacement=DisplacementObservation(
                asof=observation.asof,
                lifecycle="active",
                current_entity_id=location.source_displacement_id,
                current_direction=Direction.LONG,
                current_state_observed_at=observation.asof,
                current_started_at=(
                    location.formed_at - pd.Timedelta(minutes=10)
                ),
                current_active_at=(
                    location.formed_at - pd.Timedelta(minutes=5)
                ),
                current_last_admitted_at=observation.asof,
                current_metrics=(
                    ("origin_price", 99.0),
                    ("net_points", 2.0),
                    ("mean_overlap_ratio", 0.1),
                ),
                latest_transition=None,
            ),
        )

    graph = TemporalMarketSceneGraph()
    brain = _brain()
    first = with_displacement(forming)
    brain.update(
        first,
        scene_graph=graph,
        scene_delta=graph.update(first),
    )
    second = with_displacement(triggered)
    belief = brain.update(
        second,
        scene_graph=graph,
        scene_delta=graph.update(second),
    )
    hypothesis = belief.hypotheses[
        "displacement_first_pullback:long"
    ]

    assert hypothesis.phase is PlaybookPhase.EXECUTABLE
    assert hypothesis.market_thesis_action_bound
    assert len(hypothesis.market_thesis_ids) == 1
    assert hypothesis.market_thesis_match_status == "exact_root_bound"
    assert hypothesis.bound_market_thesis_id == hypothesis.market_thesis_id
    thesis = next(
        item
        for item in belief.market_theses
        if item.thesis_id == hypothesis.market_thesis_ids[0]
    )
    assert thesis.root_id == second.entry_locations[0].source_displacement_id
    assert thesis.mechanism == "directional_displacement"
    assert "previous-day:dfp-primary-draw" in thesis.draw_candidate_ids
    assert "swing:dfp-h4-draw" in thesis.draw_candidate_ids
    assert hypothesis.plan is not None
    assert (
        hypothesis.plan.selected_draw_id
        == "previous-day:dfp-primary-draw"
    )
    assert hypothesis.market_thesis_root_id == thesis.root_id
    assert hypothesis.market_thesis_mechanism == thesis.mechanism


def test_absent_root_live_frozen_path_continues_exact_entry_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, forming, triggered = _dfp_fixture()

    def with_displacement(
        observation: MarketObservation,
    ) -> MarketObservation:
        location = observation.entry_locations[0]
        return replace(
            observation,
            displacement=DisplacementObservation(
                asof=observation.asof,
                lifecycle="active",
                current_entity_id=location.source_displacement_id,
                current_direction=Direction.LONG,
                current_state_observed_at=observation.asof,
                current_started_at=(
                    location.formed_at - pd.Timedelta(minutes=10)
                ),
                current_active_at=(
                    location.formed_at - pd.Timedelta(minutes=5)
                ),
                current_last_admitted_at=observation.asof,
                current_metrics=(("origin_price", 99.0),),
                latest_transition=None,
            ),
        )

    graph = TemporalMarketSceneGraph()
    brain = _mapped_brain()
    first = with_displacement(forming)
    forming_belief = brain.update(
        first,
        scene_graph=graph,
        scene_delta=graph.update(first),
    )
    candidate_id, forming_candidate = (
        forming_belief.action_candidate_items()[0]
    )
    assert forming_candidate.episode_id is not None
    assert forming_candidate.entry_location_id is not None

    monkeypatch.setattr(
        "brain.core.playbooks.build_open_market_theses",
        lambda *_args, **_kwargs: (),
    )
    advanced = with_displacement(triggered)
    advanced_belief = brain.update(
        advanced,
        scene_graph=graph,
        scene_delta=graph.update(advanced),
    )

    continued = dict(advanced_belief.action_candidate_items())[candidate_id]
    assert advanced_belief.retained_episode_candidates == {}
    assert continued.record_kind == "root_candidate"
    assert continued.required_root_id == forming_candidate.required_root_id
    assert continued.episode_id == forming_candidate.episode_id
    assert continued.entry_location_id == forming_candidate.entry_location_id
    assert continued.phase is PlaybookPhase.EXECUTABLE
    assert continued.sequence is not None and continued.sequence.complete
    assert all(continued.hard_gate_results.values())
    assert continued.plan is not None
    assert continued.selected_trigger is not None


def test_absent_lsr_root_keeps_only_its_live_frozen_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _lsr_observation()
    waiting_at = observation.asof - pd.Timedelta(minutes=1)
    pool_path, entry_path = observation.path_sequences
    waiting_frames = {
        timeframe: replace(
            frame,
            cutoff=waiting_at,
            structure_breaks=(
                ()
                if timeframe is Timeframe.M1
                else frame.structure_breaks
            ),
        )
        for timeframe, frame in observation.frames.items()
    }
    waiting = replace_market_observation(
        observation,
        asof=waiting_at,
        frames=waiting_frames,
        execution=_execution(waiting_at, cost=0.025),
        entry_locations=(
            replace(
                observation.entry_locations[0],
                last_updated_at=waiting_at,
                age_real_1m_bars=1,
                state_duration_real_1m_bars=0,
            ),
        ),
        micro_bos_references=(),
        path_sequences=(
            replace(
                pool_path,
                lifecycle=PathSequenceLifecycle.ACTIVE,
                state_started_at=pool_path.steps[-2].observed_at,
                last_updated_at=waiting_at,
                age_real_1m_bars=4,
                state_duration_real_1m_bars=1,
                steps=pool_path.steps[:-1],
                ended_at=None,
                transition_reason="context_registered",
            ),
            replace(
                entry_path,
                lifecycle=PathSequenceLifecycle.ACTIVE,
                state_started_at=waiting_at,
                last_updated_at=waiting_at,
                age_real_1m_bars=1,
                state_duration_real_1m_bars=0,
                steps=entry_path.steps[:-1],
                ended_at=None,
                transition_reason="context_registered",
            ),
        ),
    )
    graph = TemporalMarketSceneGraph()
    brain = _mapped_brain()
    initial = brain.update(
        waiting,
        scene_graph=graph,
        scene_delta=graph.update(waiting),
    )
    candidate_id, candidate = next(
        (identity, item)
        for identity, item in initial.action_candidate_items()
        if item.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
        and item.direction is Direction.SHORT
        and item.market_thesis_match_status == "exact_root_bound"
    )
    assert candidate.phase is PlaybookPhase.WAITING_TRIGGER
    assert candidate.plan is not None
    assert candidate.selected_trigger is None
    assert candidate.entry_path_id == entry_path.sequence_id

    monkeypatch.setattr(
        "brain.core.playbooks.build_open_market_theses",
        lambda *_args, **_kwargs: (),
    )
    unresolved_at = waiting_at + pd.Timedelta(seconds=30)
    unresolved_observation = replace(
        _advance_observation(waiting, unresolved_at),
        entry_locations=(),
        path_sequences=(),
    )
    unresolved_belief = brain.update(
        unresolved_observation,
        scene_graph=graph,
        scene_delta=graph.update(unresolved_observation),
    )
    assert unresolved_belief.action_candidate_items() == ()
    dormant = unresolved_belief.retained_episode_candidates[candidate_id]
    assert dormant.record_kind == "retained_episode"
    assert dormant.entry_path_id == entry_path.sequence_id
    assert dormant.plan == candidate.plan
    assert dormant.plan_feasibility is not None
    assert not dormant.plan_feasibility.valid
    assert dormant.selected_trigger is None
    assert all(not value for value in dormant.hard_gate_results.values())
    assert (
        unresolved_belief.entry_episodes[candidate_id].entry_path_id
        == entry_path.sequence_id
    )

    continued_belief = brain.update(
        observation,
        scene_graph=graph,
        scene_delta=graph.update(observation),
    )

    continued = dict(continued_belief.action_candidate_items())[candidate_id]
    assert continued_belief.retained_episode_candidates == {}
    assert continued.required_root_id == candidate.required_root_id
    assert continued.episode_id == candidate.episode_id
    assert continued.entry_location_id == candidate.entry_location_id
    assert continued.selected_trigger is not None
    assert continued.phase is PlaybookPhase.EXECUTABLE
    assert continued.plan is not None
    assert continued.plan.entry_path_id == candidate.entry_path_id


def test_graph_visible_ineligible_then_absent_root_retains_without_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, forming, triggered = _dfp_fixture()

    def with_displacement(
        observation: MarketObservation,
    ) -> MarketObservation:
        location = observation.entry_locations[0]
        return replace(
            observation,
            displacement=DisplacementObservation(
                asof=observation.asof,
                lifecycle="active",
                current_entity_id=location.source_displacement_id,
                current_direction=Direction.LONG,
                current_state_observed_at=observation.asof,
                current_started_at=(
                    location.formed_at - pd.Timedelta(minutes=10)
                ),
                current_active_at=(
                    location.formed_at - pd.Timedelta(minutes=5)
                ),
                current_last_admitted_at=observation.asof,
                current_metrics=(("origin_price", 99.0),),
                latest_transition=None,
            ),
        )

    graph = TemporalMarketSceneGraph()
    brain = _mapped_brain()
    first = with_displacement(forming)
    brain.update(
        first,
        scene_graph=graph,
        scene_delta=graph.update(first),
    )
    entry_observation = with_displacement(triggered)
    entry_belief = brain.update(
        entry_observation,
        scene_graph=graph,
        scene_delta=graph.update(entry_observation),
    )
    candidate_id, executable = entry_belief.action_candidate_items()[0]
    assert executable.phase is PlaybookPhase.EXECUTABLE
    assert executable.plan is not None
    assert executable.selected_trigger is not None
    assert executable.episode_id is not None
    frozen_plan = executable.plan
    frozen_trigger = executable.selected_trigger
    frozen_episode_id = executable.episode_id
    context_id = executable.context_thesis_id
    assert context_id is not None

    current_theses: list[tuple[object, ...]] = [()]
    monkeypatch.setattr(
        "brain.core.playbooks.build_open_market_theses",
        lambda *_args, **_kwargs: current_theses[0],
    )

    # A root may remain visible while its current mechanism projection no
    # longer matches the playbook.  The frozen episode remains observable by
    # lifecycle diagnostics, but visibility alone must not restore action or
    # Focus authority.
    visible_at = entry_observation.asof + pd.Timedelta(minutes=1)
    visible_root = replace(
        next(
            thesis
            for thesis in entry_belief.market_theses
            if thesis.root_id == executable.required_root_id
        ),
        updated_at=visible_at,
        mechanism="structured_episode",
    )
    current_theses[0] = (visible_root,)
    visible_observation = replace(
        _advance_observation(
            replace(entry_observation, displacement=None),
            visible_at,
        ),
        entry_locations=(),
        path_sequences=(),
        displacement=replace(
            entry_observation.displacement,
            asof=visible_at,
        ),
    )
    visible_belief = brain.update(
        visible_observation,
        scene_graph=graph,
        scene_delta=graph.update(visible_observation),
    )

    assert visible_root.root_id in {
        thesis.root_id for thesis in visible_belief.market_theses
    }
    assert visible_belief.action_candidate_items() == ()
    assert visible_belief.focus_state is not None
    assert visible_belief.focus_state.hypothesis_id is None
    visible_retained = visible_belief.retained_episode_candidates[candidate_id]
    assert visible_retained.required_root_id == visible_root.root_id
    assert visible_retained.record_kind == "retained_episode"
    assert visible_retained.phase is not PlaybookPhase.EXECUTABLE
    assert (
        _ready_decision_layer(brain)
        .decide(visible_observation, visible_belief)
        .selected_action.value
        != "enter"
    )

    current_theses[0] = ()
    absent_at = visible_at + pd.Timedelta(minutes=1)
    absent = replace(
        _advance_observation(
            replace(entry_observation, displacement=None),
            absent_at,
        ),
        entry_locations=(),
        path_sequences=(),
        displacement=replace(
            entry_observation.displacement,
            asof=absent_at,
        ),
    )
    dormant_belief = brain.update(
        absent,
        scene_graph=graph,
        scene_delta=graph.update(absent),
    )

    assert dormant_belief.action_candidate_items() == ()
    assert dormant_belief.focus_state is not None
    assert dormant_belief.focus_state.hypothesis_id is None
    assert dormant_belief.focus_state.question == (
        "which registered mechanism explains the active market root"
    )
    assert tuple(dormant_belief.retained_episode_candidates) == (
        candidate_id,
    )
    dormant = dormant_belief.retained_episode_candidates[candidate_id]
    assert dormant.record_kind == "retained_episode"
    assert dormant.phase is PlaybookPhase.WEAKENING
    assert dormant.plan == frozen_plan
    assert dormant.selected_trigger == frozen_trigger
    assert dormant.episode_id == frozen_episode_id
    assert dormant.entry_location_id == frozen_plan.entry_location_id
    assert all(not value for value in dormant.hard_gate_results.values())
    assert dormant.plan_feasibility is not None
    assert not dormant.plan_feasibility.valid
    assert dormant_belief.context_theses[context_id].lifecycle == "active"
    assert (
        dormant_belief.entry_episodes[candidate_id].episode_id
        == frozen_episode_id
    )
    assert (
        _ready_decision_layer(brain)
        .decide(absent, dormant_belief)
        .selected_action.value
        != "enter"
    )

    reappeared_at = absent_at + pd.Timedelta(minutes=1)
    current_theses[0] = tuple(
        replace(thesis, updated_at=reappeared_at)
        for thesis in entry_belief.market_theses
    )
    reappeared_observation = _advance_observation(
        replace(entry_observation, displacement=None),
        reappeared_at,
    )
    reappeared_observation = replace(
        reappeared_observation,
        displacement=replace(
            entry_observation.displacement,
            asof=reappeared_at,
        ),
    )
    reappeared = brain.update(
        reappeared_observation,
        scene_graph=graph,
        scene_delta=graph.update(reappeared_observation),
    )
    restored = dict(reappeared.action_candidate_items())[candidate_id]
    assert restored.record_kind == "root_candidate"
    assert restored.episode_id == frozen_episode_id
    assert restored.plan == frozen_plan
    assert restored.selected_trigger == frozen_trigger
    assert restored.phase is PlaybookPhase.EXECUTABLE

    # A fill acknowledgement may arrive after the analytical root has closed.
    # The lifecycle-only snapshot must still resolve the exact frozen owner.
    current_theses[0] = ()
    second_absent_at = reappeared_at + pd.Timedelta(minutes=1)
    second_absent = replace(
        _advance_observation(
            replace(entry_observation, displacement=None),
            second_absent_at,
        ),
        entry_locations=(),
        path_sequences=(),
        displacement=replace(
            entry_observation.displacement,
            asof=second_absent_at,
        ),
    )
    brain.update(
        second_absent,
        scene_graph=graph,
        scene_delta=graph.update(second_absent),
    )
    position = PositionSnapshot(
        thesis_hash="delayed-dormant-position",
        symbol=entry_observation.symbol,
        instrument_id=entry_observation.instrument_id,
        playbook=executable.playbook,
        direction=executable.direction,
        entry_price=frozen_plan.planned_entry,
        original_invalidation=frozen_plan.invalidation,
        current_stop=frozen_plan.invalidation.price,
        primary_target=frozen_plan.targets[0],
        opened_at=reappeared_at,
        deadline=frozen_plan.deadline,
        quantity=1,
        unrealized_R=0.1,
        elapsed_minutes=2,
        mfe_R=0.1,
        mae_R=0.0,
        setup_id=frozen_plan.setup_id,
        entry_location_id=frozen_plan.entry_location_id,
        entry_path_id=frozen_plan.entry_path_id,
    )
    position_at = second_absent_at + pd.Timedelta(minutes=1)
    position_observation = replace(
        _advance_observation(
            replace(entry_observation, displacement=None),
            position_at,
        ),
        entry_locations=(),
        path_sequences=(),
        displacement=replace(
            entry_observation.displacement,
            asof=position_at,
        ),
    )
    position_belief = brain.update(
        position_observation,
        position=position,
        scene_graph=graph,
        scene_delta=graph.update(position_observation),
    )
    assert position_belief.action_candidate_items() == ()
    managed = dict(position_belief.position_candidate_items())[candidate_id]
    assert managed.plan == frozen_plan
    assert managed.selected_trigger == frozen_trigger
    assert managed.episode_id == frozen_episode_id
    account = AccountState(equity=100_000.0, position=position)
    management = _ready_decision_layer(brain).decide(
        position_observation,
        position_belief,
        account,
    )
    assert management.selected_action.value == "hold"


def test_context_keeps_sibling_episodes_isolated_and_cascades_once() -> None:
    _, _, _, forming, triggered = _dfp_fixture()

    def with_displacement(
        observation: MarketObservation,
    ) -> MarketObservation:
        location = observation.entry_locations[0]
        return replace(
            observation,
            displacement=DisplacementObservation(
                asof=observation.asof,
                lifecycle="active",
                current_entity_id=location.source_displacement_id,
                current_direction=Direction.LONG,
                current_state_observed_at=observation.asof,
                current_started_at=location.formed_at - pd.Timedelta(minutes=10),
                current_active_at=location.formed_at - pd.Timedelta(minutes=5),
                current_last_admitted_at=observation.asof,
                current_metrics=(("origin_price", 99.0),),
                latest_transition=None,
            ),
        )

    graph = TemporalMarketSceneGraph()
    brain = _mapped_brain()
    first = with_displacement(forming)
    brain.update(
        first,
        scene_graph=graph,
        scene_delta=graph.update(first),
    )
    observation = with_displacement(triggered)
    belief = brain.update(
        observation,
        scene_graph=graph,
        scene_delta=graph.update(observation),
    )
    candidate_id, episode_a = belief.action_candidate_items()[0]
    thesis_a = next(
        thesis
        for thesis in belief.market_theses
        if thesis.root_id == episode_a.required_root_id
    )
    assert episode_a.plan is not None
    assert episode_a.sequence is not None
    assert episode_a.selected_trigger is not None
    assert episode_a.plan_feasibility is not None
    context_id = episode_a.context_thesis_id
    assert context_id is not None

    root_b = "displacement:sibling-b"
    thesis_b = replace(
        thesis_a,
        thesis_id="market-thesis:sibling-b",
        root_id=root_b,
        mechanism_event_ids=(root_b,),
        evidence_state=None,
    )
    candidate_b_id = _thesis_candidate_id(
        belief.global_context.market_epoch_id,
        root_b,
        episode_a.playbook,
        episode_a.direction,
    )
    episode_b_id = "dfp-episode:sibling-b"
    location_b = "entry-location:sibling-b"
    path_b = "entry-path:sibling-b"
    trigger_b = replace(
        episode_a.selected_trigger,
        trigger_id="trigger:sibling-b",
        setup_id=episode_b_id,
        entry_location_id=location_b,
        entry_path_id=path_b,
        source_entity_id="micro-bos:sibling-b",
        source_event_id="bos:sibling-b",
    )
    plan_b = replace(
        episode_a.plan,
        setup_id=episode_b_id,
        entry_location_id=location_b,
        entry_path_id=path_b,
    )
    sequence_b = replace(episode_a.sequence, setup_id=episode_b_id)
    episode_b = replace(
        episode_a,
        candidate_id=candidate_b_id,
        required_root_id=root_b,
        market_thesis_ids=(thesis_b.thesis_id,),
        market_thesis_id=thesis_b.thesis_id,
        bound_market_thesis_id=thesis_b.thesis_id,
        market_thesis_root_id=root_b,
        setup_context_id=episode_b_id,
        episode_id=episode_b_id,
        initiating_event_id=root_b,
        entry_location_id=location_b,
        entry_path_id=path_b,
        sequence=sequence_b,
        selected_trigger=trigger_b,
        plan=plan_b,
        invalidation=plan_b.invalidation,
        deliverable_targets=plan_b.targets,
        plan_feasibility=replace(
            episode_a.plan_feasibility,
            planned_entry=plan_b.planned_entry,
            invalidation=plan_b.invalidation,
            target=plan_b.targets[0],
            deadline=plan_b.deadline,
        ),
    )
    local_terminal_at = observation.asof + pd.Timedelta(minutes=1)
    terminal_a = replace(
        episode_a,
        phase=PlaybookPhase.INVALIDATED,
        phase_started_at=local_terminal_at,
        terminal_at=local_terminal_at,
        terminal_reason="entry_zone_left_or_failed",
        terminal_source_ids=(episode_a.entry_location_id,),
    )
    later = _advance_observation(
        replace(observation, displacement=None),
        local_terminal_at,
    )
    later_context = replace(
        belief.global_context,
        updated_at=local_terminal_at,
    )
    contexts, episodes = _build_context_episode_views(
        later,
        later_context,
        {candidate_id: terminal_a, candidate_b_id: episode_b},
        {candidate_id: thesis_a, candidate_b_id: thesis_b},
        belief.context_theses,
    )
    context = contexts[context_id]
    assert context.lifecycle == "active"
    assert set(context.child_episode_ids) == {
        episode_a.episode_id,
        episode_b.episode_id,
    }
    assert episode_a.entry_location_id not in context.opposing_event_ids
    assert episodes[candidate_id].phase is PlaybookPhase.INVALIDATED
    assert episodes[candidate_b_id].phase is PlaybookPhase.EXECUTABLE
    assert episodes[candidate_b_id].entry_location_id == location_b
    assert episodes[candidate_b_id].entry_path_id == path_b
    assert episodes[candidate_b_id].selected_trigger == trigger_b

    # A terminal child may leave an empty interval without deleting its parent;
    # the current-child projection is bounded and the later sibling rejoins the
    # same stable Context identity.
    gap_at = local_terminal_at + pd.Timedelta(minutes=1)
    gap_observation = _advance_observation(later, gap_at)
    gap_context = replace(later_context, updated_at=gap_at)
    gap_contexts, gap_episodes = _build_context_episode_views(
        gap_observation,
        gap_context,
        {},
        {},
        contexts,
    )
    assert gap_episodes == {}
    assert gap_contexts[context_id].lifecycle == "active"
    assert gap_contexts[context_id].child_episode_ids == ()
    resumed_at = gap_at + pd.Timedelta(minutes=1)
    resumed_observation = _advance_observation(gap_observation, resumed_at)
    resumed_context, resumed_episodes = _build_context_episode_views(
        resumed_observation,
        replace(gap_context, updated_at=resumed_at),
        {candidate_b_id: episode_b},
        {candidate_b_id: thesis_b},
        gap_contexts,
    )
    assert resumed_context[context_id].context_thesis_id == context_id
    assert resumed_context[context_id].child_episode_ids == (episode_b_id,)
    assert tuple(resumed_episodes) == (candidate_b_id,)

    # A closure discovered from one sibling on this completed bar must expose
    # the canonical parent reason immediately.  Which cascaded child sorts
    # first cannot leak a transient ``parent_context_*`` reason into the
    # Context Thesis view.
    context_terminal_a = replace(
        episode_a,
        phase=PlaybookPhase.INVALIDATED,
        phase_started_at=resumed_at,
        terminal_at=resumed_at,
        terminal_reason="opposed_structure",
        terminal_source_ids=("h4-structure:sibling-test",),
    )
    same_bar_cascade = _cascade_context_terminals(
        resumed_observation,
        {
            candidate_id: context_terminal_a,
            candidate_b_id: episode_b,
        },
        {candidate_id: thesis_a, candidate_b_id: thesis_b},
    )
    same_bar_contexts, _ = _build_context_episode_views(
        resumed_observation,
        replace(gap_context, updated_at=resumed_at),
        same_bar_cascade,
        {candidate_id: thesis_a, candidate_b_id: thesis_b},
        gap_contexts,
    )
    assert same_bar_contexts[context_id].terminal_reason == "opposed_structure"
    assert same_bar_contexts[context_id].terminal_at == resumed_at

    # A parent closure is immutable and cascades to every later child without
    # moving its original terminal clock.
    closure = _ContextClosure(
        phase=PlaybookPhase.INVALIDATED,
        reason="context_frozen_source_invalidated",
        source_ids=(context_id, "protected-h4:sibling-test"),
        terminal_at=local_terminal_at,
        retain_until=resumed_at + pd.Timedelta(minutes=1),
    )
    cascaded = _cascade_context_terminals(
        resumed_observation,
        {candidate_b_id: episode_b},
        {candidate_b_id: thesis_b},
        {context_id: closure},
    )[candidate_b_id]
    assert cascaded.phase is PlaybookPhase.INVALIDATED
    assert cascaded.terminal_at == local_terminal_at
    assert cascaded.terminal_reason == (
        "parent_context_context_frozen_source_invalidated"
    )
    closed_contexts, closed_episodes = _build_context_episode_views(
        resumed_observation,
        replace(gap_context, updated_at=resumed_at),
        {candidate_b_id: cascaded},
        {candidate_b_id: thesis_b},
        gap_contexts,
        {context_id: closure},
    )
    assert closed_contexts[context_id].lifecycle == "invalidated"
    assert closed_contexts[context_id].terminal_at == local_terminal_at
    assert closed_contexts[context_id].terminal_reason == (
        "context_frozen_source_invalidated"
    )
    assert closed_episodes[candidate_b_id].terminal_at == local_terminal_at


def test_dfp_context_terminal_roles_exclude_mutable_legs_and_local_episode() -> None:
    _, _, draw, forming, triggered = _dfp_fixture()
    structure_id = "h4-long-structure"
    context_draw_id = draw.item_id
    mutable_high_id = "h4-mutable-high"
    mutable_low_id = "h4-mutable-low"
    protected_id = "h4-frozen-protected"

    def with_distinct_context_roles(
        observation: MarketObservation,
    ) -> MarketObservation:
        frames = dict(observation.frames)
        structure = frames[Timeframe.H4].structures[0]
        frames[Timeframe.H4] = replace(
            frames[Timeframe.H4],
            structures=(
                replace(
                    structure,
                    latest_high_id=mutable_high_id,
                    latest_low_id=mutable_low_id,
                    protected_swing_id=protected_id,
                ),
            ),
        )
        return replace(observation, frames=frames)

    forming = with_distinct_context_roles(forming)
    triggered = with_distinct_context_roles(triggered)
    brain = _mapped_brain()
    protocol = brain.registry.for_playbook(
        Playbook.DISPLACEMENT_FIRST_PULLBACK
    )
    evaluation = _typed_dfp(
        triggered,
        Direction.LONG,
        protocol,
        None,
        brain.config,
    )
    context_signal = evaluation.sequence_signals[
        "h4_structure_and_draw"
    ]
    assert context_signal.source_ids == (
        structure_id,
        context_draw_id,
        mutable_high_id,
        mutable_low_id,
        protected_id,
    )
    assert _dfp_context_terminal_source_ids(evaluation) == (
        structure_id,
        context_draw_id,
        protected_id,
    )
    short_context_candidate = replace(
        evaluation,
        sequence_signals={
            **evaluation.sequence_signals,
            "h4_structure_and_draw": replace(
                context_signal,
                source_ids=context_signal.source_ids[:4],
            ),
        },
    )
    assert _dfp_context_terminal_source_ids(short_context_candidate) == ()
    epoch_id = "epoch:dfp-context-identity"
    valid_context_id = _stable_context_thesis_id(
        epoch_id,
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        evaluation,
        None,
    )
    assert valid_context_id is not None
    assert (
        _stable_context_thesis_id(
            epoch_id,
            Playbook.DISPLACEMENT_FIRST_PULLBACK,
            Direction.LONG,
            short_context_candidate,
            None,
        )
        is None
    )
    assert (
        _stable_context_thesis_id(
            epoch_id,
            Playbook.DISPLACEMENT_FIRST_PULLBACK,
            Direction.LONG,
            replace(evaluation, sequence_signals={}),
            None,
        )
        is None
    )

    # A cached root cannot resurrect a parent merely because the short-lived
    # prior still names it.  The parent itself must remain in the prior
    # Context view or in the bounded closure tombstones.
    summary_prior = _mapped_brain().update(triggered).hypotheses[
        "displacement_first_pullback:long"
    ]
    missing_current_context = replace(
        short_context_candidate,
        context_identity=None,
    )
    stale_prior = replace(
        summary_prior,
        context_id=None,
        context_thesis_id="context-thesis:stale-without-native-context",
        parent_context_thesis_id=None,
        episode_id=None,
        episode_deadline=None,
    )
    assert (
        _stable_context_thesis_id(
            epoch_id,
            Playbook.DISPLACEMENT_FIRST_PULLBACK,
            Direction.LONG,
            missing_current_context,
            stale_prior,
        )
        is None
    )
    live_prior = replace(
        stale_prior,
        context_id="dfp-native-context:live",
        context_thesis_id=valid_context_id,
    )
    assert (
        _stable_context_thesis_id(
            epoch_id,
            Playbook.DISPLACEMENT_FIRST_PULLBACK,
            Direction.LONG,
            missing_current_context,
            live_prior,
        )
        is None
    )
    assert _stable_context_thesis_id(
        epoch_id,
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        missing_current_context,
        live_prior,
        known_context_ids={valid_context_id},
    ) == valid_context_id

    # Exact replay failure shape: the DFP evaluator still exposes its native
    # Context identity, but its frozen H4 authority roles are incomplete and
    # its formerly named parent is absent from both lifecycle stores.  The
    # stale short-term prior cannot recreate an authority-less Context.
    matched_stale_prior = replace(
        live_prior,
        context_id=short_context_candidate.context_identity,
    )
    assert (
        _stable_context_thesis_id(
            epoch_id,
            Playbook.DISPLACEMENT_FIRST_PULLBACK,
            Direction.LONG,
            short_context_candidate,
            matched_stale_prior,
        )
        is None
    )
    # Complete current roles remain independently authoritative and rebuild
    # the canonical parent rather than inheriting an unknown prior identity.
    assert _stable_context_thesis_id(
        epoch_id,
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        evaluation,
        replace(
            matched_stale_prior,
            context_thesis_id="context-thesis:unknown-stale-parent",
        ),
    ) == valid_context_id

    closed_context_id = "context-thesis:bounded-closure-bridge"
    closed_prior = replace(
        matched_stale_prior,
        context_thesis_id=closed_context_id,
    )
    assert _stable_context_thesis_id(
        epoch_id,
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        short_context_candidate,
        closed_prior,
        known_context_ids={closed_context_id},
    ) == closed_context_id

    global_context = replace(
        _brain_global_context(direction=Direction.LONG),
        updated_at=triggered.asof,
    )
    for mutable_id in (mutable_high_id, mutable_low_id):
        routed = _route_global_context(
            Playbook.DISPLACEMENT_FIRST_PULLBACK,
            Direction.LONG,
            None,
            evaluation,
            replace(
                global_context,
                invalidated_source_ids=(mutable_id,),
            ),
        )
        assert routed.invalidated is False

    brain.update(forming)
    belief = brain.update(triggered)
    candidate = belief.hypotheses[
        "displacement_first_pullback:long"
    ]
    assert candidate.entry_location_id is not None
    assert candidate.entry_path_id is not None
    assert candidate.context_thesis_id is not None
    contexts, _ = _build_context_episode_views(
        triggered,
        global_context,
        {"episode-a": candidate},
        {},
        {},
    )
    context = contexts[candidate.context_thesis_id]
    assert context.authority_ids == (structure_id, protected_id)
    assert context.structural_invalidation is not None
    assert context.structural_invalidation.source_level_id == protected_id
    assert mutable_high_id in context.supporting_event_ids
    assert mutable_low_id in context.supporting_event_ids

    def global_terminal(
        source_id: str,
    ) -> object:
        return replace(
            candidate,
            phase=PlaybookPhase.INVALIDATED,
            phase_started_at=triggered.asof,
            terminal_at=triggered.asof,
            terminal_reason="global_frozen_source_invalidated",
            terminal_source_ids=(source_id,),
        )

    local_ids = (
        candidate.entry_location_id,
        candidate.entry_path_id,
    )
    for source_id in (
        *local_ids,
        candidate.context_id,
        mutable_high_id,
        mutable_low_id,
    ):
        terminal = global_terminal(source_id)
        assert _context_level_terminal(terminal, None, context) is None
        terminal_evaluation = replace(
            evaluation,
            invalidated=True,
            terminal_reason="global_frozen_source_invalidated",
            terminal_source_ids=(source_id,),
        )
        assert (
            _position_context_terminal_phase(
                candidate,
                terminal_evaluation,
                triggered,
            )
            is None
        )

    for source_id in local_ids:
        routed = _route_global_context(
            Playbook.DISPLACEMENT_FIRST_PULLBACK,
            Direction.LONG,
            candidate,
            evaluation,
            replace(
                global_context,
                invalidated_source_ids=(source_id,),
            ),
            context_thesis=context,
        )
        assert routed.invalidated is True
        assert routed.terminal_reason == "global_frozen_source_invalidated"
        assert routed.terminal_source_ids == (source_id,)

    for source_id in (structure_id, protected_id):
        terminal = global_terminal(source_id)
        assert _context_level_terminal(terminal, None, context) == (
            "invalidated",
            "global_frozen_source_invalidated",
        )

    sibling = candidate
    local_terminal = global_terminal(candidate.entry_location_id)
    child_only = _cascade_context_terminals(
        triggered,
        {"episode-a": local_terminal, "episode-b": sibling},
        {},
        context_theses={candidate.context_thesis_id: context},
    )
    assert child_only["episode-b"].phase is candidate.phase
    protected_terminal = global_terminal(protected_id)
    context_closed = _cascade_context_terminals(
        triggered,
        {"episode-a": protected_terminal, "episode-b": sibling},
        {},
        context_theses={candidate.context_thesis_id: context},
    )
    assert context_closed["episode-b"].phase is PlaybookPhase.INVALIDATED
    assert context_closed["episode-b"].terminal_reason == (
        "parent_context_global_frozen_source_invalidated"
    )

    consumed_draw = replace(
        draw,
        lifecycle=LiquidityInventoryLifecycle.CONSUMED,
        consumed_at=triggered.asof,
        lifecycle_reason="swing_swept",
    )
    consumed_observation = replace(
        triggered,
        liquidity_inventory=tuple(
            consumed_draw if item.item_id == draw.item_id else item
            for item in triggered.liquidity_inventory
        ),
    )
    draw_closed = _cascade_context_terminals(
        consumed_observation,
        {"episode-a": sibling, "episode-b": candidate},
        {},
        context_theses={candidate.context_thesis_id: context},
    )
    assert all(
        item.phase is PlaybookPhase.COMPLETED
        for item in draw_closed.values()
    )

    new_protected_id = "h4-new-supporting-protected"
    revised_steps = tuple(
        replace(
            step,
            source_ids=(
                structure_id,
                context_draw_id,
                "h4-new-high",
                "h4-new-low",
                new_protected_id,
            ),
        )
        if step.step_id == "h4_structure_and_draw"
        else step
        for step in candidate.sequence.steps
    )
    revised_candidate = replace(
        candidate,
        sequence=replace(candidate.sequence, steps=revised_steps),
    )
    revised_contexts, _ = _build_context_episode_views(
        triggered,
        global_context,
        {"episode-c": revised_candidate},
        {},
        {candidate.context_thesis_id: context},
    )
    revised_context = revised_contexts[candidate.context_thesis_id]
    assert revised_context.authority_ids == context.authority_ids
    assert revised_context.structural_invalidation == (
        context.structural_invalidation
    )
    assert new_protected_id in revised_context.supporting_event_ids
    new_protected_terminal = replace(
        revised_candidate,
        phase=PlaybookPhase.INVALIDATED,
        terminal_at=triggered.asof,
        terminal_reason="global_frozen_source_invalidated",
        terminal_source_ids=(new_protected_id,),
    )
    assert (
        _context_level_terminal(
            new_protected_terminal,
            None,
            revised_context,
        )
        is None
    )


def test_runtime_root_does_not_promote_incomplete_native_context_to_parent() -> None:
    """The exact replay failure remains a diagnostic root, not a ghost Context."""

    _, _, _, _, observation = _dfp_fixture()
    candidate_id = "thesis-candidate:incomplete-dfp-context"
    root_id = "displacement:incomplete-dfp-context"
    thesis_id = "market-thesis:incomplete-dfp-context"
    candidate = HypothesisBelief(
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        direction=Direction.SHORT,
        probability=0.0,
        phase=PlaybookPhase.INVALIDATED,
        phase_started_at=observation.asof,
        supporting=(),
        contradicting=(),
        invalidation=None,
        deliverable_targets=(),
        remaining_path_R=None,
        uncertainty=1.0,
        plan=None,
        thesis_strength=0.0,
        sequence_progress=0.0,
        location_quality=0.0,
        entry_readiness=0.0,
        delivery_quality=0.0,
        evidence_group_scores={
            name: 0.0
            for name in (
                "structure",
                "displacement",
                "location",
                "liquidity",
                "trigger",
                "execution",
            )
        },
        hard_gate_results={"h4_structure_and_draw": False},
        raw_quality_dimensions={
            "thesis_strength": 0.0,
            "sequence_progress": 0.0,
            "location_quality": 0.0,
            "entry_readiness": 0.0,
            "delivery_quality": 0.0,
            "uncertainty": 1.0,
        },
        context_id="dfp-native-context:incomplete",
        context_thesis_id=None,
        parent_context_thesis_id=None,
        episode_id=None,
        episode_deadline=None,
        terminal_at=observation.asof,
        terminal_reason="global_frozen_source_invalidated",
        terminal_source_ids=(root_id,),
        market_thesis_ids=(thesis_id,),
        market_thesis_id=thesis_id,
        bound_market_thesis_id=None,
        market_thesis_root_id=root_id,
        market_thesis_mechanism="directional_displacement",
        market_thesis_authority_relation="local_countertrend",
        playbook_match_strength=0.5,
        market_thesis_binding_required=True,
        market_thesis_action_bound=False,
        market_thesis_match_status="root_identity_unbound",
        candidate_id=candidate_id,
        required_root_id=root_id,
        record_kind="root_candidate",
    )
    assert candidate.context_id == "dfp-native-context:incomplete"
    assert candidate.context_thesis_id is None
    assert candidate.episode_id is None
    assert candidate.episode_deadline is None

    global_context = replace(
        _brain_global_context(direction=Direction.SHORT),
        updated_at=observation.asof,
    )
    contexts, episodes = _build_context_episode_views(
        observation,
        global_context,
        {candidate_id: candidate},
        {},
        {},
    )
    assert contexts == {}
    assert episodes == {}

    # The legacy fallback remains limited to graph-free/summary beliefs.
    summary = replace(
        candidate,
        candidate_id=None,
        required_root_id=None,
        record_kind="summary",
        context_thesis_id=None,
    )
    assert summary.context_thesis_id == candidate.context_id


def test_terminal_prior_sequence_restore_cannot_create_unknown_context_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale local closure cannot take custody of a newly derived parent."""

    _, _, _, forming, triggered = _dfp_fixture()

    def seeded_case(*, known_parent: bool):
        graph = TemporalMarketSceneGraph()
        brain = _mapped_brain()
        seed = brain.update(
            forming,
            scene_graph=graph,
            scene_delta=graph.update(forming),
        )
        candidate_id, candidate = next(
            (identity, item)
            for identity, item in seed.thesis_candidates.items()
            if item.playbook
            is Playbook.DISPLACEMENT_FIRST_PULLBACK
            and item.direction is Direction.LONG
        )
        assert candidate.context_thesis_id is not None
        assert candidate.sequence is not None
        stale_sequence = replace(
            candidate.sequence,
            setup_id="stale-terminal-setup",
            steps=tuple(
                replace(
                    step,
                    satisfied=False,
                    value=0.0,
                    observed_at=None,
                    source_ids=(),
                )
                if step.step_id == "h4_structure_and_draw"
                else step
                for step in candidate.sequence.steps
            ),
        )
        parent_id = (
            candidate.context_thesis_id
            if known_parent
            else "context-thesis:unknown-stale-terminal"
        )
        stale = replace(
            candidate,
            sequence=stale_sequence,
            setup_context_id=stale_sequence.setup_id,
            context_id="dfp-native:stale-terminal",
            context_thesis_id=parent_id,
            parent_context_thesis_id=parent_id,
            episode_id=stale_sequence.setup_id,
            phase=PlaybookPhase.ARMED,
        )
        brain._candidate_priors[candidate_id] = stale
        if not known_parent:
            # Preserve the diagnostic root but remove all proof that its
            # historical parent remains a live or bounded-closed Context.
            diagnostic = replace(
                stale,
                context_thesis_id=None,
                parent_context_thesis_id=None,
                episode_id=None,
                episode_deadline=None,
            )
            brain._belief = replace(
                seed,
                context_theses={},
                entry_episodes={},
                thesis_candidates={candidate_id: diagnostic},
            )
        return graph, brain, candidate_id, candidate.context_thesis_id

    unknown_graph, unknown_brain, unknown_id, _ = seeded_case(
        known_parent=False
    )
    known_graph, known_brain, known_id, known_parent_id = seeded_case(
        known_parent=True
    )
    original = _typed_dfp

    def terminal_setup_mismatch(*args, **kwargs):
        evaluation = original(*args, **kwargs)
        if args[1] is Direction.LONG:
            return replace(
                evaluation,
                invalidated=True,
                terminal_reason="diagnostic_terminal_setup_mismatch",
                terminal_source_ids=("diagnostic-terminal-source",),
            )
        return evaluation

    monkeypatch.setattr(
        "brain.core.playbooks._typed_dfp",
        terminal_setup_mismatch,
    )
    unknown = unknown_brain.update(
        triggered,
        scene_graph=unknown_graph,
        scene_delta=unknown_graph.update(triggered),
    )
    unknown_candidate = unknown.thesis_candidates[unknown_id]
    assert unknown_candidate.phase is PlaybookPhase.INVALIDATED
    assert unknown_candidate.sequence.setup_id == "stale-terminal-setup"
    assert unknown_candidate.context_thesis_id is None
    assert unknown_candidate.episode_id is None
    assert unknown.context_theses == {}
    assert unknown.entry_episodes == {}

    known = known_brain.update(
        triggered,
        scene_graph=known_graph,
        scene_delta=known_graph.update(triggered),
    )
    known_candidate = known.thesis_candidates[known_id]
    assert known_candidate.phase is PlaybookPhase.INVALIDATED
    assert known_candidate.sequence.setup_id == "stale-terminal-setup"
    assert known_candidate.context_thesis_id == known_parent_id
    assert known_candidate.episode_id == "stale-terminal-setup"
    assert known.context_theses[known_parent_id].authority_ids


def test_public_brain_fails_closed_when_two_context_roots_claim_one_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, forming, triggered = _dfp_fixture()

    def with_displacement(
        observation: MarketObservation,
    ) -> MarketObservation:
        location = observation.entry_locations[0]
        return replace(
            observation,
            displacement=DisplacementObservation(
                asof=observation.asof,
                lifecycle="active",
                current_entity_id=location.source_displacement_id,
                current_direction=Direction.LONG,
                current_state_observed_at=observation.asof,
                current_started_at=location.formed_at - pd.Timedelta(minutes=10),
                current_active_at=location.formed_at - pd.Timedelta(minutes=5),
                current_last_admitted_at=observation.asof,
                current_metrics=(("origin_price", 99.0),),
                latest_transition=None,
            ),
        )

    seed_graph = TemporalMarketSceneGraph()
    seed_brain = _mapped_brain()
    seed_brain.update(
        with_displacement(forming),
        scene_graph=seed_graph,
        scene_delta=seed_graph.update(with_displacement(forming)),
    )
    seed_observation = with_displacement(triggered)
    seed_belief = seed_brain.update(
        seed_observation,
        scene_graph=seed_graph,
        scene_delta=seed_graph.update(seed_observation),
    )
    primary = next(
        thesis
        for thesis in seed_belief.market_theses
        if thesis.mechanism == "directional_displacement"
        and thesis.direction is Direction.LONG
    )
    secondary = replace(
        primary,
        thesis_id="market-thesis:ambiguous-dfp-owner",
        root_id="displacement:ambiguous-dfp-owner",
        # The same mechanism event makes both projections exact-bound while
        # their canonical Context roots remain distinct.  The ownership
        # resolver must therefore reject both rather than pick one.
        mechanism_event_ids=(
            "displacement:ambiguous-dfp-owner",
            *primary.mechanism_event_ids,
        ),
        evidence_state=None,
    )

    monkeypatch.setattr(
        "brain.core.playbooks.build_open_market_theses",
        lambda _previous, current, *_args, **_kwargs: (
            replace(primary, updated_at=current.asof),
            replace(secondary, updated_at=current.asof),
        ),
    )
    graph = TemporalMarketSceneGraph()
    brain = _mapped_brain()
    initial = with_displacement(forming)
    brain.update(
        initial,
        scene_graph=graph,
        scene_delta=graph.update(initial),
    )
    belief = brain.update(
        seed_observation,
        scene_graph=graph,
        scene_delta=graph.update(seed_observation),
    )

    assert len(belief.market_theses) == 2
    assert belief.action_candidate_items() == ()
    assert belief.dominant_hypothesis_id is None
    assert belief.competing_hypothesis_ids == ()
    assert belief.focus_state is not None
    assert belief.focus_state.hypothesis_id is None
    assert all(
        summary.summary_source_candidate_id is None
        for summary in belief.hypotheses.values()
    )
    decision = _ready_decision_layer(brain).decide(seed_observation, belief)
    assert decision.selected_action.value == "abstain"


def test_context_tombstones_are_pruned_after_their_causal_grace() -> None:
    observation = market_observation()
    graph = TemporalMarketSceneGraph()
    brain = _brain()
    brain.update(
        observation,
        scene_graph=graph,
        scene_delta=graph.update(observation),
    )
    brain._closed_contexts["context-thesis:expired-test"] = _ContextClosure(
        phase=PlaybookPhase.INVALIDATED,
        reason="context_frozen_source_invalidated",
        source_ids=("protected-h4:expired-test",),
        terminal_at=observation.asof - pd.Timedelta(minutes=1),
        retain_until=observation.asof,
    )
    later = _advance_observation(
        observation,
        observation.asof + pd.Timedelta(minutes=1),
    )
    brain.update(
        later,
        scene_graph=graph,
        scene_delta=graph.update(later),
    )
    assert "context-thesis:expired-test" not in brain._closed_contexts


def test_lsr_context_child_spawn_guard_requires_a_live_known_parent() -> None:
    observation = _lsr_without_entry_trigger(_lsr_observation())
    graph = TemporalMarketSceneGraph()
    belief = _brain().update(
        observation,
        scene_graph=graph,
        scene_delta=graph.update(observation),
    )
    context = next(iter(belief.context_theses.values()))
    context_id = context.context_thesis_id

    assert _lsr_context_can_spawn_child(
        observation,
        context_id,
        belief.context_theses,
        {},
    )
    assert not _lsr_context_can_spawn_child(
        observation,
        context_id,
        {
            context_id: replace(
                context,
                lifecycle="invalidated",
                terminal_at=observation.asof,
                terminal_reason="frozen_invalidation_breached",
            )
        },
        {},
    )
    assert not _lsr_context_can_spawn_child(
        observation,
        context_id,
        {
            context_id: replace(
                context,
                thesis_deadline=observation.asof,
            )
        },
        {},
    )
    assert not _lsr_context_can_spawn_child(
        observation,
        context_id,
        belief.context_theses,
        {
            context_id: _ContextClosure(
                phase=PlaybookPhase.INVALIDATED,
                reason="context_frozen_source_invalidated",
                source_ids=(context_id,),
                terminal_at=observation.asof,
                retain_until=observation.asof + pd.Timedelta(minutes=1),
            )
        },
    )
    # A formerly known parent cannot rearm merely because its bounded view and
    # tombstone have disappeared.  Only an unseen manipulation root (which has
    # no stable Context identity yet) may establish a new parent.
    assert not _lsr_context_can_spawn_child(
        observation,
        context_id,
        {},
        {},
    )
    assert _lsr_context_can_spawn_child(
        observation,
        None,
        {},
        {},
    )


def test_graph_backed_position_retains_frozen_root_after_open_thesis_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, forming, triggered = _dfp_fixture()

    def with_displacement(
        observation: MarketObservation,
    ) -> MarketObservation:
        location = observation.entry_locations[0]
        return replace(
            observation,
            displacement=DisplacementObservation(
                asof=observation.asof,
                lifecycle="active",
                current_entity_id=location.source_displacement_id,
                current_direction=Direction.LONG,
                current_state_observed_at=observation.asof,
                current_started_at=(
                    location.formed_at - pd.Timedelta(minutes=10)
                ),
                current_active_at=(
                    location.formed_at - pd.Timedelta(minutes=5)
                ),
                current_last_admitted_at=observation.asof,
                current_metrics=(
                    ("origin_price", 99.0),
                    ("net_points", 2.0),
                    ("mean_overlap_ratio", 0.1),
                ),
                latest_transition=None,
            ),
        )

    graph = TemporalMarketSceneGraph()
    brain = _mapped_brain()
    first = with_displacement(forming)
    brain.update(
        first,
        scene_graph=graph,
        scene_delta=graph.update(first),
    )
    entry_observation = with_displacement(triggered)
    entry_belief = brain.update(
        entry_observation,
        scene_graph=graph,
        scene_delta=graph.update(entry_observation),
    )
    candidate_id, entry_candidate = entry_belief.action_candidate_items()[0]
    assert entry_candidate.phase is PlaybookPhase.EXECUTABLE
    assert entry_candidate.plan is not None
    frozen_plan = entry_candidate.plan
    frozen_trigger = entry_candidate.selected_trigger
    assert frozen_trigger is not None

    entry_decision = _ready_decision_layer(brain).decide(
        entry_observation,
        entry_belief,
    )
    assert entry_decision.selected_action.value == "enter"
    assert entry_decision.best_hypothesis_key == candidate_id

    position = PositionSnapshot(
        thesis_hash="graph-backed-filled-dfp",
        symbol=entry_observation.symbol,
        instrument_id=entry_observation.instrument_id,
        playbook=entry_candidate.playbook,
        direction=entry_candidate.direction,
        entry_price=frozen_plan.planned_entry,
        original_invalidation=frozen_plan.invalidation,
        current_stop=frozen_plan.invalidation.price,
        primary_target=frozen_plan.targets[0],
        opened_at=entry_observation.asof,
        deadline=frozen_plan.deadline,
        quantity=1,
        unrealized_R=0.25,
        elapsed_minutes=1,
        mfe_R=0.25,
        mae_R=0.0,
        setup_id=frozen_plan.setup_id,
        entry_location_id=frozen_plan.entry_location_id,
        entry_path_id=frozen_plan.entry_path_id,
    )

    # Reproduce a graph-backed clock where the market root is no longer an
    # *open* analytical thesis.  The filled position still owns its exact
    # frozen candidate identity; losing it from the current discovery set is
    # not itself a structural exit condition.
    monkeypatch.setattr(
        "brain.core.playbooks.build_open_market_theses",
        lambda *_args, **_kwargs: (),
    )
    primary_inventory = next(
        item
        for item in entry_observation.liquidity_inventory
        if item.item_id == frozen_plan.selected_draw_id
    )
    active_barriers = [
        DeliveryObstruction(
            obstruction_id="position-self-target-obstruction",
            timeframe=Timeframe.H1,
            direction=None,
            side=primary_inventory.side,
            lower_bound=primary_inventory.price - 0.5,
            upper_bound=primary_inventory.price - 0.5,
            hard=True,
            source_kind="protected_swing",
            source_ids=primary_inventory.source_ids,
            structural_scope="external",
            acceptance_state="confirmed",
        )
    ]

    def with_position_barrier(
        previous: GlobalMarketContext | None,
        current_observation: MarketObservation,
        current_delta,
        _graph: TemporalMarketSceneGraph,
    ) -> GlobalMarketContext:
        context = previous or entry_belief.global_context
        assert context is not None
        views = dict(context.obstruction_views)
        views[Direction.LONG.value] = DirectionalObstructionView(
            direction=Direction.LONG,
            nearest_draw_id=primary_inventory.item_id,
            nearest_draw_price=primary_inventory.price,
            hard_barriers=tuple(active_barriers),
            soft_frictions=(),
        )
        return replace(
            context,
            updated_at=current_observation.asof,
            scene_revision_id=current_delta.revision_id,
            obstruction_views=views,
        )

    monkeypatch.setattr(
        "brain.core.playbooks.update_global_market_context",
        with_position_barrier,
    )
    next_asof = entry_observation.asof + pd.Timedelta(minutes=1)
    next_observation = replace_market_observation(
        entry_observation,
        asof=next_asof,
        price=100.75,
        frames={
            timeframe: replace(frame, cutoff=next_asof)
            for timeframe, frame in entry_observation.frames.items()
        },
        execution=_execution(next_asof, cost=0.10),
        displacement=replace(
            entry_observation.displacement,
            asof=next_asof,
        ),
    )
    next_belief = brain.update(
        next_observation,
        position=position,
        scene_graph=graph,
        scene_delta=graph.update(next_observation),
    )

    assert next_belief.market_theses == ()
    assert next_belief.action_candidate_items() == ()
    assert tuple(
        identity
        for identity, _ in next_belief.position_candidate_items()
    ) == (candidate_id,)
    retained = next_belief.resolve_hypothesis(candidate_id)
    assert retained is not None
    assert retained.required_root_id == entry_candidate.required_root_id
    assert retained.phase is PlaybookPhase.DELIVERING
    assert retained.plan == frozen_plan
    assert retained.selected_trigger == frozen_trigger
    assert retained.setup_context_id == frozen_plan.setup_id
    assert retained.entry_location_id == frozen_plan.entry_location_id
    assert retained.plan.entry_path_id == frozen_plan.entry_path_id
    assert set(brain._candidate_priors) == {candidate_id}
    assert set(brain._candidate_theses) == {candidate_id}

    account = AccountState(equity=100_000.0, position=position)
    management = _ready_decision_layer(brain).decide(
        next_observation,
        next_belief,
        account,
    )
    assert management.selected_action.value == "hold"
    assert management.best_hypothesis_key == candidate_id

    flat_recheck = _ready_decision_layer(brain).decide(
        next_observation,
        next_belief,
    )
    assert flat_recheck.selected_action.value != "enter"

    # Retention is continuous rather than a one-clock grace period.  A new
    # unrelated accepted barrier before the target weakens delivery without
    # rewriting the frozen position plan or trigger.
    active_barriers[:] = [
        replace(
            active_barriers[0],
            obstruction_id="position-unrelated-pre-target-obstruction",
            source_ids=("position-unrelated-pre-target-source",),
        )
    ]
    later_asof = next_asof + pd.Timedelta(minutes=1)
    later_observation = replace_market_observation(
        next_observation,
        asof=later_asof,
        frames={
            timeframe: replace(frame, cutoff=later_asof)
            for timeframe, frame in next_observation.frames.items()
        },
        execution=_execution(later_asof, cost=0.10),
        displacement=replace(
            next_observation.displacement,
            asof=later_asof,
        ),
    )
    later_position = replace(position, elapsed_minutes=2)
    later_belief = brain.update(
        later_observation,
        position=later_position,
        scene_graph=graph,
        scene_delta=graph.update(later_observation),
    )
    later_retained = later_belief.resolve_hypothesis(candidate_id)
    assert later_retained is not None
    assert later_retained.phase is PlaybookPhase.WEAKENING
    assert later_retained.plan == frozen_plan
    assert later_retained.selected_trigger == frozen_trigger
    assert set(brain._candidate_priors) == {candidate_id}
    assert set(brain._candidate_theses) == {candidate_id}

    # The terminal lifecycle clock may still resolve the frozen root.  Once
    # the portfolio releases that lifecycle position, the management-only
    # candidate is removed rather than becoming a latent entry candidate.
    terminal_asof = later_asof + pd.Timedelta(minutes=1)
    terminal_observation = replace_market_observation(
        later_observation,
        asof=terminal_asof,
        frames={
            timeframe: replace(frame, cutoff=terminal_asof)
            for timeframe, frame in later_observation.frames.items()
        },
        execution=_execution(terminal_asof, cost=0.10),
        displacement=replace(
            later_observation.displacement,
            asof=terminal_asof,
        ),
    )
    terminal_belief = brain.update(
        terminal_observation,
        position=replace(
            later_position,
            elapsed_minutes=3,
            status="completed",
        ),
        scene_graph=graph,
        scene_delta=graph.update(terminal_observation),
    )
    terminal_candidate = terminal_belief.resolve_hypothesis(candidate_id)
    assert terminal_candidate is not None
    assert terminal_candidate.phase is PlaybookPhase.COMPLETED
    assert terminal_candidate.plan == frozen_plan
    assert set(brain._candidate_priors) == {candidate_id}
    assert set(brain._candidate_theses) == {candidate_id}

    released_asof = terminal_asof + pd.Timedelta(minutes=1)
    released_observation = replace_market_observation(
        terminal_observation,
        asof=released_asof,
        frames={
            timeframe: replace(frame, cutoff=released_asof)
            for timeframe, frame in terminal_observation.frames.items()
        },
        execution=_execution(released_asof, cost=0.10),
        displacement=replace(
            terminal_observation.displacement,
            asof=released_asof,
        ),
    )
    released_belief = brain.update(
        released_observation,
        scene_graph=graph,
        scene_delta=graph.update(released_observation),
    )
    assert released_belief.position_management_candidates == {}
    assert released_belief.resolve_hypothesis(candidate_id) is None
    assert candidate_id not in brain._candidate_priors
    assert candidate_id not in brain._candidate_theses


def test_dfp_held_reacceptance_is_an_alternative_entry_trigger() -> None:
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
        identity="dfp-held-fvg",
        lower=99.0,
        upper=100.0,
    )
    draw = _inventory_draw(
        identity="swing:dfp-held-draw",
        timeframe=Timeframe.H4,
        side="above",
        price=103.0,
        confirmed_at=BASE - pd.Timedelta(hours=3),
        source_id="dfp-held-draw",
    )
    reducer.on_completed_1m(
        formation,
        fair_value_gaps=(fvg,),
        liquidity_inventory=(draw,),
        m1_atr=1.0,
    )
    bars = (
        _m1(1, open_=100.5, high=100.75, low=99.5, close=99.75),
        _m1(2, open_=99.75, high=100.5, low=99.5, close=100.25),
        _m1(3, open_=100.25, high=100.75, low=100.0, close=100.5),
    )
    output = None
    for bar in bars:
        output = reducer.on_completed_1m(
            bar,
            fair_value_gaps=(fvg,),
            liquidity_inventory=(draw,),
            m1_atr=1.0,
        )
    assert output is not None
    assert output.path_sequences[0].steps[-1].kind == "reacceptance_held"

    hypothesis = _brain().update(
        _dfp_observation(
            asof=bars[-1].end,
            price=bars[-1].close,
            output=output,
            fvg=fvg,
            draw=draw,
        )
    ).hypotheses["displacement_first_pullback:long"]

    assert hypothesis.phase is PlaybookPhase.EXECUTABLE
    assert hypothesis.hard_gate_results["typed_entry_trigger"]


def test_dfp_aligned_micro_bos_is_an_alternative_entry_trigger() -> None:
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
        identity="dfp-micro-fvg",
        lower=99.0,
        upper=100.0,
    )
    draw = _inventory_draw(
        identity="swing:dfp-micro-draw",
        timeframe=Timeframe.H4,
        side="above",
        price=103.0,
        confirmed_at=BASE - pd.Timedelta(hours=3),
        source_id="dfp-micro-draw",
    )
    formation_output = reducer.on_completed_1m(
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
    pullback_output = reducer.on_completed_1m(
        pullback,
        fair_value_gaps=(fvg,),
        liquidity_inventory=(draw,),
        m1_atr=1.0,
    )
    trigger_bar = _m1(
        2,
        open_=100.5,
        high=101.25,
        low=100.25,
        close=101.0,
    )
    bos = replace(
        _local_bos(trigger_bar.end),
        bos_id="dfp-micro-bos",
        direction=Direction.LONG,
        target_swing_id="dfp-micro-swing",
    )
    output = reducer.on_completed_1m(
        trigger_bar,
        fair_value_gaps=(fvg,),
        liquidity_inventory=(draw,),
        m1_bos=(bos,),
        m1_atr=1.0,
    )
    observation = _dfp_observation(
        asof=trigger_bar.end,
        price=trigger_bar.close,
        output=output,
        fvg=fvg,
        draw=draw,
    )
    frames = dict(observation.frames)
    frames[Timeframe.M1] = replace(
        frames[Timeframe.M1],
        structure_breaks=(bos,),
    )
    observation = replace(observation, frames=frames)

    started_at = formation.end - pd.Timedelta(minutes=10)
    active_at = formation.end - pd.Timedelta(minutes=5)

    def active_displacement(asof: pd.Timestamp) -> DisplacementObservation:
        return DisplacementObservation(
            asof=asof,
            lifecycle="active",
            current_entity_id=fvg.source_displacement_id,
            current_direction=Direction.LONG,
            current_state_observed_at=asof,
            current_started_at=started_at,
            current_active_at=active_at,
            current_last_admitted_at=asof,
            current_metrics=(("net_points", 2.0),),
            latest_transition=None,
        )

    graph = TemporalMarketSceneGraph()
    brain = _brain()
    for asof, price, state in (
        (formation.end, formation.close, formation_output),
        (pullback.end, pullback.close, pullback_output),
    ):
        prior_observation = _dfp_observation(
            asof=asof,
            price=price,
            output=state,
            fvg=fvg,
            draw=draw,
        )
        prior_observation = replace(
            prior_observation,
            displacement=active_displacement(asof),
        )
        brain.update(
            prior_observation,
            scene_graph=graph,
            scene_delta=graph.update(prior_observation),
        )

    terminal = DisplacementTransitionObservation(
        transition_id="transition:dfp-micro-exhausted",
        entity_id=fvg.source_displacement_id,
        lifecycle="exhausted",
        reason="lost_forward_progress",
        observed_at=trigger_bar.end,
        direction=Direction.LONG,
        started_at=started_at,
        active_at=active_at,
        state_started_at=trigger_bar.end,
        terminal_at=trigger_bar.end,
        prefix_last_admitted_at=pullback.end,
        terminal_evidence_candle_id="dfp-micro-terminal-candle",
        admitted_candle_ids=("dfp-micro-seed",),
        state_metrics=(("net_points", 2.0),),
    )
    observation = replace(
        observation,
        displacement=DisplacementObservation(
            asof=trigger_bar.end,
            lifecycle="idle",
            current_entity_id=None,
            current_direction=None,
            current_state_observed_at=None,
            current_started_at=None,
            current_active_at=None,
            current_last_admitted_at=None,
            current_metrics=None,
            latest_transition=terminal,
            recent_transitions=(terminal,),
            transitions_this_update=(terminal,),
        ),
    )
    belief = brain.update(
        observation,
        scene_graph=graph,
        scene_delta=graph.update(observation),
    )
    hypothesis = belief.hypotheses[
        "displacement_first_pullback:long"
    ]

    assert output.path_sequences[0].steps[-1].kind == "micro_bos_confirmed"
    assert output.path_sequences[0].lifecycle is PathSequenceLifecycle.CLOSED
    assert hypothesis.phase is PlaybookPhase.EXECUTABLE
    assert hypothesis.market_thesis_action_bound
    thesis = next(
        thesis
        for thesis in belief.market_theses
        if thesis.root_id == fvg.source_displacement_id
    )
    assert output.micro_bos_references[0].reference_id in (
        thesis.trigger_event_ids
    )

    next_asof = trigger_bar.end + pd.Timedelta(minutes=1)
    next_observation = replace_market_observation(
        observation,
        asof=next_asof,
        price=trigger_bar.close,
        frames={
            timeframe: replace(frame, cutoff=next_asof)
            for timeframe, frame in observation.frames.items()
        },
        execution=_execution(next_asof, cost=0.10),
        displacement=replace(
            observation.displacement,
            asof=next_asof,
            transitions_this_update=(),
        ),
    )
    next_belief = brain.update(
        next_observation,
        scene_graph=graph,
        scene_delta=graph.update(next_observation),
    )

    assert all(
        item.root_id != fvg.source_displacement_id
        for item in next_belief.market_theses
    )
    assert next_belief.hypotheses[
        "displacement_first_pullback:long"
    ].phase is not PlaybookPhase.EXECUTABLE
    assert hypothesis.entry_readiness == pytest.approx(bos.strength)


def test_dfp_first_pullback_from_another_zone_fails_closed() -> None:
    _, _, _, _, triggered = _dfp_fixture()
    path = triggered.path_sequences[0]
    steps = tuple(
        replace(step, source_entity_id="other-entry-zone")
        if step.kind == "first_pullback"
        else step
        for step in path.steps
    )
    malformed = replace(
        triggered,
        path_sequences=(replace(path, steps=steps),),
    )

    hypothesis = _brain().update(malformed).hypotheses[
        "displacement_first_pullback:long"
    ]

    assert hypothesis.phase is not PlaybookPhase.EXECUTABLE
    assert not hypothesis.hard_gate_results[
        "first_pullback_to_frozen_zone"
    ]
    assert not hypothesis.hard_gate_results["typed_entry_trigger"]


def test_same_episode_sequence_history_does_not_regress_with_trimmed_snapshot() -> None:
    _, _, _, forming, triggered = _dfp_fixture()
    brain = _brain()
    key = "displacement_first_pullback:long"
    brain.update(forming)
    complete = brain.update(triggered).hypotheses[key]
    assert complete.sequence is not None
    assert complete.sequence.complete

    path = triggered.path_sequences[0]
    trimmed_path = replace(
        path,
        steps=path.steps[:2],
        transition_reason="context_registered",
    )
    later = triggered.asof + pd.Timedelta(minutes=1)
    trimmed = replace(
        _advance_observation(triggered, later),
        path_sequences=(trimmed_path,),
        qualified_reacceptances=(),
        micro_bos_references=(),
    )

    belief = brain.update(trimmed)
    updated = belief.hypotheses[key]
    assert updated.sequence == complete.sequence
    assert updated.sequence.complete
    assert not updated.hard_gate_results[
        "first_pullback_to_frozen_zone"
    ]
    assert not updated.hard_gate_results["typed_entry_trigger"]
    assert (
        UtilityDecisionLayer().decide(trimmed, belief).selected_action.value
        != "enter"
    )


def test_missing_draw_keeps_episode_history_but_fails_current_gates() -> None:
    _, _, _, forming, triggered = _dfp_fixture()
    brain = _brain()
    key = "displacement_first_pullback:long"
    brain.update(forming)
    complete = brain.update(triggered).hypotheses[key]
    assert complete.sequence is not None
    assert complete.sequence.complete

    later = triggered.asof + pd.Timedelta(minutes=1)
    missing_draw = replace(
        _advance_observation(triggered, later),
        liquidity_inventory=(),
    )
    dormant = brain.update(missing_draw).hypotheses[key]

    assert dormant.phase is PlaybookPhase.WEAKENING
    assert dormant.sequence == complete.sequence
    assert dormant.sequence.complete
    assert not dormant.hard_gate_results[
        "h4_structure_and_draw"
    ]
    assert dormant.terminal_reason is None
    assert dormant.terminal_at is None
    assert UtilityDecisionLayer().decide(
        missing_draw,
        graph_free_action_belief(brain.current),
    ).selected_action.value != "enter"




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
        == "previous-day:dfp-primary-draw"
    )
    assert initial.thesis_draw is not None
    assert initial.thesis_draw.level_id == draw.item_id

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

    assert updated.phase is PlaybookPhase.WAITING_LOCATION
    assert updated.deliverable_targets == ()
    assert updated.plan is None
    assert not updated.hard_gate_results["h4_structure_and_draw"]
    assert updated.setup_context_id == initial.setup_context_id
    assert updated.context_id == initial.context_id
    assert updated.thesis_draw == initial.thesis_draw
    assert updated.terminal_at is None
    assert updated.terminal_reason is None
    assert replacement_draw.item_id not in {
        target.level_id for target in updated.deliverable_targets
    }


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
    assert initial.thesis_strength == pytest.approx(0.7)
    assert initial.evidence_group_scores["structure"] == 1.0

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
    assert revised.thesis_strength == pytest.approx(0.75)
    assert (
        revised.raw_quality_dimensions["thesis_strength"]
        == pytest.approx(0.75)
    )
    assert revised.evidence_revision_id != initial.evidence_revision_id
    # Missing execution authority is not market uncertainty, and future DFP
    # steps are sequence progress rather than missing current evidence.
    assert revised.uncertainty == 0.0
    assert revised.context_metadata["uncertainty_total"] == "0"
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
        0.6
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
        0.75
    )

    repeated = brain.update(
        _advance_observation(
            revised_observation,
            revised_at + pd.Timedelta(minutes=1),
        )
    ).hypotheses[key]
    assert repeated.evidence_revision_id == revised.evidence_revision_id
    assert repeated.raw_quality_dimensions["thesis_strength"] == pytest.approx(
        0.75
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
    assert raw_thesis == pytest.approx(0.75)
    assert calibrated.thesis_strength == pytest.approx(
        0.1 + raw_thesis * 0.8
    )
    assert calibrated.raw_probability == raw_thesis
    assert calibrated.probability == calibrated.thesis_strength
    assert calibrated.calibration_version == "typed-test-ready"
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
    same_identity_fvg = _fvg(
        rearm_bar.end,
        direction=Direction.LONG,
        identity="dfp-fvg",
        lower=99.0,
        upper=100.0,
    )
    same_identity_output = CausalGroup5Reducer(
        _group5_protocol()
    ).on_completed_1m(
        rearm_bar,
        fair_value_gaps=(same_identity_fvg,),
        liquidity_inventory=(draw,),
        m1_atr=1.0,
    )
    same_identity_observation = _dfp_observation(
        asof=rearm_bar.end,
        price=rearm_bar.close,
        output=same_identity_output,
        fvg=same_identity_fvg,
        draw=draw,
    )
    same_identity = brain.update(
        same_identity_observation
    ).hypotheses[key]
    assert same_identity is terminal

    rearm_bar = _m1(
        11,
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
    later_observation = replace_market_observation(
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
    new_intermediate = replace(
        triggered.liquidity_inventory[0],
        item_id="previous-session:new-nearer-draw",
        kind="previous_session_high",
        price=101.5,
        lower_bound=101.5,
        upper_bound=101.5,
        source_ids=("new-nearer-draw-source",),
        confirmed_at=later,
        formed_at=later,
    )
    missing_sources = replace(
        _advance_observation(triggered, later, price=100.75),
        liquidity_inventory=(
            new_intermediate,
            *triggered.liquidity_inventory,
        ),
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
    assert updated.plan.selected_draw_id != new_intermediate.item_id
    decision_layer = UtilityDecisionLayer()
    decision = decision_layer.decide(
        missing_sources,
        graph_free_action_belief(belief),
        AccountState(equity=100_000.0, position=position),
    )
    hold = next(
        item for item in decision.utilities if item.action.value == "hold"
    )
    # Post-entry management is structural until a separately calibrated
    # continuation probability exists.  A weakening snapshot may not reuse
    # the entry-time delivery/uncertainty scores as a synthetic hold
    # probability, while the exact frozen thesis still remains identifiable.
    assert hold.components["frozen_thesis_valid"] == 1.0
    assert hold.components["hard_exit_absent"] == 1.0
    assert "uncertainty" not in hold.components
    assert "probability" not in hold.components


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
    market_episode_deadline = first.episode_deadline
    market_thesis_deadline = first.thesis_deadline

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
    assert tightened.episode_deadline == market_episode_deadline
    assert tightened.thesis_deadline == market_thesis_deadline

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
    assert relaxed.episode_deadline == market_episode_deadline
    assert relaxed.thesis_deadline == market_thesis_deadline


def test_context_and_entry_episode_use_separate_deadline_clocks() -> None:
    _, _, _, _, triggered = _dfp_fixture()
    observed_execution = replace(
        triggered.execution,
        source="synthetic_observed_execution",
        minutes_to_deadline=50,
        anomalies=(),
    )
    missing_source_execution = replace(
        observed_execution,
        source="missing",
    )
    short_execution = replace(
        observed_execution,
        minutes_to_deadline=5,
    )

    observed = _brain().update(
        replace(triggered, execution=observed_execution)
    ).hypotheses["displacement_first_pullback:long"]
    missing_source = _brain().update(
        replace(triggered, execution=missing_source_execution)
    ).hypotheses["displacement_first_pullback:long"]
    short = _brain().update(
        replace(triggered, execution=short_execution)
    ).hypotheses["displacement_first_pullback:long"]

    assert (
        observed.thesis_deadline
        == missing_source.thesis_deadline
        == short.thesis_deadline
        is None
    )
    assert observed.plan is not None
    assert missing_source.plan is not None
    assert short.plan is not None
    assert observed.plan.deadline == triggered.asof + pd.Timedelta(minutes=50)
    assert missing_source.plan.deadline == observed.plan.deadline
    assert short.plan.deadline == triggered.asof + pd.Timedelta(minutes=5)
    assert observed.episode_deadline == observed.plan.deadline
    assert missing_source.episode_deadline == missing_source.plan.deadline
    assert short.episode_deadline == short.plan.deadline


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
        source_timeframe=Timeframe.H1,
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
        reentry_candidate_at=None,
        reentry_candidate_price=None,
        inside_hold_bars=0,
        reentry_failed_at=None,
        outside_run=0,
        outside_run_side=None,
        deadline_at=None,
        deadline_elapsed=False,
        crossed_source_ids=("pool-above",),
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
        break_bar_id="lsr-trigger-break-bar",
        break_distance_atr=0.5,
        post_break_state=BOSPostBreakState.PENDING,
    )


def _opposed_m5_bos(
    clock: pd.Timestamp,
    *,
    displacement_id: str,
    break_bar_id: str,
    direction: Direction = Direction.SHORT,
) -> BreakOfStructureState:
    return BreakOfStructureState(
        bos_id=f"opposed-return-mss:{direction.value}",
        timeframe=Timeframe.M5,
        direction=direction,
        lifecycle=BOSLifecycle.CONFIRMED,
        scope=BOSScope.OPPOSED,
        target_swing_id=f"opposed-protected:{direction.value}",
        source_structure_id=(
            "prior-structure:short"
            if direction is Direction.LONG
            else "prior-structure:long"
        ),
        target_price=100.5,
        target_ticks=402,
        pending_at=clock - pd.Timedelta(minutes=15),
        resolved_at=clock,
        age_bars=1,
        strength=0.8,
        break_bar_id=break_bar_id,
        break_distance_atr=0.5,
        source_displacement_id=displacement_id,
        mss_qualified=True,
        post_break_state=BOSPostBreakState.PENDING,
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


def _lsr_pool_source(
    swept_at: pd.Timestamp,
    resolved_at: pd.Timestamp,
) -> tuple[LiquidityPoolState, LiquidityInventoryItem]:
    touch_one = swept_at - pd.Timedelta(minutes=5)
    touch_two = swept_at - pd.Timedelta(minutes=1)
    pool = LiquidityPoolState(
        pool_id="pool-above",
        timeframe=Timeframe.H1,
        side="above",
        lower_bound=100.75,
        upper_bound=101.0,
        midpoint=100.875,
        formed_at=touch_one,
        confirmed_at=touch_two,
        lifecycle=LiquidityPoolLifecycle.REJECTED,
        member_swing_ids=("pool-high-1", "pool-high-2"),
        touch_times=(touch_one, touch_two),
        age_bars=6,
        strength=0.8,
        swept_at=swept_at,
        sweep_extreme=101.25,
        close_outside_on_sweep=False,
        resolved_at=resolved_at,
        resolution_reason="close_returned_inside",
    )
    inventory = LiquidityInventoryItem(
        item_id="pool:pool-above",
        timeframe=Timeframe.H1,
        side="above",
        kind="equal_highs",
        price=101.0,
        lower_bound=100.75,
        upper_bound=101.0,
        formed_at=touch_one,
        confirmed_at=touch_two,
        lifecycle=LiquidityInventoryLifecycle.CONSUMED,
        source_ids=pool.member_swing_ids,
        age_bars=6,
        strength=0.8,
        consumed_at=swept_at,
        lifecycle_reason="pool_swept",
    )
    return pool, inventory


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
    candidate = _m1(
        1,
        open_=100.5,
        high=100.75,
        low=100.25,
        close=100.5,
    )
    candidate_manipulation = replace(
        manipulation,
        last_updated_at=candidate.end,
        age_1m_bars=1,
        reentry_candidate_at=candidate.end,
        reentry_candidate_price=candidate.close,
    )
    reducer.on_completed_1m(
        candidate,
        manipulations=(candidate_manipulation,),
        liquidity_inventory=(target,),
        m1_atr=1.0,
    )
    reaccept = _m1(
        2,
        open_=100.5,
        high=100.75,
        low=100.25,
        close=100.5,
    )
    reaccepted_manipulation = replace(
        candidate_manipulation,
        lifecycle=ManipulationLifecycle.REACCEPTED,
        reaccepted_at=reaccept.end,
        resolved_at=reaccept.end,
        state_started_at=reaccept.end,
        last_updated_at=reaccept.end,
        reentry_price=candidate.close,
        inside_hold_bars=1,
        age_1m_bars=2,
        transition_reason="reentry_held_inside_swept_boundary",
    )
    reducer.on_completed_1m(
        reaccept,
        manipulations=(reaccepted_manipulation,),
        liquidity_inventory=(target,),
        m1_atr=1.0,
    )
    formation = _m1(
        3,
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
        manipulations=(reaccepted_manipulation,),
        liquidity_inventory=(target,),
        m1_atr=1.0,
    )
    pullback = _m1(
        4,
        open_=100.75,
        high=101.5,
        low=100.5,
        close=101.0,
    )
    reducer.on_completed_1m(
        pullback,
        fair_value_gaps=(fvg,),
        manipulations=(reaccepted_manipulation,),
        liquidity_inventory=(target,),
        m1_atr=1.0,
    )
    trigger = _m1(
        5,
        open_=101.0,
        high=101.25,
        low=100.5,
        close=101.0,
    )
    trigger_bos = _local_bos(trigger.end)
    output = reducer.on_completed_1m(
        trigger,
        fair_value_gaps=(fvg,),
        manipulations=(reaccepted_manipulation,),
        liquidity_inventory=(target,),
        m1_bos=(trigger_bos,),
        m1_atr=1.0,
    )
    base = market_observation(
        asof=trigger.end,
        price=trigger.close,
    )
    frames = dict(base.frames)
    pool, pool_inventory = _lsr_pool_source(
        sweep_bar.end,
        candidate.end,
    )
    frames[Timeframe.H4] = replace(
        frames[Timeframe.H4],
        cutoff=trigger.end,
    )
    frames[Timeframe.H1] = replace(
        frames[Timeframe.H1],
        cutoff=trigger.end,
        swings=(swing,),
        liquidity_pools=(pool,),
    )
    frames[Timeframe.M5] = replace(
        frames[Timeframe.M5],
        cutoff=trigger.end,
        fair_value_gaps=(fvg,),
        structure_breaks=(
            _opposed_m5_bos(
                formation.end,
                displacement_id=fvg.source_displacement_id,
                break_bar_id=fvg.source_candle_ids[-1],
            ),
        ),
    )
    frames[Timeframe.M1] = replace(
        frames[Timeframe.M1],
        cutoff=trigger.end,
        structure_breaks=(trigger_bos,),
    )
    return brain_observation_view(replace(
        base,
        frames=frames,
        execution=_execution(trigger.end, cost=0.025),
        liquidity_inventory=(target, pool_inventory),
        liquidity_pool_states=(pool,),
        manipulations=(reaccepted_manipulation,),
        interaction_update=_interaction_update_from_brain_fields(
            entry_locations=output.entry_locations,
            qualified_reacceptances=output.qualified_reacceptances,
            micro_bos_references=output.micro_bos_references,
            path_sequences=output.path_sequences,
        ),
    ))


def test_lsr_context_identity_requires_manipulation_and_displacement() -> None:
    observation = _lsr_observation()
    brain = _brain()
    protocol = brain.registry.for_playbook(
        Playbook.LIQUIDITY_SWEEP_REVERSAL
    )
    evaluation = _typed_lsr(
        observation,
        Direction.SHORT,
        protocol,
        None,
        BrainConfig(),
    )
    assert "formed_pool_sweep" in evaluation.sequence_signals
    assert evaluation.initiating_event_id is not None
    assert evaluation.lsr_displacement_id is not None
    assert _stable_context_thesis_id(
        "epoch:lsr-context-identity",
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        evaluation,
        None,
    ) is not None
    assert (
        _stable_context_thesis_id(
            "epoch:lsr-context-identity",
            Playbook.LIQUIDITY_SWEEP_REVERSAL,
            Direction.SHORT,
            replace(evaluation, sequence_signals={}),
            None,
        )
        is None
    )
    assert (
        _stable_context_thesis_id(
            "epoch:lsr-context-identity",
            Playbook.LIQUIDITY_SWEEP_REVERSAL,
            Direction.SHORT,
            replace(evaluation, initiating_event_id=None),
            None,
        )
        is None
    )
    assert (
        _stable_context_thesis_id(
            "epoch:lsr-context-identity",
            Playbook.LIQUIDITY_SWEEP_REVERSAL,
            Direction.SHORT,
            replace(evaluation, lsr_displacement_id=None),
            None,
        )
        is None
    )


def test_public_lsr_root_without_displacement_cannot_create_parent_or_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _lsr_observation()
    original = _typed_lsr

    def without_displacement(*args, **kwargs):
        evaluation = original(*args, **kwargs)
        if (
            args[1] is Direction.SHORT
            and "formed_pool_sweep" in evaluation.sequence_signals
        ):
            return replace(evaluation, lsr_displacement_id=None)
        return evaluation

    monkeypatch.setattr(
        "brain.core.playbooks._typed_lsr",
        without_displacement,
    )
    graph = TemporalMarketSceneGraph()
    belief = _brain().update(
        observation,
        scene_graph=graph,
        scene_delta=graph.update(observation),
    )
    candidates = tuple(
        candidate
        for candidate in belief.thesis_candidates.values()
        if candidate.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
        and candidate.direction is Direction.SHORT
    )
    assert candidates
    assert all(
        candidate.context_thesis_id is None
        and candidate.episode_id is None
        and candidate.episode_deadline is None
        for candidate in candidates
    )
    assert all(
        context.direction is not Direction.SHORT
        for context in belief.context_theses.values()
    )


def _lsr_without_entry_trigger(
    observation: MarketObservation,
) -> MarketObservation:
    """Keep the LSR Context and first pullback, but no zone-local trigger."""

    paths = []
    for path in observation.path_sequences:
        if path.context_kind != "zone_return":
            paths.append(path)
            continue
        retained = tuple(
            step
            for step in path.steps
            if step.kind not in {
                "wick_rejection",
                "reacceptance_held",
                "micro_bos_confirmed",
            }
        )
        paths.append(
            replace(
                path,
                lifecycle=PathSequenceLifecycle.ACTIVE,
                state_started_at=retained[-1].observed_at,
                last_updated_at=observation.asof,
                state_duration_real_1m_bars=max(
                    0,
                    int(
                        (
                            observation.asof
                            - retained[-1].observed_at
                        ).total_seconds()
                        // 60
                    ),
                ),
                steps=retained,
                ended_at=None,
                transition_reason="context_registered",
            )
        )
    frames = dict(observation.frames)
    frames[Timeframe.M1] = replace(
        frames[Timeframe.M1],
        structure_breaks=(),
    )
    return replace(
        observation,
        frames=frames,
        micro_bos_references=tuple(
            reference
            for reference in observation.micro_bos_references
            if reference.context_kind != "zone_return"
        ),
        path_sequences=tuple(paths),
    )


def _lsr_waiting_for_first_pullback(
    observation: MarketObservation,
) -> MarketObservation:
    """Keep an eligible LSR zone provisional before its first return."""

    waiting = _lsr_without_entry_trigger(observation)
    location = waiting.entry_locations[0]
    waiting_location = replace(
        location,
        lifecycle=EntryLocationLifecycle.APPROACHING,
        state_started_at=location.departure_confirmed_at,
        last_updated_at=waiting.asof,
        state_duration_real_1m_bars=max(
            0,
            int(
                (
                    waiting.asof - location.departure_confirmed_at
                ).total_seconds()
                // 60
            ),
        ),
        current_price=waiting.price,
        distance_to_zone_points=max(
            0.0,
            location.near_edge - waiting.price,
        ),
        distance_to_failure_points=abs(
            location.failure_boundary - waiting.price
        ),
        first_entered_at=None,
        entry_mode=None,
        contact_reference_price=None,
        first_penetration_fraction=0.0,
        rejected_at=None,
        left_at=None,
        reaction_atr=0.0,
        transition_reason="departure_confirmed",
    )
    paths = []
    for path in waiting.path_sequences:
        if path.context_id != location.location_id:
            paths.append(path)
            continue
        retained = tuple(
            step for step in path.steps if step.kind != "first_pullback"
        )
        paths.append(
            replace(
                path,
                lifecycle=PathSequenceLifecycle.ACTIVE,
                state_started_at=retained[-1].observed_at,
                last_updated_at=waiting.asof,
                state_duration_real_1m_bars=max(
                    0,
                    int(
                        (
                            waiting.asof - retained[-1].observed_at
                        ).total_seconds()
                        // 60
                    ),
                ),
                steps=retained,
                ended_at=None,
                transition_reason="context_registered",
            )
        )
    return replace(
        waiting,
        entry_locations=(
            waiting_location,
            *waiting.entry_locations[1:],
        ),
        path_sequences=tuple(paths),
    )


def _lsr_register_first_pullback(
    observation: MarketObservation,
    pullback_at: pd.Timestamp,
) -> MarketObservation:
    location = observation.entry_locations[0]
    entered = replace(
        location,
        lifecycle=EntryLocationLifecycle.IN_ZONE,
        state_started_at=pullback_at,
        last_updated_at=pullback_at,
        age_real_1m_bars=max(
            0,
            int((pullback_at - location.formed_at).total_seconds() // 60),
        ),
        state_duration_real_1m_bars=0,
        current_price=location.near_edge,
        distance_to_zone_points=0.0,
        distance_to_failure_points=abs(
            location.failure_boundary - location.near_edge
        ),
        first_entered_at=pullback_at,
        entry_mode="crossed_near_edge",
        contact_reference_price=location.near_edge,
        first_penetration_fraction=0.0,
        transition_reason="first_pullback",
    )
    paths = []
    for path in observation.path_sequences:
        if path.context_id != location.location_id:
            paths.append(path)
            continue
        pullback = PathSequenceStep(
            step_id="lsr-provisional-target-first-pullback",
            kind="first_pullback",
            observed_at=pullback_at,
            source_event_id=None,
            source_entity_id=location.location_id,
            predecessor_step_ids=(path.steps[-1].step_id,),
            same_clock_relation="strictly_after",
            direction=Direction.SHORT,
            strength=0.5,
            reason="crossed_near_edge",
        )
        paths.append(
            replace(
                path,
                lifecycle=PathSequenceLifecycle.ACTIVE,
                state_started_at=pullback_at,
                last_updated_at=pullback_at,
                state_duration_real_1m_bars=0,
                steps=(*path.steps, pullback),
                ended_at=None,
                transition_reason="context_registered",
            )
        )
    return replace_market_observation(
        _advance_observation(observation, pullback_at),
        price=location.near_edge,
        entry_locations=(entered, *observation.entry_locations[1:]),
        path_sequences=tuple(paths),
    )


def _lsr_add_zone_episode(
    observation: MarketObservation,
    *,
    suffix: str,
    formed_at: pd.Timestamp,
    pullback_at: pd.Timestamp | None,
    trigger_at: pd.Timestamp | None,
) -> MarketObservation:
    """Add one independent FVG Episode under the frozen LSR displacement."""

    source_fvg = observation.frames[Timeframe.M5].fair_value_gaps[0]
    source_location = observation.entry_locations[0]
    zone_id = f"lsr-short-fvg-{suffix}"
    location_id = f"lsr-location-{suffix}"
    fvg = replace(
        _fvg(
            formed_at,
            direction=Direction.SHORT,
            identity=zone_id,
            lower=100.75,
            upper=101.75,
        ),
        source_displacement_id=source_fvg.source_displacement_id,
        source_active_transition_id=(
            source_fvg.source_active_transition_id
        ),
        source_displacement_protocol_hash=(
            source_fvg.source_displacement_protocol_hash
        ),
        source_displacement_started_at=(
            source_fvg.source_displacement_started_at
        ),
        source_displacement_active_at=(
            source_fvg.source_displacement_active_at
        ),
        source_displacement_prefix_commitment=(
            source_fvg.source_displacement_prefix_commitment
        ),
    )
    if pullback_at is None:
        location = replace(
            source_location,
            location_id=location_id,
            source_zone_id=zone_id,
            source_displacement_id=fvg.source_displacement_id,
            lower_bound=fvg.lower_bound,
            upper_bound=fvg.upper_bound,
            midpoint=fvg.midpoint,
            near_edge=fvg.lower_bound,
            far_edge=fvg.upper_bound,
            failure_boundary=fvg.upper_bound,
            formed_at=formed_at,
            lifecycle=EntryLocationLifecycle.APPROACHING,
            state_started_at=formed_at,
            last_updated_at=observation.asof,
            age_real_1m_bars=max(
                0,
                int(
                    (observation.asof - formed_at).total_seconds() // 60
                ),
            ),
            state_duration_real_1m_bars=max(
                0,
                int(
                    (observation.asof - formed_at).total_seconds() // 60
                ),
            ),
            current_price=observation.price,
            distance_to_zone_points=0.25,
            distance_to_failure_points=abs(
                fvg.upper_bound - observation.price
            ),
            departure_confirmed_at=formed_at,
            first_entered_at=None,
            entry_mode=None,
            contact_reference_price=None,
            first_penetration_fraction=0.0,
            rejected_at=None,
            left_at=None,
            reaction_atr=0.0,
            transition_reason="departure_confirmed",
        )
    else:
        location = replace(
            source_location,
            location_id=location_id,
            source_zone_id=zone_id,
            source_displacement_id=fvg.source_displacement_id,
            lower_bound=fvg.lower_bound,
            upper_bound=fvg.upper_bound,
            midpoint=fvg.midpoint,
            near_edge=fvg.lower_bound,
            far_edge=fvg.upper_bound,
            failure_boundary=fvg.upper_bound,
            formed_at=formed_at,
            lifecycle=EntryLocationLifecycle.IN_ZONE,
            state_started_at=pullback_at,
            last_updated_at=observation.asof,
            age_real_1m_bars=max(
                0,
                int(
                    (observation.asof - formed_at).total_seconds() // 60
                ),
            ),
            state_duration_real_1m_bars=max(
                0,
                int(
                    (observation.asof - pullback_at).total_seconds() // 60
                ),
            ),
            current_price=observation.price,
            distance_to_zone_points=0.0,
            distance_to_failure_points=abs(
                fvg.upper_bound - observation.price
            ),
            departure_confirmed_at=formed_at,
            first_entered_at=pullback_at,
            entry_mode="crossed_near_edge",
            contact_reference_price=min(
                fvg.upper_bound,
                max(fvg.lower_bound, observation.price),
            ),
            first_penetration_fraction=0.5,
            rejected_at=None,
            left_at=None,
            reaction_atr=0.0,
            transition_reason="first_pullback",
        )
    if trigger_at is not None and pullback_at is not None:
        location = replace(
            location,
            lifecycle=EntryLocationLifecycle.REJECTED,
            state_started_at=trigger_at,
            last_updated_at=max(location.last_updated_at, trigger_at),
            state_duration_real_1m_bars=max(
                0,
                int(
                    (observation.asof - trigger_at).total_seconds() // 60
                ),
            ),
            rejected_at=trigger_at,
            reaction_atr=0.7,
            transition_reason="later_zone_rejection",
        )
    visible = PathSequenceStep(
        step_id=f"lsr-{suffix}-zone-visible",
        kind="zone_visible",
        observed_at=formed_at,
        source_event_id=zone_id,
        source_entity_id=zone_id,
        predecessor_step_ids=(),
        same_clock_relation="origin",
        direction=Direction.SHORT,
        strength=0.4,
        reason="typed_entry_zone_registered",
    )
    departure = PathSequenceStep(
        step_id=f"lsr-{suffix}-departure",
        kind="departure_confirmed",
        observed_at=formed_at,
        source_event_id=None,
        source_entity_id=location_id,
        predecessor_step_ids=(visible.step_id,),
        same_clock_relation="same_clock_known",
        direction=Direction.SHORT,
        strength=0.4,
        reason="formation_close_on_delivery_side",
    )
    steps = [visible, departure]
    if pullback_at is not None:
        steps.append(
            PathSequenceStep(
                step_id=f"lsr-{suffix}-first-pullback",
                kind="first_pullback",
                observed_at=pullback_at,
                source_event_id=None,
                source_entity_id=location_id,
                predecessor_step_ids=(steps[-1].step_id,),
                same_clock_relation="strictly_after",
                direction=Direction.SHORT,
                strength=0.5,
                reason="crossed_near_edge",
            )
        )
    if trigger_at is not None:
        steps.append(
            PathSequenceStep(
                step_id=f"lsr-{suffix}-wick-trigger",
                kind="wick_rejection",
                observed_at=trigger_at,
                source_event_id=None,
                source_entity_id=location_id,
                predecessor_step_ids=(steps[-1].step_id,),
                same_clock_relation=(
                    "same_clock_known"
                    if trigger_at == steps[-1].observed_at
                    else "strictly_after"
                ),
                direction=Direction.SHORT,
                strength=0.7,
                reason="later_zone_rejection",
            )
        )
    path = replace(
        next(
            item
            for item in observation.path_sequences
            if item.context_kind == "zone_return"
        ),
        sequence_id=f"lsr-zone-path-{suffix}",
        context_id=location_id,
        # A qualified wick rejection is an active milestone, not a terminal
        # path.  This mirrors Group5._mark_active_path_milestone and lets Risk
        # verify the real registered trigger contract.
        lifecycle=PathSequenceLifecycle.ACTIVE,
        formed_at=formed_at,
        state_started_at=steps[-1].observed_at,
        last_updated_at=observation.asof,
        age_real_1m_bars=max(
            0,
            int((observation.asof - formed_at).total_seconds() // 60),
        ),
        state_duration_real_1m_bars=max(
            0,
            int(
                (
                    observation.asof - steps[-1].observed_at
                ).total_seconds()
                // 60
            ),
        ),
        steps=tuple(steps),
        ended_at=None,
        transition_reason=(
            "zone_rejection_observed"
            if trigger_at is not None
            else "context_registered"
        ),
    )
    frames = dict(observation.frames)
    frames[Timeframe.M5] = replace(
        frames[Timeframe.M5],
        fair_value_gaps=(
            *frames[Timeframe.M5].fair_value_gaps,
            fvg,
        ),
    )
    return replace(
        observation,
        frames=frames,
        entry_locations=(*observation.entry_locations, location),
        path_sequences=(*observation.path_sequences, path),
    )


def _lsr_mark_first_zone_left(
    observation: MarketObservation,
    *,
    left_at: pd.Timestamp,
) -> MarketObservation:
    first_location = observation.entry_locations[0]
    locations = (
        replace(
            first_location,
            lifecycle=EntryLocationLifecycle.LEFT,
            state_started_at=left_at,
            last_updated_at=left_at,
            state_duration_real_1m_bars=0,
            current_price=first_location.upper_bound + 0.25,
            distance_to_zone_points=0.25,
            distance_to_failure_points=0.25,
            left_at=left_at,
            transition_reason="failure_boundary_breached",
        ),
        *observation.entry_locations[1:],
    )
    paths = []
    for path in observation.path_sequences:
        if path.context_id != first_location.location_id:
            paths.append(path)
            continue
        left_step = PathSequenceStep(
            step_id="lsr-first-zone-left",
            kind="location_left",
            observed_at=left_at,
            source_event_id=None,
            source_entity_id=first_location.location_id,
            predecessor_step_ids=(path.steps[-1].step_id,),
            same_clock_relation="strictly_after",
            direction=Direction.SHORT,
            strength=1.0,
            reason="close_beyond_far_edge",
        )
        paths.append(
            replace(
                path,
                lifecycle=PathSequenceLifecycle.CLOSED,
                state_started_at=left_at,
                last_updated_at=left_at,
                state_duration_real_1m_bars=0,
                steps=(*path.steps, left_step),
                ended_at=left_at,
                transition_reason="location_left",
            )
        )
    return replace(
        observation,
        entry_locations=locations,
        path_sequences=tuple(paths),
    )


def _short_lsr_candidates(belief) -> dict[str, object]:
    return {
        identity: candidate
        for identity, candidate in {
            **belief.thesis_candidates,
            **belief.retained_episode_candidates,
            **belief.position_management_candidates,
        }.items()
        if candidate.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
        and candidate.direction is Direction.SHORT
    }


def test_lsr_sibling_zones_have_independent_episode_identity_and_do_not_borrow() -> None:
    observation = _lsr_without_entry_trigger(_lsr_observation())
    observation = _lsr_add_zone_episode(
        observation,
        suffix="b-no-pullback",
        formed_at=observation.asof - pd.Timedelta(minutes=2),
        pullback_at=None,
        trigger_at=observation.asof,
    )
    graph = TemporalMarketSceneGraph()
    belief = _brain().update(
        observation,
        scene_graph=graph,
        scene_delta=graph.update(observation),
    )
    candidates = _short_lsr_candidates(belief)

    assert len(candidates) == 2
    assert len({item.context_thesis_id for item in candidates.values()}) == 1
    assert len({item.episode_id for item in candidates.values()}) == 2
    assert len({item.candidate_id for item in candidates.values()}) == 2
    assert len({item.entry_location_id for item in candidates.values()}) == 2
    assert len({item.entry_path_id for item in candidates.values()}) == 2
    assert len({item.evidence_revision_id for item in candidates.values()}) == 1
    assert all(
        item.context_metadata["lsr_manipulation_id"]
        == "manipulation:pool-above"
        and item.context_metadata["lsr_displacement_id"]
        == "displacement:lsr-short-fvg"
        for item in candidates.values()
    )
    no_pullback = next(
        item
        for item in candidates.values()
        if item.context_metadata["lsr_entry_zone_id"]
        == "lsr-short-fvg-b-no-pullback"
    )
    first_zone = next(
        item for item in candidates.values() if item is not no_pullback
    )
    assert first_zone.hard_gate_results["reversal_first_pullback"]
    assert not first_zone.hard_gate_results["typed_entry_trigger"]
    assert not no_pullback.hard_gate_results["reversal_first_pullback"]
    assert not no_pullback.hard_gate_results["typed_entry_trigger"]
    assert no_pullback.selected_trigger is None
    assert all(
        item.phase is not PlaybookPhase.EXECUTABLE
        for item in candidates.values()
    )
    context = next(iter(belief.context_theses.values()))
    assert context.context_draw is None


def test_lsr_closed_context_path_survives_local_zone_failure_and_late_zone_executes() -> None:
    waiting = _lsr_without_entry_trigger(_lsr_observation())
    graph = TemporalMarketSceneGraph()
    brain = _brain()
    initial = brain.update(
        waiting,
        scene_graph=graph,
        scene_delta=graph.update(waiting),
    )
    first = next(iter(_short_lsr_candidates(initial).values()))
    assert first.phase is PlaybookPhase.WAITING_TRIGGER
    pool_path = next(
        path
        for path in waiting.path_sequences
        if path.context_kind == "pool_reversal"
    )
    assert pool_path.lifecycle is PathSequenceLifecycle.CLOSED
    assert pool_path.transition_reason == "pool_reversal_sequence_observed"

    later_at = waiting.asof + pd.Timedelta(minutes=3)
    later = _advance_observation(waiting, later_at)
    later = _lsr_add_zone_episode(
        later,
        suffix="b-late",
        formed_at=waiting.asof + pd.Timedelta(minutes=1),
        pullback_at=waiting.asof + pd.Timedelta(minutes=2),
        trigger_at=later_at,
    )
    later = _lsr_mark_first_zone_left(
        later,
        left_at=waiting.asof + pd.Timedelta(minutes=1),
    )
    belief = brain.update(
        later,
        scene_graph=graph,
        scene_delta=graph.update(later),
    )
    candidates = _short_lsr_candidates(belief)
    late = next(
        item
        for item in candidates.values()
        if item.entry_location_id == "lsr-location-b-late"
    )
    failed = next(
        item
        for item in candidates.values()
        if item.entry_location_id == first.entry_location_id
    )

    assert failed.phase is PlaybookPhase.INVALIDATED
    assert failed.terminal_reason == "entry_zone_left_or_failed"
    assert late.phase is PlaybookPhase.EXECUTABLE
    assert late.plan is not None
    assert late.selected_trigger is not None
    assert late.context_thesis_id == first.context_thesis_id
    assert belief.context_theses[late.context_thesis_id].lifecycle == "active"
    assert first.candidate_id not in brain._candidate_priors
    assert brain._lsr_terminal_episode_ids[late.context_thesis_id] == {
        first.candidate_id
    }


def test_lsr_same_update_full_executable_clock_is_ambiguous_and_checkpointed() -> None:
    observation = _lsr_observation()
    first_location = observation.entry_locations[0]
    paths = []
    for path in observation.path_sequences:
        if path.context_id != first_location.location_id:
            paths.append(path)
            continue
        pullback = next(
            step for step in path.steps if step.kind == "first_pullback"
        )
        early_pullback_at = pullback.observed_at - pd.Timedelta(seconds=30)
        pullback = replace(
            pullback,
            observed_at=early_pullback_at,
        )
        wick = replace(
            path.steps[-1],
            step_id="lsr-a-early-wick",
            kind="wick_rejection",
            observed_at=early_pullback_at + pd.Timedelta(seconds=30),
            source_event_id=None,
            source_entity_id=first_location.location_id,
            predecessor_step_ids=(pullback.step_id,),
            same_clock_relation="strictly_after",
            reason="later_zone_rejection",
        )
        paths.append(
            replace(path, steps=(*path.steps[:-2], pullback, wick))
        )
    observation = replace(
        observation,
        path_sequences=tuple(paths),
        micro_bos_references=tuple(
            reference
            for reference in observation.micro_bos_references
            if reference.context_kind != "zone_return"
        ),
    )
    observation = _lsr_add_zone_episode(
        observation,
        suffix="b-same-update",
        formed_at=observation.asof - pd.Timedelta(minutes=2),
        pullback_at=observation.asof - pd.Timedelta(minutes=1),
        trigger_at=observation.asof,
    )
    graph = TemporalMarketSceneGraph()
    brain = _brain()
    belief = brain.update(
        observation,
        scene_graph=graph,
        scene_delta=graph.update(observation),
    )
    candidates = _short_lsr_candidates(belief)
    context_id = next(iter(candidates.values())).context_thesis_id

    assert len(candidates) == 2
    assert {
        item.selected_trigger.observed_at
        for item in candidates.values()
        if item.selected_trigger is not None
    } == {
        observation.asof - pd.Timedelta(minutes=1),
        observation.asof,
    }
    assert {item.phase_started_at for item in candidates.values()} == {
        observation.asof
    }
    assert brain._lsr_execution_winners[context_id] == (
        f"lsr-ambiguous-same-clock:{context_id}"
    )
    assert all(item.plan is None for item in candidates.values())
    assert all(
        item.context_metadata["lsr_context_execution_selection"]
        == "ambiguous_same_clock_fail_closed"
        for item in candidates.values()
    )
    assert belief.focus_state is not None
    assert belief.focus_state.hypothesis_id not in candidates

    restored = pickle.loads(pickle.dumps(brain))
    assert restored._lsr_execution_winners == brain._lsr_execution_winners
    assert restored._lsr_execution_owner_snapshots == (
        brain._lsr_execution_owner_snapshots
    )
    assert restored._lsr_context_seed_ids == brain._lsr_context_seed_ids
    assert restored._lsr_context_bindings == brain._lsr_context_bindings


def test_lsr_execution_owner_freezes_plan_trigger_summary_and_focus() -> None:
    initial_observation = _lsr_observation()
    graph = TemporalMarketSceneGraph()
    brain = _brain()
    initial_belief = brain.update(
        initial_observation,
        scene_graph=graph,
        scene_delta=graph.update(initial_observation),
    )
    initial = next(iter(_short_lsr_candidates(initial_belief).values()))
    assert initial.phase is PlaybookPhase.EXECUTABLE
    assert initial.plan is not None and initial.selected_trigger is not None

    later_at = initial_observation.asof + pd.Timedelta(minutes=3)
    later = _advance_observation(initial_observation, later_at)
    retained_paths = []
    for path in later.path_sequences:
        if path.context_kind != "zone_return":
            retained_paths.append(path)
            continue
        retained_trigger = replace(
            path.steps[-1],
            step_id="lsr-owner-retained-wick",
            kind="wick_rejection",
            source_event_id=None,
            source_entity_id=path.context_id,
            reason="later_zone_rejection",
        )
        retained_paths.append(
            replace(
                path,
                lifecycle=PathSequenceLifecycle.ACTIVE,
                state_started_at=retained_trigger.observed_at,
                last_updated_at=later_at,
                state_duration_real_1m_bars=3,
                steps=(*path.steps[:-1], retained_trigger),
                ended_at=None,
                transition_reason="zone_rejection_observed",
            )
        )
    retained_locations = tuple(
        replace(
            location,
            last_updated_at=later_at,
            state_duration_real_1m_bars=(
                location.state_duration_real_1m_bars + 3
            ),
        )
        for location in later.entry_locations
    )
    later = replace(
        later,
        path_sequences=tuple(retained_paths),
        entry_locations=retained_locations,
        micro_bos_references=tuple(
            reference
            for reference in later.micro_bos_references
            if reference.context_kind != "zone_return"
        ),
    )
    later = _lsr_add_zone_episode(
        later,
        suffix="b-stronger-late",
        formed_at=initial_observation.asof + pd.Timedelta(minutes=1),
        pullback_at=initial_observation.asof + pd.Timedelta(minutes=2),
        trigger_at=later_at,
    )
    belief = brain.update(
        later,
        scene_graph=graph,
        scene_delta=graph.update(later),
    )
    candidates = _short_lsr_candidates(belief)
    winner = candidates[initial.candidate_id]
    sibling = next(
        item
        for identity, item in candidates.items()
        if identity != initial.candidate_id
    )
    summary = belief.hypotheses["liquidity_sweep_reversal:short"]

    assert winner.phase is PlaybookPhase.EXECUTABLE
    assert winner.plan is not None and initial.plan is not None
    assert replace(
        winner.plan,
        remaining_path_R=initial.plan.remaining_path_R,
    ) == initial.plan
    assert winner.plan.remaining_path_R == pytest.approx(
        max(
            0.0,
            winner.direction.sign
            * (initial.plan.targets[0].price - later.price),
        )
        / initial.plan.risk_points
    )
    assert winner.selected_trigger == initial.selected_trigger
    assert winner.phase_started_at == initial.phase_started_at
    assert sibling.phase is PlaybookPhase.WEAKENING
    assert sibling.plan is None
    assert sibling.context_metadata["lsr_context_execution_owner"] == (
        initial.candidate_id
    )
    assert summary.summary_source_candidate_id == initial.candidate_id
    assert summary.plan == winner.plan
    assert summary.selected_trigger == initial.selected_trigger
    assert belief.focus_state is not None
    assert belief.focus_state.hypothesis_id == initial.candidate_id
    owner = brain._lsr_execution_owner_snapshots[initial.context_thesis_id]
    assert owner.candidate_id == initial.candidate_id
    assert owner.plan == initial.plan
    assert owner.trigger == initial.selected_trigger
    assert owner.first_executable_at == initial.phase_started_at
    restored = pickle.loads(pickle.dumps(brain))
    assert restored._lsr_execution_owner_snapshots == (
        brain._lsr_execution_owner_snapshots
    )
    with pytest.raises(ValueError, match="timezone aware"):
        _LSRExecutionOwner(
            candidate_id=owner.candidate_id,
            plan=owner.plan,
            trigger=owner.trigger,
            first_executable_at=pd.Timestamp("2025-01-07T09:36:00"),
        )


def test_lsr_execution_owner_freezes_route_on_opposed_micro_bos_terminal() -> None:
    observation = _lsr_observation()
    owner_location = observation.entry_locations[0]
    owner_paths = []
    for path in observation.path_sequences:
        if path.context_id != owner_location.location_id:
            owner_paths.append(path)
            continue
        retained = tuple(
            step for step in path.steps if step.kind != "micro_bos_confirmed"
        )
        wick = PathSequenceStep(
            step_id="lsr-owner-wick-before-opposed-terminal",
            kind="wick_rejection",
            observed_at=observation.asof,
            source_event_id=None,
            source_entity_id=owner_location.location_id,
            predecessor_step_ids=(retained[-1].step_id,),
            same_clock_relation="strictly_after",
            direction=path.direction,
            strength=0.5,
            reason="later_zone_rejection",
        )
        owner_paths.append(
            replace(
                path,
                lifecycle=PathSequenceLifecycle.ACTIVE,
                state_started_at=observation.asof,
                last_updated_at=observation.asof,
                state_duration_real_1m_bars=0,
                steps=(*retained, wick),
                ended_at=None,
                transition_reason="zone_rejection_observed",
            )
        )
    observation = replace(
        observation,
        entry_locations=(
            replace(
                owner_location,
                lifecycle=EntryLocationLifecycle.REJECTED,
                state_started_at=observation.asof,
                last_updated_at=observation.asof,
                state_duration_real_1m_bars=0,
                rejected_at=observation.asof,
                reaction_atr=0.5,
                transition_reason="qualified_zone_rejection",
            ),
        ),
        micro_bos_references=tuple(
            reference
            for reference in observation.micro_bos_references
            if reference.context_kind != "zone_return"
        ),
        path_sequences=tuple(owner_paths),
    )
    graph = TemporalMarketSceneGraph()
    brain = _brain()
    initial_belief = brain.update(
        observation,
        scene_graph=graph,
        scene_delta=graph.update(observation),
    )
    owner = next(iter(_short_lsr_candidates(initial_belief).values()))
    assert owner.phase is PlaybookPhase.EXECUTABLE
    assert owner.plan is not None and owner.selected_trigger is not None
    assert owner.plan.liquidity_route is not None
    brain = pickle.loads(pickle.dumps(brain))
    restored_owner = brain._lsr_execution_owner_snapshots[
        owner.context_thesis_id
    ]
    assert restored_owner.plan == owner.plan
    assert restored_owner.trigger == owner.selected_trigger

    terminal_at = observation.asof + pd.Timedelta(minutes=1)
    terminal_observation = _advance_observation(observation, terminal_at)
    original_target = next(
        item
        for item in terminal_observation.liquidity_inventory
        if item.item_id == owner.plan.selected_draw_id
    )
    later_intermediate = replace(
        original_target,
        item_id="previous_day_low:after-lsr-owner",
        kind="previous_day_low",
        price=99.5,
        lower_bound=99.5,
        upper_bound=99.5,
        formed_at=original_target.formed_at - pd.Timedelta(days=1),
        confirmed_at=original_target.confirmed_at - pd.Timedelta(minutes=1),
        source_ids=("previous_day_low:after-lsr-owner",),
        strength=0.9,
    )
    terminal_paths = []
    opposed_references = []
    for path in terminal_observation.path_sequences:
        if (
            path.context_kind != "zone_return"
            or path.context_id != owner.plan.entry_location_id
        ):
            terminal_paths.append(path)
            continue
        location = next(
            item
            for item in terminal_observation.entry_locations
            if item.location_id == path.context_id
        )
        opposed_references.append(
            MicroBOSReference(
                reference_id="lsr-opposed-reference-after-owner",
                protocol_hash=path.protocol_hash,
                context_kind=path.context_kind,
                context_id=path.context_id,
                expected_direction=path.direction,
                anchor_at=location.first_entered_at,
                bos_id="lsr-opposed-bos-after-owner",
                bos_direction=Direction.LONG,
                target_swing_id="lsr-opposed-swing-after-owner",
                scope=BOSScope.LOCAL,
                pending_at=terminal_at - pd.Timedelta(minutes=1),
                resolved_at=terminal_at,
                relation="strictly_after",
                outcome="opposed",
                qualified=False,
                strength=1.0,
            )
        )
        opposed = PathSequenceStep(
            step_id="lsr-opposed-after-execution-owner",
            kind="micro_bos_opposed",
            observed_at=terminal_at,
            source_event_id="lsr-opposed-bos-after-owner",
            source_entity_id="lsr-opposed-swing-after-owner",
            predecessor_step_ids=(path.steps[-1].step_id,),
            same_clock_relation="strictly_after",
            direction=path.direction,
            strength=1.0,
            reason="opposed_micro_bos_after_execution_owner",
        )
        terminal_paths.append(
            replace(
                path,
                lifecycle=PathSequenceLifecycle.CLOSED,
                state_started_at=terminal_at,
                last_updated_at=terminal_at,
                state_duration_real_1m_bars=0,
                steps=(*path.steps, opposed),
                ended_at=terminal_at,
                transition_reason="micro_bos_opposed",
            )
        )
    terminal_observation = replace(
        terminal_observation,
        liquidity_inventory=(
            *terminal_observation.liquidity_inventory,
            later_intermediate,
        ),
        micro_bos_references=(
            *terminal_observation.micro_bos_references,
            *opposed_references,
        ),
        path_sequences=tuple(terminal_paths),
    )
    position = PositionSnapshot(
        thesis_hash="lsr-owner-terminal-route-position",
        symbol=terminal_observation.symbol,
        instrument_id=terminal_observation.instrument_id,
        playbook=owner.playbook,
        direction=owner.direction,
        entry_price=owner.plan.planned_entry,
        original_invalidation=owner.plan.invalidation,
        current_stop=owner.plan.invalidation.price,
        primary_target=owner.plan.targets[0],
        opened_at=observation.asof,
        deadline=owner.plan.deadline,
        quantity=1,
        unrealized_R=0.0,
        elapsed_minutes=1,
        mfe_R=0.0,
        mae_R=0.0,
        setup_id=owner.plan.setup_id,
        entry_location_id=owner.plan.entry_location_id,
        entry_path_id=owner.plan.entry_path_id,
    )

    terminal_belief = brain.update(
        terminal_observation,
        scene_graph=graph,
        scene_delta=graph.update(terminal_observation),
    )
    terminal = _short_lsr_candidates(terminal_belief)[owner.candidate_id]

    assert terminal.phase is PlaybookPhase.INVALIDATED
    assert terminal.phase_started_at == terminal_at
    assert terminal.terminal_at == terminal_at
    assert terminal.terminal_reason == "micro_bos_opposed"
    assert terminal.plan is not None
    assert replace(
        terminal.plan,
        remaining_path_R=owner.plan.remaining_path_R,
    ) == owner.plan
    assert terminal.plan.liquidity_route == owner.plan.liquidity_route
    assert later_intermediate.item_id not in (
        terminal.plan.liquidity_route.intermediate_liquidity_ids
    )
    assert terminal.selected_trigger == owner.selected_trigger
    assert terminal.context_metadata["lsr_context_execution_owner"] == (
        owner.candidate_id
    )
    assert terminal.context_metadata["lsr_context_execution_selection"] == (
        "first_fully_executable_plan"
    )
    assert terminal.context_metadata["lsr_first_executable_at"] == (
        owner.phase_started_at.isoformat()
    )
    assert terminal.context_metadata["playbook_plan_delivery_valid"] == str(
        terminal.plan_feasibility is not None
        and terminal.plan_feasibility.valid
    ).lower()
    terminal_decision = _ready_decision_layer(brain).decide(
        terminal_observation,
        terminal_belief,
    )
    assert terminal_decision.selected_action is Action.ABSTAIN
    terminal_management = _ready_decision_layer(brain).decide(
        terminal_observation,
        terminal_belief,
        AccountState(equity=100_000.0, position=position),
    )
    assert terminal_management.selected_action is Action.EXIT


def test_lsr_terminal_trigger_kinds_preserve_owner_prefix_and_append() -> None:
    observation = _lsr_observation()
    graph = TemporalMarketSceneGraph()
    brain = _brain()
    belief = brain.update(
        observation,
        scene_graph=graph,
        scene_delta=graph.update(observation),
    )
    owner = next(iter(_short_lsr_candidates(belief).values()))
    assert owner.selected_trigger is not None
    assert owner.context_thesis_id is not None
    assert owner.candidate_id is not None

    snapshot = brain._lsr_execution_owner_snapshots[
        owner.context_thesis_id
    ]
    owner_kinds = tuple(
        dict.fromkeys(
            (
                "wick_rejection",
                *snapshot.trigger.available_trigger_kinds,
            )
        )
    )
    snapshot = replace(
        snapshot,
        trigger=replace(
            snapshot.trigger,
            available_trigger_kinds=owner_kinds,
        ),
    )
    terminal_at = observation.asof + pd.Timedelta(minutes=1)
    fresh_terminal_trigger = replace(
        owner.selected_trigger,
        available_trigger_kinds=(
            owner.selected_trigger.trigger_kind,
            "reacceptance_held",
        ),
    )
    terminal = replace(
        owner,
        phase=PlaybookPhase.INVALIDATED,
        phase_started_at=terminal_at,
        terminal_at=terminal_at,
        terminal_reason="micro_bos_opposed",
        terminal_source_ids=("opposed-after-owner",),
        selected_trigger=fresh_terminal_trigger,
        context_metadata={},
    )
    projected = _freeze_lsr_context_execution_owner(
        _advance_observation(observation, terminal_at),
        {owner.candidate_id: terminal},
        {owner.context_thesis_id: owner.candidate_id},
        {owner.context_thesis_id: snapshot},
        belief.global_context,
        brain.config,
    )[owner.candidate_id]

    assert projected.phase is PlaybookPhase.INVALIDATED
    assert projected.terminal_at == terminal_at
    assert projected.terminal_reason == "micro_bos_opposed"
    assert projected.terminal_source_ids == ("opposed-after-owner",)
    assert projected.selected_trigger is not None
    assert projected.selected_trigger.available_trigger_kinds == (
        *owner_kinds,
        "reacceptance_held",
    )
    assert replace(
        projected.selected_trigger,
        available_trigger_kinds=snapshot.trigger.available_trigger_kinds,
    ) == snapshot.trigger
    assert projected.context_metadata["lsr_context_execution_owner"] == (
        owner.candidate_id
    )
    assert projected.context_metadata["lsr_first_executable_at"] == (
        snapshot.first_executable_at.isoformat()
    )


def test_lsr_evicted_root_and_all_terminal_children_still_allow_late_new_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waiting = _lsr_without_entry_trigger(_lsr_observation())
    graph = TemporalMarketSceneGraph()
    brain = _brain()
    initial = brain.update(
        waiting,
        scene_graph=graph,
        scene_delta=graph.update(waiting),
    )
    first = next(iter(_short_lsr_candidates(initial).values()))
    context_id = first.context_thesis_id

    failed_at = waiting.asof + pd.Timedelta(minutes=1)
    failed_observation = _lsr_mark_first_zone_left(
        _advance_observation(waiting, failed_at),
        left_at=failed_at,
    )
    failed = brain.update(
        failed_observation,
        scene_graph=graph,
        scene_delta=graph.update(failed_observation),
    )
    assert all(
        item.phase is PlaybookPhase.INVALIDATED
        for item in _short_lsr_candidates(failed).values()
    )
    assert failed.context_theses[context_id].lifecycle == "active"
    assert first.candidate_id not in brain._candidate_priors
    assert len(brain._lsr_context_bindings) == 1

    monkeypatch.setattr(
        "brain.core.playbooks.build_open_market_theses",
        lambda *_args, **_kwargs: (),
    )
    gap_at = failed_at + pd.Timedelta(minutes=1)
    gap = _advance_observation(failed_observation, gap_at)
    gap = replace(
        gap,
        manipulations=(),
        qualified_reacceptances=(),
        micro_bos_references=(),
        path_sequences=tuple(
            path
            for path in gap.path_sequences
            if path.context_kind == "zone_return"
        ),
    )
    childless = brain.update(
        gap,
        scene_graph=graph,
        scene_delta=graph.update(gap),
    )
    assert _short_lsr_candidates(childless) == {}
    assert childless.context_theses[context_id].lifecycle == "active"
    assert childless.context_theses[context_id].child_episode_ids == ()

    late_at = gap_at + pd.Timedelta(minutes=3)
    late = _advance_observation(gap, late_at)
    late = _lsr_add_zone_episode(
        late,
        suffix="c-after-eviction",
        formed_at=gap_at + pd.Timedelta(minutes=1),
        pullback_at=gap_at + pd.Timedelta(minutes=2),
        trigger_at=late_at,
    )
    belief = brain.update(
        late,
        scene_graph=graph,
        scene_delta=graph.update(late),
    )
    late_candidates = _short_lsr_candidates(belief)
    assert len(late_candidates) == 1
    late_candidate = next(iter(late_candidates.values()))
    assert late_candidate.candidate_id != first.candidate_id
    assert late_candidate.context_thesis_id == context_id
    assert late_candidate.entry_location_id == "lsr-location-c-after-eviction"
    assert late_candidate.phase is PlaybookPhase.EXECUTABLE
    assert late_candidate.market_thesis_match_status == "exact_root_bound"
    assert late_candidate.market_thesis_action_bound
    assert late_candidate.context_metadata["lsr_pool_path_id"] == (
        next(iter(brain._lsr_context_seeds.values())).path.sequence_id
    )


def test_lsr_same_bar_wick_cannot_trigger_but_next_bar_wick_can() -> None:
    observation = _lsr_without_entry_trigger(_lsr_observation())
    entry_path = next(
        path
        for path in observation.path_sequences
        if path.context_kind == "zone_return"
    )
    first_pullback = entry_path.steps[-1]
    location = observation.entry_locations[0]
    same_bar_wick = PathSequenceStep(
        step_id="lsr-same-bar-wick",
        kind="wick_rejection",
        observed_at=first_pullback.observed_at,
        source_event_id=None,
        source_entity_id=location.location_id,
        predecessor_step_ids=(first_pullback.step_id,),
        same_clock_relation="same_clock_known",
        direction=Direction.SHORT,
        strength=0.8,
        reason="same_bar_wick_rejection",
    )
    same_bar = replace(
        observation,
        path_sequences=tuple(
            replace(path, steps=(*path.steps, same_bar_wick))
            if path.sequence_id == entry_path.sequence_id
            else path
            for path in observation.path_sequences
        ),
    )
    brain = _brain()
    waiting = brain.update(same_bar).hypotheses[
        "liquidity_sweep_reversal:short"
    ]
    assert waiting.phase is PlaybookPhase.WAITING_TRIGGER
    assert waiting.selected_trigger is None

    later_at = observation.asof + pd.Timedelta(minutes=1)
    later_wick = replace(
        same_bar_wick,
        step_id="lsr-next-bar-wick",
        observed_at=later_at,
        same_clock_relation="strictly_after",
        reason="later_zone_rejection",
    )
    later = _advance_observation(observation, later_at)
    later = replace(
        later,
        entry_locations=(
            replace(
                location,
                lifecycle=EntryLocationLifecycle.REJECTED,
                state_started_at=later_at,
                last_updated_at=later_at,
                state_duration_real_1m_bars=0,
                rejected_at=later_at,
                reaction_atr=0.5,
                transition_reason="later_bar_wick_rejection",
            ),
        ),
        path_sequences=tuple(
            replace(
                path,
                lifecycle=PathSequenceLifecycle.CLOSED,
                state_started_at=later_at,
                last_updated_at=later_at,
                state_duration_real_1m_bars=0,
                steps=(*path.steps, later_wick),
                ended_at=later_at,
                transition_reason="zone_rejection_observed",
            )
            if path.sequence_id == entry_path.sequence_id
            else path
            for path in later.path_sequences
        ),
    )
    executable = brain.update(later).hypotheses[
        "liquidity_sweep_reversal:short"
    ]
    assert executable.phase is PlaybookPhase.EXECUTABLE
    assert executable.selected_trigger is not None
    assert executable.selected_trigger.trigger_id == later_wick.step_id


def test_lsr_provisional_target_delivery_keeps_episode_waiting_for_pullback() -> None:
    # Replay analogue: zone formation 09:55, provisional route target consumed
    # at 09:56, and the zone's physical first return only at 10:02.  The
    # pre-owner target is descriptive and must not tombstone the Episode.
    formation = _lsr_waiting_for_first_pullback(_lsr_observation())
    graph = TemporalMarketSceneGraph()
    brain = _brain()
    initial_belief = brain.update(
        formation,
        scene_graph=graph,
        scene_delta=graph.update(formation),
    )
    initial = next(iter(_short_lsr_candidates(initial_belief).values()))
    assert initial.phase is PlaybookPhase.WAITING_LOCATION
    assert initial.plan is not None
    assert initial.context_metadata.get("lsr_context_execution_owner") is None
    context_id = initial.context_thesis_id
    target_id = initial.plan.targets[0].level_id

    consumed_at = formation.asof + pd.Timedelta(minutes=1)
    consumed = _advance_observation(formation, consumed_at)
    consumed = replace(
        consumed,
        liquidity_inventory=tuple(
            replace(
                item,
                lifecycle=LiquidityInventoryLifecycle.CONSUMED,
                consumed_at=consumed_at,
                lifecycle_reason="swing_swept",
            )
            if item.item_id == target_id
            else item
            for item in consumed.liquidity_inventory
        ),
    )
    consumed_belief = brain.update(
        consumed,
        scene_graph=graph,
        scene_delta=graph.update(consumed),
    )
    still_waiting = _short_lsr_candidates(consumed_belief)[
        initial.candidate_id
    ]

    assert still_waiting.phase is PlaybookPhase.WAITING_LOCATION
    assert still_waiting.terminal_at is None
    assert still_waiting.terminal_reason is None
    assert still_waiting.context_thesis_id == context_id
    assert consumed_belief.context_theses[context_id].lifecycle == "active"

    pullback_at = formation.asof + pd.Timedelta(minutes=7)
    pulled_back = _lsr_register_first_pullback(consumed, pullback_at)
    pullback_belief = brain.update(
        pulled_back,
        scene_graph=graph,
        scene_delta=graph.update(pulled_back),
    )
    advanced = _short_lsr_candidates(pullback_belief)[initial.candidate_id]

    assert advanced.phase is PlaybookPhase.WAITING_TRIGGER
    assert advanced.terminal_at is None
    assert advanced.terminal_reason is None
    assert advanced.context_thesis_id == context_id
    assert advanced.hard_gate_results["reversal_first_pullback"]


def test_lsr_child_target_delivery_does_not_complete_parent_context() -> None:
    observation = _lsr_observation()
    graph = TemporalMarketSceneGraph()
    brain = _brain()
    initial_belief = brain.update(
        observation,
        scene_graph=graph,
        scene_delta=graph.update(observation),
    )
    first = next(iter(_short_lsr_candidates(initial_belief).values()))
    assert first.phase is PlaybookPhase.EXECUTABLE
    assert first.plan is not None
    assert first.selected_trigger is not None
    assert (
        first.context_metadata["lsr_context_execution_owner"]
        == first.candidate_id
    )
    context_id = first.context_thesis_id
    target_id = first.plan.targets[0].level_id

    later_at = observation.asof + pd.Timedelta(minutes=2)
    later = _advance_observation(observation, later_at)
    later = _lsr_add_zone_episode(
        later,
        suffix="b-after-child-delivery",
        formed_at=observation.asof + pd.Timedelta(seconds=30),
        pullback_at=observation.asof + pd.Timedelta(minutes=1),
        trigger_at=None,
    )
    later = replace(
        later,
        liquidity_inventory=tuple(
            replace(
                item,
                lifecycle=LiquidityInventoryLifecycle.CONSUMED,
                consumed_at=later_at,
                lifecycle_reason="swing_swept",
            )
            if item.item_id == target_id
            else item
            for item in later.liquidity_inventory
        ),
    )
    belief = brain.update(
        later,
        scene_graph=graph,
        scene_delta=graph.update(later),
    )
    candidates = _short_lsr_candidates(belief)
    delivered = candidates[first.candidate_id]
    sibling = next(
        item
        for identity, item in candidates.items()
        if identity != first.candidate_id
    )

    assert delivered.phase is PlaybookPhase.COMPLETED
    assert delivered.plan is not None
    assert first.plan is not None
    assert replace(
        delivered.plan,
        remaining_path_R=first.plan.remaining_path_R,
    ) == first.plan
    assert delivered.selected_trigger == first.selected_trigger
    assert delivered.context_metadata["lsr_context_execution_owner"] == (
        first.candidate_id
    )
    assert delivered.context_metadata["lsr_first_executable_at"] == (
        first.phase_started_at.isoformat()
    )
    assert sibling.phase not in {
        PlaybookPhase.COMPLETED,
        PlaybookPhase.INVALIDATED,
    }
    assert sibling.entry_location_id == "lsr-location-b-after-child-delivery"
    assert belief.context_theses[context_id].lifecycle == "active"
    assert belief.context_theses[context_id].context_draw is None
    assert context_id not in brain._closed_contexts
    delivered_decision = _ready_decision_layer(brain).decide(later, belief)
    assert delivered_decision.selected_action is Action.ABSTAIN


def test_lsr_sweep_breach_cascades_siblings_and_same_root_cannot_rearm() -> None:
    observation = _lsr_without_entry_trigger(_lsr_observation())
    observation = _lsr_add_zone_episode(
        observation,
        suffix="b-before-context-terminal",
        formed_at=observation.asof - pd.Timedelta(minutes=2),
        pullback_at=observation.asof - pd.Timedelta(minutes=1),
        trigger_at=None,
    )
    graph = TemporalMarketSceneGraph()
    brain = _brain()
    initial = brain.update(
        observation,
        scene_graph=graph,
        scene_delta=graph.update(observation),
    )
    initial_candidates = _short_lsr_candidates(initial)
    assert len(initial_candidates) == 2
    context_id = next(iter(initial_candidates.values())).context_thesis_id

    breach_at = observation.asof + pd.Timedelta(minutes=1)
    breached_observation = _advance_observation(
        observation,
        breach_at,
        price=101.5,
    )
    breached = brain.update(
        breached_observation,
        scene_graph=graph,
        scene_delta=graph.update(breached_observation),
    )
    breached_candidates = _short_lsr_candidates(breached)
    assert set(breached_candidates) == set(initial_candidates)
    assert all(
        item.phase is PlaybookPhase.INVALIDATED
        for item in breached_candidates.values()
    )
    assert breached.context_theses[context_id].lifecycle == "invalidated"
    assert breached.context_theses[context_id].terminal_reason == (
        "frozen_invalidation_breached"
    )

    after_at = breach_at + pd.Timedelta(minutes=2)
    after = _advance_observation(observation, after_at, price=100.5)
    after = _lsr_add_zone_episode(
        after,
        suffix="c-same-root-after-terminal",
        formed_at=breach_at + pd.Timedelta(seconds=30),
        pullback_at=breach_at + pd.Timedelta(minutes=1),
        trigger_at=after_at,
    )
    repeated = brain.update(
        after,
        scene_graph=graph,
        scene_delta=graph.update(after),
    )
    assert context_id not in repeated.context_theses or (
        repeated.context_theses[context_id].lifecycle == "invalidated"
    )
    assert brain._closed_contexts[context_id].reason == (
        "frozen_invalidation_breached"
    )
    assert all(
        item.phase is not PlaybookPhase.EXECUTABLE
        for item in _short_lsr_candidates(repeated).values()
    )
    assert all(
        item.entry_location_id
        != "lsr-location-c-same-root-after-terminal"
        for item in _short_lsr_candidates(repeated).values()
    )
    assert all(
        episode.entry_location_id
        != "lsr-location-c-same-root-after-terminal"
        for episode in repeated.entry_episodes.values()
    )
    exemplar = next(iter(initial_candidates.values()))
    retired_seed_key = (
        repeated.global_context.market_epoch_id,
        "manipulation:pool-above",
        Direction.SHORT,
    )
    assert retired_seed_key in brain._lsr_retired_seed_keys
    assert retired_seed_key in pickle.loads(
        pickle.dumps(brain)
    )._lsr_retired_seed_keys
    assert (
        repeated.global_context.market_epoch_id,
        "manipulation:new-root",
        Direction.SHORT,
    ) not in brain._lsr_retired_seed_keys
    assert _lsr_zone_candidate_id(
        repeated.global_context.market_epoch_id,
        "manipulation:new-root",
        exemplar.context_metadata["lsr_displacement_id"],
        exemplar.context_metadata["lsr_entry_zone_id"],
        Direction.SHORT,
    ) != exemplar.candidate_id


def test_lsr_exact_source_invalidation_cascades_all_zone_episodes() -> None:
    observation = _lsr_without_entry_trigger(_lsr_observation())
    observation = _lsr_add_zone_episode(
        observation,
        suffix="b-before-source-terminal",
        formed_at=observation.asof - pd.Timedelta(minutes=2),
        pullback_at=observation.asof - pd.Timedelta(minutes=1),
        trigger_at=None,
    )
    graph = TemporalMarketSceneGraph()
    belief = _brain().update(
        observation,
        scene_graph=graph,
        scene_delta=graph.update(observation),
    )
    candidates = _short_lsr_candidates(belief)
    assert len(candidates) == 2
    first_id, first = next(iter(candidates.items()))
    sibling_id, sibling = next(
        item for item in candidates.items() if item[0] != first_id
    )
    context = belief.context_theses[first.context_thesis_id]
    manipulation_id = first.context_metadata["lsr_manipulation_id"]
    assert manipulation_id in context.authority_ids
    source_terminal = replace(
        first,
        phase=PlaybookPhase.INVALIDATED,
        phase_started_at=observation.asof,
        terminal_at=observation.asof,
        terminal_reason="global_frozen_source_invalidated",
        terminal_source_ids=(manipulation_id,),
    )

    cascaded = _cascade_context_terminals(
        observation,
        {first_id: source_terminal, sibling_id: sibling},
        {},
        context_theses={first.context_thesis_id: context},
    )
    assert cascaded[first_id].phase is PlaybookPhase.INVALIDATED
    assert cascaded[sibling_id].phase is PlaybookPhase.INVALIDATED
    assert cascaded[sibling_id].terminal_reason == (
        "parent_context_global_frozen_source_invalidated"
    )


def test_lsr_established_context_ignores_late_accepted_outside_rollover() -> None:
    waiting = _lsr_without_entry_trigger(_lsr_observation())
    graph = TemporalMarketSceneGraph()
    brain = _brain()
    initial = brain.update(
        waiting,
        scene_graph=graph,
        scene_delta=graph.update(waiting),
    )
    first = next(iter(_short_lsr_candidates(initial).values()))
    context_id = first.context_thesis_id

    later_at = waiting.asof + pd.Timedelta(minutes=3)
    later = _advance_observation(waiting, later_at)
    later = _lsr_add_zone_episode(
        later,
        suffix="b-after-rollover",
        formed_at=waiting.asof + pd.Timedelta(minutes=1),
        pullback_at=waiting.asof + pd.Timedelta(minutes=2),
        trigger_at=later_at,
    )
    later = replace(
        later,
        manipulations=tuple(
            replace(
                item,
                lifecycle=ManipulationLifecycle.ACCEPTED_OUTSIDE,
                state_started_at=later_at,
                last_updated_at=later_at,
                reaccepted_at=None,
                accepted_outside_at=later_at,
                resolved_at=later_at,
                reentry_price=None,
                resolved_side=item.side,
                outside_completed_bars=2,
                age_1m_bars=max(item.age_1m_bars, 3),
                inside_hold_bars=0,
                outside_run=2,
                outside_run_side=item.side,
                reentry_candidate_at=None,
                reentry_candidate_price=None,
                transition_reason="later_lifecycle_rollover",
            )
            for item in later.manipulations
        ),
    )
    belief = brain.update(
        later,
        scene_graph=graph,
        scene_delta=graph.update(later),
    )
    late = next(
        item
        for item in _short_lsr_candidates(belief).values()
        if item.entry_location_id == "lsr-location-b-after-rollover"
    )
    assert late.context_thesis_id == context_id
    assert late.phase is PlaybookPhase.EXECUTABLE
    assert late.plan is not None
    assert belief.context_theses[context_id].lifecycle == "active"

    # Risk validates the frozen reacceptance/displacement prefix and exact
    # current identity.  A later Group4 lifecycle projection may omit the
    # historical reaccepted_at field without erasing an established Context.
    decision = Decision(
        asof=later.asof,
        selected_action=Action.ENTER,
        utilities=(
            ActionUtility(
                action=Action.ENTER,
                utility=1.0,
                components={},
                hypothesis_key=late.key,
                reason="exercise independent Risk provenance validation",
            ),
        ),
        best_hypothesis_key=late.key,
        advantage=1.0,
        reasons=("exercise independent Risk provenance validation",),
        plan=late.plan,
    )
    risk = StructuralRiskEngine().review(decision, later)
    assert risk.passed

    # The rollover exception is not a broad provenance fallback: an exact
    # manipulation identity with a changed protocol is still rejected.
    spoofed = replace(
        later,
        manipulations=tuple(
            replace(item, protocol_hash="b" * 64)
            for item in later.manipulations
        ),
    )
    spoofed_risk = StructuralRiskEngine().review(decision, spoofed)
    assert not spoofed_risk.passed
    assert spoofed_risk.final_action.value == "abstain"


def test_first_qualified_trigger_is_frozen_until_a_new_setup_rearms() -> None:
    belief = _mapped_brain().update(_lsr_observation())
    prior = belief.hypotheses["liquidity_sweep_reversal:short"]
    frozen = prior.selected_trigger
    assert frozen is not None

    later = PathSequenceStep(
        step_id="later-stronger-trigger",
        kind="reacceptance_held",
        observed_at=frozen.observed_at + pd.Timedelta(minutes=1),
        source_event_id="later-trigger-event",
        source_entity_id="later-trigger-state",
        predecessor_step_ids=(frozen.trigger_id,),
        same_clock_relation="strictly_after",
        direction=prior.direction,
        strength=0.95,
        reason="later_real_completed_hold",
    )
    retained, support = _select_frozen_trigger(
        (later,),
        prior=prior,
        setup_id=frozen.setup_id,
        entry_path_id=frozen.entry_path_id,
        entry_location_id=frozen.entry_location_id,
        direction=prior.direction,
    )

    assert retained is not None
    assert retained.trigger_id == frozen.trigger_id
    assert retained.trigger_kind == frozen.trigger_kind
    assert retained.observed_at == frozen.observed_at
    assert retained.available_trigger_kinds == (
        frozen.trigger_kind,
        "reacceptance_held",
    )
    assert support is later

    rearmed, _ = _select_frozen_trigger(
        (later,),
        prior=prior,
        setup_id="new-setup",
        entry_path_id="new-path",
        entry_location_id="new-location",
        direction=prior.direction,
    )
    assert rearmed is not None
    assert rearmed.trigger_id == later.step_id
    assert rearmed.observed_at == later.observed_at
    assert rearmed.setup_id == "new-setup"


def test_lsr_prior_trigger_cannot_precede_the_exact_zone_first_pullback() -> None:
    prior = _mapped_brain().update(_lsr_observation()).hypotheses[
        "liquidity_sweep_reversal:short"
    ]
    frozen = prior.selected_trigger
    assert frozen is not None
    later_pullback = frozen.observed_at + pd.Timedelta(minutes=1)

    retained, support = _select_frozen_trigger(
        (),
        prior=prior,
        setup_id=frozen.setup_id,
        entry_path_id=frozen.entry_path_id,
        entry_location_id=frozen.entry_location_id,
        direction=prior.direction,
        trigger_must_follow=later_pullback,
        require_strictly_after=True,
    )

    assert retained is None
    assert support is None
    assert (
        _retain_matching_prior_trigger(
            frozen,
            prior=prior,
            setup_id=frozen.setup_id,
            entry_location_id=frozen.entry_location_id,
            entry_path_id=frozen.entry_path_id,
            direction=prior.direction,
            trigger_must_follow=later_pullback,
            require_strictly_after=True,
        )
        is None
    )


def test_lsr_entry_episode_rejects_trigger_without_causal_first_pullback() -> None:
    observation = _lsr_observation()
    candidate = _mapped_brain().update(observation).hypotheses[
        "liquidity_sweep_reversal:short"
    ]
    assert candidate.selected_trigger is not None
    assert candidate.episode_id is not None
    assert candidate.context_thesis_id is not None
    assert candidate.entry_location_id is not None
    assert candidate.entry_path_id is not None
    assert candidate.episode_deadline is not None

    with pytest.raises(
        ValueError,
        match="entry episode identity or lifecycle is invalid",
    ):
        EntryEpisodeState(
            episode_id=candidate.episode_id,
            parent_context_thesis_id=candidate.context_thesis_id,
            candidate_id="candidate:causally-impossible-trigger",
            playbook=candidate.playbook,
            direction=candidate.direction,
            initiating_event_id=candidate.initiating_event_id,
            entry_location_id=candidate.entry_location_id,
            entry_path_id=candidate.entry_path_id,
            first_pullback_at=None,
            selected_trigger=candidate.selected_trigger,
            plan=candidate.plan,
            invalidation=candidate.invalidation,
            deadline=candidate.episode_deadline,
            phase=candidate.phase,
            formed_at=candidate.phase_started_at,
            updated_at=observation.asof,
        )


def test_prior_trigger_is_not_borrowed_when_same_setup_changes_entry_path() -> None:
    prior = _mapped_brain().update(_lsr_observation()).hypotheses[
        "liquidity_sweep_reversal:short"
    ]
    frozen = prior.selected_trigger
    assert frozen is not None

    assert (
        _retain_matching_prior_trigger(
            None,
            prior=prior,
            setup_id=frozen.setup_id,
            entry_location_id="replacement-location",
            entry_path_id=frozen.entry_path_id,
            direction=prior.direction,
        )
        is None
    )
    assert (
        _retain_matching_prior_trigger(
            None,
            prior=prior,
            setup_id=frozen.setup_id,
            entry_location_id=frozen.entry_location_id,
            entry_path_id="replacement-path",
            direction=prior.direction,
        )
        is None
    )
    assert (
        _retain_matching_prior_trigger(
            None,
            prior=prior,
            setup_id=frozen.setup_id,
            entry_location_id=frozen.entry_location_id,
            entry_path_id=frozen.entry_path_id,
            direction=prior.direction,
        )
        is frozen
    )


def test_selected_trigger_is_rejected_when_effective_path_has_changed() -> None:
    prior = _mapped_brain().update(_lsr_observation()).hypotheses[
        "liquidity_sweep_reversal:short"
    ]
    frozen = prior.selected_trigger
    assert frozen is not None

    assert (
        _retain_matching_prior_trigger(
            frozen,
            prior=None,
            setup_id=frozen.setup_id,
            entry_location_id=frozen.entry_location_id,
            entry_path_id="replacement-effective-path",
            direction=prior.direction,
        )
        is None
    )


def test_public_brain_evaluates_compatible_lsr_roots_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _lsr_observation()

    # First obtain the naturally derived, fully connected LSR root.  The
    # second thesis deliberately has the same mechanism/direction but no
    # graph closure, draw, entry zone or trigger.  Only thesis discovery is
    # replaced below; both roots still pass through the public Brain update
    # and the normal typed evaluator/binding/phase pipeline.
    seed_graph = TemporalMarketSceneGraph()
    seed_belief = _brain().update(
        observation,
        scene_graph=seed_graph,
        scene_delta=seed_graph.update(observation),
    )
    complete_thesis = next(
        thesis
        for thesis in seed_belief.market_theses
        if thesis.mechanism == "liquidity_sweep"
        and thesis.direction is Direction.SHORT
        and thesis.root_id == "manipulation:pool-above"
    )
    incomplete_thesis = replace(
        complete_thesis,
        thesis_id="market-thesis:independent-incomplete-sweep",
        root_id="manipulation:independent-incomplete-sweep",
        mechanism_event_ids=(
            "manipulation:independent-incomplete-sweep",
        ),
        entry_location_ids=(),
        trigger_event_ids=(),
        draw_candidate_ids=(),
        evidence_state=None,
    )

    def _frozen_two_roots(
        _previous: object,
        current_observation: MarketObservation,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[object, object]:
        return (
            replace(complete_thesis, updated_at=current_observation.asof),
            replace(incomplete_thesis, updated_at=current_observation.asof),
        )

    monkeypatch.setattr(
        "brain.core.playbooks.build_open_market_theses",
        _frozen_two_roots,
    )

    graph = TemporalMarketSceneGraph()
    brain = _brain()
    belief = brain.update(
        observation,
        scene_graph=graph,
        scene_delta=graph.update(observation),
    )

    lsr_candidates = {
        candidate.required_root_id: (candidate_id, candidate)
        for candidate_id, candidate in belief.action_candidate_items()
        if candidate.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
        and candidate.direction is Direction.SHORT
    }
    assert set(lsr_candidates) == {
        complete_thesis.root_id,
        incomplete_thesis.root_id,
    }

    complete_id, complete = lsr_candidates[complete_thesis.root_id]
    incomplete_id, incomplete = lsr_candidates[incomplete_thesis.root_id]
    assert complete_id != incomplete_id
    assert complete_id == _lsr_zone_candidate_id(
        belief.global_context.market_epoch_id,
        complete_thesis.root_id,
        complete.context_metadata["lsr_displacement_id"],
        complete.context_metadata["lsr_entry_zone_id"],
        Direction.SHORT,
    )
    assert incomplete_id == _thesis_candidate_id(
        belief.global_context.market_epoch_id,
        incomplete_thesis.root_id,
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
    )

    assert complete.phase is PlaybookPhase.EXECUTABLE
    assert complete.market_thesis_action_bound
    assert complete.plan is not None
    assert complete.plan_feasibility is not None
    assert complete.plan_feasibility.valid
    assert complete.plan.selected_draw_id == "swing:lsr-target-low"

    assert incomplete.phase is not PlaybookPhase.EXECUTABLE
    assert not incomplete.market_thesis_action_bound
    assert incomplete.market_thesis_match_status == "root_identity_unbound"
    assert incomplete.plan is None
    assert incomplete.plan_feasibility is not None
    assert incomplete.plan_feasibility.failure_reason == (
        "market_thesis_root_unbound"
    )
    assert incomplete.deliverable_targets == ()
    assert incomplete.entry_location_id is None
    assert incomplete.selected_trigger is None
    assert incomplete.required_root_id == incomplete_thesis.root_id
    assert incomplete.setup_context_id != complete.setup_context_id

    # Decision consumes the root candidates, not the six-slot summary.  Give
    # the already-computed delivery dimension an explicit ready calibration
    # identity so this assertion isolates causal eligibility: the complete
    # root is authorized, while the incomplete root remains visible but may
    # not borrow its sibling's draw/zone/trigger.
    calibration_version = "typed-calibration:root-candidate-test"
    decision_belief = replace(
        belief,
        thesis_candidates={
            identity: replace(
                candidate,
                calibration_version=calibration_version,
                delivery_quality=(
                    0.80 if identity == complete_id else candidate.delivery_quality
                ),
            )
            for identity, candidate in belief.thesis_candidates.items()
        },
    )
    decision = UtilityDecisionLayer(
        calibration_ready=True,
        calibration_version=calibration_version,
    ).decide(observation, decision_belief)
    assert decision_belief.owns_actionable_entry_episode(
        complete_id,
        decision_belief.thesis_candidates[complete_id],
    )
    assert not decision_belief.owns_actionable_entry_episode(
        incomplete_id,
        decision_belief.thesis_candidates[incomplete_id],
    )
    with pytest.raises(
        ValueError,
        match="candidate and entry episode lifecycle disagree",
    ):
        replace(
            decision_belief,
            entry_episodes={
                **decision_belief.entry_episodes,
                complete_id: replace(
                    decision_belief.entry_episodes[complete_id],
                    phase=PlaybookPhase.WAITING_TRIGGER,
                ),
            },
        )
    complete_enter = next(
        item
        for item in decision.utilities
        if item.action.value == "enter"
        and item.hypothesis_key == complete_id
    )
    incomplete_enter = next(
        item
        for item in decision.utilities
        if item.action.value == "enter"
        and item.hypothesis_key == incomplete_id
    )
    assert complete_enter.components["action_authorized"] == 1.0
    assert incomplete_enter.components["causal_eligible"] == 0.0
    assert incomplete_enter.components["action_authorized"] == 0.0

    # The six playbook-direction views remain summaries; the independent
    # action identities and their priors are retained by candidate_id.
    assert belief.hypotheses[
        "liquidity_sweep_reversal:short"
    ].candidate_id is None
    assert brain._candidate_priors[complete_id] is complete
    assert brain._candidate_priors[incomplete_id] is incomplete

    terminal = replace(
        complete,
        phase=PlaybookPhase.INVALIDATED,
        phase_started_at=observation.asof,
        terminal_at=observation.asof,
        terminal_reason="complete_root_invalidated_for_isolation_test",
        terminal_source_ids=(complete_thesis.root_id,),
    )
    brain._candidate_priors[complete_id] = terminal
    repeated_observation = _advance_observation(
        observation,
        observation.asof + pd.Timedelta(minutes=1),
    )
    repeated = brain.update(
        repeated_observation,
        scene_graph=graph,
        scene_delta=graph.update(repeated_observation),
    )
    repeated_candidates = dict(repeated.action_candidate_items())
    assert repeated_candidates[complete_id] == terminal
    assert repeated_candidates[incomplete_id] is not incomplete
    assert repeated_candidates[incomplete_id].phase is not PlaybookPhase.INVALIDATED
    assert (
        repeated_candidates[incomplete_id].required_root_id
        == incomplete_thesis.root_id
    )


def test_lsr_accepts_alternative_zone_trigger_and_original_sweep_stop() -> None:
    observation = _lsr_observation()
    brain = _mapped_brain()
    belief = brain.update(observation)
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

    graph = TemporalMarketSceneGraph()
    graph_belief = _brain().update(
        observation,
        scene_graph=graph,
        scene_delta=graph.update(observation),
    )
    graph_hypothesis = graph_belief.hypotheses[
        "liquidity_sweep_reversal:short"
    ]
    assert graph_hypothesis.phase is PlaybookPhase.EXECUTABLE
    assert graph_hypothesis.market_thesis_action_bound
    assert graph_hypothesis.candidate_id is None
    root_candidates = graph_belief.action_candidate_items()
    candidate_id, root_candidate = next(
        item
        for item in root_candidates
        if item[1].playbook is graph_hypothesis.playbook
        and item[1].direction is graph_hypothesis.direction
    )
    assert candidate_id == root_candidate.candidate_id
    assert root_candidate.required_root_id == "manipulation:pool-above"
    assert root_candidate.market_thesis_action_bound
    assert root_candidate.phase is PlaybookPhase.EXECUTABLE
    assert root_candidate.plan == graph_hypothesis.plan
    assert any(
        thesis.root_id == "manipulation:pool-above"
        for thesis in graph_belief.market_theses
    )
    assert len(graph_belief.candidates()) == len(graph_belief.hypotheses)
    assert belief.action_candidate_items() == ()

    primary_thesis = next(
        thesis
        for thesis in graph_belief.market_theses
        if thesis.thesis_id == graph_hypothesis.bound_market_thesis_id
    )
    secondary_thesis = replace(
        primary_thesis,
        thesis_id="market-thesis:secondary-compatible-root",
        root_id="manipulation:secondary-root",
        mechanism_event_ids=("manipulation:secondary-root",),
        trigger_event_ids=(),
        evidence_state=None,
    )

    # Each compatible root owns an independent evaluator/prior identity.  A
    # second root that has no draw/location closure cannot borrow the primary
    # root's executable plan.
    second_context = replace(
        graph_belief.global_context,
        open_market_theses=(primary_thesis, secondary_thesis),
    )
    primary_id = root_candidate.candidate_id
    secondary_id = _thesis_candidate_id(
        second_context.market_epoch_id,
        secondary_thesis.root_id,
        graph_hypothesis.playbook,
        graph_hypothesis.direction,
    )
    assert primary_id != secondary_id
    conflicting_projection = replace(
        root_candidate,
        candidate_id=secondary_id,
        required_root_id=secondary_thesis.root_id,
        market_thesis_ids=(secondary_thesis.thesis_id,),
        market_thesis_id=secondary_thesis.thesis_id,
        bound_market_thesis_id=secondary_thesis.thesis_id,
        market_thesis_root_id=secondary_thesis.root_id,
    )
    # A graph ambiguity that projects one frozen path onto two roots is not a
    # process-fatal invariant and never chooses an arbitrary action owner.
    assert _fail_closed_shared_episode_ownership(
        observation,
        {
            primary_id: root_candidate,
            secondary_id: conflicting_projection,
        },
        {},
    ) == {}
    assert _fail_closed_shared_episode_ownership(
        observation,
        {secondary_id: conflicting_projection},
        {primary_id: root_candidate},
    ) == {}
    terminal_primary = replace(
        root_candidate,
        phase=PlaybookPhase.INVALIDATED,
        phase_started_at=observation.asof,
        terminal_at=observation.asof,
        terminal_reason="frozen_invalidation_breached",
    )
    assert _fail_closed_shared_episode_ownership(
        observation,
        {
            primary_id: terminal_primary,
            secondary_id: conflicting_projection,
        },
        {},
    ) == {
        primary_id: terminal_primary,
        secondary_id: conflicting_projection,
    }
    secondary_evaluation = _typed_lsr(
        observation,
        graph_hypothesis.direction,
        _brain().registry.for_playbook(graph_hypothesis.playbook),
        None,
        BrainConfig(),
        global_context=second_context,
        scene_graph=graph,
        required_thesis=secondary_thesis,
    )
    assert secondary_evaluation.setup_identity is None
    assert secondary_evaluation.plan is None
    assert not any(secondary_evaluation.hard_gate_results.values())

    decision = _ready_decision_layer(brain).decide(
        observation,
        graph_free_action_belief(belief),
    )
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
    location = observation.entry_locations[0]
    paths = []
    for path in observation.path_sequences:
        if path.context_kind != "zone_return":
            paths.append(path)
            continue
        retained_steps = tuple(
            step
            for step in path.steps
            if step.kind != "micro_bos_confirmed"
        )
        first_pullback = retained_steps[-1]
        wick = replace(
            path.steps[-1],
            step_id="lsr-alternative-wick-trigger",
            kind="wick_rejection",
            observed_at=observation.asof,
            source_event_id=None,
            source_entity_id=location.location_id,
            predecessor_step_ids=(first_pullback.step_id,),
            same_clock_relation="strictly_after",
            strength=0.25,
            reason="later_zone_rejection",
        )
        paths.append(
            replace(
                path,
                lifecycle=PathSequenceLifecycle.ACTIVE,
                state_started_at=wick.observed_at,
                last_updated_at=wick.observed_at,
                state_duration_real_1m_bars=0,
                steps=(*retained_steps, wick),
                ended_at=None,
                transition_reason="zone_rejection_observed",
            )
        )
    orphaned_micro = replace(
        observation,
        frames=orphaned_frames,
        entry_locations=(
                replace(
                    location,
                    lifecycle=EntryLocationLifecycle.REJECTED,
                    state_started_at=observation.asof,
                    last_updated_at=observation.asof,
                    state_duration_real_1m_bars=0,
                    rejected_at=observation.asof,
                    reaction_atr=0.25,
                    transition_reason="later_bar_wick_rejection",
                ),
        ),
        micro_bos_references=(),
        path_sequences=tuple(paths),
    )
    orphaned_brain = _mapped_brain()
    orphaned_belief = orphaned_brain.update(orphaned_micro)
    orphaned_hypothesis = orphaned_belief.hypotheses[
        "liquidity_sweep_reversal:short"
    ]
    assert orphaned_hypothesis.phase is PlaybookPhase.EXECUTABLE
    assert orphaned_hypothesis.hard_gate_results["typed_entry_trigger"]
    # The alternative rejection satisfies the causal trigger gate.  Its
    # readiness score is not reused as the probability of target delivery.
    assert _ready_decision_layer(orphaned_brain).decide(
        orphaned_micro,
        graph_free_action_belief(orphaned_belief),
    ).selected_action.value == "enter"

    no_mss_frames = dict(observation.frames)
    no_mss_frames[Timeframe.M5] = replace(
        no_mss_frames[Timeframe.M5],
        structure_breaks=(),
    )
    no_mss = replace(observation, frames=no_mss_frames)
    no_mss_hypothesis = _brain().update(no_mss).hypotheses[
        "liquidity_sweep_reversal:short"
    ]
    assert no_mss_hypothesis.phase is PlaybookPhase.EXECUTABLE
    assert no_mss_hypothesis.hard_gate_results[
        "reverse_displacement_zone"
    ]
    assert next(
        item
        for item in hypothesis.supporting
        if item.primitive == "opposed_mss_confirmation"
    ).value == 1.0
    assert all(
        item.primitive != "opposed_mss_confirmation"
        for item in no_mss_hypothesis.supporting
    )


def test_lsr_shared_physical_episode_keeps_active_contexts_but_no_children() -> None:
    observation = _lsr_observation()
    graph = TemporalMarketSceneGraph()
    belief = _brain().update(
        observation,
        scene_graph=graph,
        scene_delta=graph.update(observation),
    )
    primary_id, primary = next(
        item
        for item in belief.action_candidate_items()
        if item[1].playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
    )
    assert primary.sequence is not None
    assert primary.selected_trigger is not None
    assert primary.plan is not None
    assert primary.context_thesis_id is not None

    secondary_id = "lsr-zone-candidate:secondary-active-context"
    secondary_context_id = "context-thesis:secondary-active-context"
    secondary_episode_id = "lsr-entry-episode:secondary-active-context"
    secondary = replace(
        primary,
        candidate_id=secondary_id,
        required_root_id="manipulation:secondary-active-context",
        market_thesis_ids=("market-thesis:secondary-active-context",),
        market_thesis_id="market-thesis:secondary-active-context",
        bound_market_thesis_id="market-thesis:secondary-active-context",
        market_thesis_root_id="manipulation:secondary-active-context",
        setup_context_id=secondary_episode_id,
        episode_id=secondary_episode_id,
        context_thesis_id=secondary_context_id,
        parent_context_thesis_id=secondary_context_id,
        sequence=replace(
            primary.sequence,
            setup_id=secondary_episode_id,
        ),
        selected_trigger=replace(
            primary.selected_trigger,
            setup_id=secondary_episode_id,
        ),
        plan=replace(
            primary.plan,
            setup_id=secondary_episode_id,
        ),
    )
    projected = {
        primary_id: primary,
        secondary_id: secondary,
    }

    resolved = _fail_closed_shared_episode_ownership(
        observation,
        projected,
        {},
    )
    assert resolved == {}
    contexts, episodes = _build_context_episode_views(
        observation,
        belief.global_context,
        projected,
        {},
        {},
        episode_candidates=resolved,
    )
    assert set(contexts) == {
        primary.context_thesis_id,
        secondary_context_id,
    }
    assert all(context.lifecycle == "active" for context in contexts.values())
    assert all(not context.child_episode_ids for context in contexts.values())
    assert episodes == {}

    # A local entry-window terminal remains path history under its still-live
    # parent and cannot be recycled by another active Context.
    local_terminal = replace(
        primary,
        phase=PlaybookPhase.INVALIDATED,
        phase_started_at=observation.asof,
        terminal_at=observation.asof,
        terminal_reason="entry_window_expired",
    )
    assert _fail_closed_shared_episode_ownership(
        observation,
        {
            primary_id: local_terminal,
            secondary_id: secondary,
        },
        {},
    ) == {}


def test_lsr_late_mss_is_supporting_evidence_not_a_terminal_gate() -> None:
    observation = _lsr_observation()
    frames = dict(observation.frames)
    late_mss = replace(
        frames[Timeframe.M5].structure_breaks[0],
        resolved_at=observation.asof,
    )
    frames[Timeframe.M5] = replace(
        frames[Timeframe.M5],
        structure_breaks=(late_mss,),
    )
    late = replace(observation, frames=frames)

    hypothesis = _brain().update(late).hypotheses[
        "liquidity_sweep_reversal:short"
    ]

    assert hypothesis.phase is PlaybookPhase.EXECUTABLE
    assert hypothesis.terminal_reason is None
    assert hypothesis.hard_gate_results[
        "reverse_displacement_zone"
    ]


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
        liquidity_inventory=(target, observation.liquidity_inventory[1]),
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
    assert hypothesis.phase is PlaybookPhase.WEAKENING


def _lsr_source_at(
    observation: MarketObservation,
    timeframe: Timeframe,
) -> MarketObservation:
    manipulation = replace(
        observation.manipulations[0],
        source_timeframe=timeframe,
    )
    pool = replace(
        observation.liquidity_pool_states[0],
        timeframe=timeframe,
    )
    pool_inventory = replace(
        observation.liquidity_inventory[1],
        timeframe=timeframe,
    )
    frames = {
        key: replace(
            frame,
            liquidity_pools=(pool,) if key is timeframe else (),
        )
        for key, frame in observation.frames.items()
    }
    return replace(
        observation,
        frames=frames,
        manipulations=(manipulation,),
        liquidity_pool_states=(pool,),
        liquidity_inventory=(
            observation.liquidity_inventory[0],
            pool_inventory,
        ),
    )


def test_lsr_authority_tiers_keep_internal_m1_non_root_but_allow_external_rank() -> None:
    base = _lsr_observation()
    h1_candidate, _ = _select_pool_path(base, Direction.SHORT, None)
    assert h1_candidate is not None
    assert h1_candidate.tier == "A"
    assert h1_candidate.eligible_root
    assert h1_candidate.structural_rank == "internal"

    context = SimpleNamespace(
        authority_source_ids=("pool:pool-above",),
        external_draw_candidates={"above": (), "below": ()},
    )
    isolated_m1 = _lsr_source_at(base, Timeframe.M1)
    pool_path = next(
        path
        for path in isolated_m1.path_sequences
        if path.context_kind == "pool_reversal"
    )
    isolated = _lsr_candidate(
        isolated_m1,
        pool_path,
        global_context=None,
        scene_graph=None,
    )
    nested = _lsr_candidate(
        isolated_m1,
        pool_path,
        global_context=context,
        scene_graph=None,
    )
    assert isolated is not None and isolated.tier == "C"
    assert not isolated.eligible_root and not isolated.nested_source
    assert nested is not None and nested.tier == "C"
    assert not nested.eligible_root and nested.nested_source

    external_inventory = replace(
        isolated_m1.liquidity_inventory[1],
        structural_rank="external",
    )
    external_m1 = replace(
        isolated_m1,
        liquidity_inventory=(
            isolated_m1.liquidity_inventory[0],
            external_inventory,
        ),
    )
    external = _lsr_candidate(
        external_m1,
        pool_path,
        global_context=None,
        scene_graph=None,
    )
    assert external is not None and external.tier == "A"
    assert external.eligible_root


def test_lsr_tier_b_requires_global_identity_connection() -> None:
    observation = _lsr_source_at(_lsr_observation(), Timeframe.M5)
    selected, _ = _select_pool_path(
        observation,
        Direction.SHORT,
        None,
    )
    assert selected is None

    connected, _ = _select_pool_path(
        observation,
        Direction.SHORT,
        None,
        global_context=SimpleNamespace(
            authority_source_ids=("pool:pool-above",),
            external_draw_candidates={"above": (), "below": ()},
        ),
    )
    assert connected is not None
    assert connected.tier == "B"
    assert connected.eligible_root

    external_inventory = replace(
        observation.liquidity_inventory[1],
        structural_rank="external",
    )
    external_observation = replace(
        observation,
        liquidity_inventory=(
            observation.liquidity_inventory[0],
            external_inventory,
        ),
    )
    external, _ = _select_pool_path(
        external_observation,
        Direction.SHORT,
        None,
    )
    assert external is not None and external.tier == "A"
    assert external.eligible_root


def test_lsr_active_identity_is_frozen_and_higher_tier_is_competing() -> None:
    clock = BASE + pd.Timedelta(minutes=20)
    path_b = SimpleNamespace(
        sequence_id="path-b",
        context_kind="pool_reversal",
        context_id="manip-b",
        direction=Direction.SHORT,
        lifecycle=PathSequenceLifecycle.ACTIVE,
        ended_at=None,
        formed_at=clock - pd.Timedelta(minutes=2),
        steps=(),
    )
    path_a = SimpleNamespace(
        sequence_id="path-a",
        context_kind="pool_reversal",
        context_id="manip-a",
        direction=Direction.SHORT,
        lifecycle=PathSequenceLifecycle.ACTIVE,
        ended_at=None,
        formed_at=clock - pd.Timedelta(minutes=1),
        steps=(),
    )
    manipulation_b = SimpleNamespace(
        manipulation_id="manip-b",
        source_kind="formed_liquidity_pool",
        source_timeframe=Timeframe.M5,
        source_id="pool-b",
        source_inventory_item_id="inventory-b",
        crossed_source_ids=("pool-b",),
        coincident_source_ids=(),
    )
    manipulation_a = SimpleNamespace(
        manipulation_id="manip-a",
        source_kind="formed_liquidity_pool",
        source_timeframe=Timeframe.H1,
        source_id="pool-a",
        source_inventory_item_id="inventory-a",
        crossed_source_ids=("pool-a",),
        coincident_source_ids=(),
    )
    observation = SimpleNamespace(
        asof=clock,
        path_sequences=(path_b, path_a),
        manipulations=(manipulation_b, manipulation_a),
        liquidity_inventory=(
            SimpleNamespace(
                item_id="inventory-b",
                source_ids=("pool-b",),
                structural_rank="internal",
            ),
            SimpleNamespace(
                item_id="inventory-a",
                source_ids=("pool-a",),
                structural_rank="external",
            ),
        ),
        liquidity_pool_states=(),
    )
    prior = SimpleNamespace(
        phase=PlaybookPhase.WAITING_TRIGGER,
        setup_context_id="path-b",
        hard_gate_results={"formed_pool_sweep": True},
    )

    selected, competing = _select_pool_path(
        observation,
        Direction.SHORT,
        prior,
    )
    assert selected is not None
    assert selected.path.sequence_id == "path-b"
    assert selected.tier == "B"
    assert selected.authority_latched
    assert competing == ("path-a",)

    terminal_prior = SimpleNamespace(
        phase=PlaybookPhase.INVALIDATED,
        phase_started_at=clock,
        terminal_at=clock,
        episode_id="path-b",
        setup_context_id="path-b",
        sequence=None,
        competing_episode_ids=competing,
    )
    promoted, remaining = _select_pool_path(
        observation,
        Direction.SHORT,
        terminal_prior,
    )
    assert promoted is not None
    assert promoted.path.sequence_id == "path-a"
    assert remaining == ()
    assert _terminal_candidate_is_new(
        terminal_prior,
        promoted.path.sequence_id,
        promoted.path.formed_at,
    )


def test_countertrend_lsr_uses_local_target_and_stops_before_blocker() -> None:
    observation = _lsr_observation()
    local = replace(
        observation.liquidity_inventory[0],
        item_id="local-m5-low",
        timeframe=Timeframe.M5,
        kind="previous_session_low",
        price=99.5,
        lower_bound=99.5,
        upper_bound=99.5,
        source_ids=("local-m5-low",),
    )
    far_barrier = replace(
        observation.liquidity_inventory[0],
        item_id="h4-protected-far",
        timeframe=Timeframe.H4,
        kind="previous_day_low",
        price=98.0,
        lower_bound=98.0,
        upper_bound=98.0,
        source_ids=("h4-protected-source",),
        structural_rank="external",
        is_protected_swing=True,
    )
    target_observation = replace(
        observation,
        liquidity_inventory=(
            local,
            far_barrier,
            *observation.liquidity_inventory,
        ),
    )
    context = _brain_global_context(
        direction=Direction.LONG,
        hard_barriers=(
            DeliveryObstruction(
                obstruction_id=far_barrier.item_id,
                timeframe=far_barrier.timeframe,
                direction=None,
                side=far_barrier.side,
                lower_bound=far_barrier.lower_bound,
                upper_bound=far_barrier.upper_bound,
                hard=True,
                source_kind="protected_swing",
                source_ids=far_barrier.source_ids,
                structural_scope="external",
                acceptance_state="confirmed",
            ),
        ),
    )
    selected = _select_countertrend_lsr_target(
        target_observation,
        Direction.SHORT,
        101.0,
        BrainConfig(),
        context,
        context_draw=None,
    )
    assert selected is not None
    assert selected.level_id == "local-m5-low"

    blocker = replace(
        observation.liquidity_inventory[0],
        item_id="h1-protected-blocker",
        kind="previous_day_low",
        price=100.0,
        lower_bound=100.0,
        upper_bound=100.0,
        source_ids=("h1-protected-blocker",),
        structural_rank="external",
        is_protected_swing=True,
    )
    blocked_observation = replace(
        target_observation,
        liquidity_inventory=(
            local,
            blocker,
            *observation.liquidity_inventory,
        ),
    )
    assert _select_countertrend_lsr_target(
        blocked_observation,
        Direction.SHORT,
        101.0,
        BrainConfig(),
        _brain_global_context(
            direction=Direction.LONG,
            hard_barriers=(
                DeliveryObstruction(
                    obstruction_id=blocker.item_id,
                    timeframe=blocker.timeframe,
                    direction=None,
                    side=blocker.side,
                    lower_bound=blocker.lower_bound,
                    upper_bound=blocker.upper_bound,
                    hard=True,
                    source_kind="protected_swing",
                    source_ids=blocker.source_ids,
                    structural_scope="external",
                    acceptance_state="confirmed",
                ),
            ),
        ),
        context_draw=None,
    ) is None


def test_root_bound_countertrend_lsr_can_use_visible_nonclosure_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _lsr_observation()
    original_target, swept_pool = base.liquidity_inventory
    valid_internal_target = replace(
        original_target,
        item_id="h1-internal-before-h4-barrier",
        price=99.5,
        lower_bound=99.5,
        upper_bound=99.5,
        structural_rank="internal",
        is_protected_swing=False,
    )
    wrong_side = replace(
        original_target,
        item_id="wrong-side-visible",
        side="above",
        kind="previous_day_high",
        price=102.0,
        lower_bound=102.0,
        upper_bound=102.0,
        source_ids=("wrong-side-visible-source",),
        structural_rank="internal",
        is_protected_swing=False,
    )
    consumed = replace(
        original_target,
        item_id="consumed-below",
        kind="previous_day_low",
        price=99.25,
        lower_bound=99.25,
        upper_bound=99.25,
        source_ids=("consumed-below-source",),
        lifecycle=LiquidityInventoryLifecycle.CONSUMED,
        consumed_at=base.asof,
        lifecycle_reason="reference_level_swept",
        structural_rank="internal",
        is_protected_swing=False,
    )
    beyond_barrier = replace(
        original_target,
        item_id="m5-beyond-h4-barrier",
        timeframe=Timeframe.M5,
        kind="previous_session_low",
        price=98.5,
        lower_bound=98.5,
        upper_bound=98.5,
        source_ids=("m5-beyond-h4-barrier-source",),
        structural_rank="internal",
        is_protected_swing=False,
    )
    observation = replace(
        base,
        liquidity_inventory=(
            valid_internal_target,
            wrong_side,
            consumed,
            beyond_barrier,
            swept_pool,
        ),
    )
    graph = TemporalMarketSceneGraph()
    scene_delta = graph.update(observation)
    seed_belief = _brain().update(
        observation,
        scene_graph=graph,
        scene_delta=scene_delta,
    )
    root_thesis = next(
        thesis
        for thesis in seed_belief.market_theses
        if thesis.mechanism == "liquidity_sweep"
        and thesis.direction is Direction.SHORT
        and thesis.root_id == "manipulation:pool-above"
    )
    barrier = DeliveryObstruction(
        obstruction_id="h4-protected-barrier",
        timeframe=Timeframe.H4,
        direction=None,
        side="below",
        lower_bound=99.0,
        upper_bound=99.0,
        hard=True,
        source_kind="protected_swing",
        source_ids=("h4-protected-barrier-source",),
        structural_scope="external",
        acceptance_state="confirmed",
    )

    # Root binding protects the sweep/zone/trigger identity.  The thesis draw
    # view must also expose separately visible downstream inventory so LSR can
    # apply its path and authority-barrier rules without borrowing an event
    # identity from another setup.
    required_thesis = root_thesis
    context = replace(
        _brain_global_context(
            direction=Direction.LONG,
            hard_barriers=(barrier,),
        ),
        updated_at=observation.asof,
        market_epoch_id=required_thesis.market_epoch_id,
        scene_revision_id=seed_belief.global_context.scene_revision_id,
        open_market_theses=(required_thesis,),
    )
    monkeypatch.setattr(
        "brain.core.playbooks.update_global_market_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        "brain.core.playbooks.build_open_market_theses",
        lambda *_args, **_kwargs: (required_thesis,),
    )

    belief = _brain().update(
        observation,
        scene_graph=graph,
        scene_delta=scene_delta,
    )
    candidate = next(
        item
        for _, item in belief.action_candidate_items()
        if item.playbook is Playbook.LIQUIDITY_SWEEP_REVERSAL
        and item.direction is Direction.SHORT
        and item.required_root_id == required_thesis.root_id
    )

    assert candidate.market_thesis_match_status == "exact_root_bound"
    assert all(candidate.hard_gate_results.values())
    assert valid_internal_target.item_id in required_thesis.draw_candidate_ids
    assert beyond_barrier.item_id in required_thesis.draw_candidate_ids
    assert wrong_side.item_id not in required_thesis.draw_candidate_ids
    assert consumed.item_id not in required_thesis.draw_candidate_ids
    assert candidate.plan_feasibility is not None
    assert candidate.plan_feasibility.valid
    assert candidate.plan is not None
    assert candidate.plan.selected_draw_id == valid_internal_target.item_id
    assert candidate.plan.selected_draw_id not in {
        wrong_side.item_id,
        consumed.item_id,
        beyond_barrier.item_id,
    }


def test_open_thesis_draw_view_changes_without_revising_causal_evidence() -> None:
    observation = _lsr_observation()
    graph = TemporalMarketSceneGraph()
    brain = _brain()
    first_delta = graph.update(observation)
    first_belief = brain.update(
        observation,
        scene_graph=graph,
        scene_delta=first_delta,
    )
    first = next(
        thesis
        for thesis in first_belief.market_theses
        if thesis.root_id == "manipulation:pool-above"
        and thesis.direction is Direction.SHORT
    )
    original_target = observation.liquidity_inventory[0]
    later_draw = replace(
        original_target,
        item_id="later-visible-below-draw",
        kind="previous_day_low",
        price=original_target.price - 0.25,
        lower_bound=original_target.price - 0.25,
        upper_bound=original_target.price - 0.25,
        source_ids=("later-visible-below-draw-source",),
        confirmed_at=observation.asof,
    )
    later = replace(
        observation,
        liquidity_inventory=(later_draw, *observation.liquidity_inventory),
    )
    later_theses = build_open_market_theses(
        (first,),
        later,
        first_delta,
        graph,
        first_belief.global_context,
    )
    second = next(
        thesis
        for thesis in later_theses
        if thesis.root_id == first.root_id
        and thesis.direction is first.direction
    )

    assert later_draw.item_id in second.draw_candidate_ids
    assert later_draw.item_id not in second.supporting_event_ids
    assert second.evidence_revision_id == first.evidence_revision_id
    assert second.updated_at == first.updated_at


def test_global_routing_preserves_sequence_and_distinguishes_ambiguity() -> None:
    signal = _SequenceSignal(1.0, BASE, ("frozen-source",))
    evaluation = _Evaluation(
        evidence=(),
        trigger_ready=True,
        setup_clock=BASE,
        sequence_signals={"step": signal},
        setup_identity="episode",
        thesis_target=0.8,
        location_quality=0.7,
        entry_readiness=0.9,
        delivery_quality=0.8,
        typed_uncertainty=0.1,
        uncertainty_authority_missing=0.1,
        hard_gate_results={"gate": True},
    )
    ambiguous = _route_global_context(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        None,
        evaluation,
        _brain_global_context(
            direction=Direction.SHORT,
            ambiguous_evidence=("frozen-source",),
        ),
    )
    assert not ambiguous.invalidated
    assert ambiguous.sequence_signals == evaluation.sequence_signals
    assert ambiguous.entry_readiness == 0.0
    assert ambiguous.typed_uncertainty >= 0.5
    assert ambiguous.ambiguity_count == 1

    invalidated = _route_global_context(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        None,
        evaluation,
        _brain_global_context(
            direction=Direction.SHORT,
            invalidated_source_ids=("frozen-source",),
        ),
    )
    assert invalidated.invalidated
    assert invalidated.sequence_signals == evaluation.sequence_signals
    assert invalidated.thesis_target == 0.0
    assert invalidated.location_quality == 0.0
    assert invalidated.entry_readiness == 0.0
    assert invalidated.delivery_quality == 0.0
    assert invalidated.typed_uncertainty == evaluation.typed_uncertainty

    unresolved = _route_global_context(
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
        Direction.SHORT,
        None,
        evaluation,
        _brain_global_context(
            direction=Direction.SHORT,
            market_mode=MarketMode.UNCERTAIN,
            unknown_evidence=("frozen-source",),
        ),
    )
    assert unresolved.sequence_signals == evaluation.sequence_signals
    assert unresolved.thesis_target == evaluation.thesis_target
    assert unresolved.entry_readiness == evaluation.entry_readiness
    assert unresolved.typed_uncertainty >= 0.5


def _lsr_without_plan_or_trigger() -> MarketObservation:
    observation = _lsr_observation()
    frames = {
        timeframe: replace(frame, swings=())
        for timeframe, frame in observation.frames.items()
    }
    frames[Timeframe.M1] = replace(
        frames[Timeframe.M1],
        structure_breaks=(),
    )
    paths = []
    for path in observation.path_sequences:
        if path.context_kind != "zone_return":
            paths.append(path)
            continue
        steps = tuple(
            step
            for step in path.steps
            if step.kind
            not in {
                "wick_rejection",
                "reacceptance_held",
                "micro_bos_confirmed",
            }
        )
        paths.append(
            replace(
                path,
                lifecycle=PathSequenceLifecycle.ACTIVE,
                state_started_at=steps[-1].observed_at,
                last_updated_at=steps[-1].observed_at,
                state_duration_real_1m_bars=0,
                steps=steps,
                ended_at=None,
                transition_reason="context_registered",
            )
        )
    return replace(
        observation,
        frames=frames,
        liquidity_inventory=tuple(
            item
            for item in observation.liquidity_inventory
            if item.kind in {"equal_highs", "equal_lows"}
        ),
        micro_bos_references=(),
        path_sequences=tuple(paths),
    )


def test_pre_entry_episode_freezes_invalidation_and_closes_on_breach() -> None:
    observation = _lsr_without_plan_or_trigger()
    brain = _brain()
    key = "liquidity_sweep_reversal:short"

    active_belief = brain.update(observation)
    active = active_belief.hypotheses[key]
    assert active.phase is PlaybookPhase.WAITING_TRIGGER
    assert active.plan is None
    assert active.episode_deadline == pd.Timestamp(
        "2025-01-07T17:00:00-05:00"
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

    reclaimed = brain.update(
        _advance_observation(
            observation,
            breached_observation.asof + pd.Timedelta(minutes=1),
            price=active.invalidation.price - 0.25,
        )
    ).hypotheses[key]
    assert reclaimed is terminal


def test_entry_episode_deadline_terminalizes_lsr_episode() -> None:
    armed_observation = _lsr_without_plan_or_trigger()
    brain = _brain()
    key = "liquidity_sweep_reversal:short"

    active = brain.update(armed_observation).hypotheses[key]
    assert active.phase is PlaybookPhase.WAITING_TRIGGER
    assert active.plan is None
    assert active.episode_id is not None
    assert active.episode_deadline is not None
    assert active.thesis_deadline is not None

    deadline_observation = _advance_observation(
        armed_observation,
        active.episode_deadline,
    )
    expired = brain.update(deadline_observation).hypotheses[key]

    assert expired.phase is PlaybookPhase.INVALIDATED
    assert expired.terminal_at == active.episode_deadline
    assert expired.terminal_reason == "episode_deadline_elapsed"
    assert expired.episode_deadline == active.episode_deadline
    assert expired.thesis_deadline == active.thesis_deadline
    assert not expired.eligible


def test_lsr_armed_sweep_registers_thesis_before_entry_zone() -> None:
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
        qualified_reacceptances=(),
        micro_bos_references=pool_references,
        path_sequences=pool_paths,
    )
    belief = _brain().update(armed_observation)
    hypothesis = belief.hypotheses["liquidity_sweep_reversal:short"]

    assert hypothesis.phase is PlaybookPhase.ARMED
    assert hypothesis.plan is None
    assert hypothesis.episode_id is None
    assert hypothesis.episode_deadline is None
    assert hypothesis.context_thesis_id is not None
    assert hypothesis.thesis_deadline is not None
    assert hypothesis.thesis_draw is None

    recorder = BrainCalibrationRecorder()
    source_bar = Bar(
        start=armed_observation.asof - pd.Timedelta(minutes=1),
        open=armed_observation.price,
        high=armed_observation.price,
        low=armed_observation.price,
        close=armed_observation.price,
        volume=1.0,
        symbol=armed_observation.symbol,
        instrument_id=armed_observation.instrument_id,
    )
    recorder.observe(
        SimpleNamespace(
            observation=armed_observation,
            belief=graph_free_action_belief(belief),
        ),
        source_bar=source_bar,
    )

    thesis = [
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
        and sample.hypothesis_key == hypothesis.key
    ]
    assert len(thesis) == 1
    assert thesis[0].draw_id is None
    assert thesis[0].calibration_unit_id == hypothesis.context_thesis_id


def test_expired_episode_cannot_publish_a_late_entry_plan() -> None:
    observation = _lsr_observation()
    armed_observation = _lsr_without_plan_or_trigger()
    brain = _brain()
    key = "liquidity_sweep_reversal:short"

    active = brain.update(armed_observation).hypotheses[key]
    assert active.episode_deadline is not None
    assert active.plan is None

    late_location = _advance_observation(
        armed_observation,
        active.episode_deadline,
    )
    expired = brain.update(late_location).hypotheses[key]

    assert expired.episode_deadline == active.episode_deadline
    assert expired.plan is None
    assert expired.phase is PlaybookPhase.INVALIDATED
    assert expired.terminal_at == active.episode_deadline
    assert expired.terminal_reason == "episode_deadline_elapsed"


def test_prior_plan_deadline_cannot_be_backdated_onto_new_draw() -> None:
    observation = _lsr_observation()
    brain = _brain()
    key = "liquidity_sweep_reversal:short"

    belief = brain.update(observation)
    planned = belief.hypotheses[key]
    assert planned.plan is not None
    # A market episode can outlive an earlier frozen order-entry deadline.
    # Preserve that valid distinction without relaxing either clock.
    longer_episode_deadline = planned.plan.deadline + pd.Timedelta(minutes=60)
    hypotheses = dict(belief.hypotheses)
    hypotheses[key] = replace(
        planned,
        episode_deadline=longer_episode_deadline,
    )
    brain._belief = replace(belief, hypotheses=hypotheses)

    expired_asof = planned.plan.deadline
    later_plan = _advance_observation(observation, expired_asof)
    expired = brain.update(later_plan).hypotheses[key]

    assert expired.episode_deadline == longer_episode_deadline
    assert expired.plan is None
    assert expired.phase is not PlaybookPhase.EXECUTABLE


def test_execution_reality_does_not_change_market_dimensions_or_phase() -> None:
    observation = _lsr_observation()
    missing_execution = replace(
        observation.execution,
        source="missing",
        spread_points=20.0,
        expected_round_trip_cost_points=40.0,
        minutes_to_deadline=1,
        fillability=0.0,
        anomalies=(
            "spread_missing_used_one_tick",
            "deadline_missing",
            "insufficient_top_of_book_depth",
        ),
    )

    authoritative = _brain().update(observation).hypotheses[
        "liquidity_sweep_reversal:short"
    ]
    unavailable = _brain().update(
        replace(observation, execution=missing_execution)
    ).hypotheses["liquidity_sweep_reversal:short"]

    assert unavailable.phase is authoritative.phase
    assert unavailable.sequence == authoritative.sequence
    assert unavailable.thesis_strength == authoritative.thesis_strength
    assert unavailable.sequence_progress == authoritative.sequence_progress
    assert unavailable.location_quality == authoritative.location_quality
    assert unavailable.entry_readiness == authoritative.entry_readiness
    assert unavailable.delivery_quality == authoritative.delivery_quality
    assert unavailable.uncertainty == authoritative.uncertainty
    assert unavailable.raw_quality_dimensions == authoritative.raw_quality_dimensions
    assert unavailable.evidence_group_scores["execution"] == 0.0
    assert authoritative.evidence_group_scores["execution"] > 0.0


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
    wrong_side_zone = replace_market_observation(
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
