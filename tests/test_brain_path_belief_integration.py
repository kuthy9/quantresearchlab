from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

import smc_trader.playbooks as playbooks_module
from smc_trader.decision import UtilityDecisionLayer
from smc_trader.dol_probability import (
    DOLProbabilityModelArtifact,
    load_dol_probability_protocol,
)
from smc_trader.dol_ranking import load_dol_ranking_protocol
from smc_trader.engine import ContinuousSMCEngine
from smc_trader.model import (
    AuthorityLayer,
    CandleStructureState,
    ContextThesisState,
    DeliveryObstruction,
    Direction,
    DirectionalObstructionView,
    EventKind,
    EventOrigin,
    GlobalMarketContext,
    MarketBelief,
    MarketEvent,
    MarketMode,
    OpenMarketThesis,
    ScaleRelation,
    ScaleRelationState,
    SMC_SEMANTIC_VERSION,
    ThesisEvidenceState,
    Timeframe,
    to_primitive,
)
from smc_trader.path_belief import (
    PathKind,
    PathStatus,
    load_path_belief_protocol,
)
from smc_trader.playbooks import PlaybookBrain
from smc_trader.scene_graph import TemporalMarketSceneGraph
from smc_trader.signal_policy import (
    AdmittedDOLCalibrationArtifact,
    AdmittedPathLikelihoodArtifact,
    SetupDeliveryModel,
    SetupFamily,
    SignalArtifactPins,
    TargetBeforeInvalidationArtifact,
    load_signal_policy_protocol,
)

from .helpers import flat_account, market_observation
from .test_global_market_context import (
    _add_authority_and_draws,
    _delta,
    _node,
)
from .test_signal_policy_trade_intent import _typed_setup


TZ = "America/New_York"
ROOT = Path(__file__).resolve().parents[1]


def _clock(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz=TZ)


def _canonical_semantic_event(event: MarketEvent) -> MarketEvent:
    identity_payload = {
        "semantic_version": event.semantic_version,
        "semantic_type": event.kind.value,
        "event_time": event.event_time,
        "known_at": event.known_at,
        "timeframe": event.timeframe.value,
        "side": event.side,
        "price": event.price,
        "direction": None if event.direction is None else event.direction.value,
        "source_event_ids": event.source_event_ids,
        "source_data_ids": event.source_data_ids,
        "source_entity_ids": event.source_entity_ids,
        "context_event_ids": event.context_event_ids,
        "origin": event.origin.value,
        "evidence": event.evidence,
        "zone": event.zone,
    }
    event_id = hashlib.sha256(
        json.dumps(
            to_primitive(identity_payload),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    return replace(event, event_id=event_id)


def _candle_geometry(
    asof: pd.Timestamp,
    *,
    real_completed: bool,
) -> CandleStructureState:
    return CandleStructureState(
        timeframe=Timeframe.M1,
        start=asof - pd.Timedelta(minutes=1),
        observed_at=asof,
        range_points=2.0,
        body_points=1.0,
        upper_wick_points=0.5,
        lower_wick_points=0.5,
        body_ratio=0.5,
        upper_wick_ratio=0.25,
        lower_wick_ratio=0.25,
        close_location=0.75,
        direction=1,
        real_completed=real_completed,
        zero_range=False,
        body_class="normal",
        range_class="normal",
        dominant_wick="balanced",
        close_class="near_high",
        anomalies=(
            ()
            if real_completed
            else ("synthetic_or_partial_completed_candle",)
        ),
    )


def _event(
    event_id: str,
    entity_id: str,
    asof: pd.Timestamp,
) -> MarketEvent:
    return MarketEvent(
        event_id=event_id,
        kind=EventKind.STRUCTURE_STATE,
        observed_at=asof,
        timeframe=Timeframe.H4,
        side=None,
        price=100.0,
        strength=1.0,
        entity_id=entity_id,
        lifecycle="confirmed",
        formed_at=asof,
        confirmed_at=asof,
    )


def _observation(
    asof: pd.Timestamp,
    *,
    real_completed: bool,
    events: tuple[MarketEvent, ...] = (),
):
    base = market_observation(asof=asof, price=100.0)
    m1 = replace(
        base.frame(Timeframe.M1),
        cutoff=asof,
        candle_structure=_candle_geometry(
            asof,
            real_completed=real_completed,
        ),
    )
    return replace(
        base,
        frames={
            timeframe: replace(frame, cutoff=asof)
            for timeframe, frame in base.frames.items()
        }
        | {Timeframe.M1: m1},
        recent_events=(*base.recent_events, *events),
    )


def _relation_states(
    authority_structure_id: str,
) -> dict[str, ScaleRelationState]:
    return {
        timeframe.value: ScaleRelationState(
            timeframe=timeframe,
            relation=(
                ScaleRelation.ALIGNED
                if timeframe is Timeframe.H4
                else ScaleRelation.UNKNOWN
            ),
            direction=(Direction.LONG if timeframe is Timeframe.H4 else None),
            authority_layer_id=(
                authority_structure_id
                if timeframe is Timeframe.H4
                else None
            ),
            evidence_ids=(
                (authority_structure_id,)
                if timeframe is Timeframe.H4
                else ()
            ),
            evidence_kind=("structure" if timeframe is Timeframe.H4 else None),
            structural_scope=(
                "external" if timeframe is Timeframe.H4 else None
            ),
            acceptance_state=(
                "confirmed" if timeframe is Timeframe.H4 else None
            ),
            since=None,
            age_bars=0,
            graph_connected=timeframe is Timeframe.H4,
            ambiguous=False,
        )
        for timeframe in Timeframe
    }


def _thesis(
    *,
    root_id: str,
    draw_id: str,
    direction: Direction,
    relation: str,
    mechanism: str,
    epoch: str,
    asof: pd.Timestamp,
) -> OpenMarketThesis:
    evidence = ThesisEvidenceState(
        lifecycle="active",
        revision_id=f"revision:{root_id}",
        changed_at=asof,
        supporting_event_ids=(root_id,),
        new_supporting_event_ids=(root_id,),
    )
    return OpenMarketThesis(
        thesis_id=f"thesis:{root_id}",
        root_id=root_id,
        market_epoch_id=epoch,
        formed_at=asof,
        updated_at=asof,
        direction=direction,
        source_timeframe=Timeframe.M1,
        structural_scale="internal",
        mechanism=mechanism,
        authority_relation=relation,
        authority_source_ids=("authority-structure",),
        mechanism_event_ids=(root_id,),
        draw_candidate_ids=(draw_id,),
        evidence_state=evidence,
    )


def _final_context(
    *,
    asof: pd.Timestamp,
    scene_revision_id: str,
    epoch: str = "epoch:0",
    authority_structure_id: str = "authority-structure",
) -> GlobalMarketContext:
    hard = DeliveryObstruction(
        obstruction_id="hard-obstacle",
        timeframe=Timeframe.M5,
        direction=Direction.SHORT,
        side="above",
        lower_bound=101.0,
        upper_bound=101.25,
        hard=True,
        source_kind="accepted_bos",
        source_ids=("hard-obstacle",),
        structural_scope="external",
        acceptance_state="accepted",
    )
    return GlobalMarketContext(
        updated_at=asof,
        scene_revision_id=scene_revision_id,
        market_epoch_id=epoch,
        authority_stack=(
            AuthorityLayer(
                timeframe=Timeframe.H4,
                direction=Direction.LONG,
                structure_id=authority_structure_id,
                confirmed_at=asof,
                protected_level_id=None,
                structural_scope="external",
                acceptance_state="confirmed",
                status="intact",
                source_ids=(authority_structure_id,),
            ),
        ),
        market_mode=MarketMode.DIRECTIONAL,
        scale_relation_details=_relation_states(authority_structure_id),
        external_draw_candidates={
            "above": ("above-level",),
            "below": ("below-level",),
        },
        obstruction_views={
            Direction.LONG.value: DirectionalObstructionView(
                direction=Direction.LONG,
                nearest_draw_id="above-level",
                nearest_draw_price=103.0,
                hard_barriers=(hard,),
                soft_frictions=(),
            ),
            Direction.SHORT.value: DirectionalObstructionView(
                direction=Direction.SHORT,
                nearest_draw_id="below-level",
                nearest_draw_price=98.0,
                hard_barriers=(),
                soft_frictions=(),
            ),
        },
        material_conflicts=(),
        unknown_evidence=(),
        ambiguous_evidence=(),
        dislocations_by_scale={timeframe.value: () for timeframe in Timeframe},
        open_market_theses=(
            _thesis(
                root_id="root-long",
                draw_id="above-level",
                direction=Direction.LONG,
                relation="aligned",
                mechanism="directional_displacement",
                epoch=epoch,
                asof=asof,
            ),
            _thesis(
                root_id="root-short",
                draw_id="below-level",
                direction=Direction.SHORT,
                relation="local_countertrend",
                mechanism="liquidity_sweep",
                epoch=epoch,
                asof=asof,
            ),
        ),
    )


def _heartbeat(
    graph: TemporalMarketSceneGraph,
    asof: pd.Timestamp,
) -> tuple[str, ...]:
    node = graph.add_node(
        _node(
            f"heartbeat:{asof.isoformat()}",
            "candle_structure",
            Timeframe.M1,
            asof,
            direction=None,
        )
    )
    graph._last_asof = asof
    return (node.node_id,)


def _fitted_dol_artifact(path_protocol=None):
    protocol = load_dol_probability_protocol("configs/dol_probability.json")
    path_protocol = path_protocol or load_path_belief_protocol(
        "configs/path_hypotheses.json"
    )
    semantic = {
        "schema_version": 2,
        "artifact_id": "dol-artifact:brain-integration-test-v1",
        "model_version": protocol.model_version,
        "protocol_fingerprint": protocol.fingerprint,
        "ranking_protocol_fingerprint": protocol.ranking_protocol_fingerprint,
        "source_path_protocol_fingerprint": path_protocol.fingerprint,
        "source_path_model_version": path_protocol.model_version,
        "fit_status": "fitted",
        "admission_status": "admitted",
        "calibration_status": "fitted_admitted",
        "authority": "shadow_only",
        "action_authority": False,
        "parameters": protocol.development_model.payload(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            semantic,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return DOLProbabilityModelArtifact(
        schema_version=semantic["schema_version"],
        artifact_id=semantic["artifact_id"],
        model_version=semantic["model_version"],
        protocol_fingerprint=semantic["protocol_fingerprint"],
        ranking_protocol_fingerprint=semantic[
            "ranking_protocol_fingerprint"
        ],
        source_path_protocol_fingerprint=semantic[
            "source_path_protocol_fingerprint"
        ],
        source_path_model_version=semantic["source_path_model_version"],
        fit_status=semantic["fit_status"],
        admission_status=semantic["admission_status"],
        calibration_status=semantic["calibration_status"],
        authority=semantic["authority"],
        action_authority=False,
        parameters=protocol.development_model,
        fingerprint=fingerprint,
    )


def _admitted_path_protocol_file(tmp_path: Path) -> Path:
    payload = json.loads(
        (ROOT / "configs/path_hypotheses.json").read_text(encoding="utf-8")
    )
    payload["model_version"] = "path-bayesian-brain-test-fitted-v1"
    payload["model_admission_status"] = "diagnostic_likelihood_admitted"
    payload["likelihood_artifact_status"] = "diagnostic_admitted"
    conditional = {path.value: 0.5 for path in PathKind}
    for rule in payload["evidence_rules"].values():
        rule.pop("log_likelihood_increment")
        rule["conditional_likelihood"] = conditional
    target = tmp_path / "path-hypotheses-admitted.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def test_public_brain_shadow_path_is_same_clock_and_decision_neutral() -> None:
    asof = _clock("2025-01-06 10:00")
    authority_event = _event("event:authority", "h4-up", asof)
    observation = _observation(
        asof,
        real_completed=True,
        events=(authority_event,),
    )
    graph = TemporalMarketSceneGraph()
    nodes = _add_authority_and_draws(graph, asof)
    graph._refresh_current_path_block_edges(observation)
    graph._last_asof = asof
    brain = PlaybookBrain()

    belief = brain.update(
        observation,
        scene_graph=graph,
        scene_delta=_delta(
            graph,
            asof,
            nodes=tuple(node.node_id for node in nodes),
        ),
    )

    state = belief.path_competition_state
    assert state is not None
    assert state.formed_at == asof == state.asof
    assert state.real_completed_bar_count == 1
    assert sum(member.probability for member in state.members) == pytest.approx(1.0)
    assert belief.path_protocol_status == "development_unvalidated"
    assert belief.path_authority == "shadow_only"
    assert len(belief.path_update_records_this_clock) == 1
    initial = belief.path_update_records_this_clock[0]
    assert initial.initialization
    assert initial.from_asof == initial.asof == asof
    assert {
        source_id
        for contribution in initial.applied_contributions
        for source_id in contribution.source_event_ids
    } == set()
    assert state.evidence_ledger == ()
    assert not brain.hypothesis_manager.action_authority
    assert set(belief.dol_rankings) == {"long", "short"}
    assert belief.dol_probabilities == {}
    assert belief.trade_intents == {}
    with pytest.raises(TypeError, match="immutable"):
        belief.dol_probabilities["long"] = None
    assert belief.shadow_signal_rejections["__admission__"] == (
        "dol_probability_model_artifact_missing",
        "path_likelihood_artifact_missing",
        "dol_calibration_artifact_missing",
        "outcome_model_artifact_missing",
        "signal_artifact_pins_missing",
    )

    baseline = replace(
        belief,
        path_competition_state=None,
        path_update_records_this_clock=(),
        dol_rankings={},
        dol_candidate_exclusions={},
    )
    assert belief.ranked() == baseline.ranked()
    layer = UtilityDecisionLayer()
    assert layer.decide(observation, belief, flat_account()) == layer.decide(
        observation,
        baseline,
        flat_account(),
    )


def test_bayesian_path_protocol_cannot_self_admit_without_external_pins(
    tmp_path: Path,
) -> None:
    path_source = _admitted_path_protocol_file(tmp_path)

    with pytest.raises(
        ValueError,
        match="require a separately pinned artifact set",
    ):
        PlaybookBrain(path_hypotheses_protocol=path_source)


def test_fitted_pinned_brain_fixture_builds_deterministic_shadow_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asof = _clock("2026-08-21 09:30")
    path_source = _admitted_path_protocol_file(tmp_path)
    path_protocol = load_path_belief_protocol(path_source)
    ranking_protocol = load_dol_ranking_protocol(path_source)
    probability_protocol = load_dol_probability_protocol(
        "configs/dol_probability.json"
    )
    assert (
        probability_protocol.ranking_protocol_fingerprint
        == ranking_protocol.fingerprint
    )
    dol_model = _fitted_dol_artifact(path_protocol)
    signal_policy = load_signal_policy_protocol("configs/signal_policy.json")
    path_artifact = AdmittedPathLikelihoodArtifact(
        protocol_id="path-admission:brain-test:v1",
        model_id="path-likelihood:brain-test",
        model_version="path-likelihood-brain-test-v1",
        calibration_id="path-calibration:brain-test:v1",
        source_dataset_id="dataset:brain-test:train",
        coverage_id="path-coverage:brain-test",
        source_path_protocol_fingerprint=path_protocol.fingerprint,
        source_path_model_version=path_protocol.model_version,
        trained_through=asof - pd.Timedelta(days=2),
        valid_from=asof - pd.Timedelta(days=1),
        expires_at=asof + pd.Timedelta(hours=1),
    )
    dol_artifact = AdmittedDOLCalibrationArtifact(
        protocol_id="dol-admission:brain-test:v1",
        model_id="dol-calibration:brain-test",
        model_version="dol-calibration-brain-test-v1",
        calibration_id="dol-calibration-id:brain-test:v1",
        source_dataset_id="dataset:brain-test:train",
        coverage_id="dol-coverage:brain-test",
        source_dol_protocol_fingerprint=probability_protocol.fingerprint,
        source_dol_model_version=probability_protocol.model_version,
        source_path_protocol_fingerprint=path_protocol.fingerprint,
        source_path_model_version=path_protocol.model_version,
        source_dol_model_fingerprint=dol_model.fingerprint,
        trained_through=asof - pd.Timedelta(days=2),
        valid_from=asof - pd.Timedelta(days=1),
        expires_at=asof + pd.Timedelta(hours=1),
    )
    outcome_artifact = TargetBeforeInvalidationArtifact(
        protocol_id="delivery-admission:brain-test:v1",
        model_id="target-before-invalidation:brain-test",
        model_version="target-before-invalidation-brain-test-v1",
        calibration_id="delivery-calibration:brain-test:v1",
        source_dataset_id="dataset:brain-test:train",
        coverage_id="delivery-coverage:brain-test",
        path_likelihood_artifact_id=path_artifact.artifact_id,
        dol_calibration_artifact_id=dol_artifact.artifact_id,
        signal_policy_fingerprint=signal_policy.fingerprint,
        trained_through=asof - pd.Timedelta(days=2),
        valid_from=asof - pd.Timedelta(days=1),
        expires_at=asof + pd.Timedelta(hours=1),
        setup_models=(
            SetupDeliveryModel(
                setup_family=SetupFamily.DFP,
                intercept=3.0,
                path_logit_coefficient=0.0,
                dol_logit_coefficient=0.0,
                half_life_real_completed_bars=10,
            ),
            SetupDeliveryModel(
                setup_family=SetupFamily.LSR,
                intercept=3.0,
                path_logit_coefficient=0.0,
                dol_logit_coefficient=0.0,
                half_life_real_completed_bars=8,
            ),
        ),
        minimum_supported_coverage=0.9,
    )
    pins = SignalArtifactPins(
        signal_policy_fingerprint=signal_policy.fingerprint,
        path_protocol_fingerprint=path_protocol.fingerprint,
        path_likelihood_artifact_id=path_artifact.artifact_id,
        dol_calibration_artifact_id=dol_artifact.artifact_id,
        outcome_model_artifact_id=outcome_artifact.artifact_id,
        dol_probability_model_fingerprint=dol_model.fingerprint,
    )
    with pytest.raises(
        ValueError,
        match="shadow artifact admission binding is stale",
    ):
        PlaybookBrain(
            path_hypotheses_protocol=path_source,
            dol_probability_model_artifact=dol_model,
            path_likelihood_artifact=path_artifact,
            dol_calibration_artifact=dol_artifact,
            outcome_model_artifact=outcome_artifact,
        )
    brain = PlaybookBrain(
        path_hypotheses_protocol=path_source,
        dol_probability_model_artifact=dol_model,
        path_likelihood_artifact=path_artifact,
        dol_calibration_artifact=dol_artifact,
        outcome_model_artifact=outcome_artifact,
        signal_artifact_pins=pins,
    )
    scope_start = asof - pd.Timedelta(minutes=1)
    authority_event = _event("event:authority", "h4-up", scope_start)
    initial_observation = _observation(
        scope_start,
        real_completed=True,
        events=(authority_event,),
    )
    graph = TemporalMarketSceneGraph()
    nodes = _add_authority_and_draws(graph, scope_start)
    graph._refresh_current_path_block_edges(initial_observation)
    graph._last_asof = scope_start
    monkeypatch.setattr(
        playbooks_module,
        "_finalize_global_context",
        lambda _context, current_observation, current_graph, _hypotheses: (
            _final_context(
                asof=current_observation.asof,
                scene_revision_id=current_graph.revision_id,
                epoch=current_graph._market_epoch_id,
            )
        ),
    )
    brain.update(
        initial_observation,
        scene_graph=graph,
        scene_delta=_delta(
            graph,
            scope_start,
            nodes=tuple(node.node_id for node in nodes),
        ),
    )
    observation = _observation(
        asof,
        real_completed=True,
        events=(authority_event,),
    )
    observation = replace(
        observation,
        execution=replace(
            observation.execution,
            expected_round_trip_cost_points=0.1,
        ),
    )
    base = brain.update(
        observation,
        scene_graph=graph,
        scene_delta=_delta(graph, asof, nodes=_heartbeat(graph, asof)),
    )
    state = base.path_competition_state
    assert state is not None
    assert state.formed_at == scope_start
    assert state.real_completed_bar_count == 2
    probability = base.dol_probabilities[Direction.LONG.value]
    draw = next(
        item
        for item in probability.ranked_candidates
        if PathKind.CONTINUATION in item.supported_paths
    )
    candidate, episode = _typed_setup()
    assert candidate.plan is not None
    target = replace(
        candidate.plan.targets[0],
        level_id=draw.candidate_id,
        price=draw.target_price,
    )
    plan = replace(
        candidate.plan,
        targets=(target,),
        selected_draw_id=draw.candidate_id,
    )
    candidate = replace(
        candidate,
        deliverable_targets=(target,),
        plan=plan,
        phase_started_at=asof,
    )
    episode = replace(episode, plan=plan, updated_at=asof)
    context = ContextThesisState(
        context_thesis_id=candidate.context_thesis_id,
        market_epoch_id=state.market_epoch_id,
        direction=candidate.direction,
        authority_ids=(state.authority_structure_id,),
        context_draw=target,
        structural_invalidation=candidate.invalidation,
        supporting_event_ids=(candidate.initiating_event_id,),
        opposing_event_ids=(),
        lifecycle="active",
        formed_at=asof,
        updated_at=asof,
        thesis_deadline=candidate.thesis_deadline,
        child_episode_ids=(episode.episode_id,),
    )
    focus = (
        None
        if base.focus_state is None
        else replace(base.focus_state, hypothesis_id=None)
    )
    injected = replace(
        base,
        hypotheses={
            key: replace(value, summary_source_candidate_id=None)
            for key, value in base.hypotheses.items()
        },
        context_hypotheses={},
        dominant_hypothesis_id=None,
        competing_hypothesis_ids=(),
        focus_state=focus,
        thesis_candidates={candidate.candidate_id: candidate},
        retained_episode_candidates={},
        position_management_candidates={},
        context_theses={context.context_thesis_id: context},
        entry_episodes={candidate.candidate_id: episode},
        signal_assessments={},
        trade_intents={},
        shadow_signal_rejections={},
    )

    future_target = replace(
        target,
        confirmed_at=asof + pd.Timedelta(minutes=1),
    )
    future_plan = replace(plan, targets=(future_target,))
    future_candidate = replace(
        candidate,
        deliverable_targets=(future_target,),
        plan=future_plan,
    )
    future_episode = replace(episode, plan=future_plan)
    with pytest.raises(
        ValueError,
        match="root-specific thesis candidate mapping is invalid",
    ):
        replace(
            injected,
            thesis_candidates={candidate.candidate_id: future_candidate},
            entry_episodes={candidate.candidate_id: future_episode},
        )

    assessed = brain._project_shadow_signal_assessments(observation, injected)
    assessment = assessed.signal_assessments[candidate.candidate_id]
    assert assessment.eligible
    assert assessment.age_real_completed_bars == 0
    assert assessment.half_life_real_completed_bars == 10
    next_asof = asof + pd.Timedelta(minutes=1)
    next_observation = _observation(
        next_asof,
        real_completed=True,
        events=(authority_event,),
    )
    next_observation = replace(
        next_observation,
        execution=replace(
            next_observation.execution,
            expected_round_trip_cost_points=0.1,
        ),
    )
    _heartbeat(graph, next_asof)
    next_context = _final_context(
        asof=next_asof,
        scene_revision_id=graph.revision_id,
        epoch=graph._market_epoch_id,
    )
    next_state, next_records, next_rankings, next_exclusions = (
        brain._update_shadow_path_diagnostics(
            next_observation,
            next_context,
            assessed.context_theses,
            assessed.thesis_candidates,
            assessed,
        )
    )
    assert next_state is not None
    next_focus = (
        None
        if assessed.focus_state is None
        else replace(assessed.focus_state, asof=next_asof)
    )
    next_belief = replace(
        assessed,
        asof=next_asof,
        focus_state=next_focus,
        scene_revision_id=next_context.scene_revision_id,
        global_context=next_context,
        path_competition_state=next_state,
        path_update_records_this_clock=next_records,
        dol_rankings=next_rankings,
        dol_probabilities=brain._dol_probability_results,
        dol_candidate_exclusions=next_exclusions,
        signal_assessments={},
        trade_intents={},
        shadow_signal_rejections={},
    )
    progressed = brain._project_shadow_signal_assessments(
        next_observation,
        next_belief,
    )
    progressed_assessment = progressed.signal_assessments[candidate.candidate_id]
    assert progressed_assessment.eligible
    assert progressed_assessment.age_real_completed_bars == 1

    brain._belief = progressed
    first = brain.project_shadow_trade_intents(progressed, flat_account())
    brain._belief = progressed
    replay = brain.project_shadow_trade_intents(progressed, flat_account())

    assert first.trade_intents == replay.trade_intents
    intent = first.trade_intents[candidate.candidate_id]
    assert not intent.submission_allowed
    layer = UtilityDecisionLayer()
    assert layer.decide(
        next_observation,
        progressed,
        flat_account(),
    ) == layer.decide(
        next_observation,
        first,
        flat_account(),
    )
    restored = pickle.loads(pickle.dumps(brain))
    assert restored._signal_real_completed_bar_anchors == (
        brain._signal_real_completed_bar_anchors
    )
    brain.reset()
    assert brain._signal_real_completed_bar_anchors == {}


def test_shadow_path_synthetic_decay_reset_and_pickle_replay() -> None:
    t0 = _clock("2025-01-06 10:00")
    authority_event = _event("event:authority", "h4-up", t0)
    observation0 = _observation(
        t0,
        real_completed=True,
        events=(authority_event,),
    )
    graph = TemporalMarketSceneGraph()
    nodes = _add_authority_and_draws(graph, t0)
    graph._last_asof = t0
    brain = PlaybookBrain()
    first = brain.update(
        observation0,
        scene_graph=graph,
        scene_delta=_delta(
            graph,
            t0,
            nodes=tuple(node.node_id for node in nodes),
        ),
    )
    assert first.path_competition_state is not None
    original_id = first.path_competition_state.competition_set_id
    restored = pickle.loads(pickle.dumps(brain))

    t1 = t0 + pd.Timedelta(minutes=1)
    observation1 = _observation(
        t1,
        real_completed=False,
        events=(authority_event,),
    )
    delta1 = _delta(graph, t1, nodes=_heartbeat(graph, t1))
    uninterrupted = brain.update(
        observation1,
        scene_graph=graph,
        scene_delta=delta1,
    )
    replayed = restored.update(
        observation1,
        scene_graph=graph,
        scene_delta=delta1,
    )
    assert replayed.path_competition_state == uninterrupted.path_competition_state
    assert (
        replayed.path_update_records_this_clock
        == uninterrupted.path_update_records_this_clock
    )
    assert replayed.dol_rankings == uninterrupted.dol_rankings
    assert replayed.dol_probabilities == uninterrupted.dol_probabilities
    assert replayed.signal_assessments == uninterrupted.signal_assessments
    assert replayed.trade_intents == uninterrupted.trade_intents
    synthetic_record = uninterrupted.path_update_records_this_clock[0]
    assert not synthetic_record.decay_applied
    assert synthetic_record.applied_contributions == ()
    assert uninterrupted.path_competition_state.real_completed_bar_count == 1

    t2 = t1 + pd.Timedelta(minutes=1)
    observation2 = _observation(
        t2,
        real_completed=True,
        events=(authority_event,),
    )
    second_real = brain.update(
        observation2,
        scene_graph=graph,
        scene_delta=_delta(graph, t2, nodes=_heartbeat(graph, t2)),
    )
    assert not second_real.path_update_records_this_clock[0].decay_applied
    assert second_real.path_competition_state.real_completed_bar_count == 2

    brain._signal_real_completed_bar_anchors["candidate:test"] = (
        second_real.path_competition_state.competition_set_id,
        "episode:test",
        t0,
        1,
    )
    brain.reset()
    assert brain.current is None
    assert brain._path_competition_state is None
    assert brain._path_scope_key is None
    assert brain._path_seen_evidence_tokens == set()
    assert brain._dol_probability_results == {}
    assert brain._signal_real_completed_bar_anchors == {}
    t3 = t2 + pd.Timedelta(minutes=1)
    reset_belief = brain.update(
        _observation(t3, real_completed=True, events=(authority_event,)),
        scene_graph=graph,
        scene_delta=_delta(graph, t3, nodes=_heartbeat(graph, t3)),
    )
    assert reset_belief.path_competition_state is not None
    assert reset_belief.path_competition_state.competition_set_id != original_id
    assert reset_belief.path_competition_state.real_completed_bar_count == 1


def test_public_brain_exposes_exact_dol_rankings_and_obstacles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asof = _clock("2025-01-06 10:00")
    events = (
        _event("event:authority", "authority-structure", asof),
        _event("event:root-long", "root-long", asof),
        _event("event:root-short", "root-short", asof),
        _event("event:hard-obstacle", "hard-obstacle", asof),
    )
    observation = _observation(asof, real_completed=True, events=events)
    graph = TemporalMarketSceneGraph()
    nodes = _add_authority_and_draws(graph, asof)
    graph._last_asof = asof
    final_context = _final_context(
        asof=asof,
        scene_revision_id=graph.revision_id,
        epoch=graph._market_epoch_id,
    )
    monkeypatch.setattr(
        playbooks_module,
        "_finalize_global_context",
        lambda *_args, **_kwargs: final_context,
    )

    belief = PlaybookBrain().update(
        observation,
        scene_graph=graph,
        scene_delta=_delta(
            graph,
            asof,
            nodes=tuple(node.node_id for node in nodes),
        ),
    )

    state = belief.path_competition_state
    assert state is not None
    assert {member.path for member in state.members} == set(PathKind)
    assert sum(member.probability for member in state.members) == pytest.approx(1.0)
    long_result = belief.dol_rankings[Direction.LONG.value]
    short_result = belief.dol_rankings[Direction.SHORT.value]
    assert [item.candidate_id for item in long_result.ranked_candidates] == [
        "above-level"
    ]
    assert [item.candidate_id for item in short_result.ranked_candidates] == [
        "below-level"
    ]
    assert long_result.ranked_candidates[0].hard_obstacle_ids == (
        "hard-obstacle",
    )
    assert long_result.ranked_candidates[0].path is PathKind.CONTINUATION
    assert short_result.ranked_candidates[0].path is PathKind.DEEPER_RETRACEMENT
    assert sum(
        item.normalized_diagnostic_weight
        for item in long_result.ranked_candidates
    ) == pytest.approx(1.0)
    assert sum(
        item.normalized_diagnostic_weight
        for item in short_result.ranked_candidates
    ) == pytest.approx(1.0)
    assert belief.dol_candidate_exclusions == {"long": (), "short": ()}

    contributed_sources = {
        source_id
        for record in belief.path_update_records_this_clock
        for contribution in record.applied_contributions
        for source_id in contribution.source_event_ids
    }
    assert contributed_sources.issubset({event.event_id for event in events})
    assert contributed_sources == set()


def test_non_event_fact_identities_never_enter_path_event_ancestry() -> None:
    asof = _clock("2025-01-06 10:00")
    observation = _observation(asof, real_completed=True)
    context = _final_context(
        asof=asof,
        scene_revision_id="revision:test",
    )
    state, records, _rankings, _exclusions = (
        PlaybookBrain()._update_shadow_path_diagnostics(
            observation,
            context,
            {},
            {},
            None,
        )
    )

    assert state is not None
    assert len(records) == 1
    assert records[0].applied_contributions == ()
    assert state.applied_contribution_ids == ()


def test_only_phase6_canonical_events_enter_source_only_path_ledger() -> None:
    asof = _clock("2025-01-06 10:00")
    acceptance = _canonical_semantic_event(MarketEvent(
        event_id="event:acceptance:phase6",
        kind=EventKind.ACCEPTANCE_CONFIRMED,
        observed_at=asof,
        timeframe=Timeframe.M5,
        side="above",
        price=101.0,
        strength=0.75,
        direction=Direction.LONG,
        event_time=asof - pd.Timedelta(minutes=1),
        known_at=asof,
        evidence={
            "crossing_generation_id": "crossing-generation:1",
            "level_id": "level:1",
        },
        origin=EventOrigin.SEMANTIC_ATOMIC,
    ))
    displacement = _canonical_semantic_event(MarketEvent(
        event_id="event:displacement:phase6",
        kind=EventKind.DISPLACEMENT_OBSERVED,
        observed_at=asof,
        timeframe=Timeframe.M5,
        side="above",
        price=None,
        strength=0.8,
        direction=Direction.LONG,
        event_time=asof - pd.Timedelta(minutes=1),
        known_at=asof,
        evidence={
            "displacement_id": "displacement:1",
            "lifecycle": "active",
        },
        origin=EventOrigin.SEMANTIC_ATOMIC,
    ))
    legacy_transport = replace(
        acceptance,
        event_id="event:acceptance:legacy-transport",
        origin=EventOrigin.LEGACY_TRANSPORT,
    )
    non_m5 = _canonical_semantic_event(replace(
        acceptance,
        event_id="event:acceptance:h1-out-of-scope",
        timeframe=Timeframe.H1,
    ))
    wrong_semantic_version = _canonical_semantic_event(replace(
        acceptance,
        event_id="event:acceptance:wrong-semantic-version",
        semantic_version="bogus-v99",
    ))
    forged_identity = replace(
        acceptance,
        event_id="event:acceptance:not-canonical",
        semantic_version=SMC_SEMANTIC_VERSION,
    )
    observation = _observation(
        asof,
        real_completed=True,
        events=(
            acceptance,
            displacement,
            legacy_transport,
            non_m5,
            wrong_semantic_version,
            forged_identity,
        ),
    )
    graph = TemporalMarketSceneGraph()
    nodes = _add_authority_and_draws(graph, asof)
    graph._last_asof = asof

    brain = PlaybookBrain()
    belief = brain.update(
        observation,
        scene_graph=graph,
        scene_delta=_delta(
            graph,
            asof,
            nodes=tuple(node.node_id for node in nodes),
        ),
    )

    state = belief.path_competition_state
    assert state is not None
    assert len(state.evidence_ledger) == 2
    by_rule = {item.rule_id: item for item in state.evidence_ledger}
    assert set(by_rule) == {
        "acceptance_continuation",
        "displacement_impact",
    }
    assert by_rule["acceptance_continuation"].source_event_ids == (
        acceptance.event_id,
    )
    assert by_rule["acceptance_continuation"].correlation_key == (
        "semantic-entity:crossing-generation:1"
    )
    assert by_rule["displacement_impact"].source_event_ids == (
        displacement.event_id,
    )
    assert by_rule["displacement_impact"].correlation_key == (
        "semantic-entity:displacement:1"
    )

    unresolved_cluster = "unresolved-dependency-cluster:test-competition-set"
    admitted_specs = playbooks_module._shadow_path_contribution_specs(
        observation,
        _final_context(
            asof=asof,
            scene_revision_id=graph.revision_id,
            epoch=graph._market_epoch_id,
        ),
        seen_tokens=(),
        unresolved_dependency_cluster=unresolved_cluster,
    )
    assert len(admitted_specs) == 2
    assert {item[3] for item in admitted_specs} == {unresolved_cluster}
    assert all(
        not {
            legacy_transport.event_id,
            non_m5.event_id,
            wrong_semantic_version.event_id,
            forged_identity.event_id,
        }.intersection(
            item.source_event_ids
        )
        for item in state.evidence_ledger
    )
    assert belief.path_update_records_this_clock[0].evidence_admission_only
    assert all(
        member.probability == pytest.approx(1.0 / 6.0)
        for member in state.members
    )


def test_local_entry_episode_invalidation_never_terminalizes_global_path() -> None:
    t0 = _clock("2025-01-06 10:00")
    context0 = _final_context(
        asof=t0,
        scene_revision_id="revision:local-invalidation",
    )
    brain = PlaybookBrain()
    first, _, _, _ = brain._update_shadow_path_diagnostics(
        _observation(t0, real_completed=True),
        context0,
        {},
        {},
        None,
    )
    assert first is not None

    t1 = t0 + pd.Timedelta(minutes=1)
    local_terminal = SimpleNamespace(
        phase="invalidated",
        terminal_at=t1,
        terminal_source_ids=("event:local-entry-invalidated",),
        playbook="displacement_first_pullback",
    )
    second, records, _, _ = brain._update_shadow_path_diagnostics(
        _observation(t1, real_completed=True),
        replace(context0, updated_at=t1),
        {},
        {"entry-episode:local": local_terminal},
        None,
    )

    assert second is not None
    assert second.competition_set_id == first.competition_set_id
    assert all(member.status is PathStatus.ACTIVE for member in second.members)
    assert records[0].applied_terminal_events == ()


def test_new_session_first_clock_is_not_consumed_only_expiring_old_scope() -> None:
    old_clock = _clock("2025-01-06 16:59")
    brain = PlaybookBrain()
    old_state, _, _, _ = brain._update_shadow_path_diagnostics(
        _observation(old_clock, real_completed=True),
        _final_context(
            asof=old_clock,
            scene_revision_id="revision:old-session",
        ),
        {},
        {},
        None,
    )
    assert old_state is not None

    new_clock = _clock("2025-01-06 18:00")
    acceptance = _canonical_semantic_event(MarketEvent(
        event_id="event:acceptance:new-session-open",
        kind=EventKind.ACCEPTANCE_CONFIRMED,
        observed_at=new_clock,
        timeframe=Timeframe.M5,
        side="above",
        price=101.0,
        strength=0.75,
        direction=Direction.LONG,
        event_time=new_clock,
        known_at=new_clock,
        evidence={
            "crossing_generation_id": "crossing-generation:new-session",
            "level_id": "level:new-session",
        },
        origin=EventOrigin.SEMANTIC_ATOMIC,
    ))
    new_state, records, _, _ = brain._update_shadow_path_diagnostics(
        _observation(
            new_clock,
            real_completed=True,
            events=(acceptance,),
        ),
        _final_context(
            asof=new_clock,
            scene_revision_id="revision:new-session",
        ),
        {},
        {},
        None,
    )

    assert new_state is not None
    assert new_state.competition_set_id != old_state.competition_set_id
    assert new_state.formed_at == new_clock
    assert new_state.status is PathStatus.ACTIVE
    assert len(records) == 1 and records[0].initialization
    assert new_state.evidence_ledger[0].source_event_ids == (
        acceptance.event_id,
    )
    assert brain.hypothesis_manager.state == new_state


def test_existing_scope_update_commits_only_after_dol_ranking_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t0 = _clock("2025-01-06 10:00")
    context = _final_context(
        asof=t0,
        scene_revision_id="revision:transactional-path",
    )
    brain = PlaybookBrain()
    first, _, _, _ = brain._update_shadow_path_diagnostics(
        _observation(t0, real_completed=True),
        context,
        {},
        {},
        None,
    )
    assert first is not None
    original_ranker = playbooks_module.rank_dol_candidates

    def fail_ranking(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected DOL ranking failure")

    monkeypatch.setattr(
        playbooks_module,
        "rank_dol_candidates",
        fail_ranking,
    )
    t1 = t0 + pd.Timedelta(minutes=1)
    with pytest.raises(RuntimeError, match="injected DOL"):
        brain._update_shadow_path_diagnostics(
            _observation(t1, real_completed=True),
            replace(context, updated_at=t1),
            {},
            {},
            None,
        )

    assert brain._path_competition_state == first
    assert brain.hypothesis_manager.state == first
    monkeypatch.setattr(
        playbooks_module,
        "rank_dol_candidates",
        original_ranker,
    )
    retried, records, _, _ = brain._update_shadow_path_diagnostics(
        _observation(t1, real_completed=True),
        replace(context, updated_at=t1),
        {},
        {},
        None,
    )
    assert retried is not None and retried.asof == t1
    assert len(records) == 1
    assert brain.hypothesis_manager.state == retried


def test_full_update_rolls_back_path_dol_after_post_commit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t0 = _clock("2025-01-06 10:00")
    authority_event = _event("event:authority", "h4-up", t0)
    graph = TemporalMarketSceneGraph()
    nodes = _add_authority_and_draws(graph, t0)
    graph._last_asof = t0
    brain = PlaybookBrain()
    previous_belief = brain.update(
        _observation(t0, real_completed=True, events=(authority_event,)),
        scene_graph=graph,
        scene_delta=_delta(
            graph,
            t0,
            nodes=tuple(node.node_id for node in nodes),
        ),
    )
    previous_manager_payload = brain.hypothesis_manager.checkpoint_payload()
    previous_path_state = brain._path_competition_state
    previous_scope_key = brain._path_scope_key
    previous_seen_tokens = set(brain._path_seen_evidence_tokens)
    previous_dol_probabilities = dict(brain._dol_probability_results)
    brain._signal_real_completed_bar_anchors["candidate:test"] = (
        previous_path_state.competition_set_id,
        "episode:test",
        t0,
        previous_path_state.real_completed_bar_count,
    )
    previous_signal_anchors = dict(
        brain._signal_real_completed_bar_anchors
    )

    t1 = t0 + pd.Timedelta(minutes=1)
    observation1 = _observation(
        t1,
        real_completed=True,
        events=(authority_event,),
    )
    delta1 = _delta(graph, t1, nodes=_heartbeat(graph, t1))

    def fail_after_path_commit(
        current_brain: PlaybookBrain,
        current_observation,
        belief: MarketBelief,
    ) -> MarketBelief:
        assert current_observation.asof == t1
        assert belief.path_competition_state is not None
        assert belief.path_competition_state.asof == t1
        assert current_brain.hypothesis_manager.state == (
            belief.path_competition_state
        )
        assert all(
            result.path_asof == t1
            for result in current_brain._dol_probability_results.values()
        )
        current_brain._signal_real_completed_bar_anchors["candidate:test"] = (
            belief.path_competition_state.competition_set_id,
            "episode:test",
            t0,
            belief.path_competition_state.real_completed_bar_count,
        )
        raise RuntimeError("injected post-path-commit failure")

    monkeypatch.setattr(
        PlaybookBrain,
        "_project_shadow_signal_assessments",
        fail_after_path_commit,
    )
    with pytest.raises(RuntimeError, match="post-path-commit"):
        brain.update(
            observation1,
            scene_graph=graph,
            scene_delta=delta1,
        )

    assert brain.current is previous_belief
    assert (
        brain.hypothesis_manager.checkpoint_payload()
        == previous_manager_payload
    )
    assert brain._path_competition_state == previous_path_state
    assert brain._path_scope_key == previous_scope_key
    assert brain._path_seen_evidence_tokens == previous_seen_tokens
    assert brain._dol_probability_results == previous_dol_probabilities
    assert brain._signal_real_completed_bar_anchors == previous_signal_anchors


@pytest.mark.parametrize(
    ("source_ids", "structural_rank", "expected_reason"),
    [
        ((), "external", "candidate_source_ids_missing"),
        (("source:draw",), "unregistered", "candidate_structural_rank_unregistered"),
    ],
)
def test_dol_adapter_excludes_invalid_inventory_fact_without_crashing_brain(
    monkeypatch: pytest.MonkeyPatch,
    source_ids: tuple[str, ...],
    structural_rank: str,
    expected_reason: str,
) -> None:
    asof = _clock("2025-01-06 10:00")
    observation = _observation(asof, real_completed=True)
    context = _final_context(
        asof=asof,
        scene_revision_id="revision:test",
    )
    context = replace(
        context,
        external_draw_candidates={"above": ("above-level",), "below": ()},
        open_market_theses=(context.open_market_theses[0],),
    )
    invalid = SimpleNamespace(
        item_id="above-level",
        timeframe=Timeframe.H1,
        side="above",
        price=103.0,
        kind="swing",
        source_ids=source_ids,
        structural_rank=structural_rank,
        strength=0.5,
        age_bars=1,
    )
    monkeypatch.setattr(
        playbooks_module,
        "_inventory_item_map",
        lambda _observation: {"above-level": invalid},
    )

    candidates, _obstructions, exclusions = playbooks_module._shadow_dol_facts(
        observation,
        context,
    )

    assert candidates[Direction.LONG.value] == ()
    assert exclusions[Direction.LONG.value] == (
        ("above-level", expected_reason),
    )


def test_market_belief_accepts_expired_path_without_stale_dol() -> None:
    asof = _clock("2025-01-06 10:00")
    observation = _observation(asof, real_completed=True)
    context = _final_context(
        asof=asof,
        scene_revision_id="revision:test",
    )
    brain = PlaybookBrain()
    state, records, rankings, exclusions = brain._update_shadow_path_diagnostics(
        observation,
        context,
        {},
        {},
        None,
    )
    active = MarketBelief(
        asof=asof,
        hypotheses={},
        path_competition_state=state,
        path_update_records_this_clock=records,
        dol_rankings=rankings,
        dol_candidate_exclusions=exclusions,
    )
    assert active.path_competition_state is state

    expiry = state.common_expires_at
    brain._signal_real_completed_bar_anchors["candidate:test"] = (
        state.competition_set_id,
        "episode:test",
        asof,
        state.real_completed_bar_count,
    )
    expired_state, expired_records, expired_rankings, expired_exclusions = (
        brain._update_shadow_path_diagnostics(
            _observation(expiry, real_completed=True),
            replace(context, updated_at=expiry),
            {},
            {},
            active,
        )
    )
    expired = MarketBelief(
        asof=expiry,
        hypotheses={},
        path_competition_state=expired_state,
        path_update_records_this_clock=expired_records,
        dol_rankings=expired_rankings,
        dol_candidate_exclusions=expired_exclusions,
    )
    assert expired.path_competition_state.status.value == "expired"
    assert expired.dol_rankings == {}
    assert expired.dol_candidate_exclusions == {}
    assert brain._signal_real_completed_bar_anchors == {}


def test_engine_binds_exact_shadow_protocol_fingerprints_outside_repo_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    engine = ContinuousSMCEngine.from_config(
        ROOT / "configs/model.json",
        runtime_mode="development",
    )
    binding = json.loads(
        (ROOT / "configs/model.json").read_text(encoding="utf-8")
    )
    path_binding = binding["path_hypotheses"]
    assert engine.brain.path_protocol.fingerprint == path_binding[
        "path_protocol_fingerprint"
    ]
    assert engine.brain.dol_protocol.fingerprint == path_binding[
        "dol_protocol_fingerprint"
    ]
    assert engine.brain.dol_probability_protocol.fingerprint == binding[
        "dol_probability"
    ]["protocol_fingerprint"]
    assert engine.brain.signal_policy.fingerprint == binding[
        "signal_policy"
    ]["protocol_fingerprint"]

    payload = json.loads(
        (ROOT / "configs/model.json").read_text(encoding="utf-8")
    )
    payload["path_hypotheses"]["path_protocol_fingerprint"] = "0" * 64
    stale = tmp_path / "model.json"
    stale.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="path protocol fingerprint is stale"):
        ContinuousSMCEngine.from_config(stale, runtime_mode="development")

    payload = json.loads(
        (ROOT / "configs/model.json").read_text(encoding="utf-8")
    )
    payload["dol_probability"]["protocol_fingerprint"] = "0" * 64
    stale_dol = tmp_path / "model-stale-dol.json"
    stale_dol.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="DOL probability protocol fingerprint is stale",
    ):
        ContinuousSMCEngine.from_config(
            stale_dol,
            runtime_mode="development",
        )

    payload = json.loads(
        (ROOT / "configs/model.json").read_text(encoding="utf-8")
    )
    payload["signal_policy"]["protocol_fingerprint"] = "0" * 64
    stale_signal = tmp_path / "model-stale-signal.json"
    stale_signal.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Signal Policy fingerprint is stale"):
        ContinuousSMCEngine.from_config(
            stale_signal,
            runtime_mode="development",
        )
