from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from smc_trader.dol_probability import (
    DOLProbabilityModelArtifact,
    load_dol_probability_model_artifact,
    load_dol_probability_protocol,
    marginalize_dol_probabilities,
)
from smc_trader.dol_ranking import (
    DOLCandidateFact,
    DOLDirection,
    DOLObstructionViewFact,
    load_dol_ranking_protocol,
    rank_dol_candidates,
)
from smc_trader.model import (
    AccountState,
    Direction,
    EntryEpisodeState,
    Evidence,
    FrozenLSRContext,
    FrozenTriggerState,
    HypothesisBelief,
    LiquidityLevel,
    Playbook,
    PlaybookPhase,
    StructuralLevel,
    Timeframe,
    TradePlan,
)
from smc_trader.path_belief import (
    PathKind,
    create_path_competition_set,
    load_path_belief_protocol,
)
from smc_trader.signal_policy import (
    AdmittedDOLCalibrationArtifact,
    AdmittedPathLikelihoodArtifact,
    CancelConditionKind,
    SetupDeliveryModel,
    SetupFamily,
    SignalDisposition,
    SignalEvaluationContext,
    SignalArtifactPins,
    SignalPolicyProtocol,
    SignalPolicyProtocolError,
    SignalRejection,
    TargetBeforeInvalidationArtifact,
    assess_signal,
    load_signal_policy_protocol,
)
from smc_trader.trade_intent import (
    EntryMethod,
    TradeIntentError,
    build_trade_intent,
)


T0 = pd.Timestamp("2026-08-21 09:30:00", tz="America/New_York")
EXPIRY = T0 + pd.Timedelta(hours=1)
INSTRUMENT_ID = "NQ:front"
ENTRY_PREFERENCES = (
    EntryMethod.FVG_50_LIMIT,
    EntryMethod.OB_50_LIMIT,
    EntryMethod.RECLAIM_ENTRY,
    EntryMethod.MARKET_ENTRY,
)


def _policy(
    *,
    minimum_probability: float = 0.30,
    minimum_edge_R: float = 0.0,
    maximum_input_age: str = "5min",
) -> SignalPolicyProtocol:
    return SignalPolicyProtocol(
        protocol_id="signal-policy:test:v1",
        protocol_version="signal_policy_test_v1",
        minimum_p_target_before_invalidation=minimum_probability,
        minimum_net_edge_R=minimum_edge_R,
        maximum_estimated_cost_R=0.25,
        minimum_coverage=0.90,
        maximum_input_age=pd.Timedelta(maximum_input_age),
        maximum_half_life_real_completed_bars=20,
    )


def test_signal_policy_loader_is_cwd_independent_and_fail_closed(
    tmp_path,
    monkeypatch,
) -> None:
    expected = load_signal_policy_protocol("configs/signal_policy.json")
    monkeypatch.chdir(tmp_path)
    assert load_signal_policy_protocol() == expected

    payload = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/signal_policy.json")
        .read_text(encoding="utf-8")
    )
    payload["unexpected"] = True
    malformed = tmp_path / "signal-policy.json"
    malformed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        SignalPolicyProtocolError,
        match="fields are not frozen exactly",
    ):
        load_signal_policy_protocol(malformed)


def _admitted_path_protocol():
    base = load_path_belief_protocol("configs/path_hypotheses.json")
    conditional = tuple((path, 0.5) for path, _ in base.prior_log_weights)
    return replace(
        base,
        model_version="path-bayesian-test-fitted-v1",
        model_admission_status="diagnostic_likelihood_admitted",
        likelihood_artifact_status="diagnostic_admitted",
        evidence_rules=tuple(
            replace(
                rule,
                conditional_likelihoods=conditional,
                log_likelihood_increments=tuple(
                    (path, math.log(value)) for path, value in conditional
                ),
            )
            for rule in base.evidence_rules
        ),
        fingerprint="f" * 64,
    )


def _path_state():
    protocol = _admitted_path_protocol()
    return create_path_competition_set(
        protocol,
        instrument_id=INSTRUMENT_ID,
        market_epoch_id="epoch:nq:2026-08-21",
        authority_structure_id="h1-structure:1",
        horizon_id="ny-am:2026-08-21",
        formed_at=T0,
        common_expires_at=EXPIRY,
    )


def _candidate_facts(*, path=PathKind.CONTINUATION):
    return (
        DOLCandidateFact(
            candidate_id="draw:primary",
            timeframe="1H",
            side="above",
            target_price=110.0,
            source_kind="candidate_liquidity_level",
            source_ids=("event:draw:primary",),
            structural_rank="external",
            strength=0.8,
            age_real_completed_bars=2,
            path=path,
        ),
        DOLCandidateFact(
            candidate_id="draw:alternate",
            timeframe="1H",
            side="above",
            target_price=112.0,
            source_kind="candidate_liquidity_level",
            source_ids=("event:draw:alternate",),
            structural_rank="external",
            strength=0.4,
            age_real_completed_bars=5,
            path=path,
        ),
    )


def _ranking(path_state, *, path=PathKind.CONTINUATION):
    protocol = load_dol_ranking_protocol("configs/path_hypotheses.json")
    return rank_dol_candidates(
        protocol,
        direction=DOLDirection.LONG,
        current_price=100.0,
        external_draw_candidates=_candidate_facts(path=path),
        obstruction_view=DOLObstructionViewFact(
            direction=DOLDirection.LONG,
            hard_barriers=(),
            soft_frictions=(),
        ),
        path_state=path_state,
    )


def _fitted_dol_model():
    protocol = load_dol_probability_protocol("configs/dol_probability.json")
    path_protocol = _admitted_path_protocol()
    semantic = {
        "schema_version": 2,
        "artifact_id": "dol-artifact:signal-policy-default-test-v1",
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


def _probability(path_state):
    probability_protocol = load_dol_probability_protocol(
        "configs/dol_probability.json"
    )
    return marginalize_dol_probabilities(
        probability_protocol,
        ranking_protocol=load_dol_ranking_protocol(
            "configs/path_hypotheses.json"
        ),
        direction=DOLDirection.LONG,
        current_price=100.0,
        external_draw_candidates=_candidate_facts(),
        obstruction_view=DOLObstructionViewFact(
            direction=DOLDirection.LONG,
            hard_barriers=(),
            soft_frictions=(),
        ),
        path_state=path_state,
        model_artifact=_fitted_dol_model(),
    )


def _typed_setup():
    invalidation = StructuralLevel(
        price=98.0,
        side="below",
        source_level_id="protected-low:1",
        observed_at=T0 - pd.Timedelta(minutes=5),
        rationale="protected swing",
    )
    target = LiquidityLevel(
        level_id="draw:primary",
        timeframe=Timeframe.H1,
        side="above",
        price=110.0,
        formed_at=T0 - pd.Timedelta(hours=1),
        confirmed_at=T0 - pd.Timedelta(minutes=30),
        touches=0,
    )
    plan = TradePlan(
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        direction=Direction.LONG,
        planned_entry=100.0,
        invalidation=invalidation,
        targets=(target,),
        risk_points=2.0,
        primary_target_R=5.0,
        remaining_path_R=5.0,
        deadline=EXPIRY,
        setup_id="episode:dfp:1",
        entry_location_id="fvg:1",
        entry_path_id="entry-path:1",
        entry_zone_lower=99.0,
        entry_zone_upper=101.0,
        selected_draw_id=target.level_id,
    )
    quality = {
        "structure": 0.5,
        "displacement": 0.5,
        "location": 0.5,
        "liquidity": 0.5,
        "trigger": 0.5,
        "execution": 0.5,
    }
    dimensions = {
        "thesis_strength": 0.5,
        "sequence_progress": 0.5,
        "location_quality": 0.5,
        "entry_readiness": 0.5,
        "delivery_quality": 0.5,
        "uncertainty": 0.2,
    }
    candidate = HypothesisBelief(
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        direction=Direction.LONG,
        # Deliberately extreme: the Signal Policy must never consume it.
        probability=0.99,
        phase=PlaybookPhase.EXECUTABLE,
        phase_started_at=T0,
        supporting=(),
        contradicting=(),
        invalidation=invalidation,
        deliverable_targets=(target,),
        remaining_path_R=5.0,
        uncertainty=0.2,
        plan=plan,
        raw_probability=0.98,
        calibration_version="legacy-playbook-test-only",
        thesis_strength=0.5,
        sequence_progress=0.5,
        location_quality=0.5,
        entry_readiness=0.5,
        delivery_quality=0.5,
        evidence_group_scores=quality,
        hard_gate_results={"typed_setup": True},
        setup_context_id="episode:dfp:1",
        entry_location_id="fvg:1",
        entry_path_id="entry-path:1",
        context_id="context:dfp:1",
        episode_id="episode:dfp:1",
        context_thesis_id="context-thesis:1",
        parent_context_thesis_id="context-thesis:1",
        thesis_deadline=EXPIRY,
        episode_deadline=EXPIRY,
        initiating_event_id="event:dfp:root",
        evidence_revision_id="evidence-revision:1",
        raw_quality_dimensions=dimensions,
        market_thesis_ids=("market-thesis:1",),
        market_thesis_id="market-thesis:1",
        bound_market_thesis_id="market-thesis:1",
        market_thesis_root_id="root:1",
        market_thesis_mechanism="dfp",
        market_thesis_authority_relation="aligned",
        playbook_match_strength=0.8,
        market_thesis_binding_required=True,
        market_thesis_action_bound=True,
        market_thesis_match_status="exact_root_bound",
        candidate_id="candidate:dfp:1",
        required_root_id="root:1",
        record_kind="root_candidate",
    )
    episode = EntryEpisodeState(
        episode_id="episode:dfp:1",
        parent_context_thesis_id="context-thesis:1",
        candidate_id="candidate:dfp:1",
        playbook=Playbook.DISPLACEMENT_FIRST_PULLBACK,
        direction=Direction.LONG,
        initiating_event_id="event:dfp:root",
        entry_location_id="fvg:1",
        entry_path_id="entry-path:1",
        first_pullback_at=T0 - pd.Timedelta(minutes=1),
        selected_trigger=None,
        plan=plan,
        invalidation=invalidation,
        deadline=EXPIRY,
        phase=PlaybookPhase.EXECUTABLE,
        formed_at=T0 - pd.Timedelta(minutes=2),
        updated_at=T0,
    )
    return candidate, episode


def _typed_lsr_setup():
    candidate, episode = _typed_setup()
    assert candidate.plan is not None
    context = FrozenLSRContext(
        manipulation_id="protected-low:1",
        manipulation_protocol_hash="manipulation-protocol:test",
        source_pool_id="ssl-pool:1",
        pool_path_id="pool-path:1",
        pool_path_protocol_hash="pool-path-protocol:test",
        displacement_id="displacement:lsr:1",
        direction=Direction.LONG,
        swept_at=T0 - pd.Timedelta(minutes=5),
        reaccepted_at=T0 - pd.Timedelta(minutes=4),
        displacement_active_at=T0 - pd.Timedelta(minutes=3),
        displacement_observed_at=T0 - pd.Timedelta(minutes=2),
        sweep_extreme=98.0,
    )
    plan = replace(
        candidate.plan,
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        lsr_context=context,
    )
    return (
        replace(
            candidate,
            playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
            plan=plan,
            market_thesis_mechanism="lsr",
        ),
        replace(
            episode,
            playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
            plan=plan,
        ),
    )


def _evaluation(
    *,
    coverage: float = 1.0,
    ood: bool = False,
    cost_R: float | None = 0.05,
    real_completed_bars_since_setup: int | None = 0,
) -> SignalEvaluationContext:
    return SignalEvaluationContext(
        coverage_id="coverage:1",
        coverage_fraction=coverage,
        ood_assessment_id="ood-check:1",
        out_of_distribution=ood,
        cost_estimate_id=None if cost_R is None else "cost-estimate:1",
        estimated_cost_R=cost_R,
        source_event_ids=("event:coverage:1", "event:session:1"),
        real_completed_bars_since_setup=real_completed_bars_since_setup,
    )


def _artifacts(policy, path_state, ranking, *, path_expiry=EXPIRY):
    path = AdmittedPathLikelihoodArtifact(
        protocol_id="path-admission:test:v1",
        model_id="path-likelihood:test",
        model_version="path-likelihood-test-v1",
        calibration_id="path-calibration:test:v1",
        source_dataset_id="dataset:test:train",
        coverage_id="path-coverage:test",
        source_path_protocol_fingerprint=path_state.protocol_fingerprint,
        source_path_model_version=path_state.model_version,
        trained_through=T0 - pd.Timedelta(days=2),
        valid_from=T0 - pd.Timedelta(days=1),
        expires_at=path_expiry,
        temperature=0.9,
    )
    dol = AdmittedDOLCalibrationArtifact(
        protocol_id="dol-admission:test:v1",
        model_id="dol-calibration:test",
        model_version="dol-calibration-test-v1",
        calibration_id="dol-calibration-id:test:v1",
        source_dataset_id="dataset:test:train",
        coverage_id="dol-coverage:test",
        source_dol_protocol_fingerprint=ranking.protocol_fingerprint,
        source_dol_model_version=ranking.model_version,
        source_dol_model_fingerprint=getattr(
            ranking,
            "model_fingerprint",
            None,
        ),
        source_path_protocol_fingerprint=ranking.path_protocol_fingerprint,
        source_path_model_version=ranking.path_model_version,
        trained_through=T0 - pd.Timedelta(days=2),
        valid_from=T0 - pd.Timedelta(days=1),
        expires_at=EXPIRY,
        temperature=1.1,
    )
    outcome = TargetBeforeInvalidationArtifact(
        protocol_id="delivery-admission:test:v1",
        model_id="target-before-invalidation:test",
        model_version="target-before-invalidation-test-v1",
        calibration_id="delivery-calibration:test:v1",
        source_dataset_id="dataset:test:train",
        coverage_id="delivery-coverage:test",
        path_likelihood_artifact_id=path.artifact_id,
        dol_calibration_artifact_id=dol.artifact_id,
        signal_policy_fingerprint=policy.fingerprint,
        trained_through=T0 - pd.Timedelta(days=2),
        valid_from=T0 - pd.Timedelta(days=1),
        expires_at=EXPIRY,
        setup_models=(
            SetupDeliveryModel(
                setup_family=SetupFamily.DFP,
                intercept=2.0,
                path_logit_coefficient=0.10,
                dol_logit_coefficient=0.10,
                half_life_real_completed_bars=10,
            ),
            SetupDeliveryModel(
                setup_family=SetupFamily.LSR,
                intercept=1.8,
                path_logit_coefficient=0.10,
                dol_logit_coefficient=0.10,
                half_life_real_completed_bars=8,
            ),
        ),
        minimum_supported_coverage=0.90,
    )
    return path, dol, outcome


def _artifact_pins(
    policy,
    path_state,
    artifacts,
    *,
    dol_model_fingerprint=None,
):
    if dol_model_fingerprint is None:
        dol_model_fingerprint = artifacts[1].source_dol_model_fingerprint
    return SignalArtifactPins(
        signal_policy_fingerprint=policy.fingerprint,
        path_protocol_fingerprint=path_state.protocol_fingerprint,
        path_likelihood_artifact_id=artifacts[0].artifact_id,
        dol_calibration_artifact_id=artifacts[1].artifact_id,
        outcome_model_artifact_id=artifacts[2].artifact_id,
        dol_probability_model_fingerprint=dol_model_fingerprint,
    )


def _inputs(*, policy=None):
    policy = _policy() if policy is None else policy
    path_state = _path_state()
    ranking = _probability(path_state)
    candidate, episode = _typed_setup()
    artifacts = _artifacts(policy, path_state, ranking)
    return policy, path_state, ranking, candidate, episode, artifacts


def _assess(
    *,
    policy=None,
    asof=T0,
    evaluation=None,
    artifacts="default",
    candidate=None,
    episode=None,
):
    values = _inputs(policy=policy)
    policy, path_state, ranking, default_candidate, default_episode, admitted = values
    candidate = default_candidate if candidate is None else candidate
    episode = default_episode if episode is None else episode
    selected_artifacts = admitted if artifacts == "default" else artifacts
    pins = (
        None
        if any(artifact is None for artifact in selected_artifacts)
        else _artifact_pins(policy, path_state, selected_artifacts)
    )
    return assess_signal(
        policy,
        asof=asof,
        symbol="NQH6",
        instrument_id=INSTRUMENT_ID,
        path_state=path_state,
        dol_ranking=ranking,
        dol_candidate_id="draw:primary",
        setup_candidate=candidate,
        entry_episode=episode,
        evaluation=_evaluation() if evaluation is None else evaluation,
        path_likelihood_artifact=selected_artifacts[0],
        dol_calibration_artifact=selected_artifacts[1],
        outcome_model_artifact=selected_artifacts[2],
        dol_probability_model_artifact=_fitted_dol_model(),
        path_protocol=_admitted_path_protocol(),
        artifact_pins=pins,
    )


def test_signal_policy_requires_all_three_separate_admission_artifacts() -> None:
    assessment = _assess(artifacts=(None, None, None))

    assert assessment.disposition is SignalDisposition.REJECTED
    assert assessment.rejection_reasons == (
        SignalRejection.MISSING_PATH_LIKELIHOOD_ARTIFACT,
        SignalRejection.MISSING_DOL_CALIBRATION_ARTIFACT,
        SignalRejection.MISSING_OUTCOME_MODEL_ARTIFACT,
    )
    assert not assessment.can_authorize_trade


def test_eligible_signal_is_deterministic_and_uses_distinct_estimands() -> None:
    first = _assess()
    replay = _assess()

    assert first == replay
    assert first.signal_id == replay.signal_id
    assert first.eligible
    assert first.authority == "shadow_only"
    assert not first.can_authorize_trade
    assert not first.playbook_probability_consumed
    assert first.p_path_hypothesis is not None
    assert first.p_dol_candidate is not None
    assert first.p_target_before_invalidation is not None
    assert first.p_target_before_invalidation != pytest.approx(first.p_path_hypothesis)
    assert first.p_target_before_invalidation != pytest.approx(0.99)
    assert first.raw_path_probability is not None
    assert first.raw_dol_probability is not None
    with pytest.raises(FrozenInstanceError):
        first.net_edge_R = 999.0  # type: ignore[misc]


def test_signal_rejects_future_causal_plan_and_setup_observations() -> None:
    candidate, episode = _typed_setup()
    assert candidate.plan is not None
    future = T0 + pd.Timedelta(minutes=1)

    future_target = replace(
        candidate.plan.targets[0],
        confirmed_at=future,
    )
    target_plan = replace(candidate.plan, targets=(future_target,))
    target_candidate = replace(
        candidate,
        deliverable_targets=(future_target,),
        plan=target_plan,
    )
    target_episode = replace(episode, plan=target_plan)

    future_invalidation = replace(
        candidate.plan.invalidation,
        observed_at=future,
    )
    invalidation_plan = replace(
        candidate.plan,
        invalidation=future_invalidation,
    )
    invalidation_candidate = replace(
        candidate,
        invalidation=future_invalidation,
        plan=invalidation_plan,
    )
    invalidation_episode = replace(
        episode,
        invalidation=future_invalidation,
        plan=invalidation_plan,
    )

    evidence_candidate = replace(
        candidate,
        supporting=(
            Evidence(
                primitive="future_test_evidence",
                value=1.0,
                weight=1.0,
                supports=True,
                observed_at=future,
                explanation="causal clock regression",
            ),
        ),
    )

    future_trigger = FrozenTriggerState(
        trigger_id="trigger:future",
        trigger_kind="wick_rejection",
        observed_at=future,
        setup_id=episode.episode_id,
        entry_path_id=episode.entry_path_id,
        entry_location_id=episode.entry_location_id,
        direction=episode.direction,
        source_entity_id="source:future-trigger",
        strength=0.8,
        available_trigger_kinds=("wick_rejection",),
    )
    trigger_candidate = replace(candidate, selected_trigger=future_trigger)
    trigger_episode = replace(episode, selected_trigger=future_trigger)

    pairs = (
        (target_candidate, target_episode),
        (invalidation_candidate, invalidation_episode),
        (evidence_candidate, episode),
        (trigger_candidate, trigger_episode),
    )
    for future_candidate, future_episode in pairs:
        assessment = _assess(
            candidate=future_candidate,
            episode=future_episode,
        )
        assert not assessment.eligible
        assert SignalRejection.INPUT_FROM_FUTURE in (
            assessment.rejection_reasons
        )


def test_signal_rejects_future_path_and_dol_snapshot_clocks() -> None:
    policy = _policy()
    future_path = replace(
        _path_state(),
        asof=T0 + pd.Timedelta(minutes=1),
    )
    probability = _probability(future_path)
    candidate, episode = _typed_setup()
    artifacts = _artifacts(policy, future_path, probability)

    assessment = assess_signal(
        policy,
        asof=T0,
        symbol="NQH6",
        instrument_id=INSTRUMENT_ID,
        path_state=future_path,
        dol_ranking=probability,
        dol_candidate_id="draw:primary",
        setup_candidate=candidate,
        entry_episode=episode,
        evaluation=_evaluation(),
        path_likelihood_artifact=artifacts[0],
        dol_calibration_artifact=artifacts[1],
        outcome_model_artifact=artifacts[2],
        dol_probability_model_artifact=_fitted_dol_model(),
        path_protocol=_admitted_path_protocol(),
        artifact_pins=_artifact_pins(policy, future_path, artifacts),
    )

    assert not assessment.eligible
    assert SignalRejection.INPUT_FROM_FUTURE in assessment.rejection_reasons


def test_artifacts_cannot_self_admit_or_wrap_a_missing_path_model() -> None:
    policy, path_state, ranking, candidate, episode, artifacts = _inputs()
    self_admitted = assess_signal(
        policy,
        asof=T0,
        symbol="NQH6",
        instrument_id=INSTRUMENT_ID,
        path_state=path_state,
        dol_ranking=ranking,
        dol_candidate_id="draw:primary",
        setup_candidate=candidate,
        entry_episode=episode,
        evaluation=_evaluation(),
        path_likelihood_artifact=artifacts[0],
        dol_calibration_artifact=artifacts[1],
        outcome_model_artifact=artifacts[2],
        dol_probability_model_artifact=_fitted_dol_model(),
        path_protocol=_admitted_path_protocol(),
        artifact_pins=None,
    )

    missing_protocol = load_path_belief_protocol("configs/path_hypotheses.json")
    missing_state = create_path_competition_set(
        missing_protocol,
        instrument_id=INSTRUMENT_ID,
        market_epoch_id="epoch:nq:missing-model",
        authority_structure_id="h1-structure:missing-model",
        horizon_id="ny-am:missing-model",
        formed_at=T0,
        common_expires_at=EXPIRY,
    )
    missing_ranking = _ranking(missing_state)
    missing_artifacts = _artifacts(policy, missing_state, missing_ranking)
    wrapped_missing = assess_signal(
        policy,
        asof=T0,
        symbol="NQH6",
        instrument_id=INSTRUMENT_ID,
        path_state=missing_state,
        dol_ranking=missing_ranking,
        dol_candidate_id="draw:primary",
        setup_candidate=candidate,
        entry_episode=episode,
        evaluation=_evaluation(),
        path_likelihood_artifact=missing_artifacts[0],
        dol_calibration_artifact=missing_artifacts[1],
        outcome_model_artifact=missing_artifacts[2],
        path_protocol=missing_protocol,
        artifact_pins=_artifact_pins(
            policy,
            missing_state,
            missing_artifacts,
        ),
    )

    assert not self_admitted.eligible
    assert SignalRejection.ARTIFACT_NOT_ADMITTED in self_admitted.rejection_reasons
    assert not wrapped_missing.eligible
    assert SignalRejection.ARTIFACT_IDENTITY_MISMATCH in (
        wrapped_missing.rejection_reasons
    )


def test_fitted_dol_result_requires_its_exact_model_artifact() -> None:
    policy, path_state, probability, candidate, episode, artifacts = _inputs()
    assessment = assess_signal(
        policy,
        asof=T0,
        symbol="NQH6",
        instrument_id=INSTRUMENT_ID,
        path_state=path_state,
        dol_ranking=probability,
        dol_candidate_id="draw:primary",
        setup_candidate=candidate,
        entry_episode=episode,
        evaluation=_evaluation(),
        path_likelihood_artifact=artifacts[0],
        dol_calibration_artifact=artifacts[1],
        outcome_model_artifact=artifacts[2],
        path_protocol=_admitted_path_protocol(),
        artifact_pins=_artifact_pins(policy, path_state, artifacts),
    )

    assert not assessment.eligible
    assert (
        SignalRejection.MISSING_DOL_PROBABILITY_MODEL_ARTIFACT
        in assessment.rejection_reasons
    )


def test_legacy_dol_ranking_must_match_the_typed_setup_path() -> None:
    policy = _policy()
    path_state = _path_state()
    mismatched = _ranking(path_state, path=PathKind.REVERSAL)
    candidate, episode = _typed_setup()
    artifacts = _artifacts(policy, path_state, mismatched)
    assessment = assess_signal(
        policy,
        asof=T0,
        symbol="NQH6",
        instrument_id=INSTRUMENT_ID,
        path_state=path_state,
        dol_ranking=mismatched,
        dol_candidate_id="draw:primary",
        setup_candidate=candidate,
        entry_episode=episode,
        evaluation=_evaluation(),
        path_likelihood_artifact=artifacts[0],
        dol_calibration_artifact=artifacts[1],
        outcome_model_artifact=artifacts[2],
        path_protocol=_admitted_path_protocol(),
        artifact_pins=_artifact_pins(policy, path_state, artifacts),
    )

    assert not assessment.eligible
    assert SignalRejection.DOL_IDENTITY_MISMATCH in assessment.rejection_reasons


def test_fitted_path_marginal_dol_probability_keeps_no_target_in_policy(
    tmp_path,
) -> None:
    policy = _policy()
    path_state = _path_state()
    ranking_protocol = load_dol_ranking_protocol(
        "configs/path_hypotheses.json"
    )
    probability_protocol = load_dol_probability_protocol(
        "configs/dol_probability.json"
    )
    candidates = (
        DOLCandidateFact(
            candidate_id="draw:primary",
            timeframe="1H",
            side="above",
            target_price=110.0,
            source_kind="candidate_liquidity_level",
            source_ids=("event:draw:primary",),
            structural_rank="external",
            strength=0.8,
            age_real_completed_bars=2,
            path=PathKind.CONTINUATION,
        ),
        DOLCandidateFact(
            candidate_id="draw:alternate",
            timeframe="1H",
            side="above",
            target_price=112.0,
            source_kind="candidate_liquidity_level",
            source_ids=("event:draw:alternate",),
            structural_rank="external",
            strength=0.4,
            age_real_completed_bars=5,
            path=PathKind.CONTINUATION,
        ),
    )
    parameters = probability_protocol.development_model.payload()
    semantic = {
        "schema_version": 2,
        "artifact_id": "dol-artifact:signal-policy-test-v1",
        "model_version": probability_protocol.model_version,
        "protocol_fingerprint": probability_protocol.fingerprint,
        "ranking_protocol_fingerprint": ranking_protocol.fingerprint,
        "source_path_protocol_fingerprint": path_state.protocol_fingerprint,
        "source_path_model_version": path_state.model_version,
        "fit_status": "fitted",
        "admission_status": "admitted",
        "calibration_status": "fitted_admitted",
        "authority": "shadow_only",
        "action_authority": False,
        "parameters": parameters,
    }
    artifact_payload = {
        **semantic,
        "artifact_fingerprint": hashlib.sha256(
            json.dumps(
                semantic,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
    }
    artifact_path = tmp_path / "dol-artifact.json"
    artifact_path.write_text(json.dumps(artifact_payload), encoding="utf-8")
    model_artifact = load_dol_probability_model_artifact(
        artifact_path,
        protocol=probability_protocol,
        expected_fingerprint=artifact_payload["artifact_fingerprint"],
    )
    probability = marginalize_dol_probabilities(
        probability_protocol,
        ranking_protocol=ranking_protocol,
        direction=DOLDirection.LONG,
        current_price=100.0,
        external_draw_candidates=candidates,
        obstruction_view=DOLObstructionViewFact(
            direction=DOLDirection.LONG,
            hard_barriers=(),
            soft_frictions=(),
        ),
        path_state=path_state,
        model_artifact=model_artifact,
    )
    path_artifact = AdmittedPathLikelihoodArtifact(
        protocol_id="path-admission:test:v1",
        model_id="path-likelihood:test",
        model_version="path-likelihood-test-v1",
        calibration_id="path-calibration:test:v1",
        source_dataset_id="dataset:test:train",
        coverage_id="path-coverage:test",
        source_path_protocol_fingerprint=path_state.protocol_fingerprint,
        source_path_model_version=path_state.model_version,
        trained_through=T0 - pd.Timedelta(days=2),
        valid_from=T0 - pd.Timedelta(days=1),
        expires_at=EXPIRY,
    )
    dol_artifact = AdmittedDOLCalibrationArtifact(
        protocol_id="dol-admission:test:v2",
        model_id="dol-calibration:test:v2",
        model_version="dol-calibration-test-v2",
        calibration_id="dol-calibration-id:test:v2",
        source_dataset_id="dataset:test:train",
        coverage_id="dol-coverage:test",
        source_dol_protocol_fingerprint=probability.protocol_fingerprint,
        source_dol_model_version=probability.model_version,
        source_path_protocol_fingerprint=probability.path_protocol_fingerprint,
        source_path_model_version=probability.path_model_version,
        source_dol_model_fingerprint=probability.model_fingerprint,
        trained_through=T0 - pd.Timedelta(days=2),
        valid_from=T0 - pd.Timedelta(days=1),
        expires_at=EXPIRY,
    )
    outcome = TargetBeforeInvalidationArtifact(
        protocol_id="delivery-admission:test:v2",
        model_id="target-before-invalidation:test:v2",
        model_version="target-before-invalidation-test-v2",
        calibration_id="delivery-calibration:test:v2",
        source_dataset_id="dataset:test:train",
        coverage_id="delivery-coverage:test:v2",
        path_likelihood_artifact_id=path_artifact.artifact_id,
        dol_calibration_artifact_id=dol_artifact.artifact_id,
        signal_policy_fingerprint=policy.fingerprint,
        trained_through=T0 - pd.Timedelta(days=2),
        valid_from=T0 - pd.Timedelta(days=1),
        expires_at=EXPIRY,
        setup_models=(
            SetupDeliveryModel(
                setup_family=SetupFamily.DFP,
                intercept=2.0,
                path_logit_coefficient=0.10,
                dol_logit_coefficient=0.10,
                half_life_real_completed_bars=10,
            ),
            SetupDeliveryModel(
                setup_family=SetupFamily.LSR,
                intercept=1.8,
                path_logit_coefficient=0.10,
                dol_logit_coefficient=0.10,
                half_life_real_completed_bars=8,
            ),
        ),
        minimum_supported_coverage=0.9,
    )
    candidate, episode = _typed_setup()

    first = assess_signal(
        policy,
        asof=T0,
        symbol="NQH6",
        instrument_id=INSTRUMENT_ID,
        path_state=path_state,
        dol_ranking=probability,
        dol_candidate_id="draw:primary",
        setup_candidate=candidate,
        entry_episode=episode,
        evaluation=_evaluation(),
        path_likelihood_artifact=path_artifact,
        dol_calibration_artifact=dol_artifact,
        outcome_model_artifact=outcome,
        dol_probability_model_artifact=model_artifact,
        path_protocol=_admitted_path_protocol(),
        artifact_pins=_artifact_pins(
            policy,
            path_state,
            (path_artifact, dol_artifact, outcome),
            dol_model_fingerprint=probability.model_fingerprint,
        ),
    )
    replay = assess_signal(
        policy,
        asof=T0,
        symbol="NQH6",
        instrument_id=INSTRUMENT_ID,
        path_state=path_state,
        dol_ranking=probability,
        dol_candidate_id="draw:primary",
        setup_candidate=candidate,
        entry_episode=episode,
        evaluation=_evaluation(),
        path_likelihood_artifact=path_artifact,
        dol_calibration_artifact=dol_artifact,
        outcome_model_artifact=outcome,
        dol_probability_model_artifact=model_artifact,
        path_protocol=_admitted_path_protocol(),
        artifact_pins=_artifact_pins(
            policy,
            path_state,
            (path_artifact, dol_artifact, outcome),
            dol_model_fingerprint=probability.model_fingerprint,
        ),
    )

    admitted = dol_artifact.probabilities(probability)
    assert first == replay
    assert first.eligible
    assert first.dol_probability_id == probability.probability_id
    assert first.raw_dol_probability == probability.ranked_candidates[0].probability
    assert first.p_dol_candidate == admitted["draw:primary"]
    assert sum(admitted.values()) < 1.0
    assert probability.no_target_probability > 0.0
    account = AccountState(
        equity=100_000.0,
        requested_risk_fraction=0.005,
        quantity=1,
        point_value=20.0,
    )
    intent = build_trade_intent(
        first,
        asof=T0,
        setup_candidate=candidate,
        entry_episode=episode,
        account=account,
        account_snapshot_id="account-snapshot:path-marginal",
        risk_budget_id="risk-budget:path-marginal",
        entry_method_preferences=ENTRY_PREFERENCES,
    )
    replay_intent = build_trade_intent(
        replay,
        asof=T0,
        setup_candidate=candidate,
        entry_episode=episode,
        account=account,
        account_snapshot_id="account-snapshot:path-marginal",
        risk_budget_id="risk-budget:path-marginal",
        entry_method_preferences=ENTRY_PREFERENCES,
    )
    assert intent == replay_intent
    assert intent.dol_ranking_id == probability.probability_id
    assert not intent.submission_allowed


@pytest.mark.parametrize(
    ("evaluation", "reason"),
    [
        (_evaluation(coverage=0.80), SignalRejection.COVERAGE_INCOMPLETE),
        (_evaluation(ood=True), SignalRejection.OUT_OF_DISTRIBUTION),
        (_evaluation(cost_R=None), SignalRejection.COST_UNAVAILABLE),
        (_evaluation(cost_R=0.30), SignalRejection.COST_EXCEEDS_LIMIT),
    ],
)
def test_signal_policy_fail_closed_context_gates(evaluation, reason) -> None:
    assessment = _assess(evaluation=evaluation)

    assert not assessment.eligible
    assert reason in assessment.rejection_reasons


def test_low_probability_and_low_edge_each_reject() -> None:
    high_probability_policy = _policy(minimum_probability=0.99)
    probability = _assess(policy=high_probability_policy)
    high_edge_policy = _policy(minimum_edge_R=10.0)
    edge = _assess(policy=high_edge_policy)

    assert SignalRejection.PROBABILITY_BELOW_MINIMUM in (probability.rejection_reasons)
    assert SignalRejection.EDGE_BELOW_MINIMUM in edge.rejection_reasons


def test_stale_input_and_expired_signal_reject() -> None:
    policy = _policy(maximum_input_age="2min")
    stale = _assess(policy=policy, asof=T0 + pd.Timedelta(minutes=3))
    expired = _assess(asof=EXPIRY)

    assert SignalRejection.INPUT_STALE in stale.rejection_reasons
    assert SignalRejection.SIGNAL_EXPIRED in expired.rejection_reasons


def test_half_life_counts_real_completed_bars_not_wall_clock_minutes() -> None:
    policy = _policy(maximum_input_age="20min")
    synthetic_gap = _assess(
        policy=policy,
        asof=T0 + pd.Timedelta(minutes=11),
        evaluation=_evaluation(real_completed_bars_since_setup=9),
    )
    elapsed = _assess(
        policy=policy,
        asof=T0 + pd.Timedelta(minutes=11),
        evaluation=_evaluation(real_completed_bars_since_setup=10),
    )

    assert synthetic_gap.eligible
    assert synthetic_gap.half_life_expires_at is None
    assert synthetic_gap.age_real_completed_bars == 9
    assert synthetic_gap.half_life_real_completed_bars == 10
    assert SignalRejection.SIGNAL_HALF_LIFE_ELAPSED in elapsed.rejection_reasons


def test_stale_probability_artifact_rejects() -> None:
    policy, path_state, ranking, candidate, episode, _ = _inputs()
    artifacts = _artifacts(
        policy,
        path_state,
        ranking,
        path_expiry=T0 + pd.Timedelta(minutes=1),
    )
    assessment = assess_signal(
        policy,
        asof=T0 + pd.Timedelta(minutes=2),
        symbol="NQH6",
        instrument_id=INSTRUMENT_ID,
        path_state=path_state,
        dol_ranking=ranking,
        dol_candidate_id="draw:primary",
        setup_candidate=candidate,
        entry_episode=episode,
        evaluation=_evaluation(),
        path_likelihood_artifact=artifacts[0],
        dol_calibration_artifact=artifacts[1],
        outcome_model_artifact=artifacts[2],
        dol_probability_model_artifact=_fitted_dol_model(),
    )

    assert not assessment.eligible
    assert SignalRejection.ARTIFACT_OUTSIDE_VALIDITY in (assessment.rejection_reasons)


def test_exact_episode_identity_mismatch_rejects() -> None:
    policy, path_state, ranking, candidate, episode, artifacts = _inputs()
    mismatched_episode = replace(episode, candidate_id="candidate:other")
    assessment = assess_signal(
        policy,
        asof=T0,
        symbol="NQH6",
        instrument_id=INSTRUMENT_ID,
        path_state=path_state,
        dol_ranking=ranking,
        dol_candidate_id="draw:primary",
        setup_candidate=candidate,
        entry_episode=mismatched_episode,
        evaluation=_evaluation(),
        path_likelihood_artifact=artifacts[0],
        dol_calibration_artifact=artifacts[1],
        outcome_model_artifact=artifacts[2],
        dol_probability_model_artifact=_fitted_dol_model(),
    )

    assert SignalRejection.SETUP_IDENTITY_MISMATCH in (assessment.rejection_reasons)


def test_favr_remains_parked() -> None:
    policy, path_state, ranking, candidate, episode, artifacts = _inputs()
    parked_candidate = replace(
        candidate,
        playbook=Playbook.FAILED_AUCTION_VALUE_RETURN,
    )
    parked_episode = replace(
        episode,
        playbook=Playbook.FAILED_AUCTION_VALUE_RETURN,
    )
    assessment = assess_signal(
        policy,
        asof=T0,
        symbol="NQH6",
        instrument_id=INSTRUMENT_ID,
        path_state=path_state,
        dol_ranking=ranking,
        dol_candidate_id="draw:primary",
        setup_candidate=parked_candidate,
        entry_episode=parked_episode,
        evaluation=_evaluation(),
        path_likelihood_artifact=artifacts[0],
        dol_calibration_artifact=artifacts[1],
        outcome_model_artifact=artifacts[2],
        dol_probability_model_artifact=_fitted_dol_model(),
    )

    assert SignalRejection.SETUP_FAMILY_PARKED in assessment.rejection_reasons


def test_lsr_is_the_second_admitted_setup_family() -> None:
    policy, path_state, ranking, _, _, artifacts = _inputs()
    candidate, episode = _typed_lsr_setup()
    assessment = assess_signal(
        policy,
        asof=T0,
        symbol="NQH6",
        instrument_id=INSTRUMENT_ID,
        path_state=path_state,
        dol_ranking=ranking,
        dol_candidate_id="draw:primary",
        setup_candidate=candidate,
        entry_episode=episode,
        evaluation=_evaluation(),
        path_likelihood_artifact=artifacts[0],
        dol_calibration_artifact=artifacts[1],
        outcome_model_artifact=artifacts[2],
        dol_probability_model_artifact=_fitted_dol_model(),
        path_protocol=_admitted_path_protocol(),
        artifact_pins=_artifact_pins(policy, path_state, artifacts),
    )

    assert assessment.eligible
    assert assessment.setup_family is SetupFamily.LSR


def test_trade_intent_freezes_plan_risk_methods_and_cancel_conditions() -> None:
    policy, path_state, ranking, candidate, episode, artifacts = _inputs()
    assessment = assess_signal(
        policy,
        asof=T0,
        symbol="NQH6",
        instrument_id=INSTRUMENT_ID,
        path_state=path_state,
        dol_ranking=ranking,
        dol_candidate_id="draw:primary",
        setup_candidate=candidate,
        entry_episode=episode,
        evaluation=_evaluation(),
        path_likelihood_artifact=artifacts[0],
        dol_calibration_artifact=artifacts[1],
        outcome_model_artifact=artifacts[2],
        dol_probability_model_artifact=_fitted_dol_model(),
        path_protocol=_admitted_path_protocol(),
        artifact_pins=_artifact_pins(policy, path_state, artifacts),
    )
    account = AccountState(
        equity=100_000.0,
        requested_risk_fraction=0.005,
        quantity=1,
        point_value=20.0,
    )
    first = build_trade_intent(
        assessment,
        asof=T0,
        setup_candidate=candidate,
        entry_episode=episode,
        account=account,
        account_snapshot_id="account-snapshot:1",
        risk_budget_id="risk-budget:1",
        entry_method_preferences=ENTRY_PREFERENCES,
    )
    replay = build_trade_intent(
        assessment,
        asof=T0,
        setup_candidate=candidate,
        entry_episode=episode,
        account=account,
        account_snapshot_id="account-snapshot:1",
        risk_budget_id="risk-budget:1",
        entry_method_preferences=ENTRY_PREFERENCES,
    )

    assert first == replay
    assert first.intent_id == replay.intent_id
    assert first.authority == "shadow_only"
    assert not first.submission_allowed
    assert first.signal_id == assessment.signal_id
    assert first.entry_method_preferences == ENTRY_PREFERENCES
    assert first.planned_entry == candidate.plan.planned_entry
    assert first.invalidation == candidate.plan.invalidation
    assert first.targets == candidate.plan.targets
    assert first.cancel_conditions == assessment.cancel_conditions
    assert {condition.kind for condition in first.cancel_conditions} >= {
        CancelConditionKind.SIGNAL_EXPIRY_REACHED,
        CancelConditionKind.SIGNAL_HALF_LIFE_ELAPSED,
        CancelConditionKind.STRUCTURAL_INVALIDATION_REACHED,
        CancelConditionKind.PATH_HYPOTHESIS_NO_LONGER_ACTIVE,
        CancelConditionKind.DOL_CANDIDATE_UNAVAILABLE,
        CancelConditionKind.COST_EXCEEDS_LIMIT,
        CancelConditionKind.DELIVERY_PROBABILITY_BELOW_MINIMUM,
    }
    with pytest.raises(FrozenInstanceError):
        first.quantity = 99  # type: ignore[misc]


def test_trade_intent_rejects_rejected_signal_stale_signal_and_risk_overrun() -> None:
    policy, path_state, ranking, candidate, episode, artifacts = _inputs()
    accepted = assess_signal(
        policy,
        asof=T0,
        symbol="NQH6",
        instrument_id=INSTRUMENT_ID,
        path_state=path_state,
        dol_ranking=ranking,
        dol_candidate_id="draw:primary",
        setup_candidate=candidate,
        entry_episode=episode,
        evaluation=_evaluation(),
        path_likelihood_artifact=artifacts[0],
        dol_calibration_artifact=artifacts[1],
        outcome_model_artifact=artifacts[2],
        dol_probability_model_artifact=_fitted_dol_model(),
        path_protocol=_admitted_path_protocol(),
        artifact_pins=_artifact_pins(policy, path_state, artifacts),
    )
    rejected = _assess(artifacts=(None, None, None))
    account = AccountState(equity=100_000.0, quantity=1, point_value=20.0)

    with pytest.raises(TradeIntentError, match="rejected signal"):
        build_trade_intent(
            rejected,
            asof=T0,
            setup_candidate=candidate,
            entry_episode=episode,
            account=account,
            account_snapshot_id="account-snapshot:1",
            risk_budget_id="risk-budget:1",
            entry_method_preferences=ENTRY_PREFERENCES,
        )
    with pytest.raises(TradeIntentError, match="stale or expired"):
        build_trade_intent(
            accepted,
            asof=accepted.expires_at,
            setup_candidate=candidate,
            entry_episode=episode,
            account=account,
            account_snapshot_id="account-snapshot:1",
            risk_budget_id="risk-budget:1",
            entry_method_preferences=ENTRY_PREFERENCES,
        )
    with pytest.raises(TradeIntentError, match="exceeds the risk budget"):
        build_trade_intent(
            accepted,
            asof=T0,
            setup_candidate=candidate,
            entry_episode=episode,
            account=AccountState(
                equity=1_000.0,
                requested_risk_fraction=0.001,
                quantity=2,
                point_value=20.0,
            ),
            account_snapshot_id="account-snapshot:1",
            risk_budget_id="risk-budget:1",
            entry_method_preferences=ENTRY_PREFERENCES,
        )
