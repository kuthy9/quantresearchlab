from __future__ import annotations

from datetime import timedelta
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from smc_trader.model import Timeframe
from smc_trader.probability_admission import (
    AdmissionThresholds,
    ProbabilityPrediction,
    RollingAssignment,
    RollingSession,
    evaluate_probability_admission,
    validate_rolling_group_session_split,
    write_json_no_overwrite,
)
from smc_trader.probability_cohorts import (
    DOLCandidateSnapshot,
    EvidenceHistoryTransition,
    PATH_LABELS,
    PathCompetitionArchive,
    PathRiskClock,
    build_path_risk_intervals,
    derive_dol_candidate_hits,
    label_dol_candidate_set,
)
from smc_trader.probability_fit import (
    DOLChoiceSetSample,
    EvidenceContributionAtPrediction,
    PathProbabilitySample,
    PriorReversionSample,
    fit_competing_risk_life_table,
    fit_dol_softmax,
    fit_evidence_prior_reversion,
    fit_history_conditional_likelihood,
    fit_path_temperature_bias,
)
from smc_trader.structural_outcome import OutcomeBar


SOURCE_SHA = "a" * 64
MANIFEST_SHA = "b" * 64
RANKING_SHA = "c" * 64
T0 = pd.Timestamp("2024-06-03T10:00:00Z")


def _path_values(value: float) -> tuple[tuple[str, float], ...]:
    return tuple((path, value) for path in PATH_LABELS)


def _residual_archive() -> PathCompetitionArchive:
    return PathCompetitionArchive.residual_at_horizon(
        competition_set_id="competition:one",
        instrument_id="NQM4:13743",
        market_epoch_id="epoch:one",
        authority_structure_id="structure:one",
        horizon_id="session:one",
        formed_at=T0,
        common_expires_at=T0 + timedelta(minutes=3),
        source_event_ids=("bar:horizon",),
        split_role="development_fit",
        fold_id="fit:one",
    )


def test_residual_archive_and_real_bar_risk_intervals_are_sourced_and_deterministic() -> None:
    archive = _residual_archive()
    assert archive.realized_path == "residual_unknown"
    assert archive.terminal_cause == "common_horizon_without_registered_winner"
    assert archive.source_event_ids == ("bar:horizon",)
    assert archive == _residual_archive()
    assert archive.archive_id == _residual_archive().archive_id

    clocks = (
        PathRiskClock(
            asof=T0 + timedelta(minutes=1),
            real_completed_bar=True,
            evidence_history_id="none",
            correlation_cluster_id="cluster:one",
            market_state_id="state:one",
            source_event_ids=("bar:1",),
        ),
        PathRiskClock(
            asof=T0 + timedelta(minutes=2),
            real_completed_bar=False,
            evidence_history_id="none",
            correlation_cluster_id="cluster:one",
            market_state_id="state:one",
            source_event_ids=("clock:synthetic",),
        ),
        PathRiskClock(
            asof=T0 + timedelta(minutes=3),
            real_completed_bar=True,
            evidence_history_id="none",
            correlation_cluster_id="cluster:one",
            market_state_id="state:one",
            source_event_ids=("bar:horizon",),
        ),
    )
    intervals = build_path_risk_intervals(archive, clocks)
    assert len(intervals) == len(PATH_LABELS) * 3
    residual = [row for row in intervals if row.path == "residual_unknown"]
    assert [row.age_real_completed_bars for row in residual] == [1, 1, 2]
    assert residual[-1].terminal_status == "realized"
    assert "bar:horizon" in residual[-1].source_event_ids

    artifact = fit_competing_risk_life_table(
        intervals,
        source_dataset_sha256=SOURCE_SHA,
        manifest_sha256=MANIFEST_SHA,
    )
    assert artifact.real_at_risk_interval_count == len(PATH_LABELS) * 2
    assert artifact.ignored_synthetic_interval_count == len(PATH_LABELS)
    assert artifact == fit_competing_risk_life_table(
        intervals,
        source_dataset_sha256=SOURCE_SHA,
        manifest_sha256=MANIFEST_SHA,
    )


def _candidate(candidate_id: str, price: float) -> DOLCandidateSnapshot:
    return DOLCandidateSnapshot(
        competition_set_id="competition:dol",
        candidate_set_id="candidate-set:one",
        candidate_id=candidate_id,
        path="continuation",
        symbol="NQM4",
        instrument_id=13743,
        prediction_known_at=T0,
        common_expires_at=T0 + timedelta(minutes=5),
        candidate_eligible=True,
        candidate_feature_schema_id="dol-features:v1",
        raw_candidate_probability=0.4,
        target_price=price,
        source_event_ids=(f"level:{candidate_id}",),
        split_role="development_fit",
        fold_id="fit:one",
    )


def _bar(*, minute: int, low: float, high: float) -> OutcomeBar:
    return OutcomeBar(
        bar_event_id=f"bar:{minute}",
        symbol="NQM4",
        instrument_id=13743,
        timeframe=Timeframe.M1,
        known_at=T0 + timedelta(minutes=minute),
        open=low,
        high=high,
        low=low,
        close=high,
    )


def test_dol_first_hit_is_strictly_future_and_same_bar_multi_hit_is_censored() -> None:
    candidates = (_candidate("candidate:a", 100.0), _candidate("candidate:b", 101.0))
    hits = derive_dol_candidate_hits(
        candidates,
        (_bar(minute=1, low=99.0, high=102.0),),
    )
    labels = label_dol_candidate_set(
        candidates,
        hits,
        observed_through=T0 + timedelta(minutes=5),
        observation_source_event_ids=("bar:5",),
    )
    assert {row.censor_reason for row in labels} == {"ambiguous_same_bar"}
    assert {row.ambiguous_candidate_ids for row in labels} == {
        ("candidate:a", "candidate:b")
    }
    assert all(row.first_hit_candidate_id is None for row in labels)
    assert all(not row.no_target_before_horizon for row in labels)

    one_hit = derive_dol_candidate_hits(
        candidates,
        (_bar(minute=1, low=99.5, high=100.5),),
    )
    resolved = label_dol_candidate_set(
        candidates,
        one_hit,
        observed_through=T0 + timedelta(minutes=5),
        observation_source_event_ids=("bar:5",),
    )
    assert {row.first_hit_candidate_id for row in resolved} == {"candidate:a"}
    assert {row.censor_reason for row in resolved} == {None}

    no_target = label_dol_candidate_set(
        candidates,
        (),
        observed_through=T0 + timedelta(minutes=5),
        observation_source_event_ids=("bar:5",),
    )
    assert all(row.no_target_before_horizon for row in no_target)
    assert all(row.outcome_known_at == T0 + timedelta(minutes=5) for row in no_target)
    assert all(row.outcome_source_event_ids == ("bar:5",) for row in no_target)

    horizon_hit = derive_dol_candidate_hits(
        candidates,
        (_bar(minute=5, low=99.5, high=100.5),),
    )
    horizon_resolved = label_dol_candidate_set(
        candidates,
        horizon_hit,
        observed_through=T0 + timedelta(minutes=5),
        observation_source_event_ids=("bar:5",),
    )
    assert {row.first_hit_candidate_id for row in horizon_resolved} == {
        "candidate:a"
    }


def _history_row(index: int, path: str, next_history: str) -> EvidenceHistoryTransition:
    known = T0 + timedelta(seconds=index + 1)
    return EvidenceHistoryTransition(
        competition_set_id=f"competition:history:{index}",
        asof=known,
        known_at=known,
        common_expires_at=T0 + timedelta(hours=1),
        previous_history_id="none",
        evidence_history_id=next_history,
        evidence_rule_ids=(
            "acceptance_continuation"
            if next_history == "acceptance_only"
            else "displacement_impact",
        ),
        evidence_observed=True,
        correlation_cluster_id=f"cluster:{index}",
        market_state_id="state:pooled",
        realized_path=path,
        outcome_known_at=T0 + timedelta(minutes=30),
        censor_reason=None,
        split_role="development_fit",
        fold_id="fit:one",
        source_event_ids=(f"event:{index}",),
    )


def test_history_likelihood_is_jeffreys_smoothed_and_history_conditioned() -> None:
    rows = tuple(
        [_history_row(index, "continuation", "acceptance_only") for index in range(3)]
        + [_history_row(10, "continuation", "displacement_only")]
        + [_history_row(20 + index, "reversal", "displacement_only") for index in range(3)]
    )
    artifact = fit_history_conditional_likelihood(
        rows,
        source_dataset_sha256=SOURCE_SHA,
        manifest_sha256=MANIFEST_SHA,
    )
    assert artifact.likelihood("none", "acceptance_only", "continuation") == pytest.approx(
        3.5 / 5.5
    )
    assert sum(
        artifact.likelihood("none", state, "failed_breakout")
        for state in ("acceptance_only", "displacement_only", "same_clock_joint")
    ) == pytest.approx(1.0)
    assert artifact.artifact_id == fit_history_conditional_likelihood(
        rows,
        source_dataset_sha256=SOURCE_SHA,
        manifest_sha256=MANIFEST_SHA,
    ).artifact_id


def _reversion_sample(age: int, realized: str, index: int) -> PriorReversionSample:
    increments = {path: 0.0 for path in PATH_LABELS}
    increments["continuation"] = 2.0
    return PriorReversionSample(
        competition_set_id=f"competition:reversion:{index}",
        prediction_known_at=T0 + timedelta(minutes=index),
        outcome_known_at=T0 + timedelta(hours=2),
        realized_path=realized,
        prior_log_weights=_path_values(0.0),
        contributions=(
            EvidenceContributionAtPrediction(
                contribution_id=f"contribution:{index}",
                age_real_completed_bars=age,
                path_log_likelihoods=tuple(increments.items()),
            ),
        ),
        split_role="development_fit",
        fold_id="fit:one",
    )


def test_evidence_half_life_reverts_to_prior_instead_of_subtracting_drift() -> None:
    samples = (
        _reversion_sample(0, "continuation", 0),
        _reversion_sample(100, "reversal", 1),
    )
    artifact = fit_evidence_prior_reversion(
        samples,
        source_dataset_sha256=SOURCE_SHA,
        manifest_sha256=MANIFEST_SHA,
        candidate_half_lives=(5.0,),
    )
    fresh = artifact.probabilities(samples[0])
    old = artifact.probabilities(samples[1])
    uniform = 1.0 / len(PATH_LABELS)
    assert fresh["continuation"] > old["continuation"]
    assert abs(old["continuation"] - uniform) < abs(fresh["continuation"] - uniform)


def test_path_calibration_fixes_residual_bias_gauge() -> None:
    raw = _path_values(1.0 / len(PATH_LABELS))
    samples = tuple(
        PathProbabilitySample(
            competition_set_id=f"competition:path-cal:{index}",
            prediction_known_at=T0 + timedelta(minutes=index),
            outcome_known_at=T0 + timedelta(hours=1),
            realized_path="continuation" if index < 9 else "reversal",
            raw_probabilities=raw,
            split_role="calibration",
            fold_id="calibration:one",
        )
        for index in range(12)
    )
    artifact = fit_path_temperature_bias(
        samples,
        source_dataset_sha256=SOURCE_SHA,
        manifest_sha256=MANIFEST_SHA,
        source_model_artifact_id="path-model:one",
    )
    biases = dict(artifact.path_log_biases)
    assert biases["residual_unknown"] == 0.0
    assert biases["continuation"] > biases["reversal"]
    assert sum(artifact.probabilities(dict(raw)).values()) == pytest.approx(1.0)


def test_dol_softmax_fixes_candidate_gauge_and_keeps_no_target() -> None:
    samples = []
    for index in range(20):
        path = "continuation" if index < 10 else "residual_unknown"
        outcome = "candidate:a" if path == "continuation" else "no_target_before_common_horizon"
        samples.append(
            DOLChoiceSetSample(
                competition_set_id=f"competition:dol-fit:{index}",
                candidate_set_id=f"candidate-set:{index}",
                path=path,
                prediction_known_at=T0 + timedelta(minutes=index),
                outcome_known_at=T0 + timedelta(hours=1),
                candidate_scores=(("candidate:a", 2.0), ("candidate:b", 0.0)),
                outcome_id=outcome,
                split_role="development_fit",
                fold_id="fit:one",
            )
        )
    artifact = fit_dol_softmax(
        samples,
        source_dataset_sha256=SOURCE_SHA,
        manifest_sha256=MANIFEST_SHA,
        source_ranking_fingerprint=RANKING_SHA,
    )
    assert artifact.candidate_logit_scale >= 0.0
    assert set(value for _, value in artifact.candidate_path_log_weight_adjustment) == {0.0}
    continuation = artifact.probabilities(
        "continuation", (("candidate:a", 2.0), ("candidate:b", 0.0))
    )
    residual = artifact.probabilities(
        "residual_unknown", (("candidate:a", 2.0), ("candidate:b", 0.0))
    )
    assert continuation["candidate:a"] > continuation["candidate:b"]
    assert residual["no_target_before_common_horizon"] > continuation[
        "no_target_before_common_horizon"
    ]
    assert sum(residual.values()) == pytest.approx(1.0)


def test_rolling_split_requires_group_session_purge_and_embargo() -> None:
    sessions = tuple(
        RollingSession(
            session_id=f"session:{index:02d}",
            start=pd.Timestamp("2024-01-01T00:00:00Z") + timedelta(days=index),
            end_exclusive=pd.Timestamp("2024-01-01T23:00:00Z")
            + timedelta(days=index),
        )
        for index in range(25)
    )
    assignments = (
        RollingAssignment(
            competition_set_id="competition:fit",
            market_epoch_id="epoch:fit",
            session_id="session:00",
            prediction_known_at=sessions[0].start + timedelta(hours=1),
            outcome_known_at=sessions[0].start + timedelta(hours=2),
            role="fit",
            fold_id="fold:one",
        ),
        RollingAssignment(
            competition_set_id="competition:validation",
            market_epoch_id="epoch:validation",
            session_id="session:20",
            prediction_known_at=sessions[20].start + timedelta(hours=1),
            outcome_known_at=sessions[20].start + timedelta(hours=2),
            role="validation",
            fold_id="fold:one",
        ),
    )
    validate_rolling_group_session_split(assignments, sessions)
    leaking = assignments + (
        RollingAssignment(
            competition_set_id="competition:late-fit",
            market_epoch_id="epoch:late-fit",
            session_id="session:19",
            prediction_known_at=sessions[19].start + timedelta(hours=1),
            outcome_known_at=sessions[19].start + timedelta(hours=2),
            role="fit",
            fold_id="fold:one",
        ),
    )
    with pytest.raises(ValueError, match="purge|embargo"):
        validate_rolling_group_session_split(leaking, sessions)


def test_bootstrap_admission_is_hash_bound_shadow_only_and_no_overwrite(tmp_path: Path) -> None:
    records = []
    for index in range(8):
        realized = "a" if index % 2 == 0 else "b"
        model = (("a", 0.9), ("b", 0.1)) if realized == "a" else (("a", 0.1), ("b", 0.9))
        records.append(
            ProbabilityPrediction(
                unit_id=f"unit:{index}",
                cluster_id=f"session:{index // 2}",
                fold_id=f"fold:{index // 4}",
                realized_label=realized,
                support_label=realized,
                model_probabilities=model,
                baseline_probabilities=(("a", 0.5), ("b", 0.5)),
            )
        )
    thresholds = AdmissionThresholds(
        minimum_resolved_units=8,
        minimum_support_units=4,
        minimum_prediction_coverage=1.0,
        maximum_ece=0.11,
        maximum_fold_log_loss_degradation=0.05,
        minimum_improving_fold_fraction=1.0,
        bootstrap_replicates=100,
    )
    receipt = evaluate_probability_admission(
        records,
        artifact_kind="path_probability",
        model_artifact_id="model:one",
        source_dataset_sha256=SOURCE_SHA,
        manifest_sha256=MANIFEST_SHA,
        cohort_identity_sha256="d" * 64,
        cohort_role="rolling_oof",
        prediction_coverage=1.0,
        required_support_labels=("a", "b"),
        thresholds=thresholds,
    )
    assert receipt.admitted is True
    assert receipt.authority == "shadow_only"
    assert receipt.action_authority is False
    assert receipt.receipt_id == evaluate_probability_admission(
        records,
        artifact_kind="path_probability",
        model_artifact_id="model:one",
        source_dataset_sha256=SOURCE_SHA,
        manifest_sha256=MANIFEST_SHA,
        cohort_identity_sha256="d" * 64,
        cohort_role="rolling_oof",
        prediction_coverage=1.0,
        required_support_labels=("a", "b"),
        thresholds=thresholds,
    ).receipt_id
    destination = tmp_path / "receipt.json"
    write_json_no_overwrite(destination, receipt.to_dict())
    with pytest.raises(FileExistsError):
        write_json_no_overwrite(destination, receipt.to_dict())


def test_preregistration_freezes_runtime_words_and_contains_no_artifact_results() -> None:
    path = Path("configs/phase7_foundation_v2_empirical.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    runtime = payload["runtime_resolution_contract"]
    assert runtime["protocol_version"] == (
        "path_runtime_resolution_phase7_foundation_v2_empirical_v1.0"
    )
    assert runtime["residual_unknown_winner_rule"] == (
        "clean_common_horizon_without_registered_winner_realizes_residual_unknown"
    )
    assert runtime["residual_unknown_factual_outcome_reason"] == (
        "common_horizon_without_registered_winner"
    )
    assert runtime["expiry_rules"]["residual_unknown"] == "shared_common_horizon_only"
    assert set(payload["artifact_slots"].values()) == {None}
    assert payload["authority"]["sealed_oos_opened"] is False

    manifest_path = Path(
        "experiments/manifests/phase7_foundation_v2_empirical_preregistration.yaml"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["authority"]["artifact_fit_authorized"] is False
    assert manifest["authority"]["sealed_oos_reveal_authorized"] is False
    assert set(manifest["artifact_bindings"].values()) == {None}
    for binding in (
        manifest["protocol_binding"],
        *manifest["implementation_bindings"].values(),
    ):
        digest = hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest()
        assert digest == binding["sha256"]
