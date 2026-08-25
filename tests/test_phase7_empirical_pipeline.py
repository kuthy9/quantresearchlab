from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_phase7_empirical_pipeline as pipeline
from smc_trader.path_belief import create_path_competition_set, load_path_belief_protocol
from smc_trader.probability_admission import (
    AdmissionThresholds,
    ProbabilityAdmissionReceipt,
)
from smc_trader.probability_cohorts import (
    DOLCandidateOutcomeRow,
    EvidenceHistoryTransition,
    PATH_LABELS,
    PathCompetitionArchive,
)
from smc_trader.probability_fit import fit_history_conditional_likelihood


T0 = pd.Timestamp("2024-06-03T00:00:00Z")
SHA_A = "a" * 64
SHA_B = "b" * 64
COMMIT = "c" * 40
PYTHON_IDENTITY = f"{pipeline.sys.version_info.major}.{pipeline.sys.version_info.minor}"


def _empty_cohorts(
    *,
    archives: tuple[PathCompetitionArchive, ...] = (),
    predictions: tuple[pipeline._PredictionSeed, ...] = (),
    dol_snapshots: tuple[pipeline.DOLCandidateSnapshot, ...] = (),
    dol_outcomes: tuple[DOLCandidateOutcomeRow, ...] = (),
    dol_groups: tuple[pipeline._DOLGroup, ...] = (),
) -> pipeline.MaterializedCohorts:
    return pipeline.MaterializedCohorts(
        archives=archives,
        risk_intervals=(),
        history_transitions=(),
        dol_snapshots=dol_snapshots,
        dol_outcomes=dol_outcomes,
        predictions=predictions,
        dol_groups=dol_groups,
        outcome_bars=(),
        counts_by_window={},
    )


def _output_manifest(
    tmp_path: Path,
    *,
    run_id: str,
    fit_authorized: bool = False,
    inputs: tuple[pipeline.InputWindowSpec, ...] = (),
) -> pipeline.RunManifest:
    return pipeline.RunManifest(
        run_id=run_id,
        source_path=tmp_path / "manifest.json",
        source_sha256=SHA_A,
        model_config_path=pipeline.MODEL_CONFIG,
        model_config_sha256=pipeline._sha256_file(pipeline.MODEL_CONFIG),
        phase7_protocol_sha256=pipeline._sha256_file(pipeline.PHASE7_PROTOCOL),
        preregistration_sha256=pipeline._sha256_file(pipeline.PREREGISTRATION),
        repository_commit=COMMIT,
        python_major_minor=PYTHON_IDENTITY,
        code_bundle_identity=SHA_B,
        materialization_authorized=True,
        fit_and_validate_authorized=fit_authorized,
        inputs=inputs,
    )


def _fitted_source_sha(manifest: pipeline.RunManifest) -> str:
    return pipeline.canonical_sha256(
        {
            "schema_version": pipeline.PIPELINE_SCHEMA_VERSION,
            "inputs": tuple(
                (spec.window_id, spec.source_sha256, spec.row_count)
                for spec in manifest.inputs
                if not spec.registered.rolling_diagnostic_only
            ),
        }
    )


def _specs(root: Path, *windows: str) -> tuple[pipeline.InputWindowSpec, ...]:
    return tuple(
        pipeline.InputWindowSpec(
            registered=pipeline.REGISTERED_WINDOWS[window],
            path=root / f"{window}.jsonl",
            source_sha256=SHA_A if window == "W1" else SHA_B,
            row_count=pipeline.REGISTERED_WINDOWS[window].expected_rows,
            fold_id=window,
        )
        for window in windows
    )


def _fitted(
    manifest: pipeline.RunManifest,
    admissions: dict[str, object],
) -> dict[str, object]:
    return {
        "source_dataset_sha256": _fitted_source_sha(manifest),
        "artifacts": {},
        "admission_receipts": admissions,
        "artifact_blockers": {},
        "artifact_statuses": {},
        "admission_gate": {"state": "closed", "reasons": ["NO_ACTION_AUTHORITY"]},
    }


def _publish(
    destination: Path,
    manifest: pipeline.RunManifest,
    *,
    mode: str = "materialize-only",
    fitted: dict[str, object] | None = None,
) -> dict[str, object]:
    return dict(
        pipeline.write_pipeline_outputs(
            destination,
            manifest=manifest,
            cohorts=_empty_cohorts(),
            mode=mode,
            include_rolling_diagnostics=False,
            fitted=fitted,
        )
    )


def _archive(
    index: int,
    path: str,
    *,
    session: int = 1,
    competition_set_id: str | None = None,
) -> PathCompetitionArchive:
    formed = T0 + timedelta(days=session - 1, minutes=index)
    terminal = formed + timedelta(minutes=5)
    return PathCompetitionArchive(
        competition_set_id=competition_set_id or f"competition:{index}",
        instrument_id="NQM4:13743",
        market_epoch_id=f"epoch:{session}",
        authority_structure_id=f"structure:{index}",
        horizon_id=f"market-session:{session}",
        formed_at=formed,
        common_expires_at=formed + timedelta(minutes=10),
        terminal_at=terminal,
        terminal_status="realized",
        terminal_cause=f"registered:{path}",
        realized_path=path,
        outcome_known_at=terminal,
        source_event_ids=(f"event:{index}",),
        censor_reason=None,
        split_role="development_fit",
        fold_id="W1",
    )


def _prediction(archive: PathCompetitionArchive) -> pipeline._PredictionSeed:
    return pipeline._PredictionSeed(
        competition_set_id=archive.competition_set_id,
        window_id="W1",
        prediction_known_at=archive.formed_at,
        common_expires_at=archive.common_expires_at,
        raw_probabilities=tuple((path, 1.0 / len(PATH_LABELS)) for path in PATH_LABELS),
        prior_log_weights=tuple((path, 0.0) for path in PATH_LABELS),
        history_seeds=(),
        real_completed_bar_count=0,
        cluster_id=f"cluster:{archive.horizon_id}",
        split_role="development_fit",
        fold_id="W1",
        prediction_reason=("competition_formation",),
    )


def _history_row(index: int, path: str, next_state: str) -> EvidenceHistoryTransition:
    known = T0 + timedelta(seconds=index + 1)
    rule = (
        "acceptance_continuation"
        if next_state == "acceptance_only"
        else "displacement_impact"
    )
    return EvidenceHistoryTransition(
        competition_set_id=f"history-competition:{index}",
        asof=known,
        known_at=known,
        common_expires_at=T0 + timedelta(hours=2),
        previous_history_id="none",
        evidence_history_id=next_state,
        evidence_rule_ids=(rule,),
        evidence_observed=True,
        correlation_cluster_id=f"cluster:{index}",
        market_state_id="state:one",
        realized_path=path,
        outcome_known_at=T0 + timedelta(hours=1),
        censor_reason=None,
        split_role="development_fit",
        fold_id="W1",
        source_event_ids=(f"event:{index}",),
    )


def _dol_group(
    candidate_set_id: str,
    *,
    target_prices: tuple[float, ...],
    competition_set_id: str = "competition:dol",
    window_id: str = "W1",
    split_role: str = "development_fit",
    fold_id: str = "W1",
    prediction_known_at: pd.Timestamp = T0,
    cluster_id: str = "cluster:dol",
) -> pipeline._DOLGroup:
    snapshots = tuple(
        pipeline.DOLCandidateSnapshot(
            competition_set_id=competition_set_id,
            candidate_set_id=candidate_set_id,
            candidate_id=f"{candidate_set_id}:candidate:{index}",
            path="continuation",
            symbol="NQM4",
            instrument_id=13743,
            prediction_known_at=prediction_known_at,
            common_expires_at=prediction_known_at + timedelta(minutes=3),
            candidate_eligible=True,
            candidate_feature_schema_id="dol-features:test",
            raw_candidate_probability=1.0 / len(target_prices),
            target_price=price,
            source_event_ids=(f"level:{candidate_set_id}:{index}",),
            split_role=split_role,
            fold_id=fold_id,
        )
        for index, price in enumerate(target_prices)
    )
    return pipeline._DOLGroup(
        window_id=window_id,
        snapshots=snapshots,
        candidate_scores=tuple(
            (snapshot.candidate_id, 0.0) for snapshot in snapshots
        ),
        cluster_id=cluster_id,
    )


def _dol_labels(
    group: pipeline._DOLGroup,
    *,
    no_target: bool,
) -> tuple[DOLCandidateOutcomeRow, ...]:
    outcome_known_at = (
        group.snapshots[0].common_expires_at
        if no_target
        else group.snapshots[0].prediction_known_at + timedelta(minutes=1)
    )
    return tuple(
        DOLCandidateOutcomeRow(
            snapshot=snapshot,
            first_hit_candidate_id=(
                None if no_target else group.snapshots[0].candidate_id
            ),
            no_target_before_horizon=no_target,
            outcome_known_at=outcome_known_at,
            censor_reason=None,
            ambiguous_candidate_ids=(),
            outcome_source_event_ids=(f"bar:{group.candidate_set_id}",),
        )
        for snapshot in group.snapshots
    )


def _outcome_bar(minute: int, *, low: float, high: float) -> pipeline.OutcomeBar:
    return pipeline.OutcomeBar(
        bar_event_id=f"bar:{minute}",
        symbol="NQM4",
        instrument_id=13743,
        timeframe=pipeline.Timeframe.M1,
        known_at=T0 + timedelta(minutes=minute),
        open=low,
        high=high,
        low=low,
        close=high,
    )


def test_history_converter_emits_five_state_transition_rules_not_two_marginals() -> None:
    rows = tuple(
        _history_row(index, path, state)
        for index, (path, state) in enumerate(
            (
                ("continuation", "acceptance_only"),
                ("continuation", "displacement_only"),
                ("reversal", "acceptance_only"),
            )
        )
    )
    artifact = fit_history_conditional_likelihood(
        rows,
        source_dataset_sha256=SHA_A,
        manifest_sha256=SHA_B,
    )
    payload = pipeline.history_transition_rules_payload(artifact)
    assert payload["legacy_marginal_rule_projection_allowed"] is False
    assert payload["one_rule_equals_one_history_conditioned_contribution"] is True
    transitions = payload["transitions"]
    assert {row["rule_id"] for row in transitions} == {
        "history_none_to_acceptance_only",
        "history_none_to_displacement_only",
        "history_none_to_same_clock_joint",
        "history_acceptance_only_to_acceptance_then_displacement",
        "history_displacement_only_to_displacement_then_acceptance",
    }
    assert {row["evidence_family"] for row in transitions} == {
        "path_evidence_history_transition"
    }
    assert all(set(row["conditional_likelihood_by_path"]) == set(PATH_LABELS) for row in transitions)


def test_path_base_rate_is_jeffreys_fitted_with_residual_gauge() -> None:
    archives = (
        _archive(1, "continuation"),
        _archive(2, "continuation"),
        _archive(3, "reversal"),
    )
    artifact = pipeline.fit_path_base_rate_prior(
        _empty_cohorts(archives=archives),
        source_dataset_sha256=SHA_A,
        manifest_sha256=SHA_B,
    )
    denominator = 3 + 0.5 * len(PATH_LABELS)
    assert artifact["probabilities"]["continuation"] == pytest.approx(2.5 / denominator)
    assert artifact["probabilities"]["residual_unknown"] == pytest.approx(0.5 / denominator)
    assert artifact["prior_log_weights"]["residual_unknown"] == 0.0
    assert artifact["prior_log_weights"]["continuation"] > 0.0


def test_w1_inner_crossfit_is_forward_whole_session_and_not_formal_oof() -> None:
    archives = (
        _archive(1, "continuation", session=1),
        _archive(2, "reversal", session=2),
        _archive(3, "balance", session=3),
    )
    cohorts = _empty_cohorts(
        archives=archives,
        predictions=tuple(_prediction(row) for row in archives),
    )
    result = pipeline._w1_inner_rolling_crossfit(
        cohorts,
        source_dataset_sha256=SHA_A,
        manifest_sha256=SHA_B,
    )
    assert result["whole_session_split"] is True
    assert result["whole_generation_split"] is True
    assert result["formal_rolling_oof"] is False
    predictions = result["path_predictions"]
    assert {row["held_session"] for row in predictions} == {
        "market-session:2",
        "market-session:3",
    }
    assert all(row["held_session"] not in row["train_sessions"] for row in predictions)
    assert len(result["path_calibration_samples"]) == 2


def test_dol_temperature_uses_inner_w1_distributions_and_w2_stays_closed() -> None:
    source_fingerprint = "d" * 64
    inner_rows = (
        pipeline._dol_temperature_observation(
            candidate_set_id="candidate-set:inner-hit",
            competition_set_id="competition:inner-hit",
            prediction_known_at=T0,
            outcome_known_at=T0 + timedelta(minutes=1),
            probabilities={
                "candidate:inner-hit": 0.75,
                pipeline.NO_TARGET_OUTCOME: 0.25,
            },
            outcome_id="candidate:inner-hit",
            window_id="W1",
            fold_id="inner-fold:1",
            cluster_id="cluster:inner-hit",
            source_model_fingerprint=source_fingerprint,
        ),
        pipeline._dol_temperature_observation(
            candidate_set_id="candidate-set:inner-none",
            competition_set_id="competition:inner-none",
            prediction_known_at=T0,
            outcome_known_at=T0 + timedelta(minutes=1),
            probabilities={
                "candidate:inner-none": 0.75,
                pipeline.NO_TARGET_OUTCOME: 0.25,
            },
            outcome_id=pipeline.NO_TARGET_OUTCOME,
            window_id="W1",
            fold_id="inner-fold:1",
            cluster_id="cluster:inner-none",
            source_model_fingerprint=source_fingerprint,
        ),
    )
    fit_cohort = pipeline._signal_outcome_cohort(
        inner_rows,
        cohort_name="test_W1_inner_dol_temperature",
        cohort_role="development_cross_fit",
        window_id="W1",
        source_dataset_sha256=SHA_A,
        manifest_sha256=SHA_B,
        split_protocol_sha256=SHA_A,
        eligible_units=2,
    )
    lineage = pipeline.DOLModelLineage(
        source_dol_protocol_fingerprint=pipeline.DOL_PROTOCOL_FINGERPRINT,
        source_dol_model_version="test-dol-v1",
        source_dol_model_fingerprint=source_fingerprint,
        source_path_protocol_fingerprint=pipeline.PATH_PROTOCOL_FINGERPRINT,
        source_path_model_version="test-path-v1",
    )
    artifact = pipeline.fit_dol_temperature(
        inner_rows,
        cohort=fit_cohort,
        lineage=lineage,
    )
    assert artifact.fit_status == "fitted_not_admitted"
    assert artifact.admission_status == "CLOSED"

    validation_rows = (
        pipeline._dol_temperature_observation(
            candidate_set_id="candidate-set:validation-hit",
            competition_set_id="competition:validation-hit",
            prediction_known_at=T0 + timedelta(minutes=2),
            outcome_known_at=T0 + timedelta(minutes=3),
            probabilities={
                "candidate:validation-hit": 0.6,
                pipeline.NO_TARGET_OUTCOME: 0.4,
            },
            outcome_id="candidate:validation-hit",
            window_id="W2",
            fold_id="W2",
            cluster_id="cluster:validation-hit",
            source_model_fingerprint=source_fingerprint,
        ),
        pipeline._dol_temperature_observation(
            candidate_set_id="candidate-set:validation-none",
            competition_set_id="competition:validation-none",
            prediction_known_at=T0 + timedelta(minutes=2),
            outcome_known_at=T0 + timedelta(minutes=3),
            probabilities={
                "candidate:validation-none": 0.6,
                pipeline.NO_TARGET_OUTCOME: 0.4,
            },
            outcome_id=pipeline.NO_TARGET_OUTCOME,
            window_id="W2",
            fold_id="W2",
            cluster_id="cluster:validation-none",
            source_model_fingerprint=source_fingerprint,
        ),
    )
    validation_cohort = pipeline._signal_outcome_cohort(
        validation_rows,
        cohort_name="test_W2_dol_temperature_validation",
        cohort_role="historical_validation",
        window_id="W2",
        source_dataset_sha256=SHA_A,
        manifest_sha256=SHA_B,
        split_protocol_sha256=SHA_A,
        eligible_units=2,
    )
    receipt = pipeline.evaluate_dol_temperature_admission(
        artifact,
        validation_rows,
        cohort=validation_cohort,
    )
    assert receipt.status == "CLOSED"
    assert receipt.admitted is False
    assert "COHORT_NOT_ROLLING_OOF" in receipt.blockers
    assert "JUNE_2024_DEVELOPMENT_PERMANENTLY_CLOSED" in receipt.blockers


def test_fit_pipeline_attempts_dol_temperature_and_keeps_june_closed() -> None:
    archives = tuple(
        _archive(
            index,
            path,
            session=session,
            competition_set_id=f"competition:dol:{session}",
        )
        for index, session, path in (
            (1, 1, "continuation"),
            (2, 2, "reversal"),
            (3, 3, "balance"),
        )
    )
    w1_groups = tuple(
        _dol_group(
            f"candidate-set:W1:{session}",
            target_prices=(100.0 + session,),
            competition_set_id=archive.competition_set_id,
            prediction_known_at=archive.formed_at,
            cluster_id=f"cluster:W1:{session}",
        )
        for session, archive in enumerate(archives, start=1)
    )
    w2_groups = (
        _dol_group(
            "candidate-set:W2:hit",
            target_prices=(110.0,),
            competition_set_id="competition:W2:hit",
            window_id="W2",
            split_role="historical_validation",
            fold_id="W2",
            prediction_known_at=T0 + timedelta(days=7),
            cluster_id="cluster:W2:hit",
        ),
        _dol_group(
            "candidate-set:W2:none",
            target_prices=(120.0,),
            competition_set_id="competition:W2:none",
            window_id="W2",
            split_role="historical_validation",
            fold_id="W2",
            prediction_known_at=T0 + timedelta(days=7, minutes=10),
            cluster_id="cluster:W2:none",
        ),
    )
    groups = (*w1_groups, *w2_groups)
    outcomes = (
        *_dol_labels(w1_groups[0], no_target=False),
        *_dol_labels(w1_groups[1], no_target=False),
        *_dol_labels(w1_groups[2], no_target=True),
        *_dol_labels(w2_groups[0], no_target=False),
        *_dol_labels(w2_groups[1], no_target=True),
    )
    cohorts = _empty_cohorts(
        archives=archives,
        predictions=tuple(_prediction(row) for row in archives),
        dol_snapshots=tuple(
            snapshot for group in groups for snapshot in group.snapshots
        ),
        dol_outcomes=outcomes,
        dol_groups=groups,
    )
    specs = tuple(
        pipeline.InputWindowSpec(
            registered=pipeline.REGISTERED_WINDOWS[window],
            path=Path(f"{window}.jsonl"),
            source_sha256=("a" if window == "W1" else "b") * 64,
            row_count=pipeline.REGISTERED_WINDOWS[window].expected_rows,
            fold_id=window,
        )
        for window in ("W1", "W2")
    )
    manifest = pipeline.RunManifest(
        run_id="dol-temperature-attempt",
        source_path=Path("manifest.json"),
        source_sha256=SHA_A,
        model_config_path=pipeline.MODEL_CONFIG,
        model_config_sha256=pipeline._sha256_file(pipeline.MODEL_CONFIG),
        phase7_protocol_sha256=pipeline._sha256_file(pipeline.PHASE7_PROTOCOL),
        preregistration_sha256=pipeline._sha256_file(pipeline.PREREGISTRATION),
        repository_commit=COMMIT,
        python_major_minor=PYTHON_IDENTITY,
        code_bundle_identity=SHA_B,
        materialization_authorized=True,
        fit_and_validate_authorized=True,
        inputs=specs,
    )
    result = pipeline.fit_and_validate(
        manifest,
        cohorts,
        include_rolling_diagnostics=False,
    )

    artifact = result["artifacts"]["dol_temperature_calibration"]
    receipt = result["admission_receipts"]["dol_temperature_W2"]
    assert artifact["fit_status"] == "fitted_not_admitted"
    assert artifact["admission_status"] == "CLOSED"
    assert artifact["fit_sample_count"] == 2
    assert receipt["status"] == "CLOSED"
    assert receipt["validation_cohort_role"] == "historical_validation"
    assert "COHORT_NOT_ROLLING_OOF" in receipt["blockers"]
    assert result["artifact_statuses"]["dol_temperature_calibration"][
        "status"
    ] == "fitted_not_admitted"
    assert result["empirical_authority"] is False
    assert json.loads(pipeline._canonical_bytes(result))["admission_gate"][
        "state"
    ] == "closed"


def test_temporal_candidate_expands_piecewise_survival_and_never_expires_residual() -> None:
    cells = tuple(
        SimpleNamespace(
            path=path,
            bin_index=0,
            age_start_bar=1,
            at_risk_intervals=30,
            cause_hazards=(
                ("expired", 0.1),
                ("falsified", 0.1),
                ("realized", 0.2),
                ("scope_superseded", 0.1),
            ),
            no_event_probability=0.5,
        )
        for path in PATH_LABELS
    )
    temporal = SimpleNamespace(
        artifact_id="temporal:one",
        age_bin_ends=(None,),
        cells=cells,
    )
    reversion = SimpleNamespace(
        artifact_id="reversion:one",
        path_half_lives=tuple((path, 15.0) for path in PATH_LABELS),
    )
    payload = pipeline.temporal_runtime_candidate_payload(temporal, reversion)
    assert payload["hypothesis_expiry_bars"]["continuation"] == 4
    assert payload["hypothesis_expiry_bars"]["residual_unknown"] is None
    assert payload["prior_reversion_half_life_real_bars"]["reversal"] == 15.0
    assert payload["status"] == "research_runtime_candidate_not_admitted"

    under_supported = pipeline.temporal_runtime_candidate_payload(
        SimpleNamespace(
            artifact_id="temporal:thin",
            age_bin_ends=(None,),
            cells=tuple(
                SimpleNamespace(
                    path=path,
                    bin_index=0,
                    age_start_bar=1,
                    at_risk_intervals=1,
                    cause_hazards=(),
                    no_event_probability=0.5,
                )
                for path in PATH_LABELS
            ),
        ),
        None,
    )
    assert under_supported["hypothesis_expiry_bars"]["continuation"] is None
    assert "insufficient_life_table_support" in under_supported["support_blockers"][
        "continuation"
    ]


def test_dol_candidate_sets_remain_frozen_until_exact_common_horizon() -> None:
    first_hit = _dol_group("candidate-set:hit", target_prices=(100.0, 110.0))
    no_target = _dol_group("candidate-set:none", target_prices=(200.0,))
    builder = pipeline.Phase7CohortBuilder()
    builder._window_bars = [
        _outcome_bar(1, low=90.0, high=91.0),
        _outcome_bar(2, low=99.0, high=101.0),
        _outcome_bar(3, low=92.0, high=93.0),
    ]
    builder._pending_dol = [first_hit, no_target]
    builder._settle_due_dol(T0 + timedelta(minutes=3))

    assert builder._pending_dol == []
    hit_rows = tuple(
        row
        for row in builder._dol_outcomes
        if row.snapshot.candidate_set_id == "candidate-set:hit"
    )
    none_rows = tuple(
        row
        for row in builder._dol_outcomes
        if row.snapshot.candidate_set_id == "candidate-set:none"
    )
    assert {row.first_hit_candidate_id for row in hit_rows} == {
        "candidate-set:hit:candidate:0"
    }
    assert {row.outcome_known_at for row in hit_rows} == {
        T0 + timedelta(minutes=2)
    }
    assert all(row.no_target_before_horizon for row in none_rows)
    assert {row.outcome_source_event_ids for row in none_rows} == {("bar:3",)}


def test_dol_scope_supersession_excludes_new_scope_terminal_bar() -> None:
    group = _dol_group("candidate-set:superseded", target_prices=(100.0,))
    builder = pipeline.Phase7CohortBuilder()
    builder._window_bars = [
        _outcome_bar(1, low=90.0, high=91.0),
        _outcome_bar(2, low=99.0, high=101.0),
    ]
    builder._pending_dol = [group]
    builder._censor_pending_dol(
        terminal_at=T0 + timedelta(minutes=2),
        terminal_cause="scope_superseded",
        source_event_ids=("bar:2",),
    )
    assert builder._pending_dol == []
    assert {row.first_hit_candidate_id for row in builder._dol_outcomes} == {None}
    assert {row.censor_reason for row in builder._dol_outcomes} == {
        "scope_superseded"
    }


def _manifest(tmp_path: Path, *, input_path: str) -> Path:
    payload = {
        "schema_version": pipeline.RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": "phase7-test",
        "authority": {
            "cohort_materialization_authorized": True,
            "fit_and_validate_authorized": False,
            "sealed_source_read_authorized": False,
            "sealed_oos_opened": False,
            "action_authority": False,
        },
        "bindings": {
            "model_config": {
                "path": "configs/model.json",
                "sha256": pipeline._sha256_file(pipeline.MODEL_CONFIG),
            },
            "phase7_protocol": {
                "path": "configs/phase7_foundation_v2_empirical.json",
                "sha256": pipeline._sha256_file(pipeline.PHASE7_PROTOCOL),
            },
            "preregistration_manifest": {
                "path": "experiments/manifests/phase7_foundation_v2_empirical_preregistration.yaml",
                "sha256": pipeline._sha256_file(pipeline.PREREGISTRATION),
            },
            "foundation_identity": pipeline.FOUNDATION_IDENTITY,
            "path_protocol_fingerprint": pipeline.PATH_PROTOCOL_FINGERPRINT,
            "dol_protocol_fingerprint": pipeline.DOL_PROTOCOL_FINGERPRINT,
            "dol_ranking_fingerprint": pipeline.DOL_RANKING_FINGERPRINT,
            "code_bundle": pipeline.current_code_bundle_binding(
                require_clean_repository=False
            ),
        },
        "inputs": [
            {
                "window_id": "W1",
                "path": input_path,
                "sha256": SHA_A,
                "row_count": 6900,
                "split_role": "development_fit",
                "fold_id": "W1",
                "purpose": "fit",
                "rolling_diagnostic_only": False,
                "source_authority": "opened_june_2024_development",
                "contains_sealed_oos": False,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("link_kind", ("final", "parent", "missing", "loop"))
def test_direct_file_guard_rejects_symlinks_without_resolve_read_or_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    link_kind: str,
) -> None:
    opened = tmp_path / "opened"
    opened.mkdir()
    source = opened / "w1.jsonl"
    source.write_text("{}\n")
    final_link, missing_link, loop_link = (
        tmp_path / name for name in ("final.jsonl", "missing.jsonl", "loop.jsonl")
    )
    final_link.symlink_to(source)
    missing_link.symlink_to(tmp_path / "absent.jsonl")
    loop_link.symlink_to(loop_link)
    parent_link = tmp_path / "opened_alias"
    parent_link.symlink_to(opened, target_is_directory=True)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("symlink target was resolved, read, or hashed")

    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(pipeline, "_sha256_file", forbidden)
    guarded = {
        "final": final_link,
        "parent": parent_link / "w1.jsonl",
        "missing": missing_link,
        "loop": loop_link,
    }[link_kind]
    with pytest.raises(pipeline.Phase7PipelineError) as captured:
        pipeline._trusted_regular_file(guarded, name="W1 input")
    assert str(captured.value) == "W1 input must be a direct regular file"


def _clock(start: pd.Timestamp, end: pd.Timestamp, ordinal: int) -> SimpleNamespace:
    return SimpleNamespace(
        bar=SimpleNamespace(start=start, end=end, symbol="NQM4", instrument_id=13743),
        feed_event_id=f"feed:{ordinal}",
        input_digest=f"digest:{ordinal}",
        approved_intents=(),
        execution_events=(),
        account=SimpleNamespace(position=None),
    )


@pytest.mark.parametrize("bad_case", ("first", "last", "duration"))
def test_input_clock_validation_covers_boundaries_and_one_minute_duration(
    bad_case: str,
) -> None:
    registered = pipeline.REGISTERED_WINDOWS["W1"]
    w2 = pipeline.REGISTERED_WINDOWS["W2"]
    assert (registered.first_decision_clock, registered.last_decision_clock) == (
        pd.Timestamp("2024-06-02T22:01:00Z"),
        pd.Timestamp("2024-06-07T21:00:00Z"),
    )
    assert w2.last_decision_clock == pd.Timestamp("2024-06-14T21:00:00Z")
    middle = registered.start + timedelta(minutes=2)
    good = (
        _clock(registered.start, registered.first_decision_clock, 1),
        _clock(middle, middle + timedelta(minutes=1), 2),
        _clock(
            registered.last_decision_clock - timedelta(minutes=1),
            registered.last_decision_clock,
            3,
        ),
    )
    spec = replace(_specs(Path("."), "W1")[0], row_count=len(good))
    pipeline._validate_input_clocks(spec, good)
    invalid = {
        "first": (_clock(registered.start + timedelta(minutes=1), middle, 1), *good[1:]),
        "last": (*good[:-1], _clock(registered.last_decision_clock, registered.end_exclusive, 3)),
        "duration": (good[0], _clock(middle, middle + timedelta(minutes=2), 2), good[2]),
    }
    with pytest.raises(pipeline.Phase7PipelineError, match=bad_case):
        pipeline._validate_input_clocks(spec, invalid[bad_case])


def test_manifest_rejects_sealed_path_before_attempting_to_hash_it(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, input_path=str(tmp_path / "sealed_holdout.jsonl"))
    with pytest.raises(pipeline.Phase7PipelineError, match="opened JSONL"):
        pipeline.load_run_manifest(manifest)


def test_manifest_accepts_exact_opened_source_and_current_code_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = tmp_path / "opened_w1.jsonl"
    opened.write_text("{}\n", encoding="utf-8")
    manifest = _manifest(tmp_path, input_path=str(opened))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["inputs"][0]["sha256"] = pipeline._sha256_file(opened)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    commit = payload["bindings"]["code_bundle"]["repository_commit"]
    monkeypatch.setattr(pipeline, "_repository_identity", lambda: commit)
    monkeypatch.setattr(pipeline, "_assert_code_bundle_committed", lambda: None)

    loaded = pipeline.load_run_manifest(manifest)
    assert loaded.repository_commit == commit
    assert loaded.python_major_minor == PYTHON_IDENTITY
    assert len(loaded.code_bundle_identity) == 64
    assert loaded.inputs[0].source_sha256 == pipeline._sha256_file(opened)


def test_code_bundle_binds_selected_critical_runtime_files_and_rejects_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = pipeline.current_code_bundle_binding(require_clean_repository=False)
    monkeypatch.setattr(
        pipeline,
        "_repository_identity",
        lambda: binding["repository_commit"],
    )
    monkeypatch.setattr(pipeline, "_assert_code_bundle_committed", lambda: None)
    commit, python_identity, identity = pipeline._validate_code_bundle(binding)
    assert commit == binding["repository_commit"]
    assert python_identity == PYTHON_IDENTITY
    assert len(identity) == 64
    assert {
        name: item["path"] for name, item in binding["files"].items()
    } == {
        "runner": "scripts/run_phase7_empirical_pipeline.py",
        "engine": "smc_trader/engine.py",
        "playbooks": "smc_trader/playbooks.py",
        "path_belief": "smc_trader/path_belief.py",
        "probability_cohorts": "smc_trader/probability_cohorts.py",
        "probability_fit": "smc_trader/probability_fit.py",
        "probability_admission": "smc_trader/probability_admission.py",
        "signal_outcome_fit": "smc_trader/signal_outcome_fit.py",
        "model_config": "configs/model.json",
        "uv_lock": "uv.lock",
    }

    drifted = json.loads(json.dumps(binding))
    drifted["files"]["engine"]["sha256"] = "0" * 64
    with pytest.raises(pipeline.Phase7PipelineError, match="engine hash differs"):
        pipeline._validate_code_bundle(drifted)


def test_repository_binding_fails_closed_on_tracked_dirty_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(tuple(command))
        if command[1] == "rev-parse":
            return SimpleNamespace(returncode=0, stdout=COMMIT + "\n")
        return SimpleNamespace(returncode=0, stdout=" M smc_trader/engine.py\n")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    with pytest.raises(pipeline.Phase7PipelineError, match="tracked tree must be clean"):
        pipeline._repository_identity()
    assert calls[-1] == (
        "git",
        "status",
        "--porcelain",
        "--untracked-files=no",
    )


def test_code_bundle_rejects_an_uncommitted_runtime_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=""),
    )
    with pytest.raises(pipeline.Phase7PipelineError, match="committed at HEAD"):
        pipeline._assert_code_bundle_committed()


def test_fit_reads_the_exact_protocol_bytes_bound_by_the_manifest(
    tmp_path: Path,
) -> None:
    manifest = replace(
        _output_manifest(
            tmp_path,
            run_id="protocol-snapshot",
            fit_authorized=True,
            inputs=_specs(tmp_path, "W1", "W2"),
        ),
        phase7_protocol_sha256="0" * 64,
    )
    with pytest.raises(pipeline.Phase7PipelineError, match="protocol changed"):
        pipeline.fit_and_validate(
            manifest, _empty_cohorts(), include_rolling_diagnostics=False
        )


def test_zero_support_fit_returns_blockers_and_closed_non_oof_receipts() -> None:
    specs = tuple(
        pipeline.InputWindowSpec(
            registered=pipeline.REGISTERED_WINDOWS[window],
            path=Path(f"{window}.jsonl"),
            source_sha256=("a" if window == "W1" else "b") * 64,
            row_count=pipeline.REGISTERED_WINDOWS[window].expected_rows,
            fold_id=window,
        )
        for window in ("W1", "W2")
    )
    manifest = pipeline.RunManifest(
        run_id="zero-support",
        source_path=Path("manifest.json"),
        source_sha256=SHA_A,
        model_config_path=pipeline.MODEL_CONFIG,
        model_config_sha256=pipeline._sha256_file(pipeline.MODEL_CONFIG),
        phase7_protocol_sha256=pipeline._sha256_file(pipeline.PHASE7_PROTOCOL),
        preregistration_sha256=pipeline._sha256_file(pipeline.PREREGISTRATION),
        repository_commit=COMMIT,
        python_major_minor=PYTHON_IDENTITY,
        code_bundle_identity=SHA_B,
        materialization_authorized=True,
        fit_and_validate_authorized=True,
        inputs=specs,
    )
    result = pipeline.fit_and_validate(
        manifest,
        _empty_cohorts(),
        include_rolling_diagnostics=False,
    )
    assert result["admission_gate"]["state"] == "closed"
    assert result["artifact_statuses"]["path_base_rate_prior"]["status"] == "blocked"
    assert result["artifact_statuses"]["validation_predictions"]["status"] == (
        "blocked_with_research_artifact"
    )
    assert result["artifacts"]["validation_predictions"]["windows"] == {"W2": {}}
    assert result["admission_receipts"]["path_probability_W2"]["admitted"] is False
    assert "COHORT_NOT_ROLLING_OOF" in result["admission_receipts"]["path_probability_W2"]["blockers"]
    assert result["artifact_blockers"]["dol_temperature_calibration"] == (
        "W1_INNER_CROSSFIT_DOL_ROWS_EMPTY"
    )
    assert result["admission_receipts"]["dol_temperature_W2"]["status"] == (
        "CLOSED"
    )
    delivery = result["artifacts"]["delivery_target_before_invalidation"]
    assert delivery["blocker"] == "FORMAL_NONZERO_INTENT_LEDGER_EMPTY"
    assert delivery["formal_nonzero_intent_case_count"] == 0
    assert delivery["fabricated_setup_row_count"] == 0
    assert result["admission_receipts"]["delivery_target_before_invalidation"][
        "status"
    ] == "CLOSED"
    assert result["admission_receipts"]["delivery_target_before_invalidation"][
        "fit_status"
    ] == "no_fit_blocked"
    assert result["empirical_authority"] is False


def test_staging_inventory_requires_exact_files_directories_and_hashes(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    relative = "cohorts/rows.jsonl"
    digest = pipeline._write_staged_file(staging, relative, b"{}\n")
    expected = {relative: digest}
    pipeline._verify_staging_inventory(staging, expected)

    (staging / relative).write_bytes(b'{"changed":true}\n')
    with pytest.raises(pipeline.Phase7PipelineError, match="payload hash"):
        pipeline._verify_staging_inventory(staging, expected)
    (staging / relative).write_bytes(b"{}\n")
    (staging / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(pipeline.Phase7PipelineError, match="inventory differs"):
        pipeline._verify_staging_inventory(staging, expected)


@pytest.mark.parametrize("target_kind", ("directory", "dangling"))
def test_output_directory_is_strict_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing-output", target_is_directory=True)
    manifest = _output_manifest(tmp_path, run_id="no-clobber")
    revalidations: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "_revalidate_run_manifest",
        lambda value: revalidations.append(value.run_id),
    )
    guarded = destination if target_kind == "directory" else dangling
    with pytest.raises(FileExistsError):
        _publish(guarded, manifest)
    assert revalidations == []


def test_output_publication_revalidates_inventory_then_commits_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    manifest = _output_manifest(tmp_path, run_id="publish-once")
    events: list[str] = []
    real_verify = pipeline._verify_staging_inventory

    def verify_staging(root: Path, expected: dict[str, str]) -> None:
        real_verify(root, expected)
        events.append("inventory")

    monkeypatch.setattr(pipeline, "_verify_staging_inventory", verify_staging)
    monkeypatch.setattr(
        pipeline,
        "_revalidate_run_manifest",
        lambda value: events.append(f"revalidate:{value.run_id}"),
    )
    result = _publish(destination, manifest)
    assert result["status"] == "materialized_only"
    receipt = json.loads((destination / "receipt.json").read_text())
    assert receipt["status"] == "complete_research_receipt"
    assert receipt["schema_version"] == "phase7_empirical_pipeline_v2"
    assert pipeline.RUN_MANIFEST_SCHEMA_VERSION == "phase7_empirical_run_manifest_v1"
    assert (destination / "result.json").is_file()
    assert events == [
        "revalidate:publish-once",
        "revalidate:publish-once",
        "inventory",
    ]


def test_final_inventory_detects_tampering_during_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "tampered"
    manifest = _output_manifest(tmp_path, run_id="tampered")
    revalidations = 0

    def revalidate(_: pipeline.RunManifest) -> None:
        nonlocal revalidations
        revalidations += 1
        if revalidations == 2:
            staging = next(tmp_path.glob(".tampered.staging.*"))
            (staging / "cohorts/path_competition_archive.jsonl").write_bytes(
                b"tampered-after-staging\n"
            )

    monkeypatch.setattr(pipeline, "_revalidate_run_manifest", revalidate)

    with pytest.raises(pipeline.Phase7PipelineError, match="payload hash differs"):
        _publish(destination, manifest)

    assert revalidations == 2
    assert not destination.exists()
    assert list(tmp_path.glob(".tampered.staging.*")) == []


def test_admission_file_is_hash_bound_and_decision_clocks_are_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "admission-bound"
    manifest = _output_manifest(
        tmp_path,
        run_id="admission-bound",
        fit_authorized=True,
        inputs=_specs(tmp_path, "W1"),
    )
    admission = {
        "schema_version": "phase7_admission_not_evaluable_v1",
        "source_dataset_sha256": _fitted_source_sha(manifest),
        "manifest_sha256": SHA_A,
        "admitted": False,
        "status": "CLOSED",
        "receipt_id": None,
        "blockers": ["COHORT_NOT_ROLLING_OOF"],
        "action_authority": False,
    }
    monkeypatch.setattr(pipeline, "_revalidate_run_manifest", lambda _: None)
    result = _publish(
        destination, manifest,
        mode="fit-and-validate",
        fitted=_fitted(manifest, {"path_probability_W2": admission}),
    )
    admission_path = destination / "admission/path_probability_W2.json"
    binding = result["admission_bindings"]["path_probability_W2"]
    assert binding["sha256"] == pipeline._sha256_file(admission_path)
    assert set(binding) == {"path", "sha256", "schema_version", "receipt_id"}
    assert binding["receipt_id"] is None
    receipt = json.loads((destination / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["admission_bindings_sha256"] == pipeline.canonical_sha256(
        result["admission_bindings"]
    )
    assert "admission_bindings" not in receipt
    assert "staged_payload_inventory" not in receipt
    input_binding = json.loads((destination / "cohort_manifest.json").read_text())[
        "inputs"
    ][0]
    assert input_binding["first_decision_clock"] == "2024-06-02T22:01:00+00:00"
    assert input_binding["last_decision_clock"] == "2024-06-07T21:00:00+00:00"


@pytest.mark.parametrize(
    ("bad_field", "bad_value"),
    (("schema_version", 2), ("manifest_sha256", "0" * 64)),
)
def test_writer_accepts_canonical_integer_probability_admission_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_field: str,
    bad_value: object,
) -> None:
    destination = tmp_path / "canonical-probability-admission"
    manifest = _output_manifest(
        tmp_path, run_id="canonical-probability-admission", fit_authorized=True
    )
    admission = ProbabilityAdmissionReceipt(
        artifact_kind="path_probability",
        model_artifact_id="artifact:path-probability",
        source_dataset_sha256=_fitted_source_sha(manifest),
        manifest_sha256=manifest.source_sha256,
        cohort_identity_sha256=SHA_B,
        cohort_role="historical_validation",
        thresholds=AdmissionThresholds(),
        metrics=(),
        support_units=(),
        blockers=("COHORT_NOT_ROLLING_OOF",),
        admitted=False,
        status="rejected_shadow",
    ).to_dict()
    monkeypatch.setattr(pipeline, "_revalidate_run_manifest", lambda _: None)
    result = _publish(
        destination, manifest,
        mode="fit-and-validate",
        fitted=_fitted(manifest, {"path_probability_W2": admission}),
    )
    binding = result["admission_bindings"]["path_probability_W2"]
    persisted = json.loads((destination / binding["path"]).read_text())
    assert type(binding["schema_version"]) is int
    assert binding["schema_version"] == pipeline.PROBABILITY_ADMISSION_SCHEMA_VERSION
    assert persisted["schema_version"] == binding["schema_version"]
    assert binding["receipt_id"] == admission["receipt_id"]
    bad_destination = tmp_path / f"bad-{bad_field}"
    bad = {**admission, bad_field: bad_value}
    with pytest.raises(pipeline.Phase7PipelineError, match="binding differs"):
        _publish(
            bad_destination,
            manifest,
            mode="fit-and-validate",
            fitted=_fitted(manifest, {"path_probability_W2": bad}),
        )
    assert not bad_destination.exists()


@pytest.mark.parametrize(
    ("mode", "fitted", "fit_authorized", "message"),
    (
        ("materialize-only", {}, False, "cannot contain fitted outputs"),
        ("fit-and-validate", None, True, "requires fitted outputs"),
        ("fit-and-validate", {}, False, "does not authorize fitting"),
    ),
)
def test_writer_rejects_mode_fitted_and_authority_mismatch_before_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    fitted: dict[str, object] | None,
    fit_authorized: bool,
    message: str,
) -> None:
    manifest = _output_manifest(
        tmp_path, run_id=f"mismatch-{mode}-{fit_authorized}", fit_authorized=fit_authorized
    )
    monkeypatch.setattr(
        pipeline,
        "_revalidate_run_manifest",
        lambda _: pytest.fail("revalidated before mode/authority validation"),
    )
    with pytest.raises(pipeline.Phase7PipelineError, match=message):
        _publish(tmp_path / "out", manifest, mode=mode, fitted=fitted)


def test_run_pipeline_revalidates_immediately_after_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _output_manifest(tmp_path, run_id="revalidate-order")
    events: list[str] = []
    monkeypatch.setattr(pipeline, "load_run_manifest", lambda _: manifest)
    monkeypatch.setattr(
        pipeline, "materialize_cohorts",
        lambda *_args, **_kwargs: events.append("materialize") or _empty_cohorts(),
    )
    monkeypatch.setattr(
        pipeline, "_revalidate_run_manifest",
        lambda _: events.append("revalidate"),
    )
    monkeypatch.setattr(
        pipeline, "write_pipeline_outputs",
        lambda *_args, **_kwargs: events.append("publish") or {"status": "ok"},
    )
    result = pipeline.run_pipeline(
        tmp_path / "manifest.json",
        tmp_path / "out",
        mode="materialize-only",
        include_rolling_diagnostics=False,
    )
    assert result == {"status": "ok"}
    assert events == ["materialize", "revalidate", "publish"]


def test_run_pipeline_rejects_unauthorized_fit_before_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _output_manifest(tmp_path, run_id="fit-not-authorized")
    monkeypatch.setattr(pipeline, "load_run_manifest", lambda _: manifest)
    monkeypatch.setattr(
        pipeline,
        "materialize_cohorts",
        lambda *_args, **_kwargs: pytest.fail("materialized without fit authority"),
    )
    with pytest.raises(pipeline.Phase7PipelineError, match="does not authorize fitting"):
        pipeline.run_pipeline(
            tmp_path / "manifest.json",
            tmp_path / "out",
            mode="fit-and-validate",
            include_rolling_diagnostics=False,
        )


def test_external_scope_replacement_has_exact_superseded_terminal_clock() -> None:
    state = create_path_competition_set(
        load_path_belief_protocol("configs/path_hypotheses.json"),
        instrument_id="NQM4:13743",
        market_epoch_id="epoch:one",
        authority_structure_id="structure:one",
        horizon_id="market-session:one",
        formed_at=T0,
        common_expires_at=T0 + timedelta(minutes=10),
    )
    generation = pipeline._Generation(
        window_id="W1",
        split_role="development_fit",
        fold_id="W1",
        initial_state=state,
        state=state,
    )
    builder = pipeline.Phase7CohortBuilder()
    builder._counts["W1"] = {"clocks": 1}
    terminal = T0 + timedelta(minutes=1)
    builder._finalize_external(
        generation,
        terminal_at=terminal,
        terminal_status="superseded",
        terminal_cause="scope_superseded",
        censor_reason="scope_superseded",
        source_event_ids=("feed:scope-replacement",),
        real_completed_bar=True,
    )
    assert len(builder._archives) == 1
    assert builder._archives[0].terminal_status == "superseded"
    assert builder._archives[0].terminal_at == terminal
    assert len(builder._intervals) == len(PATH_LABELS)
    assert {row.terminal_status for row in builder._intervals} == {"superseded"}
    assert {row.terminal_known_at for row in builder._intervals} == {terminal}


def test_same_clock_window_censor_does_not_invent_zero_length_exposure() -> None:
    state = create_path_competition_set(
        load_path_belief_protocol("configs/path_hypotheses.json"),
        instrument_id="NQM4:13743",
        market_epoch_id="epoch:one",
        authority_structure_id="structure:one",
        horizon_id="market-session:one",
        formed_at=T0,
        common_expires_at=T0 + timedelta(minutes=10),
    )
    generation = pipeline._Generation(
        window_id="W1",
        split_role="development_fit",
        fold_id="W1",
        initial_state=state,
        state=state,
    )
    builder = pipeline.Phase7CohortBuilder()
    builder._counts["W1"] = {"clocks": 1}
    builder._finalize_external(
        generation,
        terminal_at=T0,
        terminal_status="censored",
        terminal_cause="observation_window_end",
        censor_reason="observation_window_end",
        source_event_ids=("feed:last",),
    )
    assert builder._archives[0].terminal_at == T0
    assert builder._intervals == []
