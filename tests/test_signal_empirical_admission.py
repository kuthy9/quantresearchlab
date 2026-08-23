from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from smc_trader.path_belief import PATH_KINDS, load_path_belief_protocol
from smc_trader.probability_admission import (
    AdmissionThresholds,
    ProbabilityAdmissionReceipt,
)
from smc_trader.probability_cohorts import PATH_LABELS
from smc_trader.probability_fit import PathCalibrationArtifact
from smc_trader.signal_empirical_admission import (
    EmpiricalPathAdmissionBinding,
    EmpiricalSignalOutcomeAdmissionBinding,
    SignalEmpiricalAdmissionError,
    SignalEmpiricalBlocker,
    assess_signal_empirical_admission,
    convert_path_calibration_to_admitted,
    convert_signal_outcomes_to_admitted,
)
from smc_trader.signal_outcome_fit import (
    DOLModelLineage,
    DOLTemperatureFitArtifact,
    DeliveryModelFitArtifact,
    DeliveryModelLineage,
    DeliverySetupModelFit,
    SignalOutcomeAdmissionReceipt,
    SignalOutcomeAdmissionThresholds,
    SignalOutcomeArtifactKind,
)
from smc_trader.signal_policy import (
    SetupFamily,
    load_dol_calibration_artifact,
    load_outcome_model_artifact,
    load_path_likelihood_artifact,
)


SOURCE_SHA = "a" * 64
MANIFEST_SHA = "b" * 64
COHORT_SHA = "c" * 64
TRAINED = pd.Timestamp("2025-01-31T21:00:00Z")
VALID_FROM = pd.Timestamp("2025-02-03T22:00:00Z")
EXPIRES = pd.Timestamp("2025-03-01T00:00:00Z")


def _fit() -> PathCalibrationArtifact:
    return PathCalibrationArtifact(
        model_version="phase7_path_temperature_bias_test_v1",
        source_dataset_sha256=SOURCE_SHA,
        manifest_sha256=MANIFEST_SHA,
        source_model_artifact_id="path-model:rolling-oof:test-v1",
        temperature=1.25,
        path_log_biases=tuple(
            (path, 0.0 if path == "residual_unknown" else index / 100.0)
            for index, path in enumerate(PATH_LABELS)
        ),
        gauge_path="residual_unknown",
        weighted_log_loss=0.8,
        fit_sample_count=240,
    )


def _receipt(fit: PathCalibrationArtifact) -> ProbabilityAdmissionReceipt:
    return ProbabilityAdmissionReceipt(
        artifact_kind="path_probability",
        model_artifact_id=fit.artifact_id,
        source_dataset_sha256=SOURCE_SHA,
        manifest_sha256=MANIFEST_SHA,
        cohort_identity_sha256=COHORT_SHA,
        cohort_role="rolling_oof",
        thresholds=AdmissionThresholds(),
        metrics=(("model_log_loss", 0.8),),
        support_units=tuple((path, 40) for path in PATH_LABELS),
        blockers=(),
        admitted=True,
        status="admitted_shadow",
    )


def _binding(
    fit: PathCalibrationArtifact,
    receipt: ProbabilityAdmissionReceipt,
    *,
    window_id: str = "rolling-oof:2025-fold-1",
    cohort_role: str = "rolling_oof",
) -> EmpiricalPathAdmissionBinding:
    protocol = load_path_belief_protocol()
    return EmpiricalPathAdmissionBinding(
        window_id=window_id,
        cohort_role=cohort_role,
        source_dataset_sha256=SOURCE_SHA,
        manifest_sha256=MANIFEST_SHA,
        cohort_identity_sha256=COHORT_SHA,
        expected_fit_artifact_id=fit.artifact_id,
        expected_admission_receipt_id=receipt.receipt_id,
        expected_path_protocol_fingerprint=protocol.fingerprint,
        expected_path_model_version=protocol.model_version,
        trained_through=TRAINED,
        valid_from=VALID_FROM,
        expires_at=EXPIRES,
    )


def _outcome_evidence(
    path_artifact_id: str,
    *,
    path_protocol_fingerprint: str,
    path_model_version: str,
) -> tuple[
    EmpiricalSignalOutcomeAdmissionBinding,
    DOLTemperatureFitArtifact,
    SignalOutcomeAdmissionReceipt,
    DeliveryModelFitArtifact,
    SignalOutcomeAdmissionReceipt,
]:
    trained = TRAINED - pd.Timedelta(days=1)
    split_sha = "d" * 64
    dol_fit_identity = "e" * 64
    delivery_fit_identity = "f" * 64
    dol_fit = DOLTemperatureFitArtifact(
        model_version="signal_dol_temperature_test_v1",
        lineage=DOLModelLineage(
            source_dol_protocol_fingerprint="1" * 64,
            source_dol_model_version="dol-model-test-v1",
            source_dol_model_fingerprint="2" * 64,
            source_path_protocol_fingerprint=path_protocol_fingerprint,
            source_path_model_version=path_model_version,
        ),
        source_cohort_id="dol-fit-cohort:test-v1",
        source_cohort_role="development_cross_fit",
        source_window_ids=("train-2025",),
        source_fold_ids=("fit",),
        source_dataset_sha256=SOURCE_SHA,
        manifest_sha256=MANIFEST_SHA,
        cohort_identity_sha256=dol_fit_identity,
        split_protocol_sha256=split_sha,
        source_observation_ids=("dol-fit-observation:1", "dol-fit-observation:2"),
        source_case_ids=("dol-fit-case:1", "dol-fit-case:2"),
        source_cluster_ids=("dol-fit-session:1", "dol-fit-session:2"),
        trained_through=trained,
        temperature=0.8,
        raw_log_loss=0.7,
        calibrated_log_loss=0.65,
        raw_brier=0.3,
        calibrated_brier=0.28,
        fit_sample_count=2,
        support_units=(
            ("candidate_target", 1),
            ("no_target_before_common_horizon", 1),
        ),
    )
    thresholds = SignalOutcomeAdmissionThresholds(
        minimum_resolved_units=2,
        minimum_units_per_support=1,
        minimum_rolling_folds=2,
    )
    dol_receipt = SignalOutcomeAdmissionReceipt(
        artifact_kind=SignalOutcomeArtifactKind.DOL_TEMPERATURE,
        model_artifact_id=dol_fit.artifact_id,
        fit_cohort_id=dol_fit.source_cohort_id,
        fit_cohort_role=dol_fit.source_cohort_role,
        fit_cohort_identity_sha256=dol_fit.cohort_identity_sha256,
        fit_window_ids=dol_fit.source_window_ids,
        fit_fold_ids=dol_fit.source_fold_ids,
        validation_cohort_id="dol-oof-cohort:test-v1",
        validation_cohort_role="rolling_oof",
        validation_window_ids=("rolling-oof-2025",),
        validation_fold_ids=("oof-1", "oof-2"),
        source_dataset_sha256="4" * 64,
        manifest_sha256="5" * 64,
        cohort_identity_sha256="6" * 64,
        split_protocol_sha256=split_sha,
        validation_observation_ids=("dol-oof-observation:1", "dol-oof-observation:2"),
        validation_case_ids=("dol-oof-case:1", "dol-oof-case:2"),
        fit_cluster_ids=dol_fit.source_cluster_ids,
        validation_cluster_ids=("dol-oof-session:1", "dol-oof-session:2"),
        thresholds=thresholds,
        metrics=(("model_log_loss", 0.6),),
        support_units=(
            ("candidate_target", 1),
            ("no_target_before_common_horizon", 1),
        ),
        blockers=(),
        admitted=True,
        status="admitted_shadow",
    )
    delivery_fit = DeliveryModelFitArtifact(
        model_version="signal_delivery_model_test_v1",
        lineage=DeliveryModelLineage(
            path_likelihood_artifact_id=path_artifact_id,
            dol_calibration_fit_artifact_id=dol_fit.artifact_id,
            signal_policy_fingerprint="7" * 64,
        ),
        source_cohort_id="delivery-fit-cohort:test-v1",
        source_cohort_role="development_cross_fit",
        source_window_ids=("train-2025",),
        source_fold_ids=("fit",),
        source_dataset_sha256=SOURCE_SHA,
        manifest_sha256=MANIFEST_SHA,
        cohort_identity_sha256=delivery_fit_identity,
        split_protocol_sha256=split_sha,
        source_observation_ids=tuple(
            f"delivery-fit-observation:{index}" for index in range(4)
        ),
        source_case_ids=tuple(f"delivery-fit-case:{index}" for index in range(4)),
        source_cluster_ids=("delivery-fit-session:1", "delivery-fit-session:2"),
        trained_through=trained,
        setup_models=(
            DeliverySetupModelFit(
                setup_family=SetupFamily.DFP,
                intercept=-0.2,
                path_logit_coefficient=0.8,
                dol_logit_coefficient=0.5,
                half_life_real_completed_bars=10,
                fit_prevalence=0.5,
                fit_log_loss=0.6,
                fit_brier=0.2,
                fit_sample_count=2,
                positive_count=1,
                negative_count=1,
            ),
            DeliverySetupModelFit(
                setup_family=SetupFamily.LSR,
                intercept=-0.1,
                path_logit_coefficient=0.7,
                dol_logit_coefficient=0.4,
                half_life_real_completed_bars=8,
                fit_prevalence=0.5,
                fit_log_loss=0.62,
                fit_brier=0.21,
                fit_sample_count=2,
                positive_count=1,
                negative_count=1,
            ),
        ),
        minimum_supported_coverage=0.95,
        fit_sample_count=4,
    )
    delivery_receipt = SignalOutcomeAdmissionReceipt(
        artifact_kind=SignalOutcomeArtifactKind.DELIVERY_MODEL,
        model_artifact_id=delivery_fit.artifact_id,
        fit_cohort_id=delivery_fit.source_cohort_id,
        fit_cohort_role=delivery_fit.source_cohort_role,
        fit_cohort_identity_sha256=delivery_fit.cohort_identity_sha256,
        fit_window_ids=delivery_fit.source_window_ids,
        fit_fold_ids=delivery_fit.source_fold_ids,
        validation_cohort_id="delivery-oof-cohort:test-v1",
        validation_cohort_role="rolling_oof",
        validation_window_ids=("rolling-oof-2025",),
        validation_fold_ids=("oof-1", "oof-2"),
        source_dataset_sha256="8" * 64,
        manifest_sha256="9" * 64,
        cohort_identity_sha256="0" * 64,
        split_protocol_sha256=split_sha,
        validation_observation_ids=tuple(
            f"delivery-oof-observation:{index}" for index in range(4)
        ),
        validation_case_ids=tuple(
            f"delivery-oof-case:{index}" for index in range(4)
        ),
        fit_cluster_ids=delivery_fit.source_cluster_ids,
        validation_cluster_ids=(
            "delivery-oof-session:1",
            "delivery-oof-session:2",
        ),
        thresholds=thresholds,
        metrics=(("model_log_loss", 0.55),),
        support_units=(
            ("dfp:negative", 1),
            ("dfp:positive", 1),
            ("lsr:negative", 1),
            ("lsr:positive", 1),
        ),
        blockers=(),
        admitted=True,
        status="admitted_shadow",
    )
    binding = EmpiricalSignalOutcomeAdmissionBinding(
        expected_dol_fit_artifact_id=dol_fit.artifact_id,
        expected_dol_admission_receipt_id=dol_receipt.receipt_id,
        expected_delivery_fit_artifact_id=delivery_fit.artifact_id,
        expected_delivery_admission_receipt_id=delivery_receipt.receipt_id,
        expected_path_likelihood_artifact_id=path_artifact_id,
        valid_from=VALID_FROM,
        expires_at=EXPIRES,
    )
    return binding, dol_fit, dol_receipt, delivery_fit, delivery_receipt


def test_exact_rolling_oof_path_conversion_matches_production_loader(
    tmp_path: Path,
) -> None:
    fit = _fit()
    receipt = _receipt(fit)
    protocol = load_path_belief_protocol()
    binding = _binding(fit, receipt)

    conversion = convert_path_calibration_to_admitted(
        binding,
        path_fit=fit,
        path_receipt=receipt,
        path_protocol=protocol,
    )
    artifact = conversion.artifact
    assert artifact.temperature == fit.temperature
    assert tuple(path for path, _ in artifact.path_log_biases) == PATH_KINDS
    assert tuple(value for _, value in artifact.path_log_biases) == tuple(
        value for _, value in fit.path_log_biases
    )
    assert artifact.source_path_protocol_fingerprint == protocol.fingerprint
    assert artifact.source_path_model_version == protocol.model_version
    assert artifact.authority == "shadow_only"
    assert artifact.action_authority_ready is False
    assert conversion.source_fit_artifact_id == fit.artifact_id
    assert conversion.source_admission_receipt_id == receipt.receipt_id
    assert (
        conversion.payload_sha256
        == hashlib.sha256(
            json.dumps(
                conversion.payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
    )

    path = tmp_path / "path-admitted.json"
    path.write_text(
        json.dumps(conversion.payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    assert (
        load_path_likelihood_artifact(
            path,
            expected_artifact_id=artifact.artifact_id,
        )
        == artifact
    )

    assessment = assess_signal_empirical_admission(
        binding,
        path_fit=fit,
        path_receipt=receipt,
        path_protocol=protocol,
    )
    assert assessment.path_conversion_ready is True
    assert assessment.full_signal_bundle_ready is False
    assert assessment.status == "path_conversion_ready_full_bundle_closed"
    assert assessment.blockers == tuple(
        sorted(
                (
                    SignalEmpiricalBlocker.DELIVERY_FIT_MISSING,
                    SignalEmpiricalBlocker.DELIVERY_RECEIPT_MISSING,
                    SignalEmpiricalBlocker.DOL_FIT_MISSING,
                    SignalEmpiricalBlocker.DOL_RECEIPT_MISSING,
                    SignalEmpiricalBlocker.EXTERNAL_SIGNAL_ARTIFACT_PINS_MISSING,
                    SignalEmpiricalBlocker.OUTCOME_BINDING_MISSING,
            ),
            key=lambda item: item.value,
        )
    )


def test_exact_outcome_conversions_match_production_loaders_without_pins(
    tmp_path: Path,
) -> None:
    path_fit = _fit()
    path_receipt = _receipt(path_fit)
    path_protocol = load_path_belief_protocol()
    path_binding = _binding(path_fit, path_receipt)
    path_conversion = convert_path_calibration_to_admitted(
        path_binding,
        path_fit=path_fit,
        path_receipt=path_receipt,
        path_protocol=path_protocol,
    )
    (
        outcome_binding,
        dol_fit,
        dol_receipt,
        delivery_fit,
        delivery_receipt,
    ) = _outcome_evidence(
        path_conversion.artifact.artifact_id,
        path_protocol_fingerprint=(
            path_conversion.artifact.source_path_protocol_fingerprint
        ),
        path_model_version=path_conversion.artifact.source_path_model_version,
    )

    assessment = assess_signal_empirical_admission(
        path_binding,
        path_fit=path_fit,
        path_receipt=path_receipt,
        path_protocol=path_protocol,
        outcome_binding=outcome_binding,
        dol_fit=dol_fit,
        dol_receipt=dol_receipt,
        delivery_fit=delivery_fit,
        delivery_receipt=delivery_receipt,
    )
    assert assessment.path_conversion_ready is True
    assert assessment.dol_conversion_ready is True
    assert assessment.delivery_conversion_ready is True
    assert assessment.full_signal_bundle_ready is False
    assert assessment.status == "all_conversions_ready_external_pins_closed"
    assert assessment.blockers == (
        SignalEmpiricalBlocker.EXTERNAL_SIGNAL_ARTIFACT_PINS_MISSING,
    )

    conversion = convert_signal_outcomes_to_admitted(
        outcome_binding,
        path_conversion=path_conversion,
        dol_fit=dol_fit,
        dol_receipt=dol_receipt,
        delivery_fit=delivery_fit,
        delivery_receipt=delivery_receipt,
    )
    assert conversion.pins_issued is False
    assert conversion.action_authority is False
    assert conversion.dol.artifact.temperature == dol_fit.temperature
    assert conversion.dol.artifact.source_dol_model_fingerprint == (
        dol_fit.lineage.source_dol_model_fingerprint
    )
    assert conversion.delivery.artifact.path_likelihood_artifact_id == (
        path_conversion.artifact.artifact_id
    )
    assert conversion.delivery.artifact.dol_calibration_artifact_id == (
        conversion.dol.artifact.artifact_id
    )
    assert tuple(
        (item.setup_family, item.half_life_real_completed_bars)
        for item in conversion.delivery.artifact.setup_models
    ) == ((SetupFamily.DFP, 10), (SetupFamily.LSR, 8))
    for component in (conversion.dol, conversion.delivery):
        assert component.payload_sha256 == hashlib.sha256(
            json.dumps(
                component.payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()

    dol_path = tmp_path / "dol-admitted.json"
    dol_path.write_text(
        json.dumps(conversion.dol.payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    delivery_path = tmp_path / "delivery-admitted.json"
    delivery_path.write_text(
        json.dumps(
            conversion.delivery.payload,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    assert load_dol_calibration_artifact(
        dol_path,
        expected_artifact_id=conversion.dol.artifact.artifact_id,
    ) == conversion.dol.artifact
    assert load_outcome_model_artifact(
        delivery_path,
        expected_artifact_id=conversion.delivery.artifact.artifact_id,
    ) == conversion.delivery.artifact


def test_june_fit_artifact_cannot_cross_outcome_admission_boundary() -> None:
    path_fit = _fit()
    path_receipt = _receipt(path_fit)
    path_protocol = load_path_belief_protocol()
    path_binding = _binding(path_fit, path_receipt)
    path_conversion = convert_path_calibration_to_admitted(
        path_binding,
        path_fit=path_fit,
        path_receipt=path_receipt,
        path_protocol=path_protocol,
    )
    (
        outcome_binding,
        dol_fit,
        dol_receipt,
        delivery_fit,
        delivery_receipt,
    ) = _outcome_evidence(
        path_conversion.artifact.artifact_id,
        path_protocol_fingerprint=(
            path_conversion.artifact.source_path_protocol_fingerprint
        ),
        path_model_version=path_conversion.artifact.source_path_model_version,
    )
    june_fit = replace(
        dol_fit,
        source_cohort_role="historical_validation",
        source_window_ids=("2024-06-week-1",),
    )
    june_binding = replace(
        outcome_binding,
        expected_dol_fit_artifact_id=june_fit.artifact_id,
    )
    assessment = assess_signal_empirical_admission(
        path_binding,
        path_fit=path_fit,
        path_receipt=path_receipt,
        path_protocol=path_protocol,
        outcome_binding=june_binding,
        dol_fit=june_fit,
        dol_receipt=dol_receipt,
        delivery_fit=delivery_fit,
        delivery_receipt=delivery_receipt,
    )
    assert assessment.dol_conversion_ready is False
    assert SignalEmpiricalBlocker.DOL_JUNE_SOURCE_CLOSED in assessment.blockers
    with pytest.raises(SignalEmpiricalAdmissionError, match="dol_june_source_closed"):
        convert_signal_outcomes_to_admitted(
            june_binding,
            path_conversion=path_conversion,
            dol_fit=june_fit,
            dol_receipt=dol_receipt,
            delivery_fit=delivery_fit,
            delivery_receipt=delivery_receipt,
        )


def test_each_outcome_receipt_must_be_independently_rolling_oof_admitted() -> None:
    path_fit = _fit()
    path_receipt = _receipt(path_fit)
    path_protocol = load_path_belief_protocol()
    path_binding = _binding(path_fit, path_receipt)
    path_conversion = convert_path_calibration_to_admitted(
        path_binding,
        path_fit=path_fit,
        path_receipt=path_receipt,
        path_protocol=path_protocol,
    )
    (
        outcome_binding,
        dol_fit,
        dol_receipt,
        delivery_fit,
        delivery_receipt,
    ) = _outcome_evidence(
        path_conversion.artifact.artifact_id,
        path_protocol_fingerprint=(
            path_conversion.artifact.source_path_protocol_fingerprint
        ),
        path_model_version=path_conversion.artifact.source_path_model_version,
    )
    rejected_dol = replace(
        dol_receipt,
        validation_cohort_role="historical_validation",
        blockers=("COHORT_NOT_ROLLING_OOF",),
        admitted=False,
        status="CLOSED",
    )
    rejected_delivery = replace(
        delivery_receipt,
        validation_cohort_role="historical_validation",
        blockers=("COHORT_NOT_ROLLING_OOF",),
        admitted=False,
        status="CLOSED",
    )
    rejected_binding = replace(
        outcome_binding,
        expected_dol_admission_receipt_id=rejected_dol.receipt_id,
        expected_delivery_admission_receipt_id=rejected_delivery.receipt_id,
    )
    assessment = assess_signal_empirical_admission(
        path_binding,
        path_fit=path_fit,
        path_receipt=path_receipt,
        path_protocol=path_protocol,
        outcome_binding=rejected_binding,
        dol_fit=dol_fit,
        dol_receipt=rejected_dol,
        delivery_fit=delivery_fit,
        delivery_receipt=rejected_delivery,
    )
    assert assessment.dol_conversion_ready is False
    assert assessment.delivery_conversion_ready is False
    assert SignalEmpiricalBlocker.DOL_RECEIPT_NOT_ADMITTED in assessment.blockers
    assert (
        SignalEmpiricalBlocker.DELIVERY_RECEIPT_NOT_ADMITTED
        in assessment.blockers
    )
    with pytest.raises(
        SignalEmpiricalAdmissionError,
        match="dol_receipt_not_admitted",
    ):
        convert_signal_outcomes_to_admitted(
            rejected_binding,
            path_conversion=path_conversion,
            dol_fit=dol_fit,
            dol_receipt=rejected_dol,
            delivery_fit=delivery_fit,
            delivery_receipt=rejected_delivery,
        )


@pytest.mark.parametrize(
    ("window_id", "cohort_role"),
    (
        ("W1", "development_cross_fit"),
        ("2024-06-week-2", "historical_validation"),
    ),
)
def test_june_w1_w2_remain_closed_and_cannot_be_relabelled_rolling_oof(
    window_id: str,
    cohort_role: str,
) -> None:
    fit = _fit()
    receipt = _receipt(fit)
    protocol = load_path_belief_protocol()
    binding = _binding(
        fit,
        receipt,
        window_id=window_id,
        cohort_role=cohort_role,
    )
    assessment = assess_signal_empirical_admission(
        binding,
        path_fit=fit,
        path_receipt=receipt,
        path_protocol=protocol,
    )
    assert assessment.path_conversion_ready is False
    assert SignalEmpiricalBlocker.SOURCE_NOT_ROLLING_OOF in assessment.blockers
    with pytest.raises(
        SignalEmpiricalAdmissionError,
        match="source_not_rolling_oof",
    ):
        convert_path_calibration_to_admitted(
            binding,
            path_fit=fit,
            path_receipt=receipt,
            path_protocol=protocol,
        )
    with pytest.raises(ValueError, match="cannot be relabelled rolling_oof"):
        _binding(
            fit,
            receipt,
            window_id=window_id,
            cohort_role="rolling_oof",
        )


@pytest.mark.parametrize(
    ("field_name", "replacement", "expected"),
    (
        (
            "expected_fit_artifact_id",
            "path-calibration-artifact:stale",
            SignalEmpiricalBlocker.PATH_FIT_PIN_MISMATCH,
        ),
        (
            "expected_admission_receipt_id",
            "probability-admission:stale",
            SignalEmpiricalBlocker.PATH_RECEIPT_PIN_MISMATCH,
        ),
        (
            "source_dataset_sha256",
            "d" * 64,
            SignalEmpiricalBlocker.PATH_SOURCE_HASH_MISMATCH,
        ),
        (
            "expected_path_protocol_fingerprint",
            "e" * 64,
            SignalEmpiricalBlocker.PATH_PROTOCOL_MISMATCH,
        ),
    ),
)
def test_path_conversion_rejects_every_hash_pin_or_protocol_drift(
    field_name: str,
    replacement: str,
    expected: SignalEmpiricalBlocker,
) -> None:
    fit = _fit()
    receipt = _receipt(fit)
    protocol = load_path_belief_protocol()
    binding = replace(_binding(fit, receipt), **{field_name: replacement})
    assessment = assess_signal_empirical_admission(
        binding,
        path_fit=fit,
        path_receipt=receipt,
        path_protocol=protocol,
    )
    assert expected in assessment.blockers
    assert assessment.path_conversion_ready is False
    with pytest.raises(SignalEmpiricalAdmissionError, match=expected.value):
        convert_path_calibration_to_admitted(
            binding,
            path_fit=fit,
            path_receipt=receipt,
            path_protocol=protocol,
        )


def test_non_admitted_receipt_and_wrong_path_bias_gauge_are_rejected() -> None:
    protocol = load_path_belief_protocol()
    fit = _fit()
    admitted = _receipt(fit)
    rejected = replace(
        admitted,
        cohort_role="historical_validation",
        blockers=("COHORT_NOT_ROLLING_OOF",),
        admitted=False,
        status="rejected_shadow",
    )
    rejected_binding = replace(
        _binding(fit, admitted),
        expected_admission_receipt_id=rejected.receipt_id,
    )
    rejected_assessment = assess_signal_empirical_admission(
        rejected_binding,
        path_fit=fit,
        path_receipt=rejected,
        path_protocol=protocol,
    )
    assert (
        SignalEmpiricalBlocker.PATH_RECEIPT_NOT_ADMITTED in rejected_assessment.blockers
    )

    wrong_gauge = replace(fit, gauge_path="continuation")
    wrong_gauge_receipt = _receipt(wrong_gauge)
    gauge_assessment = assess_signal_empirical_admission(
        _binding(wrong_gauge, wrong_gauge_receipt),
        path_fit=wrong_gauge,
        path_receipt=wrong_gauge_receipt,
        path_protocol=protocol,
    )
    assert (
        SignalEmpiricalBlocker.PATH_CALIBRATION_GAUGE_MISMATCH
        in gauge_assessment.blockers
    )


def test_missing_fit_receipt_and_unsupported_bundle_components_stay_closed() -> None:
    fit = _fit()
    receipt = _receipt(fit)
    binding = _binding(fit, receipt)
    assessment = assess_signal_empirical_admission(
        binding,
        path_fit=None,
        path_receipt=None,
        path_protocol=None,
    )
    assert assessment.status == "closed"
    assert set(assessment.blockers) == {
        SignalEmpiricalBlocker.PATH_FIT_MISSING,
        SignalEmpiricalBlocker.PATH_RECEIPT_MISSING,
        SignalEmpiricalBlocker.PATH_PROTOCOL_MISMATCH,
        SignalEmpiricalBlocker.OUTCOME_BINDING_MISSING,
        SignalEmpiricalBlocker.DOL_FIT_MISSING,
        SignalEmpiricalBlocker.DOL_RECEIPT_MISSING,
        SignalEmpiricalBlocker.DELIVERY_FIT_MISSING,
        SignalEmpiricalBlocker.DELIVERY_RECEIPT_MISSING,
        SignalEmpiricalBlocker.EXTERNAL_SIGNAL_ARTIFACT_PINS_MISSING,
    }
