from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from smc_trader.signal_outcome_fit import (
    DOLModelLineage,
    DOLSupportLabel,
    DOLTemperatureObservation,
    DeliveryModelLineage,
    DeliveryOutcomeObservation,
    RealCompletedBarEvidence,
    SignalOutcomeAdmissionThresholds,
    SignalOutcomeCohort,
    SignalOutcomeFitError,
    evaluate_delivery_model_admission,
    evaluate_dol_temperature_admission,
    fit_delivery_model,
    fit_dol_temperature,
)
from smc_trader.signal_policy import SetupFamily


SOURCE_MODEL_SHA = "1" * 64
SOURCE_DATA_SHA = "2" * 64
MANIFEST_SHA = "3" * 64
SPLIT_SHA = "4" * 64
PATH_PROTOCOL_SHA = "5" * 64
DOL_PROTOCOL_SHA = "6" * 64
SIGNAL_POLICY_SHA = "7" * 64
PATH_ARTIFACT_ID = "path-likelihood-artifact:synthetic-oof-v1"
DOL_FIT_SOURCE_ID = "dol-temperature-fit:synthetic-v1"
NO_TARGET = "no_target_before_common_horizon"


def _cohort(
    *,
    name: str,
    role: str,
    windows: tuple[str, ...],
    folds: tuple[str, ...],
    identity: str,
) -> SignalOutcomeCohort:
    return SignalOutcomeCohort(
        cohort_name=name,
        cohort_role=role,
        window_ids=windows,
        fold_ids=folds,
        source_dataset_sha256=SOURCE_DATA_SHA,
        manifest_sha256=MANIFEST_SHA,
        cohort_identity_sha256=identity,
        split_protocol_sha256=SPLIT_SHA,
        prediction_coverage=0.99,
    )


def _dol_lineage() -> DOLModelLineage:
    return DOLModelLineage(
        source_dol_protocol_fingerprint=DOL_PROTOCOL_SHA,
        source_dol_model_version="dol-probability-synthetic-v1",
        source_dol_model_fingerprint=SOURCE_MODEL_SHA,
        source_path_protocol_fingerprint=PATH_PROTOCOL_SHA,
        source_path_model_version="path-synthetic-v1",
    )


def _dol_rows(
    *,
    start: pd.Timestamp,
    window: str,
    folds: tuple[str, ...],
    prefix: str,
    count: int = 8,
) -> tuple[DOLTemperatureObservation, ...]:
    values = []
    for index in range(count):
        prediction = start + pd.Timedelta(minutes=index * 3)
        candidate_wins = index % 2 == 0
        candidate_probability = 0.65 if candidate_wins else 0.35
        values.append(
            DOLTemperatureObservation(
                case_id=f"{prefix}:case:{index}",
                prediction_id=f"{prefix}:prediction:{index}",
                outcome_event_id=f"{prefix}:outcome:{index}",
                window_id=window,
                fold_id=folds[index % len(folds)],
                cluster_id=f"{prefix}:session:{index // 2}",
                prediction_known_at=prediction,
                outcome_known_at=prediction + pd.Timedelta(minutes=1),
                outcome_probabilities=(
                    ("candidate:A", candidate_probability),
                    (NO_TARGET, 1.0 - candidate_probability),
                ),
                realized_outcome_id=("candidate:A" if candidate_wins else NO_TARGET),
                support_label=(
                    DOLSupportLabel.CANDIDATE_TARGET
                    if candidate_wins
                    else DOLSupportLabel.NO_TARGET
                ),
                source_dol_model_fingerprint=SOURCE_MODEL_SHA,
            )
        )
    return tuple(values)


def _bar_evidence(
    *,
    prefix: str,
    prediction: pd.Timestamp,
    prediction_ordinal: int,
    count: int,
) -> tuple[RealCompletedBarEvidence, ...]:
    return tuple(
        RealCompletedBarEvidence(
            bar_id=f"{prefix}:bar:{offset}",
            real_completed_bar_ordinal=prediction_ordinal + offset,
            known_at=prediction + pd.Timedelta(minutes=offset),
        )
        for offset in range(1, count + 1)
    )


def _delivery_rows(
    *,
    start: pd.Timestamp,
    window: str,
    folds: tuple[str, ...],
    prefix: str,
) -> tuple[DeliveryOutcomeObservation, ...]:
    rows = []
    probability_pairs = (
        (0.80, 0.75, True),
        (0.75, 0.70, True),
        (0.70, 0.65, True),
        (0.65, 0.60, True),
        (0.35, 0.40, False),
        (0.30, 0.35, False),
        (0.25, 0.30, False),
        (0.20, 0.25, False),
    )
    counts = {
        SetupFamily.DFP: (2, 4, 6, 8, 2, 4, 6, 8),
        SetupFamily.LSR: (1, 3, 5, 7, 1, 3, 5, 7),
    }
    for setup_index, setup in enumerate((SetupFamily.DFP, SetupFamily.LSR)):
        for index, (p_path, p_dol, outcome) in enumerate(probability_pairs):
            prediction = start + pd.Timedelta(hours=setup_index * 2, minutes=index * 10)
            ordinal = 1000 + setup_index * 100 + index * 10
            bars = _bar_evidence(
                prefix=f"{prefix}:{setup.value}:{index}",
                prediction=prediction,
                prediction_ordinal=ordinal,
                count=counts[setup][index],
            )
            rows.append(
                DeliveryOutcomeObservation(
                    case_id=f"{prefix}:{setup.value}:case:{index}",
                    prediction_id=f"{prefix}:{setup.value}:prediction:{index}",
                    outcome_event_id=f"{prefix}:{setup.value}:outcome:{index}",
                    window_id=window,
                    fold_id=folds[index % len(folds)],
                    cluster_id=f"{prefix}:{setup.value}:session:{index // 2}",
                    setup_family=setup,
                    prediction_known_at=prediction,
                    outcome_known_at=bars[-1].known_at,
                    p_path=p_path,
                    p_dol=p_dol,
                    target_before_invalidation=outcome,
                    prediction_real_completed_bar_ordinal=ordinal,
                    real_completed_bars=bars,
                    path_likelihood_artifact_id=PATH_ARTIFACT_ID,
                    dol_calibration_fit_artifact_id=DOL_FIT_SOURCE_ID,
                )
            )
    return tuple(rows)


def _delivery_lineage() -> DeliveryModelLineage:
    return DeliveryModelLineage(
        path_likelihood_artifact_id=PATH_ARTIFACT_ID,
        dol_calibration_fit_artifact_id=DOL_FIT_SOURCE_ID,
        signal_policy_fingerprint=SIGNAL_POLICY_SHA,
    )


def _lenient_thresholds() -> SignalOutcomeAdmissionThresholds:
    return SignalOutcomeAdmissionThresholds(
        minimum_resolved_units=4,
        minimum_units_per_support=1,
        minimum_prediction_coverage=0.9,
        minimum_rolling_folds=2,
        maximum_ece=1.0,
        maximum_log_loss_degradation=0.0,
        maximum_brier_degradation=0.0,
        minimum_improving_fold_fraction=0.5,
        maximum_single_fold_log_loss_degradation=0.0,
    )


def test_prediction_clock_must_strictly_precede_outcome_and_bars_are_real() -> None:
    clock = pd.Timestamp("2025-01-02T14:00:00Z")
    with pytest.raises(SignalOutcomeFitError, match="complete causal"):
        DOLTemperatureObservation(
            case_id="case",
            prediction_id="prediction",
            outcome_event_id="outcome",
            window_id="train",
            fold_id="fold",
            cluster_id="session",
            prediction_known_at=clock,
            outcome_known_at=clock,
            outcome_probabilities=(("candidate:A", 0.6), (NO_TARGET, 0.4)),
            realized_outcome_id="candidate:A",
            support_label=DOLSupportLabel.CANDIDATE_TARGET,
            source_dol_model_fingerprint=SOURCE_MODEL_SHA,
        )
    with pytest.raises(SignalOutcomeFitError, match="real, gap-free"):
        RealCompletedBarEvidence(
            bar_id="synthetic",
            real_completed_bar_ordinal=1,
            known_at=clock + pd.Timedelta(minutes=1),
            synthetic_no_trade=True,
        )

    valid = _delivery_rows(
        start=clock,
        window="train",
        folds=("fit",),
        prefix="leakage",
    )[0]
    with pytest.raises(SignalOutcomeFitError, match="ancestry"):
        replace(
            valid,
            outcome_known_at=valid.outcome_known_at + pd.Timedelta(minutes=1),
        )


def test_dol_temperature_fit_is_deterministic_closed_and_losslessly_mappable() -> None:
    cohort = _cohort(
        name="fit",
        role="development_cross_fit",
        windows=("train-2025",),
        folds=("fit",),
        identity="8" * 64,
    )
    rows = _dol_rows(
        start=pd.Timestamp("2025-01-02T14:00:00Z"),
        window="train-2025",
        folds=("fit",),
        prefix="dol-fit",
    )
    artifact = fit_dol_temperature(rows, cohort=cohort, lineage=_dol_lineage())
    repeated = fit_dol_temperature(rows[::-1], cohort=cohort, lineage=_dol_lineage())
    assert repeated == artifact
    assert artifact.temperature < 1.0
    assert artifact.calibrated_log_loss < artifact.raw_log_loss
    assert artifact.fit_status == "fitted_not_admitted"
    assert artifact.admission_status == "CLOSED"
    assert artifact.action_authority is False
    assert artifact.pins_issued is False
    payload = artifact.to_payload()
    assert payload["source_window_ids"] == ["train-2025"]
    assert payload["source_fold_ids"] == ["fit"]
    assert payload["lineage"]["source_dol_protocol_fingerprint"] == DOL_PROTOCOL_SHA
    assert payload["lineage"]["source_dol_model_fingerprint"] == SOURCE_MODEL_SHA
    assert payload["lineage"]["source_path_protocol_fingerprint"] == PATH_PROTOCOL_SHA
    assert payload["temperature"] == artifact.temperature


def test_delivery_fit_has_dfp_lsr_logits_and_real_bar_half_lives() -> None:
    cohort = _cohort(
        name="fit",
        role="development_cross_fit",
        windows=("train-2025",),
        folds=("fit",),
        identity="8" * 64,
    )
    rows = _delivery_rows(
        start=pd.Timestamp("2025-01-02T14:00:00Z"),
        window="train-2025",
        folds=("fit",),
        prefix="delivery-fit",
    )
    artifact = fit_delivery_model(
        rows,
        cohort=cohort,
        lineage=_delivery_lineage(),
    )
    assert tuple(model.setup_family for model in artifact.setup_models) == (
        SetupFamily.DFP,
        SetupFamily.LSR,
    )
    assert artifact.model_for(SetupFamily.DFP).half_life_real_completed_bars == 4
    assert artifact.model_for(SetupFamily.LSR).half_life_real_completed_bars == 3
    assert all(model.path_logit_coefficient > 0.0 for model in artifact.setup_models)
    assert all(model.dol_logit_coefficient > 0.0 for model in artifact.setup_models)
    assert artifact.fit_status == "fitted_not_admitted"
    assert artifact.admission_status == "CLOSED"
    payload = artifact.to_payload()
    assert payload["estimand"] == "target_before_invalidation"
    assert payload["lineage"]["path_likelihood_artifact_id"] == PATH_ARTIFACT_ID
    assert payload["lineage"]["dol_calibration_fit_artifact_id"] == DOL_FIT_SOURCE_ID
    assert [item["setup_family"] for item in payload["setup_models"]] == ["dfp", "lsr"]


def test_independent_rolling_oof_can_receive_separate_evidence_receipts() -> None:
    fit_cohort = _cohort(
        name="fit",
        role="development_cross_fit",
        windows=("train-2025",),
        folds=("fit",),
        identity="8" * 64,
    )
    dol_fit_rows = _dol_rows(
        start=pd.Timestamp("2025-01-02T14:00:00Z"),
        window="train-2025",
        folds=("fit",),
        prefix="dol-fit",
    )
    delivery_fit_rows = _delivery_rows(
        start=pd.Timestamp("2025-01-02T16:00:00Z"),
        window="train-2025",
        folds=("fit",),
        prefix="delivery-fit",
    )
    dol_artifact = fit_dol_temperature(
        dol_fit_rows,
        cohort=fit_cohort,
        lineage=_dol_lineage(),
    )
    delivery_artifact = fit_delivery_model(
        delivery_fit_rows,
        cohort=fit_cohort,
        lineage=_delivery_lineage(),
    )

    validation_cohort = _cohort(
        name="rolling-oof",
        role="rolling_oof",
        windows=("rolling-oof-2025",),
        folds=("oof-1", "oof-2"),
        identity="9" * 64,
    )
    validation_start = pd.Timestamp("2025-02-03T14:00:00Z")
    dol_validation = _dol_rows(
        start=validation_start,
        window="rolling-oof-2025",
        folds=("oof-1", "oof-2"),
        prefix="dol-oof",
    )
    delivery_validation = _delivery_rows(
        start=validation_start,
        window="rolling-oof-2025",
        folds=("oof-1", "oof-2"),
        prefix="delivery-oof",
    )
    dol_receipt = evaluate_dol_temperature_admission(
        dol_artifact,
        dol_validation,
        cohort=validation_cohort,
        thresholds=_lenient_thresholds(),
    )
    delivery_receipt = evaluate_delivery_model_admission(
        delivery_artifact,
        delivery_validation,
        cohort=validation_cohort,
        thresholds=_lenient_thresholds(),
    )
    assert dol_receipt.admitted is True
    assert delivery_receipt.admitted is True
    assert dol_receipt.receipt_id != delivery_receipt.receipt_id
    assert dol_receipt.artifact_kind.value == "dol_probability_calibration"
    assert delivery_receipt.artifact_kind.value == "target_before_invalidation_model"
    assert dol_receipt.action_authority is False
    assert delivery_receipt.action_authority is False
    assert dol_receipt.pins_issued is False
    assert delivery_receipt.pins_issued is False
    assert dol_receipt.fit_window_ids == ("train-2025",)
    assert dol_receipt.validation_window_ids == ("rolling-oof-2025",)
    assert dol_receipt.validation_fold_ids == ("oof-1", "oof-2")


def test_admission_rejects_cluster_overlap_and_split_protocol_drift() -> None:
    fit_cohort = _cohort(
        name="fit",
        role="development_cross_fit",
        windows=("train-2025",),
        folds=("fit",),
        identity="8" * 64,
    )
    fit_rows = _dol_rows(
        start=pd.Timestamp("2025-01-02T14:00:00Z"),
        window="train-2025",
        folds=("fit",),
        prefix="independence-fit",
    )
    artifact = fit_dol_temperature(
        fit_rows,
        cohort=fit_cohort,
        lineage=_dol_lineage(),
    )
    validation_cohort = replace(
        _cohort(
            name="rolling-oof",
            role="rolling_oof",
            windows=("rolling-oof-2025",),
            folds=("oof-1", "oof-2"),
            identity="9" * 64,
        ),
        split_protocol_sha256="c" * 64,
    )
    validation_rows = tuple(
        replace(item, cluster_id=f"independence-fit:session:{index // 2}")
        for index, item in enumerate(
            _dol_rows(
                start=pd.Timestamp("2025-02-03T14:00:00Z"),
                window="rolling-oof-2025",
                folds=("oof-1", "oof-2"),
                prefix="independence-validation",
            )
        )
    )
    receipt = evaluate_dol_temperature_admission(
        artifact,
        validation_rows,
        cohort=validation_cohort,
        thresholds=_lenient_thresholds(),
    )
    assert receipt.status == "CLOSED"
    assert "FIT_VALIDATION_CLUSTER_OVERLAP" in receipt.blockers
    assert "FIT_VALIDATION_SPLIT_PROTOCOL_MISMATCH" in receipt.blockers


def test_june_w1_w2_fit_artifacts_and_receipts_remain_closed() -> None:
    june = _cohort(
        name="june-development",
        role="historical_validation",
        windows=("2024-06-week-1",),
        folds=("W1", "W2"),
        identity="a" * 64,
    )
    with pytest.raises(SignalOutcomeFitError, match="cannot be relabelled"):
        replace(june, cohort_role="rolling_oof")

    dol_rows = _dol_rows(
        start=pd.Timestamp("2024-06-03T14:00:00Z"),
        window="2024-06-week-1",
        folds=("W1", "W2"),
        prefix="june-dol",
    )
    delivery_rows = _delivery_rows(
        start=pd.Timestamp("2024-06-03T16:00:00Z"),
        window="2024-06-week-1",
        folds=("W1", "W2"),
        prefix="june-delivery",
    )
    dol_artifact = fit_dol_temperature(
        dol_rows,
        cohort=june,
        lineage=_dol_lineage(),
    )
    delivery_artifact = fit_delivery_model(
        delivery_rows,
        cohort=june,
        lineage=_delivery_lineage(),
    )
    assert dol_artifact.fit_status == "fitted_not_admitted"
    assert delivery_artifact.fit_status == "fitted_not_admitted"
    assert dol_artifact.admission_status == "CLOSED"
    assert delivery_artifact.admission_status == "CLOSED"

    dol_receipt = evaluate_dol_temperature_admission(
        dol_artifact,
        dol_rows,
        cohort=june,
        thresholds=_lenient_thresholds(),
    )
    delivery_receipt = evaluate_delivery_model_admission(
        delivery_artifact,
        delivery_rows,
        cohort=june,
        thresholds=_lenient_thresholds(),
    )
    assert dol_receipt.status == "CLOSED"
    assert delivery_receipt.status == "CLOSED"
    assert dol_receipt.admitted is False
    assert delivery_receipt.admitted is False
    assert "COHORT_NOT_ROLLING_OOF" in dol_receipt.blockers
    assert "JUNE_2024_DEVELOPMENT_PERMANENTLY_CLOSED" in dol_receipt.blockers

    future_oof = _cohort(
        name="future-oof",
        role="rolling_oof",
        windows=("rolling-oof-2025",),
        folds=("oof-1", "oof-2"),
        identity="b" * 64,
    )
    future_start = pd.Timestamp("2025-02-03T14:00:00Z")
    future_dol = _dol_rows(
        start=future_start,
        window="rolling-oof-2025",
        folds=("oof-1", "oof-2"),
        prefix="june-fit-future-dol",
    )
    future_delivery = _delivery_rows(
        start=future_start,
        window="rolling-oof-2025",
        folds=("oof-1", "oof-2"),
        prefix="june-fit-future-delivery",
    )
    future_dol_receipt = evaluate_dol_temperature_admission(
        dol_artifact,
        future_dol,
        cohort=future_oof,
        thresholds=_lenient_thresholds(),
    )
    future_delivery_receipt = evaluate_delivery_model_admission(
        delivery_artifact,
        future_delivery,
        cohort=future_oof,
        thresholds=_lenient_thresholds(),
    )
    assert future_dol_receipt.status == "CLOSED"
    assert future_delivery_receipt.status == "CLOSED"
    assert (
        "FIT_SOURCE_JUNE_2024_PERMANENTLY_CLOSED"
        in future_dol_receipt.blockers
    )
    assert (
        "FIT_SOURCE_JUNE_2024_PERMANENTLY_CLOSED"
        in future_delivery_receipt.blockers
    )
