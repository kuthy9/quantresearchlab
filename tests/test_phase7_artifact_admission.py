from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from smc_trader.dol_probability import load_dol_probability_protocol
from smc_trader.dol_ranking import load_dol_ranking_protocol
from smc_trader.engine import ContinuousSMCEngine
from smc_trader.path_belief import load_path_belief_protocol
from smc_trader.signal_policy import (
    AdmittedDOLCalibrationArtifact,
    AdmittedPathLikelihoodArtifact,
    SetupDeliveryModel,
    SetupFamily,
    SignalArtifactPins,
    SignalPolicyArtifactError,
    TargetBeforeInvalidationArtifact,
    load_dol_calibration_artifact,
    load_outcome_model_artifact,
    load_path_likelihood_artifact,
    load_signal_artifact_pins,
    load_signal_policy_protocol,
)

from .test_brain_path_belief_integration import (
    ROOT,
    _admitted_path_protocol_file,
    _fitted_dol_artifact,
)


def _common_payload(artifact, *, artifact_kind: str) -> dict[str, object]:
    return {
        "artifact_kind": artifact_kind,
        "schema_version": artifact.schema_version,
        "protocol_id": artifact.protocol_id,
        "model_id": artifact.model_id,
        "model_version": artifact.model_version,
        "calibration_id": artifact.calibration_id,
        "source_dataset_id": artifact.source_dataset_id,
        "coverage_id": artifact.coverage_id,
        "trained_through": artifact.trained_through.isoformat(),
        "valid_from": artifact.valid_from.isoformat(),
        "expires_at": artifact.expires_at.isoformat(),
        "status": artifact.status,
        "authority": artifact.authority,
        "action_authority_ready": artifact.action_authority_ready,
        "artifact_id": artifact.artifact_id,
    }


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def _artifact_fixture(tmp_path: Path):
    valid_from = pd.Timestamp("2026-08-21 09:00", tz="America/New_York")
    path_source = _admitted_path_protocol_file(tmp_path)
    path_protocol = load_path_belief_protocol(path_source)
    ranking_protocol = load_dol_ranking_protocol(path_source)
    dol_protocol = load_dol_probability_protocol(
        ROOT / "configs/dol_probability.json"
    )
    signal_policy = load_signal_policy_protocol(
        ROOT / "configs/signal_policy.json"
    )
    dol_model = _fitted_dol_artifact(path_protocol)
    assert (
        dol_protocol.ranking_protocol_fingerprint
        == ranking_protocol.fingerprint
    )

    path_artifact = AdmittedPathLikelihoodArtifact(
        protocol_id="path-admission:loader-test:v1",
        model_id="path-likelihood:loader-test",
        model_version="path-likelihood-loader-test-v1",
        calibration_id="path-calibration:loader-test:v1",
        source_dataset_id="dataset:loader-test:train",
        coverage_id="path-coverage:loader-test",
        source_path_protocol_fingerprint=path_protocol.fingerprint,
        source_path_model_version=path_protocol.model_version,
        trained_through=valid_from - pd.Timedelta(days=2),
        valid_from=valid_from,
        expires_at=valid_from + pd.Timedelta(days=2),
    )
    dol_artifact = AdmittedDOLCalibrationArtifact(
        protocol_id="dol-admission:loader-test:v1",
        model_id="dol-calibration:loader-test",
        model_version="dol-calibration-loader-test-v1",
        calibration_id="dol-calibration-id:loader-test:v1",
        source_dataset_id="dataset:loader-test:train",
        coverage_id="dol-coverage:loader-test",
        source_dol_protocol_fingerprint=dol_protocol.fingerprint,
        source_dol_model_version=dol_protocol.model_version,
        source_dol_model_fingerprint=dol_model.fingerprint,
        source_path_protocol_fingerprint=path_protocol.fingerprint,
        source_path_model_version=path_protocol.model_version,
        trained_through=valid_from - pd.Timedelta(days=2),
        valid_from=valid_from,
        expires_at=valid_from + pd.Timedelta(days=2),
    )
    outcome_artifact = TargetBeforeInvalidationArtifact(
        protocol_id="outcome-admission:loader-test:v1",
        model_id="target-before-invalidation:loader-test",
        model_version="target-before-invalidation-loader-test-v1",
        calibration_id="outcome-calibration:loader-test:v1",
        source_dataset_id="dataset:loader-test:train",
        coverage_id="outcome-coverage:loader-test",
        path_likelihood_artifact_id=path_artifact.artifact_id,
        dol_calibration_artifact_id=dol_artifact.artifact_id,
        signal_policy_fingerprint=signal_policy.fingerprint,
        trained_through=valid_from - pd.Timedelta(days=2),
        valid_from=valid_from,
        expires_at=valid_from + pd.Timedelta(days=2),
        setup_models=(
            SetupDeliveryModel(
                setup_family=SetupFamily.DFP,
                intercept=1.0,
                path_logit_coefficient=0.5,
                dol_logit_coefficient=0.5,
                half_life_real_completed_bars=10,
            ),
            SetupDeliveryModel(
                setup_family=SetupFamily.LSR,
                intercept=0.5,
                path_logit_coefficient=0.4,
                dol_logit_coefficient=0.6,
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

    path_payload = {
        **_common_payload(
            path_artifact,
            artifact_kind="path_likelihood_calibration",
        ),
        "source_path_protocol_fingerprint": (
            path_artifact.source_path_protocol_fingerprint
        ),
        "source_path_model_version": (
            path_artifact.source_path_model_version
        ),
        "temperature": path_artifact.temperature,
        "path_log_biases": {
            path.value: value
            for path, value in path_artifact.path_log_biases
        },
    }
    dol_payload = {
        **_common_payload(
            dol_artifact,
            artifact_kind="dol_probability_calibration",
        ),
        "source_dol_protocol_fingerprint": (
            dol_artifact.source_dol_protocol_fingerprint
        ),
        "source_dol_model_version": dol_artifact.source_dol_model_version,
        "source_dol_model_fingerprint": (
            dol_artifact.source_dol_model_fingerprint
        ),
        "source_path_protocol_fingerprint": (
            dol_artifact.source_path_protocol_fingerprint
        ),
        "source_path_model_version": (
            dol_artifact.source_path_model_version
        ),
        "temperature": dol_artifact.temperature,
    }
    outcome_payload = {
        **_common_payload(
            outcome_artifact,
            artifact_kind="target_before_invalidation_model",
        ),
        "path_likelihood_artifact_id": (
            outcome_artifact.path_likelihood_artifact_id
        ),
        "dol_calibration_artifact_id": (
            outcome_artifact.dol_calibration_artifact_id
        ),
        "signal_policy_fingerprint": (
            outcome_artifact.signal_policy_fingerprint
        ),
        "setup_models": [
            {
                "setup_family": model.setup_family.value,
                "intercept": model.intercept,
                "path_logit_coefficient": model.path_logit_coefficient,
                "dol_logit_coefficient": model.dol_logit_coefficient,
                "half_life_real_completed_bars": (
                    model.half_life_real_completed_bars
                ),
            }
            for model in outcome_artifact.setup_models
        ],
        "minimum_supported_coverage": (
            outcome_artifact.minimum_supported_coverage
        ),
        "estimand": outcome_artifact.estimand,
    }
    pins_payload = {
        "artifact_kind": "signal_artifact_pins",
        "schema_version": 1,
        "signal_policy_fingerprint": pins.signal_policy_fingerprint,
        "path_protocol_fingerprint": pins.path_protocol_fingerprint,
        "path_likelihood_artifact_id": (
            pins.path_likelihood_artifact_id
        ),
        "dol_calibration_artifact_id": pins.dol_calibration_artifact_id,
        "outcome_model_artifact_id": pins.outcome_model_artifact_id,
        "dol_probability_model_fingerprint": (
            pins.dol_probability_model_fingerprint
        ),
        "admission_id": pins.admission_id,
    }
    model_payload = {
        "schema_version": dol_model.schema_version,
        "artifact_id": dol_model.artifact_id,
        "model_version": dol_model.model_version,
        "protocol_fingerprint": dol_model.protocol_fingerprint,
        "ranking_protocol_fingerprint": (
            dol_model.ranking_protocol_fingerprint
        ),
        "source_path_protocol_fingerprint": (
            dol_model.source_path_protocol_fingerprint
        ),
        "source_path_model_version": dol_model.source_path_model_version,
        "fit_status": dol_model.fit_status,
        "admission_status": dol_model.admission_status,
        "calibration_status": dol_model.calibration_status,
        "authority": dol_model.authority,
        "action_authority": dol_model.action_authority,
        "parameters": dol_model.parameters.payload(),
        "artifact_fingerprint": dol_model.fingerprint,
    }
    paths = {
        "path": _write(tmp_path / "path-likelihood.json", path_payload),
        "dol": _write(tmp_path / "dol-calibration.json", dol_payload),
        "outcome": _write(tmp_path / "outcome-model.json", outcome_payload),
        "pins": _write(tmp_path / "signal-pins.json", pins_payload),
        "dol_model": _write(tmp_path / "dol-model.json", model_payload),
    }
    return {
        "path_source": path_source,
        "path_protocol": path_protocol,
        "ranking_protocol": ranking_protocol,
        "dol_protocol": dol_protocol,
        "signal_policy": signal_policy,
        "dol_model": dol_model,
        "path_artifact": path_artifact,
        "dol_artifact": dol_artifact,
        "outcome_artifact": outcome_artifact,
        "pins": pins,
        "paths": paths,
    }


def test_versioned_artifact_loaders_require_external_exact_pins(
    tmp_path: Path,
) -> None:
    fixture = _artifact_fixture(tmp_path)
    paths = fixture["paths"]
    pins = load_signal_artifact_pins(
        paths["pins"],
        expected_admission_id=fixture["pins"].admission_id,
    )

    assert pins == fixture["pins"]
    assert load_path_likelihood_artifact(
        paths["path"],
        expected_artifact_id=pins.path_likelihood_artifact_id,
    ) == fixture["path_artifact"]
    assert load_dol_calibration_artifact(
        paths["dol"],
        expected_artifact_id=pins.dol_calibration_artifact_id,
    ) == fixture["dol_artifact"]
    assert load_outcome_model_artifact(
        paths["outcome"],
        expected_artifact_id=pins.outcome_model_artifact_id,
    ) == fixture["outcome_artifact"]

    tampered = json.loads(paths["path"].read_text(encoding="utf-8"))
    tampered["temperature"] = 2.0
    _write(paths["path"], tampered)
    with pytest.raises(SignalPolicyArtifactError, match="pin is stale"):
        load_path_likelihood_artifact(
            paths["path"],
            expected_artifact_id=pins.path_likelihood_artifact_id,
        )


def test_engine_admits_only_one_complete_externally_pinned_artifact_set(
    tmp_path: Path,
) -> None:
    fixture = _artifact_fixture(tmp_path)
    paths = fixture["paths"]
    config = json.loads(
        (ROOT / "configs/model.json").read_text(encoding="utf-8")
    )
    config["path_hypotheses"] = {
        "protocol": str(fixture["path_source"]),
        "path_protocol_fingerprint": fixture["path_protocol"].fingerprint,
        "dol_protocol_fingerprint": fixture["ranking_protocol"].fingerprint,
    }
    config["dol_probability"]["model_artifact"] = str(paths["dol_model"])
    config["dol_probability"]["model_artifact_fingerprint"] = fixture[
        "dol_model"
    ].fingerprint
    config["signal_policy"].update(
        {
            "path_likelihood_artifact": str(paths["path"]),
            "dol_calibration_artifact": str(paths["dol"]),
            "outcome_model_artifact": str(paths["outcome"]),
            "artifact_pins": {
                "path": str(paths["pins"]),
                "admission_id": fixture["pins"].admission_id,
            },
        }
    )
    admitted_config = _write(tmp_path / "model-admitted.json", config)

    engine = ContinuousSMCEngine.from_config(
        admitted_config,
        runtime_mode="development",
    )
    assert (
        engine.brain.path_likelihood_artifact.artifact_id
        == fixture["path_artifact"].artifact_id
    )
    assert (
        engine.brain.dol_calibration_artifact.artifact_id
        == fixture["dol_artifact"].artifact_id
    )
    assert (
        engine.brain.outcome_model_artifact.artifact_id
        == fixture["outcome_artifact"].artifact_id
    )
    assert engine.brain.signal_artifact_pins == fixture["pins"]

    partial = json.loads(
        (ROOT / "configs/model.json").read_text(encoding="utf-8")
    )
    partial["signal_policy"]["path_likelihood_artifact"] = str(paths["path"])
    partial_config = _write(tmp_path / "model-partial.json", partial)
    with pytest.raises(ValueError, match="one exact set"):
        ContinuousSMCEngine.from_config(
            partial_config,
            runtime_mode="development",
        )
