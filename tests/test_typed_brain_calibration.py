from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from smc_trader.calibration import (
    CalibrationError,
    TYPED_CALIBRATION_DIMENSIONS,
    TypedBrainCalibrator,
)
from smc_trader.engine import ContinuousSMCEngine
from smc_trader.model import Playbook


DFP = Playbook.DISPLACEMENT_FIRST_PULLBACK
LSR = Playbook.LIQUIDITY_SWEEP_REVERSAL
FAVR = Playbook.FAILED_AUCTION_VALUE_RETURN
REGISTRY_HASH = "a" * 64


def _dimension_map(offset: float = 0.0) -> dict[str, object]:
    return {
        "episodes": 100,
        "points": [
            {
                "raw_value": 0.0,
                "calibrated_value": offset,
                "episodes": 50,
            },
            {
                "raw_value": 1.0,
                "calibrated_value": 1.0,
                "episodes": 50,
            },
        ],
    }


def _artifact() -> dict[str, object]:
    return {
        "calibration_version": "typed-test",
        "status": "ready",
        "playbook_registry_hash": REGISTRY_HASH,
        "playbooks": {
            DFP.value: {
                "status": "active",
                "dimensions": {
                    dimension: _dimension_map(0.2)
                    for dimension in TYPED_CALIBRATION_DIMENSIONS
                },
            },
            LSR.value: {
                "status": "active",
                "dimensions": {
                    dimension: _dimension_map(0.1)
                    for dimension in TYPED_CALIBRATION_DIMENSIONS
                },
            },
            FAVR.value: {
                "status": "parked_missing_natural_authority",
                "dimensions": {},
            },
        },
    }


def _write_artifact(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "typed_calibration.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return path


def _load(tmp_path: Path, payload: dict[str, object]) -> TypedBrainCalibrator:
    return TypedBrainCalibrator.from_file(
        _write_artifact(tmp_path, payload),
        expected_registry_hash=REGISTRY_HASH,
    )


def test_typed_calibrator_maps_active_dimensions_and_passes_sequence(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, _artifact())
    calibrator = TypedBrainCalibrator.from_file(
        path,
        expected_registry_hash=REGISTRY_HASH,
    )

    assert calibrator.version == "typed-test"
    assert calibrator.apply(DFP, "thesis_strength", 0.5) == pytest.approx(0.6)
    assert calibrator.apply(LSR, "uncertainty", 0.5) == pytest.approx(0.55)
    assert calibrator.apply(DFP, "sequence_progress", 0.37) == 0.37
    with pytest.raises(CalibrationError, match="parked"):
        calibrator.apply(FAVR, "thesis_strength", 0.5)


def test_typed_identity_returns_raw_for_every_dimension_and_playbook() -> None:
    calibrator = TypedBrainCalibrator.identity()
    for playbook in Playbook:
        for dimension in (*TYPED_CALIBRATION_DIMENSIONS, "sequence_progress"):
            assert calibrator.apply(playbook, dimension, 0.43) == 0.43


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("status", "draft"), "only a ready"),
        (
            lambda value: value.__setitem__("playbook_registry_hash", "c" * 64),
            "registry hash is stale",
        ),
        (
            lambda value: value["playbooks"][DFP.value]["dimensions"].pop(
                "delivery_quality"
            ),
            "exactly the five",
        ),
        (
            lambda value: value["playbooks"][DFP.value]["dimensions"][
                "thesis_strength"
            ]["points"][1].__setitem__("calibrated_value", 0.1),
            "not monotone",
        ),
        (
            lambda value: value["playbooks"][FAVR.value].__setitem__(
                "status", "active"
            ),
            "must be parked",
        ),
    ],
)
def test_typed_artifact_fails_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = copy.deepcopy(_artifact())
    mutation(payload)
    with pytest.raises(CalibrationError, match=message):
        _load(tmp_path, payload)


def test_engine_accepts_current_typed_config_and_rejects_incomplete_current_config(
    tmp_path: Path,
) -> None:
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )

    assert isinstance(engine.brain.calibrator, TypedBrainCalibrator)
    assert engine.runtime_mode == "development"
    with pytest.raises(TypeError, match="runtime_mode"):
        ContinuousSMCEngine.from_config("configs/model.json")
    with pytest.raises(RuntimeError, match="readiness gate"):
        ContinuousSMCEngine(
            reader=engine.reader,
            observer=engine.observer,
            brain=engine.brain,
            decision=engine.decision,
            risk=engine.risk,
            runtime_mode="live",
        )
    incomplete = tmp_path / "model.json"
    incomplete.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="model.scales"):
        ContinuousSMCEngine.from_config(
            incomplete,
            runtime_mode="development",
        )

    payload = json.loads(Path("configs/model.json").read_text(encoding="utf-8"))
    payload["observer"].pop("group5_protocol")
    incomplete.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="group5_protocol"):
        ContinuousSMCEngine.from_config(
            incomplete,
            runtime_mode="development",
        )


def test_engine_live_mode_has_one_fail_closed_release_gate(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="live execution readiness"):
        ContinuousSMCEngine.from_config(
            "configs/model.json",
            runtime_mode="live",
        )

    group5 = json.loads(
        Path("configs/primitives_entry.json").read_text(encoding="utf-8")
    )
    group5["authority"]["natural_authority_validated"] = True
    group5_path = tmp_path / "primitives_entry.json"
    group5_path.write_text(json.dumps(group5), encoding="utf-8")

    model = json.loads(
        Path("configs/model.json").read_text(encoding="utf-8")
    )
    model["observer"]["group5_protocol"] = str(group5_path)
    model["release_readiness"] = {
        "active_model_natural_authority_validated": True,
        "economic_validation_complete": True,
        "live_execution_allowed": True,
    }
    model_path = tmp_path / "model-live.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")

    engine = ContinuousSMCEngine.from_config(
        model_path,
        runtime_mode="live",
    )
    assert isinstance(engine, ContinuousSMCEngine)
    assert engine.runtime_mode == "live"

    with pytest.raises(ValueError, match="runtime_mode"):
        ContinuousSMCEngine.from_config(
            "configs/model.json",
            runtime_mode="paper",
        )
