from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from smc_trader.calibration import (
    CalibrationError,
    MODEL_CODE_FILES,
    ProbabilityCalibrator,
    TYPED_CALIBRATION_DIMENSIONS,
    TypedBrainCalibrator,
)
from smc_trader.engine import (
    ContinuousSMCEngine,
    _configured_primitive_protocol_hashes,
)
from smc_trader.model import Playbook


DFP = Playbook.DISPLACEMENT_FIRST_PULLBACK
LSR = Playbook.LIQUIDITY_SWEEP_REVERSAL
FAVR = Playbook.FAILED_AUCTION_VALUE_RETURN
REGISTRY_HASH = "a" * 64
CODE_HASH = "b" * 64
BRAIN_INPUT_CONTRACT_HASH = "d" * 64
PROTOCOL_HASHES = {
    "structure_protocol": "1" * 64,
    "liquidity_protocol": "2" * 64,
    "displacement_protocol": "3" * 64,
    "group3_protocol": "4" * 64,
    "group4_protocol": "5" * 64,
    "group5_protocol": "6" * 64,
}


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
        "calibration_version": "4.0.0-typed-test.1",
        "status": "ready",
        "playbook_registry_hash": REGISTRY_HASH,
        "model_code_hash": CODE_HASH,
        "primitive_protocol_hashes": dict(PROTOCOL_HASHES),
        "brain_input_contract_hash": BRAIN_INPUT_CONTRACT_HASH,
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
        expected_code_hash=CODE_HASH,
        expected_primitive_protocol_hashes=PROTOCOL_HASHES,
        expected_brain_input_contract_hash=BRAIN_INPUT_CONTRACT_HASH,
    )


def test_typed_calibrator_maps_active_dimensions_and_passes_sequence(tmp_path: Path) -> None:
    path = _write_artifact(tmp_path, _artifact())
    calibrator = TypedBrainCalibrator.from_file(
        path,
        expected_registry_hash=REGISTRY_HASH,
        expected_code_hash=CODE_HASH,
        expected_primitive_protocol_hashes=PROTOCOL_HASHES,
        expected_brain_input_contract_hash=BRAIN_INPUT_CONTRACT_HASH,
    )

    assert calibrator.version == "4.0.0-typed-test.1"
    assert calibrator.brain_input_contract_hash == BRAIN_INPUT_CONTRACT_HASH
    assert calibrator.fingerprint == hashlib.sha256(path.read_bytes()).hexdigest()
    assert calibrator.apply(DFP, "thesis_strength", 0.5) == pytest.approx(0.6)
    assert calibrator.apply(LSR, "uncertainty", 0.5) == pytest.approx(0.55)
    assert calibrator.apply(DFP, "sequence_progress", 0.37) == 0.37
    with pytest.raises(CalibrationError, match="parked"):
        calibrator.apply(FAVR, "thesis_strength", 0.5)


def test_scene_graph_is_part_of_model_code_identity() -> None:
    assert "scene_graph.py" in MODEL_CODE_FILES


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
            lambda value: value.__setitem__("model_code_hash", "c" * 64),
            "model-code hash is stale",
        ),
        (
            lambda value: value["primitive_protocol_hashes"].__setitem__(
                "group5_protocol", "c" * 64
            ),
            "primitive protocol hashes are stale",
        ),
        (
            lambda value: value.pop("brain_input_contract_hash"),
            "Brain input contract hash is stale",
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


def test_engine_hashes_exact_configured_protocol_bytes(tmp_path: Path) -> None:
    observer: dict[str, str] = {}
    for index, field in enumerate(PROTOCOL_HASHES, start=1):
        protocol = tmp_path / f"{field}.json"
        protocol.write_bytes(f"protocol-{index}\n".encode())
        observer[field] = str(protocol)

    hashes = _configured_primitive_protocol_hashes(observer)

    assert hashes == {
        field: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for field, path in observer.items()
    }


def test_engine_selects_calibrator_family_from_registry_version() -> None:
    typed = ContinuousSMCEngine.from_config("configs/model_v3_development.json")
    legacy = ContinuousSMCEngine.from_config(
        "configs/model_v2_1_belief_identity.json"
    )

    assert isinstance(typed.brain.calibrator, TypedBrainCalibrator)
    assert isinstance(legacy.brain.calibrator, ProbabilityCalibrator)
