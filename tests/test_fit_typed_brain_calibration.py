from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.fit_typed_brain_calibration import (
    FITTED_DIMENSIONS,
    UNCERTAINTY_FORMULA_VERSION,
    fit_typed_brain_calibration,
    resolve_model_bindings,
)
from smc_trader.calibration import (
    CalibrationError,
    DimensionReliabilityMap,
    DimensionReliabilityPoint,
    TypedBrainCalibrator,
    monotone_reliability_points,
)
from smc_trader.model import Playbook


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "configs/model_v3_development.json"
VALIDATION_PROTOCOL = ROOT / "configs/validation_protocol_v2.json"
DFP = Playbook.DISPLACEMENT_FIRST_PULLBACK
LSR = Playbook.LIQUIDITY_SWEEP_REVERSAL
FAVR = Playbook.FAILED_AUCTION_VALUE_RETURN


def _identity_fields(bindings: dict[str, object], token: str) -> dict[str, object]:
    return {
        "hypothesis_key": token,
        "scene_hypothesis_id": f"scene:{token}",
        "competing_scene_hypothesis_ids": "[]",
        "context_root_ids": json.dumps([f"root:{token}"], separators=(",", ":")),
        "scene_revision_id": "scene-revision-1",
        "liquidity_route_id": f"route:{token}",
        "context_draw_id": f"context-draw:{token}",
        "intermediate_liquidity_ids": "[]",
        "primary_deliverable_target_id": f"target:{token}",
        "terminal_draw_id": f"terminal:{token}",
        "path_blocker_ids": "[]",
        "source_path_ids": json.dumps(
            [f"target:{token}"],
            separators=(",", ":"),
        ),
        "brain_input_contract_hash": bindings[
            "brain_input_contract_hash"
        ],
    }


def _rows() -> pd.DataFrame:
    bindings = resolve_model_bindings(MODEL_CONFIG)
    primitive_json = json.dumps(
        bindings["primitive_protocol_hashes"],
        sort_keys=True,
        separators=(",", ":"),
    )
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2022-02-01T09:31:00-05:00")
    for playbook in (DFP, LSR):
        version = bindings["playbook_protocol_versions"][playbook.value]
        for dimension_index, dimension in enumerate(FITTED_DIMENSIONS):
            for index in range(8):
                sampled_at = start + pd.Timedelta(
                    days=dimension_index,
                    minutes=index,
                )
                rows.append(
                    {
                        **_identity_fields(
                            bindings,
                            f"{playbook.value}:{dimension}:{index}",
                        ),
                        "sample_id": f"{playbook.value}:{dimension}:{index}",
                        "playbook": playbook.value,
                        "direction": "long" if index % 2 == 0 else "short",
                        "dimension": dimension,
                        "sampled_at": sampled_at,
                        "resolved_at": sampled_at + pd.Timedelta(minutes=5),
                        "raw_value": (index + 1) / 10.0,
                        "outcome_value": float(index >= 4),
                        "resolution": "causal_target_resolved",
                        "censored": False,
                        "fit_eligible": True,
                        "protocol_version": version,
                        "protocol_hash": bindings["registry_hash"],
                        "registry_hash": bindings["registry_hash"],
                        "model_code_hash": bindings["model_code_hash"],
                        "config_hash": bindings["config_hash"],
                        "primitive_protocol_hashes": primitive_json,
                    }
                )
            # This future-unresolved row proves censorship is excluded instead
            # of being silently treated as a zero outcome.
            rows.append(
                {
                    **_identity_fields(
                        bindings,
                        f"{playbook.value}:{dimension}:censored",
                    ),
                    "sample_id": f"{playbook.value}:{dimension}:censored",
                    "playbook": playbook.value,
                    "direction": "long",
                    "dimension": dimension,
                    "sampled_at": start + pd.Timedelta(days=10),
                    "resolved_at": start + pd.Timedelta(days=11),
                    "raw_value": 0.99,
                    "outcome_value": None,
                    "resolution": "window_end_censored",
                    "censored": True,
                    "fit_eligible": False,
                    "protocol_version": version,
                    "protocol_hash": bindings["registry_hash"],
                    "registry_hash": bindings["registry_hash"],
                    "model_code_hash": bindings["model_code_hash"],
                    "config_hash": bindings["config_hash"],
                    "primitive_protocol_hashes": primitive_json,
                }
            )
        for dimension in ("sequence_progress", "uncertainty"):
            rows.append(
                {
                    **_identity_fields(
                        bindings,
                        f"{playbook.value}:{dimension}:observed",
                    ),
                    "sample_id": f"{playbook.value}:{dimension}:observed",
                    "playbook": playbook.value,
                    "direction": "short",
                    "dimension": dimension,
                    "sampled_at": start,
                    "resolved_at": start,
                    "raw_value": 0.6,
                    "outcome_value": None,
                    "resolution": f"{dimension}_observed",
                    "censored": False,
                    "fit_eligible": False,
                    "protocol_version": version,
                    "protocol_hash": bindings["registry_hash"],
                    "registry_hash": bindings["registry_hash"],
                    "model_code_hash": bindings["model_code_hash"],
                    "config_hash": bindings["config_hash"],
                    "primitive_protocol_hashes": primitive_json,
                }
            )
    return pd.DataFrame(rows)


def _fit(tmp_path: Path, frame: pd.DataFrame) -> tuple[Path, dict[str, object]]:
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    output = tmp_path / "typed-calibration.json"
    payload = fit_typed_brain_calibration(
        row_paths=[rows],
        output=output,
        model_config=MODEL_CONFIG,
        validation_protocol=VALIDATION_PROTOCOL,
        bins=4,
        minimum_bin_samples=2,
        minimum_dimension_samples=8,
        calibration_version="4.0.0-typed-test-fit.1",
    )
    return output, payload


def test_fitter_builds_loader_valid_typed_artifact_without_pnl(tmp_path: Path) -> None:
    output, payload = _fit(tmp_path, _rows())
    bindings = resolve_model_bindings(MODEL_CONFIG)

    calibrator = TypedBrainCalibrator.from_file(
        output,
        expected_registry_hash=bindings["registry_hash"],
        expected_code_hash=bindings["model_code_hash"],
        expected_primitive_protocol_hashes=bindings["primitive_protocol_hashes"],
        expected_brain_input_contract_hash=bindings[
            "brain_input_contract_hash"
        ],
    )

    assert calibrator.status == "ready"
    assert payload["method"]["pnl_labels_used"] is False
    assert payload["method"]["threshold_search"] is False
    assert payload["method"]["sequence_progress"] == (
        "deterministic_passthrough_not_fitted"
    )
    assert payload["method"]["uncertainty_formula_version"] == (
        UNCERTAINTY_FORMULA_VERSION
    )
    for playbook in (DFP, LSR):
        dimensions = payload["playbooks"][playbook.value]["dimensions"]
        assert set(dimensions) == {*FITTED_DIMENSIONS, "uncertainty"}
        assert all(dimensions[name]["episodes"] == 8 for name in FITTED_DIMENSIONS)
        assert dimensions["uncertainty"]["status"] == (
            "authorized_formula_passthrough"
        )
        assert calibrator.apply(playbook, "uncertainty", 0.37) == 0.37
        assert calibrator.apply(playbook, "sequence_progress", 0.61) == 0.61
    assert payload["playbooks"][FAVR.value] == {
        "status": "parked_missing_natural_authority",
        "dimensions": {},
    }


def test_fitter_fails_closed_when_one_dimension_has_one_raw_level(tmp_path: Path) -> None:
    frame = _rows()
    mask = frame["playbook"].eq(DFP.value) & frame["dimension"].eq(
        "thesis_strength"
    ) & frame["fit_eligible"]
    frame.loc[mask, "raw_value"] = 0.5
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)

    with pytest.raises(CalibrationError, match="fewer than two unique raw values"):
        fit_typed_brain_calibration(
            row_paths=[rows],
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
            minimum_dimension_samples=8,
        )


def test_fitter_keeps_large_equal_raw_ties_in_one_support_level(
    tmp_path: Path,
) -> None:
    frame = _rows()
    mask = (
        frame["playbook"].eq(LSR.value)
        & frame["dimension"].eq("thesis_strength")
        & frame["fit_eligible"]
    )
    indices = frame.index[mask].tolist()
    frame.loc[indices[:4], "raw_value"] = 0.0
    frame.loc[indices[4:], "raw_value"] = [0.2, 0.4, 0.6, 0.8]

    _, payload = _fit(tmp_path, frame)
    points = payload["playbooks"][LSR.value]["dimensions"][
        "thesis_strength"
    ]["points"]

    raw_values = [point["raw_value"] for point in points]
    assert len(raw_values) >= 2
    assert all(
        right > left
        for left, right in zip(raw_values[:-1], raw_values[1:])
    )
    assert any(point["episodes"] >= 4 for point in points)


def test_reliability_bins_choose_a_legal_boundary_before_a_trailing_tie() -> None:
    points = monotone_reliability_points(
        [0.1] * 3 + [0.9] * 7,
        [0.0] * 3 + [1.0] * 7,
        bins=2,
        minimum_bin_episodes=2,
    )

    assert [point.raw_probability for point in points] == pytest.approx([0.1, 0.9])
    assert [point.episodes for point in points] == [3, 7]


def test_weighted_pava_preserves_a_flat_map_across_pooled_bins() -> None:
    points = monotone_reliability_points(
        [0.1, 0.1, 0.3, 0.3, 0.7, 0.7],
        [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        bins=3,
        minimum_bin_episodes=2,
    )
    mapping = DimensionReliabilityMap(
        playbook=LSR,
        dimension="thesis_strength",
        episodes=6,
        points=tuple(
            DimensionReliabilityPoint(
                raw_value=point.raw_probability,
                calibrated_value=point.calibrated_probability,
                episodes=point.episodes,
            )
            for point in points
        ),
    )

    assert [point.raw_probability for point in points] == pytest.approx(
        [0.1, 0.3, 0.7]
    )
    assert points[0].calibrated_probability == pytest.approx(
        points[1].calibrated_probability
    )
    assert mapping.apply(0.2) == pytest.approx(points[0].calibrated_probability)
    assert mapping.apply(0.3) == pytest.approx(points[1].calibrated_probability)


def test_fitter_rejects_stale_primitive_binding(tmp_path: Path) -> None:
    frame = _rows()
    stale = json.loads(frame.loc[0, "primitive_protocol_hashes"])
    stale["group5_protocol"] = "0" * 64
    frame.loc[0, "primitive_protocol_hashes"] = json.dumps(
        stale,
        sort_keys=True,
        separators=(",", ":"),
    )
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)

    with pytest.raises(ValueError, match="primitive protocol hashes are stale"):
        fit_typed_brain_calibration(
            row_paths=[rows],
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
            minimum_dimension_samples=8,
        )


def test_fitter_rejects_legacy_rows_without_input_contract(tmp_path: Path) -> None:
    frame = _rows().drop(columns=["brain_input_contract_hash"])
    rows = tmp_path / "legacy-rows.parquet"
    frame.to_parquet(rows, index=False)

    with pytest.raises(ValueError, match="omit fields"):
        fit_typed_brain_calibration(
            row_paths=[rows],
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
            minimum_dimension_samples=8,
        )


def test_fitter_rejects_stale_brain_input_contract(tmp_path: Path) -> None:
    frame = _rows()
    frame.loc[0, "brain_input_contract_hash"] = "0" * 64
    rows = tmp_path / "stale-contract.parquet"
    frame.to_parquet(rows, index=False)

    with pytest.raises(ValueError, match="mix or omit brain_input_contract_hash"):
        fit_typed_brain_calibration(
            row_paths=[rows],
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
            minimum_dimension_samples=8,
        )


def test_fitter_rejects_economic_labels_in_recorder_rows(tmp_path: Path) -> None:
    frame = _rows()
    frame["net_R"] = 0.0
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)

    with pytest.raises(ValueError, match="must not contain economic labels"):
        fit_typed_brain_calibration(
            row_paths=[rows],
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
            minimum_dimension_samples=8,
        )
