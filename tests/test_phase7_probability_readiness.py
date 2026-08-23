from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_phase7_probability_readiness.py"
CONTRACT = ROOT / "configs/phase7_probability_fit_admission.json"


def _module():
    spec = importlib.util.spec_from_file_location("phase7_readiness", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_2024_06_evidence_fails_closed_without_artifact(tmp_path: Path) -> None:
    report_path = tmp_path / "readiness.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--report", str(report_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    codes = {item["code"] for item in report["blockers"]}
    assert "PHASE6_SOURCE_FORBIDS_ARTIFACT_FIT" in codes
    assert "PHASE6_SOURCE_IS_MECHANISM_ASSOCIATION_ONLY" in codes
    assert "CONDITIONAL_LIKELIHOOD_LABELED_COHORT_MISSING" in codes
    assert "HYPOTHESIS_SPECIFIC_TEMPORAL_LABELED_COHORT_MISSING" in codes
    assert "PATH_CALIBRATION_LABELED_COHORT_MISSING" in codes
    assert "DOL_CALIBRATION_LABELED_COHORT_MISSING" in codes
    assert "REQUESTED_2024_06_COVERAGE_INCOMPLETE" in codes
    assert report["requested_window"]["verified_registry_role"] == "development"
    assert report["phase6_evidence_family_allowlist"] == [
        "acceptance_continuation",
        "displacement_impact",
    ]
    assert report["artifacts_written"] == []
    assert not list(tmp_path.glob("*artifact*"))


def test_required_schema_preserves_path_risk_and_no_target_labels() -> None:
    report = _module().audit_readiness(CONTRACT)

    assert "realized_path" in report["required_schema_gaps"]["conditional_likelihood"]
    assert "at_risk" in report["required_schema_gaps"]["hypothesis_specific_temporal"]
    assert "raw_path_probabilities" in report["required_schema_gaps"]["path_calibration"]
    assert "no_target_before_horizon" in report["required_schema_gaps"]["dol_calibration"]
    assert report["dol_no_target_outcome"] == "no_target_before_common_horizon"
    assert not report["ready_for_offline_fit"]


def test_phase6_hash_drift_is_error_not_fit_permission(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["phase6_source"]["result"]["sha256"] = "0" * 64
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(payload), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--contract", str(stale)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    error = json.loads(completed.stderr)
    assert "fail closed" in error["error"]
    assert error["artifacts_written"] == []


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__("status", "ready_for_offline_fit"),
        lambda value: value.__setitem__("protocol_version", "garbage_v999"),
        lambda value: value["requested_window"].__setitem__("id", "sealed-oos"),
        lambda value: value["runtime_bindings"].__setitem__(
            "production_override", True
        ),
    ),
)
def test_readiness_contract_identity_rejects_authority_drift(
    mutation,
    tmp_path: Path,
) -> None:
    payload = deepcopy(json.loads(CONTRACT.read_text(encoding="utf-8")))
    mutation(payload)
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    module = _module()

    with pytest.raises(module.ReadinessError, match="fail closed"):
        module.audit_readiness(changed)
