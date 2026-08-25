from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_semantic_signal_research as semantic_runner
from scripts import run_mbo_mechanism_research as phase6_runner
from scripts import run_phase6_mbo_foundation_comparison as comparison
from smc_trader.mbo_mechanism_research import (
    LEGACY_SYNTHETIC_SEMANTIC_EXCEPTION_POLICY,
    SYNTHETIC_SEMANTIC_EXCEPTION_POLICY,
    Phase6ResearchError,
)
from smc_trader.model import Bar, to_primitive
from smc_trader.scene_graph import scale_registry_id


_HISTORICAL_MANIFEST_SHA256 = {
    "W1": "1b832e72084684735bfe95b827c83282d0285ecdc1a5bdd83643036019b68b98",
    "W2": "4c6595c19bab5ed1d547edc4306e8c0647413242894ce7ffbf497df44be22584",
}


@pytest.mark.parametrize("window_key", ("W1", "W2"))
def test_committed_manifest_is_immutable_historical_runtime(window_key: str) -> None:
    path = comparison.ROOT / comparison.WINDOWS[window_key]["manifest"]
    committed = path.read_bytes()

    assert hashlib.sha256(committed).hexdigest() == (
        _HISTORICAL_MANIFEST_SHA256[window_key]
    )
    assert committed.decode("utf-8") != comparison.canonical_manifest_text(window_key)
    with pytest.raises(Phase6ResearchError, match="manifest drifted"):
        comparison.validate_window(window_key)


@pytest.mark.parametrize(
    ("window_key", "role", "window_id", "synthetic_clock"),
    (
        (
            "W1",
            "development_comparison",
            "2024-06-week-1",
            "2024-06-07T03:10:00Z",
        ),
        (
            "W2",
            "historical_validation_comparison",
            "2024-06-week-2",
            "2024-06-10T04:14:00Z",
        ),
    ),
)
def test_validate_only_binds_current_foundation_window_and_data(
    window_key: str,
    role: str,
    window_id: str,
    synthetic_clock: str,
) -> None:
    payload = comparison.build_manifest(window_key)
    comparison_contract = payload["comparison_contract"]
    active_window = payload["windows"][
        "primary" if window_key == "W1" else "underpowered_extension"
    ]

    assert active_window["id"] == window_id
    assert active_window["expected_rows"] == 6900
    assert comparison_contract["comparison_role"] == role
    assert comparison_contract["expected_completed_clocks"] == 6900
    assert comparison_contract["expected_real_completed_clocks"] == 6899
    assert comparison_contract["synthetic_decision_clocks"] == [
        synthetic_clock
    ]
    assert payload["foundation_version"] == "smc_semantic_foundation_v2.0"
    assert payload["foundation_registry_identity"] == (
        "ac04636919931d774309a0c306764fdf8eb53aee41df0f31d4d94e5b9125732b"
    )
    assert payload["authority"][
        "phase7_evidence_allowed_only_if_supported"
    ] is False
    assert comparison_contract["execution_authority"] == {
        "comparison_only": True,
        "rolling_oof": False,
        "sealed_oos": False,
        "model_admission": False,
        "trading": False,
    }
    assert comparison_contract["processed_phase6_artifact"] == payload[
        "identity_bindings"
    ]["mbo_feature_artifact"]
    assert comparison_contract["raw_partition_manifest"] == payload[
        "identity_bindings"
    ]["raw_mbo_partition_manifest"]
    assert payload["synthetic_semantic_exception_policy"] == (
        SYNTHETIC_SEMANTIC_EXCEPTION_POLICY
    )
    assert comparison_contract[
        "synthetic_semantic_exception_policy_lineage"
    ]["effective_comparison_policy"] == "current_full_interval_roots"
    assert comparison_contract[
        "synthetic_semantic_exception_policy_lineage"
    ]["historical_source_manifest_mutated"] is False


@pytest.mark.parametrize(
    ("window_key", "source_policy", "source_policy_version"),
    (
        (
            "W1",
            LEGACY_SYNTHETIC_SEMANTIC_EXCEPTION_POLICY,
            "legacy_single_clock_root",
        ),
        (
            "W2",
            SYNTHETIC_SEMANTIC_EXCEPTION_POLICY,
            "current_full_interval_roots",
        ),
    ),
)
def test_comparison_uses_current_policy_without_mutating_historical_source(
    window_key: str,
    source_policy: object,
    source_policy_version: str,
) -> None:
    source_path = (
        comparison.ROOT / comparison.WINDOWS[window_key]["source_manifest"]
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    effective = comparison.build_manifest(window_key)

    assert source["synthetic_semantic_exception_policy"] == source_policy
    assert effective["synthetic_semantic_exception_policy"] == (
        SYNTHETIC_SEMANTIC_EXCEPTION_POLICY
    )
    assert effective["comparison_contract"][
        "synthetic_semantic_exception_policy_lineage"
    ] == {
        "source_manifest_policy": source_policy_version,
        "effective_comparison_policy": "current_full_interval_roots",
        "historical_source_manifest_mutated": False,
    }


def test_comparison_loader_rejects_legacy_policy_before_identity_reads(
    tmp_path: Path,
) -> None:
    payload = comparison.build_manifest("W1")
    payload["synthetic_semantic_exception_policy"] = deepcopy(
        LEGACY_SYNTHETIC_SEMANTIC_EXCEPTION_POLICY
    )
    manifest = tmp_path / "comparison.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        Phase6ResearchError,
        match="comparison requires the current synthetic semantic exception policy",
    ):
        comparison.load_frozen_phase6_contract(
            manifest,
            root=tmp_path,
            verify_raw_partition_hashes=False,
            comparison_validation_only=True,
        )


def test_comparison_builder_rejects_source_policy_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_read_json = comparison._read_json

    def drifted_read_json(path: Path, *, label: str) -> dict[str, object]:
        payload = original_read_json(path, label=label)
        if label == "source Phase-6 manifest":
            payload["synthetic_semantic_exception_policy"] = {"drifted": True}
        return payload

    monkeypatch.setattr(comparison, "_read_json", drifted_read_json)

    with pytest.raises(
        Phase6ResearchError,
        match="source Phase-6 synthetic semantic exception policy changed",
    ):
        comparison.build_manifest("W1")


def test_phase6_eye_builder_exactly_matches_phase45_foundation_runtime() -> None:
    model_path = comparison.ROOT / comparison.MODEL_PATH
    phase45_reader, phase45_observer = semantic_runner._build_eye(model_path)
    phase6_reader, phase6_observer = phase6_runner._build_eye(model_path)

    assert phase6_observer.config.canonical_foundation_enabled is True
    assert phase6_observer._foundation_adapter is not None
    assert phase6_observer.config == phase45_observer.config
    assert scale_registry_id(phase6_reader.scale_specs) == scale_registry_id(
        phase45_reader.scale_specs
    )

    start = pd.Timestamp("2025-01-05 18:00", tz="America/New_York")
    bars = tuple(
        Bar(
            start=start + pd.Timedelta(index, unit="min"),
            open=20_000.0 + index * 0.25,
            high=20_000.5 + index * 0.25,
            low=19_999.5 + index * 0.25,
            close=20_000.25 + index * 0.25,
            volume=100.0 + index,
            symbol="NQH5",
            instrument_id=1,
        )
        for index in range(3)
    )
    for bar in bars:
        phase45_observation = phase45_observer.observe(
            phase45_reader.on_bar(bar)
        )
        phase6_observation = phase6_observer.observe(
            phase6_reader.on_bar(bar)
        )
        assert to_primitive(phase6_observation) == to_primitive(
            phase45_observation
        )

    assert len(phase6_observer.audit_store) == len(
        phase45_observer.audit_store
    )
    assert phase6_observer.audit_store.fingerprint() == (
        phase45_observer.audit_store.fingerprint()
    )
    assert phase6_observer._foundation_adapter.checkpoint() == (
        phase45_observer._foundation_adapter.checkpoint()
    )


def test_w2_prior_result_is_gate_and_warmup_only() -> None:
    payload = comparison.build_manifest("W2")

    assert payload["week1_gate_result"] is not None
    assert payload["comparison_contract"]["prior_week1_policy"] == (
        "gate_and_warmup_audit_only_prior_pairs_excluded_from_inference"
    )
    assert payload["comparison_contract"][
        "historical_manifests_and_results_mutable"
    ] is False


def test_manifest_tampering_fails_closed() -> None:
    payload = comparison.build_manifest("W1")
    tampered = deepcopy(payload)
    tampered["comparison_contract"]["expected_completed_clocks"] = 6899

    with pytest.raises(Phase6ResearchError, match="manifest drifted"):
        comparison._validate_exact_payload(tampered, "W1")


def test_comparison_outputs_cannot_target_historical_results() -> None:
    registered = {
        (comparison.ROOT / str(item["output"])).resolve()
        for item in comparison.WINDOWS.values()
    }
    historical = set(
        (comparison.ROOT / "experiments/results").glob(
            "smc_semantics_v1_2_2024_06_phase6_mbo*.json"
        )
    )

    assert registered.isdisjoint(path.resolve() for path in historical)
    assert not any(path.exists() for path in registered)


def test_wrapper_delegates_to_existing_runner_with_comparison_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_validate(
        window_key: str,
        *,
        verify_raw_partition_hashes: bool = False,
    ) -> None:
        captured["validated"] = (
            window_key,
            verify_raw_partition_hashes,
        )

    def fake_run(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(comparison, "validate_window", fake_validate)
    monkeypatch.setattr(phase6_runner, "run", fake_run)

    assert comparison.execute("W1") == {"status": "ok"}
    assert captured["validated"] == ("W1", False)
    assert captured["manifest_path"] == (
        comparison.ROOT / comparison.WINDOWS["W1"]["manifest"]
    ).resolve()
    assert captured["output"] == (
        comparison.ROOT / comparison.WINDOWS["W1"]["output"]
    ).resolve()
    assert captured["verify_raw_partition_hashes"] is True
    assert captured["comparison_validation_only"] is True


def test_base_runner_seam_defaults_to_historical_behavior() -> None:
    run_default = inspect.signature(phase6_runner.run).parameters[
        "comparison_validation_only"
    ].default
    loader_default = inspect.signature(
        comparison.load_frozen_phase6_contract
    ).parameters["comparison_validation_only"].default

    assert run_default is False
    assert loader_default is False


def test_comparison_manifest_requires_explicit_loader_seam() -> None:
    manifest = (
        comparison.ROOT / comparison.WINDOWS["W1"]["manifest"]
    ).resolve()

    with pytest.raises(Phase6ResearchError, match="authority must fail closed"):
        comparison.load_frozen_phase6_contract(
            manifest,
            root=comparison.ROOT,
            verify_raw_partition_hashes=False,
        )


def test_comparison_report_cannot_claim_evidence_admission() -> None:
    historical = json.loads(
        (
            comparison.ROOT
            / "experiments/results/"
            "smc_semantics_v1_2_2024_06_phase6_mbo_week1_v5.json"
        ).read_text(encoding="utf-8")
    )
    historical["comparison_validation_only"] = True
    historical["comparison_supported_mechanisms_not_admitted"] = [
        "displacement_impact"
    ]
    historical["phase7_evidence_allowlist"] = []
    historical["coverage"]["synthetic_semantic_exception_gate"][
        "allowed_by_clock_scope"
    ] = {}

    report = phase6_runner._report(historical)

    assert "## Comparison containment" in report
    assert "No mechanism is admitted" in report
    assert "## Phase-7 evidence admission" not in report


def test_base_runner_no_clobber_precedes_contract_or_data_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "already-exists.json"
    output.write_text(json.dumps({"sentinel": True}), encoding="utf-8")
    monkeypatch.setattr(phase6_runner, "ROOT", tmp_path)

    with pytest.raises(FileExistsError, match="already exists"):
        phase6_runner.run(
            manifest_path=tmp_path / "not-opened.yaml",
            output=output,
            comparison_validation_only=True,
        )


def test_manifests_remain_json_subset_yaml() -> None:
    for window in comparison.WINDOWS.values():
        path = comparison.ROOT / window["manifest"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["comparison_contract_version"] == 1
        assert payload["comparison_status"] == comparison.COMPARISON_STATUS
