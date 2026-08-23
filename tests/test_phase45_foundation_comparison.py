from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import run_phase45_foundation_comparison as comparison
from scripts import run_semantic_signal_research as semantic_runner
from smc_trader.signal_research import ResearchContractError


@pytest.mark.parametrize("window_key", ("W1", "W2"))
def test_committed_manifest_is_exact_canonical_render(window_key: str) -> None:
    path = comparison.ROOT / comparison.WINDOWS[window_key]["manifest"]

    assert path.read_text(encoding="utf-8") == comparison.canonical_manifest_text(
        window_key
    )


@pytest.mark.parametrize(
    ("window_key", "role", "emitted"),
    (
        ("W1", "development_comparison", 13560),
        ("W2", "historical_validation_comparison", 20460),
    ),
)
def test_contract_only_validation_binds_window_foundation_and_census(
    window_key: str,
    role: str,
    emitted: int,
) -> None:
    contract = comparison.validate_window(window_key)
    payload = contract.payload

    assert payload["comparison_contract"]["comparison_role"] == role
    assert payload["comparison_contract"]["expected_contracts"] == [
        {"symbol": "NQM4", "instrument_id": 13743}
    ]
    assert payload["foundation_version"] == "smc_semantic_foundation_v2.0"
    assert (
        payload["foundation_registry_identity"]
        == "ac04636919931d774309a0c306764fdf8eb53aee41df0f31d4d94e5b9125732b"
    )
    assert payload["input_census"][
        "expected_emitted_bars_including_warmup"
    ] == emitted
    assert payload["input_census"]["expected_diagnostic_completed_bars"] == 6900
    assert payload["input_census"]["expected_diagnostic_real_rows"] == 6899
    assert payload["input_census"]["expected_diagnostic_synthetic_bars"] == 1
    assert payload["input_census"]["expected_synthetic_clocks"] == payload[
        "comparison_contract"
    ]["expected_synthetic_clocks"]
    assert payload["event_definition"]["snapshot_authority"] == (
        "atomic_event_reducer"
    )


def test_manifest_payload_tampering_fails_closed() -> None:
    payload = comparison.build_manifest("W1")
    tampered = deepcopy(payload)
    tampered["input_census"]["expected_diagnostic_completed_bars"] = 6899

    with pytest.raises(ResearchContractError, match="exact contract"):
        comparison._validate_exact_payload(tampered, "W1")


def test_manifest_hash_tampering_fails_closed() -> None:
    payload = comparison.build_manifest("W2")
    tampered = deepcopy(payload)
    tampered["identity_bindings"]["runtime_semantic_foundation"]["sha256"] = (
        "0" * 64
    )

    with pytest.raises(ResearchContractError, match="exact contract"):
        comparison._validate_exact_payload(tampered, "W2")


def test_new_manifest_labels_do_not_claim_a_stronger_evidence_class() -> None:
    fragments = tuple(
        token.replace("_", "") for token in comparison._DISALLOWED_LABEL_FRAGMENTS
    )
    for window in comparison.WINDOWS.values():
        path = comparison.ROOT / window["manifest"]
        normalized = path.read_text(encoding="utf-8").lower().replace("_", "")
        assert all(fragment not in normalized for fragment in fragments)


def test_wrapper_delegates_to_existing_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(semantic_runner, "run", fake_run)
    output = comparison.ROOT / "experiments/results/not-created.json"

    result = comparison.execute(window_key="W1", output=output, max_bars=7)

    assert result == {"status": "ok"}
    assert captured["manifest_path"] == (
        comparison.ROOT / comparison.WINDOWS["W1"]["manifest"]
    ).resolve()
    assert captured["contract_loader"] is comparison.load_comparison_contract
    assert captured["split_validator"] is comparison.validate_comparison_split
    assert captured["force"] is False
    assert captured["max_bars"] == 7


def test_current_eye_builder_enables_bound_foundation_projection() -> None:
    _, observer = semantic_runner._build_eye(comparison.ROOT / comparison.MODEL_PATH)

    assert observer.config.canonical_foundation_enabled is True


def test_comparison_report_uses_only_comparison_scope_language() -> None:
    summary = {
        "signals": 0,
        "resolved_n": 0,
        "laplace_success_rate": 0.5,
        "delta_vs_prior": None,
        "minimum_sample_met": False,
    }
    result = {
        "semantic_version": "smc_semantics_v1.2",
        "status": comparison.STATUS,
        "validation_state": "fixed_historical_comparison_unvalidated",
        "artifact_classification": {"complete_registered_window": True},
        "nested_chain": {
            name: dict(summary) for name in semantic_runner.V3_NESTED_STAGE_ORDER
        },
        "non_nested_comparisons": {},
        "control_comparisons": {
            name: {"requested": 0, "matched": 0}
            for name in semantic_runner.V3_HOLM_FAMILY
        },
        "inference": {
            "exact_mcnemar": {
                name: {"p_value": 1.0} for name in semantic_runner.V3_HOLM_FAMILY
            },
            "holm_fixed_family": {
                "adjusted_p_values": {
                    name: 1.0 for name in semantic_runner.V3_HOLM_FAMILY
                }
            },
        },
    }

    report = semantic_runner._report_v3(result)

    assert "Fixed Phase-4/5 comparison" in report
    normalized = report.lower().replace("_", "")
    assert all(
        token.replace("_", "") not in normalized
        for token in comparison._DISALLOWED_LABEL_FRAGMENTS
    )


def test_full_census_rejects_contract_identity_drift() -> None:
    expected = {
        "expected_emitted_bars_including_warmup": 1,
        "expected_diagnostic_completed_bars": 1,
        "expected_diagnostic_real_rows": 1,
        "expected_diagnostic_ready_real_rows": 1,
        "expected_diagnostic_synthetic_bars": 0,
        "expected_warmup_data_gap_resets": 0,
        "expected_diagnostic_data_gap_resets": 0,
        "expected_contract_changes": 0,
        "expected_first_diagnostic_asof": "2024-06-02T22:01:00Z",
        "expected_last_processed_asof": "2024-06-02T22:01:00Z",
        "expected_last_diagnostic_asof": "2024-06-02T22:01:00Z",
        "expected_synthetic_clocks": [],
        "expected_contracts": [{"symbol": "NQM4", "instrument_id": 13743}],
    }
    actual = {
        "emitted_bars_including_warmup": 1,
        "diagnostic_completed_bars": 1,
        "diagnostic_real_rows": 1,
        "diagnostic_ready_real_rows": 1,
        "diagnostic_synthetic_bars": 0,
        "warmup_data_gap_resets": 0,
        "diagnostic_data_gap_resets": 0,
        "contract_changes": 0,
        "first_diagnostic_asof": "2024-06-02T22:01:00Z",
        "last_processed_asof": "2024-06-02T22:01:00Z",
        "last_diagnostic_asof": "2024-06-02T22:01:00Z",
        "synthetic_clocks": [],
        "contracts": [{"symbol": "NQU4", "instrument_id": 999}],
    }

    with pytest.raises(ResearchContractError, match="contracts"):
        semantic_runner._assert_full_input_census(expected, actual)


def test_committed_manifests_are_json_subset_yaml() -> None:
    for window in comparison.WINDOWS.values():
        path = comparison.ROOT / window["manifest"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 2
        assert Path(payload["research_design_binding"]["path"]) == Path(
            comparison.DESIGN_PATH
        )
