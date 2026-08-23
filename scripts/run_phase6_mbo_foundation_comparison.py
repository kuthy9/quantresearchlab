#!/usr/bin/env python3
"""Validate or execute frozen Foundation-v2 Phase-6 MBO comparisons.

This wrapper owns only the W1/W2 comparison contract.  Detection, episode
construction, matching, inference, and ledger materialization remain in
``run_mbo_mechanism_research``.  Contract validation never opens a sealed
source and execution is no-clobber at one registered output stem per window.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_mbo_mechanism_research as phase6_runner  # noqa: E402
from smc_trader.foundation_registry import (  # noqa: E402
    FOUNDATION_VERSION,
    load_foundation_registry,
)
from smc_trader.mbo_mechanism_research import (  # noqa: E402
    PHASE6_EXECUTABLE_STATUS,
    REQUIRED_PHASE6_IDENTITY_BINDINGS,
    FrozenPhase6Contract,
    Phase6ResearchError,
    load_frozen_phase6_contract,
    sha256_file,
)


SCHEMA_VERSION = 1
FROZEN_AT = "2026-08-23T00:00:00-04:00"
FOUNDATION_REGISTRY_PATH = "semantics/foundation_v2_0.yaml"
MODEL_PATH = "configs/model.json"
COMPARISON_STATUS = "frozen_foundation_v2_phase6_mbo_comparison"

WINDOWS: Mapping[str, Mapping[str, Any]] = {
    "W1": {
        "source_manifest": (
            "experiments/manifests/"
            "smc_semantics_v1_2_2024_06_phase6_mbo_week1_v5.yaml"
        ),
        "manifest": (
            "experiments/manifests/"
            "foundation_v2_2024_06_phase6_mbo_w1_"
            "development_comparison_v1.yaml"
        ),
        "output": (
            "experiments/results/"
            "foundation_v2_2024_06_phase6_mbo_w1_"
            "development_comparison_v1.json"
        ),
        "experiment_id": (
            "foundation_v2_2024_06_phase6_mbo_w1_development_comparison_v1"
        ),
        "window_id": "2024-06-week-1",
        "comparison_role": "development_comparison",
        "start": "2024-06-02T22:00:00Z",
        "end_exclusive": "2024-06-07T21:01:00Z",
        "synthetic_decision_clock": "2024-06-07T03:10:00Z",
        "prior_week1_policy": "not_applicable",
    },
    "W2": {
        "source_manifest": (
            "experiments/manifests/"
            "smc_semantics_v1_2_2024_06_phase6_mbo_week2_extension_v3.yaml"
        ),
        "manifest": (
            "experiments/manifests/"
            "foundation_v2_2024_06_phase6_mbo_w2_"
            "historical_validation_comparison_v1.yaml"
        ),
        "output": (
            "experiments/results/"
            "foundation_v2_2024_06_phase6_mbo_w2_"
            "historical_validation_comparison_v1.json"
        ),
        "experiment_id": (
            "foundation_v2_2024_06_phase6_mbo_w2_"
            "historical_validation_comparison_v1"
        ),
        "window_id": "2024-06-week-2",
        "comparison_role": "historical_validation_comparison",
        "start": "2024-06-09T22:00:00Z",
        "end_exclusive": "2024-06-14T21:01:00Z",
        "synthetic_decision_clock": "2024-06-10T04:14:00Z",
        "prior_week1_policy": (
            "gate_and_warmup_audit_only_prior_pairs_excluded_from_inference"
        ),
    },
}

_COMMON_COMPARISON_IDENTITIES: Mapping[str, str] = {
    "comparison_wrapper": "scripts/run_phase6_mbo_foundation_comparison.py",
    "foundation_registry": FOUNDATION_REGISTRY_PATH,
    "runtime_foundation_registry": "smc_trader/foundation_registry.py",
    "runtime_semantic_foundation": "smc_trader/semantic_foundation.py",
    "runtime_semantic_lifecycle": "smc_trader/semantic_lifecycle.py",
    "runtime_foundation_adapter": "smc_trader/foundation_adapter.py",
    "runtime_structural_outcome": "smc_trader/structural_outcome.py",
}


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Phase6ResearchError(f"comparison manifest repeats key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase6ResearchError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise Phase6ResearchError(f"{label} must be an object")
    return payload


def _bound_file(relative_path: str, *, label: str) -> Path:
    path = (ROOT / relative_path).resolve(strict=False)
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise Phase6ResearchError(f"{label} escapes the repository") from error
    if not path.is_file() or path.is_symlink():
        raise Phase6ResearchError(f"{label} is not a regular file")
    return path


def _binding(relative_path: str, *, label: str) -> dict[str, str]:
    return {
        "path": relative_path,
        "sha256": sha256_file(_bound_file(relative_path, label=label)),
    }


def _window(window_key: str) -> Mapping[str, Any]:
    try:
        return WINDOWS[window_key]
    except KeyError as error:
        raise Phase6ResearchError(
            f"unknown Phase-6 comparison window: {window_key}"
        ) from error


def build_manifest(window_key: str) -> dict[str, Any]:
    """Build the sole accepted manifest for one registered comparison."""

    window = _window(window_key)
    source_path = _bound_file(
        str(window["source_manifest"]),
        label="source Phase-6 manifest",
    )
    payload = deepcopy(_read_json(source_path, label="source Phase-6 manifest"))
    bindings = payload.get("identity_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(
        REQUIRED_PHASE6_IDENTITY_BINDINGS
    ):
        raise Phase6ResearchError("source Phase-6 identity surface changed")
    payload["identity_bindings"] = {
        name: _binding(str(binding["path"]), label=name)
        for name, binding in sorted(bindings.items())
    }
    foundation = load_foundation_registry(
        _bound_file(FOUNDATION_REGISTRY_PATH, label="foundation_registry")
    )
    comparison_bindings = {
        name: _binding(relative_path, label=name)
        for name, relative_path in sorted(
            {
                **_COMMON_COMPARISON_IDENTITIES,
                "source_phase6_manifest": str(window["source_manifest"]),
            }.items()
        )
    }
    payload.update(
        {
            "status": PHASE6_EXECUTABLE_STATUS,
            "experiment_id": window["experiment_id"],
            "frozen_at": FROZEN_AT,
            "frozen_before_run": True,
            "authority": {
                "mbo_development_only": True,
                "mechanism_association_only": True,
                "artifact_fit_allowed": False,
                "phase7_evidence_allowed_only_if_supported": False,
                "trading_authority": False,
                "sealed_holdout_opened": False,
            },
            "foundation_version": foundation.foundation_version,
            "foundation_registry": FOUNDATION_REGISTRY_PATH,
            "foundation_registry_identity": foundation.identity,
            "comparison_contract_version": SCHEMA_VERSION,
            "comparison_status": COMPARISON_STATUS,
            "comparison_identity_bindings": comparison_bindings,
            "comparison_contract": {
                "window_key": window_key,
                "window_id": window["window_id"],
                "comparison_role": window["comparison_role"],
                "phase_scope": "phase_6_mbo_mechanism",
                "foundation_mode": "canonical_foundation_v2_runtime",
                "detector_and_model_source": (
                    "scripts/run_mbo_mechanism_research.py"
                ),
                "start": window["start"],
                "end_exclusive": window["end_exclusive"],
                "expected_completed_clocks": 6900,
                "expected_real_completed_clocks": 6899,
                "expected_synthetic_no_trade_clocks": 1,
                "synthetic_decision_clocks": [
                    window["synthetic_decision_clock"]
                ],
                "contract": {"symbol": "NQM4", "instrument_id": 13743},
                "source_phase6_manifest": str(window["source_manifest"]),
                "processed_phase6_artifact": payload["identity_bindings"][
                    "mbo_feature_artifact"
                ],
                "processed_phase6_manifest": payload["identity_bindings"][
                    "mbo_feature_manifest"
                ],
                "raw_partition_manifest": payload["identity_bindings"][
                    "raw_mbo_partition_manifest"
                ],
                "prior_week1_policy": window["prior_week1_policy"],
                "source_artifacts_reused_without_mutation": True,
                "historical_manifests_and_results_mutable": False,
                "execution_authority": {
                    "comparison_only": True,
                    "rolling_oof": False,
                    "sealed_oos": False,
                    "model_admission": False,
                    "trading": False,
                },
            },
            "claim_scope": (
                "fixed open-development Foundation v2 mechanism comparison "
                "only; no causal, OOF, sealed OOS, model-fit, model-admission, "
                "or trading claim"
            ),
            "notes": (
                "FROZEN BEFORE COMPARISON. Reuses the exact historical Phase-6 "
                "window, processed minute artifact, raw partition manifest, "
                "detector, matching and inference code under the current "
                "hash-bound Foundation v2 runtime. W1 is development comparison; "
                "W2 is historical validation comparison. Results cannot enter "
                "rolling OOF, sealed OOS, model admission, or trading."
            ),
        }
    )
    return payload


def canonical_manifest_text(window_key: str) -> str:
    return json.dumps(
        build_manifest(window_key),
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def _manifest_window_key(payload: Mapping[str, Any]) -> str:
    comparison = payload.get("comparison_contract")
    if not isinstance(comparison, Mapping):
        raise Phase6ResearchError("comparison_contract is required")
    key = comparison.get("window_key")
    if key not in WINDOWS:
        raise Phase6ResearchError("comparison window is not registered")
    return str(key)


def _validate_exact_payload(payload: Mapping[str, Any], window_key: str) -> None:
    if dict(payload) != build_manifest(window_key):
        raise Phase6ResearchError("Phase-6 comparison manifest drifted")


def validate_window(
    window_key: str,
    *,
    verify_raw_partition_hashes: bool = False,
) -> FrozenPhase6Contract:
    """Validate identities and frozen design without opening market frames."""

    window = _window(window_key)
    source = (ROOT / str(window["manifest"])).resolve()
    if not source.is_file() or source.is_symlink():
        raise Phase6ResearchError("comparison manifest is not a regular file")
    payload = _read_json(source, label="comparison manifest")
    actual_key = _manifest_window_key(payload)
    if actual_key != window_key:
        raise Phase6ResearchError("comparison manifest window changed")
    _validate_exact_payload(payload, window_key)
    for name, binding in payload["comparison_identity_bindings"].items():
        if _binding(str(binding["path"]), label=name) != dict(binding):
            raise Phase6ResearchError(
                f"comparison identity binding changed: {name}"
            )
    foundation = load_foundation_registry(
        _bound_file(FOUNDATION_REGISTRY_PATH, label="foundation_registry"),
        expected_identity=str(payload["foundation_registry_identity"]),
    )
    if foundation.foundation_version != FOUNDATION_VERSION:
        raise Phase6ResearchError("comparison Foundation version changed")
    model = _read_json(_bound_file(MODEL_PATH, label="model_config"), label="model")
    observer = model.get("observer")
    if (
        not isinstance(observer, Mapping)
        or observer.get("canonical_foundation_enabled") is not True
        or observer.get("canonical_foundation_registry")
        != FOUNDATION_REGISTRY_PATH
        or observer.get("canonical_foundation_identity") != foundation.identity
    ):
        raise Phase6ResearchError("model does not bind canonical Foundation v2")
    contract = load_frozen_phase6_contract(
        source,
        root=ROOT,
        verify_raw_partition_hashes=verify_raw_partition_hashes,
        comparison_validation_only=True,
    )
    if (
        contract.active_window.window_id != window["window_id"]
        or contract.active_window.start.isoformat().replace("+00:00", "Z")
        != window["start"]
        or contract.active_window.end_exclusive.isoformat().replace(
            "+00:00", "Z"
        )
        != window["end_exclusive"]
        or contract.active_window.expected_rows != 6900
    ):
        raise Phase6ResearchError("comparison active window changed")
    return contract


def execute(window_key: str) -> dict[str, Any]:
    """Execute the fixed no-clobber output after contract-only validation."""

    window = _window(window_key)
    manifest = (ROOT / str(window["manifest"])).resolve()
    output = (ROOT / str(window["output"])).resolve()
    validate_window(window_key, verify_raw_partition_hashes=False)
    return phase6_runner.run(
        manifest_path=manifest,
        output=output,
        verify_raw_partition_hashes=True,
        comparison_validation_only=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-id", choices=tuple(WINDOWS), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render-manifest", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--verify-raw-partitions", action="store_true")
    args = parser.parse_args()
    if args.verify_raw_partitions and not args.validate_only:
        raise ValueError(
            "--verify-raw-partitions is valid only with --validate-only"
        )
    if args.render_manifest:
        print(canonical_manifest_text(args.window_id), end="")
        return
    if args.validate_only:
        contract = validate_window(
            args.window_id,
            verify_raw_partition_hashes=args.verify_raw_partitions,
        )
        print(
            json.dumps(
                {
                    "comparison_role": contract.payload[
                        "comparison_contract"
                    ]["comparison_role"],
                    "manifest": str(contract.manifest_path.relative_to(ROOT)),
                    "manifest_sha256": contract.manifest_sha256,
                    "raw_partition_hashes_reverified": (
                        args.verify_raw_partitions
                    ),
                    "status": "valid",
                    "window_id": contract.active_window.window_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    result = execute(args.window_id)
    print(
        json.dumps(
            {
                "output": WINDOWS[args.window_id]["output"],
                "status": result["status"],
                "window_id": result["active_window"]["id"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
