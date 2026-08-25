#!/usr/bin/env python3
"""Validate or execute the frozen June-2024 Phase-4/5 comparisons.

The execution path delegates to ``run_semantic_signal_research.run``.  This
file owns only the two-window contract boundary and never implements a market
detector, event selector, outcome engine, or matcher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_semantic_signal_research as semantic_runner  # noqa: E402
from smc_trader.foundation_registry import (  # noqa: E402
    FOUNDATION_VERSION,
    load_foundation_registry,
)
from smc_trader.semantics import (  # noqa: E402
    SemanticRegistry,
    load_semantic_selection,
)
from smc_trader.signal_research import (  # noqa: E402
    FrozenResearchContract,
    REQUIRED_IDENTITY_BINDINGS,
    REQUIRED_RUNTIME_CODE_BINDINGS,
    ResearchContractError,
    sha256_file,
)


SCHEMA_VERSION = 1
FROZEN_AT = "2026-08-23T00:00:00-04:00"
STATUS = "frozen_foundation_v2_phase45_semantic_comparison"
DESIGN_PATH = "experiments/manifests/semantic_event_study_v3_template.yaml"
DATASET_PATH = (
    "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"
)
DATASET_MANIFEST_PATH = (
    "data/processed/"
    "nq_1m_previous_session_front_v2_3_2017_2026.manifest.json"
)
SPLIT_REGISTRY_PATH = "configs/data_splits.json"
SEMANTIC_REGISTRY_PATH = "semantics/registry_v1_2.yaml"
FOUNDATION_REGISTRY_PATH = "semantics/foundation_v2_0.yaml"
MODEL_PATH = "configs/model.json"

_DESIGN_FIELDS = (
    "event_definition",
    "control_definition",
    "inference_definition",
    "ledger_definition",
    "primary_outcome",
    "secondary_outcomes",
    "nested_chain_metric",
    "minimum_sample_requirement",
)
_DISALLOWED_LABEL_FRAGMENTS = (
    "rolling_" + "oof",
    "seal" + "ed",
    "trad" + "ing",
    "admis" + "sion",
)

_BASE_IDENTITY_PATHS: Mapping[str, str] = {
    "semantic_registry": SEMANTIC_REGISTRY_PATH,
    "semantic_parameters": "semantics/parameters_v1_2.yaml",
    "dataset_manifest": DATASET_MANIFEST_PATH,
    "split_registry": SPLIT_REGISTRY_PATH,
    "model_config": MODEL_PATH,
    "structure_protocol": "configs/primitives_structure_liquidity.json",
    "liquidity_protocol": "configs/primitives_structure_liquidity.json",
    "displacement_protocol": "configs/primitives_displacement.json",
    "group3_protocol": "configs/primitives_zones.json",
    "group4_protocol": "configs/primitives_range.json",
    "group5_protocol": "configs/primitives_entry.json",
    "runner": "scripts/run_semantic_signal_research.py",
    "pyproject": "pyproject.toml",
    "lockfile": "uv.lock",
    **dict(REQUIRED_RUNTIME_CODE_BINDINGS),
}
if set(_BASE_IDENTITY_PATHS) != set(REQUIRED_IDENTITY_BINDINGS):
    raise RuntimeError("comparison wrapper and base runner identity surfaces differ")

_ADDITIVE_IDENTITY_PATHS: Mapping[str, str] = {
    "comparison_wrapper": "scripts/run_phase45_foundation_comparison.py",
    "research_design": DESIGN_PATH,
    "foundation_registry": FOUNDATION_REGISTRY_PATH,
    "runtime_foundation_registry": "smc_trader/foundation_registry.py",
    "runtime_semantic_foundation": "smc_trader/semantic_foundation.py",
    "runtime_semantic_lifecycle": "smc_trader/semantic_lifecycle.py",
    "runtime_foundation_adapter": "smc_trader/foundation_adapter.py",
    "runtime_structural_outcome": "smc_trader/structural_outcome.py",
}
IDENTITY_PATHS: Mapping[str, str] = {
    **_BASE_IDENTITY_PATHS,
    **_ADDITIVE_IDENTITY_PATHS,
}

WINDOWS: Mapping[str, Mapping[str, Any]] = {
    "W1": {
        "manifest": (
            "experiments/manifests/"
            "foundation_v2_2024_06_phase45_w1_development_comparison_v1.yaml"
        ),
        "experiment_id": "foundation_v2_2024_06_phase45_w1_comparison_v1",
        "window_id": "2024-06-week-1",
        "comparison_role": "development_comparison",
        "warmup_start": "2024-05-26T22:00:00Z",
        "start": "2024-06-02T22:00:00Z",
        "end_exclusive": "2024-06-07T21:01:00Z",
        "source_first_completed_clock": "2024-06-02T22:01:00Z",
        "source_last_completed_clock": "2024-06-07T21:00:00Z",
        "synthetic_clocks": ["2024-06-07T03:10:00Z"],
        "emitted_including_warmup": 13560,
    },
    "W2": {
        "manifest": (
            "experiments/manifests/"
            "foundation_v2_2024_06_phase45_w2_historical_validation_comparison_v1.yaml"
        ),
        "experiment_id": "foundation_v2_2024_06_phase45_w2_comparison_v1",
        "window_id": "2024-06-week-2",
        "comparison_role": "historical_validation_comparison",
        "warmup_start": "2024-05-26T22:00:00Z",
        "start": "2024-06-09T22:00:00Z",
        "end_exclusive": "2024-06-14T21:01:00Z",
        "source_first_completed_clock": "2024-06-09T22:01:00Z",
        "source_last_completed_clock": "2024-06-14T21:00:00Z",
        "synthetic_clocks": ["2024-06-10T04:14:00Z"],
        "emitted_including_warmup": 20460,
    },
}


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ResearchContractError(f"comparison manifest repeats key: {key}")
        output[key] = value
    return output


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ResearchContractError(f"cannot load {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ResearchContractError(f"{label} must be an object")
    return payload


def _bound_path(relative_path: str, *, label: str) -> Path:
    path = (ROOT / relative_path).resolve(strict=False)
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ResearchContractError(f"{label} escapes the repository") from error
    if not path.is_file() or path.is_symlink():
        raise ResearchContractError(f"{label} is not a regular file")
    return path


def _identity_bindings() -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": relative_path,
            "sha256": sha256_file(_bound_path(relative_path, label=name)),
        }
        for name, relative_path in sorted(IDENTITY_PATHS.items())
    }


def build_manifest(window_key: str) -> dict[str, Any]:
    """Render the only accepted manifest bytes for one registered window."""

    try:
        window = WINDOWS[window_key]
    except KeyError as error:
        raise ResearchContractError(f"unknown comparison window: {window_key}") from error
    semantic_registry = SemanticRegistry.from_file(
        _bound_path(SEMANTIC_REGISTRY_PATH, label="semantic_registry"),
        required_version="smc_semantics_v1.2",
    )
    foundation_registry = load_foundation_registry(
        _bound_path(FOUNDATION_REGISTRY_PATH, label="foundation_registry")
    )
    bindings = _identity_bindings()
    dataset_sha = sha256_file(_bound_path(DATASET_PATH, label="dataset"))
    dataset_manifest_sha = bindings["dataset_manifest"]["sha256"]
    split_registry_sha = bindings["split_registry"]["sha256"]
    return {
        "schema_version": 2,
        "research_protocol_version": 3,
        "comparison_contract_version": SCHEMA_VERSION,
        "status": STATUS,
        "experiment_id": window["experiment_id"],
        "frozen_at": FROZEN_AT,
        "frozen_before_run": True,
        "authority": {
            "comparison_only": True,
            "artifact_fit_allowed": False,
            "parameter_change_allowed": False,
            "model_action_allowed": False,
        },
        "comparison_contract": {
            "window_key": window_key,
            "window_id": window["window_id"],
            "comparison_role": window["comparison_role"],
            "phase_scope": ["phase_4", "phase_5"],
            "input_mode": "causal_completed_ohlcv",
            "detector_source": "scripts/run_semantic_signal_research.py",
            "foundation_mode": "additive_canonical_projection",
            "source_first_completed_clock": window[
                "source_first_completed_clock"
            ],
            "source_last_completed_clock": window["source_last_completed_clock"],
            "expected_contracts": [{"symbol": "NQM4", "instrument_id": 13743}],
            "expected_synthetic_clocks": window["synthetic_clocks"],
            "prior_results_mutable": False,
        },
        "semantic_version": semantic_registry.semantic_version,
        "semantic_registry": SEMANTIC_REGISTRY_PATH,
        "semantic_registry_identity": semantic_registry.identity,
        "foundation_version": foundation_registry.foundation_version,
        "foundation_registry": FOUNDATION_REGISTRY_PATH,
        "foundation_registry_identity": foundation_registry.identity,
        "research_design_binding": bindings["research_design"],
        "identity_bindings": bindings,
        "dataset_version": {
            "path": DATASET_PATH,
            "sha256": dataset_sha,
            "manifest_path": DATASET_MANIFEST_PATH,
            "manifest_sha256": dataset_manifest_sha,
            "split_registry": SPLIT_REGISTRY_PATH,
            "split_registry_sha256": split_registry_sha,
            "split_role": window["comparison_role"],
            "use_restriction": "fixed_semantic_comparison_only",
        },
        "instrument_universe": ["NQM4:13743"],
        "contract_handling": (
            "strict_previous_completed_session_front; "
            "reset causal state on instrument_id change"
        ),
        "session_definition": (
            "independent cross-timeframe SessionState on "
            "America/New_York completed 1m clock"
        ),
        "warmup_period": {
            "start": window["warmup_start"],
            "end_exclusive": window["start"],
            "outcomes_opened": False,
        },
        "diagnostic_period": {
            "start": window["start"],
            "end_exclusive": window["end_exclusive"],
        },
        "input_census": {
            "diagnostic_data_gap_policy": (
                "fail_closed_on_diagnostic_data_gap_history_reset"
            ),
            "expected_emitted_bars_including_warmup": window[
                "emitted_including_warmup"
            ],
            "expected_diagnostic_completed_bars": 6900,
            "expected_diagnostic_real_rows": 6899,
            "expected_diagnostic_ready_real_rows": 6899,
            "expected_diagnostic_synthetic_bars": 1,
            "expected_warmup_data_gap_resets": 0,
            "expected_diagnostic_data_gap_resets": 0,
            "expected_contract_changes": 0,
            "expected_first_diagnostic_asof": window["start"],
            "expected_last_processed_asof": window["end_exclusive"],
            "expected_last_diagnostic_asof": window[
                "source_last_completed_clock"
            ],
            "expected_synthetic_clocks": window["synthetic_clocks"],
            "expected_contracts": [{"symbol": "NQM4", "instrument_id": 13743}],
        },
        "parameter_search_space": {},
        "comparison_constraints": [
            "fixed historical comparison; no parameter changes are permitted",
            "the price-bar event study does not evaluate order-book mechanisms",
            "results cannot change the current model or authorize model actions",
        ],
        "limitations": list(semantic_runner.REQUIRED_RESEARCH_LIMITATIONS),
        "notes": (
            "Strict W1/W2 Phase-4/5 comparison over the current hash-bound "
            "Foundation v2 runtime. The detector, event study, controls and "
            "outcome code are reused from the bound semantic research runner."
        ),
    }


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
        raise ResearchContractError("comparison_contract is required")
    window_key = comparison.get("window_key")
    if window_key not in WINDOWS:
        raise ResearchContractError("comparison window is not registered")
    return str(window_key)


def _validate_exact_payload(payload: Mapping[str, Any], window_key: str) -> None:
    expected = build_manifest(window_key)
    if payload != expected:
        raise ResearchContractError("comparison manifest drifted from its exact contract")
    serialized = json.dumps(payload, sort_keys=True).lower()
    if any(token in serialized for token in _DISALLOWED_LABEL_FRAGMENTS):
        raise ResearchContractError("comparison manifest contains a forbidden label")


def load_comparison_contract(
    manifest_path: str | Path,
    *,
    root: str | Path,
    actual_semantic_registry_identity: str,
) -> FrozenResearchContract:
    """Load the exact committed comparison contract without opening data."""

    if Path(root).resolve() != ROOT:
        raise ResearchContractError("comparison contract root changed")
    source = Path(manifest_path).resolve()
    if not source.is_file() or source.is_symlink():
        raise ResearchContractError("comparison manifest is not a regular file")
    payload = _read_json_object(source, label="comparison manifest")
    window_key = _manifest_window_key(payload)
    expected_path = (ROOT / str(WINDOWS[window_key]["manifest"])).resolve()
    if source != expected_path:
        raise ResearchContractError("comparison manifest path is not registered")
    _validate_exact_payload(payload, window_key)

    identity_paths = {
        name: _bound_path(binding["path"], label=name)
        for name, binding in payload["identity_bindings"].items()
    }
    semantic_registry = SemanticRegistry.from_file(
        identity_paths["semantic_registry"],
        required_version=str(payload["semantic_version"]),
    )
    if (
        semantic_registry.identity != actual_semantic_registry_identity
        or semantic_registry.identity != payload["semantic_registry_identity"]
    ):
        raise ResearchContractError("comparison semantic registry identity changed")
    foundation = load_foundation_registry(
        identity_paths["foundation_registry"],
        expected_identity=str(payload["foundation_registry_identity"]),
    )
    if foundation.foundation_version != FOUNDATION_VERSION:
        raise ResearchContractError("comparison Foundation version changed")
    model = _read_json_object(identity_paths["model_config"], label="model config")
    try:
        selection = load_semantic_selection(
            model.get("semantic_selection"),
            root=ROOT,
        )
    except ValueError as error:
        raise ResearchContractError(
            "comparison model semantic_selection is invalid"
        ) from error
    if (
        selection.atomic_definition_identity != semantic_registry.identity
        or selection.foundation_registry_path != FOUNDATION_REGISTRY_PATH
        or selection.foundation_registry_identity != foundation.identity
        or selection.parent_atomic_semantics_version
        != selection.atomic_semantics_version
    ):
        raise ResearchContractError("comparison model does not bind Foundation v2")

    design = _read_json_object(identity_paths["research_design"], label="research design")
    effective_payload = dict(payload)
    for field in _DESIGN_FIELDS:
        if field not in design:
            raise ResearchContractError(f"research design lacks {field}")
        effective_payload[field] = design[field]

    comparison = payload["comparison_contract"]
    warmup = pd.Timestamp(payload["warmup_period"]["start"])
    start = pd.Timestamp(payload["diagnostic_period"]["start"])
    end = pd.Timestamp(payload["diagnostic_period"]["end_exclusive"])
    if any(clock.tzinfo is None for clock in (warmup, start, end)):
        raise ResearchContractError("comparison clocks must be timezone aware")
    dataset = payload["dataset_version"]
    raw = source.read_bytes()
    return FrozenResearchContract(
        manifest_path=source,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        payload=effective_payload,
        dataset_path=_bound_path(str(dataset["path"]), label="dataset"),
        dataset_relative_path=str(dataset["path"]),
        dataset_sha256=str(dataset["sha256"]),
        split_registry_path=identity_paths["split_registry"],
        split_registry_sha256=str(dataset["split_registry_sha256"]),
        split_role=str(comparison["comparison_role"]),
        semantic_registry_path=identity_paths["semantic_registry"],
        semantic_registry_identity=semantic_registry.identity,
        model_path=identity_paths["model_config"],
        warmup_start=warmup.tz_convert("America/New_York"),
        diagnostic_start=start.tz_convert("America/New_York"),
        diagnostic_end=end.tz_convert("America/New_York"),
        allowed_diagnostic_roles=(str(comparison["comparison_role"]),),
        allowed_warmup_roles=("registered_context",),
        identity_paths=identity_paths,
    )


def validate_comparison_split(contract: FrozenResearchContract, validation: Any) -> None:
    """Verify source identity and that the exact window remains registered."""

    if validation.fingerprint != contract.split_registry_sha256:
        raise ResearchContractError("comparison split registry identity changed")
    source = validation.causal_source
    if (
        source.path != contract.dataset_relative_path
        or source.sha256 != contract.dataset_sha256
        or source.manifest_path != DATASET_MANIFEST_PATH
        or source.manifest_sha256
        != contract.payload["dataset_version"]["manifest_sha256"]
    ):
        raise ResearchContractError("comparison causal source identity changed")
    # Classification is used only as a containment check.  The public
    # comparison role remains the exact role frozen in this wrapper contract.
    validation.classify_ohlcv(contract.warmup_start, contract.diagnostic_end)


def validate_window(window_key: str) -> FrozenResearchContract:
    path = (ROOT / str(WINDOWS[window_key]["manifest"])).resolve()
    contract, _, _, _ = semantic_runner._load_contract_and_registry(
        path,
        contract_loader=load_comparison_contract,
        split_validator=validate_comparison_split,
    )
    semantic_runner._validated_research_design(contract.payload)
    return contract


def execute(
    *,
    window_key: str,
    output: Path,
    max_bars: int | None = None,
) -> dict[str, Any]:
    manifest = (ROOT / str(WINDOWS[window_key]["manifest"])).resolve()
    return semantic_runner.run(
        output=output.resolve(),
        max_bars=max_bars,
        manifest_path=manifest,
        force=False,
        contract_loader=load_comparison_contract,
        split_validator=validate_comparison_split,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-id", choices=tuple(WINDOWS), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--render-manifest", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--output", type=Path)
    parser.add_argument("--max-bars", type=int)
    args = parser.parse_args()
    if args.max_bars is not None and args.max_bars < 1:
        raise ValueError("--max-bars must be positive")
    if args.max_bars is not None and args.output is None:
        raise ValueError("--max-bars is valid only with --output")
    if args.render_manifest:
        print(canonical_manifest_text(args.window_id), end="")
        return
    if args.validate_only:
        contract = validate_window(args.window_id)
        print(
            json.dumps(
                {
                    "manifest": str(contract.manifest_path.relative_to(ROOT)),
                    "manifest_sha256": contract.manifest_sha256,
                    "status": "valid",
                    "window_id": contract.payload["comparison_contract"][
                        "window_id"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    result = execute(
        window_key=args.window_id,
        output=args.output,
        max_bars=args.max_bars,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "rows": result["coverage"]["diagnostic_real_rows"],
                "status": result["status"],
                "window_id": result["split_authority"]["window_id"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
