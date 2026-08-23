#!/usr/bin/env python3
"""Audit Phase 7 probability-fit readiness without fitting any artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.dol_probability import load_dol_probability_protocol  # noqa: E402
from smc_trader.mbo_mechanism_research import validate_phase6_design  # noqa: E402
from smc_trader.path_belief import load_path_belief_protocol  # noqa: E402
from smc_trader.validation import load_validation_protocol  # noqa: E402


DEFAULT_CONTRACT = ROOT / "configs/phase7_probability_fit_admission.json"
ESTIMANDS = (
    "conditional_likelihood",
    "hypothesis_specific_temporal",
    "path_calibration",
    "dol_calibration",
)
EXPECTED_PROTOCOL_VERSION = "phase7_probability_fit_readiness_v1.0"
EXPECTED_STATUS = "blocked_inputs_incomplete"
EXPECTED_WINDOW = {
    "id": "2024-06-mbo-development",
    "start": "2024-06-02T00:00:00Z",
    "end_exclusive": "2024-07-01T00:00:00Z",
    "validation_or_oos_claim_allowed": False,
}
EXPECTED_RUNTIME_BINDINGS = frozenset(
    {
        "dol_protocol",
        "model_config",
        "path_protocol",
        "required_temporal_model_slot",
        "split_registry",
    }
)
EXPECTED_CONTRACT_IDENTITY = (
    "21d8d587d2d3bec3ff35628a8851295183a8e7943852ca13b90ec768d1035378"
)


class ReadinessError(ValueError):
    pass


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json(path: Path, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReadinessError(f"cannot load {name}: {path}") from error
    if not isinstance(value, Mapping):
        raise ReadinessError(f"{name} must be an object")
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReadinessError(f"{name} must be an object")
    return value


def _bound_json(value: Any, *, name: str) -> tuple[Path, Mapping[str, Any]]:
    binding = _mapping(value, name=name)
    if set(binding) != {"path", "sha256"}:
        raise ReadinessError(f"{name} binding is incomplete")
    path = _resolve(str(binding["path"]))
    if not path.is_file() or _hash(path) != binding["sha256"]:
        raise ReadinessError(f"{name} hash mismatch")
    return path, _json(path, name=name)


def _phase6_ledger_columns(
    result: Mapping[str, Any],
) -> tuple[dict[str, set[str]], int]:
    ledgers = _mapping(result.get("ledgers"), name="Phase 6 ledgers")
    schemas: dict[str, set[str]] = {}
    total_rows = 0
    for name in ("episodes", "matched_pairs", "unmatched"):
        binding = _mapping(ledgers.get(name), name=f"Phase 6 {name} ledger")
        if set(binding) != {"path", "rows", "sha256"}:
            raise ReadinessError(f"Phase 6 {name} ledger binding is incomplete")
        path = _resolve(str(binding["path"]))
        if not path.is_file() or _hash(path) != binding["sha256"]:
            raise ReadinessError(f"Phase 6 {name} ledger hash mismatch")
        observed_rows = 0
        columns: set[str] = set()
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ReadinessError(f"invalid {name} row {line_number}") from error
                if not isinstance(row, Mapping):
                    raise ReadinessError(f"invalid {name} row {line_number}")
                columns.update(row)
                observed_rows += 1
        if observed_rows != binding["rows"]:
            raise ReadinessError(f"Phase 6 {name} row count mismatch")
        schemas[name] = columns
        total_rows += observed_rows
    return schemas, total_rows


def _timestamp(value: Any, *, name: str) -> pd.Timestamp:
    try:
        clock = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ReadinessError(f"{name} is invalid") from error
    if clock.tzinfo is None:
        raise ReadinessError(f"{name} must be timezone-aware")
    return clock


def audit_readiness(path: str | Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract_path = _resolve(path)
    contract = _json(contract_path, name="Phase 7 readiness contract")
    expected_root = {
        "schema_version", "protocol_version", "status", "authority",
        "requested_window", "phase6_source", "runtime_bindings",
        "required_labeled_cohort_schema", "invariants",
    }
    invariants = _mapping(contract.get("invariants"), name="invariants")
    if (
        set(contract) != expected_root
        or contract["schema_version"] != 1
        or contract.get("protocol_version") != EXPECTED_PROTOCOL_VERSION
        or contract.get("status") != EXPECTED_STATUS
        or contract["authority"] != "readiness_only_no_artifact_fit"
        or any(value is not True for value in invariants.values())
        or _canonical_identity(contract) != EXPECTED_CONTRACT_IDENTITY
    ):
        raise ReadinessError("Phase 7 readiness contract is not fail closed")

    source = _mapping(contract["phase6_source"], name="phase6_source")
    _, manifest = _bound_json(source.get("manifest"), name="Phase 6 manifest")
    _, result = _bound_json(source.get("result"), name="Phase 6 result")
    try:
        validate_phase6_design(manifest)
    except ValueError as error:
        raise ReadinessError(f"Phase 6 design is invalid: {error}") from error
    authority = _mapping(manifest.get("authority"), name="Phase 6 authority")

    runtime = _mapping(contract["runtime_bindings"], name="runtime_bindings")
    if (
        set(runtime) != EXPECTED_RUNTIME_BINDINGS
        or runtime.get("required_temporal_model_slot") != "path_temporal_artifact"
    ):
        raise ReadinessError("runtime_bindings changed from the readiness contract")
    try:
        path_protocol = load_path_belief_protocol(_resolve(runtime["path_protocol"]))
        dol_protocol = load_dol_probability_protocol(_resolve(runtime["dol_protocol"]))
        split_protocol = load_validation_protocol(_resolve(runtime["split_registry"]))
    except (KeyError, ValueError) as error:
        raise ReadinessError(f"current protocol binding is invalid: {error}") from error
    model = _json(_resolve(runtime["model_config"]), name="model config")

    manifest_sha = _mapping(source["manifest"], name="manifest binding")["sha256"]
    result_sha = _mapping(source["result"], name="result binding")["sha256"]
    if (
        result.get("manifest_sha256") != manifest_sha
        or path_protocol.phase6_manifest_sha256 != manifest_sha
        or path_protocol.phase6_result_sha256 != result_sha
        or path_protocol.phase6_result_identity != result.get("result_identity")
        or list(path_protocol.evidence_allowlist)
        != result.get("phase7_evidence_allowlist")
    ):
        raise ReadinessError("Phase 6 result and Path protocol bindings are stale")

    requested = _mapping(contract["requested_window"], name="requested_window")
    if dict(requested) != EXPECTED_WINDOW:
        raise ReadinessError("requested_window changed from the frozen 2024-06 scope")
    start = _timestamp(requested.get("start"), name="requested_window.start")
    end = _timestamp(requested.get("end_exclusive"), name="requested_window.end")
    try:
        registered_window = split_protocol.classify_mbo(start, end)
    except ValueError as error:
        raise ReadinessError(f"requested MBO window is unregistered: {error}") from error
    if (
        registered_window.role != "development"
        or requested.get("validation_or_oos_claim_allowed") is not False
    ):
        raise ReadinessError("2024-06 must remain development-only")

    ledger_schemas, ledger_rows = _phase6_ledger_columns(result)
    schemas = _mapping(
        contract["required_labeled_cohort_schema"],
        name="required_labeled_cohort_schema",
    )
    if set(schemas) != set(ESTIMANDS):
        raise ReadinessError("Phase 7 estimand schema set changed")
    schema_gaps: dict[str, list[str]] = {}
    for estimand in ESTIMANDS:
        required = schemas[estimand]
        if not isinstance(required, list) or not required:
            raise ReadinessError(f"{estimand} required schema is invalid")
        required_fields = set(required)
        schema_gaps[estimand] = min(
            (sorted(required_fields - columns) for columns in ledger_schemas.values()),
            key=lambda missing: (len(missing), missing),
        )

    signal = _mapping(model.get("signal_policy"), name="model.signal_policy")
    dol_binding = _mapping(model.get("dol_probability"), name="model.dol_probability")
    blockers: list[dict[str, str]] = []
    if authority.get("artifact_fit_allowed") is not True:
        blockers.append({
            "code": "PHASE6_SOURCE_FORBIDS_ARTIFACT_FIT",
            "detail": "frozen Phase 6 authority sets artifact_fit_allowed=false",
        })
    if authority.get("mechanism_association_only") is not False:
        blockers.append({
            "code": "PHASE6_SOURCE_IS_MECHANISM_ASSOCIATION_ONLY",
            "detail": "Phase 6 does not estimate path or DOL outcome likelihoods",
        })
    blockers.append({
        "code": "INDEPENDENT_FIT_VALIDATION_COHORTS_MISSING",
        "detail": "2024-06 development evidence has no separate fit and validation cohorts",
    })
    registered_phase6_end = max(
        _timestamp(window["end_exclusive"], name="Phase 6 window end")
        for window in _mapping(manifest["windows"], name="Phase 6 windows").values()
        if isinstance(window, Mapping) and "end_exclusive" in window
    )
    if registered_phase6_end < end:
        blockers.append({
            "code": "REQUESTED_2024_06_COVERAGE_INCOMPLETE",
            "detail": (
                "frozen Phase 6 evidence ends at "
                f"{registered_phase6_end.isoformat()}, before the requested month end"
            ),
        })
    for estimand, missing in schema_gaps.items():
        if missing:
            blockers.append({
                "code": f"{estimand.upper()}_LABELED_COHORT_MISSING",
                "detail": f"Phase 6 ledgers omit required fields: {missing}",
            })
    model_gaps = {
        "fitted_conditional_likelihood": (
            path_protocol.likelihood_artifact_status != "missing"
            and "no_likelihood_artifact" not in path_protocol.model_version
        ),
        "fitted_dol_model": (
            dol_binding.get("model_artifact") is not None
            and dol_binding.get("model_artifact_fingerprint") is not None
        ),
        "path_calibration": signal.get("path_likelihood_artifact") is not None,
        "dol_calibration": signal.get("dol_calibration_artifact") is not None,
        "hypothesis_specific_temporal": str(
            runtime.get("required_temporal_model_slot", "")
        ) in model,
    }
    for name, complete in model_gaps.items():
        if not complete:
            blockers.append({
                "code": f"{name.upper()}_MISSING",
                "detail": f"current model has no admitted {name} component",
            })
    blockers.sort(key=lambda item: item["code"])
    return {
        "schema_version": 1,
        "protocol_version": contract["protocol_version"],
        "contract_path": str(contract_path),
        "contract_sha256": _hash(contract_path),
        "requested_window": {**requested, "verified_registry_role": registered_window.role},
        "phase6_evidence_family_allowlist": list(path_protocol.evidence_allowlist),
        "phase6_ledger_rows_inspected": ledger_rows,
        "required_schema_gaps": schema_gaps,
        "model_components_complete": model_gaps,
        "ready_for_offline_fit": not blockers,
        "blockers": blockers,
        "artifacts_written": [],
        "claim_scope": "readiness_only_no_probability_validation_oos_or_trading_claim",
        "dol_no_target_outcome": dol_protocol.no_target_outcome,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--report", help="optional readiness JSON; never an artifact")
    args = parser.parse_args(argv)
    try:
        report = audit_readiness(args.contract)
    except ReadinessError as error:
        print(json.dumps({
            "ready_for_offline_fit": False,
            "error": str(error),
            "artifacts_written": [],
        }, sort_keys=True), file=sys.stderr)
        return 1
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        destination = Path(args.report).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0 if report["ready_for_offline_fit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
