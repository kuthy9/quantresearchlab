"""Fail-closed Phase 8 paired execution-research runner.

The runner is intentionally separate from the production Engine and order
FSM.  It accepts only a hash-bound Phase 8 v2 intent/research-case ledger and
either the registered 2024-06 Week-1/Week-2 OHLCV + Phase-6 minute artifacts
or a separately manifested minute-execution artifact derived from those exact
sources.

The current Phase-6 ``F`` fields are side-aggregate passive-fill proxies, not
price-level fills.  A method whose apparent fill would depend on that proxy is
therefore censored here.  Likewise, a variant that needs future cancel-state
facts or an entry-zone failure boundary absent from the input ledger is emitted
as censored.  Missing facts are never imputed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .execution import TopOfBook
from .execution_research import (
    LIMIT_EXECUTION_METHODS,
    ORDERED_EXECUTION_METHODS,
    ExecutionMethod,
    ExecutionMethodOutcome,
    ExecutionResearchConfig,
    ExecutionResearchError,
    MinuteExecutionInput,
    _evaluate_method,
    load_execution_research_config,
    minute_execution_inputs_from_phase6_frame,
)
from .execution_research_v2 import (
    EXECUTION_RESEARCH_V2_CONFIG_SHA256,
    RISK_ADMISSION_PROTOCOL_SHA256,
    ExecutionVariant,
    IntentLedgerRecord,
    MethodAvailability,
    MethodPriceProvenance,
    MethodPriceSet,
    Phase8AppendOnlyLedger,
    Phase8ExecutionResearchProtocol,
    ResearchCaseLedgerRecord,
    load_execution_research_v2_config,
    load_risk_admission_protocol,
)
from .io import iter_completed_bars, load_ohlcv
from .mbo_mechanism import load_mbo_mechanism_artifact
from .model import Bar, Direction, aware_timestamp


EXECUTION_RUNNER_SCHEMA_VERSION = "phase8_execution_research_runner_v1"
RUN_MANIFEST_SCHEMA_VERSION = "phase8_execution_research_run_manifest_v1"
PAIRED_ROWS_SCHEMA_VERSION = "phase8_execution_variant_paired_rows_v1"
SUMMARY_SCHEMA_VERSION = "phase8_execution_variant_summary_v1"
OUTPUT_MANIFEST_SCHEMA_VERSION = "phase8_execution_research_output_manifest_v1"
MINUTE_ARTIFACT_MANIFEST_SCHEMA_VERSION = "phase8_minute_execution_artifact_manifest_v1"
RUNNER_CONFIG_SHA256 = (
    "708cf08cf5f99ae0377c71d537778f547b3062c09cd48bb0e97d05bbfd608165"
)


class Phase8RunnerError(ExecutionResearchError):
    """Raised when the formal runner cannot prove an input or output fact."""


class SourceMode(str, Enum):
    REGISTERED_PHASE6 = "registered_phase6_plus_ohlcv"
    MINUTE_ARTIFACT = "minute_execution_artifact"


class MethodEvaluationStatus(str, Enum):
    EVALUATED = "evaluated"
    CENSORED = "censored"
    UNAVAILABLE = "unavailable_preoutcome"


def _timestamp(value: Any, *, name: str) -> pd.Timestamp:
    try:
        result = aware_timestamp(pd.Timestamp(value), name=name)
    except (TypeError, ValueError) as exc:
        raise Phase8RunnerError(f"{name} must be timezone aware") from exc
    return result.tz_convert("UTC")


def _identity(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in ("\n", "\r", "\x00"))
    ):
        raise Phase8RunnerError(f"{name} must be canonical non-empty text")
    return value


def _sha(value: Any, *, name: str) -> str:
    result = _identity(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise Phase8RunnerError(f"{name} must be a lowercase SHA-256")
    return result


def _normal(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if is_dataclass(value):
        return {
            item.name: _normal(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {
            str(key): _normal(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normal(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise Phase8RunnerError("canonical output cannot contain non-finite floats")
    if hasattr(value, "item"):
        return _normal(value.item())
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _normal(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise Phase8RunnerError("value is not canonical-JSON serializable") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Phase8RunnerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes(raw: bytes | str, *, name: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_duplicate_guard)
    except Phase8RunnerError:
        raise
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase8RunnerError(f"{name} is not valid duplicate-free JSON") from exc


def _exact_keys(value: Any, expected: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise Phase8RunnerError(
            f"{name} keys are not exact; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )
    return value


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_sealed_path(path: Path, *, name: str) -> None:
    if any(
        "sealed" in part.lower() or "holdout" in part.lower()
        for part in path.parts
    ):
        raise Phase8RunnerError(f"{name} cannot reference sealed/holdout paths")


def _resolve_repository_path(root: Path, value: Any, *, name: str) -> Path:
    raw = _identity(value, name=name)
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise Phase8RunnerError(f"{name} must be repository-relative")
    lowered = {part.lower() for part in relative.parts}
    if any("sealed" in part or "holdout" in part for part in lowered):
        raise Phase8RunnerError(f"{name} cannot reference sealed/holdout paths")
    result = (root / relative).resolve()
    if root != result and root not in result.parents:
        raise Phase8RunnerError(f"{name} escapes repository root")
    return result


def _require_file_binding(
    root: Path,
    binding: Any,
    *,
    name: str,
    expected_path: str | None = None,
    expected_sha256: str | None = None,
) -> tuple[Path, str]:
    typed = _exact_keys(binding, {"path", "sha256"}, name=name)
    path = _resolve_repository_path(root, typed["path"], name=f"{name} path")
    expected = _sha(typed["sha256"], name=f"{name} SHA-256")
    if path.is_symlink() or not path.is_file():
        raise Phase8RunnerError(f"{name} is not a regular file")
    if expected_path is not None and str(typed["path"]) != expected_path:
        raise Phase8RunnerError(f"{name} path differs from preregistration")
    if expected_sha256 is not None and expected != expected_sha256:
        raise Phase8RunnerError(f"{name} declared SHA differs from preregistration")
    if _file_sha(path) != expected:
        raise Phase8RunnerError(f"{name} file SHA-256 mismatch")
    return path, expected


@dataclass(frozen=True)
class RegisteredExecutionWindow:
    window_id: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    expected_rows: int
    expected_real_rows: int
    synthetic_decision_clocks: tuple[pd.Timestamp, ...]
    phase6_manifest_path: str
    phase6_manifest_sha256: str
    mbo_artifact_path: str
    mbo_artifact_sha256: str
    mbo_manifest_path: str
    mbo_manifest_sha256: str

    def __post_init__(self) -> None:
        _identity(self.window_id, name="window_id")
        start = _timestamp(self.start, name="window start")
        end = _timestamp(self.end_exclusive, name="window end_exclusive")
        synthetic = tuple(
            _timestamp(value, name="synthetic decision clock")
            for value in self.synthetic_decision_clocks
        )
        if (
            end <= start
            or type(self.expected_rows) is not int
            or type(self.expected_real_rows) is not int
            or self.expected_rows <= 0
            or self.expected_real_rows < 0
            or self.expected_real_rows + len(synthetic) != self.expected_rows
            or len(synthetic) != len(set(synthetic))
            or any(clock < start or clock >= end for clock in synthetic)
        ):
            raise Phase8RunnerError("registered window census is invalid")
        for name in (
            "phase6_manifest_path",
            "mbo_artifact_path",
            "mbo_manifest_path",
        ):
            _identity(getattr(self, name), name=f"window {name}")
        for name in (
            "phase6_manifest_sha256",
            "mbo_artifact_sha256",
            "mbo_manifest_sha256",
        ):
            _sha(getattr(self, name), name=f"window {name}")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end_exclusive", end)
        object.__setattr__(self, "synthetic_decision_clocks", synthetic)


_RUNNER_PROTOCOL_SEAL = object()


@dataclass(frozen=True)
class ExecutionResearchRunnerProtocol:
    config_sha256: str
    canonical_config_sha256: str
    windows: tuple[RegisteredExecutionWindow, ...]
    logical_instrument_id: str
    vendor_instrument_id: int
    vendor_symbol: str
    mapping_id: str
    mapping_sha256: str
    tick_size: float
    point_value: float
    ohlcv_path: str
    ohlcv_sha256: str
    ohlcv_manifest_path: str
    ohlcv_manifest_sha256: str
    analysis_minutes_after_expiry: int
    bootstrap_replicates: int
    bootstrap_seed: int
    bootstrap_confidence: float
    minimum_complete_pairs: int
    payload: Mapping[str, Any] = field(repr=False, compare=False)
    _loader_seal: object = field(init=False, default=None, repr=False, compare=False)

    def window_for(self, clock: pd.Timestamp) -> RegisteredExecutionWindow:
        typed = _timestamp(clock, name="case clock")
        matches = tuple(
            item for item in self.windows if item.start <= typed < item.end_exclusive
        )
        if len(matches) != 1:
            raise Phase8RunnerError("case clock is outside the registered W1/W2 windows")
        return matches[0]


def load_execution_research_runner_protocol(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> ExecutionResearchRunnerProtocol:
    source = Path(path)
    _reject_sealed_path(source, name="runner config")
    if source.is_symlink() or not source.is_file():
        raise Phase8RunnerError("runner config is not a regular file")
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != RUNNER_CONFIG_SHA256:
        raise Phase8RunnerError("runner config bytes differ from preregistration")
    if expected_sha256 is not None and actual != _sha(expected_sha256, name="expected runner config SHA"):
        raise Phase8RunnerError("runner config SHA-256 mismatch")
    payload = _load_json_bytes(raw, name="runner config")
    typed = _exact_keys(
        payload,
        {
            "schema_version",
            "status",
            "authority",
            "execution_protocols",
            "instrument_mapping",
            "windows",
            "shared_ohlcv",
            "case_contract",
            "fill_evidence",
            "variant_evaluability",
            "inference",
            "output_contract",
        },
        name="runner config",
    )
    if (
        typed["schema_version"] != EXECUTION_RUNNER_SCHEMA_VERSION
        or typed["status"]
        != "preregistered_2024_06_development_only_not_oos_not_trading_authority"
        or typed["authority"]
        != {
            "research_only": True,
            "order_submission": False,
            "production_engine_integration": False,
            "sealed_oos_reveal": False,
        }
        or typed["fill_evidence"]["aggregate_F_as_exact_fill_allowed"] is not False
        or typed["fill_evidence"]["queue_position_claimed"] is not False
        or typed["output_contract"]["no_clobber"] is not True
        or typed["output_contract"]["append_only_records"] is not True
    ):
        raise Phase8RunnerError("runner config authority or evidence contract is invalid")
    protocols = typed["execution_protocols"]
    if protocols != {
        "v1_config_sha256": "8212939f9dc00b11c285063a78d56f8ddbc5257a728b02829017a6d60e0d08b7",
        "v1_runtime_sha256": "1b2f338c8196ff0f963a65df66f41d48805b1eee1caaee1bf9f615d5e3341e17",
        "v2_config_sha256": EXECUTION_RESEARCH_V2_CONFIG_SHA256,
        "v2_runtime_sha256": "42e6d9b679e8a7260f77d155bd50aa3fd3e7ff626f60e0df7685d8b84ba01e40",
        "risk_config_sha256": RISK_ADMISSION_PROTOCOL_SHA256,
    }:
        raise Phase8RunnerError("runner execution protocol identities drifted")
    mapping = typed["instrument_mapping"]
    ohlcv = typed["shared_ohlcv"]
    case = typed["case_contract"]
    inference = typed["inference"]
    windows = tuple(
        RegisteredExecutionWindow(
            window_id=item["window_id"],
            start=item["start"],
            end_exclusive=item["end_exclusive"],
            expected_rows=item["expected_rows"],
            expected_real_rows=item["expected_real_rows"],
            synthetic_decision_clocks=tuple(item["synthetic_decision_clocks"]),
            phase6_manifest_path=item["phase6_manifest_path"],
            phase6_manifest_sha256=item["phase6_manifest_sha256"],
            mbo_artifact_path=item["mbo_artifact_path"],
            mbo_artifact_sha256=item["mbo_artifact_sha256"],
            mbo_manifest_path=item["mbo_manifest_path"],
            mbo_manifest_sha256=item["mbo_manifest_sha256"],
        )
        for item in typed["windows"]
    )
    if tuple(item.window_id for item in windows) != (
        "2024-06-week-1",
        "2024-06-week-2",
    ):
        raise Phase8RunnerError("runner windows differ from fixed W1/W2 order")
    result = ExecutionResearchRunnerProtocol(
        config_sha256=actual,
        canonical_config_sha256=_digest(typed),
        windows=windows,
        logical_instrument_id=mapping["logical_instrument_id"],
        vendor_instrument_id=int(mapping["vendor_instrument_id"]),
        vendor_symbol=mapping["vendor_symbol"],
        mapping_id=mapping["mapping_id"],
        mapping_sha256=mapping["mapping_sha256"],
        tick_size=float(mapping["tick_size"]),
        point_value=float(mapping["point_value"]),
        ohlcv_path=ohlcv["path"],
        ohlcv_sha256=ohlcv["sha256"],
        ohlcv_manifest_path=ohlcv["manifest_path"],
        ohlcv_manifest_sha256=ohlcv["manifest_sha256"],
        analysis_minutes_after_expiry=int(case["position_analysis_minutes_after_intent_expiry"]),
        bootstrap_replicates=int(inference["bootstrap_replicates"]),
        bootstrap_seed=int(inference["bootstrap_seed"]),
        bootstrap_confidence=float(inference["bootstrap_confidence"]),
        minimum_complete_pairs=int(inference["minimum_complete_pairs"]),
        payload=typed,
    )
    if (
        result.logical_instrument_id != "NQ:front"
        or result.vendor_instrument_id != 13743
        or result.vendor_symbol != "NQM4"
        or result.mapping_id != "instrument-map:NQ-front-to-NQM4-13743:v1"
        or result.mapping_sha256
        != "f820a6f4f329e1df667a3b876e2e0e3738f5e7b122638e9846453a4fe5b362bc"
        or not math.isclose(result.tick_size, 0.25, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(result.point_value, 20.0, rel_tol=0.0, abs_tol=1e-12)
        or result.analysis_minutes_after_expiry != 120
        or result.bootstrap_replicates != 10000
        or result.bootstrap_seed != 20240602
        or not math.isclose(result.bootstrap_confidence, 0.95)
        or result.minimum_complete_pairs != 2
    ):
        raise Phase8RunnerError("runner numeric/mapping preregistration drifted")
    object.__setattr__(result, "_loader_seal", _RUNNER_PROTOCOL_SEAL)
    return result


_RUNTIME_BINDING_PATHS = {
    "runner_config": "configs/execution_research_runner_v1.json",
    "execution_v1_config": "configs/execution_research_v1.json",
    "execution_v2_config": "configs/execution_research_v2.json",
    "risk_config": "configs/risk_admission_v1.json",
    "execution_v1_runtime": "smc_trader/execution_research.py",
    "execution_v2_runtime": "smc_trader/execution_research_v2.py",
    "runner_runtime": "smc_trader/execution_research_runner.py",
    "runner_cli": "scripts/run_execution_research_v2.py",
    "instrument_mapping_registry": "configs/shadow_live_v1.json",
}


@dataclass(frozen=True)
class Phase8RunManifestContract:
    source_path: Path
    manifest_sha256: str
    status: str
    experiment_id: str | None
    frozen_at: pd.Timestamp | None
    source_mode: SourceMode | None
    ledger_path: Path | None
    ledger_sha256: str | None
    minute_artifact_path: Path | None
    minute_artifact_sha256: str | None
    minute_manifest_path: Path | None
    minute_manifest_sha256: str | None
    output_paths: Mapping[str, Path | None]
    input_bindings: Mapping[str, Any] = field(repr=False, compare=False)
    payload: Mapping[str, Any] = field(repr=False, compare=False)
    blockers: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.blockers


def _nullable_binding(
    root: Path,
    value: Any,
    *,
    name: str,
    validate_file: bool,
) -> tuple[Path | None, str | None]:
    if value == {"path": None, "sha256": None}:
        return None, None
    typed = _exact_keys(value, {"path", "sha256"}, name=name)
    path = _resolve_repository_path(root, typed["path"], name=f"{name} path")
    digest = _sha(typed["sha256"], name=f"{name} SHA-256")
    if validate_file:
        if path.is_symlink() or not path.is_file() or _file_sha(path) != digest:
            raise Phase8RunnerError(f"{name} file identity mismatch")
    return path, digest


def validate_phase8_run_manifest(
    path: str | Path,
    *,
    project_root: str | Path,
    validate_input_files: bool = False,
) -> Phase8RunManifestContract:
    """Validate a blocked template or a frozen development run manifest."""

    root = Path(project_root).resolve()
    source = Path(path)
    _reject_sealed_path(source, name="run manifest")
    if source.is_symlink() or not source.is_file():
        raise Phase8RunnerError("run manifest is not a regular file")
    resolved_source = source.resolve()
    if root != resolved_source and root not in resolved_source.parents:
        raise Phase8RunnerError("run manifest must remain under the repository root")
    source = resolved_source
    raw = source.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    payload = _load_json_bytes(raw, name="run manifest")
    typed = _exact_keys(
        payload,
        {
            "schema_version",
            "status",
            "authority",
            "experiment_id",
            "frozen_at",
            "frozen_before_run",
            "window_ids",
            "runtime_bindings",
            "input_bindings",
            "outputs",
            "readiness_blockers",
        },
        name="run manifest",
    )
    status = typed["status"]
    if (
        typed["schema_version"] != RUN_MANIFEST_SCHEMA_VERSION
        or status
        not in {
            "template_incomplete_not_authorized_to_run",
            "frozen_2024_06_development_research_not_oos_not_trading_authority",
        }
        or typed["authority"]
        != {
            "research_only": True,
            "order_submission": False,
            "production_engine_integration": False,
            "sealed_oos_reveal": False,
        }
        or tuple(typed["window_ids"])
        != ("2024-06-week-1", "2024-06-week-2")
    ):
        raise Phase8RunnerError("run manifest authority/window contract is invalid")

    runtime_bindings = _exact_keys(
        typed["runtime_bindings"],
        set(_RUNTIME_BINDING_PATHS),
        name="run manifest runtime_bindings",
    )
    for name, expected_path in _RUNTIME_BINDING_PATHS.items():
        expected_sha = None
        if name == "runner_config":
            expected_sha = RUNNER_CONFIG_SHA256
        elif name == "execution_v1_config":
            expected_sha = "8212939f9dc00b11c285063a78d56f8ddbc5257a728b02829017a6d60e0d08b7"
        elif name == "execution_v2_config":
            expected_sha = EXECUTION_RESEARCH_V2_CONFIG_SHA256
        elif name == "risk_config":
            expected_sha = RISK_ADMISSION_PROTOCOL_SHA256
        elif name == "execution_v1_runtime":
            expected_sha = "1b2f338c8196ff0f963a65df66f41d48805b1eee1caaee1bf9f615d5e3341e17"
        elif name == "execution_v2_runtime":
            expected_sha = "42e6d9b679e8a7260f77d155bd50aa3fd3e7ff626f60e0df7685d8b84ba01e40"
        elif name == "instrument_mapping_registry":
            expected_sha = "d8e543f54bc6e81bb820be51e02382b5477f8a3e8e676063aa685f0d599e7961"
        _require_file_binding(
            root,
            runtime_bindings[name],
            name=f"runtime binding {name}",
            expected_path=expected_path,
            expected_sha256=expected_sha,
        )

    inputs = _exact_keys(
        typed["input_bindings"],
        {
            "intent_research_case_ledger",
            "source_mode",
            "minute_execution_artifact",
            "minute_execution_manifest",
        },
        name="run manifest input_bindings",
    )
    ledger_path, ledger_sha = _nullable_binding(
        root,
        inputs["intent_research_case_ledger"],
        name="intent/research-case ledger",
        validate_file=validate_input_files,
    )
    raw_mode = inputs["source_mode"]
    try:
        source_mode = None if raw_mode is None else SourceMode(raw_mode)
    except (TypeError, ValueError) as exc:
        raise Phase8RunnerError("run manifest source_mode is invalid") from exc
    minute_path, minute_sha = _nullable_binding(
        root,
        inputs["minute_execution_artifact"],
        name="minute execution artifact",
        validate_file=validate_input_files,
    )
    minute_manifest_path, minute_manifest_sha = _nullable_binding(
        root,
        inputs["minute_execution_manifest"],
        name="minute execution manifest",
        validate_file=validate_input_files,
    )
    if source_mode is SourceMode.REGISTERED_PHASE6 and any(
        value is not None
        for value in (minute_path, minute_sha, minute_manifest_path, minute_manifest_sha)
    ):
        raise Phase8RunnerError("registered Phase6 mode cannot bind a minute artifact")
    if source_mode is SourceMode.MINUTE_ARTIFACT and any(
        value is None
        for value in (minute_path, minute_sha, minute_manifest_path, minute_manifest_sha)
    ):
        raise Phase8RunnerError("minute-artifact mode requires artifact and manifest bindings")

    output_values = _exact_keys(
        typed["outputs"],
        {"paired_rows", "summary", "output_manifest"},
        name="run manifest outputs",
    )
    output_paths: dict[str, Path | None] = {}
    for name, value in output_values.items():
        if value is None:
            output_paths[name] = None
            continue
        target = _resolve_repository_path(root, value, name=f"output {name}")
        results_root = (root / "experiments/results").resolve()
        if results_root != target.parent and results_root not in target.parents:
            raise Phase8RunnerError("run outputs must remain under experiments/results")
        output_paths[name] = target
    nonnull_outputs = tuple(value for value in output_paths.values() if value is not None)
    if len(nonnull_outputs) != len(set(nonnull_outputs)):
        raise Phase8RunnerError("run output paths must be distinct")

    experiment_id = typed["experiment_id"]
    if experiment_id is not None:
        experiment_id = _identity(experiment_id, name="run experiment_id")
    blockers: list[str] = []
    if ledger_path is None:
        blockers.append("intent_research_case_ledger_not_bound")
    if source_mode is None:
        blockers.append("minute_source_mode_not_bound")
    if any(value is None for value in output_paths.values()):
        blockers.append("outputs_not_registered")
    if experiment_id is None:
        blockers.append("experiment_identity_missing")
    if typed["frozen_at"] is None:
        blockers.append("frozen_clock_missing")
    if typed["frozen_before_run"] is not True:
        blockers.append("manifest_not_frozen")
    if not isinstance(typed["readiness_blockers"], list) or any(
        not isinstance(item, str) or not item
        for item in typed["readiness_blockers"]
    ):
        raise Phase8RunnerError("run manifest readiness blockers are invalid")
    declared_blockers = tuple(typed["readiness_blockers"])
    if tuple(blockers) != declared_blockers:
        raise Phase8RunnerError("run manifest readiness blockers are not deterministic")
    frozen_at = (
        None
        if typed["frozen_at"] is None
        else _timestamp(typed["frozen_at"], name="run manifest frozen_at")
    )
    if status == "template_incomplete_not_authorized_to_run" and not blockers:
        raise Phase8RunnerError("incomplete template cannot claim readiness")
    if status.startswith("frozen_") and blockers:
        raise Phase8RunnerError("frozen run manifest cannot retain readiness blockers")
    return Phase8RunManifestContract(
        source_path=source.resolve(),
        manifest_sha256=manifest_sha,
        status=status,
        experiment_id=experiment_id,
        frozen_at=frozen_at,
        source_mode=source_mode,
        ledger_path=ledger_path,
        ledger_sha256=ledger_sha,
        minute_artifact_path=minute_path,
        minute_artifact_sha256=minute_sha,
        minute_manifest_path=minute_manifest_path,
        minute_manifest_sha256=minute_manifest_sha,
        output_paths=output_paths,
        input_bindings=inputs,
        payload=typed,
        blockers=tuple(blockers),
    )


_MINUTE_PAYLOAD_FIELDS = {
    "input_id",
    "decision_time",
    "symbol",
    "instrument_id",
    "vendor_instrument_id",
    "instrument_mapping_id",
    "instrument_mapping_sha256",
    "bar",
    "book",
    "book_valid",
    "invalid_reason",
    "passive_bid_fill_volume",
    "passive_ask_fill_volume",
    "displayed_bid_add_volume",
    "displayed_ask_add_volume",
    "displayed_bid_cancel_volume",
    "displayed_ask_cancel_volume",
    "aggressor_buy_volume",
    "aggressor_sell_volume",
    "source_reset",
    "synthetic_source",
    "source_artifact_id",
    "source_artifact_sha256",
    "source_row_sha256",
}


def minute_execution_input_to_payload(value: MinuteExecutionInput) -> dict[str, Any]:
    if not isinstance(value, MinuteExecutionInput):
        raise TypeError("minute serializer requires MinuteExecutionInput")
    bar = value.bar
    book = value.book
    return {
        "input_id": value.input_id,
        "decision_time": value.decision_time.isoformat(),
        "symbol": value.symbol,
        "instrument_id": value.instrument_id,
        "vendor_instrument_id": value.vendor_instrument_id,
        "instrument_mapping_id": value.instrument_mapping_id,
        "instrument_mapping_sha256": value.instrument_mapping_sha256,
        "bar": {
            "start": bar.start.isoformat(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "symbol": bar.symbol,
            "instrument_id": bar.instrument_id,
            "synthetic_no_trade": bar.synthetic_no_trade,
            "data_gap_before_minutes": bar.data_gap_before_minutes,
        },
        "book": (
            None
            if book is None
            else {
                "observed_at": book.observed_at.isoformat(),
                "bid": book.bid,
                "ask": book.ask,
                "bid_size": book.bid_size,
                "ask_size": book.ask_size,
            }
        ),
        "book_valid": value.book_valid,
        "invalid_reason": value.invalid_reason,
        "passive_bid_fill_volume": value.passive_bid_fill_volume,
        "passive_ask_fill_volume": value.passive_ask_fill_volume,
        "displayed_bid_add_volume": value.displayed_bid_add_volume,
        "displayed_ask_add_volume": value.displayed_ask_add_volume,
        "displayed_bid_cancel_volume": value.displayed_bid_cancel_volume,
        "displayed_ask_cancel_volume": value.displayed_ask_cancel_volume,
        "aggressor_buy_volume": value.aggressor_buy_volume,
        "aggressor_sell_volume": value.aggressor_sell_volume,
        "source_reset": value.source_reset,
        "synthetic_source": value.synthetic_source,
        "source_artifact_id": value.source_artifact_id,
        "source_artifact_sha256": value.source_artifact_sha256,
        "source_row_sha256": value.source_row_sha256,
    }


def minute_execution_input_from_payload(payload: Mapping[str, Any]) -> MinuteExecutionInput:
    typed = _exact_keys(payload, _MINUTE_PAYLOAD_FIELDS, name="minute execution row")
    bar_payload = _exact_keys(
        typed["bar"],
        {
            "start",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
            "instrument_id",
            "synthetic_no_trade",
            "data_gap_before_minutes",
        },
        name="minute execution bar",
    )
    bar = Bar(**bar_payload)
    book_payload = typed["book"]
    book = None
    if book_payload is not None:
        book = TopOfBook(**_exact_keys(
            book_payload,
            {"observed_at", "bid", "ask", "bid_size", "ask_size"},
            name="minute execution book",
        ))
    result = MinuteExecutionInput(
        decision_time=typed["decision_time"],
        symbol=typed["symbol"],
        instrument_id=typed["instrument_id"],
        vendor_instrument_id=typed["vendor_instrument_id"],
        instrument_mapping_id=typed["instrument_mapping_id"],
        instrument_mapping_sha256=typed["instrument_mapping_sha256"],
        bar=bar,
        book=book,
        book_valid=typed["book_valid"],
        invalid_reason=typed["invalid_reason"],
        passive_bid_fill_volume=typed["passive_bid_fill_volume"],
        passive_ask_fill_volume=typed["passive_ask_fill_volume"],
        displayed_bid_add_volume=typed["displayed_bid_add_volume"],
        displayed_ask_add_volume=typed["displayed_ask_add_volume"],
        displayed_bid_cancel_volume=typed["displayed_bid_cancel_volume"],
        displayed_ask_cancel_volume=typed["displayed_ask_cancel_volume"],
        aggressor_buy_volume=typed["aggressor_buy_volume"],
        aggressor_sell_volume=typed["aggressor_sell_volume"],
        source_reset=typed["source_reset"],
        synthetic_source=typed["synthetic_source"],
        source_artifact_id=typed["source_artifact_id"],
        source_artifact_sha256=typed["source_artifact_sha256"],
        source_row_sha256=typed["source_row_sha256"],
    )
    if result.input_id != typed["input_id"]:
        raise Phase8RunnerError("minute execution input identity conflicts with content")
    return result


def _validate_minute_census(
    inputs: Sequence[MinuteExecutionInput],
    protocol: ExecutionResearchRunnerProtocol,
) -> tuple[MinuteExecutionInput, ...]:
    ordered = tuple(sorted(inputs, key=lambda item: item.decision_time))
    if (
        len({item.input_id for item in ordered}) != len(ordered)
        or len({item.decision_time for item in ordered}) != len(ordered)
    ):
        raise Phase8RunnerError("minute execution artifact has duplicate identities/clocks")
    for window in protocol.windows:
        rows = tuple(
            item
            for item in ordered
            if window.start <= item.decision_time < window.end_exclusive
        )
        synthetic = tuple(
            item.decision_time for item in rows if item.synthetic_source
        )
        if (
            len(rows) != window.expected_rows
            or sum(not item.synthetic_source for item in rows)
            != window.expected_real_rows
            or synthetic != window.synthetic_decision_clocks
            or any(
                item.source_artifact_id != f"phase6-mbo:{window.window_id}"
                or item.source_artifact_sha256 != window.mbo_artifact_sha256
                for item in rows
            )
        ):
            raise Phase8RunnerError(
                f"minute execution census differs for {window.window_id}"
            )
    if len(ordered) != sum(window.expected_rows for window in protocol.windows):
        raise Phase8RunnerError("minute artifact contains clocks outside registered W1/W2")
    if any(
        item.symbol != protocol.vendor_symbol
        or item.instrument_id != protocol.logical_instrument_id
        or item.vendor_instrument_id != protocol.vendor_instrument_id
        or item.instrument_mapping_id != protocol.mapping_id
        or item.instrument_mapping_sha256 != protocol.mapping_sha256
        for item in ordered
    ):
        raise Phase8RunnerError("minute execution artifact contract mapping differs")
    return ordered


def load_minute_execution_artifact(
    artifact_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_artifact_sha256: str,
    expected_manifest_sha256: str,
    protocol: ExecutionResearchRunnerProtocol,
) -> tuple[MinuteExecutionInput, ...]:
    """Load a fully manifested W1/W2 JSONL minute artifact."""

    if protocol._loader_seal is not _RUNNER_PROTOCOL_SEAL:
        raise TypeError("minute artifact loader requires loaded runner protocol")
    artifact = Path(artifact_path)
    manifest_source = Path(manifest_path)
    _reject_sealed_path(artifact, name="minute artifact")
    _reject_sealed_path(manifest_source, name="minute artifact manifest")
    if (
        artifact.is_symlink()
        or manifest_source.is_symlink()
        or not artifact.is_file()
        or not manifest_source.is_file()
    ):
        raise Phase8RunnerError("minute artifact/manifest must be regular files")
    artifact_sha = _sha(expected_artifact_sha256, name="minute artifact SHA")
    manifest_sha = _sha(expected_manifest_sha256, name="minute manifest SHA")
    if _file_sha(artifact) != artifact_sha or _file_sha(manifest_source) != manifest_sha:
        raise Phase8RunnerError("minute artifact/manifest SHA mismatch")
    manifest = _load_json_bytes(manifest_source.read_bytes(), name="minute artifact manifest")
    typed = _exact_keys(
        manifest,
        {
            "schema_version",
            "artifact_kind",
            "status",
            "authority",
            "output",
            "window_ids",
            "upstream_source_sha256",
            "instrument_mapping",
            "fill_evidence",
        },
        name="minute artifact manifest",
    )
    expected_upstream = {
        "ohlcv": protocol.ohlcv_sha256,
        "ohlcv_manifest": protocol.ohlcv_manifest_sha256,
        "week1_phase6_mbo": protocol.windows[0].mbo_artifact_sha256,
        "week1_phase6_mbo_manifest": protocol.windows[0].mbo_manifest_sha256,
        "week1_phase6_source_manifest": protocol.windows[0].phase6_manifest_sha256,
        "week2_phase6_mbo": protocol.windows[1].mbo_artifact_sha256,
        "week2_phase6_mbo_manifest": protocol.windows[1].mbo_manifest_sha256,
        "week2_phase6_source_manifest": protocol.windows[1].phase6_manifest_sha256,
    }
    if (
        typed["schema_version"] != MINUTE_ARTIFACT_MANIFEST_SCHEMA_VERSION
        or typed["artifact_kind"] != "phase8_minute_execution_inputs"
        or typed["status"] != "frozen_2024_06_development_input_not_oos"
        or typed["authority"]
        != {"research_only": True, "sealed_oos": False, "order_submission": False}
        or tuple(typed["window_ids"])
        != ("2024-06-week-1", "2024-06-week-2")
        or typed["upstream_source_sha256"] != expected_upstream
        or typed["instrument_mapping"]
        != {
            "mapping_id": protocol.mapping_id,
            "mapping_sha256": protocol.mapping_sha256,
            "logical_instrument_id": protocol.logical_instrument_id,
            "vendor_instrument_id": protocol.vendor_instrument_id,
            "vendor_symbol": protocol.vendor_symbol,
        }
        or typed["fill_evidence"]
        != "top_of_book_plus_side_aggregate_passive_proxy_not_price_level"
        or typed["output"]
        != {
            "sha256": artifact_sha,
            "rows": sum(window.expected_rows for window in protocol.windows),
        }
    ):
        raise Phase8RunnerError("minute artifact manifest contract differs")
    rows: list[MinuteExecutionInput] = []
    with artifact.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith("\n") or not line.strip():
                raise Phase8RunnerError(
                    f"minute artifact line {line_number} is not canonical JSONL"
                )
            payload = _load_json_bytes(line, name=f"minute artifact line {line_number}")
            if _canonical_json(payload) + "\n" != line:
                raise Phase8RunnerError(
                    f"minute artifact line {line_number} is not canonical JSON"
                )
            rows.append(minute_execution_input_from_payload(payload))
    return _validate_minute_census(rows, protocol)


def _validate_phase6_source_manifest(
    path: Path,
    *,
    window: RegisteredExecutionWindow,
    protocol: ExecutionResearchRunnerProtocol,
) -> None:
    payload = _load_json_bytes(path.read_bytes(), name=f"{window.window_id} Phase6 manifest")
    if not isinstance(payload, Mapping):
        raise Phase8RunnerError("Phase6 source manifest root is invalid")
    identity = payload.get("identity_bindings")
    windows = payload.get("windows")
    authority = payload.get("authority")
    if (
        payload.get("status")
        != "frozen_mbo_development_mechanism_validation_not_oos_not_trading_authority"
        or payload.get("frozen_before_run") is not True
        or not isinstance(identity, Mapping)
        or not isinstance(windows, Mapping)
        or not isinstance(authority, Mapping)
        or authority.get("sealed_holdout_opened") is not False
        or payload.get("contract")
        != {"symbol": protocol.vendor_symbol, "instrument_id": protocol.vendor_instrument_id}
        or identity.get("ohlcv_artifact")
        != {"path": protocol.ohlcv_path, "sha256": protocol.ohlcv_sha256}
        or identity.get("mbo_feature_artifact")
        != {"path": window.mbo_artifact_path, "sha256": window.mbo_artifact_sha256}
        or identity.get("mbo_feature_manifest")
        != {"path": window.mbo_manifest_path, "sha256": window.mbo_manifest_sha256}
    ):
        raise Phase8RunnerError("Phase6 source manifest does not bind exact registered inputs")
    expected_window = next(
        (
            item
            for item in windows.values()
            if isinstance(item, Mapping) and item.get("id") == window.window_id
        ),
        None,
    )
    if expected_window != {
        "id": window.window_id,
        "start": window.start.isoformat().replace("+00:00", "Z"),
        "end_exclusive": window.end_exclusive.isoformat().replace("+00:00", "Z"),
        "expected_rows": window.expected_rows,
    }:
        raise Phase8RunnerError("Phase6 source window differs from runner preregistration")


def load_registered_phase6_minute_inputs(
    project_root: str | Path,
    protocol: ExecutionResearchRunnerProtocol,
) -> tuple[MinuteExecutionInput, ...]:
    """Load the exact open-development W1/W2 sources; never discover paths."""

    if protocol._loader_seal is not _RUNNER_PROTOCOL_SEAL:
        raise TypeError("Phase6 loader requires loaded runner protocol")
    root = Path(project_root).resolve()
    ohlcv_path = _resolve_repository_path(root, protocol.ohlcv_path, name="OHLCV path")
    ohlcv_manifest = _resolve_repository_path(
        root, protocol.ohlcv_manifest_path, name="OHLCV manifest path"
    )
    for path, digest, name in (
        (ohlcv_path, protocol.ohlcv_sha256, "OHLCV artifact"),
        (ohlcv_manifest, protocol.ohlcv_manifest_sha256, "OHLCV manifest"),
    ):
        if path.is_symlink() or not path.is_file() or _file_sha(path) != digest:
            raise Phase8RunnerError(f"{name} identity differs from runner preregistration")

    results: list[MinuteExecutionInput] = []
    for window in protocol.windows:
        phase6_manifest = _resolve_repository_path(
            root, window.phase6_manifest_path, name="Phase6 source manifest path"
        )
        mbo_path = _resolve_repository_path(
            root, window.mbo_artifact_path, name="Phase6 MBO artifact path"
        )
        mbo_manifest = _resolve_repository_path(
            root, window.mbo_manifest_path, name="Phase6 MBO manifest path"
        )
        for path, digest, name in (
            (phase6_manifest, window.phase6_manifest_sha256, "Phase6 source manifest"),
            (mbo_path, window.mbo_artifact_sha256, "Phase6 MBO artifact"),
            (mbo_manifest, window.mbo_manifest_sha256, "Phase6 MBO manifest"),
        ):
            if path.is_symlink() or not path.is_file() or _file_sha(path) != digest:
                raise Phase8RunnerError(f"{name} identity mismatch for {window.window_id}")
        _validate_phase6_source_manifest(
            phase6_manifest,
            window=window,
            protocol=protocol,
        )
        frame = load_mbo_mechanism_artifact(
            mbo_path,
            manifest_path=mbo_manifest,
            verify_lineage=False,
            expected_manifest_sha256=window.mbo_manifest_sha256,
            expected_start=window.start,
            expected_end=window.end_exclusive,
            expected_symbol=protocol.vendor_symbol,
            expected_instrument_id=protocol.vendor_instrument_id,
            expected_rows=window.expected_rows,
        )
        loaded = load_ohlcv(
            ohlcv_path,
            start=window.start,
            end=window.end_exclusive,
        )
        if not loaded.contract_selection_causal:
            raise Phase8RunnerError("registered OHLCV contract selection is not causal")
        bars = tuple(
            iter_completed_bars(
                loaded.frame,
                maximum_no_trade_gap_minutes=5,
                allow_data_gap_reset=False,
            )
        )
        if (
            len(bars) != window.expected_rows
            or sum(not bar.synthetic_no_trade for bar in bars) != window.expected_real_rows
            or tuple(bar.end.tz_convert("UTC") for bar in bars if bar.synthetic_no_trade)
            != window.synthetic_decision_clocks
        ):
            raise Phase8RunnerError(f"OHLCV reader census differs for {window.window_id}")
        adapted = minute_execution_inputs_from_phase6_frame(
            frame,
            bars,
            source_artifact_id=f"phase6-mbo:{window.window_id}",
            source_artifact_sha256=window.mbo_artifact_sha256,
            logical_instrument_id=protocol.logical_instrument_id,
            instrument_mapping_id=protocol.mapping_id,
            instrument_mapping_sha256=protocol.mapping_sha256,
        )
        results.extend(adapted)
    return _validate_minute_census(results, protocol)


@dataclass(frozen=True)
class _PerMethodIntent:
    source_trade_intent_id: str
    research_intent_id: str
    created_at: pd.Timestamp
    expires_at: pd.Timestamp
    analysis_ends_at: pd.Timestamp
    symbol: str
    instrument_id: str | int
    vendor_instrument_id: int
    instrument_mapping_id: str
    instrument_mapping_sha256: str
    side: Direction
    quantity: int
    tick_size: float
    point_value: float
    arrival_mid: float
    invalidation_price: float
    target_price: float
    method: ExecutionMethod
    order_price: float | None

    @property
    def entry_expires_at(self) -> pd.Timestamp:
        return self.expires_at

    def price_for(self, method: ExecutionMethod) -> float | None:
        typed = ExecutionMethod(method)
        if typed is not self.method:
            raise Phase8RunnerError("per-method evaluator crossed method identity")
        return self.order_price


def _intent_payload(record: IntentLedgerRecord) -> Mapping[str, Any]:
    value = _load_json_bytes(record.canonical_intent_json, name="TradeIntent ledger snapshot")
    if not isinstance(value, Mapping):
        raise Phase8RunnerError("TradeIntent ledger snapshot root is invalid")
    required = {
        "intent_id",
        "created_at",
        "expires_at",
        "symbol",
        "instrument_id",
        "side",
        "quantity",
        "point_value",
        "invalidation",
        "targets",
        "cancel_conditions",
        "setup_family",
        "source_event_ids",
        "source_identity_ids",
    }
    if not required.issubset(value):
        raise Phase8RunnerError(
            f"TradeIntent ledger snapshot lacks runner fields: {sorted(required-set(value))}"
        )
    return value


def _required_case_source_ids(
    window: RegisteredExecutionWindow,
    protocol: ExecutionResearchRunnerProtocol,
) -> frozenset[str]:
    return frozenset(
        {
            f"phase6-mbo-sha256:{window.mbo_artifact_sha256}",
            f"ohlcv-sha256:{protocol.ohlcv_sha256}",
            f"instrument-mapping-sha256:{protocol.mapping_sha256}",
        }
    )


def _variant_entry_expiry(
    variant: ExecutionVariant,
    *,
    created_at: pd.Timestamp,
    native_expiry: pd.Timestamp,
    inputs: Sequence[MinuteExecutionInput],
) -> pd.Timestamp | None:
    count = variant.wait_limit_completed_m1_bars
    if count is None:
        return native_expiry
    real = tuple(
        item.decision_time
        for item in inputs
        if item.decision_time > created_at and not item.synthetic_source
    )
    if len(real) < count:
        return None
    return min(native_expiry, real[count - 1])


def _cancel_conditions_supported(payload: Mapping[str, Any]) -> bool:
    conditions = payload.get("cancel_conditions")
    if not isinstance(conditions, list) or not conditions:
        return False
    allowed = {"signal_expiry_reached", "structural_invalidation_reached"}
    for item in conditions:
        if not isinstance(item, Mapping) or item.get("kind") not in allowed:
            return False
        if item.get("kind") == "signal_expiry_reached":
            trigger = item.get("trigger_at")
            if trigger is None or _timestamp(trigger, name="signal expiry cancel clock") != _timestamp(
                payload["expires_at"], name="intent expiry"
            ):
                return False
    return True


def _method_price_matches_arrival(
    provenance: MethodPriceProvenance,
    *,
    direction: Direction,
    arrival: MinuteExecutionInput,
) -> bool:
    if provenance.method is not ExecutionMethod.MARKET:
        return True
    if arrival.book is None or provenance.price_ticks is None:
        return False
    executable = (
        arrival.book.ask if direction is Direction.LONG else arrival.book.bid
    )
    ticks = executable / provenance.tick_size
    return math.isclose(ticks, provenance.price_ticks, rel_tol=0.0, abs_tol=1e-9)


def _marketable(direction: Direction, order_price: float, book: TopOfBook) -> bool:
    return (
        order_price >= book.ask
        if direction is Direction.LONG
        else order_price <= book.bid
    )


def _limit_touched(direction: Direction, order_price: float, bar: Bar) -> bool:
    return bar.low <= order_price if direction is Direction.LONG else bar.high >= order_price


def _unproven_resting_fill_required(
    *,
    direction: Direction,
    order_price: float,
    entry_expiry: pd.Timestamp,
    created_at: pd.Timestamp,
    inputs: Sequence[MinuteExecutionInput],
    execution_config: ExecutionResearchConfig,
) -> bool:
    """Detect the first fill opportunity that would consume side-aggregate F."""

    for item in inputs:
        if item.decision_time < created_at or item.decision_time >= entry_expiry:
            continue
        if item.source_censor_reason(execution_config) is not None or item.book is None:
            return False
        if _marketable(direction, order_price, item.book):
            return False
        if item.decision_time <= created_at or not _limit_touched(
            direction, order_price, item.bar
        ):
            continue
        passive = (
            item.passive_bid_fill_volume
            if direction is Direction.LONG
            else item.passive_ask_fill_volume
        )
        if passive is not None and float(passive) >= 1.0:
            return True
    return False


def _method_intent(
    *,
    payload: Mapping[str, Any],
    provenance: MethodPriceProvenance,
    variant: ExecutionVariant,
    entry_expiry: pd.Timestamp,
    analysis_end: pd.Timestamp,
    arrival_mid: float,
    protocol: ExecutionResearchRunnerProtocol,
    case_id: str,
) -> _PerMethodIntent:
    side = Direction(payload["side"])
    invalidation = float(payload["invalidation"]["price"])
    primary_target = float(payload["targets"][0]["price"])
    order_price = provenance.price
    target = primary_target
    if variant.variant_id == "target_1r_capped_dol_v1":
        entry = arrival_mid if order_price is None else order_price
        risk = abs(entry - invalidation)
        target = (
            min(entry + risk, primary_target)
            if side is Direction.LONG
            else max(entry - risk, primary_target)
        )
    identity_payload = {
        "case_id": case_id,
        "variant_id": variant.variant_id,
        "method": provenance.method.value,
        "entry_expiry": entry_expiry.isoformat(),
        "analysis_end": analysis_end.isoformat(),
        "order_price": order_price,
        "invalidation": invalidation,
        "target": target,
    }
    return _PerMethodIntent(
        source_trade_intent_id=payload["intent_id"],
        research_intent_id=f"execution-v2-method-intent:{_digest(identity_payload)}",
        created_at=_timestamp(payload["created_at"], name="intent created_at"),
        expires_at=entry_expiry,
        analysis_ends_at=analysis_end,
        symbol=payload["symbol"],
        instrument_id=payload["instrument_id"],
        vendor_instrument_id=protocol.vendor_instrument_id,
        instrument_mapping_id=protocol.mapping_id,
        instrument_mapping_sha256=protocol.mapping_sha256,
        side=side,
        quantity=int(payload["quantity"]),
        tick_size=protocol.tick_size,
        point_value=float(payload["point_value"]),
        arrival_mid=arrival_mid,
        invalidation_price=invalidation,
        target_price=target,
        method=provenance.method,
        order_price=(None if provenance.method is ExecutionMethod.MARKET else order_price),
    )


def _evaluability_reason(
    variant: ExecutionVariant,
    *,
    intent_payload: Mapping[str, Any],
) -> str | None:
    if variant.variant_id in {
        "primary_v1",
        "wait_1m_v1",
        "wait_5m_v1",
        "target_1r_capped_dol_v1",
    } and not _cancel_conditions_supported(intent_payload):
        return "future_frozen_cancel_condition_state_ledger_unavailable"
    if variant.variant_id == "cancel_gtt_only_v1":
        return "pending_vs_post_fill_terminal_split_not_representable_by_v1_1"
    if variant.variant_id == "stop_entry_zone_failure_v1":
        return "entry_zone_failure_boundary_ticks_unavailable_preoutcome"
    return None


def _method_result(
    provenance: MethodPriceProvenance,
    *,
    variant_reason: str | None,
    intent: _PerMethodIntent | None,
    inputs: tuple[MinuteExecutionInput, ...],
    execution_config: ExecutionResearchConfig,
) -> dict[str, Any]:
    base = {
        "method": provenance.method.value,
        "method_price_provenance_id": provenance.provenance_id,
        "preoutcome_availability": provenance.availability.value,
        "price_ticks": provenance.price_ticks,
        "evaluation_status": None,
        "censor_reasons": [],
        "outcome": None,
    }
    if provenance.availability is not MethodAvailability.AVAILABLE:
        base["evaluation_status"] = MethodEvaluationStatus.UNAVAILABLE.value
        base["censor_reasons"] = [
            f"method_{provenance.availability.value}:{provenance.availability_reason}"
        ]
        return base
    if variant_reason is not None:
        base["evaluation_status"] = MethodEvaluationStatus.CENSORED.value
        base["censor_reasons"] = [variant_reason]
        return base
    if intent is None:
        raise AssertionError("evaluable method is missing its internal intent")
    if (
        provenance.method is not ExecutionMethod.MARKET
        and intent.order_price is not None
        and _unproven_resting_fill_required(
            direction=intent.side,
            order_price=intent.order_price,
            entry_expiry=intent.entry_expires_at,
            created_at=intent.created_at,
            inputs=inputs,
            execution_config=execution_config,
        )
    ):
        base["evaluation_status"] = MethodEvaluationStatus.CENSORED.value
        base["censor_reasons"] = [
            "phase6_side_aggregate_passive_F_cannot_prove_method_price_fill"
        ]
        return base
    outcome: ExecutionMethodOutcome = _evaluate_method(
        intent,
        inputs,
        execution_config,
        provenance.method,
    )
    base["evaluation_status"] = MethodEvaluationStatus.EVALUATED.value
    base["outcome"] = _normal(outcome)
    return base


def _paired_contrasts(method_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_method = {item["method"]: item for item in method_results}
    market = by_method[ExecutionMethod.MARKET.value]
    contrasts: list[dict[str, Any]] = []
    for method in LIMIT_EXECUTION_METHODS:
        treatment = by_method[method.value]
        reasons: list[str] = []
        difference: float | None = None
        if market["evaluation_status"] != MethodEvaluationStatus.EVALUATED.value:
            reasons.append("market_comparator_not_evaluated")
        if treatment["evaluation_status"] != MethodEvaluationStatus.EVALUATED.value:
            reasons.append("treatment_method_not_evaluated")
        if not reasons:
            market_value = market["outcome"]["implementation_shortfall_points"]
            treatment_value = treatment["outcome"]["implementation_shortfall_points"]
            if market_value is None or treatment_value is None:
                reasons.append("implementation_shortfall_pair_incomplete")
            else:
                difference = float(treatment_value) - float(market_value)
        contrasts.append(
            {
                "method": method.value,
                "comparator": ExecutionMethod.MARKET.value,
                "complete_pair": difference is not None,
                "difference_implementation_shortfall_points": difference,
                "unavailable_reasons": sorted(set(reasons)),
            }
        )
    return contrasts


def _tick_aligned(value: float, tick_size: float) -> bool:
    ratio = float(value) / float(tick_size)
    return math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9)


def _validate_intent_case_contract(
    intent_record: IntentLedgerRecord,
    case_record: ResearchCaseLedgerRecord,
    *,
    payload: Mapping[str, Any],
    arrival: MinuteExecutionInput,
    window: RegisteredExecutionWindow,
    runner_protocol: ExecutionResearchRunnerProtocol,
    execution_protocol: Phase8ExecutionResearchProtocol,
    risk_protocol_id: str,
    risk_protocol_sha256: str,
) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """Validate all facts needed before an outcome evaluator may be called."""

    if (
        case_record.source_intent_record_id != intent_record.record_id
        or case_record.source_trade_intent_id != intent_record.source_trade_intent_id
        or case_record.created_at != intent_record.created_at
        or case_record.execution_protocol_id != execution_protocol.protocol_id
        or case_record.execution_protocol_sha256
        != execution_protocol.source_file_sha256
        or case_record.risk_protocol_id != risk_protocol_id
        or case_record.risk_protocol_sha256 != risk_protocol_sha256
        or case_record.variant_registry_id
        != execution_protocol.variant_registry_id
    ):
        raise Phase8RunnerError("research case protocol/intent identity drifted")
    if set(case_record.source_artifact_ids) != set(
        _required_case_source_ids(window, runner_protocol)
    ):
        raise Phase8RunnerError("research case does not bind the exact registered sources")

    created = _timestamp(payload["created_at"], name="intent created_at")
    expires = _timestamp(payload["expires_at"], name="intent expires_at")
    analysis_end = expires + pd.Timedelta(
        runner_protocol.analysis_minutes_after_expiry,
        unit="m",
    )
    if (
        created != intent_record.created_at
        or expires <= created
        or expires >= window.end_exclusive
        or analysis_end >= window.end_exclusive
    ):
        raise Phase8RunnerError("intent horizon is invalid or crosses a registered window")
    if (
        payload["intent_id"] != intent_record.source_trade_intent_id
        or payload["symbol"] != runner_protocol.vendor_symbol
        or payload["instrument_id"] != runner_protocol.logical_instrument_id
        or type(payload["quantity"]) is not int
        or payload["quantity"] <= 0
        or not math.isclose(
            float(payload["point_value"]),
            runner_protocol.point_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise Phase8RunnerError("TradeIntent contract differs from the registered instrument")
    try:
        side = Direction(payload["side"])
        invalidation_payload = payload["invalidation"]
        target_payload = payload["targets"][0]
        invalidation = float(invalidation_payload["price"])
        target = float(target_payload["price"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise Phase8RunnerError("TradeIntent stop/target snapshot is invalid") from exc
    if (
        not isinstance(invalidation_payload, Mapping)
        or not isinstance(target_payload, Mapping)
        or _timestamp(
            invalidation_payload.get("observed_at"),
            name="intent invalidation observed_at",
        )
        > created
        or _timestamp(
            target_payload.get("confirmed_at"),
            name="intent target confirmed_at",
        )
        > created
        or not _tick_aligned(invalidation, runner_protocol.tick_size)
        or not _tick_aligned(target, runner_protocol.tick_size)
    ):
        raise Phase8RunnerError("TradeIntent stop/target was not causally frozen on tick")

    if (
        arrival.decision_time != created
        or arrival.symbol != runner_protocol.vendor_symbol
        or arrival.instrument_id != runner_protocol.logical_instrument_id
        or arrival.vendor_instrument_id != runner_protocol.vendor_instrument_id
        or arrival.instrument_mapping_id != runner_protocol.mapping_id
        or arrival.instrument_mapping_sha256 != runner_protocol.mapping_sha256
        or not arrival.book_valid
        or arrival.book is None
    ):
        raise Phase8RunnerError("TradeIntent has no exact causal arrival BBO")
    arrival_mid = (float(arrival.book.bid) + float(arrival.book.ask)) / 2.0
    if (
        (side is Direction.LONG and not invalidation < arrival_mid < target)
        or (side is Direction.SHORT and not target < arrival_mid < invalidation)
    ):
        raise Phase8RunnerError("TradeIntent arrival/stop/target geometry is invalid")

    method_prices = case_record.method_price_set
    if (
        method_prices.source_trade_intent_id != intent_record.source_trade_intent_id
        or method_prices.snapshot_asof != created
        or not math.isclose(
            method_prices.tick_size,
            runner_protocol.tick_size,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise Phase8RunnerError("method-price set is not frozen with its TradeIntent")
    execution_protocol.validate_method_price_set(method_prices)
    for provenance in method_prices.methods:
        if (
            provenance.source_known_at is not None
            and provenance.source_known_at > created
        ):
            raise Phase8RunnerError("method price contains future-known provenance")
        if provenance.availability is MethodAvailability.AVAILABLE and (
            provenance.price is None
            or not _tick_aligned(provenance.price, runner_protocol.tick_size)
        ):
            raise Phase8RunnerError("available method price is missing or off tick")
        if (
            provenance.availability is MethodAvailability.AVAILABLE
            and provenance.source_semantic_type == "causal_arrival_bbo"
            and arrival.input_id not in provenance.derivation_input_ids
        ):
            raise Phase8RunnerError(
                "arrival-BBO method provenance does not bind the exact arrival input"
            )
    market = method_prices.for_method(ExecutionMethod.MARKET)
    if not _method_price_matches_arrival(market, direction=side, arrival=arrival):
        raise Phase8RunnerError("market method price differs from the causal executable quote")
    return expires, analysis_end, arrival_mid


def _ordered_case_records(
    ledger: Phase8AppendOnlyLedger,
) -> tuple[tuple[IntentLedgerRecord, ResearchCaseLedgerRecord], ...]:
    intents = {
        record.record_id: record
        for record in ledger.records
        if isinstance(record, IntentLedgerRecord)
    }
    cases = tuple(
        record
        for record in ledger.records
        if isinstance(record, ResearchCaseLedgerRecord)
    )
    if not intents or len(intents) != len(cases):
        raise Phase8RunnerError(
            "ledger must contain exactly one research case after every intent"
        )
    paired: list[tuple[IntentLedgerRecord, ResearchCaseLedgerRecord]] = []
    used: set[str] = set()
    for case in cases:
        intent = intents.get(case.source_intent_record_id)
        if intent is None or intent.record_id in used:
            raise Phase8RunnerError("research case has no unique exact intent record")
        used.add(intent.record_id)
        paired.append((intent, case))
    if used != set(intents):
        raise Phase8RunnerError("ledger contains an intent without a research case")
    return tuple(
        sorted(
            paired,
            key=lambda pair: (pair[0].created_at, pair[0].source_trade_intent_id),
        )
    )


def evaluate_phase8_research_cases(
    ledger: Phase8AppendOnlyLedger,
    minute_inputs: Sequence[MinuteExecutionInput],
    *,
    runner_protocol: ExecutionResearchRunnerProtocol,
    execution_protocol: Phase8ExecutionResearchProtocol,
    execution_config: ExecutionResearchConfig,
    risk_protocol_id: str,
    risk_protocol_sha256: str,
) -> tuple[dict[str, Any], ...]:
    """Evaluate all fixed primary/OFAT rows from an already verified ledger.

    This function intentionally accepts typed, already-loaded contracts.  The
    formal CLI additionally verifies all file hashes and the full W1/W2 minute
    census before it calls this core.
    """

    if not isinstance(ledger, Phase8AppendOnlyLedger):
        raise TypeError("Phase 8 evaluation requires Phase8AppendOnlyLedger")
    if runner_protocol._loader_seal is not _RUNNER_PROTOCOL_SEAL:
        raise TypeError("Phase 8 evaluation requires a loaded runner protocol")
    if (
        execution_protocol.source_file_sha256
        != EXECUTION_RESEARCH_V2_CONFIG_SHA256
        or execution_config.config_id
        != "execution-research-config:d04f5020aec6d48d66e9c22df574ffde34b2f0e4363753e0041af0a0b1e80701"
        or risk_protocol_sha256 != RISK_ADMISSION_PROTOCOL_SHA256
    ):
        raise Phase8RunnerError("execution/risk protocol identity drifted")
    _identity(risk_protocol_id, name="risk protocol id")
    ordered_inputs = tuple(sorted(minute_inputs, key=lambda item: item.decision_time))
    if (
        not ordered_inputs
        or len({item.input_id for item in ordered_inputs}) != len(ordered_inputs)
        or len({item.decision_time for item in ordered_inputs}) != len(ordered_inputs)
        or any(
            item.symbol != runner_protocol.vendor_symbol
            or item.instrument_id != runner_protocol.logical_instrument_id
            or item.vendor_instrument_id != runner_protocol.vendor_instrument_id
            or item.instrument_mapping_id != runner_protocol.mapping_id
            or item.instrument_mapping_sha256 != runner_protocol.mapping_sha256
            for item in ordered_inputs
        )
    ):
        raise Phase8RunnerError("minute inputs are empty, duplicated, or contract-inconsistent")
    by_clock = {item.decision_time: item for item in ordered_inputs}

    rows: list[dict[str, Any]] = []
    for intent_record, case_record in _ordered_case_records(ledger):
        payload = _intent_payload(intent_record)
        window = runner_protocol.window_for(intent_record.created_at)
        arrival = by_clock.get(intent_record.created_at)
        if arrival is None:
            raise Phase8RunnerError("TradeIntent creation clock is absent from minute inputs")
        expires, analysis_end, arrival_mid = _validate_intent_case_contract(
            intent_record,
            case_record,
            payload=payload,
            arrival=arrival,
            window=window,
            runner_protocol=runner_protocol,
            execution_protocol=execution_protocol,
            risk_protocol_id=risk_protocol_id,
            risk_protocol_sha256=risk_protocol_sha256,
        )
        inputs = tuple(
            item
            for item in ordered_inputs
            if intent_record.created_at <= item.decision_time <= analysis_end
        )
        if not inputs or inputs[0].decision_time != intent_record.created_at:
            raise Phase8RunnerError("case inputs do not start at the exact intent clock")

        for variant in execution_protocol.variants:
            entry_expiry = _variant_entry_expiry(
                variant,
                created_at=intent_record.created_at,
                native_expiry=expires,
                inputs=inputs,
            )
            variant_reason = _evaluability_reason(variant, intent_payload=payload)
            if entry_expiry is None:
                variant_reason = "registered_wait_horizon_missing_real_completed_bars"
                entry_expiry = expires
            method_results: list[dict[str, Any]] = []
            for provenance in case_record.method_price_set.methods:
                method_reason = variant_reason
                if (
                    method_reason is None
                    and variant.variant_id == "target_1r_capped_dol_v1"
                    and provenance.availability is MethodAvailability.AVAILABLE
                ):
                    entry = arrival_mid if provenance.price is None else provenance.price
                    risk = abs(entry - float(payload["invalidation"]["price"]))
                    if risk <= 0.0:
                        method_reason = "one_r_target_has_nonpositive_risk_unit"
                per_method = None
                if (
                    provenance.availability is MethodAvailability.AVAILABLE
                    and method_reason is None
                ):
                    per_method = _method_intent(
                        payload=payload,
                        provenance=provenance,
                        variant=variant,
                        entry_expiry=entry_expiry,
                        analysis_end=analysis_end,
                        arrival_mid=arrival_mid,
                        protocol=runner_protocol,
                        case_id=case_record.record_id,
                    )
                method_results.append(
                    _method_result(
                        provenance,
                        variant_reason=method_reason,
                        intent=per_method,
                        inputs=inputs,
                        execution_config=execution_config,
                    )
                )
            if tuple(item["method"] for item in method_results) != tuple(
                method.value for method in ORDERED_EXECUTION_METHODS
            ):
                raise AssertionError("method output order changed")
            row_payload: dict[str, Any] = {
                "schema_version": PAIRED_ROWS_SCHEMA_VERSION,
                "case_record_id": case_record.record_id,
                "source_intent_record_id": intent_record.record_id,
                "source_trade_intent_id": intent_record.source_trade_intent_id,
                "window_id": window.window_id,
                "created_at": intent_record.created_at.isoformat(),
                "native_entry_expires_at": expires.isoformat(),
                "variant_entry_expires_at": entry_expiry.isoformat(),
                "analysis_ends_at": analysis_end.isoformat(),
                "variant": variant.to_payload(),
                "method_price_set_id": case_record.method_price_set.method_price_set_id,
                "source_artifact_ids": list(case_record.source_artifact_ids),
                "source_input_ids": [item.input_id for item in inputs],
                "method_results": method_results,
                "paired_contrasts": _paired_contrasts(method_results),
            }
            row_payload["row_id"] = f"phase8-paired-row:{_digest(row_payload)}"
            rows.append(row_payload)
    expected = len(_ordered_case_records(ledger)) * len(execution_protocol.variants)
    if len(rows) != expected:
        raise AssertionError("paired row cardinality changed")
    return tuple(rows)


def paired_bootstrap_interval(
    differences: Sequence[float],
    *,
    replicates: int,
    seed: int,
    confidence: float,
    minimum_pairs: int,
) -> tuple[float | None, float | None]:
    """Deterministic percentile CI over same-case paired differences."""

    values = np.asarray(tuple(float(value) for value in differences), dtype=float)
    if (
        values.ndim != 1
        or np.any(~np.isfinite(values))
        or type(replicates) is not int
        or replicates <= 0
        or type(seed) is not int
        or not 0.0 < float(confidence) < 1.0
        or type(minimum_pairs) is not int
        or minimum_pairs <= 0
    ):
        raise Phase8RunnerError("paired bootstrap inputs are invalid")
    if len(values) < minimum_pairs:
        return None, None
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    remaining = replicates
    while remaining:
        size = min(1024, max(1, 2_000_000 // len(values)), remaining)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means.append(values[indices].mean(axis=1))
        remaining -= size
    estimates = np.concatenate(means)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, (alpha, 1.0 - alpha), method="linear")
    return float(lower), float(upper)


def summarize_phase8_paired_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    runner_protocol: ExecutionResearchRunnerProtocol,
    execution_protocol: Phase8ExecutionResearchProtocol,
) -> dict[str, Any]:
    """Summarize fixed same-case contrasts without selecting a winner."""

    typed_rows = tuple(rows)
    if not typed_rows or any(not isinstance(item, Mapping) for item in typed_rows):
        raise Phase8RunnerError("paired rows are empty, duplicated, or invalid")
    row_ids = tuple(item.get("row_id") for item in typed_rows)
    if (
        any(item.get("schema_version") != PAIRED_ROWS_SCHEMA_VERSION for item in typed_rows)
        or len(row_ids) != len(set(row_ids))
        or any(
            item.get("row_id")
            != f"phase8-paired-row:{_digest({key: value for key, value in item.items() if key != 'row_id'})}"
            for item in typed_rows
        )
    ):
        raise Phase8RunnerError("paired rows are empty, duplicated, or invalid")
    scopes: tuple[tuple[str, frozenset[str]], ...] = (
        (
            "pooled_w1_w2",
            frozenset(window.window_id for window in runner_protocol.windows),
        ),
        *tuple(
            (window.window_id, frozenset({window.window_id}))
            for window in runner_protocol.windows
        ),
    )
    estimates: list[dict[str, Any]] = []
    for scope_name, window_ids in scopes:
        for variant in execution_protocol.variants:
            selected = tuple(
                item
                for item in typed_rows
                if item["window_id"] in window_ids
                and item["variant"]["variant_id"] == variant.variant_id
            )
            for method in LIMIT_EXECUTION_METHODS:
                contrasts = tuple(
                    next(
                        contrast
                        for contrast in item["paired_contrasts"]
                        if contrast["method"] == method.value
                    )
                    for item in selected
                )
                differences = tuple(
                    float(item["difference_implementation_shortfall_points"])
                    for item in contrasts
                    if item["complete_pair"]
                )
                reason_counts: Counter[str] = Counter()
                for item in contrasts:
                    reason_counts.update(item["unavailable_reasons"])
                seed_salt = int(
                    hashlib.sha256(
                        f"{scope_name}|{variant.variant_id}|{method.value}".encode("utf-8")
                    ).hexdigest()[:16],
                    16,
                )
                lower, upper = paired_bootstrap_interval(
                    differences,
                    replicates=runner_protocol.bootstrap_replicates,
                    seed=(runner_protocol.bootstrap_seed ^ seed_salt) % (2**63 - 1),
                    confidence=runner_protocol.bootstrap_confidence,
                    minimum_pairs=runner_protocol.minimum_complete_pairs,
                )
                estimates.append(
                    {
                        "scope": scope_name,
                        "variant_id": variant.variant_id,
                        "method": method.value,
                        "comparator": ExecutionMethod.MARKET.value,
                        "available_cases": len(selected),
                        "complete_pairs": len(differences),
                        "mean_difference_implementation_shortfall_points": (
                            None if not differences else float(np.mean(differences))
                        ),
                        "bootstrap_ci_lower": lower,
                        "bootstrap_ci_upper": upper,
                        "bootstrap_status": (
                            "estimated"
                            if lower is not None
                            else "insufficient_complete_pairs"
                        ),
                        "incomplete_pair_reason_counts": dict(sorted(reason_counts.items())),
                    }
                )
    method_status_counts: Counter[str] = Counter()
    terminal_status_counts: Counter[str] = Counter()
    method_censor_reason_counts: Counter[str] = Counter()
    for row in typed_rows:
        for result in row["method_results"]:
            method_status_counts[
                f"{result['method']}:{result['evaluation_status']}"
            ] += 1
            if result["outcome"] is not None:
                terminal_status_counts[
                    f"{result['method']}:{result['outcome']['terminal_status']}"
                ] += 1
            for reason in result["censor_reasons"]:
                method_censor_reason_counts[f"{result['method']}:{reason}"] += 1
    payload: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "development_descriptive_not_oos_not_trading_authority",
        "case_count": len({item["case_record_id"] for item in typed_rows}),
        "paired_row_count": len(typed_rows),
        "variant_count": len(execution_protocol.variants),
        "method_count": len(ORDERED_EXECUTION_METHODS),
        "bootstrap": {
            "kind": "deterministic_paired_case_percentile",
            "replicates": runner_protocol.bootstrap_replicates,
            "base_seed": runner_protocol.bootstrap_seed,
            "contrast_seed_rule": (
                "base_seed_xor_first_64_bits_sha256_scope_pipe_variant_pipe_method_"
                "mod_2pow63_minus1"
            ),
            "confidence": runner_protocol.bootstrap_confidence,
            "minimum_complete_pairs": runner_protocol.minimum_complete_pairs,
        },
        "method_evaluation_status_counts": dict(sorted(method_status_counts.items())),
        "method_terminal_status_counts": dict(sorted(terminal_status_counts.items())),
        "method_censor_reason_counts": dict(sorted(method_censor_reason_counts.items())),
        "estimates": estimates,
        "source_row_ids": list(row_ids),
    }
    payload["summary_id"] = f"phase8-paired-summary:{_digest(payload)}"
    return payload


def load_phase8_intent_case_ledger(
    path: str | Path,
    *,
    expected_sha256: str,
) -> Phase8AppendOnlyLedger:
    """Load exact canonical append-only JSONL without accepting rewrites."""

    source = Path(path)
    _reject_sealed_path(source, name="intent/research-case ledger")
    expected = _sha(expected_sha256, name="intent/research-case ledger SHA-256")
    if source.is_symlink() or not source.is_file() or _file_sha(source) != expected:
        raise Phase8RunnerError("intent/research-case ledger identity mismatch")
    try:
        text_value = source.read_bytes().decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise Phase8RunnerError("intent/research-case ledger is not UTF-8") from exc
    ledger = Phase8AppendOnlyLedger.from_jsonl(text_value)
    if not text_value or ledger.to_jsonl() != text_value:
        raise Phase8RunnerError("intent/research-case ledger is not canonical JSONL")
    _ordered_case_records(ledger)
    return ledger


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise Phase8RunnerError("artifact path is outside the repository") from exc


def _canonical_json_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _canonical_jsonl_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    if not values:
        raise Phase8RunnerError("append-only JSONL output cannot be empty")
    return "".join(_canonical_json(value) + "\n" for value in values).encode("utf-8")


def _validate_output_payloads(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    expected_variants = (
        "primary_v1",
        "wait_1m_v1",
        "wait_5m_v1",
        "cancel_gtt_only_v1",
        "cancel_price_terminal_only_v1",
        "stop_entry_zone_failure_v1",
        "target_1r_capped_dol_v1",
    )
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    row_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise Phase8RunnerError("paired output row is not an object")
        row_id = row.get("row_id")
        expected_id = f"phase8-paired-row:{_digest({key: value for key, value in row.items() if key != 'row_id'})}"
        method_results = row.get("method_results")
        if (
            row.get("schema_version") != PAIRED_ROWS_SCHEMA_VERSION
            or row_id != expected_id
            or not isinstance(method_results, list)
            or tuple(item.get("method") for item in method_results)
            != tuple(method.value for method in ORDERED_EXECUTION_METHODS)
        ):
            raise Phase8RunnerError("paired output row identity/method registry is invalid")
        row_ids.append(str(row_id))
        by_case.setdefault(str(row.get("case_record_id")), []).append(row)
    if not by_case or len(row_ids) != len(set(row_ids)):
        raise Phase8RunnerError("paired output rows are empty or duplicated")
    for case_rows in by_case.values():
        if (
            tuple(item["variant"]["variant_id"] for item in case_rows)
            != expected_variants
            or len({tuple(item["source_input_ids"]) for item in case_rows}) != 1
        ):
            raise Phase8RunnerError("case rows do not contain the exact same-input OFAT registry")
    if not isinstance(summary, Mapping):
        raise Phase8RunnerError("paired summary is not an object")
    expected_summary_id = (
        "phase8-paired-summary:"
        + _digest({key: value for key, value in summary.items() if key != "summary_id"})
    )
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA_VERSION
        or summary.get("summary_id") != expected_summary_id
        or summary.get("source_row_ids") != row_ids
        or summary.get("paired_row_count") != len(row_ids)
        or summary.get("case_count") != len(by_case)
    ):
        raise Phase8RunnerError("paired summary identity/source rows are invalid")


def _publish_no_clobber(path: Path, content: bytes) -> str:
    """Publish immutable bytes with an atomic hard-link create operation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise Phase8RunnerError(f"output already exists and cannot be overwritten: {path}")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, path)
    except FileExistsError as exc:
        raise Phase8RunnerError(
            f"output already exists and cannot be overwritten: {path}"
        ) from exc
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    digest = hashlib.sha256(content).hexdigest()
    if _file_sha(path) != digest:
        raise Phase8RunnerError("published output failed its content identity check")
    return digest


def _source_identity_payload(
    contract: Phase8RunManifestContract,
    protocol: ExecutionResearchRunnerProtocol,
) -> dict[str, Any]:
    mapping_identity = {
        "mapping_id": protocol.mapping_id,
        "mapping_sha256": protocol.mapping_sha256,
        "logical_instrument_id": protocol.logical_instrument_id,
        "vendor_instrument_id": protocol.vendor_instrument_id,
        "vendor_symbol": protocol.vendor_symbol,
        "registry_path": protocol.payload["instrument_mapping"]["registry_path"],
        "registry_sha256": protocol.payload["instrument_mapping"]["registry_sha256"],
    }
    if contract.source_mode is SourceMode.REGISTERED_PHASE6:
        return {
            "source_mode": SourceMode.REGISTERED_PHASE6.value,
            "instrument_mapping": mapping_identity,
            "shared_ohlcv": {
                "path": protocol.ohlcv_path,
                "sha256": protocol.ohlcv_sha256,
                "manifest_path": protocol.ohlcv_manifest_path,
                "manifest_sha256": protocol.ohlcv_manifest_sha256,
            },
            "windows": [
                {
                    "window_id": window.window_id,
                    "phase6_manifest_path": window.phase6_manifest_path,
                    "phase6_manifest_sha256": window.phase6_manifest_sha256,
                    "mbo_artifact_path": window.mbo_artifact_path,
                    "mbo_artifact_sha256": window.mbo_artifact_sha256,
                    "mbo_manifest_path": window.mbo_manifest_path,
                    "mbo_manifest_sha256": window.mbo_manifest_sha256,
                }
                for window in protocol.windows
            ],
        }
    if contract.source_mode is SourceMode.MINUTE_ARTIFACT:
        assert contract.minute_artifact_path is not None
        assert contract.minute_artifact_sha256 is not None
        assert contract.minute_manifest_path is not None
        assert contract.minute_manifest_sha256 is not None
        return {
            "source_mode": SourceMode.MINUTE_ARTIFACT.value,
            "instrument_mapping": mapping_identity,
            "minute_execution_artifact": {
                "path": str(contract.input_bindings["minute_execution_artifact"]["path"]),
                "sha256": contract.minute_artifact_sha256,
            },
            "minute_execution_manifest": {
                "path": str(contract.input_bindings["minute_execution_manifest"]["path"]),
                "sha256": contract.minute_manifest_sha256,
            },
        }
    raise Phase8RunnerError("run manifest did not select a minute source mode")


@dataclass(frozen=True)
class Phase8RunResult:
    experiment_id: str
    paired_rows_path: Path
    summary_path: Path
    output_manifest_path: Path
    paired_rows_sha256: str
    summary_sha256: str
    output_manifest_sha256: str
    case_count: int
    paired_row_count: int


def write_phase8_research_outputs(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    contract: Phase8RunManifestContract,
    runner_protocol: ExecutionResearchRunnerProtocol,
    project_root: str | Path,
) -> Phase8RunResult:
    """Write rows, summary, then manifest; every target is create-only."""

    if (
        not contract.ready
        or contract.status
        != "frozen_2024_06_development_research_not_oos_not_trading_authority"
        or contract.experiment_id is None
        or contract.frozen_at is None
        or contract.ledger_sha256 is None
        or contract.source_mode is None
    ):
        raise Phase8RunnerError("only a frozen ready run manifest may write outputs")
    root = Path(project_root).resolve()
    outputs = contract.output_paths
    paired_path = outputs.get("paired_rows")
    summary_path = outputs.get("summary")
    manifest_path = outputs.get("output_manifest")
    if paired_path is None or summary_path is None or manifest_path is None:
        raise Phase8RunnerError("all three output paths must be registered")
    if paired_path.suffix != ".jsonl" or summary_path.suffix != ".json" or manifest_path.suffix != ".json":
        raise Phase8RunnerError("Phase 8 outputs require JSONL rows and JSON summary/manifest")
    for path in (paired_path, summary_path, manifest_path):
        if os.path.lexists(path):
            raise Phase8RunnerError(f"output already exists and cannot be overwritten: {path}")

    row_values = tuple(rows)
    _validate_output_payloads(row_values, summary)
    row_bytes = _canonical_jsonl_bytes(row_values)
    summary_bytes = _canonical_json_bytes(summary)
    rows_sha = hashlib.sha256(row_bytes).hexdigest()
    summary_sha = hashlib.sha256(summary_bytes).hexdigest()
    case_count = len({item["case_record_id"] for item in row_values})
    manifest_payload: dict[str, Any] = {
        "schema_version": OUTPUT_MANIFEST_SCHEMA_VERSION,
        "status": "development_completed_not_oos_not_trading_authority",
        "authority": {
            "research_only": True,
            "order_submission": False,
            "production_engine_integration": False,
            "sealed_oos_reveal": False,
        },
        "experiment_id": contract.experiment_id,
        "frozen_at": contract.frozen_at.isoformat(),
        "run_manifest": {
            "path": _relative_path(root, contract.source_path),
            "sha256": contract.manifest_sha256,
        },
        "intent_research_case_ledger": {
            "path": str(contract.input_bindings["intent_research_case_ledger"]["path"]),
            "sha256": contract.ledger_sha256,
        },
        "runner_protocol": {
            "config_sha256": runner_protocol.config_sha256,
            "canonical_config_sha256": runner_protocol.canonical_config_sha256,
            "execution_protocols": runner_protocol.payload["execution_protocols"],
        },
        "source_inputs": _source_identity_payload(contract, runner_protocol),
        "counts": {
            "cases": case_count,
            "paired_rows": len(row_values),
            "variants_per_case": 7,
            "methods_per_variant": 7,
        },
        "outputs": {
            "paired_rows": {
                "path": _relative_path(root, paired_path),
                "sha256": rows_sha,
                "rows": len(row_values),
                "schema_version": PAIRED_ROWS_SCHEMA_VERSION,
            },
            "summary": {
                "path": _relative_path(root, summary_path),
                "sha256": summary_sha,
                "schema_version": SUMMARY_SCHEMA_VERSION,
            },
        },
        "manifest_written_last": True,
    }
    manifest_payload["manifest_id"] = (
        f"phase8-output-manifest:{_digest(manifest_payload)}"
    )
    manifest_bytes = _canonical_json_bytes(manifest_payload)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    # Precomputed bytes are published in this order by protocol.  If a process
    # is interrupted, the absence of the final manifest proves incompleteness.
    _publish_no_clobber(paired_path, row_bytes)
    _publish_no_clobber(summary_path, summary_bytes)
    _publish_no_clobber(manifest_path, manifest_bytes)
    return Phase8RunResult(
        experiment_id=contract.experiment_id,
        paired_rows_path=paired_path,
        summary_path=summary_path,
        output_manifest_path=manifest_path,
        paired_rows_sha256=rows_sha,
        summary_sha256=summary_sha,
        output_manifest_sha256=manifest_sha,
        case_count=case_count,
        paired_row_count=len(row_values),
    )


def run_phase8_execution_research(
    manifest_path: str | Path,
    *,
    project_root: str | Path,
) -> Phase8RunResult:
    """Run the exact open-development W1/W2 contract after manifest freeze."""

    root = Path(project_root).resolve()
    contract = validate_phase8_run_manifest(
        manifest_path,
        project_root=root,
        validate_input_files=True,
    )
    if not contract.ready:
        raise Phase8RunnerError(
            "run manifest is blocked: " + ",".join(contract.blockers)
        )
    runner_protocol = load_execution_research_runner_protocol(
        root / _RUNTIME_BINDING_PATHS["runner_config"],
        expected_sha256=RUNNER_CONFIG_SHA256,
    )
    execution_config = load_execution_research_config(
        root / _RUNTIME_BINDING_PATHS["execution_v1_config"],
        expected_sha256="8212939f9dc00b11c285063a78d56f8ddbc5257a728b02829017a6d60e0d08b7",
    )
    execution_protocol = load_execution_research_v2_config(
        root / _RUNTIME_BINDING_PATHS["execution_v2_config"],
        expected_sha256=EXECUTION_RESEARCH_V2_CONFIG_SHA256,
    )
    risk_protocol = load_risk_admission_protocol(
        root / _RUNTIME_BINDING_PATHS["risk_config"],
        expected_sha256=RISK_ADMISSION_PROTOCOL_SHA256,
    )
    assert contract.ledger_path is not None and contract.ledger_sha256 is not None
    ledger = load_phase8_intent_case_ledger(
        contract.ledger_path,
        expected_sha256=contract.ledger_sha256,
    )
    if contract.source_mode is SourceMode.REGISTERED_PHASE6:
        minute_inputs = load_registered_phase6_minute_inputs(root, runner_protocol)
    elif contract.source_mode is SourceMode.MINUTE_ARTIFACT:
        assert contract.minute_artifact_path is not None
        assert contract.minute_artifact_sha256 is not None
        assert contract.minute_manifest_path is not None
        assert contract.minute_manifest_sha256 is not None
        minute_inputs = load_minute_execution_artifact(
            contract.minute_artifact_path,
            contract.minute_manifest_path,
            expected_artifact_sha256=contract.minute_artifact_sha256,
            expected_manifest_sha256=contract.minute_manifest_sha256,
            protocol=runner_protocol,
        )
    else:
        raise Phase8RunnerError("run manifest did not select a minute source mode")
    rows = evaluate_phase8_research_cases(
        ledger,
        minute_inputs,
        runner_protocol=runner_protocol,
        execution_protocol=execution_protocol,
        execution_config=execution_config,
        risk_protocol_id=risk_protocol.protocol_id,
        risk_protocol_sha256=risk_protocol.source_file_sha256,
    )
    summary = summarize_phase8_paired_rows(
        rows,
        runner_protocol=runner_protocol,
        execution_protocol=execution_protocol,
    )
    return write_phase8_research_outputs(
        rows,
        summary,
        contract=contract,
        runner_protocol=runner_protocol,
        project_root=root,
    )


__all__ = [
    "EXECUTION_RUNNER_SCHEMA_VERSION",
    "MINUTE_ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "OUTPUT_MANIFEST_SCHEMA_VERSION",
    "PAIRED_ROWS_SCHEMA_VERSION",
    "RUNNER_CONFIG_SHA256",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "ExecutionResearchRunnerProtocol",
    "MethodEvaluationStatus",
    "Phase8RunManifestContract",
    "Phase8RunResult",
    "Phase8RunnerError",
    "RegisteredExecutionWindow",
    "SourceMode",
    "evaluate_phase8_research_cases",
    "load_execution_research_runner_protocol",
    "load_minute_execution_artifact",
    "load_phase8_intent_case_ledger",
    "load_registered_phase6_minute_inputs",
    "minute_execution_input_from_payload",
    "minute_execution_input_to_payload",
    "paired_bootstrap_interval",
    "run_phase8_execution_research",
    "summarize_phase8_paired_rows",
    "validate_phase8_run_manifest",
    "write_phase8_research_outputs",
]
