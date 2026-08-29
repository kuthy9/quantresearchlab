#!/usr/bin/env python3
"""Materialize and, when separately authorized, fit Phase-7 research cohorts.

This runner consumes only hash-bound, already-opened June 2024
``ShadowClockInput`` JSONL files.  It replays the production development
Engine one completed clock at a time and projects immutable research cohorts
from the resulting ``MarketBelief`` and the normalized input BAR.  It never
discovers a data source, opens sealed OOS data, edits a production config, or
grants action authority.

The frozen Phase-7 preregistration is a design receipt and intentionally does
not authorize execution.  A separate run manifest is therefore required for
every materialization.  Fitting additionally requires an explicit execution
authorization in that run manifest.  Every output is written into a new
directory; overwrite and resume semantics are deliberately absent.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat as stat_module
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.engine import ContinuousSMCEngine  # noqa: E402
from smc_trader.model import EngineSnapshot, NeutralEngineSnapshot, Timeframe  # noqa: E402
from smc_trader.path_belief import (  # noqa: E402
    PATH_KINDS,
    PathCompetitionSetState,
    PathKind,
    PathStatus,
    load_path_belief_protocol,
)
from smc_trader.probability_admission import (  # noqa: E402
    AdmissionThresholds,
    PROBABILITY_ADMISSION_SCHEMA_VERSION,
    ProbabilityPrediction,
    dol_support_label,
    evaluate_probability_admission,
)
from smc_trader.probability_cohorts import (  # noqa: E402
    DOLCandidateOutcomeRow,
    DOLCandidateSnapshot,
    EvidenceHistoryTransition,
    NO_TARGET_OUTCOME,
    PATH_LABELS,
    PathCompetitionArchive,
    PathRiskClock,
    PathRiskInterval,
    PathTerminalDisposition,
    build_path_risk_intervals,
    canonical_identity,
    canonical_sha256,
    derive_dol_candidate_hits,
    label_dol_candidate_set,
)
from smc_trader.probability_fit import (  # noqa: E402
    DOLChoiceSetSample,
    EvidenceContributionAtPrediction,
    HistoryConditionalLikelihoodArtifact,
    PathProbabilitySample,
    PriorReversionSample,
    fit_competing_risk_life_table,
    fit_dol_softmax,
    fit_evidence_prior_reversion,
    fit_history_conditional_likelihood,
    fit_path_temperature_bias,
)
from smc_trader.signal_outcome_fit import (  # noqa: E402
    DOLModelLineage,
    DOLSupportLabel,
    DOLTemperatureObservation,
    SIGNAL_OUTCOME_ADMISSION_SCHEMA,
    SignalOutcomeAdmissionThresholds,
    SignalOutcomeCohort,
    SignalOutcomeFitError,
    evaluate_dol_temperature_admission,
    fit_dol_temperature,
)
from smc_trader.shadow_live import ShadowClockInput  # noqa: E402
from smc_trader.semantics import load_semantic_selection  # noqa: E402
from smc_trader.structural_outcome import OutcomeBar  # noqa: E402
from scripts.run_shadow_file_pilot import (  # noqa: E402
    INPUT_SCHEMA_VERSION as SHADOW_INPUT_SCHEMA_VERSION,
    iter_shadow_clock_file,
)


PIPELINE_SCHEMA_VERSION = "phase7_empirical_pipeline_v2"
RUN_MANIFEST_SCHEMA_VERSION = "phase7_empirical_run_manifest_v1"
AUTHORITY = "research_shadow_only"
_ADMISSION_STRING_SCHEMA_VERSIONS = frozenset(
    {
        SIGNAL_OUTCOME_ADMISSION_SCHEMA,
        "phase7_admission_not_evaluable_v1",
        "phase7_signal_outcome_not_evaluable_v1",
    }
)
MODEL_CONFIG = ROOT / "configs/model.json"
PHASE7_PROTOCOL = ROOT / "configs/phase7_foundation_v2_empirical.json"
PREREGISTRATION = (
    ROOT
    / "experiments/manifests/phase7_foundation_v2_empirical_preregistration.yaml"
)
FOUNDATION_IDENTITY = (
    "0c49da28e103f0515d3eb93ab03e8659e334d2477449f5174df3b5e8b0b84cc6"
)
PATH_PROTOCOL_FINGERPRINT = (
    "d897635cb91fc1cb84647f41eb4c5a6b8a06d05d2e44a1b3471e8e9ff67b0482"
)
DOL_PROTOCOL_FINGERPRINT = (
    "a7f588d35c380d4d9b38a2d5c3aa90ee5e30c49c241696f0d2847fb90b7ad7fe"
)
DOL_RANKING_FINGERPRINT = (
    "be9177601fbdac6e36072dbf7dd8bde46c793e7197d0a9e3662f835ceebbb4be"
)
# Selected critical runtime-file index for receipt readability.  This is not
# an import closure; repository_commit plus the clean tracked-tree gate binds
# the complete tracked runtime closure.
CODE_BUNDLE_FILES: Mapping[str, Path] = {
    "runner": ROOT / "scripts/run_phase7_empirical_pipeline.py",
    "engine": ROOT / "smc_trader/engine.py",
    "playbooks": ROOT / "smc_trader/playbooks.py",
    "path_belief": ROOT / "smc_trader/path_belief.py",
    "probability_cohorts": ROOT / "smc_trader/probability_cohorts.py",
    "probability_fit": ROOT / "smc_trader/probability_fit.py",
    "probability_admission": ROOT / "smc_trader/probability_admission.py",
    "signal_outcome_fit": ROOT / "smc_trader/signal_outcome_fit.py",
    "model_config": MODEL_CONFIG,
    "uv_lock": ROOT / "uv.lock",
}

_RESET_ANOMALIES = frozenset(
    {
        "contract_change_history_reset",
        "data_gap_history_reset",
        "semantic_reset",
        "data_reset",
    }
)
_EVIDENCE_RULES = frozenset(
    {"acceptance_continuation", "displacement_impact"}
)
_PREDICTION_AGE_BOUNDARIES = frozenset({1, 5, 15, 30, 60, 120, 240})
_FIT_ROLES = frozenset({"development_fit", "development_cross_fit", "calibration"})
_HISTORY_TRANSITION_RULE_IDS = {
    ("none", "acceptance_only"): "history_none_to_acceptance_only",
    ("none", "displacement_only"): "history_none_to_displacement_only",
    ("none", "same_clock_joint"): "history_none_to_same_clock_joint",
    (
        "acceptance_only",
        "acceptance_then_displacement",
    ): "history_acceptance_only_to_acceptance_then_displacement",
    (
        "displacement_only",
        "displacement_then_acceptance",
    ): "history_displacement_only_to_displacement_then_acceptance",
}


@dataclass(frozen=True)
class RegisteredWindow:
    window_id: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    expected_rows: int
    split_role: str
    purpose: str
    rolling_diagnostic_only: bool
    allowed_contracts: tuple[tuple[str, int], ...]

    @property
    def first_decision_clock(self) -> pd.Timestamp:
        return self.start + pd.Timedelta(1, unit="min")

    @property
    def last_decision_clock(self) -> pd.Timestamp:
        return self.end_exclusive - pd.Timedelta(1, unit="min")


REGISTERED_WINDOWS: Mapping[str, RegisteredWindow] = {
    "W1": RegisteredWindow(
        window_id="W1",
        start=pd.Timestamp("2024-06-02T22:00:00Z"),
        end_exclusive=pd.Timestamp("2024-06-07T21:01:00Z"),
        expected_rows=6900,
        split_role="development_fit",
        purpose="fit",
        rolling_diagnostic_only=False,
        allowed_contracts=(("NQM4", 13743),),
    ),
    "W2": RegisteredWindow(
        window_id="W2",
        start=pd.Timestamp("2024-06-09T22:00:00Z"),
        end_exclusive=pd.Timestamp("2024-06-14T21:01:00Z"),
        expected_rows=6900,
        split_role="historical_validation",
        purpose="validation",
        rolling_diagnostic_only=False,
        allowed_contracts=(("NQM4", 13743),),
    ),
    "W3": RegisteredWindow(
        window_id="W3",
        start=pd.Timestamp("2024-06-16T22:00:00Z"),
        end_exclusive=pd.Timestamp("2024-06-21T21:01:00Z"),
        expected_rows=6660,
        split_role="development_cross_fit",
        purpose="rolling_diagnostic",
        rolling_diagnostic_only=True,
        allowed_contracts=(("NQM4", 13743), ("NQU4", 4358)),
    ),
    "W4": RegisteredWindow(
        window_id="W4",
        start=pd.Timestamp("2024-06-23T22:00:00Z"),
        end_exclusive=pd.Timestamp("2024-06-28T21:01:00Z"),
        expected_rows=6900,
        split_role="development_cross_fit",
        purpose="rolling_diagnostic",
        rolling_diagnostic_only=True,
        allowed_contracts=(("NQU4", 4358),),
    ),
}


class Phase7PipelineError(ValueError):
    """Raised when the empirical pipeline cannot preserve its frozen contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, name: str) -> str:
    text = str(value).strip()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise Phase7PipelineError(f"{name} must be a lowercase SHA-256")
    return text


def _aware(value: Any, *, name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise Phase7PipelineError(f"{name} is not a timestamp") from error
    if result.tzinfo is None:
        raise Phase7PipelineError(f"{name} must be timezone aware")
    return result


def _resolve(path: str | Path) -> Path:
    source = Path(path)
    return source if source.is_absolute() else ROOT / source


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _trusted_regular_file(path: Path, *, name: str) -> Path:
    """Return one direct regular file without following any path indirection."""

    lexical = _lexical_absolute(path)
    error_message = f"{name} must be a direct regular file"
    current = Path(lexical.anchor)
    try:
        metadata = os.lstat(current)
        if stat_module.S_ISLNK(metadata.st_mode) or not stat_module.S_ISDIR(
            metadata.st_mode
        ):
            raise Phase7PipelineError(error_message)
        parts = lexical.parts[1:]
        for index, part in enumerate(parts):
            current /= part
            metadata = os.lstat(current)
            if stat_module.S_ISLNK(metadata.st_mode):
                raise Phase7PipelineError(error_message)
            final = index == len(parts) - 1
            if final:
                if not stat_module.S_ISREG(metadata.st_mode):
                    raise Phase7PipelineError(error_message)
            elif not stat_module.S_ISDIR(metadata.st_mode):
                raise Phase7PipelineError(error_message)
    except OSError as error:
        raise Phase7PipelineError(error_message) from error
    if not parts:
        raise Phase7PipelineError(error_message)
    return lexical


def _assert_opened_jsonl_path(path: Path, *, name: str) -> None:
    lowered = "/".join(_lexical_absolute(path).parts).lower()
    if path.suffix != ".jsonl" or any(
        token in lowered for token in ("sealed", "holdout", "2026-04")
    ):
        raise Phase7PipelineError(f"{name} path is not an opened JSONL source")


def _repository_identity() -> str:
    """Return exact clean HEAD or fail before any empirical replay begins."""

    try:
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise Phase7PipelineError("repository identity is unavailable") from error
    commit = head.stdout.strip() if isinstance(head.stdout, str) else ""
    dirty = status.stdout if isinstance(status.stdout, str) else ""
    if (
        head.returncode != 0
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise Phase7PipelineError("repository HEAD is invalid")
    if status.returncode != 0 or dirty:
        raise Phase7PipelineError("repository tracked tree must be clean")
    return commit


def _assert_code_bundle_committed() -> None:
    relative_paths = tuple(
        sorted(
            path.relative_to(ROOT).as_posix()
            for path in {
                *CODE_BUNDLE_FILES.values(),
                PHASE7_PROTOCOL,
                PREREGISTRATION,
            }
        )
    )
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", *relative_paths],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise Phase7PipelineError("code bundle commit membership is unavailable") from error
    tracked = tuple(sorted(line.strip() for line in result.stdout.splitlines() if line.strip()))
    if result.returncode != 0 or tracked != relative_paths:
        raise Phase7PipelineError("every runtime binding must be committed at HEAD")


def current_code_bundle_binding(*, require_clean_repository: bool = True) -> dict[str, Any]:
    """Build the exact code binding for a post-commit run manifest.

    The returned object is data only; it does not authorize a run.  The normal
    path requires a clean committed tree.  Tests may opt out of the repository
    check while still exercising exact file and runtime identities.
    """

    if require_clean_repository:
        repository_commit = _repository_identity()
        _assert_code_bundle_committed()
    else:
        repository_commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    files = {
        name: {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256_file(path),
        }
        for name, path in CODE_BUNDLE_FILES.items()
    }
    return {
        "repository_commit": repository_commit,
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "files": files,
    }


def _validate_code_bundle(value: Any) -> tuple[str, str, str]:
    binding = _exact_mapping(
        value,
        {"repository_commit", "python_major_minor", "files"},
        name="bindings.code_bundle",
    )
    commit = str(binding["repository_commit"]).strip()
    python_identity = str(binding["python_major_minor"]).strip()
    expected_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if python_identity != expected_python:
        raise Phase7PipelineError("bindings.code_bundle Python runtime differs")
    files = _exact_mapping(
        binding["files"], CODE_BUNDLE_FILES, name="bindings.code_bundle.files"
    )
    normalized_files: dict[str, Any] = {}
    for name, expected_path in CODE_BUNDLE_FILES.items():
        item = _exact_mapping(
            files[name], {"path", "sha256"}, name=f"code bundle file {name}"
        )
        path = _resolve(item["path"])
        digest = _sha256(item["sha256"], name=f"code bundle file {name} sha256")
        if path.resolve() != expected_path.resolve():
            raise Phase7PipelineError(f"code bundle file {name} path differs")
        if not path.is_file() or _sha256_file(path) != digest:
            raise Phase7PipelineError(f"code bundle file {name} hash differs")
        normalized_files[name] = {
            "path": expected_path.relative_to(ROOT).as_posix(),
            "sha256": digest,
        }
    if _repository_identity() != commit:
        raise Phase7PipelineError("bindings.code_bundle repository commit differs")
    _assert_code_bundle_committed()
    identity = canonical_sha256(
        {
            "repository_commit": commit,
            "python_major_minor": python_identity,
            "files": normalized_files,
        }
    )
    return commit, python_identity, identity


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise Phase7PipelineError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def _reject_nonfinite_json(value: str) -> Any:
    raise Phase7PipelineError(f"non-finite JSON constant: {value}")


def _load_json_with_sha256(path: Path) -> tuple[Mapping[str, Any], str]:
    """Parse and hash the same immutable byte snapshot."""

    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase7PipelineError(f"JSON is unreadable: {path}") from error
    if not isinstance(payload, Mapping):
        raise Phase7PipelineError(f"JSON root must be an object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    return _load_json_with_sha256(path)[0]


def _exact_mapping(value: Any, fields: Iterable[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Phase7PipelineError(f"{name} must be an object")
    expected = set(fields)
    actual = set(value)
    if actual != expected:
        raise Phase7PipelineError(
            f"{name} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return dict(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or value == -math.inf:
            raise Phase7PipelineError("output contains a non-finite number")
        return "infinity" if value == math.inf else value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported output value: {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Any]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


@dataclass(frozen=True)
class InputWindowSpec:
    registered: RegisteredWindow
    path: Path
    source_sha256: str
    row_count: int
    fold_id: str

    @property
    def window_id(self) -> str:
        return self.registered.window_id

    @property
    def split_role(self) -> str:
        return self.registered.split_role


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    source_path: Path
    source_sha256: str
    model_config_path: Path
    model_config_sha256: str
    phase7_protocol_sha256: str
    preregistration_sha256: str
    repository_commit: str
    python_major_minor: str
    code_bundle_identity: str
    materialization_authorized: bool
    fit_and_validate_authorized: bool
    inputs: tuple[InputWindowSpec, ...]


def load_run_manifest(path: str | Path) -> RunManifest:
    """Load and verify one exact execution manifest and its opened inputs."""

    source = _trusted_regular_file(
        _resolve(path), name="Phase-7 run manifest"
    )
    manifest_payload, manifest_sha256 = _load_json_with_sha256(source)
    payload = _exact_mapping(
        manifest_payload,
        {
            "schema_version",
            "run_id",
            "authority",
            "bindings",
            "inputs",
        },
        name="Phase-7 run manifest",
    )
    if payload["schema_version"] != RUN_MANIFEST_SCHEMA_VERSION:
        raise Phase7PipelineError("Phase-7 run manifest schema differs")
    if not isinstance(payload["run_id"], str) or not payload["run_id"].strip():
        raise Phase7PipelineError("run_id is required")
    authority = _exact_mapping(
        payload["authority"],
        {
            "cohort_materialization_authorized",
            "fit_and_validate_authorized",
            "sealed_source_read_authorized",
            "sealed_oos_opened",
            "action_authority",
        },
        name="run authority",
    )
    if authority["cohort_materialization_authorized"] is not True:
        raise Phase7PipelineError("cohort materialization lacks explicit authority")
    for forbidden in (
        "sealed_source_read_authorized",
        "sealed_oos_opened",
        "action_authority",
    ):
        if authority[forbidden] is not False:
            raise Phase7PipelineError(f"Phase-7 runner requires {forbidden}=false")
    if type(authority["fit_and_validate_authorized"]) is not bool:
        raise Phase7PipelineError("fit_and_validate_authorized must be boolean")

    # Reject forbidden source names before validating or hashing any runtime or
    # input binding.  A manifest cannot use code-bundle failure as an oracle
    # for a sealed path, and the runner never discovers sealed data.
    raw_inputs = payload["inputs"]
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise Phase7PipelineError("run manifest requires at least one input")
    for index, raw in enumerate(raw_inputs):
        if (
            not isinstance(raw, Mapping)
            or "path" not in raw
            or not isinstance(raw["path"], str)
            or not raw["path"].strip()
        ):
            raise Phase7PipelineError(f"inputs[{index}] path is required")
        window_label = str(raw.get("window_id", f"inputs[{index}]"))
        guarded_path = _resolve(raw["path"])
        _assert_opened_jsonl_path(guarded_path, name=window_label)
        guarded_path = _trusted_regular_file(
            guarded_path, name=f"{window_label} input"
        )
        _assert_opened_jsonl_path(guarded_path, name=window_label)

    bindings = _exact_mapping(
        payload["bindings"],
        {
            "model_config",
            "phase7_protocol",
            "preregistration_manifest",
            "foundation_identity",
            "path_protocol_fingerprint",
            "dol_protocol_fingerprint",
            "dol_ranking_fingerprint",
            "code_bundle",
        },
        name="run bindings",
    )

    def bound_file(name: str, expected_default: Path) -> tuple[Path, str]:
        binding = _exact_mapping(
            bindings[name], {"path", "sha256"}, name=f"bindings.{name}"
        )
        resolved = _resolve(binding["path"])
        expected_sha = _sha256(binding["sha256"], name=f"bindings.{name}.sha256")
        if resolved.resolve() != expected_default.resolve():
            raise Phase7PipelineError(f"bindings.{name}.path is not the frozen file")
        if not resolved.is_file() or _sha256_file(resolved) != expected_sha:
            raise Phase7PipelineError(f"bindings.{name} hash differs")
        return resolved, expected_sha

    model_path, model_sha = bound_file("model_config", MODEL_CONFIG)
    _, protocol_sha = bound_file("phase7_protocol", PHASE7_PROTOCOL)
    _, prereg_sha = bound_file("preregistration_manifest", PREREGISTRATION)
    repository_commit, python_identity, code_bundle_identity = _validate_code_bundle(
        bindings["code_bundle"]
    )
    exact_identities = {
        "foundation_identity": FOUNDATION_IDENTITY,
        "path_protocol_fingerprint": PATH_PROTOCOL_FINGERPRINT,
        "dol_protocol_fingerprint": DOL_PROTOCOL_FINGERPRINT,
        "dol_ranking_fingerprint": DOL_RANKING_FINGERPRINT,
    }
    for key, expected in exact_identities.items():
        if bindings[key] != expected:
            raise Phase7PipelineError(f"bindings.{key} differs from preregistration")

    model_payload, loaded_model_sha = _load_json_with_sha256(model_path)
    if loaded_model_sha != model_sha:
        raise Phase7PipelineError("model config changed while loading")
    try:
        semantic_selection = load_semantic_selection(
            model_payload.get("semantic_selection"),
            root=ROOT,
        )
    except ValueError as error:
        raise Phase7PipelineError(
            "model semantic_selection is invalid"
        ) from error
    paths = model_payload.get("path_hypotheses")
    dol = model_payload.get("dol_probability")
    if (
        semantic_selection.foundation_registry_identity != FOUNDATION_IDENTITY
        or semantic_selection.parent_atomic_semantics_version
        != semantic_selection.atomic_semantics_version
        or not isinstance(paths, Mapping)
        or paths.get("path_protocol_fingerprint") != PATH_PROTOCOL_FINGERPRINT
        or paths.get("dol_protocol_fingerprint") != DOL_RANKING_FINGERPRINT
        or not isinstance(dol, Mapping)
        or dol.get("protocol_fingerprint") != DOL_PROTOCOL_FINGERPRINT
    ):
        raise Phase7PipelineError("model config is not bound to current Phase-7 identities")

    specs: list[InputWindowSpec] = []
    seen_windows: set[str] = set()
    for index, raw in enumerate(raw_inputs):
        item = _exact_mapping(
            raw,
            {
                "window_id",
                "path",
                "sha256",
                "row_count",
                "split_role",
                "fold_id",
                "purpose",
                "rolling_diagnostic_only",
                "source_authority",
                "contains_sealed_oos",
            },
            name=f"inputs[{index}]",
        )
        window_id = str(item["window_id"])
        registered = REGISTERED_WINDOWS.get(window_id)
        if registered is None or window_id in seen_windows:
            raise Phase7PipelineError("inputs must use each registered W1-W4 at most once")
        seen_windows.add(window_id)
        if (
            item["split_role"] != registered.split_role
            or item["purpose"] != registered.purpose
            or item["rolling_diagnostic_only"]
            is not registered.rolling_diagnostic_only
            or item["source_authority"] != "opened_june_2024_development"
            or item["contains_sealed_oos"] is not False
        ):
            raise Phase7PipelineError(f"{window_id} role/authority differs from registry")
        if item["fold_id"] != window_id:
            raise Phase7PipelineError(f"{window_id} fold_id must equal its window id")
        if type(item["row_count"]) is not int or item["row_count"] != registered.expected_rows:
            raise Phase7PipelineError(f"{window_id} row census differs")
        input_path = _resolve(item["path"])
        _assert_opened_jsonl_path(input_path, name=window_id)
        input_path = _trusted_regular_file(input_path, name=f"{window_id} input")
        _assert_opened_jsonl_path(input_path, name=window_id)
        input_sha = _sha256(item["sha256"], name=f"{window_id} sha256")
        # Only after the sealed/path guard is established may the runner touch
        # the declared source.
        if _sha256_file(input_path) != input_sha:
            raise Phase7PipelineError(f"{window_id} input hash differs")
        specs.append(
            InputWindowSpec(
                registered=registered,
                path=input_path,
                source_sha256=input_sha,
                row_count=item["row_count"],
                fold_id=item["fold_id"],
            )
        )
    specs.sort(key=lambda item: item.registered.start)
    if (
        _trusted_regular_file(source, name="Phase-7 run manifest") != source
        or _sha256_file(source) != manifest_sha256
    ):
        raise Phase7PipelineError("Phase-7 run manifest changed while validating")
    return RunManifest(
        run_id=payload["run_id"].strip(),
        source_path=source,
        source_sha256=manifest_sha256,
        model_config_path=model_path,
        model_config_sha256=model_sha,
        phase7_protocol_sha256=protocol_sha,
        preregistration_sha256=prereg_sha,
        repository_commit=repository_commit,
        python_major_minor=python_identity,
        code_bundle_identity=code_bundle_identity,
        materialization_authorized=True,
        fit_and_validate_authorized=authority["fit_and_validate_authorized"],
        inputs=tuple(specs),
    )


def _revalidate_run_manifest(manifest: RunManifest) -> None:
    """Re-open every authority and data binding and require exact equality."""

    if load_run_manifest(manifest.source_path) != manifest:
        raise Phase7PipelineError("Phase-7 run manifest binding changed")


def _validate_input_clocks(
    spec: InputWindowSpec,
    clocks: Sequence[ShadowClockInput],
) -> None:
    if len(clocks) != spec.row_count:
        raise Phase7PipelineError(f"{spec.window_id} decoded row census differs")
    if not clocks:
        raise Phase7PipelineError(f"{spec.window_id} is empty")
    starts = tuple(clock.bar.start for clock in clocks)
    decisions = tuple(clock.bar.end for clock in clocks)
    if (
        starts[0] != spec.registered.start
        or decisions[0] != spec.registered.first_decision_clock
    ):
        raise Phase7PipelineError(
            f"{spec.window_id} first bar-start/decision-clock boundary differs"
        )
    if (
        starts[-1]
        != spec.registered.last_decision_clock - pd.Timedelta(1, unit="min")
        or decisions[-1] != spec.registered.last_decision_clock
    ):
        raise Phase7PipelineError(
            f"{spec.window_id} last decision-clock boundary differs"
        )
    if (
        starts != tuple(sorted(starts))
        or len(starts) != len(set(starts))
        or decisions != tuple(sorted(decisions))
        or len(decisions) != len(set(decisions))
        or any(
            decision != start + pd.Timedelta(1, unit="min")
            for start, decision in zip(starts, decisions)
        )
        or len({clock.feed_event_id for clock in clocks}) != len(clocks)
        or len({clock.input_digest for clock in clocks}) != len(clocks)
    ):
        raise Phase7PipelineError(
            f"{spec.window_id} bar-start/decision-clock order or duration differs"
        )
    allowed = set(spec.registered.allowed_contracts)
    if any((clock.bar.symbol, clock.bar.instrument_id) not in allowed for clock in clocks):
        raise Phase7PipelineError(f"{spec.window_id} contains an unregistered contract")
    if any(
        clock.approved_intents
        or clock.execution_events
        or clock.account.position is not None
        for clock in clocks
    ):
        raise Phase7PipelineError("Phase-7 research input must be flat and action-free")


@dataclass(frozen=True)
class _HistorySeed:
    previous_history_id: str
    evidence_history_id: str
    rule_ids: tuple[str, ...]
    known_at: pd.Timestamp
    asof: pd.Timestamp
    source_event_ids: tuple[str, ...]
    correlation_cluster_id: str
    market_state_id: str
    real_bar_ordinal: int
    transition_key: str


@dataclass(frozen=True)
class _PredictionSeed:
    competition_set_id: str
    window_id: str
    prediction_known_at: pd.Timestamp
    common_expires_at: pd.Timestamp
    raw_probabilities: tuple[tuple[str, float], ...]
    prior_log_weights: tuple[tuple[str, float], ...]
    history_seeds: tuple[_HistorySeed, ...]
    real_completed_bar_count: int
    cluster_id: str
    split_role: str
    fold_id: str
    prediction_reason: tuple[str, ...]


@dataclass(frozen=True)
class _ReversionPredictionInput:
    """Prediction-only view; unlike fit DTOs it preserves validation role."""

    competition_set_id: str
    prediction_known_at: pd.Timestamp
    outcome_known_at: pd.Timestamp
    realized_path: str
    prior_log_weights: tuple[tuple[str, float], ...]
    contributions: tuple[EvidenceContributionAtPrediction, ...]
    split_role: str
    fold_id: str


@dataclass(frozen=True)
class _DOLPredictionInput:
    competition_set_id: str
    candidate_set_id: str
    path: str
    prediction_known_at: pd.Timestamp
    outcome_known_at: pd.Timestamp
    candidate_scores: tuple[tuple[str, float], ...]
    outcome_id: str
    split_role: str
    fold_id: str


@dataclass(frozen=True)
class _DOLGroup:
    window_id: str
    snapshots: tuple[DOLCandidateSnapshot, ...]
    candidate_scores: tuple[tuple[str, float], ...]
    cluster_id: str

    @property
    def candidate_set_id(self) -> str:
        return self.snapshots[0].candidate_set_id


@dataclass
class _Generation:
    window_id: str
    split_role: str
    fold_id: str
    initial_state: PathCompetitionSetState
    state: PathCompetitionSetState
    risk_clocks: list[PathRiskClock] = field(default_factory=list)
    history_seeds: list[_HistorySeed] = field(default_factory=list)
    predictions: list[_PredictionSeed] = field(default_factory=list)
    dol_groups: list[_DOLGroup] = field(default_factory=list)
    seen_prediction_clocks: set[pd.Timestamp] = field(default_factory=set)
    seen_dol_sets: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class MaterializedCohorts:
    archives: tuple[PathCompetitionArchive, ...]
    risk_intervals: tuple[PathRiskInterval, ...]
    history_transitions: tuple[EvidenceHistoryTransition, ...]
    dol_snapshots: tuple[DOLCandidateSnapshot, ...]
    dol_outcomes: tuple[DOLCandidateOutcomeRow, ...]
    predictions: tuple[_PredictionSeed, ...]
    dol_groups: tuple[_DOLGroup, ...]
    outcome_bars: tuple[tuple[str, OutcomeBar], ...]
    counts_by_window: Mapping[str, Mapping[str, int]]


def _history_projection(
    state: PathCompetitionSetState,
) -> tuple[str, tuple[tuple[str, str, tuple[Any, ...]], ...]]:
    """Return the first-occurrence state machine, never marginal evidence."""

    first: dict[str, Any] = {}
    for contribution in state.evidence_ledger:
        if contribution.rule_id in _EVIDENCE_RULES and contribution.rule_id not in first:
            first[contribution.rule_id] = contribution
    acceptance = first.get("acceptance_continuation")
    displacement = first.get("displacement_impact")
    if acceptance is None and displacement is None:
        return "none", ()
    if acceptance is not None and displacement is not None:
        if acceptance.known_at == displacement.known_at:
            return (
                "same_clock_joint",
                (("none", "same_clock_joint", (acceptance, displacement)),),
            )
        if acceptance.known_at < displacement.known_at:
            return (
                "acceptance_then_displacement",
                (
                    ("none", "acceptance_only", (acceptance,)),
                    (
                        "acceptance_only",
                        "acceptance_then_displacement",
                        (displacement,),
                    ),
                ),
            )
        return (
            "displacement_then_acceptance",
            (
                ("none", "displacement_only", (displacement,)),
                (
                    "displacement_only",
                    "displacement_then_acceptance",
                    (acceptance,),
                ),
            ),
        )
    if acceptance is not None:
        return "acceptance_only", (("none", "acceptance_only", (acceptance,)),)
    return "displacement_only", (("none", "displacement_only", (displacement,)),)


def _correlation_cluster(contributions: Sequence[Any], competition_set_id: str) -> str:
    keys = tuple(sorted({item.correlation_key for item in contributions}))
    return canonical_identity(
        "path-correlation-cluster",
        {"competition_set_id": competition_set_id, "correlation_keys": keys},
    )


def _market_state_id(snapshot: EngineSnapshot, state: PathCompetitionSetState) -> str:
    revision = snapshot.belief.scene_revision_id or snapshot.observation.scene_revision_id
    if not revision:
        raise Phase7PipelineError("active path state lacks an exact market revision")
    return canonical_identity(
        "path-market-state",
        {
            "competition_set_id": state.competition_set_id,
            "market_epoch_id": state.market_epoch_id,
            "scene_revision_id": revision,
        },
    )


def _path_values_from_members(state: PathCompetitionSetState) -> tuple[tuple[str, float], ...]:
    values = tuple((member.path.value, float(member.probability)) for member in state.members)
    if not math.isclose(math.fsum(value for _, value in values), 1.0, abs_tol=1e-12):
        raise Phase7PipelineError("active path probabilities are not complete")
    return values


def _prior_log_weights() -> tuple[tuple[str, float], ...]:
    protocol = load_path_belief_protocol(ROOT / "configs/path_hypotheses.json")
    return tuple((path.value, float(protocol.prior(path))) for path in PATH_KINDS)


_FROZEN_PRIOR_LOG_WEIGHTS = _prior_log_weights()


class Phase7CohortBuilder:
    """Stateful causal projector for one or more independently replayed windows."""

    def __init__(self) -> None:
        self._archives: list[PathCompetitionArchive] = []
        self._intervals: list[PathRiskInterval] = []
        self._history_rows: list[EvidenceHistoryTransition] = []
        self._dol_snapshots: list[DOLCandidateSnapshot] = []
        self._dol_outcomes: list[DOLCandidateOutcomeRow] = []
        self._predictions: list[_PredictionSeed] = []
        self._dol_groups: list[_DOLGroup] = []
        self._pending_dol: list[_DOLGroup] = []
        self._bars: list[tuple[str, OutcomeBar]] = []
        self._active: _Generation | None = None
        self._last_clock: ShadowClockInput | None = None
        self._window_bars: list[OutcomeBar] = []
        self._window_id: str | None = None
        self._seen_generation_ids: set[str] = set()
        self._terminalized_generation_ids: set[str] = set()
        self._counts: dict[str, dict[str, int]] = {}

    def start_window(self, spec: InputWindowSpec) -> None:
        if self._window_id is not None:
            raise Phase7PipelineError("prior window was not closed")
        self._window_id = spec.window_id
        self._window_bars = []
        self._last_clock = None
        self._active = None
        self._counts.setdefault(spec.window_id, {"clocks": 0})

    def _bar(self, clock: ShadowClockInput) -> OutcomeBar:
        return OutcomeBar(
            bar_event_id=clock.feed_event_id,
            symbol=clock.bar.symbol,
            instrument_id=clock.bar.instrument_id,
            timeframe=Timeframe.M1,
            known_at=clock.bar.end,
            open=clock.bar.open,
            high=clock.bar.high,
            low=clock.bar.low,
            close=clock.bar.close,
        )

    def observe(
        self,
        spec: InputWindowSpec,
        clock: ShadowClockInput,
        snapshot: EngineSnapshot | NeutralEngineSnapshot,
    ) -> None:
        if self._window_id != spec.window_id:
            raise Phase7PipelineError("builder window differs")
        if self._last_clock is not None and clock.bar.start <= self._last_clock.bar.start:
            raise Phase7PipelineError("builder clocks must advance strictly")
        self._last_clock = clock
        bar = self._bar(clock)
        self._window_bars.append(bar)
        self._bars.append((spec.window_id, bar))
        self._counts[spec.window_id]["clocks"] += 1

        observation_anomalies = tuple(getattr(snapshot.observation, "anomalies", ()))
        reset = bool(
            _RESET_ANOMALIES.intersection(
                {*observation_anomalies, *tuple(clock.execution.anomalies)}
            )
        )
        if reset and self._active is not None:
            self._finalize_external(
                self._active,
                terminal_at=clock.bar.end,
                terminal_status="censored",
                terminal_cause="runtime_reset",
                censor_reason="runtime_reset",
                source_event_ids=(clock.feed_event_id,),
                real_completed_bar=not clock.bar.synthetic_no_trade,
            )
            self._active = None
        if reset:
            self._censor_pending_dol(
                terminal_at=clock.bar.end,
                terminal_cause="runtime_reset",
                source_event_ids=(clock.feed_event_id,),
            )
        else:
            self._settle_due_dol(clock.bar.end)

        if isinstance(snapshot, NeutralEngineSnapshot):
            if self._active is not None:
                self._finalize_external(
                    self._active,
                    terminal_at=clock.bar.end,
                    terminal_status="censored",
                    terminal_cause="path_authority_missing",
                    censor_reason="path_authority_missing",
                    source_event_ids=(clock.feed_event_id,),
                    real_completed_bar=not clock.bar.synthetic_no_trade,
                )
                self._active = None
            self._censor_pending_dol(
                terminal_at=clock.bar.end,
                terminal_cause="path_authority_missing",
                source_event_ids=(clock.feed_event_id,),
            )
            return
        if not isinstance(snapshot, EngineSnapshot):
            raise TypeError("Phase-7 replay requires an Engine snapshot")
        state = snapshot.belief.path_competition_state
        if state is not None and not isinstance(state, PathCompetitionSetState):
            raise TypeError("MarketBelief path state is not canonical")
        if state is None:
            if self._active is not None:
                self._finalize_external(
                    self._active,
                    terminal_at=clock.bar.end,
                    terminal_status="censored",
                    terminal_cause="path_authority_missing",
                    censor_reason="path_authority_missing",
                    source_event_ids=(clock.feed_event_id,),
                    real_completed_bar=not clock.bar.synthetic_no_trade,
                )
                self._active = None
            self._censor_pending_dol(
                terminal_at=clock.bar.end,
                terminal_cause="path_authority_missing",
                source_event_ids=(clock.feed_event_id,),
            )
            return

        if (
            state.protocol_fingerprint != PATH_PROTOCOL_FINGERPRINT
            or state.authority != "shadow_only"
            or state.asof != clock.bar.end
        ):
            raise Phase7PipelineError("path state identity or clock differs")
        if state.competition_set_id in self._terminalized_generation_ids:
            if state.status is PathStatus.ACTIVE:
                raise Phase7PipelineError("terminal competition generation became active")
            # The production Brain causally carries an immutable terminal set
            # until its owning scope rolls.  That carry is not a new sample.
            return
        if self._active is not None and (
            self._active.state.competition_set_id != state.competition_set_id
        ):
            self._finalize_external(
                self._active,
                terminal_at=clock.bar.end,
                terminal_status="superseded",
                terminal_cause="scope_superseded",
                censor_reason="scope_superseded",
                source_event_ids=(clock.feed_event_id,),
                real_completed_bar=not clock.bar.synthetic_no_trade,
            )
            self._active = None
        self._censor_pending_dol(
            terminal_at=clock.bar.end,
            terminal_cause="scope_superseded",
            source_event_ids=(clock.feed_event_id,),
            except_competition_set_id=state.competition_set_id,
        )
        if self._active is None:
            if state.competition_set_id in self._seen_generation_ids:
                raise Phase7PipelineError("terminal competition generation reappeared")
            self._seen_generation_ids.add(state.competition_set_id)
            self._active = _Generation(
                window_id=spec.window_id,
                split_role=spec.split_role,
                fold_id=spec.fold_id,
                initial_state=state,
                state=state,
            )
        generation = self._active
        assert generation is not None
        generation.state = state
        market_state_id = _market_state_id(snapshot, state)
        history_id, transitions = _history_projection(state)
        correlation = _correlation_cluster(state.evidence_ledger, state.competition_set_id)
        existing_keys = {item.transition_key for item in generation.history_seeds}
        ordinal_by_id = dict(state.evidence_real_completed_bar_ordinals)
        history_changed = False
        for previous, current, contributions in transitions:
            key = canonical_identity(
                "history-transition-source",
                {
                    "competition_set_id": state.competition_set_id,
                    "previous": previous,
                    "current": current,
                    "contribution_ids": tuple(item.contribution_id for item in contributions),
                },
            )
            if key in existing_keys:
                continue
            known_at = max(item.known_at for item in contributions)
            source_ids = tuple(
                sorted({source for item in contributions for source in item.source_event_ids})
            )
            real_ordinal = max(
                (ordinal_by_id.get(item.contribution_id, 0) for item in contributions),
                default=0,
            )
            generation.history_seeds.append(
                _HistorySeed(
                    previous_history_id=previous,
                    evidence_history_id=current,
                    rule_ids=tuple(sorted(item.rule_id for item in contributions)),
                    known_at=known_at,
                    asof=state.asof,
                    source_event_ids=source_ids,
                    correlation_cluster_id=correlation,
                    market_state_id=market_state_id,
                    real_bar_ordinal=real_ordinal,
                    transition_key=key,
                )
            )
            existing_keys.add(key)
            history_changed = True

        if state.asof > state.formed_at:
            generation.risk_clocks.append(
                PathRiskClock(
                    asof=state.asof,
                    real_completed_bar=not clock.bar.synthetic_no_trade,
                    evidence_history_id=history_id,
                    correlation_cluster_id=correlation,
                    market_state_id=market_state_id,
                    source_event_ids=(clock.feed_event_id,),
                )
            )

        reasons: list[str] = []
        if state.asof == state.formed_at:
            reasons.append("competition_formation")
        if history_changed:
            reasons.append("registered_evidence_history_transition")
        if state.real_completed_bar_count in _PREDICTION_AGE_BOUNDARIES:
            reasons.append("registered_hazard_age_bin_boundary")
        if (
            state.status is PathStatus.ACTIVE
            and reasons
            and state.asof not in generation.seen_prediction_clocks
        ):
            generation.seen_prediction_clocks.add(state.asof)
            generation.predictions.append(
                _PredictionSeed(
                    competition_set_id=state.competition_set_id,
                    window_id=spec.window_id,
                    prediction_known_at=state.asof,
                    common_expires_at=state.common_expires_at,
                    raw_probabilities=_path_values_from_members(state),
                    prior_log_weights=_FROZEN_PRIOR_LOG_WEIGHTS,
                    history_seeds=tuple(generation.history_seeds),
                    real_completed_bar_count=state.real_completed_bar_count,
                    cluster_id=f"{state.market_epoch_id}:{state.horizon_id}",
                    split_role=spec.split_role,
                    fold_id=spec.fold_id,
                    prediction_reason=tuple(sorted(reasons)),
                )
            )
            self._capture_dol(generation, snapshot, clock)

        if state.status is not PathStatus.ACTIVE:
            self._finalize_state(generation, state)
            self._active = None

    def _capture_dol(
        self,
        generation: _Generation,
        snapshot: EngineSnapshot,
        clock: ShadowClockInput,
    ) -> None:
        belief = snapshot.belief
        for direction, ranking in sorted(belief.dol_rankings.items()):
            if (
                ranking.competition_set_id != generation.state.competition_set_id
                or ranking.path_asof != generation.state.asof
                or ranking.protocol_fingerprint != DOL_RANKING_FINGERPRINT
                or ranking.path_protocol_fingerprint != PATH_PROTOCOL_FINGERPRINT
            ):
                raise Phase7PipelineError("DOL ranking binding differs from path state")
            for path in PATH_LABELS:
                candidates = tuple(
                    item
                    for item in ranking.ranked_candidates
                    if item.path.value == path
                )
                if not candidates:
                    continue
                candidate_set_id = canonical_identity(
                    "dol-candidate-set",
                    {
                        "competition_set_id": generation.state.competition_set_id,
                        "prediction_known_at": generation.state.asof,
                        "direction": ranking.direction.value,
                        "path": path,
                        "ranking_id": ranking.ranking_id,
                        "candidate_ids": tuple(item.candidate_id for item in candidates),
                    },
                )
                if candidate_set_id in generation.seen_dol_sets:
                    continue
                generation.seen_dol_sets.add(candidate_set_id)
                snapshots = tuple(
                    DOLCandidateSnapshot(
                        competition_set_id=generation.state.competition_set_id,
                        candidate_set_id=candidate_set_id,
                        candidate_id=item.candidate_id,
                        path=path,
                        symbol=clock.bar.symbol,
                        instrument_id=clock.bar.instrument_id,
                        prediction_known_at=generation.state.asof,
                        common_expires_at=generation.state.common_expires_at,
                        candidate_eligible=True,
                        candidate_feature_schema_id=(
                            f"dol-ranking-features:{ranking.protocol_fingerprint}"
                        ),
                        raw_candidate_probability=float(
                            item.normalized_diagnostic_weight
                        ),
                        target_price=float(item.target_price),
                        source_event_ids=item.candidate_source_ids,
                        split_role=generation.split_role,
                        fold_id=generation.fold_id,
                    )
                    for item in candidates
                )
                generation.dol_groups.append(
                    _DOLGroup(
                        window_id=generation.window_id,
                        snapshots=snapshots,
                        candidate_scores=tuple(
                            sorted(
                                (
                                    item.candidate_id,
                                    float(item.conditional_log_weight),
                                )
                                for item in candidates
                            )
                        ),
                        cluster_id=(
                            f"{generation.state.market_epoch_id}:"
                            f"{generation.state.horizon_id}"
                        ),
                    )
                )

    def _settle_dol_group(
        self,
        group: _DOLGroup,
        *,
        observed_through: pd.Timestamp,
        observation_source_event_ids: Sequence[str],
        censor_reason: str | None,
        include_terminal_bar: bool,
    ) -> None:
        reference = group.snapshots[0]
        self._dol_snapshots.extend(group.snapshots)
        counts = self._counts.setdefault(group.window_id, {"clocks": 0})
        counts["dol_candidate_snapshot_rows"] = counts.get(
            "dol_candidate_snapshot_rows", 0
        ) + len(group.snapshots)
        # A candidate set frozen on the final observation clock has no strictly
        # future exposure.  Preserve its frozen snapshot, but do not invent a
        # zero-length outcome row that the canonical label DTO forbids.
        if observed_through <= reference.prediction_known_at:
            counts["dol_zero_future_exposure_sets"] = counts.get(
                "dol_zero_future_exposure_sets", 0
            ) + 1
            return
        bars = tuple(
            bar
            for bar in self._window_bars
            if (
                bar.known_at <= observed_through
                if include_terminal_bar
                else bar.known_at < observed_through
            )
        )
        hits = derive_dol_candidate_hits(group.snapshots, bars)
        labels = label_dol_candidate_set(
            group.snapshots,
            hits,
            observed_through=observed_through,
            observation_source_event_ids=observation_source_event_ids,
            censor_reason=censor_reason,
        )
        self._dol_outcomes.extend(labels)
        counts["dol_candidate_outcome_rows"] = counts.get(
            "dol_candidate_outcome_rows", 0
        ) + len(labels)

    def _settle_due_dol(self, asof: pd.Timestamp) -> None:
        retained: list[_DOLGroup] = []
        for group in self._pending_dol:
            reference = group.snapshots[0]
            if reference.common_expires_at > asof:
                retained.append(group)
                continue
            exact_horizon_sources = tuple(
                bar.bar_event_id
                for bar in self._window_bars
                if bar.known_at == reference.common_expires_at
                and bar.symbol == reference.symbol
                and bar.instrument_id == reference.instrument_id
            )
            if not exact_horizon_sources:
                raise Phase7PipelineError(
                    "DOL common horizon lacks an exact normalized BAR source"
                )
            self._settle_dol_group(
                group,
                observed_through=reference.common_expires_at,
                observation_source_event_ids=exact_horizon_sources,
                censor_reason=None,
                include_terminal_bar=True,
            )
        self._pending_dol = retained

    def _censor_pending_dol(
        self,
        *,
        terminal_at: pd.Timestamp,
        terminal_cause: str,
        source_event_ids: Sequence[str],
        except_competition_set_id: str | None = None,
    ) -> None:
        retained: list[_DOLGroup] = []
        for group in self._pending_dol:
            reference = group.snapshots[0]
            if reference.competition_set_id == except_competition_set_id:
                retained.append(group)
                continue
            self._settle_dol_group(
                group,
                observed_through=terminal_at,
                observation_source_event_ids=source_event_ids,
                censor_reason=terminal_cause,
                include_terminal_bar=False,
            )
        self._pending_dol = retained

    def _archive(
        self,
        generation: _Generation,
        *,
        terminal_at: pd.Timestamp,
        terminal_status: str,
        terminal_cause: str,
        realized_path: str | None,
        outcome_known_at: pd.Timestamp | None,
        source_event_ids: Sequence[str],
        censor_reason: str | None,
    ) -> PathCompetitionArchive:
        state = generation.state
        return PathCompetitionArchive(
            competition_set_id=state.competition_set_id,
            instrument_id=state.instrument_id,
            market_epoch_id=state.market_epoch_id,
            authority_structure_id=state.authority_structure_id,
            horizon_id=state.horizon_id,
            formed_at=state.formed_at,
            common_expires_at=state.common_expires_at,
            terminal_at=terminal_at,
            terminal_status=terminal_status,
            terminal_cause=terminal_cause,
            realized_path=realized_path,
            outcome_known_at=outcome_known_at,
            source_event_ids=tuple(source_event_ids),
            censor_reason=censor_reason,
            split_role=generation.split_role,
            fold_id=generation.fold_id,
        )

    def _finalize_external(
        self,
        generation: _Generation,
        *,
        terminal_at: pd.Timestamp,
        terminal_status: str,
        terminal_cause: str,
        censor_reason: str,
        source_event_ids: Sequence[str],
        real_completed_bar: bool = False,
    ) -> None:
        if terminal_at < generation.state.formed_at:
            raise Phase7PipelineError("external censor precedes generation formation")
        if terminal_at > generation.state.common_expires_at:
            raise Phase7PipelineError("external censor follows the registered horizon")
        if terminal_at > generation.state.formed_at and (
            not generation.risk_clocks
            or generation.risk_clocks[-1].asof < terminal_at
        ):
            history_id, _ = _history_projection(generation.state)
            generation.risk_clocks.append(
                PathRiskClock(
                    asof=terminal_at,
                    real_completed_bar=real_completed_bar,
                    evidence_history_id=history_id,
                    correlation_cluster_id=_correlation_cluster(
                        generation.state.evidence_ledger,
                        generation.state.competition_set_id,
                    ),
                    market_state_id=canonical_identity(
                        "path-market-state-terminal",
                        {
                            "competition_set_id": (
                                generation.state.competition_set_id
                            ),
                            "terminal_cause": terminal_cause,
                            "terminal_at": terminal_at,
                        },
                    ),
                    source_event_ids=tuple(source_event_ids),
                )
            )
        archive = self._archive(
            generation,
            terminal_at=terminal_at,
            terminal_status=terminal_status,
            terminal_cause=terminal_cause,
            realized_path=None,
            outcome_known_at=None,
            source_event_ids=source_event_ids,
            censor_reason=censor_reason,
        )
        dispositions = tuple(
            PathTerminalDisposition(
                path=path,
                status=("superseded" if terminal_status == "superseded" else "censored"),
                cause=terminal_cause,
                known_at=terminal_at,
                source_event_ids=tuple(source_event_ids),
                censor_reason=censor_reason,
            )
            for path in PATH_LABELS
        )
        self._finish_generation(generation, archive, dispositions)

    def _finalize_state(
        self,
        generation: _Generation,
        state: PathCompetitionSetState,
    ) -> None:
        if state.status is PathStatus.REALIZED:
            assert state.winner_path is not None and state.realized_at is not None
            winner = state.member(state.winner_path)
            archive = self._archive(
                generation,
                terminal_at=state.realized_at,
                terminal_status="realized",
                terminal_cause=winner.terminal_reason or "registered_path_outcome",
                realized_path=state.winner_path.value,
                outcome_known_at=state.realized_at,
                source_event_ids=state.outcome_source_event_ids,
                censor_reason=None,
            )
        elif state.status is PathStatus.EXPIRED:
            archive = self._archive(
                generation,
                terminal_at=state.asof,
                terminal_status="expired",
                terminal_cause="common_horizon_elapsed",
                realized_path=None,
                outcome_known_at=None,
                source_event_ids=(),
                censor_reason="path_competition_expired_without_factual_winner",
            )
        else:
            raise Phase7PipelineError("unsupported terminal path competition state")
        dispositions: list[PathTerminalDisposition] = []
        for member in state.members:
            if member.terminal_at is None or not member.terminal_reason:
                raise Phase7PipelineError("terminal path member lacks provenance")
            status = (
                "realized"
                if member.status is PathStatus.REALIZED
                else "falsified"
                if member.status is PathStatus.INVALIDATED
                else "expired"
            )
            dispositions.append(
                PathTerminalDisposition(
                    path=member.path.value,
                    status=status,
                    cause=member.terminal_reason,
                    known_at=member.terminal_at,
                    source_event_ids=member.terminal_source_event_ids,
                )
            )
        self._finish_generation(generation, archive, tuple(dispositions))

    def _finish_generation(
        self,
        generation: _Generation,
        archive: PathCompetitionArchive,
        dispositions: Sequence[PathTerminalDisposition],
    ) -> None:
        clocks = tuple(
            clock for clock in generation.risk_clocks if clock.asof <= archive.terminal_at
        )
        if clocks:
            self._intervals.extend(
                build_path_risk_intervals(
                    archive,
                    clocks,
                    terminal_dispositions=dispositions,
                )
            )
        elif archive.terminal_at > archive.formed_at:
            raise Phase7PipelineError("terminal generation lacks causal risk clocks")
        self._archives.append(archive)
        self._terminalized_generation_ids.add(archive.competition_set_id)
        self._predictions.extend(generation.predictions)
        self._dol_groups.extend(generation.dol_groups)

        for seed in generation.history_seeds:
            resolved = bool(
                archive.terminal_status == "realized"
                and archive.outcome_known_at is not None
                and archive.outcome_known_at > seed.asof
            )
            self._history_rows.append(
                EvidenceHistoryTransition(
                    competition_set_id=archive.competition_set_id,
                    asof=seed.asof,
                    known_at=seed.known_at,
                    common_expires_at=archive.common_expires_at,
                    previous_history_id=seed.previous_history_id,
                    evidence_history_id=seed.evidence_history_id,
                    evidence_rule_ids=seed.rule_ids,
                    evidence_observed=True,
                    correlation_cluster_id=seed.correlation_cluster_id,
                    market_state_id=seed.market_state_id,
                    realized_path=archive.realized_path if resolved else None,
                    outcome_known_at=archive.outcome_known_at if resolved else None,
                    censor_reason=(
                        None
                        if resolved
                        else archive.censor_reason
                        or "outcome_not_strictly_future_of_transition"
                    ),
                    split_role=archive.split_role,
                    fold_id=archive.fold_id,
                    source_event_ids=seed.source_event_ids,
                )
            )

        for group in generation.dol_groups:
            if archive.terminal_status in {"censored", "superseded"}:
                self._settle_dol_group(
                    group,
                    observed_through=archive.terminal_at,
                    observation_source_event_ids=archive.source_event_ids,
                    censor_reason=archive.censor_reason,
                    include_terminal_bar=False,
                )
            elif archive.terminal_at == archive.common_expires_at:
                exact_horizon_sources = tuple(
                    bar.bar_event_id
                    for bar in self._window_bars
                    if bar.known_at == archive.common_expires_at
                    and bar.symbol == group.snapshots[0].symbol
                    and bar.instrument_id == group.snapshots[0].instrument_id
                )
                if not exact_horizon_sources:
                    raise Phase7PipelineError(
                        "DOL terminal horizon lacks an exact normalized BAR source"
                    )
                self._settle_dol_group(
                    group,
                    observed_through=archive.common_expires_at,
                    observation_source_event_ids=exact_horizon_sources,
                    censor_reason=None,
                    include_terminal_bar=True,
                )
            else:
                self._pending_dol.append(group)

        counts = self._counts[generation.window_id]
        counts["competition_archives"] = counts.get("competition_archives", 0) + 1
        counts["path_risk_intervals"] = counts.get("path_risk_intervals", 0) + sum(
            row.competition_set_id == archive.competition_set_id for row in self._intervals
        )
        counts["history_transitions"] = counts.get("history_transitions", 0) + len(
            generation.history_seeds
        )
        counts["path_prediction_seeds"] = counts.get("path_prediction_seeds", 0) + len(
            generation.predictions
        )
        counts["dol_candidate_sets"] = counts.get("dol_candidate_sets", 0) + len(
            generation.dol_groups
        )

    def finish_window(self, spec: InputWindowSpec) -> None:
        if self._window_id != spec.window_id or self._last_clock is None:
            raise Phase7PipelineError("cannot close an empty or different window")
        if self._active is not None:
            self._finalize_external(
                self._active,
                terminal_at=self._last_clock.bar.end,
                terminal_status="censored",
                terminal_cause="observation_window_end",
                censor_reason="observation_window_end",
                source_event_ids=(self._last_clock.feed_event_id,),
            )
            self._active = None
        self._censor_pending_dol(
            terminal_at=self._last_clock.bar.end,
            terminal_cause="observation_window_end",
            source_event_ids=(self._last_clock.feed_event_id,),
        )
        self._window_id = None
        self._window_bars = []
        self._last_clock = None

    def result(self) -> MaterializedCohorts:
        if self._window_id is not None:
            raise Phase7PipelineError("builder result requested before window close")
        if self._pending_dol:
            raise Phase7PipelineError("builder result retains unresolved DOL horizons")
        identities = {
            "archives": [row.archive_id for row in self._archives],
            "intervals": [row.interval_id for row in self._intervals],
            "history": [row.transition_id for row in self._history_rows],
            "dol_snapshots": [row.snapshot_id for row in self._dol_snapshots],
            "dol_outcomes": [row.label_id for row in self._dol_outcomes],
        }
        for name, values in identities.items():
            if len(values) != len(set(values)):
                raise Phase7PipelineError(f"materialized {name} repeat an identity")
        return MaterializedCohorts(
            archives=tuple(sorted(self._archives, key=lambda row: row.archive_id)),
            risk_intervals=tuple(sorted(self._intervals, key=lambda row: row.interval_id)),
            history_transitions=tuple(
                sorted(self._history_rows, key=lambda row: row.transition_id)
            ),
            dol_snapshots=tuple(
                sorted(self._dol_snapshots, key=lambda row: row.snapshot_id)
            ),
            dol_outcomes=tuple(
                sorted(self._dol_outcomes, key=lambda row: row.label_id)
            ),
            predictions=tuple(
                sorted(
                    self._predictions,
                    key=lambda row: (
                        row.window_id,
                        row.prediction_known_at,
                        row.competition_set_id,
                    ),
                )
            ),
            dol_groups=tuple(
                sorted(self._dol_groups, key=lambda row: row.candidate_set_id)
            ),
            outcome_bars=tuple(self._bars),
            counts_by_window={
                key: dict(sorted(value.items())) for key, value in sorted(self._counts.items())
            },
        )


EngineFactory = Callable[[], ContinuousSMCEngine]
ClockLoader = Callable[[Path], Sequence[ShadowClockInput]]


def _default_engine_factory() -> ContinuousSMCEngine:
    return ContinuousSMCEngine.from_config(MODEL_CONFIG, runtime_mode="development")


def _default_clock_loader(path: Path) -> Sequence[ShadowClockInput]:
    return tuple(iter_shadow_clock_file(path))


def materialize_cohorts(
    manifest: RunManifest,
    *,
    include_rolling_diagnostics: bool,
    engine_factory: EngineFactory = _default_engine_factory,
    clock_loader: ClockLoader = _default_clock_loader,
) -> MaterializedCohorts:
    if _repository_identity() != manifest.repository_commit:
        raise Phase7PipelineError("repository changed after run manifest validation")
    _assert_code_bundle_committed()
    selected = tuple(
        spec
        for spec in manifest.inputs
        if include_rolling_diagnostics or not spec.registered.rolling_diagnostic_only
    )
    if not selected:
        raise Phase7PipelineError("no selected input windows")
    builder = Phase7CohortBuilder()
    for spec in selected:
        if _sha256_file(spec.path) != spec.source_sha256:
            raise Phase7PipelineError(f"{spec.window_id} input changed before replay")
        clocks = tuple(clock_loader(spec.path))
        if _sha256_file(spec.path) != spec.source_sha256:
            raise Phase7PipelineError(f"{spec.window_id} input changed while loading")
        _validate_input_clocks(spec, clocks)
        if _sha256_file(manifest.model_config_path) != manifest.model_config_sha256:
            raise Phase7PipelineError("model config changed before Engine construction")
        engine = engine_factory()
        if (
            engine.model_config_sha256 != manifest.model_config_sha256
            or engine.foundation_registry_identity != FOUNDATION_IDENTITY
            or _sha256_file(manifest.model_config_path) != manifest.model_config_sha256
        ):
            raise Phase7PipelineError("Engine runtime identity differs from run manifest")
        builder.start_window(spec)
        for clock in clocks:
            snapshot = engine.on_bar(
                clock.bar,
                execution=clock.execution,
                account=clock.account,
            )
            builder.observe(spec, clock, snapshot)
        builder.finish_window(spec)
    result = builder.result()
    if _repository_identity() != manifest.repository_commit:
        raise Phase7PipelineError("repository changed during empirical replay")
    _assert_code_bundle_committed()
    return result


def _resolved_archive_by_id(
    cohorts: MaterializedCohorts,
) -> Mapping[str, PathCompetitionArchive]:
    return {
        archive.competition_set_id: archive
        for archive in cohorts.archives
        if archive.terminal_status == "realized"
        and archive.realized_path is not None
        and archive.outcome_known_at is not None
    }


def fit_path_base_rate_prior(
    cohorts: MaterializedCohorts,
    *,
    source_dataset_sha256: str,
    manifest_sha256: str,
    smoothing_alpha: float = 0.5,
    competition_ids: set[str] | None = None,
) -> Mapping[str, Any]:
    """Fit the W1 competition-unit Jeffreys path prior with a fixed gauge."""

    rows = tuple(
        archive
        for archive in cohorts.archives
        if archive.split_role == "development_fit"
        and archive.terminal_status == "realized"
        and archive.realized_path is not None
        and (
            competition_ids is None
            or archive.competition_set_id in competition_ids
        )
    )
    if not rows:
        raise ValueError("path base-rate prior has zero resolved W1 competitions")
    counts = {path: 0 for path in PATH_LABELS}
    for row in rows:
        assert row.realized_path is not None
        counts[row.realized_path] += 1
    denominator = len(rows) + smoothing_alpha * len(PATH_LABELS)
    probabilities = {
        path: (counts[path] + smoothing_alpha) / denominator
        for path in PATH_LABELS
    }
    gauge_path = PathKind.RESIDUAL_UNKNOWN.value
    gauge = math.log(probabilities[gauge_path])
    log_weights = {
        path: math.log(probabilities[path]) - gauge for path in PATH_LABELS
    }
    payload = {
        "schema_version": "phase7_path_base_rate_prior_v1",
        "model_version": "phase7_path_base_rate_prior_jeffreys_v1",
        "source_dataset_sha256": source_dataset_sha256,
        "manifest_sha256": manifest_sha256,
        "smoothing": "dirichlet_jeffreys",
        "smoothing_alpha": smoothing_alpha,
        "fit_competition_count": len(rows),
        "observed_counts": counts,
        "probabilities": probabilities,
        "prior_log_weights": log_weights,
        "additive_gauge_path": gauge_path,
        "additive_gauge_log_weight": 0.0,
        "authority": "research_shadow_only",
        "action_authority": False,
    }
    return {
        **payload,
        "artifact_id": canonical_identity("path-base-rate-prior", payload),
    }


def _reversion_inputs(
    cohorts: MaterializedCohorts,
    artifact: HistoryConditionalLikelihoodArtifact,
    *,
    windows: set[str],
    fitted_prior_log_weights: Mapping[str, float],
    competition_ids: set[str] | None = None,
) -> tuple[_ReversionPredictionInput, ...]:
    archives = _resolved_archive_by_id(cohorts)
    rows: list[_ReversionPredictionInput] = []
    for seed in cohorts.predictions:
        if seed.window_id not in windows:
            continue
        if competition_ids is not None and seed.competition_set_id not in competition_ids:
            continue
        archive = archives.get(seed.competition_set_id)
        if archive is None or archive.outcome_known_at <= seed.prediction_known_at:
            continue
        contributions = tuple(
            EvidenceContributionAtPrediction(
                contribution_id=(
                    _HISTORY_TRANSITION_RULE_IDS[
                        (
                            history.previous_history_id,
                            history.evidence_history_id,
                        )
                    ]
                    + ":"
                    + history.transition_key
                ),
                age_real_completed_bars=max(
                    0,
                    seed.real_completed_bar_count
                    - 1
                    - history.real_bar_ordinal,
                ),
                path_log_likelihoods=tuple(
                    (
                        path,
                        artifact.log_increment(
                            history.previous_history_id,
                            history.evidence_history_id,
                            path,
                        ),
                    )
                    for path in PATH_LABELS
                ),
            )
            for history in seed.history_seeds
        )
        rows.append(
            _ReversionPredictionInput(
                competition_set_id=seed.competition_set_id,
                prediction_known_at=seed.prediction_known_at,
                outcome_known_at=archive.outcome_known_at,
                realized_path=archive.realized_path,
                prior_log_weights=tuple(
                    (path, float(fitted_prior_log_weights[path]))
                    for path in PATH_LABELS
                ),
                contributions=contributions,
                split_role=seed.split_role,
                fold_id=seed.fold_id,
            )
        )
    return tuple(rows)


def _prior_fit_samples(
    inputs: Sequence[_ReversionPredictionInput],
) -> tuple[PriorReversionSample, ...]:
    return tuple(
        PriorReversionSample(
            competition_set_id=row.competition_set_id,
            prediction_known_at=row.prediction_known_at,
            outcome_known_at=row.outcome_known_at,
            realized_path=row.realized_path,
            prior_log_weights=row.prior_log_weights,
            contributions=row.contributions,
            split_role=row.split_role,
            fold_id=row.fold_id,
        )
        for row in inputs
        if row.split_role in _FIT_ROLES
    )


def history_transition_rules_payload(
    artifact: HistoryConditionalLikelihoodArtifact,
) -> Mapping[str, Any]:
    """Publish the exact history-state converter consumed by reversion fits.

    A cell is one ``previous_history -> next_history`` likelihood conditioned
    on the eventual path.  It is intentionally not representable as either of
    the two legacy marginal evidence-rule increments.
    """

    transitions: list[Mapping[str, Any]] = []
    for previous, next_states in artifact.allowed_transitions:
        for next_state in next_states:
            likelihoods = {
                path: artifact.likelihood(previous, next_state, path)
                for path in PATH_LABELS
            }
            transitions.append(
                {
                    "rule_id": _HISTORY_TRANSITION_RULE_IDS[(previous, next_state)],
                    "evidence_family": "path_evidence_history_transition",
                    "previous_history_id": previous,
                    "evidence_history_id": next_state,
                    "conditional_likelihood_by_path": likelihoods,
                    "log_likelihood_increment_by_path": {
                        path: math.log(value) for path, value in likelihoods.items()
                    },
                }
            )
    return {
        "schema_version": "phase7_history_transition_rules_v1",
        "artifact_id": artifact.artifact_id,
        "model_version": artifact.model_version,
        "application_unit": "one_registered_history_state_transition",
        "first_occurrence_only": True,
        "one_rule_equals_one_history_conditioned_contribution": True,
        "legacy_marginal_rule_projection_allowed": False,
        "legacy_static_evidence_rules_replaced": False,
        "runtime_authority": "research_artifact_only",
        "action_authority": False,
        "transitions": transitions,
    }


def _artifact_payload(value: Any) -> Mapping[str, Any]:
    payload = asdict(value)
    payload["artifact_id"] = value.artifact_id
    return payload


def temporal_runtime_candidate_payload(
    temporal_artifact: Any,
    reversion_artifact: Any | None,
    *,
    survival_threshold: float = 0.10,
    minimum_at_risk_intervals_per_cell: int = 30,
) -> Mapping[str, Any]:
    """Derive non-authoritative per-path expiry candidates from life-table cells."""

    if (
        not 0.0 < survival_threshold < 1.0
        or type(minimum_at_risk_intervals_per_cell) is not int
        or minimum_at_risk_intervals_per_cell < 1
    ):
        raise ValueError("temporal runtime candidate support contract is invalid")

    cells_by_path: dict[str, dict[int, Any]] = {path: {} for path in PATH_LABELS}
    for cell in temporal_artifact.cells:
        cells_by_path[cell.path][cell.bin_index] = cell
    expiry: dict[str, int | None] = {}
    blockers: dict[str, str] = {}
    hazard_rows: dict[str, list[Mapping[str, Any]]] = {}
    half_lives = (
        {}
        if reversion_artifact is None
        else dict(reversion_artifact.path_half_lives)
    )
    for path in PATH_LABELS:
        path_cells = cells_by_path[path]
        hazard_rows[path] = []
        for index, end in enumerate(temporal_artifact.age_bin_ends):
            cell = path_cells.get(index)
            if cell is None:
                continue
            hazard_rows[path].append(
                {
                    "age_start_bar": cell.age_start_bar,
                    "age_end_bar": end,
                    "at_risk_intervals": cell.at_risk_intervals,
                    "cause_hazards": dict(cell.cause_hazards),
                    "no_event_probability_per_real_bar": (
                        cell.no_event_probability
                    ),
                }
            )
        if path == PathKind.RESIDUAL_UNKNOWN.value:
            expiry[path] = None
            continue
        survival = 1.0
        candidate: int | None = None
        expected_start = 1
        for index, end in enumerate(temporal_artifact.age_bin_ends):
            cell = path_cells.get(index)
            if cell is None or cell.age_start_bar != expected_start:
                blockers[path] = (
                    f"insufficient_contiguous_life_table_support_at_age_{expected_start}"
                )
                break
            if cell.at_risk_intervals < minimum_at_risk_intervals_per_cell:
                blockers[path] = (
                    "insufficient_life_table_support_at_age_"
                    f"{expected_start}:observed={cell.at_risk_intervals}:"
                    f"required={minimum_at_risk_intervals_per_cell}"
                )
                break
            no_event = float(cell.no_event_probability)
            if end is None:
                if 0.0 < no_event < 1.0:
                    remaining = math.log(survival_threshold / survival) / math.log(
                        no_event
                    )
                    steps = max(1, math.ceil(remaining - 1e-15))
                    candidate = expected_start + steps - 1
                elif no_event == 0.0:
                    candidate = expected_start
                else:
                    blockers[path] = "survival_never_reaches_threshold_in_open_bin"
                break
            for age in range(expected_start, end + 1):
                survival *= no_event
                if survival <= survival_threshold:
                    candidate = age
                    break
            if candidate is not None:
                break
            expected_start = end + 1
        expiry[path] = candidate
        if candidate is None and path not in blockers:
            blockers[path] = "expiry_threshold_not_reached_with_supported_cells"
    payload = {
        "schema_version": "phase7_temporal_runtime_candidate_v1",
        "source_temporal_artifact_id": temporal_artifact.artifact_id,
        "source_prior_reversion_artifact_id": (
            None if reversion_artifact is None else reversion_artifact.artifact_id
        ),
        "clock": "real_completed_M1_bar",
        "piecewise_convention": (
            "each life-table cell no_event_probability is applied once per "
            "real completed bar throughout its registered inclusive age bin"
        ),
        "survival_threshold": survival_threshold,
        "minimum_at_risk_intervals_per_expiry_cell": (
            minimum_at_risk_intervals_per_cell
        ),
        "hypothesis_expiry_bars": expiry,
        "non_residual_runtime_cap": "min(expiry_bars,shared_common_horizon)",
        "residual_unknown_expiry_bars": None,
        "cause_hazards": hazard_rows,
        "prior_reversion_half_life_real_bars": {
            path: half_lives.get(path) for path in PATH_LABELS
        },
        "support_blockers": blockers,
        "status": "research_runtime_candidate_not_admitted",
        "authority": "shadow_only",
        "action_authority": False,
    }
    return {
        **payload,
        "artifact_id": canonical_identity("temporal-runtime-candidate", payload),
    }


def _dol_choice_samples(
    cohorts: MaterializedCohorts,
    *,
    windows: set[str],
) -> tuple[DOLChoiceSetSample, ...]:
    outcomes: dict[str, tuple[DOLCandidateOutcomeRow, ...]] = {}
    for row in cohorts.dol_outcomes:
        outcomes.setdefault(row.snapshot.candidate_set_id, ())
        outcomes[row.snapshot.candidate_set_id] = (
            *outcomes[row.snapshot.candidate_set_id],
            row,
        )
    samples: list[DOLChoiceSetSample] = []
    for group in cohorts.dol_groups:
        if group.window_id not in windows:
            continue
        labels = outcomes.get(group.candidate_set_id, ())
        if not labels:
            continue
        first = labels[0]
        if first.censor_reason is not None:
            continue
        outcome_id = (
            NO_TARGET_OUTCOME
            if first.no_target_before_horizon
            else first.first_hit_candidate_id
        )
        if outcome_id is None:
            continue
        samples.append(
            DOLChoiceSetSample(
                competition_set_id=first.snapshot.competition_set_id,
                candidate_set_id=first.snapshot.candidate_set_id,
                path=first.snapshot.path,
                prediction_known_at=first.snapshot.prediction_known_at,
                outcome_known_at=first.outcome_known_at,
                candidate_scores=group.candidate_scores,
                outcome_id=outcome_id,
                split_role=first.snapshot.split_role,
                fold_id=first.snapshot.fold_id,
            )
        )
    return tuple(samples)


def _dol_prediction_inputs(
    cohorts: MaterializedCohorts,
    *,
    window_id: str,
) -> tuple[_DOLPredictionInput, ...]:
    outcomes: dict[str, tuple[DOLCandidateOutcomeRow, ...]] = {}
    for row in cohorts.dol_outcomes:
        outcomes[row.snapshot.candidate_set_id] = (
            *outcomes.get(row.snapshot.candidate_set_id, ()),
            row,
        )
    rows: list[_DOLPredictionInput] = []
    for group in cohorts.dol_groups:
        if group.window_id != window_id:
            continue
        labels = outcomes.get(group.candidate_set_id, ())
        if not labels or labels[0].censor_reason is not None:
            continue
        first = labels[0]
        outcome_id = (
            NO_TARGET_OUTCOME
            if first.no_target_before_horizon
            else first.first_hit_candidate_id
        )
        if outcome_id is None:
            continue
        rows.append(
            _DOLPredictionInput(
                competition_set_id=first.snapshot.competition_set_id,
                candidate_set_id=first.snapshot.candidate_set_id,
                path=first.snapshot.path,
                prediction_known_at=first.snapshot.prediction_known_at,
                outcome_known_at=first.outcome_known_at,
                candidate_scores=group.candidate_scores,
                outcome_id=outcome_id,
                split_role=first.snapshot.split_role,
                fold_id=first.snapshot.fold_id,
            )
        )
    return tuple(rows)


def _path_predictions(
    cohorts: MaterializedCohorts,
    reversion_artifact: Any,
    history_artifact: HistoryConditionalLikelihoodArtifact,
    *,
    window_id: str,
    calibration_artifact: Any | None,
    fitted_prior_log_weights: Mapping[str, float],
) -> tuple[ProbabilityPrediction, ...]:
    samples = _reversion_inputs(
        cohorts,
        history_artifact,
        windows={window_id},
        fitted_prior_log_weights=fitted_prior_log_weights,
    )
    raw_by_key = {
        (seed.competition_set_id, seed.prediction_known_at): seed
        for seed in cohorts.predictions
        if seed.window_id == window_id
    }
    output: list[ProbabilityPrediction] = []
    for sample in samples:
        model = reversion_artifact.probabilities(sample)
        if calibration_artifact is not None:
            model = calibration_artifact.probabilities(model)
        seed = raw_by_key[(sample.competition_set_id, sample.prediction_known_at)]
        output.append(
            ProbabilityPrediction(
                unit_id=canonical_identity(
                    "path-prediction-unit",
                    {
                        "competition_set_id": sample.competition_set_id,
                        "prediction_known_at": sample.prediction_known_at,
                    },
                ),
                cluster_id=seed.cluster_id,
                fold_id=sample.fold_id,
                realized_label=sample.realized_path,
                support_label=sample.realized_path,
                model_probabilities=tuple(model.items()),
                baseline_probabilities=seed.raw_probabilities,
            )
        )
    return tuple(output)


def _base_rate_predictions(
    cohorts: MaterializedCohorts,
    base_prior_artifact: Mapping[str, Any],
    *,
    window_id: str,
    calibration_artifact: Any | None = None,
) -> tuple[ProbabilityPrediction, ...]:
    archives = _resolved_archive_by_id(cohorts)
    base_model = {
        path: float(base_prior_artifact["probabilities"][path])
        for path in PATH_LABELS
    }
    if calibration_artifact is not None:
        base_model = dict(calibration_artifact.probabilities(base_model))
    model = tuple(
        (path, base_model[path])
        for path in PATH_LABELS
    )
    output: list[ProbabilityPrediction] = []
    for seed in cohorts.predictions:
        if seed.window_id != window_id:
            continue
        archive = archives.get(seed.competition_set_id)
        if (
            archive is None
            or archive.outcome_known_at is None
            or archive.outcome_known_at <= seed.prediction_known_at
            or archive.realized_path is None
        ):
            continue
        output.append(
            ProbabilityPrediction(
                unit_id=canonical_identity(
                    "path-prediction-unit",
                    {
                        "competition_set_id": seed.competition_set_id,
                        "prediction_known_at": seed.prediction_known_at,
                    },
                ),
                cluster_id=seed.cluster_id,
                fold_id=seed.fold_id,
                realized_label=archive.realized_path,
                support_label=archive.realized_path,
                model_probabilities=model,
                baseline_probabilities=seed.raw_probabilities,
            )
        )
    return tuple(output)


def _dol_predictions(
    cohorts: MaterializedCohorts,
    artifact: Any,
    *,
    window_id: str,
) -> tuple[ProbabilityPrediction, ...]:
    samples = _dol_prediction_inputs(cohorts, window_id=window_id)
    cluster_by_set = {
        group.candidate_set_id: group.cluster_id for group in cohorts.dol_groups
    }
    output: list[ProbabilityPrediction] = []
    for sample in samples:
        model = artifact.probabilities(sample.path, sample.candidate_scores)
        labels = tuple(identity for identity, _ in sample.candidate_scores) + (
            NO_TARGET_OUTCOME,
        )
        baseline = tuple((identity, 1.0 / len(labels)) for identity in labels)
        output.append(
            ProbabilityPrediction(
                unit_id=sample.candidate_set_id,
                cluster_id=cluster_by_set[sample.candidate_set_id],
                fold_id=sample.fold_id,
                realized_label=sample.outcome_id,
                support_label=dol_support_label(sample.outcome_id),
                model_probabilities=tuple(model.items()),
                baseline_probabilities=baseline,
            )
        )
    return tuple(output)


def _w1_inner_rolling_crossfit(
    cohorts: MaterializedCohorts,
    *,
    source_dataset_sha256: str,
    manifest_sha256: str,
) -> Mapping[str, Any]:
    """Create whole-session/whole-generation W1 development OOF predictions.

    This is an inner development cross-fit used only to fit calibration.  It
    is never a rolling-OOF admission cohort.  Each held session is predicted
    solely from strictly earlier complete sessions, and every competition
    generation remains wholly in one side of the fold.
    """

    archives = tuple(
        archive
        for archive in cohorts.archives
        if archive.split_role == "development_fit"
    )
    by_session: dict[str, list[PathCompetitionArchive]] = {}
    for archive in archives:
        by_session.setdefault(archive.horizon_id, []).append(archive)
    session_order = tuple(
        session
        for session, _ in sorted(
            (
                (session, min(row.formed_at for row in rows))
                for session, rows in by_session.items()
            ),
            key=lambda item: (item[1], item[0]),
        )
    )
    path_samples: list[PathProbabilitySample] = []
    path_rows: list[Mapping[str, Any]] = []
    dol_rows: list[Mapping[str, Any]] = []
    blockers: list[str] = []
    archive_by_id = {row.competition_set_id: row for row in archives}
    dol_inputs = _dol_prediction_inputs(cohorts, window_id="W1")

    for fold_index, held_session in enumerate(session_order[1:], start=1):
        train_sessions = session_order[:fold_index]
        train_ids = {
            row.competition_set_id
            for session in train_sessions
            for row in by_session[session]
        }
        held_ids = {
            row.competition_set_id for row in by_session[held_session]
        }
        if train_ids.intersection(held_ids):
            raise Phase7PipelineError("inner cross-fit leaks a competition generation")
        fold_id = canonical_identity(
            "w1-inner-session-fold",
            {
                "train_sessions": train_sessions,
                "held_session": held_session,
                "train_competition_ids": tuple(sorted(train_ids)),
                "held_competition_ids": tuple(sorted(held_ids)),
            },
        )
        try:
            base = fit_path_base_rate_prior(
                cohorts,
                source_dataset_sha256=source_dataset_sha256,
                manifest_sha256=manifest_sha256,
                competition_ids=train_ids,
            )
        except ValueError as error:
            blockers.append(f"{held_session}:path_base_rate:{error}")
            continue

        history = None
        reversion = None
        try:
            history = fit_history_conditional_likelihood(
                tuple(
                    row
                    for row in cohorts.history_transitions
                    if row.competition_set_id in train_ids
                    and row.split_role == "development_fit"
                ),
                source_dataset_sha256=source_dataset_sha256,
                manifest_sha256=manifest_sha256,
            )
            training_inputs = _reversion_inputs(
                cohorts,
                history,
                windows={"W1"},
                fitted_prior_log_weights=base["prior_log_weights"],
                competition_ids=train_ids,
            )
            reversion = fit_evidence_prior_reversion(
                _prior_fit_samples(training_inputs),
                source_dataset_sha256=source_dataset_sha256,
                manifest_sha256=manifest_sha256,
            )
        except ValueError as error:
            blockers.append(f"{held_session}:history_or_reversion:{error}")

        if history is not None and reversion is not None:
            held_inputs = _reversion_inputs(
                cohorts,
                history,
                windows={"W1"},
                fitted_prior_log_weights=base["prior_log_weights"],
                competition_ids=held_ids,
            )
            path_values = tuple(
                (row, tuple(reversion.probabilities(row).items()))
                for row in held_inputs
            )
        else:
            held_inputs = ()
            path_values = tuple(
                (
                    seed,
                    tuple(
                        (path, float(base["probabilities"][path]))
                        for path in PATH_LABELS
                    ),
                )
                for seed in cohorts.predictions
                if seed.competition_set_id in held_ids
                and seed.competition_set_id in archive_by_id
                and archive_by_id[seed.competition_set_id].outcome_known_at
                is not None
                and archive_by_id[seed.competition_set_id].outcome_known_at
                > seed.prediction_known_at
            )
        for source, probabilities in path_values:
            archive = archive_by_id[source.competition_set_id]
            assert archive.outcome_known_at is not None
            assert archive.realized_path is not None
            prediction_at = source.prediction_known_at
            sample = PathProbabilitySample(
                competition_set_id=source.competition_set_id,
                prediction_known_at=prediction_at,
                outcome_known_at=archive.outcome_known_at,
                realized_path=archive.realized_path,
                raw_probabilities=probabilities,
                split_role="development_cross_fit",
                fold_id=fold_id,
            )
            path_samples.append(sample)
            path_rows.append(
                {
                    "schema_version": "phase7_w1_inner_path_prediction_v1",
                    "competition_set_id": source.competition_set_id,
                    "prediction_known_at": prediction_at,
                    "outcome_known_at": archive.outcome_known_at,
                    "realized_path": archive.realized_path,
                    "probabilities": dict(probabilities),
                    "fold_id": fold_id,
                    "held_session": held_session,
                    "train_sessions": train_sessions,
                    "whole_generation_split": True,
                    "formal_rolling_oof": False,
                    "authority": "development_inner_crossfit_only",
                }
            )

        train_dol = tuple(
            row for row in dol_inputs if row.competition_set_id in train_ids
        )
        held_dol = tuple(
            row for row in dol_inputs if row.competition_set_id in held_ids
        )
        try:
            fold_dol = fit_dol_softmax(
                tuple(
                    DOLChoiceSetSample(
                        competition_set_id=row.competition_set_id,
                        candidate_set_id=row.candidate_set_id,
                        path=row.path,
                        prediction_known_at=row.prediction_known_at,
                        outcome_known_at=row.outcome_known_at,
                        candidate_scores=row.candidate_scores,
                        outcome_id=row.outcome_id,
                        split_role="development_fit",
                        fold_id=fold_id,
                    )
                    for row in train_dol
                ),
                source_dataset_sha256=source_dataset_sha256,
                manifest_sha256=manifest_sha256,
                source_ranking_fingerprint=DOL_RANKING_FINGERPRINT,
            )
        except ValueError as error:
            blockers.append(f"{held_session}:dol_softmax:{error}")
            continue
        for row in held_dol:
            dol_rows.append(
                {
                    "schema_version": "phase7_w1_inner_dol_prediction_v1",
                    "competition_set_id": row.competition_set_id,
                    "candidate_set_id": row.candidate_set_id,
                    "path": row.path,
                    "prediction_known_at": row.prediction_known_at,
                    "outcome_known_at": row.outcome_known_at,
                    "outcome_id": row.outcome_id,
                    "probabilities": fold_dol.probabilities(
                        row.path, row.candidate_scores
                    ),
                    "fold_id": fold_id,
                    "held_session": held_session,
                    "train_sessions": train_sessions,
                    "whole_generation_split": True,
                    "formal_rolling_oof": False,
                    "authority": "development_inner_crossfit_only",
                }
            )
    return {
        "path_calibration_samples": tuple(path_samples),
        "path_predictions": tuple(path_rows),
        "dol_predictions": tuple(dol_rows),
        "session_order": session_order,
        "blockers": tuple(sorted(blockers)),
        "whole_session_split": True,
        "whole_generation_split": True,
        "formal_rolling_oof": False,
    }


def _inner_crossfit_path_records(
    cohorts: MaterializedCohorts,
    inner_crossfit: Mapping[str, Any],
    calibration_artifact: Any | None,
) -> tuple[ProbabilityPrediction, ...]:
    seeds = {
        (row.competition_set_id, row.prediction_known_at): row
        for row in cohorts.predictions
        if row.window_id == "W1"
    }
    output: list[ProbabilityPrediction] = []
    for row in inner_crossfit["path_predictions"]:
        key = (row["competition_set_id"], row["prediction_known_at"])
        seed = seeds[key]
        probabilities = dict(row["probabilities"])
        if calibration_artifact is not None:
            probabilities = dict(calibration_artifact.probabilities(probabilities))
        output.append(
            ProbabilityPrediction(
                unit_id=canonical_identity(
                    "w1-inner-path-prediction-unit",
                    {
                        "competition_set_id": row["competition_set_id"],
                        "prediction_known_at": row["prediction_known_at"],
                    },
                ),
                cluster_id=seed.cluster_id,
                fold_id=row["fold_id"],
                realized_label=row["realized_path"],
                support_label=row["realized_path"],
                model_probabilities=tuple(probabilities.items()),
                baseline_probabilities=seed.raw_probabilities,
            )
        )
    return tuple(output)


def _inner_crossfit_dol_records(
    cohorts: MaterializedCohorts,
    inner_crossfit: Mapping[str, Any],
) -> tuple[ProbabilityPrediction, ...]:
    groups = {row.candidate_set_id: row for row in cohorts.dol_groups}
    output: list[ProbabilityPrediction] = []
    for row in inner_crossfit["dol_predictions"]:
        group = groups[row["candidate_set_id"]]
        labels = tuple(identity for identity, _ in group.candidate_scores) + (
            NO_TARGET_OUTCOME,
        )
        output.append(
            ProbabilityPrediction(
                unit_id=row["candidate_set_id"],
                cluster_id=group.cluster_id,
                fold_id=row["fold_id"],
                realized_label=row["outcome_id"],
                support_label=dol_support_label(row["outcome_id"]),
                model_probabilities=tuple(row["probabilities"].items()),
                baseline_probabilities=tuple(
                    (label, 1.0 / len(labels)) for label in labels
                ),
            )
        )
    return tuple(output)


def _dol_temperature_lineage(
    dol_artifact: Any,
    inner_crossfit_artifact_id: str,
) -> DOLModelLineage:
    """Bind one model-family identity across W1 cross-fit and W2 inference."""

    path_protocol = load_path_belief_protocol(ROOT / "configs/path_hypotheses.json")
    source_model_fingerprint = canonical_sha256(
        {
            "procedure": "phase7_w1_dol_softmax_inner_crossfit_v1",
            "final_w1_model_artifact_id": dol_artifact.artifact_id,
            "inner_crossfit_artifact_id": inner_crossfit_artifact_id,
            "source_ranking_fingerprint": DOL_RANKING_FINGERPRINT,
        }
    )
    return DOLModelLineage(
        source_dol_protocol_fingerprint=DOL_PROTOCOL_FINGERPRINT,
        source_dol_model_version=dol_artifact.model_version,
        source_dol_model_fingerprint=source_model_fingerprint,
        source_path_protocol_fingerprint=PATH_PROTOCOL_FINGERPRINT,
        source_path_model_version=path_protocol.model_version,
    )


def _dol_temperature_observation(
    *,
    candidate_set_id: str,
    competition_set_id: str,
    prediction_known_at: pd.Timestamp,
    outcome_known_at: pd.Timestamp,
    probabilities: Mapping[str, float],
    outcome_id: str,
    window_id: str,
    fold_id: str,
    cluster_id: str,
    source_model_fingerprint: str,
) -> DOLTemperatureObservation:
    return DOLTemperatureObservation(
        # The signal-fit contract requires one unique case per distribution.
        # Candidate-set identity is the frozen DOL competition unit; the Path
        # competition identity remains bound into the outcome-event identity.
        case_id=candidate_set_id,
        prediction_id=candidate_set_id,
        outcome_event_id=canonical_identity(
            "phase7-dol-temperature-outcome",
            {
                "candidate_set_id": candidate_set_id,
                "competition_set_id": competition_set_id,
                "outcome_id": outcome_id,
                "outcome_known_at": outcome_known_at,
            },
        ),
        window_id=window_id,
        fold_id=fold_id,
        cluster_id=cluster_id,
        prediction_known_at=prediction_known_at,
        outcome_known_at=outcome_known_at,
        outcome_probabilities=tuple(sorted(probabilities.items())),
        realized_outcome_id=outcome_id,
        support_label=(
            DOLSupportLabel.NO_TARGET
            if outcome_id == NO_TARGET_OUTCOME
            else DOLSupportLabel.CANDIDATE_TARGET
        ),
        source_dol_model_fingerprint=source_model_fingerprint,
    )


def _inner_dol_temperature_observations(
    cohorts: MaterializedCohorts,
    inner_crossfit: Mapping[str, Any],
    *,
    source_model_fingerprint: str,
) -> tuple[DOLTemperatureObservation, ...]:
    groups = {group.candidate_set_id: group for group in cohorts.dol_groups}
    return tuple(
        _dol_temperature_observation(
            candidate_set_id=row["candidate_set_id"],
            competition_set_id=row["competition_set_id"],
            prediction_known_at=row["prediction_known_at"],
            outcome_known_at=row["outcome_known_at"],
            probabilities=row["probabilities"],
            outcome_id=row["outcome_id"],
            window_id="W1",
            fold_id=row["fold_id"],
            cluster_id=groups[row["candidate_set_id"]].cluster_id,
            source_model_fingerprint=source_model_fingerprint,
        )
        for row in inner_crossfit["dol_predictions"]
    )


def _window_dol_temperature_observations(
    cohorts: MaterializedCohorts,
    dol_artifact: Any,
    *,
    window_id: str,
    source_model_fingerprint: str,
) -> tuple[DOLTemperatureObservation, ...]:
    groups = {group.candidate_set_id: group for group in cohorts.dol_groups}
    return tuple(
        _dol_temperature_observation(
            candidate_set_id=row.candidate_set_id,
            competition_set_id=row.competition_set_id,
            prediction_known_at=row.prediction_known_at,
            outcome_known_at=row.outcome_known_at,
            probabilities=dol_artifact.probabilities(
                row.path, row.candidate_scores
            ),
            outcome_id=row.outcome_id,
            window_id=window_id,
            fold_id=row.fold_id,
            cluster_id=groups[row.candidate_set_id].cluster_id,
            source_model_fingerprint=source_model_fingerprint,
        )
        for row in _dol_prediction_inputs(cohorts, window_id=window_id)
    )


def _signal_outcome_cohort(
    observations: Sequence[DOLTemperatureObservation],
    *,
    cohort_name: str,
    cohort_role: str,
    window_id: str,
    source_dataset_sha256: str,
    manifest_sha256: str,
    split_protocol_sha256: str,
    eligible_units: int,
) -> SignalOutcomeCohort:
    if not observations:
        raise SignalOutcomeFitError("signal outcome cohort has zero observations")
    if eligible_units < len(observations):
        raise SignalOutcomeFitError("signal outcome cohort exceeds eligible units")
    return SignalOutcomeCohort(
        cohort_name=cohort_name,
        cohort_role=cohort_role,
        window_ids=(window_id,),
        fold_ids=tuple(sorted({row.fold_id for row in observations})),
        source_dataset_sha256=source_dataset_sha256,
        manifest_sha256=manifest_sha256,
        cohort_identity_sha256=canonical_sha256(
            {
                "cohort_name": cohort_name,
                "cohort_role": cohort_role,
                "observation_ids": tuple(
                    sorted(row.observation_id for row in observations)
                ),
            }
        ),
        split_protocol_sha256=split_protocol_sha256,
        prediction_coverage=(
            0.0 if eligible_units == 0 else len(observations) / eligible_units
        ),
    )


def _closed_signal_outcome_receipt(
    *,
    artifact_kind: str,
    model_artifact_id: str,
    source_dataset_sha256: str,
    manifest_sha256: str,
    fit_cohort_role: str,
    validation_cohort_role: str,
    blockers: Sequence[str],
    fit_status: str | None = None,
) -> Mapping[str, Any]:
    resolved_fit_status = fit_status or (
        "not_fitted"
        if model_artifact_id.startswith("unavailable:")
        else "fitted_not_admitted"
    )
    payload = {
        "schema_version": "phase7_signal_outcome_not_evaluable_v1",
        "artifact_kind": artifact_kind,
        "model_artifact_id": model_artifact_id,
        "source_dataset_sha256": source_dataset_sha256,
        "manifest_sha256": manifest_sha256,
        "fit_cohort_role": fit_cohort_role,
        "validation_cohort_role": validation_cohort_role,
        "fit_status": resolved_fit_status,
        "admitted": False,
        "status": "CLOSED",
        "blockers": tuple(sorted(set(blockers))),
        "authority": "shadow_evidence_only",
        "action_authority": False,
        "pins_issued": False,
    }
    return {
        **payload,
        "receipt_id": canonical_identity("phase7-signal-outcome-closed", payload),
    }


def _delivery_no_fit_payload(
    *,
    source_dataset_sha256: str,
    manifest_sha256: str,
) -> Mapping[str, Any]:
    payload = {
        "schema_version": "phase7_delivery_no_fit_v1",
        "estimand": "target_before_invalidation",
        "setup_families": ("dfp", "lsr"),
        "fit_status": "no_fit_blocked",
        "admission_status": "CLOSED",
        "blocker": "FORMAL_NONZERO_INTENT_LEDGER_EMPTY",
        "formal_nonzero_intent_case_count": 0,
        "fabricated_setup_row_count": 0,
        "source_dataset_sha256": source_dataset_sha256,
        "manifest_sha256": manifest_sha256,
        "authority": "research_only",
        "action_authority": False,
        "pins_issued": False,
    }
    return {
        **payload,
        "artifact_id": canonical_identity("phase7-delivery-no-fit", payload),
    }


def _admission_or_blocked(
    records: Sequence[ProbabilityPrediction],
    *,
    artifact_kind: str,
    artifact_id: str,
    source_dataset_sha256: str,
    manifest_sha256: str,
    cohort_identity_sha256: str,
    cohort_role: str,
    coverage: float,
    required_support: Sequence[str],
    thresholds: AdmissionThresholds,
) -> Mapping[str, Any]:
    try:
        return evaluate_probability_admission(
            records,
            artifact_kind=artifact_kind,
            model_artifact_id=artifact_id,
            source_dataset_sha256=source_dataset_sha256,
            manifest_sha256=manifest_sha256,
            cohort_identity_sha256=cohort_identity_sha256,
            cohort_role=cohort_role,
            prediction_coverage=coverage,
            required_support_labels=required_support,
            thresholds=thresholds,
        ).to_dict()
    except ValueError as error:
        blockers = [f"ADMISSION_NOT_EVALUABLE:{type(error).__name__}:{error}"]
        if cohort_role != "rolling_oof":
            blockers.append("COHORT_NOT_ROLLING_OOF")
        return {
            "schema_version": "phase7_admission_not_evaluable_v1",
            "artifact_kind": artifact_kind,
            "model_artifact_id": artifact_id,
            "source_dataset_sha256": source_dataset_sha256,
            "manifest_sha256": manifest_sha256,
            "cohort_identity_sha256": cohort_identity_sha256,
            "cohort_role": cohort_role,
            "admitted": False,
            "status": "not_evaluable_shadow",
            "blockers": sorted(blockers),
            "authority": "shadow_only",
            "action_authority": False,
        }


def fit_and_validate(
    manifest: RunManifest,
    cohorts: MaterializedCohorts,
    *,
    include_rolling_diagnostics: bool,
) -> Mapping[str, Any]:
    if not manifest.fit_and_validate_authorized:
        raise Phase7PipelineError("run manifest does not authorize fitting")
    windows = {spec.window_id for spec in manifest.inputs}
    if not {"W1", "W2"}.issubset(windows):
        raise Phase7PipelineError("fit-and-validate requires W1 fit and W2 validation")
    if include_rolling_diagnostics and not {"W3", "W4"}.issubset(windows):
        raise Phase7PipelineError(
            "--include-rolling-diagnostics requires both registered W3 and W4"
        )
    dataset_sha = canonical_sha256(
        {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "inputs": tuple(
                (spec.window_id, spec.source_sha256, spec.row_count)
                for spec in manifest.inputs
                if include_rolling_diagnostics
                or not spec.registered.rolling_diagnostic_only
            ),
        }
    )
    protocol, protocol_sha256 = _load_json_with_sha256(PHASE7_PROTOCOL)
    if protocol_sha256 != manifest.phase7_protocol_sha256:
        raise Phase7PipelineError("Phase-7 protocol changed before fitting")
    thresholds_raw = protocol["admission_thresholds"]
    thresholds = AdmissionThresholds(
        minimum_resolved_units=int(thresholds_raw["minimum_resolved_units"]),
        minimum_support_units=int(
            thresholds_raw["minimum_units_per_required_support"]
        ),
        minimum_prediction_coverage=float(
            thresholds_raw["minimum_prediction_coverage"]
        ),
        maximum_ece=float(thresholds_raw["maximum_top_label_ece"]),
        maximum_fold_log_loss_degradation=float(
            thresholds_raw["maximum_single_fold_log_loss_degradation"]
        ),
        minimum_improving_fold_fraction=float(
            thresholds_raw["minimum_improving_fold_fraction"]
        ),
        bootstrap_confidence=float(thresholds_raw["bootstrap_confidence"]),
        bootstrap_replicates=int(thresholds_raw["bootstrap_replicates"]),
        bootstrap_seed=int(thresholds_raw["bootstrap_seed"]),
        probability_tolerance=float(thresholds_raw["probability_sum_tolerance"]),
    )
    fit_rows = tuple(
        row
        for row in cohorts.history_transitions
        if row.split_role == "development_fit"
    )
    fit_intervals = tuple(
        row for row in cohorts.risk_intervals if row.split_role == "development_fit"
    )
    artifacts: dict[str, Any] = {}
    blockers: dict[str, str] = {}

    base_prior = None
    try:
        base_prior = fit_path_base_rate_prior(
            cohorts,
            source_dataset_sha256=dataset_sha,
            manifest_sha256=manifest.source_sha256,
        )
        artifacts["path_base_rate_prior"] = base_prior
    except ValueError as error:
        blockers["path_base_rate_prior"] = str(error)

    history = None
    try:
        history = fit_history_conditional_likelihood(
            fit_rows,
            source_dataset_sha256=dataset_sha,
            manifest_sha256=manifest.source_sha256,
        )
        artifacts["history_likelihood"] = _artifact_payload(history)
        artifacts["history_transition_rules"] = history_transition_rules_payload(history)
    except ValueError as error:
        blockers["history_likelihood"] = str(error)

    temporal = None
    try:
        temporal = fit_competing_risk_life_table(
            fit_intervals,
            source_dataset_sha256=dataset_sha,
            manifest_sha256=manifest.source_sha256,
        )
        artifacts["path_temporal"] = _artifact_payload(temporal)
    except ValueError as error:
        blockers["path_temporal"] = str(error)

    reversion = None
    if history is not None and base_prior is not None:
        try:
            w1_inputs = _reversion_inputs(
                cohorts,
                history,
                windows={"W1"},
                fitted_prior_log_weights=base_prior["prior_log_weights"],
            )
            reversion = fit_evidence_prior_reversion(
                _prior_fit_samples(w1_inputs),
                source_dataset_sha256=dataset_sha,
                manifest_sha256=manifest.source_sha256,
            )
            artifacts["prior_reversion"] = _artifact_payload(reversion)
        except ValueError as error:
            blockers["prior_reversion"] = str(error)
    else:
        blockers["prior_reversion"] = (
            "requires fitted W1 base-rate prior and history-transition likelihood"
        )

    dol = None
    try:
        dol = fit_dol_softmax(
            _dol_choice_samples(cohorts, windows={"W1"}),
            source_dataset_sha256=dataset_sha,
            manifest_sha256=manifest.source_sha256,
            source_ranking_fingerprint=DOL_RANKING_FINGERPRINT,
        )
        artifacts["dol_probability"] = _artifact_payload(dol)
    except ValueError as error:
        blockers["dol_probability"] = str(error)

    inner_crossfit = _w1_inner_rolling_crossfit(
        cohorts,
        source_dataset_sha256=dataset_sha,
        manifest_sha256=manifest.source_sha256,
    )
    inner_payload = {
        key: value
        for key, value in inner_crossfit.items()
        if key != "path_calibration_samples"
    }
    inner_payload = {
        "schema_version": "phase7_w1_inner_crossfit_bundle_v1",
        **inner_payload,
        "source_dataset_sha256": dataset_sha,
        "manifest_sha256": manifest.source_sha256,
        "artifact_id": canonical_identity(
            "w1-inner-crossfit-bundle",
            {
                "source_dataset_sha256": dataset_sha,
                "manifest_sha256": manifest.source_sha256,
                "path_predictions": inner_crossfit["path_predictions"],
                "dol_predictions": inner_crossfit["dol_predictions"],
                "session_order": inner_crossfit["session_order"],
                "blockers": inner_crossfit["blockers"],
            },
        ),
        "authority": "development_inner_crossfit_only",
        "formal_rolling_oof": False,
        "action_authority": False,
    }
    artifacts["w1_inner_crossfit_predictions"] = inner_payload

    calibration = None
    try:
        calibration_samples = tuple(inner_crossfit["path_calibration_samples"])
        if not calibration_samples:
            raise ValueError("W1 inner rolling cross-fit produced zero path predictions")
        calibration = fit_path_temperature_bias(
            calibration_samples,
            source_dataset_sha256=dataset_sha,
            manifest_sha256=manifest.source_sha256,
            source_model_artifact_id=inner_payload["artifact_id"],
        )
        artifacts["path_calibration"] = _artifact_payload(calibration)
    except ValueError as error:
        blockers["path_calibration"] = str(error)

    signal_admissions: dict[str, Any] = {}
    signal_thresholds = SignalOutcomeAdmissionThresholds(
        minimum_resolved_units=thresholds.minimum_resolved_units,
        minimum_units_per_support=thresholds.minimum_support_units,
        minimum_prediction_coverage=thresholds.minimum_prediction_coverage,
        maximum_ece=thresholds.maximum_ece,
        minimum_improving_fold_fraction=thresholds.minimum_improving_fold_fraction,
        maximum_single_fold_log_loss_degradation=(
            thresholds.maximum_fold_log_loss_degradation
        ),
    )
    dol_temperature = None
    inner_dol_rows = tuple(inner_crossfit["dol_predictions"])
    if not inner_dol_rows:
        reason = "W1_INNER_CROSSFIT_DOL_ROWS_EMPTY"
        blockers["dol_temperature_calibration"] = reason
        signal_admissions["dol_temperature_W2"] = _closed_signal_outcome_receipt(
            artifact_kind="dol_probability_calibration",
            model_artifact_id="unavailable:dol_temperature",
            source_dataset_sha256=dataset_sha,
            manifest_sha256=manifest.source_sha256,
            fit_cohort_role="development_cross_fit",
            validation_cohort_role="historical_validation",
            blockers=(reason,),
        )
    elif dol is None:
        reason = "W1_DOL_SOFTMAX_ARTIFACT_UNAVAILABLE"
        blockers["dol_temperature_calibration"] = reason
        signal_admissions["dol_temperature_W2"] = _closed_signal_outcome_receipt(
            artifact_kind="dol_probability_calibration",
            model_artifact_id="unavailable:dol_temperature",
            source_dataset_sha256=dataset_sha,
            manifest_sha256=manifest.source_sha256,
            fit_cohort_role="development_cross_fit",
            validation_cohort_role="historical_validation",
            blockers=(reason,),
        )
    else:
        try:
            dol_lineage = _dol_temperature_lineage(
                dol,
                inner_payload["artifact_id"],
            )
            inner_temperature_rows = _inner_dol_temperature_observations(
                cohorts,
                inner_crossfit,
                source_model_fingerprint=(
                    dol_lineage.source_dol_model_fingerprint
                ),
            )
            fit_cohort = _signal_outcome_cohort(
                inner_temperature_rows,
                cohort_name="phase7_W1_inner_dol_temperature",
                cohort_role="development_cross_fit",
                window_id="W1",
                source_dataset_sha256=dataset_sha,
                manifest_sha256=manifest.source_sha256,
                split_protocol_sha256=manifest.phase7_protocol_sha256,
                eligible_units=sum(
                    group.window_id == "W1" for group in cohorts.dol_groups
                ),
            )
            dol_temperature = fit_dol_temperature(
                inner_temperature_rows,
                cohort=fit_cohort,
                lineage=dol_lineage,
            )
            artifacts["dol_temperature_calibration"] = dol_temperature.to_payload()

            w2_temperature_rows = _window_dol_temperature_observations(
                cohorts,
                dol,
                window_id="W2",
                source_model_fingerprint=(
                    dol_lineage.source_dol_model_fingerprint
                ),
            )
            if not w2_temperature_rows:
                reason = "W2_DOL_HISTORICAL_VALIDATION_ROWS_EMPTY"
                blockers["dol_temperature_validation"] = reason
                signal_admissions["dol_temperature_W2"] = (
                    _closed_signal_outcome_receipt(
                        artifact_kind="dol_probability_calibration",
                        model_artifact_id=dol_temperature.artifact_id,
                        source_dataset_sha256=dataset_sha,
                        manifest_sha256=manifest.source_sha256,
                        fit_cohort_role="development_cross_fit",
                        validation_cohort_role="historical_validation",
                        blockers=(reason,),
                    )
                )
            else:
                validation_cohort = _signal_outcome_cohort(
                    w2_temperature_rows,
                    cohort_name="phase7_W2_dol_temperature_validation",
                    cohort_role="historical_validation",
                    window_id="W2",
                    source_dataset_sha256=dataset_sha,
                    manifest_sha256=manifest.source_sha256,
                    split_protocol_sha256=manifest.phase7_protocol_sha256,
                    eligible_units=sum(
                        group.window_id == "W2" for group in cohorts.dol_groups
                    ),
                )
                signal_admissions["dol_temperature_W2"] = (
                    evaluate_dol_temperature_admission(
                        dol_temperature,
                        w2_temperature_rows,
                        cohort=validation_cohort,
                        thresholds=signal_thresholds,
                    ).to_payload()
                )
        except ValueError as error:
            reason = f"DOL_TEMPERATURE_NOT_EVALUABLE:{type(error).__name__}:{error}"
            blocker_key = (
                "dol_temperature_calibration"
                if dol_temperature is None
                else "dol_temperature_validation"
            )
            blockers[blocker_key] = reason
            signal_admissions["dol_temperature_W2"] = (
                _closed_signal_outcome_receipt(
                    artifact_kind="dol_probability_calibration",
                    model_artifact_id=(
                        "unavailable:dol_temperature"
                        if dol_temperature is None
                        else dol_temperature.artifact_id
                    ),
                    source_dataset_sha256=dataset_sha,
                    manifest_sha256=manifest.source_sha256,
                    fit_cohort_role="development_cross_fit",
                    validation_cohort_role="historical_validation",
                    blockers=(reason,),
                )
            )

    delivery_no_fit = _delivery_no_fit_payload(
        source_dataset_sha256=dataset_sha,
        manifest_sha256=manifest.source_sha256,
    )
    artifacts["delivery_target_before_invalidation"] = delivery_no_fit
    blockers["delivery_target_before_invalidation"] = (
        "FORMAL_NONZERO_INTENT_LEDGER_EMPTY"
    )
    signal_admissions["delivery_target_before_invalidation"] = (
        _closed_signal_outcome_receipt(
            artifact_kind="target_before_invalidation_model",
            model_artifact_id=delivery_no_fit["artifact_id"],
            source_dataset_sha256=dataset_sha,
            manifest_sha256=manifest.source_sha256,
            fit_cohort_role="development_fit",
            validation_cohort_role="historical_validation",
            blockers=("FORMAL_NONZERO_INTENT_LEDGER_EMPTY",),
            fit_status="no_fit_blocked",
        )
    )

    if temporal is not None:
        runtime_candidate = temporal_runtime_candidate_payload(
            temporal,
            reversion,
            minimum_at_risk_intervals_per_cell=thresholds.minimum_support_units,
        )
        artifacts["temporal_runtime_candidate"] = runtime_candidate
        incomplete_paths = tuple(
            path
            for path in PATH_LABELS
            if path != PathKind.RESIDUAL_UNKNOWN.value
            and runtime_candidate["hypothesis_expiry_bars"][path] is None
        )
        if reversion is None or incomplete_paths:
            reasons = []
            if reversion is None:
                reasons.append("prior_reversion_artifact_unavailable")
            if incomplete_paths:
                reasons.append(
                    "expiry_support_blocked_paths=" + ",".join(incomplete_paths)
                )
            blockers["temporal_runtime_candidate"] = ";".join(reasons)

    admissions: dict[str, Any] = dict(signal_admissions)
    validation_predictions: dict[str, dict[str, Sequence[ProbabilityPrediction]]] = {}
    if base_prior is not None:
        for window_id in ("W2", "W4"):
            if window_id not in windows or (window_id == "W4" and not include_rolling_diagnostics):
                continue
            records = (
                _path_predictions(
                    cohorts,
                    reversion,
                    history,
                    window_id=window_id,
                    calibration_artifact=calibration,
                    fitted_prior_log_weights=base_prior["prior_log_weights"],
                )
                if history is not None and reversion is not None
                else _base_rate_predictions(
                    cohorts,
                    base_prior,
                    window_id=window_id,
                    calibration_artifact=calibration,
                )
            )
            candidates = sum(
                seed.window_id == window_id for seed in cohorts.predictions
            )
            validation_predictions.setdefault(window_id, {})[
                "path_probability"
            ] = records
            role = REGISTERED_WINDOWS[window_id].split_role
            admissions[f"path_probability_{window_id}"] = _admission_or_blocked(
                records,
                artifact_kind="path_probability",
                artifact_id=(
                    calibration.artifact_id
                    if calibration is not None
                    else (
                        reversion.artifact_id
                        if reversion is not None
                        else base_prior["artifact_id"]
                    )
                ),
                source_dataset_sha256=dataset_sha,
                manifest_sha256=manifest.source_sha256,
                cohort_identity_sha256=canonical_sha256(
                    {"window_id": window_id, "records": records}
                ),
                cohort_role=role,
                coverage=(0.0 if candidates == 0 else len(records) / candidates),
                required_support=PATH_LABELS,
                thresholds=thresholds,
            )
    if dol is not None:
        for window_id in ("W2", "W4"):
            if window_id not in windows or (window_id == "W4" and not include_rolling_diagnostics):
                continue
            records = _dol_predictions(cohorts, dol, window_id=window_id)
            candidate_sets = sum(
                group.window_id == window_id for group in cohorts.dol_groups
            )
            validation_predictions.setdefault(window_id, {})[
                "dol_probability"
            ] = records
            admissions[f"dol_probability_{window_id}"] = _admission_or_blocked(
                records,
                artifact_kind="dol_probability",
                artifact_id=dol.artifact_id,
                source_dataset_sha256=dataset_sha,
                manifest_sha256=manifest.source_sha256,
                cohort_identity_sha256=canonical_sha256(
                    {"window_id": window_id, "records": records}
                ),
                cohort_role=REGISTERED_WINDOWS[window_id].split_role,
                coverage=(
                    0.0 if candidate_sets == 0 else len(records) / candidate_sets
                ),
                required_support=("first_hit_candidate", NO_TARGET_OUTCOME),
                thresholds=thresholds,
            )
    validation_windows = tuple(
        window_id
        for window_id in ("W2", "W4")
        if window_id in windows
        and (window_id != "W4" or include_rolling_diagnostics)
    )
    validation_payload = {
        "schema_version": "phase7_validation_predictions_v1",
        "source_dataset_sha256": dataset_sha,
        "manifest_sha256": manifest.source_sha256,
        "source_model_artifact_ids": tuple(
            sorted(
                {
                    artifact_id
                    for artifact_id in (
                        None if base_prior is None else base_prior["artifact_id"],
                        None if reversion is None else reversion.artifact_id,
                        None if calibration is None else calibration.artifact_id,
                        None if dol is None else dol.artifact_id,
                        (
                            None
                            if dol_temperature is None
                            else dol_temperature.artifact_id
                        ),
                    )
                    if artifact_id is not None
                }
            )
        ),
        "windows": {
            window_id: {
                kind: tuple(asdict(row) for row in records)
                for kind, records in sorted(
                    validation_predictions.get(window_id, {}).items()
                )
            }
            for window_id in validation_windows
        },
        "cohort_roles": {
            window_id: REGISTERED_WINDOWS[window_id].split_role
            for window_id in validation_windows
        },
        "formal_rolling_oof": False,
        "purpose": (
            "independent_historical_validation_and_downstream_"
            "development_calibration_input"
        ),
        "authority": "research_shadow_only",
        "action_authority": False,
    }
    validation_payload = {
        **validation_payload,
        "artifact_id": canonical_identity(
            "phase7-validation-predictions", validation_payload
        ),
    }
    artifacts["validation_predictions"] = validation_payload
    if not any(
        records
        for value in validation_predictions.values()
        for records in value.values()
    ):
        blockers["validation_predictions"] = (
            "no_supported_W2_or_optional_W4_validation_predictions"
        )
    inner_path_records = _inner_crossfit_path_records(
        cohorts,
        inner_crossfit,
        calibration,
    )
    admissions["path_probability_W1_inner_development"] = _admission_or_blocked(
        inner_path_records,
        artifact_kind="path_probability",
        artifact_id=(
            calibration.artifact_id
            if calibration is not None
            else inner_payload["artifact_id"]
        ),
        source_dataset_sha256=dataset_sha,
        manifest_sha256=manifest.source_sha256,
        cohort_identity_sha256=canonical_sha256(
            {"window_id": "W1-inner", "records": inner_path_records}
        ),
        cohort_role="development_cross_fit",
        coverage=(1.0 if inner_path_records else 0.0),
        required_support=PATH_LABELS,
        thresholds=thresholds,
    )
    inner_dol_records = _inner_crossfit_dol_records(cohorts, inner_crossfit)
    admissions["dol_probability_W1_inner_development"] = _admission_or_blocked(
        inner_dol_records,
        artifact_kind="dol_probability",
        artifact_id=inner_payload["artifact_id"],
        source_dataset_sha256=dataset_sha,
        manifest_sha256=manifest.source_sha256,
        cohort_identity_sha256=canonical_sha256(
            {"window_id": "W1-inner", "records": inner_dol_records}
        ),
        cohort_role="development_cross_fit",
        coverage=(1.0 if inner_dol_records else 0.0),
        required_support=("first_hit_candidate", NO_TARGET_OUTCOME),
        thresholds=thresholds,
    )
    for window_id in ("W2", "W4"):
        if window_id not in windows or (
            window_id == "W4" and not include_rolling_diagnostics
        ):
            continue
        for kind in ("path_probability", "dol_probability"):
            key = f"{kind}_{window_id}"
            if key in admissions:
                continue
            admissions[key] = _admission_or_blocked(
                (),
                artifact_kind=kind,
                artifact_id=f"unavailable:{kind}",
                source_dataset_sha256=dataset_sha,
                manifest_sha256=manifest.source_sha256,
                cohort_identity_sha256=canonical_sha256(
                    {"window_id": window_id, "records": ()}
                ),
                cohort_role=REGISTERED_WINDOWS[window_id].split_role,
                coverage=0.0,
                required_support=(
                    PATH_LABELS
                    if kind == "path_probability"
                    else ("first_hit_candidate", NO_TARGET_OUTCOME)
                ),
                thresholds=thresholds,
            )
    artifact_statuses: dict[str, Mapping[str, Any]] = {}
    for name in (
        "path_base_rate_prior",
        "history_likelihood",
        "history_transition_rules",
        "path_temporal",
        "temporal_runtime_candidate",
        "prior_reversion",
        "path_calibration",
        "dol_probability",
        "dol_temperature_calibration",
        "delivery_target_before_invalidation",
        "w1_inner_crossfit_predictions",
        "validation_predictions",
    ):
        if name not in artifacts:
            artifact_statuses[name] = {
                "status": "blocked",
                "reason": blockers.get(name, "unavailable"),
            }
        elif name in blockers:
            artifact_statuses[name] = {
                "status": "blocked_with_research_artifact",
                "artifact_id": artifacts[name].get("artifact_id"),
                "reason": blockers[name],
            }
        else:
            artifact_statuses[name] = {
                "status": "fitted_not_admitted",
                "artifact_id": artifacts[name].get("artifact_id"),
            }
    return {
        "source_dataset_sha256": dataset_sha,
        "artifacts": artifacts,
        "artifact_blockers": dict(sorted(blockers.items())),
        "artifact_statuses": artifact_statuses,
        "admission_receipts": admissions,
        "admission_gate": {
            "state": "closed",
            "reasons": [
                "W1_IS_INNER_DEVELOPMENT_NOT_FORMAL_ROLLING_OOF",
                "W2_IS_HISTORICAL_VALIDATION_NOT_ROLLING_OOF",
                "JUNE_2024_ALREADY_OPENED_DEVELOPMENT",
                "NO_ACTION_AUTHORITY",
            ],
        },
        "history_runtime_integration": (
            "explicit_history_state_transition_rules_not_legacy_marginal_rules"
        ),
        "production_config_updated": False,
        "empirical_authority": False,
        "action_authority": False,
    }


_COHORT_FILES = {
    "competition_archive": "cohorts/path_competition_archive.jsonl",
    "path_risk_intervals": "cohorts/path_risk_intervals.jsonl",
    "evidence_history_transitions": "cohorts/evidence_history_transitions.jsonl",
    "dol_candidate_snapshots": "cohorts/dol_candidate_snapshots.jsonl",
    "dol_candidate_outcomes": "cohorts/dol_candidate_outcomes.jsonl",
    "path_prediction_seeds": "cohorts/path_prediction_seeds.jsonl",
    "dol_choice_sets": "cohorts/dol_choice_sets.jsonl",
}


def _write_staged_file(root: Path, relative: str, data: bytes) -> str:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_file(destination)


def _verify_staging_inventory(
    root: Path,
    expected_files: Mapping[str, str],
) -> None:
    """Require the staging tree to contain exactly the hash-bound payloads."""

    try:
        root_metadata = os.stat(root, follow_symlinks=False)
    except OSError as error:
        raise Phase7PipelineError("Phase-7 staging directory is unavailable") from error
    if not stat_module.S_ISDIR(root_metadata.st_mode):
        raise Phase7PipelineError("Phase-7 staging root must be a direct directory")
    expected: dict[str, str] = {}
    expected_directories: set[str] = set()
    for relative, digest in expected_files.items():
        candidate = Path(relative)
        if (
            not relative
            or candidate.is_absolute()
            or candidate.as_posix() != relative
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise Phase7PipelineError("staging inventory path is not canonical")
        expected[relative] = _sha256(digest, name=f"{relative} sha256")
        expected_directories.update(
            parent.as_posix()
            for parent in candidate.parents
            if parent != Path(".")
        )

    actual_files: dict[str, Path] = {}
    actual_directories: set[str] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        try:
            metadata = os.stat(candidate, follow_symlinks=False)
        except OSError as error:
            raise Phase7PipelineError("staging inventory changed while reading") from error
        if stat_module.S_ISDIR(metadata.st_mode):
            actual_directories.add(relative)
        elif stat_module.S_ISREG(metadata.st_mode):
            actual_files[relative] = candidate
        else:
            raise Phase7PipelineError(
                f"staging inventory contains a non-regular entry: {relative}"
            )

    if set(actual_files) != set(expected) or actual_directories != expected_directories:
        raise Phase7PipelineError("staging inventory differs from expected payload set")
    for relative, candidate in actual_files.items():
        if _sha256_file(candidate) != expected[relative]:
            raise Phase7PipelineError(f"staging payload hash differs: {relative}")


def _cohort_rows(cohorts: MaterializedCohorts) -> Mapping[str, Sequence[Any]]:
    return {
        "competition_archive": cohorts.archives,
        "path_risk_intervals": cohorts.risk_intervals,
        "evidence_history_transitions": cohorts.history_transitions,
        "dol_candidate_snapshots": cohorts.dol_snapshots,
        "dol_candidate_outcomes": tuple(
            row.to_cohort_dict() for row in cohorts.dol_outcomes
        ),
        "path_prediction_seeds": cohorts.predictions,
        "dol_choice_sets": cohorts.dol_groups,
    }


def write_pipeline_outputs(
    destination: str | Path,
    *,
    manifest: RunManifest,
    cohorts: MaterializedCohorts,
    mode: str,
    include_rolling_diagnostics: bool,
    fitted: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Publish a complete run directory without replacing an existing path."""

    target = _resolve(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Phase-7 output already exists: {target}")
    if mode == "materialize-only":
        if fitted is not None:
            raise Phase7PipelineError(
                "materialize-only publication cannot contain fitted outputs"
            )
    elif mode == "fit-and-validate":
        if fitted is None:
            raise Phase7PipelineError(
                "fit-and-validate publication requires fitted outputs"
            )
        if not manifest.fit_and_validate_authorized:
            raise Phase7PipelineError("run manifest does not authorize fitting")
    else:
        raise Phase7PipelineError("unsupported Phase-7 pipeline mode")
    _revalidate_run_manifest(manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging.", dir=target.parent)
    )
    try:
        staged_inventory: dict[str, str] = {}
        cohort_bindings: dict[str, Any] = {}
        rows_by_name = _cohort_rows(cohorts)
        for name, relative in _COHORT_FILES.items():
            rows = tuple(rows_by_name[name])
            digest = _write_staged_file(staging, relative, _jsonl_bytes(rows))
            staged_inventory[relative] = digest
            cohort_bindings[name] = {
                "path": relative,
                "sha256": digest,
                "rows": len(rows),
            }
        cohort_manifest = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "run_id": manifest.run_id,
            "authority": AUTHORITY,
            "mode": mode,
            "shadow_input_schema_version": SHADOW_INPUT_SCHEMA_VERSION,
            "run_manifest_sha256": manifest.source_sha256,
            "repository_commit": manifest.repository_commit,
            "python_major_minor": manifest.python_major_minor,
            "code_bundle_identity": manifest.code_bundle_identity,
            "model_config_sha256": manifest.model_config_sha256,
            "phase7_protocol_sha256": manifest.phase7_protocol_sha256,
            "preregistration_sha256": manifest.preregistration_sha256,
            "foundation_identity": FOUNDATION_IDENTITY,
            "path_protocol_fingerprint": PATH_PROTOCOL_FINGERPRINT,
            "dol_protocol_fingerprint": DOL_PROTOCOL_FINGERPRINT,
            "dol_ranking_fingerprint": DOL_RANKING_FINGERPRINT,
            "inputs": [
                {
                    "window_id": spec.window_id,
                    "sha256": spec.source_sha256,
                    "rows": spec.row_count,
                    "split_role": spec.split_role,
                    "fold_id": spec.fold_id,
                    "rolling_diagnostic_only": spec.registered.rolling_diagnostic_only,
                    "rolling_oof": False,
                    "first_decision_clock": spec.registered.first_decision_clock,
                    "last_decision_clock": spec.registered.last_decision_clock,
                }
                for spec in manifest.inputs
                if include_rolling_diagnostics
                or not spec.registered.rolling_diagnostic_only
            ],
            "cohorts": cohort_bindings,
            "counts_by_window": cohorts.counts_by_window,
            "sealed_oos_opened": False,
            "empirical_authority": False,
            "action_authority": False,
        }
        cohort_manifest_sha = _write_staged_file(
            staging,
            "cohort_manifest.json",
            _canonical_bytes(cohort_manifest),
        )
        staged_inventory["cohort_manifest.json"] = cohort_manifest_sha
        artifact_bindings: dict[str, Any] = {}
        admission_bindings: dict[str, Any] = {}
        admissions: Mapping[str, Any] = {}
        artifact_blockers: Mapping[str, Any] = {}
        artifact_statuses: Mapping[str, Any] = {}
        admission_gate: Mapping[str, Any] = {
            "state": "closed",
            "reasons": ["MATERIALIZATION_ONLY_NO_EMPIRICAL_ADMISSION"],
        }
        if fitted is not None:
            expected_source_sha = canonical_sha256(
                {
                    "schema_version": PIPELINE_SCHEMA_VERSION,
                    "inputs": tuple(
                        (spec.window_id, spec.source_sha256, spec.row_count)
                        for spec in manifest.inputs
                        if include_rolling_diagnostics
                        or not spec.registered.rolling_diagnostic_only
                    ),
                }
            )
            fitted_source_sha = _sha256(
                fitted["source_dataset_sha256"],
                name="fitted source dataset sha256",
            )
            if fitted_source_sha != expected_source_sha:
                raise Phase7PipelineError("fitted source dataset binding differs")
            admission_gate = fitted["admission_gate"]
            if (
                not isinstance(admission_gate, Mapping)
                or admission_gate.get("state") != "closed"
            ):
                raise Phase7PipelineError("Phase-7 admission gate must remain closed")
            for name, payload in fitted["artifacts"].items():
                if not isinstance(name, str) or not name or Path(name).name != name:
                    raise Phase7PipelineError("artifact output name is unsafe")
                relative = f"artifacts/{name}.json"
                digest = _write_staged_file(staging, relative, _canonical_bytes(payload))
                staged_inventory[relative] = digest
                artifact_bindings[name] = {
                    "path": relative,
                    "sha256": digest,
                    "artifact_id": payload.get("artifact_id"),
                }
            admissions = fitted["admission_receipts"]
            for name, payload in admissions.items():
                if not isinstance(name, str) or not name or Path(name).name != name:
                    raise Phase7PipelineError("admission output name is unsafe")
                if not isinstance(payload, Mapping):
                    raise Phase7PipelineError("admission receipt must be a mapping")
                receipt_id = payload.get("receipt_id")
                schema_version = payload.get("schema_version")
                known_schema = (
                    type(schema_version) is int
                    and schema_version == PROBABILITY_ADMISSION_SCHEMA_VERSION
                ) or (
                    type(schema_version) is str
                    and schema_version in _ADMISSION_STRING_SCHEMA_VERSIONS
                )
                if (
                    payload.get("source_dataset_sha256") != fitted_source_sha
                    or payload.get("manifest_sha256") != manifest.source_sha256
                    or not known_schema
                    or (
                        receipt_id is not None
                        and (not isinstance(receipt_id, str) or not receipt_id)
                    )
                    or type(payload.get("admitted")) is not bool
                    or payload.get("admitted") is not False
                    or payload.get("action_authority") is not False
                ):
                    raise Phase7PipelineError(
                        f"admission receipt binding differs or is not closed: {name}"
                    )
                relative = f"admission/{name}.json"
                digest = _write_staged_file(
                    staging,
                    relative,
                    _canonical_bytes(payload),
                )
                staged_inventory[relative] = digest
                admission_bindings[name] = {
                    "path": relative,
                    "sha256": digest,
                    "schema_version": schema_version,
                    "receipt_id": receipt_id,
                }
            artifact_blockers = fitted["artifact_blockers"]
            artifact_statuses = fitted["artifact_statuses"]
        result = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "status": (
                "materialized_only"
                if mode == "materialize-only"
                else "fit_and_validation_completed_with_explicit_blockers"
            ),
            "authority": AUTHORITY,
            "run_id": manifest.run_id,
            "run_manifest_sha256": manifest.source_sha256,
            "repository_commit": manifest.repository_commit,
            "python_major_minor": manifest.python_major_minor,
            "code_bundle_identity": manifest.code_bundle_identity,
            "cohort_manifest_sha256": cohort_manifest_sha,
            "cohort_bindings": cohort_bindings,
            "artifact_bindings": artifact_bindings,
            "admission_bindings": admission_bindings,
            "artifact_blockers": artifact_blockers,
            "artifact_statuses": artifact_statuses,
            "admission_gate": admission_gate,
            "admission_receipts": {
                name: {
                    "admitted": bool(payload.get("admitted", False)),
                    "status": payload.get("status"),
                    "receipt_id": payload.get("receipt_id"),
                    "blockers": payload.get("blockers", []),
                    "metrics": payload.get("metrics", []),
                    "support_units": payload.get("support_units", []),
                }
                for name, payload in admissions.items()
            },
            "rolling_diagnostics_included": include_rolling_diagnostics,
            "rolling_oof_claimed": False,
            "sealed_oos_opened": False,
            "production_config_updated": False,
            "empirical_authority": False,
            "action_authority": False,
        }
        result_sha = _write_staged_file(
            staging, "result.json", _canonical_bytes(result)
        )
        staged_inventory["result.json"] = result_sha
        receipt = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "status": "complete_research_receipt",
            "run_id": manifest.run_id,
            "mode": mode,
            "run_manifest_sha256": manifest.source_sha256,
            "repository_commit": manifest.repository_commit,
            "python_major_minor": manifest.python_major_minor,
            "code_bundle_identity": manifest.code_bundle_identity,
            "cohort_manifest_sha256": cohort_manifest_sha,
            "result_sha256": result_sha,
            "runner_sha256": _sha256_file(Path(__file__)),
            "source_input_sha256": {
                spec.window_id: spec.source_sha256
                for spec in manifest.inputs
                if include_rolling_diagnostics
                or not spec.registered.rolling_diagnostic_only
            },
            "model_config_sha256": manifest.model_config_sha256,
            "foundation_identity": FOUNDATION_IDENTITY,
            "history_likelihood_application": (
                "history_state_transition_rules_not_two_marginal_rules"
            ),
            "same_month_rolling_diagnostics_are_oof": False,
            "admission_gate": admission_gate,
            "admission_bindings_sha256": canonical_sha256(admission_bindings),
            "sealed_oos_opened": False,
            "empirical_authority": False,
            "action_authority": False,
        }
        receipt_sha = _write_staged_file(
            staging,
            "receipt.json",
            _canonical_bytes(receipt),
        )
        staged_inventory["receipt.json"] = receipt_sha
        _revalidate_run_manifest(manifest)
        _verify_staging_inventory(staging, staged_inventory)
        # Claim the final path with an exclusive mkdir.  Renaming a directory
        # can replace a concurrently-created empty directory on some systems;
        # this two-step publication never replaces any existing inode.  The
        # receipt moves last and therefore doubles as the completion marker.
        target.mkdir(exist_ok=False)
        children = tuple(staging.iterdir())
        for child in sorted(
            children,
            key=lambda item: (item.name == "receipt.json", item.name),
        ):
            os.rename(child, target / child.name)
        staging.rmdir()
        return result
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def run_pipeline(
    manifest_path: str | Path,
    output_directory: str | Path,
    *,
    mode: str,
    include_rolling_diagnostics: bool,
    engine_factory: EngineFactory = _default_engine_factory,
    clock_loader: ClockLoader = _default_clock_loader,
) -> Mapping[str, Any]:
    if mode not in {"materialize-only", "fit-and-validate"}:
        raise Phase7PipelineError("unsupported Phase-7 pipeline mode")
    manifest = load_run_manifest(manifest_path)
    if include_rolling_diagnostics and not {
        "W3",
        "W4",
    }.issubset({spec.window_id for spec in manifest.inputs}):
        raise Phase7PipelineError(
            "rolling diagnostics require exact hash-bound W3 and W4 inputs"
        )
    if mode == "fit-and-validate" and not manifest.fit_and_validate_authorized:
        raise Phase7PipelineError("run manifest does not authorize fitting")
    cohorts = materialize_cohorts(
        manifest,
        include_rolling_diagnostics=include_rolling_diagnostics,
        engine_factory=engine_factory,
        clock_loader=clock_loader,
    )
    _revalidate_run_manifest(manifest)
    fitted = (
        None
        if mode == "materialize-only"
        else fit_and_validate(
            manifest,
            cohorts,
            include_rolling_diagnostics=include_rolling_diagnostics,
        )
    )
    return write_pipeline_outputs(
        output_directory,
        manifest=manifest,
        cohorts=cohorts,
        mode=mode,
        include_rolling_diagnostics=include_rolling_diagnostics,
        fitted=fitted,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("materialize-only", "fit-and-validate"),
    )
    parser.add_argument(
        "--include-rolling-diagnostics",
        action="store_true",
        help=(
            "Include exact W3/W4 same-month development_cross_fit diagnostics; "
            "these are never labeled rolling OOF."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_pipeline(
        arguments.manifest,
        arguments.output_directory,
        mode=arguments.mode,
        include_rolling_diagnostics=arguments.include_rolling_diagnostics,
    )
    print(json.dumps(_jsonable(result), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
