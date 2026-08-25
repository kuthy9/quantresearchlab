"""Phase-9 historical simulation admission and recovery primitives.

This module deliberately does not provide a market-data socket or claim that a
historical file is a real-time pilot.  It supplies the smaller contracts that a
future no-order service must share: exact bundle admission, an fsync-before-
process WAL, ordered reconnect/backfill, effective-dated instrument mappings,
compact cursor checkpoints, and externally issued capacity authorization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from .artifact_stream import sha256_file
from .market_clock import is_registered_trading_minute, scheduled_gap_kind
from .model import to_primitive
from .shadow_live import ShadowClockInput, shadow_runtime_bindings_from_model_config


PHASE9_HISTORICAL_WINDOW_BUNDLE_SCHEMA_VERSION = (
    "phase9_shadow_historical_window_input_bundle_v3"
)
# Compatibility import for the first v3 implementation.  The v3 schema itself
# is now honestly named for both registered June development windows.
PHASE9_WEEK1_BUNDLE_SCHEMA_VERSION = PHASE9_HISTORICAL_WINDOW_BUNDLE_SCHEMA_VERSION
OPERATIONAL_PROTOCOL_SCHEMA_VERSION = "phase9_shadow_operational_v1"
WAL_SCHEMA_VERSION = "phase9_shadow_wal_v1"
COMPACT_CURSOR_SCHEMA_VERSION = "phase9_shadow_compact_cursor_v1"
CAPACITY_AUTHORIZATION_SCHEMA_VERSION = "phase9_capacity_authorization_v1"


@dataclass(frozen=True)
class HistoricalShadowWindow:
    """Frozen, previously opened June development-window census."""

    alias: str
    window_id: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    synthetic_clocks: tuple[pd.Timestamp, ...]
    manifest_binding: str
    rows: int = 6900
    real_rows: int = 6899
    symbol: str = "NQM4"
    instrument_id: int = 13743
    tick_size: float = 0.25
    point_value: float = 20.0

    @property
    def first_decision_clock(self) -> pd.Timestamp:
        return self.start + pd.Timedelta(minutes=1)

    @property
    def last_decision_clock(self) -> pd.Timestamp:
        return self.end_exclusive - pd.Timedelta(minutes=1)


WEEK1_WINDOW = HistoricalShadowWindow(
    alias="W1",
    window_id="2024-06-week-1",
    start=pd.Timestamp("2024-06-02T22:00:00Z"),
    end_exclusive=pd.Timestamp("2024-06-07T21:01:00Z"),
    synthetic_clocks=(pd.Timestamp("2024-06-07T03:10:00Z"),),
    manifest_binding="week1_v5_manifest",
)
WEEK2_WINDOW = HistoricalShadowWindow(
    alias="W2",
    window_id="2024-06-week-2",
    start=pd.Timestamp("2024-06-09T22:00:00Z"),
    end_exclusive=pd.Timestamp("2024-06-14T21:01:00Z"),
    synthetic_clocks=(pd.Timestamp("2024-06-10T04:14:00Z"),),
    manifest_binding="week2_extension_v2_manifest",
)
_HISTORICAL_WINDOWS = {
    value: window
    for window in (WEEK1_WINDOW, WEEK2_WINDOW)
    for value in (window.alias, window.window_id)
}


def phase9_historical_window(value: str) -> HistoricalShadowWindow:
    """Resolve only the two preregistered 2024-06 development windows."""

    if not isinstance(value, str):
        raise Phase9OperationalError("Phase-9 window id must be W1 or W2")
    try:
        return _HISTORICAL_WINDOWS[value]
    except KeyError as exc:
        raise Phase9OperationalError("Phase-9 window id must be W1 or W2") from exc


WEEK1_WINDOW_ID = WEEK1_WINDOW.window_id
WEEK1_START = WEEK1_WINDOW.start
WEEK1_END_EXCLUSIVE = WEEK1_WINDOW.end_exclusive
WEEK1_FIRST_DECISION_CLOCK = WEEK1_WINDOW.first_decision_clock
WEEK1_LAST_DECISION_CLOCK = WEEK1_WINDOW.last_decision_clock
WEEK1_SYNTHETIC_CLOCKS = WEEK1_WINDOW.synthetic_clocks
WEEK1_ROWS = WEEK1_WINDOW.rows
WEEK1_REAL_ROWS = WEEK1_WINDOW.real_rows
WEEK1_SYMBOL = WEEK1_WINDOW.symbol
WEEK1_INSTRUMENT_ID = WEEK1_WINDOW.instrument_id
WEEK1_TICK_SIZE = WEEK1_WINDOW.tick_size
WEEK1_POINT_VALUE = WEEK1_WINDOW.point_value

_REQUIRED_COMMON_SOURCE_BINDINGS = frozenset(
    {
        "mbo_feature_artifact",
        "mbo_feature_manifest",
        "ohlcv_artifact",
        "ohlcv_manifest",
    }
)
_RUNTIME_CODE_PATHS = (
    "smc_trader/engine.py",
    "smc_trader/current_facts.py",
    "smc_trader/market_state.py",
    "smc_trader/observation.py",
    "smc_trader/shadow_live.py",
    "smc_trader/shadow_operational.py",
    "scripts/materialize_shadow_week1_input.py",
    "scripts/run_shadow_file_pilot.py",
    "scripts/run_shadow_file_pilot_v3.py",
    "configs/model.json",
    "configs/shadow_live_v1.json",
    "configs/phase9_shadow_operational_v1.json",
    "configs/phase9_current_contract_mapping_v1.template.json",
    "semantics/foundation_v2_0.yaml",
)


class Phase9OperationalError(ValueError):
    """Raised when a Phase-9 operational invariant fails closed."""


class CapacityAuthorizationError(Phase9OperationalError):
    """Raised when an external capacity authorization is absent or invalid."""


@dataclass(frozen=True)
class CompactShadowRuntimeEnvelope:
    """Pickle payload containing runtime state but no journal/record history."""

    runtime_state: Mapping[str, Any]
    committed_clocks: int
    wal_fingerprint: str
    protocol_id: str
    mapping_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.runtime_state, Mapping)
            or type(self.committed_clocks) is not int
            or self.committed_clocks < 0
            or not self.protocol_id.startswith("phase9-shadow-operational:")
            or not self.mapping_id.startswith("shadow-instrument-mapping:")
        ):
            raise Phase9OperationalError("compact shadow runtime envelope is invalid")
        _sha256(self.wal_fingerprint, name="compact WAL fingerprint")


class ShadowBackfillRequired(Phase9OperationalError):
    """Raised before journaling a future clock whose predecessors are absent."""

    def __init__(self, missing_completed_clocks: Sequence[pd.Timestamp]) -> None:
        self.missing_completed_clocks = tuple(
            _aware(value, name="missing completed clock")
            for value in missing_completed_clocks
        )
        super().__init__(
            "exact ordered backfill is required before the future clock: "
            + ",".join(value.isoformat() for value in self.missing_completed_clocks)
        )


def _aware(value: Any, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise Phase9OperationalError(f"{name} is not a timestamp") from exc
    if timestamp.tzinfo is None:
        raise Phase9OperationalError(f"{name} must be timezone aware")
    return timestamp.tz_convert("UTC")


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Phase9OperationalError(f"{name} must be a lowercase SHA-256")
    return value


def _identity(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise Phase9OperationalError(f"{name} must be a non-empty identity")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        to_primitive(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _duplicate_guard(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Phase9OperationalError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, *, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Phase9OperationalError(f"{name} must be a trusted regular file")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_guard,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase9OperationalError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise Phase9OperationalError(f"{name} root must be an object")
    return payload


@dataclass(frozen=True)
class ShadowOperationalProtocol:
    schema_version: str
    status: str
    historical_simulation_only: bool
    real_time_live_claim_allowed: bool
    external_submission_allowed: bool
    external_capacity_authorization_required: bool
    self_authorization_allowed: bool
    clock_policy: Mapping[str, Any]
    durability_policy: Mapping[str, Any]
    mapping_policy: Mapping[str, Any]
    operational_metrics: Mapping[str, int]
    authority: Mapping[str, bool]
    source_sha256: str
    protocol_id: str = field(init=False)

    def __post_init__(self) -> None:
        _sha256(self.source_sha256, name="operational protocol source")
        expected_clock = {
            "calendar": "registered_cme_equity_index_minutes",
            "completed_bar_interval_seconds": 60,
            "duplicate": "same_feed_id_same_digest_idempotent",
            "conflicting_duplicate": "terminal_fail_closed",
            "out_of_order": "terminal_fail_closed",
            "gap": "pause_before_journal_and_require_exact_ordered_backfill",
        }
        expected_durability = {
            "wal": "append_flush_fsync_before_process",
            "commit": "append_flush_fsync_after_process",
            "checkpoint": "content_addressed_engine_plus_compact_cursor",
            "checkpoint_interval_clocks": 250,
            "recovery_point_objective_accepted_clocks": 0,
        }
        required_metrics = {
            "minimum_complete_sessions",
            "maximum_recovery_time_seconds",
            "maximum_conflicting_duplicates",
            "maximum_accepted_out_of_order_clocks",
            "maximum_unrecovered_expected_clock_gaps",
            "maximum_external_submission_attempts",
            "maximum_parity_mismatches",
            "maximum_semantic_event_duplicate_identities",
            "maximum_relation_restart_without_termination",
            "maximum_evidence_belief_orphans",
            "maximum_signal_expiry_violations",
            "maximum_dol_identity_drifts",
        }
        integrity_metrics = required_metrics - {
            "minimum_complete_sessions",
            "maximum_recovery_time_seconds",
        }
        if (
            self.schema_version != OPERATIONAL_PROTOCOL_SCHEMA_VERSION
            or self.status != "engineering_historical_simulation_only"
            or self.historical_simulation_only is not True
            or self.real_time_live_claim_allowed is not False
            or self.external_submission_allowed is not False
            or self.external_capacity_authorization_required is not True
            or self.self_authorization_allowed is not False
            or dict(self.clock_policy) != expected_clock
            or dict(self.durability_policy) != expected_durability
            or set(self.operational_metrics) != required_metrics
            or any(
                type(value) is not int or value < 0
                for value in self.operational_metrics.values()
            )
            or any(self.operational_metrics[name] != 0 for name in integrity_metrics)
            or self.operational_metrics["minimum_complete_sessions"] < 2
            or self.operational_metrics["maximum_recovery_time_seconds"] < 1
            or self.mapping_policy.get("silent_rollover_allowed") is not False
            or any(value is not False for value in self.authority.values())
        ):
            raise Phase9OperationalError(
                "Phase-9 operational preregistration changed or self-authorizes"
            )
        payload = {
            "schema_version": self.schema_version,
            "status": self.status,
            "historical_simulation_only": self.historical_simulation_only,
            "real_time_live_claim_allowed": self.real_time_live_claim_allowed,
            "external_submission_allowed": self.external_submission_allowed,
            "external_capacity_authorization_required": (
                self.external_capacity_authorization_required
            ),
            "self_authorization_allowed": self.self_authorization_allowed,
            "clock_policy": dict(self.clock_policy),
            "durability_policy": dict(self.durability_policy),
            "mapping_policy": dict(self.mapping_policy),
            "operational_metrics": dict(self.operational_metrics),
            "authority": dict(self.authority),
        }
        object.__setattr__(
            self,
            "protocol_id",
            f"phase9-shadow-operational:{_digest(payload)}",
        )

    @property
    def recovery_point_objective_accepted_clocks(self) -> int:
        return int(self.durability_policy["recovery_point_objective_accepted_clocks"])


def load_shadow_operational_protocol(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> ShadowOperationalProtocol:
    source = Path(path)
    payload = _read_json(source, name="Phase-9 operational protocol")
    source_sha = sha256_file(source)
    if expected_sha256 is not None and source_sha != _sha256(
        expected_sha256,
        name="expected operational protocol SHA",
    ):
        raise Phase9OperationalError("operational protocol SHA-256 differs")
    expected_keys = {
        "schema_version",
        "status",
        "historical_simulation_only",
        "real_time_live_claim_allowed",
        "external_submission_allowed",
        "external_capacity_authorization_required",
        "self_authorization_allowed",
        "clock_policy",
        "durability_policy",
        "mapping_policy",
        "operational_metrics",
        "authority",
    }
    if set(payload) != expected_keys:
        raise Phase9OperationalError("operational protocol fields changed")
    return ShadowOperationalProtocol(
        **payload,
        source_sha256=source_sha,
    )


@dataclass(frozen=True)
class InstrumentMappingBinding:
    mapping_id: str
    mapping_version: str
    mode: str
    logical_instrument_id: str
    vendor_symbol: str
    vendor_instrument_id: int
    tick_size: float
    point_value: float
    effective_from: pd.Timestamp
    effective_until: pd.Timestamp
    source_identity: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "effective_from",
            _aware(self.effective_from, name="mapping effective_from"),
        )
        object.__setattr__(
            self,
            "effective_until",
            _aware(self.effective_until, name="mapping effective_until"),
        )
        if (
            not self.mapping_id.startswith("shadow-instrument-mapping:")
            or not self.mapping_version
            or self.mode not in {"historical_simulation", "current_live_shadow"}
            or not self.logical_instrument_id
            or not self.vendor_symbol
            or type(self.vendor_instrument_id) is not int
            or self.vendor_instrument_id <= 0
            or not math.isfinite(float(self.tick_size))
            or float(self.tick_size) <= 0.0
            or not math.isfinite(float(self.point_value))
            or float(self.point_value) <= 0.0
            or self.effective_until <= self.effective_from
        ):
            raise Phase9OperationalError("instrument mapping is invalid")
        _sha256(self.source_identity, name="instrument mapping source identity")

    def require_input(self, value: ShadowClockInput, *, mode: str) -> None:
        if mode != self.mode:
            raise Phase9OperationalError("instrument mapping mode differs")
        clock = value.bar.end.tz_convert("UTC")
        if not self.effective_from <= clock < self.effective_until:
            raise Phase9OperationalError("input is outside mapping effective interval")
        if (
            value.bar.symbol != self.vendor_symbol
            or value.bar.instrument_id != self.vendor_instrument_id
        ):
            raise Phase9OperationalError(
                "input differs from versioned instrument mapping"
            )


def load_instrument_mapping(
    path: str | Path,
    *,
    expected_sha256: str,
    required_mode: str,
) -> InstrumentMappingBinding:
    source = Path(path)
    if sha256_file(source) != _sha256(
        expected_sha256,
        name="expected mapping artifact SHA",
    ):
        raise Phase9OperationalError("instrument mapping artifact SHA differs")
    payload = _read_json(source, name="instrument mapping")
    if (
        payload.get("schema_version") != "phase9_shadow_instrument_mapping_v1"
        or payload.get("status") != "externally_frozen_no_order_shadow_mapping"
        or payload.get("external_submission_allowed") is not False
    ):
        raise Phase9OperationalError(
            "instrument mapping is not frozen for no-order use"
        )
    fields = {
        "mapping_id",
        "mapping_version",
        "mode",
        "logical_instrument_id",
        "vendor_symbol",
        "vendor_instrument_id",
        "tick_size",
        "point_value",
        "effective_from",
        "effective_until",
        "source_identity",
    }
    mapping = InstrumentMappingBinding(**{name: payload.get(name) for name in fields})
    if mapping.mode != required_mode:
        raise Phase9OperationalError("instrument mapping mode differs")
    return mapping


def _git_output(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase9OperationalError("repository identity is unavailable") from exc
    return result.stdout.strip()


def runtime_code_environment_identity(
    root: str | Path,
    *,
    require_clean: bool = False,
) -> dict[str, Any]:
    """Return the implementation/environment identity omitted by Shadow v1.2."""

    repository = Path(root).resolve()
    code_hashes: dict[str, str] = {}
    for relative in _RUNTIME_CODE_PATHS:
        path = repository / relative
        if path.is_symlink() or not path.is_file():
            raise Phase9OperationalError(f"runtime identity file is absent: {relative}")
        code_hashes[relative] = sha256_file(path)
    lockfile = repository / "uv.lock"
    if lockfile.is_symlink() or not lockfile.is_file():
        raise Phase9OperationalError("uv.lock is absent from runtime identity")
    status = _git_output(repository, "status", "--porcelain=v1")
    clean = not bool(status)
    if require_clean and not clean:
        raise Phase9OperationalError("repository must be clean before v3 admission")
    payload: dict[str, Any] = {
        "git_commit": _git_output(repository, "rev-parse", "HEAD"),
        "git_tree": _git_output(repository, "rev-parse", "HEAD^{tree}"),
        "repository_clean": clean,
        "working_tree_status_sha256": hashlib.sha256(
            status.encode("utf-8")
        ).hexdigest(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "uv_lock_sha256": sha256_file(lockfile),
        "code_sha256": code_hashes,
        "model_runtime_bindings": dict(
            shadow_runtime_bindings_from_model_config(repository / "configs/model.json")
        ),
    }
    payload["runtime_identity_sha256"] = _digest(payload)
    return payload


def build_phase9_bundle_v3_payload(
    base_payload: Mapping[str, Any],
    *,
    runtime_identity: Mapping[str, Any],
    operational_protocol: ShadowOperationalProtocol,
) -> dict[str, Any]:
    """Upgrade a registered W1/W2 sidecar without changing input bytes."""

    payload = json.loads(json.dumps(dict(base_payload)))
    window_payload = payload.get("window")
    if not isinstance(window_payload, Mapping):
        raise Phase9OperationalError("v3 bundle window is absent")
    window = phase9_historical_window(window_payload.get("id"))
    required_sources = _REQUIRED_COMMON_SOURCE_BINDINGS | {window.manifest_binding}
    payload["format_version"] = 3
    payload["schema_version"] = PHASE9_HISTORICAL_WINDOW_BUNDLE_SCHEMA_VERSION
    payload["status"] = "complete_current_bound_historical_simulation_input"
    payload["runtime_code_environment"] = dict(runtime_identity)
    payload["current_runtime_bindings"] = {
        "model_runtime_protocols": dict(
            runtime_identity.get("model_runtime_bindings", {})
        ),
        "runtime_identity_sha256": runtime_identity.get("runtime_identity_sha256"),
    }
    payload["operational_protocol"] = {
        "schema_version": operational_protocol.schema_version,
        "protocol_id": operational_protocol.protocol_id,
        "sha256": operational_protocol.source_sha256,
        "historical_simulation_only": True,
        "real_time_live_claim_allowed": False,
    }
    payload["data_role"] = {
        "ohlcv": "previously_opened_2024_06_development_not_oof",
        "mbo": "development_execution_reality_only",
        "rolling_oof_allowed": False,
        "sealed_oos_opened": False,
    }
    payload["admission_contract"] = {
        "exact_rows": window.rows,
        "exact_window_id": window.window_id,
        "exact_source_bindings": sorted(required_sources),
        "current_runtime_code_environment_required": True,
        "external_capacity_authorization_required": True,
        "sidecar_required_before_input_open": True,
    }
    payload["authorization"] = {
        "full_6900_file_replay_authorized": False,
        "real_time_live_authorized": False,
        "sealed_oos_authorized": False,
        "self_authorization_allowed": False,
    }
    return payload


def build_week1_bundle_v3_payload(
    base_payload: Mapping[str, Any],
    *,
    runtime_identity: Mapping[str, Any],
    operational_protocol: ShadowOperationalProtocol,
) -> dict[str, Any]:
    """Backward-compatible name for the generic registered-window upgrader."""

    return build_phase9_bundle_v3_payload(
        base_payload,
        runtime_identity=runtime_identity,
        operational_protocol=operational_protocol,
    )


def validate_phase9_bundle_v3_payload(
    payload: Mapping[str, Any],
    *,
    output_path: str | Path,
    output_sha256: str,
    output_bytes: int,
    output_rows: int,
    runtime_identity: Mapping[str, Any],
    operational_protocol: ShadowOperationalProtocol,
    root: str | Path | None = None,
    verify_source_files: bool = True,
) -> None:
    """Validate the current v3 sidecar before an Engine may be constructed."""

    contract = payload.get("contract")
    window = payload.get("window")
    census = payload.get("census")
    sources = payload.get("source_bindings")
    output = payload.get("output")
    limits = payload.get("limits")
    projection = payload.get("projection_contract")
    admission = payload.get("admission_contract")
    authorization = payload.get("authorization")
    protocol = payload.get("operational_protocol")
    if not isinstance(window, Mapping):
        raise Phase9OperationalError("v3 bundle window is absent")
    window_contract = phase9_historical_window(window.get("id"))
    required_sources = _REQUIRED_COMMON_SOURCE_BINDINGS | {
        window_contract.manifest_binding
    }
    if (
        payload.get("format_version") != 3
        or payload.get("schema_version")
        != PHASE9_HISTORICAL_WINDOW_BUNDLE_SCHEMA_VERSION
        or payload.get("input_schema_version") != "phase9_shadow_file_input_v2"
        or payload.get("status") != "complete_current_bound_historical_simulation_input"
        or payload.get("authority") != "historical_engineering_file_input_only"
    ):
        raise Phase9OperationalError("v3 bundle identity changed")
    if contract != {
        "symbol": window_contract.symbol,
        "instrument_id": window_contract.instrument_id,
        "tick_size": window_contract.tick_size,
        "point_value": window_contract.point_value,
    }:
        raise Phase9OperationalError("v3 bundle contract changed")
    if (
        window.get("id") != window_contract.window_id
        or _aware(window.get("start"), name="bundle window start")
        != window_contract.start
        or _aware(window.get("end_exclusive"), name="bundle window end")
        != window_contract.end_exclusive
    ):
        raise Phase9OperationalError("v3 bundle window changed")
    if (
        not isinstance(census, Mapping)
        or census.get("rows") != window_contract.rows
        or census.get("real_completed") != window_contract.real_rows
        or census.get("synthetic_no_trade") != len(window_contract.synthetic_clocks)
        or tuple(
            _aware(value, name="bundle synthetic clock")
            for value in census.get("synthetic_decision_clocks", ())
        )
        != window_contract.synthetic_clocks
        or _aware(
            census.get("first_decision_clock"),
            name="bundle first decision clock",
        )
        != window_contract.first_decision_clock
        or _aware(
            census.get("last_decision_clock"),
            name="bundle last decision clock",
        )
        != window_contract.last_decision_clock
        or census.get("data_gap_resets") != 0
        or census.get("contract_changes") != 0
        or type(output_rows) is not int
        or output_rows != window_contract.rows
    ):
        raise Phase9OperationalError("v3 bundle census changed")
    if (
        not isinstance(output, Mapping)
        or output.get("path") != str(Path(output_path).resolve())
        or output.get("sha256") != _sha256(output_sha256, name="output SHA")
        or output.get("bytes") != output_bytes
        or output.get("rows") != output_rows
    ):
        raise Phase9OperationalError("v3 bundle output identity changed")
    if payload.get("runtime_code_environment") != dict(runtime_identity):
        raise Phase9OperationalError("v3 bundle runtime/code/environment differs")
    if payload.get("current_runtime_bindings") != {
        "model_runtime_protocols": dict(
            runtime_identity.get("model_runtime_bindings", {})
        ),
        "runtime_identity_sha256": runtime_identity.get("runtime_identity_sha256"),
    }:
        raise Phase9OperationalError("v3 bundle runtime/code/environment differs")
    if protocol != {
        "schema_version": operational_protocol.schema_version,
        "protocol_id": operational_protocol.protocol_id,
        "sha256": operational_protocol.source_sha256,
        "historical_simulation_only": True,
        "real_time_live_claim_allowed": False,
    }:
        raise Phase9OperationalError("v3 bundle operational protocol differs")
    if payload.get("data_role") != {
        "ohlcv": "previously_opened_2024_06_development_not_oof",
        "mbo": "development_execution_reality_only",
        "rolling_oof_allowed": False,
        "sealed_oos_opened": False,
    }:
        raise Phase9OperationalError("v3 bundle development data role changed")
    if projection != {
        "execution_provider": "TopOfBookExecutionProvider",
        "source_depth_imbalance": "validated_top5_not_serialized",
        "serialized_depth_imbalance": "recomputed_best_level",
        "data_age_seconds": "scalar_decision_clock_minus_book_observed_at",
        "received_delay_milliseconds": 5,
        "execution_deadline_minutes": 60,
        "account": {
            "equity": 100000.0,
            "flat": True,
            "open_risk_fraction": 0.0,
            "requested_risk_fraction": 0.0,
        },
        "approved_intents": 0,
        "execution_events": 0,
    }:
        raise Phase9OperationalError("v3 bundle projection contract changed")
    if (
        admission
        != {
            "exact_rows": window_contract.rows,
            "exact_window_id": window_contract.window_id,
            "exact_source_bindings": sorted(required_sources),
            "current_runtime_code_environment_required": True,
            "external_capacity_authorization_required": True,
            "sidecar_required_before_input_open": True,
        }
        or authorization
        != {
            "full_6900_file_replay_authorized": False,
            "real_time_live_authorized": False,
            "sealed_oos_authorized": False,
            "self_authorization_allowed": False,
        }
        or limits
        != {
            "cold_start": True,
            "warm_state_restored": False,
            "warmup_history_included": False,
            "real_time_live": False,
            "multi_day_live_pilot": False,
            "broker_submission": False,
            "live_account_state": False,
            "raw_mbo_read": False,
            "sealed_holdout_read": False,
            "phase9_gate_closed": True,
        }
    ):
        raise Phase9OperationalError("v3 bundle admission/authority changed")
    if not isinstance(sources, Mapping) or set(sources) != required_sources:
        raise Phase9OperationalError("v3 bundle source bindings changed")
    repository = Path(root).resolve() if root is not None else None
    for name, binding in sources.items():
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or not isinstance(binding.get("path"), str)
            or Path(binding["path"]).is_absolute()
        ):
            raise Phase9OperationalError(f"v3 bundle source {name} is invalid")
        expected_hash = _sha256(binding.get("sha256"), name=f"source {name} SHA")
        if verify_source_files:
            if repository is None:
                raise Phase9OperationalError(
                    "source verification requires repository root"
                )
            source = repository / binding["path"]
            if source.is_symlink() or not source.is_file():
                raise Phase9OperationalError(f"v3 bundle source {name} is absent")
            if sha256_file(source) != expected_hash:
                raise Phase9OperationalError(f"v3 bundle source {name} SHA differs")


def validate_week1_bundle_v3_payload(
    payload: Mapping[str, Any],
    *,
    output_path: str | Path,
    output_sha256: str,
    output_bytes: int,
    output_rows: int,
    runtime_identity: Mapping[str, Any],
    operational_protocol: ShadowOperationalProtocol,
    root: str | Path | None = None,
    verify_source_files: bool = True,
) -> None:
    """Backward-compatible name for registered W1/W2 v3 validation."""

    validate_phase9_bundle_v3_payload(
        payload,
        output_path=output_path,
        output_sha256=output_sha256,
        output_bytes=output_bytes,
        output_rows=output_rows,
        runtime_identity=runtime_identity,
        operational_protocol=operational_protocol,
        root=root,
        verify_source_files=verify_source_files,
    )


@dataclass(frozen=True)
class WALTransaction:
    sequence: int
    input_value: ShadowClockInput
    result_digest: str | None = None
    result_payload: Any | None = None

    @property
    def committed(self) -> bool:
        return self.result_digest is not None


class DurableShadowWAL:
    """Append-only attempt/commit journal with fsync on both boundaries."""

    def __init__(
        self,
        path: str | Path,
        *,
        encode_input: Callable[[ShadowClockInput], Mapping[str, Any]],
        decode_input: Callable[[Mapping[str, Any]], ShadowClockInput],
    ) -> None:
        self.path = Path(path)
        self._encode_input = encode_input
        self._decode_input = decode_input
        self._transactions: list[WALTransaction] = []
        self._committed_count = 0
        self._feed_digests: dict[str, str] = {}
        self._clock_feed_ids: dict[pd.Timestamp, str] = {}
        self._fingerprint = _digest((WAL_SCHEMA_VERSION,))
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        if self.path.is_symlink() or not self.path.is_file():
            raise Phase9OperationalError("shadow WAL must be a trusted regular file")
        pending: WALTransaction | None = None
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise Phase9OperationalError("shadow WAL contains a blank row")
                try:
                    record = json.loads(line, object_pairs_hook=_duplicate_guard)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise Phase9OperationalError(
                        f"shadow WAL row {line_number} is invalid"
                    ) from exc
                self._fingerprint = _digest((self._fingerprint, _digest(record)))
                kind = record.get("kind")
                sequence = record.get("sequence")
                if (
                    record.get("schema_version") != WAL_SCHEMA_VERSION
                    or type(sequence) is not int
                    or sequence <= 0
                ):
                    raise Phase9OperationalError("shadow WAL identity changed")
                if kind == "attempt":
                    if pending is not None or sequence != len(self._transactions) + 1:
                        raise Phase9OperationalError(
                            "shadow WAL attempt sequence changed"
                        )
                    payload = record.get("input_payload")
                    if not isinstance(payload, Mapping):
                        raise Phase9OperationalError(
                            "shadow WAL input payload is invalid"
                        )
                    value = self._decode_input(payload)
                    if (
                        record.get("feed_event_id") != value.feed_event_id
                        or record.get("input_digest") != value.input_digest
                        or _aware(
                            record.get("completed_clock"),
                            name="WAL completed clock",
                        )
                        != value.bar.end
                    ):
                        raise Phase9OperationalError(
                            "shadow WAL attempt binding differs"
                        )
                    self._register_attempt(value)
                    pending = WALTransaction(sequence=sequence, input_value=value)
                    self._transactions.append(pending)
                elif kind == "commit":
                    if (
                        pending is None
                        or sequence != pending.sequence
                        or record.get("input_digest")
                        != pending.input_value.input_digest
                    ):
                        raise Phase9OperationalError(
                            "shadow WAL commit lacks its attempt"
                        )
                    result_payload = record.get("result_payload")
                    result_digest = _sha256(
                        record.get("result_digest"),
                        name="WAL result digest",
                    )
                    if _digest(result_payload) != result_digest:
                        raise Phase9OperationalError("shadow WAL result digest differs")
                    committed = WALTransaction(
                        sequence=sequence,
                        input_value=pending.input_value,
                        result_digest=result_digest,
                        result_payload=result_payload,
                    )
                    self._transactions[-1] = committed
                    self._committed_count += 1
                    pending = None
                else:
                    raise Phase9OperationalError("shadow WAL record kind changed")

    def _register_attempt(self, value: ShadowClockInput) -> None:
        prior = self._feed_digests.get(value.feed_event_id)
        if prior is not None:
            if prior != value.input_digest:
                raise Phase9OperationalError(
                    "feed event identity conflicts with durable WAL"
                )
            raise Phase9OperationalError("durable WAL repeats an attempt")
        clock = value.bar.end.tz_convert("UTC")
        if clock in self._clock_feed_ids:
            raise Phase9OperationalError("completed clock conflicts in durable WAL")
        if self._transactions and clock <= self._transactions[-1].input_value.bar.end:
            raise Phase9OperationalError("durable WAL clocks are out of order")
        self._feed_digests[value.feed_event_id] = value.input_digest
        self._clock_feed_ids[clock] = value.feed_event_id

    def _append_record(self, record: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise Phase9OperationalError("shadow WAL cannot be a symlink")
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        payload = _canonical_bytes(record) + b"\n"
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("short WAL write")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._fingerprint = _digest((self._fingerprint, _digest(record)))

    @property
    def transactions(self) -> tuple[WALTransaction, ...]:
        return tuple(self._transactions)

    @property
    def committed_count(self) -> int:
        return self._committed_count

    @property
    def attempted_count(self) -> int:
        return len(self._transactions)

    @property
    def pending_input(self) -> ShadowClockInput | None:
        if self._transactions and not self._transactions[-1].committed:
            return self._transactions[-1].input_value
        return None

    @property
    def last_committed_input(self) -> ShadowClockInput | None:
        for transaction in reversed(self._transactions):
            if transaction.committed:
                return transaction.input_value
        return None

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def prior_input_digest(self, feed_event_id: str) -> str | None:
        return self._feed_digests.get(feed_event_id)

    def prior_feed_id_for_clock(self, clock: pd.Timestamp) -> str | None:
        return self._clock_feed_ids.get(_aware(clock, name="completed clock"))

    def append_attempt(self, value: ShadowClockInput) -> None:
        if self.pending_input is not None:
            raise Phase9OperationalError(
                "pending durable attempt must recover before another input"
            )
        sequence = len(self._transactions) + 1
        self._register_attempt(value)
        record = {
            "schema_version": WAL_SCHEMA_VERSION,
            "kind": "attempt",
            "sequence": sequence,
            "feed_event_id": value.feed_event_id,
            "input_digest": value.input_digest,
            "completed_clock": value.bar.end.isoformat(),
            "input_payload": dict(self._encode_input(value)),
        }
        try:
            self._append_record(record)
        except BaseException:
            self._feed_digests.pop(value.feed_event_id, None)
            self._clock_feed_ids.pop(value.bar.end.tz_convert("UTC"), None)
            raise
        self._transactions.append(WALTransaction(sequence=sequence, input_value=value))

    def append_commit(self, result: Any) -> None:
        if self.pending_input is None:
            raise Phase9OperationalError("shadow WAL commit has no pending attempt")
        transaction = self._transactions[-1]
        result_payload = to_primitive(result)
        result_digest = _digest(result_payload)
        record = {
            "schema_version": WAL_SCHEMA_VERSION,
            "kind": "commit",
            "sequence": transaction.sequence,
            "input_digest": transaction.input_value.input_digest,
            "result_digest": result_digest,
            "result_payload": result_payload,
        }
        self._append_record(record)
        self._transactions[-1] = WALTransaction(
            sequence=transaction.sequence,
            input_value=transaction.input_value,
            result_digest=result_digest,
            result_payload=result_payload,
        )
        self._committed_count += 1


def expected_missing_completed_clocks(
    last_completed_clock: pd.Timestamp,
    next_completed_clock: pd.Timestamp,
) -> tuple[pd.Timestamp, ...]:
    """Return registered completed clocks strictly between two inputs."""

    left = _aware(last_completed_clock, name="last completed clock")
    right = _aware(next_completed_clock, name="next completed clock")
    minute = pd.Timedelta(1, unit="min")
    if right <= left:
        raise Phase9OperationalError("next completed clock is out of order")
    current_start = right - minute
    if not is_registered_trading_minute(current_start):
        raise Phase9OperationalError("next input is outside registered market minutes")
    if scheduled_gap_kind(left, current_start) is not None:
        return ()
    missing: list[pd.Timestamp] = []
    bar_start = left
    while bar_start < current_start:
        if is_registered_trading_minute(bar_start):
            missing.append(bar_start + minute)
        bar_start += minute
    return tuple(missing)


@dataclass(frozen=True)
class OperationalIngestResult:
    result: Any | None
    duplicate: bool


class OperationalShadowSession:
    """Minimal reconnect state machine around an externally supplied processor."""

    def __init__(
        self,
        *,
        protocol: ShadowOperationalProtocol,
        mapping: InstrumentMappingBinding,
        wal: DurableShadowWAL,
        mode: str,
    ) -> None:
        if mode not in {"historical_simulation", "current_live_shadow"}:
            raise Phase9OperationalError("shadow operational mode is invalid")
        if (
            mode == "current_live_shadow"
            and protocol.real_time_live_claim_allowed is False
        ):
            # The state machine may be reused by a later versioned protocol,
            # but this frozen engineering protocol cannot claim that mode.
            raise Phase9OperationalError(
                "current live shadow requires a later versioned protocol"
            )
        if mapping.mode != mode:
            raise Phase9OperationalError("instrument mapping mode differs")
        self.protocol = protocol
        self.mapping = mapping
        self.wal = wal
        self.mode = mode
        self.status = (
            "recovery_required" if wal.pending_input is not None else "connected"
        )
        self._required_backfill: tuple[pd.Timestamp, ...] = ()
        self._metrics = {
            "disconnects": 0,
            "reconnects": 0,
            "exact_duplicates": 0,
            "conflicting_duplicates": 0,
            "out_of_order_clocks": 0,
            "gap_incidents": 0,
            "backfilled_clocks": 0,
            "processing_failures": 0,
        }

    @property
    def metrics(self) -> Mapping[str, int]:
        return dict(self._metrics)

    def disconnect(self, *, reason: str) -> None:
        _identity(reason, name="disconnect reason")
        if self.status not in {"connected", "backfill_required"}:
            raise Phase9OperationalError("shadow session cannot disconnect now")
        self.status = "disconnected"
        self._metrics["disconnects"] += 1

    def reconnect(self) -> None:
        if self.status not in {
            "disconnected",
            "recovery_required",
            "failed_recovery_required",
        }:
            raise Phase9OperationalError("shadow session reconnect is not required")
        self.status = (
            "recovery_required" if self.wal.pending_input is not None else "connected"
        )
        self._metrics["reconnects"] += 1

    def _require_new_input(
        self,
        value: ShadowClockInput,
        *,
        backfill: bool,
    ) -> OperationalIngestResult | None:
        prior_digest = self.wal.prior_input_digest(value.feed_event_id)
        if prior_digest is not None:
            if prior_digest != value.input_digest:
                self._metrics["conflicting_duplicates"] += 1
                raise Phase9OperationalError(
                    "feed event identity conflicts with durable WAL"
                )
            transaction = next(
                item
                for item in self.wal.transactions
                if item.input_value.feed_event_id == value.feed_event_id
            )
            if not transaction.committed:
                raise Phase9OperationalError(
                    "exact duplicate is a pending attempt and requires recovery"
                )
            self._metrics["exact_duplicates"] += 1
            return OperationalIngestResult(
                result=transaction.result_payload,
                duplicate=True,
            )
        clock = value.bar.end.tz_convert("UTC")
        existing_feed = self.wal.prior_feed_id_for_clock(clock)
        if existing_feed is not None:
            self._metrics["conflicting_duplicates"] += 1
            raise Phase9OperationalError("completed clock conflicts with durable WAL")
        last = self.wal.last_committed_input
        if last is not None and clock <= last.bar.end:
            self._metrics["out_of_order_clocks"] += 1
            raise Phase9OperationalError("new shadow clock is out of order")
        if self._required_backfill:
            if not backfill or clock != self._required_backfill[0]:
                raise ShadowBackfillRequired(self._required_backfill)
        elif last is not None:
            missing = expected_missing_completed_clocks(last.bar.end, clock)
            if missing:
                self._required_backfill = missing
                self.status = "backfill_required"
                self._metrics["gap_incidents"] += 1
                raise ShadowBackfillRequired(missing)
        return None

    def ingest(
        self,
        value: ShadowClockInput,
        *,
        process: Callable[[ShadowClockInput], Any],
        backfill: bool = False,
    ) -> OperationalIngestResult:
        if self.status == "disconnected":
            raise Phase9OperationalError("shadow session is disconnected")
        if self.status in {"recovery_required", "failed_recovery_required"}:
            raise Phase9OperationalError("pending attempt must recover before ingest")
        if self.status == "backfill_required" and not backfill:
            raise ShadowBackfillRequired(self._required_backfill)
        self.mapping.require_input(value, mode=self.mode)
        duplicate = self._require_new_input(value, backfill=backfill)
        if duplicate is not None:
            return duplicate
        self.wal.append_attempt(value)
        try:
            result = process(value)
        except BaseException:
            self.status = "failed_recovery_required"
            self._metrics["processing_failures"] += 1
            raise
        self.wal.append_commit(result)
        if self._required_backfill:
            self._required_backfill = self._required_backfill[1:]
            self._metrics["backfilled_clocks"] += 1
            self.status = (
                "backfill_required" if self._required_backfill else "connected"
            )
        return OperationalIngestResult(result=result, duplicate=False)

    def recover_pending(
        self,
        *,
        process: Callable[[ShadowClockInput], Any],
    ) -> Any:
        if self.status != "recovery_required" or self.wal.pending_input is None:
            raise Phase9OperationalError("shadow session has no pending recovery")
        value = self.wal.pending_input
        assert value is not None
        self.mapping.require_input(value, mode=self.mode)
        try:
            result = process(value)
        except BaseException:
            self.status = "failed_recovery_required"
            self._metrics["processing_failures"] += 1
            raise
        self.wal.append_commit(result)
        self.status = "connected"
        return result

    def compact_checkpoint_payload(
        self,
        *,
        engine_checkpoint_sha256: str,
    ) -> dict[str, Any]:
        engine_sha = _sha256(
            engine_checkpoint_sha256,
            name="engine checkpoint SHA",
        )
        last = self.wal.last_committed_input
        return {
            "schema_version": COMPACT_CURSOR_SCHEMA_VERSION,
            "status": self.status,
            "protocol_id": self.protocol.protocol_id,
            "mapping_id": self.mapping.mapping_id,
            "mode": self.mode,
            "attempted_clocks": self.wal.attempted_count,
            "committed_clocks": self.wal.committed_count,
            "last_feed_event_id": None if last is None else last.feed_event_id,
            "last_completed_clock": None if last is None else last.bar.end.isoformat(),
            "wal_fingerprint": self.wal.fingerprint,
            "engine_checkpoint_sha256": engine_sha,
            "required_backfill": [
                value.isoformat() for value in self._required_backfill
            ],
            "metrics": dict(self._metrics),
            "real_time_live_claim": False,
            "external_submission_allowed": False,
        }


@dataclass(frozen=True)
class CapacityAuthorization:
    authorization_id: str
    input_bundle_sha256: str
    input_sidecar_sha256: str
    runtime_identity_sha256: str
    authorized_rows: int
    full_6900_file_replay_authorized: bool
    real_time_live_authorized: bool
    sealed_oos_authorized: bool
    maximum_checkpoint_bytes: int
    maximum_peak_working_set_bytes: int
    minimum_available_memory_bytes: int
    minimum_available_disk_bytes: int
    frozen_at: pd.Timestamp
    artifact_sha256: str


def capacity_authorization_id(payload: Mapping[str, Any]) -> str:
    values = dict(payload)
    values.pop("authorization_id", None)
    return f"phase9-capacity-authorization:{_digest(values)}"


def load_capacity_authorization(
    path: str | Path,
    *,
    expected_sha256: str,
    input_bundle_sha256: str,
    input_sidecar_sha256: str,
    runtime_identity_sha256: str,
    required_rows: int,
) -> CapacityAuthorization:
    source = Path(path)
    expected_artifact = _sha256(
        expected_sha256,
        name="expected capacity artifact SHA",
    )
    if (
        source.is_symlink()
        or not source.is_file()
        or sha256_file(source) != expected_artifact
    ):
        raise CapacityAuthorizationError("capacity authorization artifact SHA differs")
    try:
        payload = _read_json(source, name="capacity authorization")
    except Phase9OperationalError as exc:
        raise CapacityAuthorizationError(str(exc)) from exc
    if (
        payload.get("schema_version") != CAPACITY_AUTHORIZATION_SCHEMA_VERSION
        or payload.get("status")
        != "externally_frozen_historical_simulation_capacity_authorized"
        or payload.get("authority") != "independent_capacity_review"
        or payload.get("full_6900_file_replay_authorized") is not True
    ):
        raise CapacityAuthorizationError(
            "capacity authorization was not externally frozen"
        )
    if (
        payload.get("real_time_live_authorized") is not False
        or payload.get("sealed_oos_authorized") is not False
    ):
        raise CapacityAuthorizationError(
            "historical capacity artifact cannot authorize live or sealed use"
        )
    if (
        payload.get("input_bundle_sha256")
        != _sha256(input_bundle_sha256, name="input bundle SHA")
        or payload.get("input_sidecar_sha256")
        != _sha256(input_sidecar_sha256, name="input sidecar SHA")
        or payload.get("runtime_identity_sha256")
        != _sha256(runtime_identity_sha256, name="runtime identity SHA")
        or type(payload.get("authorized_rows")) is not int
        or payload.get("authorized_rows") != required_rows
        or required_rows != WEEK1_ROWS
    ):
        raise CapacityAuthorizationError("capacity authorization binding differs")
    if payload.get("authorization_id") != capacity_authorization_id(payload):
        raise CapacityAuthorizationError("capacity authorization identity differs")
    maximum_checkpoint = payload.get("maximum_checkpoint_bytes")
    maximum_peak = payload.get("maximum_peak_working_set_bytes")
    minimum_memory = payload.get("minimum_available_memory_bytes")
    minimum_disk = payload.get("minimum_available_disk_bytes")
    if (
        type(maximum_checkpoint) is not int
        or maximum_checkpoint <= 0
        or type(maximum_peak) is not int
        or maximum_peak <= maximum_checkpoint
        or type(minimum_memory) is not int
        or minimum_memory <= 0
        or type(minimum_disk) is not int
        or minimum_disk <= 0
    ):
        raise CapacityAuthorizationError("capacity authorization limits are invalid")
    return CapacityAuthorization(
        authorization_id=payload["authorization_id"],
        input_bundle_sha256=payload["input_bundle_sha256"],
        input_sidecar_sha256=payload["input_sidecar_sha256"],
        runtime_identity_sha256=payload["runtime_identity_sha256"],
        authorized_rows=payload["authorized_rows"],
        full_6900_file_replay_authorized=True,
        real_time_live_authorized=False,
        sealed_oos_authorized=False,
        maximum_checkpoint_bytes=maximum_checkpoint,
        maximum_peak_working_set_bytes=maximum_peak,
        minimum_available_memory_bytes=minimum_memory,
        minimum_available_disk_bytes=minimum_disk,
        frozen_at=_aware(payload.get("frozen_at"), name="capacity frozen_at"),
        artifact_sha256=expected_artifact,
    )
