#!/usr/bin/env python3
"""Run a bounded Phase-9 engineering file replay through ShadowLiveRunner.

This is a local operational rehearsal, not a real-time or multi-day live
pilot.  It accepts only flat-account clocks with no risk approvals and no
execution events, uses the production development Engine plus
``NullExecutionGateway``, checkpoints to a trusted local pickle, and proves a
second cold replay from the durable input journal before completion.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import pandas as pd
import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.artifact_stream import (  # noqa: E402
    atomic_bytes,
    sha256_file,
)
from smc_trader.calibration_replay import ReplayCheckpointStore  # noqa: E402
from smc_trader.engine import ContinuousSMCEngine  # noqa: E402
from smc_trader.model import AccountState, Bar, to_primitive  # noqa: E402
from smc_trader.observation import ExecutionRealityInput  # noqa: E402
from smc_trader.shadow_live import (  # noqa: E402
    SHADOW_LIVE_AUTHORITY,
    NullExecutionGateway,
    ShadowClockInput,
    ShadowInputJournal,
    ShadowLiveError,
    ShadowLiveRunner,
    audit_shadow_parity,
    load_shadow_live_protocol,
    replay_shadow_journal,
    shadow_runtime_bindings_from_model_config,
)


PILOT_SCHEMA_VERSION = "phase9_shadow_file_replay_v2"
PILOT_STATUS = "engineering_file_replay_only"
INPUT_SCHEMA_VERSION = "phase9_shadow_file_input_v2"
MODEL_CONFIG = ROOT / "configs/model.json"
SHADOW_PROTOCOL = ROOT / "configs/shadow_live_v1.json"
RUN_MANIFEST = "run_manifest.json"
INPUT_JOURNAL = "input_journal.jsonl"
PARITY_RECORDS = "parity_records.jsonl"
COMPLETED = "COMPLETED.json"
FAILED = "FAILED.json"

_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "feed_event_id",
        "received_at",
        "bar",
        "execution",
        "execution_observed_at",
        "execution_known_at",
        "execution_source_event_id",
        "account",
        "account_observed_at",
        "account_known_at",
        "account_snapshot_id",
        "source_event_ids",
        "approved_intents",
        "execution_events",
    }
)
_BAR_FIELDS = frozenset(item.name for item in fields(Bar))
_EXECUTION_FIELDS = frozenset(item.name for item in fields(ExecutionRealityInput))
_ACCOUNT_FIELDS = frozenset(item.name for item in fields(AccountState))
_CHECKPOINT_FIELDS = frozenset(
    {
        "replay",
        "processed_bars",
        "decision_rows",
        "next_shard_index",
        "committed_shards",
        "last_source_start",
        "source_rows_consumed",
        "pilot_schema_version",
        "record_fingerprint",
        "journal_fingerprint",
        "attempt_fingerprint",
    }
)


class ShadowFilePilotError(ValueError):
    """Raised when the bounded file-pilot contract fails closed."""


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ShadowFilePilotError(
                f"shadow file input contains duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def _exact_mapping(
    value: Any,
    expected: frozenset[str],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ShadowFilePilotError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != set(expected):
        raise ShadowFilePilotError(
            f"{name} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return dict(value)


def _timestamp(value: Any, *, name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ShadowFilePilotError(f"{name} is not a timestamp") from exc
    if result.tzinfo is None:
        raise ShadowFilePilotError(f"{name} must be timezone aware")
    return result


def shadow_clock_input_from_payload(payload: Mapping[str, Any]) -> ShadowClockInput:
    """Decode the intentionally narrow, zero-order pilot JSON schema."""

    root = _exact_mapping(payload, _ROOT_FIELDS, name="shadow clock input")
    if root["schema_version"] != INPUT_SCHEMA_VERSION:
        raise ShadowFilePilotError("shadow clock input schema version differs")
    approvals = root["approved_intents"]
    execution_events = root["execution_events"]
    if approvals != [] or execution_events != []:
        raise ShadowFilePilotError(
            "file pilot v1 requires zero approved_intents and execution_events"
        )

    bar_payload = _exact_mapping(root["bar"], _BAR_FIELDS, name="bar")
    bar_payload["start"] = _timestamp(bar_payload["start"], name="bar.start")
    execution_payload = _exact_mapping(
        root["execution"],
        _EXECUTION_FIELDS,
        name="execution",
    )
    if execution_payload["deadline"] is not None:
        execution_payload["deadline"] = _timestamp(
            execution_payload["deadline"],
            name="execution.deadline",
        )
    if not isinstance(execution_payload["anomalies"], list):
        raise ShadowFilePilotError("execution.anomalies must be a JSON array")
    execution_payload["anomalies"] = tuple(execution_payload["anomalies"])

    account_payload = _exact_mapping(
        root["account"],
        _ACCOUNT_FIELDS,
        name="account",
    )
    if account_payload["position"] is not None:
        raise ShadowFilePilotError("file pilot v1 requires a flat account")
    sources = root["source_event_ids"]
    if not isinstance(sources, list):
        raise ShadowFilePilotError("source_event_ids must be a JSON array")
    try:
        return ShadowClockInput(
            feed_event_id=root["feed_event_id"],
            received_at=_timestamp(root["received_at"], name="received_at"),
            bar=Bar(**bar_payload),
            execution=ExecutionRealityInput(**execution_payload),
            execution_observed_at=_timestamp(
                root["execution_observed_at"],
                name="execution_observed_at",
            ),
            execution_known_at=_timestamp(
                root["execution_known_at"],
                name="execution_known_at",
            ),
            execution_source_event_id=root["execution_source_event_id"],
            account=AccountState(**account_payload),
            account_observed_at=_timestamp(
                root["account_observed_at"],
                name="account_observed_at",
            ),
            account_known_at=_timestamp(
                root["account_known_at"],
                name="account_known_at",
            ),
            account_snapshot_id=root["account_snapshot_id"],
            source_event_ids=tuple(sources),
            approved_intents=(),
            execution_events=(),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ShadowFilePilotError):
            raise
        raise ShadowFilePilotError(
            f"shadow clock input violates the Phase-9 contract: {exc}"
        ) from exc


def shadow_clock_input_payload(value: ShadowClockInput) -> dict[str, Any]:
    if not isinstance(value, ShadowClockInput):
        raise TypeError("shadow journal serialization requires ShadowClockInput")
    if value.approved_intents or value.execution_events:
        raise ShadowFilePilotError(
            "file pilot v1 cannot serialize approvals or execution events"
        )
    if value.account.position is not None:
        raise ShadowFilePilotError("file pilot v1 cannot serialize a position")
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "feed_event_id": value.feed_event_id,
        "received_at": value.received_at.isoformat(),
        "bar": to_primitive(value.bar),
        "execution": to_primitive(value.execution),
        "execution_observed_at": value.execution_observed_at.isoformat(),
        "execution_known_at": value.execution_known_at.isoformat(),
        "execution_source_event_id": value.execution_source_event_id,
        "account": to_primitive(value.account),
        "account_observed_at": value.account_observed_at.isoformat(),
        "account_known_at": value.account_known_at.isoformat(),
        "account_snapshot_id": value.account_snapshot_id,
        "source_event_ids": list(value.source_event_ids),
        "approved_intents": [],
        "execution_events": [],
    }


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _jsonl(values: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_json(value) + b"\n" for value in values)


def _read_clock_file(path: Path) -> tuple[ShadowClockInput, ...]:
    if path.is_symlink() or not path.is_file():
        raise ShadowFilePilotError(
            f"shadow input must be a trusted regular file: {path}"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ShadowFilePilotError("shadow input is not readable UTF-8") from exc
    lines = text.splitlines()
    if not lines:
        raise ShadowFilePilotError("shadow input must contain at least one clock")
    values: list[ShadowClockInput] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ShadowFilePilotError(
                f"shadow input contains a blank JSONL row: {line_number}"
            )
        try:
            payload = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ShadowFilePilotError(
                f"shadow input row {line_number} is invalid JSON"
            ) from exc
        try:
            values.append(shadow_clock_input_from_payload(payload))
        except (ShadowFilePilotError, ShadowLiveError) as exc:
            raise ShadowFilePilotError(
                f"shadow input row {line_number} is invalid: {exc}"
            ) from exc
    return tuple(values)


def _engine() -> ContinuousSMCEngine:
    return ContinuousSMCEngine.from_config(
        MODEL_CONFIG,
        runtime_mode="development",
    )


def _new_runner() -> ShadowLiveRunner:
    return ShadowLiveRunner(
        engine=_engine(),
        protocol=load_shadow_live_protocol(SHADOW_PROTOCOL),
        runtime_bindings=shadow_runtime_bindings_from_model_config(MODEL_CONFIG),
        gateway=NullExecutionGateway(),
        model_config_path=MODEL_CONFIG,
    )


def _checkpoint_state(
    runner: ShadowLiveRunner,
    *,
    source_rows_consumed: int,
) -> dict[str, Any]:
    attempts = runner.journal.attempts
    if len(attempts) != source_rows_consumed:
        raise ShadowFilePilotError(
            "checkpoint cursor differs from the in-memory attempt journal"
        )
    return {
        "replay": runner,
        "processed_bars": source_rows_consumed,
        "decision_rows": len(runner.records),
        "next_shard_index": 0,
        "committed_shards": [],
        "last_source_start": None if not attempts else attempts[-1].bar.start,
        "source_rows_consumed": source_rows_consumed,
        "pilot_schema_version": PILOT_SCHEMA_VERSION,
        "record_fingerprint": runner.record_fingerprint,
        "journal_fingerprint": runner.journal.fingerprint,
        "attempt_fingerprint": runner.journal.attempt_fingerprint,
    }


def _require_checkpoint_state(
    state: Mapping[str, Any],
    source: Sequence[ShadowClockInput],
) -> tuple[ShadowLiveRunner, int]:
    if set(state) != set(_CHECKPOINT_FIELDS):
        raise ShadowFilePilotError("shadow checkpoint state fields changed")
    runner = state["replay"]
    cursor = state["source_rows_consumed"]
    if (
        not isinstance(runner, ShadowLiveRunner)
        or type(cursor) is not int
        or cursor < 0
        or cursor > len(source)
        or state["processed_bars"] != cursor
        or state["decision_rows"] != len(runner.records)
        or state["next_shard_index"] != 0
        or state["committed_shards"] != []
        or state["pilot_schema_version"] != PILOT_SCHEMA_VERSION
        or state["record_fingerprint"] != runner.record_fingerprint
        or state["journal_fingerprint"] != runner.journal.fingerprint
        or state["attempt_fingerprint"] != runner.journal.attempt_fingerprint
        or len(runner.journal.attempts) != cursor
        or runner.failure is not None
    ):
        raise ShadowFilePilotError("shadow checkpoint state is inconsistent")
    if any(
        left.input_digest != right.input_digest
        for left, right in zip(runner.journal.attempts, source[:cursor])
    ):
        raise ShadowFilePilotError("shadow checkpoint is not a source prefix")
    return runner, cursor


def _sync_durable_journal(
    path: Path,
    attempts: Sequence[ShadowClockInput],
) -> None:
    existing: tuple[ShadowClockInput, ...] = ()
    if path.exists():
        existing = _read_clock_file(path)
        if len(existing) > len(attempts) or any(
            left.input_digest != right.input_digest
            for left, right in zip(existing, attempts)
        ):
            raise ShadowFilePilotError(
                "durable shadow journal is not a checkpoint prefix"
            )
    atomic_bytes(
        path,
        _jsonl(tuple(shadow_clock_input_payload(item) for item in attempts)),
    )


def _checkpoint(
    store: ReplayCheckpointStore,
    runner: ShadowLiveRunner,
    *,
    bindings: Mapping[str, Any],
    source_rows_consumed: int,
    journal_path: Path,
) -> None:
    # The content-addressed pickle is committed first.  If the process stops
    # before the JSONL replacement, resume accepts the older JSONL only when
    # it is an exact prefix and deterministically catches it up from pickle.
    store.save(
        _checkpoint_state(
            runner,
            source_rows_consumed=source_rows_consumed,
        ),
        bindings=bindings,
    )
    _sync_durable_journal(journal_path, runner.journal.attempts)


def _cold_journal(values: Sequence[ShadowClockInput]) -> ShadowInputJournal:
    journal = ShadowInputJournal()
    for value in values:
        journal.record_attempt(value)
        journal.append(value)
    journal.require_consistent()
    return journal


def preflight_file_pilot_capacity(
    input_path: str | Path,
    *,
    reference_output_directory: str | Path,
    output_parent: str | Path | None = None,
) -> dict[str, Any]:
    """Estimate a full replay from a bounded historical prefix without running it.

    The estimate deliberately stays a lower-bound planning diagnostic.  It
    never unpickles the reference checkpoint, creates an Engine, replays a
    clock, or promotes a historical binding as current evidence.
    """

    source = Path(input_path)
    if source.is_symlink() or not source.is_file():
        raise ShadowFilePilotError("capacity input must be a trusted regular file")
    raw = source.read_bytes()
    lines = raw.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ShadowFilePilotError("capacity input must be non-empty JSONL")
    target_rows = len(lines)

    reference = Path(reference_output_directory)
    run_manifest_path = reference / RUN_MANIFEST
    completed_path = reference / COMPLETED
    checkpoint_manifest_path = reference / "_checkpoint/manifest.json"
    for path, name in (
        (run_manifest_path, "reference run manifest"),
        (completed_path, "reference completion marker"),
        (checkpoint_manifest_path, "reference checkpoint manifest"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ShadowFilePilotError(f"{name} is absent or untrusted")
    try:
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        checkpoint_manifest = json.loads(
            checkpoint_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShadowFilePilotError("capacity reference metadata is invalid") from exc
    reference_rows = completed.get("accepted_clocks")
    state_file = checkpoint_manifest.get("state_file")
    state_sha256 = checkpoint_manifest.get("state_sha256")
    run_manifest_sha = sha256_file(run_manifest_path)
    checkpoint_manifest_sha = sha256_file(checkpoint_manifest_path)
    if (
        run_manifest.get("schema_version") != PILOT_SCHEMA_VERSION
        or completed.get("schema_version") != PILOT_SCHEMA_VERSION
        or completed.get("status") != "complete_engineering_file_replay"
        or completed.get("gate_pass") is not True
        or completed.get("parity_exact") is not True
        or completed.get("coverage_complete") is not True
        or type(reference_rows) is not int
        or reference_rows <= 0
        or completed.get("source_rows") != reference_rows
        or completed.get("attempted_clocks") != reference_rows
        or completed.get("parity_records") != reference_rows
        or run_manifest.get("input_rows") != reference_rows
        or checkpoint_manifest.get("processed_bars") != reference_rows
        or checkpoint_manifest.get("decision_rows") != reference_rows
        or completed.get("run_manifest_sha256") != run_manifest_sha
        or completed.get("checkpoint_manifest_sha256")
        != checkpoint_manifest_sha
        or checkpoint_manifest.get("bindings")
        != {"run_manifest_sha256": run_manifest_sha}
        or not isinstance(state_file, str)
        or not state_file
        or not isinstance(state_sha256, str)
        or state_file != f"state-{state_sha256}.pkl"
    ):
        raise ShadowFilePilotError("capacity reference is not a completed exact prefix")
    state_path = reference / "_checkpoint" / state_file
    journal_path = reference / INPUT_JOURNAL
    records_path = reference / PARITY_RECORDS
    for path, name in (
        (state_path, "reference checkpoint state"),
        (journal_path, "reference input journal"),
        (records_path, "reference parity records"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ShadowFilePilotError(f"{name} is absent or untrusted")
    if (
        state_sha256 != sha256_file(state_path)
        or completed.get("input_journal_sha256") != sha256_file(journal_path)
        or completed.get("parity_records_sha256") != sha256_file(records_path)
        or len(journal_path.read_bytes().splitlines()) != reference_rows
        or len(records_path.read_bytes().splitlines()) != reference_rows
        or state_path.stat().st_size < reference_rows
    ):
        raise ShadowFilePilotError("capacity reference artifact hash differs")

    scale = target_rows / reference_rows
    projected_checkpoint = math.ceil(state_path.stat().st_size * scale)
    projected_journal = math.ceil(journal_path.stat().st_size * scale)
    projected_records = math.ceil(records_path.stat().st_size * scale)
    projected_output = projected_checkpoint + projected_journal + projected_records
    # Live and cold runners coexist at final parity. Pickle bytes are only a
    # storage proxy for each runner, so this remains explicitly a lower bound.
    projected_peak_lower_bound = math.ceil(
        2 * projected_checkpoint + 2 * len(raw) + projected_records
    )
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(Path(output_parent or source.parent))
    current_bindings = dict(shadow_runtime_bindings_from_model_config(MODEL_CONFIG))
    reference_bindings = run_manifest.get("runtime_bindings")
    bindings_match = reference_bindings == current_bindings
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "bounded_capacity_preflight_only",
        "real_time_live": False,
        "multi_day_live_pilot": False,
        "broker_submission": False,
        "engine_created": False,
        "clocks_replayed": 0,
        "input_path": str(source.resolve()),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "target_rows": target_rows,
        "reference_output_directory": str(reference.resolve()),
        "reference_rows": reference_rows,
        "reference_run_manifest_sha256": run_manifest_sha,
        "reference_runtime_bindings": reference_bindings,
        "current_runtime_bindings": current_bindings,
        "reference_runtime_bindings_match_current": bindings_match,
        "historical_reference_promoted_as_current_result": False,
        "estimation_model": "linear_prefix_extrapolation_lower_bound_only",
        "projected_checkpoint_bytes": projected_checkpoint,
        "projected_journal_bytes": projected_journal,
        "projected_parity_record_bytes": projected_records,
        "projected_output_bytes": projected_output,
        "projected_peak_working_set_lower_bound_bytes": projected_peak_lower_bound,
        "available_memory_bytes_at_preflight": int(memory.available),
        "available_disk_bytes_at_preflight": int(disk.free),
        "memory_lower_bound_fits_now": memory.available >= projected_peak_lower_bound,
        "disk_lower_bound_fits_now": disk.free >= projected_output,
        "known_nonlinear_consistency_residual": True,
        "full_6900_replay_authorized": False,
        "capacity_conclusion": "unproven_requires_bounded_benchmark_and_capacity_fix",
    }


def run_file_pilot(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    resume: bool = False,
    checkpoint_clocks: int = 250,
    stop_after_clocks: int | None = None,
) -> dict[str, Any]:
    """Run or resume the bounded engineering replay.

    ``stop_after_clocks`` is a deterministic graceful-pause hook used to
    exercise local checkpoint recovery.  A paused prefix never writes a
    completion claim.
    """

    if type(resume) is not bool:
        raise TypeError("resume must be boolean")
    if type(checkpoint_clocks) is not int or checkpoint_clocks < 1:
        raise ShadowFilePilotError("checkpoint_clocks must be positive")
    if stop_after_clocks is not None and (
        type(stop_after_clocks) is not int or stop_after_clocks < 1
    ):
        raise ShadowFilePilotError("stop_after_clocks must be positive")

    requested_source = Path(input_path)
    if requested_source.is_symlink():
        raise ShadowFilePilotError("shadow input cannot be a symlink")
    source_path = requested_source.resolve()
    source = _read_clock_file(source_path)
    requested_destination = Path(output_directory)
    if requested_destination.is_symlink():
        raise ShadowFilePilotError("shadow output directory cannot be a symlink")
    destination = requested_destination.resolve()
    manifest_payload = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": PILOT_STATUS,
        "authority": SHADOW_LIVE_AUTHORITY,
        "real_time_live": False,
        "multi_day_live_pilot": False,
        "broker_submission": False,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "input_path": str(source_path),
        "input_sha256": sha256_file(source_path),
        "input_rows": len(source),
        "approved_intents_allowed": False,
        "execution_events_allowed": False,
        "account_positions_allowed": False,
        "checkpoint_clocks": checkpoint_clocks,
        "model_config_path": str(MODEL_CONFIG.relative_to(ROOT)),
        "model_config_sha256": sha256_file(MODEL_CONFIG),
        "shadow_protocol_path": str(SHADOW_PROTOCOL.relative_to(ROOT)),
        "shadow_protocol_sha256": sha256_file(SHADOW_PROTOCOL),
        "shadow_protocol_id": load_shadow_live_protocol(
            SHADOW_PROTOCOL
        ).protocol_id,
        "runtime_bindings": dict(
            shadow_runtime_bindings_from_model_config(MODEL_CONFIG)
        ),
        "pilot_script_sha256": sha256_file(Path(__file__)),
    }
    manifest_bytes = _canonical_json(manifest_payload)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_path = destination / RUN_MANIFEST
    completed_path = destination / COMPLETED
    failed_path = destination / FAILED
    if resume:
        if completed_path.exists():
            raise ShadowFilePilotError("completed file pilot cannot be resumed")
        if failed_path.exists():
            raise ShadowFilePilotError("failed file pilot is terminal")
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ShadowFilePilotError("resume requires a trusted run manifest")
        if manifest_path.read_bytes() != manifest_bytes:
            raise ShadowFilePilotError("run manifest differs from requested replay")
    else:
        if destination.exists() and any(destination.iterdir()):
            raise ShadowFilePilotError(
                "new file pilot requires an empty output directory"
            )
        # Validate the exact production Engine/runtime bindings before
        # materializing a new run directory.  A setup failure must not leave
        # behind a manifest that looks resumable.
        runner = _new_runner()
        cursor = 0
        destination.mkdir(parents=True, exist_ok=True)
        atomic_bytes(manifest_path, manifest_bytes)

    checkpoint_store = ReplayCheckpointStore(destination / "_checkpoint")
    checkpoint_bindings = {"run_manifest_sha256": manifest_sha256}
    journal_path = destination / INPUT_JOURNAL
    if resume:
        state = checkpoint_store.load(
            expected_bindings=checkpoint_bindings,
            expected_replay_type=ShadowLiveRunner,
        )
        runner, cursor = _require_checkpoint_state(state, source)
        _sync_durable_journal(journal_path, runner.journal.attempts)

    while cursor < len(source):
        value = source[cursor]
        try:
            runner.process(value)
        except Exception as exc:
            attempts = runner.journal.attempts
            attempted_cursor = len(attempts)
            prefix_valid = (
                attempted_cursor <= len(source)
                and all(
                    left.input_digest == right.input_digest
                    for left, right in zip(
                        attempts,
                        source[:attempted_cursor],
                    )
                )
            )
            checkpointed_failure = False
            if prefix_valid and attempted_cursor == cursor + 1:
                _checkpoint(
                    checkpoint_store,
                    runner,
                    bindings=checkpoint_bindings,
                    source_rows_consumed=attempted_cursor,
                    journal_path=journal_path,
                )
                checkpointed_failure = True
            elif not (prefix_valid and attempted_cursor == cursor):
                raise ShadowFilePilotError(
                    "failed shadow update corrupted its source/attempt cursor"
                ) from exc
            failure_payload = {
                "schema_version": PILOT_SCHEMA_VERSION,
                "status": "terminal_failure",
                "real_time_live": False,
                "source_rows_consumed": attempted_cursor,
                "attempted_clock_journaled": attempted_cursor == cursor + 1,
                "checkpointed_failure": checkpointed_failure,
                "failure": to_primitive(runner.failure),
                "error_type": type(exc).__name__,
                "error_message": str(exc).strip() or "exception_without_message",
            }
            atomic_bytes(failed_path, _canonical_json(failure_payload))
            raise
        cursor += 1
        due = cursor % checkpoint_clocks == 0 or cursor == len(source)
        pausing = (
            stop_after_clocks is not None
            and cursor >= stop_after_clocks
            and cursor < len(source)
        )
        if due or pausing:
            _checkpoint(
                checkpoint_store,
                runner,
                bindings=checkpoint_bindings,
                source_rows_consumed=cursor,
                journal_path=journal_path,
            )
        if pausing:
            return {
                "schema_version": PILOT_SCHEMA_VERSION,
                "status": "paused_checkpointed_prefix",
                "real_time_live": False,
                "source_rows_consumed": cursor,
                "input_rows": len(source),
                "record_fingerprint": runner.record_fingerprint,
                "journal_fingerprint": runner.journal.fingerprint,
            }

    durable_values = _read_clock_file(journal_path)
    if len(durable_values) != len(source):
        raise ShadowFilePilotError("durable journal does not cover the source")
    durable_journal = _cold_journal(durable_values)
    if (
        durable_journal.fingerprint != runner.journal.fingerprint
        or durable_journal.attempt_fingerprint
        != runner.journal.attempt_fingerprint
    ):
        raise ShadowFilePilotError("durable journal differs from live replay")
    cold = replay_shadow_journal(
        durable_journal,
        engine_factory=_engine,
        protocol=runner.protocol,
        runtime_bindings=runner.runtime_bindings,
    )
    audit = audit_shadow_parity(runner, cold)
    audit.require_exact()
    records_path = destination / PARITY_RECORDS
    atomic_bytes(
        records_path,
        _jsonl(tuple(to_primitive(item) for item in runner.records)),
    )
    checkpoint_manifest = checkpoint_store.manifest_path
    result = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "complete_engineering_file_replay",
        "authority": SHADOW_LIVE_AUTHORITY,
        "real_time_live": False,
        "multi_day_live_pilot_completed": False,
        "broker_submission_authorized": False,
        "source_rows": len(source),
        "accepted_clocks": len(runner.journal.events),
        "attempted_clocks": len(runner.journal.attempts),
        "parity_records": len(runner.records),
        "approved_intents": 0,
        "execution_events": 0,
        "external_submission_attempts": runner.gateway.submission_attempts,
        "run_manifest_sha256": manifest_sha256,
        "input_journal_sha256": sha256_file(journal_path),
        "parity_records_sha256": sha256_file(records_path),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
        "live_record_fingerprint": runner.record_fingerprint,
        "cold_record_fingerprint": cold.record_fingerprint,
        "live_journal_fingerprint": runner.journal.fingerprint,
        "cold_journal_fingerprint": cold.journal.fingerprint,
        "parity_audit_id": audit.audit_id,
        "parity_exact": audit.exact_match,
        "coverage_complete": audit.coverage_complete,
        "gate_pass": audit.gate_pass,
        "checkpoint_trust_boundary": "trusted_local_pickle_only",
    }
    atomic_bytes(completed_path, _canonical_json(result))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--capacity-preflight-only",
        action="store_true",
        help="estimate from a completed bounded prefix without creating an Engine",
    )
    parser.add_argument(
        "--reference-output-directory",
        type=Path,
        help="completed bounded prefix used only as a capacity-size reference",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-clocks", type=int, default=250)
    parser.add_argument(
        "--stop-after-clocks",
        type=int,
        help="gracefully checkpoint a bounded prefix without a completion claim",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.capacity_preflight_only:
        if args.reference_output_directory is None:
            raise ShadowFilePilotError(
                "capacity preflight requires --reference-output-directory"
            )
        result = preflight_file_pilot_capacity(
            args.input,
            reference_output_directory=args.reference_output_directory,
            output_parent=(
                None
                if args.output_directory is None
                else args.output_directory.parent
            ),
        )
    else:
        if args.output_directory is None:
            raise ShadowFilePilotError("file replay requires --output-directory")
        if args.reference_output_directory is not None:
            raise ShadowFilePilotError(
                "--reference-output-directory is capacity-preflight-only"
            )
        result = run_file_pilot(
            args.input,
            args.output_directory,
            resume=args.resume,
            checkpoint_clocks=args.checkpoint_clocks,
            stop_after_clocks=args.stop_after_clocks,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
