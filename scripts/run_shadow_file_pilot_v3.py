#!/usr/bin/env python3
"""Run an externally capacity-authorized Phase-9 v3 historical simulation.

This command is intentionally not a real-time pilot.  It requires a current,
exact 6,900-clock v3 input sidecar and a separately frozen capacity artifact.
Inputs are streamed, every attempt is flushed and fsynced before Engine
processing, runtime checkpoints exclude WAL/record history, and final parity is
verified after releasing the live runner.  No artifact produced here can
authorize live trading, a sealed reveal, or a real-time operational claim.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterator, Mapping, Sequence

import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_shadow_file_pilot import (  # noqa: E402
    _new_runner,
    iter_shadow_clock_file,
    shadow_clock_input_from_payload,
    shadow_clock_input_payload,
)
from smc_trader.artifact_stream import atomic_bytes, sha256_file  # noqa: E402
from smc_trader.calibration_replay import ReplayCheckpointStore  # noqa: E402
from smc_trader.shadow_live import (  # noqa: E402
    ShadowClockInput,
    ShadowLiveRunner,
    ShadowParityRecord,
)
from smc_trader.shadow_operational import (  # noqa: E402
    CAPACITY_AUTHORIZATION_SCHEMA_VERSION,
    CompactShadowRuntimeEnvelope,
    DurableShadowWAL,
    HistoricalShadowWindow,
    InstrumentMappingBinding,
    OperationalShadowSession,
    Phase9OperationalError,
    load_capacity_authorization,
    load_shadow_operational_protocol,
    phase9_historical_window,
    runtime_code_environment_identity,
    validate_phase9_bundle_v3_payload,
)


V3_RUN_SCHEMA_VERSION = "phase9_shadow_file_replay_v3"
V3_RUN_STATUS = "historical_engineering_simulation_only"
OPERATIONAL_PROTOCOL_PATH = ROOT / "configs/phase9_shadow_operational_v1.json"
RUN_MANIFEST = "run_manifest.json"
COMPLETED = "COMPLETED.json"
FAILED = "FAILED.json"
WAL = "shadow_attempt_commit.wal.jsonl"
CURSOR = "compact_cursor.json"


class ShadowFilePilotV3Error(Phase9OperationalError):
    """Raised when the v3 historical simulation fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _duplicate_guard(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ShadowFilePilotV3Error(f"JSON contains duplicate key {key!r}")
        payload[key] = value
    return payload


def _read_json(path: Path, *, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ShadowFilePilotV3Error(f"{name} must be a trusted regular file")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_guard,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShadowFilePilotV3Error(f"{name} is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ShadowFilePilotV3Error(f"{name} root must be an object")
    return payload


class _CanonicalStringSequenceDigest:
    def __init__(self) -> None:
        self._hasher = hashlib.sha256()
        self._hasher.update(b"[")
        self._count = 0

    def append(self, value: str) -> None:
        if self._count:
            self._hasher.update(b",")
        self._hasher.update(_canonical_bytes(value))
        self._count += 1

    @property
    def fingerprint(self) -> str:
        current = self._hasher.copy()
        current.update(b"]")
        return current.hexdigest()


def _input_sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _stream_file_metadata(path: Path) -> tuple[str, int, int]:
    if path.is_symlink() or not path.is_file():
        raise ShadowFilePilotV3Error("v3 input must be a trusted regular file")
    digest = hashlib.sha256()
    rows = 0
    total_bytes = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            total_bytes += len(line)
            if not line.strip():
                raise ShadowFilePilotV3Error("v3 input contains a blank row")
            rows += 1
    if rows == 0:
        raise ShadowFilePilotV3Error("v3 input is empty")
    return digest.hexdigest(), total_bytes, rows


def _validate_clock_census(
    path: Path,
    *,
    expected_digest_sequence: str,
    window: HistoricalShadowWindow,
) -> None:
    count = 0
    real = 0
    synthetic: list[Any] = []
    first = None
    last = None
    digest_sequence = _CanonicalStringSequenceDigest()
    for value in iter_shadow_clock_file(path):
        count += 1
        first = value.bar.end if first is None else first
        last = value.bar.end
        digest_sequence.append(value.input_digest)
        if value.bar.synthetic_no_trade:
            synthetic.append(value.bar.end)
        else:
            real += 1
        if (
            value.bar.symbol != window.symbol
            or value.bar.instrument_id != window.instrument_id
            or value.bar.data_gap_before_minutes != 0
            or value.approved_intents
            or value.execution_events
            or value.account.position is not None
            or value.account.equity != 100_000.0
            or value.account.open_risk_fraction != 0.0
            or value.account.requested_risk_fraction != 0.0
            or value.account.quantity != 1
            or value.account.point_value != window.point_value
        ):
            raise ShadowFilePilotV3Error(
                "v3 input contract/account/zero-order census changed"
            )
    if (
        count != window.rows
        or real != window.real_rows
        or tuple(synthetic) != window.synthetic_clocks
        or first != window.first_decision_clock
        or last != window.last_decision_clock
        or digest_sequence.fingerprint != expected_digest_sequence
    ):
        raise ShadowFilePilotV3Error("v3 input semantic census differs")


def admit_week1_bundle_v3(
    input_path: str | Path,
    *,
    sidecar_path: str | Path | None = None,
    expected_window_id: str = "W1",
) -> tuple[Mapping[str, Any], Mapping[str, Any], str, str]:
    """Admit sidecar/source/runtime identity before an Engine is constructed."""

    source = Path(input_path).resolve()
    sidecar = Path(sidecar_path or _input_sidecar_path(source)).resolve()
    payload = _read_json(sidecar, name="v3 input sidecar")
    window = phase9_historical_window(expected_window_id)
    declared_window = payload.get("window")
    if (
        not isinstance(declared_window, Mapping)
        or declared_window.get("id") != window.window_id
    ):
        raise ShadowFilePilotV3Error(
            "v3 input sidecar does not match the requested W1/W2 window"
        )
    protocol = load_shadow_operational_protocol(OPERATIONAL_PROTOCOL_PATH)
    runtime_identity = runtime_code_environment_identity(ROOT, require_clean=True)
    declared_output = payload.get("output")
    if not isinstance(declared_output, Mapping):
        raise ShadowFilePilotV3Error("v3 sidecar output binding is absent")
    # First pass validates every non-input dependency before opening JSONL.
    validate_phase9_bundle_v3_payload(
        payload,
        output_path=source,
        output_sha256=declared_output.get("sha256"),
        output_bytes=declared_output.get("bytes"),
        output_rows=declared_output.get("rows"),
        runtime_identity=runtime_identity,
        operational_protocol=protocol,
        root=ROOT,
        verify_source_files=True,
    )
    input_sha, input_bytes, input_rows = _stream_file_metadata(source)
    validate_phase9_bundle_v3_payload(
        payload,
        output_path=source,
        output_sha256=input_sha,
        output_bytes=input_bytes,
        output_rows=input_rows,
        runtime_identity=runtime_identity,
        operational_protocol=protocol,
        root=ROOT,
        verify_source_files=True,
    )
    _validate_clock_census(
        source,
        expected_digest_sequence=declared_output.get("input_digest_sequence_sha256"),
        window=window,
    )
    return payload, runtime_identity, input_sha, sha256_file(sidecar)


def _mapping(
    input_sha256: str,
    window: HistoricalShadowWindow,
) -> InstrumentMappingBinding:
    return InstrumentMappingBinding(
        mapping_id=(f"shadow-instrument-mapping:{window.alias.lower()}:{input_sha256}"),
        mapping_version=f"2024-06-{window.alias.lower()}-v3",
        mode="historical_simulation",
        logical_instrument_id="NQ:front",
        vendor_symbol=window.symbol,
        vendor_instrument_id=window.instrument_id,
        tick_size=window.tick_size,
        point_value=window.point_value,
        effective_from=window.start,
        effective_until=window.end_exclusive,
        source_identity=input_sha256,
    )


def _record_from_payload(payload: Any) -> ShadowParityRecord:
    if not isinstance(payload, Mapping):
        raise ShadowFilePilotV3Error("WAL parity result is not an object")
    init_fields = {item.name for item in fields(ShadowParityRecord) if item.init}
    if set(payload) != init_fields | {"record_id"}:
        raise ShadowFilePilotV3Error("WAL parity result fields changed")
    record = ShadowParityRecord(**{name: payload[name] for name in init_fields})
    if record.record_id != payload["record_id"]:
        raise ShadowFilePilotV3Error("WAL parity record identity differs")
    return record


def _history(
    transactions,
    count: int,
) -> tuple[tuple[ShadowClockInput, ...], tuple[ShadowParityRecord, ...]]:
    selected = tuple(transactions[:count])
    if len(selected) != count or any(not item.committed for item in selected):
        raise ShadowFilePilotV3Error("checkpoint WAL prefix is incomplete")
    return (
        tuple(item.input_value for item in selected),
        tuple(_record_from_payload(item.result_payload) for item in selected),
    )


def _checkpoint(
    store: ReplayCheckpointStore,
    runner: ShadowLiveRunner,
    session: OperationalShadowSession,
    *,
    bindings: Mapping[str, Any],
    cursor_path: Path,
) -> Mapping[str, Any]:
    cursor = session.wal.committed_count
    envelope = CompactShadowRuntimeEnvelope(
        runtime_state=runner.compact_runtime_checkpoint(),
        committed_clocks=cursor,
        wal_fingerprint=session.wal.fingerprint,
        protocol_id=session.protocol.protocol_id,
        mapping_id=session.mapping.mapping_id,
    )
    state = {
        "replay": envelope,
        "processed_bars": cursor,
        "decision_rows": cursor,
        "next_shard_index": 0,
        "committed_shards": [],
        "last_source_start": (
            None
            if session.wal.last_committed_input is None
            else session.wal.last_committed_input.bar.start
        ),
    }
    manifest = store.save(state, bindings=bindings)
    cursor_payload = session.compact_checkpoint_payload(
        engine_checkpoint_sha256=manifest["state_sha256"],
    )
    cursor_payload["checkpoint_manifest_sha256"] = sha256_file(store.manifest_path)
    atomic_bytes(cursor_path, _canonical_bytes(cursor_payload))
    return manifest


def _restore(
    store: ReplayCheckpointStore,
    session: OperationalShadowSession,
    *,
    bindings: Mapping[str, Any],
    cursor_path: Path,
) -> ShadowLiveRunner:
    state = store.load(
        expected_bindings=bindings,
        expected_replay_type=CompactShadowRuntimeEnvelope,
    )
    envelope = state["replay"]
    if (
        envelope.committed_clocks != state["processed_bars"]
        or envelope.protocol_id != session.protocol.protocol_id
        or envelope.mapping_id != session.mapping.mapping_id
    ):
        raise ShadowFilePilotV3Error("compact runtime envelope binding differs")
    cursor = _read_json(cursor_path, name="compact cursor")
    checkpoint_manifest = _read_json(
        store.manifest_path,
        name="checkpoint manifest",
    )
    if (
        cursor.get("schema_version") != "phase9_shadow_compact_cursor_v1"
        or cursor.get("real_time_live_claim") is not False
        or cursor.get("external_submission_allowed") is not False
        or cursor.get("attempted_clocks") != envelope.committed_clocks
        or cursor.get("committed_clocks") != envelope.committed_clocks
        or cursor.get("wal_fingerprint") != envelope.wal_fingerprint
        or cursor.get("engine_checkpoint_sha256")
        != checkpoint_manifest.get("state_sha256")
        or cursor.get("checkpoint_manifest_sha256") != sha256_file(store.manifest_path)
        or cursor.get("protocol_id") != envelope.protocol_id
        or cursor.get("mapping_id") != envelope.mapping_id
    ):
        raise ShadowFilePilotV3Error("compact cursor differs from checkpoint")
    inputs, records = _history(
        session.wal.transactions,
        envelope.committed_clocks,
    )
    runner = ShadowLiveRunner.from_compact_runtime_checkpoint(
        envelope.runtime_state,
        journal_events=inputs,
        records=records,
    )
    # WAL commits after the last checkpoint are replayed and matched exactly.
    for transaction in session.wal.transactions[envelope.committed_clocks :]:
        if not transaction.committed:
            break
        record = runner.process(transaction.input_value)
        if record != _record_from_payload(transaction.result_payload):
            raise ShadowFilePilotV3Error(
                "post-checkpoint WAL replay differs from durable result"
            )
    if session.wal.pending_input is not None:
        session.reconnect()
        recovered = session.recover_pending(process=runner.process)
        if not isinstance(recovered, ShadowParityRecord):
            raise ShadowFilePilotV3Error("pending recovery did not publish parity")
    return runner


def _source_stream_after_wal(
    path: Path,
    wal: DurableShadowWAL,
) -> Iterator[ShadowClockInput]:
    durable = wal.transactions
    for index, value in enumerate(iter_shadow_clock_file(path)):
        if index < len(durable):
            if value.input_digest != durable[index].input_value.input_digest:
                raise ShadowFilePilotV3Error("WAL is not an exact input prefix")
            continue
        yield value


def run_shadow_file_pilot_v3(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    capacity_authorization_path: str | Path,
    capacity_authorization_sha256: str,
    sidecar_path: str | Path | None = None,
    resume: bool = False,
    window_id: str = "W1",
) -> Mapping[str, Any]:
    """Run the capacity-authorized historical simulation; never self-authorize."""

    source = Path(input_path).resolve()
    destination = Path(output_directory).resolve()
    window = phase9_historical_window(window_id)
    bundle, runtime_identity, input_sha, sidecar_sha = admit_week1_bundle_v3(
        source,
        sidecar_path=sidecar_path,
        expected_window_id=window.alias,
    )
    protocol = load_shadow_operational_protocol(OPERATIONAL_PROTOCOL_PATH)
    authorization = load_capacity_authorization(
        capacity_authorization_path,
        expected_sha256=capacity_authorization_sha256,
        input_bundle_sha256=input_sha,
        input_sidecar_sha256=sidecar_sha,
        runtime_identity_sha256=runtime_identity["runtime_identity_sha256"],
        required_rows=window.rows,
    )
    resource_parent = Path(output_directory).resolve().parent
    if (
        psutil.virtual_memory().available < authorization.minimum_available_memory_bytes
        or shutil.disk_usage(resource_parent).free
        < authorization.minimum_available_disk_bytes
    ):
        raise ShadowFilePilotV3Error(
            "current resources are below the external capacity authorization"
        )
    mapping = _mapping(input_sha, window)
    run_manifest = {
        "schema_version": V3_RUN_SCHEMA_VERSION,
        "status": V3_RUN_STATUS,
        "historical_simulation_only": True,
        "real_time_live": False,
        "multi_day_live_pilot": False,
        "external_submission_allowed": False,
        "sealed_oos_opened": False,
        "input_path": str(source),
        "input_sha256": input_sha,
        "input_rows": window.rows,
        "window_id": window.window_id,
        "input_sidecar_sha256": sidecar_sha,
        "runtime_code_environment": runtime_identity,
        "operational_protocol_id": protocol.protocol_id,
        "operational_protocol_sha256": protocol.source_sha256,
        "mapping_id": mapping.mapping_id,
        "capacity_authorization_schema": CAPACITY_AUTHORIZATION_SCHEMA_VERSION,
        "capacity_authorization_id": authorization.authorization_id,
        "capacity_authorization_artifact_sha256": authorization.artifact_sha256,
        "runner_sha256": sha256_file(Path(__file__)),
        "checkpoint_interval_clocks": int(
            protocol.durability_policy["checkpoint_interval_clocks"]
        ),
        "source_manifest_status": bundle["status"],
    }
    run_manifest_bytes = _canonical_bytes(run_manifest)
    run_manifest_sha = hashlib.sha256(run_manifest_bytes).hexdigest()
    manifest_path = destination / RUN_MANIFEST
    completed_path = destination / COMPLETED
    failed_path = destination / FAILED
    if resume:
        if completed_path.exists() or failed_path.exists():
            raise ShadowFilePilotV3Error("terminal v3 run cannot resume")
        if (
            not manifest_path.is_file()
            or manifest_path.read_bytes() != run_manifest_bytes
        ):
            raise ShadowFilePilotV3Error("v3 run manifest differs on resume")
    else:
        if destination.exists() and any(destination.iterdir()):
            raise ShadowFilePilotV3Error("new v3 run requires an empty directory")
        destination.mkdir(parents=True, exist_ok=True)
        atomic_bytes(manifest_path, run_manifest_bytes)

    wal = DurableShadowWAL(
        destination / WAL,
        encode_input=shadow_clock_input_payload,
        decode_input=shadow_clock_input_from_payload,
    )
    session = OperationalShadowSession(
        protocol=protocol,
        mapping=mapping,
        wal=wal,
        mode="historical_simulation",
    )
    checkpoint_store = ReplayCheckpointStore(destination / "_checkpoint")
    checkpoint_bindings = {"run_manifest_sha256": run_manifest_sha}
    cursor_path = destination / CURSOR
    try:
        runner = (
            _restore(
                checkpoint_store,
                session,
                bindings=checkpoint_bindings,
                cursor_path=cursor_path,
            )
            if resume
            else _new_runner()
        )
        checkpoint_interval = int(
            protocol.durability_policy["checkpoint_interval_clocks"]
        )
        for value in _source_stream_after_wal(source, wal):
            outcome = session.ingest(value, process=runner.process)
            if outcome.duplicate:
                raise ShadowFilePilotV3Error(
                    "source stream unexpectedly repeated a durable input"
                )
            if wal.committed_count % checkpoint_interval == 0:
                manifest = _checkpoint(
                    checkpoint_store,
                    runner,
                    session,
                    bindings=checkpoint_bindings,
                    cursor_path=cursor_path,
                )
                state_path = checkpoint_store.root / manifest["state_file"]
                if state_path.stat().st_size > authorization.maximum_checkpoint_bytes:
                    raise ShadowFilePilotV3Error(
                        "checkpoint exceeded external capacity authorization"
                    )
                if (
                    psutil.Process().memory_info().rss
                    > authorization.maximum_peak_working_set_bytes
                ):
                    raise ShadowFilePilotV3Error(
                        "working set exceeded external capacity authorization"
                    )
        if wal.committed_count != window.rows or wal.pending_input is not None:
            raise ShadowFilePilotV3Error(
                f"v3 WAL does not cover exact {window.alias} census"
            )
        _checkpoint(
            checkpoint_store,
            runner,
            session,
            bindings=checkpoint_bindings,
            cursor_path=cursor_path,
        )
        live_record_fingerprint = runner.record_fingerprint
        live_journal_fingerprint = runner.journal.fingerprint
        if runner.gateway.submission_attempts != 0:
            raise ShadowFilePilotV3Error("NullGateway submission count changed")
        del runner
        gc.collect()

        cold = _new_runner()
        parity_mismatches = 0
        for transaction in wal.transactions:
            if not transaction.committed:
                raise ShadowFilePilotV3Error("cold replay encountered pending WAL")
            record = cold.process(transaction.input_value)
            if record != _record_from_payload(transaction.result_payload):
                parity_mismatches += 1
                break
        if (
            parity_mismatches
            or cold.record_fingerprint != live_record_fingerprint
            or cold.journal.fingerprint != live_journal_fingerprint
            or cold.gateway.submission_attempts != 0
        ):
            raise ShadowFilePilotV3Error("v3 cold replay parity differs")
        result = {
            "schema_version": V3_RUN_SCHEMA_VERSION,
            "status": "complete_historical_engineering_simulation",
            "historical_simulation_only": True,
            "real_time_live": False,
            "multi_day_live_pilot_completed": False,
            "operational_gate_pass": False,
            "full_6900_file_replay_complete": True,
            "input_rows": window.rows,
            "window_id": window.window_id,
            "attempted_clocks": wal.attempted_count,
            "committed_clocks": wal.committed_count,
            "record_fingerprint": live_record_fingerprint,
            "journal_fingerprint": live_journal_fingerprint,
            "wal_fingerprint": wal.fingerprint,
            "external_submission_attempts": 0,
            "capacity_authorization_id": authorization.authorization_id,
            "run_manifest_sha256": run_manifest_sha,
            "checkpoint_manifest_sha256": sha256_file(checkpoint_store.manifest_path),
            "compact_cursor_sha256": sha256_file(cursor_path),
            "metrics": dict(session.metrics),
            "sealed_oos_opened": False,
        }
        atomic_bytes(completed_path, _canonical_bytes(result))
        return result
    except BaseException as exc:
        failure = {
            "schema_version": V3_RUN_SCHEMA_VERSION,
            "status": "terminal_failure",
            "real_time_live": False,
            "committed_clocks": wal.committed_count,
            "attempted_clocks": wal.attempted_count,
            "wal_fingerprint": wal.fingerprint,
            "error_type": type(exc).__name__,
            "error_message": str(exc).strip() or "exception_without_message",
        }
        atomic_bytes(failed_path, _canonical_bytes(failure))
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-sidecar", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--capacity-authorization", required=True, type=Path)
    parser.add_argument("--capacity-authorization-sha256", required=True)
    parser.add_argument(
        "--window-id",
        choices=("W1", "W2"),
        default="W1",
        help="registered June development window (default: W1)",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_shadow_file_pilot_v3(
        args.input,
        args.output_directory,
        capacity_authorization_path=args.capacity_authorization,
        capacity_authorization_sha256=args.capacity_authorization_sha256,
        sidecar_path=args.input_sidecar,
        resume=args.resume,
        window_id=args.window_id,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
