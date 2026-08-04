#!/usr/bin/env python3
"""Verify the fixed EXP016 causal projection; governance stays in its wrapper."""
from __future__ import annotations

import hashlib, json, math, os, stat, struct, subprocess, sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "EXP-SMC-3.0.2-016-DISPLACEMENT-PRODUCTION-IDENTITY-CUSTODY"
PREREG = "6a092dcfbce1845bc4b0eff946a69fc7315bbbcfc7827642a2dceba7c3aaf301"
SOURCE = Path("data/processed/nq_1m_previous_session_front_v2_3_pre_holdout_2017_20260331.parquet")
EXPLICIT = Path(f"{SOURCE}.manifest.json")
DERIVED = SOURCE.with_suffix(".manifest.json")
OUTPUT = Path("data/processed/nq_1m_previous_session_front_v3_exp016_discovery_2017_2021.parquet")
OUTPUT_MANIFEST = Path(f"{OUTPUT}.manifest.json")
TEMPORARY = Path(f"{OUTPUT}.tmp")
MATERIALIZER = Path("scripts/materialize_ohlcv_preholdout.py")
ATTEMPT_ROOT = Path("artifacts/v3_exp016_displacement_gate2_physical_001")
CORE_RESULT = ATTEMPT_ROOT / "CORE_RESULT.json"
SEMANTIC_ROOT = Path("outputs/v3_exp016_displacement/discovery-2017-2021-attempt001")
SOURCE_SHA = "6d2f0b36097e779867277a070073bc157bf58891769478d41f4efc8763ba0204"
SOURCE_BYTES = 53_010_541
EXPLICIT_SHA = "6b1c2a750543434b546526fdf80ba63200290b46efdc0c7bcdf423d342c8ee04"
MATERIALIZER_SHA = "82cd593ca7fa8b6659316634e6eadeb872a7be582e24edfed0815d8d63c43fbe"
START = "2017-01-03T18:00:00-05:00"
CUTOFF = "2022-01-01T00:00:00-05:00"
UPSTREAM_END = "2026-04-01T00:00:00-04:00"
TIMEZONE = "America/New_York"
BATCH_ROWS = 4096
ROW_GROUP_ROWS = 4096
FIELDS = ("open", "high", "low", "close", "volume", "symbol", "instrument_id", "ts")
STABLE_FIELDS = ("st_dev", "st_ino", "st_mode", "st_size", "st_blocks", "st_mtime_ns", "st_ctime_ns", "st_flags_raw")
SF_DATALESS = 0x40000000
SELECTION = "highest total volume from the strictly prior completed Globex session"
DOMAIN = "EXP014-ORDERED-TYPED-ROW-COMMITMENT-v1"
DENIALS = {
    "semantic_runner_authorized": False, "mbo_authorized": False,
    "sealed_data_authorized": False, "future_or_action_authorized": False,
    "economic_evaluation_authorized": False, "gate3_authorized": False,
    "production_release_authorized": False,
}


def _path(relative: Path) -> Path: return ROOT / relative


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _parent_chain(relative: Path) -> None:
    current = ROOT
    for part in relative.parts[:-1]:
        current /= part
        try:
            value = os.lstat(current)
        except FileNotFoundError:
            break
        _require(stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode), f"unsafe ancestor: {relative}")


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def _stable(value: os.stat_result) -> dict[str, int]:
    names = ("st_dev", "st_ino", "st_mode", "st_size", "st_blocks", "st_mtime_ns", "st_ctime_ns", "st_flags")
    _require(all(hasattr(value, name) for name in names), "stable stat unavailable")
    return {name: int(getattr(value, "st_flags" if name == "st_flags_raw" else name)) for name in STABLE_FIELDS}


def _leaf(relative: Path, allocated: bool, size: int | None) -> dict[str, int]:
    _parent_chain(relative)
    result = _stable(os.lstat(_path(relative)))
    _require(stat.S_ISREG(result["st_mode"]), f"not an ordinary leaf: {relative}")
    if allocated:
        _require(not result["st_flags_raw"] & SF_DATALESS, f"dataless leaf: {relative}")
        _require(result["st_blocks"] > 0, f"leaf has no local allocation: {relative}")
    if size is not None:
        _require(result["st_size"] == size, f"logical size mismatch: {relative}")
    return result


def _absent(relative: Path) -> None:
    _parent_chain(relative)
    try:
        os.lstat(_path(relative))
    except FileNotFoundError:
        return
    raise FileExistsError(f"create-once target exists: {relative}")


def _directory(relative: Path) -> None:
    _parent_chain(relative)
    value = os.lstat(_path(relative))
    _require(stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode), f"not an ordinary directory: {relative}")


@contextmanager
def _fenced(relative: Path, *, allocated: bool = False, size: int | None = None):
    _require(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW unavailable")
    before = _leaf(relative, allocated, size)
    descriptor = os.open(_path(relative), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        _require(_stable(os.fstat(descriptor)) == before, f"descriptor/path mismatch: {relative}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            yield handle, before
        _require(_stable(os.fstat(descriptor)) == before, f"descriptor changed: {relative}")
        _require(_leaf(relative, allocated, size) == before, f"path changed: {relative}")
    finally:
        os.close(descriptor)


def _hash_file(relative: Path, *, allocated: bool = False, size: int | None = None) -> tuple[str, int, dict[str, int]]:
    digest = hashlib.sha256()
    count = 0
    with _fenced(relative, allocated=allocated, size=size) as (handle, snapshot):
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
            count += len(block)
    return digest.hexdigest(), count, snapshot


def _json_file(relative: Path) -> tuple[dict[str, object], str, int, dict[str, int]]:
    digest = hashlib.sha256()
    data = bytearray()
    with _fenced(relative) as (handle, snapshot):
        while block := handle.read(65_536):
            digest.update(block)
            data.extend(block)
            _require(len(data) <= 1_048_576, f"manifest too large: {relative}")
    value = json.loads(bytes(data).decode())
    _require(isinstance(value, dict), f"manifest is not an object: {relative}")
    return value, digest.hexdigest(), len(data), snapshot


def _fsync_leaf(relative: Path, *, allocated: bool = False) -> dict[str, int]:
    with _fenced(relative, allocated=allocated) as (handle, snapshot):
        os.fsync(handle.fileno())
    return snapshot


def _fsync_dir(relative: Path) -> None:
    _parent_chain(relative)
    descriptor = os.open(_path(relative), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        _require(stat.S_ISDIR(os.fstat(descriptor).st_mode), f"not an ordinary directory: {relative}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_result(value: object) -> None:
    data = _canonical(value)
    _parent_chain(CORE_RESULT)
    descriptor = os.open(_path(CORE_RESULT), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        _require(stream.write(data) == len(data), "short CORE_RESULT write")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_dir(ATTEMPT_ROOT)


def _progress(phase: str, percent: int, **details: object) -> None:
    print(json.dumps({"phase": phase, "completed_percent": percent, **details}, sort_keys=True, separators=(",", ":")), flush=True)


def _validate_schema(schema: pa.Schema) -> None:
    _require(tuple(schema.names) == FIELDS, "schema field order/set mismatch")
    _require(schema.field("ts").type == pa.timestamp("ns", tz=TIMEZONE), "timestamp must be named-zone nanoseconds")
    _require(all(pa.types.is_floating(schema.field(name).type) for name in FIELDS[:4]), "OHLC must be floating")
    volume = schema.field("volume").type
    _require(pa.types.is_integer(volume) or pa.types.is_floating(volume), "volume must be numeric")
    symbol = schema.field("symbol").type
    symbol = symbol.value_type if pa.types.is_dictionary(symbol) else symbol
    _require(pa.types.is_string(symbol) or pa.types.is_large_string(symbol), "symbol must be a logical string")
    instrument = schema.field("instrument_id").type
    _require(pa.types.is_signed_integer(instrument) and instrument.bit_width <= 64, "instrument_id must be signed and int64-representable")


def _schema_info(parquet: pq.ParquetFile) -> dict[str, object]:
    schema = parquet.schema_arrow
    _validate_schema(schema)
    arrow = schema.serialize().to_pybytes()
    return {"_schema": schema, "field_order": list(schema.names), "arrow_schema_hex": arrow.hex(),
            "parquet_schema_rendering": str(parquet.schema)}


def _schema_only() -> dict[str, object]:
    with _fenced(SOURCE, allocated=True, size=SOURCE_BYTES) as (handle, snapshot):
        result = _schema_info(pq.ParquetFile(handle))
    result["stable_stat"] = snapshot
    return result


def _u64(value: int) -> bytes: return struct.pack(">Q", value)


def _blob(value: bytes) -> bytes: return _u64(len(value)) + value


def _metadata(digest: object, metadata: dict[bytes, bytes] | None) -> None:
    items = sorted((bytes(key), bytes(value)) for key, value in (metadata or {}).items())
    digest.update(_u64(len(items)))
    for key, value in items:
        digest.update(_blob(key) + _blob(value))


def _commitment(schema: pa.Schema):
    digest = hashlib.sha256()
    digest.update(b"D" + _blob(DOMAIN.encode()) + b"S" + _u64(len(schema)))
    for field in schema:
        digest.update(b"F" + _blob(field.name.encode()) + _blob(str(field.type).encode()))
        digest.update(b"\x01" if field.nullable else b"\x00")
        _metadata(digest, field.metadata)
    _metadata(digest, schema.metadata)
    return digest


def _scalar(value: pa.Scalar, arrow_type: pa.DataType) -> tuple[bytes, bytes]:
    _require(value.is_valid, "required field contains null")
    if pa.types.is_timestamp(arrow_type):
        return b"T", int(value.cast(pa.int64()).as_py()).to_bytes(8, "big", signed=True)
    if pa.types.is_floating(arrow_type):
        number = float(value.as_py())
        _require(math.isfinite(number), "nonfinite floating value")
        return b"F", struct.pack(">d", number)
    if pa.types.is_signed_integer(arrow_type):
        return b"I", int(value.as_py()).to_bytes(arrow_type.bit_width // 8, "big", signed=True)
    if pa.types.is_unsigned_integer(arrow_type):
        return b"U", int(value.as_py()).to_bytes(arrow_type.bit_width // 8, "big", signed=False)
    logical = arrow_type.value_type if pa.types.is_dictionary(arrow_type) else arrow_type
    if pa.types.is_string(logical) or pa.types.is_large_string(logical):
        return b"S", str(value.as_py()).encode()
    raise RuntimeError(f"unsupported commitment type: {arrow_type}")


def _timestamp_ns(value: str) -> int:
    parsed = datetime.fromisoformat(value)
    _require(parsed.tzinfo is not None, "timestamp is naive")
    return int(parsed.timestamp()) * 1_000_000_000 + parsed.microsecond * 1000


def _verify_cutoff_tail(parquet: pq.ParquetFile, group_index: int, cutoff_ns: int, checked_rows: int, expected_rows: int) -> None:
    visited = 0
    for batch in parquet.iter_batches(batch_size=BATCH_ROWS, row_groups=[group_index], columns=["ts"]):
        for value in batch.column(0):
            if visited >= checked_rows:
                _require(value.is_valid, "timestamp tail contains null")
                stamp = int(value.cast(pa.int64()).as_py())
                _require(stamp >= cutoff_ns, "later row returns inside registered interval")
            visited += 1
    _require(visited == expected_rows and checked_rows <= visited, "cutoff tail row count mismatch")


def _audit(relative: Path, *, source: bool) -> dict[str, object]:
    start_ns, cutoff_ns = _timestamp_ns(START), _timestamp_ns(CUTOFF)
    count, first, last, peak, groups, past_cutoff = 0, None, None, 0, 0, False
    with _fenced(relative, allocated=True, size=SOURCE_BYTES if source else None) as (handle, snapshot):
        parquet = pq.ParquetFile(handle)
        info = _schema_info(parquet)
        schema = info["_schema"]
        digest = _commitment(schema)
        for group_index in range(parquet.num_row_groups):
            metadata = parquet.metadata.row_group(group_index)
            columns = [metadata.column(index) for index in range(metadata.num_columns) if metadata.column(index).path_in_schema == "ts"]
            _require(len(columns) == 1 and columns[0].statistics is not None, "timestamp row-group statistics absent or ambiguous")
            minimum = pa.scalar(columns[0].statistics.min, type=schema.field("ts").type)
            if source and past_cutoff:
                _require(int(minimum.cast(pa.int64()).as_py()) >= cutoff_ns, "later row group returns inside registered interval")
                continue
            if source and int(minimum.cast(pa.int64()).as_py()) >= cutoff_ns:
                past_cutoff = True
                continue
            if not source:
                _require(metadata.num_rows <= ROW_GROUP_ROWS, "output row-group limit exceeded")
            groups += 1
            checked_rows = 0
            for batch in parquet.iter_batches(batch_size=BATCH_ROWS, row_groups=[group_index]):
                _require(batch.num_rows <= BATCH_ROWS, "audit batch limit exceeded")
                peak = max(peak, batch.num_rows)
                for row_index in range(batch.num_rows):
                    stamp = int(batch.column(len(schema) - 1)[row_index].cast(pa.int64()).as_py())
                    if source and past_cutoff:
                        _require(stamp >= cutoff_ns, "later row returns inside registered interval")
                        continue
                    if source and stamp >= cutoff_ns:
                        past_cutoff = True
                        continue
                    values = [batch.column(index)[row_index] for index in range(len(schema))]
                    _require(start_ns <= stamp < cutoff_ns and (last is None or stamp > last),
                             "timestamp interval/order/uniqueness violation")
                    _require(all(value.is_valid for value in values), "required field contains null")
                    o, high, low, close = (float(value.as_py()) for value in values[:4])
                    _require(all(math.isfinite(value) for value in (o, high, low, close))
                             and high >= max(o, close) and low <= min(o, close) and low <= high,
                             "OHLC geometry violation")
                    volume = float(values[4].as_py())
                    _require(math.isfinite(volume) and volume >= 0, "volume value violation")
                    instrument = int(values[6].as_py())
                    _require(-(1 << 63) <= instrument < (1 << 63), "instrument_id exceeds int64")
                    digest.update(b"R" + _u64(len(schema)))
                    for value, field in zip(values, schema):
                        tag, encoded = _scalar(value, field.type)
                        digest.update(tag + _blob(encoded))
                    first = stamp if first is None else first
                    last, count = stamp, count + 1
                checked_rows += batch.num_rows
                if source and past_cutoff:
                    break
            if source and past_cutoff:
                batch = values = value = None
                _verify_cutoff_tail(parquet, group_index, cutoff_ns, checked_rows, metadata.num_rows)
    _require(count > 0 and first == start_ns and last is not None and last < cutoff_ns, "half-open projection boundary violation")
    result = {key: value for key, value in info.items() if key != "_schema"}
    result.update({
        "_schema": info["_schema"], "stable_stat": snapshot, "rows": count,
        "first_timestamp_ns": first, "last_timestamp_ns": last, "first_timestamp": START,
        "last_timestamp": pa.scalar(last, type=pa.int64()).cast(pa.timestamp("ns", tz=TIMEZONE)).as_py().isoformat(),
        "typed_row_commitment": digest.hexdigest(), "row_groups_read": groups,
        "peak_batch_rows": peak, "maximum_batches_held": 1})
    return result


def _public(value: dict[str, object]) -> dict[str, object]: return {key: item for key, item in value.items() if key != "_schema"}


def _explicit_contract(value: dict[str, object]) -> dict[str, object]:
    expected = {"output": str(SOURCE), "output_sha256": SOURCE_SHA, "selection": SELECTION,
                "current_session_volume_used": False, "sealed_rows_written": False,
                "start": START, "end_exclusive": UPSTREAM_END}
    _require(all(value.get(key) == item for key, item in expected.items()), "explicit manifest causal/identity contract mismatch")
    return expected


def _child_contract(value: dict[str, object], output_sha: str, audit: dict[str, object]) -> dict[str, object]:
    expected = {
        "format_version": 1, "artifact": "causal_previous_session_front_pre_holdout",
        "selection": SELECTION, "current_session_volume_used": False, "source": str(SOURCE),
        "source_declared_sha256": SOURCE_SHA, "source_rehashed_during_materialization": False,
        "source_manifest": None, "source_manifest_sha256": None, "end_exclusive": CUTOFF,
        "sealed_rows_written": False, "rows": audit["rows"], "start": audit["first_timestamp"],
        "end": audit["last_timestamp"], "output": str(OUTPUT), "output_sha256": output_sha,
        "row_group_rows": ROW_GROUP_ROWS}
    _require(value == expected, "child manifest contract mismatch")
    return expected


def _projection_equality(source: dict[str, object], output: dict[str, object]) -> dict[str, bool]:
    _require(source["_schema"].equals(output["_schema"], check_metadata=True), "Arrow schema/metadata changed")
    keys = ("field_order", "arrow_schema_hex", "rows", "first_timestamp_ns", "last_timestamp_ns", "typed_row_commitment")
    equality = {key: source[key] == output[key] for key in keys}
    _require(all(equality.values()), "source/output projection differs")
    return equality


def main() -> None:
    _require(len(sys.argv) == 1, "custody controller accepts no arguments")
    _require(Path.cwd().resolve() == ROOT, "custody controller must run from frozen cwd")
    for directory in (Path("artifacts"), Path("data/processed"), ATTEMPT_ROOT):
        _directory(directory)
    for target in (CORE_RESULT, OUTPUT, OUTPUT_MANIFEST, TEMPORARY, DERIVED, SEMANTIC_ROOT):
        _absent(target)
    source_pre = _hash_file(SOURCE, allocated=True, size=SOURCE_BYTES)
    _require(source_pre[:2] == (SOURCE_SHA, SOURCE_BYTES), "source pre hash/size mismatch")
    explicit, explicit_pre_sha, explicit_pre_bytes, explicit_pre_stat = _json_file(EXPLICIT)
    _require(explicit_pre_sha == EXPLICIT_SHA, "explicit manifest pre hash mismatch")
    explicit_fields = _explicit_contract(explicit)
    materializer_pre = _hash_file(MATERIALIZER)
    _require(materializer_pre[0] == MATERIALIZER_SHA, "materializer pre hash mismatch")
    pre_schema = _schema_only()
    _progress("pre_child_schema", 30, fields=list(FIELDS))
    for target in (OUTPUT, OUTPUT_MANIFEST, TEMPORARY, DERIVED):
        _absent(target)
    command = [sys.executable, "-I", "-B", str(_path(MATERIALIZER)), "--source", str(SOURCE),
               "--output", str(OUTPUT), "--parent-sha256", SOURCE_SHA,
               "--end-exclusive", CUTOFF, "--batch-rows", str(BATCH_ROWS),
               "--row-group-rows", str(ROW_GROUP_ROWS)]
    _progress("materializer_start", 35, child_invocations=1)
    child = subprocess.run(command, cwd=ROOT, check=False)
    _require(child.returncode == 0, f"materializer exited {child.returncode}")
    _absent(DERIVED)
    _absent(TEMPORARY)
    output_fsync_stat = _fsync_leaf(OUTPUT, allocated=True)
    manifest_fsync_stat = _fsync_leaf(OUTPUT_MANIFEST)
    _fsync_dir(OUTPUT.parent)
    output_hash = _hash_file(OUTPUT, allocated=True)
    output_manifest, output_manifest_sha, output_manifest_bytes, output_manifest_stat = _json_file(OUTPUT_MANIFEST)
    materializer_post = _hash_file(MATERIALIZER)
    _require(materializer_post[:2] == materializer_pre[:2], "materializer changed")
    source_post = _hash_file(SOURCE, allocated=True, size=SOURCE_BYTES)
    _require(source_post[:2] == (SOURCE_SHA, SOURCE_BYTES), "source post hash/size mismatch")
    explicit_post = _json_file(EXPLICIT)
    _require(explicit_post[1:3] == (EXPLICIT_SHA, explicit_pre_bytes), "explicit manifest changed")
    _require(_explicit_contract(explicit_post[0]) == explicit_fields, "explicit manifest post contract mismatch")
    source_audit = _audit(SOURCE, source=True)
    _require(pre_schema["_schema"].equals(source_audit["_schema"], check_metadata=True), "pre-child/source-audit schema changed")
    _progress("source_audit", 75, rows=source_audit["rows"])
    output_audit = _audit(OUTPUT, source=False)
    equality = _projection_equality(source_audit, output_audit)
    child_manifest = _child_contract(output_manifest, output_hash[0], output_audit)
    _absent(SEMANTIC_ROOT)
    path_names = ("source", "explicit_manifest", "derived_manifest", "output", "output_manifest", "temporary", "materializer", "attempt_root", "core_result", "real_semantic_root")
    path_values = (SOURCE, EXPLICIT, DERIVED, OUTPUT, OUTPUT_MANIFEST, TEMPORARY, MATERIALIZER, ATTEMPT_ROOT, CORE_RESULT, SEMANTIC_ROOT)
    fixed = {
        "start": START, "end_exclusive": CUTOFF, "timezone": TIMEZONE,
        "ordered_fields": list(FIELDS), "batch_rows": BATCH_ROWS, "row_group_rows": ROW_GROUP_ROWS,
        "commitment_domain": DOMAIN, "sf_dataless_mask": SF_DATALESS, "stable_stat_fields": list(STABLE_FIELDS),
        "atime_policy": "excluded_because_authorized_reads_may_change_it",
        "single_writer_threat_model": "controlled_local_single_writer"}
    identities = {
        "source_registered_sha256": SOURCE_SHA, "source_pre_sha256": source_pre[0],
        "source_post_sha256": source_post[0], "source_bytes": source_post[1],
        "explicit_manifest_registered_sha256": EXPLICIT_SHA, "explicit_manifest_pre_sha256": explicit_pre_sha,
        "explicit_manifest_post_sha256": explicit_post[1], "explicit_manifest_bytes": explicit_post[2],
        "materializer_registered_sha256": MATERIALIZER_SHA, "materializer_pre_sha256": materializer_pre[0],
        "materializer_post_sha256": materializer_post[0], "output_sha256": output_hash[0], "output_bytes": output_hash[1],
        "output_manifest_sha256": output_manifest_sha, "output_manifest_bytes": output_manifest_bytes}
    stat_records = (
        ("source_hash_pre", source_pre[2]), ("source_schema_pre", pre_schema["stable_stat"]),
        ("source_hash_post", source_post[2]), ("source_audit_post", source_audit["stable_stat"]),
        ("explicit_manifest_pre", explicit_pre_stat), ("explicit_manifest_post", explicit_post[3]),
        ("materializer_pre", materializer_pre[2]), ("materializer_post", materializer_post[2]),
        ("output_fsync", output_fsync_stat), ("output_hash", output_hash[2]),
        ("output_audit", output_audit["stable_stat"]), ("output_manifest_fsync", manifest_fsync_stat),
        ("output_manifest_hash", output_manifest_stat))
    result = {
        "format_version": 1, "status": "complete_exact_projection",
        "classification": "CUSTODY_CORE_EXACT_PROJECTION_VERIFIED", "experiment_id": EXPERIMENT, "product_version": "3.0.2",
        "preregistration_sha256": PREREG, "gate": "gate2_physical_custody",
        "attempt_id": "physical_001",
        "paths": {key: str(value) for key, value in zip(path_names, path_values)},
        "fixed_contract": fixed, "identities": identities,
        "pre_child": {
            "schema_validation_completed_before_child": True, "source_schema": _public(pre_schema),
            "explicit_manifest_validated_fields": explicit_fields, "derived_manifest_absent": True,
            "outputs_absent": True},
        "child": {"invocations": 1, "returncode": child.returncode, "argv": command,
                  "manifest": child_manifest},
        "audits": {"source": _public(source_audit), "output": _public(output_audit), "equality": equality},
        "stable_stats": dict(stat_records),
        "durability": {
            "output_file_fsynced": True, "output_manifest_file_fsynced": True,
            "output_parent_directory_fsynced": True,
            "core_result_policy": "O_EXCL_file_fsync_direct_directory_fsync"},
        "resources": {
            "source_peak_batch_rows": source_audit["peak_batch_rows"],
            "output_peak_batch_rows": output_audit["peak_batch_rows"],
            "maximum_batches_held": 1, "full_frame_retained": False},
        "authority_denials": DENIALS, "mbo_accesses": 0,
        "current_session_volume_uses": 0, "action_or_future_fields": 0}
    _exclusive_result(result)


if __name__ == "__main__":
    _require(sys.flags.isolated == 1 and sys.dont_write_bytecode, "custody controller requires -I -B")
    main()
