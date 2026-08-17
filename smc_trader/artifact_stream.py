"""Atomic, hash-bound Parquet shard streams used by long calibration jobs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def atomic_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def _arrow_schema(field_types: Mapping[str, str]):
    import pyarrow as pa

    scalar_types = {
        "large_string": pa.large_string(),
        "float64": pa.float64(),
        "int64": pa.int64(),
        "bool": pa.bool_(),
        "timestamp_ny": pa.timestamp("ns", tz="America/New_York"),
        "timestamp_utc": pa.timestamp("ns", tz="UTC"),
    }
    unknown = sorted(set(field_types.values()) - set(scalar_types))
    if unknown:
        raise ValueError(f"unsupported registered Parquet field types: {unknown}")
    return pa.schema(
        [pa.field(name, scalar_types[type_name]) for name, type_name in field_types.items()]
    )


def _schema_fingerprint(field_types: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(field_types),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def atomic_parquet(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    field_types: Mapping[str, str] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if field_types is None:
        frame.to_parquet(temporary, index=False)
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq

        expected_columns = tuple(field_types)
        actual_columns = tuple(str(column) for column in frame.columns)
        missing = sorted(set(expected_columns) - set(actual_columns))
        extra = sorted(set(actual_columns) - set(expected_columns))
        if missing or extra:
            raise ValueError(
                "Parquet shard columns differ from the registered schema: "
                f"missing={missing}, extra={extra}"
            )
        ordered = frame.loc[:, list(expected_columns)]
        table = pa.Table.from_pandas(
            ordered,
            schema=_arrow_schema(field_types),
            preserve_index=False,
            safe=True,
        )
        pq.write_table(table, temporary)
    os.replace(temporary, destination)


def new_stream_state(
    field_types: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    state = {
        "rows": 0,
        "next_shard_index": 0,
        "committed_shards": [],
    }
    if field_types is not None:
        state["schema_fingerprint"] = _schema_fingerprint(field_types)
        state["field_types"] = dict(field_types)
    return state


def write_stream_shard(
    destination: str | Path,
    stream_name: str,
    buffer: list[dict[str, Any]],
    stream_state: dict[str, Any],
    *,
    key_column: str,
    field_types: Mapping[str, str] | None = None,
) -> None:
    if not buffer:
        return
    if "/" in stream_name or stream_name in {"", ".", ".."}:
        raise ValueError("invalid shard stream name")
    index = int(stream_state["next_shard_index"])
    relative = Path(stream_name) / f"part-{index:05d}.parquet"
    path = Path(destination) / relative
    frame = pd.DataFrame(buffer)
    if key_column not in frame:
        raise ValueError(
            f"{stream_name} shard omits registered key column {key_column}"
        )
    if field_types is not None:
        fingerprint = _schema_fingerprint(field_types)
        registered = stream_state.get("schema_fingerprint")
        if registered is not None and registered != fingerprint:
            raise ValueError("registered shard stream schema changed")
        stream_state["schema_fingerprint"] = fingerprint
        stream_state["field_types"] = dict(field_types)
    atomic_parquet(frame, path, field_types=field_types)
    stream_state["committed_shards"].append(
        {
            "index": index,
            "path": str(relative),
            "rows": int(len(frame)),
            "first_key": str(frame[key_column].iloc[0]),
            "last_key": str(frame[key_column].iloc[-1]),
            "sha256": sha256_file(path),
        }
    )
    stream_state["rows"] = int(stream_state["rows"]) + int(len(frame))
    stream_state["next_shard_index"] = index + 1
    buffer.clear()


def write_stream_shards_bounded(
    destination: str | Path,
    stream_name: str,
    buffer: list[dict[str, Any]],
    stream_state: dict[str, Any],
    *,
    key_column: str,
    maximum_rows: int,
    field_types: Mapping[str, str] | None = None,
) -> None:
    if maximum_rows < 1:
        raise ValueError("maximum shard rows must be positive")
    while buffer:
        chunk = buffer[:maximum_rows]
        del buffer[:maximum_rows]
        write_stream_shard(
            destination,
            stream_name,
            chunk,
            stream_state,
            key_column=key_column,
            field_types=field_types,
        )


def verify_stream_shards(
    destination: str | Path,
    stream_state: Mapping[str, Any],
) -> int:
    import pyarrow.parquet as pq

    registered_types = stream_state.get("field_types")
    expected_schema = (
        None if registered_types is None else _arrow_schema(registered_types)
    )
    if registered_types is not None:
        expected_fingerprint = _schema_fingerprint(registered_types)
        if stream_state.get("schema_fingerprint") != expected_fingerprint:
            raise ValueError("registered shard schema fingerprint is invalid")
    expected_index = 0
    total = 0
    for shard in stream_state.get("committed_shards", ()):
        if int(shard["index"]) != expected_index:
            raise ValueError("committed shard indices are not contiguous")
        path = Path(destination) / str(shard["path"])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"committed shard is missing: {path}")
        if sha256_file(path) != str(shard["sha256"]):
            raise ValueError(f"committed shard hash is invalid: {path}")
        rows = int(pq.ParquetFile(path).metadata.num_rows)
        if rows != int(shard["rows"]):
            raise ValueError(f"committed shard row count is invalid: {path}")
        if expected_schema is not None:
            actual_schema = pq.read_schema(path).remove_metadata()
            # JSON object keys are canonicalized when the manifest is
            # written, while Parquet preserves the recorder's column order.
            # Stream identity is the exact field-name/type set; ordering is
            # enforced by the writer and is not a semantic schema mismatch.
            expected_fields = {
                field.name: (field.type, field.nullable)
                for field in expected_schema
            }
            actual_fields = {
                field.name: (field.type, field.nullable)
                for field in actual_schema
            }
            if actual_fields != expected_fields:
                raise ValueError(f"committed shard schema is invalid: {path}")
        total += rows
        expected_index += 1
    if total != int(stream_state.get("rows", -1)):
        raise ValueError("committed shard rows are not conserved")
    if expected_index != int(stream_state.get("next_shard_index", -1)):
        raise ValueError("next shard index is not conserved")
    return total


def write_stream_manifest(
    destination: str | Path,
    stream_name: str,
    stream_state: Mapping[str, Any],
    *,
    artifact: str,
    bindings: Mapping[str, Any],
) -> Path:
    verify_stream_shards(destination, stream_state)
    manifest = {
        "format_version": 1,
        "artifact": artifact,
        "status": "complete",
        "stream": stream_name,
        "rows": int(stream_state["rows"]),
        "shards": list(stream_state["committed_shards"]),
        "bindings": dict(bindings),
    }
    if stream_state.get("schema_fingerprint") is not None:
        manifest["schema_fingerprint"] = stream_state["schema_fingerprint"]
        manifest["field_types"] = dict(stream_state["field_types"])
    path = Path(destination) / f"{stream_name}.manifest.json"
    atomic_bytes(path, canonical_json(manifest))
    return path


__all__ = [
    "atomic_bytes",
    "atomic_parquet",
    "canonical_json",
    "new_stream_state",
    "sha256_file",
    "verify_stream_shards",
    "write_stream_manifest",
    "write_stream_shard",
    "write_stream_shards_bounded",
]
