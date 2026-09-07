#!/usr/bin/env python3
"""Materialize a bounded causal OHLCV source without loading sealed rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument(
        "--end-exclusive",
        default="2026-04-01T00:00:00-04:00",
    )
    parser.add_argument("--batch-rows", type=int, default=200_000)
    parser.add_argument("--row-group-rows", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    destination = Path(args.output)
    manifest_destination = destination.with_suffix(
        destination.suffix + ".manifest.json"
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() or manifest_destination.exists() or temporary.exists():
        raise FileExistsError("refusing to overwrite bounded OHLCV artifacts")
    if len(str(args.parent_sha256)) != 64:
        raise ValueError("parent SHA-256 identity is invalid")
    end = pd.Timestamp(args.end_exclusive)
    if end.tzinfo is None:
        raise ValueError("pre-holdout boundary must be timezone aware")
    batch_rows = int(args.batch_rows)
    row_group_rows = int(args.row_group_rows)
    if batch_rows <= 0 or row_group_rows <= 0:
        raise ValueError("parquet batch and row-group sizes must be positive")

    parent_manifest = source.with_suffix(".manifest.json")
    parent_manifest_hash = (
        _sha256(parent_manifest)
        if parent_manifest.is_file()
        else None
    )
    parquet = pq.ParquetFile(source)
    schema = parquet.schema_arrow
    if "ts" not in schema.names:
        raise ValueError("causal OHLCV parquet has no ts field")
    timestamp_type = schema.field("ts").type
    cutoff = pa.scalar(end.to_pydatetime(), type=timestamp_type)
    written_rows = 0
    first_timestamp = None
    last_timestamp = None
    stop = False
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(
        temporary,
        schema,
        compression="zstd",
    )
    try:
        for row_group_index in range(parquet.num_row_groups):
            metadata = parquet.metadata.row_group(row_group_index)
            timestamp_column = next(
                metadata.column(index)
                for index in range(metadata.num_columns)
                if metadata.column(index).path_in_schema == "ts"
            )
            statistics = timestamp_column.statistics
            if statistics is None:
                raise ValueError("OHLCV timestamp row group lacks statistics")
            if pd.Timestamp(statistics.min) >= end:
                break
            for batch in parquet.iter_batches(
                batch_size=batch_rows,
                row_groups=[row_group_index],
            ):
                timestamps = batch.column(
                    batch.schema.get_field_index("ts")
                )
                keep = pc.less(timestamps, cutoff)
                kept = batch.filter(keep)
                if kept.num_rows:
                    table = pa.Table.from_batches([kept], schema=schema)
                    writer.write_table(
                        table,
                        row_group_size=row_group_rows,
                    )
                    batch_first = pd.Timestamp(
                        kept.column(
                            kept.schema.get_field_index("ts")
                        )[0].as_py()
                    )
                    batch_last = pd.Timestamp(
                        kept.column(
                            kept.schema.get_field_index("ts")
                        )[-1].as_py()
                    )
                    if first_timestamp is None:
                        first_timestamp = batch_first
                    if (
                        last_timestamp is not None
                        and batch_first <= last_timestamp
                    ):
                        raise ValueError(
                            "causal OHLCV source is not strictly chronological"
                        )
                    last_timestamp = batch_last
                    written_rows += kept.num_rows
                if kept.num_rows != batch.num_rows:
                    stop = True
                    break
            if stop:
                break
    except Exception:
        writer.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        writer.close()

    if (
        written_rows <= 0
        or first_timestamp is None
        or last_timestamp is None
        or last_timestamp >= end
    ):
        if temporary.exists():
            temporary.unlink()
        raise ValueError("bounded OHLCV materialization failed its time boundary")
    os.replace(temporary, destination)
    output_hash = _sha256(destination)
    manifest = {
        "format_version": 1,
        "artifact": "causal_previous_session_front_pre_holdout",
        "selection": (
            "highest total volume from the strictly prior completed "
            "Globex session"
        ),
        "current_session_volume_used": False,
        "source": str(source),
        "source_declared_sha256": str(args.parent_sha256),
        "source_rehashed_during_materialization": False,
        "source_manifest": (
            str(parent_manifest) if parent_manifest.is_file() else None
        ),
        "source_manifest_sha256": parent_manifest_hash,
        "end_exclusive": end.isoformat(),
        "sealed_rows_written": False,
        "rows": int(written_rows),
        "start": first_timestamp.isoformat(),
        "end": last_timestamp.isoformat(),
        "output": str(destination),
        "output_sha256": output_hash,
        "row_group_rows": row_group_rows,
    }
    manifest_destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": written_rows,
                "start": first_timestamp.isoformat(),
                "end": last_timestamp.isoformat(),
                "sha256": output_hash,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
