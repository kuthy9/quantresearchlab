#!/usr/bin/env python3
"""Materialize the preregistered Phase 6 one-week MBO minute facts."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_mbo_execution import (  # noqa: E402
    _partition_paths,
    _verify_partition_manifest,
)
from smc_trader.artifact_stream import (  # noqa: E402
    atomic_bytes,
    atomic_parquet,
    canonical_json,
    sha256_file,
)
from smc_trader.io import iter_completed_bars, load_ohlcv  # noqa: E402
from smc_trader.mbo import F_LAST, MBOReplayError  # noqa: E402
from smc_trader.mbo_mechanism import (  # noqa: E402
    EXCLUDED_FLOW_FLAGS,
    FLOW_COMPONENT_COLUMNS,
    MBO_MECHANISM_COLUMNS,
    MBO_MECHANISM_PROTOCOL,
    MBO_MECHANISM_PROTOCOL_SHA256,
    MBO_MECHANISM_SCHEMA_VERSION,
    build_minute_mechanism_frame,
    derive_flow_features,
    flow_updates,
    validate_mbo_mechanism_frame,
)
from smc_trader.validation import load_validation_protocol  # noqa: E402


DEFAULT_START = "2024-06-02T22:00:00Z"
DEFAULT_END = "2024-06-07T21:01:00Z"
DEFAULT_SYMBOL = "NQM4"
DEFAULT_INSTRUMENT_ID = 13743
DEFAULT_EXPECTED_MINUTES = 6900
FLOW_SCAN_COLUMNS = (
    "ts_recv",
    "publisher_id",
    "instrument_id",
    "action",
    "side",
    "size",
    "flags",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbo-root", default="data/raw/nq_mbo/legacy_parquet")
    parser.add_argument(
        "--ohlcv-source",
        default="data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet",
    )
    parser.add_argument(
        "--bbo-source",
        default="data/processed/mbo_execution_dev_202406_v2_3_clockfix.parquet",
    )
    parser.add_argument("--validation-protocol", default="configs/data_splits.json")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--instrument-id", type=int, default=DEFAULT_INSTRUMENT_ID)
    parser.add_argument(
        "--expected-minutes", type=int, default=DEFAULT_EXPECTED_MINUTES
    )
    parser.add_argument("--tick-size", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _aware_utc(value: Any, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone aware")
    return timestamp.tz_convert("UTC")


def _normalize_code(values: pd.Series, *, kind: str) -> pd.Series:
    normalized = values.astype(str).str.strip().str.upper().str.rsplit(".").str[-1]
    mapping = (
        {
            "ADD": "A",
            "CANCEL": "C",
            "MODIFY": "M",
            "CLEAR": "R",
            "CLEARBOOK": "R",
            "CLEAR_BOOK": "R",
            "TRADE": "T",
            "FILL": "F",
            "NONE": "N",
        }
        if kind == "action"
        else {"ASK": "A", "BID": "B", "NONE": "N"}
    )
    return normalized.replace(mapping)


def _aggregate_closed_packets(
    closed: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if closed.empty:
        return pd.DataFrame(), {
            "selected_rows": 0,
            "selected_packets": 0,
            "in_window_rows": 0,
            "eligible_in_window_rows": 0,
            "excluded_in_window_rows": 0,
            "ignored_none_in_window_rows": 0,
            "eligible_t_rows": 0,
            "eligible_f_rows": 0,
            "excluded_t_rows": 0,
            "excluded_f_rows": 0,
            "flag_value_counts": {},
            "eligible_flow_flag_histogram_in_window": {},
            "excluded_flow_flag_histogram_in_window": {},
            "publisher_ids": [],
        }
    work = closed.copy()
    work["action"] = _normalize_code(work["action"], kind="action")
    work["side"] = _normalize_code(work["side"], kind="side")
    unknown_actions = sorted(set(work["action"]) - {"A", "C", "M", "R", "T", "F", "N"})
    unknown_sides = sorted(set(work["side"]) - {"A", "B", "N"})
    if unknown_actions or unknown_sides:
        raise MBOReplayError(
            f"raw MBO contains unknown action/side: {unknown_actions}/{unknown_sides}"
        )
    work["ts_recv"] = pd.to_datetime(work["ts_recv"], errors="coerce", utc=True)
    if work["ts_recv"].isna().any():
        raise MBOReplayError("raw MBO contains an invalid ts_recv")
    work["flags"] = pd.to_numeric(work["flags"], errors="raise").astype("int64")
    work["size"] = pd.to_numeric(work["size"], errors="coerce")
    if work["size"].isna().any() or (work["size"] < 0).any():
        raise MBOReplayError("raw MBO contains invalid size")

    in_window = work["ts_recv"].ge(start) & work["ts_recv"].lt(end)
    packet_invalid_row = work["flags"].map(
        lambda value: bool(int(value) & EXCLUDED_FLOW_FLAGS)
    ) | work["action"].eq("R")
    invalid_packet_ids = set(work.loc[packet_invalid_row, "_packet_group"])
    packet_eligible = ~work["_packet_group"].isin(invalid_packet_ids)
    eligible = work.loc[in_window & packet_eligible & work["action"].isin({"A", "C", "M", "T", "F"})].copy()
    if eligible.empty:
        totals = pd.DataFrame()
    else:
        # Exact-boundary receive timestamps are already knowable at that
        # decision clock, matching the causal BBO replay's <= clock rule.
        eligible["decision_time"] = eligible["ts_recv"].dt.ceil("min")
        totals = (
            eligible.groupby(
                ["decision_time", "action", "side"],
                as_index=False,
                sort=True,
                observed=True,
            )
            .agg(volume=("size", "sum"), record_count=("size", "size"))
        )
    qc = {
        "selected_rows": int(len(work)),
        "selected_packets": int(work["_packet_group"].nunique()),
        "in_window_rows": int(in_window.sum()),
        "eligible_in_window_rows": int(
            (in_window & packet_eligible & work["action"].isin({"A", "C", "M", "T", "F"})).sum()
        ),
        "excluded_in_window_rows": int((in_window & ~packet_eligible).sum()),
        "ignored_none_in_window_rows": int(
            (in_window & packet_eligible & work["action"].eq("N")).sum()
        ),
        "eligible_t_rows": int(
            (in_window & packet_eligible & work["action"].eq("T")).sum()
        ),
        "eligible_f_rows": int(
            (in_window & packet_eligible & work["action"].eq("F")).sum()
        ),
        "excluded_t_rows": int(
            (in_window & ~packet_eligible & work["action"].eq("T")).sum()
        ),
        "excluded_f_rows": int(
            (in_window & ~packet_eligible & work["action"].eq("F")).sum()
        ),
        "flag_value_counts": {
            str(int(flag)): int(count)
            for flag, count in work.loc[in_window, "flags"].value_counts().items()
        },
        "eligible_flow_flag_histogram_in_window": {
            str(int(flag)): int(count)
            for flag, count in work.loc[
                in_window
                & packet_eligible
                & work["action"].isin({"A", "C", "M", "T", "F"}),
                "flags",
            ]
            .value_counts()
            .items()
        },
        "excluded_flow_flag_histogram_in_window": {
            str(int(flag)): int(count)
            for flag, count in work.loc[
                in_window
                & ~packet_eligible
                & work["action"].isin({"A", "C", "M", "T", "F"}),
                "flags",
            ]
            .value_counts()
            .items()
        },
        "publisher_ids": sorted(
            int(value) for value in work.loc[in_window, "publisher_id"].unique()
        ),
    }
    return totals, qc


def _aggregate_parquet_partition(payload: Sequence[Any]) -> tuple[pd.DataFrame, dict[str, Any], str]:
    path_value, start_value, end_value, instrument_id, batch_size = payload
    path = Path(path_value)
    start = _aware_utc(start_value, name="start")
    end = _aware_utc(end_value, name="end")
    import pyarrow as pa
    import pyarrow.dataset as ds

    dataset = ds.dataset(path, format="parquet")
    missing = sorted(set(FLOW_SCAN_COLUMNS) - set(dataset.schema.names))
    if missing:
        raise MBOReplayError(f"{path} is missing MBO flow fields: {missing}")
    scanner = dataset.scanner(
        columns=list(FLOW_SCAN_COLUMNS),
        batch_size=int(batch_size),
        use_threads=False,
    )
    pending = pd.DataFrame()
    packet_offset = 0
    packet_open = False
    totals: list[pd.DataFrame] = []
    qc = {
        "raw_rows": 0,
        "raw_vendor_packets": 0,
        "selected_rows": 0,
        "selected_packets": 0,
        "in_window_rows": 0,
        "eligible_in_window_rows": 0,
        "excluded_in_window_rows": 0,
        "ignored_none_in_window_rows": 0,
        "eligible_t_rows": 0,
        "eligible_f_rows": 0,
        "excluded_t_rows": 0,
        "excluded_f_rows": 0,
        "flag_value_counts": {},
        "eligible_flow_flag_histogram_in_window": {},
        "excluded_flow_flag_histogram_in_window": {},
        "publisher_ids": set(),
    }
    last_selected_recv: pd.Timestamp | None = None
    for batch in scanner.to_batches():
        qc["raw_rows"] += int(batch.num_rows)
        ids = np.asarray(
            batch.column(batch.schema.get_field_index("instrument_id")).to_numpy(
                zero_copy_only=False
            )
        )
        flags = np.asarray(
            batch.column(batch.schema.get_field_index("flags")).to_numpy(
                zero_copy_only=False
            ),
            dtype=np.int64,
        )
        last = (flags & F_LAST) != 0
        group_ids = packet_offset + np.cumsum(last, dtype=np.int64) - last
        selected_mask = ids == int(instrument_id)
        if bool(selected_mask.any()):
            selected_batch = batch.filter(pa.array(selected_mask))
            selected_frame = selected_batch.to_pandas()
            selected_frame["_packet_group"] = group_ids[selected_mask]
            pending = (
                selected_frame
                if pending.empty
                else pd.concat((pending, selected_frame), ignore_index=True)
            )
        completed = int(last.sum())
        packet_offset += completed
        qc["raw_vendor_packets"] += completed
        packet_open = not bool(last[-1]) if len(last) else packet_open
        if pending.empty:
            continue
        closed_mask = pending["_packet_group"].lt(packet_offset)
        if not bool(closed_mask.any()):
            continue
        closed = pending.loc[closed_mask].copy()
        pending = pending.loc[~closed_mask].copy()
        selected_recv = pd.to_datetime(closed["ts_recv"], errors="coerce", utc=True)
        if selected_recv.isna().any() or not selected_recv.is_monotonic_increasing:
            raise MBOReplayError(f"{path} selected ts_recv order regressed")
        if last_selected_recv is not None and selected_recv.iloc[0] < last_selected_recv:
            raise MBOReplayError(f"{path} selected ts_recv order regressed across batches")
        last_selected_recv = selected_recv.iloc[-1]
        local_totals, local_qc = _aggregate_closed_packets(
            closed,
            start=start,
            end=end,
        )
        if not local_totals.empty:
            totals.append(local_totals)
        for key in (
            "selected_rows",
            "selected_packets",
            "in_window_rows",
            "eligible_in_window_rows",
            "excluded_in_window_rows",
            "ignored_none_in_window_rows",
            "eligible_t_rows",
            "eligible_f_rows",
            "excluded_t_rows",
            "excluded_f_rows",
        ):
            qc[key] += int(local_qc[key])
        for histogram_key in (
            "flag_value_counts",
            "eligible_flow_flag_histogram_in_window",
            "excluded_flow_flag_histogram_in_window",
        ):
            for flag, count in local_qc[histogram_key].items():
                qc[histogram_key][flag] = (
                    int(qc[histogram_key].get(flag, 0)) + int(count)
                )
        qc["publisher_ids"].update(local_qc["publisher_ids"])
    if packet_open or not pending.empty:
        raise MBOReplayError(f"{path} ends inside an incomplete vendor packet")
    if totals:
        result = (
            pd.concat(totals, ignore_index=True)
            .groupby(["decision_time", "action", "side"], as_index=False, sort=True)
            .agg(volume=("volume", "sum"), record_count=("record_count", "sum"))
        )
    else:
        result = pd.DataFrame(
            columns=["decision_time", "action", "side", "volume", "record_count"]
        )
    qc["publisher_ids"] = sorted(qc["publisher_ids"])
    return result, qc, str(path)


def _combine_partition_results(
    results: Sequence[tuple[pd.DataFrame, Mapping[str, Any], str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    nonempty = [frame for frame, _, _ in results if not frame.empty]
    if nonempty:
        totals = (
            pd.concat(nonempty, ignore_index=True)
            .groupby(["decision_time", "action", "side"], as_index=False, sort=True)
            .agg(volume=("volume", "sum"), record_count=("record_count", "sum"))
        )
    else:
        totals = pd.DataFrame(
            columns=["decision_time", "action", "side", "volume", "record_count"]
        )
    qc: dict[str, Any] = {
        "raw_rows": 0,
        "raw_vendor_packets": 0,
        "selected_rows": 0,
        "selected_packets": 0,
        "in_window_rows": 0,
        "eligible_in_window_rows": 0,
        "excluded_in_window_rows": 0,
        "ignored_none_in_window_rows": 0,
        "eligible_t_rows": 0,
        "eligible_f_rows": 0,
        "excluded_t_rows": 0,
        "excluded_f_rows": 0,
        "flag_value_counts": {},
        "eligible_flow_flag_histogram_in_window": {},
        "excluded_flow_flag_histogram_in_window": {},
        "publisher_ids": set(),
    }
    histogram_keys = {
        "flag_value_counts",
        "eligible_flow_flag_histogram_in_window",
        "excluded_flow_flag_histogram_in_window",
    }
    for _, local, _ in results:
        for key in qc:
            if key == "publisher_ids":
                qc[key].update(local[key])
            elif key in histogram_keys:
                for flag, count in local[key].items():
                    qc[key][flag] = int(qc[key].get(flag, 0)) + int(count)
            else:
                qc[key] += int(local[key])
    qc["publisher_ids"] = sorted(qc["publisher_ids"])
    return totals, qc


def _action_totals_to_flow(
    totals: pd.DataFrame,
    *,
    decision_times: Sequence[Any],
) -> pd.DataFrame:
    clocks = pd.DatetimeIndex(pd.to_datetime(list(decision_times), utc=True))
    values: dict[pd.Timestamp, dict[str, float]] = {
        clock: {column: 0.0 for column in FLOW_COMPONENT_COLUMNS}
        for clock in clocks
    }
    for row in totals.itertuples(index=False):
        clock = pd.Timestamp(row.decision_time).tz_convert("UTC")
        if clock not in values:
            continue
        updates = flow_updates(row.action, row.side, float(row.volume))
        for column, value in updates.items():
            values[clock][column] += (
                float(row.record_count) if column.endswith("_count") else float(value)
            )
    return derive_flow_features(
        pd.DataFrame(
            {"decision_time": clock, **values[clock]} for clock in clocks
        )
    )


def _registered_clocks(
    source: str | Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    symbol: str,
    instrument_id: int,
    expected_minutes: int,
) -> tuple[pd.DataFrame, Any]:
    loaded = load_ohlcv(source, start=start, end=end)
    if not loaded.contract_selection_causal:
        raise RuntimeError("Phase 6 requires the causal previous-session contract")
    rows = [
        {
            "decision_time": bar.end.tz_convert("UTC"),
            "symbol": bar.symbol,
            "instrument_id": bar.instrument_id,
        }
        for bar in iter_completed_bars(loaded.frame)
        if start <= bar.end.tz_convert("UTC") < end
    ]
    clocks = pd.DataFrame(rows)
    if (
        len(clocks) != int(expected_minutes)
        or clocks["decision_time"].duplicated().any()
        or set(clocks["symbol"]) != {str(symbol)}
        or set(clocks["instrument_id"]) != {int(instrument_id)}
    ):
        raise RuntimeError(
            "causal OHLCV clocks do not match the registered Phase 6 week/contract"
        )
    return clocks, loaded


def _registered_bbo(
    source: str | Path,
    *,
    clocks: pd.DataFrame,
    registered_artifact: Any,
) -> tuple[pd.DataFrame, Mapping[str, Any], Path]:
    path = Path(source)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if sha256_file(path) != registered_artifact.sha256:
        raise RuntimeError("BBO artifact differs from its validation-protocol binding")
    if sha256_file(manifest_path) != registered_artifact.manifest_sha256:
        raise RuntimeError("BBO manifest differs from its validation-protocol binding")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output_sha256") != sha256_file(path):
        raise RuntimeError("BBO manifest does not bind the BBO artifact")
    bbo = pd.read_parquet(path)
    bbo["decision_time"] = pd.to_datetime(
        bbo["decision_time"], errors="coerce", utc=True
    )
    bbo = bbo.loc[bbo["decision_time"].isin(set(clocks["decision_time"]))].copy()
    if len(bbo) != len(clocks):
        raise RuntimeError("BBO artifact does not cover every registered Phase 6 clock")
    return bbo, manifest, manifest_path


def _find_registered_bbo(protocol: Any, source: str | Path) -> Any:
    target = Path(source).resolve()
    matches = [
        artifact
        for artifact in protocol.mbo_identity.development_execution_artifacts
        if Path(artifact.path).resolve() == target
    ]
    if len(matches) != 1:
        raise RuntimeError("BBO source is not uniquely registered in data_splits.json")
    return matches[0]


def _lineage_binding(role: str, path: str | Path, sha256: str | None = None) -> dict[str, str]:
    source = Path(path)
    return {
        "role": role,
        "path": str(source),
        "sha256": sha256 if sha256 is not None else sha256_file(source),
    }


def main() -> None:
    args = parse_args()
    start = _aware_utc(args.start, name="start")
    end = _aware_utc(args.end, name="end")
    if end <= start:
        raise ValueError("Phase 6 window must be positive")
    if not 1 <= int(args.workers) <= 4:
        raise ValueError("--workers must be between 1 and 4")
    if int(args.batch_size) < 1 or int(args.expected_minutes) < 1:
        raise ValueError("batch size and expected minutes must be positive")
    if float(args.tick_size) != float(MBO_MECHANISM_PROTOCOL["tick_size"]):
        raise ValueError("--tick-size differs from the frozen Phase 6 protocol")
    destination = Path(args.output)
    manifest_destination = destination.with_suffix(
        destination.suffix + ".manifest.json"
    )
    if destination.exists() or manifest_destination.exists():
        raise FileExistsError("refusing to overwrite Phase 6 artifact or manifest")

    protocol_path = Path(args.validation_protocol)
    protocol = load_validation_protocol(protocol_path)
    window = protocol.classify_mbo(start, end)
    if window.role == "sealed_holdout":
        raise RuntimeError("Phase 6 development materialization cannot reveal holdout MBO")
    if sha256_file(protocol_path) != protocol.fingerprint:
        raise RuntimeError("validation protocol changed during Phase 6 materialization")
    ohlcv_path = Path(args.ohlcv_source)
    ohlcv_hash = sha256_file(ohlcv_path)
    if (
        ohlcv_hash != protocol.causal_source.sha256
        or ohlcv_path.resolve() != Path(protocol.causal_source.path).resolve()
    ):
        raise RuntimeError("OHLCV source differs from the registered causal source")
    clocks, loaded = _registered_clocks(
        ohlcv_path,
        start=start,
        end=end,
        symbol=args.symbol,
        instrument_id=args.instrument_id,
        expected_minutes=args.expected_minutes,
    )

    registered_bbo = _find_registered_bbo(protocol, args.bbo_source)
    bbo, bbo_manifest, bbo_manifest_path = _registered_bbo(
        args.bbo_source,
        clocks=clocks,
        registered_artifact=registered_bbo,
    )
    if (
        bbo_manifest.get("mbo_partition_manifest_sha256")
        != protocol.mbo_identity.development_partition_manifest_sha256
    ):
        raise RuntimeError("BBO lineage refers to a different MBO partition manifest")
    raw_root = Path(args.mbo_root)
    if raw_root.resolve() != Path(protocol.mbo_identity.development_root).resolve():
        raise RuntimeError("MBO root differs from the registered development root")
    paths = _partition_paths(raw_root, start=start, end=end)
    if len(paths) != 6:
        raise RuntimeError("registered Phase 6 week must resolve to exactly six UTC partitions")
    partition_manifest_hash, verified_partition_hashes = _verify_partition_manifest(
        raw_root, paths
    )
    if (
        partition_manifest_hash
        != protocol.mbo_identity.development_partition_manifest_sha256
        or verified_partition_hashes != len(paths)
    ):
        raise RuntimeError("MBO partition identity differs from data_splits.json")

    tasks = [
        (str(path), start.isoformat(), end.isoformat(), int(args.instrument_id), int(args.batch_size))
        for path in paths
    ]
    if int(args.workers) == 1:
        results = []
        for task in tasks:
            result = _aggregate_parquet_partition(task)
            results.append(result)
            print(json.dumps({"mbo_flow_partition_completed": result[2]}), flush=True)
    else:
        results = []
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = [executor.submit(_aggregate_parquet_partition, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps({"mbo_flow_partition_completed": result[2]}), flush=True)
    totals, flow_qc = _combine_partition_results(results)
    flow = _action_totals_to_flow(
        totals,
        decision_times=clocks["decision_time"],
    )
    output = build_minute_mechanism_frame(
        clocks,
        flow,
        bbo,
        tick_size=float(args.tick_size),
    )
    output = validate_mbo_mechanism_frame(
        output,
        expected_start=start,
        expected_end=end,
        expected_symbol=args.symbol,
        expected_instrument_id=args.instrument_id,
        expected_rows=args.expected_minutes,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(output, destination)
    partition_manifest = json.loads(
        (raw_root / "manifest.json").read_text(encoding="utf-8")
    )
    registered_partition_hashes = {
        str(item["path"]): str(item["sha256"])
        for item in partition_manifest["files"]
    }
    raw_partitions = []
    for path in paths:
        relative = str(path.relative_to(raw_root))
        raw_partitions.append(
            {
                "path": str(path),
                "relative_path": relative,
                "sha256": registered_partition_hashes[relative],
                "verified_during_materialization": True,
            }
        )
    ohlcv_manifest_path = Path(protocol.causal_source.manifest_path or "")
    module_path = ROOT / "smc_trader/mbo_mechanism.py"
    script_path = Path(__file__).resolve()
    direct_dependencies = (
        ROOT / "scripts/materialize_mbo_execution.py",
        ROOT / "smc_trader/mbo.py",
        ROOT / "smc_trader/artifact_stream.py",
        ROOT / "smc_trader/io.py",
        ROOT / "smc_trader/validation.py",
    )
    lineage = [
        _lineage_binding("validation_protocol", protocol_path, protocol.fingerprint),
        _lineage_binding("ohlcv_source", ohlcv_path, ohlcv_hash),
        _lineage_binding(
            "ohlcv_manifest",
            ohlcv_manifest_path,
            protocol.causal_source.manifest_sha256,
        ),
        _lineage_binding(
            "mbo_partition_manifest",
            raw_root / "manifest.json",
            partition_manifest_hash,
        ),
        _lineage_binding("bbo_artifact", args.bbo_source, registered_bbo.sha256),
        _lineage_binding(
            "bbo_manifest", bbo_manifest_path, registered_bbo.manifest_sha256
        ),
        _lineage_binding("feature_module", module_path),
        _lineage_binding("materializer", script_path),
        *(
            _lineage_binding(f"direct_dependency:{path.relative_to(ROOT)}", path)
            for path in direct_dependencies
        ),
    ]
    manifest = {
        "format_version": 1,
        "artifact_kind": "mbo_minute_mechanism_features",
        "feature_schema_version": MBO_MECHANISM_SCHEMA_VERSION,
        "feature_protocol": dict(MBO_MECHANISM_PROTOCOL),
        "feature_protocol_sha256": MBO_MECHANISM_PROTOCOL_SHA256,
        "schema_columns": list(MBO_MECHANISM_COLUMNS),
        "schema_columns_sha256": hashlib.sha256(
            json.dumps(list(MBO_MECHANISM_COLUMNS), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "validation_schema_version": protocol.schema_version,
        "validation_protocol_hash": protocol.fingerprint,
        "validation_window_role": window.role,
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "symbol": str(args.symbol),
        "instrument_id": int(args.instrument_id),
        "tick_size": float(args.tick_size),
        "availability_clock": "ts_recv",
        "event_time_materialization": (
            "ts_event remains in the hash-bound raw partitions but is not copied "
            "into this minute artifact; availability and binning use ts_recv"
        ),
        "contract_selection_causal": bool(loaded.contract_selection_causal),
        "raw_partitions": raw_partitions,
        "verified_partition_hashes": int(verified_partition_hashes),
        "flow_qc": flow_qc,
        "bbo_rebind": {
            "source_values_changed": False,
            "upstream_declared_ohlcv_sha256": bbo_manifest.get("ohlcv_source_sha256"),
            "current_ohlcv_sha256": ohlcv_hash,
            "exact_clock_symbol_instrument_match": True,
            "reason": "new Phase 6 derivative binds the current causal OHLCV identity",
        },
        "load_verified_lineage": lineage,
        "output": {
            "path": str(destination),
            "rows": int(len(output)),
            "valid_book_rows": int(output["book_valid"].sum()),
            "valid_book_change_rows": int(output["book_change_valid"].sum()),
            "sha256": sha256_file(destination),
        },
        "sealed_holdout_read": False,
    }
    atomic_bytes(manifest_destination, canonical_json(manifest))
    print(
        json.dumps(
            {
                "output": str(destination),
                "rows": len(output),
                "valid_book_rows": int(output["book_valid"].sum()),
                "valid_book_change_rows": int(output["book_change_valid"].sum()),
                "manifest_sha256": sha256_file(manifest_destination),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
