#!/usr/bin/env python3
"""Stream level-3 MBO into one causal execution-reality row per OHLCV minute."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shares.core.io import iter_completed_bars, load_ohlcv  # noqa: E402
from execution.core.mbo import (  # noqa: E402
    F_LAST,
    F_SNAPSHOT,
    MBO_COLUMNS,
    MBOOrderBook,
    MBORecord,
    MBOReplayError,
    assert_mbo_source_allowed,
    iter_complete_events,
)
from shares.core.validation import load_validation_protocol  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbo-root", default="data/raw/nq_mbo/legacy_parquet")
    parser.add_argument(
        "--ohlcv-source",
        default="data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet",
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--validation-protocol",
        default="configs/data_splits.json",
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--reveal-sealed-holdout",
        action="store_true",
        help="explicitly permit a final DBN holdout reveal after all hashes are frozen",
    )
    return parser.parse_args()


def _partition_paths(
    root: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[Path]:
    dates = {
        str(value.date())
        for value in pd.date_range(
            start.tz_convert("UTC").normalize() - pd.Timedelta(days=1),
            end.tz_convert("UTC").normalize(),
            freq="D",
        )
    }
    paths = [
        path
        for path in root.glob("date=*/part-*.parquet")
        if path.parent.name.removeprefix("date=") in dates
    ]
    if not paths:
        raise FileNotFoundError("no MBO partitions cover the requested interval")
    return sorted(paths)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify_partition_manifest(
    root: Path,
    paths: list[Path],
) -> tuple[str, int]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"partitioned MBO source has no integrity manifest: {manifest_path}"
        )
    raw = manifest_path.read_bytes()
    payload = json.loads(raw)
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise MBOReplayError("partitioned MBO integrity manifest has no files")
    expected = {
        str(item.get("path")): str(item.get("sha256"))
        for item in entries
        if isinstance(item, dict)
    }
    verified = 0
    for path in paths:
        try:
            relative = str(path.relative_to(root))
        except ValueError as error:
            raise MBOReplayError(
                f"MBO partition escapes its registered root: {path}"
            ) from error
        expected_hash = expected.get(relative)
        if not expected_hash:
            raise MBOReplayError(
                f"MBO partition is absent from the integrity manifest: {relative}"
            )
        actual_hash = _sha256_file(path)
        if actual_hash != expected_hash:
            raise MBOReplayError(
                f"MBO partition hash mismatch: {relative}"
            )
        verified += 1
    return hashlib.sha256(raw).hexdigest(), verified


def _invalid_row(bar, reason: str) -> dict:
    return {
        "decision_time": bar.end,
        "symbol": bar.symbol,
        "instrument_id": bar.instrument_id,
        "book_observed_at": pd.NaT,
        "publisher_id": pd.NA,
        "sequence": pd.NA,
        "bid": float("nan"),
        "ask": float("nan"),
        "bid_size": float("nan"),
        "ask_size": float("nan"),
        "top5_bid_size": float("nan"),
        "top5_ask_size": float("nan"),
        "depth_imbalance": float("nan"),
        "book_valid": False,
        "invalid_reason": reason,
    }


def _valid_row(bar, snapshot) -> dict:
    return {
        "decision_time": bar.end,
        "symbol": bar.symbol,
        "instrument_id": bar.instrument_id,
        "book_observed_at": snapshot.observed_at,
        "publisher_id": snapshot.publisher_id,
        "sequence": snapshot.sequence,
        "bid": snapshot.bid,
        "ask": snapshot.ask,
        "bid_size": snapshot.bid_size,
        "ask_size": snapshot.ask_size,
        "top5_bid_size": snapshot.top5_bid_size,
        "top5_ask_size": snapshot.top5_ask_size,
        "depth_imbalance": snapshot.depth_imbalance,
        "book_valid": True,
        "invalid_reason": "",
    }


def _replay_records(
    records,
    bars,
    *,
    end: pd.Timestamp,
    require_initial_snapshot: bool,
    selected_instrument_ids: set[int] | None = None,
) -> tuple[list[dict], int]:
    books: dict[tuple[int, int], MBOOrderBook] = {}
    latest: dict[tuple[int, int], MBOOrderBook] = {}
    invalid_reason: dict[tuple[int, int], str] = {}
    rows: list[dict] = []
    bar_index = 0

    def flush(before: pd.Timestamp | None = None) -> None:
        nonlocal bar_index
        while bar_index < len(bars) and (
            before is None or bars[bar_index].end < before
        ):
            bar = bars[bar_index]
            candidates = [
                book
                for (_, instrument_id), book in latest.items()
                if instrument_id == bar.instrument_id
                and book.last_recv is not None
                and book.last_recv <= bar.end
            ]
            selected_book = (
                max(candidates, key=lambda value: value.last_recv)
                if candidates
                else None
            )
            snapshot = (
                None if selected_book is None else selected_book.snapshot()
            )
            if snapshot is None:
                reasons = sorted(
                    {
                        reason
                        for (_, instrument_id), reason in invalid_reason.items()
                        if instrument_id == bar.instrument_id
                    }
                )
                rows.append(
                    _invalid_row(
                        bar,
                        "|".join(reasons) if reasons else "no_complete_book",
                    )
                )
            else:
                rows.append(_valid_row(bar, snapshot))
            bar_index += 1

    event_count = 0
    last_event_time: pd.Timestamp | None = None
    initialized_keys: set[tuple[int, int]] = set()
    snapshot_clear_preambles: dict[
        tuple[int, int], tuple[MBORecord, ...]
    ] = {}
    selected = (
        None
        if selected_instrument_ids is None
        else {int(value) for value in selected_instrument_ids}
    )
    for event in iter_complete_events(records):
        event_time = max(record.ts_recv for record in event)
        if last_event_time is not None and event_time < last_event_time:
            raise RuntimeError("MBO source is not globally ordered by receive time")
        last_event_time = event_time
        if event_time >= end:
            break
        flush(event_time)
        subevents: dict[tuple[int, int], list[MBORecord]] = {}
        for record in event:
            if selected is not None and record.instrument_id not in selected:
                continue
            if (
                record.action != "R"
                and record.price is not None
                and record.price <= 0
            ):
                raise MBOReplayError(
                    "selected execution instrument contains a nonpositive "
                    f"price: instrument_id={record.instrument_id}, "
                    f"ts_recv={record.ts_recv.isoformat()}"
                )
            key = (record.publisher_id, record.instrument_id)
            subevents.setdefault(key, []).append(record)
        if not subevents:
            continue
        event_count += 1
        for key, values in subevents.items():
            # F_LAST closes the complete vendor packet, not necessarily each
            # instrument inside a multi-book packet. Preserve packet atomicity
            # first, then make each book-specific subevent independently
            # complete for the single-key order-book state machine.
            values[-1] = replace(values[-1], flags=values[-1].flags | F_LAST)
            subevent = tuple(values)
            if key not in initialized_keys:
                snapshot_flags = [
                    bool(record.flags & F_SNAPSHOT) for record in subevent
                ]
                is_standalone_snapshot = (
                    subevent[0].action == "R"
                    and any(snapshot_flags)
                )
                is_clear_preamble = (
                    not any(snapshot_flags)
                    and len(subevent) == 1
                    and subevent[0].action == "R"
                )
                is_snapshot_continuation = (
                    key in snapshot_clear_preambles
                    and all(snapshot_flags)
                    and all(record.action == "A" for record in subevent)
                )
                if require_initial_snapshot and is_clear_preamble:
                    snapshot_clear_preambles[key] = tuple(
                        replace(record, flags=record.flags & ~F_LAST)
                        for record in subevent
                    )
                    continue
                if require_initial_snapshot and is_snapshot_continuation:
                    subevent = (
                        *snapshot_clear_preambles.pop(key),
                        *subevent,
                    )
                elif require_initial_snapshot and not is_standalone_snapshot:
                    raise MBOReplayError(
                        "daily parallel replay requires a complete initial "
                        f"snapshot for book {key}"
                    )
                initialized_keys.add(key)
            book = books.setdefault(key, MBOOrderBook(*key))
            book.apply_complete_event(subevent, capture_snapshot=False)
            if not book.valid:
                latest.pop(key, None)
                invalid_reason[key] = book.invalid_reason
            else:
                latest[key] = book
                invalid_reason.pop(key, None)
    flush()
    if require_initial_snapshot:
        initialized_instruments = {
            instrument_id for _, instrument_id in initialized_keys
        }
        missing = sorted((selected or set()) - initialized_instruments)
        if missing:
            raise MBOReplayError(
                "daily parallel replay ended before a complete initial "
                f"snapshot for instruments {missing}"
            )
    return rows, event_count


def _iter_selected_parquet_records_preserving_packets(
    path: Path,
    *,
    batch_size: int,
    selected_instrument_ids: set[int],
):
    """Select records only after observing each complete vendor packet.

    Every raw row is inspected for ``F_LAST``. Only selected outright records
    are parsed into MBORecord objects, so legitimate zero/negative calendar
    spread prices cannot weaken the outright-price invariant or add needless
    object-construction cost.
    """

    import pyarrow.dataset as ds

    selected = {int(value) for value in selected_instrument_ids}
    if not selected:
        return
    dataset = ds.dataset(path, format="parquet")
    missing = sorted(set(MBO_COLUMNS) - set(dataset.schema.names))
    if missing:
        raise MBOReplayError(f"{path} is missing MBO fields: {missing}")
    scanner = dataset.scanner(
        columns=list(MBO_COLUMNS),
        batch_size=int(batch_size),
        use_threads=False,
    )
    pending = pd.DataFrame()
    packet_offset = 0
    packet_open = False
    for batch in scanner.to_batches():
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
        selected_mask = np.isin(ids, tuple(selected))
        if bool(selected_mask.any()):
            import pyarrow as pa

            selected_batch = batch.filter(pa.array(selected_mask))
            frame = selected_batch.to_pandas()
            frame["_packet_group"] = group_ids[selected_mask]
            pending = (
                frame
                if pending.empty
                else pd.concat((pending, frame), ignore_index=True)
            )
        packet_offset += int(last.sum())
        packet_open = not bool(last[-1]) if len(last) else packet_open
        if not pending.empty:
            closed_mask = pending["_packet_group"] < packet_offset
            if bool(closed_mask.any()):
                closed = pending.loc[closed_mask].copy()
                pending = pending.loc[~closed_mask].copy()
                closed_flags = (
                    closed["flags"].to_numpy(dtype=np.int64) & ~F_LAST
                )
                closed_groups = closed["_packet_group"].to_numpy(
                    dtype=np.int64
                )
                group_end = np.r_[
                    closed_groups[1:] != closed_groups[:-1],
                    True,
                ]
                closed_flags[group_end] |= F_LAST
                closed["flags"] = closed_flags
                closed.drop(columns=["_packet_group"], inplace=True)
                for row in closed.itertuples(index=False):
                    yield MBORecord.from_value(row)
    if packet_open or not pending.empty:
        raise MBOReplayError(f"{path} ends inside an incomplete vendor event")


def _iter_selected_dbn_records_preserving_packets(
    path: Path,
    *,
    selected_instrument_ids: set[int],
    allow_sealed_holdout: bool,
):
    """Stream selected DBN records without losing vendor packet boundaries."""

    try:
        import databento as db
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "DBN MBO replay requires the 'dbn' optional dependency"
        ) from error
    from shares.core.io import require_materialized

    source = assert_mbo_source_allowed(
        path,
        allow_sealed_holdout=allow_sealed_holdout,
    )
    require_materialized(source)
    selected = {int(value) for value in selected_instrument_ids}
    if not selected:
        return
    store = db.DBNStore.from_file(source)
    pending: list[MBORecord] = []
    packet_open = False
    for value in store:
        if not hasattr(value, "action") or not hasattr(value, "instrument_id"):
            continue
        packet_open = True
        flags = int(value.flags)
        if int(value.instrument_id) in selected:
            record = MBORecord.from_value(value, fixed_point_price=True)
            pending.append(replace(record, flags=record.flags & ~F_LAST))
        if flags & F_LAST:
            if pending:
                pending[-1] = replace(
                    pending[-1],
                    flags=pending[-1].flags | F_LAST,
                )
                yield from pending
                pending.clear()
            packet_open = False
    if packet_open:
        raise MBOReplayError(f"{source} ends inside an incomplete vendor event")


def _parquet_partition_worker(payload) -> tuple[list[dict], int, str]:
    path, bars, batch_size, end = payload
    instrument_ids = sorted({bar.instrument_id for bar in bars})
    records = _iter_selected_parquet_records_preserving_packets(
        path,
        batch_size=batch_size,
        selected_instrument_ids=set(instrument_ids),
    )
    rows, event_count = _replay_records(
        records,
        bars,
        end=end,
        require_initial_snapshot=True,
        selected_instrument_ids=set(instrument_ids),
    )
    return rows, event_count, str(path)


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    protocol = load_validation_protocol(args.validation_protocol)
    if not 1 <= int(args.workers) <= 4:
        raise ValueError("--workers must be between 1 and 4")
    window = protocol.classify_mbo(start, end)
    if window.role == "sealed_holdout" and not args.reveal_sealed_holdout:
        raise RuntimeError(
            "sealed MBO materialization requires --reveal-sealed-holdout"
        )
    root = assert_mbo_source_allowed(
        args.mbo_root,
        allow_sealed_holdout=args.reveal_sealed_holdout,
    )
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    ohlcv_hash = _sha256_file(args.ohlcv_source)
    if ohlcv_hash != protocol.causal_source.sha256:
        raise RuntimeError(
            "MBO execution materialization requires the preregistered "
            "strict previous-session contract series"
        )
    loaded = load_ohlcv(
        args.ohlcv_source,
        start=start - pd.Timedelta(minutes=1),
        end=end,
    )
    if not loaded.contract_selection_causal:
        raise RuntimeError("MBO materialization requires the causal contract series")
    bars = [
        bar
        for bar in iter_completed_bars(loaded.frame)
        if start <= bar.end < end
    ]
    if not bars:
        raise ValueError("requested interval contains no causal OHLCV bars")
    instrument_ids = sorted({bar.instrument_id for bar in bars})
    is_dbn = root.is_file() and (
        root.name.lower().endswith(".dbn")
        or root.name.lower().endswith(".dbn.zst")
    )
    if is_dbn:
        paths = [root]
        partition_manifest_hash = None
        verified_partition_hashes = 0
        records = _iter_selected_dbn_records_preserving_packets(
            root,
            selected_instrument_ids=set(instrument_ids),
            allow_sealed_holdout=bool(args.reveal_sealed_holdout),
        )
        rows, event_count = _replay_records(
            records,
            bars,
            end=end,
            require_initial_snapshot=False,
            selected_instrument_ids=set(instrument_ids),
        )
    else:
        paths = _partition_paths(root, start=start, end=end)
        paths_by_date = {
            path.parent.name.removeprefix("date="): path for path in paths
        }
        bars_by_date: dict[str, list] = {}
        for bar in bars:
            key = str(bar.end.tz_convert("UTC").date())
            bars_by_date.setdefault(key, []).append(bar)
        missing_dates = sorted(set(bars_by_date) - set(paths_by_date))
        rows = [
            _invalid_row(bar, "mbo_partition_missing")
            for key in missing_dates
            for bar in bars_by_date[key]
        ]
        tasks = []
        used_paths: list[Path] = []
        for key in sorted(set(bars_by_date) & set(paths_by_date)):
            path = paths_by_date[key]
            local_bars = bars_by_date[key]
            local_end = min(
                end,
                pd.Timestamp(key, tz="UTC") + pd.Timedelta(days=1),
            )
            tasks.append((path, local_bars, int(args.batch_size), local_end))
            used_paths.append(path)
        (
            partition_manifest_hash,
            verified_partition_hashes,
        ) = _verify_partition_manifest(root, used_paths)
        if int(args.workers) == 1:
            results = [_parquet_partition_worker(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
                futures = [
                    executor.submit(_parquet_partition_worker, task)
                    for task in tasks
                ]
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print(
                        json.dumps({"mbo_partition_completed": result[2]}),
                        flush=True,
                    )
        event_count = 0
        for local_rows, local_events, completed_path in results:
            rows.extend(local_rows)
            event_count += local_events
            if int(args.workers) == 1:
                print(
                    json.dumps({"mbo_partition_completed": completed_path}),
                    flush=True,
                )
        paths = used_paths

    output = pd.DataFrame(rows).sort_values("decision_time", kind="stable")
    if len(output) != len(bars) or output["decision_time"].duplicated().any():
        raise AssertionError("MBO minute materialization lost or duplicated OHLCV clocks")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(destination, index=False)
    manifest = {
        "format_version": 1,
        "validation_schema_version": protocol.schema_version,
        "validation_protocol_hash": protocol.fingerprint,
        "validation_window_role": window.role,
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "mbo_root": str(root),
        "mbo_partitions": [str(path) for path in paths],
        "mbo_partition_manifest_sha256": partition_manifest_hash,
        "verified_partition_hashes": verified_partition_hashes,
        "ohlcv_source": str(loaded.source),
        "ohlcv_source_sha256": ohlcv_hash,
        "contract_selection_causal": loaded.contract_selection_causal,
        "instrument_ids": instrument_ids,
        "complete_vendor_events": event_count,
        "workers": 1 if is_dbn else int(args.workers),
        "minute_rows": len(output),
        "valid_book_rows": int(output["book_valid"].sum()),
        "invalid_book_rows": int((~output["book_valid"]).sum()),
        "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "sealed_holdout_read": window.role == "sealed_holdout",
    }
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({key: manifest[key] for key in ("minute_rows", "valid_book_rows")}))


if __name__ == "__main__":
    main()
