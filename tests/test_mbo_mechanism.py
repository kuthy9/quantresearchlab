from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_mbo_mechanism import (
    _aggregate_closed_packets,
    _aggregate_parquet_partition,
)
from smc_trader.artifact_stream import atomic_bytes, canonical_json
from smc_trader.mbo import (
    F_BAD_TS_RECV,
    F_LAST,
    F_PUBLISHER_SPECIFIC,
    F_SNAPSHOT,
    MBORecord,
)
from smc_trader.mbo_mechanism import (
    MBO_MECHANISM_PROTOCOL_SHA256,
    MBO_MECHANISM_SCHEMA_VERSION,
    MBOMechanismArtifactError,
    aggregate_flow_records,
    build_minute_mechanism_frame,
    completed_minute_clock,
    flow_updates,
    load_mbo_mechanism_artifact,
)


def _record(
    clock: str | pd.Timestamp,
    *,
    action: str,
    side: str,
    size: float,
    flags: int,
    sequence: int,
) -> MBORecord:
    timestamp = pd.Timestamp(clock)
    return MBORecord(
        ts_recv=timestamp,
        ts_event=timestamp - pd.Timedelta(milliseconds=1),
        publisher_id=1,
        instrument_id=13743,
        action=action,
        side=side,
        price=20_000.0 if action != "R" else None,
        size=size,
        order_id=sequence,
        flags=flags,
        sequence=sequence,
    )


def test_completed_clock_uses_receive_time_and_exact_boundary_is_inclusive() -> None:
    assert completed_minute_clock("2024-06-02T22:00:00Z") == pd.Timestamp(
        "2024-06-02T22:00:00Z"
    )
    assert completed_minute_clock("2024-06-02T22:00:00.000000001Z") == pd.Timestamp(
        "2024-06-02T22:01:00Z"
    )
    assert completed_minute_clock("2024-06-02T22:00:59.999999999Z") == pd.Timestamp(
        "2024-06-02T22:01:00Z"
    )
    with pytest.raises(Exception, match="timezone aware"):
        completed_minute_clock("2024-06-02 22:00:01")


def test_window_boundary_and_t_f_accounting_are_causal_and_not_double_counted() -> None:
    records = (
        # Exact start maps to the absent pre-window decision clock and drops.
        _record(
            "2024-06-02T22:00:00Z",
            action="T",
            side="B",
            size=99,
            flags=F_LAST,
            sequence=1,
        ),
        # A complete T/F vendor packet: T is trade volume, F is passive fill.
        _record(
            "2024-06-02T22:00:00.000000001Z",
            action="T",
            side="B",
            size=3,
            flags=0,
            sequence=2,
        ),
        _record(
            "2024-06-02T22:00:00.000000002Z",
            action="F",
            side="A",
            size=3,
            flags=F_LAST,
            sequence=3,
        ),
        # Exact end maps to end and is not one of the registered clocks.
        _record(
            "2024-06-02T22:02:00Z",
            action="T",
            side="A",
            size=77,
            flags=F_LAST,
            sequence=4,
        ),
    )
    frame = aggregate_flow_records(
        records,
        decision_times=[
            pd.Timestamp("2024-06-02T22:01:00Z"),
        ],
        instrument_id=13743,
    )
    row = frame.iloc[0]
    assert row["aggressor_trade_volume"] == 3
    assert row["aggressor_buy_volume"] == 3
    assert row["passive_ask_fill_volume"] == 3
    assert row["aggressor_trade_volume"] != (
        row["aggressor_buy_volume"] + row["passive_ask_fill_volume"]
    )


def test_snapshot_or_bad_packet_excludes_all_selected_t_and_f_records() -> None:
    records = (
        _record(
            "2024-06-02T22:00:01Z",
            action="T",
            side="B",
            size=2,
            flags=0,
            sequence=1,
        ),
        _record(
            "2024-06-02T22:00:01.000000001Z",
            action="F",
            side="A",
            size=2,
            flags=F_SNAPSHOT | F_LAST,
            sequence=2,
        ),
        _record(
            "2024-06-02T22:00:02Z",
            action="T",
            side="A",
            size=4,
            flags=F_BAD_TS_RECV | F_LAST,
            sequence=3,
        ),
    )
    frame = aggregate_flow_records(
        records,
        decision_times=[pd.Timestamp("2024-06-02T22:01:00Z")],
        instrument_id=13743,
    )
    assert frame.iloc[0]["aggressor_trade_volume"] == 0
    assert frame.iloc[0]["passive_ask_fill_volume"] == 0


def test_publisher_specific_flag_is_audited_but_does_not_erase_live_flow() -> None:
    frame = aggregate_flow_records(
        [
            _record(
                "2024-06-02T22:00:01Z",
                action="A",
                side="B",
                size=12,
                flags=F_PUBLISHER_SPECIFIC | F_LAST,
                sequence=1,
            )
        ],
        decision_times=[pd.Timestamp("2024-06-02T22:01:00Z")],
        instrument_id=13743,
    )
    assert frame.iloc[0]["displayed_bid_add_volume"] == 12


def test_modify_size_is_named_as_reported_size_not_additive_volume() -> None:
    updates = flow_updates("M", "B", 12)
    assert updates == {
        "displayed_bid_modify_reported_size": 12.0,
        "displayed_bid_modify_count": 1,
    }
    assert not any("add_volume" in name for name in updates)


def test_vector_packet_qc_reports_excluded_live_t_and_f_rows() -> None:
    start = pd.Timestamp("2024-06-02T22:00:00Z")
    closed = pd.DataFrame(
        [
            {
                "ts_recv": start + pd.Timedelta(seconds=1),
                "publisher_id": 1,
                "instrument_id": 13743,
                "action": "T",
                "side": "B",
                "size": 2,
                "flags": 0,
                "_packet_group": 0,
            },
            {
                "ts_recv": start + pd.Timedelta(seconds=1, nanoseconds=1),
                "publisher_id": 1,
                "instrument_id": 13743,
                "action": "F",
                "side": "A",
                "size": 2,
                "flags": F_SNAPSHOT | F_LAST,
                "_packet_group": 0,
            },
            {
                "ts_recv": start + pd.Timedelta(seconds=2),
                "publisher_id": 1,
                "instrument_id": 13743,
                "action": "T",
                "side": "A",
                "size": 1,
                "flags": F_LAST,
                "_packet_group": 1,
            },
        ]
    )
    totals, qc = _aggregate_closed_packets(
        closed,
        start=start,
        end=start + pd.Timedelta(minutes=1),
    )
    assert qc["excluded_t_rows"] == 1
    assert qc["excluded_f_rows"] == 1
    assert qc["eligible_t_rows"] == 1
    assert qc["eligible_f_rows"] == 0
    assert qc["flag_value_counts"] == {
        "0": 1,
        "128": 1,
        "160": 1,
    }
    assert totals["volume"].sum() == 1


def test_partition_scan_preserves_packet_boundary_before_instrument_selection(
    tmp_path: Path,
) -> None:
    start = pd.Timestamp("2024-06-02T22:00:00Z")
    path = tmp_path / "raw.parquet"
    rows = [
        # The selected T row is not F_LAST; an unselected record closes packet 0.
        (start + pd.Timedelta(seconds=1), 1, 13743, "T", "B", 2, 0),
        (start + pd.Timedelta(seconds=1, nanoseconds=1), 1, 999, "F", "A", 2, F_LAST),
        # Packet 1 is snapshot-flagged and must contribute no selected flow.
        (start + pd.Timedelta(seconds=2), 1, 13743, "T", "A", 9, F_SNAPSHOT),
        (
            start + pd.Timedelta(seconds=2, nanoseconds=1),
            1,
            13743,
            "F",
            "B",
            9,
            F_SNAPSHOT | F_LAST,
        ),
    ]
    pd.DataFrame(
        rows,
        columns=[
            "ts_recv",
            "publisher_id",
            "instrument_id",
            "action",
            "side",
            "size",
            "flags",
        ],
    ).to_parquet(path, index=False)
    totals, qc, completed_path = _aggregate_parquet_partition(
        (path, start, start + pd.Timedelta(minutes=1), 13743, 1)
    )
    assert completed_path == str(path)
    assert qc["raw_vendor_packets"] == 2
    assert qc["eligible_t_rows"] == 1
    assert qc["excluded_t_rows"] == 1
    assert qc["excluded_f_rows"] == 1
    assert totals.loc[totals["action"].eq("T"), "volume"].sum() == 2


def _bbo_frame(clocks: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_time": clocks["decision_time"],
            "symbol": clocks["symbol"],
            "instrument_id": clocks["instrument_id"],
            "book_observed_at": [
                clocks["decision_time"].iloc[0] - pd.Timedelta(milliseconds=10),
                clocks["decision_time"].iloc[1],
            ],
            "publisher_id": [1, 1],
            "sequence": [1, 2],
            "bid": [100.0, 100.25],
            "ask": [100.25, 100.5],
            "bid_size": [5.0, 7.0],
            "ask_size": [4.0, 3.0],
            "top5_bid_size": [20.0, 22.0],
            "top5_ask_size": [18.0, 17.0],
            "depth_imbalance": [0.05, 0.12],
            "book_valid": [True, True],
            "invalid_reason": ["", ""],
        }
    )


def test_minute_frame_derives_causal_bbo_ofi_and_impact() -> None:
    clocks = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                ["2024-06-02T22:01:00Z", "2024-06-02T22:02:00Z"], utc=True
            ),
            "symbol": ["NQM4", "NQM4"],
            "instrument_id": [13743, 13743],
        }
    )
    flow = pd.DataFrame(
        {
            "decision_time": clocks["decision_time"],
            "aggressor_buy_volume": [0.0, 2.0],
        }
    )
    output = build_minute_mechanism_frame(clocks, flow, _bbo_frame(clocks))
    assert np.isnan(output.loc[0, "best_level_ofi_contracts"])
    # Bid and ask both moved up: current bid queue + previous ask queue.
    assert output.loc[1, "best_level_ofi_contracts"] == 11.0
    assert output.loc[1, "mid_change_ticks"] == 1.0
    assert output.loc[1, "absolute_mid_impact_ticks_per_aggressor_contract"] == 0.5
    assert output.loc[1, "book_age_seconds"] == 0.0


def test_loader_verifies_output_and_lineage_hashes(tmp_path: Path) -> None:
    clocks = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                ["2024-06-02T22:01:00Z", "2024-06-02T22:02:00Z"], utc=True
            ),
            "symbol": ["NQM4", "NQM4"],
            "instrument_id": [13743, 13743],
        }
    )
    artifact = tmp_path / "features.parquet"
    frame = build_minute_mechanism_frame(
        clocks,
        pd.DataFrame({"decision_time": clocks["decision_time"]}),
        _bbo_frame(clocks),
    )
    frame.to_parquet(artifact, index=False)
    lineage = tmp_path / "protocol.json"
    lineage.write_text("{}", encoding="utf-8")
    manifest = {
        "artifact_kind": "mbo_minute_mechanism_features",
        "feature_schema_version": MBO_MECHANISM_SCHEMA_VERSION,
        "feature_protocol_sha256": MBO_MECHANISM_PROTOCOL_SHA256,
        "start": "2024-06-02T22:00:00Z",
        "end_exclusive": "2024-06-02T22:03:00Z",
        "symbol": "NQM4",
        "instrument_id": 13743,
        "load_verified_lineage": [
            {
                "role": "protocol",
                "path": str(lineage),
                "sha256": hashlib.sha256(lineage.read_bytes()).hexdigest(),
            }
        ],
        "output": {
            "rows": 2,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
    }
    manifest_path = artifact.with_suffix(".parquet.manifest.json")
    atomic_bytes(manifest_path, canonical_json(manifest))
    loaded = load_mbo_mechanism_artifact(artifact)
    assert len(loaded) == 2
    lineage.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(MBOMechanismArtifactError, match="lineage SHA-256"):
        load_mbo_mechanism_artifact(artifact)
