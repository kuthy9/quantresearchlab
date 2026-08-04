from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_decision_audit_batch import (
    build_audit_batch,
    read_verified_case,
)
from smc_trader.artifact_stream import (
    atomic_bytes,
    canonical_json,
    new_stream_state,
    sha256_file,
    write_stream_manifest,
    write_stream_shards_bounded,
)


FIELD_TYPES = {
    "asof": "timestamp_ny",
    "snapshot_hash": "large_string",
    "model_action": "large_string",
    "risk_action": "large_string",
    "top_playbook": "large_string",
    "top_direction": "large_string",
    "top_phase": "large_string",
    "top_setup_id": "large_string",
    "decision_hypothesis_key": "large_string",
    "decision_playbook": "large_string",
    "decision_direction": "large_string",
    "decision_phase": "large_string",
    "decision_setup_id": "large_string",
    "h4_regime": "large_string",
    "position_open": "bool",
    "position_thesis_hash": "large_string",
    "position_setup_id": "large_string",
    "position_playbook": "large_string",
    "position_direction": "large_string",
    "observation_anomalies": "large_string",
}
ACTIONS = ("wait", "abstain", "enter", "hold", "protect", "exit")
PLAYBOOKS = (
    "displacement_first_pullback",
    "liquidity_sweep_reversal",
    "failed_auction_value_return",
)
H4_REGIMES = ("h4_down", "h4_flat", "h4_up", "h4_unready")


def _clocks() -> list[pd.Timestamp]:
    clocks: list[pd.Timestamp] = []
    for day in range(2):
        date = pd.Timestamp(
            "2023-03-06", tz="America/New_York"
        ) + pd.Timedelta(days=day)
        for hour in (1, 9, 13):
            clocks.extend(
                date + pd.Timedelta(hours=hour, minutes=minute)
                for minute in range(10)
            )
    return clocks


def _write_replay(
    root: Path,
    *,
    duplicate_clock: bool = False,
    include_traces_binding: bool = False,
) -> None:
    clocks = _clocks()
    if duplicate_clock:
        clocks[10] = clocks[9]
    rows = []
    for index, asof in enumerate(clocks):
        action = ACTIONS[index % len(ACTIONS)]
        risk_action = (
            "abstain" if index % 11 == 0 and action != "abstain" else action
        )
        playbook = PLAYBOOKS[index % len(PLAYBOOKS)]
        direction = "long" if index % 2 == 0 else "short"
        top_setup_id = f"top-only-{index // 6}"
        decision_bound = action == "enter"
        position_bound = action in {"hold", "protect", "exit"}
        decision_key = (
            f"{playbook}:{direction}" if decision_bound else None
        )
        rows.append(
            {
                "asof": asof,
                "snapshot_hash": f"{index + 1:064x}",
                "model_action": action,
                "risk_action": risk_action,
                "top_playbook": playbook,
                "top_direction": direction,
                "top_phase": (
                    "waiting_trigger" if index == 55 else "armed"
                ),
                "top_setup_id": top_setup_id,
                "decision_hypothesis_key": decision_key,
                "decision_playbook": playbook if decision_bound else None,
                "decision_direction": direction if decision_bound else None,
                "decision_phase": "executable" if decision_bound else None,
                "decision_setup_id": (
                    f"decision-{index // 6}" if decision_bound else None
                ),
                "h4_regime": H4_REGIMES[index % len(H4_REGIMES)],
                "position_open": position_bound,
                "position_thesis_hash": (
                    f"thesis-{index // 6}" if position_bound else None
                ),
                "position_setup_id": (
                    f"position-{index // 6}" if position_bound else None
                ),
                "position_playbook": playbook if position_bound else None,
                "position_direction": direction if position_bound else None,
                "observation_anomalies": json.dumps(
                    ["contract_change_history_reset"] if index == 30 else []
                ),
            }
        )
    bindings = {
        "runner": "continuous_development_stream_v1",
        "include_decision_traces": include_traces_binding,
        "decision_trace_schema_version": (
            1 if include_traces_binding else None
        ),
    }
    state = new_stream_state(FIELD_TYPES)
    buffer = list(rows)
    write_stream_shards_bounded(
        root,
        "decision_shards",
        buffer,
        state,
        key_column="snapshot_hash",
        maximum_rows=13,
        field_types=FIELD_TYPES,
    )
    manifest = write_stream_manifest(
        root,
        "decision_shards",
        state,
        artifact="continuous_development_decision_shards",
        bindings=bindings,
    )
    atomic_bytes(
        root / "COMPLETED.json",
        canonical_json(
            {
                "format_version": 1,
                "status": "complete",
                "bindings": bindings,
                "stream_manifest_sha256": {
                    "decision_shards": sha256_file(manifest)
                },
            }
        ),
    )


def test_builds_deterministic_batch_from_lightweight_rows(tmp_path: Path) -> None:
    replay = tmp_path / "replay"
    _write_replay(replay)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest_path = build_audit_batch(
        replay,
        first,
        count=20,
        max_trace_rows=12,
        unbound_context_rows=4,
    )
    second_manifest_path = build_audit_batch(
        replay,
        second,
        count=20,
        max_trace_rows=12,
        unbound_context_rows=4,
    )
    assert first_manifest_path.read_bytes() == second_manifest_path.read_bytes()
    manifest = json.loads(first_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "anchors_selected"
    assert manifest["case_count"] == 20
    assert manifest["selection"]["full_trace_fields_read"] is False
    assert manifest["source"]["include_decision_traces"] is False
    assert all(item["complete"] for item in manifest["coverage"].values())
    assert manifest["coverage"]["phase"] == {
        "available": [
            "armed",
            "executable",
            "position_open",
            "waiting_trigger",
        ],
        "selected": [
            "armed",
            "executable",
            "position_open",
            "waiting_trigger",
        ],
        "complete": True,
    }

    for item in manifest["cases"]:
        payload = read_verified_case(first / item["file"])
        anchor = payload["anchor"]
        request = payload["sampled_trajectory_request"]
        assert payload["status"] == "anchor_selected"
        assert request["rows"] == (
            request["end_ordinal"] - request["start_ordinal"] + 1
        )
        assert request["end_ordinal"] == anchor["ordinal"]
        assert request["rows"] <= (
            4 if anchor["setup_scope"] == "unbound" else 12
        )
        if anchor["ordinal"] >= 30:
            assert request["start_ordinal"] >= 30
        if (
            anchor["hypothesis_key"] is None
            and anchor["model_action"] in {"wait", "abstain"}
        ):
            assert anchor["setup_scope"] == "unbound"
            assert anchor["setup_id"] is None


def test_rejects_noncausal_clock_input(tmp_path: Path) -> None:
    replay = tmp_path / "replay"
    _write_replay(replay, duplicate_clock=True)
    with pytest.raises(ValueError, match="unique and strictly increasing"):
        build_audit_batch(replay, tmp_path / "output", count=20)


def test_rejects_trace_enabled_history_replay(tmp_path: Path) -> None:
    replay = tmp_path / "replay"
    _write_replay(replay, include_traces_binding=True)
    with pytest.raises(ValueError, match="lightweight replay"):
        build_audit_batch(replay, tmp_path / "output", count=20)


def test_rejects_incomplete_replay(tmp_path: Path) -> None:
    replay = tmp_path / "replay"
    _write_replay(replay)
    completed_path = replay / "COMPLETED.json"
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    completed["status"] = "failed"
    atomic_bytes(completed_path, canonical_json(completed))
    with pytest.raises(ValueError, match="not complete"):
        build_audit_batch(replay, tmp_path / "output", count=20)
