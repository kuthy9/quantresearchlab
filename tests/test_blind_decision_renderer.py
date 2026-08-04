from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.render_blind_decision_batch as renderer


def _snapshot(ordinal: int) -> SimpleNamespace:
    asof = pd.Timestamp("2023-03-06 09:30", tz="America/New_York") + pd.Timedelta(
        minutes=ordinal
    )
    return SimpleNamespace(
        snapshot_hash=f"hash-{ordinal}",
        observation=SimpleNamespace(asof=asof),
        decision=SimpleNamespace(
            selected_action=SimpleNamespace(value="wait"),
            best_hypothesis_key=None,
        ),
        risk=SimpleNamespace(final_action=SimpleNamespace(value="wait")),
        belief=SimpleNamespace(hypotheses={}),
    )


def _target(start: int, end: int, case_id: str) -> dict:
    return {
        "case_id": case_id,
        "trace_start_ordinal": start,
        "anchor_ordinal": end,
        "asof": _snapshot(end).observation.asof,
        "snapshot_hash": f"hash-{end}",
        "trace_records": [],
        "trajectory_filename": f"sampled_trajectories/{case_id}.json",
    }


def test_trace_builder_runs_once_per_union_clock(monkeypatch) -> None:
    calls: list[str] = []

    def fake_build(snapshot, *_args, **_kwargs):
        calls.append(snapshot.snapshot_hash)
        return {
            "schema_version": 1,
            "decision_hash": snapshot.snapshot_hash,
            "future_path_included": False,
        }

    monkeypatch.setattr(renderer, "build_decision_trace", fake_build)
    first = _target(1, 3, "first")
    second = _target(3, 5, "second")
    targets = [first, second]
    for ordinal in range(7):
        renderer._capture_sampled_trace(
            ordinal=ordinal,
            targets=targets,
            snapshot=_snapshot(ordinal),
            previous_snapshot=None,
            source_bar=None,
            account_state=None,
            belief_position_input=None,
        )

    assert calls == [f"hash-{ordinal}" for ordinal in range(1, 6)]
    assert [row["ordinal"] for row in first["trace_records"]] == [1, 2, 3]
    assert [row["ordinal"] for row in second["trace_records"]] == [3, 4, 5]
    assert first["trace_records"][-1] is second["trace_records"][0]


def test_sampled_trajectory_is_persisted_and_bound(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        renderer,
        "build_decision_trace",
        lambda snapshot, *_args, **_kwargs: {
            "schema_version": 1,
            "decision_hash": snapshot.snapshot_hash,
            "future_path_included": False,
        },
    )
    target = _target(1, 3, "case-1")
    for ordinal in range(1, 4):
        renderer._capture_sampled_trace(
            ordinal=ordinal,
            targets=[target],
            snapshot=_snapshot(ordinal),
            previous_snapshot=None,
            source_bar=None,
            account_state=None,
            belief_position_input=None,
        )
    (tmp_path / "sampled_trajectories").mkdir()
    path = renderer._write_sampled_trajectory(
        target=target,
        destination=tmp_path,
        batch_manifest_sha256="a" * 64,
        replay_completed_sha256="b" * 64,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["rows"] == 3
    assert payload["maximum_market_time"] == target["asof"].isoformat()
    assert payload["future_path_included"] is False
    assert payload["source_binding"] == {
        "anchor_snapshot_hash": "hash-3",
        "batch_manifest_sha256": "a" * 64,
        "replay_completed_sha256": "b" * 64,
    }


def test_unbound_anchor_cannot_inherit_top_setup() -> None:
    snapshot = _snapshot(0)
    target = {
        "case_id": "unbound",
        "asof": snapshot.observation.asof,
        "snapshot_hash": snapshot.snapshot_hash,
        "model_action": "wait",
        "risk_action": "wait",
        "hypothesis_key": None,
        "setup_scope": "unbound",
        "setup_id": None,
    }
    renderer._validate_anchor_identity(target, snapshot, None)
    target["setup_id"] = "top-only-setup"
    with pytest.raises(ValueError, match="invalid setup binding"):
        renderer._validate_anchor_identity(target, snapshot, None)


def _write_bound_mbo(tmp_path: Path) -> tuple[Path, dict]:
    clock = pd.Timestamp("2024-06-24 09:31", tz="America/New_York")
    source = tmp_path / "minute_execution.parquet"
    pd.DataFrame(
        {
            "decision_time": [clock],
            "instrument_id": [1],
            "book_observed_at": [clock],
            "bid": [100.0],
            "ask": [100.25],
            "bid_size": [5.0],
            "ask_size": [6.0],
            "book_valid": [True],
        }
    ).to_parquet(source, index=False)
    validation_hash = "v" * 64
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "output_sha256": renderer.sha256_file(source),
                "validation_protocol_hash": validation_hash,
            }
        ),
        encoding="utf-8",
    )
    return source, {
        "mbo_execution_sha256": renderer.sha256_file(source),
        "mbo_execution_manifest_sha256": renderer.sha256_file(manifest_path),
        "validation_protocol_hash": validation_hash,
    }


def test_bound_mbo_loader_verifies_parquet_manifest_and_validation(
    tmp_path: Path,
) -> None:
    source, bindings = _write_bound_mbo(tmp_path)
    store = renderer._load_bound_mbo_execution(source, bindings)
    assert len(store.frame) == 1

    with pytest.raises(ValueError, match="parquet differs"):
        renderer._load_bound_mbo_execution(
            source,
            {**bindings, "mbo_execution_sha256": "x" * 64},
        )

    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    with pytest.raises(ValueError, match="manifest differs"):
        renderer._load_bound_mbo_execution(
            source,
            {**bindings, "mbo_execution_manifest_sha256": "x" * 64},
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"] = "x" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    invalid_output_bindings = {
        **bindings,
        "mbo_execution_manifest_sha256": renderer.sha256_file(manifest_path),
    }
    with pytest.raises(ValueError, match="output hash is invalid"):
        renderer._load_bound_mbo_execution(source, invalid_output_bindings)

    manifest["output_sha256"] = renderer.sha256_file(source)
    manifest["validation_protocol_hash"] = "other"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    invalid_validation_bindings = {
        **bindings,
        "mbo_execution_manifest_sha256": renderer.sha256_file(manifest_path),
    }
    with pytest.raises(ValueError, match="validation binding is invalid"):
        renderer._load_bound_mbo_execution(source, invalid_validation_bindings)
