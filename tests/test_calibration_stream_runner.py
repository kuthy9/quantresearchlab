from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
import pickle
import subprocess
import sys

import pandas as pd

from scripts.run_continuous_replay import (
    BRAIN_CALIBRATION_FIELD_TYPES,
    DECISION_FIELD_TYPES,
)
from smc_trader.brain_calibration import BrainCalibrationRecord


ROOT = Path(__file__).resolve().parents[1]


def test_stream_schemas_match_the_lightweight_contract() -> None:
    assert tuple(BRAIN_CALIBRATION_FIELD_TYPES) == tuple(
        field.name for field in fields(BrainCalibrationRecord)
    )
    assert DECISION_FIELD_TYPES["asof"] == "timestamp_ny"
    assert DECISION_FIELD_TYPES["model_action"] == "large_string"
    assert DECISION_FIELD_TYPES["invalidation_source_id"] == "large_string"
    assert DECISION_FIELD_TYPES["target_ids"] == "large_string"


def _write_source(tmp_path: Path) -> Path:
    index = pd.date_range(
        "2022-06-06 18:00",
        periods=32,
        freq="min",
        tz="America/New_York",
        name="ts",
    )
    steps = pd.Series(range(len(index)), index=index, dtype=float)
    close = 12_500.0 + 0.25 * (steps % 9)
    frame = pd.DataFrame(
        {
            "open": close.shift(1, fill_value=close.iloc[0]),
            "close": close,
            "volume": 10.0 + (steps % 5),
            "symbol": "NQU2",
            "instrument_id": 1,
        },
        index=index,
    )
    frame["high"] = frame[["open", "close"]].max(axis=1) + 0.25
    frame["low"] = frame[["open", "close"]].min(axis=1) - 0.25
    source = tmp_path / "research_previous_session_front.parquet"
    frame.to_parquet(source)
    return source


def _command(
    source: Path,
    output: Path,
    *,
    resume: bool = False,
    stop_after: int = 0,
    brain_calibration: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_continuous_replay.py",
        "--source",
        str(source),
        "--output",
        str(output),
        "--config",
        "configs/model.json",
        "--validation-protocol",
        "configs/data_splits.json",
        "--start",
        "2022-06-06T18:00:00-04:00",
        "--end",
        "2022-06-06T18:32:00-04:00",
        "--warmup-days",
        "0",
        "--shard-rows",
        "7",
        "--checkpoint-bars",
        "5",
        "--acknowledge-research-roll-lineage",
    ]
    if resume:
        command.append("--resume")
    if stop_after:
        command.extend(["--diagnostic-stop-after-bars", str(stop_after)])
    if brain_calibration:
        command.append("--brain-calibration")
    return command


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_default_replay_is_lightweight_resumable_and_deterministic(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    resumed_output = tmp_path / "resumed"
    interrupted = _run(_command(source, resumed_output, stop_after=11))
    assert interrupted.returncode != 0

    progress = json.loads((resumed_output / "progress.json").read_text())
    assert progress["status"] == "failed"
    assert progress["resume_supported"] is True
    assert 0.0 < progress["completed_percent"] < 100.0

    checkpoint_manifest = json.loads(
        (resumed_output / "_checkpoint/manifest.json").read_text()
    )
    checkpoint = pickle.loads(
        (
            resumed_output / "_checkpoint" / checkpoint_manifest["state_file"]
        ).read_bytes()
    )
    assert set(checkpoint["buffers"]) == {"decision_shards"}
    assert len(checkpoint["buffers"]["decision_shards"]) < 7

    resumed = _run(_command(source, resumed_output, resume=True))
    assert resumed.returncode == 0, resumed.stderr

    uninterrupted_output = tmp_path / "uninterrupted"
    uninterrupted = _run(_command(source, uninterrupted_output))
    assert uninterrupted.returncode == 0, uninterrupted.stderr

    resumed_summary = json.loads((resumed_output / "summary.json").read_text())
    uninterrupted_summary = json.loads(
        (uninterrupted_output / "summary.json").read_text()
    )
    assert resumed_summary["decision_rows"] == uninterrupted_summary["decision_rows"]
    assert resumed_summary["rolling_state_commitment"] == (
        uninterrupted_summary["rolling_state_commitment"]
    )
    assert resumed_summary["resume_count"] == 1
    assert resumed_summary["full_snapshot_hash_per_minute"] is False
    assert max(resumed_summary["peak_buffer_rows"].values()) <= 7

    assert (resumed_output / "decision_shards.manifest.json").is_file()
    for retired in (
        "funnel_transition_shards.manifest.json",
        "path_test_shards.manifest.json",
        "decision_trace_shards.manifest.json",
        "ai_primitive_proposals.json",
    ):
        assert not (resumed_output / retired).exists()


def test_brain_calibration_is_the_only_optional_default_stream(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    output = tmp_path / "calibration"
    completed = _run(_command(source, output, brain_calibration=True))
    assert completed.returncode == 0, completed.stderr
    assert (output / "decision_shards.manifest.json").is_file()
    assert (output / "brain_calibration_shards.manifest.json").is_file()
    state = json.loads((output / "summary.json").read_text())
    assert set(state["peak_buffer_rows"]) == {
        "decision_shards",
        "brain_calibration_shards",
    }
