from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from brain.scripts._run_identity import RunWindow
from brain.scripts.run_llm_brain import IBKR_MAX_TAPE_AGE, tape_is_current

NOW = pd.Timestamp("2026-09-17T14:00:00Z")


def test_a_historical_window_is_not_current() -> None:
    assert tape_is_current("2022-01-03 12:00", now=NOW) is False


def test_a_window_ending_within_the_allowed_age_is_current() -> None:
    assert IBKR_MAX_TAPE_AGE == pd.Timedelta(days=1)
    assert tape_is_current("2026-09-17 09:00", now=NOW) is True
    assert tape_is_current("2026-09-16 09:00", now=NOW) is False


def test_model_label_carries_the_reasoning_effort_override() -> None:
    from brain.scripts.run_llm_brain import model_label

    assert model_label("deepseek", "deepseek-flash", None) == "deepseek:deepseek-flash"
    assert model_label("deepseek", "deepseek-flash", "max") == "deepseek:deepseek-flash@max"


def test_drive_records_eye_and_bar_timings(tmp_path) -> None:
    from brain.scripts._run_identity import drive
    from shares.core.timing import Timings

    pytest.importorskip("pyarrow")
    source = Path(__file__).resolve().parents[2] / "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"
    if not source.exists():
        pytest.skip("materialized NQ tape not linked")
    timings = Timings()
    seen = []
    drive(
        RunWindow(source, "2022-01-03 09:30", "2022-01-03 09:35", "2022-01-03 09:40"),
        model_path=Path(__file__).resolve().parents[2] / "configs/model.json", root=Path(__file__).resolve().parents[2],
        on_observation=lambda obs, emitting, bar: seen.append(emitting), timings=timings,
    )
    summary = timings.summary()
    assert summary["eye"]["count"] == len(seen) >= 5 and summary["bar"]["count"] == sum(seen) >= 1


def test_drive_stops_when_asked(tmp_path) -> None:
    from brain.scripts._run_identity import drive

    pytest.importorskip("pyarrow")
    source = Path(__file__).resolve().parents[2] / "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet"
    if not source.exists():
        pytest.skip("materialized NQ tape not linked")
    emitted = []
    logs = []
    seen = drive(
        RunWindow(source, "2022-01-03 09:30", "2022-01-03 09:35", "2022-01-03 10:40"),
        model_path=Path(__file__).resolve().parents[2] / "configs/model.json", root=Path(__file__).resolve().parents[2],
        on_observation=lambda obs, emitting, bar: emitted.append(emitting) if emitting else None, stop=lambda: len(emitted) >= 2, log=logs.append,
    )
    assert len(emitted) == 2 and seen < 60 and any(log.startswith("stopped after") for log in logs)
