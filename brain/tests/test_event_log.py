"""The event log records every transition event, and nothing else, at the
clock it was published."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brain.research.event_log import (
    EVENT_COLUMNS,
    TRANSITION_KINDS,
    is_transition,
    load_blocks,
    save_block,
)
from brain.research.trajectory_dataset import build_dataset
from contract.eye import EventKind
from shares.tests.helpers import session_bars, write_synthetic_ohlcv

ROOT = Path(__file__).resolve().parents[2]


def test_transition_kinds_exclude_state_republication_and_heartbeats() -> None:
    assert len(TRANSITION_KINDS) == 49
    assert "bar_completed" not in TRANSITION_KINDS
    assert "market_epoch_reset" not in TRANSITION_KINDS
    assert not any(kind.endswith("_state") for kind in TRANSITION_KINDS)
    assert is_transition(EventKind.SWEEP_CONFIRMED)
    assert not is_transition(EventKind.BAR_COMPLETED)


@pytest.fixture(scope="module")
def synthetic_block(tmp_path_factory) -> Path:
    source = write_synthetic_ohlcv(session_bars(3), tmp_path_factory.mktemp("tape") / "synthetic.parquet")
    dataset = build_dataset(
        source=source,
        warmup_start="2025-01-05",
        emit_start="2025-01-07",
        end="2025-01-09",
        model_path=ROOT / "configs" / "model.json",
        root=ROOT,
    )
    out = tmp_path_factory.mktemp("blocks") / "2025-01-06"
    # emit_end past the last bar: nothing is cut, but the cut path runs.
    save_block(dataset, out, emit_end="2025-01-12T18:00")
    return out.parent


def test_emit_end_drops_clocks_and_events_at_or_after_it(tmp_path) -> None:
    source = write_synthetic_ohlcv(session_bars(3), tmp_path / "synthetic.parquet")
    dataset = build_dataset(
        source=source, warmup_start="2025-01-05", emit_start="2025-01-07", end="2025-01-09",
        model_path=ROOT / "configs" / "model.json", root=ROOT,
    )
    save_block(dataset, tmp_path / "block", emit_end="2025-01-08T12:00")
    data = load_blocks(tmp_path)
    limit = pd.Timestamp("2025-01-08T12:00", tz="America/New_York")
    assert (pd.DatetimeIndex(data["index"]) < limit).all()
    assert (data["events"]["known_at"] < limit).all()
    assert len(data["index"]) < len(dataset.index)


def test_the_log_has_one_row_per_transition_event_at_its_known_at(synthetic_block) -> None:
    data = load_blocks(synthetic_block)
    events = data["events"]
    assert list(events.columns) == list(EVENT_COLUMNS)
    assert len(events) > 0
    assert set(events["kind"]).issubset(set(TRANSITION_KINDS))
    assert events["known_at"].is_monotonic_increasing
    # Every event is stamped at the close of a completed bar inside the emit
    # window. The log is not restricted to sampled clocks: a clock the dataset
    # drops (no complete future, the session close) still publishes events,
    # and they are history for every later clock.
    closes = pd.DatetimeIndex([bar.start + pd.Timedelta(minutes=1) for bar in session_bars(3)]).tz_convert("UTC")
    known = events["known_at"].dt.tz_convert("UTC")
    assert known.isin(closes).all()
    assert (known >= pd.Timestamp("2025-01-07", tz="America/New_York")).all()
    clocks = pd.DatetimeIndex(data["index"]).tz_convert("UTC")
    assert known.isin(clocks).mean() > 0.9


def test_blocks_concatenate_in_order(synthetic_block) -> None:
    data = load_blocks(synthetic_block)
    assert data["features"].shape[0] == data["index"].shape[0] == data["prices"].shape[0]
    assert np.all(np.diff(pd.DatetimeIndex(data["index"]).asi8) > 0)


def test_a_clock_whose_future_crosses_a_session_gap_is_not_sampled(synthetic_block) -> None:
    """The last hour before a session close has no sixty consecutive traded
    minutes ahead of it, so it carries events but no observation point."""

    data = load_blocks(synthetic_block)
    local = pd.DatetimeIndex(data["index"]).tz_convert("America/New_York")
    minute_of_day = local.hour * 60 + local.minute
    # session_bars closes each synthetic day at 17:00. The dataset's future
    # window is the sixty rows after the row at ``asof`` (closes at asof+2 ..
    # asof+61, a one-minute offset the research path has always carried), so
    # the last clock with a complete traded future is 15:59.
    assert not ((minute_of_day >= 16 * 60) & (minute_of_day <= 17 * 60)).any()
    assert (minute_of_day == 15 * 60 + 59).any()
    assert (minute_of_day >= 18 * 60).any()  # the evening half of the session is sampled


def test_blocks_carry_a_path_log_cut_at_emit_end(tmp_path) -> None:
    from brain.research.path_log import PATH_COLUMNS

    source = write_synthetic_ohlcv(session_bars(3), tmp_path / "synthetic.parquet")
    dataset = build_dataset(
        source=source, warmup_start="2025-01-05", emit_start="2025-01-07", end="2025-01-09",
        model_path=ROOT / "configs" / "model.json", root=ROOT, record_paths=True,
    )
    assert list(dataset.paths.columns) == list(PATH_COLUMNS)
    save_block(dataset, tmp_path / "block", emit_end="2025-01-08T12:00")
    assert (tmp_path / "block" / "paths.parquet").exists()
    data = load_blocks(tmp_path)
    assert list(data["paths"].columns) == list(PATH_COLUMNS)
    limit = pd.Timestamp("2025-01-08T12:00", tz="America/New_York")
    if len(data["paths"]):
        assert (data["paths"]["known_at"] < limit).all()
        assert data["paths"]["known_at"].dt.tz is not None


def test_a_populated_path_log_round_trips_through_the_block_cache(tmp_path) -> None:
    """The synthetic tape forms no Group-5 context, so the recorder's output is
    built by hand here: mixed None / Timestamp columns, JSON strings and
    bools must survive parquet, the emit_end cut and the sort."""
    from dataclasses import replace
    from types import SimpleNamespace

    from brain.research.path_log import PATH_COLUMNS, path_rows
    from brain.core.hypothesis_proposer import FEATURE_NAMES
    from contract.market import Direction

    source = write_synthetic_ohlcv(session_bars(3), tmp_path / "synthetic.parquet")
    dataset = build_dataset(
        source=source, warmup_start="2025-01-05", emit_start="2025-01-07", end="2025-01-09",
        model_path=ROOT / "configs" / "model.json", root=ROOT,
    )
    rows = []
    for i, at in enumerate(("2025-01-08T10:00", "2025-01-08T11:30", "2025-01-08T13:00")):
        asof = pd.Timestamp(at, tz="America/New_York").tz_convert("UTC")
        step = SimpleNamespace(step_id=f"s{i}:0", kind="zone_visible", reason="typed_entry_zone_registered",
                               strength=0.5, observed_at=asof)
        path = SimpleNamespace(sequence_id=f"s{i}", context_kind="zone_return", context_id=f"loc-{i}",
                               direction=Direction.LONG, formed_at=asof, lifecycle="active", steps=(step,))
        location = SimpleNamespace(location_id=f"loc-{i}", lower_bound=99.0, upper_bound=99.8, near_edge=99.8,
                                   far_edge=99.0, failure_boundary=98.7, source_zone_kind="fvg", entry_mode="touch",
                                   first_penetration_fraction=0.25, nearest_visible_draw_distance_points=3.0)
        update = SimpleNamespace(milestone_transitions=((f"s{i}", step),), interaction_paths=(path,),
                                 interaction_path_transitions=(), zone_interactions=(location,) if i else (),
                                 reacceptance_interactions=())
        observation = SimpleNamespace(market_snapshot=SimpleNamespace(asof=asof, timeframe_states={}),
                                      interaction_update=update, manipulations=())
        rows += path_rows(observation, close=100.0, high=100.5, low=99.5, atr=2.0,
                          history=[100.0] * 70, features=tuple(float(j) for j in range(len(FEATURE_NAMES))))
    frame = pd.DataFrame(rows, columns=list(PATH_COLUMNS))
    assert len(frame) == 3 and frame["context_found"].tolist() == [False, True, True]
    save_block(replace(dataset, paths=frame), tmp_path / "block", emit_end="2025-01-08T12:00")
    data = load_blocks(tmp_path)
    paths = data["paths"]
    assert list(paths.columns) == list(PATH_COLUMNS)
    assert len(paths) == 2  # the 13:00 step is at or after emit_end
    assert paths["known_at"].dt.tz is not None and paths["known_at"].is_monotonic_increasing
    assert paths["context_found"].tolist() == [False, True]
    assert paths["steps_so_far"].iloc[0] == '[["zone_visible", "typed_entry_zone_registered", 0.5]]'
    assert paths["step_ordinal"].tolist() == [0, 0] and paths["source_zone_kind"].tolist() == [None, "fvg"]
    assert np.isnan(paths["failure_boundary"].iloc[0]) and paths["failure_boundary"].iloc[1] == 98.7


def test_a_block_without_a_path_log_still_loads(tmp_path) -> None:
    source = write_synthetic_ohlcv(session_bars(3), tmp_path / "synthetic.parquet")
    dataset = build_dataset(
        source=source, warmup_start="2025-01-05", emit_start="2025-01-07", end="2025-01-09",
        model_path=ROOT / "configs" / "model.json", root=ROOT,
    )
    save_block(dataset, tmp_path / "block", emit_end=None)
    (tmp_path / "block" / "paths.parquet").unlink()
    data = load_blocks(tmp_path)
    assert len(data["paths"]) == 0
