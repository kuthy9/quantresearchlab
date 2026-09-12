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
    save_block(dataset, out, emit_end=None)
    return out.parent


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
