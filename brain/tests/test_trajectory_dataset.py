"""The sixty-minute future of an observation point starts at the first bar
the Eye had not yet seen at ``asof``."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from brain.research.trajectory_dataset import FUTURE_HORIZON_MINUTES, build_dataset
from shares.tests.helpers import session_bars, write_synthetic_ohlcv

ROOT = Path(__file__).resolve().parents[2]


def test_the_future_window_is_the_sixty_bars_from_asof(tmp_path) -> None:
    # Every close is unique, so a one-row misalignment is a value mismatch,
    # not a coincidence. Bars open at the prior close and overlap widely so
    # the ramp does not flood the Eye's Group-3 zone capacity.
    bars: list = []
    previous = 20_000.0
    for k, bar in enumerate(session_bars(3)):
        close = 20_000.0 + 0.25 * k
        bars.append(
            replace(
                bar, open=previous, close=close,
                high=max(previous, close) + 2.0, low=min(previous, close) - 2.0,
            )
        )
        previous = close
    source = write_synthetic_ohlcv(bars, tmp_path / "synthetic.parquet")
    dataset = build_dataset(
        source=source, warmup_start="2025-01-05", emit_start="2025-01-07", end="2025-01-09",
        model_path=ROOT / "configs" / "model.json", root=ROOT,
    )
    tape = pd.DataFrame(
        {"close": [b.close for b in bars], "high": [b.high for b in bars], "low": [b.low for b in bars]},
        index=pd.DatetimeIndex([b.start for b in bars]).tz_convert("UTC"),
    )
    minute = pd.Timedelta(minutes=1)
    assert len(dataset) > 0
    for n in range(len(dataset)):
        asof = dataset.index[n].tz_convert("UTC")
        # ``asof`` is the end of the bar just completed, i.e. the tape row
        # before it; the row *at* ``asof`` is the first bar not yet seen.
        assert dataset.prices["close"].iloc[n] == tape.loc[asof - minute, "close"]
        assert dataset.future_closes[n][0] == tape.loc[asof, "close"]
        window = tape.loc[asof : asof + (FUTURE_HORIZON_MINUTES - 1) * minute]
        assert len(window) == FUTURE_HORIZON_MINUTES
        np.testing.assert_array_equal(dataset.future_closes[n], window["close"].to_numpy())
        np.testing.assert_array_equal(dataset.future_highs[n], window["high"].to_numpy())
        np.testing.assert_array_equal(dataset.future_lows[n], window["low"].to_numpy())
