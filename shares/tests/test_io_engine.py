from __future__ import annotations

from dataclasses import replace
import pandas as pd
import pytest

from shares.core.engine import ContinuousSMCEngine
from execution.core.execution import TopOfBook, TopOfBookExecutionProvider
from shares.core.io import (
    DataContinuityError,
    build_previous_session_front,
    inspect_source,
    iter_completed_bars,
    load_ohlcv,
)
from shares.core.model import AccountState, Direction, Timeframe
from execution.core.execution import ExecutionRealityInput

from shares.tests.helpers import session_bars


def _raw_contract_rows() -> pd.DataFrame:
    rows = []
    sessions = (
        ("2025-01-05 18:00", (100, 40)),
        ("2025-01-06 18:00", (10, 200)),
        ("2025-01-07 18:00", (50, 50)),
    )
    for start, volumes in sessions:
        for minute in range(2):
            timestamp = pd.Timestamp(start, tz="America/New_York") + pd.Timedelta(minutes=minute)
            for symbol, instrument, volume, price in (
                ("NQH5", 1, volumes[0], 20_000.0),
                ("NQM5", 2, volumes[1], 20_010.0),
            ):
                rows.append(
                    {
                        "ts": timestamp,
                        "open": price,
                        "high": price + 1,
                        "low": price - 1,
                        "close": price + 0.25,
                        "volume": volume,
                        "symbol": symbol,
                        "instrument_id": instrument,
                    }
                )
    return pd.DataFrame(rows)


def test_previous_session_front_never_uses_current_session_winner() -> None:
    bars, roll = build_previous_session_front(_raw_contract_rows())
    first_selected_session = roll.iloc[0]
    second_selected_session = roll.iloc[1]
    assert first_selected_session["symbol"] == "NQH5"
    assert second_selected_session["symbol"] == "NQM5"
    first_session_rows = bars.loc[bars.index.date == pd.Timestamp("2025-01-06").date()]
    assert set(first_session_rows["symbol"]) == {"NQH5"}


def test_previous_session_front_excludes_off_session_volume_before_selection() -> None:
    rows = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp(
                    "2025-01-05 18:00",
                    tz="America/New_York",
                ),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 10.0,
                "symbol": "NQH5",
                "instrument_id": 1,
            },
            {
                "ts": pd.Timestamp(
                    "2025-01-06 17:00",
                    tz="America/New_York",
                ),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000_000.0,
                "symbol": "NQM5",
                "instrument_id": 2,
            },
            {
                "ts": pd.Timestamp(
                    "2025-01-06 18:00",
                    tz="America/New_York",
                ),
                "open": 101.0,
                "high": 102.0,
                "low": 100.0,
                "close": 101.0,
                "volume": 10.0,
                "symbol": "NQH5",
                "instrument_id": 1,
            },
        ]
    )
    bars, selection = build_previous_session_front(rows)
    assert len(selection) == 1
    assert selection.iloc[0]["symbol"] == "NQH5"
    assert set(bars["symbol"]) == {"NQH5"}


def test_large_same_contract_gap_can_be_marked_for_causal_reset() -> None:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-06 10:00", tz="America/New_York"),
            pd.Timestamp("2025-01-06 10:07", tz="America/New_York"),
        ],
        name="ts",
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [100.25, 101.25],
            "low": [99.75, 100.75],
            "close": [100.0, 101.0],
            "volume": [10.0, 10.0],
            "symbol": ["NQH5", "NQH5"],
            "instrument_id": [1, 1],
        },
        index=index,
    )
    bars = list(
        iter_completed_bars(
            frame,
            allow_data_gap_reset=True,
        )
    )
    assert len(bars) == 2
    assert bars[1].data_gap_before_minutes == 6
    assert not bars[1].synthetic_no_trade


def test_february_2019_same_contract_gap_is_one_184_minute_reset() -> None:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2019-02-26 19:40", tz="America/New_York"),
            pd.Timestamp("2019-02-26 22:45", tz="America/New_York"),
        ],
        name="ts",
    )
    frame = pd.DataFrame(
        {
            "open": [7_100.0, 7_101.0],
            "high": [7_100.25, 7_101.25],
            "low": [7_099.75, 7_100.75],
            "close": [7_100.0, 7_101.0],
            "volume": [10.0, 10.0],
            "symbol": ["NQH9", "NQH9"],
            "instrument_id": [15657, 15657],
        },
        index=index,
    )

    bars = list(iter_completed_bars(frame, allow_data_gap_reset=True))

    assert len(bars) == 2
    assert bars[1].start == pd.Timestamp(
        "2019-02-26 22:45", tz="America/New_York"
    )
    assert bars[1].data_gap_before_minutes == 184
    assert not bars[1].synthetic_no_trade


def test_large_same_contract_gap_remains_fail_closed_by_default() -> None:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-06 10:00", tz="America/New_York"),
            pd.Timestamp("2025-01-06 10:07", tz="America/New_York"),
        ],
        name="ts",
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [100.25, 101.25],
            "low": [99.75, 100.75],
            "close": [100.0, 101.0],
            "volume": [10.0, 10.0],
            "symbol": ["NQH5", "NQH5"],
            "instrument_id": [1, 1],
        },
        index=index,
    )

    with pytest.raises(DataContinuityError, match="cap=5, same_contract=True"):
        list(iter_completed_bars(frame))


def test_data_gap_reset_exception_never_crosses_contracts() -> None:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-01-06 10:00", tz="America/New_York"),
            pd.Timestamp("2025-01-06 10:07", tz="America/New_York"),
        ],
        name="ts",
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [100.25, 101.25],
            "low": [99.75, 100.75],
            "close": [100.0, 101.0],
            "volume": [10.0, 10.0],
            "symbol": ["NQH5", "NQM5"],
            "instrument_id": [1, 2],
        },
        index=index,
    )

    with pytest.raises(DataContinuityError, match="same_contract=False"):
        list(iter_completed_bars(frame, allow_data_gap_reset=True))


def test_engine_runs_all_layers_once_per_completed_minute() -> None:
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    snapshot = None
    for bar in session_bars(1)[:300]:
        open_price = round(bar.open / 0.25) * 0.25
        close = round(bar.close / 0.25) * 0.25
        bar = replace(
            bar,
            open=open_price,
            high=max(open_price, close) + 0.5,
            low=min(open_price, close) - 0.5,
            close=close,
        )
        snapshot = engine.on_bar(
            bar,
            execution=ExecutionRealityInput(
                spread_points=0.25,
                deadline=bar.end + pd.Timedelta(minutes=90),
                size_available=10,
            ),
            account=AccountState(100_000.0),
        )
    assert snapshot is not None
    assert len(snapshot.belief.hypotheses) == 6
    assert engine.last_snapshot is snapshot
    assert all(
        frame.cutoff <= snapshot.observation.asof
        for frame in snapshot.observation.frames.values()
    )
    expected_timeframes = (
        Timeframe.H4,
        Timeframe.H1,
        Timeframe.M15,
        Timeframe.M5,
        Timeframe.M1,
    )
    assert engine.reader.active_timeframes == expected_timeframes
    assert engine.observer._active_timeframes == expected_timeframes
    assert snapshot.observation.active_timeframes == expected_timeframes
    assert tuple(snapshot.observation.frames) == expected_timeframes
    assert (
        engine.reader.scale_registry_id
        == engine.observer._scale_registry_id
        == snapshot.observation.scale_registry_id
    )
    assert (
        snapshot.observation.frame(Timeframe.M15).cutoff
        <= snapshot.observation.asof
    )
    assert snapshot.risk.final_action.value in {
        "enter", "wait", "hold", "protect", "exit", "abstain"
    }


def test_top_of_book_cannot_be_later_than_decision_clock() -> None:
    clock = pd.Timestamp("2025-01-06 10:00", tz="America/New_York")
    book = TopOfBook(clock + pd.Timedelta(seconds=1), 20_000.0, 20_000.25, 10, 10)
    with pytest.raises(ValueError):
        TopOfBookExecutionProvider().observe(
            book,
            decision_clock=clock,
            deadline=clock + pd.Timedelta(minutes=30),
            direction=Direction.LONG,
        )


def test_source_materialization_check_is_python39_compatible(tmp_path) -> None:
    source = tmp_path / "bars.parquet"
    source.write_bytes(b"materialized")
    status = inspect_source(source)
    assert status.exists and status.materialized and not status.dataless


def test_parquet_window_is_pushed_into_source_read(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "causal_front.parquet"
    timestamps = pd.date_range(
        "2025-01-06 09:29",
        periods=4,
        freq="1min",
        tz="America/New_York",
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1, 2, 3, 4],
            "symbol": ["NQH5"] * 4,
            "instrument_id": [1] * 4,
        },
        index=timestamps.rename("ts"),
    )
    frame.to_parquet(source)
    original = pd.read_parquet
    captured = {}

    def recording_read_parquet(path, **kwargs):
        captured.update(kwargs)
        return original(path, **kwargs)

    monkeypatch.setattr(
        "shares.core.io.pd.read_parquet",
        recording_read_parquet,
    )
    loaded = load_ohlcv(
        source,
        start=timestamps[1],
        end=timestamps[3],
    )
    assert list(loaded.frame.index) == list(timestamps[1:3])
    assert captured["filters"] == [
        ("ts", ">=", timestamps[1]),
        ("ts", "<", timestamps[3]),
    ]


def test_engine_surfaces_missing_execution_deadline_to_the_risk_layer() -> None:
    """Execution scoring is Engine-owned, and its anomalies still reach risk."""

    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )

    snapshot = engine.on_bar(session_bars(1)[0])

    assert "deadline_missing" in snapshot.observation.anomalies
    assert snapshot.observation.execution.source == "assumed_default"


def test_invalid_execution_reality_fails_before_the_eye_is_entered() -> None:
    """A bad reality input is rejected while the Eye is still untouched."""

    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )

    with pytest.raises(ValueError, match="spread cannot be negative"):
        engine.on_bar(
            session_bars(1)[0],
            execution=ExecutionRealityInput(spread_points=-0.25),
        )

    observer = engine.observer
    assert observer._prior is None
    assert observer.memory.last_minute_end is None
    assert observer.memory.clock_coverage_start is None


def test_engine_labels_an_unobserved_execution_as_an_assumed_model() -> None:
    """With no broker/feed input there is nothing observed, only assumed."""

    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )

    snapshot = engine.on_bar(session_bars(1)[0])
    execution = snapshot.observation.execution

    assert execution.source == "assumed_default"
    assert "execution_assumed_default_model" in execution.anomalies


def test_engine_labels_a_supplied_execution_as_observed() -> None:
    """A real input keeps the caller's own provenance label."""

    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )

    snapshot = engine.on_bar(
        session_bars(1)[0],
        execution=ExecutionRealityInput(
            spread_points=0.25,
            source="top_of_book",
            size_available=10.0,
        ),
    )
    execution = snapshot.observation.execution

    assert execution.source == "top_of_book"
    assert "execution_assumed_default_model" not in execution.anomalies
