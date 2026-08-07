from __future__ import annotations

import pandas as pd
import pytest

from smc_trader.causal import CausalClockError, CausalMarketReader
from smc_trader.model import Bar, Candle, Timeframe

from .helpers import MODEL_SCALE_SPECS, session_bars


def test_reader_emits_only_completed_higher_timeframe_bars() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    update = None
    for bar in session_bars(1)[:300]:
        update = reader.on_bar(bar)
    assert update is not None
    assert len(update.histories[Timeframe.M5]) == 60
    assert len(update.histories[Timeframe.H1]) == 5
    assert len(update.histories[Timeframe.H4]) == 1
    for candles in update.histories.values():
        assert all(candle.complete and candle.end <= update.asof for candle in candles)


def test_duplicate_and_unregistered_gap_fail_closed() -> None:
    first, second, third = session_bars(1)[:3]
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    reader.on_bar(first)
    with pytest.raises(CausalClockError):
        reader.on_bar(first)
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    reader.on_bar(first)
    with pytest.raises(CausalClockError):
        reader.on_bar(third)


def test_explicit_data_gap_resets_histories_and_emits_hard_anomaly() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    first = Bar(
        pd.Timestamp("2025-01-06 10:00", tz="America/New_York"),
        100.0,
        100.25,
        99.75,
        100.0,
        10,
        "NQH5",
        1,
    )
    reader.on_bar(first)
    later = Bar(
        pd.Timestamp("2025-01-06 10:07", tz="America/New_York"),
        101.0,
        101.25,
        100.75,
        101.0,
        10,
        "NQH5",
        1,
        data_gap_before_minutes=6,
    )
    update = reader.on_bar(later)
    assert "data_gap_history_reset" in update.anomalies
    assert len(update.histories[Timeframe.M1]) == 1
    assert not update.histories[Timeframe.M5]
    assert not update.histories[Timeframe.H1]
    assert not update.histories[Timeframe.H4]


def test_scheduled_maintenance_gap_is_explicitly_allowed() -> None:
    bars = session_bars(2)
    close_bar = next(bar for bar in bars if bar.start.strftime("%Y-%m-%d %H:%M") == "2025-01-06 16:59")
    next_open = next(bar for bar in bars if bar.start.strftime("%Y-%m-%d %H:%M") == "2025-01-06 18:00")
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    reader.on_bar(close_bar)
    update = reader.on_bar(next_open)
    assert "scheduled_market_closure" in update.anomalies


def test_scheduled_weekend_and_registered_full_session_closures_are_exact() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    friday_close = Bar(
        pd.Timestamp("2021-01-08 16:59", tz="America/New_York"),
        100.0,
        100.25,
        99.75,
        100.0,
        10,
        "NQH1",
        1,
    )
    sunday_open = Bar(
        pd.Timestamp("2021-01-10 18:00", tz="America/New_York"),
        100.0,
        100.25,
        99.75,
        100.0,
        10,
        "NQH1",
        1,
    )
    reader.on_bar(friday_close)
    update = reader.on_bar(sunday_open)
    assert "scheduled_weekend_closure" in update.anomalies

    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    new_year_close = Bar(
        pd.Timestamp("2020-12-31 16:59", tz="America/New_York"),
        100.0,
        100.25,
        99.75,
        100.0,
        10,
        "NQH1",
        1,
    )
    new_year_open = Bar(
        pd.Timestamp("2021-01-03 18:00", tz="America/New_York"),
        100.0,
        100.25,
        99.75,
        100.0,
        10,
        "NQH1",
        1,
    )
    reader.on_bar(new_year_close)
    update = reader.on_bar(new_year_open)
    assert "registered_full_session_closure" in update.anomalies


def test_arbitrary_multi_day_maintenance_shaped_gap_fails_closed() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    reader.on_bar(
        Bar(
            pd.Timestamp("2021-01-04 16:59", tz="America/New_York"),
            100.0,
            100.25,
            99.75,
            100.0,
            10,
            "NQH1",
            1,
        )
    )
    with pytest.raises(CausalClockError):
        reader.on_bar(
            Bar(
                pd.Timestamp("2021-01-06 18:00", tz="America/New_York"),
                100.0,
                100.25,
                99.75,
                100.0,
                10,
                "NQH1",
                1,
            )
        )


def test_historical_settlement_pause_keeps_h1_bucket_complete() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    timestamps = list(
        pd.date_range(
            "2021-01-05 16:00",
            "2021-01-05 16:14",
            freq="min",
            tz="America/New_York",
        )
    ) + list(
        pd.date_range(
            "2021-01-05 16:30",
            "2021-01-05 16:59",
            freq="min",
            tz="America/New_York",
        )
    )
    update = None
    for index, timestamp in enumerate(timestamps):
        update = reader.on_bar(
            Bar(timestamp, 100.0, 100.25, 99.75, 100.0, 10, "NQH1", 1)
        )
        if index == 15:
            assert "historical_settlement_pause" in update.anomalies
    assert update is not None
    candle = update.histories[Timeframe.H1][-1]
    assert candle.complete
    assert candle.observed_minutes == 45
    assert candle.expected_minutes == 45


def test_unregistered_post_cutoff_settlement_gap_fails_closed() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    first = Bar(
        pd.Timestamp("2021-06-28 16:14", tz="America/New_York"),
        100.0,
        100.25,
        99.75,
        100.0,
        10,
        "NQU1",
        1,
    )
    second = Bar(
        pd.Timestamp("2021-06-28 16:30", tz="America/New_York"),
        100.0,
        100.25,
        99.75,
        100.0,
        10,
        "NQU1",
        1,
    )
    reader.on_bar(first)
    with pytest.raises(CausalClockError):
        reader.on_bar(second)


def test_registered_special_close_emits_complete_shortened_h4() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    update = None
    for timestamp in pd.date_range(
        "2021-01-18 10:00",
        "2021-01-18 12:59",
        freq="min",
        tz="America/New_York",
    ):
        update = reader.on_bar(
            Bar(timestamp, 100.0, 100.25, 99.75, 100.0, 10, "NQH1", 1)
        )
    assert update is not None
    candle = update.histories[Timeframe.H4][-1]
    assert candle.complete
    assert candle.end.hour == 13
    assert candle.observed_minutes == 180
    reopened = reader.on_bar(
        Bar(
            pd.Timestamp("2021-01-18 18:00", tz="America/New_York"),
            100.0,
            100.25,
            99.75,
            100.0,
            10,
            "NQH1",
            1,
        )
    )
    assert "registered_special_session_closure" in reopened.anomalies


def test_post_thanksgiving_close_emits_complete_final_partial_hour() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    update = None
    for timestamp in pd.date_range(
        "2021-11-26 12:00",
        "2021-11-26 13:14",
        freq="min",
        tz="America/New_York",
    ):
        update = reader.on_bar(
            Bar(timestamp, 100.0, 100.25, 99.75, 100.0, 10, "NQZ1", 1)
        )
    assert update is not None
    candle = update.histories[Timeframe.H1][-1]
    assert candle.complete
    assert candle.start == pd.Timestamp(
        "2021-11-26 13:00",
        tz="America/New_York",
    )
    assert candle.end == pd.Timestamp(
        "2021-11-26 13:15",
        tz="America/New_York",
    )
    assert candle.observed_minutes == 15
    assert candle.expected_minutes == 15


@pytest.mark.parametrize(
    "session_date",
    (
        "2018-12-24",
        "2019-12-24",
        "2020-12-24",
        "2024-12-24",
        "2025-12-24",
    ),
)
def test_christmas_eve_nq_session_runs_through_1315(
    session_date: str,
) -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    update = None
    for timestamp in pd.date_range(
        f"{session_date} 13:00",
        f"{session_date} 13:14",
        freq="min",
        tz="America/New_York",
    ):
        update = reader.on_bar(
            Bar(timestamp, 100.0, 100.25, 99.75, 100.0, 10, "NQ", 1)
        )
    assert update is not None
    candle = update.histories[Timeframe.H1][-1]
    assert candle.start.hour == 13
    assert candle.start.minute == 0
    assert candle.end.hour == 13
    assert candle.end.minute == 15
    assert candle.observed_minutes == 15
    assert candle.expected_minutes == 15


@pytest.mark.parametrize("session_date", ("2018-12-05", "2025-01-09"))
def test_national_day_of_mourning_nq_session_runs_through_0930(
    session_date: str,
) -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    update = None
    for timestamp in pd.date_range(
        f"{session_date} 09:00",
        f"{session_date} 09:29",
        freq="min",
        tz="America/New_York",
    ):
        update = reader.on_bar(
            Bar(timestamp, 100.0, 100.25, 99.75, 100.0, 10, "NQ", 1)
        )
    assert update is not None
    candle = update.histories[Timeframe.H1][-1]
    assert candle.end.hour == 9
    assert candle.end.minute == 30
    assert candle.observed_minutes == 30
    assert candle.expected_minutes == 30


def test_contract_change_resets_multitimeframe_history() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    bars = session_bars(1)
    for bar in bars[:60]:
        reader.on_bar(bar)
    source = bars[60]
    changed = Bar(
        source.start,
        source.open,
        source.high,
        source.low,
        source.close,
        source.volume,
        "NQM5",
        2,
    )
    update = reader.on_bar(changed)
    assert "contract_change_history_reset" in update.anomalies
    assert len(update.histories[Timeframe.H1]) == 0


def test_real_and_synthetic_minute_provenance_reaches_higher_timeframes() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    update = None
    for index in range(5):
        update = reader.on_bar(
            Bar(
                start=pd.Timestamp(
                    f"2025-01-06 10:0{index}",
                    tz="America/New_York",
                ),
                open=100.0,
                high=100.25,
                low=99.75,
                close=100.0,
                volume=0.0 if index == 2 else 10.0,
                symbol="NQH5",
                instrument_id=1,
                synthetic_no_trade=index == 2,
            )
        )
        minute = update.completed_1m
        assert minute.synthetic_minutes == (1 if index == 2 else 0)
        assert minute.real_minutes == (0 if index == 2 else 1)
        assert minute.real_completed is (index != 2)
    assert update is not None
    five_minute = update.histories[Timeframe.M5][-1]
    assert five_minute.complete
    assert five_minute.observed_minutes == 5
    assert five_minute.real_minutes == 4
    assert five_minute.synthetic_minutes == 1
    assert not five_minute.real_completed


def test_candle_rejects_inconsistent_provenance_coverage() -> None:
    with pytest.raises(ValueError, match="provenance"):
        Candle(
            timeframe=Timeframe.M1,
            start=pd.Timestamp(
                "2025-01-06 10:00",
                tz="America/New_York",
            ),
            end=pd.Timestamp(
                "2025-01-06 10:01",
                tz="America/New_York",
            ),
            open=100.0,
            high=100.25,
            low=99.75,
            close=100.0,
            volume=10.0,
            symbol="NQH5",
            instrument_id=1,
            observed_minutes=1,
            expected_minutes=1,
            complete=True,
            real_minutes=1,
            synthetic_minutes=1,
        )
    with pytest.raises(ValueError, match="coverage"):
        Candle(
            timeframe=Timeframe.M5,
            start=pd.Timestamp(
                "2025-01-06 10:00",
                tz="America/New_York",
            ),
            end=pd.Timestamp(
                "2025-01-06 10:05",
                tz="America/New_York",
            ),
            open=100.0,
            high=100.25,
            low=99.75,
            close=100.0,
            volume=10.0,
            symbol="NQH5",
            instrument_id=1,
            observed_minutes=4,
            expected_minutes=5,
            complete=True,
            real_minutes=4,
            synthetic_minutes=0,
        )
