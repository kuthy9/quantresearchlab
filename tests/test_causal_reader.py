from __future__ import annotations

from decimal import Decimal, localcontext

import pandas as pd
import pytest

from smc_trader.causal import CausalClockError, CausalMarketReader
from smc_trader.model import (
    Bar,
    Candle,
    Timeframe,
    candle_identity,
    price_to_ticks,
    ticks_to_price,
)

from .helpers import MODEL_SCALE_SPECS, session_bars


@pytest.mark.parametrize("off_grid", (100.125, 100.375, 100.1))
def test_exact_price_grid_rejects_half_ticks_without_reader_mutation(
    off_grid: float,
) -> None:
    reader = CausalMarketReader(
        scale_specs=MODEL_SCALE_SPECS,
        tick_size=0.25,
    )
    reader.on_bar(
        Bar(
            pd.Timestamp("2025-01-06 10:00", tz="America/New_York"),
            100.0,
            100.25,
            99.75,
            100.0,
            10.0,
            "NQH5",
            1,
        )
    )
    before = (
        reader.last_asof,
        reader._contract,
        {
            timeframe: reader.window(timeframe, 100)
            for timeframe in reader.active_timeframes
        },
        {
            timeframe: (
                aggregator._bucket_start,
                aggregator._bucket_end,
                tuple(aggregator._bars),
            )
            for timeframe, aggregator in reader._aggregators.items()
        },
    )
    rejected = Bar(
        pd.Timestamp("2025-01-06 10:07", tz="America/New_York"),
        off_grid,
        off_grid + 0.25,
        off_grid - 0.25,
        off_grid,
        10.0,
        "NQM5",
        2,
        data_gap_before_minutes=6,
    )

    with pytest.raises(ValueError, match="off-grid"):
        reader.on_bar(rejected)

    after = (
        reader.last_asof,
        reader._contract,
        {
            timeframe: reader.window(timeframe, 100)
            for timeframe in reader.active_timeframes
        },
        {
            timeframe: (
                aggregator._bucket_start,
                aggregator._bucket_end,
                tuple(aggregator._bars),
            )
            for timeframe, aggregator in reader._aggregators.items()
        },
    )
    assert after == before


@pytest.mark.parametrize(
    ("tick_size", "coordinates"),
    (
        (0.25, (-401, -1, 0, 1, 401)),
        (0.1, (-1001, -3, 0, 3, 1001)),
    ),
)
def test_exact_price_tick_roundtrip(
    tick_size: float,
    coordinates: tuple[int, ...],
) -> None:
    for coordinate in coordinates:
        price = ticks_to_price(coordinate, tick_size)
        assert price_to_ticks(price, tick_size) == coordinate


@pytest.mark.parametrize(
    ("price", "tick_size"),
    (
        (Decimal("10000000000000000000000000000.1"), Decimal("1")),
        (Decimal("1.00000000000000000000000000001"), Decimal("1")),
    ),
)
def test_exact_price_grid_does_not_depend_on_decimal_context_precision(
    price: Decimal,
    tick_size: Decimal,
) -> None:
    with pytest.raises(ValueError, match="off-grid"):
        price_to_ticks(price, tick_size)

    exact_coordinate = 10**40
    assert price_to_ticks(Decimal(exact_coordinate), tick_size) == (
        exact_coordinate
    )


def test_tick_projection_does_not_depend_on_decimal_context_precision() -> None:
    with localcontext() as context:
        context.prec = 3
        projected = ticks_to_price(123457, Decimal("0.25"))

    assert projected == 30864.25
    assert price_to_ticks(projected, Decimal("0.25")) == 123457


def test_candle_identity_reuses_exact_price_grid() -> None:
    candle = Candle(
        timeframe=Timeframe.M1,
        start=pd.Timestamp("2025-01-06 10:00", tz="America/New_York"),
        end=pd.Timestamp("2025-01-06 10:01", tz="America/New_York"),
        open=100.0,
        high=100.25,
        low=100.0,
        close=100.1,
        volume=10.0,
        symbol="NQH5",
        instrument_id=1,
        observed_minutes=1,
        expected_minutes=1,
        complete=True,
    )
    with pytest.raises(ValueError, match="off-grid"):
        candle_identity(candle, tick_size=0.25)


def test_reader_aggregates_and_exposes_integer_ohlc_ticks() -> None:
    reader = CausalMarketReader(
        scale_specs=MODEL_SCALE_SPECS,
        tick_size=0.25,
    )
    rows = (
        (100.0, 100.25, 99.75, 100.25),
        (100.25, 100.75, 100.0, 100.5),
        (100.5, 101.0, 99.5, 100.75),
        (100.75, 100.75, 100.25, 100.5),
        (100.5, 100.75, 100.0, 100.25),
    )
    update = None
    for index, (open_, high, low, close) in enumerate(rows):
        update = reader.on_bar(
            Bar(
                pd.Timestamp(
                    f"2025-01-06 10:0{index}",
                    tz="America/New_York",
                ),
                open_,
                high,
                low,
                close,
                10.0,
                "NQH5",
                1,
            )
        )
        assert update.completed_1m.normalized_ohlc_ticks == tuple(
            price_to_ticks(value, 0.25)
            for value in (open_, high, low, close)
        )
    assert update is not None
    aggregated = update.newly_completed[Timeframe.M5][0]
    assert aggregated.normalized_ohlc_ticks == (400, 404, 398, 401)
    assert (
        aggregated.open,
        aggregated.high,
        aggregated.low,
        aggregated.close,
    ) == (100.0, 101.0, 99.5, 100.25)


def test_reader_tick_aggregation_ignores_decimal_context_precision() -> None:
    reader = CausalMarketReader(
        scale_specs=MODEL_SCALE_SPECS,
        tick_size=0.25,
    )
    update = None
    with localcontext() as context:
        context.prec = 2
        for index in range(5):
            update = reader.on_bar(
                Bar(
                    pd.Timestamp(
                        f"2025-01-06 10:0{index}",
                        tz="America/New_York",
                    ),
                    30864.25,
                    30864.75 + 0.25 * index,
                    30864.0,
                    30864.5,
                    10.0,
                    "NQH5",
                    1,
                )
            )

    assert update is not None
    aggregated = update.newly_completed[Timeframe.M5][0]
    assert aggregated.normalized_ohlc_ticks == (
        123457,
        123463,
        123456,
        123458,
    )
    assert reader.last_asof == update.asof
    assert len(reader.window(Timeframe.M1, 10)) == 5
    assert len(reader.window(Timeframe.M5, 10)) == 1


def test_reader_rejects_pre_normalized_bar_from_a_different_grid() -> None:
    reader = CausalMarketReader(
        scale_specs=MODEL_SCALE_SPECS,
        tick_size=0.25,
    )
    bar = Bar(
        pd.Timestamp("2025-01-06 10:00", tz="America/New_York"),
        100.0,
        100.5,
        99.5,
        100.0,
        10.0,
        "NQH5",
        1,
        price_tick_size=0.5,
    )

    with pytest.raises(ValueError, match="grid disagrees"):
        reader.on_bar(bar)
    assert reader.last_asof is None
    assert all(
        reader.window(timeframe, 1) == ()
        for timeframe in reader.active_timeframes
    )


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
