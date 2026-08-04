"""CME equity-index session clock for the registered 2017-2026 sample."""
from __future__ import annotations

from datetime import date
from functools import lru_cache

import pandas as pd


MARKET_TIMEZONE = "America/New_York"
SETTLEMENT_PAUSE_END_EXCLUSIVE = date(2021, 6, 28)
CLOCK_START = pd.Timestamp("2016-12-01")
CLOCK_END = pd.Timestamp("2027-01-31")

# Good Friday is normally absent from the generic CME session calendar.
# Equity index futures opened for an abbreviated release session on these
# Employment Situation Fridays and closed at 08:15 CT / 09:15 ET.
ABBREVIATED_GOOD_FRIDAY_CLOSES = {
    date(2021, 4, 2): (9, 15),
    date(2023, 4, 7): (9, 15),
    date(2026, 4, 3): (9, 15),
}

# exchange_calendars 4.11 predates the finalized 2026 equity-index
# Juneteenth hours present in this dataset.
EQUITY_INDEX_CLOSE_OVERRIDES = {
    **ABBREVIATED_GOOD_FRIDAY_CLOSES,
    date(2017, 7, 3): (13, 15),
    date(2017, 11, 24): (13, 15),
    date(2018, 7, 3): (13, 15),
    date(2018, 11, 23): (13, 15),
    date(2018, 12, 5): (9, 30),
    date(2018, 12, 24): (13, 15),
    date(2019, 7, 3): (13, 15),
    date(2019, 11, 29): (13, 15),
    date(2019, 12, 24): (13, 15),
    date(2020, 11, 27): (13, 15),
    date(2020, 12, 24): (13, 15),
    date(2021, 11, 26): (13, 15),
    date(2022, 6, 20): (13, 0),
    date(2022, 11, 25): (13, 15),
    date(2023, 6, 19): (13, 0),
    date(2023, 7, 3): (13, 15),
    date(2023, 11, 24): (13, 15),
    date(2024, 6, 19): (13, 0),
    date(2024, 7, 3): (13, 15),
    date(2024, 11, 29): (13, 15),
    date(2024, 12, 24): (13, 15),
    date(2025, 1, 9): (9, 30),
    date(2025, 6, 19): (13, 0),
    date(2025, 7, 3): (13, 15),
    date(2025, 11, 28): (13, 15),
    date(2025, 12, 24): (13, 15),
    date(2026, 6, 19): (13, 0),
    date(2026, 11, 27): (13, 15),
    date(2026, 12, 24): (13, 15),
}


@lru_cache(maxsize=1)
def _session_schedule() -> pd.DataFrame:
    try:
        import exchange_calendars as xcals
    except ImportError as error:  # pragma: no cover - declared dependency
        raise RuntimeError(
            "the registered CME clock requires exchange-calendars==4.11"
        ) from error
    calendar = xcals.get_calendar("CMES")
    schedule = calendar.schedule.loc[CLOCK_START:CLOCK_END].copy()
    if schedule.empty:
        raise RuntimeError("registered CME calendar returned no sessions")
    return schedule


def _session_position(label: pd.Timestamp) -> int | None:
    index = _session_schedule().index
    try:
        return int(index.get_loc(pd.Timestamp(label).tz_localize(None).normalize()))
    except KeyError:
        return None


def _consecutive_registered_sessions(
    left_label: pd.Timestamp,
    right_label: pd.Timestamp,
) -> bool:
    left_position = _session_position(left_label)
    right_position = _session_position(right_label)
    return bool(
        left_position is not None
        and right_position is not None
        and right_position == left_position + 1
    )


@lru_cache(maxsize=4096)
def _special_session_close_for_date(
    label_date: date,
) -> pd.Timestamp | None:
    """Return the frozen close shared by every minute of a local date."""

    explicit = EQUITY_INDEX_CLOSE_OVERRIDES.get(label_date)
    if explicit is not None:
        return pd.Timestamp(
            year=label_date.year,
            month=label_date.month,
            day=label_date.day,
            hour=explicit[0],
            minute=explicit[1],
            tz=MARKET_TIMEZONE,
        )
    label = pd.Timestamp(label_date)
    schedule = _session_schedule()
    if label not in schedule.index:
        return None
    close = pd.Timestamp(schedule.loc[label, "close"]).tz_convert(MARKET_TIMEZONE)
    # CMES represents an ordinary session through the following 18:00 open.
    # The NQ reader separately registers its actual 17:00-18:00 maintenance
    # closure; only a close earlier than 17:00 is a holiday exception here.
    if (close.hour, close.minute) >= (17, 0):
        return None
    return close


@lru_cache(maxsize=4096)
def special_session_close(timestamp: pd.Timestamp) -> pd.Timestamp | None:
    """Return a registered nonstandard equity-index close on the local date."""

    local = pd.Timestamp(timestamp).tz_convert(MARKET_TIMEZONE)
    return _special_session_close_for_date(local.date())


@lru_cache(maxsize=4096)
def _session_bounds(label: date) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    explicit = EQUITY_INDEX_CLOSE_OVERRIDES.get(label)
    schedule = _session_schedule()
    key = pd.Timestamp(label)
    if key in schedule.index:
        row = schedule.loc[key]
        opened = pd.Timestamp(row["open"]).tz_convert(MARKET_TIMEZONE)
        close = pd.Timestamp(row["close"]).tz_convert(MARKET_TIMEZONE)
        if explicit is not None:
            close = pd.Timestamp(
                year=label.year,
                month=label.month,
                day=label.day,
                hour=explicit[0],
                minute=explicit[1],
                tz=MARKET_TIMEZONE,
            )
        elif (close.hour, close.minute) >= (17, 0):
            close = pd.Timestamp(
                year=label.year,
                month=label.month,
                day=label.day,
                hour=17,
                tz=MARKET_TIMEZONE,
            )
        return opened, close
    if label in ABBREVIATED_GOOD_FRIDAY_CLOSES:
        close_hour, close_minute = ABBREVIATED_GOOD_FRIDAY_CLOSES[label]
        close = pd.Timestamp(
            year=label.year,
            month=label.month,
            day=label.day,
            hour=close_hour,
            minute=close_minute,
            tz=MARKET_TIMEZONE,
        )
        opened = (
            pd.Timestamp(label)
            - pd.Timedelta(days=1)
            + pd.Timedelta(hours=18)
        ).tz_localize(MARKET_TIMEZONE)
        return opened, close
    return None


def is_registered_trading_minute(timestamp: pd.Timestamp) -> bool:
    """Return whether an OHLCV interval start is inside registered NQ hours."""

    # Round on the UTC timeline before converting to local time.  Flooring a
    # timezone-aware local timestamp during the fall-back hour asks pandas to
    # infer an ambiguous wall-clock offset and can raise despite an unambiguous
    # source instant.
    local = (
        pd.Timestamp(timestamp)
        .tz_convert("UTC")
        .floor("min")
        .tz_convert(MARKET_TIMEZONE)
    )
    label = (
        (pd.Timestamp(local.date()) + pd.Timedelta(days=1)).date()
        if (local.hour, local.minute) >= (18, 0)
        else local.date()
    )
    bounds = _session_bounds(label)
    if bounds is None:
        return False
    opened, close = bounds
    if not opened <= local < close:
        return False
    if (
        local.date() < SETTLEMENT_PAUSE_END_EXCLUSIVE
        and (local.hour, local.minute) >= (16, 15)
        and (local.hour, local.minute) < (16, 30)
    ):
        return False
    return True


def scheduled_gap_kind(
    left_end: pd.Timestamp,
    right_start: pd.Timestamp,
) -> str | None:
    """Identify only maintenance, weekend, or registered holiday closures."""

    left = pd.Timestamp(left_end).tz_convert(MARKET_TIMEZONE)
    right = pd.Timestamp(right_start).tz_convert(MARKET_TIMEZONE)
    if right <= left:
        return None

    if (
        left.hour == 17
        and left.minute == 0
        and right.hour == 18
        and right.minute == 0
    ):
        elapsed = right - left
        if elapsed == pd.Timedelta(hours=1):
            return "scheduled_market_closure"
        if (
            left.weekday() == 4
            and right.weekday() == 6
            and elapsed == pd.Timedelta(hours=49)
        ):
            return "scheduled_weekend_closure"
        left_label = pd.Timestamp(left.date())
        right_label = pd.Timestamp((right + pd.Timedelta(days=1)).date())
        if _consecutive_registered_sessions(left_label, right_label):
            return "registered_full_session_closure"

    special_close = special_session_close(left)
    if (
        special_close is not None
        and left == special_close
        and right.hour == 18
        and right.minute == 0
    ):
        if left.date() in ABBREVIATED_GOOD_FRIDAY_CLOSES and right.weekday() == 6:
            return "registered_abbreviated_good_friday_closure"
        left_label = pd.Timestamp(left.date())
        right_label = pd.Timestamp((right + pd.Timedelta(days=1)).date())
        if _consecutive_registered_sessions(left_label, right_label):
            return "registered_special_session_closure"

    if (
        left.date() == right.date()
        and left.date() < SETTLEMENT_PAUSE_END_EXCLUSIVE
        and left.hour == 16
        and left.minute == 15
        and right.hour == 16
        and right.minute == 30
    ):
        return "historical_settlement_pause"

    # Exact endpoint rules above retain useful diagnostic names.  This final
    # independent-calendar reconciliation also accepts gaps whose last/first
    # trade is not exactly at a session boundary.  Any open-market minute keeps
    # the gap unresolved so the caller must densify it or fail closed.
    cursor = left.tz_convert("UTC").floor("min").tz_convert(MARKET_TIMEZONE)
    while cursor < right:
        if is_registered_trading_minute(cursor):
            return None
        cursor += pd.Timedelta(minutes=1)
    return "registered_exchange_closure"


def expected_trading_minutes(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> int:
    """Return wall-clock minutes less registered within-bucket pauses."""

    start_local = pd.Timestamp(start).tz_convert(MARKET_TIMEZONE)
    end_local = pd.Timestamp(end).tz_convert(MARKET_TIMEZONE)
    start_naive = start_local.tz_localize(None)
    end_naive = end_local.tz_localize(None)
    expected = int((end_naive - start_naive).total_seconds() // 60)
    for day in {start_local.date(), end_local.date()}:
        if day >= SETTLEMENT_PAUSE_END_EXCLUSIVE:
            continue
        base = pd.Timestamp(day)
        pause_start = base + pd.Timedelta(hours=16, minutes=15)
        pause_end = base + pd.Timedelta(hours=16, minutes=30)
        if start_naive <= pause_start and pause_end <= end_naive:
            expected -= 15
    if expected <= 0:
        raise ValueError("timeframe bucket has no expected trading minutes")
    return expected


__all__ = [
    "ABBREVIATED_GOOD_FRIDAY_CLOSES",
    "CLOCK_END",
    "CLOCK_START",
    "EQUITY_INDEX_CLOSE_OVERRIDES",
    "MARKET_TIMEZONE",
    "SETTLEMENT_PAUSE_END_EXCLUSIVE",
    "expected_trading_minutes",
    "is_registered_trading_minute",
    "scheduled_gap_kind",
    "special_session_close",
]
