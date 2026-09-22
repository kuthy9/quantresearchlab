from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from brain.core.event_calendar import CalendarEvent, EventFilter, EventRule, ScheduledEvent, parse_ics

# Two blocks in the BLS feed's own shape, one under EST and one under EDT, a folded SUMMARY, a Zulu time and an all-day event.
ICS = """BEGIN:VCALENDAR
PRODID:-//Department of Labor//Bureau of Labor Statistics//EN
VERSION:2.0
X-WR-TIMEZONE:US-Eastern
BEGIN:VTIMEZONE
TZID:US-Eastern
END:VTIMEZONE
BEGIN:VEVENT
SEQUENCE:1
CLASS:PUBLIC
UID:6f82e6b8-8b66-4da2-a29c-72643b3440bb
DTSTART;TZID=US-Eastern:20250115T083000
DURATION:PT0M
SUMMARY:Consumer Price Index
LOCATION:Washington\\, DC
TRANSP:TRANSPARENT
CATEGORIES:IMPORTANT, BLS
END:VEVENT
BEGIN:VEVENT
UID:7d36662a-e652-4966-a432-df5d287cac50
DTSTART;TZID=US-Eastern:20250703T083000
SUMMARY:Employment
  Situation
CATEGORIES:IMPORTANT, BLS
END:VEVENT
BEGIN:VEVENT
UID:fomc-1
DTSTART:20220727T180000Z
SUMMARY:FOMC Statement
DESCRIPTION:https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
END:VEVENT
BEGIN:VEVENT
UID:veterans
DTSTART;TZID=US-Eastern:20220421T100000
SUMMARY:Employment Situation of Veterans
END:VEVENT
BEGIN:VEVENT
UID:allday
DTSTART;VALUE=DATE:20250704
SUMMARY:Independence Day
END:VEVENT
BEGIN:VEVENT
UID:floating
DTSTART:20250110T083000
SUMMARY:Employment Situation
END:VEVENT
END:VCALENDAR
"""

RULES = (
    EventRule("CPI", "Consumer Price Index", 60, 30),
    EventRule("NFP", "Employment Situation", 60, 30),
    EventRule("FOMC", "FOMC Statement", 60, 90),
)


def test_parse_ics_reads_tzid_zulu_folded_and_floating_times_and_skips_all_day_events() -> None:
    events = parse_ics(ICS)
    by_uid = {event.uid: event for event in events}
    assert set(by_uid) == {"6f82e6b8-8b66-4da2-a29c-72643b3440bb", "7d36662a-e652-4966-a432-df5d287cac50", "fomc-1", "veterans", "floating"}
    assert by_uid["6f82e6b8-8b66-4da2-a29c-72643b3440bb"].at == pd.Timestamp("2025-01-15T13:30:00Z")  # EST
    assert by_uid["7d36662a-e652-4966-a432-df5d287cac50"].at == pd.Timestamp("2025-07-03T12:30:00Z")  # EDT
    assert by_uid["7d36662a-e652-4966-a432-df5d287cac50"].summary == "Employment Situation"  # unfolded
    assert by_uid["fomc-1"].at == pd.Timestamp("2022-07-27T18:00:00Z")
    assert by_uid["floating"].at == pd.Timestamp("2025-01-10T13:30:00Z")  # X-WR-TIMEZONE
    assert by_uid["6f82e6b8-8b66-4da2-a29c-72643b3440bb"].categories == ("IMPORTANT", "BLS")
    assert isinstance(events[0], CalendarEvent)


def test_rules_match_the_whole_summary_and_build_the_sleep_window() -> None:
    filt = EventFilter.from_events(parse_ics(ICS), RULES, sha256="x")
    kinds = [(event.kind, event.at.isoformat()) for event in filt.events]
    assert ("NFP", "2022-04-21T14:00:00+00:00") not in kinds  # "Employment Situation of Veterans" is not the jobs report
    assert kinds == sorted(kinds, key=lambda item: item[1])
    cpi = next(event for event in filt.events if event.kind == "CPI")
    assert isinstance(cpi, ScheduledEvent) and cpi.name == "Consumer Price Index"
    assert cpi.start == pd.Timestamp("2025-01-15T12:30:00Z") and cpi.end == pd.Timestamp("2025-01-15T14:00:00Z")
    fomc = next(event for event in filt.events if event.kind == "FOMC")
    assert fomc.start == pd.Timestamp("2022-07-27T17:00:00Z") and fomc.end == pd.Timestamp("2022-07-27T19:30:00Z")


def test_active_and_ended_between_boundaries() -> None:
    filt = EventFilter.from_events(parse_ics(ICS), RULES, sha256="x")
    t = pd.Timestamp
    assert filt.active(t("2025-01-15T12:29:00Z")) is None
    assert filt.active(t("2025-01-15T12:30:00Z")).kind == "CPI"  # the window's first minute
    assert filt.active(t("2025-01-15T13:59:00Z")).kind == "CPI"
    assert filt.active(t("2025-01-15T14:00:00Z")) is None  # end is exclusive
    assert filt.ended_between(t("2025-01-15T13:59:00Z"), t("2025-01-15T14:00:00Z")).kind == "CPI"
    assert filt.ended_between(t("2025-01-15T14:00:00Z"), t("2025-01-15T14:01:00Z")) is None  # fires once
    assert filt.ended_between(None, t("2025-01-15T14:00:00Z")) is None  # no previous bar: nothing ended between
    assert filt.reason(filt.active(t("2025-01-15T12:30:00Z"))) == "CPI:2025-01-15T13:30:00Z"


def test_from_config_loads_the_calendar_and_hashes_it(tmp_path: Path) -> None:
    calendar = tmp_path / "cal.ics"
    calendar.write_text(ICS, encoding="utf-8")
    payload = {
        "calendar": "cal.ics",
        "rules": [
            {"kind": "CPI", "summary_pattern": "Consumer Price Index", "sleep_before_minutes": 60, "sleep_after_minutes": 30},
            {"kind": "FOMC", "summary_pattern": "FOMC Statement", "sleep_before_minutes": 60, "sleep_after_minutes": 90},
        ],
    }
    filt = EventFilter.from_config(payload, root=tmp_path)
    assert {event.kind for event in filt.events} == {"CPI", "FOMC"}
    assert filt.sha256 == hashlib.sha256(ICS.encode("utf-8")).hexdigest()
    with pytest.raises(ValueError):
        EventFilter.from_config({**payload, "rules": [{"kind": "CPI", "summary_pattern": "x", "sleep_before_minutes": -1, "sleep_after_minutes": 0}]}, root=tmp_path)
    empty = EventFilter.none()
    assert empty.active(pd.Timestamp("2025-01-15T13:00:00Z")) is None and empty.events == ()
