from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from brain.core.event_calendar import EventFilter, EventRule, parse_ics
from brain.scripts.build_event_calendar import BLS_SCHEDULE, BLS_SUBSET, CALENDAR, FOMC_HTML, build_calendar, fomc_statement_dates, main

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "brain" / "configs" / "calendar_sources"

FOMC_SNIPPET = """
<div class="panel panel-default"><div class="panel-heading"><h4><a id="1">2024 FOMC Meetings</a></h4></div>
<div class="row fomc-meeting">
<div class="fomc-meeting__month col-xs-5"><strong>January</strong></div>
<div class="fomc-meeting__date col-xs-4">30-31</div>
</div>
<div class="fomc-meeting--shaded row fomc-meeting" ">
<div class="fomc-meeting--shaded fomc-meeting__month col-xs-5"><strong>March</strong></div>
<div class="fomc-meeting__date col-xs-4">19-20*</div>
</div>
<div class="row fomc-meeting">
<div class="fomc-meeting__month col-xs-5"><strong>Apr/May</strong></div>
<div class="fomc-meeting__date col-xs-4">30-1</div>
</div>
<div class="row fomc-meeting">
<div class="fomc-meeting__month col-xs-5"><strong>October</strong></div>
<div class="fomc-meeting__date col-xs-4">6 (unscheduled)</div>
</div>
</div>
<div class="panel panel-default"><div class="panel-heading"><h4><a id="2">2022 FOMC Meetings</a></h4></div>
<div class="row fomc-meeting">
<div class="fomc-meeting__month col-xs-5"><strong>July</strong></div>
<div class="fomc-meeting__date col-xs-4">26-27</div>
</div>
</div>
"""


def test_fomc_statement_dates_take_the_last_day_of_every_scheduled_meeting() -> None:
    assert fomc_statement_dates(FOMC_SNIPPET) == [date(2022, 7, 27), date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1)]


def test_the_real_fed_page_gives_eight_statements_in_2022() -> None:
    dates = fomc_statement_dates((SOURCES / FOMC_HTML).read_text(encoding="utf-8"))
    assert [d for d in dates if d.year == 2022] == [
        date(2022, 1, 26), date(2022, 3, 16), date(2022, 5, 4), date(2022, 6, 15), date(2022, 7, 27), date(2022, 9, 21), date(2022, 11, 2), date(2022, 12, 14),
    ]
    assert {d.year for d in dates} >= {2021, 2022, 2023, 2024, 2025, 2026}


def test_build_calendar_puts_every_source_in_the_feeds_shape() -> None:
    text = build_calendar(
        bls_subset=(SOURCES / BLS_SUBSET).read_text(encoding="utf-8"),
        bls_schedule=(SOURCES / BLS_SCHEDULE).read_text(encoding="utf-8"),
        fomc_html=(SOURCES / FOMC_HTML).read_text(encoding="utf-8"),
    )
    assert text.startswith("BEGIN:VCALENDAR") and "TZID:US-Eastern" in text and "DTSTART;TZID=US-Eastern:20221013T083000" in text
    assert "DTSTART;TZID=US-Eastern:20220727T140000" in text and "SUMMARY:FOMC Statement" in text
    events = parse_ics(text)
    in_2022 = [event for event in events if event.at.year == 2022]
    assert sum(event.summary == "Consumer Price Index" for event in in_2022) == 12
    assert sum(event.summary == "Employment Situation" for event in in_2022) == 12
    assert sum(event.summary == "FOMC Statement" for event in in_2022) == 8
    assert sum(event.summary in ("Consumer Price Index", "Employment Situation") for event in events if event.at.year >= 2025) == 46
    assert all("Veterans" not in event.summary for event in events)
    cpi = next(event for event in events if event.uid == "6f82e6b8-8b66-4da2-a29c-72643b3440bb")
    assert cpi.at == pd.Timestamp("2025-01-15T13:30:00Z") and cpi.summary == "Consumer Price Index"
    filt = EventFilter.from_events(events, (EventRule("CPI", "Consumer Price Index", 60, 30), EventRule("FOMC", "FOMC Statement", 60, 90)), sha256="x")
    assert filt.active(pd.Timestamp("2022-10-13T12:31:00Z")).kind == "CPI"  # 08:31 New York, the crash bar of the third pass
    assert filt.active(pd.Timestamp("2022-07-27T19:29:00Z")).kind == "FOMC" and filt.active(pd.Timestamp("2022-07-27T19:30:00Z")) is None


def test_the_committed_calendar_is_what_the_script_builds(tmp_path: Path) -> None:
    out = tmp_path / "economic_calendar.ics"
    assert main(["--output", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == (ROOT / CALENDAR).read_text(encoding="utf-8")
