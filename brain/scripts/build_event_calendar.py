"""Build ``brain/configs/economic_calendar.ics`` from the committed sources.

    .venv/bin/python -m brain.scripts.build_event_calendar

Sources under ``brain/configs/calendar_sources/`` (each file's header names
its URL and fetch date):

- ``bls_news_release_subset.tsv`` — the BLS calendar feed's Consumer Price
  Index and Employment Situation events (UID, DTSTART in US-Eastern,
  SUMMARY); the feed's blocks all share the other fields, emitted back here.
- ``bls_schedule_2022.txt`` — the 2022 rows of the same two releases from
  the BLS yearly schedule page (the feed reaches back to 2025-01 only).
- ``fomc_calendar.html`` — the Federal Reserve's FOMC calendar page; every
  scheduled two-day meeting's last day carries the statement at 14:00
  US-Eastern.

The output is one VCALENDAR in the BLS feed's shape (its ``US-Eastern``
VTIMEZONE), every event with a ``DESCRIPTION`` naming its source.  The
committed calendar must equal this script's output (a test checks it)."""
from __future__ import annotations

import argparse
from datetime import date, datetime
import html as html_module
from pathlib import Path
import re
import sys

from brain.scripts._run_identity import ROOT

SOURCES = "brain/configs/calendar_sources"
BLS_SUBSET = "bls_news_release_subset.tsv"
BLS_SCHEDULE = "bls_schedule_2022.txt"
FOMC_HTML = "fomc_calendar.html"
CALENDAR = "brain/configs/economic_calendar.ics"

BLS_FEED_URL = "https://www.bls.gov/schedule/news_release/bls.ics"
BLS_SCHEDULE_URL = "https://www.bls.gov/schedule/{year}/home.htm"
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FOMC_STATEMENT_TIME = "140000"  # 2:00 p.m. US-Eastern

HEADER = """BEGIN:VCALENDAR
PRODID:-//smc_trader//economic calendar (BLS news releases and FOMC statements)//EN
VERSION:2.0
CALSCALE:GREGORIAN
METHOD:PUBLISH
X-WR-CALNAME:Scheduled releases the Brain sleeps through
X-WR-TIMEZONE:US-Eastern
BEGIN:VTIMEZONE
TZID:US-Eastern
BEGIN:DAYLIGHT
TZOFFSETFROM:-0500
TZOFFSETTO:-0400
DTSTART:20070311T020000
RRULE:FREQ=YEARLY;BYMONTH=3;BYDAY=2SU
TZNAME:EDT
END:DAYLIGHT
BEGIN:STANDARD
TZOFFSETFROM:-0400
TZOFFSETTO:-0500
DTSTART:20071104T020000
RRULE:FREQ=YEARLY;BYMONTH=11;BYDAY=1SU
TZNAME:EST
END:STANDARD
END:VTIMEZONE
"""

MONTHS = {name: index for index, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"), 1
)}
MONTH_ABBREVIATIONS = {name[:3]: index for name, index in MONTHS.items()}


def _event(uid: str, dtstart: str, summary: str, description: str) -> str:
    return (
        "BEGIN:VEVENT\nSEQUENCE:1\nCLASS:PUBLIC\n"
        f"UID:{uid}\nDTSTART;TZID=US-Eastern:{dtstart}\nDURATION:PT0M\nSUMMARY:{summary}\n"
        f"DESCRIPTION:{description}\nLOCATION:Washington\\, DC\nTRANSP:TRANSPARENT\nCATEGORIES:IMPORTANT, BLS\nEND:VEVENT\n"
    )


def bls_subset_events(text: str) -> list[tuple[str, str, str]]:
    """``(uid, dtstart, summary)`` rows of the feed subset, comments skipped."""
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        uid, dtstart, summary = (item.strip() for item in line.split("\t"))
        if not re.fullmatch(r"\d{8}T\d{6}", dtstart):
            raise ValueError(f"bad DTSTART in the BLS subset: {line!r}")
        rows.append((uid, dtstart, summary))
    return rows


def bls_schedule_events(text: str) -> list[tuple[str, str, str]]:
    """``(uid, dtstart, summary)`` from ``Month d, yyyy | hh:mm AM | Release for <month>`` rows."""
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        day, clock, release = (item.strip() for item in line.split("|"))
        when = datetime.strptime(f"{day} {clock}", "%B %d, %Y %I:%M %p")
        summary = re.sub(r" for .*$", "", release)
        dtstart = when.strftime("%Y%m%dT%H%M%S")
        rows.append((f"bls-schedule-{when.strftime('%Y%m%d')}-{summary.lower().replace(' ', '-')}", dtstart, summary))
    return rows


def fomc_statement_dates(html: str) -> list[date]:
    """The last day of every scheduled meeting on the Fed's calendar page:
    ``Month d-d``, ``Mon/Mon d-d`` (the second day in the second month),
    ``d-d*`` (a projections meeting); a single-day or ``(unscheduled)`` row
    is not a scheduled statement and is skipped.  Sorted."""
    text = html_module.unescape(html)
    dates: list[date] = []
    year: int | None = None
    # A meeting row is a div whose class names both ``row`` and ``fomc-meeting``
    # (the projection meetings add ``fomc-meeting--shaded`` in front).
    row_start = r'<div class="(?=[^"]*\brow\b)(?=[^"]*\bfomc-meeting\b)[^"]*"'
    pattern = re.compile(
        rf'(\d{{4}}) FOMC Meetings|{row_start}[^>]*>(.*?)(?={row_start}|<div class="panel panel-default|$)', re.S
    )
    for match in pattern.finditer(text):
        if match.group(1):
            year = int(match.group(1))
            continue
        if year is None:
            continue
        row = match.group(2)
        month_cell = re.search(r'fomc-meeting__month[^>]*>(.*?)</div>', row, re.S)
        date_cell = re.search(r'fomc-meeting__date[^>]*>(.*?)</div>', row, re.S)
        if month_cell is None or date_cell is None:
            continue
        months = re.sub(r"<[^>]+>", "", month_cell.group(1)).strip()
        days = re.sub(r"<[^>]+>", "", date_cell.group(1)).strip()
        span = re.fullmatch(r"(\d{1,2})-(\d{1,2})\*?", days)
        if span is None:
            continue  # a one-day or unscheduled meeting: no 14:00 statement on a second day
        names = months.split("/")
        last_month = names[-1].strip()
        month = MONTHS.get(last_month) or MONTH_ABBREVIATIONS.get(last_month[:3])
        if month is None:
            raise ValueError(f"unknown month {months!r} on the FOMC calendar")
        dates.append(date(year, month, int(span.group(2))))
    return sorted(dates)


def build_calendar(*, bls_subset: str, bls_schedule: str, fomc_html: str) -> str:
    parts = [HEADER]
    for uid, dtstart, summary in bls_subset_events(bls_subset):
        parts.append(_event(uid, dtstart, summary, BLS_FEED_URL))
    for uid, dtstart, summary in bls_schedule_events(bls_schedule):
        parts.append(_event(uid, dtstart, summary, BLS_SCHEDULE_URL.format(year=dtstart[:4])))
    for day in fomc_statement_dates(fomc_html):
        stamp = day.strftime("%Y%m%d")
        parts.append(_event(f"fomc-statement-{stamp}", f"{stamp}T{FOMC_STATEMENT_TIME}", "FOMC Statement", FOMC_URL))
    parts.append("END:VCALENDAR\n")
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", default=SOURCES)
    parser.add_argument("--output", default=CALENDAR)
    args = parser.parse_args(argv)
    sources = Path(args.sources) if Path(args.sources).is_absolute() else ROOT / args.sources
    text = build_calendar(
        bls_subset=(sources / BLS_SUBSET).read_text(encoding="utf-8"),
        bls_schedule=(sources / BLS_SCHEDULE).read_text(encoding="utf-8"),
        fomc_html=(sources / FOMC_HTML).read_text(encoding="utf-8"),
    )
    output = Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output} ({text.count('BEGIN:VEVENT')} events)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
