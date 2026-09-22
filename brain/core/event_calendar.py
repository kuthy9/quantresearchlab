"""Scheduled releases the Brain sleeps through.

An ``.ics`` calendar (the BLS news-release feed's shape, plus the FOMC
statements the build script adds) is parsed into ``CalendarEvent``s; the
controller's ``events`` rules turn the matching ones into
``ScheduledEvent`` windows — ``[at − before, at + after)`` — and the
``EventFilter`` answers two questions per bar: is this bar inside a window,
and did a window end since the previous bar.  No date or time lives in
code: ``brain/configs/economic_calendar.ics`` is the data, built by
``brain/scripts/build_event_calendar.py`` from the committed sources."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

import pandas as pd

# The BLS feed's TZID is not an IANA name.
TZID_ALIASES: Mapping[str, str] = {"US-Eastern": "America/New_York"}


@dataclass(frozen=True)
class CalendarEvent:
    uid: str
    summary: str
    at: pd.Timestamp  # UTC
    categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventRule:
    """Which summaries are one kind of release, and how long the Brain
    sleeps around each: ``sleep_before_minutes`` before the release,
    ``sleep_after_minutes`` after it."""

    kind: str
    summary_pattern: str
    sleep_before_minutes: int
    sleep_after_minutes: int

    def __post_init__(self) -> None:
        if not self.kind or not self.summary_pattern:
            raise ValueError("an event rule needs a kind and a summary_pattern")
        if int(self.sleep_before_minutes) < 0 or int(self.sleep_after_minutes) < 0:
            raise ValueError(f"event rule {self.kind}: sleep minutes must not be negative")
        object.__setattr__(self, "sleep_before_minutes", int(self.sleep_before_minutes))
        object.__setattr__(self, "sleep_after_minutes", int(self.sleep_after_minutes))
        re.compile(self.summary_pattern)

    def matches(self, summary: str) -> bool:
        return re.fullmatch(self.summary_pattern, summary) is not None


@dataclass(frozen=True)
class ScheduledEvent:
    kind: str
    name: str
    at: pd.Timestamp
    start: pd.Timestamp
    end: pd.Timestamp


def _unfold(text: str) -> list[str]:
    """RFC 5545 line unfolding: a line starting with a space or tab
    continues the previous one."""
    lines: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw[:1] in (" ", "\t") and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def _property(line: str) -> tuple[str, dict[str, str], str] | None:
    """``NAME;PARAM=VALUE:value`` → (name, params, value); None for a line
    without a colon."""
    if ":" not in line:
        return None
    head, value = line.split(":", 1)
    parts = head.split(";")
    params = {}
    for part in parts[1:]:
        if "=" in part:
            key, item = part.split("=", 1)
            params[key.strip().upper()] = item.strip()
    return parts[0].strip().upper(), params, value


def _unescape(value: str) -> str:
    return value.replace("\\,", ",").replace("\\;", ";").replace("\\n", "\n").replace("\\\\", "\\").strip()


def _timestamp(value: str, params: Mapping[str, str], default_tz: str | None) -> pd.Timestamp | None:
    """A DTSTART value in UTC; None for an all-day event."""
    if params.get("VALUE", "").upper() == "DATE" or re.fullmatch(r"\d{8}", value.strip()):
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        return pd.Timestamp(raw[:-1]).tz_localize("UTC")
    zone = params.get("TZID") or default_tz
    if zone is None:
        raise ValueError(f"DTSTART {raw!r} has no time zone and the calendar names none")
    zone = TZID_ALIASES.get(zone, zone)
    return pd.Timestamp(raw).tz_localize(zone).tz_convert("UTC")


def parse_ics(text: str) -> tuple[CalendarEvent, ...]:
    """Every timed ``VEVENT`` of the calendar, in file order."""
    default_tz: str | None = None
    events: list[CalendarEvent] = []
    block: dict[str, Any] | None = None
    for line in _unfold(text):
        prop = _property(line)
        if prop is None:
            continue
        name, params, value = prop
        if name == "X-WR-TIMEZONE":
            default_tz = TZID_ALIASES.get(value.strip(), value.strip())
        elif name == "BEGIN" and value.strip().upper() == "VEVENT":
            block = {}
        elif name == "END" and value.strip().upper() == "VEVENT":
            if block is not None and block.get("at") is not None:
                events.append(CalendarEvent(
                    uid=str(block.get("uid", "")), summary=str(block.get("summary", "")), at=block["at"],
                    categories=tuple(block.get("categories", ())),
                ))
            block = None
        elif block is not None:
            if name == "UID":
                block["uid"] = _unescape(value)
            elif name == "SUMMARY":
                block["summary"] = _unescape(value)
            elif name == "DTSTART":
                block["at"] = _timestamp(value, params, default_tz)
            elif name == "CATEGORIES":
                block["categories"] = tuple(item.strip() for item in _unescape(value).split(",") if item.strip())
    return tuple(events)


@dataclass(frozen=True)
class EventFilter:
    events: tuple[ScheduledEvent, ...]
    sha256: str

    @classmethod
    def none(cls) -> "EventFilter":
        return cls((), hashlib.sha256(b"").hexdigest())

    @classmethod
    def from_events(cls, events: Iterable[CalendarEvent], rules: Sequence[EventRule], *, sha256: str) -> "EventFilter":
        scheduled: list[ScheduledEvent] = []
        for event in events:
            for rule in rules:
                if rule.matches(event.summary):
                    scheduled.append(ScheduledEvent(
                        kind=rule.kind, name=event.summary, at=event.at,
                        start=event.at - pd.Timedelta(minutes=rule.sleep_before_minutes),
                        end=event.at + pd.Timedelta(minutes=rule.sleep_after_minutes),
                    ))
                    break
        scheduled.sort(key=lambda item: (item.start, item.kind))
        return cls(tuple(scheduled), sha256)

    @classmethod
    def from_config(cls, payload: Mapping[str, Any], *, root: Path) -> "EventFilter":
        """``{"calendar": <path relative to root>, "rules": [...]}``."""
        rules = tuple(
            EventRule(str(rule["kind"]), str(rule["summary_pattern"]), int(rule["sleep_before_minutes"]), int(rule["sleep_after_minutes"]))
            for rule in payload["rules"]
        )
        if not rules:
            raise ValueError("events.rules must name at least one rule")
        path = Path(payload["calendar"])
        raw = (path if path.is_absolute() else Path(root) / path).read_bytes()
        return cls.from_events(parse_ics(raw.decode("utf-8")), rules, sha256=hashlib.sha256(raw).hexdigest())

    def active(self, known_at: pd.Timestamp) -> ScheduledEvent | None:
        """The event whose ``[start, end)`` contains the bar."""
        at = pd.Timestamp(known_at).tz_convert("UTC")
        for event in self.events:
            if event.start <= at < event.end:
                return event
        return None

    def ended_between(self, previous: pd.Timestamp | None, known_at: pd.Timestamp) -> ScheduledEvent | None:
        """The event whose ``end`` lies in ``(previous, known_at]``: the
        first bar at or after the window's end."""
        if previous is None:
            return None
        before = pd.Timestamp(previous).tz_convert("UTC")
        at = pd.Timestamp(known_at).tz_convert("UTC")
        for event in self.events:
            if before < event.end <= at:
                return event
        return None

    @staticmethod
    def reason(event: ScheduledEvent) -> str:
        return f"{event.kind}:{event.at.strftime('%Y-%m-%dT%H:%M:%SZ')}"


__all__ = ["TZID_ALIASES", "CalendarEvent", "EventFilter", "EventRule", "ScheduledEvent", "parse_ics"]
