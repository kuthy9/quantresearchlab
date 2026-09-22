"""The Sleep Controller: a gate, not a forecaster.

Once per completed 1m bar it answers one question from the Eye's transition
events alone — is there something worth the Main Brain's reasoning?  Asleep,
it wakes on the configured reaction kinds; awake, it asks for an update when
new evidence arrived or a watched object on one of ``relation_change_timeframes``
changed its side of price (since 2026-09-17: a 5m pool crossing price is
no longer a reason to reason), and ticks otherwise.  Since 2026-09-21 it
also reads the clock against the event calendar: inside a scheduled
release's window (CPI, NFP, FOMC statement — ``events`` in the config) an
asleep Brain stays asleep and an active one is put to sleep
(``EVENT_SLEEP``), and the first bar after the window wakes it whatever the
Eye says.  It carries no weights, no direction and no price threshold, and
the decision to go back to sleep is otherwise the reducer's, not its own."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path

import pandas as pd

from brain.core.event_calendar import EventFilter
from brain.core.eye_view import EvidenceRule, evidence_id
from contract.eye import MarketEvent
from contract.market.primitives import FrozenDict

SLEEP_CONTROLLER_SCHEMA_VERSION = 5  # 5 (2026-09-21): events (the calendar and its sleep windows)


class Decision(str, Enum):
    WAKE = "WAKE"
    STAY_ASLEEP = "STAY_ASLEEP"
    UPDATE = "UPDATE"
    TICK = "TICK"
    # An active episode put to sleep by a scheduled release's window (2026-09-21).
    EVENT_SLEEP = "EVENT_SLEEP"


@dataclass(frozen=True)
class ControllerConfig:
    wake_timeframes_any_reaction: frozenset[str]
    # Formation, touch and level-bookkeeping kinds: evidence for the next
    # call, never a wake and never an UPDATE trigger on their own.
    bookkeeping_kinds: frozenset[str]
    wake_timeframe_specific: Mapping[str, frozenset[str]]
    # The scales whose watched objects' relation flips trigger an UPDATE.
    relation_change_timeframes: frozenset[str]
    # A watched alias's relation flip triggers an UPDATE at most once per
    # this many 1m bars (the runtime keeps the last trigger per alias).
    relation_change_debounce_bars: int
    evidence: EvidenceRule
    tape_timeframe: str
    tape_reaction_kinds: frozenset[str]
    tape_recent_limit: int
    # Consecutive accepted updates with no opportunity and an unchanged
    # understanding after which the episode is archived; None disables it.
    idle_archive_after_updates: int | None
    # The scheduled releases and their sleep windows (schema 5).
    events: EventFilter
    # Over the config bytes and the calendar bytes: a run's identity changes
    # with either.
    sha256: str

    @classmethod
    def from_json(cls, path: Path, *, root: Path | None = None) -> "ControllerConfig":
        """``root`` resolves the calendar path; by default the repository
        root two levels above the config's directory."""
        path = Path(path)
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
        if payload.get("schema_version") != SLEEP_CONTROLLER_SCHEMA_VERSION:
            raise ValueError("unsupported sleep_controller schema_version")
        wake = payload["wake"]
        evidence = payload["evidence"]
        idle = payload["idle_archive_after_updates"]
        base = Path(root) if root is not None else path.resolve().parents[2]
        events = EventFilter.from_config(payload["events"], root=base)
        return cls(
            wake_timeframes_any_reaction=frozenset(wake["timeframes_any_reaction"]),
            bookkeeping_kinds=frozenset(payload["bookkeeping_kinds"]),
            wake_timeframe_specific=FrozenDict(
                {tf: frozenset(kinds) for tf, kinds in wake["timeframe_specific_kinds"].items()}
            ),
            relation_change_timeframes=frozenset(str(tf) for tf in payload["relation_change_timeframes"]),
            relation_change_debounce_bars=max(0, int(payload["relation_change_debounce_bars"])),
            evidence=EvidenceRule(
                timeframes=frozenset(evidence["timeframes"]),
                heartbeat_kinds=frozenset(evidence["heartbeat_kinds_excluded"]),
                state_suffix=str(evidence["state_republication_suffix"]),
            ),
            tape_timeframe=str(evidence["tape_timeframe"]),
            tape_reaction_kinds=frozenset(evidence["tape_reaction_kinds"]),
            tape_recent_limit=int(evidence["tape_recent_limit"]),
            idle_archive_after_updates=None if idle is None else int(idle),
            events=events,
            sha256=hashlib.sha256(raw_bytes + events.sha256.encode("ascii")).hexdigest(),
        )

    def is_wake_event(self, event: MarketEvent) -> bool:
        kind = event.kind.value
        if not self.evidence.is_transition(kind):
            return False
        tf = event.timeframe.value
        if tf in self.wake_timeframes_any_reaction and kind not in self.bookkeeping_kinds:
            return True
        return kind in self.wake_timeframe_specific.get(tf, frozenset())

    def is_update_event(self, event: MarketEvent) -> bool:
        """Evidence that is worth a call: a reaction, not bookkeeping."""
        return self.evidence.is_evidence(event) and event.kind.value not in self.bookkeeping_kinds

    def is_tape_event(self, event: MarketEvent) -> bool:
        return (
            event.timeframe.value == self.tape_timeframe
            and event.kind.value in self.tape_reaction_kinds
            and self.evidence.is_transition(event.kind.value)
        )


@dataclass(frozen=True)
class ControllerDecision:
    decision: Decision
    reasons: tuple[str, ...]


def decide(
    events: Sequence[MarketEvent],
    *,
    active: bool,
    config: ControllerConfig,
    relation_changes: Sequence[str] = (),
    known_at: pd.Timestamp | None = None,
    previous_known_at: pd.Timestamp | None = None,
) -> ControllerDecision:
    """``known_at`` is this bar's clock and ``previous_known_at`` the last
    bar's; without a clock the calendar is not consulted."""
    if known_at is not None:
        window = config.events.active(known_at)
        if window is not None:
            reason = f"event:{config.events.reason(window)}"
            return ControllerDecision(Decision.EVENT_SLEEP if active else Decision.STAY_ASLEEP, (reason,))
        if not active:
            ended = config.events.ended_between(previous_known_at, known_at)
            if ended is not None:
                return ControllerDecision(Decision.WAKE, (f"event_ended:{config.events.reason(ended)}",))
    if not active:
        reasons = tuple(evidence_id(event) for event in events if config.is_wake_event(event))
        if reasons:
            return ControllerDecision(Decision.WAKE, reasons)
        return ControllerDecision(Decision.STAY_ASLEEP, ())
    reasons = tuple(evidence_id(event) for event in events if config.is_update_event(event))
    reasons += tuple(str(alias) for alias in relation_changes)
    if reasons:
        return ControllerDecision(Decision.UPDATE, reasons)
    return ControllerDecision(Decision.TICK, ())


__all__ = [
    "SLEEP_CONTROLLER_SCHEMA_VERSION",
    "ControllerConfig",
    "ControllerDecision",
    "Decision",
    "decide",
]
