"""The Sleep Controller: a gate, not a forecaster.

Once per completed 1m bar it answers one question from the Eye's transition
events alone — is there something worth the Main Brain's reasoning?  Asleep,
it wakes on the configured reaction kinds; awake, it asks for an update when
new evidence arrived or a watched object's price relation changed, and ticks
otherwise.  It carries no weights, no direction and no price threshold, and
the decision to go back to sleep is the reducer's, not its own."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path

from brain.core.eye_view import EvidenceRule, evidence_id
from contract.eye import MarketEvent
from contract.market.primitives import FrozenDict

SLEEP_CONTROLLER_SCHEMA_VERSION = 1


class Decision(str, Enum):
    WAKE = "WAKE"
    STAY_ASLEEP = "STAY_ASLEEP"
    UPDATE = "UPDATE"
    TICK = "TICK"


@dataclass(frozen=True)
class ControllerConfig:
    wake_timeframes_any_reaction: frozenset[str]
    wake_excluded_kinds: frozenset[str]
    wake_timeframe_specific: Mapping[str, frozenset[str]]
    evidence: EvidenceRule
    tape_timeframe: str
    tape_reaction_kinds: frozenset[str]
    tape_recent_limit: int
    sha256: str

    @classmethod
    def from_json(cls, path: Path) -> "ControllerConfig":
        raw_bytes = Path(path).read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
        if payload.get("schema_version") != SLEEP_CONTROLLER_SCHEMA_VERSION:
            raise ValueError("unsupported sleep_controller schema_version")
        wake = payload["wake"]
        evidence = payload["evidence"]
        return cls(
            wake_timeframes_any_reaction=frozenset(wake["timeframes_any_reaction"]),
            wake_excluded_kinds=frozenset(wake["reaction_kinds_excluded"]),
            wake_timeframe_specific=FrozenDict(
                {tf: frozenset(kinds) for tf, kinds in wake["timeframe_specific_kinds"].items()}
            ),
            evidence=EvidenceRule(
                timeframes=frozenset(evidence["timeframes"]),
                heartbeat_kinds=frozenset(evidence["heartbeat_kinds_excluded"]),
                state_suffix=str(evidence["state_republication_suffix"]),
            ),
            tape_timeframe=str(evidence["tape_timeframe"]),
            tape_reaction_kinds=frozenset(evidence["tape_reaction_kinds"]),
            tape_recent_limit=int(evidence["tape_recent_limit"]),
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
        )

    def is_wake_event(self, event: MarketEvent) -> bool:
        kind = event.kind.value
        if not self.evidence.is_transition(kind):
            return False
        tf = event.timeframe.value
        if tf in self.wake_timeframes_any_reaction and kind not in self.wake_excluded_kinds:
            return True
        return kind in self.wake_timeframe_specific.get(tf, frozenset())

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
) -> ControllerDecision:
    if not active:
        reasons = tuple(evidence_id(event) for event in events if config.is_wake_event(event))
        if reasons:
            return ControllerDecision(Decision.WAKE, reasons)
        return ControllerDecision(Decision.STAY_ASLEEP, ())
    reasons = tuple(evidence_id(event) for event in events if config.evidence.is_evidence(event))
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
