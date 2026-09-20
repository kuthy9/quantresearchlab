"""The LLM Brain's persistent state.

``BrainState`` is what the Main Brain carries from one bar to the next: an
episode's market understanding, its active expectation, the evidence ledger,
the objects it watches and the opportunity it may have named.  It changes only
through ``brain/core/reducer.py`` and is written in full to the journal after
every revision, so a state is also what a replay compares against.

Every object the state refers to is a registry alias (``FVG_5m_3``); the
registry maps aliases to the Eye's entity ids so a state is self-describing."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
import re
import json
from typing import Any

import pandas as pd

from contract.market.primitives import FrozenDict, aware_timestamp

BRAIN_STATE_SCHEMA_VERSION = 2  # 2 (2026-09-18): ``bias``


class BrainStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"


class OpportunityState(str, Enum):
    NONE = "NONE"
    DEVELOPING = "DEVELOPING"
    ACTIONABLE = "ACTIONABLE"


class TradeDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class ThesisGrade(str, Enum):
    BASE = "BASE"
    A_PLUS = "A_PLUS"


class InvalidationMode(str, Enum):
    TOUCH = "TOUCH"
    CLOSE_BEYOND = "CLOSE_BEYOND"


# The scales a thesis may rest on (the 1m scale is never governing).
GOVERNING_TIMEFRAMES: tuple[str, ...] = ("4H", "1H", "15m", "5m")
# The scales that may set the bias: the 5m is an execution scale, never the
# direction (run B of 2026-09-18 set the bias on the 5m 18 times).
BIAS_SCALES: tuple[str, ...] = ("4H", "1H", "15m")
THESIS_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


class BiasDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class Confidence(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Verdict(str, Enum):
    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    NEUTRAL = "NEUTRAL"
    RESOLVE = "RESOLVE"


def isoformat_utc(timestamp: pd.Timestamp) -> str:
    return pd.Timestamp(timestamp).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text: str) -> pd.Timestamp:
    return aware_timestamp(pd.Timestamp(text), name="timestamp")


def _enum(kind: type[Enum], value: Any, *, name: str) -> Any:
    try:
        return kind(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be one of {[item.value for item in kind]}") from error


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    return value


def _texts(values: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence of text")
    return tuple(_text(value, name=name) for value in values)


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    known_at: pd.Timestamp
    kind: str
    timeframe: str
    object_id: str | None
    verdict: Verdict | None = None
    note: str = ""
    direction: str | None = None
    side: str | None = None

    def __post_init__(self) -> None:
        _text(self.evidence_id, name="evidence_id")
        object.__setattr__(self, "known_at", aware_timestamp(self.known_at, name="evidence.known_at"))
        _text(self.kind, name="evidence.kind")
        _text(self.timeframe, name="evidence.timeframe")
        if self.verdict is not None:
            object.__setattr__(self, "verdict", _enum(Verdict, self.verdict, name="evidence.verdict"))
        _text(self.note, name="evidence.note")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "known_at": isoformat_utc(self.known_at),
            "kind": self.kind,
            "timeframe": self.timeframe,
            "object_id": self.object_id,
            "verdict": None if self.verdict is None else self.verdict.value,
            "note": self.note,
            "direction": self.direction,
            "side": self.side,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceItem":
        return cls(
            evidence_id=payload["evidence_id"],
            known_at=parse_utc(payload["known_at"]),
            kind=payload["kind"],
            timeframe=payload["timeframe"],
            object_id=payload.get("object_id"),
            verdict=payload.get("verdict"),
            note=payload.get("note", ""),
            direction=payload.get("direction"),
            side=payload.get("side"),
        )


@dataclass(frozen=True)
class ActiveExpectation:
    thesis: str = ""
    expected_next: tuple[str, ...] = ()
    should_not_happen: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.thesis, name="active_expectation.thesis")
        object.__setattr__(self, "expected_next", _texts(self.expected_next, name="expected_next"))
        object.__setattr__(
            self, "should_not_happen", _texts(self.should_not_happen, name="should_not_happen")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "thesis": self.thesis,
            "expected_next": list(self.expected_next),
            "should_not_happen": list(self.should_not_happen),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActiveExpectation":
        return cls(payload["thesis"], tuple(payload["expected_next"]), tuple(payload["should_not_happen"]))


@dataclass(frozen=True)
class WatchItem:
    object_id: str
    question: str

    def __post_init__(self) -> None:
        _text(self.object_id, name="watch_next.object_id")
        _text(self.question, name="watch_next.question")

    def to_dict(self) -> dict[str, Any]:
        return {"object_id": self.object_id, "question": self.question}


@dataclass(frozen=True)
class Opportunity:
    """The trade the Brain proposes: three objects, and since 2026-09-17 the
    thesis it expresses (``thesis_id``, stable while the reading holds), the
    scale the thesis rests on, the Brain's grade and how the invalidation
    object falsifies it.  ``thesis_id`` and ``governing_timeframe`` are
    ``None`` only when the state is NONE; the grade and the mode default for
    states journaled before they existed."""

    state: OpportunityState = OpportunityState.NONE
    direction: TradeDirection | None = None
    entry_object_id: str | None = None
    invalidation_object_id: str | None = None
    target_object_id: str | None = None
    thesis_id: str | None = None
    governing_timeframe: str | None = None
    grade: ThesisGrade = ThesisGrade.BASE
    invalidation_mode: InvalidationMode = InvalidationMode.TOUCH

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", _enum(OpportunityState, self.state, name="opportunity.state"))
        if self.direction is not None:
            object.__setattr__(
                self, "direction", _enum(TradeDirection, self.direction, name="opportunity.direction")
            )
        object.__setattr__(self, "grade", _enum(ThesisGrade, self.grade, name="opportunity.grade"))
        object.__setattr__(
            self, "invalidation_mode", _enum(InvalidationMode, self.invalidation_mode, name="opportunity.invalidation_mode")
        )
        ids = (self.entry_object_id, self.invalidation_object_id, self.target_object_id)
        if self.state is OpportunityState.NONE:
            if self.direction is not None or any(value is not None for value in ids):
                raise ValueError("opportunity NONE carries no direction and no object ids")
            if self.thesis_id is not None or self.governing_timeframe is not None:
                raise ValueError("opportunity NONE carries no thesis_id and no governing_timeframe")
            return
        if self.direction is None:
            raise ValueError("opportunity direction is required when the state is not NONE")
        if any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("opportunity requires entry, invalidation and target object ids")
        if len(set(ids)) != 3:
            raise ValueError("opportunity object ids must be distinct")
        if self.thesis_id is not None and (not isinstance(self.thesis_id, str) or not THESIS_ID_PATTERN.match(self.thesis_id)):
            raise ValueError("opportunity thesis_id must match [A-Za-z0-9_-]{1,32}")
        if self.governing_timeframe is not None and self.governing_timeframe not in GOVERNING_TIMEFRAMES:
            raise ValueError(f"opportunity governing_timeframe must be one of {list(GOVERNING_TIMEFRAMES)}")

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(
            value
            for value in (self.entry_object_id, self.invalidation_object_id, self.target_object_id)
            if value is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "direction": None if self.direction is None else self.direction.value,
            "entry_object_id": self.entry_object_id,
            "invalidation_object_id": self.invalidation_object_id,
            "target_object_id": self.target_object_id,
            "thesis_id": self.thesis_id,
            "governing_timeframe": self.governing_timeframe,
            "grade": None if self.state is OpportunityState.NONE else self.grade.value,
            "invalidation_mode": None if self.state is OpportunityState.NONE else self.invalidation_mode.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Opportunity":
        return cls(
            payload["state"],
            payload.get("direction"),
            payload.get("entry_object_id"),
            payload.get("invalidation_object_id"),
            payload.get("target_object_id"),
            thesis_id=payload.get("thesis_id"),
            governing_timeframe=payload.get("governing_timeframe"),
            grade=payload.get("grade") or ThesisGrade.BASE,
            invalidation_mode=payload.get("invalidation_mode") or InvalidationMode.TOUCH,
        )


@dataclass(frozen=True)
class Bias:
    """The Brain's direction (2026-09-18): one side or none, the scale whose
    live delivery sets it, and the one-sentence basis.  The reducer drops an
    opportunity that goes against it or rests on a scale above it."""

    direction: BiasDirection = BiasDirection.NEUTRAL
    scale: str = "15m"
    basis: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", _enum(BiasDirection, self.direction, name="bias.direction"))
        if self.scale not in BIAS_SCALES:
            raise ValueError(f"bias.scale must be one of {list(BIAS_SCALES)}")
        _text(self.basis, name="bias.basis")

    def to_dict(self) -> dict[str, Any]:
        return {"direction": self.direction.value, "scale": self.scale, "basis": self.basis}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Bias":
        return cls(payload["direction"], payload["scale"], payload.get("basis", ""))


@dataclass(frozen=True)
class RegisteredObject:
    entity_id: str
    kind: str
    timeframe: str

    def __post_init__(self) -> None:
        _text(self.entity_id, name="registry.entity_id")
        _text(self.kind, name="registry.kind")
        _text(self.timeframe, name="registry.timeframe")

    def to_dict(self) -> dict[str, str]:
        return {"entity_id": self.entity_id, "kind": self.kind, "timeframe": self.timeframe}


@dataclass(frozen=True)
class EvidenceLedger:
    supporting: tuple[EvidenceItem, ...] = ()
    contradicting: tuple[EvidenceItem, ...] = ()
    unresolved: tuple[EvidenceItem, ...] = ()

    def __post_init__(self) -> None:
        for name in ("supporting", "contradicting", "unresolved"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        ids = [item.evidence_id for item in self.supporting + self.contradicting + self.unresolved]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence ids must be unique across the ledger")

    def to_dict(self) -> dict[str, Any]:
        return {
            "supporting": [item.to_dict() for item in self.supporting],
            "contradicting": [item.to_dict() for item in self.contradicting],
            "unresolved": [item.to_dict() for item in self.unresolved],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceLedger":
        return cls(
            tuple(EvidenceItem.from_dict(item) for item in payload["supporting"]),
            tuple(EvidenceItem.from_dict(item) for item in payload["contradicting"]),
            tuple(EvidenceItem.from_dict(item) for item in payload["unresolved"]),
        )


_VERDICT_KEYS = tuple(item.value for item in Verdict)


@dataclass(frozen=True)
class LastUpdate:
    known_at: pd.Timestamp
    llm_called: bool
    verdicts: Mapping[str, int]
    incident: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "known_at", aware_timestamp(self.known_at, name="last_update.known_at"))
        if type(self.llm_called) is not bool:
            raise ValueError("last_update.llm_called must be a bool")
        counts = {key: int(self.verdicts.get(key, 0)) for key in _VERDICT_KEYS}
        object.__setattr__(self, "verdicts", FrozenDict(counts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "known_at": isoformat_utc(self.known_at),
            "llm_called": self.llm_called,
            "verdicts": dict(self.verdicts),
            "incident": self.incident,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LastUpdate":
        return cls(parse_utc(payload["known_at"]), payload["llm_called"], payload["verdicts"], payload.get("incident"))


@dataclass(frozen=True)
class BrainState:
    episode_id: str
    status: BrainStatus
    revision: int
    started_at: pd.Timestamp
    updated_at: pd.Timestamp
    market_understanding: str
    active_expectation: ActiveExpectation
    evidence: EvidenceLedger
    watch_next: tuple[WatchItem, ...]
    destination_candidates: tuple[str, ...]
    opportunity: Opportunity
    reasoning_confidence: Confidence
    continue_active: bool
    object_registry: Mapping[str, RegisteredObject]
    last_update: LastUpdate
    bias: Bias = field(default_factory=Bias)
    schema_version: int = BRAIN_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BRAIN_STATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported BrainState schema_version {self.schema_version!r}")
        _text(self.episode_id, name="episode_id")
        object.__setattr__(self, "status", _enum(BrainStatus, self.status, name="status"))
        if int(self.revision) < 0:
            raise ValueError("revision must be non-negative")
        object.__setattr__(self, "revision", int(self.revision))
        object.__setattr__(self, "started_at", aware_timestamp(self.started_at, name="started_at"))
        object.__setattr__(self, "updated_at", aware_timestamp(self.updated_at, name="updated_at"))
        if self.updated_at < self.started_at:
            raise ValueError("updated_at precedes started_at")
        _text(self.market_understanding, name="market_understanding")
        object.__setattr__(self, "watch_next", tuple(self.watch_next))
        object.__setattr__(
            self, "destination_candidates", _texts(self.destination_candidates, name="destination_candidates")
        )
        object.__setattr__(
            self,
            "reasoning_confidence",
            _enum(Confidence, self.reasoning_confidence, name="reasoning_confidence"),
        )
        if type(self.continue_active) is not bool:
            raise ValueError("continue_active must be a bool")
        registry = {
            _text(alias, name="registry alias"): (
                value if isinstance(value, RegisteredObject) else RegisteredObject(**value)
            )
            for alias, value in self.object_registry.items()
        }
        object.__setattr__(self, "object_registry", FrozenDict(registry))
        for item in self.watch_next:
            if item.object_id not in registry:
                raise ValueError(f"watch_next names an unregistered object {item.object_id!r}")
        for alias in self.destination_candidates:
            if alias not in registry:
                raise ValueError(f"destination_candidates names an unregistered object {alias!r}")
        for alias in self.opportunity.object_ids:
            if alias not in registry:
                raise ValueError(f"opportunity names an unregistered object {alias!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "status": self.status.value,
            "revision": self.revision,
            "started_at": isoformat_utc(self.started_at),
            "updated_at": isoformat_utc(self.updated_at),
            "market_understanding": self.market_understanding,
            "active_expectation": self.active_expectation.to_dict(),
            "evidence": self.evidence.to_dict(),
            "watch_next": [item.to_dict() for item in self.watch_next],
            "destination_candidates": list(self.destination_candidates),
            "opportunity": self.opportunity.to_dict(),
            "reasoning_confidence": self.reasoning_confidence.value,
            "continue_active": self.continue_active,
            "object_registry": {
                alias: entry.to_dict() for alias, entry in sorted(self.object_registry.items())
            },
            "last_update": self.last_update.to_dict(),
            "bias": self.bias.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BrainState":
        expected = {item.name for item in fields(cls)}
        unknown = set(payload) - expected
        if unknown:
            raise ValueError(f"BrainState payload has unknown keys {sorted(unknown)}")
        missing = expected - set(payload)
        if payload.get("schema_version") == 1:
            missing -= {"bias"}  # schema 1 (before 2026-09-18) had no bias
        if missing:
            raise ValueError(f"BrainState payload is missing {sorted(missing)}")
        if payload["schema_version"] not in (1, BRAIN_STATE_SCHEMA_VERSION):
            raise ValueError(f"unsupported BrainState schema_version {payload['schema_version']!r}")
        return cls(
            episode_id=payload["episode_id"],
            status=payload["status"],
            revision=payload["revision"],
            started_at=parse_utc(payload["started_at"]),
            updated_at=parse_utc(payload["updated_at"]),
            market_understanding=payload["market_understanding"],
            active_expectation=ActiveExpectation.from_dict(payload["active_expectation"]),
            evidence=EvidenceLedger.from_dict(payload["evidence"]),
            watch_next=tuple(WatchItem(**item) for item in payload["watch_next"]),
            destination_candidates=tuple(payload["destination_candidates"]),
            opportunity=Opportunity.from_dict(payload["opportunity"]),
            reasoning_confidence=payload["reasoning_confidence"],
            continue_active=payload["continue_active"],
            object_registry={
                alias: RegisteredObject(**entry) for alias, entry in payload["object_registry"].items()
            },
            last_update=LastUpdate.from_dict(payload["last_update"]),
            bias=Bias.from_dict(payload["bias"]) if "bias" in payload else Bias(),
            schema_version=BRAIN_STATE_SCHEMA_VERSION,
        )

    @classmethod
    def from_json(cls, text: str) -> "BrainState":
        return cls.from_dict(json.loads(text))


__all__ = [
    "BRAIN_STATE_SCHEMA_VERSION",
    "ActiveExpectation",
    "BIAS_SCALES",
    "Bias",
    "BiasDirection",
    "BrainState",
    "BrainStatus",
    "Confidence",
    "EvidenceItem",
    "EvidenceLedger",
    "LastUpdate",
    "Opportunity",
    "OpportunityState",
    "RegisteredObject",
    "TradeDirection",
    "Verdict",
    "WatchItem",
    "isoformat_utc",
    "parse_utc",
]
