"""What crosses the wire between the Main Brain and the LLM.

``LLMInput`` is the exact payload sent as the user turn — its canonical JSON
is hashed into the journal so a replay can prove the same bytes were sent.
``LLMUpdate`` is the one JSON object the model may reply with.  ``parse_update``
is the gate: every id must be an alias the input published, every enum must
be in vocabulary, no key may be missing or extra, and nothing numeric that
could be read as a price exists in the schema at all."""
from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import pandas as pd

from contract.market.primitives import FrozenDict, aware_timestamp
from contract.brain.state import (
    GOVERNING_TIMEFRAMES,
    ActiveExpectation,
    Confidence,
    InvalidationMode,
    Opportunity,
    OpportunityState,
    ThesisGrade,
    TradeDirection,
    Verdict,
    WatchItem,
    isoformat_utc,
)

LLM_INPUT_SCHEMA_VERSION = 1
FRAMEWORK_STEPS: tuple[str, ...] = tuple(f"step_{i}" for i in range(1, 15))
LLM_UPDATE_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "evidence_verdicts",
        "understanding_holds",
        "market_understanding",
        "active_expectation",
        "watch_next",
        "destination_candidates",
        "opportunity",
        "reasoning_confidence",
        "continue_active",
        "framework_trace",
    }
)
_VERDICT_KEYS = frozenset({"evidence_id", "verdict", "note", "resolves_evidence_id", "resolution"})
_EXPECTATION_KEYS = frozenset({"thesis", "expected_next", "should_not_happen"})
_WATCH_KEYS = frozenset({"object_id", "question"})
_OPPORTUNITY_KEYS = frozenset(
    {
        "state", "direction", "entry_object_id", "invalidation_object_id", "target_object_id",
        "thesis_id", "governing_timeframe", "grade", "invalidation_mode",
    }
)
_THESIS_KEYS = ("thesis_id", "governing_timeframe", "grade", "invalidation_mode")


class MalformedReply(ValueError):
    """The model's reply does not satisfy the update contract."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class EvidenceVerdict:
    evidence_id: str
    verdict: Verdict
    note: str = ""
    resolves_evidence_id: str | None = None
    resolution: Verdict | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "verdict": self.verdict.value,
            "note": self.note,
            "resolves_evidence_id": self.resolves_evidence_id,
            "resolution": None if self.resolution is None else self.resolution.value,
        }


@dataclass(frozen=True)
class LLMUpdate:
    evidence_verdicts: tuple[EvidenceVerdict, ...]
    understanding_holds: bool
    market_understanding: str
    active_expectation: ActiveExpectation
    watch_next: tuple[WatchItem, ...]
    destination_candidates: tuple[str, ...]
    opportunity: Opportunity
    reasoning_confidence: Confidence
    continue_active: bool
    framework_trace: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_verdicts", tuple(self.evidence_verdicts))
        object.__setattr__(self, "watch_next", tuple(self.watch_next))
        object.__setattr__(self, "destination_candidates", tuple(self.destination_candidates))
        object.__setattr__(self, "framework_trace", FrozenDict(dict(self.framework_trace)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_verdicts": [item.to_dict() for item in self.evidence_verdicts],
            "understanding_holds": self.understanding_holds,
            "market_understanding": self.market_understanding,
            "active_expectation": self.active_expectation.to_dict(),
            "watch_next": [item.to_dict() for item in self.watch_next],
            "destination_candidates": list(self.destination_candidates),
            "opportunity": self.opportunity.to_dict(),
            "reasoning_confidence": self.reasoning_confidence.value,
            "continue_active": self.continue_active,
            "framework_trace": dict(self.framework_trace),
        }


LLM_UPDATE_EXAMPLE: dict[str, Any] = {
    "evidence_verdicts": [
        {
            "evidence_id": "ev_<id from new_evidence>",
            "verdict": "SUPPORT",
            "note": "why this evidence supports, contradicts, is neutral to, or resolves the expectation",
            "resolves_evidence_id": None,
            "resolution": None,
        }
    ],
    "understanding_holds": True,
    "market_understanding": "one paragraph: HTF/LTF structure, the active range and where price sits in it, delivery, protected structure, the liquidity that matters",
    "active_expectation": {
        "thesis": "what should happen next if the understanding is right",
        "expected_next": ["observable behaviour that would confirm it"],
        "should_not_happen": ["observable behaviour that would deny it"],
    },
    "watch_next": [{"object_id": "FVG_5m_3", "question": "does price accept or reject here?"}],
    "destination_candidates": ["BSL_1H_1"],
    "opportunity": {
        "state": "NONE",
        "direction": None,
        "entry_object_id": None,
        "invalidation_object_id": None,
        "target_object_id": None,
        "thesis_id": None,
        "governing_timeframe": None,
        "grade": None,
        "invalidation_mode": None,
    },
    "reasoning_confidence": "LOW",
    "continue_active": True,
    "framework_trace": {step: "one sentence, or n/a" for step in FRAMEWORK_STEPS},
}


def _fail(message: str) -> MalformedReply:
    return MalformedReply(message)


def _require_keys(payload: Any, keys: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise _fail(f"{name}: expected a JSON object")
    present = set(payload)
    if present != keys:
        missing = sorted(keys - present)
        extra = sorted(present - keys)
        raise _fail(f"{name}: keys mismatch (missing {missing}, unexpected {extra})")
    return payload


def _str(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise _fail(f"{name}: expected text")
    return value


def _opt_str(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _str(value, name=name)


def _bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise _fail(f"{name}: expected true or false")
    return value


def _str_list(value: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise _fail(f"{name}: expected a list")
    return tuple(_str(item, name=f"{name}[]") for item in value)


def _enum(kind: type, value: Any, *, name: str) -> Any:
    try:
        return kind(value)
    except (TypeError, ValueError):
        raise _fail(f"{name}: {value!r} is not one of {[item.value for item in kind]}") from None


def parse_update(
    text: str, *, evidence_ids: Collection[str], object_ids: Collection[str]
) -> LLMUpdate:
    """Validate a reply against the contract; ``MalformedReply`` on any defect."""

    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as error:
        raise _fail(f"reply is not JSON: {error}") from None
    payload = _require_keys(payload, LLM_UPDATE_REQUIRED_KEYS, name="reply")
    known_evidence = set(evidence_ids)
    known_objects = set(object_ids)

    raw_verdicts = payload["evidence_verdicts"]
    if not isinstance(raw_verdicts, list):
        raise _fail("evidence_verdicts: expected a list")
    verdicts: list[EvidenceVerdict] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_verdicts):
        name = f"evidence_verdicts[{index}]"
        raw = _require_keys(raw, _VERDICT_KEYS, name=name)
        evidence_id = _str(raw["evidence_id"], name=f"{name}.evidence_id")
        if evidence_id not in known_evidence:
            raise _fail(f"{name}: {evidence_id!r} is not in new_evidence")
        if evidence_id in seen:
            raise _fail(f"{name}: duplicate verdict for {evidence_id!r}")
        seen.add(evidence_id)
        verdict = _enum(Verdict, raw["verdict"], name=f"{name}.verdict")
        note = _str(raw["note"], name=f"{name}.note")
        resolves = _opt_str(raw["resolves_evidence_id"], name=f"{name}.resolves_evidence_id")
        resolution_raw = raw["resolution"]
        if verdict is Verdict.RESOLVE:
            if resolves is None:
                raise _fail(f"{name}: RESOLVE needs resolves_evidence_id")
            resolution = _enum(Verdict, resolution_raw, name=f"{name}.resolution")
            if resolution not in (Verdict.SUPPORT, Verdict.CONTRADICT):
                raise _fail(f"{name}: resolution must be SUPPORT or CONTRADICT")
        else:
            if resolves is not None or resolution_raw is not None:
                raise _fail(f"{name}: only RESOLVE carries resolves_evidence_id / resolution")
            resolution = None
        verdicts.append(EvidenceVerdict(evidence_id, verdict, note, resolves, resolution))

    understanding_holds = _bool(payload["understanding_holds"], name="understanding_holds")
    market_understanding = _str(payload["market_understanding"], name="market_understanding")

    raw_expectation = _require_keys(payload["active_expectation"], _EXPECTATION_KEYS, name="active_expectation")
    expectation = ActiveExpectation(
        _str(raw_expectation["thesis"], name="active_expectation.thesis"),
        _str_list(raw_expectation["expected_next"], name="active_expectation.expected_next"),
        _str_list(raw_expectation["should_not_happen"], name="active_expectation.should_not_happen"),
    )

    raw_watch = payload["watch_next"]
    if not isinstance(raw_watch, list):
        raise _fail("watch_next: expected a list")
    watch: list[WatchItem] = []
    for index, raw in enumerate(raw_watch):
        name = f"watch_next[{index}]"
        raw = _require_keys(raw, _WATCH_KEYS, name=name)
        alias = _str(raw["object_id"], name=f"{name}.object_id")
        if alias not in known_objects:
            raise _fail(f"{name}: {alias!r} is not a published object")
        watch.append(WatchItem(alias, _str(raw["question"], name=f"{name}.question")))

    destinations = _str_list(payload["destination_candidates"], name="destination_candidates")
    for alias in destinations:
        if alias not in known_objects:
            raise _fail(f"destination_candidates: {alias!r} is not a published object")

    raw_opportunity = _require_keys(payload["opportunity"], _OPPORTUNITY_KEYS, name="opportunity")
    state = _enum(OpportunityState, raw_opportunity["state"], name="opportunity.state")
    direction_raw = raw_opportunity["direction"]
    direction = None if direction_raw is None else _enum(TradeDirection, direction_raw, name="opportunity.direction")
    ids = {
        key: _opt_str(raw_opportunity[key], name=f"opportunity.{key}")
        for key in ("entry_object_id", "invalidation_object_id", "target_object_id")
    }
    for key, alias in ids.items():
        if alias is not None and alias not in known_objects:
            raise _fail(f"opportunity.{key}: {alias!r} is not a published object")
    thesis: dict[str, Any] = {}
    if state is OpportunityState.NONE:
        for key in _THESIS_KEYS:
            if raw_opportunity[key] is not None:
                raise _fail(f"opportunity.{key} must be null when the state is NONE")
    else:
        thesis["thesis_id"] = _str(raw_opportunity["thesis_id"], name="opportunity.thesis_id")
        timeframe = _str(raw_opportunity["governing_timeframe"], name="opportunity.governing_timeframe")
        if timeframe not in GOVERNING_TIMEFRAMES:
            raise _fail(f"opportunity.governing_timeframe must be one of {list(GOVERNING_TIMEFRAMES)}")
        thesis["governing_timeframe"] = timeframe
        thesis["grade"] = _enum(ThesisGrade, raw_opportunity["grade"], name="opportunity.grade")
        thesis["invalidation_mode"] = _enum(InvalidationMode, raw_opportunity["invalidation_mode"], name="opportunity.invalidation_mode")
    try:
        opportunity = Opportunity(
            state, direction, ids["entry_object_id"], ids["invalidation_object_id"], ids["target_object_id"], **thesis
        )
    except ValueError as error:
        raise _fail(f"opportunity: {error}") from None

    confidence = _enum(Confidence, payload["reasoning_confidence"], name="reasoning_confidence")
    continue_active = _bool(payload["continue_active"], name="continue_active")

    raw_trace = payload["framework_trace"]
    if not isinstance(raw_trace, Mapping) or set(raw_trace) != set(FRAMEWORK_STEPS):
        raise _fail(f"framework_trace: expected exactly the keys {list(FRAMEWORK_STEPS)}")
    trace = {step: _str(raw_trace[step], name=f"framework_trace.{step}") for step in FRAMEWORK_STEPS}

    return LLMUpdate(
        evidence_verdicts=tuple(verdicts),
        understanding_holds=understanding_holds,
        market_understanding=market_understanding,
        active_expectation=expectation,
        watch_next=tuple(watch),
        destination_candidates=destinations,
        opportunity=opportunity,
        reasoning_confidence=confidence,
        continue_active=continue_active,
        framework_trace=trace,
    )


def _plain(value: Any) -> Any:
    """Tuples to lists, mappings to dicts, so the payload is plain JSON."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class LLMInput:
    episode_id: str
    known_at: pd.Timestamp
    trigger: Mapping[str, Any]
    bar: Mapping[str, float]
    session: Mapping[str, Any]
    scales: Mapping[str, Any]
    interaction: tuple[Mapping[str, Any], ...]
    new_evidence: tuple[Mapping[str, Any], ...]
    tape_since_last_update: Mapping[str, Any]
    price_relations: tuple[Mapping[str, Any], ...]
    prior_state: Mapping[str, Any] | None
    schema_version: int = LLM_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "known_at", aware_timestamp(self.known_at, name="llm_input.known_at"))
        object.__setattr__(self, "trigger", FrozenDict(_plain(self.trigger)))
        object.__setattr__(self, "bar", FrozenDict(_plain(self.bar)))
        object.__setattr__(self, "session", FrozenDict(_plain(self.session)))
        object.__setattr__(self, "scales", FrozenDict(_plain(self.scales)))
        object.__setattr__(self, "interaction", tuple(FrozenDict(_plain(item)) for item in self.interaction))
        object.__setattr__(self, "new_evidence", tuple(FrozenDict(_plain(item)) for item in self.new_evidence))
        object.__setattr__(self, "tape_since_last_update", FrozenDict(_plain(self.tape_since_last_update)))
        object.__setattr__(self, "price_relations", tuple(FrozenDict(_plain(item)) for item in self.price_relations))
        object.__setattr__(
            self, "prior_state", None if self.prior_state is None else FrozenDict(_plain(self.prior_state))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "known_at": isoformat_utc(self.known_at),
            "trigger": _plain(self.trigger),
            "bar": _plain(self.bar),
            "session": _plain(self.session),
            "scales": _plain(self.scales),
            "interaction": _plain(self.interaction),
            "new_evidence": _plain(self.new_evidence),
            "tape_since_last_update": _plain(self.tape_since_last_update),
            "price_relations": _plain(self.price_relations),
            "prior_state": None if self.prior_state is None else _plain(self.prior_state),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def input_sha(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


__all__ = [
    "FRAMEWORK_STEPS",
    "LLM_INPUT_SCHEMA_VERSION",
    "LLM_UPDATE_EXAMPLE",
    "LLM_UPDATE_REQUIRED_KEYS",
    "EvidenceVerdict",
    "LLMInput",
    "LLMUpdate",
    "MalformedReply",
    "canonical_json",
    "parse_update",
]
