from __future__ import annotations

import json

import pandas as pd
import pytest

from contract.brain.llm import LLM_UPDATE_EXAMPLE, LLMInput, MalformedReply, parse_update
from contract.brain.state import Bias, BiasDirection, InvalidationMode, OpportunityState, ThesisGrade, Verdict

EVIDENCE = {"ev_1", "ev_2"}
OBJECTS = {"FVG_5m_3", "SSL_5m_2", "BSL_1H_1"}


def _reply(**overrides) -> str:
    payload = json.loads(json.dumps(LLM_UPDATE_EXAMPLE))
    payload["evidence_verdicts"] = [
        {"evidence_id": "ev_1", "verdict": "SUPPORT", "note": "swept and reclaimed", "resolves_evidence_id": None, "resolution": None},
        {"evidence_id": "ev_2", "verdict": "NEUTRAL", "note": "", "resolves_evidence_id": None, "resolution": None},
    ]
    payload.update(overrides)
    return json.dumps(payload)


def _actionable(**over) -> dict:
    payload = {
        "state": "ACTIONABLE", "direction": "LONG", "entry_object_id": "FVG_5m_3", "invalidation_object_id": "SSL_5m_2",
        "target_object_id": "BSL_1H_1", "thesis_id": "T1", "governing_timeframe": "15m", "grade": "BASE", "invalidation_mode": "TOUCH",
    }
    payload.update(over)
    return payload


def test_example_reply_parses() -> None:
    update = parse_update(_reply(), evidence_ids=EVIDENCE, object_ids=OBJECTS)
    assert update.evidence_verdicts[0].verdict is Verdict.SUPPORT
    assert update.opportunity.state is OpportunityState.NONE
    assert update.framework_trace["step_1"]
    assert update.to_dict()["continue_active"] is True


def test_resolve_verdict_carries_its_resolution() -> None:
    update = parse_update(
        _reply(evidence_verdicts=[
            {"evidence_id": "ev_1", "verdict": "RESOLVE", "note": "closed", "resolves_evidence_id": "ev_0", "resolution": "CONTRADICT"},
        ]),
        evidence_ids=EVIDENCE, object_ids=OBJECTS,
    )
    assert update.evidence_verdicts[0].resolution is Verdict.CONTRADICT


@pytest.mark.parametrize("bad", [
    "not json", "", "[]", '{"evidence_verdicts": []}',
    _reply(extra_key=1),
    _reply(reasoning_confidence="SURE"),
    _reply(continue_active="yes"),
    _reply(opportunity=_actionable(target_object_id="NOPE_1")),
    _reply(opportunity=_actionable(target_object_id=4500.0)),
    _reply(opportunity={k: v for k, v in _actionable().items() if k != "target_object_id"}),
    _reply(opportunity=_actionable(thesis_id=None)),
    _reply(opportunity=_actionable(thesis_id="has space")),
    _reply(opportunity=_actionable(thesis_id="x" * 33)),
    _reply(opportunity=_actionable(governing_timeframe="1m")),
    _reply(opportunity=_actionable(grade="A+")),
    _reply(opportunity=_actionable(invalidation_mode="CLOSE")),
    _reply(opportunity={**_actionable(), "state": "NONE", "direction": None, "entry_object_id": None, "invalidation_object_id": None, "target_object_id": None}),
    _reply(opportunity={k: v for k, v in _actionable().items() if k != "thesis_id"}),
    _reply(evidence_verdicts=[{"evidence_id": "ev_9", "verdict": "SUPPORT", "note": "", "resolves_evidence_id": None, "resolution": None}]),
    _reply(evidence_verdicts=[{"evidence_id": "ev_1", "verdict": "RESOLVE", "note": "", "resolves_evidence_id": "ev_0", "resolution": None}]),
    _reply(evidence_verdicts=[{"evidence_id": "ev_1", "verdict": "SUPPORT", "note": "", "resolves_evidence_id": None, "resolution": "SUPPORT"}]),
    _reply(evidence_verdicts=[
        {"evidence_id": "ev_1", "verdict": "SUPPORT", "note": "", "resolves_evidence_id": None, "resolution": None},
        {"evidence_id": "ev_1", "verdict": "NEUTRAL", "note": "", "resolves_evidence_id": None, "resolution": None},
    ]),
    _reply(watch_next=[{"object_id": "GHOST_1", "question": "?"}]),
    _reply(destination_candidates=["GHOST_1"]),
    _reply(active_expectation={"thesis": "t", "expected_next": "not a list", "should_not_happen": []}),
    _reply(framework_trace={"step_1": "only one"}),
])
def test_malformed_replies_are_refused(bad: str) -> None:
    with pytest.raises(MalformedReply):
        parse_update(bad, evidence_ids=EVIDENCE, object_ids=OBJECTS)


def test_llm_input_hashes_canonically() -> None:
    known_at = pd.Timestamp("2022-01-04T14:41:00Z")
    a = LLMInput(
        "EP_1", known_at, {"kind": "WAKE", "reasons": ["ev_1"]},
        {"close": 1.0, "high": 1.0, "low": 1.0, "atr_1m": 0.1}, {"name": "rth"}, {"5m": {"z": 1, "a": 2}},
        (), ({"evidence_id": "ev_1"},), {"bars": 0, "counts": {}, "recent": []}, (), None,
    )
    b = LLMInput(
        "EP_1", known_at, {"reasons": ["ev_1"], "kind": "WAKE"},
        {"atr_1m": 0.1, "low": 1.0, "high": 1.0, "close": 1.0}, {"name": "rth"}, {"5m": {"a": 2, "z": 1}},
        (), ({"evidence_id": "ev_1"},), {"recent": [], "counts": {}, "bars": 0}, (), None,
    )
    assert a.input_sha == b.input_sha
    assert json.loads(a.to_json())["known_at"] == "2022-01-04T14:41:00Z"
    assert json.loads(a.to_json())["schema_version"] == 2


def test_actionable_opportunity_carries_thesis_scale_grade_and_mode() -> None:
    update = parse_update(_reply(opportunity=_actionable(grade="A_PLUS", invalidation_mode="CLOSE_BEYOND")), evidence_ids=EVIDENCE, object_ids=OBJECTS)
    o = update.opportunity
    assert o.thesis_id == "T1" and o.governing_timeframe == "15m"
    assert o.grade is ThesisGrade.A_PLUS and o.invalidation_mode is InvalidationMode.CLOSE_BEYOND
    assert update.to_dict()["opportunity"]["invalidation_mode"] == "CLOSE_BEYOND"


def test_example_none_opportunity_carries_null_thesis_fields() -> None:
    assert LLM_UPDATE_EXAMPLE["opportunity"] == {
        "state": "NONE", "direction": None, "entry_object_id": None, "invalidation_object_id": None, "target_object_id": None,
        "thesis_id": None, "governing_timeframe": None, "grade": None, "invalidation_mode": None,
    }


def test_the_example_carries_a_bias_and_a_reply_without_one_is_refused() -> None:
    assert LLM_UPDATE_EXAMPLE["bias"]["direction"] == "NEUTRAL" and LLM_UPDATE_EXAMPLE["bias"]["scale"] == "15m"
    payload = json.loads(_reply())
    del payload["bias"]
    with pytest.raises(MalformedReply):
        parse_update(json.dumps(payload), evidence_ids=EVIDENCE, object_ids=OBJECTS)
    update = parse_update(_reply(bias={"direction": "SHORT", "scale": "1H", "basis": "1H MSS short with the active leg short"}), evidence_ids=EVIDENCE, object_ids=OBJECTS)
    assert update.bias == Bias(BiasDirection.SHORT, "1H", "1H MSS short with the active leg short")
    with pytest.raises(MalformedReply):
        parse_update(_reply(bias={"direction": "SHORT", "scale": "1m", "basis": ""}), evidence_ids=EVIDENCE, object_ids=OBJECTS)
    with pytest.raises(MalformedReply):
        parse_update(_reply(bias={"direction": "SHORT", "scale": "5m", "basis": "the 5m never sets the bias"}), evidence_ids=EVIDENCE, object_ids=OBJECTS)
