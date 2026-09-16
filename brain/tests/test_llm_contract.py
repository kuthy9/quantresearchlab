from __future__ import annotations

import json

import pandas as pd
import pytest

from contract.brain.llm import LLM_UPDATE_EXAMPLE, LLMInput, MalformedReply, parse_update
from contract.brain.state import OpportunityState, Verdict

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
    _reply(opportunity={"state": "ACTIONABLE", "direction": "LONG", "entry_object_id": "FVG_5m_3", "invalidation_object_id": "SSL_5m_2", "target_object_id": "NOPE_1"}),
    _reply(opportunity={"state": "ACTIONABLE", "direction": "LONG", "entry_object_id": "FVG_5m_3", "invalidation_object_id": "SSL_5m_2", "target_object_id": 4500.0}),
    _reply(opportunity={"state": "ACTIONABLE", "direction": "LONG", "entry_object_id": "FVG_5m_3", "invalidation_object_id": "SSL_5m_2"}),
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
    assert json.loads(a.to_json())["schema_version"] == 1
