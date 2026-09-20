from __future__ import annotations

import json

import pandas as pd
import pytest

from contract.brain.state import (
    ActiveExpectation,
    Bias,
    BiasDirection,
    BrainState,
    BrainStatus,
    Confidence,
    EvidenceItem,
    EvidenceLedger,
    InvalidationMode,
    LastUpdate,
    Opportunity,
    OpportunityState,
    RegisteredObject,
    ThesisGrade,
    TradeDirection,
    Verdict,
    WatchItem,
)

T0 = pd.Timestamp("2022-01-04T14:35:00Z")
T1 = pd.Timestamp("2022-01-04T14:41:00Z")


def make_state(**overrides) -> BrainState:
    base = dict(
        episode_id="EP_20220104_001",
        status=BrainStatus.ACTIVE,
        revision=1,
        started_at=T0,
        updated_at=T1,
        market_understanding="5m in a retracement inside the 15m range",
        active_expectation=ActiveExpectation(
            "expect a return to FVG_5m_3", ("hold above SSL_5m_2",), ("close below SSL_5m_2",)
        ),
        evidence=EvidenceLedger(
            supporting=(
                EvidenceItem("ev_1", T1, "sweep_confirmed", "5m", "SSL_5m_2", Verdict.SUPPORT, "swept"),
            )
        ),
        watch_next=(WatchItem("FVG_5m_3", "does it hold?"),),
        destination_candidates=("BSL_1H_1",),
        opportunity=Opportunity(),
        reasoning_confidence=Confidence.LOW,
        continue_active=True,
        object_registry={
            "FVG_5m_3": RegisteredObject("a" * 24, "fvg", "5m"),
            "SSL_5m_2": RegisteredObject("b" * 24, "ssl", "5m"),
            "BSL_1H_1": RegisteredObject("c" * 24, "bsl", "1H"),
        },
        last_update=LastUpdate(T1, True, {"SUPPORT": 1, "CONTRADICT": 0, "NEUTRAL": 0, "RESOLVE": 0}),
    )
    base.update(overrides)
    return BrainState(**base)


def test_round_trips_through_json_with_sorted_keys() -> None:
    state = make_state()
    text = state.to_json()
    assert BrainState.from_json(text) == state
    payload = json.loads(text)
    assert list(payload) == sorted(payload)
    assert payload["started_at"] == "2022-01-04T14:35:00Z"
    assert payload["opportunity"] == {
        "state": "NONE",
        "direction": None,
        "entry_object_id": None,
        "invalidation_object_id": None,
        "target_object_id": None,
        "thesis_id": None,
        "governing_timeframe": None,
        "grade": None,
        "invalidation_mode": None,
    }
    assert payload["object_registry"]["FVG_5m_3"] == {"entity_id": "a" * 24, "kind": "fvg", "timeframe": "5m"}


def test_opportunity_requires_registered_ids_and_a_direction() -> None:
    with pytest.raises(ValueError, match="opportunity"):
        make_state(
            opportunity=Opportunity(
                OpportunityState.DEVELOPING, TradeDirection.LONG, "FVG_5m_3", "SSL_5m_2", "BSL_9H_9"
            )
        )
    with pytest.raises(ValueError, match="direction"):
        make_state(
            opportunity=Opportunity(OpportunityState.DEVELOPING, None, "FVG_5m_3", "SSL_5m_2", "BSL_1H_1")
        )
    with pytest.raises(ValueError, match="distinct"):
        make_state(
            opportunity=Opportunity(
                OpportunityState.DEVELOPING, TradeDirection.LONG, "FVG_5m_3", "FVG_5m_3", "BSL_1H_1"
            )
        )
    make_state(
        opportunity=Opportunity(
            OpportunityState.DEVELOPING, TradeDirection.LONG, "FVG_5m_3", "SSL_5m_2", "BSL_1H_1"
        )
    )


def test_evidence_ids_are_unique_and_timestamps_ordered() -> None:
    dup = EvidenceItem("ev_1", T1, "x", "5m", None, Verdict.NEUTRAL)
    with pytest.raises(ValueError, match="evidence"):
        make_state(evidence=EvidenceLedger(supporting=(dup,), unresolved=(dup,)))
    with pytest.raises(ValueError, match="updated_at"):
        make_state(updated_at=T0 - pd.Timedelta(minutes=1))


def test_watch_and_destination_aliases_must_be_registered() -> None:
    with pytest.raises(ValueError, match="watch_next"):
        make_state(watch_next=(WatchItem("GHOST_1", "?"),))
    with pytest.raises(ValueError, match="destination"):
        make_state(destination_candidates=("GHOST_1",))


def test_naive_timestamps_are_refused() -> None:
    with pytest.raises(ValueError):
        make_state(started_at=pd.Timestamp("2022-01-04T14:35:00"))


def test_from_dict_rejects_unknown_keys_and_other_schemas() -> None:
    payload = make_state().to_dict()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        BrainState.from_dict(payload)
    payload = make_state().to_dict()
    payload["schema_version"] = 3
    with pytest.raises(ValueError, match="schema"):
        BrainState.from_dict(payload)


def test_opportunity_from_dict_defaults_the_thesis_fields_of_an_old_state() -> None:
    o = Opportunity.from_dict({"state": "ACTIONABLE", "direction": "SHORT", "entry_object_id": "a", "invalidation_object_id": "b", "target_object_id": "c"})
    assert o.thesis_id is None and o.governing_timeframe is None
    assert o.grade is ThesisGrade.BASE and o.invalidation_mode is InvalidationMode.TOUCH
    assert Opportunity.from_dict(o.to_dict()) == o
    full = Opportunity("ACTIONABLE", "SHORT", "a", "b", "c", thesis_id="T-1", governing_timeframe="4H", grade="A_PLUS", invalidation_mode="CLOSE_BEYOND")
    assert Opportunity.from_dict(full.to_dict()) == full and full.to_dict()["grade"] == "A_PLUS"
    assert Opportunity().to_dict()["thesis_id"] is None and Opportunity().to_dict()["grade"] is None


@pytest.mark.parametrize("bad", [
    dict(thesis_id="bad id"), dict(governing_timeframe="1m"), dict(thesis_id="x" * 33),
])
def test_opportunity_refuses_bad_thesis_fields(bad) -> None:
    fields = dict(thesis_id="T1", governing_timeframe="15m")
    fields.update(bad)
    with pytest.raises(ValueError):
        Opportunity("ACTIONABLE", "SHORT", "a", "b", "c", **fields)
    with pytest.raises(ValueError, match="NONE"):
        Opportunity("NONE", thesis_id="T1")


def test_bias_round_trips_and_a_schema_1_state_reads_as_neutral() -> None:
    state = make_state(bias=Bias(BiasDirection.LONG, "15m", "15m active leg long past one ATR after the MSS"))
    again = BrainState.from_json(state.to_json())
    assert again.bias == state.bias and again.schema_version == 2
    payload = json.loads(state.to_json())
    del payload["bias"]
    payload["schema_version"] = 1
    old = BrainState.from_dict(payload)
    assert old.bias == Bias() and old.schema_version == 2


@pytest.mark.parametrize("bad", [dict(scale="1m"), dict(scale="5m"), dict(scale="4h"), dict(direction="UP")])
def test_bias_refuses_a_bad_scale_or_direction(bad) -> None:
    """The 5m is an execution scale, never the bias scale (run B set it 18 times)."""
    with pytest.raises(ValueError):
        Bias(**{"direction": BiasDirection.LONG, "scale": "15m", "basis": "", **bad})
