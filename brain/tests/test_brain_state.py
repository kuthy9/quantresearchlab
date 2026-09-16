from __future__ import annotations

import json

import pandas as pd
import pytest

from contract.brain.state import (
    ActiveExpectation,
    BrainState,
    BrainStatus,
    Confidence,
    EvidenceItem,
    EvidenceLedger,
    LastUpdate,
    Opportunity,
    OpportunityState,
    RegisteredObject,
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
    payload["schema_version"] = 2
    with pytest.raises(ValueError, match="schema"):
        BrainState.from_dict(payload)
