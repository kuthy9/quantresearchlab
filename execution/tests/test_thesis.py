from __future__ import annotations

import pandas as pd

from contract.brain.state import TradeDirection
from contract.decision import OpportunityGeometry
from contract.risk import ObjectRef, TradePlan
from execution.core.thesis import ThesisBook, ThesisRecord
from risk.core.gate import ThesisConfig

T0 = pd.Timestamp("2022-01-03T14:12:00Z")


def plan(thesis_id: str = "T1", direction=TradeDirection.SHORT, target_id: str = "swing:c") -> TradePlan:
    return TradePlan(
        "EP", 1, T0, direction,
        ObjectRef("FVG_5m_10", "fvg:a", "fvg", "5m"), ObjectRef("BSL_15m_3", "swing:b", "bsl", "15m"), ObjectRef("SSL_4H_2", target_id, "ssl", "4H"),
        OpportunityGeometry(16387.5, 16411.5, 16330.0, 2.4, ("e", "s", "t")), close=16390.0, thesis_id=thesis_id, governing_timeframe="1H",
    )


def book() -> ThesisBook:
    b = ThesisBook(ThesisConfig(max_expressions=2, stop_cooldown_bars=30))
    b.start_episode("EP")
    return b


def test_first_expression_is_admitted_then_the_same_thesis_is_refused_while_engaged() -> None:
    b = book()
    p = plan()
    assert b.admit(p, 0) is None
    b.expressed(p)
    b.engaged(p.signature, True)
    assert b.admit(plan(target_id="swing:d"), 1) == "thesis_engaged"
    b.engaged(p.signature, False)
    assert b.admit(plan(target_id="swing:d"), 2) is None


def test_a_stop_closes_the_thesis_and_starts_the_cooldown_for_every_thesis() -> None:
    b = book()
    p = plan()
    b.admit(p, 0)
    b.expressed(p)
    b.engaged(p.signature, True)
    b.outcome(p, "position_closed", exit_role="stop", bar_index=10)
    b.engaged(p.signature, False)
    assert b.admit(plan(target_id="swing:d"), 11) == "thesis_closed"
    assert b.admit(plan(thesis_id="T2"), 11) == "stop_cooldown" and b.cooldown_bars_left(11) == 29
    assert b.cooldown_bars_left(40) == 0 and b.admit(plan(thesis_id="T2"), 40) is None
    record = b.view(11)["theses"][0]
    assert record["status"] == "CLOSED" and record["closed_reason"] == "stopped" and record["last_outcome"] == "position_closed:stop"


def test_expressions_are_capped_and_a_target_closes_as_achieved() -> None:
    b = book()
    for i in range(2):
        p = plan(target_id=f"swing:{i}")
        assert b.admit(p, i) is None
        b.expressed(p)
        b.outcome(p, "cancelled", exit_role=None, bar_index=i + 1)  # a cancel spends the expression (an expiry would give it back)
    assert b.admit(plan(target_id="swing:z"), 5) == "expressions_exhausted"
    assert b.view(5)["theses"][0]["closed_reason"] == "expressions_exhausted" and b.view(5)["theses"][0]["expressions"] == 2
    q = plan(thesis_id="T2")
    b.admit(q, 6)
    b.expressed(q)
    b.outcome(q, "position_closed", exit_role="target", bar_index=9)
    assert b.view(9)["theses"][1] == {
        "thesis_id": "T2", "direction": "SHORT", "governing_timeframe": "1H", "status": "CLOSED",
        "closed_reason": "achieved", "expressions": 1, "last_outcome": "position_closed:target",
    }
    assert b.cooldown_bars_left(9) == 0, "a target starts no cooldown"


def test_a_thesis_that_flips_direction_is_closed_and_a_new_episode_forgets_everything() -> None:
    b = book()
    b.admit(plan(), 0)
    assert b.admit(plan(direction=TradeDirection.LONG), 1) == "direction_changed"
    assert b.view(1)["theses"][0]["closed_reason"] == "direction_changed"
    b.start_episode("EP2")
    assert b.admit(plan(direction=TradeDirection.LONG), 2) is None and b.view(2) == {"theses": [{
        "thesis_id": "T1", "direction": "LONG", "governing_timeframe": "1H", "status": "OPEN", "closed_reason": None, "expressions": 0, "last_outcome": None,
    }], "cooldown_bars_left": 0}


def test_a_plan_without_a_thesis_id_is_admitted_and_never_recorded() -> None:
    b = book()
    p = plan(thesis_id="")
    assert b.admit(p, 0) is None
    b.expressed(p)
    b.outcome(p, "position_closed", exit_role="stop", bar_index=3)
    assert b.view(3)["theses"] == [] and b.cooldown_bars_left(4) == 29, "the cooldown still guards the next entry"
    assert isinstance(ThesisRecord("T1", TradeDirection.SHORT, "1H", T0), ThesisRecord)


def test_an_expiry_gives_the_expression_back_but_a_cancel_does_not() -> None:
    b = book()
    first = plan()
    assert b.admit(first, 0) is None
    b.expressed(first)
    b.outcome(first, "expired", exit_role=None, bar_index=16)
    second = plan(target_id="swing:d")
    assert b.admit(second, 17) is None
    b.expressed(second)
    b.outcome(second, "expired", exit_role=None, bar_index=33)
    third = plan(target_id="swing:e")
    assert b.admit(third, 34) is None, "two expiries spent nothing"
    b.expressed(third)
    b.outcome(third, "cancelled", exit_role=None, bar_index=40)
    fourth = plan(target_id="swing:f")
    assert b.admit(fourth, 41) is None
    b.expressed(fourth)
    b.outcome(fourth, "cancelled", exit_role=None, bar_index=43)
    assert b.admit(plan(target_id="swing:h"), 44) == "expressions_exhausted", "two cancels are two expressions"


def test_a_structural_reversal_closes_the_thesis_without_a_cooldown() -> None:
    b = book()
    p = plan()
    assert b.admit(p, 0) is None
    b.expressed(p)
    b.outcome(p, "position_closed", exit_role="structure_reversed", bar_index=10)
    assert b.view(11)["theses"][0]["closed_reason"] == "structure_reversed" and b.cooldown_bars_left(11) == 0
    c = book()
    q = plan(thesis_id="T9")
    assert c.admit(q, 0) is None
    c.expressed(q)
    c.outcome(q, "cancelled", exit_role=None, bar_index=5, reason="structure_reversed")  # the working entry withdrawn on the reversal
    assert c.view(6)["theses"][0]["closed_reason"] == "structure_reversed" and c.admit(plan(thesis_id="T9", target_id="swing:d"), 6) == "thesis_closed"
    assert b.admit(plan(target_id="swing:d"), 11) == "thesis_closed"


def test_a_replacement_gives_the_expression_back_but_a_dropped_plan_does_not() -> None:
    for reason, left in (("signature_changed", 0), ("entry_object_not_visible", 0), ("plan_dropped", 1)):
        b = book()
        p = plan()
        assert b.admit(p, 0) is None
        b.expressed(p)
        b.outcome(p, "cancelled", exit_role=None, bar_index=1, reason=reason)
        assert b.view(1)["theses"][0]["expressions"] == left, reason


def test_an_event_sleep_refunds_a_cancelled_entry_and_closes_a_flattened_thesis_without_a_cooldown() -> None:
    from execution.core.thesis import REPLACEMENT_REASONS

    assert "event_sleep" in REPLACEMENT_REASONS
    b = book()
    p = plan("T1")
    assert b.admit(p, 1) is None
    b.expressed(p)
    b.outcome(p, "cancelled", exit_role=None, bar_index=2, reason="event_sleep")
    assert b.view(2)["theses"][0]["expressions"] == 0 and b.view(2)["theses"][0]["status"] == "OPEN"
    b.expressed(p)
    b.outcome(p, "position_closed", exit_role="event_sleep", bar_index=3)
    record = b.view(3)["theses"][0]
    assert record["closed_reason"] == "event_sleep" and record["last_outcome"] == "position_closed:event_sleep"
    assert b.view(3)["cooldown_bars_left"] == 0 and b.admit(p, 4) == "thesis_closed"
