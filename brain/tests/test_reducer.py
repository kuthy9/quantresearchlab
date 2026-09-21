from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from brain.core.reducer import ReduceContext, apply, empty_state, sleep_blockers
from brain.tests.test_brain_state import T1, make_state
from contract.brain.llm import EvidenceVerdict, LLMUpdate
from contract.brain.state import (
    ActiveExpectation,
    Bias,
    BiasDirection,
    Confidence,
    EvidenceItem,
    EvidenceLedger,
    Opportunity,
    OpportunityState,
    RegisteredObject,
    TradeDirection,
    Verdict,
    WatchItem,
)

REG = {
    "FVG_5m_3": RegisteredObject("a" * 24, "fvg", "5m"),
    "SSL_5m_2": RegisteredObject("b" * 24, "ssl", "5m"),
    "BSL_1H_1": RegisteredObject("c" * 24, "bsl", "1H"),
}
T2 = T1 + pd.Timedelta(minutes=1)


def ctx(**over) -> ReduceContext:
    base = dict(
        known_at=T2, has_open_position=False, open_interaction=False,
        visible_aliases=frozenset(REG), registry=REG, coherence=lambda o: None,
        idle_updates=0, idle_archive_after=None,
    )
    base.update(over)
    return ReduceContext(**base)


def upd(**over) -> LLMUpdate:
    base = dict(
        evidence_verdicts=(), understanding_holds=True, market_understanding="same",
        active_expectation=ActiveExpectation("t", (), ()), watch_next=(), destination_candidates=(),
        opportunity=Opportunity(), reasoning_confidence=Confidence.LOW, continue_active=True,
        framework_trace={f"step_{i}": "n/a" for i in range(1, 15)},
        bias=Bias(BiasDirection.LONG, "4H", "scripted"),  # the tests' opportunities are LONG; 4H admits every governing scale
    )
    base.update(over)
    return LLMUpdate(**base)


def item(i: str, kind: str = "sweep_confirmed") -> EvidenceItem:
    return EvidenceItem(i, T2, kind, "5m", "SSL_5m_2")


def quiet(**over):
    base = dict(watch_next=(), destination_candidates=())
    base.update(over)
    return make_state(**base)


def test_wake_builds_revision_zero_from_the_update() -> None:
    r = apply(
        None, episode_id="EP_1", evidence=[item("ev_11")],
        update=upd(evidence_verdicts=(EvidenceVerdict("ev_11", Verdict.SUPPORT, "swept"),), market_understanding="fresh"),
        ctx=ctx(),
    )
    assert r.state.revision == 0 and r.state.started_at == T2 and r.state.market_understanding == "fresh"
    assert [e.evidence_id for e in r.state.evidence.supporting] == ["ev_11"] and r.rejections == ()
    assert r.state.evidence.supporting[0].note == "swept"


def test_verdicts_route_and_resolve() -> None:
    prev = quiet(evidence=EvidenceLedger(unresolved=(item("ev_0", "level_reached"),)))
    verdicts = (
        EvidenceVerdict("ev_11", Verdict.CONTRADICT, "no"),
        EvidenceVerdict("ev_12", Verdict.NEUTRAL, ""),
        EvidenceVerdict("ev_13", Verdict.RESOLVE, "closed", resolves_evidence_id="ev_0", resolution=Verdict.SUPPORT),
    )
    r = apply(
        prev, episode_id=prev.episode_id, evidence=[item("ev_11"), item("ev_12"), item("ev_13")],
        update=upd(evidence_verdicts=verdicts), ctx=ctx(),
    )
    s = r.state
    assert s.revision == prev.revision + 1 and s.updated_at == T2 and r.rejections == ()
    assert {e.evidence_id for e in s.evidence.supporting} == {"ev_13"}
    assert {e.evidence_id for e in s.evidence.contradicting} == {"ev_11"}
    assert {e.evidence_id for e in s.evidence.unresolved} == {"ev_12"}
    assert s.last_update.verdicts == {"SUPPORT": 0, "CONTRADICT": 1, "NEUTRAL": 1, "RESOLVE": 1}


def test_resolve_of_an_unknown_target_is_recorded() -> None:
    prev = quiet()
    verdicts = (EvidenceVerdict("ev_11", Verdict.RESOLVE, "", resolves_evidence_id="ev_nope", resolution=Verdict.CONTRADICT),)
    r = apply(prev, episode_id=prev.episode_id, evidence=[item("ev_11")], update=upd(evidence_verdicts=verdicts), ctx=ctx())
    assert "resolve_target_not_unresolved:ev_nope" in r.rejections
    assert {e.evidence_id for e in r.state.evidence.contradicting} == {"ev_11"}


def test_missing_verdict_lands_in_unresolved_with_a_rejection() -> None:
    prev = quiet()
    r = apply(prev, episode_id=prev.episode_id, evidence=[item("ev_11")], update=upd(), ctx=ctx())
    assert r.rejections == ("evidence_without_verdict:ev_11",)
    assert r.state.evidence.unresolved[0].evidence_id == "ev_11"


def test_understanding_not_replaced_rejects_the_update() -> None:
    prev = quiet()
    r = apply(
        prev, episode_id=prev.episode_id, evidence=[item("ev_11")],
        update=upd(understanding_holds=False, market_understanding=prev.market_understanding, active_expectation=prev.active_expectation),
        ctx=ctx(),
    )
    assert r.incident == "understanding_not_replaced"
    assert r.state.revision == prev.revision + 1 and r.state.market_understanding == prev.market_understanding
    assert r.state.last_update.llm_called is True and "evidence_without_verdict:ev_11" in r.rejections


def test_replaced_understanding_is_accepted() -> None:
    prev = quiet()
    r = apply(
        prev, episode_id=prev.episode_id, evidence=[],
        update=upd(understanding_holds=False, market_understanding="new reading", active_expectation=ActiveExpectation("new thesis")),
        ctx=ctx(),
    )
    assert r.incident is None and r.state.market_understanding == "new reading"


def test_opportunity_downgrades_on_unknown_alias_or_incoherence() -> None:
    prev = quiet()
    bad = Opportunity(OpportunityState.ACTIONABLE, TradeDirection.LONG, "FVG_5m_3", "SSL_5m_2", "BSL_1H_1")
    r = apply(prev, episode_id=prev.episode_id, evidence=[], update=upd(opportunity=bad), ctx=ctx(visible_aliases=frozenset({"FVG_5m_3", "SSL_5m_2"})))
    assert r.state.opportunity == Opportunity() and r.rejections == ("opportunity_object_not_visible:BSL_1H_1",)
    r = apply(prev, episode_id=prev.episode_id, evidence=[], update=upd(opportunity=bad), ctx=ctx(coherence=lambda o: "target below entry"))
    assert r.state.opportunity == Opportunity() and r.rejections == ("opportunity_incoherent:target below entry",)
    r = apply(prev, episode_id=prev.episode_id, evidence=[], update=upd(opportunity=bad), ctx=ctx())
    assert r.state.opportunity == bad and r.rejections == ()


def test_open_position_forces_active() -> None:
    prev = quiet()
    r = apply(prev, episode_id=prev.episode_id, evidence=[], update=upd(continue_active=False), ctx=ctx(has_open_position=True))
    assert r.state.continue_active is True and r.slept is False and "sleep_refused:open_position" in r.rejections


@pytest.mark.parametrize("blocker,over,ctx_over", [
    ("open_interaction", {}, {"open_interaction": True}),
    ("unresolved_evidence", {"evidence": EvidenceLedger(unresolved=(item("ev_0"),))}, {}),
    ("watch_next", {"watch_next": (WatchItem("FVG_5m_3", "?"),)}, {}),
    ("opportunity", {"opportunity": Opportunity(OpportunityState.DEVELOPING, TradeDirection.LONG, "FVG_5m_3", "SSL_5m_2", "BSL_1H_1")}, {}),
])
def test_each_sleep_condition_blocks(blocker, over, ctx_over) -> None:
    prev = quiet(**over)
    update = upd(continue_active=False, watch_next=prev.watch_next, opportunity=prev.opportunity)
    r = apply(prev, episode_id=prev.episode_id, evidence=[], update=update, ctx=ctx(**ctx_over))
    assert r.slept is False and f"sleep_refused:{blocker}" in r.rejections
    assert r.state.continue_active is True


def test_sleep_when_all_five_hold() -> None:
    prev = quiet()
    r = apply(prev, episode_id=prev.episode_id, evidence=[], update=upd(continue_active=False), ctx=ctx())
    assert r.slept is True and r.state.continue_active is False and r.state.status.value == "ARCHIVED"
    assert sleep_blockers(r.state, has_open_position=False, open_interaction=False) == ()


def test_tick_only_advances_bookkeeping() -> None:
    prev = quiet()
    r = apply(prev, episode_id=prev.episode_id, evidence=[], update=None, ctx=ctx())
    assert r.state.revision == prev.revision + 1 and r.state.last_update.llm_called is False
    assert r.state.market_understanding == prev.market_understanding and r.state.evidence == prev.evidence
    assert r.state.updated_at == T2 and r.rejections == ()


def test_llm_incident_carries_forward_with_evidence_unresolved() -> None:
    prev = quiet()
    r = apply(prev, episode_id=prev.episode_id, evidence=[item("ev_11")], update=None, ctx=ctx(), incident="LLMTimeout")
    assert r.incident == "LLMTimeout" and r.state.last_update.llm_called is True
    assert r.state.last_update.incident == "LLMTimeout"
    assert [e.evidence_id for e in r.state.evidence.unresolved] == ["ev_11"]


def test_incident_on_wake_opens_an_empty_state() -> None:
    s = empty_state("EP_1", T2, REG, incident="LLMTimeout")
    assert s.revision == 0 and s.continue_active is True and s.last_update.incident == "LLMTimeout"
    r = apply(None, episode_id="EP_1", evidence=[item("ev_11")], update=None, ctx=ctx(), incident="LLMTimeout")
    assert r.state == s


def test_unknown_watch_alias_is_dropped_with_a_rejection() -> None:
    prev = quiet()
    r = apply(
        prev, episode_id=prev.episode_id, evidence=[],
        update=upd(watch_next=(WatchItem("GHOST_1", "?"),), destination_candidates=("GHOST_2",)), ctx=ctx(),
    )
    assert r.state.watch_next == () and set(r.rejections) == {"watch_object_unknown:GHOST_1", "destination_object_unknown:GHOST_2"}


def test_duplicate_evidence_is_skipped_with_a_rejection() -> None:
    prev = quiet()  # already holds ev_1 in supporting
    r = apply(prev, episode_id=prev.episode_id, evidence=[item("ev_1")], update=upd(evidence_verdicts=(EvidenceVerdict("ev_1", Verdict.CONTRADICT, ""),)), ctx=ctx())
    assert r.rejections == ("evidence_duplicate:ev_1",)
    assert [e.evidence_id for e in r.state.evidence.contradicting] == []
    r = apply(prev, episode_id=prev.episode_id, evidence=[item("ev_1")], update=None, ctx=ctx(), incident="LLMTimeout")
    assert r.rejections == ("evidence_duplicate:ev_1",) and r.state.evidence == prev.evidence


def test_registry_grows_across_revisions() -> None:
    prev = quiet()
    newer = dict(REG, OB_15m_1=RegisteredObject("d" * 24, "ob", "15m"))
    r = apply(prev, episode_id=prev.episode_id, evidence=[], update=upd(), ctx=ctx(registry=newer, visible_aliases=frozenset(newer)))
    assert set(r.state.object_registry) == set(newer)


def test_determinism() -> None:
    prev = quiet()
    kwargs = dict(episode_id=prev.episode_id, evidence=[item("ev_11")], update=upd(evidence_verdicts=(EvidenceVerdict("ev_11", Verdict.SUPPORT, "x"),)), ctx=ctx())
    assert apply(prev, **kwargs).state.to_json() == apply(prev, **kwargs).state.to_json()


def pending(i: str) -> EvidenceItem:
    """Evidence the LLM never saw: parked by an incident bar, verdict None."""
    return item(i)


def neutral(i: str) -> EvidenceItem:
    return EvidenceItem(i, T2, "sweep_confirmed", "5m", "SSL_5m_2", Verdict.NEUTRAL, "not bearing on the thesis")


def test_neutral_evidence_is_judged_and_does_not_block_sleep() -> None:
    prev = quiet(evidence=EvidenceLedger(unresolved=(neutral("ev_7"),)))
    assert sleep_blockers(prev, has_open_position=False, open_interaction=False) == ()
    r = apply(prev, episode_id=prev.episode_id, evidence=[], update=upd(continue_active=False), ctx=ctx())
    assert r.slept is True and r.rejections == ()
    assert [e.evidence_id for e in r.state.evidence.unresolved] == ["ev_7"]


def test_pending_evidence_blocks_sleep_until_it_is_verdicted() -> None:
    prev = quiet(evidence=EvidenceLedger(unresolved=(pending("ev_7"),)))
    assert sleep_blockers(prev, has_open_position=False, open_interaction=False) == ("unresolved_evidence",)
    r = apply(prev, episode_id=prev.episode_id, evidence=[item("ev_7")], update=upd(continue_active=False), ctx=ctx())
    assert r.slept is False
    assert set(r.rejections) == {"evidence_without_verdict:ev_7", "sleep_refused:unresolved_evidence"}
    assert [e.evidence_id for e in r.state.evidence.unresolved] == ["ev_7"]


def test_pending_evidence_reoffered_takes_its_verdict_without_a_duplicate() -> None:
    prev = quiet(evidence=EvidenceLedger(unresolved=(pending("ev_7"),)))
    r = apply(
        prev, episode_id=prev.episode_id, evidence=[item("ev_7"), item("ev_8")],
        update=upd(
            evidence_verdicts=(EvidenceVerdict("ev_7", Verdict.SUPPORT, "late"), EvidenceVerdict("ev_8", Verdict.CONTRADICT, "")),
            continue_active=False,
        ),
        ctx=ctx(),
    )
    assert r.rejections == () and r.slept is True
    assert [e.evidence_id for e in r.state.evidence.supporting] == ["ev_7"]
    assert [e.evidence_id for e in r.state.evidence.contradicting] == ["ev_8"]
    assert r.state.evidence.unresolved == ()
    assert r.state.last_update.verdicts == {"SUPPORT": 1, "CONTRADICT": 1, "NEUTRAL": 0, "RESOLVE": 0}


def test_pending_evidence_resolved_by_a_carrier_is_not_re_filed() -> None:
    prev = quiet(evidence=EvidenceLedger(unresolved=(pending("ev_7"),)))
    r = apply(
        prev, episode_id=prev.episode_id, evidence=[item("ev_8"), item("ev_7")],
        update=upd(evidence_verdicts=(EvidenceVerdict("ev_8", Verdict.RESOLVE, "", "ev_7", Verdict.CONTRADICT),)),
        ctx=ctx(),
    )
    assert r.rejections == ()
    assert [e.evidence_id for e in r.state.evidence.contradicting] == ["ev_8"]
    assert r.state.evidence.unresolved == ()


def test_pending_evidence_stays_pending_through_another_incident() -> None:
    prev = quiet(evidence=EvidenceLedger(unresolved=(pending("ev_7"),)))
    r = apply(
        prev, episode_id=prev.episode_id, evidence=[item("ev_7"), item("ev_8")], update=None, ctx=ctx(), incident="LLMTimeout",
    )
    assert r.rejections == ("evidence_without_verdict:ev_8",)
    assert [e.evidence_id for e in r.state.evidence.unresolved] == ["ev_7", "ev_8"]


def test_idle_archive_after_n_updates_without_opportunity_or_new_understanding() -> None:
    prev = make_state()  # watch_next is non-empty: the LLM would never be allowed to sleep on its own
    r = apply(prev, episode_id=prev.episode_id, evidence=[], update=upd(market_understanding=prev.market_understanding), ctx=ctx(idle_updates=5, idle_archive_after=6))
    assert r.slept is True and r.sleep_reason == "idle" and r.state.status.value == "ARCHIVED" and r.state.continue_active is False
    r = apply(prev, episode_id=prev.episode_id, evidence=[], update=upd(), ctx=ctx(idle_updates=4, idle_archive_after=6))
    assert r.slept is False and r.sleep_reason is None
    r = apply(quiet(), episode_id=prev.episode_id, evidence=[], update=upd(continue_active=False), ctx=ctx())
    assert r.slept is True and r.sleep_reason == "continue_active=false"


@pytest.mark.parametrize("over,ctx_over", [
    ({"opportunity": Opportunity(OpportunityState.DEVELOPING, TradeDirection.LONG, "FVG_5m_3", "SSL_5m_2", "BSL_1H_1")}, {}),
    ({"understanding_holds": False, "market_understanding": "new", "active_expectation": ActiveExpectation("new t", (), ())}, {}),
    ({}, {"has_open_position": True}),
    ({}, {"idle_archive_after": None}),
])
def test_idle_archive_is_not_granted_when(over, ctx_over) -> None:
    prev = make_state()
    base = dict(idle_updates=5, idle_archive_after=6); base.update(ctx_over)
    r = apply(prev, episode_id=prev.episode_id, evidence=[], update=upd(**over), ctx=ctx(**base))
    assert r.slept is False


def test_idle_archive_waits_for_pending_evidence() -> None:
    prev = make_state(evidence=EvidenceLedger(unresolved=(pending("ev_7"),)))
    r = apply(prev, episode_id=prev.episode_id, evidence=[], update=upd(), ctx=ctx(idle_updates=5, idle_archive_after=6))
    assert r.slept is False


def _scale(alias: str) -> str:
    return alias.split("_")[-2]


def test_invalidation_more_than_one_scale_below_the_governing_one_is_refused() -> None:
    opportunity = Opportunity("ACTIONABLE", "SHORT", "FVG_5m_3", "SSL_5m_2", "BSL_1H_1", thesis_id="T1", governing_timeframe="1H")
    res = apply(None, episode_id="EP", evidence=(), update=upd(opportunity=opportunity), ctx=ctx(timeframe_of=_scale))
    assert "opportunity_invalidation_scale:SSL_5m_2" in res.rejections and res.state.opportunity.state is OpportunityState.NONE


def test_invalidation_on_the_governing_scale_or_one_below_is_accepted() -> None:
    opportunity = Opportunity("ACTIONABLE", "SHORT", "FVG_5m_3", "SSL_5m_2", "BSL_1H_1", thesis_id="T1", governing_timeframe="15m")
    short = Bias(BiasDirection.SHORT, "4H", "b")
    res = apply(None, episode_id="EP", evidence=(), update=upd(opportunity=opportunity, bias=short), ctx=ctx(timeframe_of=_scale))
    assert res.state.opportunity.state is OpportunityState.ACTIONABLE and res.rejections == ()
    unknown = apply(None, episode_id="EP", evidence=(), update=upd(opportunity=opportunity, bias=short), ctx=ctx(timeframe_of=lambda alias: None))
    assert unknown.state.opportunity.state is OpportunityState.ACTIONABLE, "an object of unknown scale is not judged"


def test_a_thesis_id_that_flips_direction_is_refused() -> None:
    first = apply(None, episode_id="EP", evidence=(), update=upd(opportunity=Opportunity("DEVELOPING", "SHORT", "FVG_5m_3", "SSL_5m_2", "BSL_1H_1", thesis_id="T1", governing_timeframe="5m"), bias=Bias(BiasDirection.SHORT, "4H", "b")), ctx=ctx())
    flipped = Opportunity("ACTIONABLE", "LONG", "BSL_1H_1", "SSL_5m_2", "FVG_5m_3", thesis_id="T1", governing_timeframe="5m")
    res = apply(first.state, episode_id="EP", evidence=(), update=upd(opportunity=flipped), ctx=ctx())
    assert "thesis_direction_changed:T1" in res.rejections and res.state.opportunity.state is OpportunityState.NONE
    renamed = apply(first.state, episode_id="EP", evidence=(), update=upd(opportunity=replace(flipped, thesis_id="T2")), ctx=ctx())
    assert renamed.state.opportunity.state is OpportunityState.ACTIONABLE


# ------------------------------------------------------------ rule 4b: the bias (2026-09-18)

def _opp(direction=TradeDirection.LONG, tf="5m"):
    return Opportunity(OpportunityState.DEVELOPING, direction, "FVG_5m_3", "SSL_5m_2", "BSL_1H_1", thesis_id="T1", governing_timeframe=tf)


def test_an_opportunity_against_the_bias_is_dropped_and_the_bias_kept() -> None:
    result = apply(quiet(), episode_id="EP", evidence=(), update=upd(opportunity=_opp(TradeDirection.SHORT), bias=Bias(BiasDirection.LONG, "15m", "b")), ctx=ctx())
    assert result.state.opportunity == Opportunity() and result.state.bias.direction is BiasDirection.LONG
    assert "opportunity_against_bias:SHORT" in result.rejections


def test_no_opportunity_under_a_neutral_bias() -> None:
    result = apply(quiet(), episode_id="EP", evidence=(), update=upd(opportunity=_opp(), bias=Bias()), ctx=ctx())
    assert result.state.opportunity == Opportunity() and "opportunity_against_bias:NEUTRAL" in result.rejections


def test_a_thesis_above_the_bias_scale_is_dropped() -> None:
    update = upd(opportunity=_opp(tf="1H"), bias=Bias(BiasDirection.LONG, "15m", "b"))
    result = apply(quiet(), episode_id="EP", evidence=(), update=update, ctx=ctx())
    assert result.state.opportunity == Opportunity() and "opportunity_scale_above_bias:1H" in result.rejections
    update = upd(opportunity=_opp(tf="5m"), bias=Bias(BiasDirection.LONG, "15m", "b"))
    assert apply(quiet(), episode_id="EP", evidence=(), update=update, ctx=ctx()).state.opportunity.state is OpportunityState.DEVELOPING


def test_a_carried_forward_state_keeps_its_bias() -> None:
    prev = quiet(bias=Bias(BiasDirection.SHORT, "1H", "b"))
    result = apply(prev, episode_id="EP", evidence=(), update=None, ctx=ctx())
    assert result.state.bias == prev.bias


def test_pending_evidence_is_bounded_and_the_oldest_expires() -> None:
    """Run X (2026-09-19): 25 empty replies in a row re-offered 124 pending
    items and the input grew until no reply could come.  The ledger keeps
    the newest ``max_pending`` pending items; the rest expire, journaled."""
    old = tuple(item(f"ev_p{i}", "level_reached") for i in range(3))
    prev = quiet(evidence=EvidenceLedger(unresolved=old))
    # an incident bar (no update) carries forward and files one more unjudged item
    r = apply(prev, episode_id="EP_1", evidence=[item("ev_new")], update=None, ctx=ctx(max_pending=2))
    pending = [e.evidence_id for e in r.state.evidence.unresolved if e.verdict is None]
    assert pending == ["ev_p2", "ev_new"]
    assert "evidence_expired:ev_p0" in r.rejections and "evidence_expired:ev_p1" in r.rejections
    # an update that leaves them unjudged is bounded the same way
    r2 = apply(r.state, episode_id="EP_1", evidence=[item("ev_p2"), item("ev_new"), item("ev_later")], update=upd(evidence_verdicts=()), ctx=ctx(max_pending=2))
    pending = [e.evidence_id for e in r2.state.evidence.unresolved if e.verdict is None]
    assert pending == ["ev_new", "ev_later"] and "evidence_expired:ev_p2" in r2.rejections
    # unbounded by default
    r3 = apply(prev, episode_id="EP_1", evidence=[item("ev_new")], update=None, ctx=ctx())
    assert len([e for e in r3.state.evidence.unresolved if e.verdict is None]) == 4


def test_the_state_records_why_the_opportunity_was_dropped() -> None:
    prev = quiet()
    bad = Opportunity(OpportunityState.ACTIONABLE, TradeDirection.LONG, "FVG_5m_3", "SSL_5m_2", "BSL_1H_1", thesis_id="T1", governing_timeframe="5m")
    r = apply(prev, episode_id=prev.episode_id, evidence=[], update=upd(opportunity=bad), ctx=ctx(coherence=lambda o: "LONG entry 105 lies above price 103"))
    assert r.state.opportunity.state is OpportunityState.NONE
    assert r.state.last_update.rejections == ("opportunity_incoherent:LONG entry 105 lies above price 103",) == r.rejections
    clean = apply(r.state, episode_id=prev.episode_id, evidence=[], update=upd(), ctx=ctx())
    assert clean.state.last_update.rejections == ()
