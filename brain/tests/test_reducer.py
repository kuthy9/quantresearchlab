from __future__ import annotations

import pandas as pd
import pytest

from brain.core.reducer import ReduceContext, apply, empty_state, sleep_blockers
from brain.tests.test_brain_state import T1, make_state
from contract.brain.llm import EvidenceVerdict, LLMUpdate
from contract.brain.state import (
    ActiveExpectation,
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
    )
    base.update(over)
    return ReduceContext(**base)


def upd(**over) -> LLMUpdate:
    base = dict(
        evidence_verdicts=(), understanding_holds=True, market_understanding="same",
        active_expectation=ActiveExpectation("t", (), ()), watch_next=(), destination_candidates=(),
        opportunity=Opportunity(), reasoning_confidence=Confidence.LOW, continue_active=True,
        framework_trace={f"step_{i}": "n/a" for i in range(1, 15)},
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
