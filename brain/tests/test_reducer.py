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
    # 2026-09-22: a RESOLVE files the resolved item under its resolution too — judged, not erased (the decay reads it)
    assert {e.evidence_id for e in s.evidence.supporting} == {"ev_13", "ev_0"}
    assert next(e for e in s.evidence.supporting if e.evidence_id == "ev_0").verdict is Verdict.SUPPORT
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
    assert r.state.opportunity == replace(bad, governing_timeframe="1H") and r.rejections == ()  # the thesis scale of the 4H bias


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
    assert [e.evidence_id for e in r.state.evidence.contradicting] == ["ev_7", "ev_8"]  # the resolved item filed under its resolution, then the carrier
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
    opportunity = Opportunity("ACTIONABLE", "SHORT", "FVG_5m_3", "SSL_5m_2", "BSL_1H_1", thesis_id="T1")
    res = apply(None, episode_id="EP", evidence=(), update=upd(opportunity=opportunity, bias=Bias(BiasDirection.SHORT, "4H", "b")), ctx=ctx(timeframe_of=_scale))  # a 1H thesis
    assert "opportunity_invalidation_scale:SSL_5m_2" in res.rejections and res.state.opportunity.state is OpportunityState.NONE


def test_invalidation_on_the_governing_scale_or_one_below_is_accepted() -> None:
    opportunity = Opportunity("ACTIONABLE", "SHORT", "FVG_5m_3", "SSL_5m_2", "BSL_1H_1", thesis_id="T1")
    short = Bias(BiasDirection.SHORT, "1H", "b")  # a 15m thesis: a 5m invalidation is one below
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


# ------------------------------------------------------------ the thesis scale and the target scale (2026-09-22)

def _tf(alias: str) -> str:
    return alias.split("_")[-2]


def test_the_thesis_scale_is_the_biass_and_the_reply_cannot_set_it() -> None:
    opp = Opportunity(OpportunityState.ACTIONABLE, TradeDirection.LONG, "FVG_5m_3", "SSL_5m_2", "BSL_1H_1", thesis_id="T1")
    res = apply(None, episode_id="EP", evidence=[], update=upd(opportunity=opp, bias=Bias(BiasDirection.LONG, "1H", "b")), ctx=ctx(timeframe_of=_tf))
    assert res.state.opportunity.governing_timeframe == "15m" and res.state.opportunity.state is OpportunityState.ACTIONABLE and res.rejections == ()
    res = apply(None, episode_id="EP", evidence=[], update=upd(opportunity=opp, bias=Bias(BiasDirection.LONG, "15m", "b")), ctx=ctx(timeframe_of=_tf))
    assert res.state.opportunity.governing_timeframe == "15m"
    res = apply(None, episode_id="EP", evidence=[], update=upd(opportunity=opp, bias=Bias(BiasDirection.LONG, "4H", "b")), ctx=ctx(timeframe_of=_tf))
    assert "opportunity_invalidation_scale:SSL_5m_2" in res.rejections, "a 1H thesis is not falsified by a 5m pool"
    # the reply's value, if a caller still passes one, is overwritten, not judged
    res = apply(None, episode_id="EP", evidence=[], update=upd(opportunity=replace(opp, governing_timeframe="5m"), bias=Bias(BiasDirection.LONG, "1H", "b")), ctx=ctx(timeframe_of=_tf))
    assert res.state.opportunity.governing_timeframe == "15m" and not any(r.startswith("opportunity_scale") for r in res.rejections)


def test_a_target_below_the_thesis_scale_is_refused() -> None:
    reg = {**REG, "BSL_5m_9": RegisteredObject("d" * 24, "bsl", "5m")}
    opp = Opportunity(OpportunityState.ACTIONABLE, TradeDirection.LONG, "FVG_5m_3", "SSL_5m_2", "BSL_5m_9", thesis_id="T1")
    res = apply(None, episode_id="EP", evidence=[], update=upd(opportunity=opp, bias=Bias(BiasDirection.LONG, "15m", "b")),
                ctx=ctx(registry=reg, visible_aliases=frozenset(reg), timeframe_of=_tf))
    assert "opportunity_target_scale:BSL_5m_9" in res.rejections and res.state.opportunity == Opportunity()
    unknown = apply(None, episode_id="EP", evidence=[], update=upd(opportunity=opp, bias=Bias(BiasDirection.LONG, "15m", "b")),
                    ctx=ctx(registry=reg, visible_aliases=frozenset(reg), timeframe_of=lambda alias: None))
    assert unknown.state.opportunity.state is OpportunityState.ACTIONABLE, "an object of unknown scale is not judged"


# ------------------------------------------------------------ rule 4d: the bias decay (2026-09-22)

def struct(i: str, tf: str, direction: str, kind: str = "displacement_observed", minutes: int = 0) -> EvidenceItem:
    return EvidenceItem(i, T2 + pd.Timedelta(minutes=minutes), kind, tf, None, direction=direction)


def _judged(items, verdict=Verdict.NEUTRAL):
    return tuple(EvidenceVerdict(item.evidence_id, verdict, "n") for item in items)


def test_a_bias_keeps_its_since_while_the_pair_holds_and_restarts_on_a_new_pair() -> None:
    first = apply(None, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.SHORT, "4H", "a")), ctx=ctx())
    assert first.state.bias.since == T2 and first.state.bias.decayed is None
    later = ctx(known_at=T2 + pd.Timedelta(minutes=5))
    second = apply(first.state, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.SHORT, "4H", "b")), ctx=later)
    assert second.state.bias.since == T2 and second.state.bias.basis == "b"
    third = apply(second.state, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.LONG, "15m", "c")), ctx=later)
    assert third.state.bias.since == later.known_at
    neutral = apply(third.state, episode_id="EP", evidence=[], update=upd(bias=Bias()), ctx=later)
    assert neutral.state.bias.since == later.known_at and neutral.state.bias.decayed is None


def test_two_structural_events_against_the_bias_on_the_scales_below_decay_it() -> None:
    prev = apply(None, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.SHORT, "4H", "a")), ctx=ctx()).state
    later = ctx(known_at=T2 + pd.Timedelta(minutes=30))
    evidence = [struct("ev_1", "15m", "long", minutes=10), struct("ev_2", "15m", "long", minutes=20)]
    res = apply(prev, episode_id="EP", evidence=evidence, update=upd(evidence_verdicts=_judged(evidence, Verdict.CONTRADICT), bias=Bias(BiasDirection.SHORT, "4H", "a")), ctx=later)
    bias = res.state.bias
    assert bias.direction is BiasDirection.NEUTRAL and bias.scale == "4H" and bias.decayed == "SHORT@4H" and bias.since == later.known_at
    assert "bias_decayed:SHORT@4H:2" in res.rejections and bias.basis.startswith("code: 2 structural events against SHORT on 4H/1H/15m since ")
    assert res.state.last_update.rejections == res.rejections


def test_a_same_direction_event_resets_the_count_and_the_5m_counts_only_under_a_15m_bias() -> None:
    prev = apply(None, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.SHORT, "4H", "a")), ctx=ctx()).state
    later = ctx(known_at=T2 + pd.Timedelta(minutes=40))
    evidence = [struct("ev_1", "15m", "long", minutes=10), struct("ev_2", "1H", "short", minutes=20), struct("ev_3", "15m", "long", minutes=30), struct("ev_4", "5m", "long", minutes=35)]
    res = apply(prev, episode_id="EP", evidence=evidence, update=upd(evidence_verdicts=_judged(evidence), bias=Bias(BiasDirection.SHORT, "4H", "a")), ctx=later)
    assert res.state.bias.direction is BiasDirection.SHORT and not any(r.startswith("bias_") for r in res.rejections)
    prev15 = apply(None, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.LONG, "15m", "a")), ctx=ctx()).state
    evidence = [struct("ev_5", "5m", "short", minutes=10), struct("ev_6", "5m", "short", minutes=20)]
    res = apply(prev15, episode_id="EP", evidence=evidence, update=upd(evidence_verdicts=_judged(evidence), bias=Bias(BiasDirection.LONG, "15m", "a")), ctx=later)
    assert res.state.bias.decayed == "LONG@15m"
    # events before the bias was set do not count, nor do non-structural ones
    prior_items = [struct("ev_7", "15m", "long", minutes=-10), struct("ev_8", "15m", "long", minutes=-5)]
    old = make_state(bias=Bias(BiasDirection.SHORT, "4H", "a", since=T2), evidence=EvidenceLedger(contradicting=tuple(prior_items)), watch_next=(), destination_candidates=())
    sweep = EvidenceItem("ev_9", T2 + pd.Timedelta(minutes=10), "sweep_confirmed", "15m", None, direction="long")
    res = apply(old, episode_id="EP", evidence=[sweep], update=upd(evidence_verdicts=_judged([sweep]), bias=Bias(BiasDirection.SHORT, "4H", "a")), ctx=later)
    assert res.state.bias.direction is BiasDirection.SHORT


def test_an_mss_against_the_bias_on_its_own_scale_ends_it_at_once() -> None:
    prev = apply(None, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.LONG, "15m", "a")), ctx=ctx()).state
    later = ctx(known_at=T2 + pd.Timedelta(minutes=15))
    mss = struct("ev_1", "15m", "short", kind="mss_core_confirmed", minutes=10)
    res = apply(prev, episode_id="EP", evidence=[mss], update=upd(evidence_verdicts=_judged([mss]), bias=Bias(BiasDirection.LONG, "15m", "a")), ctx=later)
    assert res.state.bias.decayed == "LONG@15m" and "bias_decayed:LONG@15m:1" in res.rejections and res.state.bias.decayed_at == later.known_at
    assert "ended by mss_core_confirmed on 15m" in res.state.bias.basis
    # a 5m MSS against a 15m bias is one event, not the end
    mss5 = struct("ev_2", "5m", "short", kind="mss_core_confirmed", minutes=10)
    res = apply(prev, episode_id="EP", evidence=[mss5], update=upd(evidence_verdicts=_judged([mss5]), bias=Bias(BiasDirection.LONG, "15m", "a")), ctx=later)
    assert res.state.bias.direction is BiasDirection.LONG


def test_a_decayed_bias_is_reasserted_by_structure_on_its_scale_only() -> None:
    decayed = make_state(bias=Bias(BiasDirection.NEUTRAL, "4H", "code", since=T2, decayed="SHORT@4H", decayed_at=T2), watch_next=(), destination_candidates=())
    later = ctx(known_at=T2 + pd.Timedelta(minutes=30))
    disp = struct("ev_1", "15m", "short", minutes=10)
    res = apply(decayed, episode_id="EP", evidence=[disp], update=upd(evidence_verdicts=_judged([disp], Verdict.SUPPORT), bias=Bias(BiasDirection.SHORT, "4H", "again")), ctx=later)
    assert res.state.bias == decayed.bias and "bias_reassert_refused:SHORT@4H" in res.rejections
    bos = struct("ev_2", "4H", "short", kind="qualified_bos", minutes=20)
    res = apply(decayed, episode_id="EP", evidence=[bos], update=upd(evidence_verdicts=_judged([bos], Verdict.SUPPORT), bias=Bias(BiasDirection.SHORT, "4H", "again")), ctx=later)
    assert res.state.bias.direction is BiasDirection.SHORT and res.state.bias.since == later.known_at and res.state.bias.decayed is None and res.rejections == ()
    res = apply(decayed, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.LONG, "15m", "other")), ctx=later)
    assert res.state.bias.direction is BiasDirection.LONG and res.state.bias.decayed == "SHORT@4H", "another pair is free; the memory travels with it"
    stays = apply(decayed, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.NEUTRAL, "15m", "still")), ctx=later)
    assert stays.state.bias == Bias(BiasDirection.NEUTRAL, "4H", "still", since=T2, decayed="SHORT@4H", decayed_at=T2), "a NEUTRAL reply keeps the decay memory"


def test_an_opportunity_under_a_decayed_bias_is_dropped() -> None:
    prev = apply(None, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.LONG, "15m", "a")), ctx=ctx()).state
    later = ctx(known_at=T2 + pd.Timedelta(minutes=15), timeframe_of=_tf)
    opp = Opportunity(OpportunityState.ACTIONABLE, TradeDirection.LONG, "FVG_5m_3", "SSL_5m_2", "BSL_1H_1", thesis_id="T1")
    mss = struct("ev_1", "15m", "short", kind="mss_core_confirmed", minutes=10)
    res = apply(prev, episode_id="EP", evidence=[mss], update=upd(evidence_verdicts=_judged([mss]), bias=Bias(BiasDirection.LONG, "15m", "a"), opportunity=opp), ctx=later)
    assert res.state.opportunity == Opportunity() and "opportunity_against_bias:NEUTRAL" in res.rejections and "bias_decayed:LONG@15m:1" in res.rejections


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


def test_a_wake_carries_the_archived_bias_into_the_new_episode() -> None:
    """The decay memory lives across episodes (2026-09-22): a fresh wake after a decay is held to the same rule."""
    carried = Bias(BiasDirection.NEUTRAL, "1H", "code", since=T2, decayed="SHORT@1H", decayed_at=T2)
    later = ctx(known_at=T2 + pd.Timedelta(minutes=30), carried_bias=carried)
    disp = struct("ev_1", "15m", "short", minutes=10)
    res = apply(None, episode_id="EP_2", evidence=[disp], update=upd(evidence_verdicts=_judged([disp], Verdict.SUPPORT), bias=Bias(BiasDirection.SHORT, "1H", "again")), ctx=later)
    assert res.state.bias == carried and "bias_reassert_refused:SHORT@1H" in res.rejections and res.state.revision == 0
    bos = struct("ev_2", "1H", "short", kind="qualified_bos", minutes=20)
    res = apply(None, episode_id="EP_2", evidence=[bos], update=upd(evidence_verdicts=_judged([bos], Verdict.SUPPORT), bias=Bias(BiasDirection.SHORT, "1H", "again")), ctx=later)
    assert res.state.bias.direction is BiasDirection.SHORT and res.state.bias.decayed is None and res.rejections == ()
    res = apply(None, episode_id="EP_2", evidence=[], update=upd(bias=Bias(BiasDirection.LONG, "15m", "other")), ctx=later)
    assert res.state.bias.direction is BiasDirection.LONG and res.state.bias.since == later.known_at
    # the same pair as the archived bias keeps its since across the wake
    held = ctx(known_at=T2 + pd.Timedelta(minutes=30), carried_bias=Bias(BiasDirection.SHORT, "4H", "a", since=T2))
    res = apply(None, episode_id="EP_2", evidence=[], update=upd(bias=Bias(BiasDirection.SHORT, "4H", "b")), ctx=held)
    assert res.state.bias.since == T2 and res.state.bias.basis == "b"
    assert apply(None, episode_id="EP_2", evidence=[], update=upd(bias=Bias(BiasDirection.SHORT, "4H", "b")), ctx=ctx()).state.bias.since == T2  # no carry: since is now


# ------------------------------------------------------------ the review's fixes (2026-09-22, second pass)

def test_an_incident_on_the_wake_call_keeps_the_carried_bias() -> None:
    carried = Bias(BiasDirection.NEUTRAL, "4H", "code", since=T2, decayed="SHORT@4H", decayed_at=T2)
    later = ctx(known_at=T2 + pd.Timedelta(minutes=30), carried_bias=carried)
    res = apply(None, episode_id="EP_2", evidence=[], update=None, ctx=later, incident="llm_timeout")
    assert res.state.revision == 0 and res.state.bias == carried
    again = apply(res.state, episode_id="EP_2", evidence=[], update=upd(bias=Bias(BiasDirection.SHORT, "4H", "again")), ctx=later)
    assert again.state.bias.direction is BiasDirection.NEUTRAL and "bias_reassert_refused:SHORT@4H" in again.rejections


def test_a_resolved_structural_item_still_counts_toward_the_decay() -> None:
    first = struct("ev_1", "15m", "long", minutes=5)
    prev = make_state(
        bias=Bias(BiasDirection.SHORT, "4H", "a", since=T2), watch_next=(), destination_candidates=(),
        evidence=EvidenceLedger(unresolved=(replace(first, verdict=Verdict.NEUTRAL),)),
    )
    later = ctx(known_at=T2 + pd.Timedelta(minutes=30))
    carrier = EvidenceItem("ev_2", T2 + pd.Timedelta(minutes=20), "sweep_confirmed", "15m", None)
    second = struct("ev_3", "15m", "long", minutes=25)
    verdicts = (
        EvidenceVerdict("ev_2", Verdict.RESOLVE, "closes ev_1", resolves_evidence_id="ev_1", resolution=Verdict.CONTRADICT),
        EvidenceVerdict("ev_3", Verdict.NEUTRAL, "n"),
    )
    res = apply(prev, episode_id="EP", evidence=[carrier, second], update=upd(evidence_verdicts=verdicts, bias=Bias(BiasDirection.SHORT, "4H", "a")), ctx=later)
    ids = {e.evidence_id for e in res.state.evidence.contradicting}
    assert {"ev_1", "ev_2"} <= ids, "the resolved item stays in the ledger under its resolution"
    assert res.state.bias.decayed == "SHORT@4H" and "bias_decayed:SHORT@4H:2" in res.rejections


def test_a_scale_change_in_the_same_direction_keeps_since() -> None:
    prev = apply(None, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.SHORT, "1H", "a")), ctx=ctx()).state
    later = ctx(known_at=T2 + pd.Timedelta(minutes=30))
    evidence = [struct("ev_1", "15m", "long", minutes=10), struct("ev_2", "15m", "long", minutes=20)]
    res = apply(prev, episode_id="EP", evidence=evidence, update=upd(evidence_verdicts=_judged(evidence), bias=Bias(BiasDirection.SHORT, "4H", "up a scale")), ctx=later)
    assert res.state.bias.decayed == "SHORT@4H", "the two 15m events since the SHORT was first set still count on the 4H"
    up = apply(prev, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.SHORT, "4H", "up a scale")), ctx=later)
    assert up.state.bias.since == T2


@pytest.mark.parametrize("ids", [("ev_a", "ev_b"), ("ev_b", "ev_a")])
def test_same_bar_items_count_one_against_whatever_their_ids(ids) -> None:
    prev = apply(None, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.LONG, "15m", "a")), ctx=ctx()).state
    later = ctx(known_at=T2 + pd.Timedelta(minutes=20))
    same = struct(ids[0], "15m", "long", minutes=10)
    against = struct(ids[1], "15m", "short", minutes=10)
    res = apply(prev, episode_id="EP", evidence=[same, against], update=upd(evidence_verdicts=_judged([same, against]), bias=Bias(BiasDirection.LONG, "15m", "a")), ctx=later)
    assert res.state.bias.direction is BiasDirection.LONG, "one bar: the same-direction event resets, the against one counts one"
    more = struct("ev_c", "5m", "short", minutes=15)
    res = apply(prev, episode_id="EP", evidence=[same, against, more], update=upd(evidence_verdicts=_judged([same, against, more]), bias=Bias(BiasDirection.LONG, "15m", "a")), ctx=later)
    assert res.state.bias.decayed == "LONG@15m"


def test_the_decay_memory_survives_a_detour_through_another_pair() -> None:
    decayed = make_state(bias=Bias(BiasDirection.NEUTRAL, "4H", "code", since=T2, decayed="SHORT@4H", decayed_at=T2), watch_next=(), destination_candidates=())
    later = ctx(known_at=T2 + pd.Timedelta(minutes=30))
    detour = apply(decayed, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.LONG, "15m", "other")), ctx=later)
    assert detour.state.bias.direction is BiasDirection.LONG and detour.state.bias.decayed == "SHORT@4H" and detour.state.bias.decayed_at == T2
    back = apply(detour.state, episode_id="EP", evidence=[], update=upd(bias=Bias(BiasDirection.SHORT, "4H", "again")), ctx=ctx(known_at=T2 + pd.Timedelta(minutes=40)))
    assert back.state.bias.direction is BiasDirection.NEUTRAL and back.state.bias.scale == "4H" and "bias_reassert_refused:SHORT@4H" in back.rejections
    assert back.state.bias.decayed == "SHORT@4H" and back.state.bias.since == T2 + pd.Timedelta(minutes=40)
    bos = struct("ev_9", "4H", "short", kind="qualified_bos", minutes=35)
    confirmed = apply(detour.state, episode_id="EP", evidence=[bos], update=upd(evidence_verdicts=_judged([bos], Verdict.SUPPORT), bias=Bias(BiasDirection.SHORT, "4H", "again")), ctx=ctx(known_at=T2 + pd.Timedelta(minutes=40)))
    assert confirmed.state.bias.direction is BiasDirection.SHORT and confirmed.state.bias.decayed is None and confirmed.state.bias.decayed_at is None
