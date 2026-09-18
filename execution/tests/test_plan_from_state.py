from __future__ import annotations

from pathlib import Path

import pytest

from brain.core.eye_view import EvidenceRule, build_eye_context
from brain.core.object_registry import ObjectRegistry
from brain.tests.test_brain_state import make_state
from contract.brain.state import Opportunity, OpportunityState, TradeDirection
from execution.core.plan import plan_from_state
from shares.core.eye_factory import build_eye
from shares.tests.helpers import session_bars

ROOT = Path(__file__).resolve().parents[2]
RULE = EvidenceRule(frozenset({"5m", "15m", "1H", "4H"}), frozenset({"bar_completed", "market_epoch_reset"}), "_state")


@pytest.fixture(scope="module")
def contexts():
    reader, observer = build_eye(ROOT / "configs" / "model.json", root=ROOT, audit_journal_dir=None)
    registry = ObjectRegistry()
    built = []
    for bar in session_bars(1):
        obs = observer.observe(reader.on_bar(bar))
        if obs.market_snapshot is not None:
            built.append(build_eye_context(obs, registry, rule=RULE))
    return built, registry


def long_triple(ctx):
    zones = [v for v in ctx.objects if v.kind in {"fvg", "ob"}]
    above = [v for v in ctx.objects if v.kind == "bsl" and v.lower > ctx.close]
    below = [v for v in ctx.objects if v.kind == "ssl" and v.upper < ctx.close]
    if not (zones and above and below):
        pytest.skip("synthetic tape has no zone/pool triple on this bar")
    zone = min(zones, key=lambda v: abs(v.anchor - ctx.close))
    return zone.alias, min(below, key=lambda v: v.upper).alias, max(above, key=lambda v: v.lower).alias


def state_with(ctx, registry, opportunity: Opportunity):
    return make_state(
        episode_id="EP_20250105_001", started_at=ctx.known_at, updated_at=ctx.known_at,
        opportunity=opportunity, watch_next=(), destination_candidates=(), object_registry=dict(registry.snapshot()),
    )


def test_none_and_developing_give_no_plan(contexts) -> None:
    ctxs, registry = contexts
    ctx = ctxs[-1]
    assert plan_from_state(state_with(ctx, registry, Opportunity()), ctx, tick=0.25) is None
    entry, stop, target = long_triple(ctx)
    developing = Opportunity(OpportunityState.DEVELOPING, TradeDirection.LONG, entry, stop, target)
    assert plan_from_state(state_with(ctx, registry, developing), ctx, tick=0.25) is None


def test_actionable_becomes_a_plan_with_entity_ids_and_a_stable_signature(contexts) -> None:
    ctxs, registry = contexts
    ctx = ctxs[-1]
    entry, stop, target = long_triple(ctx)
    actionable = Opportunity(OpportunityState.ACTIONABLE, TradeDirection.LONG, entry, stop, target)
    state = state_with(ctx, registry, actionable)
    plan = plan_from_state(state, ctx, tick=0.25)
    assert plan is not None and plan.direction is TradeDirection.LONG
    assert plan.entry.alias == entry and plan.entry.entity_id == registry.get(entry).entity_id
    assert plan.target.kind == "bsl" and plan.geometry.stop_price < plan.geometry.entry_price < plan.geometry.target_price
    assert plan.close == ctx.close and plan.episode_id == state.episode_id and plan.revision == state.revision
    again = plan_from_state(state, ctx, tick=0.25)
    assert again is not None and again.signature == plan.signature


def test_a_vanished_object_gives_no_plan(contexts) -> None:
    ctxs, registry = contexts
    ctx = ctxs[-1]
    entry, stop, target = long_triple(ctx)
    actionable = Opportunity(OpportunityState.ACTIONABLE, TradeDirection.LONG, entry, stop, target)
    state = state_with(ctx, registry, actionable)
    earlier = next((c for c in ctxs if entry not in c.visible_aliases()), None)
    if earlier is None:
        pytest.skip("the entry object is visible on every synthetic bar")
    assert plan_from_state(state, earlier, tick=0.25) is None


def test_a_plan_carries_the_thesis_fields_and_the_invalidation_objects_far_edge(contexts) -> None:
    ctxs, registry = contexts
    ctx = ctxs[-1]
    entry, stop, target = long_triple(ctx)
    opportunity = Opportunity(OpportunityState.ACTIONABLE, TradeDirection.LONG, entry, stop, target, thesis_id="T7", governing_timeframe="5m", grade="A_PLUS", invalidation_mode="TOUCH")
    plan = plan_from_state(state_with(ctx, registry, opportunity), ctx, tick=0.25)
    assert plan is not None
    assert plan.thesis_id == "T7" and plan.governing_timeframe == "5m" and plan.grade.value == "A_PLUS" and plan.invalidation_mode.value == "TOUCH"
    assert plan.invalidation_level == ctx.geometries()[stop].lower, "a LONG is invalidated below the object's lower edge"
    assert plan.to_dict()["invalidation_level"] == plan.invalidation_level
