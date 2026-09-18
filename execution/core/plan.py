"""From the Brain's state to a ``TradePlan``.

Only an ACTIONABLE opportunity becomes a plan.  The geometry is resolved
again on the *current* context (the objects may have moved since the
revision that named them) by the same ``opportunity_geometry`` rules the
reducer vetted, and the entity ids come from the episode's registry, so the
plan is the Brain's ``data_id`` for the executor: alias, Eye entity id, and
the prices code derived from that entity on this bar."""
from __future__ import annotations

from brain.core.eye_view import EyeContext
from brain.core.opportunity_geometry import SWING_KINDS, resolve_geometry
from contract.brain.state import BrainState, OpportunityState, TradeDirection
from contract.decision import GeometryError
from contract.risk import ObjectRef, TradePlan


def plan_from_state(state: BrainState, context: EyeContext, *, tick: float) -> TradePlan | None:
    opportunity = state.opportunity
    if opportunity.state is not OpportunityState.ACTIONABLE or opportunity.direction is None:
        return None
    geometries = context.geometries()
    try:
        geometry = resolve_geometry(opportunity, geometries, close=context.close, tick=tick, atr_1m=context.atr_1m)
    except GeometryError:
        return None
    invalidation = geometries[opportunity.invalidation_object_id or ""]
    if invalidation.kind in SWING_KINDS:
        level = invalidation.anchor
    else:
        level = invalidation.lower if opportunity.direction is TradeDirection.LONG else invalidation.upper
    refs = []
    for alias in (opportunity.entry_object_id, opportunity.invalidation_object_id, opportunity.target_object_id):
        registered = state.object_registry.get(alias or "")
        if registered is None:
            return None
        refs.append(ObjectRef(str(alias), registered.entity_id, registered.kind, registered.timeframe))
    return TradePlan(
        episode_id=state.episode_id, revision=state.revision, known_at=context.known_at, direction=opportunity.direction,
        entry=refs[0], invalidation=refs[1], target=refs[2], geometry=geometry, close=context.close,
        thesis_id=opportunity.thesis_id or "", governing_timeframe=opportunity.governing_timeframe or "",
        grade=opportunity.grade, invalidation_mode=opportunity.invalidation_mode, invalidation_level=level,
    )


__all__ = ["plan_from_state"]
