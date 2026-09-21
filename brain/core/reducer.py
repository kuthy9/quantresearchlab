"""BrainState_t + NewEvidence_t+1 + LLMUpdate → BrainState_t+1.

A pure function.  The LLM proposes; this module keeps the books: every piece
of evidence gets exactly one verdict or lands in ``unresolved``, an abandoned
understanding must be replaced, an opportunity must name visible objects
whose geometry is coherent, an open position forbids sleep, and sleep itself
is granted only when the five exit conditions hold.  Nothing here reads the
Eye; it reads the ``ReduceContext`` the Main Brain hands it.

``unresolved`` holds two things: NEUTRAL items (judged — they bear on nothing
yet, and may be RESOLVEd later) and *pending* items whose ``verdict`` is
``None`` because an incident bar left them unjudged.  Only pending items
refuse sleep, and the Main Brain re-offers them in ``new_evidence`` on every
later call until the LLM verdicts them."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from contract.brain.llm import LLMUpdate
from contract.brain.state import (
    GOVERNING_TIMEFRAMES,
    ActiveExpectation,
    Bias,
    BiasDirection,
    BrainState,
    BrainStatus,
    Confidence,
    EvidenceItem,
    EvidenceLedger,
    LastUpdate,
    Opportunity,
    OpportunityState,
    RegisteredObject,
    Verdict,
    WatchItem,
)

SLEEP_CONDITIONS: tuple[str, ...] = (
    "open_position",
    "open_interaction",
    "unresolved_evidence",
    "watch_next",
    "opportunity",
)


@dataclass(frozen=True)
class ReduceContext:
    known_at: pd.Timestamp
    has_open_position: bool
    open_interaction: bool
    visible_aliases: frozenset[str]
    registry: Mapping[str, RegisteredObject]
    coherence: Callable[[Opportunity], str | None]
    # Accepted updates so far in a row that proposed no opportunity and kept
    # the understanding; with ``idle_archive_after`` set, the update that
    # makes the run reach it archives the episode (rule 6b).
    idle_updates: int = 0
    idle_archive_after: int | None = None
    # The scale of a visible object (``None`` when unknown: the invalidation
    # scale rule then does not judge it).
    timeframe_of: Callable[[str], str | None] = lambda alias: None
    # The ledger keeps at most this many pending (unjudged) items; the oldest
    # expire (``evidence_expired``).  None = unbounded.  Run X of 2026-09-19:
    # 25 empty replies in a row re-offered 124 pending items and the input
    # grew until no reply could come.
    max_pending: int | None = None


# The scale ladder: an invalidation object may sit on the governing scale or
# one step below it (4H → 1H, 1H → 15m, 15m → 5m, 5m → 1m).
_SCALE_LADDER: tuple[str, ...] = GOVERNING_TIMEFRAMES + ("1m",)


def scale_gap(governing: str, invalidation: str) -> int | None:
    """How many steps below the governing scale the invalidation scale is;
    negative above it, ``None`` when either scale is not on the ladder."""
    if governing not in _SCALE_LADDER or invalidation not in _SCALE_LADDER:
        return None
    return _SCALE_LADDER.index(invalidation) - _SCALE_LADDER.index(governing)


@dataclass(frozen=True)
class ReduceResult:
    state: BrainState
    rejections: tuple[str, ...]
    slept: bool
    incident: str | None
    # Why the episode slept: the LLM's own request, or the idle rule.
    sleep_reason: str | None = None


def _zero_verdicts() -> dict[str, int]:
    return {item.value: 0 for item in Verdict}


def empty_state(
    episode_id: str,
    known_at: pd.Timestamp,
    registry: Mapping[str, RegisteredObject],
    *,
    incident: str | None,
) -> BrainState:
    """The revision-0 state of an episode whose first reasoning never arrived."""
    return BrainState(
        episode_id=episode_id,
        status=BrainStatus.ACTIVE,
        revision=0,
        started_at=known_at,
        updated_at=known_at,
        market_understanding="",
        active_expectation=ActiveExpectation(),
        bias=Bias(),
        evidence=EvidenceLedger(),
        watch_next=(),
        destination_candidates=(),
        opportunity=Opportunity(),
        reasoning_confidence=Confidence.LOW,
        continue_active=True,
        object_registry=registry,
        last_update=LastUpdate(known_at, True, _zero_verdicts(), incident),
    )


def pending_evidence(ledger: EvidenceLedger) -> tuple[EvidenceItem, ...]:
    """The unjudged items: parked by an incident bar, still awaiting a verdict."""
    return tuple(item for item in ledger.unresolved if item.verdict is None)


def _expire_pending(unresolved: list[EvidenceItem], max_pending: int | None, rejections: list[str]) -> list[EvidenceItem]:
    """Keep the newest ``max_pending`` pending items (ledger order is arrival
    order); the older ones leave the ledger as ``evidence_expired``."""
    if max_pending is None:
        return unresolved
    pending_ids = [item.evidence_id for item in unresolved if item.verdict is None]
    excess = set(pending_ids[: max(0, len(pending_ids) - max(0, int(max_pending)))])
    if not excess:
        return unresolved
    for evidence_id in pending_ids:
        if evidence_id in excess:
            rejections.append(f"evidence_expired:{evidence_id}")
    return [item for item in unresolved if item.evidence_id not in excess]


def sleep_blockers(
    state: BrainState, *, has_open_position: bool, open_interaction: bool
) -> tuple[str, ...]:
    """The exit conditions that currently fail, in ``SLEEP_CONDITIONS`` order."""
    failing = []
    if has_open_position:
        failing.append("open_position")
    if open_interaction:
        failing.append("open_interaction")
    if pending_evidence(state.evidence):
        failing.append("unresolved_evidence")
    if state.watch_next:
        failing.append("watch_next")
    if state.opportunity.state is not OpportunityState.NONE:
        failing.append("opportunity")
    return tuple(failing)


def _carry_forward(
    prev: BrainState,
    *,
    evidence: Sequence[EvidenceItem],
    ctx: ReduceContext,
    llm_called: bool,
    incident: str | None,
    rejections: list[str],
) -> BrainState:
    """A TICK-shaped revision: bookkeeping advances, reasoning does not."""
    unresolved = list(prev.evidence.unresolved)
    known = _ledger_ids(prev.evidence)
    pending = {item.evidence_id for item in pending_evidence(prev.evidence)}
    for item in evidence:
        if item.evidence_id in pending:
            continue  # re-offered and still unjudged: it stays parked, once
        if item.evidence_id in known:
            rejections.append(f"evidence_duplicate:{item.evidence_id}")
            continue
        rejections.append(f"evidence_without_verdict:{item.evidence_id}")
        unresolved.append(item)
    unresolved = _expire_pending(unresolved, ctx.max_pending, rejections)
    return BrainState(
        episode_id=prev.episode_id,
        status=BrainStatus.ACTIVE,
        revision=prev.revision + 1,
        started_at=prev.started_at,
        updated_at=ctx.known_at,
        market_understanding=prev.market_understanding,
        active_expectation=prev.active_expectation,
        bias=prev.bias,
        evidence=EvidenceLedger(prev.evidence.supporting, prev.evidence.contradicting, tuple(unresolved)),
        watch_next=prev.watch_next,
        destination_candidates=prev.destination_candidates,
        opportunity=prev.opportunity,
        reasoning_confidence=prev.reasoning_confidence,
        continue_active=True,
        object_registry=_merge_registry(prev.object_registry, ctx.registry),
        last_update=LastUpdate(ctx.known_at, llm_called, _zero_verdicts(), incident, rejections=tuple(rejections)),
    )


def _ledger_ids(ledger: EvidenceLedger) -> set[str]:
    return {
        item.evidence_id
        for item in ledger.supporting + ledger.contradicting + ledger.unresolved
    }


def _merge_registry(
    prev: Mapping[str, RegisteredObject], current: Mapping[str, RegisteredObject]
) -> dict[str, RegisteredObject]:
    merged = dict(prev)
    merged.update(current)
    return merged


def apply(
    prev: BrainState | None,
    *,
    episode_id: str,
    evidence: Sequence[EvidenceItem],
    update: LLMUpdate | None,
    ctx: ReduceContext,
    incident: str | None = None,
) -> ReduceResult:
    rejections: list[str] = []

    if update is None:
        if prev is None:
            state = empty_state(episode_id, ctx.known_at, ctx.registry, incident=incident)
            return ReduceResult(state, (), False, incident)
        state = _carry_forward(
            prev, evidence=evidence, ctx=ctx, llm_called=incident is not None, incident=incident, rejections=rejections
        )
        return ReduceResult(state, tuple(rejections), False, incident)

    # Rule 3 — an abandoned understanding must be replaced, else the whole
    # update is refused and the state carries forward.
    if prev is not None and not update.understanding_holds:
        if (
            update.market_understanding == prev.market_understanding
            or update.active_expectation.thesis == prev.active_expectation.thesis
        ):
            state = _carry_forward(
                prev, evidence=evidence, ctx=ctx, llm_called=True,
                incident="understanding_not_replaced", rejections=rejections,
            )
            return ReduceResult(state, tuple(rejections), False, "understanding_not_replaced")

    # Rule 2 — evidence bookkeeping.
    supporting = list(prev.evidence.supporting) if prev else []
    contradicting = list(prev.evidence.contradicting) if prev else []
    unresolved = list(prev.evidence.unresolved) if prev else []
    counts = _zero_verdicts()
    verdicts = {item.evidence_id: item for item in update.evidence_verdicts}
    known = _ledger_ids(prev.evidence) if prev else set()
    pending = {item.evidence_id for item in pending_evidence(prev.evidence)} if prev else set()
    resolved_now = {
        item.resolves_evidence_id for item in update.evidence_verdicts if item.verdict is Verdict.RESOLVE
    }
    for item in evidence:
        reoffered = item.evidence_id in pending
        if item.evidence_id in known and not reoffered:
            rejections.append(f"evidence_duplicate:{item.evidence_id}")
            continue
        if reoffered and item.evidence_id in resolved_now:
            continue  # a RESOLVE in this same update files it through its carrier
        verdict = verdicts.get(item.evidence_id)
        if verdict is None:
            rejections.append(f"evidence_without_verdict:{item.evidence_id}")
            if not reoffered:
                unresolved.append(item)
            continue
        if reoffered:
            unresolved = [pending_item for pending_item in unresolved if pending_item.evidence_id != item.evidence_id]
        counts[verdict.verdict.value] += 1
        recorded = EvidenceItem(
            item.evidence_id, item.known_at, item.kind, item.timeframe, item.object_id,
            verdict.verdict, verdict.note, item.direction, item.side,
        )
        if verdict.verdict is Verdict.SUPPORT:
            supporting.append(recorded)
        elif verdict.verdict is Verdict.CONTRADICT:
            contradicting.append(recorded)
        elif verdict.verdict is Verdict.NEUTRAL:
            unresolved.append(recorded)
        else:  # RESOLVE
            target = verdict.resolves_evidence_id
            before = len(unresolved)
            unresolved = [pending for pending in unresolved if pending.evidence_id != target]
            if len(unresolved) == before:
                rejections.append(f"resolve_target_not_unresolved:{target}")
            if verdict.resolution is Verdict.SUPPORT:
                supporting.append(recorded)
            else:
                contradicting.append(recorded)

    unresolved = _expire_pending(unresolved, ctx.max_pending, rejections)

    # Rule 4 — opportunity validation: visible objects, the invalidation on
    # the thesis's scale (or one below), a thesis id that keeps its
    # direction, and coherent geometry.
    opportunity = update.opportunity
    if opportunity.state is not OpportunityState.NONE:
        for alias in opportunity.object_ids:
            if alias not in ctx.visible_aliases:
                rejections.append(f"opportunity_object_not_visible:{alias}")
        invalidation = opportunity.invalidation_object_id or ""
        scale = ctx.timeframe_of(invalidation)
        gap = None if scale is None or opportunity.governing_timeframe is None else scale_gap(opportunity.governing_timeframe, scale)
        if gap is not None and gap > 1:
            rejections.append(f"opportunity_invalidation_scale:{invalidation}")
        if (
            prev is not None
            and opportunity.thesis_id is not None
            and prev.opportunity.thesis_id == opportunity.thesis_id
            and prev.opportunity.direction is not None
            and prev.opportunity.direction is not opportunity.direction
        ):
            rejections.append(f"thesis_direction_changed:{opportunity.thesis_id}")
        # Rule 4b (2026-09-18) — the opportunity follows the bias: no side
        # against it, nothing under a NEUTRAL bias, no thesis on a scale
        # above the bias scale.
        bias = update.bias
        if bias.direction is BiasDirection.NEUTRAL:
            rejections.append("opportunity_against_bias:NEUTRAL")
        elif opportunity.direction is not None and opportunity.direction.value != bias.direction.value:
            rejections.append(f"opportunity_against_bias:{opportunity.direction.value}")
        governing = opportunity.governing_timeframe
        if governing is not None and _SCALE_LADDER.index(governing) < _SCALE_LADDER.index(bias.scale):
            rejections.append(f"opportunity_scale_above_bias:{governing}")
        if not any(r.startswith("opportunity_") or r.startswith("thesis_") for r in rejections):
            reason = ctx.coherence(opportunity)
            if reason is not None:
                rejections.append(f"opportunity_incoherent:{reason}")
        if any(r.startswith("opportunity_") or r.startswith("thesis_") for r in rejections):
            opportunity = Opportunity()

    registry = _merge_registry(prev.object_registry, ctx.registry) if prev else dict(ctx.registry)
    watch: list[WatchItem] = []
    for item in update.watch_next:
        if item.object_id in registry:
            watch.append(item)
        else:
            rejections.append(f"watch_object_unknown:{item.object_id}")
    destinations: list[str] = []
    for alias in update.destination_candidates:
        if alias in registry:
            destinations.append(alias)
        else:
            rejections.append(f"destination_object_unknown:{alias}")

    started_at = prev.started_at if prev else ctx.known_at
    revision = prev.revision + 1 if prev else 0
    draft = BrainState(
        episode_id=episode_id,
        status=BrainStatus.ACTIVE,
        revision=revision,
        started_at=started_at,
        updated_at=ctx.known_at,
        market_understanding=update.market_understanding,
        active_expectation=update.active_expectation,
        bias=update.bias,
        evidence=EvidenceLedger(tuple(supporting), tuple(contradicting), tuple(unresolved)),
        watch_next=tuple(watch),
        destination_candidates=tuple(destinations),
        opportunity=opportunity,
        reasoning_confidence=update.reasoning_confidence,
        continue_active=True,
        object_registry=registry,
        last_update=LastUpdate(ctx.known_at, True, counts, incident),
    )

    # Rules 5 and 6 — position and the sleep conditions.
    continue_active = True
    slept = False
    sleep_reason: str | None = None
    if not update.continue_active:
        blockers = sleep_blockers(
            draft, has_open_position=ctx.has_open_position, open_interaction=ctx.open_interaction
        )
        if blockers:
            rejections.extend(f"sleep_refused:{name}" for name in blockers)
        else:
            continue_active = False
            slept = True
            sleep_reason = "continue_active=false"
    # Rule 6b — the idle rule: the LLM keeps a standing watch list forever, so
    # after ``idle_archive_after`` consecutive updates with no opportunity and
    # an unchanged understanding the episode is archived regardless of
    # ``watch_next`` and of the interaction gate.  A position or an unjudged
    # item still holds it open.
    if (
        not slept
        and ctx.idle_archive_after is not None
        and update.understanding_holds
        and opportunity.state is OpportunityState.NONE
        and ctx.idle_updates + 1 >= ctx.idle_archive_after
        and not ctx.has_open_position
        and not pending_evidence(draft.evidence)
    ):
        continue_active = False
        slept = True
        sleep_reason = "idle"

    state = BrainState(
        episode_id=draft.episode_id,
        status=BrainStatus.ARCHIVED if slept else BrainStatus.ACTIVE,
        revision=draft.revision,
        started_at=draft.started_at,
        updated_at=draft.updated_at,
        market_understanding=draft.market_understanding,
        active_expectation=draft.active_expectation,
        bias=draft.bias,
        evidence=draft.evidence,
        watch_next=draft.watch_next,
        destination_candidates=draft.destination_candidates,
        opportunity=draft.opportunity,
        reasoning_confidence=draft.reasoning_confidence,
        continue_active=continue_active,
        object_registry=draft.object_registry,
        # The rejections travel with the state (2026-09-20): the next call reads them in prior_state.last_update.
        last_update=LastUpdate(ctx.known_at, True, draft.last_update.verdicts, incident, rejections=tuple(rejections)),
    )
    return ReduceResult(state, tuple(rejections), slept, incident, sleep_reason)


__all__ = [
    "SLEEP_CONDITIONS",
    "ReduceContext",
    "ReduceResult",
    "apply",
    "empty_state",
    "pending_evidence",
    "sleep_blockers",
]
