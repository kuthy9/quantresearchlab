"""BrainState_t + NewEvidence_t+1 + LLMUpdate → BrainState_t+1.

A pure function.  The LLM proposes; this module keeps the books: every piece
of evidence gets exactly one verdict or lands in ``unresolved``, an abandoned
understanding must be replaced, an opportunity must name visible objects
whose geometry is coherent, an open position forbids sleep, and sleep itself
is granted only when the five exit conditions hold.  Nothing here reads the
Eye; it reads the ``ReduceContext`` the Main Brain hands it."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from contract.brain.llm import LLMUpdate
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


@dataclass(frozen=True)
class ReduceResult:
    state: BrainState
    rejections: tuple[str, ...]
    slept: bool
    incident: str | None


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
        evidence=EvidenceLedger(),
        watch_next=(),
        destination_candidates=(),
        opportunity=Opportunity(),
        reasoning_confidence=Confidence.LOW,
        continue_active=True,
        object_registry=registry,
        last_update=LastUpdate(known_at, True, _zero_verdicts(), incident),
    )


def sleep_blockers(
    state: BrainState, *, has_open_position: bool, open_interaction: bool
) -> tuple[str, ...]:
    """The exit conditions that currently fail, in ``SLEEP_CONDITIONS`` order."""
    failing = []
    if has_open_position:
        failing.append("open_position")
    if open_interaction:
        failing.append("open_interaction")
    if state.evidence.unresolved:
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
    for item in evidence:
        if item.evidence_id in known:
            rejections.append(f"evidence_duplicate:{item.evidence_id}")
            continue
        rejections.append(f"evidence_without_verdict:{item.evidence_id}")
        unresolved.append(item)
    return BrainState(
        episode_id=prev.episode_id,
        status=BrainStatus.ACTIVE,
        revision=prev.revision + 1,
        started_at=prev.started_at,
        updated_at=ctx.known_at,
        market_understanding=prev.market_understanding,
        active_expectation=prev.active_expectation,
        evidence=EvidenceLedger(prev.evidence.supporting, prev.evidence.contradicting, tuple(unresolved)),
        watch_next=prev.watch_next,
        destination_candidates=prev.destination_candidates,
        opportunity=prev.opportunity,
        reasoning_confidence=prev.reasoning_confidence,
        continue_active=True,
        object_registry=_merge_registry(prev.object_registry, ctx.registry),
        last_update=LastUpdate(ctx.known_at, llm_called, _zero_verdicts(), incident),
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
    for item in evidence:
        if item.evidence_id in known:
            rejections.append(f"evidence_duplicate:{item.evidence_id}")
            continue
        verdict = verdicts.get(item.evidence_id)
        if verdict is None:
            rejections.append(f"evidence_without_verdict:{item.evidence_id}")
            unresolved.append(item)
            continue
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

    # Rule 4 — opportunity validation.
    opportunity = update.opportunity
    if opportunity.state is not OpportunityState.NONE:
        for alias in opportunity.object_ids:
            if alias not in ctx.visible_aliases:
                rejections.append(f"opportunity_object_not_visible:{alias}")
        if not any(r.startswith("opportunity_object_not_visible") for r in rejections):
            reason = ctx.coherence(opportunity)
            if reason is not None:
                rejections.append(f"opportunity_incoherent:{reason}")
        if any(r.startswith("opportunity_") for r in rejections):
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
    if not update.continue_active:
        blockers = sleep_blockers(
            draft, has_open_position=ctx.has_open_position, open_interaction=ctx.open_interaction
        )
        if blockers:
            rejections.extend(f"sleep_refused:{name}" for name in blockers)
        else:
            continue_active = False
            slept = True

    state = BrainState(
        episode_id=draft.episode_id,
        status=BrainStatus.ARCHIVED if slept else BrainStatus.ACTIVE,
        revision=draft.revision,
        started_at=draft.started_at,
        updated_at=draft.updated_at,
        market_understanding=draft.market_understanding,
        active_expectation=draft.active_expectation,
        evidence=draft.evidence,
        watch_next=draft.watch_next,
        destination_candidates=draft.destination_candidates,
        opportunity=draft.opportunity,
        reasoning_confidence=draft.reasoning_confidence,
        continue_active=continue_active,
        object_registry=draft.object_registry,
        last_update=draft.last_update,
    )
    return ReduceResult(state, tuple(rejections), slept, incident)


__all__ = [
    "SLEEP_CONDITIONS",
    "ReduceContext",
    "ReduceResult",
    "apply",
    "empty_state",
    "sleep_blockers",
]
