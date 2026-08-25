"""Brain-owned interpretation of canonical physical interaction facts.

This module is deliberately stateless.  It does not detect market events or
maintain a second reducer; it classifies the exact raw facts admitted by
``InteractionSemantics`` for setup/playbook consumers.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

import pandas as pd

from .model import (
    Direction,
    EntryLocationState,
    InteractionUpdate,
    MicroBOSReference,
    MicroBreakFact,
    PathSequenceLifecycle,
    PathSequenceState,
    PathSequenceStep,
    QualifiedReacceptanceState,
)


_SUCCESSFUL_PATH_REASONS = frozenset(
    {"micro_bos_aligned", "pool_reversal_sequence_observed"}
)
_LOCAL_TERMINAL_PATH_REASONS = frozenset(
    {
        "location_left",
        "reacceptance_failed",
        "micro_bos_opposed",
        "micro_bos_ambiguous_same_clock",
    }
)


def _identity(*parts: object) -> str:
    return hashlib.sha256(
        "|".join(
            value.isoformat()
            if isinstance(value, pd.Timestamp)
            else str(value)
            for value in parts
        ).encode("utf-8")
    ).hexdigest()


def interpret_micro_break_facts(
    facts: tuple[MicroBreakFact, ...],
) -> tuple[MicroBOSReference, ...]:
    """Classify raw break direction/order once for Brain consumers."""

    strict_counts: dict[tuple[str, str, pd.Timestamp], int] = {}
    for fact in facts:
        if fact.relation == "strictly_after":
            key = (fact.context_kind, fact.context_id, fact.resolved_at)
            strict_counts[key] = strict_counts.get(key, 0) + 1
    references: list[MicroBOSReference] = []
    for fact in facts:
        if fact.relation == "same_clock_unknown":
            outcome = "simultaneous_unknown"
        elif strict_counts[
            (fact.context_kind, fact.context_id, fact.resolved_at)
        ] != 1:
            outcome = "ambiguous_same_clock"
        elif fact.bos_direction is fact.context_direction:
            outcome = "aligned"
        else:
            outcome = "opposed"
        references.append(
            MicroBOSReference(
                reference_id=fact.reference_id,
                protocol_hash=fact.protocol_hash,
                context_kind=fact.context_kind,
                context_id=fact.context_id,
                expected_direction=fact.context_direction,
                anchor_at=fact.anchor_at,
                bos_id=fact.bos_id,
                bos_direction=fact.bos_direction,
                target_swing_id=fact.target_swing_id,
                scope=fact.scope,
                pending_at=fact.pending_at,
                resolved_at=fact.resolved_at,
                relation=fact.relation,
                outcome=outcome,
                qualified=bool(
                    fact.relation == "strictly_after"
                    and outcome == "aligned"
                ),
                strength=fact.strength,
            )
        )
    return tuple(
        sorted(references, key=lambda item: (item.resolved_at, item.reference_id))
    )


def _legacy_micro_step(
    *,
    path: PathSequenceState,
    ordinal: int,
    physical: PathSequenceStep,
    outcome: str,
    predecessor_step_id: str | None,
) -> PathSequenceStep:
    kind = {
        "simultaneous_unknown": "micro_bos_simultaneous",
        "aligned": "micro_bos_confirmed",
        "opposed": "micro_bos_opposed",
        "ambiguous_same_clock": "micro_bos_ambiguous",
    }[outcome]
    reason = (
        "anchor_clock_order_unknown"
        if outcome == "simultaneous_unknown"
        else outcome
    )
    relation = (
        "origin"
        if ordinal == 0
        else (
            "strictly_after"
            if predecessor_step_id is not None
            and physical.same_clock_relation == "strictly_after"
            else physical.same_clock_relation
        )
    )
    return PathSequenceStep(
        step_id=_identity(
            "group5-step-v1",
            path.protocol_hash,
            path.sequence_id,
            ordinal,
            kind,
            physical.observed_at,
            physical.source_entity_id,
        ),
        kind=kind,
        observed_at=physical.observed_at,
        source_event_id=physical.source_event_id,
        source_entity_id=physical.source_entity_id,
        predecessor_step_ids=(
            () if predecessor_step_id is None else (predecessor_step_id,)
        ),
        same_clock_relation=relation,
        direction=physical.direction,
        strength=physical.strength,
        reason=reason,
    )


def interpret_interaction_paths(
    paths: tuple[PathSequenceState, ...],
    references: tuple[MicroBOSReference, ...],
    *,
    allow_censored_without_facts: bool = False,
) -> tuple[PathSequenceState, ...]:
    """Project physical milestone paths into the legacy Brain vocabulary."""

    references_by_source = {
        (item.context_kind, item.context_id, item.bos_id): item
        for item in references
    }
    interpreted: list[PathSequenceState] = []
    for path in paths:
        strict_break_counts: dict[pd.Timestamp, int] = {}
        for physical in path.steps:
            if (
                physical.kind == "micro_break_observed"
                and physical.reason
                == "first_strictly_later_confirmed_m1_break"
            ):
                strict_break_counts[physical.observed_at] = (
                    strict_break_counts.get(physical.observed_at, 0) + 1
                )
        steps: list[PathSequenceStep] = []
        for ordinal, physical in enumerate(path.steps):
            predecessor = None if not steps else steps[-1].step_id
            if physical.kind == "micro_break_observed":
                reference = references_by_source.get(
                    (
                        path.context_kind,
                        path.context_id,
                        physical.source_event_id or "",
                    )
                )
                if (
                    reference is None
                    and not (
                        allow_censored_without_facts
                        and path.lifecycle is PathSequenceLifecycle.CENSORED
                    )
                ):
                    raise ValueError(
                        "physical micro-break milestone lacks its exact fact"
                    )
                if reference is not None:
                    outcome = reference.outcome
                elif (
                    physical.reason
                    == "confirmed_m1_break_at_anchor_clock"
                ):
                    outcome = "simultaneous_unknown"
                elif strict_break_counts.get(physical.observed_at) != 1:
                    outcome = "ambiguous_same_clock"
                elif physical.direction is path.direction:
                    outcome = "aligned"
                else:
                    outcome = "opposed"
                step = _legacy_micro_step(
                    path=path,
                    ordinal=ordinal,
                    physical=physical,
                    outcome=outcome,
                    predecessor_step_id=predecessor,
                )
            else:
                step = replace(
                    physical,
                    predecessor_step_ids=(
                        () if predecessor is None else (predecessor,)
                    ),
                )
            steps.append(step)
        reason = path.transition_reason
        if reason == "first_strict_micro_break_observed":
            strict = tuple(
                reference
                for reference in references_by_source.values()
                if reference.context_kind == path.context_kind
                and reference.context_id == path.context_id
                and reference.relation == "strictly_after"
            )
            outcomes = {reference.outcome for reference in strict}
            if "ambiguous_same_clock" in outcomes:
                reason = "micro_bos_ambiguous_same_clock"
            elif "opposed" in outcomes:
                reason = "micro_bos_opposed"
            elif path.context_kind == "pool_reversal":
                reason = "pool_reversal_sequence_observed"
            else:
                reason = "micro_bos_aligned"
        interpreted.append(replace(path, steps=tuple(steps), transition_reason=reason))
    return tuple(interpreted)


def interpret_milestone_transitions(
    paths: tuple[PathSequenceState, ...],
    references: tuple[MicroBOSReference, ...],
    transitions: tuple[tuple[str, PathSequenceStep], ...],
) -> tuple[tuple[str, PathSequenceStep], ...]:
    interpreted = {
        path.sequence_id: path
        for path in interpret_interaction_paths(paths, references)
    }
    physical = {path.sequence_id: path for path in paths}
    result: list[tuple[str, PathSequenceStep]] = []
    for sequence_id, step in transitions:
        source = physical[sequence_id]
        ordinal = source.steps.index(step)
        result.append((sequence_id, interpreted[sequence_id].steps[ordinal]))
    return tuple(result)


def brain_path_terminal_role(path: PathSequenceState) -> str:
    """Classify one already-interpreted path without reinterpreting facts."""

    if path.lifecycle is PathSequenceLifecycle.CENSORED:
        return "censored"
    if path.lifecycle is not PathSequenceLifecycle.CLOSED:
        return "active"
    if path.transition_reason in _SUCCESSFUL_PATH_REASONS:
        return "successful"
    if path.transition_reason in _LOCAL_TERMINAL_PATH_REASONS:
        return "local_terminal"
    return "unknown"


@dataclass(frozen=True)
class BrainInteractionView:
    zone_interactions: tuple[EntryLocationState, ...]
    reacceptance_interactions: tuple[QualifiedReacceptanceState, ...]
    micro_bos_references: tuple[MicroBOSReference, ...]
    path_sequences: tuple[PathSequenceState, ...]


class BrainObservationView:
    """Read-only Brain adapter over one immutable Eye observation."""

    __slots__ = ("_observation", "_interaction")

    def __init__(self, observation: object) -> None:
        self._observation = observation
        self._interaction = brain_interaction_view(observation)

    @property
    def interaction(self) -> BrainInteractionView:
        return self._interaction

    @property
    def entry_locations(self) -> tuple[EntryLocationState, ...]:
        return self._interaction.zone_interactions

    @property
    def qualified_reacceptances(
        self,
    ) -> tuple[QualifiedReacceptanceState, ...]:
        return self._interaction.reacceptance_interactions

    @property
    def micro_bos_references(self) -> tuple[MicroBOSReference, ...]:
        return self._interaction.micro_bos_references

    @property
    def path_sequences(self) -> tuple[PathSequenceState, ...]:
        return self._interaction.path_sequences

    def __getattr__(self, name: str) -> object:
        return getattr(self._observation, name)


def interpret_interaction_update(update: InteractionUpdate) -> BrainInteractionView:
    if type(update) is not InteractionUpdate:
        raise TypeError("Brain interaction input must be an InteractionUpdate")
    references = interpret_micro_break_facts(update.micro_break_facts)
    return BrainInteractionView(
        zone_interactions=update.zone_interactions,
        reacceptance_interactions=update.reacceptance_interactions,
        micro_bos_references=references,
        path_sequences=interpret_interaction_paths(
            update.interaction_paths,
            references,
        ),
    )


def brain_interaction_view(observation: object) -> BrainInteractionView:
    """Interpret the canonical schema-v3 physical interaction payload."""

    if isinstance(observation, BrainObservationView):
        return observation.interaction
    update = getattr(observation, "interaction_update", None)
    if isinstance(update, InteractionUpdate):
        return interpret_interaction_update(update)
    return BrainInteractionView(
        zone_interactions=(),
        reacceptance_interactions=(),
        micro_bos_references=(),
        path_sequences=(),
    )


def brain_observation_view(observation: object) -> BrainObservationView:
    if isinstance(observation, BrainObservationView):
        return observation
    return BrainObservationView(observation)


__all__ = [
    "BrainInteractionView",
    "BrainObservationView",
    "brain_interaction_view",
    "brain_observation_view",
    "brain_path_terminal_role",
    "interpret_interaction_paths",
    "interpret_interaction_update",
    "interpret_micro_break_facts",
]
