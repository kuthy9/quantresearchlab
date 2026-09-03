"""Test-only adapter over the sole ``InteractionSemantics`` reducer.

This was ``smc_trader.group5``: a production module that existed purely so
historical readers could keep asking for the interpreted Group-5 shape.  Nothing
in the runtime imported it, so it now lives with the tests that actually use it.
It is not an authority, and no checkpoint resumes through it -- replay canonical
events into ``InteractionSemantics`` instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smc_trader.brain_entry_sequence import (
    interpret_interaction_paths,
    interpret_micro_break_facts,
    interpret_milestone_transitions,
)
from smc_trader.interaction import InteractionProtocol, InteractionSemantics
from smc_trader.model import (
    EntryLocationState,
    InteractionUpdate,
    MicroBOSReference,
    PathSequenceState,
    PathSequenceStep,
    ReacceptanceState,
)


Group5Protocol = InteractionProtocol


@dataclass(frozen=True)
class Group5Update:
    """Historical interpreted shape; never embedded in an observation."""

    entry_locations: tuple[EntryLocationState, ...]
    qualified_reacceptances: tuple[ReacceptanceState, ...]
    micro_bos_references: tuple[MicroBOSReference, ...]
    path_sequences: tuple[PathSequenceState, ...]
    path_transitions: tuple[PathSequenceState, ...] = ()
    reacceptance_transitions: tuple[ReacceptanceState, ...] = ()
    step_transitions: tuple[tuple[str, PathSequenceStep], ...] = ()
    cold_source_ids: tuple[str, ...] = ()
    boundary_reason: str | None = None


def _legacy_update(update: InteractionUpdate) -> Group5Update:
    references = interpret_micro_break_facts(update.micro_break_facts)
    paths = interpret_interaction_paths(update.interaction_paths, references)
    return Group5Update(
        entry_locations=update.zone_interactions,
        qualified_reacceptances=update.reacceptance_interactions,
        micro_bos_references=references,
        path_sequences=paths,
        path_transitions=interpret_interaction_paths(
            update.interaction_path_transitions,
            references,
            allow_censored_without_facts=(update.boundary_reason is not None),
        ),
        reacceptance_transitions=update.reacceptance_interaction_transitions,
        step_transitions=interpret_milestone_transitions(
            update.interaction_paths,
            references,
            update.milestone_transitions,
        ),
        cold_source_ids=update.cold_source_ids,
        boundary_reason=update.boundary_reason,
    )


class CausalGroup5Reducer:
    """Interpreted view over one ``InteractionSemantics`` reducer.

    One canonical update interprets to one object.  Callers rely on that
    identity to prove the adapter re-reads rather than re-derives.
    """

    def __init__(self, protocol: Group5Protocol) -> None:
        self._semantics = InteractionSemantics(protocol)
        self._last_canonical: InteractionUpdate | None = None
        self._last_legacy: Group5Update | None = None

    @property
    def protocol(self) -> InteractionProtocol:
        return self._semantics.protocol

    def _adapt(self, update: InteractionUpdate) -> Group5Update:
        if update is self._last_canonical and self._last_legacy is not None:
            return self._last_legacy
        legacy = _legacy_update(update)
        self._last_canonical = update
        self._last_legacy = legacy
        return legacy

    def snapshot(self) -> Group5Update:
        return self._adapt(self._semantics.snapshot())

    def on_completed_1m(self, *args: Any, **kwargs: Any) -> Group5Update:
        return self._adapt(self._semantics.on_completed_1m(*args, **kwargs))

    def on_boundary(self, *args: Any, **kwargs: Any) -> Group5Update:
        return self._adapt(self._semantics.on_boundary(*args, **kwargs))
