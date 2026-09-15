"""A DTO that passed canonical re-admission once is not reconstructed again.

``InteractionUpdate.validate_canonical_bindings`` re-admits every nested DTO
through its constructor on every bar the update is built -- and the update
carries every closed context path until ``maximum_context_states`` evicts it,
so at bar 2,000 on the real tape that was 18 s per 500 bars and growing.  A
nested DTO is a frozen dataclass: an instance that was canonical stays
canonical, so the validator now re-admits only objects it has not seen.  The
memo is keyed by ``id`` under a weak reference, so a freed object never leaves
an entry a recycled id could hit.
"""
from __future__ import annotations

from dataclasses import fields, replace
import gc

import contract.eye.interaction as interaction_contract
from contract.eye.interaction import InteractionUpdate, _is_admitted

from eyes.tests.test_interaction_eye_brain_boundary import _aligned_interaction


def _rebuild(update: InteractionUpdate) -> InteractionUpdate:
    """Build the update again from the same nested DTOs, as every bar does."""

    return InteractionUpdate(
        **{item.name: getattr(update, item.name) for item in fields(InteractionUpdate)}
    )


def test_a_path_is_remembered_after_its_first_admission() -> None:
    update, _ = _aligned_interaction()
    path = update.interaction_paths[0]
    assert _is_admitted(path)
    for step in path.steps:
        assert _is_admitted(step)
    for zone in update.zone_interactions:
        assert _is_admitted(zone)


def test_an_equal_but_distinct_object_is_admitted_on_its_own() -> None:
    update, _ = _aligned_interaction()
    path = update.interaction_paths[0]
    twin = replace(path)
    assert _is_admitted(path) and not _is_admitted(twin)
    _rebuild(replace(update, interaction_paths=(twin,)))
    assert _is_admitted(twin)


def test_a_tampered_admitted_object_is_not_trusted() -> None:
    update, _ = _aligned_interaction()
    fact = update.micro_break_facts[0]
    assert _is_admitted(fact)
    object.__setattr__(fact, "scope", "continuation")
    assert not _is_admitted(fact)


def test_a_freed_object_does_not_leave_a_stale_entry() -> None:
    update, _ = _aligned_interaction()
    key = id(update.interaction_paths[0])
    assert key in interaction_contract._ADMITTED
    del update
    gc.collect()
    assert key not in interaction_contract._ADMITTED
