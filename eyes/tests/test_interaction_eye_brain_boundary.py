from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pickle

import pandas as pd
import pytest

import shares
from shares.tests import legacy_group5
from brain.core.brain_entry_sequence import (
    brain_interaction_view,
    brain_observation_view,
    interpret_interaction_update,
)
from eyes.core.interaction import (
    INTERACTION_ARTIFACT_COLLECTION_NAMES,
    InteractionProtocol,
    InteractionSemantics,
    interaction_artifact_collections,
    interaction_update_from_artifact_collections,
)
from contract.market import (
    Direction,
    to_primitive,
)
from contract.eye import (
    INTERACTION_UPDATE_SCHEMA_VERSION,
    InteractionUpdate,
    MARKET_OBSERVATION_SCHEMA_VERSION,
    MicroBreakFact,
    PathSequenceLifecycle,
    QualifiedReacceptanceLifecycle,
    QualifiedReacceptanceState,
    ReacceptanceLifecycle,
    ReacceptanceState,
)
from shares.core.scene_graph import TemporalMarketSceneGraph

from shares.tests.helpers import market_observation, replace_market_observation
from eyes.tests.test_v3_group5_primitives import (
    PROTOCOL_PATH as LEGACY_PROTOCOL_PATH,
    _bos,
    _m1,
    _zone_formation,
)


PROTOCOL_PATH = "configs/primitives_interaction.json"


def test_hot_interaction_protocol_has_distinct_physical_identity() -> None:
    current_path = Path(PROTOCOL_PATH)
    legacy_path = Path(LEGACY_PROTOCOL_PATH)
    current_raw = current_path.read_text(encoding="utf-8")
    current = InteractionProtocol.from_file(current_path)
    legacy = InteractionProtocol.from_file(legacy_path)

    assert current.protocol_hash != legacy.protocol_hash
    assert current.later_hold_bars == legacy.later_hold_bars
    assert all(
        token not in current_raw
        for token in (
            "QualifiedReacceptance",
            "qualified_reacceptance",
            "micro_bos_aligned",
            "micro_bos_opposed",
            "micro_bos_confirmed",
            "typed_state_available",
            "brain_input_allowed",
            "dfp_lsr_input_authority_validated",
            "favr_natural_authority_validated",
            "favr_enabled",
            "independent_action_authority",
        )
    )
    assert {
        "typed_state_available",
        "brain_input_allowed",
        "dfp_lsr_input_authority_validated",
        "favr_natural_authority_validated",
        "favr_enabled",
        "independent_action_authority",
    }.isdisjoint(InteractionProtocol.__dataclass_fields__)


def _strict_interaction(*directions: Direction):
    semantics = InteractionSemantics(
        InteractionProtocol.from_file(PROTOCOL_PATH)
    )
    _, fvg, _ = _zone_formation(semantics)
    pullback = _m1(
        1,
        open_=100.5,
        high=100.75,
        low=99.5,
        close=100.5,
    )
    semantics.on_completed_1m(
        pullback,
        fair_value_gaps=(fvg,),
        m1_atr=1.0,
    )
    trigger = _m1(2)
    update = semantics.on_completed_1m(
        trigger,
        fair_value_gaps=(fvg,),
        m1_bos=tuple(
            _bos(
                identity=f"bos:raw-boundary:{index}",
                resolved_at=trigger.end,
                direction=direction,
            )
            for index, direction in enumerate(directions)
        ),
        m1_atr=1.0,
    )
    return update, trigger.end


def _aligned_interaction():
    return _strict_interaction(Direction.LONG)


def _reacceptance_interactions():
    semantics = InteractionSemantics(
        InteractionProtocol.from_file(PROTOCOL_PATH)
    )
    _, fvg, _ = _zone_formation(semantics)
    left = semantics.on_completed_1m(
        _m1(
            1,
            open_=100.5,
            high=100.75,
            low=99.5,
            close=99.75,
        ),
        fair_value_gaps=(fvg,),
        m1_atr=1.0,
    )
    boundary = semantics.on_boundary(
        "contract_change_reset",
        _m1(2).end,
        symbol="NQH5",
        instrument_id=2,
    )
    return left, boundary


def _held_interaction() -> tuple[InteractionSemantics, InteractionUpdate]:
    semantics = InteractionSemantics(
        InteractionProtocol.from_file(PROTOCOL_PATH)
    )
    _, fvg, _ = _zone_formation(semantics)
    semantics.on_completed_1m(
        _m1(
            1,
            open_=100.5,
            high=100.75,
            low=99.5,
            close=99.75,
        ),
        fair_value_gaps=(fvg,),
        m1_atr=1.0,
    )
    semantics.on_completed_1m(
        _m1(
            2,
            open_=99.75,
            high=100.5,
            low=99.5,
            close=100.25,
        ),
        fair_value_gaps=(fvg,),
        m1_atr=1.0,
    )
    held = semantics.on_completed_1m(
        _m1(
            3,
            open_=100.25,
            high=100.75,
            low=100.0,
            close=100.5,
        ),
        fair_value_gaps=(fvg,),
        m1_atr=1.0,
    )
    return semantics, held


def _held_boundary_interaction() -> InteractionUpdate:
    semantics, _ = _held_interaction()
    return semantics.on_boundary(
        "data_gap_reset",
        _m1(4).end,
        symbol="NQH5",
        instrument_id=1,
    )


def _assert_nested_tamper_rejected(
    update: InteractionUpdate,
    target: object,
    attribute: str,
    value: object,
) -> None:
    serialized_state = update.__getstate__()
    object.__setattr__(target, attribute, value)
    with pytest.raises(ValueError, match="canonical state changed"):
        update.validate_canonical_bindings()
    with pytest.raises(ValueError, match="canonical state changed"):
        pickle.dumps(update)
    with pytest.raises(ValueError, match="canonical state changed"):
        InteractionUpdate.__setstate__(
            object.__new__(InteractionUpdate),
            serialized_state,
        )


def test_eye_publishes_raw_break_and_brain_owns_qualification() -> None:
    update, _ = _aligned_interaction()

    fact = update.micro_break_facts[0]
    physical_path = update.interaction_paths[0]
    assert isinstance(fact, MicroBreakFact)
    assert not hasattr(fact, "outcome")
    assert not hasattr(fact, "qualified")
    assert not hasattr(update, "micro_bos_references")
    assert not hasattr(update, "path_sequences")
    assert physical_path.steps[-1].kind == "micro_break_observed"
    assert physical_path.transition_reason == (
        "first_strict_micro_break_observed"
    )

    # Frozen readers resolve the same IDs through the single Brain
    # interpreter; there is no second detector or semantic identity.
    interpreted = interpret_interaction_update(update)
    reference = interpreted.micro_bos_references[0]
    interpreted_path = interpreted.path_sequences[0]
    assert reference.reference_id == fact.reference_id
    assert reference.outcome == "aligned"
    assert reference.qualified
    assert interpreted_path.steps[-1].kind == "micro_bos_confirmed"
    assert interpreted_path.transition_reason == "micro_bos_aligned"


def test_reacceptance_is_physical_and_only_brain_names_qualification() -> None:
    _, update = _held_interaction()
    physical = update.reacceptance_interactions[0]
    path = update.interaction_paths[0]

    assert INTERACTION_UPDATE_SCHEMA_VERSION == 2
    assert type(physical) is ReacceptanceState
    assert physical.lifecycle is ReacceptanceLifecycle.HELD
    assert ReacceptanceState.__name__ == "ReacceptanceState"
    assert ReacceptanceLifecycle.__name__ == "ReacceptanceLifecycle"
    assert QualifiedReacceptanceState is ReacceptanceState
    assert QualifiedReacceptanceLifecycle is ReacceptanceLifecycle
    assert pickle.loads(
        b"ccontract.eye.entities\nQualifiedReacceptanceState\n."
    ) is ReacceptanceState
    assert pickle.loads(
        b"ccontract.eye.vocabulary\nQualifiedReacceptanceLifecycle\n."
    ) is ReacceptanceLifecycle
    legacy_pickle = (
        pickle.dumps(physical, protocol=0)
        .replace(
            b"ReacceptanceState",
            b"QualifiedReacceptanceState",
        )
        .replace(
            b"ReacceptanceLifecycle",
            b"QualifiedReacceptanceLifecycle",
        )
    )
    restored_legacy = pickle.loads(legacy_pickle)
    assert type(restored_legacy) is ReacceptanceState
    assert restored_legacy == physical
    assert {
        "QualifiedReacceptanceState",
        "QualifiedReacceptanceLifecycle",
    }.isdisjoint(shares.__all__)
    assert update.zone_interactions[0].transition_reason == "hold_completed"
    assert path.transition_reason == "hold_completed"
    physical_vocabulary = (
        path.transition_reason,
        *(step.kind for step in path.steps),
        *(step.reason for step in path.steps),
    )
    assert all(
        token not in value
        for value in physical_vocabulary
        for token in ("qualified", "aligned", "opposed", "successful")
    )

    interpreted = interpret_interaction_update(update)
    assert interpreted.path_sequences[0].transition_reason == (
        "qualified_reacceptance_held"
    )

    collections = to_primitive(dict(interaction_artifact_collections(update)))
    collections["interaction_reacceptance_interactions"][0][
        "qualified"
    ] = True
    with pytest.raises(ValueError, match="interaction artifact"):
        interaction_update_from_artifact_collections(collections)


@pytest.mark.parametrize(
    ("directions", "outcomes", "reason"),
    (
        ((Direction.LONG,), ("aligned",), "micro_bos_aligned"),
        ((Direction.SHORT,), ("opposed",), "micro_bos_opposed"),
        (
            (Direction.LONG, Direction.SHORT),
            ("ambiguous_same_clock", "ambiguous_same_clock"),
            "micro_bos_ambiguous_same_clock",
        ),
    ),
)
def test_brain_classifies_three_states_without_rewriting_eye_graph(
    directions: tuple[Direction, ...],
    outcomes: tuple[str, ...],
    reason: str,
) -> None:
    update, asof = _strict_interaction(*directions)
    assert {
        path.transition_reason for path in update.interaction_paths
    } == {"first_strict_micro_break_observed"}

    interpreted = interpret_interaction_update(update)
    assert tuple(
        reference.outcome
        for reference in interpreted.micro_bos_references
    ) == outcomes
    assert interpreted.path_sequences[0].transition_reason == reason

    observation = replace_market_observation(
        market_observation(asof=asof),
        interaction_update=update,
    )
    graph = TemporalMarketSceneGraph()
    graph.update(observation)
    path_node = next(
        node
        for node in graph.nodes_asof(asof)
        if node.kind == "path_sequence"
    )
    assert path_node.resolution_reason == (
        "first_strict_micro_break_observed"
    )
    micro_nodes = tuple(
        node
        for node in graph.nodes_asof(asof)
        if node.kind == "micro_bos"
    )
    assert len(micro_nodes) == len(directions)
    assert all(
        {"outcome", "qualified"}.isdisjoint(
            dict(node.semantic_attributes)
        )
        for node in micro_nodes
    )


def test_canonical_binding_rejects_forgery_and_remains_failure_atomic() -> None:
    update, _ = _aligned_interaction()
    baseline = interpret_interaction_update(update)
    fact = update.micro_break_facts[0]

    with pytest.raises(ValueError, match="micro-break fact custody"):
        replace(
            update,
            micro_break_facts=(
                replace(fact, context_direction=Direction.SHORT),
            ),
        )
    with pytest.raises(ValueError, match="micro-break fact custody"):
        replace(
            update,
            micro_break_facts=(
                replace(fact, target_swing_id="swing:forged-source"),
            ),
        )
    with pytest.raises(ValueError, match="zone-return path custody"):
        replace(
            update,
            interaction_paths=(
                replace(
                    update.interaction_paths[0],
                    protocol_hash="forged-protocol",
                ),
            ),
        )
    with pytest.raises(ValueError, match="path identities repeat"):
        replace(
            update,
            interaction_paths=(
                update.interaction_paths[0],
                update.interaction_paths[0],
            ),
        )
    transition = update.milestone_transitions[0]
    with pytest.raises(ValueError, match="milestone transitions repeat"):
        replace(
            update,
            milestone_transitions=(transition, transition),
        )
    path_without_break = replace(
        update.interaction_paths[0],
        steps=update.interaction_paths[0].steps[:-1],
    )
    with pytest.raises(ValueError, match="exact closing fact"):
        replace(
            update,
            micro_break_facts=(),
            interaction_paths=(path_without_break,),
            interaction_path_transitions=(),
            milestone_transitions=(),
        )
    with pytest.raises(ValueError, match="closing reason changed"):
        replace(
            update,
            interaction_paths=(
                replace(
                    update.interaction_paths[0],
                    transition_reason="location_left",
                ),
            ),
            interaction_path_transitions=(),
            milestone_transitions=(),
        )

    assert interpret_interaction_update(update) == baseline


def test_interaction_pickle_is_exact_and_rejects_legacy_nested_fields() -> None:
    update, _ = _aligned_interaction()
    assert pickle.loads(pickle.dumps(update)) == update

    state = dict(update.__getstate__())
    state["schema_version"] = 0
    with pytest.raises(ValueError, match="pickle schema changed"):
        InteractionUpdate.__setstate__(object.__new__(InteractionUpdate), state)
    missing = dict(update.__getstate__())
    missing.pop("schema_version")
    with pytest.raises(ValueError, match="pickle schema changed"):
        InteractionUpdate.__setstate__(
            object.__new__(InteractionUpdate),
            missing,
        )

    forged_fact = replace(update.micro_break_facts[0])
    object.__setattr__(forged_fact, "outcome", "aligned")
    with pytest.raises(ValueError, match="micro-break shape changed"):
        replace(update, micro_break_facts=(forged_fact,))

    forged_update = pickle.loads(pickle.dumps(update))
    object.__setattr__(forged_update, "legacy_group5_state", ())
    with pytest.raises(ValueError, match="interaction update shape changed"):
        pickle.dumps(forged_update)


def test_interaction_readmission_rejects_all_nested_dto_tampering() -> None:
    update, _ = _aligned_interaction()
    zone = update.zone_interactions[0]
    _assert_nested_tamper_rejected(
        update,
        zone,
        "formed_at",
        zone.formed_at.tz_localize(None),
    )

    update, _ = _aligned_interaction()
    fact = update.micro_break_facts[0]
    _assert_nested_tamper_rejected(update, fact, "scope", "continuation")

    update, _ = _aligned_interaction()
    path = update.interaction_paths[0]
    _assert_nested_tamper_rejected(
        update,
        path,
        "last_updated_at",
        path.formed_at - pd.Timedelta(minutes=1),
    )

    update, _ = _aligned_interaction()
    step = update.interaction_paths[0].steps[0]
    _assert_nested_tamper_rejected(update, step, "strength", 999.0)

    update, _ = _aligned_interaction()
    milestone = update.milestone_transitions[0][1]
    _assert_nested_tamper_rejected(update, milestone, "reason", "")

    update, _ = _aligned_interaction()
    transition_step = update.interaction_path_transitions[0].steps[0]
    _assert_nested_tamper_rejected(
        update,
        transition_step,
        "observed_at",
        transition_step.observed_at.tz_localize(None),
    )

    current, boundary = _reacceptance_interactions()
    reacceptance = current.reacceptance_interactions[0]
    _assert_nested_tamper_rejected(
        current,
        reacceptance,
        "lifecycle",
        "left",
    )
    boundary_reacceptance = boundary.reacceptance_interaction_transitions[0]
    _assert_nested_tamper_rejected(
        boundary,
        boundary_reacceptance,
        "strength",
        999.0,
    )


def test_trusted_brain_and_artifact_reads_reuse_one_dto_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    update, asof = _aligned_interaction()
    admissions = 0
    original = InteractionUpdate.validate_canonical_bindings

    def counted(value: InteractionUpdate) -> None:
        nonlocal admissions
        admissions += 1
        original(value)

    monkeypatch.setattr(
        InteractionUpdate,
        "validate_canonical_bindings",
        counted,
    )
    admitted = replace(update)
    assert admissions == 1

    observation = replace_market_observation(
        market_observation(asof=asof),
        interaction_update=admitted,
    )
    brain_observation_view(observation)
    interaction_artifact_collections(admitted)
    assert admissions == 1

    pickle.dumps(admitted)
    assert admissions == 2


def test_cold_source_identity_is_canonical_across_all_transports() -> None:
    update, _ = _aligned_interaction()
    with pytest.raises(ValueError, match="unique and sorted"):
        replace(update, cold_source_ids=("source:b", "source:a"))

    admitted = replace(
        update,
        cold_source_ids=("source:a", "source:b"),
    )
    assert pickle.loads(pickle.dumps(admitted)) == admitted
    collections = to_primitive(dict(interaction_artifact_collections(admitted)))
    assert interaction_update_from_artifact_collections(collections) == admitted


@pytest.mark.parametrize(
    ("collection", "nested_key", "nested_value"),
    (
        ("interaction_micro_break_facts", "expected_direction", "long"),
        ("interaction_micro_break_facts", "outcome", "aligned"),
        ("interaction_micro_break_facts", "qualified", True),
        ("interaction_micro_break_facts", "brain_response", {}),
    ),
)
def test_raw_artifact_gate_rejects_nested_brain_vocabulary(
    collection: str,
    nested_key: str,
    nested_value: object,
) -> None:
    update, _ = _aligned_interaction()
    collections = to_primitive(dict(interaction_artifact_collections(update)))
    collections[collection][0][nested_key] = nested_value

    with pytest.raises(ValueError, match="interaction artifact"):
        interaction_update_from_artifact_collections(collections)


def test_raw_artifact_gate_rejects_interpreted_path_and_extra_collection() -> None:
    update, _ = _aligned_interaction()
    collections = to_primitive(dict(interaction_artifact_collections(update)))
    collections["interaction_path_transitions"] = []
    collections["interaction_milestone_transitions"] = []
    path = collections["interaction_paths"][0]
    path["steps"][-1]["kind"] = "micro_bos_confirmed"
    path["steps"][-1]["reason"] = "micro_bos_aligned"
    path["transition_reason"] = "micro_bos_aligned"
    with pytest.raises(ValueError, match="physical vocabulary"):
        interaction_update_from_artifact_collections(collections)

    collections = to_primitive(dict(interaction_artifact_collections(update)))
    collections["brain_response"] = []
    assert tuple(collections) != INTERACTION_ARTIFACT_COLLECTION_NAMES
    with pytest.raises(ValueError, match="collection schema"):
        interaction_update_from_artifact_collections(collections)


def test_boundary_update_is_terminal_only_and_keeps_path_custody() -> None:
    semantics = InteractionSemantics(
        InteractionProtocol.from_file(PROTOCOL_PATH)
    )
    _, _, current = _zone_formation(semantics)
    boundary = semantics.on_boundary(
        "data_gap_reset",
        _m1(1).end,
        symbol="NQH5",
        instrument_id=1,
    )
    assert boundary.interaction_paths == ()
    assert boundary.zone_interactions == ()
    assert boundary.interaction_path_transitions

    with pytest.raises(
        ValueError,
        match="boundary cannot publish current entities",
    ):
        replace(
            boundary,
            zone_interactions=current.zone_interactions,
        )
    terminal_path = boundary.interaction_path_transitions[0]
    forged_origin = replace(
        terminal_path.steps[0],
        source_entity_id="zone:forged",
    )
    with pytest.raises(ValueError, match="zone-return path custody changed"):
        replace(
            boundary,
            interaction_path_transitions=(
                replace(
                    terminal_path,
                    steps=(forged_origin, *terminal_path.steps[1:]),
                ),
            ),
        )


def test_boundary_reacceptance_lineage_is_exact_and_context_unique() -> None:
    _, boundary = _reacceptance_interactions()
    assert len(boundary.reacceptance_interaction_transitions) == 1

    with pytest.raises(
        ValueError,
        match="reacceptance changed custody|live reacceptance lineage differs",
    ):
        replace(boundary, reacceptance_interaction_transitions=())

    terminal_path = boundary.interaction_path_transitions[0]
    without_reference = replace(
        terminal_path,
        steps=terminal_path.steps[:-1],
    )
    with pytest.raises(
        ValueError,
        match="reacceptance changed custody|live reacceptance lineage differs",
    ):
        replace(
            boundary,
            interaction_path_transitions=(without_reference,),
        )

    state = boundary.reacceptance_interaction_transitions[0]
    with pytest.raises(ValueError, match="contexts repeat"):
        replace(
            boundary,
            reacceptance_interaction_transitions=(
                state,
                replace(state, reacceptance_id="reacceptance:forged-sibling"),
            ),
        )
    collections = to_primitive(dict(interaction_artifact_collections(boundary)))
    forged_state = dict(collections["interaction_reacceptance_transitions"][0])
    forged_state["reacceptance_id"] = "reacceptance:forged-sibling"
    collections["interaction_reacceptance_transitions"].append(forged_state)
    with pytest.raises(ValueError, match="contexts repeat"):
        interaction_update_from_artifact_collections(collections)

    held_boundary = _held_boundary_interaction()
    assert held_boundary.reacceptance_interaction_transitions == ()
    assert any(
        step.kind == "reference_left"
        for step in held_boundary.interaction_path_transitions[0].steps
    )
    assert any(
        step.kind == "reacceptance_held"
        for step in held_boundary.interaction_path_transitions[0].steps
    )
    assert pickle.loads(pickle.dumps(held_boundary)) == held_boundary


def test_schema_v3_serializes_only_canonical_interaction_facts() -> None:
    update, asof = _aligned_interaction()
    observation = replace_market_observation(
        market_observation(asof=asof),
        interaction_update=update,
    )

    payload = to_primitive(observation)
    assert MARKET_OBSERVATION_SCHEMA_VERSION == 6
    assert "interaction_update" in payload
    assert "micro_break_facts" in payload["interaction_update"]
    assert "micro_bos_references" not in payload
    assert "path_sequences" not in payload
    assert "outcome" not in payload["interaction_update"][
        "micro_break_facts"
    ][0]
    view = brain_interaction_view(observation)
    assert view.micro_bos_references[0].qualified
    observation_view = brain_observation_view(observation)
    assert brain_observation_view(observation_view) is observation_view
    assert (
        brain_interaction_view(observation_view)
        is observation_view.interaction
    )
    legacy_fields = {
        "group5_typed_available",
        "group5_entry_location_transitions_this_update",
        "group5_reacceptance_transitions_this_update",
        "group5_micro_bos_transitions_this_update",
        "group5_path_transitions_this_update",
        "group5_step_transitions_this_update",
        "entry_locations",
        "qualified_reacceptances",
        "micro_bos_references",
        "path_sequences",
        "group5_boundary_path_transitions",
        "group5_boundary_reacceptance_transitions",
    }
    assert legacy_fields.isdisjoint(type(observation).__dataclass_fields__)
    assert all(not hasattr(observation, name) for name in legacy_fields)
    assert legacy_fields.isdisjoint(payload)
    restored = pickle.loads(pickle.dumps(observation))
    assert restored == observation
    assert all(not hasattr(restored, name) for name in legacy_fields)
    assert {
        "InteractionProtocol",
        "InteractionSemantics",
        "InteractionUpdate",
    }.isdisjoint(shares.__all__)

    with pytest.raises(TypeError, match="unexpected keyword"):
        replace_market_observation(
            observation,
            micro_bos_references=view.micro_bos_references,
        )


def test_raw_interaction_reasons_and_milestones_are_canonical() -> None:
    update, _ = _aligned_interaction()
    path = update.interaction_paths[0]
    forged_step = replace(
        path.steps[-1],
        reason="brain_qualified_outcome_aligned",
    )
    with pytest.raises(ValueError, match="physical vocabulary"):
        replace(
            update,
            interaction_paths=(
                replace(path, steps=(*path.steps[:-1], forged_step)),
            ),
            milestone_transitions=(),
        )

    first, second = path.steps[:2]
    ordered = replace(
        update,
        milestone_transitions=(
            (path.sequence_id, first),
            (path.sequence_id, second),
        ),
    )
    assert pickle.loads(pickle.dumps(ordered)) == ordered
    with pytest.raises(ValueError, match="predecessor order"):
        replace(
            update,
            milestone_transitions=(
                (path.sequence_id, second),
                (path.sequence_id, first),
            ),
        )


def test_legacy_boundary_adapter_rebuilds_censored_micro_step() -> None:
    reducer = legacy_group5.CausalGroup5Reducer(
        InteractionProtocol.from_file(LEGACY_PROTOCOL_PATH)
    )
    _, fvg, _ = _zone_formation(reducer)
    pullback = _m1(
        1,
        open_=100.5,
        high=100.75,
        low=99.5,
        close=100.5,
    )
    same_clock = reducer.on_completed_1m(
        pullback,
        fair_value_gaps=(fvg,),
        m1_bos=(
            _bos(
                identity="bos:same-clock-boundary",
                resolved_at=pullback.end,
                direction=Direction.LONG,
            ),
        ),
        m1_atr=1.0,
    )
    assert same_clock.micro_bos_references[0].outcome == (
        "simultaneous_unknown"
    )

    boundary = reducer.on_boundary(
        "data_gap_reset",
        _m1(2).end,
        symbol="NQH5",
        instrument_id=1,
    )
    path = boundary.path_transitions[0]
    assert path.lifecycle is PathSequenceLifecycle.CENSORED
    assert path.steps[-1].kind == "micro_bos_simultaneous"


