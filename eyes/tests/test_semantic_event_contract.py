from __future__ import annotations

from dataclasses import replace
import json
import pickle

import pandas as pd
import pytest

from eyes.core.event_store import EventStore
from contract.market import (
    SMC_SEMANTIC_VERSION,
    Timeframe,
)
from contract.eye import EventKind
from eyes.core.event_memory import EventMemory
from eyes.core.semantic_event_emitter import SemanticEventEmitter, _event
from eyes.core.semantics import (
    SEMANTIC_EVENT_BINDING_STATUSES,
    SemanticRegistry,
    SemanticRegistryError,
)

from eyes.tests.test_event_provenance_contract import (
    _legacy_range_boundary_candidate,
    _normalized_bar,
)


def _clock(minutes: int) -> pd.Timestamp:
    return pd.Timestamp("2026-08-19 10:00", tz="America/New_York") + pd.Timedelta(
        minutes=minutes
    )


def _swing_event(*, semantic_version: str = SMC_SEMANTIC_VERSION):
    return _event(
        EventKind.SWING_STATE,
        _clock(10),
        Timeframe.M5,
        "above",
        21_500.0,
        0.8,
        ("pivot-candle",),
        {
            "lower_bound": 21_499.75,
            "upper_bound": 21_500.25,
            "features": ["prominence", "duration"],
        },
        entity_id="swing:example",
        lifecycle="confirmed",
        formed_at=_clock(0),
        confirmed_at=_clock(10),
        event_time=_clock(0),
        semantic_version=semantic_version,
    )


def test_registered_v1_concepts_have_all_five_preregistration_sections() -> None:
    registry = SemanticRegistry.from_file()

    assert registry.semantic_version == SMC_SEMANTIC_VERSION
    assert len(registry.identity) == 64
    assert {
        "confirmed_swing",
        "structural_leg",
        "candidate_liquidity_level",
        "liquidity_sweep",
        "acceptance",
        "displacement",
        "raw_boundary_break",
        "structure_direction",
        "qualified_bos",
        "protected_swing",
        "mss_core",
        "fvg",
        "base_origin_core",
        "qualified_origin_zone",
        "structural_range",
        "balance_range",
        "premium_discount_irl_erl",
        "structure_regime",
        "delivery_phase",
        "dol_candidate",
    } <= set(registry.concepts)
    for concept in registry.concepts.values():
        assert concept.domain_meaning
        assert concept.operational_definition
        assert concept.hypothesized_relations
        assert concept.falsification_conditions
        assert concept.oos_pass_criteria
    assert registry.parameters.parameters["prominence_ATR"]["value"] is None
    assert {
        binding.status for binding in registry.event_bindings
    } <= SEMANTIC_EVENT_BINDING_STATUSES
    assert {
        binding.concept for binding in registry.event_bindings
    } == set(registry.concepts)


def test_v1_3_event_bindings_match_the_emitted_and_reserved_surface() -> None:
    registry = SemanticRegistry.from_file()
    by_kind = registry.event_binding_by_kind

    assert registry.canonical_emitted_event_kinds == {
        EventKind.SWING_CONFIRMED,
        EventKind.STRUCTURAL_LEG_CREATED,
        EventKind.LIQUIDITY_LEVEL_CREATED,
        EventKind.LEVEL_TOUCHED,
        EventKind.LEVEL_PENETRATED,
        EventKind.SWEEP_CONFIRMED,
        EventKind.ACCEPTANCE_CONFIRMED,
        EventKind.LEVEL_REACHED,
        EventKind.LEVEL_INVALIDATED,
        EventKind.DISPLACEMENT_OBSERVED,
        EventKind.RAW_BOUNDARY_BREAK,
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        EventKind.QUALIFIED_BOS,
        EventKind.PROTECTED_SWING_ASSIGNED,
        EventKind.MSS_CORE_CONFIRMED,
        EventKind.FVG_CREATED,
        EventKind.FVG_FIRST_RETEST,
        EventKind.FVG_PARTIALLY_FILLED,
        EventKind.FVG_MIDPOINT_TOUCHED,
        EventKind.FVG_FULLY_FILLED,
        EventKind.FVG_INVALIDATED,
        EventKind.DEALING_RANGE_CREATED,
        EventKind.BALANCE_RANGE_OBSERVED,
        EventKind.DEALING_RANGE_INVALIDATED,
        EventKind.DEALING_RANGE_REPLACED,
        EventKind.BASE_ORIGIN_CORE_CREATED,
        EventKind.QUALIFIED_ORIGIN_ZONE_CREATED,
        EventKind.ORIGIN_ZONE_MITIGATED,
        EventKind.ORIGIN_ZONE_INVALIDATED,
        EventKind.DELIVERY_PHASE_ENTERED,
        EventKind.DELIVERY_PHASE_UPDATED,
        EventKind.DELIVERY_PHASE_EXITED,
    }
    assert by_kind[EventKind.FVG_TOUCHED].status == (
        "compatibility_alias_not_emitted"
    )
    for kind in (
        EventKind.FVG_EXPIRED,
        EventKind.DEALING_RANGE_EXTENDED,
        # The Structural Range lifecycle has no maturity state to publish.
        EventKind.BALANCE_RANGE_MATURED,
    ):
        assert by_kind[kind].status == "reserved_not_emitted"
    for kind in (
        EventKind.DELIVERY_PHASE_CHANGED,
        EventKind.ORIGIN_ZONE_TOUCHED,
        EventKind.ORIGIN_ZONE_CREATED,
        EventKind.DEALING_RANGE_ACTIVATED,
    ):
        assert by_kind[kind].status == "compatibility_alias_not_emitted"
    assert {
        binding.concept
        for binding in registry.event_bindings
        if binding.status == "snapshot_derived"
    } == {
        "premium_discount_irl_erl",
        "structure_regime",
        "delivery_phase",
        "dol_candidate",
    }


def test_registry_rejects_invalid_event_binding_identity(tmp_path) -> None:
    registry = SemanticRegistry.from_file()
    original = json.loads(registry.source_path.read_text(encoding="utf-8"))

    mutations = (
        ("status", "invented_status", "invalid status"),
        ("event_kind", "not_an_event_kind", "unknown EventKind"),
        ("concept", "not_a_concept", "unknown concept"),
    )
    for field, value, message in mutations:
        payload = json.loads(json.dumps(original))
        payload["event_bindings"][0][field] = value
        changed = tmp_path / f"registry-{field}.yaml"
        changed.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SemanticRegistryError, match=message):
            SemanticRegistry.from_file(changed)


def test_registry_rejects_duplicate_json_keys_at_any_nesting(tmp_path) -> None:
    registry = SemanticRegistry.from_file()
    payload = registry.source_path.read_text(encoding="utf-8")
    changed_payload = payload.replace(
        '"owner": "eye",',
        '"owner": "eye",\n      "owner": "eye",',
        1,
    )
    assert changed_payload != payload
    changed = tmp_path / "duplicate-key-registry.yaml"
    changed.write_text(changed_payload, encoding="utf-8")

    with pytest.raises(SemanticRegistryError, match="duplicate JSON key"):
        SemanticRegistry.from_file(changed)


def test_v1_preregistration_matches_executable_semantic_boundaries() -> None:
    registry = SemanticRegistry.from_file()
    parameters = registry.parameters.parameters

    prominence = parameters["prominence_ATR"]
    assert prominence["value"] is None
    assert prominence["detector_default"] == 0.0
    assert "no positive hard prominence cutoff" in prominence["threshold_semantics"]
    assert "no positive hard threshold" in (
        registry.concepts["confirmed_swing"].operational_definition
    )

    expiry = parameters["fvg_expiry_bars"]
    assert expiry["value"] is None
    assert expiry["status"] == (
        "reserved_explicit_event_only_no_v1_3_detector_age_threshold"
    )
    fvg_definition = registry.concepts["fvg"].operational_definition
    assert "no age-based expiry threshold" in fvg_definition
    assert "FVG_EXPIRED is reserved" in fvg_definition

    acceptance = parameters["acceptance_bars"]["values"]
    assert acceptance[
        "active_dealing_range_external_acceptance_completed_h1_closes"
    ] == 1
    assert acceptance["range_manipulation_real_1m_outside_closes"] == 2
    range_definition = registry.concepts[
        "structural_range"
    ].operational_definition
    assert "first completed H1 close strictly outside" in range_definition
    assert "no M-bar hold is claimed" in range_definition

    extension = parameters["dealing_range_extension"]
    assert extension["value"] is None
    assert extension["status"] == "reserved_not_emitted_v1_3"
    assert "DEALING_RANGE_EXTENDED is reserved" in range_definition

    phase_definition = registry.concepts["delivery_phase"].operational_definition
    assert "entered/updated/exited lifecycle" in phase_definition
    assert "not a value recomputed each bar" in phase_definition
    assert (
        "Range-extension and volatility-compression are not v1.3 phase inputs"
    ) in phase_definition
    phase_binding = registry.concepts["delivery_phase"].existing_binding
    assert "DELIVERY_PHASE_CHANGED is a compatibility projection alias" in phase_binding
    # The binding has to say which of the three lifecycle kinds are live and
    # on what cadence, so the definition cannot imply a per-bar stream.
    assert (
        "an update only when a registered phase input moves while the phase"
        " does not"
    ) in phase_binding
    assert registry.semantic_version == "smc_semantics_v1.3"


def test_registry_and_parameters_must_share_one_semantic_version(tmp_path) -> None:
    registry = SemanticRegistry.from_file()
    payload = registry.parameters.source_path.read_text(encoding="utf-8").replace(
        SMC_SEMANTIC_VERSION,
        "smc_semantics_v1.4",
    )
    changed = tmp_path / "parameters.yaml"
    changed.write_text(payload, encoding="utf-8")

    with pytest.raises(SemanticRegistryError, match="versions differ"):
        type(registry.parameters).from_file(
            changed,
            expected_version=SMC_SEMANTIC_VERSION,
        )


def test_semantic_event_separates_market_time_from_first_knowledge_time() -> None:
    event = _swing_event()

    assert event.event_time == _clock(0)
    assert event.known_at == _clock(10)
    assert event.observed_at == event.known_at
    assert event.semantic_type == "swing_state"
    assert event.source_event_ids == ("pivot-candle",)
    assert event.zone == (21_499.75, 21_500.25)
    assert event.evidence == event.details


def test_lifecycle_transition_defaults_event_time_to_transition_clock() -> None:
    event = _event(
        EventKind.FVG_STATE,
        _clock(15),
        Timeframe.M5,
        "below",
        21_490.0,
        0.4,
        entity_id="fvg:example",
        lifecycle="invalidated",
        formed_at=_clock(0),
        ended_at=_clock(15),
        transition_reason="close_through_far_edge",
    )

    assert event.event_time == _clock(15)
    assert event.formed_at == _clock(0)


def test_semantic_event_payload_is_deeply_immutable_and_pickle_safe() -> None:
    event = _swing_event()

    with pytest.raises(TypeError, match="immutable"):
        event.details["lower_bound"] = 1.0
    assert event.details["features"] == ("prominence", "duration")
    assert pickle.loads(pickle.dumps(event)) == event


def test_known_at_is_the_only_replay_availability_gate() -> None:
    event = _swing_event()
    store = EventStore.from_events((event,))

    before = store.replay((), lambda state, item: (*state, item.event_id), known_at=_clock(5))
    at_confirmation = store.replay(
        (),
        lambda state, item: (*state, item.event_id),
        known_at=_clock(10),
    )

    assert before.events_applied == 0
    assert before.state == ()
    assert at_confirmation.events_applied == 1
    assert at_confirmation.state == (event.event_id,)


def test_event_store_is_idempotent_append_only_and_version_isolated() -> None:
    event = _swing_event()
    store = EventStore()

    assert store.append(event) is True
    assert store.append(event) is False
    with pytest.raises(ValueError, match="immutable history"):
        store.append(replace(event, price=21_501.0))
    with pytest.raises(ValueError, match="semantic versions"):
        store.append(replace(event, event_id="new-id", semantic_version="v2"))
    assert store.events() == (event,)


def test_event_store_treats_transport_resequencing_as_an_idempotent_retry() -> None:
    event = _swing_event()
    store = EventStore()

    assert store.append(event) is True
    assert store.append(replace(event, sequence_no=event.sequence_no + 99)) is False
    assert store.events() == (event,)


def test_event_store_batch_is_atomic_and_replay_is_repeatable() -> None:
    first = _swing_event()
    second = _event(
        EventKind.STRUCTURE_BREAK,
        _clock(15),
        Timeframe.M5,
        "above",
        21_501.0,
        0.7,
        (first.event_id,),
        direction=None,
    )
    invalid = replace(second, event_id="version-mismatch", semantic_version="v2")
    store = EventStore()

    with pytest.raises(ValueError, match="semantic versions"):
        store.append_batch((first, invalid))
    assert len(store) == 0

    assert store.append_batch((first, second)) == 2
    left = store.replay([], lambda state, event: [*state, event.event_id])
    right = store.replay([], lambda state, event: [*state, event.event_id])
    assert left == right
    assert left.event_fingerprint == store.fingerprint()


def test_event_memory_rejects_cross_version_timeline() -> None:
    memory = EventMemory(8)
    memory.append(_swing_event())

    with pytest.raises(ValueError, match="cannot mix semantic versions"):
        memory.append(
            replace(
                _event(
                    EventKind.LIQUIDITY_SWEEP,
                    _clock(15),
                    Timeframe.M5,
                    "above",
                    21_501.0,
                    0.5,
                ),
                semantic_version="smc_semantics_v2.0",
            )
        )


def test_failed_event_memory_append_does_not_bind_a_semantic_version() -> None:
    memory = EventMemory(8)

    with pytest.raises(ValueError, match="sequence floor"):
        memory.append(_swing_event(semantic_version="v2"), sequence_floor=-1)
    assert memory.semantic_version is None
    memory.append(_swing_event())
    assert memory.semantic_version == SMC_SEMANTIC_VERSION


def test_production_event_identity_is_bound_to_semantic_version() -> None:
    first = _swing_event()
    changed = _swing_event(
        semantic_version=f"{SMC_SEMANTIC_VERSION}.identity-test"
    )

    assert first.event_id != changed.event_id


def test_canonical_retry_keeps_first_known_strength_after_hot_key_eviction() -> None:
    emitter = object.__new__(SemanticEventEmitter)
    emitter.semantic_registry = SemanticRegistry.from_file()
    emitter.audit_store = EventStore()
    emitter.memory = EventMemory(1, audit_store=emitter.audit_store)
    range_context, candidate = _legacy_range_boundary_candidate(
        prefix="canonical-retry",
        minutes=9,
        timeframe=Timeframe.M5,
        side="above",
        level_id="level-1",
        price=21_500.0,
    )
    crossing_bar = _normalized_bar(
        "canonical-retry-bar",
        10,
        timeframe=Timeframe.M5,
        high=21_501.0,
        low=21_499.0,
        close=21_500.0,
    )
    emitter.audit_store.append_batch(
        (range_context, candidate, crossing_bar)
    )
    # Production records the completed bar through the same memory before any
    # semantic event derived from it, so the bar owns sequence 0 at that clock
    # and its derived events follow.  Without this the emitted event ties the
    # bar on (known_at, sequence_no) and canonical order falls back to the
    # content-addressed event_id.
    emitter.memory.append(crossing_bar, include_in_recent=False, audit=False)
    source_ids = (candidate.event_id, crossing_bar.event_id)
    crossing_clock = crossing_bar.known_at

    first = emitter._append_semantic_atomic(
        EventKind.LEVEL_TOUCHED,
        crossing_clock,
        Timeframe.M5,
        "above",
        21_500.0,
        0.25,
        source_ids,
        {"level_id": "level-1", "touch_ordinal": 1},
        event_time=crossing_clock,
    )
    emitter.memory.flush_audit()
    retry = emitter._append_semantic_atomic(
        EventKind.LEVEL_TOUCHED,
        crossing_clock,
        Timeframe.M5,
        "above",
        21_500.0,
        0.75,
        source_ids,
        {"level_id": "level-1", "touch_ordinal": 1},
        event_time=crossing_clock,
    )

    assert retry == first
    assert retry.strength == 0.25
    assert len(emitter.audit_store) == 4
    assert emitter.audit_store.get(first.event_id) == first


def test_bos_lifecycle_is_carried_by_its_terminal_kinds_only() -> None:
    """A pending BOS is already stated by RAW_BOUNDARY_BREAK's absence of a terminal.

    ``bos_state`` was an unregistered transport that only ever carried
    ``pending`` and was cited by nothing, so the BOS timeline now begins at its
    terminal kinds instead of duplicating what the structure-break events say.
    """

    from eyes.core.event_memory import EventMemory
    from contract.eye import BOSLifecycle

    initial = EventMemory._COMPLETE_INITIAL_LIFECYCLES["bos"]
    transitions = EventMemory._TIMELINE_TRANSITIONS["bos"]

    assert BOSLifecycle.PENDING.value not in initial
    assert BOSLifecycle.PENDING.value not in transitions
    assert initial == frozenset(
        {BOSLifecycle.CONFIRMED.value, BOSLifecycle.FAILED.value}
    )


def test_bos_state_is_not_an_emittable_event_kind() -> None:
    """Nothing may emit the retired pending-BOS transport."""

    assert not hasattr(EventKind, "BOS_STATE")
