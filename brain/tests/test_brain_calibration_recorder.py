from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from brain.core.brain_calibration import BrainCalibrationRecorder
from brain.core.brain_entry_sequence import (
    BrainInteractionView,
    BrainObservationView,
)
from shares.core.model import (
    BOSLifecycle,
    BOSScope,
    Bar,
    Direction,
    EntryLocationLifecycle,
    Playbook,
    PlaybookPhase,
    StructureLifecycle,
    Timeframe,
)


TZ = "America/New_York"


class _BrainFixtureObservation(BrainObservationView):
    """Mutable test harness around the internal Brain consumer view."""

    __slots__ = ()

    def __init__(
        self,
        observation: SimpleNamespace,
        *,
        entry_locations: tuple[object, ...] = (),
        path_sequences: tuple[object, ...] = (),
    ) -> None:
        object.__setattr__(self, "_observation", observation)
        object.__setattr__(
            self,
            "_interaction",
            BrainInteractionView(
                zone_interactions=entry_locations,
                reacceptance_interactions=(),
                micro_bos_references=(),
                path_sequences=path_sequences,
            ),
        )

    def __setattr__(self, name: str, value: object) -> None:
        interaction_field = {
            "entry_locations": "zone_interactions",
            "qualified_reacceptances": "reacceptance_interactions",
            "micro_bos_references": "micro_bos_references",
            "path_sequences": "path_sequences",
        }.get(name)
        if interaction_field is not None:
            object.__setattr__(
                self,
                "_interaction",
                replace(
                    self._interaction,
                    **{interaction_field: tuple(value)},
                ),
            )
            return
        setattr(self._observation, name, value)


def _recorder() -> BrainCalibrationRecorder:
    return BrainCalibrationRecorder()


def _clock(minute: int = 0) -> pd.Timestamp:
    return pd.Timestamp("2025-01-06 10:00", tz=TZ) + pd.Timedelta(
        minutes=minute
    )


def _location(
    lifecycle: EntryLocationLifecycle = EntryLocationLifecycle.IN_ZONE,
) -> SimpleNamespace:
    return SimpleNamespace(
        location_id="location-1",
        lifecycle=lifecycle,
    )


def _hypothesis(
    *,
    playbook: Playbook = Playbook.DISPLACEMENT_FIRST_PULLBACK,
    phase: PlaybookPhase = PlaybookPhase.EXECUTABLE,
    complete: bool = True,
    revision: str = "revision-1",
    setup_id: str = "setup-1",
    context_id: str = "context-1",
    episode_id: str | None = None,
    terminal_at: pd.Timestamp | None = None,
    terminal_reason: str | None = None,
    terminal_source_ids: tuple[str, ...] = (),
    readiness: float | None = None,
    delivery: float = 0.75,
    hard_gates: dict[str, bool] | None = None,
) -> SimpleNamespace:
    direction = Direction.LONG
    deadline = _clock(10)
    invalidation = SimpleNamespace(
        price=99.0,
        source_level_id="invalidation-1",
    )
    target = SimpleNamespace(level_id="draw-1", price=103.0)
    draw = SimpleNamespace(draw_id="draw-1", price=103.0)
    liquidity_route = SimpleNamespace(
        route_id="route-1",
        context_draw_id="context-draw-1",
        intermediate_liquidity_ids=("liquidity-1",),
        primary_deliverable_target_id="draw-1",
        terminal_draw_id="terminal-draw-1",
        path_blocker_ids=(),
        source_path_ids=("context-draw-1", "draw-1"),
    )
    plan = SimpleNamespace(
        deadline=deadline,
        invalidation=invalidation,
        entry_path_id="path-1",
        liquidity_route=liquidity_route,
    )
    steps = tuple(SimpleNamespace() for _ in range(5))
    sequence = SimpleNamespace(
        protocol_version="4.0.0",
        protocol_hash="a" * 64,
        setup_id=setup_id,
        completed_steps=5 if complete else 3,
        complete=complete,
        steps=steps,
    )
    key = f"{playbook.value}:{direction.value}"
    resolved_episode_id = setup_id if episode_id is None else episode_id
    readiness_value = (
        0.8 if complete else 0.0
    ) if readiness is None else readiness
    selected_trigger = (
        None
        if not complete or readiness_value <= 0.0
        else SimpleNamespace(
            trigger_id="trigger-1",
            trigger_kind="micro_bos_confirmed",
            observed_at=_clock(),
            entry_path_id="path-1",
            available_trigger_kinds=("micro_bos_confirmed",),
        )
    )
    return SimpleNamespace(
        key=key,
        playbook=playbook,
        direction=direction,
        phase=phase,
        sequence=sequence,
        raw_quality_dimensions={
            "thesis_strength": 0.7,
            "sequence_progress": 1.0 if complete else 0.6,
            "location_quality": 0.65,
            "entry_readiness": readiness_value,
            "delivery_quality": delivery,
            "uncertainty": 0.3,
        },
        raw_probability=0.7,
        thesis_strength=0.7,
        sequence_progress=1.0 if complete else 0.6,
        location_quality=0.65,
        entry_readiness=readiness_value,
        delivery_quality=delivery,
        uncertainty=0.3,
        setup_context_id=setup_id,
        episode_id=resolved_episode_id,
        context_id=context_id,
        context_thesis_id=context_id,
        parent_context_thesis_id=(
            context_id if resolved_episode_id is not None else None
        ),
        evidence_revision_id=revision,
        entry_location_id="location-1",
        invalidation=invalidation,
        draw_selection=draw,
        deliverable_targets=(target,),
        thesis_deadline=deadline,
        episode_deadline=deadline,
        plan=plan,
        liquidity_route=liquidity_route,
        terminal_at=terminal_at,
        terminal_reason=terminal_reason,
        terminal_source_ids=terminal_source_ids,
        hard_gate_results=(
            {"market_sequence": True}
            if hard_gates is None
            else hard_gates
        ),
        selected_trigger=selected_trigger,
    )


def _snapshot(
    minute: int = 0,
    *,
    hypotheses: tuple[SimpleNamespace, ...] | None = None,
    root_candidates: tuple[tuple[str, SimpleNamespace], ...] | None = None,
    lifecycle_candidates: tuple[
        tuple[str, SimpleNamespace], ...
    ] | None = None,
    location: SimpleNamespace | None = None,
    anomalies: tuple[str, ...] = (),
    price: float = 100.0,
    context_theses: dict[str, SimpleNamespace] | None = None,
) -> SimpleNamespace:
    hypotheses = hypotheses or (_hypothesis(),)
    raw_observation = SimpleNamespace(
        asof=_clock(minute),
        symbol="NQH5",
        instrument_id=1,
        price=price,
        anomalies=anomalies,
    )
    observation = _BrainFixtureObservation(
        raw_observation,
        entry_locations=(() if location is None else (location,)),
    )
    contexts = {
        f"scene:{item.key}": SimpleNamespace(
            hypothesis_id=f"scene:{item.key}",
            playbook=item.playbook,
            direction=item.direction,
            context_root_ids=(item.context_id, item.setup_context_id),
        )
        for item in hypotheses
    }
    belief = SimpleNamespace(
        hypotheses={item.key: item for item in hypotheses},
        context_hypotheses=contexts,
        scene_revision_id="scene-revision-1",
        candidates=lambda: tuple(hypotheses),
        context_theses={} if context_theses is None else context_theses,
    )
    if root_candidates is not None:
        belief.action_candidate_items = lambda: root_candidates
    if lifecycle_candidates is not None:
        belief.lifecycle_candidate_items = lambda: lifecycle_candidates
    return SimpleNamespace(
        observation=observation,
        belief=belief,
    )


def _context_thesis(
    context_id: str,
    *,
    lifecycle: str = "active",
    terminal_at: pd.Timestamp | None = None,
    terminal_reason: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        context_thesis_id=context_id,
        direction=Direction.LONG,
        lifecycle=lifecycle,
        terminal_at=terminal_at,
        terminal_reason=terminal_reason,
    )


def _bind_dfp_h4_context(
    hypothesis: SimpleNamespace,
    *,
    structure_id: str = "h4-structure-1",
) -> SimpleNamespace:
    hypothesis.sequence.steps = (
        SimpleNamespace(
            step_id="h4_structure_and_draw",
            satisfied=True,
            observed_at=_clock(-4),
            source_ids=(structure_id, "draw-1"),
        ),
    )
    hypothesis.sequence.completed_steps = 1
    return SimpleNamespace(
        structure_id=structure_id,
        direction=Direction.LONG,
        lifecycle=StructureLifecycle.CONFIRMED,
        confirmed_at=_clock(-5),
    )


def _with_structure_frames(
    snapshot: SimpleNamespace,
    *,
    h4_structures: tuple[SimpleNamespace, ...],
    h1_breaks: tuple[SimpleNamespace, ...] = (),
) -> SimpleNamespace:
    frames = {
        Timeframe.H4: SimpleNamespace(structures=h4_structures),
        Timeframe.H1: SimpleNamespace(structure_breaks=h1_breaks),
    }
    snapshot.observation.frame = lambda timeframe: frames[timeframe]
    return snapshot


def test_root_candidates_share_context_thesis_but_keep_entry_samples_distinct() -> None:
    hypothesis = _hypothesis()
    candidates = (
        ("root-a|dfp|long", hypothesis),
        ("root-b|dfp|long", hypothesis),
    )
    recorder = _recorder()

    recorder.observe(
        _snapshot(
            hypotheses=(hypothesis,),
            root_candidates=candidates,
            location=_location(),
        ),
        source_bar=_bar(-1),
    )

    immediate = recorder.drain_rows()
    assert len(immediate) == 4
    assert {row.hypothesis_key for row in immediate} == {
        candidate_id for candidate_id, _ in candidates
    }
    assert len(recorder.open_samples) == 7
    assert {sample.hypothesis_key for sample in recorder.open_samples} == {
        candidate_id for candidate_id, _ in candidates
    }
    assert len({sample.sample_id for sample in recorder.open_samples}) == 7
    thesis = [
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0].calibration_unit_id == "context-1"


def test_graph_free_action_override_can_fallback_from_empty_lifecycle_view(
) -> None:
    hypothesis = _hypothesis()
    candidate_id = "root-a|dfp|long"
    recorder = _recorder()

    recorder.observe(
        _snapshot(
            hypotheses=(hypothesis,),
            root_candidates=((candidate_id, hypothesis),),
            # Mirrors the graph-free MarketBelief test helper: action items
            # are explicit while the inherited production maps stay empty.
            lifecycle_candidates=(),
            location=_location(),
        ),
        source_bar=_bar(-1),
    )

    assert recorder.drain_rows()
    assert recorder.open_samples
    assert {
        sample.hypothesis_key for sample in recorder.open_samples
    } == {candidate_id}


def test_calibration_rows_freeze_global_market_mode_at_sample_clock() -> None:
    hypothesis = _hypothesis()
    snapshot = _snapshot(
        hypotheses=(hypothesis,),
        root_candidates=(("root-a|dfp|long", hypothesis),),
        location=_location(),
    )
    snapshot.belief.global_context = SimpleNamespace(
        market_mode=SimpleNamespace(value="balanced")
    )
    recorder = _recorder()

    recorder.observe(snapshot, source_bar=_bar(-1))

    assert {row.global_market_mode for row in recorder.drain_rows()} == {
        "balanced"
    }
    assert {sample.global_market_mode for sample in recorder.open_samples} == {
        "balanced"
    }


def test_calibration_rows_freeze_explicit_context_and_episode_ownership() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(
        phase=PlaybookPhase.WAITING_LOCATION,
        complete=False,
        setup_id="entry-episode-1",
        context_id="legacy-context-1",
        episode_id="entry-episode-1",
    )
    hypothesis.context_thesis_id = "context-thesis-1"
    hypothesis.parent_context_thesis_id = "context-thesis-1"

    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )
    restored = BrainCalibrationRecorder.from_state(recorder.state_dict())

    frozen = (*restored.rows, *restored.open_samples)
    assert frozen
    assert {item.context_id for item in frozen} == {"legacy-context-1"}
    assert {item.context_thesis_id for item in frozen} == {
        "context-thesis-1"
    }
    assert {item.parent_context_thesis_id for item in frozen} == {
        "context-thesis-1"
    }
    thesis = next(
        item
        for item in restored.open_samples
        if item.dimension == "thesis_strength"
    )
    assert thesis.calibration_unit_id == "context-thesis-1"
    assert thesis.calibration_unit_kind == "dfp_context_thesis"


def test_prime_deduplicates_context_thesis_without_entry_slot_collision() -> None:
    hypothesis = _hypothesis()
    candidates = (
        ("root-a|dfp|long", hypothesis),
        ("root-b|dfp|long", hypothesis),
    )
    recorder = _recorder()

    recorder.prime(
        _snapshot(
            hypotheses=(hypothesis,),
            root_candidates=candidates,
            location=_location(),
        ),
        source_bar=_bar(-1),
    )

    sample_payloads = tuple(
        json.loads(sample_id)
        for sample_id in recorder.state_dict()["seen_sample_ids"]
    )
    assert len(sample_payloads) == 9
    assert {payload["owner_key"] for payload in sample_payloads} >= {
        candidate_id for candidate_id, _ in candidates
    }
    thesis = [
        payload
        for payload in sample_payloads
        if payload["dimension"] == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0]["owner_key"] == "context-1"
    assert {
        tuple(item)
        for item in recorder.state_dict()["seen_location_ids"]
    } == {
        (candidate_id, "location-1") for candidate_id, _ in candidates
    }


def _bar(
    minute: int,
    *,
    high: float = 100.5,
    low: float = 99.5,
    close: float = 100.0,
    instrument_id: int = 1,
    data_gap: int = 0,
    synthetic: bool = False,
) -> Bar:
    return Bar(
        start=_clock(minute),
        open=100.0,
        high=high,
        low=low,
        close=close,
        volume=10.0,
        symbol="NQH5" if instrument_id == 1 else "NQM5",
        instrument_id=instrument_id,
        synthetic_no_trade=synthetic,
        data_gap_before_minutes=data_gap,
    )


def test_registers_four_targets_and_two_descriptive_rows_once() -> None:
    recorder = _recorder()
    recorder.observe(
        _snapshot(location=_location()),
        source_bar=_bar(-1),
    )

    immediate = recorder.drain_rows()
    assert {row.dimension for row in immediate} == {
        "sequence_progress",
        "uncertainty",
    }
    assert all(row.outcome_value is None for row in immediate)
    assert all(not row.fit_eligible and not row.censored for row in immediate)
    assert {item.dimension for item in recorder.open_samples} == {
        "thesis_strength",
        "location_quality",
        "entry_readiness",
        "delivery_quality",
    }
    expected_key = (
        f"{Playbook.DISPLACEMENT_FIRST_PULLBACK.value}:long"
    )
    assert all(row.hypothesis_key == expected_key for row in immediate)
    assert all(row.evidence_revision_id == "revision-1" for row in immediate)
    assert all(row.liquidity_route_id == "route-1" for row in immediate)
    assert all(row.selected_trigger_id == "trigger-1" for row in immediate)
    assert all(
        row.selected_trigger_kind == "micro_bos_confirmed"
        for row in immediate
    )
    assert all(row.selected_trigger_at == _clock() for row in immediate)
    assert all(
        json.loads(row.available_trigger_kinds)
        == ["micro_bos_confirmed"]
        for row in immediate
    )
    assert all(
        json.loads(row.sample_id)["dimension"] == row.dimension
        for row in immediate
    )

    # An unchanged evidence revision and phase do not create minute labels.
    recorder.on_bar(_bar(0, close=100.0))
    repeated = _hypothesis()
    # Route snapshots may receive a new per-bar route ID while the evidence
    # revision remains unchanged; that must not manufacture minute labels.
    repeated.liquidity_route.route_id = "route-2"
    recorder.observe(
        _snapshot(1, hypotheses=(repeated,), location=_location()),
        source_bar=_bar(0),
    )
    later = recorder.drain_rows()
    # A close that does not yet exceed the frozen trigger extreme keeps the
    # readiness target open; the unchanged evidence/phase creates no row.
    assert later == ()
    assert {item.dimension for item in recorder.open_samples} == {
        "thesis_strength",
        "location_quality",
        "entry_readiness",
        "delivery_quality",
    }


def test_recorder_freezes_hypothesis_specific_context_covariates() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis()
    hypothesis.context_metadata = {
        "authority_relation": "challenges_incumbent",
        "authority_rank_gap": "2",
        "conflict_role": "authority_transition_candidate",
        "conflict_scope": "external",
        "acceptance_state": "accepted",
        "obstruction_distance_R": "1.25",
        "free_path_R": "0.75",
        "soft_obstruction_count": "3",
        "hard_barrier_before_target": "false",
        "ambiguity_count": "1",
    }
    snapshot = _snapshot(
        hypotheses=(hypothesis,),
        location=_location(),
    )
    snapshot.belief.global_context = SimpleNamespace(
        material_conflicts=(),
        ambiguous_evidence=(),
        obstruction_views={},
        scale_relation_details={},
    )

    recorder.observe(snapshot, source_bar=_bar(-1))

    rows = (*recorder.rows, *recorder.open_samples)
    assert rows
    assert {row.authority_relation for row in rows} == {
        "challenges_incumbent"
    }
    assert {row.authority_rank_gap for row in rows} == {2}
    assert {row.conflict_role for row in rows} == {
        "authority_transition_candidate"
    }
    assert {row.conflict_scope for row in rows} == {"external"}
    assert {row.acceptance_state for row in rows} == {"accepted"}
    assert {row.obstruction_distance_R for row in rows} == {1.25}
    assert {row.free_path_R for row in rows} == {0.75}
    assert {row.soft_obstruction_count for row in rows} == {3}
    assert {row.hard_barrier_before_target for row in rows} == {False}
    assert {row.ambiguity_count for row in rows} == {1}


def test_recorder_freezes_compact_open_thesis_binding_diagnostics() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis()
    hypothesis.market_thesis_id = "market-thesis:1"
    hypothesis.bound_market_thesis_id = "market-thesis:1"
    hypothesis.market_thesis_root_id = "displacement:1"
    hypothesis.market_thesis_mechanism = "directional_displacement"
    hypothesis.market_thesis_authority_relation = "aligned"
    hypothesis.playbook_match_strength = 0.8
    hypothesis.market_thesis_binding_required = True
    hypothesis.market_thesis_action_bound = True
    hypothesis.market_thesis_match_status = "exact_root_bound"
    hypothesis.hard_gate_results = {
        "causal_order": True,
        "entry_trigger": False,
    }
    hypothesis.context_metadata = {
        "playbook_plan_delivery_valid": "true"
    }

    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )
    restored = BrainCalibrationRecorder.from_state(recorder.state_dict())
    frozen = (*restored.rows, *restored.open_samples)
    assert frozen
    assert {row.market_thesis_id for row in frozen} == {"market-thesis:1"}
    assert {row.bound_market_thesis_id for row in frozen} == {
        "market-thesis:1"
    }
    assert {row.market_thesis_root_id for row in frozen} == {
        "displacement:1"
    }
    assert {row.market_thesis_mechanism for row in frozen} == {
        "directional_displacement"
    }
    assert {row.playbook_match_strength for row in frozen} == {0.8}
    assert {row.market_thesis_match_status for row in frozen} == {
        "exact_root_bound"
    }
    assert {row.playbook_first_failed_hard_gate_id for row in frozen} == {
        "entry_trigger"
    }
    assert all(row.playbook_plan_delivery_valid for row in frozen)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("obstruction_distance_R", False),
        ("free_path_R", True),
    ),
)
def test_recorder_rejects_boolean_distance_metadata(
    field: str,
    value: bool,
) -> None:
    recorder = _recorder()
    hypothesis = _hypothesis()
    hypothesis.context_metadata = {field: value}

    with pytest.raises(
        ValueError,
        match="calibration numeric metadata cannot be boolean",
    ):
        recorder.observe(
            _snapshot(hypotheses=(hypothesis,), location=_location()),
            source_bar=_bar(-1),
        )


def test_unbound_single_global_conflict_does_not_contaminate_row() -> None:
    recorder = _recorder()
    snapshot = _snapshot(location=_location())
    snapshot.belief.global_context = SimpleNamespace(
        material_conflicts=(
            SimpleNamespace(
                affected_hypothesis_ids=("another-setup",),
                observed_at=_clock(),
                role=SimpleNamespace(value="authority_transition_candidate"),
                structural_scale="external",
                source_timeframe=Timeframe.H1,
            ),
        ),
        ambiguous_evidence=(),
        obstruction_views={},
        scale_relation_details={},
    )

    recorder.observe(snapshot, source_bar=_bar(-1))

    rows = (*recorder.rows, *recorder.open_samples)
    assert {row.authority_relation for row in rows} == {"unrelated"}
    assert {row.conflict_role for row in rows} == {"none"}


def test_broad_global_slot_conflict_never_reinterprets_unrelated_hypothesis() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis()
    hypothesis.context_metadata = {
        "authority_relation": "unrelated",
        "authority_rank_gap": 0,
        "conflict_role": "none",
        "conflict_scope": "none",
        "acceptance_state": "unknown",
        "soft_obstruction_count": 0,
        "ambiguity_count": 0,
    }
    snapshot = _snapshot(hypotheses=(hypothesis,), location=_location())
    snapshot.belief.global_context = SimpleNamespace(
        material_conflicts=(
            SimpleNamespace(
                # Global context lists fixed directional slots, not proof of
                # an identity connection to this exact setup.
                affected_hypothesis_ids=(hypothesis.key,),
                observed_at=_clock(),
                role=SimpleNamespace(value="authority_transition_candidate"),
                structural_scale="external",
                source_timeframe=Timeframe.H1,
            ),
        ),
        ambiguous_evidence=("unrelated:ambiguity",),
        obstruction_views={
            Direction.LONG.value: SimpleNamespace(
                hard_barriers=(SimpleNamespace(),),
                soft_frictions=(SimpleNamespace(),),
            )
        },
        scale_relation_details={
            Timeframe.H1.value: SimpleNamespace(
                relation=SimpleNamespace(value="material_opposition"),
                structural_scope="external",
                acceptance_state="accepted",
            )
        },
    )

    recorder.observe(snapshot, source_bar=_bar(-1))

    rows = (*recorder.rows, *recorder.open_samples)
    assert {row.authority_relation for row in rows} == {"unrelated"}
    assert {row.conflict_role for row in rows} == {"none"}
    assert {row.soft_obstruction_count for row in rows} == {0}
    assert {row.ambiguity_count for row in rows} == {0}


def test_recorder_uses_action_candidates_without_scene_projection_metadata() -> None:
    recorder = _recorder()
    snapshot = _snapshot(location=_location())
    hypothesis = snapshot.belief.candidates()[0]
    snapshot.belief.context_hypotheses = {}
    snapshot.belief.scene_revision_id = None
    snapshot.belief.hypotheses = {}

    recorder.observe(snapshot, source_bar=_bar(-1))

    rows = recorder.drain_rows()
    assert rows
    assert all(row.hypothesis_key == hypothesis.key for row in rows)
    assert all("scene_hypothesis_id" not in row.to_dict() for row in rows)


def test_favr_is_parked_and_never_registered() -> None:
    recorder = _recorder()
    favr = _hypothesis(playbook=Playbook.FAILED_AUCTION_VALUE_RETURN)
    recorder.observe(
        _snapshot(hypotheses=(favr,), location=_location()),
        source_bar=_bar(-1),
    )
    assert recorder.rows == ()
    assert recorder.open_samples == ()


def test_same_bar_invalidation_wins_over_draw() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL
    )
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    terminal_bar = _bar(0, high=104.0, low=98.5, close=103.5)
    recorder.on_bar(terminal_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(hypothesis,),
            location=_location(),
            price=terminal_bar.close,
        ),
        source_bar=terminal_bar,
    )
    rows = recorder.drain_rows()
    assert len(rows) == 4
    assert {row.dimension for row in rows} == {
        "thesis_strength",
        "location_quality",
        "entry_readiness",
        "delivery_quality",
    }
    assert all(row.outcome_value == 0.0 for row in rows)
    assert all(row.resolution == "invalidation_touched" for row in rows)


@pytest.mark.parametrize("anomaly", ("data_anomaly", "tick_size_mismatch"))
def test_same_bar_observation_anomaly_censors_pending_invalidation(
    anomaly: str,
) -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL
    )
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    anomalous_bar = _bar(0, high=104.0, low=98.5, close=103.5)
    recorder.on_bar(anomalous_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(hypothesis,),
            location=_location(),
            anomalies=(anomaly,),
            price=anomalous_bar.close,
        ),
        source_bar=anomalous_bar,
    )
    rows = recorder.drain_rows()

    assert {row.dimension for row in rows} == {
        "thesis_strength",
        "location_quality",
        "entry_readiness",
        "delivery_quality",
    }
    assert all(row.outcome_value is None for row in rows)
    assert all(row.censored and not row.fit_eligible for row in rows)
    assert all(row.resolution == "observation_boundary" for row in rows)


def test_dfp_thesis_does_not_inherit_entry_zone_invalidation() -> None:
    recorder = _recorder()
    recorder.observe(
        _snapshot(location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    invalidation_bar = _bar(0, high=100.5, low=98.5, close=99.5)
    recorder.on_bar(invalidation_bar)
    recorder.observe(
        _snapshot(
            1,
            location=_location(),
            price=invalidation_bar.close,
        ),
        source_bar=invalidation_bar,
    )
    rows = recorder.drain_rows()
    assert {row.dimension for row in rows} == {
        "location_quality",
        "entry_readiness",
        "delivery_quality",
    }
    thesis = [
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0].invalidation_price is None


def test_readiness_accepts_the_first_advance_before_deadline() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(phase=PlaybookPhase.WAITING_TRIGGER)
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    advance = _bar(0, high=100.8, close=100.75)
    recorder.on_bar(advance)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(hypothesis,),
            location=_location(),
            price=advance.close,
        ),
        source_bar=advance,
    )
    rows = recorder.drain_rows()
    readiness = [row for row in rows if row.dimension == "entry_readiness"]
    assert len(readiness) == 1
    assert readiness[0].outcome_value == 1.0
    assert readiness[0].resolution == "close_advanced_before_deadline"
    assert {item.dimension for item in recorder.open_samples} == {
        "thesis_strength",
        "location_quality",
        "delivery_quality",
    }


def test_valid_trigger_registers_readiness_without_delivery_plan() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis()
    hypothesis.plan = None
    snapshot = _snapshot(
        hypotheses=(hypothesis,),
        location=_location(),
    )
    snapshot.observation.path_sequences = (
        SimpleNamespace(
            context_kind="zone_return",
            context_id="location-1",
            direction=Direction.LONG,
            last_updated_at=_clock(0),
            sequence_id="path-1",
        ),
    )

    recorder.observe(snapshot, source_bar=_bar(-1))

    dimensions = {sample.dimension for sample in recorder.open_samples}
    assert "entry_readiness" in dimensions
    assert "delivery_quality" not in dimensions
    readiness = next(
        sample
        for sample in recorder.open_samples
        if sample.dimension == "entry_readiness"
    )
    assert readiness.target_deadline_kind == "entry_deadline"
    assert readiness.deadline == hypothesis.episode_deadline
    assert readiness.draw_id == "draw-1"


def test_planless_waiting_trigger_uses_explicit_frozen_entry_path() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(phase=PlaybookPhase.WAITING_TRIGGER)
    hypothesis.plan = None
    hypothesis.entry_path_id = "frozen-waiting-trigger-path"
    hypothesis.selected_trigger.entry_path_id = "frozen-waiting-trigger-path"

    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )

    path_owned = {
        sample.dimension: sample.calibration_unit_id
        for sample in recorder.open_samples
        if sample.dimension in {
            "location_quality",
            "entry_readiness",
            "delivery_quality",
        }
    }
    assert path_owned == {
        "location_quality": "frozen-waiting-trigger-path",
        "entry_readiness": "frozen-waiting-trigger-path",
    }


def test_recorder_rejects_conflicting_belief_and_plan_entry_paths() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis()
    hypothesis.entry_path_id = "belief-path"
    hypothesis.plan.entry_path_id = "different-plan-path"

    with pytest.raises(
        ValueError,
        match="belief and plan entry path identities disagree",
    ):
        recorder.observe(
            _snapshot(hypotheses=(hypothesis,), location=_location()),
            source_bar=_bar(-1),
        )


def test_recorder_rejects_explicit_entry_path_with_wrong_scene_owner() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis()
    hypothesis.entry_path_id = "path-1"
    snapshot = _snapshot(hypotheses=(hypothesis,), location=_location())
    snapshot.observation.path_sequences = (
        SimpleNamespace(
            context_kind="zone_return",
            context_id="different-location",
            direction=Direction.LONG,
            last_updated_at=_clock(0),
            sequence_id="path-1",
        ),
    )

    with pytest.raises(
        ValueError,
        match="frozen entry path disagrees with its belief owner",
    ):
        recorder.observe(snapshot, source_bar=_bar(-1))


def test_completed_episode_does_not_backfill_thesis_or_location_success() -> None:
    recorder = _recorder()
    active = _hypothesis(
        phase=PlaybookPhase.WAITING_LOCATION,
        complete=False,
    )
    recorder.observe(
        _snapshot(hypotheses=(active,), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    quiet_bar = _bar(0, high=100.5, low=99.5, close=100.0)
    recorder.on_bar(quiet_bar)
    completed = _hypothesis(
        phase=PlaybookPhase.COMPLETED,
        complete=False,
        terminal_at=_clock(1),
        terminal_reason="position_completed",
        terminal_source_ids=("context-1", "path-1"),
    )
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(completed,),
            location=_location(EntryLocationLifecycle.IN_ZONE),
            price=quiet_bar.close,
        ),
        source_bar=quiet_bar,
    )

    assert recorder.drain_rows() == ()
    assert {sample.dimension for sample in recorder.open_samples} == {
        "thesis_strength",
        "location_quality",
    }


@pytest.mark.parametrize(
    "playbook,reason",
    [
        (
            Playbook.DISPLACEMENT_FIRST_PULLBACK,
            "global_authority_invalidated",
        ),
        (
            Playbook.LIQUIDITY_SWEEP_REVERSAL,
            "global_frozen_source_invalidated",
        ),
    ],
)
def test_global_authority_or_source_invalidation_is_thesis_failure(
    playbook: Playbook,
    reason: str,
) -> None:
    recorder = _recorder()
    active = _hypothesis(
        playbook=playbook,
        phase=PlaybookPhase.ARMED,
        complete=False,
    )
    recorder.observe(
        _snapshot(hypotheses=(active,)),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    quiet_bar = _bar(0)
    recorder.on_bar(quiet_bar)
    terminal = _hypothesis(
        playbook=playbook,
        phase=PlaybookPhase.INVALIDATED,
        complete=False,
        terminal_reason=reason,
        terminal_source_ids=(
            "context-1" if playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
            else "context-1",
        ),
    )
    contexts = (
        {}
        if playbook is Playbook.DISPLACEMENT_FIRST_PULLBACK
        else {
            "context-1": _context_thesis(
                "context-1",
                lifecycle="invalidated",
                terminal_at=_clock(1),
                terminal_reason=reason,
            )
        }
    )
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(terminal,),
            price=quiet_bar.close,
            context_theses=contexts,
        ),
        source_bar=quiet_bar,
    )

    thesis = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0].outcome_value == 0.0
    assert not thesis[0].censored
    assert thesis[0].resolution == f"thesis_contradicted:{reason}"


def test_replaced_dfp_context_terminal_closes_only_exact_frozen_structure(
) -> None:
    recorder = _recorder()
    candidate_id = "root-1|dfp|long"
    active = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id="old-context",
        context_id="old-context",
    )
    old_structure = _bind_dfp_h4_context(
        active,
        structure_id="old-h4-structure",
    )
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                hypotheses=(active,),
                root_candidates=((candidate_id, active),),
            ),
            h4_structures=(old_structure,),
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    frozen = next(
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
    )

    quiet_bar = _bar(0)
    recorder.on_bar(quiet_bar)
    replacement = _hypothesis(
        phase=PlaybookPhase.INVALIDATED,
        complete=False,
        setup_id="new-context",
        context_id="new-context",
        terminal_reason="global_frozen_source_invalidated",
        terminal_source_ids=("old-h4-structure",),
    )
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                1,
                hypotheses=(replacement,),
                root_candidates=((candidate_id, replacement),),
                price=quiet_bar.close,
            ),
            # The explicit terminal_source_id, rather than absence from this
            # bounded frame, owns the old context's invalidation.
            h4_structures=(),
        ),
        source_bar=quiet_bar,
    )

    rows = [
        row
        for row in recorder.drain_rows()
        if row.sample_id == frozen.sample_id
    ]
    assert len(rows) == 1
    assert rows[0].outcome_value == 0.0
    assert not rows[0].censored
    assert rows[0].resolution == (
        "thesis_contradicted:global_frozen_source_invalidated"
    )


@pytest.mark.parametrize(
    "playbook",
    [
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
    ],
)
def test_local_frozen_source_terminal_does_not_fail_context_thesis(
    playbook: Playbook,
) -> None:
    recorder = _recorder()
    active = _hypothesis(
        playbook=playbook,
        phase=PlaybookPhase.ARMED,
        complete=False,
    )
    recorder.observe(
        _snapshot(hypotheses=(active,)),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    quiet_bar = _bar(0)
    recorder.on_bar(quiet_bar)
    terminal = _hypothesis(
        playbook=playbook,
        phase=PlaybookPhase.INVALIDATED,
        complete=False,
        terminal_reason="global_frozen_source_invalidated",
        terminal_source_ids=("local-fvg-or-entry-path",),
    )
    recorder.observe(
        _snapshot(1, hypotheses=(terminal,), price=quiet_bar.close),
        source_bar=quiet_bar,
    )

    assert not any(
        row.dimension == "thesis_strength"
        for row in recorder.drain_rows()
    )
    assert any(
        sample.dimension == "thesis_strength"
        for sample in recorder.open_samples
    )


def test_missing_frozen_geometry_is_not_backfilled_from_future_snapshot() -> None:
    recorder = _recorder()
    incomplete = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
    )
    incomplete.draw_selection = None
    incomplete.deliverable_targets = ()
    recorder.observe(
        _snapshot(hypotheses=(incomplete,)),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    assert not any(
        sample.dimension == "thesis_strength"
        for sample in recorder.open_samples
    )
    assert recorder.late_registration_summary[
        "incomplete_registration_skipped"
    ] == 1

    # The same evidence revision later gains a draw. It must not be sampled
    # with this future observation's clock/raw value.
    complete_later = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
    )
    later_bar = _bar(0)
    recorder.on_bar(later_bar)
    recorder.observe(
        _snapshot(1, hypotheses=(complete_later,), price=later_bar.close),
        source_bar=later_bar,
    )
    assert not any(
        sample.dimension == "thesis_strength"
        for sample in recorder.open_samples
    )
    assert recorder.late_registration_summary[
        "incomplete_registration_skipped"
    ] == 1


def test_readiness_stays_open_through_pause_then_advances() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(phase=PlaybookPhase.WAITING_TRIGGER)
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    # The first completed bar pauses inside the trigger bar's frozen range.
    recorder.on_bar(_bar(0, high=100.4, close=100.0))
    assert recorder.drain_rows() == ()
    assert any(
        sample.dimension == "entry_readiness"
        for sample in recorder.open_samples
    )

    # A later completed bar, still before the frozen deadline, advances.
    advance = _bar(1, high=100.8, close=100.75)
    recorder.on_bar(advance)
    recorder.observe(
        _snapshot(
            2,
            hypotheses=(hypothesis,),
            location=_location(),
            price=advance.close,
        ),
        source_bar=advance,
    )
    readiness = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "entry_readiness"
    ]
    assert len(readiness) == 1
    assert readiness[0].sampled_at == _clock(0)
    assert readiness[0].resolved_at == _clock(2)
    assert readiness[0].outcome_value == 1.0
    assert readiness[0].resolution == "close_advanced_before_deadline"


def test_late_readiness_uses_checkpointed_first_trigger_bar_geometry() -> None:
    recorder = _recorder()
    trigger_bar = _bar(-1, high=100.5, low=99.5, close=100.0)
    waiting = _hypothesis(
        phase=PlaybookPhase.WAITING_TRIGGER,
        readiness=0.0,
    )
    waiting.selected_trigger = SimpleNamespace(
        trigger_id="trigger-1",
        trigger_kind="micro_bos_confirmed",
        observed_at=_clock(),
        entry_path_id="path-1",
        available_trigger_kinds=("micro_bos_confirmed",),
    )
    recorder.observe(
        _snapshot(hypotheses=(waiting,), location=_location()),
        source_bar=trigger_bar,
    )
    recorder.drain_rows()

    # Checkpointing between trigger formation and readiness must preserve the
    # exact completed trigger bar, not force a future-bar fallback.
    resumed = BrainCalibrationRecorder.from_state(recorder.state_dict())
    # Keep the later bar below the Context draw; this test isolates trigger-
    # geometry persistence and must not simultaneously terminate the thesis.
    later_bar = _bar(0, high=102.5, low=99.5, close=100.1)
    resumed.on_bar(later_bar)
    ready = _hypothesis(
        phase=PlaybookPhase.WAITING_TRIGGER,
        readiness=0.8,
    )
    ready.selected_trigger = waiting.selected_trigger
    resumed.observe(
        _snapshot(
            1,
            hypotheses=(ready,),
            location=_location(),
            price=later_bar.close,
        ),
        source_bar=later_bar,
    )
    resumed.drain_rows()

    sample = next(
        item
        for item in resumed.open_samples
        if item.dimension == "entry_readiness"
    )
    assert sample.selected_trigger_at == _clock()
    assert sample.sampled_at == _clock(1)
    assert sample.trigger_bar_high == 100.5
    assert sample.trigger_bar_low == 99.5

    # This close advances beyond the original trigger bar, but remains far
    # below the later bar's high.  Success therefore proves the frozen clock's
    # geometry survived delayed registration and resume.
    advance = _bar(1, high=100.7, low=99.5, close=100.6)
    resumed.on_bar(advance)
    resumed.observe(
        _snapshot(
            2,
            hypotheses=(ready,),
            location=_location(),
            price=advance.close,
        ),
        source_bar=advance,
    )
    readiness = [
        row
        for row in resumed.drain_rows()
        if row.dimension == "entry_readiness"
    ]
    assert len(readiness) == 1
    assert readiness[0].outcome_value == 1.0
    assert readiness[0].resolution == "close_advanced_before_deadline"


def test_later_descriptive_revision_keeps_frozen_trigger_geometry() -> None:
    recorder = _recorder()
    trigger_bar = _bar(-1, high=100.5, low=99.5, close=100.0)
    hypothesis = _hypothesis(revision="revision-1")
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=trigger_bar,
    )
    recorder.drain_rows()

    # A later evidence revision can occur after price has left the original
    # trigger bar.  The row's origin remains the current completed close,
    # while trigger geometry remains bound to the first trigger clock.
    later_bar = _bar(0, high=101.5, low=99.8, close=101.0)
    recorder.on_bar(later_bar)
    revised = _hypothesis(revision="revision-2")
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(revised,),
            location=_location(),
            price=later_bar.close,
        ),
        source_bar=later_bar,
    )

    uncertainty = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "uncertainty"
        and row.evidence_revision_id == "revision-2"
    ]
    assert len(uncertainty) == 1
    assert uncertainty[0].origin_price == 101.0
    assert uncertainty[0].trigger_bar_high == 100.5
    assert uncertainty[0].trigger_bar_low == 99.5
    assert uncertainty[0].selected_trigger_at == _clock()


def test_readiness_fails_only_when_deadline_expires_without_advance() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(phase=PlaybookPhase.WAITING_TRIGGER)
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    for minute in range(9):
        pause = _bar(minute, high=100.4, close=100.0)
        recorder.on_bar(pause)
        recorder.observe(
            _snapshot(
                minute + 1,
                hypotheses=(hypothesis,),
                location=_location(),
                price=pause.close,
            ),
            source_bar=pause,
        )
        assert not any(
            row.dimension == "entry_readiness"
            for row in recorder.drain_rows()
        )

    # The completed bar ending exactly on the frozen deadline can establish
    # only that no qualifying close advance occurred inside the horizon.
    deadline_bar = _bar(9, high=100.4, close=100.0)
    recorder.on_bar(deadline_bar)
    recorder.observe(
        _snapshot(
            10,
            hypotheses=(hypothesis,),
            location=_location(),
            price=deadline_bar.close,
        ),
        source_bar=deadline_bar,
    )
    readiness = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "entry_readiness"
    ]
    assert len(readiness) == 1
    assert readiness[0].resolved_at == _clock(10)
    assert readiness[0].outcome_value == 0.0
    assert readiness[0].resolution == "deadline_without_trigger_advance"


def test_dfp_thesis_keeps_context_owner_across_entry_path_setup() -> None:
    recorder = _recorder()
    context_id = "dfp-context:structure-1:draw-1"
    context = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="context-revision",
        setup_id=context_id,
        context_id=context_id,
    )
    recorder.observe(
        _snapshot(hypotheses=(context,)),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    thesis_before = [
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
    ]
    assert len(thesis_before) == 1
    assert thesis_before[0].setup_id == context_id
    assert thesis_before[0].calibration_unit_id == context_id
    assert thesis_before[0].calibration_unit_kind == "dfp_context_thesis"
    recorder.on_bar(_bar(0))
    entry_path = _hypothesis(
        phase=PlaybookPhase.WAITING_LOCATION,
        complete=False,
        revision="context-revision",
        setup_id="entry-path-episode-1",
        context_id=context_id,
    )
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(entry_path,),
            location=_location(),
        ),
        source_bar=_bar(0),
    )

    transition_rows = recorder.drain_rows()
    assert not any(row.censored for row in transition_rows)
    thesis_after = [
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
    ]
    assert [sample.sample_id for sample in thesis_after] == [
        thesis_before[0].sample_id
    ]
    location = next(
        sample
        for sample in recorder.open_samples
        if sample.dimension == "location_quality"
    )
    assert location.setup_id == "entry-path-episode-1"
    assert location.calibration_unit_id == "path-1"
    assert location.calibration_unit_kind == "entry_path_location"
    assert {
        row.setup_id
        for row in transition_rows
        if row.dimension in {"sequence_progress", "uncertainty"}
    } == {"entry-path-episode-1"}

    # A genuinely new thesis evidence revision remains distinct even though
    # ownership stays on the same DFP context.
    recorder.on_bar(_bar(1))
    revised = _hypothesis(
        phase=PlaybookPhase.WAITING_LOCATION,
        complete=False,
        revision="h1-bos-revision",
        setup_id="entry-path-episode-1",
        context_id=context_id,
    )
    recorder.observe(
        _snapshot(
            2,
            hypotheses=(revised,),
            location=_location(),
        ),
        source_bar=_bar(1),
    )
    thesis_revisions = [
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
    ]
    assert len(thesis_revisions) == 2
    # setup_id remains the model's per-revision setup identity. Weighting is
    # owned by the explicitly separate DFP context calibration unit.
    assert {sample.setup_id for sample in thesis_revisions} == {
        context_id,
        "entry-path-episode-1",
    }
    assert {
        sample.calibration_unit_id for sample in thesis_revisions
    } == {context_id}


@pytest.mark.parametrize(
    "playbook",
    (
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
    ),
)
def test_identity_incomplete_runtime_root_keeps_only_descriptive_diagnostics(
    playbook: Playbook,
) -> None:
    """A native evaluator context is not a causal calibration owner."""

    recorder = _recorder()
    hypothesis = _hypothesis(
        playbook=playbook,
        episode_id=None,
    )
    candidate_id = f"runtime-root:{playbook.value}"
    hypothesis.candidate_id = candidate_id
    hypothesis.episode_id = None
    hypothesis.context_thesis_id = None
    hypothesis.parent_context_thesis_id = None
    hypothesis.initiating_event_id = "event:unbound"

    recorder.observe(
        _snapshot(
            hypotheses=(hypothesis,),
            root_candidates=((candidate_id, hypothesis),),
            location=_location(),
        ),
        source_bar=_bar(-1),
    )

    rows = recorder.drain_rows()
    assert {row.dimension for row in rows} == {
        "sequence_progress",
        "uncertainty",
    }
    assert recorder.open_samples == ()
    assert recorder.late_registration_summary == {
        "counting_basis": "unique_expired_calibration_unit",
        "late_registration_skipped": 0,
        "by_playbook": {},
        "incomplete_registration_skipped": 0,
        "incomplete_by_dimension": {},
    }


@pytest.mark.parametrize(
    "playbook",
    (
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Playbook.LIQUIDITY_SWEEP_REVERSAL,
    ),
)
def test_warmup_prime_skips_fitted_units_for_identity_incomplete_runtime_root(
    playbook: Playbook,
) -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(
        playbook=playbook,
        phase=PlaybookPhase.ARMED,
        complete=False,
        episode_id=None,
    )
    candidate_id = f"runtime-root:{playbook.value}"
    hypothesis.candidate_id = candidate_id
    hypothesis.episode_id = None
    hypothesis.context_thesis_id = None
    hypothesis.parent_context_thesis_id = None

    recorder.on_bar(_bar(-1))
    recorder.prime(
        _snapshot(
            hypotheses=(hypothesis,),
            root_candidates=((candidate_id, hypothesis),),
            location=_location(),
        ),
        source_bar=_bar(-1),
    )

    state = recorder.state_dict()
    seen_dimensions = {
        json.loads(sample_id)["dimension"]
        for sample_id in state["seen_sample_ids"]
    }
    assert seen_dimensions == {"sequence_progress", "uncertainty"}
    assert state["seen_location_ids"] == []
    assert state["late_registration_keys"] == []
    assert state["incomplete_registration_keys"] == []
    assert recorder.open_samples == ()
    assert recorder.drain_rows() == ()


def test_warmup_prime_still_records_a_valid_lsr_context_and_child_episode(
) -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
    )

    recorder.on_bar(_bar(-1))
    recorder.prime(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )

    payloads = tuple(
        json.loads(sample_id)
        for sample_id in recorder.state_dict()["seen_sample_ids"]
    )
    thesis = [
        payload
        for payload in payloads
        if payload["dimension"] == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0]["calibration_unit_kind"] == "lsr_context_thesis"
    assert thesis[0]["calibration_unit_id"] == "context-1"


def test_ownerless_warmup_trigger_cannot_seed_a_later_entry_episode() -> None:
    recorder = _recorder()
    candidate_id = "runtime-root:ownerless-trigger"
    ownerless = _hypothesis()
    ownerless.candidate_id = candidate_id
    ownerless.context_thesis_id = None
    ownerless.parent_context_thesis_id = None
    ownerless.episode_id = None

    recorder.on_bar(_bar(-1))
    recorder.prime(
        _snapshot(
            hypotheses=(ownerless,),
            root_candidates=((candidate_id, ownerless),),
            location=_location(),
        ),
        source_bar=_bar(-1),
    )
    assert recorder.state_dict()["trigger_bar_geometry"] == []

    # A parent and child materialize one completed bar later, but their
    # selected trigger still points at the ownerless warmup clock.  The new
    # Episode may publish current Context/location evidence; it cannot borrow
    # trigger geometry or manufacture readiness/delivery targets.
    capture = _hypothesis(revision="revision-after-parent-materialized")
    capture_bar = _bar(0)
    recorder.on_bar(capture_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(capture,),
            root_candidates=((candidate_id, capture),),
            location=_location(),
            price=capture_bar.close,
        ),
        source_bar=capture_bar,
    )

    assert not any(
        sample.dimension in {"entry_readiness", "delivery_quality"}
        for sample in recorder.open_samples
    )
    assert not any(
        row.dimension in {"entry_readiness", "delivery_quality"}
        for row in recorder.drain_rows()
    )
    assert recorder.state_dict()["trigger_bar_geometry"] == []


def test_lsr_context_only_projection_registers_thesis_without_entry_episode(
) -> None:
    recorder = _recorder()
    context_id = "lsr-context-before-zone"
    context = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id=context_id,
        context_id=context_id,
        episode_id=None,
    )
    context.episode_id = None
    context.parent_context_thesis_id = None

    recorder.observe(
        _snapshot(
            hypotheses=(context,),
            context_theses={context_id: _context_thesis(context_id)},
        ),
        source_bar=_bar(-1),
    )

    thesis = [
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0].episode_id is None
    assert thesis[0].parent_context_thesis_id is None
    assert thesis[0].calibration_unit_id == context_id
    assert thesis[0].calibration_unit_kind == "lsr_context_thesis"
    assert thesis[0].draw_id is None
    assert thesis[0].draw_price is None
    assert thesis[0].liquidity_route_id is None
    assert thesis[0].context_draw_id is None
    assert thesis[0].primary_deliverable_target_id is None
    assert thesis[0].terminal_draw_id is None
    assert thesis[0].intermediate_liquidity_ids == "[]"
    assert thesis[0].path_blocker_ids == "[]"
    assert thesis[0].source_path_ids == "[]"


def test_execution_deadline_does_not_substitute_for_fitted_market_targets() -> None:
    recorder = _recorder()
    comparison = _recorder()
    hypothesis = _hypothesis()
    hypothesis.thesis_deadline = None
    hypothesis.episode_deadline = None
    hypothesis.plan.deadline = None
    snapshot = _snapshot(
        hypotheses=(hypothesis,),
        location=_location(),
    )
    snapshot.observation.execution = SimpleNamespace(minutes_to_deadline=30)
    comparison_snapshot = _snapshot(
        hypotheses=(hypothesis,),
        location=_location(),
    )
    comparison_snapshot.observation.execution = SimpleNamespace(
        minutes_to_deadline=5
    )

    recorder.observe(snapshot, source_bar=_bar(-1))
    comparison.observe(comparison_snapshot, source_bar=_bar(-1))

    thesis_samples = tuple(
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
    )
    assert len(thesis_samples) == 1
    assert thesis_samples[0].deadline == pd.Timestamp(
        "2025-01-06 17:00",
        tz=TZ,
    )
    assert thesis_samples[0].deadline != (
        snapshot.observation.asof + pd.Timedelta(minutes=30)
    )
    comparison_thesis = tuple(
        sample
        for sample in comparison.open_samples
        if sample.dimension == "thesis_strength"
    )
    assert len(comparison_thesis) == 1
    assert comparison_thesis[0].deadline == thesis_samples[0].deadline
    assert {row.dimension for row in recorder.drain_rows()} == {
        "sequence_progress",
        "uncertainty",
    }


@pytest.mark.parametrize("delivery", [0.0, 0.05])
def test_terminal_valid_trigger_records_low_delivery_and_resolves_causally(
    delivery: float,
) -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(
        phase=PlaybookPhase.INVALIDATED,
        terminal_reason="invalidated",
        delivery=delivery,
    )
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )

    assert recorder.drain_rows() == ()
    assert {sample.dimension for sample in recorder.open_samples} == {
        "entry_readiness",
        "delivery_quality",
    }
    recorded_delivery = next(
        sample
        for sample in recorder.open_samples
        if sample.dimension == "delivery_quality"
    )
    assert recorded_delivery.raw_value == delivery

    advance = _bar(0, high=100.8, close=100.75)
    recorder.on_bar(advance)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(hypothesis,),
            location=_location(),
            price=advance.close,
        ),
        source_bar=advance,
    )
    readiness_rows = recorder.drain_rows()
    assert [row.dimension for row in readiness_rows] == [
        "entry_readiness"
    ]
    assert readiness_rows[0].sampled_at == _clock(0)
    assert readiness_rows[0].resolved_at == _clock(1)
    assert readiness_rows[0].outcome_value == 1.0
    assert {sample.dimension for sample in recorder.open_samples} == {
        "delivery_quality"
    }

    delivery_bar = _bar(1, high=103.5, close=103.0)
    recorder.on_bar(delivery_bar)
    recorder.observe(
        _snapshot(
            2,
            hypotheses=(hypothesis,),
            location=_location(),
            price=delivery_bar.close,
        ),
        source_bar=delivery_bar,
    )
    delivery_rows = recorder.drain_rows()
    assert [row.dimension for row in delivery_rows] == [
        "delivery_quality"
    ]
    assert delivery_rows[0].outcome_value == 1.0
    assert delivery_rows[0].resolution == "draw_delivered"


def test_retained_terminal_does_not_rewrite_frozen_trigger_path() -> None:
    recorder = _recorder()
    terminal = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.INVALIDATED,
        terminal_reason="delivery_not_ready",
        terminal_source_ids=("setup-1",),
    )
    recorder.observe(
        _snapshot(hypotheses=(terminal,), location=_location()),
        source_bar=_bar(-1),
    )

    advance = _bar(0, high=100.8, close=100.75)
    recorder.on_bar(advance)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(terminal,),
            location=_location(),
            price=advance.close,
        ),
        source_bar=advance,
    )
    readiness = recorder.drain_rows()
    assert [row.dimension for row in readiness] == ["entry_readiness"]

    # Typed Brain retains a terminal episode until a causally newer setup
    # replaces it. Seeing that closure did not resolve its frozen delivery.
    assert recorder.drain_rows() == ()
    assert [sample.dimension for sample in recorder.open_samples] == [
        "delivery_quality"
    ]

    delivery_bar = _bar(1, high=103.5, close=103.0)
    recorder.on_bar(delivery_bar)
    recorder.observe(
        _snapshot(
            2,
            hypotheses=(terminal,),
            location=_location(),
            price=delivery_bar.close,
        ),
        source_bar=delivery_bar,
    )
    delivered = recorder.drain_rows()
    assert [row.dimension for row in delivered] == ["delivery_quality"]
    assert delivered[0].outcome_value == 1.0
    assert delivered[0].resolution == "draw_delivered"


def test_lsr_local_micro_bos_terminal_keeps_shared_context_thesis_open() -> None:
    recorder = _recorder()
    old_setup = "lsr-old-episode"
    active = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id=old_setup,
        context_id=old_setup,
    )
    recorder.observe(
        _snapshot(hypotheses=(active,)),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    recorder.on_bar(_bar(0))
    terminal = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.INVALIDATED,
        complete=False,
        setup_id="new-current-episode",
        context_id=old_setup,
        episode_id="new-current-episode",
        terminal_reason="micro_bos_opposed",
        terminal_source_ids=("new-current-episode", "micro-bos-event"),
    )
    terminal.probability = 0.1
    top = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id="dfp-top-episode",
        context_id="dfp-top-context",
    )
    top.probability = 0.95
    recorder.observe(
        _snapshot(1, hypotheses=(top, terminal)),
        source_bar=_bar(0),
    )

    rows = recorder.drain_rows()
    thesis = [
        row
        for row in rows
        if row.dimension == "thesis_strength"
        and row.setup_id == old_setup
    ]
    assert thesis == []
    assert any(
        sample.dimension == "thesis_strength"
        and sample.context_thesis_id == old_setup
        for sample in recorder.open_samples
    )


def test_lsr_multiple_zone_children_share_one_thesis_and_local_terminal(
) -> None:
    recorder = _recorder()
    context_id = "lsr-context-shared"
    child_a = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="shared-context-revision",
        setup_id="lsr-zone-a",
        context_id=context_id,
        episode_id="lsr-zone-a",
    )
    child_b = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="shared-context-revision",
        setup_id="lsr-zone-b",
        context_id=context_id,
        episode_id="lsr-zone-b",
    )
    recorder.observe(
        _snapshot(
            hypotheses=(child_a, child_b),
            root_candidates=(("candidate-a", child_a), ("candidate-b", child_b)),
            context_theses={context_id: _context_thesis(context_id)},
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    thesis = [
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0].calibration_unit_id == context_id
    assert thesis[0].calibration_unit_kind == "lsr_context_thesis"

    local_terminal = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.INVALIDATED,
        complete=False,
        revision="shared-context-revision",
        setup_id="lsr-zone-a",
        context_id=context_id,
        episode_id="lsr-zone-a",
        terminal_at=_clock(1),
        terminal_reason="entry_zone_left",
        terminal_source_ids=("lsr-zone-a", "location-a", "path-a"),
    )
    # The child trade target is touched, but it is not the shared reversal
    # Context target and therefore cannot settle the Context thesis.
    target_touch_bar = _bar(0, high=104.0, close=103.5)
    recorder.on_bar(target_touch_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(local_terminal, child_b),
            root_candidates=(
                ("candidate-a", local_terminal),
                ("candidate-b", child_b),
            ),
            lifecycle_candidates=(
                ("candidate-a", local_terminal),
                ("candidate-b", child_b),
            ),
            price=target_touch_bar.close,
            context_theses={context_id: _context_thesis(context_id)},
        ),
        source_bar=target_touch_bar,
    )

    assert not any(
        row.dimension == "thesis_strength" for row in recorder.drain_rows()
    )
    assert len(
        [
            sample
            for sample in recorder.open_samples
            if sample.dimension == "thesis_strength"
        ]
    ) == 1
    assert recorder.state_dict()["terminal_thesis_units"] == []


def test_lsr_exact_context_terminal_settles_all_revisions_and_blocks_children(
) -> None:
    recorder = _recorder()
    context_id = "lsr-context-terminal"
    revisions = tuple(
        _hypothesis(
            playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
            phase=PlaybookPhase.ARMED,
            complete=False,
            revision=f"context-revision-{suffix}",
            setup_id=f"zone-{suffix}",
            context_id=context_id,
            episode_id=f"zone-{suffix}",
        )
        for suffix in ("a", "b")
    )
    recorder.observe(
        _snapshot(
            hypotheses=revisions,
            root_candidates=tuple(
                (f"candidate-{index}", child)
                for index, child in enumerate(revisions)
            ),
            context_theses={context_id: _context_thesis(context_id)},
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    resumed = BrainCalibrationRecorder.from_state(recorder.state_dict())

    terminal_bar = _bar(0)
    terminal_context = _context_thesis(
        context_id,
        lifecycle="invalidated",
        terminal_at=_clock(1),
        terminal_reason="accepted_outside_or_failed",
    )
    terminal_snapshot = _snapshot(
        1,
        hypotheses=revisions,
        root_candidates=(),
        lifecycle_candidates=(),
        price=terminal_bar.close,
        context_theses={context_id: terminal_context},
    )
    for value in (recorder, resumed):
        value.on_bar(terminal_bar)
        value.observe(terminal_snapshot, source_bar=terminal_bar)

    assert [row.to_dict() for row in resumed.drain_rows()] == [
        row.to_dict() for row in recorder.drain_rows()
    ]
    terminal_units = recorder.state_dict()["terminal_thesis_units"]
    assert len(terminal_units) == 1
    assert terminal_units[0]["calibration_unit_id"] == context_id
    assert terminal_units[0]["context_thesis_id"] == context_id
    assert terminal_units[0]["calibration_unit_kind"] == "lsr_context_thesis"
    assert terminal_units[0]["scope"] == "context_terminal"

    later_child = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="context-revision-c",
        setup_id="zone-c",
        context_id=context_id,
        episode_id="zone-c",
    )
    recorder.on_bar(_bar(1))
    recorder.observe(
        _snapshot(
            2,
            hypotheses=(later_child,),
            root_candidates=(("candidate-c", later_child),),
            context_theses={context_id: terminal_context},
        ),
        source_bar=_bar(1),
    )
    assert recorder.drain_rows() == ()
    assert recorder.open_samples == ()


def test_lsr_context_deadline_settles_thesis_horizon_and_tombstones_context(
) -> None:
    recorder = _recorder()
    context_id = "lsr-context-deadline"
    context = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id="zone-a",
        context_id=context_id,
        episode_id="zone-a",
    )
    context.thesis_deadline = _clock(1)
    recorder.observe(
        _snapshot(
            hypotheses=(context,),
            context_theses={context_id: _context_thesis(context_id)},
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    deadline_bar = _bar(0)
    recorder.on_bar(deadline_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(context,),
            root_candidates=(),
            lifecycle_candidates=(),
            price=deadline_bar.close,
            context_theses={
                context_id: _context_thesis(
                    context_id,
                    lifecycle="invalidated",
                    terminal_at=_clock(1),
                    terminal_reason="context_thesis_deadline_elapsed",
                )
            },
        ),
        source_bar=deadline_bar,
    )

    rows = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "thesis_strength"
    ]
    assert len(rows) == 1
    assert rows[0].resolved_at == _clock(1)
    assert rows[0].outcome_value == 1.0
    assert rows[0].resolution == "thesis_intact_at_deadline"
    tombstone = recorder.state_dict()["terminal_thesis_units"]
    assert len(tombstone) == 1
    assert tombstone[0]["scope"] == "context_terminal"


def test_lsr_observation_horizon_allows_later_child_then_promotes_context(
) -> None:
    recorder = _recorder()
    context_id = "lsr-context-observation-horizon"
    context = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="context-revision",
        setup_id="zone-a",
        context_id=context_id,
        episode_id="zone-a",
    )
    context.thesis_deadline = _clock(1)
    recorder.observe(
        _snapshot(
            hypotheses=(context,),
            context_theses={context_id: _context_thesis(context_id)},
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    horizon_bar = _bar(0)
    recorder.on_bar(horizon_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(context,),
            price=horizon_bar.close,
            context_theses={context_id: _context_thesis(context_id)},
        ),
        source_bar=horizon_bar,
    )
    horizon = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "thesis_strength"
    ]
    assert len(horizon) == 1
    assert horizon[0].resolution == "thesis_intact_at_deadline"
    assert recorder.state_dict()["terminal_thesis_units"][0]["scope"] == (
        "thesis_observation_horizon"
    )

    child_b = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.EXECUTABLE,
        complete=True,
        revision="child-b-revision",
        setup_id="zone-b",
        context_id=context_id,
        episode_id="zone-b",
    )
    child_b.entry_location_id = "zone-b-location"
    child_b.entry_path_id = "zone-b-path"
    child_b.plan.entry_path_id = "zone-b-path"
    child_b.selected_trigger.entry_path_id = "zone-b-path"
    child_b.selected_trigger.observed_at = _clock(2)
    child_bar = _bar(1)
    recorder.on_bar(child_bar)
    recorder.observe(
        _snapshot(
            2,
            hypotheses=(child_b,),
            location=SimpleNamespace(
                location_id="zone-b-location",
                lifecycle=EntryLocationLifecycle.IN_ZONE,
            ),
            price=child_bar.close,
            context_theses={context_id: _context_thesis(context_id)},
        ),
        source_bar=child_bar,
    )
    recorder.drain_rows()
    assert {
        sample.dimension
        for sample in recorder.open_samples
        if sample.episode_id == "zone-b"
    } == {"location_quality", "entry_readiness", "delivery_quality"}

    terminal_bar = _bar(2)
    terminal_context = _context_thesis(
        context_id,
        lifecycle="invalidated",
        terminal_at=_clock(3),
        terminal_reason="accepted_outside_or_failed",
    )
    recorder.on_bar(terminal_bar)
    recorder.observe(
        _snapshot(
            3,
            hypotheses=(child_b,),
            price=terminal_bar.close,
            context_theses={context_id: terminal_context},
        ),
        source_bar=terminal_bar,
    )
    recorder.drain_rows()
    tombstone = recorder.state_dict()["terminal_thesis_units"][0]
    assert tombstone["scope"] == "context_terminal"
    assert tombstone["resolved_at"] == _clock(3).isoformat()

    child_c = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="child-c-revision",
        setup_id="zone-c",
        context_id=context_id,
        episode_id="zone-c",
    )
    next_bar = _bar(3)
    recorder.on_bar(next_bar)
    recorder.observe(
        _snapshot(
            4,
            hypotheses=(child_c,),
            price=next_bar.close,
            context_theses={context_id: terminal_context},
        ),
        source_bar=next_bar,
    )
    assert not any(
        row.episode_id == "zone-c" for row in recorder.drain_rows()
    )
    assert not any(
        sample.episode_id == "zone-c" for sample in recorder.open_samples
    )


def test_dfp_local_terminal_keeps_context_thesis_open_until_deadline() -> None:
    recorder = _recorder()
    context_id = "dfp-context:structure-1:draw-1"
    active = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id=context_id,
        context_id=context_id,
    )
    recorder.observe(
        _snapshot(hypotheses=(active,)),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    recorder.on_bar(_bar(0))
    terminal = _hypothesis(
        phase=PlaybookPhase.INVALIDATED,
        complete=False,
        setup_id="dfp-entry-episode",
        context_id=context_id,
        terminal_reason="entry_zone_left_or_failed",
        terminal_source_ids=(context_id, "dfp-entry-episode"),
    )
    recorder.observe(
        _snapshot(1, hypotheses=(terminal,)),
        source_bar=_bar(0),
    )

    thesis = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "thesis_strength"
        and row.setup_id == context_id
    ]
    assert thesis == []
    open_thesis = [
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
        and sample.setup_id == context_id
    ]
    assert len(open_thesis) == 1

    recorder.on_bar(_bar(9))
    assert recorder.drain_rows() == ()
    recorder.observe(
        _snapshot(10, hypotheses=(terminal,)),
        source_bar=_bar(9),
    )
    thesis = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "thesis_strength"
        and row.setup_id == context_id
    ]
    assert len(thesis) == 1
    assert thesis[0].resolved_at == _clock(10)
    assert thesis[0].outcome_value == 1.0
    assert not thesis[0].censored
    assert thesis[0].resolution == "thesis_intact_at_deadline"


def test_dfp_thesis_survives_root_disappearance_and_reappearance() -> None:
    recorder = _recorder()
    candidate_id = "root-1|dfp|long"
    hypothesis = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id="dfp-context-1",
        context_id="dfp-context-1",
    )
    structure = _bind_dfp_h4_context(hypothesis)
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                hypotheses=(hypothesis,),
                root_candidates=((candidate_id, hypothesis),),
            ),
            h4_structures=(structure,),
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    frozen = next(
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
    )

    first_bar = _bar(0)
    recorder.on_bar(first_bar)
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                1,
                hypotheses=(hypothesis,),
                root_candidates=(),
                price=first_bar.close,
            ),
            h4_structures=(structure,),
        ),
        source_bar=first_bar,
    )
    disappeared_rows = recorder.drain_rows()
    assert not any(
        row.sample_id == frozen.sample_id for row in disappeared_rows
    )
    assert frozen.sample_id in {
        sample.sample_id for sample in recorder.open_samples
    }

    second_bar = _bar(1)
    recorder.on_bar(second_bar)
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                2,
                hypotheses=(hypothesis,),
                root_candidates=((candidate_id, hypothesis),),
                price=second_bar.close,
            ),
            h4_structures=(structure,),
        ),
        source_bar=second_bar,
    )
    assert not any(
        row.sample_id == frozen.sample_id for row in recorder.drain_rows()
    )
    matching = [
        sample
        for sample in recorder.open_samples
        if sample.sample_id == frozen.sample_id
    ]
    assert len(matching) == 1


def test_dfp_h1_evidence_retention_does_not_end_frozen_h4_thesis() -> None:
    recorder = _recorder()
    candidate_id = "root-1|dfp|long"
    hypothesis = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id="dfp-context-1",
        context_id="dfp-context-1",
    )
    structure = _bind_dfp_h4_context(hypothesis)
    h1_bos = SimpleNamespace(
        bos_id="h1-bos-1",
        direction=Direction.LONG,
        lifecycle=BOSLifecycle.CONFIRMED,
        scope=BOSScope.CONTINUATION,
    )
    hypothesis.sequence.steps = (
        *hypothesis.sequence.steps,
        SimpleNamespace(
            step_id="h1_continuation_bos",
            satisfied=True,
            observed_at=_clock(-2),
            source_ids=(h1_bos.bos_id, "h1-target-swing"),
        ),
    )
    hypothesis.sequence.completed_steps = 2
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                hypotheses=(hypothesis,),
                root_candidates=((candidate_id, hypothesis),),
            ),
            h4_structures=(structure,),
            h1_breaks=(h1_bos,),
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    next_bar = _bar(0)
    recorder.on_bar(next_bar)
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                1,
                hypotheses=(hypothesis,),
                root_candidates=(),
                price=next_bar.close,
            ),
            h4_structures=(structure,),
            # The bounded current BOS view may drop an older supporting
            # revision.  Absence is not a confirmed H4 invalidation.
            h1_breaks=(),
        ),
        source_bar=next_bar,
    )

    assert not any(
        row.dimension == "thesis_strength"
        for row in recorder.drain_rows()
    )
    thesis = [
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0].dfp_h1_bos_id == h1_bos.bos_id


@pytest.mark.parametrize(
    ("lifecycle", "outcome", "resolution"),
    [
        (
            EntryLocationLifecycle.REJECTED,
            1.0,
            "frozen_zone_rejected",
        ),
        (EntryLocationLifecycle.LEFT, 0.0, "frozen_zone_left"),
    ],
)
def test_location_lifecycle_resolves_before_root_disappearance(
    lifecycle: EntryLocationLifecycle,
    outcome: float,
    resolution: str,
) -> None:
    recorder = _recorder()
    candidate_id = "root-1|dfp|long"
    hypothesis = _hypothesis(
        phase=PlaybookPhase.WAITING_LOCATION,
        complete=False,
    )
    recorder.observe(
        _snapshot(
            hypotheses=(hypothesis,),
            root_candidates=((candidate_id, hypothesis),),
            location=_location(),
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    frozen = next(
        sample
        for sample in recorder.open_samples
        if sample.dimension == "location_quality"
    )

    transition_bar = _bar(0)
    recorder.on_bar(transition_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(hypothesis,),
            root_candidates=(),
            location=_location(lifecycle),
            price=transition_bar.close,
        ),
        source_bar=transition_bar,
    )

    rows = [
        row
        for row in recorder.drain_rows()
        if row.sample_id == frozen.sample_id
    ]
    assert len(rows) == 1
    assert rows[0].outcome_value == outcome
    assert not rows[0].censored
    assert rows[0].resolution == resolution


def test_dfp_disappeared_root_checkpoint_resume_preserves_draw_resolution(
) -> None:
    uninterrupted = _recorder()
    candidate_id = "root-1|dfp|long"
    hypothesis = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id="dfp-context-1",
        context_id="dfp-context-1",
    )
    structure = _bind_dfp_h4_context(hypothesis)
    initial = _with_structure_frames(
        _snapshot(
            hypotheses=(hypothesis,),
            root_candidates=((candidate_id, hypothesis),),
        ),
        h4_structures=(structure,),
    )
    uninterrupted.observe(initial, source_bar=_bar(-1))
    uninterrupted.drain_rows()

    first_bar = _bar(0)
    uninterrupted.on_bar(first_bar)
    disappeared = _with_structure_frames(
        _snapshot(
            1,
            hypotheses=(hypothesis,),
            root_candidates=(),
            price=first_bar.close,
        ),
        h4_structures=(structure,),
    )
    uninterrupted.observe(disappeared, source_bar=first_bar)
    uninterrupted.drain_rows()
    resumed = BrainCalibrationRecorder.from_state(
        uninterrupted.state_dict()
    )

    delivery_bar = _bar(1, high=103.5, close=103.0)
    for recorder in (uninterrupted, resumed):
        recorder.on_bar(delivery_bar)
        recorder.observe(
            _with_structure_frames(
                _snapshot(
                    2,
                    hypotheses=(hypothesis,),
                    root_candidates=(),
                    price=delivery_bar.close,
                ),
                h4_structures=(structure,),
            ),
            source_bar=delivery_bar,
        )

    uninterrupted_rows = sorted(
        (row.to_dict() for row in uninterrupted.drain_rows()),
        key=lambda row: row["sample_id"],
    )
    resumed_rows = sorted(
        (row.to_dict() for row in resumed.drain_rows()),
        key=lambda row: row["sample_id"],
    )
    assert resumed_rows == uninterrupted_rows
    thesis = [
        row
        for row in resumed_rows
        if row["dimension"] == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0]["outcome_value"] == 1.0
    assert not thesis[0]["censored"]
    assert thesis[0]["resolution"] == "draw_delivered"
    assert resumed.state_dict() == uninterrupted.state_dict()


def test_dfp_disappeared_root_resolves_on_frozen_h4_invalidation() -> None:
    recorder = _recorder()
    candidate_id = "root-1|dfp|long"
    hypothesis = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id="dfp-context-1",
        context_id="dfp-context-1",
    )
    structure = _bind_dfp_h4_context(hypothesis)
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                hypotheses=(hypothesis,),
                root_candidates=((candidate_id, hypothesis),),
            ),
            h4_structures=(structure,),
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    first_bar = _bar(0)
    recorder.on_bar(first_bar)
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                1,
                hypotheses=(hypothesis,),
                root_candidates=(),
                price=first_bar.close,
            ),
            h4_structures=(structure,),
        ),
        source_bar=first_bar,
    )
    assert not any(
        row.dimension == "thesis_strength"
        for row in recorder.drain_rows()
    )

    broken = SimpleNamespace(
        structure_id=structure.structure_id,
        direction=structure.direction,
        lifecycle=StructureLifecycle.BROKEN,
        confirmed_at=structure.confirmed_at,
    )
    terminal_bar = _bar(1)
    recorder.on_bar(terminal_bar)
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                2,
                hypotheses=(hypothesis,),
                root_candidates=(),
                price=terminal_bar.close,
            ),
            h4_structures=(broken,),
        ),
        source_bar=terminal_bar,
    )
    thesis = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0].outcome_value == 0.0
    assert not thesis[0].censored
    assert thesis[0].resolution == "thesis_contradicted:opposed_structure"


@pytest.mark.parametrize("opposed_confirmed_minute", (-4, 0))
def test_dfp_existing_or_same_clock_h4_opposition_is_not_a_future_terminal(
    opposed_confirmed_minute: int,
) -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id="dfp-context-1",
        context_id="dfp-context-1",
    )
    structure = _bind_dfp_h4_context(hypothesis)
    opposed = SimpleNamespace(
        structure_id="h4-structure-opposed",
        direction=Direction.SHORT,
        lifecycle=StructureLifecycle.CONFIRMED,
        confirmed_at=_clock(opposed_confirmed_minute),
    )
    frames = (structure, opposed)
    recorder.observe(
        _with_structure_frames(
            _snapshot(hypotheses=(hypothesis,)),
            h4_structures=frames,
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    recorder.on_bar(_bar(0))
    recorder.observe(
        _with_structure_frames(
            _snapshot(1, hypotheses=(hypothesis,)),
            h4_structures=frames,
        ),
        source_bar=_bar(0),
    )

    assert not any(
        row.dimension == "thesis_strength"
        for row in recorder.drain_rows()
    )
    assert any(
        sample.dimension == "thesis_strength"
        for sample in recorder.open_samples
    )


def test_dfp_later_h4_opposition_terminal_survives_checkpoint_roundtrip(
) -> None:
    uninterrupted = _recorder()
    hypothesis = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id="dfp-context-1",
        context_id="dfp-context-1",
    )
    structure = _bind_dfp_h4_context(hypothesis)
    uninterrupted.observe(
        _with_structure_frames(
            _snapshot(hypotheses=(hypothesis,)),
            h4_structures=(structure,),
        ),
        source_bar=_bar(-1),
    )
    uninterrupted.drain_rows()
    resumed = BrainCalibrationRecorder.from_state(
        uninterrupted.state_dict()
    )

    opposed = SimpleNamespace(
        structure_id="h4-structure-opposed",
        direction=Direction.SHORT,
        lifecycle=StructureLifecycle.CONFIRMED,
        confirmed_at=_clock(1),
    )
    snapshot = _with_structure_frames(
        _snapshot(2, hypotheses=(hypothesis,)),
        h4_structures=(structure, opposed),
    )
    for recorder in (uninterrupted, resumed):
        recorder.on_bar(_bar(1))
        recorder.observe(snapshot, source_bar=_bar(1))

    uninterrupted_rows = tuple(
        row.to_dict() for row in uninterrupted.drain_rows()
    )
    resumed_rows = tuple(row.to_dict() for row in resumed.drain_rows())
    assert resumed_rows == uninterrupted_rows
    thesis = [
        row
        for row in resumed_rows
        if row["dimension"] == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0]["outcome_value"] == 0.0
    assert thesis[0]["resolution"] == (
        "thesis_contradicted:opposed_structure"
    )
    assert resumed.state_dict() == uninterrupted.state_dict()


def test_dfp_missing_frozen_structure_revision_is_not_a_break() -> None:
    recorder = _recorder()
    candidate_id = "root-1|dfp|long"
    hypothesis = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id="dfp-context-1",
        context_id="dfp-context-1",
    )
    structure = _bind_dfp_h4_context(hypothesis)
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                hypotheses=(hypothesis,),
                root_candidates=((candidate_id, hypothesis),),
            ),
            h4_structures=(structure,),
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    frozen = next(
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
    )

    quiet_bar = _bar(0)
    recorder.on_bar(quiet_bar)
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                1,
                hypotheses=(hypothesis,),
                root_candidates=(),
                price=quiet_bar.close,
            ),
            # A bounded current frame may omit the older structure.  Only a
            # typed BROKEN revision or later opposed structure is terminal.
            h4_structures=(),
        ),
        source_bar=quiet_bar,
    )

    assert not any(
        row.sample_id == frozen.sample_id for row in recorder.drain_rows()
    )
    assert frozen.sample_id in {
        sample.sample_id for sample in recorder.open_samples
    }


def test_dfp_disappeared_root_resolves_at_frozen_thesis_deadline() -> None:
    recorder = _recorder()
    candidate_id = "root-1|dfp|long"
    hypothesis = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id="dfp-context-1",
        context_id="dfp-context-1",
    )
    structure = _bind_dfp_h4_context(hypothesis)
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                hypotheses=(hypothesis,),
                root_candidates=((candidate_id, hypothesis),),
            ),
            h4_structures=(structure,),
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    deadline_bar = _bar(9)
    recorder.on_bar(deadline_bar)
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                10,
                hypotheses=(hypothesis,),
                root_candidates=(),
                price=deadline_bar.close,
            ),
            h4_structures=(structure,),
        ),
        source_bar=deadline_bar,
    )
    thesis = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0].resolved_at == hypothesis.thesis_deadline
    assert thesis[0].outcome_value == 1.0
    assert not thesis[0].censored
    assert thesis[0].resolution == "thesis_intact_at_deadline"


def test_dfp_disappeared_root_is_censored_only_at_data_boundary() -> None:
    recorder = _recorder()
    candidate_id = "root-1|dfp|long"
    hypothesis = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id="dfp-context-1",
        context_id="dfp-context-1",
    )
    structure = _bind_dfp_h4_context(hypothesis)
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                hypotheses=(hypothesis,),
                root_candidates=((candidate_id, hypothesis),),
            ),
            h4_structures=(structure,),
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    quiet_bar = _bar(0)
    recorder.on_bar(quiet_bar)
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                1,
                hypotheses=(hypothesis,),
                root_candidates=(),
                price=quiet_bar.close,
            ),
            h4_structures=(structure,),
        ),
        source_bar=quiet_bar,
    )
    assert not any(
        row.dimension == "thesis_strength"
        for row in recorder.drain_rows()
    )

    recorder.on_bar(_bar(1, data_gap=2))
    thesis = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0].outcome_value is None
    assert thesis[0].censored
    assert thesis[0].resolution == "data_gap_boundary"


def test_lsr_thesis_disappearance_without_terminal_remains_open() -> None:
    recorder = _recorder()
    candidate_id = "root-1|lsr|long"
    hypothesis = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
    )
    recorder.observe(
        _snapshot(
            hypotheses=(hypothesis,),
            root_candidates=((candidate_id, hypothesis),),
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    frozen = next(
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
    )

    next_bar = _bar(0)
    recorder.on_bar(next_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(hypothesis,),
            root_candidates=(),
            price=next_bar.close,
        ),
        source_bar=next_bar,
    )
    assert not any(
        row.sample_id == frozen.sample_id for row in recorder.drain_rows()
    )
    assert frozen.sample_id in {
        sample.sample_id for sample in recorder.open_samples
    }


def test_lsr_disappeared_thesis_ignores_child_draw_and_uses_frozen_sweep(
) -> None:
    recorder = _recorder()
    candidate_id = "root-1|lsr|long"
    hypothesis = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
    )
    recorder.observe(
        _snapshot(
            hypotheses=(hypothesis,),
            root_candidates=((candidate_id, hypothesis),),
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    frozen = next(
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
    )

    quiet_bar = _bar(0)
    recorder.on_bar(quiet_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(hypothesis,),
            root_candidates=(),
            price=quiet_bar.close,
        ),
        source_bar=quiet_bar,
    )
    assert recorder.drain_rows() == ()

    delivery_bar = _bar(1, high=103.5, close=103.0)
    recorder.on_bar(delivery_bar)
    recorder.observe(
        _snapshot(
            2,
            hypotheses=(hypothesis,),
            root_candidates=(),
            price=delivery_bar.close,
        ),
        source_bar=delivery_bar,
    )
    assert not any(
        row.sample_id == frozen.sample_id for row in recorder.drain_rows()
    )
    assert frozen.sample_id in {
        sample.sample_id for sample in recorder.open_samples
    }

    invalidation_bar = _bar(2, low=98.5, close=99.0)
    recorder.on_bar(invalidation_bar)
    recorder.observe(
        _snapshot(
            3,
            hypotheses=(hypothesis,),
            root_candidates=(),
            price=invalidation_bar.close,
        ),
        source_bar=invalidation_bar,
    )
    rows = [
        row
        for row in recorder.drain_rows()
        if row.sample_id == frozen.sample_id
    ]
    assert len(rows) == 1
    assert rows[0].outcome_value == 0.0
    assert not rows[0].censored
    assert rows[0].resolution == "invalidation_touched"


def test_lsr_disappeared_thesis_checkpoint_resume_preserves_invalidation(
) -> None:
    uninterrupted = _recorder()
    candidate_id = "root-1|lsr|long"
    hypothesis = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
    )
    uninterrupted.observe(
        _snapshot(
            hypotheses=(hypothesis,),
            root_candidates=((candidate_id, hypothesis),),
        ),
        source_bar=_bar(-1),
    )
    uninterrupted.drain_rows()

    quiet_bar = _bar(0)
    uninterrupted.on_bar(quiet_bar)
    uninterrupted.observe(
        _snapshot(
            1,
            hypotheses=(hypothesis,),
            root_candidates=(),
            price=quiet_bar.close,
        ),
        source_bar=quiet_bar,
    )
    uninterrupted.drain_rows()
    resumed = BrainCalibrationRecorder.from_state(
        uninterrupted.state_dict()
    )

    invalidation_bar = _bar(1, high=100.5, low=98.5, close=99.0)
    for recorder in (uninterrupted, resumed):
        recorder.on_bar(invalidation_bar)
        recorder.observe(
            _snapshot(
                2,
                hypotheses=(hypothesis,),
                root_candidates=(),
                price=invalidation_bar.close,
            ),
            source_bar=invalidation_bar,
        )

    uninterrupted_rows = sorted(
        (row.to_dict() for row in uninterrupted.drain_rows()),
        key=lambda row: row["sample_id"],
    )
    resumed_rows = sorted(
        (row.to_dict() for row in resumed.drain_rows()),
        key=lambda row: row["sample_id"],
    )
    assert resumed_rows == uninterrupted_rows
    thesis = [
        row
        for row in resumed_rows
        if row["dimension"] == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0]["outcome_value"] == 0.0
    assert not thesis[0]["censored"]
    assert thesis[0]["resolution"] == "invalidation_touched"
    assert resumed.state_dict() == uninterrupted.state_dict()


def test_lsr_disappeared_location_resolves_at_frozen_episode_deadline() -> None:
    recorder = _recorder()
    candidate_id = "root-1|lsr|long"
    hypothesis = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.WAITING_LOCATION,
        complete=False,
    )
    recorder.observe(
        _snapshot(
            hypotheses=(hypothesis,),
            root_candidates=((candidate_id, hypothesis),),
            location=_location(),
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    frozen = next(
        sample
        for sample in recorder.open_samples
        if sample.dimension == "location_quality"
    )

    quiet_bar = _bar(0)
    recorder.on_bar(quiet_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(hypothesis,),
            root_candidates=(),
            price=quiet_bar.close,
        ),
        source_bar=quiet_bar,
    )
    assert not any(
        row.sample_id == frozen.sample_id for row in recorder.drain_rows()
    )

    deadline_bar = _bar(9)
    recorder.on_bar(deadline_bar)
    recorder.observe(
        _snapshot(
            10,
            hypotheses=(hypothesis,),
            root_candidates=(),
            price=deadline_bar.close,
        ),
        source_bar=deadline_bar,
    )
    rows = [
        row
        for row in recorder.drain_rows()
        if row.sample_id == frozen.sample_id
    ]
    assert len(rows) == 1
    assert rows[0].resolved_at == hypothesis.episode_deadline
    assert rows[0].outcome_value == 0.0
    assert not rows[0].censored
    assert rows[0].resolution == "deadline_without_dimension_delivery"


def test_lsr_disappeared_thesis_accepts_matching_explicit_terminal() -> None:
    recorder = _recorder()
    candidate_id = "root-1|lsr|long"
    active = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
    )
    recorder.observe(
        _snapshot(
            hypotheses=(active,),
            root_candidates=((candidate_id, active),),
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    frozen = next(
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
    )

    quiet_bar = _bar(0)
    recorder.on_bar(quiet_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(active,),
            root_candidates=(),
            price=quiet_bar.close,
        ),
        source_bar=quiet_bar,
    )
    assert recorder.drain_rows() == ()

    terminal_bar = _bar(1)
    recorder.on_bar(terminal_bar)
    terminal = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.INVALIDATED,
        complete=False,
        terminal_reason="accepted_outside_or_failed",
        terminal_source_ids=("context-1",),
    )
    recorder.observe(
        _snapshot(
            2,
            hypotheses=(terminal,),
            root_candidates=((candidate_id, terminal),),
            price=terminal_bar.close,
            context_theses={
                "context-1": _context_thesis(
                    "context-1",
                    lifecycle="invalidated",
                    terminal_at=_clock(2),
                    terminal_reason="accepted_outside_or_failed",
                )
            },
        ),
        source_bar=terminal_bar,
    )
    rows = [
        row
        for row in recorder.drain_rows()
        if row.sample_id == frozen.sample_id
    ]
    assert len(rows) == 1
    assert rows[0].outcome_value == 0.0
    assert not rows[0].censored
    assert rows[0].resolution == (
        "thesis_contradicted:accepted_outside_or_failed"
    )


def test_lsr_dormant_lifecycle_candidate_resolves_without_action_authority(
) -> None:
    recorder = _recorder()
    candidate_id = "root-1|lsr|long"
    active = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
    )
    recorder.observe(
        _snapshot(
            hypotheses=(active,),
            root_candidates=((candidate_id, active),),
            lifecycle_candidates=((candidate_id, active),),
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    frozen = next(
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
    )

    terminal_bar = _bar(0)
    recorder.on_bar(terminal_bar)
    dormant_terminal = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.INVALIDATED,
        complete=False,
        terminal_reason="accepted_outside_or_failed",
        terminal_source_ids=("context-1",),
    )
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(dormant_terminal,),
            root_candidates=(),
            lifecycle_candidates=((candidate_id, dormant_terminal),),
            price=terminal_bar.close,
            context_theses={
                "context-1": _context_thesis(
                    "context-1",
                    lifecycle="invalidated",
                    terminal_at=_clock(1),
                    terminal_reason="accepted_outside_or_failed",
                )
            },
        ),
        source_bar=terminal_bar,
    )

    rows = [
        row
        for row in recorder.drain_rows()
        if row.sample_id == frozen.sample_id
    ]
    assert len(rows) == 1
    assert rows[0].outcome_value == 0.0
    assert not rows[0].censored
    assert rows[0].resolution == (
        "thesis_contradicted:accepted_outside_or_failed"
    )


def test_dormant_lifecycle_candidate_cannot_register_samples_or_trigger(
) -> None:
    recorder = _recorder()
    candidate_id = "root-1|lsr|long"
    dormant = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
    )

    recorder.observe(
        _snapshot(
            hypotheses=(dormant,),
            root_candidates=(),
            lifecycle_candidates=((candidate_id, dormant),),
            location=_location(),
        ),
        source_bar=_bar(-1),
    )

    assert recorder.drain_rows() == ()
    assert recorder.open_samples == ()
    state = recorder.state_dict()
    assert state["seen_sample_ids"] == []
    assert state["trigger_bar_geometry"] == []


def test_missing_then_dormant_terminal_survives_checkpoint_resume() -> None:
    uninterrupted = _recorder()
    candidate_id = "root-1|lsr|long"
    active = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
    )
    uninterrupted.observe(
        _snapshot(
            hypotheses=(active,),
            root_candidates=((candidate_id, active),),
            lifecycle_candidates=((candidate_id, active),),
        ),
        source_bar=_bar(-1),
    )
    uninterrupted.drain_rows()

    missing_bar = _bar(0)
    uninterrupted.on_bar(missing_bar)
    uninterrupted.observe(
        _snapshot(
            1,
            hypotheses=(active,),
            root_candidates=(),
            lifecycle_candidates=(),
            price=missing_bar.close,
        ),
        source_bar=missing_bar,
    )
    assert uninterrupted.drain_rows() == ()
    resumed = BrainCalibrationRecorder.from_state(
        uninterrupted.state_dict()
    )

    terminal_bar = _bar(1)
    dormant_terminal = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.INVALIDATED,
        complete=False,
        terminal_reason="accepted_outside_or_failed",
        terminal_source_ids=("context-1",),
    )
    terminal_snapshot = _snapshot(
        2,
        hypotheses=(dormant_terminal,),
        root_candidates=(),
        lifecycle_candidates=((candidate_id, dormant_terminal),),
        price=terminal_bar.close,
        context_theses={
            "context-1": _context_thesis(
                "context-1",
                lifecycle="invalidated",
                terminal_at=_clock(2),
                terminal_reason="accepted_outside_or_failed",
            )
        },
    )
    for recorder in (uninterrupted, resumed):
        recorder.on_bar(terminal_bar)
        recorder.observe(terminal_snapshot, source_bar=terminal_bar)

    uninterrupted_rows = sorted(
        (row.to_dict() for row in uninterrupted.drain_rows()),
        key=lambda row: row["sample_id"],
    )
    resumed_rows = sorted(
        (row.to_dict() for row in resumed.drain_rows()),
        key=lambda row: row["sample_id"],
    )
    assert resumed_rows == uninterrupted_rows
    thesis = [
        row
        for row in resumed_rows
        if row["dimension"] == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0]["outcome_value"] == 0.0
    assert not thesis[0]["censored"]
    assert thesis[0]["resolution"] == (
        "thesis_contradicted:accepted_outside_or_failed"
    )
    assert resumed.state_dict() == uninterrupted.state_dict()


def test_lsr_disappeared_samples_are_censored_only_at_data_boundary() -> None:
    recorder = _recorder()
    candidate_id = "root-1|lsr|long"
    hypothesis = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.WAITING_LOCATION,
        complete=False,
    )
    recorder.observe(
        _snapshot(
            hypotheses=(hypothesis,),
            root_candidates=((candidate_id, hypothesis),),
            location=_location(),
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    quiet_bar = _bar(0)
    recorder.on_bar(quiet_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(hypothesis,),
            root_candidates=(),
            price=quiet_bar.close,
        ),
        source_bar=quiet_bar,
    )
    assert recorder.drain_rows() == ()

    recorder.on_bar(_bar(1, data_gap=2))
    rows = recorder.drain_rows()
    assert rows
    assert all(row.outcome_value is None for row in rows)
    assert all(row.censored for row in rows)
    assert all(row.resolution == "data_gap_boundary" for row in rows)


def test_dfp_context_contradiction_on_deadline_bar_beats_local_terminal() -> None:
    recorder = _recorder()
    context_id = "dfp-context:h4-structure-1:draw-1"
    structure = SimpleNamespace(
        structure_id="h4-structure-1",
        direction=Direction.LONG,
        lifecycle=StructureLifecycle.CONFIRMED,
        confirmed_at=_clock(-5),
    )
    h1_bos = SimpleNamespace(
        bos_id="h1-bos-1",
        direction=Direction.LONG,
        lifecycle=BOSLifecycle.CONFIRMED,
        scope=BOSScope.CONTINUATION,
    )
    aligned_frames = {
        Timeframe.H4: SimpleNamespace(structures=(structure,)),
        Timeframe.H1: SimpleNamespace(structure_breaks=(h1_bos,)),
    }
    active = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id=context_id,
        context_id=context_id,
    )
    active.sequence.steps = (
        SimpleNamespace(
            step_id="h4_structure_and_draw",
            satisfied=True,
            observed_at=_clock(-4),
            source_ids=("h4-structure-1", "draw-1"),
        ),
        SimpleNamespace(
            step_id="h1_continuation_bos",
            satisfied=True,
            observed_at=_clock(-2),
            source_ids=("h1-bos-1", "h1-target-swing"),
        ),
    )
    active.sequence.completed_steps = 2
    active_snapshot = _snapshot(hypotheses=(active,))
    active_snapshot.observation.frame = lambda timeframe: aligned_frames[
        timeframe
    ]
    recorder.observe(active_snapshot, source_bar=_bar(-1))
    recorder.drain_rows()

    recorder.on_bar(_bar(0))
    terminal = _hypothesis(
        phase=PlaybookPhase.INVALIDATED,
        complete=False,
        setup_id="dfp-entry-episode",
        context_id=context_id,
        terminal_reason="trigger_opposed",
        terminal_source_ids=(context_id, "dfp-entry-episode"),
    )
    terminal_snapshot = _snapshot(1, hypotheses=(terminal,))
    terminal_snapshot.observation.frame = lambda timeframe: aligned_frames[
        timeframe
    ]
    recorder.observe(terminal_snapshot, source_bar=_bar(0))
    assert not any(
        row.dimension == "thesis_strength"
        for row in recorder.drain_rows()
    )

    recorder.on_bar(_bar(9))
    assert recorder.drain_rows() == ()
    opposed = SimpleNamespace(
        structure_id="h4-structure-opposed",
        direction=Direction.SHORT,
        lifecycle=StructureLifecycle.CONFIRMED,
        confirmed_at=_clock(10),
    )
    contradicted_frames = {
        Timeframe.H4: SimpleNamespace(
            structures=(structure, opposed),
        ),
        Timeframe.H1: SimpleNamespace(structure_breaks=(h1_bos,)),
    }
    deadline_snapshot = _snapshot(10, hypotheses=(terminal,))
    deadline_snapshot.observation.frame = lambda timeframe: (
        contradicted_frames[timeframe]
    )
    recorder.observe(deadline_snapshot, source_bar=_bar(9))

    thesis = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "thesis_strength"
        and row.setup_id == context_id
    ]
    assert len(thesis) == 1
    assert thesis[0].outcome_value == 0.0
    assert not thesis[0].censored
    assert thesis[0].resolution == "thesis_contradicted:opposed_structure"


def test_dfp_local_terminal_then_same_context_rearm_keeps_one_revision() -> None:
    recorder = _recorder()
    context_id = "dfp-context:structure-1:draw-1"
    active = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id=context_id,
        context_id=context_id,
    )
    recorder.observe(
        _snapshot(hypotheses=(active,)),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    recorder.on_bar(_bar(0))
    terminal = _hypothesis(
        phase=PlaybookPhase.INVALIDATED,
        complete=False,
        setup_id="dfp-entry-episode",
        context_id=context_id,
        terminal_reason="entry_window_expired",
        terminal_source_ids=(context_id, "dfp-entry-episode"),
    )
    recorder.observe(
        _snapshot(1, hypotheses=(terminal,)),
        source_bar=_bar(0),
    )
    recorder.drain_rows()

    recorder.on_bar(_bar(1))
    rearmed = _hypothesis(
        phase=PlaybookPhase.WAITING_LOCATION,
        complete=False,
        setup_id="dfp-entry-episode-2",
        context_id=context_id,
    )
    recorder.observe(
        _snapshot(2, hypotheses=(rearmed,)),
        source_bar=_bar(1),
    )

    thesis = [
        sample
        for sample in recorder.open_samples
        if sample.dimension == "thesis_strength"
        and sample.setup_id == context_id
    ]
    assert len(thesis) == 1


def test_dfp_context_terminal_settles_every_revision_and_blocks_rearm() -> None:
    recorder = _recorder()
    context_id = "context-thesis-shared"
    structure_id = "h4-structure-shared"
    first = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="context-revision-a",
        setup_id="entry-episode-a",
        context_id=context_id,
        episode_id="entry-episode-a",
    )
    second = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="context-revision-b",
        setup_id="entry-episode-b",
        context_id=context_id,
        episode_id="entry-episode-b",
    )
    structure = _bind_dfp_h4_context(first, structure_id=structure_id)
    _bind_dfp_h4_context(second, structure_id=structure_id)
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                hypotheses=(first, second),
                root_candidates=(("root-a", first), ("root-b", second)),
            ),
            h4_structures=(structure,),
        ),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    assert len(
        [
            sample
            for sample in recorder.open_samples
            if sample.dimension == "thesis_strength"
        ]
    ) == 2

    quiet_bar = _bar(0)
    recorder.on_bar(quiet_bar)
    terminal = _hypothesis(
        phase=PlaybookPhase.INVALIDATED,
        complete=False,
        revision="context-revision-a",
        setup_id="entry-episode-a",
        context_id=context_id,
        episode_id="entry-episode-a",
        terminal_reason="global_frozen_source_invalidated",
        terminal_source_ids=(structure_id,),
    )
    _bind_dfp_h4_context(terminal, structure_id=structure_id)
    recorder.observe(
        _with_structure_frames(
            _snapshot(
                1,
                hypotheses=(terminal, second),
                root_candidates=(
                    ("root-a", terminal),
                    ("root-b", second),
                ),
                price=quiet_bar.close,
            ),
            h4_structures=(structure,),
        ),
        source_bar=quiet_bar,
    )

    thesis = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "thesis_strength"
    ]
    assert len(thesis) == 2
    assert {row.calibration_unit_id for row in thesis} == {context_id}
    assert {row.resolved_at for row in thesis} == {_clock(1)}
    assert {row.outcome_value for row in thesis} == {0.0}
    assert {row.resolution for row in thesis} == {
        "thesis_contradicted:global_frozen_source_invalidated"
    }
    assert not recorder.open_samples

    recorder.on_bar(_bar(1))
    rearmed = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="context-revision-c",
        setup_id="entry-episode-c",
        context_id=context_id,
        episode_id="entry-episode-c",
    )
    recorder.observe(
        _snapshot(
            2,
            hypotheses=(rearmed,),
            root_candidates=(("root-c", rearmed),),
        ),
        source_bar=_bar(1),
    )
    assert recorder.drain_rows() == ()
    assert recorder.open_samples == ()


def test_lsr_sweep_invalidation_tombstones_context_and_blocks_rearm() -> None:
    recorder = _recorder()
    episode_id = "lsr-episode-terminal"
    active = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="lsr-revision-a",
        setup_id=episode_id,
        context_id="lsr-context",
        episode_id=episode_id,
    )
    recorder.observe(
        _snapshot(hypotheses=(active,)),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    invalidation_bar = _bar(0, low=98.5, close=99.5)
    recorder.on_bar(invalidation_bar)
    rearmed = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="lsr-revision-b",
        setup_id=episode_id,
        context_id="lsr-context",
        episode_id=episode_id,
    )
    recorder.observe(
        _snapshot(1, hypotheses=(rearmed,), price=invalidation_bar.close),
        source_bar=invalidation_bar,
    )

    rows = recorder.drain_rows()
    thesis = [row for row in rows if row.dimension == "thesis_strength"]
    assert len(thesis) == 1
    assert thesis[0].calibration_unit_id == "lsr-context"
    assert thesis[0].calibration_unit_kind == "lsr_context_thesis"
    assert thesis[0].resolution == "invalidation_touched"
    assert thesis[0].outcome_value == 0.0
    assert not any(
        row.evidence_revision_id == "lsr-revision-b" for row in rows
    )
    assert recorder.open_samples == ()

    resumed = BrainCalibrationRecorder.from_state(recorder.state_dict())
    resumed.on_bar(_bar(1))
    resumed.observe(
        _snapshot(2, hypotheses=(rearmed,)),
        source_bar=_bar(1),
    )
    assert resumed.drain_rows() == ()
    assert resumed.open_samples == ()


def test_dfp_thesis_horizon_allows_new_child_episode_targets() -> None:
    recorder = _recorder()
    context_id = "dfp-context-session-horizon"
    context = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="context-revision",
        setup_id="context-only",
        context_id=context_id,
        episode_id=None,
    )
    context.episode_id = None
    context.parent_context_thesis_id = None
    context.thesis_deadline = _clock(1)
    recorder.observe(
        _snapshot(hypotheses=(context,)),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    deadline_bar = _bar(0)
    recorder.on_bar(deadline_bar)
    recorder.observe(
        _snapshot(1, hypotheses=(context,), price=deadline_bar.close),
        source_bar=deadline_bar,
    )
    thesis = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0].resolution == "thesis_intact_at_deadline"

    recorder.on_bar(_bar(1))
    child = _hypothesis(
        phase=PlaybookPhase.EXECUTABLE,
        complete=True,
        revision="child-revision",
        setup_id="child-episode",
        context_id=context_id,
        episode_id="child-episode",
    )
    child.plan.entry_path_id = "child-path"
    child.selected_trigger.entry_path_id = "child-path"
    child.selected_trigger.observed_at = _clock(2)
    recorder.observe(
        _snapshot(2, hypotheses=(child,), location=_location()),
        source_bar=_bar(1),
    )

    rows = recorder.drain_rows()
    assert not any(row.dimension == "thesis_strength" for row in rows)
    assert {
        sample.dimension for sample in recorder.open_samples
    } == {"location_quality", "entry_readiness", "delivery_quality"}
    assert {
        sample.episode_id for sample in recorder.open_samples
    } == {"child-episode"}


def test_dfp_horizon_promotes_only_exact_context_terminal_and_blocks_rearm(
) -> None:
    recorder = _recorder()
    context_id = "dfp-context-promoted-terminal"
    context = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="context-revision",
        setup_id="context-only",
        context_id=context_id,
        episode_id=None,
    )
    context.episode_id = None
    context.parent_context_thesis_id = None
    context.thesis_deadline = _clock(1)
    recorder.observe(
        _snapshot(hypotheses=(context,)),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    deadline_bar = _bar(0)
    recorder.on_bar(deadline_bar)
    recorder.observe(
        _snapshot(1, hypotheses=(context,), price=deadline_bar.close),
        source_bar=deadline_bar,
    )
    horizon_rows = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "thesis_strength"
    ]
    assert len(horizon_rows) == 1
    assert horizon_rows[0].resolution == "thesis_intact_at_deadline"

    child_a = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="child-a-revision",
        setup_id="child-a",
        context_id=context_id,
        episode_id="child-a",
    )
    child_a.plan.entry_path_id = "child-a-path"
    child_a.entry_path_id = "child-a-path"
    child_a.entry_location_id = "child-a-location"
    child_a_location = SimpleNamespace(
        location_id="child-a-location",
        lifecycle=EntryLocationLifecycle.IN_ZONE,
    )
    child_a_bar = _bar(1)
    recorder.on_bar(child_a_bar)
    recorder.observe(
        _snapshot(
            2,
            hypotheses=(child_a,),
            location=child_a_location,
            price=child_a_bar.close,
        ),
        source_bar=child_a_bar,
    )
    recorder.drain_rows()
    assert any(
        sample.episode_id == "child-a"
        for sample in recorder.open_samples
    )

    terminal_bar = _bar(2)
    terminal_snapshot = _snapshot(
        3,
        hypotheses=(child_a,),
        location=child_a_location,
        price=terminal_bar.close,
    )
    terminal_snapshot.belief.context_theses = {
        context_id: SimpleNamespace(
            context_thesis_id=context_id,
            direction=Direction.LONG,
            lifecycle="invalidated",
            terminal_at=_clock(3),
            terminal_reason="context_structural_invalidation_breached",
        )
    }
    recorder.on_bar(terminal_bar)
    recorder.observe(terminal_snapshot, source_bar=terminal_bar)
    recorder.drain_rows()

    terminal_units = recorder.state_dict()["terminal_thesis_units"]
    assert len(terminal_units) == 1
    assert terminal_units[0]["calibration_unit_id"] == context_id
    assert terminal_units[0]["scope"] == "context_terminal"
    assert terminal_units[0]["resolved_at"] == _clock(3).isoformat()
    assert terminal_units[0]["resolution"] == (
        "context_terminal:context_structural_invalidation_breached"
    )
    # Promotion is recorder state only; the already emitted horizon row is
    # immutable and no replacement thesis result is manufactured.
    assert not any(
        row.dimension == "thesis_strength"
        for row in recorder.rows
    )

    resumed = BrainCalibrationRecorder.from_state(recorder.state_dict())
    child_b = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="child-b-revision",
        setup_id="child-b",
        context_id=context_id,
        episode_id="child-b",
    )
    child_b.plan.entry_path_id = "child-b-path"
    child_b.entry_path_id = "child-b-path"
    child_b.entry_location_id = "child-b-location"
    child_b_bar = _bar(3)
    resumed.on_bar(child_b_bar)
    child_b_snapshot = _snapshot(
        4,
        hypotheses=(child_b,),
        location=SimpleNamespace(
            location_id="child-b-location",
            lifecycle=EntryLocationLifecycle.IN_ZONE,
        ),
        price=child_b_bar.close,
    )
    child_b_snapshot.belief.context_theses = terminal_snapshot.belief.context_theses
    resumed.observe(child_b_snapshot, source_bar=child_b_bar)

    assert not any(
        row.episode_id == "child-b" for row in resumed.drain_rows()
    )
    assert not any(
        sample.episode_id == "child-b"
        for sample in resumed.open_samples
    )


def test_dfp_local_child_terminal_does_not_promote_context_horizon() -> None:
    recorder = _recorder()
    context_id = "dfp-context-local-terminal"
    context = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="context-revision",
        setup_id="context-only",
        context_id=context_id,
        episode_id=None,
    )
    context.episode_id = None
    context.parent_context_thesis_id = None
    context.thesis_deadline = _clock(1)
    recorder.observe(
        _snapshot(hypotheses=(context,)),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    deadline_bar = _bar(0)
    recorder.on_bar(deadline_bar)
    recorder.observe(
        _snapshot(1, hypotheses=(context,), price=deadline_bar.close),
        source_bar=deadline_bar,
    )
    recorder.drain_rows()

    local_terminal = _hypothesis(
        phase=PlaybookPhase.INVALIDATED,
        complete=False,
        revision="local-terminal-revision",
        setup_id="local-child-a",
        context_id=context_id,
        episode_id="local-child-a",
        terminal_at=_clock(2),
        terminal_reason="entry_window_expired",
        terminal_source_ids=("local-child-a",),
    )
    local_bar = _bar(1)
    recorder.on_bar(local_bar)
    recorder.observe(
        _snapshot(2, hypotheses=(local_terminal,), price=local_bar.close),
        source_bar=local_bar,
    )
    assert recorder.state_dict()["terminal_thesis_units"][0]["scope"] == (
        "thesis_observation_horizon"
    )

    child_b = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        revision="child-b-revision",
        setup_id="local-child-b",
        context_id=context_id,
        episode_id="local-child-b",
    )
    child_b.plan.entry_path_id = "local-child-b-path"
    child_b.entry_path_id = "local-child-b-path"
    child_b.entry_location_id = "local-child-b-location"
    child_b_bar = _bar(2)
    recorder.on_bar(child_b_bar)
    recorder.observe(
        _snapshot(
            3,
            hypotheses=(child_b,),
            location=SimpleNamespace(
                location_id="local-child-b-location",
                lifecycle=EntryLocationLifecycle.IN_ZONE,
            ),
            price=child_b_bar.close,
        ),
        source_bar=child_b_bar,
    )
    assert any(
        sample.episode_id == "local-child-b"
        for sample in recorder.open_samples
    )


def test_non_future_deadline_skips_fitted_targets_once_per_causal_unit() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis()
    hypothesis.thesis_deadline = _clock(-1)
    hypothesis.episode_deadline = _clock(-1)
    hypothesis.plan.deadline = _clock(-1)

    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )

    rows = recorder.drain_rows()
    fitted = [
        row
        for row in rows
        if row.dimension
        in {
            "thesis_strength",
            "location_quality",
            "entry_readiness",
            "delivery_quality",
        }
    ]
    assert fitted == []
    descriptive = [
        row
        for row in rows
        if row.dimension in {"sequence_progress", "uncertainty"}
    ]
    assert {row.dimension for row in descriptive} == {
        "sequence_progress",
        "uncertainty",
    }
    assert all(not row.censored for row in descriptive)
    assert all(row.deadline == _clock(-1) for row in rows)
    assert recorder.open_samples == ()
    assert recorder.late_registration_summary == {
        "counting_basis": "unique_expired_calibration_unit",
        "late_registration_skipped": 3,
        "by_playbook": {
            Playbook.DISPLACEMENT_FIRST_PULLBACK.value: 3,
        },
        "incomplete_registration_skipped": 0,
        "incomplete_by_dimension": {},
    }

    # A new revision from the same expired owners does not emit repeated
    # non-future rows or inflate the light summary.
    repeated = _hypothesis(revision="revision-2")
    repeated.thesis_deadline = _clock(-1)
    repeated.episode_deadline = _clock(-1)
    repeated.plan.deadline = _clock(-1)
    recorder.observe(
        _snapshot(1, hypotheses=(repeated,), location=_location()),
        source_bar=_bar(0),
    )
    assert {
        row.dimension for row in recorder.drain_rows()
    } == {"uncertainty"}
    assert recorder.late_registration_summary["late_registration_skipped"] == 3


def test_frozen_delivery_survives_brain_setup_rearm() -> None:
    recorder = _recorder()
    triggered = _hypothesis(
        phase=PlaybookPhase.EXECUTABLE,
        setup_id="triggered-setup",
    )
    recorder.observe(
        _snapshot(hypotheses=(triggered,), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    # The next bar resolves readiness but not the frozen draw delivery.
    recorder.on_bar(_bar(0, close=100.0))
    recorder.drain_rows()
    rearmed = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id="new-setup",
        revision="new-setup-revision",
    )
    recorder.observe(
        _snapshot(1, hypotheses=(rearmed,)),
        source_bar=_bar(0),
    )
    rows = recorder.drain_rows()
    assert not any(
        row.dimension == "delivery_quality" and row.censored
        for row in rows
    )
    delivery = [
        sample
        for sample in recorder.open_samples
        if sample.dimension == "delivery_quality"
    ]
    assert len(delivery) == 1
    assert delivery[0].setup_id == "triggered-setup"

    delivery_bar = _bar(1, high=103.5, close=103.0)
    recorder.on_bar(delivery_bar)
    recorder.observe(
        _snapshot(2, hypotheses=(rearmed,), price=delivery_bar.close),
        source_bar=delivery_bar,
    )
    resolved = recorder.drain_rows()
    delivered = [
        row for row in resolved if row.dimension == "delivery_quality"
    ]
    assert len(delivered) == 1
    assert delivered[0].outcome_value == 1.0
    assert delivered[0].resolution == "draw_delivered"


@pytest.mark.parametrize(
    "terminal_reason",
    [
        "trigger_opposed",
        "frozen_invalidation_breached",
        "micro_bos_opposed",
        "mss_confirmed_after_first_pullback",
        "manipulation_resolution_deadline",
    ],
)
def test_invalid_terminal_trigger_does_not_create_samples(
    terminal_reason: str,
) -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(
        phase=PlaybookPhase.INVALIDATED,
        terminal_reason=terminal_reason,
    )
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )

    assert recorder.rows == ()
    assert recorder.open_samples == ()


def test_location_resolves_from_authoritative_lifecycle() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(
        phase=PlaybookPhase.WAITING_LOCATION,
        complete=False,
    )
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    recorder.on_bar(_bar(0, close=100.0))

    recorder.observe(
        _snapshot(
            1,
            hypotheses=(hypothesis,),
            location=_location(EntryLocationLifecycle.REJECTED),
        ),
        source_bar=_bar(0),
    )
    rows = recorder.drain_rows()
    location_rows = [row for row in rows if row.dimension == "location_quality"]
    assert len(location_rows) == 1
    assert location_rows[0].outcome_value == 1.0
    assert location_rows[0].resolution == "frozen_zone_rejected"


def test_checkpoint_roundtrip_and_data_boundary_censor() -> None:
    recorder = _recorder()
    recorder.observe(
        _snapshot(location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    restored = BrainCalibrationRecorder.from_state(recorder.state_dict())

    assert restored.state_dict() == recorder.state_dict()
    restored.on_bar(_bar(0, data_gap=2))
    rows = restored.drain_rows()
    assert len(rows) == 4
    assert all(row.censored for row in rows)
    assert all(row.outcome_value is None for row in rows)
    assert all(row.resolution == "data_gap_boundary" for row in rows)


def test_checkpoint_with_unknown_recorder_schema_fails_closed() -> None:
    state = _recorder().state_dict()
    state["schema_version"] = 999

    with pytest.raises(ValueError, match="unsupported brain calibration recorder"):
        BrainCalibrationRecorder.from_state(state)


def test_schema_11_checkpoint_cannot_cross_causal_retention_contract() -> None:
    state = _recorder().state_dict()
    state["schema_version"] = 11

    with pytest.raises(ValueError, match="unsupported brain calibration recorder"):
        BrainCalibrationRecorder.from_state(state)


def test_schema_12_checkpoint_cannot_cross_context_target_contract() -> None:
    state = _recorder().state_dict()
    state["schema_version"] = 12

    with pytest.raises(ValueError, match="unsupported brain calibration recorder"):
        BrainCalibrationRecorder.from_state(state)


def test_schema_13_checkpoint_cannot_cross_terminal_owner_contract() -> None:
    state = _recorder().state_dict()
    state["schema_version"] = 13

    with pytest.raises(ValueError, match="unsupported brain calibration recorder"):
        BrainCalibrationRecorder.from_state(state)


def test_schema_14_checkpoint_cannot_cross_context_terminal_promotion_contract(
) -> None:
    state = _recorder().state_dict()
    state["schema_version"] = 14

    with pytest.raises(ValueError, match="unsupported brain calibration recorder"):
        BrainCalibrationRecorder.from_state(state)


def test_schema_15_checkpoint_cannot_cross_dfp_opposition_clock_contract(
) -> None:
    state = _recorder().state_dict()
    state["schema_version"] = 15

    with pytest.raises(ValueError, match="unsupported brain calibration recorder"):
        BrainCalibrationRecorder.from_state(state)


def test_schema_16_checkpoint_cannot_cross_lsr_context_owner_contract() -> None:
    state = _recorder().state_dict()
    state["schema_version"] = 16

    with pytest.raises(ValueError, match="unsupported brain calibration recorder"):
        BrainCalibrationRecorder.from_state(state)


def test_schema_17_checkpoint_cannot_cross_lsr_context_target_contract() -> None:
    state = _recorder().state_dict()
    state["schema_version"] = 17

    with pytest.raises(ValueError, match="unsupported brain calibration recorder"):
        BrainCalibrationRecorder.from_state(state)


def test_schema_18_checkpoint_cannot_cross_owner_admission_contract() -> None:
    state = _recorder().state_dict()
    state["schema_version"] = 18

    with pytest.raises(ValueError, match="unsupported brain calibration recorder"):
        BrainCalibrationRecorder.from_state(state)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("draw_id", "child-draw"),
        ("liquidity_route_id", "child-route"),
        ("source_path_ids", '["child-target"]'),
        ("invalidation_price", None),
        ("invalidation_source_id", None),
        ("deadline", None),
    ),
)
def test_schema_19_open_lsr_thesis_rejects_child_target_or_missing_sweep(
    field: str,
    value: object,
) -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
    )
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,)),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    state = recorder.state_dict()
    thesis = next(
        item
        for item in state["open_samples"]
        if item["dimension"] == "thesis_strength"
    )
    thesis[field] = value

    with pytest.raises(
        ValueError,
        match="LSR Context thesis target custody is invalid",
    ):
        BrainCalibrationRecorder.from_state(state)


def test_schema_19_queued_lsr_thesis_rejects_child_target_custody() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
    )
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,)),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    invalidation_bar = _bar(0, low=98.5, close=99.5)
    recorder.on_bar(invalidation_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(hypothesis,),
            root_candidates=(),
            lifecycle_candidates=(),
            price=invalidation_bar.close,
        ),
        source_bar=invalidation_bar,
    )
    state = recorder.state_dict()
    thesis = next(
        item
        for item in state["queued_rows"]
        if item["dimension"] == "thesis_strength"
    )
    thesis["primary_deliverable_target_id"] = "child-target"

    with pytest.raises(
        ValueError,
        match="LSR Context thesis target custody is invalid",
    ):
        BrainCalibrationRecorder.from_state(state)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("scope", "episode_terminal"),
        ("calibration_unit_kind", "lsr_sweep_thesis"),
        ("calibration_unit_id", "lsr-child-episode"),
    ),
)
def test_lsr_context_tombstone_checkpoint_fails_closed_on_old_child_owner(
    field: str,
    value: str,
) -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.ARMED,
        complete=False,
    )
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,)),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    invalidation_bar = _bar(0, low=98.5, close=99.5)
    recorder.on_bar(invalidation_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(hypothesis,),
            root_candidates=(),
            lifecycle_candidates=(),
            price=invalidation_bar.close,
        ),
        source_bar=invalidation_bar,
    )
    state = recorder.state_dict()
    assert len(state["terminal_thesis_units"]) == 1
    state["terminal_thesis_units"][0][field] = value

    with pytest.raises(ValueError, match="terminal thesis checkpoint row is invalid"):
        BrainCalibrationRecorder.from_state(state)


def test_synthetic_no_trade_bar_is_ignored_not_censored() -> None:
    recorder = _recorder()
    recorder.observe(
        _snapshot(location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    before = tuple(item.sample_id for item in recorder.open_samples)

    recorder.on_bar(_bar(0, synthetic=True))
    assert recorder.drain_rows() == ()
    assert tuple(item.sample_id for item in recorder.open_samples) == before


def test_data_gap_censor_clock_cannot_cross_frozen_deadline() -> None:
    recorder = _recorder()
    recorder.observe(
        _snapshot(location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    recorder.on_bar(_bar(20, data_gap=2))
    rows = recorder.drain_rows()

    assert len(rows) == 4
    assert all(row.censored for row in rows)
    assert all(row.resolved_at == _clock(10) for row in rows)
    assert all(row.resolution == "data_gap_boundary" for row in rows)


def test_synthetic_no_trade_bar_settles_targets_at_frozen_deadline() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis()
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    deadline_bar = _bar(9, synthetic=True)
    recorder.on_bar(deadline_bar)
    recorder.observe(
        _snapshot(
            10,
            hypotheses=(hypothesis,),
            location=_location(),
            price=deadline_bar.close,
        ),
        source_bar=deadline_bar,
    )
    rows = recorder.drain_rows()

    assert {row.dimension for row in rows} == {
        "thesis_strength",
        "location_quality",
        "entry_readiness",
        "delivery_quality",
    }
    assert all(row.resolved_at == _clock(10) for row in rows)
    thesis = next(row for row in rows if row.dimension == "thesis_strength")
    assert thesis.outcome_value == 1.0
    assert thesis.resolution == "thesis_intact_at_deadline"


def test_missing_draw_consumes_thesis_revision_to_prevent_future_backfill() -> None:
    recorder = _recorder()
    missing = _hypothesis(
        phase=PlaybookPhase.WAITING_LOCATION,
        complete=False,
    )
    missing.draw_selection = None
    missing.deliverable_targets = ()
    recorder.observe(
        _snapshot(hypotheses=(missing,), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    assert not any(
        item.dimension == "thesis_strength"
        for item in recorder.open_samples
    )

    supplied = _hypothesis(
        phase=PlaybookPhase.WAITING_LOCATION,
        complete=False,
    )
    recorder.on_bar(_bar(0))
    recorder.observe(
        _snapshot(1, hypotheses=(supplied,), location=_location()),
        source_bar=_bar(0),
    )
    assert not any(
        item.dimension == "thesis_strength"
        for item in recorder.open_samples
    )
    assert recorder.late_registration_summary[
        "incomplete_registration_skipped"
    ] == 1


def test_window_end_right_censors_without_future_label() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(
        phase=PlaybookPhase.WAITING_LOCATION,
        complete=False,
    )
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    recorder.close_unresolved(_clock(5))
    rows = recorder.drain_rows()
    assert {row.dimension for row in rows} == {
        "thesis_strength",
        "location_quality",
    }
    assert all(row.censored and row.outcome_value is None for row in rows)
    assert all(row.resolution == "window_end" for row in rows)


def test_same_bar_draw_consumption_only_wins_without_structural_terminal() -> None:
    delivered = _recorder()
    active = _hypothesis()
    delivered.observe(
        _snapshot(hypotheses=(active,), location=_location()),
        source_bar=_bar(-1),
    )
    delivered.drain_rows()

    delivery_bar = _bar(0, high=103.5, low=99.5, close=103.0)
    delivered.on_bar(delivery_bar)
    draw_terminal = _hypothesis(
        phase=PlaybookPhase.INVALIDATED,
        terminal_reason="selected_draw_consumed_or_missing",
        terminal_source_ids=("setup-1", "draw-1"),
    )
    delivered.observe(
        _snapshot(
            1,
            hypotheses=(draw_terminal,),
            location=_location(),
            price=delivery_bar.close,
        ),
        source_bar=delivery_bar,
    )
    delivered_thesis = [
        row
        for row in delivered.drain_rows()
        if row.dimension == "thesis_strength"
    ]
    assert len(delivered_thesis) == 1
    assert delivered_thesis[0].outcome_value == 1.0
    assert delivered_thesis[0].resolution == "draw_delivered"

    contradicted = _recorder()
    contradicted.observe(
        _snapshot(hypotheses=(active,), location=_location()),
        source_bar=_bar(-1),
    )
    contradicted.drain_rows()
    contradicted.on_bar(delivery_bar)
    structural_terminal = _hypothesis(
        phase=PlaybookPhase.INVALIDATED,
        terminal_reason="opposed_structure",
        terminal_source_ids=("setup-1", "draw-1"),
    )
    contradicted.observe(
        _snapshot(
            1,
            hypotheses=(structural_terminal,),
            location=_location(),
            price=delivery_bar.close,
        ),
        source_bar=delivery_bar,
    )
    contradicted_thesis = [
        row
        for row in contradicted.drain_rows()
        if row.dimension == "thesis_strength"
    ]
    assert len(contradicted_thesis) == 1
    assert contradicted_thesis[0].outcome_value == 0.0
    assert contradicted_thesis[0].resolution == (
        "thesis_contradicted:opposed_structure"
    )


def test_pending_same_bar_outcome_survives_checkpoint_resume_exactly() -> None:
    uninterrupted = _recorder()
    active = _hypothesis()
    uninterrupted.observe(
        _snapshot(hypotheses=(active,), location=_location()),
        source_bar=_bar(-1),
    )
    uninterrupted.drain_rows()
    delivery_bar = _bar(0, high=103.5, low=99.5, close=103.0)
    uninterrupted.on_bar(delivery_bar)

    resumed = BrainCalibrationRecorder.from_state(
        uninterrupted.state_dict()
    )
    active_after_bar = _hypothesis()
    snapshot = _snapshot(
        1,
        hypotheses=(active_after_bar,),
        location=_location(),
        price=delivery_bar.close,
    )
    uninterrupted.observe(snapshot, source_bar=delivery_bar)
    resumed.observe(snapshot, source_bar=delivery_bar)

    resumed_rows = sorted(
        (row.to_dict() for row in resumed.drain_rows()),
        key=lambda row: row["sample_id"],
    )
    uninterrupted_rows = sorted(
        (row.to_dict() for row in uninterrupted.drain_rows()),
        key=lambda row: row["sample_id"],
    )
    assert resumed_rows == uninterrupted_rows
    assert resumed.state_dict() == uninterrupted.state_dict()


def test_draw_identity_disappearance_without_price_touch_is_censored() -> None:
    recorder = _recorder()
    recorder.observe(
        _snapshot(hypotheses=(_hypothesis(),), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    quiet_bar = _bar(0, high=101.5, low=99.5, close=101.0)
    recorder.on_bar(quiet_bar)
    terminal = _hypothesis(
        phase=PlaybookPhase.INVALIDATED,
        terminal_reason="selected_draw_consumed_or_missing",
        terminal_source_ids=("setup-1", "draw-1"),
    )
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(terminal,),
            location=_location(),
            price=quiet_bar.close,
        ),
        source_bar=quiet_bar,
    )

    thesis = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0].outcome_value is None
    assert thesis[0].censored
    assert thesis[0].resolution == "non_thesis_terminal_unresolved"


def test_non_aligned_deadline_conservatively_censors_crossing_bar() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis()
    deadline = _clock(0) + pd.Timedelta(seconds=30)
    hypothesis.thesis_deadline = deadline
    hypothesis.episode_deadline = deadline
    hypothesis.plan.deadline = deadline
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    # This completed bar straddles the frozen deadline.  Neither its pre- nor
    # post-deadline extremes can be ordered from OHLCV, so no directional label
    # may be inferred even though it touches both draw and invalidation.
    crossing_bar = _bar(0, high=104.0, low=98.5, close=103.5)
    recorder.on_bar(crossing_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(hypothesis,),
            location=_location(),
            price=crossing_bar.close,
        ),
        source_bar=crossing_bar,
    )
    rows = recorder.drain_rows()

    assert {row.dimension for row in rows} == {
        "thesis_strength",
        "location_quality",
        "entry_readiness",
        "delivery_quality",
    }
    assert all(row.resolved_at == deadline for row in rows)
    assert all(row.outcome_value is None and row.censored for row in rows)
    assert all(
        row.resolution == "deadline_crossed_inside_completed_bar"
        for row in rows
    )


def test_dfp_non_aligned_deadline_precedes_post_deadline_h4_opposition() -> None:
    recorder = _recorder()
    context_id = "dfp-context:h4-structure-1:draw-1"
    deadline = _clock(0) + pd.Timedelta(seconds=30)
    structure = SimpleNamespace(
        structure_id="h4-structure-1",
        direction=Direction.LONG,
        lifecycle=StructureLifecycle.CONFIRMED,
        confirmed_at=_clock(-5),
    )
    h1_bos = SimpleNamespace(
        bos_id="h1-bos-1",
        direction=Direction.LONG,
        lifecycle=BOSLifecycle.CONFIRMED,
        scope=BOSScope.CONTINUATION,
    )
    active = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
        setup_id=context_id,
        context_id=context_id,
    )
    active.thesis_deadline = deadline
    active.sequence.steps = (
        SimpleNamespace(
            step_id="h4_structure_and_draw",
            satisfied=True,
            observed_at=_clock(-4),
            source_ids=("h4-structure-1", "draw-1"),
        ),
        SimpleNamespace(
            step_id="h1_continuation_bos",
            satisfied=True,
            observed_at=_clock(-2),
            source_ids=("h1-bos-1", "h1-target-swing"),
        ),
    )
    active.sequence.completed_steps = 2
    aligned_frames = {
        Timeframe.H4: SimpleNamespace(structures=(structure,)),
        Timeframe.H1: SimpleNamespace(structure_breaks=(h1_bos,)),
    }
    initial = _snapshot(hypotheses=(active,))
    initial.observation.frame = lambda timeframe: aligned_frames[timeframe]
    recorder.observe(initial, source_bar=_bar(-1))
    recorder.drain_rows()

    crossing_bar = _bar(0)
    recorder.on_bar(crossing_bar)
    opposed = SimpleNamespace(
        structure_id="h4-structure-opposed",
        direction=Direction.SHORT,
        lifecycle=StructureLifecycle.CONFIRMED,
        confirmed_at=_clock(1),
    )
    post_deadline_frames = {
        Timeframe.H4: SimpleNamespace(structures=(structure, opposed)),
        Timeframe.H1: SimpleNamespace(structure_breaks=(h1_bos,)),
    }
    post_deadline = _snapshot(
        1,
        hypotheses=(active,),
        price=crossing_bar.close,
    )
    post_deadline.observation.frame = (
        lambda timeframe: post_deadline_frames[timeframe]
    )
    recorder.observe(post_deadline, source_bar=crossing_bar)

    thesis = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "thesis_strength"
    ]
    assert len(thesis) == 1
    assert thesis[0].resolved_at == deadline
    assert thesis[0].outcome_value is None
    assert thesis[0].censored
    assert thesis[0].resolution == "deadline_crossed_inside_completed_bar"


def test_same_setup_entry_path_replacement_keeps_old_location_owner() -> None:
    recorder = _recorder()
    active = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.WAITING_LOCATION,
        complete=False,
        setup_id="shared-setup",
    )
    recorder.observe(
        _snapshot(hypotheses=(active,), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    old_location = next(
        sample
        for sample in recorder.open_samples
        if sample.dimension == "location_quality"
    )
    assert old_location.calibration_unit_id == "path-1"

    transition_bar = _bar(0)
    recorder.on_bar(transition_bar)
    replacement = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.WAITING_LOCATION,
        complete=False,
        setup_id="shared-setup",
    )
    replacement.plan.entry_path_id = "path-2"
    replacement.entry_location_id = "location-2"
    replacement_location = SimpleNamespace(
        location_id="location-2",
        lifecycle=EntryLocationLifecycle.IN_ZONE,
    )
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(replacement,),
            location=replacement_location,
            price=transition_bar.close,
        ),
        source_bar=transition_bar,
    )

    replacement_rows = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "location_quality"
    ]
    assert replacement_rows == []
    locations = [
        sample
        for sample in recorder.open_samples
        if sample.dimension == "location_quality"
    ]
    assert {sample.calibration_unit_id for sample in locations} == {
        "path-1",
        "path-2",
    }
    new_location = next(
        sample
        for sample in locations
        if sample.calibration_unit_id == "path-2"
    )
    assert new_location.entry_location_id == "location-2"


def test_same_setup_new_path_terminal_only_closes_matching_location_owner(
) -> None:
    recorder = _recorder()
    active = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.WAITING_LOCATION,
        complete=False,
        setup_id="shared-setup",
    )
    recorder.observe(
        _snapshot(hypotheses=(active,), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()
    old_location = next(
        sample
        for sample in recorder.open_samples
        if sample.dimension == "location_quality"
    )

    transition_bar = _bar(0)
    recorder.on_bar(transition_bar)
    terminal = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.INVALIDATED,
        complete=False,
        setup_id="shared-setup",
        terminal_reason="entry_zone_left",
        terminal_source_ids=("shared-setup", "path-2", "location-2"),
    )
    terminal.plan.entry_path_id = "path-2"
    terminal.entry_location_id = "location-2"
    replacement_location = SimpleNamespace(
        location_id="location-2",
        lifecycle=EntryLocationLifecycle.LEFT,
    )
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(terminal,),
            location=replacement_location,
            price=transition_bar.close,
        ),
        source_bar=transition_bar,
    )

    old_owner_rows = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "location_quality"
        and row.sample_id == old_location.sample_id
    ]
    assert old_owner_rows == []
    assert old_location.sample_id in {
        sample.sample_id for sample in recorder.open_samples
    }

    # The same projected candidate may later report a terminal for the exact
    # frozen owner.  terminal_source_ids, not the current path selection,
    # supplies the causal custody needed to close it.
    matching_bar = _bar(1)
    recorder.on_bar(matching_bar)
    matching_terminal = _hypothesis(
        playbook=Playbook.LIQUIDITY_SWEEP_REVERSAL,
        phase=PlaybookPhase.INVALIDATED,
        complete=False,
        setup_id="shared-setup",
        terminal_reason="entry_zone_left",
        # The frozen typed location is sufficient causal custody even though
        # this minute's selected candidate now projects another path.
        terminal_source_ids=("shared-setup", "location-1"),
    )
    matching_terminal.plan.entry_path_id = "path-2"
    matching_terminal.entry_location_id = "location-2"
    recorder.observe(
        _snapshot(
            2,
            hypotheses=(matching_terminal,),
            location=replacement_location,
            price=matching_bar.close,
        ),
        source_bar=matching_bar,
    )
    matched_rows = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "location_quality"
        and row.sample_id == old_location.sample_id
    ]
    assert len(matched_rows) == 1
    assert matched_rows[0].calibration_unit_id == "path-1"
    assert matched_rows[0].outcome_value == 0.0
    assert not matched_rows[0].censored
    assert matched_rows[0].resolution == "location_failed:entry_zone_left"


def test_uncertainty_dedupes_identical_state_and_revises_on_component_change() -> None:
    recorder = _recorder()
    unchanged = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
    )
    recorder.observe(
        _snapshot(hypotheses=(unchanged,)),
        source_bar=_bar(-1),
    )
    initial_uncertainty = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "uncertainty"
    ]
    assert len(initial_uncertainty) == 1

    first_bar = _bar(0)
    recorder.on_bar(first_bar)
    recorder.observe(
        _snapshot(1, hypotheses=(unchanged,), price=first_bar.close),
        source_bar=first_bar,
    )
    assert not any(
        row.dimension == "uncertainty"
        for row in recorder.drain_rows()
    )

    changed = _hypothesis(
        phase=PlaybookPhase.ARMED,
        complete=False,
    )
    changed.context_metadata = {
        "uncertainty_conflict": 0.2,
        "uncertainty_required_evidence_missing": 0.0,
        "uncertainty_authority_missing": 0.1,
        "uncertainty_graph_ambiguity": 0.0,
        "uncertainty_total": 0.28,
    }
    changed.raw_quality_dimensions["uncertainty"] = 0.28
    changed.uncertainty = 0.28
    second_bar = _bar(1)
    recorder.on_bar(second_bar)
    recorder.observe(
        _snapshot(2, hypotheses=(changed,), price=second_bar.close),
        source_bar=second_bar,
    )
    changed_uncertainty = [
        row
        for row in recorder.drain_rows()
        if row.dimension == "uncertainty"
    ]
    assert len(changed_uncertainty) == 1
    assert changed_uncertainty[0].sample_id != initial_uncertainty[0].sample_id
    assert changed_uncertainty[0].raw_value == pytest.approx(0.28)
    assert changed_uncertainty[0].uncertainty_conflict == pytest.approx(0.2)
    assert changed_uncertainty[0].uncertainty_authority_missing == pytest.approx(
        0.1
    )


def test_warmup_prime_skips_left_censored_trigger_but_accepts_new_path() -> None:
    recorder = _recorder()
    warmup = _hypothesis()
    warmup_snapshot = _snapshot(
        hypotheses=(warmup,),
        location=_location(),
    )
    recorder.on_bar(_bar(-1))
    recorder.prime(warmup_snapshot, source_bar=_bar(-1))
    assert recorder.open_samples == ()
    assert recorder.drain_rows() == ()

    first_bar = _bar(0)
    recorder.on_bar(first_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(warmup,),
            location=_location(),
            price=first_bar.close,
        ),
        source_bar=first_bar,
    )
    assert recorder.open_samples == ()
    assert recorder.drain_rows() == ()

    new_path = _hypothesis(revision="revision-2")
    new_path.plan.entry_path_id = "path-2"
    new_path.selected_trigger.entry_path_id = "path-2"
    new_path.selected_trigger.trigger_id = "trigger-2"
    new_path.selected_trigger.observed_at = _clock(2)
    second_bar = _bar(1)
    recorder.on_bar(second_bar)
    recorder.observe(
        _snapshot(
            2,
            hypotheses=(new_path,),
            location=_location(),
            price=second_bar.close,
        ),
        source_bar=second_bar,
    )
    assert any(
        sample.dimension == "entry_readiness"
        and sample.calibration_unit_id == "path-2"
        for sample in recorder.open_samples
    )


def test_warmup_trigger_without_plan_allows_first_capture_plan_delivery() -> None:
    recorder = _recorder()
    warmup = _hypothesis()
    warmup.plan = None
    warmup_snapshot = _snapshot(hypotheses=(warmup,))
    warmup_snapshot.observation.path_sequences = (
        SimpleNamespace(
            context_kind="zone_return",
            context_id="location-1",
            direction=Direction.LONG,
            last_updated_at=_clock(0),
            sequence_id="path-1",
        ),
    )
    recorder.on_bar(_bar(-1))
    recorder.prime(warmup_snapshot, source_bar=_bar(-1))

    with_plan = _hypothesis()
    capture_bar = _bar(0)
    recorder.on_bar(capture_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(with_plan,),
            price=capture_bar.close,
        ),
        source_bar=capture_bar,
    )

    assert {
        sample.dimension for sample in recorder.open_samples
    } == {"delivery_quality"}
    delivery = recorder.open_samples[0]
    assert delivery.calibration_unit_id == "path-1"
    assert delivery.sampled_at == _clock(1)
    assert delivery.target_deadline_kind == "plan_deadline"


def test_hard_barrier_delivery_is_recorded_but_not_fit_eligible() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis()
    hypothesis.context_metadata = {
        "obstruction_distance_R": "0.5",
        "free_path_R": "0.4",
        "soft_obstruction_count": "0",
        "hard_barrier_before_target": "true",
        "path_blocker_ids": '["blocker:authority"]',
    }
    # A terminal/descriptive hypothesis can retain obstruction diagnostics
    # after its plan route has disappeared. The compact Brain metadata owns
    # the frozen blocker identity in that case.
    hypothesis.liquidity_route = None
    hypothesis.plan.liquidity_route = None
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    delivery_bar = _bar(0, high=103.5, close=103.0)
    recorder.on_bar(delivery_bar)
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(hypothesis,),
            location=_location(),
            price=delivery_bar.close,
        ),
        source_bar=delivery_bar,
    )
    delivery = next(
        row
        for row in recorder.drain_rows()
        if row.dimension == "delivery_quality"
    )
    assert delivery.outcome_value == 1.0
    assert delivery.hard_barrier_before_target
    assert json.loads(delivery.path_blocker_ids) == ["blocker:authority"]
    assert not delivery.censored
    assert not delivery.fit_eligible
