from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from smc_trader.brain_calibration import BrainCalibrationRecorder
from smc_trader.model import (
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
        path_blocker_ids=("blocker-1",),
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
    readiness_value = (
        0.8 if complete else 0.0
    ) if readiness is None else readiness
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
        episode_id=setup_id if episode_id is None else episode_id,
        context_id=context_id,
        evidence_revision_id=revision,
        entry_location_id="location-1",
        invalidation=invalidation,
        draw_selection=draw,
        deliverable_targets=(target,),
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
    )


def _snapshot(
    minute: int = 0,
    *,
    hypotheses: tuple[SimpleNamespace, ...] | None = None,
    location: SimpleNamespace | None = None,
    anomalies: tuple[str, ...] = (),
    price: float = 100.0,
) -> SimpleNamespace:
    hypotheses = hypotheses or (_hypothesis(),)
    observation = SimpleNamespace(
        asof=_clock(minute),
        symbol="NQH5",
        instrument_id=1,
        price=price,
        anomalies=anomalies,
        entry_locations=(() if location is None else (location,)),
        path_sequences=(),
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
    )
    return SimpleNamespace(
        observation=observation,
        belief=belief,
    )


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
    assert {row.scene_hypothesis_id for row in immediate} == {
        f"scene:{Playbook.DISPLACEMENT_FIRST_PULLBACK.value}:long"
    }
    assert all(row.liquidity_route_id == "route-1" for row in immediate)
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
    # The only new row is the legitimately resolved readiness target; the
    # unchanged evidence/phase did not create another registration.
    assert [row.dimension for row in later] == ["entry_readiness"]
    assert {item.dimension for item in recorder.open_samples} == {
        "thesis_strength",
        "location_quality",
        "delivery_quality",
    }


def test_recorder_uses_candidates_and_preserves_competing_scene_context() -> None:
    recorder = _recorder()
    snapshot = _snapshot(location=_location())
    hypothesis = snapshot.belief.candidates()[0]
    alternate_id = f"scene-alt:{hypothesis.key}"
    snapshot.belief.context_hypotheses[alternate_id] = SimpleNamespace(
        hypothesis_id=alternate_id,
        playbook=hypothesis.playbook,
        direction=hypothesis.direction,
        context_root_ids=("alternate-root",),
    )
    snapshot.belief.hypotheses = {}

    recorder.observe(snapshot, source_bar=_bar(-1))

    rows = recorder.drain_rows()
    assert rows
    assert all(
        json.loads(row.competing_scene_hypothesis_ids) == [alternate_id]
        for row in rows
    )
    assert all(row.hypothesis_key == hypothesis.key for row in rows)


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

    recorder.on_bar(_bar(0, high=104.0, low=98.5, close=103.5))
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


def test_dfp_thesis_does_not_inherit_entry_zone_invalidation() -> None:
    recorder = _recorder()
    recorder.observe(
        _snapshot(location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    recorder.on_bar(_bar(0, high=100.5, low=98.5, close=99.5))
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


def test_readiness_uses_only_the_next_completed_bar() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis(phase=PlaybookPhase.WAITING_TRIGGER)
    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )
    recorder.drain_rows()

    recorder.on_bar(_bar(0, high=100.8, close=100.75))
    rows = recorder.drain_rows()
    readiness = [row for row in rows if row.dimension == "entry_readiness"]
    assert len(readiness) == 1
    assert readiness[0].outcome_value == 1.0
    assert readiness[0].resolution == "next_close_advanced_toward_draw"
    assert {item.dimension for item in recorder.open_samples} == {
        "thesis_strength",
        "location_quality",
        "delivery_quality",
    }


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
    assert not any(
        row.resolution == "setup_replaced_without_matching_terminal"
        for row in transition_rows
    )
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
    assert {sample.setup_id for sample in thesis_revisions} == {context_id}


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

    recorder.on_bar(_bar(0, high=100.8, close=100.75))
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

    recorder.on_bar(_bar(1, high=103.5, close=103.0))
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

    recorder.on_bar(_bar(0, high=100.8, close=100.75))
    readiness = recorder.drain_rows()
    assert [row.dimension for row in readiness] == ["entry_readiness"]

    # Typed Brain retains a terminal episode until a causally newer setup
    # replaces it.  Seeing that closure again must not resolve its frozen
    # delivery target.
    recorder.observe(
        _snapshot(
            1,
            hypotheses=(terminal,),
            location=_location(),
            price=100.75,
        ),
        source_bar=_bar(0, high=100.8, close=100.75),
    )
    assert recorder.drain_rows() == ()
    assert [sample.dimension for sample in recorder.open_samples] == [
        "delivery_quality"
    ]

    recorder.on_bar(_bar(1, high=103.5, close=103.0))
    delivered = recorder.drain_rows()
    assert [row.dimension for row in delivered] == ["delivery_quality"]
    assert delivered[0].outcome_value == 1.0
    assert delivered[0].resolution == "draw_delivered"


def test_non_top_old_episode_terminal_resolves_matching_lsr_thesis() -> None:
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
        context_id="new-current-episode",
        terminal_reason="micro_bos_opposed_or_ambiguous",
        terminal_source_ids=(old_setup, "micro-bos-event"),
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
    assert len(thesis) == 1
    assert thesis[0].resolved_at == _clock(1)
    assert thesis[0].outcome_value == 0.0
    assert not thesis[0].censored
    assert thesis[0].resolution == (
        "thesis_contradicted:micro_bos_opposed_or_ambiguous"
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
        terminal_reason="trigger_opposed_or_ambiguous",
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


def test_non_future_deadline_is_censored_not_labelled_success() -> None:
    recorder = _recorder()
    hypothesis = _hypothesis()
    hypothesis.episode_deadline = _clock(0)
    hypothesis.plan.deadline = _clock(0)

    recorder.observe(
        _snapshot(hypotheses=(hypothesis,), location=_location()),
        source_bar=_bar(-1),
    )

    fitted = [
        row
        for row in recorder.drain_rows()
        if row.dimension
        in {
            "thesis_strength",
            "location_quality",
            "entry_readiness",
            "delivery_quality",
        }
    ]
    assert {row.dimension for row in fitted} == {
        "thesis_strength",
        "location_quality",
        "entry_readiness",
        "delivery_quality",
    }
    assert all(row.outcome_value is None for row in fitted)
    assert all(row.censored and not row.fit_eligible for row in fitted)
    assert all(row.resolution == "non_future_deadline" for row in fitted)


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
        row.dimension == "delivery_quality"
        and row.resolution == "setup_replaced_without_matching_terminal"
        for row in rows
    )
    delivery = [
        sample
        for sample in recorder.open_samples
        if sample.dimension == "delivery_quality"
    ]
    assert len(delivery) == 1
    assert delivery[0].setup_id == "triggered-setup"

    recorder.on_bar(_bar(1, high=103.5, close=103.0))
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
        "trigger_opposed_or_ambiguous",
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


def test_missing_draw_does_not_consume_thesis_revision_identity() -> None:
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
    assert any(
        item.dimension == "thesis_strength"
        for item in recorder.open_samples
    )


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
