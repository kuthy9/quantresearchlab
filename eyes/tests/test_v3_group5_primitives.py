from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import pickle
from pathlib import Path

import pandas as pd
import pytest

from shares.tests.legacy_group5 import CausalGroup5Reducer, Group5Protocol
from contract.market import (
    Candle,
    Direction,
    Timeframe,
)
from contract.eye import (
    BOSLifecycle,
    BOSPostBreakState,
    BOSScope,
    BreakOfStructureState,
    EntryLocationLifecycle,
    EventKind,
    FVGQualification,
    FairValueGapLifecycle,
    FairValueGapState,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    ManipulationLifecycle,
    ManipulationState,
    PathSequenceLifecycle,
    QualifiedReacceptanceLifecycle,
)
from eyes.core.observation import CausalObserver, ObserverConfig

from shares.tests.helpers import CORE_TEST_SCALE_SPECS


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/primitives_entry.json"
GROUP12_PROTOCOL_PATH = (
    ROOT / "configs/primitives_structure_liquidity.json"
)
GROUP3_PROTOCOL_PATH = ROOT / "configs/primitives_zones.json"
GROUP4_PROTOCOL_PATH = ROOT / "configs/primitives_range.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


PROTOCOL_SHA = _sha256(PROTOCOL_PATH)
GROUP12_SHA = _sha256(GROUP12_PROTOCOL_PATH)
GROUP3_SHA = _sha256(GROUP3_PROTOCOL_PATH)
GROUP4_SHA = _sha256(GROUP4_PROTOCOL_PATH)
DISPLACEMENT_SHA = "a" * 64
BASE = pd.Timestamp("2025-01-07T09:30:00-05:00")


def _protocol() -> Group5Protocol:
    return Group5Protocol.from_file(PROTOCOL_PATH)


def _m1(
    index: int,
    *,
    open_: float | None = None,
    high: float = 101.25,
    low: float = 100.5,
    close: float = 101.0,
    synthetic: bool = False,
    symbol: str = "NQH5",
    instrument_id: int = 1,
) -> Candle:
    start = BASE + pd.Timedelta(minutes=index)
    return Candle(
        timeframe=Timeframe.M1,
        start=start,
        end=start + pd.Timedelta(minutes=1),
        open=close if open_ is None else open_,
        high=high,
        low=low,
        close=close,
        volume=0.0 if synthetic else 100.0,
        symbol=symbol,
        instrument_id=instrument_id,
        observed_minutes=1,
        expected_minutes=1,
        complete=True,
        real_minutes=0 if synthetic else 1,
        synthetic_minutes=1 if synthetic else 0,
    )


def _fvg(
    confirmed_at: pd.Timestamp,
    *,
    identity: str = "fvg-long",
    qualification: FVGQualification = FVGQualification.DISPLACEMENT_LINKED,
    direction: Direction = Direction.LONG,
    displacement_active_at: pd.Timestamp | None = None,
) -> FairValueGapState:
    starts = (
        confirmed_at - pd.Timedelta(minutes=15),
        confirmed_at - pd.Timedelta(minutes=10),
        confirmed_at - pd.Timedelta(minutes=5),
    )
    return FairValueGapState(
        fvg_id=identity,
        protocol_hash=GROUP3_SHA,
        symbol="NQH5",
        instrument_id=1,
        timeframe=Timeframe.M5,
        direction=direction,
        lifecycle=FairValueGapLifecycle.OPEN,
        qualification=qualification,
        source_displacement_id=(
            f"displacement:{identity}"
            if qualification is FVGQualification.DISPLACEMENT_LINKED
            else None
        ),
        source_active_transition_id=(
            f"active:{identity}"
            if qualification is FVGQualification.DISPLACEMENT_LINKED
            else None
        ),
        source_displacement_protocol_hash=(
            DISPLACEMENT_SHA
            if qualification is FVGQualification.DISPLACEMENT_LINKED
            else None
        ),
        source_displacement_started_at=(
            starts[1]
            if qualification is FVGQualification.DISPLACEMENT_LINKED
            else None
        ),
        source_displacement_active_at=(
            (
                starts[2]
                if displacement_active_at is None
                else displacement_active_at
            )
            if qualification is FVGQualification.DISPLACEMENT_LINKED
            else None
        ),
        source_displacement_prefix_commitment=(
            f"prefix:{identity}"
            if qualification is FVGQualification.DISPLACEMENT_LINKED
            else None
        ),
        source_candle_ids=(
            f"{identity}:one",
            f"{identity}:two",
            f"{identity}:three",
        ),
        source_candle_starts=starts,
        lower_bound=99.0,
        upper_bound=100.0,
        midpoint=99.5,
        invalidation_price=(
            99.0 if direction is Direction.LONG else 100.0
        ),
        width_points=1.0,
        width_ticks=4,
        formation_atr=1.0,
        width_atr=1.0,
        strength=0.8,
        formed_at=confirmed_at,
        confirmed_at=confirmed_at,
        state_started_at=confirmed_at,
        last_updated_at=confirmed_at,
        age_bars=0,
        max_fill_fraction=0.0,
    )


def _invalidated_fvg(
    state: FairValueGapState,
    clock: pd.Timestamp,
) -> FairValueGapState:
    return replace(
        state,
        lifecycle=FairValueGapLifecycle.INVALIDATED,
        state_started_at=clock,
        last_updated_at=clock,
        invalidated_at=clock,
        transition_reason="close_beyond_frozen_far_edge",
    )


def _draw(
    *,
    identity: str,
    side: str,
    kind: str,
    price: float,
    confirmed_at: pd.Timestamp,
) -> LiquidityInventoryItem:
    return LiquidityInventoryItem(
        item_id=identity,
        timeframe=Timeframe.M5,
        side=side,
        kind=kind,
        price=price,
        lower_bound=price,
        upper_bound=price,
        formed_at=confirmed_at - pd.Timedelta(minutes=5),
        confirmed_at=confirmed_at,
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=(f"source:{identity}",),
        age_bars=1,
        strength=0.7,
    )


def _bos(
    *,
    identity: str,
    resolved_at: pd.Timestamp,
    direction: Direction,
) -> BreakOfStructureState:
    target = 102.0 if direction is Direction.LONG else 99.0
    return BreakOfStructureState(
        bos_id=identity,
        timeframe=Timeframe.M1,
        direction=direction,
        lifecycle=BOSLifecycle.CONFIRMED,
        scope=BOSScope.LOCAL,
        target_swing_id=f"swing:{identity}",
        source_structure_id=None,
        target_price=target,
        target_ticks=round(target / 0.25),
        pending_at=resolved_at - pd.Timedelta(minutes=1),
        resolved_at=resolved_at,
        age_bars=1,
        strength=0.8,
        break_bar_id=f"break:{identity}",
        break_distance_atr=0.8,
        post_break_state=BOSPostBreakState.PENDING,
    )


def _manipulation(
    swept_at: pd.Timestamp,
    *,
    identity: str = "manipulation:pool-above",
    close_outside: bool = True,
    lifecycle: ManipulationLifecycle = ManipulationLifecycle.SWEPT,
    resolved_at: pd.Timestamp | None = None,
) -> ManipulationState:
    reaccepted_at = (
        resolved_at
        if lifecycle is ManipulationLifecycle.REACCEPTED
        else None
    )
    accepted_outside_at = (
        resolved_at
        if lifecycle is ManipulationLifecycle.ACCEPTED_OUTSIDE
        else None
    )
    last_updated_at = resolved_at or swept_at
    age = int((last_updated_at - swept_at) / pd.Timedelta(minutes=1))
    return ManipulationState(
        manipulation_id=identity,
        protocol_hash=GROUP4_SHA,
        source_group12_protocol_hash=GROUP12_SHA,
        symbol="NQH5",
        instrument_id=1,
        timeframe=Timeframe.M1,
        lifecycle=lifecycle,
        side="above",
        source_kind="formed_liquidity_pool",
        source_id="pool-above",
        source_protocol_hash=GROUP12_SHA,
        source_timeframe=Timeframe.M1,
        source_inventory_item_id="pool:pool-above",
        source_inventory_lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        coincident_source_ids=(),
        source_formed_at=swept_at - pd.Timedelta(minutes=5),
        source_eligible_at=swept_at - pd.Timedelta(minutes=1),
        source_lower_bound=100.75,
        source_upper_bound=101.0,
        formed_at=swept_at,
        confirmed_at=swept_at,
        swept_at=swept_at,
        reaccepted_at=reaccepted_at,
        accepted_outside_at=accepted_outside_at,
        resolved_at=resolved_at,
        state_started_at=resolved_at or swept_at,
        last_updated_at=last_updated_at,
        sweep_extreme=101.25,
        close_outside_on_sweep=close_outside,
        reentry_price=(
            100.75
            if lifecycle is ManipulationLifecycle.REACCEPTED
            else None
        ),
        resolved_side=(
            "above"
            if lifecycle is ManipulationLifecycle.ACCEPTED_OUTSIDE
            else None
        ),
        outside_completed_bars=(
            2
            if lifecycle is ManipulationLifecycle.ACCEPTED_OUTSIDE
            else 1 if close_outside else 0
        ),
        penetration_atr=0.25,
        strength=0.25,
        age_1m_bars=age,
        transition_reason=(
            "source_swept"
            if lifecycle is ManipulationLifecycle.SWEPT
            else (
                "reentry_held_inside_swept_boundary"
                if lifecycle is ManipulationLifecycle.REACCEPTED
                else "consecutive_closes_held_outside"
            )
        ),
        censored_at=None,
        reentry_candidate_at=(
            resolved_at - pd.Timedelta(minutes=1)
            if (
                lifecycle is ManipulationLifecycle.REACCEPTED
                and resolved_at is not None
            )
            else None
        ),
        reentry_candidate_price=(
            100.75
            if lifecycle is ManipulationLifecycle.REACCEPTED
            else None
        ),
        inside_hold_bars=(
            1 if lifecycle is ManipulationLifecycle.REACCEPTED else 0
        ),
        reentry_failed_at=None,
        outside_run=(
            2
            if lifecycle is ManipulationLifecycle.ACCEPTED_OUTSIDE
            else (
                1
                if (
                    lifecycle is ManipulationLifecycle.SWEPT
                    and close_outside
                )
                else 0
            )
        ),
        outside_run_side=(
            "above"
            if (
                lifecycle is ManipulationLifecycle.ACCEPTED_OUTSIDE
                or (
                    lifecycle is ManipulationLifecycle.SWEPT
                    and close_outside
                )
            )
            else None
        ),
        deadline_at=None,
        deadline_elapsed=False,
        crossed_source_ids=("pool-above",),
    )


def _zone_formation(
    reducer: CausalGroup5Reducer,
    *,
    inventory: tuple[LiquidityInventoryItem, ...] = (),
) -> tuple[Candle, FairValueGapState, object]:
    candle = _m1(0)
    fvg = _fvg(candle.end)
    output = reducer.on_completed_1m(
        candle,
        fair_value_gaps=(fvg,),
        liquidity_inventory=inventory,
        m1_atr=1.0,
    )
    return candle, fvg, output


def test_raw_fvg_is_visible_input_but_not_registered_by_group5() -> None:
    protocol = replace(
        _protocol(),
        source_zone_protocol_hash=GROUP3_SHA,
    )
    reducer = CausalGroup5Reducer(protocol)
    candle = _m1(0)
    raw = _fvg(
        candle.end,
        identity="fvg-raw",
        qualification=FVGQualification.RAW,
    )

    output = reducer.on_completed_1m(
        candle,
        fair_value_gaps=(raw,),
        m1_atr=1.0,
    )

    assert output.entry_locations == ()
    assert output.path_sequences == ()
    assert output.cold_source_ids == (raw.fvg_id,)


def test_group5_protocol_tracks_current_config_and_upstream_bindings() -> None:
    protocol = _protocol()
    payload = json.loads(PROTOCOL_PATH.read_bytes())
    parameters = payload["engineering_parameters"]
    assert payload["authority"] == {
        "typed_state_available": True,
        "brain_input_allowed": True,
        "dfp_lsr_input_authority_validated": True,
        "favr_natural_authority_validated": False,
        "independent_action_authority": False,
        "favr_enabled": False,
    }
    assert _sha256(PROTOCOL_PATH) == PROTOCOL_SHA
    assert protocol.protocol_hash == PROTOCOL_SHA
    assert protocol.source_group12_protocol_hash == GROUP12_SHA
    assert protocol.source_zone_protocol_hash == GROUP3_SHA
    assert protocol.source_range_auction_protocol_hash == GROUP4_SHA
    assert (
        payload["upstream"]["group12_protocol_sha256"]
        == GROUP12_SHA
    )
    assert payload["upstream"]["group3_protocol_sha256"] == GROUP3_SHA
    assert payload["upstream"]["group4_protocol_sha256"] == GROUP4_SHA
    assert protocol.protocol_version == payload["protocol_version"]
    assert protocol.tick_size == payload["tick_size"]
    assert (
        protocol.m1_atr_period,
        protocol.later_hold_bars,
        protocol.maximum_contexts,
        protocol.maximum_steps_per_path,
    ) == (
        parameters["m1_atr_period"],
        parameters["qualified_reacceptance_later_hold_bars"],
        parameters["maximum_context_states"],
        parameters["maximum_steps_per_path"],
    )
    assert isinstance(protocol.protocol_version, str)
    assert protocol.protocol_version
    assert isinstance(protocol.tick_size, (int, float))
    assert not isinstance(protocol.tick_size, bool)
    assert math.isfinite(float(protocol.tick_size))
    assert protocol.tick_size > 0.0
    assert all(
        type(item) is int and item > 0
        for item in (
            protocol.m1_atr_period,
            protocol.later_hold_bars,
            protocol.maximum_contexts,
            protocol.maximum_steps_per_path,
        )
    )


def test_first_pullback_rejection_draw_and_event_memory_are_source_bound() -> None:
    reducer = CausalGroup5Reducer(_protocol())
    formation = _m1(0)
    draw = _draw(
        identity="draw:swing-above",
        side="above",
        kind="swing",
        price=103.0,
        confirmed_at=formation.start,
    )
    parked_range = _draw(
        identity="draw:parked-range",
        side="above",
        kind="range_boundary",
        price=102.0,
        confirmed_at=formation.start,
    )
    fvg = _fvg(formation.end)
    created = reducer.on_completed_1m(
        formation,
        fair_value_gaps=(fvg,),
        liquidity_inventory=(parked_range, draw),
        m1_atr=1.0,
    )
    assert reducer._last_canonical is not None
    created_canonical = reducer._last_canonical

    assert reducer.on_completed_1m(
        formation,
        fair_value_gaps=(fvg,),
        liquidity_inventory=(parked_range, draw),
        m1_atr=1.0,
    ) is created
    assert [
        step.kind for step in created.path_sequences[0].steps
    ] == ["zone_visible", "departure_confirmed"]
    assert created.path_sequences[0].steps[1].same_clock_relation == (
        "same_clock_known"
    )

    pullback = _m1(
        1,
        open_=100.5,
        high=100.75,
        low=99.5,
        close=100.5,
    )
    resolved = reducer.on_completed_1m(
        pullback,
        fair_value_gaps=(fvg,),
        liquidity_inventory=(parked_range, draw),
        m1_atr=1.0,
    )
    assert reducer._last_canonical is not None
    resolved_canonical = reducer._last_canonical
    location = resolved.entry_locations[0]
    path = resolved.path_sequences[0]

    assert location.lifecycle is EntryLocationLifecycle.REJECTED
    assert location.first_entered_at == pullback.end
    assert location.entry_mode == "crossed_near_edge"
    assert location.contact_reference_price == 100.0
    assert location.nearest_visible_draw_id == draw.item_id
    assert location.nearest_visible_draw_distance_points == 2.5
    assert path.lifecycle is PathSequenceLifecycle.ACTIVE
    assert path.transition_reason == "zone_rejection_observed"
    assert [step.kind for step in path.steps] == [
        "zone_visible",
        "departure_confirmed",
        "first_pullback",
        "wick_rejection",
    ]
    assert path.steps[-1].same_clock_relation == "same_clock_known"

    observer = CausalObserver(
        ObserverConfig(scale_specs=CORE_TEST_SCALE_SPECS)
    )
    observer._emitter._record_interaction_events(created_canonical)
    observer._emitter._record_interaction_events(resolved_canonical)
    assert any(
        event.kind is EventKind.ENTRY_PATH_STEP
        for event in observer.memory.recent()
    )
    timeline_key = f"entry_path:{path.sequence_id}"
    observer.memory.sync_retained_entity_timelines(
        (timeline_key,),
        asof=pullback.end,
    )
    timeline = observer.memory.entity_timelines()[
        timeline_key
    ]
    assert [event.lifecycle for event in timeline] == ["active"]

    failure_bar = _m1(
        2,
        open_=100.5,
        high=100.75,
        low=98.5,
        close=98.75,
    )
    failed = reducer.on_completed_1m(
        failure_bar,
        fair_value_gaps=(fvg,),
        liquidity_inventory=(parked_range, draw),
        m1_atr=1.0,
    )
    assert failed.entry_locations[0].lifecycle is EntryLocationLifecycle.LEFT
    assert failed.path_sequences[0].lifecycle is PathSequenceLifecycle.CLOSED
    assert failed.path_sequences[0].transition_reason == "location_left"
    assert reducer._last_canonical is not None
    observer._emitter._record_interaction_events(reducer._last_canonical)
    observer.memory.sync_retained_entity_timelines(
        (timeline_key,),
        asof=failure_bar.end,
    )
    assert [
        event.lifecycle
        for event in observer.memory.entity_timelines()[timeline_key]
    ] == ["active", "closed"]


def test_same_bar_wick_remains_visible_then_later_aligned_bos_closes() -> None:
    reducer = CausalGroup5Reducer(_protocol())
    _, fvg, _ = _zone_formation(reducer)
    pullback = _m1(
        1,
        open_=100.5,
        high=100.75,
        low=99.5,
        close=100.5,
    )
    rejected = reducer.on_completed_1m(
        pullback,
        fair_value_gaps=(fvg,),
        m1_atr=1.0,
    )
    waiting = rejected.path_sequences[0]

    assert waiting.lifecycle is PathSequenceLifecycle.ACTIVE
    assert waiting.transition_reason == "zone_rejection_observed"
    assert waiting.last_updated_at == pullback.end
    assert waiting.steps[-1].kind == "wick_rejection"

    trigger_bar = _m1(
        2,
        open_=100.5,
        high=101.25,
        low=100.25,
        close=101.0,
    )
    aligned = _bos(
        identity="bos:strictly-later-aligned",
        resolved_at=trigger_bar.end,
        direction=Direction.LONG,
    )
    triggered = reducer.on_completed_1m(
        trigger_bar,
        fair_value_gaps=(fvg,),
        m1_bos=(aligned,),
        m1_atr=1.0,
    )
    path = triggered.path_sequences[0]
    reference = triggered.micro_bos_references[0]

    assert path.lifecycle is PathSequenceLifecycle.CLOSED
    assert path.transition_reason == "micro_bos_aligned"
    assert path.ended_at == trigger_bar.end
    assert path.steps[-1].kind == "micro_bos_confirmed"
    assert reference.context_id == triggered.entry_locations[0].location_id
    assert reference.anchor_at == pullback.end
    assert reference.resolved_at == trigger_bar.end
    assert reference.relation == "strictly_after"
    assert reference.outcome == "aligned"
    assert reference.qualified


def test_same_bar_leave_and_eligible_bos_freeze_failure_before_path_close() -> None:
    reducer = CausalGroup5Reducer(_protocol())
    _, fvg, _ = _zone_formation(reducer)

    first_pullback = _m1(
        1,
        open_=100.5,
        high=100.5,
        low=99.5,
        close=100.0,
    )
    waiting = reducer.on_completed_1m(
        first_pullback,
        fair_value_gaps=(fvg,),
        m1_atr=1.0,
    )
    assert waiting.entry_locations[0].lifecycle is (
        EntryLocationLifecycle.IN_ZONE
    )
    assert waiting.path_sequences[0].steps[-1].kind == "first_pullback"

    leave_and_trigger = _m1(
        2,
        open_=100.0,
        high=100.25,
        low=99.5,
        close=99.75,
    )
    resolved = reducer.on_completed_1m(
        leave_and_trigger,
        fair_value_gaps=(fvg,),
        m1_bos=(
            _bos(
                identity="bos:same-clock-as-leave",
                resolved_at=leave_and_trigger.end,
                direction=Direction.LONG,
            ),
        ),
        m1_atr=1.0,
    )
    reacceptance = resolved.qualified_reacceptances[0]
    path = resolved.path_sequences[0]

    assert reacceptance.lifecycle is QualifiedReacceptanceLifecycle.FAILED
    assert reacceptance.left_at == leave_and_trigger.end
    assert reacceptance.failed_at == reacceptance.left_at
    assert reacceptance.transition_reason == "context_closed_before_hold"
    assert path.lifecycle is PathSequenceLifecycle.CLOSED
    assert path.transition_reason == "micro_bos_aligned"
    assert [step.kind for step in path.steps] == [
        "zone_visible",
        "departure_confirmed",
        "first_pullback",
        "reference_left",
        "micro_bos_confirmed",
        "reacceptance_failed",
    ]
    assert [step.same_clock_relation for step in path.steps[-3:]] == [
        "strictly_after",
        "same_clock_unknown",
        "same_clock_known",
    ]
    assert path.ended_at == leave_and_trigger.end
    with pytest.raises(
        ValueError,
        match="reacceptance clocks are invalid",
    ):
        replace(
            reacceptance,
            transition_reason="reclaim_lost_before_hold",
        )


def test_reacceptance_requires_leave_reclaim_and_a_later_real_hold() -> None:
    reducer = CausalGroup5Reducer(_protocol())
    _, fvg, created = _zone_formation(reducer)
    initial_location = created.entry_locations[0]

    synthetic = reducer.on_completed_1m(
        _m1(
            1,
            open_=100.5,
            high=100.75,
            low=99.5,
            close=99.75,
            synthetic=True,
        ),
        fair_value_gaps=(fvg,),
        m1_atr=1.0,
    )
    assert synthetic.entry_locations[0] == initial_location

    leave = _m1(
        2,
        open_=100.5,
        high=100.75,
        low=99.5,
        close=99.75,
    )
    left = reducer.on_completed_1m(
        leave,
        fair_value_gaps=(fvg,),
        m1_atr=1.0,
    )
    reacceptance = left.qualified_reacceptances[0]
    path = left.path_sequences[0]
    assert reacceptance.lifecycle is QualifiedReacceptanceLifecycle.LEFT
    assert path.steps[-2].kind == "first_pullback"
    assert path.steps[-1].kind == "reference_left"
    assert path.steps[-1].same_clock_relation == "same_clock_known"

    reclaim = _m1(
        3,
        open_=99.75,
        high=100.5,
        low=99.5,
        close=100.25,
    )
    reclaimed = reducer.on_completed_1m(
        reclaim,
        fair_value_gaps=(fvg,),
        m1_atr=1.0,
    )
    assert (
        reclaimed.qualified_reacceptances[0].lifecycle
        is QualifiedReacceptanceLifecycle.RECLAIMED
    )
    assert reclaimed.qualified_reacceptances[0].hold_real_1m_bars == 0

    hold = _m1(
        4,
        open_=100.25,
        high=100.75,
        low=100.0,
        close=100.5,
    )
    held = reducer.on_completed_1m(
        hold,
        fair_value_gaps=(fvg,),
        m1_atr=1.0,
    )
    final_reacceptance = held.qualified_reacceptances[0]
    assert final_reacceptance.lifecycle is QualifiedReacceptanceLifecycle.HELD
    assert final_reacceptance.reclaimed_at == reclaim.end
    assert final_reacceptance.held_at == hold.end
    assert final_reacceptance.hold_real_1m_bars == 1
    assert (
        held.entry_locations[0].transition_reason
        == "hold_completed"
    )
    assert held.path_sequences[0].lifecycle is PathSequenceLifecycle.ACTIVE
    assert held.path_sequences[0].transition_reason == (
        "qualified_reacceptance_held"
    )

    trigger_bar = _m1(
        5,
        open_=100.5,
        high=101.25,
        low=100.25,
        close=101.0,
    )
    triggered = reducer.on_completed_1m(
        trigger_bar,
        fair_value_gaps=(fvg,),
        m1_bos=(
            _bos(
                identity="bos:after-held",
                resolved_at=trigger_bar.end,
                direction=Direction.LONG,
            ),
        ),
        m1_atr=1.0,
    )
    assert triggered.path_sequences[0].lifecycle is PathSequenceLifecycle.CLOSED
    assert triggered.path_sequences[0].transition_reason == "micro_bos_aligned"


def test_typed_zone_failure_precedes_derived_reacceptance_failure() -> None:
    reducer = CausalGroup5Reducer(_protocol())
    _, fvg, _ = _zone_formation(reducer)
    leave = _m1(
        1,
        open_=100.5,
        high=100.75,
        low=99.5,
        close=99.75,
    )
    reducer.on_completed_1m(
        leave,
        fair_value_gaps=(fvg,),
        m1_atr=1.0,
    )

    failure_bar = _m1(
        2,
        open_=99.75,
        high=100.0,
        low=99.25,
        close=99.5,
    )
    failed = reducer.on_completed_1m(
        failure_bar,
        fair_value_gaps=(
            _invalidated_fvg(fvg, failure_bar.end),
        ),
        m1_atr=1.0,
    )
    path = failed.path_sequences[0]
    location = failed.entry_locations[0]
    reacceptance = failed.qualified_reacceptances[0]

    assert location.lifecycle is EntryLocationLifecycle.LEFT
    assert location.transition_reason == "fvg_invalidated"
    assert reacceptance.lifecycle is QualifiedReacceptanceLifecycle.FAILED
    assert reacceptance.transition_reason == "source_invalidated"
    assert [step.kind for step in path.steps[-2:]] == [
        "location_left",
        "reacceptance_failed",
    ]
    assert path.steps[-2].source_event_id == fvg.fvg_id
    assert path.steps[-1].same_clock_relation == "same_clock_known"
    assert path.transition_reason == "location_left"


def test_pool_reaccepted_clock_is_latched_and_old_bos_cannot_be_backfilled() -> None:
    reducer = CausalGroup5Reducer(_protocol())
    sweep_bar = _m1(
        0,
        open_=101.0,
        high=101.25,
        low=100.75,
        close=101.25,
    )
    swept = _manipulation(sweep_bar.end)
    reducer.on_completed_1m(
        sweep_bar,
        manipulations=(swept,),
        m1_atr=1.0,
    )
    reducer.on_completed_1m(
        _m1(1),
        manipulations=(swept,),
        m1_atr=1.0,
    )

    source_clock = _m1(2).end
    returned = _manipulation(
        sweep_bar.end,
        lifecycle=ManipulationLifecycle.REACCEPTED,
        resolved_at=source_clock,
    )
    first_bos = _bos(
        identity="bos:first",
        resolved_at=source_clock,
        direction=Direction.SHORT,
    )
    equality_bar = _m1(
        2,
        open_=101.0,
        high=101.25,
        low=100.75,
        close=101.0,
    )
    waiting = reducer.on_completed_1m(
        equality_bar,
        manipulations=(returned,),
        m1_bos=(first_bos,),
        m1_atr=1.0,
    )
    assert waiting.qualified_reacceptances == ()
    assert waiting.path_sequences[0].steps[-1].kind == "reacceptance_held"
    assert waiting.path_sequences[0].lifecycle is PathSequenceLifecycle.ACTIVE

    displacement_bar = _m1(
        3,
        open_=101.0,
        high=101.0,
        low=100.5,
        close=100.75,
    )
    reverse_zone = _fvg(
        displacement_bar.end,
        identity="fvg:pool-reverse-short",
        direction=Direction.SHORT,
        displacement_active_at=displacement_bar.end,
    )
    displaced = reducer.on_completed_1m(
        displacement_bar,
        fair_value_gaps=(reverse_zone,),
        manipulations=(),
        m1_atr=1.0,
    )
    assert displaced.path_sequences[0].steps[-1].kind == (
        "opposite_displacement"
    )
    later_trigger = _m1(
        4,
        open_=100.75,
        high=101.0,
        low=100.0,
        close=100.25,
    )
    triggered = reducer.on_completed_1m(
        later_trigger,
        fair_value_gaps=(reverse_zone,),
        manipulations=(),
        m1_bos=(
            _bos(
                identity="bos:strict-after-reacceptance",
                resolved_at=later_trigger.end,
                direction=Direction.SHORT,
            ),
        ),
        m1_atr=1.0,
    )
    assert triggered.path_sequences[0].lifecycle is (
        PathSequenceLifecycle.CLOSED
    )
    assert triggered.path_sequences[0].transition_reason == (
        "pool_reversal_sequence_observed"
    )


def test_pool_opposite_displacement_freezes_root_not_entry_zone() -> None:
    sweep_bar = _m1(
        0,
        open_=101.0,
        high=101.25,
        low=100.75,
        close=101.25,
    )
    swept = _manipulation(sweep_bar.end)
    reaccepted_at = _m1(2).end
    returned = _manipulation(
        sweep_bar.end,
        lifecycle=ManipulationLifecycle.REACCEPTED,
        resolved_at=reaccepted_at,
    )
    displacement_active_at = _m1(3).end
    zone_confirmed_at = _m1(4).end
    pre_visibility_bos = _bos(
        identity="bos:before-zone-visible",
        resolved_at=displacement_active_at,
        direction=Direction.SHORT,
    )
    first_zone = _fvg(
        zone_confirmed_at,
        identity="fvg:pool-reverse-short:a",
        direction=Direction.SHORT,
        displacement_active_at=displacement_active_at,
    )
    second_zone = replace(
        _fvg(
            zone_confirmed_at,
            identity="fvg:pool-reverse-short:b",
            direction=Direction.SHORT,
            displacement_active_at=displacement_active_at,
        ),
        source_displacement_id=first_zone.source_displacement_id,
        source_active_transition_id=(
            first_zone.source_active_transition_id
        ),
        source_displacement_started_at=(
            first_zone.source_displacement_started_at
        ),
        source_displacement_prefix_commitment=(
            first_zone.source_displacement_prefix_commitment
        ),
    )

    frozen_steps = []
    for zones in (
        (first_zone,),
        (second_zone,),
        (first_zone, second_zone),
    ):
        reducer = CausalGroup5Reducer(_protocol())
        reducer.on_completed_1m(
            sweep_bar,
            manipulations=(swept,),
            m1_atr=1.0,
        )
        reducer.on_completed_1m(
            _m1(1),
            manipulations=(swept,),
            m1_atr=1.0,
        )
        reducer.on_completed_1m(
            _m1(2),
            manipulations=(returned,),
            m1_atr=1.0,
        )
        reducer.on_completed_1m(
            _m1(3),
            m1_bos=(pre_visibility_bos,),
            m1_atr=1.0,
        )
        output = reducer.on_completed_1m(
            _m1(4),
            fair_value_gaps=zones,
            m1_bos=(pre_visibility_bos,),
            m1_atr=1.0,
        )
        assert {
            location.source_zone_id for location in output.entry_locations
        } == {zone.fvg_id for zone in zones}
        assert all(
            reference.bos_id != pre_visibility_bos.bos_id
            for reference in output.micro_bos_references
        )
        pool_path = next(
            path
            for path in output.path_sequences
            if path.context_kind == "pool_reversal"
        )
        assert pool_path.lifecycle is PathSequenceLifecycle.ACTIVE
        frozen_steps.append(
            next(
                step
                for step in pool_path.steps
                if step.kind == "opposite_displacement"
            )
        )

    first_step, second_step, combined_step = frozen_steps
    assert first_step == second_step == combined_step
    assert first_step.source_event_id == returned.manipulation_id
    assert first_step.source_entity_id == first_zone.source_displacement_id
    assert first_step.observed_at == zone_confirmed_at
    assert first_step.source_active_at == displacement_active_at
    assert reaccepted_at < first_step.source_active_at <= first_step.observed_at
    assert first_step.direction is Direction.SHORT
    assert first_step.reason == "opposite_displacement_after_reacceptance"
    assert first_step.source_event_id not in {
        first_zone.fvg_id,
        second_zone.fvg_id,
    }
    assert pickle.loads(pickle.dumps(first_step)) == first_step
    with pytest.raises(ValueError, match="path-sequence step is invalid"):
        replace(first_step, source_active_at=None)
    with pytest.raises(ValueError, match="path-sequence step is invalid"):
        replace(
            first_step,
            source_active_at=zone_confirmed_at + pd.Timedelta(minutes=1),
        )


def test_pool_reacceptance_accepts_atr_derived_off_tick_zone_boundary() -> None:
    reducer = CausalGroup5Reducer(_protocol())
    sweep_bar = _m1(
        0,
        open_=101.0,
        high=101.25,
        low=100.75,
        close=101.25,
    )
    swept = replace(
        _manipulation(sweep_bar.end),
        source_lower_bound=100.6875,
        source_upper_bound=101.0625,
        penetration_atr=0.1875,
        strength=0.1875,
    )

    output = reducer.on_completed_1m(
        sweep_bar,
        manipulations=(swept,),
        m1_atr=1.0,
    )
    assert output.qualified_reacceptances == ()
    reducer.on_completed_1m(
        _m1(1),
        manipulations=(swept,),
        m1_atr=1.0,
    )
    returned = replace(
        _manipulation(
            sweep_bar.end,
            lifecycle=ManipulationLifecycle.REACCEPTED,
            resolved_at=_m1(2).end,
        ),
        source_lower_bound=100.6875,
        source_upper_bound=101.0625,
        penetration_atr=0.1875,
        strength=0.1875,
    )
    held = reducer.on_completed_1m(
        _m1(2),
        manipulations=(returned,),
        m1_atr=1.0,
    )
    assert held.qualified_reacceptances == ()
    assert held.path_sequences[0].steps[-1].kind == "reacceptance_held"
    assert held.path_sequences[0].steps[-1].source_entity_id == (
        returned.manipulation_id
    )
    assert (
        reducer.on_completed_1m(
            _m1(2),
            manipulations=(returned,),
            m1_atr=1.0,
        )
        is held
    )


def test_sweep_clock_bos_is_not_a_post_reacceptance_trigger() -> None:
    reducer = CausalGroup5Reducer(_protocol())
    sweep_bar = _m1(
        0,
        open_=101.0,
        high=101.25,
        low=100.5,
        close=100.75,
    )
    swept = _manipulation(
        sweep_bar.end,
        close_outside=False,
    )
    bos_z = _bos(
        identity="bos:z",
        resolved_at=sweep_bar.end,
        direction=Direction.LONG,
    )
    bos_a = _bos(
        identity="bos:a",
        resolved_at=sweep_bar.end,
        direction=Direction.SHORT,
    )

    output = reducer.on_completed_1m(
        sweep_bar,
        manipulations=(swept,),
        m1_bos=(bos_z, bos_a),
        m1_atr=1.0,
    )
    path = output.path_sequences[0]
    assert path.lifecycle is PathSequenceLifecycle.ACTIVE
    assert [step.kind for step in path.steps] == ["pool_swept"]
    assert output.micro_bos_references == ()
    assert output.qualified_reacceptances == ()


def test_hard_boundary_censors_old_epoch_with_reason_aware_identity() -> None:
    reducer = CausalGroup5Reducer(_protocol())
    _, fvg, _ = _zone_formation(reducer)
    leave = _m1(
        1,
        open_=100.5,
        high=100.75,
        low=99.5,
        close=99.75,
    )
    reducer.on_completed_1m(
        leave,
        fair_value_gaps=(fvg,),
        m1_atr=1.0,
    )
    boundary_at = _m1(2).end

    boundary = reducer.on_boundary(
        "contract_change_reset",
        boundary_at,
        symbol="NQH5",
        instrument_id=2,
    )
    assert reducer.on_boundary(
        "contract_change_reset",
        boundary_at,
        symbol="NQH5",
        instrument_id=2,
    ) is boundary
    assert boundary.entry_locations == ()
    assert boundary.path_sequences == ()
    assert len(boundary.path_transitions) == 1
    assert (
        boundary.path_transitions[0].lifecycle
        is PathSequenceLifecycle.CENSORED
    )
    assert boundary.path_transitions[0].instrument_id == 1
    assert (
        boundary.reacceptance_transitions[0].lifecycle
        is QualifiedReacceptanceLifecycle.CENSORED
    )
    assert boundary.reacceptance_transitions[0].instrument_id == 1
    assert reducer.snapshot().path_sequences == ()

    invalid = CausalGroup5Reducer(_protocol())
    _zone_formation(invalid)
    before = invalid.snapshot()
    with pytest.raises(
        ValueError,
        match="boundary identity contradicts its reason",
    ):
        invalid.on_boundary(
            "contract_change_reset",
            _m1(1).end,
            symbol="NQH5",
            instrument_id=1,
        )
    assert invalid.snapshot() == before


def test_cold_start_diagnostic_is_current_and_does_not_retain_history() -> None:
    reducer = CausalGroup5Reducer(_protocol())
    stale = _fvg(_m1(-2).end, identity="fvg-stale")

    first = reducer.on_completed_1m(
        _m1(0),
        fair_value_gaps=(stale,),
        m1_atr=1.0,
    )
    second = reducer.on_completed_1m(
        _m1(1),
        fair_value_gaps=(stale,),
        m1_atr=1.0,
    )
    empty = reducer.on_completed_1m(
        _m1(2),
        fair_value_gaps=(),
        m1_atr=1.0,
    )

    assert first.path_sequences == ()
    assert first.cold_source_ids == ("fvg-stale",)
    assert second.cold_source_ids == ("fvg-stale",)
    assert empty.cold_source_ids == ()
