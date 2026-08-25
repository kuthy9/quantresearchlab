from __future__ import annotations

from dataclasses import replace
from decimal import localcontext
import pickle

import pandas as pd
import pytest

from smc_trader.foundation_adapter import (
    FOUNDATION_ADAPTER_CHECKPOINT_SCHEMA_VERSION,
    CanonicalFoundationAdapter,
    _checkpoint_payload,
)
from smc_trader.market_state import (
    DeliveryPhase,
    RelationRole,
    RelationState,
    SwingGeometryNode,
    _validate_foundation_authoritative_sources,
)
from smc_trader.model import (
    Direction,
    EventKind,
    EventOrigin,
    MarketEvent,
    Timeframe,
    content_hash,
)
from smc_trader.semantic_lifecycle import (
    GenerationLifecycle,
    LiquidityInteractionLifecycle,
    LiquidityInteractionTerminal,
    LiquidityLevelLifecycle,
    NormalizedLifecycleTransition,
    NormalizedTransitionKind,
    StructureGenerationLifecycle,
    StructureScope,
    StructureTransitionLifecycle,
)
from smc_trader.semantic_zones import FVGStructuralLifecycle
from smc_trader.semantic_foundation import (
    FoundationObjectType,
    FoundationProjectionReducer,
)


TZ = "America/New_York"
TICK = 0.25


def _clock(minutes: int) -> pd.Timestamp:
    return pd.Timestamp("2024-06-03 09:30", tz=TZ) + pd.Timedelta(
        minutes, unit="m"
    )


def _bar(
    minutes: int,
    *,
    sequence: int = 0,
    timeframe: Timeframe = Timeframe.M1,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.0,
    event_id: str | None = None,
    real_completed: bool = True,
) -> MarketEvent:
    identity = event_id or f"bar:{timeframe.value}:{minutes}"
    event = MarketEvent(
        event_id=identity,
        kind=EventKind.BAR_COMPLETED,
        observed_at=_clock(minutes),
        timeframe=timeframe,
        side=None,
        price=close,
        strength=0.0,
        sequence_no=sequence,
        event_time=_clock(minutes),
        known_at=_clock(minutes),
        evidence={
            "high": high,
            "low": low,
            "close": close,
            "real_completed": real_completed,
            "clock_only": not real_completed,
        },
        source_data_ids=(f"data:{identity}",),
        origin=EventOrigin.NORMALIZED_DATA,
    )
    return event if real_completed else _with_clock_only_coverage(event)


def _with_clock_only_coverage(event: MarketEvent) -> MarketEvent:
    timeframe_minutes = {
        Timeframe.M1: 1,
        Timeframe.M5: 5,
        Timeframe.M15: 15,
        Timeframe.H1: 60,
        Timeframe.H4: 240,
    }[event.timeframe]
    evidence = {
        **dict(event.evidence),
        "complete": True,
        "start": event.known_at
        - pd.Timedelta(timeframe_minutes, unit="min"),
        "observed_minutes": timeframe_minutes,
        "expected_minutes": timeframe_minutes,
        "real_minutes": timeframe_minutes - 1,
        "synthetic_minutes": 1,
    }
    return replace(event, details=evidence, evidence=evidence)


def _atomic(
    event_id: str,
    kind: EventKind,
    minutes: int,
    sequence: int,
    *,
    timeframe: Timeframe = Timeframe.M1,
    source_event_ids: tuple[str, ...] = (),
    source_entity_ids: tuple[str, ...] = (),
    context_event_ids: tuple[str, ...] = (),
    evidence: dict[str, object] | None = None,
    side: str | None = None,
    price: float | None = None,
    direction: Direction | None = None,
    zone: tuple[float, float] | None = None,
) -> MarketEvent:
    return MarketEvent(
        event_id=event_id,
        kind=kind,
        observed_at=_clock(minutes),
        timeframe=timeframe,
        side=side,
        price=price,
        strength=0.5,
        sequence_no=sequence,
        event_time=_clock(minutes),
        known_at=_clock(minutes),
        evidence=evidence or {},
        direction=direction,
        zone=zone,
        source_event_ids=source_event_ids,
        source_entity_ids=source_entity_ids,
        context_event_ids=context_event_ids,
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )


def _consume(adapter: CanonicalFoundationAdapter, events) -> None:
    for event in events:
        adapter.consume(event)


def _structure_confirmation(
    facts: dict[str, MarketEvent],
    *,
    label: str,
    minute: int,
    direction: Direction,
    candidate_protected_swing_id: str | None = None,
    sequence: int = 1,
    event_id: str | None = None,
    structure_id: str | None = None,
) -> MarketEvent:
    high = facts["high"]
    low = facts["low"]
    high_id = str(high.evidence["source_entity_id"])
    low_id = str(low.evidence["source_entity_id"])
    identity = structure_id or f"structure-source:{label}"
    evidence: dict[str, object] = {"structure_id": identity}
    if candidate_protected_swing_id is not None:
        evidence.update(
            {
                "source_high_id": high_id,
                "source_low_id": low_id,
                "candidate_protected_swing_id": candidate_protected_swing_id,
            }
        )
    return _atomic(
        event_id or f"structure:{label}",
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        minute,
        sequence,
        timeframe=Timeframe.H1,
        source_event_ids=(high.event_id, low.event_id),
        evidence=evidence,
        direction=direction,
    )


def _mss_triplet(
    *,
    label: str,
    minute: int,
    direction: Direction,
    structure_event: MarketEvent,
    structure_id: str,
    bos_id: str | None = None,
    bar_event_id: str | None = None,
    raw_event_id: str | None = None,
    mss_event_id: str | None = None,
    raw_direction: Direction | None = None,
) -> tuple[MarketEvent, MarketEvent, MarketEvent]:
    bar = _bar(minute, event_id=bar_event_id)
    raw = _atomic(
        raw_event_id or f"raw:{label}",
        EventKind.RAW_BOUNDARY_BREAK,
        minute,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bar.event_id,),
        direction=raw_direction,
    )
    resolved_bos_id = bos_id or f"bos-source:{label}"
    mss = _atomic(
        mss_event_id or f"mss:{label}",
        EventKind.MSS_CORE_CONFIRMED,
        minute,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(raw.event_id, structure_event.event_id),
        source_entity_ids=(resolved_bos_id, structure_id),
        evidence={"bos_id": resolved_bos_id},
        direction=direction,
    )
    return bar, raw, mss


def _assert_adapter_replay(
    events: tuple[MarketEvent, ...],
    adapter: CanonicalFoundationAdapter,
) -> None:
    replayed = CanonicalFoundationAdapter.replay(events, tick_size=TICK)
    assert replayed.lifecycle == adapter.lifecycle
    assert replayed.projection == adapter.projection


def test_formed_pool_half_tick_midpoint_uses_frozen_near_side_anchor() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _seed_level(
        adapter,
        label="formed-half-tick",
        source_kind="formed_liquidity_pool",
        side="above",
        price=20_022.375,
        zone=(20_022.0, 20_022.75),
    )

    level = adapter.lifecycle.levels[-1]
    assert level.price_ticks == 80_088
    assert level.lower_bound_ticks == 80_088
    assert level.upper_bound_ticks == 80_091
    assert (
        level.price_anchor_rule
        == "near_side_tradable_zone_boundary_for_nontradable_midpoint"
    )
    projected_level = next(
        record
        for record in adapter.projection.latest_records
        if record.object_type.value == "liquidity_level"
    )
    assert projected_level.payload["price_anchor_rule"] == (
        "near_side_tradable_zone_boundary_for_nontradable_midpoint"
    )


@pytest.mark.parametrize(
    ("source_kind", "price"),
    (
        ("formed_liquidity_pool", 20_022.38),
        ("confirmed_swing", 20_022.375),
    ),
)
def test_nonregistered_or_inexact_off_grid_pool_anchor_fails_closed(
    source_kind: str,
    price: float,
) -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    with pytest.raises(ValueError, match="exact formed-pool midpoint"):
        _seed_level(
            adapter,
            label=f"invalid-anchor-{source_kind}",
            source_kind=source_kind,
            side="above",
            price=price,
            zone=(20_022.0, 20_022.75),
        )


def test_stage_batch_forks_only_mutable_transaction_containers() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    bar = _bar(0)
    source = _atomic(
        "source:staged-level",
        EventKind.SWING_CONFIRMED,
        0,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bar.event_id,),
        evidence={"source_entity_id": "swing:staged-level"},
        side="above",
        price=100.0,
        direction=Direction.LONG,
    )
    adapter.consume_batch((bar, source))
    prior_lifecycle = adapter.lifecycle
    prior_projection = adapter.projection

    empty_candidate, empty_updates = adapter.stage_batch(())
    assert empty_updates == ()
    assert empty_candidate.lifecycle is prior_lifecycle
    assert empty_candidate.projection is prior_projection
    assert empty_candidate._seen_event_fingerprints is not (
        adapter._seen_event_fingerprints
    )
    assert empty_candidate._real_bars is not adapter._real_bars

    level = _atomic(
        "level-created:staged-level",
        EventKind.LIQUIDITY_LEVEL_CREATED,
        0,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(source.event_id,),
        evidence={
            "level_id": "staged-level",
            "source_kind": "confirmed_swing",
        },
        side="above",
        price=100.0,
    )
    candidate, updates = adapter.stage_batch((level,))

    assert len(updates) == 1
    assert updates[0].records
    assert adapter.lifecycle is prior_lifecycle
    assert adapter.projection is prior_projection
    assert level.event_id not in adapter.known_input_event_ids
    assert level.event_id in candidate.known_input_event_ids
    assert candidate.lifecycle is not prior_lifecycle
    assert candidate.projection is not prior_projection


def test_single_input_identity_query_matches_public_snapshot() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    bar = _bar(0)
    adapter.consume(bar)

    assert adapter.is_known_input_event_id(bar.event_id) is True
    assert adapter.is_known_input_event_id("missing:event") is False
    assert adapter.known_input_event_ids == frozenset({bar.event_id})
    with pytest.raises(ValueError, match="identity must be non-empty"):
        adapter.is_known_input_event_id("")


def test_staged_candidate_rejects_nested_batch_without_destroying_state() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    candidate, updates = adapter.stage_batch(())
    assert updates == ()
    before = candidate.checkpoint()

    with pytest.raises(ValueError, match="nested foundation batch"):
        candidate.consume_batch(())
    with pytest.raises(ValueError, match="nested foundation batch"):
        candidate.consume_batch((_bar(0),))

    assert candidate.checkpoint() == before
    assert candidate._staged_transaction_open is True
    assert adapter.known_input_event_ids == frozenset()


def _seed_level(
    adapter: CanonicalFoundationAdapter,
    *,
    label: str = "level-a",
    minute: int = 0,
    source_kind: str = "confirmed_swing",
    source_timeframe: Timeframe = Timeframe.H1,
    side: str = "above",
    price: float = 100.0,
    zone: tuple[float, float] | None = None,
) -> tuple[tuple[MarketEvent, ...], MarketEvent]:
    bar = _bar(minute)
    source = _atomic(
        f"source:{label}",
        EventKind.SWING_CONFIRMED,
        minute,
        1,
        timeframe=source_timeframe,
        source_event_ids=(bar.event_id,),
        evidence={"source_entity_id": f"swing:{label}"},
        side=side,
        price=price,
        direction=Direction.LONG if side == "above" else Direction.SHORT,
    )
    level = _atomic(
        f"level-created:{label}",
        EventKind.LIQUIDITY_LEVEL_CREATED,
        minute,
        2,
        timeframe=source_timeframe,
        source_event_ids=(source.event_id,),
        evidence={"level_id": label, "source_kind": source_kind},
        side=side,
        price=price,
        zone=zone,
    )
    events = (bar, source, level)
    _consume(adapter, events)
    return events, level


def _cross_level(
    adapter: CanonicalFoundationAdapter,
    level: MarketEvent,
    *,
    crossed_minute: int,
    resolved_minute: int,
    terminal_kind: EventKind,
    protected_swing_id: str | None = None,
    protected_assignment_event_id: str | None = None,
    source_timeframe: Timeframe | None = None,
    crossing_close: float = 99.75,
    resolution_high: float = 101.0,
    intermediate_closes: dict[int, float] | None = None,
    terminal_evidence: dict[str, object] | None = None,
) -> tuple[tuple[MarketEvent, ...], MarketEvent]:
    level_id = str(level.evidence["level_id"])
    crossing_bar = _bar(
        crossed_minute,
        high=101.0,
        low=99.0,
        close=crossing_close,
    )
    touch = _atomic(
        f"touch:{level_id}:{crossed_minute}",
        EventKind.LEVEL_TOUCHED,
        crossed_minute,
        1,
        source_event_ids=(level.event_id, crossing_bar.event_id),
        evidence={
            "level_id": level_id,
            "source_kind": level.evidence["source_kind"],
        },
        side="above",
        price=100.0,
        zone=level.zone,
    )
    penetration = _atomic(
        f"penetration:{level_id}:{crossed_minute}",
        EventKind.LEVEL_PENETRATED,
        crossed_minute,
        2,
        source_event_ids=(
            level.event_id,
            touch.event_id,
            crossing_bar.event_id,
        ),
        evidence={
            "level_id": level_id,
            "crossed_at": _clock(crossed_minute).isoformat(),
        },
        side="above",
        price=101.0,
        direction=Direction.LONG,
    )
    events: list[MarketEvent] = [crossing_bar, touch, penetration]
    _consume(adapter, events)
    for minute in range(crossed_minute + 1, resolved_minute + 1):
        bar = _bar(
            minute,
            high=resolution_high,
            low=99.0,
            close=(
                intermediate_closes[minute]
                if intermediate_closes is not None
                and minute in intermediate_closes
                else 100.5
                if terminal_kind is EventKind.ACCEPTANCE_CONFIRMED
                else 99.75
            ),
        )
        adapter.consume(bar)
        events.append(bar)
    terminal_bar = next(
        item
        for item in reversed(events)
        if item.kind is EventKind.BAR_COMPLETED
    )
    terminal_sequence = 3 if resolved_minute == crossed_minute else 1
    evidence: dict[str, object] = {
        "level_id": level_id,
        "crossed_at": _clock(crossed_minute).isoformat(),
        "resolved_at": _clock(resolved_minute).isoformat(),
    }
    if terminal_evidence is not None:
        evidence.update(terminal_evidence)
    context: tuple[str, ...] = ()
    if protected_swing_id is not None:
        evidence["protected_swing_id"] = protected_swing_id
        evidence["protected_swing_event_id"] = protected_assignment_event_id
        evidence["source_timeframe"] = (
            source_timeframe or Timeframe.H1
        ).value
        context = (str(protected_assignment_event_id),)
    terminal = _atomic(
        f"terminal:{terminal_kind.value}:{level_id}:{resolved_minute}",
        terminal_kind,
        resolved_minute,
        terminal_sequence,
        source_event_ids=(penetration.event_id, terminal_bar.event_id),
        context_event_ids=context,
        evidence=evidence,
        side="above",
        price=101.0,
        direction=(
            Direction.SHORT
            if terminal_kind is EventKind.SWEEP_CONFIRMED
            else Direction.LONG
        ),
    )
    adapter.consume(terminal)
    events.append(terminal)
    return tuple(events), terminal


def test_terminal_interaction_authority_rejects_cross_level_terminal_fact() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    level_events, level = _seed_level(
        adapter,
        label="terminal-authority",
        zone=(100.0, 100.0),
    )
    crossing_events, terminal = _cross_level(
        adapter,
        level,
        crossed_minute=1,
        resolved_minute=2,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
    )
    records = adapter.projection.records
    record_index = next(
        index
        for index, record in enumerate(records)
        if record.object_type
        is FoundationObjectType.LIQUIDITY_INTERACTION_GENERATION
        and record.payload.get("terminal_event_id") == terminal.event_id
    )
    projection = FoundationProjectionReducer.initial_projection()
    for prior_record in records[:record_index]:
        projection = FoundationProjectionReducer.reduce(projection, prior_record)
    record = records[record_index]
    authoritative = {
        event.event_id: event for event in (*level_events, *crossing_events)
    }
    _validate_foundation_authoritative_sources(
        record,
        authoritative,
        projection,
    )

    forged_evidence = {
        **dict(terminal.evidence),
        "level_id": "sibling-level",
    }
    forged_terminal = replace(
        terminal,
        details=forged_evidence,
        evidence=forged_evidence,
    )
    authoritative[terminal.event_id] = forged_terminal
    with pytest.raises(ValueError, match="Liquidity Interaction terminal"):
        _validate_foundation_authoritative_sources(
            record,
            authoritative,
            projection,
        )


def _seed_structure(
    adapter: CanonicalFoundationAdapter,
    *,
    minute: int = 0,
    direction: Direction = Direction.LONG,
    label: str = "parent",
    timeframe: Timeframe = Timeframe.H1,
) -> tuple[tuple[MarketEvent, ...], dict[str, MarketEvent]]:
    bar = _bar(minute)
    high = _atomic(
        f"swing-high:{label}",
        EventKind.SWING_CONFIRMED,
        minute,
        1,
        timeframe=timeframe,
        source_event_ids=(bar.event_id,),
        evidence={"source_entity_id": f"high:{label}"},
        side="above",
        price=105.0,
        direction=Direction.LONG,
    )
    low = _atomic(
        f"swing-low:{label}",
        EventKind.SWING_CONFIRMED,
        minute,
        2,
        timeframe=timeframe,
        source_event_ids=(bar.event_id,),
        evidence={"source_entity_id": f"low:{label}"},
        side="below",
        price=100.0,
        direction=Direction.SHORT,
    )
    structure = _atomic(
        f"structure:{label}",
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        minute,
        3,
        timeframe=timeframe,
        source_event_ids=(high.event_id, low.event_id),
        evidence={
            "structure_id": f"structure-source:{label}",
            "source_high_id": f"high:{label}",
            "source_low_id": f"low:{label}",
            "candidate_protected_swing_id": f"low:{label}",
        },
        side="above" if direction is Direction.LONG else "below",
        price=100.0,
        direction=direction,
    )
    events = (bar, high, low, structure)
    _consume(adapter, events)
    return events, {"bar": bar, "high": high, "low": low, "structure": structure}


def test_candidate_zone_freezes_inward_tradable_tick_envelope() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    with localcontext() as context:
        context.prec = 3
        _, _ = _seed_level(
            adapter,
            label="off-grid-structural-zone",
            source_kind="structural_swing",
            price=20019.0,
            zone=(20018.669412345678, 20019.330698765432),
        )
    level = adapter.lifecycle.levels[-1]
    assert level.price_ticks == 80076
    assert level.lower_bound_ticks == 80075  # ceil(20018.6694 / .25)
    assert level.upper_bound_ticks == 80077  # floor(20019.3306 / .25)

    invalid = CanonicalFoundationAdapter(tick_size=TICK)
    bar = _bar(0)
    source = _atomic(
        "source:narrow-zone",
        EventKind.SWING_CONFIRMED,
        0,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bar.event_id,),
        evidence={"source_entity_id": "swing:narrow-zone"},
        price=100.0,
    )
    invalid.consume_batch((bar, source))
    narrow = _atomic(
        "level-created:narrow-zone",
        EventKind.LIQUIDITY_LEVEL_CREATED,
        0,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(source.event_id,),
        evidence={
            "level_id": "narrow-zone",
            "source_kind": "structural_swing",
        },
        side="above",
        price=100.0,
        zone=(100.01, 100.24),
    )
    frozen = invalid.checkpoint()
    with pytest.raises(ValueError, match="no valid tradable tick envelope"):
        invalid.consume(narrow)
    assert invalid.checkpoint() == frozen

    zone_rearm = CanonicalFoundationAdapter(tick_size=TICK)
    _, zone_level = _seed_level(
        zone_rearm,
        label="zone-rearm",
        source_kind="structural_swing",
        price=100.0,
        zone=(99.6694, 100.3306),
    )
    _cross_level(
        zone_rearm,
        zone_level,
        crossed_minute=1,
        resolved_minute=1,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
    )
    assert zone_rearm.lifecycle.levels[-1].lifecycle is LiquidityLevelLifecycle.DISARMED
    update = zone_rearm.consume(
        _bar(2, high=100.0, low=99.25, close=99.5)
    )
    assert zone_rearm.lifecycle.levels[-1].lifecycle is LiquidityLevelLifecycle.REARMED
    assert tuple(
        record.payload["lifecycle"]
        for record in update.records
        if record.object_type.value == "liquidity_level"
    ) == ("rearmable", "rearmed")


def test_consume_batch_commits_once_and_rolls_back_every_fact_on_failure() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    bar = _bar(0)
    source = _atomic(
        "batch-source",
        EventKind.SWING_CONFIRMED,
        0,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bar.event_id,),
        evidence={"source_entity_id": "batch-source"},
        price=100.0,
    )
    updates = adapter.consume_batch((bar, source))
    assert tuple(item.input_fact_id for item in updates) == (
        bar.event_id,
        source.event_id,
    )
    assert adapter.known_input_event_ids == frozenset(
        {bar.event_id, source.event_id}
    )

    frozen = adapter.checkpoint()
    next_bar = _bar(1)
    bad_level = _atomic(
        "batch-bad-level",
        EventKind.LIQUIDITY_LEVEL_CREATED,
        1,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=("unseen-batch-source",),
        evidence={
            "level_id": "batch-bad-level",
            "source_kind": "confirmed_swing",
        },
        side="above",
        price=100.0,
    )
    with pytest.raises(ValueError, match="unseen source events"):
        adapter.consume_batch((next_bar, bad_level))
    assert adapter.checkpoint() == frozen
    assert next_bar.event_id not in adapter.known_input_event_ids


def test_real_bar_gap_blocks_rearm_until_an_explicit_epoch_reset() -> None:
    skipped = CanonicalFoundationAdapter(tick_size=TICK)
    _, skipped_level = _seed_level(skipped, label="skipped-clock-rearm")
    _cross_level(
        skipped,
        skipped_level,
        crossed_minute=1,
        resolved_minute=1,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
    )
    assert (
        skipped.lifecycle.levels[-1].lifecycle
        is LiquidityLevelLifecycle.DISARMED
    )
    frozen = skipped.checkpoint()
    with pytest.raises(ValueError, match="exactly contiguous"):
        skipped.consume(_bar(3, high=100.0, low=99.0, close=99.75))
    assert skipped.checkpoint() == frozen

    reset_adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, prior_level = _seed_level(reset_adapter, label="pre-reset-level")
    _cross_level(
        reset_adapter,
        prior_level,
        crossed_minute=1,
        resolved_minute=1,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
    )
    reset = _atomic(
        "explicit-gap-reset",
        EventKind.MARKET_EPOCH_RESET,
        10,
        0,
        evidence={"reason": "data_gap_reset"},
    )
    reset_adapter.consume(reset)
    assert reset_adapter.lifecycle.real_bar_clocks == ()
    assert reset_adapter.lifecycle.registered_bar_clocks == ()

    new_bar = _bar(20)
    new_source = _atomic(
        "source:post-reset-level",
        EventKind.SWING_CONFIRMED,
        20,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(new_bar.event_id,),
        evidence={"source_entity_id": "swing:post-reset-level"},
        side="above",
        price=100.0,
        direction=Direction.LONG,
    )
    new_level = _atomic(
        "level-created:post-reset-level",
        EventKind.LIQUIDITY_LEVEL_CREATED,
        20,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(new_source.event_id,),
        evidence={
            "level_id": "post-reset-level",
            "source_kind": "confirmed_swing",
        },
        side="above",
        price=100.0,
    )
    _consume(reset_adapter, (new_bar, new_source, new_level))
    _cross_level(
        reset_adapter,
        new_level,
        crossed_minute=21,
        resolved_minute=21,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
    )
    reset_adapter.consume(
        _bar(22, high=100.0, low=99.0, close=99.75)
    )
    assert (
        reset_adapter.lifecycle.levels[-1].lifecycle
        is LiquidityLevelLifecycle.REARMED
    )
    assert reset_adapter.lifecycle.real_bar_clocks[-1].count == 3


def test_registered_short_h4_and_reopen_are_contiguous_without_epoch_reset() -> None:
    def bar_at(label: str, clock: str) -> MarketEvent:
        known_at = pd.Timestamp(clock, tz=TZ)
        return replace(
            _bar(
                0,
                timeframe=Timeframe.H4,
                event_id=f"registered-h4:{label}",
            ),
            observed_at=known_at,
            event_time=known_at,
            known_at=known_at,
        )

    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    for label, clock in (
        ("memorial-prior", "2024-05-27 10:00"),
        ("memorial-short-close", "2024-05-27 13:00"),
        ("memorial-reopen", "2024-05-27 22:00"),
    ):
        adapter.consume(bar_at(label, clock))
    assert adapter.lifecycle.epoch == 0
    assert adapter.lifecycle.real_bar_clocks[-1].count == 3

    skipped = CanonicalFoundationAdapter(tick_size=TICK)
    skipped.consume(bar_at("skip-prior", "2024-05-27 10:00"))
    frozen = skipped.checkpoint()
    with pytest.raises(ValueError, match="registered native clock"):
        skipped.consume(bar_at("skip-short-close", "2024-05-27 22:00"))
    assert skipped.checkpoint() == frozen


def test_clock_only_bar_advances_registered_clock_without_real_semantics() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    first = _bar(1)
    clock_only = _bar(2, real_completed=False)
    resumed = _bar(3)
    frozen_real_transition = adapter._transition(
        first,
        NormalizedTransitionKind.REAL_BAR_COMPLETED,
        payload={"bar_event_id": first.event_id, "real_completed": True},
        source_event_ids=(first.event_id,),
    )

    adapter.consume(first)
    assert (
        adapter.lifecycle.applied_transitions[0].fingerprint
        == content_hash(frozen_real_transition)
    )
    projection_before = adapter.projection
    update = adapter.consume(clock_only)

    registered = adapter.lifecycle.registered_bar_clocks[0]
    real = adapter.lifecycle.real_bar_clocks[0]
    assert update.records == ()
    assert update.ignored is False
    assert adapter.projection == projection_before
    assert registered.timeframe is Timeframe.M1
    assert registered.count == 2
    assert registered.last_completed_at == clock_only.known_at
    assert registered.last_bar_event_id == clock_only.event_id
    assert real.count == 1
    assert real.last_completed_at == first.known_at
    assert len(adapter._real_bars) == 1

    restored = CanonicalFoundationAdapter.restore(
        pickle.loads(pickle.dumps(adapter.checkpoint()))
    )
    assert (
        adapter.checkpoint().schema_version
        == FOUNDATION_ADAPTER_CHECKPOINT_SCHEMA_VERSION
    )
    adapter.consume(resumed)
    restored.consume(resumed)

    assert adapter.lifecycle == restored.lifecycle
    assert adapter.projection == restored.projection
    assert adapter.lifecycle.registered_bar_clocks[0].count == 3
    assert adapter.lifecycle.real_bar_clocks[0].count == 2
    assert adapter.lifecycle.epoch == 0


@pytest.mark.parametrize(
    ("timeframe", "clocks"),
    (
        (
            Timeframe.M1,
            ("2024-06-06 23:09", "2024-06-06 23:10", "2024-06-06 23:11"),
        ),
        (
            Timeframe.M5,
            ("2024-06-06 23:05", "2024-06-06 23:10", "2024-06-06 23:15"),
        ),
        (
            Timeframe.M15,
            ("2024-06-06 23:00", "2024-06-06 23:15", "2024-06-06 23:30"),
        ),
        (
            Timeframe.H1,
            ("2024-06-06 23:00", "2024-06-07 00:00", "2024-06-07 01:00"),
        ),
        (
            Timeframe.H4,
            ("2024-06-06 22:00", "2024-06-07 02:00", "2024-06-07 06:00"),
        ),
    ),
)
def test_registered_clock_recovers_after_clock_only_bar_on_every_timeframe(
    timeframe: Timeframe,
    clocks: tuple[str, str, str],
) -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    events = []
    for index, clock in enumerate(clocks):
        known_at = pd.Timestamp(clock, tz=TZ)
        event = replace(
            _bar(
                0,
                timeframe=timeframe,
                event_id=f"clock-recovery:{timeframe.value}:{index}",
                real_completed=index != 1,
            ),
            observed_at=known_at,
            event_time=known_at,
            known_at=known_at,
        )
        events.append(
            event
            if event.evidence["real_completed"] is True
            else _with_clock_only_coverage(event)
        )

    _consume(adapter, events)

    assert adapter.lifecycle.registered_bar_clocks[0].count == 3
    assert (
        adapter.lifecycle.registered_bar_clocks[0].last_completed_at
        == events[2].known_at
    )
    assert adapter.lifecycle.real_bar_clocks[0].count == 2
    assert adapter.lifecycle.real_bar_clocks[0].last_completed_at == events[2].known_at


def test_completed_bar_flags_are_mandatory_complements_and_atomic() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    adapter.consume(_bar(1))
    before = adapter.checkpoint()
    invalid = _bar(2)
    evidence = dict(invalid.evidence)
    evidence.pop("clock_only")
    invalid = replace(invalid, details=evidence, evidence=evidence)

    with pytest.raises(ValueError, match="exact complementary"):
        adapter.consume(invalid)

    assert adapter.checkpoint() == before


def test_clock_only_bar_coverage_is_mandatory_and_atomic() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    adapter.consume(_bar(1))
    before = adapter.checkpoint()
    invalid = _bar(2, real_completed=False)
    evidence = dict(invalid.evidence)
    for name in (
        "complete",
        "start",
        "observed_minutes",
        "expected_minutes",
        "real_minutes",
        "synthetic_minutes",
    ):
        evidence.pop(name)
    invalid = replace(invalid, details=evidence, evidence=evidence)

    with pytest.raises(ValueError, match="requires coverage"):
        adapter.consume(invalid)

    assert adapter.checkpoint() == before


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("clock_only_event_time_drift", "event_time and known_at"),
        ("partial_real_coverage", "all-or-none"),
        ("real_with_synthetic_coverage", "coverage is inconsistent"),
    ),
)
def test_direct_bar_root_contract_is_complete_and_atomic(
    case: str,
    message: str,
) -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    adapter.consume(_bar(1))
    before = adapter.checkpoint()
    if case == "clock_only_event_time_drift":
        invalid = replace(
            _bar(2, real_completed=False),
            event_time=_clock(1),
        )
    elif case == "partial_real_coverage":
        event = _bar(2)
        evidence = {
            **dict(event.evidence),
            "complete": True,
            "start": event.known_at - pd.Timedelta(1, unit="min"),
        }
        invalid = replace(event, details=evidence, evidence=evidence)
    elif case == "real_with_synthetic_coverage":
        invalid = _with_clock_only_coverage(_bar(2))
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(case)

    with pytest.raises(ValueError, match=message):
        adapter.consume(invalid)

    assert adapter.checkpoint() == before


def test_direct_adapter_pickle_rejects_missing_internal_state_schema() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    legacy_state = adapter.__getstate__()
    legacy_state.pop("_state_schema_version")
    restored = object.__new__(CanonicalFoundationAdapter)

    with pytest.raises(ValueError, match="pickle state schema changed"):
        restored.__setstate__(legacy_state)


def test_adapter_checkpoint_requires_owned_schema_and_registered_ledger() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    adapter.consume(_bar(1))
    checkpoint = adapter.checkpoint()

    missing_schema = pickle.loads(pickle.dumps(checkpoint))
    vars(missing_schema).pop("schema_version")
    with pytest.raises(ValueError, match="checkpoint integrity"):
        CanonicalFoundationAdapter.restore(missing_schema)

    missing_registered_ledger = pickle.loads(pickle.dumps(checkpoint))
    lifecycle_checkpoint = missing_registered_ledger.lifecycle_checkpoint
    vars(lifecycle_checkpoint.state).pop("registered_bar_clocks")
    vars(lifecycle_checkpoint)["state_digest"] = content_hash(
        lifecycle_checkpoint.state
    )
    vars(missing_registered_ledger)["checkpoint_digest"] = content_hash(
        _checkpoint_payload(missing_registered_ledger)
    )
    with pytest.raises(ValueError, match="lifecycle checkpoint"):
        CanonicalFoundationAdapter.restore(missing_registered_ledger)


def test_same_bar_terminal_uses_actual_event_id_and_competing_terminal_is_atomic() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, level = _seed_level(adapter)
    _, terminal = _cross_level(
        adapter,
        level,
        crossed_minute=1,
        resolved_minute=1,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
    )
    interaction = adapter.lifecycle.interactions[-1]
    frozen_state = adapter.lifecycle
    frozen_projection = adapter.projection

    assert interaction.lifecycle is LiquidityInteractionLifecycle.TERMINAL
    assert interaction.terminal_state is LiquidityInteractionTerminal.SWEEP
    assert interaction.terminal_event_id == terminal.event_id
    assert interaction.constituent_bar_roles == (
        "penetration",
        "reentry",
        "hold",
        "confirmation",
    )
    assert len(set(interaction.constituent_bar_ids)) == 1
    with pytest.raises(ValueError, match="strictly after terminal"):
        interaction.require_response_clock(terminal.known_at)

    competing = _atomic(
        "competing-acceptance",
        EventKind.ACCEPTANCE_CONFIRMED,
        1,
        4,
        source_event_ids=(
            f"penetration:{level.evidence['level_id']}:1",
            "bar:1m:1",
        ),
        evidence={
            "level_id": level.evidence["level_id"],
            "crossed_at": _clock(1).isoformat(),
            "resolved_at": _clock(1).isoformat(),
        },
        side="above",
        price=101.0,
        direction=Direction.LONG,
    )
    with pytest.raises(ValueError, match="exact active penetration generation"):
        adapter.consume(competing)
    assert adapter.lifecycle == frozen_state
    assert adapter.projection == frozen_projection


def test_terminal_role_ledger_follows_exact_completed_close_geometry() -> None:
    acceptance_reentry = CanonicalFoundationAdapter(tick_size=TICK)
    _, level = _seed_level(acceptance_reentry, label="acceptance-reentry")
    _cross_level(
        acceptance_reentry,
        level,
        crossed_minute=1,
        resolved_minute=2,
        terminal_kind=EventKind.ACCEPTANCE_CONFIRMED,
    )
    variant_b = acceptance_reentry.lifecycle.interactions[-1]
    assert variant_b.constituent_bar_roles == (
        "penetration",
        "reentry",
        "confirmation",
    )
    assert variant_b.first_inside_close_at == _clock(1)
    assert variant_b.first_outside_close_at == _clock(2)

    acceptance_hold = CanonicalFoundationAdapter(tick_size=TICK)
    _, level = _seed_level(acceptance_hold, label="acceptance-hold")
    _cross_level(
        acceptance_hold,
        level,
        crossed_minute=1,
        resolved_minute=2,
        terminal_kind=EventKind.ACCEPTANCE_CONFIRMED,
        crossing_close=100.5,
    )
    variant_a = acceptance_hold.lifecycle.interactions[-1]
    assert variant_a.constituent_bar_roles == (
        "penetration",
        "hold",
        "confirmation",
    )
    assert variant_a.first_inside_close_at is None
    assert variant_a.first_outside_close_at == _clock(1)

    delayed_acceptance = CanonicalFoundationAdapter(tick_size=TICK)
    _, level = _seed_level(delayed_acceptance, label="delayed-acceptance")
    _cross_level(
        delayed_acceptance,
        level,
        crossed_minute=1,
        resolved_minute=4,
        terminal_kind=EventKind.ACCEPTANCE_CONFIRMED,
        intermediate_closes={2: 99.75, 3: 100.5},
    )
    delayed = delayed_acceptance.lifecycle.interactions[-1]
    assert delayed.constituent_bar_roles == (
        "penetration",
        "reentry",
        "hold",
        "hold",
        "confirmation",
    )
    assert delayed.first_inside_close_at == _clock(1)
    assert delayed.first_outside_close_at == _clock(3)

    interrupted_then_accepted = CanonicalFoundationAdapter(tick_size=TICK)
    _, level = _seed_level(
        interrupted_then_accepted,
        label="interrupted-then-accepted",
    )
    _cross_level(
        interrupted_then_accepted,
        level,
        crossed_minute=1,
        resolved_minute=4,
        terminal_kind=EventKind.ACCEPTANCE_CONFIRMED,
        crossing_close=100.5,
        intermediate_closes={2: 99.75, 3: 100.5, 4: 100.5},
        terminal_evidence={"outside_completed_bars": 3, "outside_run": 2},
    )
    interrupted = interrupted_then_accepted.lifecycle.interactions[-1]
    assert interrupted.constituent_bar_roles == (
        "penetration",
        "hold",
        "hold",
        "hold",
        "confirmation",
    )
    assert interrupted.first_inside_close_at == _clock(2)
    assert interrupted.first_outside_close_at == _clock(1)

    conflicting_counts = CanonicalFoundationAdapter(tick_size=TICK)
    _, level = _seed_level(conflicting_counts, label="conflicting-counts")
    with pytest.raises(ValueError, match="outside counters"):
        _cross_level(
            conflicting_counts,
            level,
            crossed_minute=1,
            resolved_minute=4,
            terminal_kind=EventKind.ACCEPTANCE_CONFIRMED,
            crossing_close=100.5,
            intermediate_closes={2: 99.75, 3: 100.5, 4: 100.5},
            terminal_evidence={"outside_completed_bars": 2, "outside_run": 3},
        )

    two_bar_sweep = CanonicalFoundationAdapter(tick_size=TICK)
    _, level = _seed_level(two_bar_sweep, label="two-bar-sweep")
    _cross_level(
        two_bar_sweep,
        level,
        crossed_minute=1,
        resolved_minute=2,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
        crossing_close=100.5,
    )
    sweep = two_bar_sweep.lifecycle.interactions[-1]
    assert sweep.constituent_bar_roles == (
        "penetration",
        "reentry",
        "confirmation",
    )
    assert sweep.first_outside_close_at == _clock(1)
    assert sweep.first_inside_close_at == _clock(2)


def test_multibar_ancestry_is_contiguous_complete_and_excludes_post_terminal_bar() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, level = _seed_level(adapter, label="long-window")
    _cross_level(
        adapter,
        level,
        crossed_minute=1,
        resolved_minute=4,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
        resolution_high=103.0,
    )
    interaction = adapter.lifecycle.interactions[-1]
    formation_ids = set(interaction.constituent_bar_ids)

    assert formation_ids == {f"bar:1m:{minute}" for minute in range(1, 5)}
    assert interaction.constituent_bar_roles == (
        "penetration",
        "reentry",
        "hold",
        "hold",
        "confirmation",
    )
    assert interaction.max_penetration_ticks == 12
    adapter.consume(_bar(5))
    assert adapter.lifecycle.interactions[-1] == interaction
    assert "bar:1m:5" not in formation_ids
    assert interaction.require_response_clock(_clock(5)) == _clock(5)


def _maintenance_closure_terminal_chain(
    terminal_kind: EventKind,
) -> tuple[CanonicalFoundationAdapter, tuple[MarketEvent, ...]]:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    suffix = terminal_kind.value

    def at_clock(event: MarketEvent, value: str) -> MarketEvent:
        clock = pd.Timestamp(value, tz=TZ)
        return replace(
            event,
            observed_at=clock,
            event_time=clock,
            known_at=clock,
        )

    seed_bar = at_clock(
        _bar(0, event_id=f"bar:maintenance-seed:{suffix}"),
        "2024-05-28 16:59",
    )
    source = at_clock(
        _atomic(
            f"source:maintenance:{suffix}",
            EventKind.SWING_CONFIRMED,
            0,
            1,
            timeframe=Timeframe.H1,
            source_event_ids=(seed_bar.event_id,),
            evidence={"source_entity_id": f"swing-entity:maintenance:{suffix}"},
            side="above",
            price=100.0,
            direction=Direction.LONG,
        ),
        "2024-05-28 16:59",
    )
    level_id = f"maintenance-level:{suffix}"
    level = at_clock(
        _atomic(
            f"level-created:maintenance:{suffix}",
            EventKind.LIQUIDITY_LEVEL_CREATED,
            0,
            2,
            timeframe=Timeframe.H1,
            source_event_ids=(source.event_id,),
            evidence={
                "level_id": level_id,
                "source_kind": "confirmed_swing",
            },
            side="above",
            price=100.0,
        ),
        "2024-05-28 16:59",
    )
    crossing_close = (
        99.75
        if terminal_kind is EventKind.ACCEPTANCE_CONFIRMED
        else 100.5
    )
    crossing_bar = at_clock(
        _bar(
            0,
            high=101.0,
            low=99.0,
            close=crossing_close,
            event_id=f"bar:maintenance-crossing:{suffix}",
        ),
        "2024-05-28 17:00",
    )
    touch = at_clock(
        _atomic(
            f"touch:maintenance:{suffix}",
            EventKind.LEVEL_TOUCHED,
            0,
            1,
            source_event_ids=(level.event_id, crossing_bar.event_id),
            evidence={
                "level_id": level_id,
                "source_kind": "confirmed_swing",
            },
            side="above",
            price=100.0,
        ),
        "2024-05-28 17:00",
    )
    penetration = at_clock(
        _atomic(
            f"penetration:maintenance:{suffix}",
            EventKind.LEVEL_PENETRATED,
            0,
            2,
            source_event_ids=(level.event_id, touch.event_id, crossing_bar.event_id),
            evidence={
                "level_id": level_id,
                "crossed_at": pd.Timestamp(
                    "2024-05-28 17:00", tz=TZ
                ).isoformat(),
            },
            side="above",
            price=101.0,
            direction=Direction.LONG,
        ),
        "2024-05-28 17:00",
    )
    reopen_close = 100.5 if terminal_kind is EventKind.ACCEPTANCE_CONFIRMED else 99.75
    reopen_bar = at_clock(
        _bar(
            0,
            high=101.0,
            low=99.0,
            close=reopen_close,
            event_id=f"bar:maintenance-reopen:{suffix}",
        ),
        "2024-05-28 18:01",
    )
    resolution_bar = at_clock(
        _bar(
            0,
            high=101.0,
            low=99.0,
            close=reopen_close,
            event_id=f"bar:maintenance-resolution:{suffix}",
        ),
        "2024-05-28 18:02",
    )
    terminal = at_clock(
        _atomic(
            f"terminal:maintenance:{suffix}",
            terminal_kind,
            0,
            1,
            source_event_ids=(penetration.event_id, resolution_bar.event_id),
            evidence={
                "level_id": level_id,
                "crossed_at": pd.Timestamp(
                    "2024-05-28 17:00", tz=TZ
                ).isoformat(),
                "resolved_at": pd.Timestamp(
                    "2024-05-28 18:02", tz=TZ
                ).isoformat(),
            },
            side="above",
            price=101.0,
            direction=(
                Direction.LONG
                if terminal_kind is EventKind.ACCEPTANCE_CONFIRMED
                else Direction.SHORT
            ),
        ),
        "2024-05-28 18:02",
    )
    events = (
        seed_bar,
        source,
        level,
        crossing_bar,
        touch,
        penetration,
        reopen_bar,
        resolution_bar,
        terminal,
    )
    _consume(adapter, events)
    return adapter, events


@pytest.mark.parametrize(
    "terminal_kind",
    (EventKind.ACCEPTANCE_CONFIRMED, EventKind.SWEEP_CONFIRMED),
)
def test_terminal_formation_uses_registered_successor_across_maintenance(
    terminal_kind: EventKind,
) -> None:
    adapter, events = _maintenance_closure_terminal_chain(terminal_kind)
    interaction = adapter.lifecycle.interactions[-1]
    assert interaction.lifecycle is LiquidityInteractionLifecycle.TERMINAL
    assert interaction.terminal_state is (
        LiquidityInteractionTerminal.ACCEPTANCE
        if terminal_kind is EventKind.ACCEPTANCE_CONFIRMED
        else LiquidityInteractionTerminal.SWEEP
    )
    assert set(interaction.constituent_bar_ids) == {
        f"bar:maintenance-crossing:{terminal_kind.value}",
        f"bar:maintenance-reopen:{terminal_kind.value}",
        f"bar:maintenance-resolution:{terminal_kind.value}",
    }
    replayed = CanonicalFoundationAdapter.replay(events, tick_size=TICK)
    assert replayed.lifecycle == adapter.lifecycle
    assert replayed.projection == adapter.projection


def test_terminal_formation_rejects_skipped_registered_or_ordinary_bar() -> None:
    maintenance, _ = _maintenance_closure_terminal_chain(
        EventKind.SWEEP_CONFIRMED
    )
    maintenance._real_bars = [
        item
        for item in maintenance._real_bars
        if item.known_at != pd.Timestamp("2024-05-28 18:01", tz=TZ)
    ]
    with pytest.raises(ValueError, match="continuous real M1 BAR ancestry"):
        maintenance._formation_bars(
            pd.Timestamp("2024-05-28 17:00", tz=TZ),
            pd.Timestamp("2024-05-28 18:02", tz=TZ),
            Timeframe.M1,
        )

    ordinary = CanonicalFoundationAdapter(tick_size=TICK)
    _consume(ordinary, (_bar(0), _bar(1), _bar(2)))
    ordinary._real_bars = [
        item for item in ordinary._real_bars if item.known_at != _clock(1)
    ]
    with pytest.raises(ValueError, match="continuous real M1 BAR ancestry"):
        ordinary._formation_bars(_clock(0), _clock(2), Timeframe.M1)


def test_pool_sweep_preserves_inside_outside_inside_formation_path() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, level = _seed_level(adapter, label="alternating-pool-sweep")
    _cross_level(
        adapter,
        level,
        crossed_minute=1,
        resolved_minute=4,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
        crossing_close=100.0,
        intermediate_closes={2: 100.5, 3: 99.75, 4: 99.75},
    )

    interaction = adapter.lifecycle.interactions[-1]
    assert interaction.terminal_state is LiquidityInteractionTerminal.SWEEP
    assert interaction.constituent_bar_ids == (
        "bar:1m:1",
        "bar:1m:1",
        "bar:1m:2",
        "bar:1m:3",
        "bar:1m:4",
    )
    assert interaction.constituent_bar_roles == (
        "penetration",
        "reentry",
        "hold",
        "reentry",
        "confirmation",
    )
    assert interaction.first_inside_close_at == _clock(1)
    assert interaction.first_outside_close_at == _clock(2)


def test_sweep_rearm_creates_new_immutable_generation_and_equal_pool_rejects() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    level_events, level_event = _seed_level(adapter, label="swing-rearm")
    crossing_events, _ = _cross_level(
        adapter,
        level_event,
        crossed_minute=1,
        resolved_minute=1,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
    )
    first = adapter.lifecycle.interactions[-1]
    departure_bar = _bar(2, high=100.0, low=99.0, close=99.75)
    update = adapter.consume(departure_bar)
    level = adapter.lifecycle.levels[-1]
    second = adapter.lifecycle.interaction(level.active_generation_id)
    level_revisions = tuple(
        record
        for record in update.records
        if record.object_type.value == "liquidity_level"
        and record.object_id == level.level_id
    )

    assert first.terminal_state is LiquidityInteractionTerminal.SWEEP
    assert tuple(record.payload["lifecycle"] for record in level_revisions) == (
        LiquidityLevelLifecycle.REARMABLE.value,
        LiquidityLevelLifecycle.REARMED.value,
    )
    assert len({record.record_id for record in level_revisions}) == 2
    assert second.generation_id != first.generation_id
    assert second.previous_generation_id == first.generation_id
    assert second.generation_number == 2
    assert level_event.event_id in second.source_event_ids
    assert departure_bar.event_id in second.source_event_ids
    assert adapter.lifecycle.interaction(first.generation_id) == first
    replayed = CanonicalFoundationAdapter.replay(
        (*level_events, *crossing_events, departure_bar),
        tick_size=TICK,
    )
    assert replayed.lifecycle == adapter.lifecycle
    assert replayed.projection == adapter.projection

    equal_adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, equal_level = _seed_level(
        equal_adapter,
        label="equal-pool",
        source_kind="equal_highs",
    )
    _cross_level(
        equal_adapter,
        equal_level,
        crossed_minute=1,
        resolved_minute=1,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
    )
    equal_bar = _bar(2, high=100.0, low=99.0, close=99.75)
    equal_adapter.consume(equal_bar)
    assert (
        equal_adapter.lifecycle.levels[-1].lifecycle
        is LiquidityLevelLifecycle.DISARMED
    )
    with pytest.raises(ValueError, match="exact source kind"):
        equal_adapter.observe_rearm_departure(
            source_level_id="equal-pool",
            departure_bar_event_id=equal_bar.event_id,
            departure_price=99.75,
        )

    unknown_adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, unknown_level = _seed_level(
        unknown_adapter,
        label="unknown-reference-prefix",
        source_kind="previous_magic_high",
    )
    _cross_level(
        unknown_adapter,
        unknown_level,
        crossed_minute=1,
        resolved_minute=1,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
    )
    unknown_bar = _bar(2, high=100.0, low=99.0, close=99.75)
    unknown_adapter.consume(unknown_bar)
    assert (
        unknown_adapter.lifecycle.levels[-1].lifecycle
        is LiquidityLevelLifecycle.DISARMED
    )
    with pytest.raises(ValueError, match="exact source kind"):
        unknown_adapter.observe_rearm_departure(
            source_level_id="unknown-reference-prefix",
            departure_bar_event_id=unknown_bar.event_id,
            departure_price=99.75,
        )


def test_terminal_after_unregistered_same_level_recrossing_stays_noncanonical() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, level_event = _seed_level(adapter, label="same-bar-recross")
    _cross_level(
        adapter,
        level_event,
        crossed_minute=1,
        resolved_minute=1,
        terminal_kind=EventKind.SWEEP_CONFIRMED,
    )
    terminal = adapter.lifecycle.interactions[-1]

    outside_bar = _bar(2, high=101.0, low=100.0, close=100.25)
    touch = _atomic(
        "touch:unregistered-recross",
        EventKind.LEVEL_TOUCHED,
        2,
        1,
        source_event_ids=(level_event.event_id, outside_bar.event_id),
        evidence={"level_id": "same-bar-recross"},
    )
    penetration = _atomic(
        "penetration:unregistered-recross",
        EventKind.LEVEL_PENETRATED,
        2,
        2,
        source_event_ids=(level_event.event_id, touch.event_id, outside_bar.event_id),
        evidence={
            "level_id": "same-bar-recross",
            "crossed_at": _clock(2).isoformat(),
        },
        price=101.0,
        direction=Direction.LONG,
    )
    _consume(adapter, (outside_bar, touch, penetration))
    assert adapter.lifecycle.levels[-1].lifecycle is LiquidityLevelLifecycle.DISARMED

    confirmation_bar = _bar(3, high=101.0, low=99.0, close=99.75)
    acceptance = _atomic(
        "acceptance:unregistered-recross",
        EventKind.ACCEPTANCE_CONFIRMED,
        3,
        1,
        source_event_ids=(penetration.event_id, confirmation_bar.event_id),
        evidence={
            "level_id": "same-bar-recross",
            "crossed_at": _clock(2).isoformat(),
            "resolved_at": _clock(3).isoformat(),
        },
        direction=Direction.LONG,
    )
    adapter.consume(confirmation_bar)
    rearmed = adapter.lifecycle.interactions[-1]
    assert adapter.lifecycle.levels[-1].lifecycle is LiquidityLevelLifecycle.REARMED
    assert rearmed.generation_number == 2
    assert rearmed.armed_at == _clock(3)
    update = adapter.consume(acceptance)

    assert update.ignored is True
    assert adapter.lifecycle.interactions == (terminal, rearmed)
    assert rearmed.lifecycle is LiquidityInteractionLifecycle.ARMED
    assert adapter.lifecycle.levels[-1].last_terminal_generation_id == terminal.generation_id


def test_price_terminal_after_exact_administrative_expiry_is_ignored() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, level_event = _seed_level(adapter, label="administrative-expiry")
    crossing_bar = _bar(1, high=101.0, low=99.0, close=99.75)
    touch = _atomic(
        "touch:administrative-expiry",
        EventKind.LEVEL_TOUCHED,
        1,
        1,
        source_event_ids=(level_event.event_id, crossing_bar.event_id),
        evidence={"level_id": "administrative-expiry"},
    )
    penetration = _atomic(
        "penetration:administrative-expiry",
        EventKind.LEVEL_PENETRATED,
        1,
        2,
        source_event_ids=(level_event.event_id, touch.event_id, crossing_bar.event_id),
        evidence={
            "level_id": "administrative-expiry",
            "crossed_at": _clock(1).isoformat(),
        },
        price=101.0,
        direction=Direction.LONG,
    )
    _consume(adapter, (crossing_bar, touch, penetration))
    interaction_id = adapter.lifecycle.levels[-1].active_generation_id
    adapter.retire_level(
        source_level_id="administrative-expiry",
        reason="source_retired",
        source_event_ids=(penetration.event_id,),
        known_at=_clock(1),
        timeframe=Timeframe.H1,
    )
    expired = adapter.lifecycle.interaction(interaction_id)
    terminal = _atomic(
        "terminal:administrative-expiry",
        EventKind.SWEEP_CONFIRMED,
        1,
        3,
        source_event_ids=(penetration.event_id, crossing_bar.event_id),
        evidence={
            "level_id": "administrative-expiry",
            "crossed_at": _clock(1).isoformat(),
            "resolved_at": _clock(1).isoformat(),
        },
        direction=Direction.SHORT,
    )
    update = adapter.consume(terminal)

    assert update.ignored is True
    assert expired.terminal_state is LiquidityInteractionTerminal.EXPIRED
    assert adapter.lifecycle.interaction(interaction_id) == expired


def test_acceptance_permanently_retires_same_source() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, level_event = _seed_level(adapter, label="accepted-source")
    _cross_level(
        adapter,
        level_event,
        crossed_minute=1,
        resolved_minute=2,
        terminal_kind=EventKind.ACCEPTANCE_CONFIRMED,
    )
    assert adapter.lifecycle.levels[-1].lifecycle is LiquidityLevelLifecycle.RETIRED
    assert adapter.projection.active_dol_candidate_ids == ()

    bar = _bar(3)
    source = _atomic(
        "accepted-source-new-root",
        EventKind.SWING_CONFIRMED,
        3,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bar.event_id,),
        evidence={"source_entity_id": "accepted-source"},
    )
    replacement = _atomic(
        "accepted-source-recreated",
        EventKind.LIQUIDITY_LEVEL_CREATED,
        3,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(source.event_id,),
        evidence={
            "level_id": "accepted-source",
            "source_kind": "confirmed_swing",
        },
        side="above",
        price=100.0,
    )
    _consume(adapter, (bar, source))
    with pytest.raises(ValueError, match="accepted liquidity"):
        adapter.consume(replacement)


def test_post_terminal_touch_and_penetration_are_seen_but_do_not_reopen_level() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, level = _seed_level(adapter, label="stale-after-acceptance")
    _cross_level(
        adapter,
        level,
        crossed_minute=1,
        resolved_minute=2,
        terminal_kind=EventKind.ACCEPTANCE_CONFIRMED,
    )
    stale_bar = _bar(3, high=101.0, low=99.0, close=100.5)
    adapter.consume(stale_bar)
    frozen_lifecycle = adapter.lifecycle
    frozen_projection = adapter.projection
    touch = _atomic(
        "stale-terminal-touch",
        EventKind.LEVEL_TOUCHED,
        3,
        1,
        source_event_ids=(level.event_id, stale_bar.event_id),
        evidence={"level_id": "stale-after-acceptance"},
        side="above",
        price=100.0,
    )
    penetration = _atomic(
        "stale-terminal-penetration",
        EventKind.LEVEL_PENETRATED,
        3,
        2,
        source_event_ids=(level.event_id, touch.event_id, stale_bar.event_id),
        evidence={
            "level_id": "stale-after-acceptance",
            "crossed_at": _clock(3).isoformat(),
        },
        side="above",
        price=101.0,
        direction=Direction.LONG,
    )
    touch_update = adapter.consume(touch)
    penetration_update = adapter.consume(penetration)
    assert touch_update.ignored is True
    assert penetration_update.ignored is True
    assert adapter.lifecycle == frozen_lifecycle
    assert adapter.projection == frozen_projection
    assert {touch.event_id, penetration.event_id} <= adapter.known_input_event_ids


def test_retirement_and_reference_replacement_leave_no_stale_dol_candidate() -> None:
    retired_adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, old_event = _seed_level(retired_adapter, label="old-reference")
    retirement = MarketEvent(
        event_id="old-reference-retired",
        kind=EventKind.LIQUIDITY_RETIRED,
        observed_at=_clock(1),
        timeframe=Timeframe.M1,
        side="above",
        price=100.0,
        strength=0.0,
        source_ids=("old-reference",),
        evidence={"source_kind": "previous_day_high"},
        transition_reason="reference_period_replaced",
        sequence_no=0,
    )
    ignored = retired_adapter.consume(retirement)
    assert ignored.ignored is True
    assert (
        retired_adapter.lifecycle.levels[-1].lifecycle
        is LiquidityLevelLifecycle.ACTIVE
    )
    retirement_bar = _bar(1, sequence=1)
    retired_adapter.consume(retirement_bar)
    with pytest.raises(ValueError, match="normalized/atomic"):
        retired_adapter.retire_level(
            source_level_id="old-reference",
            reason="reference_rollover",
            source_event_ids=(retirement.event_id,),
            known_at=_clock(2),
            timeframe=Timeframe.H1,
        )
    retired_adapter.retire_level(
        source_level_id="old-reference",
        reason="reference_rollover",
        source_event_ids=(retirement_bar.event_id,),
        known_at=_clock(2),
        timeframe=Timeframe.H1,
    )
    old_level = retired_adapter.lifecycle.levels[-1]
    assert old_level.lifecycle is LiquidityLevelLifecycle.RETIRED
    assert old_level.retirement_reason == "reference_rollover"
    assert retired_adapter.projection.active_dol_candidate_ids == ()
    metadata = {
        event_id: origin
        for event_id, _, origin in retired_adapter.checkpoint().seen_event_metadata
    }
    assert all(
        metadata[source_id]
        in {EventOrigin.NORMALIZED_DATA, EventOrigin.SEMANTIC_ATOMIC}
        for record in retired_adapter.projection.records
        for source_id in record.source_event_ids
    )

    replaced_adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, active_old = _seed_level(replaced_adapter, label="replace-old")
    bar = _bar(1)
    source = _atomic(
        "source:replace-new",
        EventKind.SWING_CONFIRMED,
        1,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bar.event_id,),
        evidence={"source_entity_id": "replace-new"},
    )
    replacement = _atomic(
        "level-created:replace-new",
        EventKind.LIQUIDITY_LEVEL_CREATED,
        1,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(source.event_id,),
        context_event_ids=(active_old.event_id,),
        evidence={
            "level_id": "replace-new",
            "source_kind": "confirmed_swing",
            "replaces_level_id": "replace-old",
            "replaces_level_event_id": active_old.event_id,
        },
        side="above",
        price=101.0,
    )
    _consume(replaced_adapter, (bar, source, replacement))
    prior = next(
        item
        for item in replaced_adapter.lifecycle.levels
        if item.source_identity == "replace-old"
    )
    current = next(
        item
        for item in replaced_adapter.lifecycle.levels
        if item.source_identity == "replace-new"
    )
    assert prior.retirement_reason == "supersession"
    assert prior.superseded_by_level_id == current.level_id
    assert replaced_adapter.projection.active_dol_candidate_ids == (
        current.level_id,
    )

    disappeared = CanonicalFoundationAdapter(tick_size=TICK)
    _, _ = _seed_level(disappeared, label="retired-swing-source")
    disappearance_bar = _bar(1)
    disappeared.consume(disappearance_bar)
    disappeared.retire_level(
        source_level_id="retired-swing-source",
        reason="source_retired",
        source_event_ids=(disappearance_bar.event_id,),
        known_at=disappearance_bar.known_at,
        timeframe=Timeframe.H1,
    )
    retired_swing = disappeared.lifecycle.levels[-1]
    assert retired_swing.retirement_reason == "source_retired"
    assert disappeared.projection.active_dol_candidate_ids == ()


def test_mss_is_transition_evidence_and_original_direction_resumption_fails_it() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, facts = _seed_structure(adapter)
    generation = adapter.lifecycle.structure_generations[-1]

    bar1 = _bar(1)
    raw_bos = _atomic(
        "raw-bos-long",
        EventKind.RAW_BOUNDARY_BREAK,
        1,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bar1.event_id,),
    )
    bos = _atomic(
        "qualified-bos-long",
        EventKind.QUALIFIED_BOS,
        1,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(raw_bos.event_id, facts["structure"].event_id),
        evidence={
            "bos_id": "bos-long",
            "structure_id": "structure-source:parent",
        },
        direction=Direction.LONG,
    )
    _consume(adapter, (bar1, raw_bos, bos))

    bar2 = _bar(2)
    protected = _atomic(
        "protected-long",
        EventKind.PROTECTED_SWING_ASSIGNED,
        2,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bos.event_id, facts["low"].event_id),
        evidence={
            "protected_swing_id": "low:parent",
            "structure_id": "structure-source:parent",
        },
        price=100.0,
        direction=Direction.LONG,
    )
    _consume(adapter, (bar2, protected))

    bar3 = _bar(3)
    raw_mss = _atomic(
        "raw-mss-short",
        EventKind.RAW_BOUNDARY_BREAK,
        3,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bar3.event_id,),
    )
    mss = _atomic(
        "mss-short",
        EventKind.MSS_CORE_CONFIRMED,
        3,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(raw_mss.event_id, facts["structure"].event_id),
        evidence={"bos_id": "mss-short"},
        source_entity_ids=("mss-short", "structure-source:parent"),
        direction=Direction.SHORT,
    )
    _consume(adapter, (bar3, raw_mss, mss))
    still_external = adapter.lifecycle.structure(generation.generation_id)
    transition = adapter.lifecycle.structure_transitions[-1]

    assert still_external.direction is Direction.LONG
    assert still_external.lifecycle is StructureGenerationLifecycle.CONFIRMED
    assert bos.event_id in still_external.bos_event_ids
    assert still_external.protected_swing_assignment_event_id == protected.event_id
    assert transition.lifecycle is StructureTransitionLifecycle.STARTED
    assert transition.challenger_direction is Direction.SHORT
    first_internal = next(
        item
        for item in adapter.lifecycle.structure_generations
        if item.scope is StructureScope.INTERNAL
        and item.lifecycle is StructureGenerationLifecycle.FORMING
    )
    assert first_internal.origin_event_id == mss.event_id
    assert first_internal.confirmation_event_id is None
    assert first_internal.origin_swing_id == "structure-source:parent"
    assert first_internal.mss_event_ids == (mss.event_id,)

    bar4 = _bar(4)
    raw_mss_2 = _atomic(
        "raw-mss-short-2",
        EventKind.RAW_BOUNDARY_BREAK,
        4,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bar4.event_id,),
    )
    mss_2 = _atomic(
        "mss-short-2",
        EventKind.MSS_CORE_CONFIRMED,
        4,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(raw_mss_2.event_id, facts["structure"].event_id),
        source_entity_ids=("mss-short-2", "structure-source:parent"),
        evidence={"bos_id": "mss-short-2"},
        direction=Direction.SHORT,
    )
    _consume(adapter, (bar4, raw_mss_2, mss_2))
    accumulating = adapter.lifecycle.structure(first_internal.generation_id)
    assert accumulating.lifecycle is StructureGenerationLifecycle.FORMING
    assert accumulating.origin_event_id == mss.event_id
    assert accumulating.mss_event_ids == (mss.event_id, mss_2.event_id)
    assert adapter.lifecycle.structure_transitions[-1].mss_event_ids == (
        mss.event_id,
        mss_2.event_id,
    )

    bar5 = _bar(5)
    challenger_confirmed = _atomic(
        "structure-short-challenger-confirmed",
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        5,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["high"].event_id, facts["low"].event_id),
        evidence={"structure_id": "structure-short-challenger-confirmed"},
        direction=Direction.SHORT,
    )
    _consume(adapter, (bar5, challenger_confirmed))
    matured = adapter.lifecycle.structure(first_internal.generation_id)
    assert matured.lifecycle is StructureGenerationLifecycle.CONFIRMED
    assert matured.origin_event_id == mss.event_id
    assert matured.confirmation_event_id == challenger_confirmed.event_id
    assert matured.confirmed_at > matured.started_at

    bar6 = _bar(6)
    resumed = _atomic(
        "structure-long-resumed",
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        6,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bar6.event_id, facts["structure"].event_id),
        evidence={"structure_id": "structure-long-resumed"},
        direction=Direction.LONG,
    )
    _consume(adapter, (bar6, resumed))
    failed = adapter.lifecycle.structure_transitions[-1]
    assert failed.lifecycle is StructureTransitionLifecycle.FAILED
    assert failed.terminal_reason == "original_direction_resumed"
    terminated = adapter.lifecycle.structure(first_internal.generation_id)
    assert terminated.lifecycle is StructureGenerationLifecycle.TERMINATED
    assert terminated.termination_reason == "scope_rollover"


def test_distinct_bound_structure_alias_mss_restarts_exact_transition_lineage(
) -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    structure_events, facts = _seed_structure(
        adapter,
        direction=Direction.SHORT,
        label="multi-origin",
    )
    incumbent = adapter.lifecycle.structure_generations[-1]
    level_events, protected_level = _seed_level(
        adapter,
        label="multi-origin-protected",
        minute=1,
        source_timeframe=Timeframe.H1,
        side="above",
        price=100.0,
    )
    protected = _atomic(
        "protected:multi-origin",
        EventKind.PROTECTED_SWING_ASSIGNED,
        1,
        3,
        timeframe=Timeframe.H1,
        source_event_ids=(
            facts["structure"].event_id,
            level_events[1].event_id,
        ),
        evidence={
            "structure_id": "structure-source:multi-origin",
            "protected_swing_id": "multi-origin-protected",
        },
        direction=Direction.SHORT,
    )
    alias = _structure_confirmation(
        facts,
        label="multi-origin-alias",
        minute=1,
        sequence=4,
        direction=Direction.SHORT,
        candidate_protected_swing_id="high:multi-origin",
    )
    _consume(adapter, (protected, alias))
    assert (
        adapter._structure_bindings["structure-source:multi-origin-alias"]
        == incumbent.generation_id
    )

    first_bar, first_raw, first_mss = _mss_triplet(
        label="multi-origin:first",
        minute=2,
        direction=Direction.LONG,
        structure_event=facts["structure"],
        structure_id="structure-source:multi-origin",
        raw_direction=Direction.LONG,
    )
    _consume(adapter, (first_bar, first_raw, first_mss))
    old_internal = adapter._active_internal(Timeframe.H1)
    old_transition = adapter._started_transition(Timeframe.H1)
    assert old_internal is not None
    assert old_transition is not None
    assert old_internal.mss_event_ids == (first_mss.event_id,)
    assert old_transition.mss_event_ids == (first_mss.event_id,)

    second_bar, second_raw, second_mss = _mss_triplet(
        label="multi-origin:second",
        minute=3,
        direction=Direction.LONG,
        structure_event=alias,
        structure_id="structure-source:multi-origin-alias",
        raw_direction=Direction.LONG,
    )
    _consume(adapter, (second_bar, second_raw))
    restored = CanonicalFoundationAdapter.restore(
        pickle.loads(pickle.dumps(adapter.checkpoint()))
    )
    adapter.consume(second_mss)
    restored.consume(second_mss)

    replaced_internal = adapter.lifecycle.structure(old_internal.generation_id)
    replaced_transition = adapter.lifecycle.transition(
        old_transition.structure_transition_id
    )
    assert replaced_internal.lifecycle is StructureGenerationLifecycle.TERMINATED
    assert replaced_internal.termination_reason == "superseded"
    assert replaced_transition.lifecycle is StructureTransitionLifecycle.CENSORED
    assert replaced_transition.terminal_reason == "superseded"
    assert second_mss.event_id in replaced_transition.source_event_ids

    current_internal = adapter._active_internal(Timeframe.H1)
    current_transition = adapter._started_transition(Timeframe.H1)
    assert current_internal is not None
    assert current_transition is not None
    assert current_internal.generation_id != old_internal.generation_id
    assert (
        current_transition.structure_transition_id
        != old_transition.structure_transition_id
    )
    assert current_internal.origin_swing_id == (
        "structure-source:multi-origin-alias"
    )
    assert current_internal.mss_event_ids == (second_mss.event_id,)
    assert current_transition.mss_event_ids == (second_mss.event_id,)
    assert current_internal.started_at == current_transition.started_at
    assert sum(
        transition.lifecycle is StructureTransitionLifecycle.STARTED
        for transition in adapter.lifecycle.structure_transitions
    ) == 1
    assert restored.lifecycle == adapter.lifecycle
    assert restored.projection == adapter.projection

    acceptance_events, acceptance = _cross_level(
        adapter,
        protected_level,
        crossed_minute=4,
        resolved_minute=5,
        terminal_kind=EventKind.ACCEPTANCE_CONFIRMED,
        protected_swing_id="multi-origin-protected",
        protected_assignment_event_id=protected.event_id,
        source_timeframe=Timeframe.H1,
    )
    _consume(restored, acceptance_events)
    accepted_transition = adapter.lifecycle.transition(
        current_transition.structure_transition_id
    )
    assert accepted_transition.lifecycle is StructureTransitionLifecycle.STARTED
    assert accepted_transition.protected_acceptance_event_id == acceptance.event_id

    resolution_bar = _bar(6)
    resolution = _structure_confirmation(
        facts,
        label="multi-origin-promoted",
        minute=6,
        direction=Direction.LONG,
        candidate_protected_swing_id="low:multi-origin",
    )
    _consume(adapter, (resolution_bar, resolution))
    _consume(restored, (resolution_bar, resolution))

    promoted = adapter.lifecycle.transition(
        current_transition.structure_transition_id
    )
    assert promoted.lifecycle is StructureTransitionLifecycle.CONFIRMED
    assert promoted.terminal_reason == (
        "protected_acceptance_plus_opposite_confirmation"
    )
    assert promoted.opposite_confirmation_event_id == resolution.event_id
    assert (
        adapter.lifecycle.structure(current_internal.generation_id).termination_reason
        == "superseded"
    )
    external = adapter._active_external(Timeframe.H1)
    assert external is not None
    assert external.direction is Direction.LONG
    assert restored.lifecycle == adapter.lifecycle
    assert restored.projection == adapter.projection

    prefix = (
        *structure_events,
        *level_events,
        protected,
        alias,
        first_bar,
        first_raw,
        first_mss,
        second_bar,
        second_raw,
        second_mss,
    )
    _assert_adapter_replay(
        (*prefix, *acceptance_events, resolution_bar, resolution),
        adapter,
    )


def test_later_incumbent_qualified_bos_fails_forming_mss_challenger() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    structure_events, facts = _seed_structure(adapter)
    incumbent = adapter.lifecycle.structure_generations[-1]

    mss_bar, raw_mss, mss = _mss_triplet(
        label="before-continuation",
        minute=1,
        direction=Direction.SHORT,
        structure_event=facts["structure"],
        structure_id="structure-source:parent",
        bos_id="mss-before-continuation",
        raw_event_id="raw-mss-before-continuation",
        mss_event_id="mss-before-continuation",
    )
    _consume(adapter, (mss_bar, raw_mss, mss))
    transition = adapter.lifecycle.structure_transitions[-1]
    challenger = next(
        generation
        for generation in adapter.lifecycle.structure_generations
        if generation.scope is StructureScope.INTERNAL
        and generation.lifecycle is StructureGenerationLifecycle.FORMING
    )

    continuation_bar = _bar(2)
    raw_continuation = _atomic(
        "raw-incumbent-continuation",
        EventKind.RAW_BOUNDARY_BREAK,
        2,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(continuation_bar.event_id,),
        direction=Direction.LONG,
    )
    _consume(adapter, (continuation_bar, raw_continuation))
    assert (
        adapter.lifecycle.transition(transition.structure_transition_id).lifecycle
        is StructureTransitionLifecycle.STARTED
    )
    assert (
        adapter.lifecycle.structure(challenger.generation_id).lifecycle
        is StructureGenerationLifecycle.FORMING
    )

    qualified = _atomic(
        "qualified-incumbent-continuation",
        EventKind.QUALIFIED_BOS,
        2,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(
            raw_continuation.event_id,
            facts["structure"].event_id,
        ),
        evidence={
            "bos_id": "qualified-incumbent-continuation",
            "structure_id": "structure-source:parent",
        },
        direction=Direction.LONG,
    )
    adapter.consume(qualified)

    failed = adapter.lifecycle.transition(transition.structure_transition_id)
    assert failed.lifecycle is StructureTransitionLifecycle.FAILED
    assert failed.terminal_reason == "original_direction_resumed"
    assert failed.resumption_event_id == qualified.event_id
    terminated = adapter.lifecycle.structure(challenger.generation_id)
    assert terminated.lifecycle is StructureGenerationLifecycle.TERMINATED
    assert terminated.termination_reason == "scope_rollover"
    live_incumbent = adapter.lifecycle.structure(incumbent.generation_id)
    assert live_incumbent.lifecycle is StructureGenerationLifecycle.CONFIRMED
    assert live_incumbent.direction is Direction.LONG

    replayed = CanonicalFoundationAdapter.replay(
        (
            *structure_events,
            mss_bar,
            raw_mss,
            mss,
            continuation_bar,
            raw_continuation,
            qualified,
        ),
        tick_size=TICK,
    )
    assert replayed.lifecycle == adapter.lifecycle
    assert replayed.projection == adapter.projection


def test_same_clock_incumbent_qualified_bos_does_not_fail_mss() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, facts = _seed_structure(adapter)
    bar, raw_mss, mss = _mss_triplet(
        label="same-clock",
        minute=1,
        direction=Direction.SHORT,
        structure_event=facts["structure"],
        structure_id="structure-source:parent",
        bos_id="same-clock-mss",
        raw_event_id="same-clock-raw-mss",
        mss_event_id="same-clock-mss",
    )
    raw_continuation = _atomic(
        "same-clock-raw-continuation",
        EventKind.RAW_BOUNDARY_BREAK,
        1,
        3,
        timeframe=Timeframe.H1,
        source_event_ids=(bar.event_id,),
        direction=Direction.LONG,
    )
    qualified = _atomic(
        "same-clock-qualified-continuation",
        EventKind.QUALIFIED_BOS,
        1,
        4,
        timeframe=Timeframe.H1,
        source_event_ids=(
            raw_continuation.event_id,
            facts["structure"].event_id,
        ),
        evidence={
            "bos_id": "same-clock-qualified-continuation",
            "structure_id": "structure-source:parent",
        },
        direction=Direction.LONG,
    )
    _consume(adapter, (bar, raw_mss, mss, raw_continuation, qualified))

    assert (
        adapter.lifecycle.structure_transitions[-1].lifecycle
        is StructureTransitionLifecycle.STARTED
    )
    assert any(
        generation.scope is StructureScope.INTERNAL
        and generation.lifecycle is StructureGenerationLifecycle.FORMING
        for generation in adapter.lifecycle.structure_generations
    )


def test_unbound_opposite_tracker_bos_and_assignment_stay_noncanonical() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, facts = _seed_structure(adapter)
    incumbent = adapter.lifecycle.structure_generations[-1]

    opposite_bar = _bar(1)
    adapter.consume(opposite_bar)
    opposite_structure = _structure_confirmation(
        facts,
        label="unbound-opposite",
        minute=1,
        direction=Direction.SHORT,
        candidate_protected_swing_id="high:parent",
    )
    structure_update = adapter.consume(opposite_structure)
    assert structure_update.ignored is True

    bar = _bar(2)
    raw = _atomic(
        "raw:unbound-opposite",
        EventKind.RAW_BOUNDARY_BREAK,
        2,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bar.event_id,),
        direction=Direction.SHORT,
    )
    bos = _atomic(
        "bos:unbound-opposite",
        EventKind.QUALIFIED_BOS,
        2,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(raw.event_id, opposite_structure.event_id),
        source_entity_ids=(
            "bos-source:unbound-opposite",
            "structure-source:unbound-opposite",
        ),
        evidence={"bos_id": "bos-source:unbound-opposite"},
        direction=Direction.SHORT,
    )
    protected = _atomic(
        "protected:unbound-opposite",
        EventKind.PROTECTED_SWING_ASSIGNED,
        2,
        3,
        timeframe=Timeframe.H1,
        source_event_ids=(bos.event_id, facts["high"].event_id),
        evidence={
            "structure_id": "structure-source:unbound-opposite",
            "protected_swing_id": "high:parent",
        },
        direction=Direction.SHORT,
    )
    _consume(adapter, (bar, raw))
    assert adapter.consume(bos).ignored is True
    assert adapter.consume(protected).ignored is True

    unchanged = adapter.lifecycle.structure(incumbent.generation_id)
    assert unchanged.lifecycle is StructureGenerationLifecycle.CONFIRMED
    assert unchanged.direction is Direction.LONG
    assert unchanged.bos_event_ids == ()
    assert unchanged.protected_swing_assignment_event_id is None

    conflicting = _atomic(
        "bos:bound-direction-conflict",
        EventKind.QUALIFIED_BOS,
        3,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bos.event_id, facts["structure"].event_id),
        evidence={
            "bos_id": "bos-source:bound-direction-conflict",
            "structure_id": "structure-source:parent",
        },
        direction=Direction.SHORT,
    )
    with pytest.raises(ValueError, match="live external generation"):
        adapter.consume(conflicting)


def test_exact_evidence_for_rolled_internal_owner_stays_noncanonical() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, facts = _seed_structure(adapter)

    first_bar = _bar(1)
    first_raw = _atomic(
        "raw:first-internal",
        EventKind.RAW_BOUNDARY_BREAK,
        1,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(first_bar.event_id,),
        direction=Direction.SHORT,
    )
    first_mss = _atomic(
        "mss:first-internal",
        EventKind.MSS_CORE_CONFIRMED,
        1,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(first_raw.event_id, facts["structure"].event_id),
        source_entity_ids=(
            "bos-source:first-internal",
            "structure-source:parent",
        ),
        evidence={"bos_id": "bos-source:first-internal"},
        direction=Direction.SHORT,
    )
    _consume(adapter, (first_bar, first_raw, first_mss))
    first_internal = adapter.lifecycle.structure_generations[-1]

    confirmation_bar = _bar(2)
    first_confirmation = _atomic(
        "structure:first-internal",
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        2,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["high"].event_id, facts["low"].event_id),
        evidence={
            "structure_id": "structure-source:first-internal",
            "source_high_id": "high:parent",
            "source_low_id": "low:parent",
            "candidate_protected_swing_id": "high:parent",
        },
        direction=Direction.SHORT,
    )
    _consume(adapter, (confirmation_bar, first_confirmation))

    resumption_bar = _bar(3)
    resumption = _atomic(
        "structure:resume-before-new-internal",
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        3,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["high"].event_id, facts["low"].event_id),
        evidence={
            "structure_id": "structure-source:resume-before-new-internal",
            "source_high_id": "high:parent",
            "source_low_id": "low:parent",
            "candidate_protected_swing_id": "low:parent",
        },
        direction=Direction.LONG,
    )
    _consume(adapter, (resumption_bar, resumption))
    assert (
        adapter.lifecycle.structure(first_internal.generation_id).lifecycle
        is StructureGenerationLifecycle.TERMINATED
    )

    second_bar = _bar(4)
    second_raw = _atomic(
        "raw:second-internal",
        EventKind.RAW_BOUNDARY_BREAK,
        4,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(second_bar.event_id,),
        direction=Direction.SHORT,
    )
    second_mss = _atomic(
        "mss:second-internal",
        EventKind.MSS_CORE_CONFIRMED,
        4,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(second_raw.event_id, facts["structure"].event_id),
        source_entity_ids=(
            "bos-source:second-internal",
            "structure-source:parent",
        ),
        evidence={"bos_id": "bos-source:second-internal"},
        direction=Direction.SHORT,
    )
    _consume(adapter, (second_bar, second_raw, second_mss))
    second_internal = adapter.lifecycle.structure_generations[-1]
    assert second_internal.generation_id != first_internal.generation_id

    stale_bar = _bar(5)
    stale_raw = _atomic(
        "raw:stale-first-internal",
        EventKind.RAW_BOUNDARY_BREAK,
        5,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(stale_bar.event_id,),
        direction=Direction.SHORT,
    )
    stale_bos = _atomic(
        "bos:stale-first-internal",
        EventKind.QUALIFIED_BOS,
        5,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(stale_raw.event_id, first_confirmation.event_id),
        source_entity_ids=(
            "bos-source:stale-first-internal",
            "structure-source:first-internal",
        ),
        evidence={"bos_id": "bos-source:stale-first-internal"},
        direction=Direction.SHORT,
    )
    _consume(adapter, (stale_bar, stale_raw))
    assert adapter.consume(stale_bos).ignored is True
    assert (
        adapter.lifecycle.structure(second_internal.generation_id).bos_event_ids
        == ()
    )

    stale_counter_bar = _bar(6)
    stale_counter_raw = _atomic(
        "raw:counter-stale-first-internal",
        EventKind.RAW_BOUNDARY_BREAK,
        6,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(stale_counter_bar.event_id,),
        direction=Direction.LONG,
    )
    stale_counter_mss = _atomic(
        "mss:counter-stale-first-internal",
        EventKind.MSS_CORE_CONFIRMED,
        6,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(
            stale_counter_raw.event_id,
            first_confirmation.event_id,
        ),
        source_entity_ids=(
            "bos-source:counter-stale-first-internal",
            "structure-source:first-internal",
        ),
        evidence={"bos_id": "bos-source:counter-stale-first-internal"},
        direction=Direction.LONG,
    )
    _consume(adapter, (stale_counter_bar, stale_counter_raw))
    checkpoint = adapter.checkpoint()
    assert adapter.consume(stale_counter_mss).ignored is True
    assert adapter.lifecycle == checkpoint.lifecycle_checkpoint.state


def test_counter_mss_on_internal_challenger_does_not_confirm_resumption() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, facts = _seed_structure(adapter)

    mss_bar, raw_mss, mss = _mss_triplet(
        label="internal-challenger",
        minute=1,
        direction=Direction.SHORT,
        structure_event=facts["structure"],
        structure_id="structure-source:parent",
        raw_direction=Direction.SHORT,
    )
    _consume(adapter, (mss_bar, raw_mss, mss))
    transition = adapter.lifecycle.structure_transitions[-1]
    internal = adapter.lifecycle.structure_generations[-1]

    confirmation_bar = _bar(2)
    challenger_confirmation = _structure_confirmation(
        facts,
        label="internal-challenger",
        minute=2,
        direction=Direction.SHORT,
        candidate_protected_swing_id="high:parent",
    )
    _consume(adapter, (confirmation_bar, challenger_confirmation))

    counter_bar, raw_counter, counter = _mss_triplet(
        label="counter-mss",
        minute=3,
        direction=Direction.LONG,
        structure_event=challenger_confirmation,
        structure_id="structure-source:internal-challenger",
        raw_direction=Direction.LONG,
    )
    _consume(adapter, (counter_bar, raw_counter))
    update = adapter.consume(counter)

    assert update.ignored is True
    assert (
        adapter.lifecycle.transition(transition.structure_transition_id).lifecycle
        is StructureTransitionLifecycle.STARTED
    )
    assert (
        adapter.lifecycle.structure(internal.generation_id).lifecycle
        is StructureGenerationLifecycle.CONFIRMED
    )

    _, protected_level = _seed_level(
        adapter,
        label="internal-high",
        minute=4,
        source_timeframe=Timeframe.H1,
        side="above",
        price=100.0,
    )
    bos_bar = _bar(5)
    raw_bos = _atomic(
        "raw:internal-protection",
        EventKind.RAW_BOUNDARY_BREAK,
        5,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bos_bar.event_id,),
        direction=Direction.SHORT,
    )
    bos = _atomic(
        "bos:internal-protection",
        EventKind.QUALIFIED_BOS,
        5,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(raw_bos.event_id, challenger_confirmation.event_id),
        evidence={
            "bos_id": "bos-source:internal-protection",
            "structure_id": "structure-source:internal-challenger",
        },
        direction=Direction.SHORT,
    )
    protected = _atomic(
        "protected:internal-protection",
        EventKind.PROTECTED_SWING_ASSIGNED,
        5,
        3,
        timeframe=Timeframe.H1,
        source_event_ids=(bos.event_id, protected_level.event_id),
        evidence={
            "structure_id": "structure-source:internal-challenger",
            "protected_swing_id": "internal-high",
        },
        direction=Direction.SHORT,
    )
    _consume(adapter, (bos_bar, raw_bos, bos, protected))
    assert (
        adapter.lifecycle.structure(internal.generation_id).protected_swing_id
        == "internal-high"
    )

    resumption_bar = _bar(6)
    resumption = _structure_confirmation(
        facts,
        label="registered-resumption",
        minute=6,
        direction=Direction.LONG,
        candidate_protected_swing_id="low:parent",
    )
    _consume(adapter, (resumption_bar, resumption))
    assert (
        adapter.lifecycle.structure(internal.generation_id).lifecycle
        is StructureGenerationLifecycle.TERMINATED
    )

    _cross_level(
        adapter,
        protected_level,
        crossed_minute=7,
        resolved_minute=8,
        terminal_kind=EventKind.ACCEPTANCE_CONFIRMED,
        protected_swing_id="internal-high",
        protected_assignment_event_id=protected.event_id,
        source_timeframe=Timeframe.H1,
    )
    external = adapter.lifecycle.structure_generations[0]
    assert external.lifecycle is StructureGenerationLifecycle.CONFIRMED
    assert external.protected_swing_assignment_event_id is None
    assert (
        adapter.lifecycle.interactions[-1].terminal_state
        is LiquidityInteractionTerminal.ACCEPTANCE
    )


def _accepted_live_internal_challenger(
    *,
    confirm_before_acceptance: bool,
) -> tuple[
    CanonicalFoundationAdapter,
    tuple[MarketEvent, ...],
    dict[str, MarketEvent],
]:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    structure_events, facts = _seed_structure(
        adapter,
        direction=Direction.SHORT,
        label="accepted-live-internal",
    )
    level_events, protected_level = _seed_level(
        adapter,
        label="accepted-live-internal-protected",
        minute=1,
        source_timeframe=Timeframe.H1,
        side="above",
        price=100.0,
    )
    protected_source = level_events[1]
    protected = _atomic(
        "protected:accepted-live-internal",
        EventKind.PROTECTED_SWING_ASSIGNED,
        1,
        3,
        timeframe=Timeframe.H1,
        source_event_ids=(
            facts["structure"].event_id,
            protected_source.event_id,
        ),
        evidence={
            "structure_id": "structure-source:accepted-live-internal",
            "protected_swing_id": "accepted-live-internal-protected",
        },
        direction=Direction.SHORT,
    )
    adapter.consume(protected)

    mss_bar, raw_mss, mss = _mss_triplet(
        label="accepted-live-internal",
        minute=2,
        direction=Direction.LONG,
        structure_event=facts["structure"],
        structure_id="structure-source:accepted-live-internal",
        raw_direction=Direction.LONG,
    )
    _consume(adapter, (mss_bar, raw_mss, mss))

    confirmation_bar = _bar(3)
    challenger_confirmation = _structure_confirmation(
        facts,
        label="accepted-live-internal-challenger",
        minute=3,
        direction=Direction.LONG,
        candidate_protected_swing_id="low:accepted-live-internal",
    )
    if confirm_before_acceptance:
        _consume(adapter, (confirmation_bar, challenger_confirmation))
        confirmation_events = (confirmation_bar, challenger_confirmation)
    else:
        adapter.consume(confirmation_bar)
        confirmation_events = (confirmation_bar,)

    acceptance_events, acceptance = _cross_level(
        adapter,
        protected_level,
        crossed_minute=4,
        resolved_minute=5,
        terminal_kind=EventKind.ACCEPTANCE_CONFIRMED,
        protected_swing_id="accepted-live-internal-protected",
        protected_assignment_event_id=protected.event_id,
        source_timeframe=Timeframe.H1,
    )
    transition = adapter.lifecycle.structure_transitions[-1]
    internal = adapter._active_internal(Timeframe.H1)
    assert transition.lifecycle is StructureTransitionLifecycle.STARTED
    assert transition.protected_acceptance_event_id == acceptance.event_id
    assert internal is not None
    assert internal.direction is Direction.LONG
    assert internal.lifecycle is (
        StructureGenerationLifecycle.CONFIRMED
        if confirm_before_acceptance
        else StructureGenerationLifecycle.FORMING
    )
    assert adapter._active_external(Timeframe.H1) is None
    acceptance_bar = next(
        event
        for event in acceptance_events
        if event.kind is EventKind.BAR_COMPLETED
        and event.known_at == _clock(5)
    )
    facts.update(
        {
            "protected": protected,
            "mss": mss,
            "challenger_confirmation": challenger_confirmation,
            "acceptance": acceptance,
            "acceptance_bar": acceptance_bar,
        }
    )
    prefix = (
        *structure_events,
        *level_events,
        protected,
        mss_bar,
        raw_mss,
        mss,
        *confirmation_events,
        *acceptance_events,
    )
    return adapter, prefix, facts


def _accepted_internal_counter_mss(
    facts: dict[str, MarketEvent],
    *,
    minute: int = 6,
    direction: Direction = Direction.SHORT,
) -> tuple[MarketEvent, MarketEvent, MarketEvent]:
    return _mss_triplet(
        label=f"accepted-internal-counter:{minute}",
        minute=minute,
        direction=direction,
        structure_event=facts["challenger_confirmation"],
        structure_id="structure-source:accepted-live-internal-challenger",
        bar_event_id=f"bar:accepted-internal-counter:{minute}",
        raw_direction=direction,
    )


def test_post_acceptance_counter_mss_on_confirmed_internal_is_ignored() -> None:
    adapter, prefix, facts = _accepted_live_internal_challenger(
        confirm_before_acceptance=True
    )
    internal = adapter._active_internal(Timeframe.H1)
    assert internal is not None
    counter_events = _accepted_internal_counter_mss(facts)
    _consume(adapter, counter_events[:2])
    restored = CanonicalFoundationAdapter.restore(
        pickle.loads(pickle.dumps(adapter.checkpoint()))
    )
    prior_lifecycle = adapter.lifecycle
    prior_projection = adapter.projection
    prior_bindings = dict(adapter._structure_bindings)
    prior_mss_ids = internal.mss_event_ids

    update = adapter.consume(counter_events[2])
    restored_update = restored.consume(counter_events[2])

    assert update.ignored is True
    assert restored_update.ignored is True
    assert adapter.lifecycle == prior_lifecycle
    assert adapter.projection == prior_projection
    assert adapter._structure_bindings == prior_bindings
    assert (
        adapter.lifecycle.structure(internal.generation_id).mss_event_ids
        == prior_mss_ids
    )
    assert counter_events[2].event_id in adapter.known_input_event_ids
    assert restored.lifecycle == adapter.lifecycle
    assert restored.projection == adapter.projection
    _assert_adapter_replay((*prefix, *counter_events), adapter)


@pytest.mark.parametrize(
    "tamper",
    ("same_direction", "foreign_confirmation", "same_clock"),
)
def test_post_acceptance_counter_mss_requires_exact_source_and_clock(
    tamper: str,
) -> None:
    adapter, _, facts = _accepted_live_internal_challenger(
        confirm_before_acceptance=True
    )
    if tamper == "same_clock":
        raw = _atomic(
            "raw:accepted-internal-counter:same-clock",
            EventKind.RAW_BOUNDARY_BREAK,
            5,
            2,
            timeframe=Timeframe.H1,
            source_event_ids=(facts["acceptance_bar"].event_id,),
            direction=Direction.SHORT,
        )
        counter = _atomic(
            "mss:accepted-internal-counter:same-clock",
            EventKind.MSS_CORE_CONFIRMED,
            5,
            3,
            timeframe=Timeframe.H1,
            source_event_ids=(
                raw.event_id,
                facts["challenger_confirmation"].event_id,
            ),
            source_entity_ids=(
                "bos-source:accepted-internal-counter:same-clock",
                "structure-source:accepted-live-internal-challenger",
            ),
            evidence={
                "bos_id": "bos-source:accepted-internal-counter:same-clock"
            },
            direction=Direction.SHORT,
        )
        adapter.consume(raw)
    else:
        bar, raw, counter = _accepted_internal_counter_mss(
            facts,
            direction=(
                Direction.LONG
                if tamper == "same_direction"
                else Direction.SHORT
            ),
        )
        _consume(adapter, (bar, raw))
        if tamper == "foreign_confirmation":
            forged_sources = (raw.event_id, facts["structure"].event_id)
            counter = replace(
                counter,
                source_ids=forged_sources,
                source_event_ids=forged_sources,
            )
    before = adapter.checkpoint()
    with pytest.raises(
        ValueError,
        match="MSS does not oppose one confirmed external generation",
    ):
        adapter.consume(counter)
    assert adapter.checkpoint() == before


@pytest.mark.parametrize("confirm_before_acceptance", (False, True))
@pytest.mark.parametrize(
    ("direction", "expected_transition", "transition_reason", "internal_reason"),
    (
        (
            Direction.LONG,
            StructureTransitionLifecycle.CONFIRMED,
            "protected_acceptance_plus_opposite_confirmation",
            "superseded",
        ),
        (
            Direction.SHORT,
            StructureTransitionLifecycle.CENSORED,
            "scope_rollover",
            "scope_rollover",
        ),
    ),
)
def test_post_acceptance_structure_resolves_live_internal_challenger(
    confirm_before_acceptance: bool,
    direction: Direction,
    expected_transition: StructureTransitionLifecycle,
    transition_reason: str,
    internal_reason: str,
) -> None:
    adapter, prefix, facts = _accepted_live_internal_challenger(
        confirm_before_acceptance=confirm_before_acceptance
    )
    transition = adapter.lifecycle.structure_transitions[-1]
    internal = adapter._active_internal(Timeframe.H1)
    assert internal is not None
    restored = CanonicalFoundationAdapter.restore(
        pickle.loads(pickle.dumps(adapter.checkpoint()))
    )
    bar = _bar(6, event_id=f"bar:accepted-resolution:{direction.value}")
    structure = _structure_confirmation(
        facts,
        label=f"accepted-resolution:{direction.value}",
        minute=6,
        direction=direction,
        candidate_protected_swing_id=(
            "low:accepted-live-internal"
            if direction is Direction.LONG
            else "high:accepted-live-internal"
        ),
    )

    _consume(adapter, (bar, structure))
    _consume(restored, (bar, structure))

    resolved = adapter.lifecycle.transition(transition.structure_transition_id)
    assert resolved.lifecycle is expected_transition
    assert resolved.terminal_reason == transition_reason
    assert structure.event_id in resolved.source_event_ids
    assert (
        adapter.lifecycle.structure(internal.generation_id).termination_reason
        == internal_reason
    )
    external = adapter._active_external(Timeframe.H1)
    assert external is not None
    assert external.direction is direction
    if expected_transition is StructureTransitionLifecycle.CONFIRMED:
        assert resolved.opposite_confirmation_event_id == structure.event_id
        assert resolved.resumption_event_id is None
    else:
        assert resolved.opposite_confirmation_event_id is None
        assert resolved.resumption_event_id is None
    assert restored.lifecycle == adapter.lifecycle
    assert restored.projection == adapter.projection
    _assert_adapter_replay((*prefix, bar, structure), adapter)


@pytest.mark.parametrize("confirm_before_acceptance", (False, True))
def test_same_clock_later_sequence_promotes_accepted_challenger(
    confirm_before_acceptance: bool,
) -> None:
    adapter, prefix, facts = _accepted_live_internal_challenger(
        confirm_before_acceptance=confirm_before_acceptance
    )
    transition = adapter.lifecycle.structure_transitions[-1]
    internal = adapter._active_internal(Timeframe.H1)
    assert internal is not None
    assert adapter._last_order is not None
    acceptance = facts["acceptance"]
    structure = _structure_confirmation(
        facts,
        label="accepted-resolution:same-clock-later-sequence",
        minute=5,
        sequence=adapter._last_order[1] + 1,
        direction=Direction.LONG,
        candidate_protected_swing_id="low:accepted-live-internal",
    )
    assert structure.known_at == acceptance.known_at
    assert structure.sequence_no > acceptance.sequence_no
    restored = CanonicalFoundationAdapter.restore(
        pickle.loads(pickle.dumps(adapter.checkpoint()))
    )

    adapter.consume(structure)
    restored.consume(structure)

    resolved = adapter.lifecycle.transition(
        transition.structure_transition_id
    )
    assert resolved.lifecycle is StructureTransitionLifecycle.CONFIRMED
    assert resolved.opposite_confirmation_event_id == structure.event_id
    assert resolved.resumption_event_id is None
    assert (
        adapter.lifecycle.structure(internal.generation_id).termination_reason
        == "superseded"
    )
    assert restored.lifecycle == adapter.lifecycle
    assert restored.projection == adapter.projection
    _assert_adapter_replay((*prefix, structure), adapter)


def test_same_clock_nonlater_sequence_cannot_promote_accepted_challenger() -> None:
    adapter, _, facts = _accepted_live_internal_challenger(
        confirm_before_acceptance=True
    )
    assert adapter._last_order is not None
    structure = _structure_confirmation(
        facts,
        label="accepted-resolution:same-clock-not-later",
        minute=5,
        sequence=adapter._last_order[1],
        event_id="zz-structure:accepted-resolution:same-clock-not-later",
        direction=Direction.LONG,
        candidate_protected_swing_id="low:accepted-live-internal",
    )
    before = adapter.checkpoint()
    with pytest.raises(ValueError, match="exact later protected Acceptance"):
        adapter.consume(structure)
    assert adapter.checkpoint() == before


def test_later_incumbent_qualified_bos_fails_confirmed_internal_challenger() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    structure_events, facts = _seed_structure(adapter)
    incumbent = adapter.lifecycle.structure_generations[-1]
    mss_bar, raw_mss, mss = _mss_triplet(
        label="confirmed-challenger",
        minute=1,
        direction=Direction.SHORT,
        structure_event=facts["structure"],
        structure_id="structure-source:parent",
        bos_id="confirmed-challenger-mss",
        raw_event_id="confirmed-challenger-raw-mss",
        mss_event_id="confirmed-challenger-mss",
    )
    _consume(adapter, (mss_bar, raw_mss, mss))
    transition = adapter.lifecycle.structure_transitions[-1]
    challenger = next(
        generation
        for generation in adapter.lifecycle.structure_generations
        if generation.scope is StructureScope.INTERNAL
    )

    confirmation_bar = _bar(2)
    challenger_confirmation = _structure_confirmation(
        facts,
        label="confirmed-challenger",
        minute=2,
        event_id="confirmed-challenger-structure",
        structure_id="confirmed-challenger-structure",
        direction=Direction.SHORT,
    )
    _consume(adapter, (confirmation_bar, challenger_confirmation))
    assert (
        adapter.lifecycle.structure(challenger.generation_id).lifecycle
        is StructureGenerationLifecycle.CONFIRMED
    )
    assert (
        adapter.lifecycle.transition(transition.structure_transition_id).lifecycle
        is StructureTransitionLifecycle.STARTED
    )

    continuation_bar = _bar(3)
    raw_continuation = _atomic(
        "confirmed-challenger-raw-continuation",
        EventKind.RAW_BOUNDARY_BREAK,
        3,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(continuation_bar.event_id,),
        direction=Direction.LONG,
    )
    qualified = _atomic(
        "confirmed-challenger-qualified-continuation",
        EventKind.QUALIFIED_BOS,
        3,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(
            raw_continuation.event_id,
            facts["structure"].event_id,
        ),
        evidence={
            "bos_id": "confirmed-challenger-qualified-continuation",
            "structure_id": "structure-source:parent",
        },
        direction=Direction.LONG,
    )
    _consume(adapter, (continuation_bar, raw_continuation))
    assert (
        adapter.lifecycle.transition(transition.structure_transition_id).lifecycle
        is StructureTransitionLifecycle.STARTED
    )
    adapter.consume(qualified)

    assert (
        adapter.lifecycle.transition(transition.structure_transition_id).lifecycle
        is StructureTransitionLifecycle.FAILED
    )
    assert (
        adapter.lifecycle.structure(challenger.generation_id).lifecycle
        is StructureGenerationLifecycle.TERMINATED
    )
    current_incumbent = adapter.lifecycle.structure(incumbent.generation_id)
    assert current_incumbent.lifecycle is StructureGenerationLifecycle.CONFIRMED
    assert current_incumbent.direction is Direction.LONG

    replayed = CanonicalFoundationAdapter.replay(
        (
            *structure_events,
            mss_bar,
            raw_mss,
            mss,
            confirmation_bar,
            challenger_confirmation,
            continuation_bar,
            raw_continuation,
            qualified,
        ),
        tick_size=TICK,
    )
    assert replayed.lifecycle == adapter.lifecycle
    assert replayed.projection == adapter.projection


def test_exact_protected_acceptance_then_opposite_generation_confirms_transition() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    structure_events, facts = _seed_structure(adapter)
    level = _atomic(
        "level-created:protected-low",
        EventKind.LIQUIDITY_LEVEL_CREATED,
        0,
        4,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["low"].event_id,),
        evidence={
            "level_id": "swing:low:parent",
            "source_kind": "confirmed_swing",
        },
        side="above",
        price=100.0,
    )
    adapter.consume(level)

    bar1 = _bar(1)
    protected = _atomic(
        "protected-parent",
        EventKind.PROTECTED_SWING_ASSIGNED,
        1,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["structure"].event_id, facts["low"].event_id),
        evidence={
            "protected_swing_id": "low:parent",
            "structure_id": "structure-source:parent",
        },
        price=100.0,
        direction=Direction.LONG,
    )
    _consume(adapter, (bar1, protected))

    bar2 = _bar(2)
    raw_mss = _atomic(
        "raw-transition-short",
        EventKind.RAW_BOUNDARY_BREAK,
        2,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bar2.event_id,),
    )
    mss = _atomic(
        "mss-transition-short",
        EventKind.MSS_CORE_CONFIRMED,
        2,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(raw_mss.event_id, facts["structure"].event_id),
        evidence={"bos_id": "mss-transition-short"},
        source_entity_ids=(
            "mss-transition-short",
            "structure-source:parent",
        ),
        direction=Direction.SHORT,
    )
    _consume(adapter, (bar2, raw_mss, mss))
    incumbent_id = adapter.lifecycle.structure_transitions[-1].incumbent_structure_generation_id

    crossing_bar = _bar(3, high=101.0, low=99.0, close=100.5)
    touch = _atomic(
        "touch-protected",
        EventKind.LEVEL_TOUCHED,
        3,
        1,
        source_event_ids=(level.event_id, crossing_bar.event_id),
        evidence={"level_id": "swing:low:parent"},
        side="above",
        price=100.0,
    )
    penetration = _atomic(
        "penetration-protected",
        EventKind.LEVEL_PENETRATED,
        3,
        2,
        source_event_ids=(level.event_id, touch.event_id, crossing_bar.event_id),
        evidence={
            "level_id": "swing:low:parent",
            "crossed_at": _clock(3).isoformat(),
        },
        side="above",
        price=101.0,
        direction=Direction.LONG,
    )
    acceptance_bar = _bar(4, high=101.0, low=99.0, close=100.5)
    acceptance = _atomic(
        "protected-acceptance",
        EventKind.ACCEPTANCE_CONFIRMED,
        4,
        1,
        source_event_ids=(penetration.event_id, acceptance_bar.event_id),
        context_event_ids=(protected.event_id,),
        evidence={
            "level_id": "swing:low:parent",
            "crossed_at": _clock(3).isoformat(),
            "resolved_at": _clock(4).isoformat(),
            "protected_swing_id": "low:parent",
            "protected_swing_event_id": protected.event_id,
            "source_timeframe": Timeframe.H1.value,
        },
        side="above",
        price=101.0,
        direction=Direction.SHORT,
    )
    _consume(
        adapter,
        (crossing_bar, touch, penetration, acceptance_bar, acceptance),
    )
    incumbent = adapter.lifecycle.structure(incumbent_id)
    transition = adapter.lifecycle.structure_transitions[-1]

    assert incumbent.lifecycle is StructureGenerationLifecycle.TERMINATED
    assert incumbent.protected_acceptance_event_id == acceptance.event_id
    assert transition.lifecycle is StructureTransitionLifecycle.STARTED
    assert transition.protected_acceptance_event_id == acceptance.event_id
    internal = next(
        item
        for item in adapter.lifecycle.structure_generations
        if item.scope is StructureScope.INTERNAL
        and item.lifecycle is StructureGenerationLifecycle.FORMING
    )

    bar4 = _bar(5)
    opposite = _atomic(
        "structure-short-confirmed",
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        5,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["high"].event_id, facts["low"].event_id),
        evidence={
            "structure_id": "structure-short-confirmed",
            "source_high_id": "high:parent",
            "source_low_id": "low:parent",
            "candidate_protected_swing_id": "high:parent",
        },
        direction=Direction.SHORT,
    )
    _consume(adapter, (bar4, opposite))
    confirmed = adapter.lifecycle.structure_transitions[-1]
    assert confirmed.lifecycle is StructureTransitionLifecycle.CONFIRMED
    assert confirmed.opposite_confirmation_event_id == opposite.event_id
    assert (
        adapter.lifecycle.structure(internal.generation_id).lifecycle
        is StructureGenerationLifecycle.TERMINATED
    )
    assert (
        adapter.lifecycle.structure(internal.generation_id).confirmation_event_id
        == opposite.event_id
    )
    assert (
        adapter.lifecycle.structure(confirmed.opposite_structure_generation_id).direction
        is Direction.SHORT
    )


def _unbound_tracker_transition_after_acceptance() -> tuple[
    CanonicalFoundationAdapter,
    tuple[MarketEvent, ...],
    dict[str, MarketEvent],
]:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    structure_events, facts = _seed_structure(adapter)
    incumbent = adapter.lifecycle.structure_generations[-1]
    level = _atomic(
        "level-created:tracker-transition-protected-low",
        EventKind.LIQUIDITY_LEVEL_CREATED,
        0,
        4,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["low"].event_id,),
        evidence={
            "level_id": "swing:low:parent",
            "source_kind": "confirmed_swing",
        },
        side="above",
        price=100.0,
    )
    protected_bar = _bar(1)
    incumbent_protected = _atomic(
        "protected:tracker-transition-incumbent",
        EventKind.PROTECTED_SWING_ASSIGNED,
        1,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["structure"].event_id, facts["low"].event_id),
        evidence={
            "protected_swing_id": "low:parent",
            "structure_id": "structure-source:parent",
        },
        price=100.0,
        direction=Direction.LONG,
    )
    leg = _atomic(
        "leg:tracker-only-short",
        EventKind.STRUCTURAL_LEG_CREATED,
        1,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["high"].event_id, facts["low"].event_id),
        source_entity_ids=(
            "leg-entity:tracker-only-short",
            "high:parent",
            "low:parent",
        ),
        evidence={
            "leg_id": "leg-entity:tracker-only-short",
            "start_swing_id": "high:parent",
            "end_swing_id": "low:parent",
        },
        direction=Direction.SHORT,
    )
    mss_bar = _bar(2)
    tracker_only = _atomic(
        "structure:tracker-only-transition-short",
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        2,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["high"].event_id, facts["low"].event_id),
        source_entity_ids=(
            "structure-source:tracker-only-transition-short",
            "high:parent",
            "low:parent",
        ),
        evidence={
            "structure_id": "structure-source:tracker-only-transition-short",
            "source_high_id": "high:parent",
            "source_low_id": "low:parent",
            "candidate_protected_swing_id": "high:parent",
        },
        direction=Direction.SHORT,
    )
    raw_mss = _atomic(
        "raw:independent-transition-short",
        EventKind.RAW_BOUNDARY_BREAK,
        2,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(mss_bar.event_id,),
        direction=Direction.SHORT,
    )
    mss = _atomic(
        "mss:independent-transition-short",
        EventKind.MSS_CORE_CONFIRMED,
        2,
        3,
        timeframe=Timeframe.H1,
        source_event_ids=(raw_mss.event_id, facts["structure"].event_id),
        source_entity_ids=(
            "bos:independent-transition-short",
            "structure-source:parent",
        ),
        evidence={"bos_id": "bos:independent-transition-short"},
        direction=Direction.SHORT,
    )
    crossing_bar = _bar(3, high=101.0, low=99.0, close=100.5)
    touch = _atomic(
        "touch:tracker-transition-protected-low",
        EventKind.LEVEL_TOUCHED,
        3,
        1,
        source_event_ids=(level.event_id, crossing_bar.event_id),
        evidence={"level_id": "swing:low:parent"},
        side="above",
        price=100.0,
    )
    penetration = _atomic(
        "penetration:tracker-transition-protected-low",
        EventKind.LEVEL_PENETRATED,
        3,
        2,
        source_event_ids=(level.event_id, touch.event_id, crossing_bar.event_id),
        evidence={
            "level_id": "swing:low:parent",
            "crossed_at": _clock(3).isoformat(),
        },
        side="above",
        price=101.0,
        direction=Direction.LONG,
    )
    acceptance_bar = _bar(4, high=101.0, low=99.0, close=100.5)
    acceptance = _atomic(
        "acceptance:tracker-transition-protected-low",
        EventKind.ACCEPTANCE_CONFIRMED,
        4,
        1,
        source_event_ids=(penetration.event_id, acceptance_bar.event_id),
        context_event_ids=(incumbent_protected.event_id,),
        evidence={
            "level_id": "swing:low:parent",
            "crossed_at": _clock(3).isoformat(),
            "resolved_at": _clock(4).isoformat(),
            "protected_swing_id": "low:parent",
            "protected_swing_event_id": incumbent_protected.event_id,
            "source_timeframe": Timeframe.H1.value,
        },
        side="above",
        price=101.0,
        direction=Direction.SHORT,
    )
    suffix = (
        level,
        protected_bar,
        incumbent_protected,
        leg,
        mss_bar,
        tracker_only,
        raw_mss,
        mss,
        crossing_bar,
        touch,
        penetration,
        acceptance_bar,
        acceptance,
    )
    _consume(adapter, suffix)
    transition = adapter.lifecycle.structure_transitions[-1]
    internal = adapter.lifecycle.structure_generations[-1]
    assert adapter.lifecycle.structure(incumbent.generation_id).termination_reason == (
        "protected_break_accepted"
    )
    assert transition.lifecycle is StructureTransitionLifecycle.STARTED
    assert transition.protected_acceptance_event_id == acceptance.event_id
    assert internal.scope is StructureScope.INTERNAL
    assert internal.lifecycle is StructureGenerationLifecycle.FORMING
    assert internal.origin_event_id == mss.event_id
    assert (
        "structure-source:tracker-only-transition-short"
        not in adapter._structure_bindings
    )
    facts.update(
        {
            "level": level,
            "incumbent_protected": incumbent_protected,
            "leg": leg,
            "tracker_only": tracker_only,
            "mss": mss,
            "acceptance_bar": acceptance_bar,
            "acceptance": acceptance,
        }
    )
    return adapter, (*structure_events, *suffix), facts


def test_original_direction_after_acceptance_censors_exact_forming_challenger() -> None:
    adapter, prefix, facts = _unbound_tracker_transition_after_acceptance()
    transition = adapter.lifecycle.structure_transitions[-1]
    internal = adapter._active_internal(Timeframe.H1)
    assert internal is not None
    checkpoint = pickle.loads(pickle.dumps(adapter.checkpoint()))
    restored = CanonicalFoundationAdapter.restore(checkpoint)
    bar = _bar(5)
    original_direction = _atomic(
        "structure:post-acceptance-original-direction",
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        5,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["high"].event_id, facts["low"].event_id),
        evidence={
            "structure_id": "structure-source:post-acceptance-original-direction",
            "source_high_id": "high:parent",
            "source_low_id": "low:parent",
            "candidate_protected_swing_id": "low:parent",
        },
        direction=Direction.LONG,
    )

    _consume(adapter, (bar, original_direction))
    _consume(restored, (bar, original_direction))

    censored = adapter.lifecycle.transition(transition.structure_transition_id)
    assert censored.lifecycle is StructureTransitionLifecycle.CENSORED
    assert censored.terminal_reason == "scope_rollover"
    assert (
        censored.protected_acceptance_event_id
        == transition.protected_acceptance_event_id
    )
    assert censored.resumption_event_id is None
    assert original_direction.event_id in censored.source_event_ids
    rolled = adapter.lifecycle.structure(internal.generation_id)
    assert rolled.lifecycle is StructureGenerationLifecycle.TERMINATED
    assert rolled.termination_reason == "scope_rollover"
    current = adapter._active_external(Timeframe.H1)
    assert current is not None
    assert current.direction is Direction.LONG
    assert current.lifecycle is StructureGenerationLifecycle.CONFIRMED
    assert (
        adapter._structure_bindings[
            "structure-source:post-acceptance-original-direction"
        ]
        == current.generation_id
    )
    assert restored.lifecycle == adapter.lifecycle
    assert restored.projection == adapter.projection
    replayed = CanonicalFoundationAdapter.replay(
        (*prefix, bar, original_direction),
        tick_size=TICK,
    )
    assert replayed.lifecycle == adapter.lifecycle
    assert replayed.projection == adapter.projection


def _unbound_tracker_continuation(
    facts: dict[str, MarketEvent],
    *,
    minute: int = 5,
    raw_sequence: int = 1,
) -> tuple[MarketEvent, MarketEvent, MarketEvent, MarketEvent]:
    bar = _bar(minute)
    raw = _atomic(
        "raw:tracker-only-transition-continuation",
        EventKind.RAW_BOUNDARY_BREAK,
        minute,
        raw_sequence,
        timeframe=Timeframe.H1,
        source_event_ids=(bar.event_id,),
        direction=Direction.SHORT,
    )
    qualified = _atomic(
        "qualified:tracker-only-transition-continuation",
        EventKind.QUALIFIED_BOS,
        minute,
        raw_sequence + 1,
        timeframe=Timeframe.H1,
        source_event_ids=(raw.event_id, facts["tracker_only"].event_id),
        source_entity_ids=(
            "bos:tracker-only-transition-continuation",
            "structure-source:tracker-only-transition-short",
        ),
        evidence={
            "bos_id": "bos:tracker-only-transition-continuation",
            "scope": "continuation",
            "qualification": "aligned_with_confirmed_structure",
        },
        direction=Direction.SHORT,
    )
    assignment = _atomic(
        "protected:tracker-only-transition-continuation",
        EventKind.PROTECTED_SWING_ASSIGNED,
        minute,
        raw_sequence + 2,
        timeframe=Timeframe.H1,
        source_event_ids=(
            qualified.event_id,
            facts["leg"].event_id,
            facts["high"].event_id,
        ),
        source_entity_ids=(
            "bos:tracker-only-transition-continuation",
            "structure-source:tracker-only-transition-short",
            "leg-entity:tracker-only-short",
            "high:parent",
        ),
        evidence={
            "bos_id": "bos:tracker-only-transition-continuation",
            "structure_id": "structure-source:tracker-only-transition-short",
            "origin_leg_id": "leg-entity:tracker-only-short",
            "protected_swing_id": "high:parent",
            "break_standard": "later_acceptance_beyond",
        },
        direction=Direction.SHORT,
    )
    return bar, raw, qualified, assignment


def test_strictly_later_unbound_qbos_and_immediate_assignment_stay_noncanonical() -> None:
    adapter, prefix, facts = _unbound_tracker_transition_after_acceptance()
    bar, raw, qualified, assignment = _unbound_tracker_continuation(facts)
    _consume(adapter, (bar, raw))
    frozen_lifecycle = adapter.lifecycle
    frozen_projection = adapter.projection
    frozen_bindings = dict(adapter._structure_bindings)

    assert adapter.consume(qualified).ignored is True
    assert adapter.consume(assignment).ignored is True

    assert adapter.lifecycle == frozen_lifecycle
    assert adapter.projection == frozen_projection
    assert adapter._structure_bindings == frozen_bindings
    assert qualified.event_id in adapter.known_input_event_ids
    assert assignment.event_id in adapter.known_input_event_ids
    replayed = CanonicalFoundationAdapter.replay(
        (*prefix, bar, raw, qualified, assignment),
        tick_size=TICK,
    )
    assert replayed.lifecycle == adapter.lifecycle
    assert replayed.projection == adapter.projection
    assert replayed._structure_bindings == adapter._structure_bindings


@pytest.mark.parametrize(
    "case",
    (
        "scope",
        "qualification",
        "direction",
        "entities",
        "structure_source_clock",
        "context",
    ),
)
def test_unbound_transition_qbos_requires_every_exact_condition(case: str) -> None:
    adapter, _, facts = _unbound_tracker_transition_after_acceptance()
    bar, raw, qualified, _ = _unbound_tracker_continuation(facts)
    _consume(adapter, (bar, raw))
    if case in {"scope", "qualification"}:
        evidence = dict(qualified.evidence)
        evidence[case] = "opposed" if case == "scope" else "score_gate"
        qualified = replace(qualified, details=evidence, evidence=evidence)
    elif case == "direction":
        qualified = replace(qualified, direction=Direction.LONG)
    elif case == "entities":
        qualified = replace(
            qualified,
            source_entity_ids=tuple(reversed(qualified.source_entity_ids)),
        )
    elif case == "structure_source_clock":
        sources = (raw.event_id, facts["high"].event_id)
        qualified = replace(
            qualified,
            source_ids=sources,
            source_event_ids=sources,
        )
    elif case == "context":
        qualified = replace(
            qualified,
            context_event_ids=(facts["acceptance"].event_id,),
        )
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(case)
    frozen = adapter.checkpoint()

    with pytest.raises(ValueError, match="does not bind a canonical Structure"):
        adapter.consume(qualified)

    assert adapter.checkpoint() == frozen


def test_unbound_transition_qbos_requires_immediate_raw_and_strictly_later_clock() -> None:
    adapter, _, facts = _unbound_tracker_transition_after_acceptance()
    bar, raw, qualified, _ = _unbound_tracker_continuation(facts)
    interposed = _atomic(
        "interposed:after-tracker-only-raw",
        EventKind.SWING_CONFIRMED,
        5,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(bar.event_id,),
        evidence={"source_entity_id": "interposed:after-tracker-only-raw"},
    )
    qualified = replace(qualified, sequence_no=3)
    _consume(adapter, (bar, raw, interposed))
    frozen = adapter.checkpoint()
    with pytest.raises(ValueError, match="does not bind a canonical Structure"):
        adapter.consume(qualified)
    assert adapter.checkpoint() == frozen

    same_clock, _, same_facts = _unbound_tracker_transition_after_acceptance()
    raw_same_clock = _atomic(
        "raw:tracker-only-same-acceptance-clock",
        EventKind.RAW_BOUNDARY_BREAK,
        4,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(same_facts["acceptance_bar"].event_id,),
        direction=Direction.SHORT,
    )
    qualified_same_clock = _atomic(
        "qualified:tracker-only-same-acceptance-clock",
        EventKind.QUALIFIED_BOS,
        4,
        3,
        timeframe=Timeframe.H1,
        source_event_ids=(
            raw_same_clock.event_id,
            same_facts["tracker_only"].event_id,
        ),
        source_entity_ids=(
            "bos:tracker-only-same-acceptance-clock",
            "structure-source:tracker-only-transition-short",
        ),
        evidence={
            "bos_id": "bos:tracker-only-same-acceptance-clock",
            "scope": "continuation",
            "qualification": "aligned_with_confirmed_structure",
        },
        direction=Direction.SHORT,
    )
    same_clock.consume(raw_same_clock)
    frozen = same_clock.checkpoint()
    with pytest.raises(ValueError, match="does not bind a canonical Structure"):
        same_clock.consume(qualified_same_clock)
    assert same_clock.checkpoint() == frozen


def test_unbound_transition_qbos_requires_current_forming_transition_state() -> None:
    reset_adapter, _, reset_facts = _unbound_tracker_transition_after_acceptance()
    reset = _atomic(
        "reset:tracker-only-transition",
        EventKind.MARKET_EPOCH_RESET,
        5,
        0,
        evidence={"reason": "semantic_reset"},
    )
    reset_adapter.consume(reset)
    bar, raw, qualified, _ = _unbound_tracker_continuation(
        reset_facts,
        minute=6,
    )
    _consume(reset_adapter, (bar, raw))
    frozen = reset_adapter.checkpoint()
    with pytest.raises(ValueError, match="does not bind a canonical Structure"):
        reset_adapter.consume(qualified)
    assert reset_adapter.checkpoint() == frozen

    confirmed_adapter, _, confirmed_facts = (
        _unbound_tracker_transition_after_acceptance()
    )
    confirming_bar = _bar(5)
    confirmation = _atomic(
        "structure:independent-short-confirmation",
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        5,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(
            confirmed_facts["high"].event_id,
            confirmed_facts["low"].event_id,
        ),
        evidence={
            "structure_id": "structure-source:independent-short-confirmation",
            "source_high_id": "high:parent",
            "source_low_id": "low:parent",
            "candidate_protected_swing_id": "high:parent",
        },
        direction=Direction.SHORT,
    )
    _consume(confirmed_adapter, (confirming_bar, confirmation))
    assert confirmed_adapter._started_transition(Timeframe.H1) is None
    assert confirmed_adapter._active_internal(Timeframe.H1) is None
    bar, raw, qualified, _ = _unbound_tracker_continuation(
        confirmed_facts,
        minute=6,
    )
    _consume(confirmed_adapter, (bar, raw))
    frozen = confirmed_adapter.checkpoint()
    with pytest.raises(ValueError, match="does not bind a canonical Structure"):
        confirmed_adapter.consume(qualified)
    assert confirmed_adapter.checkpoint() == frozen


@pytest.mark.parametrize(
    "case",
    ("direction", "entities", "sources", "break_standard", "context"),
)
def test_unbound_transition_assignment_requires_exact_immediate_qbos_chain(
    case: str,
) -> None:
    adapter, _, facts = _unbound_tracker_transition_after_acceptance()
    bar, raw, qualified, assignment = _unbound_tracker_continuation(facts)
    _consume(adapter, (bar, raw, qualified))
    if case == "direction":
        assignment = replace(assignment, direction=Direction.LONG)
    elif case == "entities":
        assignment = replace(
            assignment,
            source_entity_ids=assignment.source_entity_ids[:-1],
        )
    elif case == "sources":
        sources = assignment.source_event_ids[:-1]
        assignment = replace(
            assignment,
            source_ids=sources,
            source_event_ids=sources,
        )
    elif case == "break_standard":
        evidence = {**dict(assignment.evidence), "break_standard": "raw_wick"}
        assignment = replace(assignment, details=evidence, evidence=evidence)
    elif case == "context":
        assignment = replace(
            assignment,
            context_event_ids=(facts["acceptance"].event_id,),
        )
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(case)
    frozen = adapter.checkpoint()

    with pytest.raises(ValueError, match="does not bind a canonical Structure"):
        adapter.consume(assignment)

    assert adapter.checkpoint() == frozen


def test_unbound_transition_assignment_rejects_an_interposed_atomic_fact() -> None:
    adapter, _, facts = _unbound_tracker_transition_after_acceptance()
    bar, raw, qualified, assignment = _unbound_tracker_continuation(facts)
    interposed = _atomic(
        "interposed:after-tracker-only-qbos",
        EventKind.SWING_CONFIRMED,
        5,
        3,
        timeframe=Timeframe.H1,
        source_event_ids=(bar.event_id,),
        evidence={"source_entity_id": "interposed:after-tracker-only-qbos"},
    )
    assignment = replace(assignment, sequence_no=4)
    _consume(adapter, (bar, raw, qualified, interposed))
    frozen = adapter.checkpoint()

    with pytest.raises(ValueError, match="does not bind a canonical Structure"):
        adapter.consume(assignment)

    assert adapter.checkpoint() == frozen


def _accepted_protected_external_before_mss() -> tuple[
    CanonicalFoundationAdapter,
    tuple[MarketEvent, ...],
    dict[str, MarketEvent],
    str,
]:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    structure_events, facts = _seed_structure(adapter)
    incumbent_id = adapter.lifecycle.structure_generations[-1].generation_id
    level = _atomic(
        "level-created:pre-mss-protected-low",
        EventKind.LIQUIDITY_LEVEL_CREATED,
        0,
        4,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["low"].event_id,),
        evidence={
            "level_id": "swing:low:parent",
            "source_kind": "confirmed_swing",
        },
        side="above",
        price=100.0,
    )
    protected_bar = _bar(1)
    protected = _atomic(
        "protected-before-mss",
        EventKind.PROTECTED_SWING_ASSIGNED,
        1,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["structure"].event_id, facts["low"].event_id),
        evidence={
            "protected_swing_id": "low:parent",
            "structure_id": "structure-source:parent",
        },
        price=100.0,
        direction=Direction.LONG,
    )
    tracker_only = _atomic(
        "tracker-only-opposite-before-acceptance",
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        1,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["high"].event_id, facts["low"].event_id),
        evidence={
            "structure_id": "structure-source:tracker-only-before-acceptance",
            "source_high_id": "high:parent",
            "source_low_id": "low:parent",
            "candidate_protected_swing_id": "high:parent",
        },
        direction=Direction.SHORT,
    )
    crossing_bar = _bar(2, high=101.0, low=99.0, close=100.5)
    touch = _atomic(
        "touch-before-mss",
        EventKind.LEVEL_TOUCHED,
        2,
        1,
        source_event_ids=(level.event_id, crossing_bar.event_id),
        evidence={"level_id": "swing:low:parent"},
        side="above",
        price=100.0,
    )
    penetration = _atomic(
        "penetration-before-mss",
        EventKind.LEVEL_PENETRATED,
        2,
        2,
        source_event_ids=(level.event_id, touch.event_id, crossing_bar.event_id),
        evidence={
            "level_id": "swing:low:parent",
            "crossed_at": _clock(2).isoformat(),
        },
        side="above",
        price=101.0,
        direction=Direction.LONG,
    )
    acceptance_bar = _bar(3, high=101.0, low=99.0, close=100.5)
    acceptance = _atomic(
        "protected-acceptance-before-mss",
        EventKind.ACCEPTANCE_CONFIRMED,
        3,
        1,
        source_event_ids=(penetration.event_id, acceptance_bar.event_id),
        context_event_ids=(protected.event_id,),
        evidence={
            "level_id": "swing:low:parent",
            "crossed_at": _clock(2).isoformat(),
            "resolved_at": _clock(3).isoformat(),
            "protected_swing_id": "low:parent",
            "protected_swing_event_id": protected.event_id,
            "source_timeframe": Timeframe.H1.value,
        },
        side="above",
        price=101.0,
        direction=Direction.SHORT,
    )
    events = (
        *structure_events,
        level,
        protected_bar,
        protected,
        tracker_only,
        crossing_bar,
        touch,
        penetration,
        acceptance_bar,
        acceptance,
    )
    _consume(
        adapter,
        (
            level,
            protected_bar,
            protected,
            tracker_only,
            crossing_bar,
            touch,
            penetration,
            acceptance_bar,
            acceptance,
        ),
    )
    incumbent = adapter.lifecycle.structure(incumbent_id)
    assert incumbent.lifecycle is StructureGenerationLifecycle.TERMINATED
    assert incumbent.termination_reason == "protected_break_accepted"
    assert (
        "structure-source:tracker-only-before-acceptance"
        not in adapter._structure_bindings
    )
    facts["tracker_only"] = tracker_only
    return adapter, events, facts, incumbent_id


def test_strictly_late_opposite_mss_against_exact_protected_terminal_is_ignored() -> None:
    adapter, events, facts, incumbent_id = _accepted_protected_external_before_mss()
    bar = _bar(4)
    raw = _atomic(
        "raw-late-mss-after-acceptance",
        EventKind.RAW_BOUNDARY_BREAK,
        4,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bar.event_id,),
        direction=Direction.SHORT,
    )
    mss = _atomic(
        "late-mss-after-acceptance",
        EventKind.MSS_CORE_CONFIRMED,
        4,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(raw.event_id, facts["structure"].event_id),
        source_entity_ids=(
            "late-mss-after-acceptance",
            "structure-source:parent",
        ),
        evidence={"bos_id": "late-mss-after-acceptance"},
        direction=Direction.SHORT,
    )
    _consume(adapter, (bar, raw))
    frozen_lifecycle = adapter.lifecycle
    frozen_projection = adapter.projection

    update = adapter.consume(mss)

    assert update.ignored is True
    assert adapter.lifecycle == frozen_lifecycle
    assert adapter.projection == frozen_projection
    assert mss.event_id in adapter.known_input_event_ids
    assert adapter.lifecycle.structure(incumbent_id).mss_event_ids == ()
    replayed = CanonicalFoundationAdapter.replay(
        (*events, bar, raw, mss),
        tick_size=TICK,
    )
    assert replayed.lifecycle == adapter.lifecycle
    assert replayed.projection == adapter.projection


def test_unbound_tracker_mss_matching_latest_protected_terminal_is_ignored() -> None:
    adapter, events, facts, incumbent_id = _accepted_protected_external_before_mss()
    bar = _bar(4)
    raw = _atomic(
        "raw-unbound-tracker-mss-after-acceptance",
        EventKind.RAW_BOUNDARY_BREAK,
        4,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bar.event_id,),
        direction=Direction.LONG,
    )
    mss = _atomic(
        "unbound-tracker-mss-after-acceptance",
        EventKind.MSS_CORE_CONFIRMED,
        4,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(raw.event_id, facts["tracker_only"].event_id),
        source_entity_ids=(
            "unbound-tracker-mss-after-acceptance",
            "structure-source:tracker-only-before-acceptance",
        ),
        evidence={"bos_id": "unbound-tracker-mss-after-acceptance"},
        direction=Direction.LONG,
    )
    _consume(adapter, (bar, raw))
    frozen_lifecycle = adapter.lifecycle
    frozen_projection = adapter.projection

    update = adapter.consume(mss)

    assert update.ignored is True
    assert adapter.lifecycle == frozen_lifecycle
    assert adapter.projection == frozen_projection
    assert mss.event_id in adapter.known_input_event_ids
    assert adapter.lifecycle.structure(incumbent_id).mss_event_ids == ()
    replayed = CanonicalFoundationAdapter.replay(
        (*events, bar, raw, mss),
        tick_size=TICK,
    )
    assert replayed.lifecycle == adapter.lifecycle
    assert replayed.projection == adapter.projection


@pytest.mark.parametrize(
    ("minute", "direction", "label"),
    (
        (3, Direction.SHORT, "same-clock"),
        (4, Direction.LONG, "same-direction"),
    ),
)
def test_nonqualifying_mss_after_protected_terminal_still_fails_closed(
    minute: int,
    direction: Direction,
    label: str,
) -> None:
    adapter, _, facts, _ = _accepted_protected_external_before_mss()
    raw_sequence = 2 if minute == 3 else 1
    mss_sequence = raw_sequence + 1
    if minute == 4:
        adapter.consume(_bar(minute))
    raw = _atomic(
        f"raw-{label}-mss-after-acceptance",
        EventKind.RAW_BOUNDARY_BREAK,
        minute,
        raw_sequence,
        timeframe=Timeframe.H1,
        source_event_ids=(_bar(minute).event_id,),
        direction=direction,
    )
    mss = _atomic(
        f"{label}-mss-after-acceptance",
        EventKind.MSS_CORE_CONFIRMED,
        minute,
        mss_sequence,
        timeframe=Timeframe.H1,
        source_event_ids=(raw.event_id, facts["structure"].event_id),
        source_entity_ids=(
            f"{label}-mss-after-acceptance",
            "structure-source:parent",
        ),
        evidence={"bos_id": f"{label}-mss-after-acceptance"},
        direction=direction,
    )
    adapter.consume(raw)
    frozen = adapter.checkpoint()

    with pytest.raises(
        ValueError,
        match="MSS does not oppose one confirmed external generation",
    ):
        adapter.consume(mss)

    assert adapter.checkpoint() == frozen


@pytest.mark.parametrize(
    ("minute", "direction", "raw_sequence", "label"),
    (
        (3, Direction.LONG, 2, "same-clock"),
        (4, Direction.SHORT, 1, "opposite-latest-direction"),
    ),
)
def test_unbound_mss_outside_latest_protected_terminal_rule_fails_closed(
    minute: int,
    direction: Direction,
    raw_sequence: int,
    label: str,
) -> None:
    adapter, _, facts, _ = _accepted_protected_external_before_mss()
    if minute == 4:
        adapter.consume(_bar(minute))
    raw = _atomic(
        f"raw-unbound-{label}",
        EventKind.RAW_BOUNDARY_BREAK,
        minute,
        raw_sequence,
        timeframe=Timeframe.H1,
        source_event_ids=(_bar(minute).event_id,),
        direction=direction,
    )
    mss = _atomic(
        f"unbound-{label}",
        EventKind.MSS_CORE_CONFIRMED,
        minute,
        raw_sequence + 1,
        timeframe=Timeframe.H1,
        source_event_ids=(raw.event_id, facts["tracker_only"].event_id),
        source_entity_ids=(
            f"unbound-{label}",
            "structure-source:tracker-only-before-acceptance",
        ),
        evidence={"bos_id": f"unbound-{label}"},
        direction=direction,
    )
    adapter.consume(raw)
    frozen = adapter.checkpoint()

    with pytest.raises(
        ValueError,
        match="MSS does not oppose one confirmed external generation",
    ):
        adapter.consume(mss)

    assert adapter.checkpoint() == frozen


def test_unbound_mss_with_live_external_still_fails_closed() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, facts = _seed_structure(adapter)
    tracker_only = _atomic(
        "tracker-only-opposite-live-incumbent",
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        1,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["high"].event_id, facts["low"].event_id),
        evidence={
            "structure_id": "structure-source:tracker-only-live-incumbent",
            "source_high_id": "high:parent",
            "source_low_id": "low:parent",
            "candidate_protected_swing_id": "high:parent",
        },
        direction=Direction.SHORT,
    )
    assert adapter.consume(_bar(1)).ignored is False
    assert adapter.consume(tracker_only).ignored is True
    raw = _atomic(
        "raw-unbound-live-incumbent",
        EventKind.RAW_BOUNDARY_BREAK,
        2,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(_bar(2).event_id,),
        direction=Direction.SHORT,
    )
    mss = _atomic(
        "unbound-live-incumbent",
        EventKind.MSS_CORE_CONFIRMED,
        2,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(raw.event_id, tracker_only.event_id),
        source_entity_ids=(
            "unbound-live-incumbent",
            "structure-source:tracker-only-live-incumbent",
        ),
        evidence={"bos_id": "unbound-live-incumbent"},
        direction=Direction.SHORT,
    )
    _consume(adapter, (_bar(2), raw))
    frozen = adapter.checkpoint()

    with pytest.raises(
        ValueError,
        match="MSS does not oppose one confirmed external generation",
    ):
        adapter.consume(mss)

    assert adapter.checkpoint() == frozen


@pytest.mark.parametrize(
    ("direction", "label"),
    (
        (Direction.LONG, "matches-nonlatest-protected"),
        (Direction.SHORT, "matches-latest-reset"),
    ),
)
def test_unbound_mss_cannot_borrow_nonlatest_or_nonacceptance_terminal(
    direction: Direction,
    label: str,
) -> None:
    adapter, _, facts, _ = _accepted_protected_external_before_mss()
    new_bar = _bar(4)
    new_external = _atomic(
        "new-external-before-reset",
        EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        4,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["high"].event_id, facts["low"].event_id),
        evidence={
            "structure_id": "structure-source:new-external-before-reset",
            "source_high_id": "high:parent",
            "source_low_id": "low:parent",
            "candidate_protected_swing_id": "high:parent",
        },
        direction=Direction.SHORT,
    )
    reset = _atomic(
        "reset-newest-external",
        EventKind.MARKET_EPOCH_RESET,
        5,
        0,
        evidence={"reason": "semantic_reset"},
    )
    _consume(adapter, (new_bar, new_external, reset))
    latest = adapter.lifecycle.structure_generations[-1]
    assert latest.direction is Direction.SHORT
    assert latest.termination_reason == "semantic_reset"
    bar = _bar(6)
    raw = _atomic(
        f"raw-unbound-{label}",
        EventKind.RAW_BOUNDARY_BREAK,
        6,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bar.event_id,),
        direction=direction,
    )
    mss = _atomic(
        f"unbound-{label}",
        EventKind.MSS_CORE_CONFIRMED,
        6,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(raw.event_id, facts["tracker_only"].event_id),
        source_entity_ids=(
            f"unbound-{label}",
            "structure-source:tracker-only-before-acceptance",
        ),
        evidence={"bos_id": f"unbound-{label}"},
        direction=direction,
    )
    _consume(adapter, (bar, raw))
    frozen = adapter.checkpoint()

    with pytest.raises(
        ValueError,
        match="MSS does not oppose one confirmed external generation",
    ):
        adapter.consume(mss)

    assert adapter.checkpoint() == frozen


def test_reset_replay_checkpoint_pickle_and_explicit_boundary_attack() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    level_events, _ = _seed_level(adapter, label="reset-level")
    reset = _atomic(
        "epoch-reset",
        EventKind.MARKET_EPOCH_RESET,
        1,
        0,
        evidence={"reason": "contract_change_reset"},
    )
    adapter.consume(reset)
    assert adapter.lifecycle.epoch == 1
    assert adapter.lifecycle.levels[-1].lifecycle is LiquidityLevelLifecycle.ARCHIVED
    assert (
        adapter.lifecycle.interactions[-1].terminal_state
        is LiquidityInteractionTerminal.CENSORED
    )

    replayed = CanonicalFoundationAdapter.replay(
        (*level_events, reset), tick_size=TICK
    )
    assert replayed.lifecycle == adapter.lifecycle
    assert replayed.projection == adapter.projection

    prefix = CanonicalFoundationAdapter.replay(level_events, tick_size=TICK)
    checkpoint = pickle.loads(pickle.dumps(prefix.checkpoint()))
    resumed = CanonicalFoundationAdapter.restore(checkpoint)
    resumed.consume(reset)
    assert resumed.lifecycle == adapter.lifecycle
    assert resumed.projection == adapter.projection
    pickled_adapter = pickle.loads(pickle.dumps(adapter))
    assert pickled_adapter.lifecycle == adapter.lifecycle
    assert pickled_adapter.projection == adapter.projection

    boundary_adapter = CanonicalFoundationAdapter(tick_size=TICK)
    structure_events, facts = _seed_structure(boundary_adapter)
    h1_bar = _bar(
        1,
        timeframe=Timeframe.H1,
        high=100.5,
        low=99.75,
        close=100.0,
        event_id="boundary-h1-bar",
    )
    boundary_adapter.consume(h1_bar)
    generation = boundary_adapter.lifecycle.structure_generations[-1]
    boundary_fact = NormalizedLifecycleTransition(
        fact_id="explicit-boundary-attack",
        kind=NormalizedTransitionKind.BOUNDARY_ATTACK_OBSERVED,
        known_at=h1_bar.known_at,
        timeframe=Timeframe.H1,
        source_event_ids=(facts["high"].event_id, h1_bar.event_id),
        payload={
            "bos_generation_id": generation.generation_id,
            "direction": Direction.LONG.value,
            "target_swing_event_id": facts["high"].event_id,
            "bar_event_id": h1_bar.event_id,
            "boundary_ticks": 400,
            "high_ticks": 402,
            "low_ticks": 399,
            "close_ticks": 400,
        },
    )
    update = boundary_adapter.observe_boundary_attack(boundary_fact)
    assert len(update.records) == 1
    assert update.lifecycle.boundary_attacks[-1].bar_event_id == h1_bar.event_id

    boundary_record = update.records[0]
    authoritative = {
        facts["high"].event_id: facts["high"],
        h1_bar.event_id: h1_bar,
    }
    forged_evidence = {
        **dict(h1_bar.evidence),
        "real_completed": False,
        "clock_only": True,
    }
    authoritative[h1_bar.event_id] = replace(
        h1_bar,
        details=forged_evidence,
        evidence=forged_evidence,
    )
    with pytest.raises(ValueError, match="Boundary Attack BAR.*exact real BAR"):
        _validate_foundation_authoritative_sources(
            boundary_record,
            authoritative,
        )

    clock_adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, clock_facts = _seed_structure(clock_adapter, label="clock-boundary")
    clock_bar = _bar(
        30,
        timeframe=Timeframe.H1,
        high=100.5,
        low=99.75,
        close=100.0,
        event_id="clock-only-boundary-h1-bar",
        real_completed=False,
    )
    clock_adapter.consume(clock_bar)
    clock_generation = clock_adapter.lifecycle.structure_generations[-1]
    clock_fact = NormalizedLifecycleTransition(
        fact_id="clock-only-explicit-boundary-attack",
        kind=NormalizedTransitionKind.BOUNDARY_ATTACK_OBSERVED,
        known_at=clock_bar.known_at,
        timeframe=Timeframe.H1,
        source_event_ids=(clock_facts["high"].event_id, clock_bar.event_id),
        payload={
            "bos_generation_id": clock_generation.generation_id,
            "direction": Direction.LONG.value,
            "target_swing_event_id": clock_facts["high"].event_id,
            "bar_event_id": clock_bar.event_id,
            "boundary_ticks": 400,
            "high_ticks": 402,
            "low_ticks": 399,
            "close_ticks": 400,
        },
    )
    with pytest.raises(ValueError, match="exact same-clock real BAR"):
        clock_adapter.observe_boundary_attack(clock_fact)


def test_unknown_event_is_noop_but_missing_sources_and_out_of_order_fail_closed() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    bar = _bar(0)
    adapter.consume(bar)
    unknown = _atomic(
        "unknown-fvg",
        EventKind.FVG_CREATED,
        0,
        1,
        source_event_ids=(bar.event_id,),
        evidence={"fvg_id": "fvg-ignored"},
    )
    update = adapter.consume(unknown)
    assert update.ignored is True
    assert update.records == ()

    missing = _atomic(
        "missing-source-level",
        EventKind.LIQUIDITY_LEVEL_CREATED,
        0,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=("not-seen",),
        evidence={
            "level_id": "missing-source",
            "source_kind": "confirmed_swing",
        },
        side="above",
        price=100.0,
    )
    frozen = adapter.checkpoint()
    with pytest.raises(ValueError, match="unseen source events"):
        adapter.consume(missing)
    assert adapter.checkpoint() == frozen

    with pytest.raises(ValueError, match="out of knowledge order"):
        adapter.consume(_bar(-1))
    assert adapter.checkpoint() == frozen


def test_mss_internal_generation_is_explicitly_censored_by_reset() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    structure_events, facts = _seed_structure(adapter, label="reset-structure")
    bar = _bar(1)
    raw = _atomic(
        "raw-mss-reset",
        EventKind.RAW_BOUNDARY_BREAK,
        1,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(bar.event_id,),
    )
    mss = _atomic(
        "mss-reset",
        EventKind.MSS_CORE_CONFIRMED,
        1,
        2,
        timeframe=Timeframe.H1,
        source_event_ids=(raw.event_id, facts["structure"].event_id),
        source_entity_ids=(
            "mss-reset",
            "structure-source:reset-structure",
        ),
        evidence={"bos_id": "mss-reset"},
        direction=Direction.SHORT,
    )
    reset = _atomic(
        "internal-reset",
        EventKind.MARKET_EPOCH_RESET,
        2,
        0,
        evidence={"reason": "semantic_reset"},
    )
    _consume(adapter, (bar, raw, mss))
    internal = next(
        item
        for item in adapter.lifecycle.structure_generations
        if item.scope is StructureScope.INTERNAL
    )
    adapter.consume(reset)

    terminal = adapter.lifecycle.structure(internal.generation_id)
    assert terminal.lifecycle is StructureGenerationLifecycle.TERMINATED
    assert terminal.termination_reason == "semantic_reset"
    assert (
        adapter.lifecycle.structure_transitions[-1].lifecycle
        is StructureTransitionLifecycle.CENSORED
    )
    replayed = CanonicalFoundationAdapter.replay(
        (*structure_events, bar, raw, mss, reset),
        tick_size=TICK,
    )
    assert replayed.lifecycle == adapter.lifecycle
    assert replayed.projection == adapter.projection


def test_append_dto_requires_seen_authoritative_sources_not_future_knowledge() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    bar = _bar(0)
    creation = _atomic(
        "fvg-source",
        EventKind.FVG_CREATED,
        0,
        1,
        source_event_ids=(bar.event_id,),
        evidence={"fvg_id": "fvg-a"},
    )
    _consume(adapter, (bar, creation))
    lifecycle = FVGStructuralLifecycle(
        fvg_id="fvg-a",
        source_creation_event_id=creation.event_id,
        symbol="ES",
        instrument_id=1,
        timeframe=Timeframe.M1,
        created_at=_clock(0),
        known_at=_clock(0),
    )

    update = adapter.append_dto(lifecycle)
    assert len(update.records) == 1
    assert creation.event_id in adapter.known_input_event_ids
    assert adapter.append_dto(lifecycle).ignored is True
    with pytest.raises(ValueError, match="differ from exact DTO ancestry"):
        adapter.append_dto(
            lifecycle,
            source_event_ids=(bar.event_id,),
        )

    geometry = SwingGeometryNode(
        swing_id="geometry-a",
        timeframe=Timeframe.M1,
        symbol="ES",
        instrument_id=1,
        window_start=_clock(0) - pd.Timedelta(1, unit="min"),
        window_end=_clock(0),
        lower_bound=99.0,
        upper_bound=101.0,
        known_at=_clock(0),
        source_candle_ids=("candle:geometry-a",),
    )
    geometry_update = adapter.append_dto(
        geometry,
        source_event_ids=(bar.event_id,),
    )
    assert len(geometry_update.records) == 1

    state_source = MarketEvent(
        event_id="state-projection-fvg-source",
        kind=EventKind.FVG_CREATED,
        observed_at=_clock(0),
        timeframe=Timeframe.M1,
        side="above",
        price=100.0,
        strength=0.0,
        sequence_no=2,
        event_time=_clock(0),
        known_at=_clock(0),
        evidence={"fvg_id": "state-projection-fvg"},
        origin=EventOrigin.STATE_PROJECTION,
    )
    adapter.consume(state_source)
    state_lifecycle = FVGStructuralLifecycle(
        fvg_id="state-projection-fvg",
        source_creation_event_id=state_source.event_id,
        symbol="ES",
        instrument_id=1,
        timeframe=Timeframe.M1,
        created_at=_clock(0),
        known_at=_clock(0),
    )
    with pytest.raises(ValueError, match="non-authoritative DTO sources"):
        adapter.append_dto(state_lifecycle)
    level_from_projection = _atomic(
        "level-from-state-projection",
        EventKind.LIQUIDITY_LEVEL_CREATED,
        0,
        3,
        timeframe=Timeframe.H1,
        source_event_ids=(state_source.event_id,),
        evidence={
            "level_id": "state-projection-level",
            "source_kind": "confirmed_swing",
        },
        side="above",
        price=100.0,
    )
    frozen = adapter.checkpoint()
    with pytest.raises(ValueError, match="non-authoritative"):
        adapter.consume(level_from_projection)
    assert adapter.checkpoint() == frozen

    future_creation = _atomic(
        "future-fvg-source",
        EventKind.FVG_CREATED,
        1,
        1,
        source_event_ids=(creation.event_id,),
        evidence={"fvg_id": "fvg-future"},
    )
    adapter.consume(future_creation)
    future_leak = FVGStructuralLifecycle(
        fvg_id="fvg-future",
        source_creation_event_id=future_creation.event_id,
        symbol="ES",
        instrument_id=1,
        timeframe=Timeframe.M1,
        created_at=_clock(0),
        known_at=_clock(0),
    )
    frozen = adapter.checkpoint()
    with pytest.raises(ValueError, match="predates exact source-event knowledge"):
        adapter.append_dto(future_leak)
    assert adapter.checkpoint() == frozen

    legacy = MarketEvent(
        event_id="legacy-fvg-source",
        kind=EventKind.FVG_CREATED,
        observed_at=_clock(2),
        timeframe=Timeframe.M1,
        side="above",
        price=100.0,
        strength=0.0,
        sequence_no=0,
        evidence={"fvg_id": "legacy-fvg"},
    )
    adapter.consume(legacy)
    legacy_lifecycle = FVGStructuralLifecycle(
        fvg_id="legacy-fvg",
        source_creation_event_id=legacy.event_id,
        symbol="ES",
        instrument_id=1,
        timeframe=Timeframe.M1,
        created_at=_clock(2),
        known_at=_clock(2),
    )
    with pytest.raises(ValueError, match="non-authoritative DTO sources"):
        adapter.append_dto(legacy_lifecycle)

    legacy_swing = MarketEvent(
        event_id="legacy-swing-source",
        kind=EventKind.SWING_CONFIRMED,
        observed_at=_clock(3),
        timeframe=Timeframe.H1,
        side="above",
        price=101.0,
        strength=0.0,
        sequence_no=0,
        evidence={"source_entity_id": "legacy-swing"},
    )
    adapter.consume(legacy_swing)
    level_from_legacy = _atomic(
        "level-from-legacy",
        EventKind.LIQUIDITY_LEVEL_CREATED,
        3,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(legacy_swing.event_id,),
        evidence={
            "level_id": "legacy-level",
            "source_kind": "confirmed_swing",
        },
        side="above",
        price=101.0,
    )
    frozen = adapter.checkpoint()
    with pytest.raises(ValueError, match="legacy transport ancestry"):
        adapter.consume(level_from_legacy)
    assert adapter.checkpoint() == frozen


def test_relation_and_delivery_public_apis_update_then_explicitly_terminate() -> None:
    adapter = CanonicalFoundationAdapter(tick_size=TICK)
    _, parent_facts = _seed_structure(adapter, label="relation-parent")
    _, child_facts = _seed_structure(
        adapter,
        minute=1,
        label="relation-child",
        timeframe=Timeframe.M5,
        direction=Direction.SHORT,
    )
    parent = next(
        item
        for item in adapter.lifecycle.structure_generations
        if item.timeframe is Timeframe.H1
        and item.scope is StructureScope.EXTERNAL
    )
    child = next(
        item
        for item in adapter.lifecycle.structure_generations
        if item.timeframe is Timeframe.M5
        and item.scope is StructureScope.EXTERNAL
    )

    relation_source = _atomic(
        "relation-observation-1",
        EventKind.RELATION_STATE_CHANGED,
        2,
        0,
        timeframe=Timeframe.M5,
        source_event_ids=(
            parent_facts["structure"].event_id,
            child_facts["structure"].event_id,
        ),
    )
    adapter.consume(relation_source)
    relation = RelationState(
        relation_id="1h:5m",
        parent_tf=Timeframe.H1,
        child_tf=Timeframe.M5,
        role=RelationRole.PARENT_RETRACEMENT,
        parent_direction=Direction.LONG,
        child_direction=Direction.SHORT,
        parent_protected_swing_intact=True,
        parent_state_invalidated=False,
        reversal_warning=False,
        child_location_in_parent_range=0.5,
        parent_invalidation_distance_atr=1.0,
        known_at=_clock(2),
    )
    adapter.observe_relation(
        relation,
        parent_structure_generation_id=parent.generation_id,
        child_structure_generation_id=child.generation_id,
        source_event_ids=(relation_source.event_id,),
    )
    relation_generation = adapter.lifecycle.relation_generations[-1]
    assert relation_generation.lifecycle is GenerationLifecycle.ACTIVE
    assert relation_generation.observation_count == 1

    restored = CanonicalFoundationAdapter.restore(
        pickle.loads(pickle.dumps(adapter.checkpoint()))
    )
    assert restored.known_input_event_ids == adapter.known_input_event_ids
    assert restored.lifecycle == adapter.lifecycle
    assert restored.projection == adapter.projection

    relation_source_2 = _atomic(
        "relation-observation-2",
        EventKind.RELATION_STATE_CHANGED,
        3,
        0,
        timeframe=Timeframe.M5,
        source_event_ids=(relation_source.event_id,),
    )
    adapter.consume(relation_source_2)
    relation_2 = RelationState(
        **{
            **relation.__dict__,
            "known_at": _clock(3),
        }
    )
    adapter.observe_relation(
        relation_2,
        parent_structure_generation_id=parent.generation_id,
        child_structure_generation_id=child.generation_id,
        source_event_ids=(relation_source_2.event_id,),
    )
    updated_relation = adapter.lifecycle.relation_generations[-1]
    assert updated_relation.generation_id == relation_generation.generation_id
    assert updated_relation.observation_count == 2

    delivery_source = _atomic(
        "delivery-observation-1",
        EventKind.DELIVERY_PHASE_CHANGED,
        4,
        0,
        timeframe=Timeframe.H1,
        source_event_ids=(parent_facts["structure"].event_id,),
    )
    adapter.consume(delivery_source)
    adapter.observe_delivery_phase(
        DeliveryPhase.RETRACEMENT,
        timeframe=Timeframe.H1,
        known_at=_clock(4),
        parent_structure_generation_id=parent.generation_id,
        origin_event_id=delivery_source.event_id,
        source_event_ids=(delivery_source.event_id,),
        current_price_ticks=400,
    )
    delivery = adapter.lifecycle.delivery_generations[-1]
    assert delivery.lifecycle is GenerationLifecycle.ACTIVE

    delivery_source_2 = _atomic(
        "delivery-observation-2",
        EventKind.DELIVERY_PHASE_CHANGED,
        5,
        0,
        timeframe=Timeframe.H1,
        source_event_ids=(delivery_source.event_id,),
    )
    adapter.consume(delivery_source_2)
    adapter.observe_delivery_phase(
        DeliveryPhase.RETRACEMENT,
        timeframe=Timeframe.H1,
        known_at=_clock(5),
        parent_structure_generation_id=parent.generation_id,
        origin_event_id=delivery_source.event_id,
        source_event_ids=(delivery_source.event_id, delivery_source_2.event_id),
        current_price_ticks=404,
    )
    updated_delivery = adapter.lifecycle.delivery_generations[-1]
    assert updated_delivery.generation_id == delivery.generation_id
    assert updated_delivery.observation_count == 2
    assert updated_delivery.max_extension_ticks == 4.0

    relation_terminal_source = _atomic(
        "relation-terminal-source",
        EventKind.RELATION_STATE_CHANGED,
        6,
        0,
        timeframe=Timeframe.M5,
        source_event_ids=(relation_source_2.event_id,),
    )
    adapter.consume(relation_terminal_source)
    frozen = adapter.checkpoint()
    with pytest.raises(ValueError, match="not preregistered"):
        adapter.terminate_relation(
            relation_generation_id=relation_generation.generation_id,
            known_at=_clock(6),
            reason="ad_hoc",
            source_event_ids=(relation_terminal_source.event_id,),
        )
    assert adapter.checkpoint() == frozen
    adapter.terminate_relation(
        relation_generation_id=relation_generation.generation_id,
        known_at=_clock(6),
        reason="parent_rollover",
        source_event_ids=(relation_terminal_source.event_id,),
    )

    delivery_terminal_source = _atomic(
        "delivery-terminal-source",
        EventKind.DELIVERY_PHASE_CHANGED,
        6,
        1,
        timeframe=Timeframe.H1,
        source_event_ids=(delivery_source_2.event_id,),
    )
    adapter.consume(delivery_terminal_source)
    adapter.terminate_delivery(
        delivery_generation_id=delivery.generation_id,
        known_at=_clock(6),
        reason="parent_structure_terminated",
        source_event_ids=(delivery_terminal_source.event_id,),
    )

    assert (
        adapter.lifecycle.relation_generations[-1].lifecycle
        is GenerationLifecycle.TERMINATED
    )
    assert (
        adapter.lifecycle.delivery_generations[-1].lifecycle
        is GenerationLifecycle.TERMINATED
    )
