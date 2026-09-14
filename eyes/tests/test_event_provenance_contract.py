from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
from typing import Mapping

import pandas as pd
import pytest

import shares
import eyes.core.event_store as event_store_module
from eyes.core.event_store import (
    EventStore,
    read_event_journal,
    validate_canonical_event,
    write_event_journal,
)
from eyes.core.foundation_registry import FOUNDATION_VERSION
from eyes.core.market_state import TimeframeEventReducer, reduce_timeframe_state
from contract.market import (
    Direction,
    SMC_SEMANTIC_VERSION,
    Timeframe,
    to_primitive,
)
from contract.eye import (
    EventKind,
    EventOrigin,
    MarketEvent,
)
from eyes.core.semantics import SemanticRegistry


TZ = "America/New_York"


def test_event_store_uses_new_public_identity_and_reads_legacy_pickle() -> None:
    store = EventStore()
    current_pickle = pickle.dumps(store, protocol=0)
    current_global = b"ceyes.core.event_store\nEventStore\n"
    legacy_global = b"ceyes.core.event_store\nImmutableEventStore\n"

    assert EventStore.__module__ == "eyes.core.event_store"
    assert EventStore.__qualname__ == "EventStore"
    assert current_global in current_pickle
    assert "EventStore" in event_store_module.__all__
    assert "ImmutableEventStore" not in event_store_module.__all__
    assert not hasattr(shares, "ImmutableEventStore")

    legacy_pickle = current_pickle.replace(current_global, legacy_global, 1)
    restored = pickle.loads(legacy_pickle)

    assert type(restored) is EventStore
    assert restored == store


def _clock(minutes: int) -> pd.Timestamp:
    return pd.Timestamp("2025-01-06 10:00", tz=TZ) + pd.Timedelta(
        minutes * 60,
        unit="s",
    )


def _event(
    event_id: str,
    minutes: int,
    *,
    canonical: bool = False,
    projection: bool = False,
    source_ids: tuple[str, ...] = (),
    source_event_ids: tuple[str, ...] = (),
    source_data_ids: tuple[str, ...] = (),
    source_entity_ids: tuple[str, ...] = (),
    context_event_ids: tuple[str, ...] = (),
    semantic_version: str = SMC_SEMANTIC_VERSION,
    kind: EventKind | None = None,
    timeframe: Timeframe = Timeframe.M5,
    details: Mapping[str, object] | None = None,
    price: float | None = 100.0,
    zone: tuple[float, float] | None = None,
    transition_reason: str | None = None,
    origin: EventOrigin | None = None,
    direction: Direction | None = None,
    side: str | None = "above",
    event_time_minutes: int | None = None,
    sequence_no: int = 0,
) -> MarketEvent:
    event_details = dict(details or {})
    if canonical:
        event_details.update(
            canonical_semantic=True,
            projection_only=projection,
        )
    if projection:
        projection_state = {"state_id": event_id, "value": 1}
        payload = json.dumps(
            to_primitive(projection_state),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        event_details.update(
            state_id=event_id,
            projection_state=projection_state,
            projection_sha256=hashlib.sha256(payload).hexdigest(),
        )
    clock = _clock(minutes)
    event_time = _clock(
        minutes if event_time_minutes is None else event_time_minutes
    )
    return MarketEvent(
        event_id=event_id,
        kind=(
            EventKind.TIMEFRAME_STATE_CHANGED
            if projection
            else (kind or EventKind.LIQUIDITY_CONSUMED)
        ),
        observed_at=clock,
        timeframe=timeframe,
        side=side,
        price=price,
        strength=0.5,
        source_ids=source_ids,
        details=event_details,
        transition_reason=transition_reason,
        direction=direction,
        sequence_no=sequence_no,
        event_time=event_time,
        known_at=clock,
        semantic_version=semantic_version,
        zone=zone,
        source_event_ids=source_event_ids,
        source_data_ids=source_data_ids,
        source_entity_ids=source_entity_ids,
        context_event_ids=context_event_ids,
        origin=origin
        or (
            EventOrigin.STATE_PROJECTION
            if projection
            else (
                EventOrigin.SEMANTIC_ATOMIC
                if canonical
                else EventOrigin.LEGACY_TRANSPORT
            )
        ),
    )


def _normalized_bar(
    event_id: str,
    minutes: int,
    *,
    timeframe: Timeframe = Timeframe.M5,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    close: float = 100.0,
    atr: float = 2.0,
    real_completed: bool = True,
    symbol: str = "NQH5",
    instrument_id: int = 750,
    sequence_no: int = 0,
) -> MarketEvent:
    bar_open = close if open_ is None else open_
    bar_high = max(bar_open, close) + 1.0 if high is None else high
    bar_low = min(bar_open, close) - 1.0 if low is None else low
    detector_id = f"detector:{event_id}"
    data_id = f"data:{event_id}"
    details = {
        "detector_candle_id": detector_id,
        "open": bar_open,
        "high": bar_high,
        "low": bar_low,
        "close": close,
        "volume": 100.0 if real_completed else 0.0,
        "atr": atr,
        "data_complete": True,
        "real_completed": real_completed,
        "clock_only": not real_completed,
        "symbol": symbol,
        "instrument_id": instrument_id,
    }
    if not real_completed:
        timeframe_minutes = {
            Timeframe.M1: 1,
            Timeframe.M5: 5,
            Timeframe.M15: 15,
            Timeframe.H1: 60,
            Timeframe.H4: 240,
        }[timeframe]
        details.update(
            complete=True,
            start=_clock(minutes)
            - pd.Timedelta(timeframe_minutes, unit="min"),
            observed_minutes=timeframe_minutes,
            expected_minutes=timeframe_minutes,
            real_minutes=timeframe_minutes - 1,
            synthetic_minutes=1,
        )
    return _event(
        event_id,
        minutes,
        kind=EventKind.BAR_COMPLETED,
        timeframe=timeframe,
        details=details,
        price=close,
        origin=EventOrigin.NORMALIZED_DATA,
        source_data_ids=(data_id,),
        sequence_no=sequence_no,
    )


def _crossing_generation_id(
    *,
    level_id: str,
    timeframe: Timeframe,
    crossed_at: pd.Timestamp,
) -> str:
    payload = (
        f"{SMC_SEMANTIC_VERSION}|crossing-v1|{timeframe.value}|"
        f"{level_id}|{crossed_at.isoformat()}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _legacy_range_boundary_candidate(
    *,
    prefix: str,
    minutes: int,
    timeframe: Timeframe,
    side: str,
    level_id: str,
    price: float = 100.0,
) -> tuple[MarketEvent, MarketEvent]:
    source_id = f"range-source:{prefix}:{level_id}"
    zone = (price, price)
    context = replace(
        _event(
            f"{prefix}-range-state",
            minutes,
            kind=EventKind.SUPPORT_RESISTANCE_STATE,
            timeframe=timeframe,
            side=side,
            price=price,
            zone=zone,
            source_ids=(level_id, source_id),
            details={
                "source_kind": "range_boundary",
                "source_ids": (source_id,),
            },
            sequence_no=0,
        ),
        formed_at=_clock(minutes),
    )
    candidate = _event(
        f"{prefix}-candidate",
        minutes,
        canonical=True,
        kind=EventKind.LIQUIDITY_LEVEL_CREATED,
        timeframe=timeframe,
        side=side,
        price=price,
        details={
            "level_id": level_id,
            "candidate_only": True,
            "source_kind": "range_boundary",
            "source_ids": (source_id,),
        },
        source_entity_ids=(level_id, source_id),
        context_event_ids=(context.event_id,),
        zone=zone,
        sequence_no=1,
    )
    return context, candidate


def _crossing_chain(
    *,
    prefix: str = "crossing",
    kind: EventKind = EventKind.ACCEPTANCE_CONFIRMED,
    timeframe: Timeframe = Timeframe.M5,
    candidate_timeframe: Timeframe | None = None,
    side: str = "above",
    level_id: str = "level-1",
    crossed_minutes: int = 2,
    resolved_minutes: int = 3,
    candidate_minutes: int = 0,
    terminal_details: Mapping[str, object] | None = None,
) -> tuple[MarketEvent, ...]:
    candidate_timeframe = candidate_timeframe or timeframe
    crossed_at = _clock(crossed_minutes)
    resolved_at = _clock(resolved_minutes)
    crossing_direction = (
        Direction.LONG if side == "above" else Direction.SHORT
    )
    terminal_direction = (
        crossing_direction
        if kind is EventKind.ACCEPTANCE_CONFIRMED
        else (
            Direction.SHORT
            if crossing_direction is Direction.LONG
            else Direction.LONG
        )
    )
    generation_id = _crossing_generation_id(
        level_id=level_id,
        timeframe=timeframe,
        crossed_at=crossed_at,
    )
    range_context, candidate = _legacy_range_boundary_candidate(
        prefix=prefix,
        minutes=candidate_minutes,
        timeframe=candidate_timeframe,
        side=side,
        level_id=level_id,
    )
    terminal_close = (
        (101.0 if side == "above" else 99.0)
        if kind is EventKind.ACCEPTANCE_CONFIRMED
        else 100.0
    )
    crossing_bar = _normalized_bar(
        f"{prefix}-crossing-bar",
        crossed_minutes,
        timeframe=timeframe,
        close=(terminal_close if resolved_minutes == crossed_minutes else 100.0),
        sequence_no=0,
    )
    touch = _event(
        f"{prefix}-touch",
        crossed_minutes,
        canonical=True,
        kind=EventKind.LEVEL_TOUCHED,
        timeframe=timeframe,
        side=side,
        details={
            "level_id": level_id,
            **(
                {"source_timeframe": candidate_timeframe.value}
                if candidate_timeframe is not timeframe
                else {}
            ),
        },
        source_event_ids=(candidate.event_id, crossing_bar.event_id),
        sequence_no=1,
    )
    penetration = _event(
        f"{prefix}-penetration",
        crossed_minutes,
        canonical=True,
        kind=EventKind.LEVEL_PENETRATED,
        timeframe=timeframe,
        side=side,
        direction=crossing_direction,
        details={
            "level_id": level_id,
            "crossing_generation_id": generation_id,
            "crossed_at": crossed_at.isoformat(),
            **(
                {"source_timeframe": candidate_timeframe.value}
                if candidate_timeframe is not timeframe
                else {}
            ),
        },
        source_event_ids=(
            candidate.event_id,
            touch.event_id,
            crossing_bar.event_id,
        ),
        sequence_no=2,
    )
    if resolved_minutes == crossed_minutes:
        resolution_bar = crossing_bar
        terminal_sequence = 3
        parents = (
            range_context,
            candidate,
            crossing_bar,
            touch,
            penetration,
        )
    else:
        resolution_bar = _normalized_bar(
            f"{prefix}-resolution-bar",
            resolved_minutes,
            timeframe=timeframe,
            close=terminal_close,
            sequence_no=0,
        )
        terminal_sequence = 1
        parents = (
            range_context,
            candidate,
            crossing_bar,
            touch,
            penetration,
            resolution_bar,
        )
    terminal = _event(
        f"{prefix}-terminal",
        resolved_minutes,
        canonical=True,
        kind=kind,
        timeframe=timeframe,
        side=side,
        direction=terminal_direction,
        details={
            **dict(terminal_details or {}),
            "level_id": level_id,
            "crossing_generation_id": generation_id,
            "crossed_at": crossed_at.isoformat(),
            "resolved_at": resolved_at.isoformat(),
            **(
                {"source_timeframe": candidate_timeframe.value}
                if candidate_timeframe is not timeframe
                else {}
            ),
        },
        source_event_ids=(penetration.event_id, resolution_bar.event_id),
        source_entity_ids=(level_id,),
        event_time_minutes=crossed_minutes,
        sequence_no=terminal_sequence,
    )
    return (*parents, terminal)


def _with_evidence(
    event: MarketEvent,
    **changes: object,
) -> MarketEvent:
    evidence = {**dict(event.evidence), **changes}
    if (
        event.kind is EventKind.BAR_COMPLETED
        and event.origin is EventOrigin.NORMALIZED_DATA
        and evidence.get("real_completed") is False
        and evidence.get("clock_only") is True
        and "start" not in evidence
    ):
        timeframe_minutes = {
            Timeframe.M1: 1,
            Timeframe.M5: 5,
            Timeframe.M15: 15,
            Timeframe.H1: 60,
            Timeframe.H4: 240,
        }[event.timeframe]
        evidence.update(
            complete=True,
            start=event.known_at
            - pd.Timedelta(timeframe_minutes, unit="min"),
            observed_minutes=timeframe_minutes,
            expected_minutes=timeframe_minutes,
            real_minutes=timeframe_minutes - 1,
            synthetic_minutes=1,
        )
    return replace(event, details=evidence, evidence=evidence)


def _foundation_structural_leg_contract(
    *,
    session_gap_minutes: int = 0,
) -> tuple[
    MarketEvent,
    dict[str, MarketEvent],
]:
    # The registered calendar truncates the bucket that closes at the daily
    # maintenance break and restarts one gap later, so the BAR closing a pivot
    # can sit more than one stride after the pivot clock.
    gap = session_gap_minutes
    prior_bars = tuple(
        _normalized_bar(
            f"foundation-prior-{minute}",
            minute,
            timeframe=Timeframe.M1,
            open_=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
        )
        for minute in range(1, 15)
    )
    path_bars = (
        _normalized_bar(
            "foundation-path-15",
            15 + gap,
            timeframe=Timeframe.M1,
            open_=100.0,
            high=101.0,
            low=98.0,
            close=100.0,
        ),
        _normalized_bar(
            "foundation-path-16",
            16 + gap,
            timeframe=Timeframe.M1,
            open_=100.0,
            high=102.0,
            low=98.5,
            close=99.5,
        ),
        _normalized_bar(
            "foundation-path-17",
            17 + gap,
            timeframe=Timeframe.M1,
            open_=99.5,
            high=103.0,
            low=97.5,
            close=102.0,
        ),
        _normalized_bar(
            "foundation-path-18",
            18 + gap,
            timeframe=Timeframe.M1,
            open_=102.0,
            high=105.0,
            low=101.0,
            close=104.0,
        ),
    )
    confirmation_bar = _normalized_bar(
        "foundation-confirmation-19",
        19 + gap,
        timeframe=Timeframe.M1,
        open_=104.0,
        high=104.5,
        low=102.0,
        close=103.0,
    )
    start_swing = _event(
        "foundation-start-swing",
        16 + gap,
        canonical=True,
        kind=EventKind.SWING_CONFIRMED,
        timeframe=Timeframe.M1,
        side="below",
        price=98.0,
        event_time_minutes=14,
        source_event_ids=(
            prior_bars[-1].event_id,
            path_bars[0].event_id,
            path_bars[1].event_id,
        ),
        source_entity_ids=("foundation-start",),
        details={
            "source_entity_id": "foundation-start",
            "side": "low",
            "relation": "none",
            "pivot_start": _clock(14).isoformat(),
            "pivot_end": _clock(15 + gap).isoformat(),
            "prominence_atr": 1.0,
            "legacy_same_side_magnitude_atr": 0.0,
            "confirmation_delay_bars": 1,
            "confirmation_delay_minutes": 2 + gap,
            "nesting_depth": 0,
            "semantic_rank": "micro",
            "delta_ticks": 0,
        },
    )
    end_swing = _event(
        "foundation-end-swing",
        19 + gap,
        canonical=True,
        kind=EventKind.SWING_CONFIRMED,
        timeframe=Timeframe.M1,
        side="above",
        price=105.0,
        event_time_minutes=17 + gap,
        source_event_ids=(
            path_bars[2].event_id,
            path_bars[3].event_id,
            confirmation_bar.event_id,
        ),
        source_entity_ids=("foundation-end",),
        details={
            "source_entity_id": "foundation-end",
            "side": "high",
            "relation": "none",
            "pivot_start": _clock(17 + gap).isoformat(),
            "pivot_end": _clock(18 + gap).isoformat(),
            "prominence_atr": 1.0,
            "legacy_same_side_magnitude_atr": 0.0,
            "confirmation_delay_bars": 1,
            "confirmation_delay_minutes": 2,
            "nesting_depth": 0,
            "semantic_rank": "micro",
            "delta_ticks": 0,
        },
    )
    atr_candle_ids = tuple(
        str(bar.evidence["detector_candle_id"]) for bar in prior_bars
    )
    path_candle_ids = tuple(
        str(bar.evidence["detector_candle_id"]) for bar in path_bars
    )
    leg = _event(
        "foundation-structural-leg",
        19 + gap,
        canonical=True,
        kind=EventKind.STRUCTURAL_LEG_CREATED,
        timeframe=Timeframe.M1,
        direction=Direction.LONG,
        side="above",
        price=105.0,
        event_time_minutes=17 + gap,
        source_event_ids=(start_swing.event_id, end_swing.event_id),
        source_data_ids=(*atr_candle_ids, *path_candle_ids),
        source_entity_ids=(
            "foundation-leg",
            "foundation-start",
            "foundation-end",
        ),
        context_event_ids=tuple(
            bar.event_id for bar in (*prior_bars, *path_bars)
        ),
        details={
            "path_class": "internal",
            "leg_id": "foundation-leg",
            "start_swing_id": "foundation-start",
            "end_swing_id": "foundation-end",
            "start_event_time": _clock(14).isoformat(),
            "end_event_time": _clock(17 + gap).isoformat(),
            "start_price": 98.0,
            "end_price": 105.0,
            "start_close": 100.0,
            "end_close": 104.0,
            "amplitude_points": 7.0,
            "amplitude_ticks": 28,
            "amplitude_atr": 3.5,
            "atr_at_leg_start": 2.0,
            "duration_bars": 4,
            "duration_minutes": 3 + gap,
            "duration_seconds": 180 + gap * 60,
            "efficiency": 0.8,
            "close_efficiency": 0.8,
            "extreme_path_efficiency": 1.0,
            "max_retracement_points": 0.5,
            "max_retracement_atr": 0.25,
            "close_mae_points": 0.5,
            "close_mae_atr": 0.25,
            "wick_mae_points": 0.5,
            "wick_mae_atr": 0.25,
            "path_candle_ids": path_candle_ids,
            "atr_source_candle_ids": atr_candle_ids,
            "foundation_version": FOUNDATION_VERSION,
            "tick_size": 0.25,
            "symbol": "NQH5",
            "instrument_id": 750,
            "rank": "internal",
        },
    )
    leg = replace(leg, strength=0.8)
    available = {
        event.event_id: event
        for event in (
            *prior_bars,
            *path_bars,
            confirmation_bar,
            start_swing,
            end_swing,
        )
    }
    return leg, available


def test_foundation_structural_leg_binds_production_bar_ancestry() -> None:
    leg, available = _foundation_structural_leg_contract()

    validate_canonical_event(leg, available_events=available)
    ordered = tuple(
        sorted(
            (*available.values(), leg),
            key=lambda event: (
                event.known_at,
                event.sequence_no,
                event.event_id,
            ),
        )
    )
    restored = EventStore.from_events(ordered)
    assert restored.get(leg.event_id) == leg


def test_structural_leg_foundation_extension_preserves_v1_2_shape() -> None:
    leg, available = _foundation_structural_leg_contract()
    foundation_names = {
        "amplitude_ticks",
        "atr_at_leg_start",
        "atr_source_candle_ids",
        "close_efficiency",
        "close_mae_atr",
        "close_mae_points",
        "duration_seconds",
        "extreme_path_efficiency",
        "foundation_version",
        "instrument_id",
        "path_candle_ids",
        "symbol",
        "tick_size",
        "wick_mae_atr",
        "wick_mae_points",
    }
    legacy_evidence = {
        key: value
        for key, value in leg.evidence.items()
        if key not in foundation_names
    }
    legacy = replace(
        leg,
        details=legacy_evidence,
        evidence=legacy_evidence,
        source_data_ids=(),
        context_event_ids=(),
    )

    validate_canonical_event(legacy, available_events=available)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("amplitude_ticks", 27),
        ("amplitude_points", 7.25),
        ("amplitude_atr", 3.25),
        ("atr_at_leg_start", 2.25),
        ("duration_seconds", 181),
        ("duration_minutes", 4),
        ("efficiency", 0.7),
        ("close_efficiency", 0.7),
        ("extreme_path_efficiency", 0.9),
        ("max_retracement_points", 0.75),
        ("max_retracement_atr", 0.5),
        ("close_mae_points", 0.75),
        ("close_mae_atr", 0.5),
        ("wick_mae_points", 0.75),
        ("wick_mae_atr", 0.5),
        ("tick_size", 0.5),
        ("symbol", "ESM4"),
        ("instrument_id", 751),
    ),
)
def test_foundation_structural_leg_rejects_tampered_metrics(
    field: str,
    value: object,
) -> None:
    leg, available = _foundation_structural_leg_contract()

    with pytest.raises(ValueError, match="foundation structural leg"):
        validate_canonical_event(
            _with_evidence(leg, **{field: value}),
            available_events=available,
        )


@pytest.mark.parametrize("mutation", ("missing", "reordered"))
def test_foundation_structural_leg_rejects_incomplete_or_unordered_bars(
    mutation: str,
) -> None:
    leg, available = _foundation_structural_leg_contract()
    context = list(leg.context_event_ids)
    if mutation == "missing":
        context.pop(0)
    else:
        context[0], context[1] = context[1], context[0]
    forged = replace(leg, context_event_ids=tuple(context))

    with pytest.raises(ValueError, match="foundation structural leg"):
        validate_canonical_event(forged, available_events=available)


@pytest.mark.parametrize("mutation", ("duplicate", "forged"))
def test_foundation_structural_leg_rejects_self_reported_atr_candle_ids(
    mutation: str,
) -> None:
    leg, available = _foundation_structural_leg_contract()
    identities = list(leg.evidence["atr_source_candle_ids"])
    identities[1] = identities[0] if mutation == "duplicate" else "forged-id"
    forged = _with_evidence(
        leg,
        atr_source_candle_ids=tuple(identities),
    )

    with pytest.raises(ValueError, match="foundation structural leg"):
        validate_canonical_event(forged, available_events=available)


def test_foundation_structural_leg_rejects_nonlatest_prior_atr_bar() -> None:
    leg, available = _foundation_structural_leg_contract()
    older = _normalized_bar(
        "foundation-older-prior",
        0,
        timeframe=Timeframe.M1,
    )
    available[older.event_id] = older
    contexts = list(leg.context_event_ids)
    contexts[0] = older.event_id
    atr_candle_ids = list(leg.evidence["atr_source_candle_ids"])
    atr_candle_ids[0] = str(older.evidence["detector_candle_id"])
    source_data_ids = list(leg.source_data_ids)
    source_data_ids[0] = atr_candle_ids[0]
    forged = _with_evidence(
        replace(
            leg,
            context_event_ids=tuple(contexts),
            source_data_ids=tuple(source_data_ids),
        ),
        atr_source_candle_ids=tuple(atr_candle_ids),
    )

    with pytest.raises(ValueError, match="exact strict-prior"):
        validate_canonical_event(forged, available_events=available)


@pytest.mark.parametrize(
    ("timeframe", "minutes", "match"),
    (
        (Timeframe.M5, 18, "real normalized BAR root"),
        (Timeframe.M1, 20, "causally ordered"),
    ),
)
def test_foundation_structural_leg_rejects_wrong_timeframe_or_future_bar(
    timeframe: Timeframe,
    minutes: int,
    match: str,
) -> None:
    leg, available = _foundation_structural_leg_contract()
    alien = _normalized_bar(
        f"foundation-alien-{timeframe.value}-{minutes}",
        minutes,
        timeframe=timeframe,
    )
    available[alien.event_id] = alien
    contexts = list(leg.context_event_ids)
    contexts[-1] = alien.event_id
    path_candle_ids = list(leg.evidence["path_candle_ids"])
    path_candle_ids[-1] = str(alien.evidence["detector_candle_id"])
    source_data_ids = list(leg.source_data_ids)
    source_data_ids[-1] = path_candle_ids[-1]
    forged = _with_evidence(
        replace(
            leg,
            context_event_ids=tuple(contexts),
            source_data_ids=tuple(source_data_ids),
        ),
        path_candle_ids=tuple(path_candle_ids),
    )

    with pytest.raises(ValueError, match=match):
        validate_canonical_event(forged, available_events=available)


def test_foundation_structural_leg_spans_a_truncated_session_bucket() -> None:
    """A leg stays valid when a session break widens the endpoint stride."""

    leg, available = _foundation_structural_leg_contract(session_gap_minutes=3)
    start = available[leg.source_event_ids[0]]
    pivot_bar = available[start.source_event_ids[1]]

    # the BAR closing the start pivot is four minutes after the pivot clock,
    # not the one arithmetic stride the contract used to assume
    assert pivot_bar.known_at - start.event_time == pd.Timedelta(minutes=4)

    validate_canonical_event(leg, available_events=available)


def test_foundation_structural_leg_rejects_endpoint_swing_pivot_drift() -> None:
    leg, available = _foundation_structural_leg_contract()
    start = available[leg.source_event_ids[0]]
    drifted_sources = (
        start.source_event_ids[0],
        leg.context_event_ids[12],
        start.source_event_ids[2],
    )
    available[start.event_id] = replace(
        start,
        source_ids=drifted_sources,
        source_event_ids=drifted_sources,
    )

    with pytest.raises(ValueError, match="path endpoints"):
        validate_canonical_event(leg, available_events=available)


def _authoritative_phase23_chain() -> tuple[MarketEvent, ...]:
    bar_left = _normalized_bar(
        "bar-left", 0, close=102.0, high=104.0, low=101.0
    )
    bar_middle = _normalized_bar(
        "bar-middle", 1, close=101.0, high=103.0, low=100.0
    )
    bar_right = _normalized_bar(
        "bar-right", 2, close=100.0, high=102.0, low=99.0
    )
    bar_three = _normalized_bar(
        "bar-three", 3, close=103.0, high=104.0, low=100.0
    )
    bar_four = _normalized_bar(
        "bar-four", 4, close=104.0, high=105.0, low=101.0
    )
    swing_left = _event(
        "swing-left",
        4,
        canonical=True,
        kind=EventKind.SWING_CONFIRMED,
        side="below",
        price=99.0,
        event_time_minutes=2,
        sequence_no=1,
        source_event_ids=(
            bar_left.event_id,
            bar_middle.event_id,
            bar_right.event_id,
            bar_three.event_id,
            bar_four.event_id,
        ),
        source_entity_ids=("swing-a",),
        details={
            "source_entity_id": "swing-a",
            "side": "low",
            "relation": "none",
            "pivot_start": _clock(2).isoformat(),
            "pivot_end": _clock(2).isoformat(),
            "prominence_atr": 2.5,
            "legacy_same_side_magnitude_atr": 0.0,
            "confirmation_delay_bars": 2,
            "nesting_depth": 0,
            "semantic_rank": "micro",
            "delta_ticks": 0,
            "confirmation_delay_minutes": 2,
        },
    )
    bar_five = _normalized_bar(
        "bar-five", 5, close=103.0, high=104.0, low=100.0
    )
    bar_six = _normalized_bar(
        "bar-six", 6, close=102.0, high=103.0, low=99.5
    )
    swing_right = _event(
        "swing-right",
        6,
        canonical=True,
        kind=EventKind.SWING_CONFIRMED,
        side="above",
        price=105.0,
        event_time_minutes=4,
        sequence_no=1,
        source_event_ids=(
            bar_right.event_id,
            bar_three.event_id,
            bar_four.event_id,
            bar_five.event_id,
            bar_six.event_id,
        ),
        source_entity_ids=("swing-b",),
        details={
            "source_entity_id": "swing-b",
            "side": "high",
            "relation": "none",
            "pivot_start": _clock(4).isoformat(),
            "pivot_end": _clock(4).isoformat(),
            "prominence_atr": 2.75,
            "legacy_same_side_magnitude_atr": 0.0,
            "confirmation_delay_bars": 2,
            "nesting_depth": 0,
            "semantic_rank": "micro",
            "delta_ticks": 0,
            "confirmation_delay_minutes": 2,
        },
    )
    direction = _event(
        "structure-direction",
        6,
        canonical=True,
        kind=EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        direction=Direction.LONG,
        side="above",
        price=99.0,
        sequence_no=2,
        source_event_ids=(swing_right.event_id, swing_left.event_id),
        source_entity_ids=("structure-a", "swing-b", "swing-a"),
        details={
            "structure_id": "structure-a",
            "direction": "long",
            "source_high_id": "swing-b",
            "source_low_id": "swing-a",
            "sequence_count": 2,
            "candidate_protected_swing_id": "swing-a",
        },
    )
    leg = _event(
        "structural-leg",
        6,
        canonical=True,
        kind=EventKind.STRUCTURAL_LEG_CREATED,
        direction=Direction.LONG,
        side="above",
        price=105.0,
        event_time_minutes=4,
        sequence_no=3,
        source_event_ids=(swing_left.event_id, swing_right.event_id),
        source_entity_ids=("leg-a", "swing-a", "swing-b"),
        details={
            "path_class": "internal",
            "leg_id": "leg-a",
            "start_swing_id": "swing-a",
            "end_swing_id": "swing-b",
            "start_event_time": _clock(2).isoformat(),
            "end_event_time": _clock(4).isoformat(),
            "start_price": 99.0,
            "end_price": 105.0,
            "start_close": 100.0,
            "end_close": 104.0,
            "amplitude_points": 6.0,
            "amplitude_atr": 3.0,
            "duration_bars": 2,
            "duration_minutes": 2,
            "efficiency": 0.8,
            "max_retracement_points": 1.0,
            "max_retracement_atr": 0.5,
            "rank": "internal",
        },
    )
    break_bar = _normalized_bar(
        "bar-break",
        7,
        close=106.0,
        high=107.0,
        low=102.0,
        sequence_no=0,
    )
    raw_break = _event(
        "raw-break",
        7,
        canonical=True,
        kind=EventKind.RAW_BOUNDARY_BREAK,
        direction=Direction.LONG,
        side="above",
        price=105.0,
        sequence_no=1,
        source_event_ids=(swing_right.event_id, break_bar.event_id),
        source_data_ids=("detector:bar-break",),
        source_entity_ids=("bos-a", "swing-b", "structure-a"),
        details={
            "bos_id": "bos-a",
            "target_swing_id": "swing-b",
            "scope": "continuation",
            "break_bar_id": "detector:bar-break",
            "break_close": 106.0,
            "break_buffer_ticks": 0,
            "comparison": "strict_close_beyond",
            "break_standard": "close_beyond_confirmed_boundary",
        },
    )
    qualified_bos = _event(
        "qualified-bos",
        7,
        canonical=True,
        kind=EventKind.QUALIFIED_BOS,
        direction=Direction.LONG,
        side="above",
        price=105.0,
        sequence_no=2,
        source_event_ids=(raw_break.event_id, direction.event_id),
        source_entity_ids=("bos-a", "structure-a"),
        details={
            "bos_id": "bos-a",
            "scope": "continuation",
            "qualification": "aligned_with_confirmed_structure",
        },
    )
    protected = _event(
        "protected-swing",
        7,
        canonical=True,
        kind=EventKind.PROTECTED_SWING_ASSIGNED,
        direction=Direction.LONG,
        side="below",
        price=99.0,
        event_time_minutes=2,
        sequence_no=3,
        source_event_ids=(
            qualified_bos.event_id,
            leg.event_id,
            swing_left.event_id,
        ),
        source_entity_ids=("bos-a", "structure-a", "leg-a", "swing-a"),
        details={
            "bos_id": "bos-a",
            "structure_id": "structure-a",
            "origin_leg_id": "leg-a",
            "protected_swing_id": "swing-a",
            "break_standard": "later_acceptance_beyond",
        },
    )
    opposed_direction = _event(
        "structure-direction-opposed",
        8,
        canonical=True,
        kind=EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        direction=Direction.SHORT,
        side="below",
        price=105.0,
        source_event_ids=(swing_right.event_id, swing_left.event_id),
        source_entity_ids=("structure-b", "swing-b", "swing-a"),
        details={
            "structure_id": "structure-b",
            "direction": "short",
            "source_high_id": "swing-b",
            "source_low_id": "swing-a",
            "sequence_count": 2,
            "candidate_protected_swing_id": "swing-b",
        },
        sequence_no=0,
    )
    mss_break_bar = _normalized_bar(
        "bar-mss-break",
        9,
        close=106.0,
        high=107.0,
        low=103.0,
        sequence_no=0,
    )
    raw_mss = _event(
        "raw-break-mss",
        9,
        canonical=True,
        kind=EventKind.RAW_BOUNDARY_BREAK,
        direction=Direction.LONG,
        side="above",
        price=105.0,
        source_event_ids=(swing_right.event_id, mss_break_bar.event_id),
        source_data_ids=("detector:bar-mss-break",),
        source_entity_ids=("bos-b", "swing-b", "structure-b"),
        details={
            "bos_id": "bos-b",
            "target_swing_id": "swing-b",
            "scope": "opposed",
            "break_bar_id": "detector:bar-mss-break",
            "break_close": 106.0,
            "break_buffer_ticks": 0,
            "comparison": "strict_close_beyond",
            "break_standard": "close_beyond_confirmed_boundary",
        },
        sequence_no=1,
    )
    mss = _event(
        "mss-core",
        9,
        canonical=True,
        kind=EventKind.MSS_CORE_CONFIRMED,
        direction=Direction.LONG,
        side="above",
        price=105.0,
        source_event_ids=(raw_mss.event_id, opposed_direction.event_id),
        source_entity_ids=("bos-b", "structure-b"),
        details={
            "bos_id": "bos-b",
            "scope": "opposed",
            "core_definition": "first_opposed_confirmed_boundary_break",
        },
        sequence_no=2,
    )
    fvg = _event(
        "fvg-created",
        11,
        canonical=True,
        kind=EventKind.FVG_CREATED,
        source_event_ids=(
            bar_left.event_id,
            bar_middle.event_id,
            bar_right.event_id,
        ),
    )
    displacement = _event(
        "displacement",
        12,
        canonical=True,
        kind=EventKind.DISPLACEMENT_OBSERVED,
        direction=Direction.LONG,
        side="above",
        price=None,
        event_time_minutes=5,
        source_event_ids=(bar_five.event_id, bar_six.event_id),
        source_data_ids=("detector:bar-five", "detector:bar-six"),
        source_entity_ids=("displacement-1",),
        details={
            "transition_id": "transition-1",
            "displacement_id": "displacement-1",
            "lifecycle": "active",
            "terminal_reason": None,
            "admitted_candle_ids": (
                "detector:bar-five",
                "detector:bar-six",
            ),
            "prefix_last_admitted_at": _clock(6).isoformat(),
            "state_metrics": {
                "age_minutes_at_last_admitted": 1.0,
                "atr0": 2.0,
                "efficiency": 1.0,
                "favorable_extreme": 105.0,
                "mean_body_fraction": 0.5,
                "mean_overlap_ratio": 0.5,
                "max_overlap_ratio": 0.5,
                "mean_directional_clv": 0.75,
                "min_directional_clv": 0.5,
                "nested_seed_observed": 0.0,
                "net_points": 2.0,
                "net_ticks": 8.0,
                "origin_price": 100.0,
                "real_episode_bar_count": 2.0,
                "relative_atr": 1.0,
                "speed_atr_per_bar": 0.5,
                "travel_points": 2.0,
                "volume_ready": 1.0,
                "protection_price": 99.0,
                "last_favorable_close": 102.0,
                "interruption_run": 0.0,
                "total_interruption_bars": 0.0,
                "directional_bar_count": 2.0,
                "neutral_bar_count": 0.0,
                "opposite_bar_count": 0.0,
                "directional_body_points": 2.0,
                "opposite_body_points": 0.0,
                "body_continuity": 1.0,
                "activation_gate_count": 6.0,
                "activation_weakest_ratio": 1.0,
                "activation_episode_bar_count_ratio": 1.0,
                "activation_relative_atr_ratio": 1.0,
                "activation_efficiency_ratio": 1.0,
                "activation_speed_ratio": 1.0,
                "activation_mean_body_fraction_ratio": 1.0,
                "activation_body_continuity_ratio": 1.0,
            },
        },
    )
    origin_zone = _event(
        "origin-zone",
        13,
        canonical=True,
        kind=EventKind.ORIGIN_ZONE_CREATED,
        source_event_ids=(
            displacement.event_id,
            raw_break.event_id,
            bar_left.event_id,
        ),
    )
    return (
        bar_left,
        bar_middle,
        bar_right,
        bar_three,
        bar_four,
        swing_left,
        bar_five,
        bar_six,
        swing_right,
        direction,
        leg,
        break_bar,
        raw_break,
        qualified_bos,
        protected,
        opposed_direction,
        mss_break_bar,
        raw_mss,
        mss,
        fvg,
        displacement,
        origin_zone,
    )


def _second_short_protected_assignment(
    *,
    high_swing: MarketEvent,
) -> tuple[MarketEvent, ...]:
    bars = (
        _normalized_bar("second-short-bar-11", 11, close=102.0, low=101.0),
        _normalized_bar("second-short-bar-12", 12, close=101.0, low=100.0),
        _normalized_bar("second-short-bar-13", 13, close=99.0, low=98.0),
        _normalized_bar("second-short-bar-14", 14, close=100.0, low=99.0),
        _normalized_bar("second-short-bar-15", 15, close=101.0, low=100.0),
    )
    low_swing = _event(
        "second-short-low-swing",
        15,
        canonical=True,
        kind=EventKind.SWING_CONFIRMED,
        side="below",
        price=98.0,
        event_time_minutes=13,
        sequence_no=1,
        source_event_ids=tuple(bar.event_id for bar in bars),
        source_entity_ids=("swing-c",),
        details={
            "source_entity_id": "swing-c",
            "side": "low",
            "relation": "lower_low",
            "pivot_start": _clock(13).isoformat(),
            "pivot_end": _clock(13).isoformat(),
            "prominence_atr": 1.5,
            "legacy_same_side_magnitude_atr": 0.5,
            "confirmation_delay_bars": 2,
            "nesting_depth": 0,
            "semantic_rank": "micro",
            "delta_ticks": -4,
            "confirmation_delay_minutes": 2,
        },
    )
    direction = _event(
        "second-short-direction",
        15,
        canonical=True,
        kind=EventKind.STRUCTURE_DIRECTION_CONFIRMED,
        direction=Direction.SHORT,
        side="below",
        price=float(high_swing.price),
        sequence_no=2,
        source_event_ids=(high_swing.event_id, low_swing.event_id),
        source_entity_ids=("structure-c", "swing-b", "swing-c"),
        details={
            "structure_id": "structure-c",
            "direction": "short",
            "source_high_id": "swing-b",
            "source_low_id": "swing-c",
            "sequence_count": 2,
            "candidate_protected_swing_id": "swing-b",
        },
    )
    leg = _event(
        "second-short-leg",
        15,
        canonical=True,
        kind=EventKind.STRUCTURAL_LEG_CREATED,
        direction=Direction.SHORT,
        side="below",
        price=98.0,
        event_time_minutes=13,
        sequence_no=3,
        source_event_ids=(high_swing.event_id, low_swing.event_id),
        source_entity_ids=("leg-c", "swing-b", "swing-c"),
        details={
            "path_class": "internal",
            "leg_id": "leg-c",
            "start_swing_id": "swing-b",
            "end_swing_id": "swing-c",
            "start_event_time": high_swing.event_time.isoformat(),
            "end_event_time": _clock(13).isoformat(),
            "start_price": float(high_swing.price),
            "end_price": 98.0,
            "start_close": 104.0,
            "end_close": 99.0,
            "amplitude_points": 7.0,
            "amplitude_atr": 3.5,
            "duration_bars": 9,
            "duration_minutes": 9,
            "efficiency": 0.8,
            "max_retracement_points": 1.0,
            "max_retracement_atr": 0.5,
            "rank": "internal",
        },
    )
    break_bar = _normalized_bar(
        "second-short-break-bar",
        16,
        close=97.0,
        high=100.0,
        low=96.0,
        sequence_no=0,
    )
    raw = _event(
        "second-short-raw",
        16,
        canonical=True,
        kind=EventKind.RAW_BOUNDARY_BREAK,
        direction=Direction.SHORT,
        side="below",
        price=98.0,
        sequence_no=1,
        source_event_ids=(low_swing.event_id, break_bar.event_id),
        source_data_ids=("detector:second-short-break-bar",),
        source_entity_ids=("bos-c", "swing-c", "structure-c"),
        details={
            "bos_id": "bos-c",
            "target_swing_id": "swing-c",
            "scope": "continuation",
            "break_bar_id": "detector:second-short-break-bar",
            "break_close": 97.0,
            "break_buffer_ticks": 0,
            "comparison": "strict_close_beyond",
            "break_standard": "close_beyond_confirmed_boundary",
        },
    )
    qualified = _event(
        "second-short-qualified",
        16,
        canonical=True,
        kind=EventKind.QUALIFIED_BOS,
        direction=Direction.SHORT,
        side="below",
        price=98.0,
        sequence_no=2,
        source_event_ids=(raw.event_id, direction.event_id),
        source_entity_ids=("bos-c", "structure-c"),
        details={
            "bos_id": "bos-c",
            "scope": "continuation",
            "qualification": "aligned_with_confirmed_structure",
        },
    )
    protected = _event(
        "second-short-protected",
        16,
        canonical=True,
        kind=EventKind.PROTECTED_SWING_ASSIGNED,
        direction=Direction.SHORT,
        side="above",
        price=float(high_swing.price),
        event_time_minutes=4,
        sequence_no=3,
        source_event_ids=(
            qualified.event_id,
            leg.event_id,
            high_swing.event_id,
        ),
        source_entity_ids=("bos-c", "structure-c", "leg-c", "swing-b"),
        details={
            "bos_id": "bos-c",
            "structure_id": "structure-c",
            "origin_leg_id": "leg-c",
            "protected_swing_id": "swing-b",
            "break_standard": "later_acceptance_beyond",
        },
    )
    return (*bars, low_swing, direction, leg, break_bar, raw, qualified, protected)


def _origin_zone_terminal_chain(
    kind: EventKind,
) -> tuple[MarketEvent, ...]:
    events = list(_authoritative_phase23_chain())
    anchor_payload = {
        **dict(events[0].evidence),
        "open": 102.0,
        "high": 104.0,
        "low": 101.0,
        "close": 102.0,
        "real_completed": True,
        "clock_only": False,
        "symbol": "NQH5",
        "instrument_id": 750,
    }
    events[0] = replace(
        events[0],
        details=anchor_payload,
        evidence=anchor_payload,
        source_data_ids=("data:origin-anchor",),
    )
    _kind_by_id = {event.event_id: event.kind for event in events}
    zone_id = "origin-zone-1"
    created_payload = {
        "origin_zone_id": zone_id,
        "geometry": "frozen_group3_order_block_range",
        "lifecycle": "created",
        "source_displacement_id": "displacement-1",
        "source_bos_id": "bos-1",
    }
    # v1.3 publishes the frozen geometry and its qualification as two facts at
    # one clock, so the terminal cites the qualified zone rather than a single
    # merged creation event.
    core = _event(
        "base-origin-core",
        13,
        canonical=True,
        kind=EventKind.BASE_ORIGIN_CORE_CREATED,
        source_event_ids=tuple(
            event_id
            for event_id in events[-1].source_event_ids
            if _kind_by_id[event_id] is EventKind.BAR_COMPLETED
        ),
        source_data_ids=("data:origin-anchor",),
        source_entity_ids=(zone_id,),
        details={
            "base_origin_core_id": zone_id,
            "geometry": "frozen_group3_order_block_range",
            "locating_impulse_id": "displacement-1",
        },
        price=100.0,
        zone=(99.0, 101.0),
        direction=Direction.LONG,
        side="below",
    )
    created = _event(
        "origin-zone",
        13,
        canonical=True,
        kind=EventKind.QUALIFIED_ORIGIN_ZONE_CREATED,
        source_event_ids=(
            core.event_id,
            *(
                event_id
                for event_id in events[-1].source_event_ids
                if _kind_by_id[event_id] is not EventKind.BAR_COMPLETED
            ),
        ),
        source_entity_ids=(zone_id, "displacement-1", "bos-1"),
        details={
            **created_payload,
            "base_origin_core_id": zone_id,
            "base_origin_core_event_id": core.event_id,
        },
        price=100.0,
        zone=(99.0, 101.0),
        direction=Direction.LONG,
        side="below",
    )
    events[-1] = core
    events.append(created)
    invalidated = kind is EventKind.ORIGIN_ZONE_INVALIDATED
    terminal_bar_payload = {
        "open": 100.0,
        "high": 100.5,
        "low": 98.0 if invalidated else 99.5,
        "close": 98.5 if invalidated else 100.25,
        "real_completed": True,
        "clock_only": False,
        "symbol": "NQH5",
        "instrument_id": 750,
    }
    terminal_bar = _event(
        "origin-terminal-bar",
        14,
        kind=EventKind.BAR_COMPLETED,
        timeframe=Timeframe.M5,
        details=terminal_bar_payload,
        price=float(terminal_bar_payload["close"]),
        side=None,
        origin=EventOrigin.NORMALIZED_DATA,
        source_data_ids=("data:origin-terminal",),
        sequence_no=0,
    )
    terminal_payload = {
        "origin_zone_id": zone_id,
        "lifecycle": "failed" if invalidated else "mitigated",
        "transition_reason": (
            "close_through_distal_edge"
            if invalidated
            else "zone_intersected"
        ),
    }
    terminal = _event(
        f"origin-terminal-{kind.value}",
        14,
        canonical=True,
        kind=kind,
        source_event_ids=(created.event_id, terminal_bar.event_id),
        source_entity_ids=(zone_id,),
        details=terminal_payload,
        price=100.0,
        zone=(99.0, 101.0),
        direction=Direction.LONG,
        side="below",
        sequence_no=1,
    )
    return (*events, terminal_bar, terminal)


def _synthetic_displacement_chain() -> tuple[MarketEvent, ...]:
    template = next(
        event
        for event in _authoritative_phase23_chain()
        if event.kind is EventKind.DISPLACEMENT_OBSERVED
    )
    source_one = _normalized_bar(
        "synthetic-displacement-source-one",
        0,
        timeframe=Timeframe.M5,
    )
    source_two = _normalized_bar(
        "synthetic-displacement-source-two",
        5,
        timeframe=Timeframe.M5,
    )
    interval_roots = tuple(
        _normalized_bar(
            f"synthetic-displacement-m1-{minute}",
            minute,
            timeframe=Timeframe.M1,
            real_completed=minute != 8,
            sequence_no=0,
        )
        for minute in range(6, 11)
    )
    synthetic_root = interval_roots[2]
    displacement = _event(
        "synthetic-displacement-terminal",
        10,
        canonical=True,
        kind=EventKind.DISPLACEMENT_OBSERVED,
        timeframe=Timeframe.M5,
        direction=Direction.LONG,
        side="above",
        price=None,
        event_time_minutes=0,
        sequence_no=1,
        source_event_ids=(source_one.event_id, source_two.event_id),
        source_data_ids=(
            "detector:synthetic-displacement-source-one",
            "detector:synthetic-displacement-source-two",
        ),
        source_entity_ids=("synthetic-displacement",),
        context_event_ids=(synthetic_root.event_id,),
        details={
            "transition_id": "synthetic-displacement-transition",
            "displacement_id": "synthetic-displacement",
            "lifecycle": "censored",
            "terminal_reason": "synthetic_interruption",
            "admitted_candle_ids": (
                "detector:synthetic-displacement-source-one",
                "detector:synthetic-displacement-source-two",
            ),
            "prefix_last_admitted_at": _clock(5).isoformat(),
            "state_metrics": dict(template.evidence["state_metrics"]),
        },
    )
    return (source_one, source_two, *interval_roots, displacement)


def _external_range_chain(
    *,
    close: float = 111.0,
    acceptance_range_id: str = "range-1",
    acceptance_timeframe: Timeframe = Timeframe.H1,
) -> tuple[MarketEvent, ...]:
    created = _event(
        "range-created",
        20,
        canonical=True,
        kind=EventKind.DEALING_RANGE_CREATED,
        timeframe=Timeframe.H1,
        details={"range_id": "range-1"},
        zone=(90.0, 110.0),
    )
    activated = _event(
        "range-activated",
        21,
        canonical=True,
        kind=EventKind.BALANCE_RANGE_MATURED,
        timeframe=Timeframe.H1,
        details={"range_id": "range-1"},
        zone=(90.0, 110.0),
        source_event_ids=(created.event_id,),
    )
    bar = _normalized_bar(
        "range-break-bar",
        22,
        timeframe=Timeframe.H1,
        close=close,
        sequence_no=0,
    )
    acceptance_bar = (
        bar
        if acceptance_timeframe is Timeframe.H1
        else _normalized_bar(
            "range-acceptance-bar",
            22,
            timeframe=acceptance_timeframe,
            close=close,
            sequence_no=1,
        )
    )
    level_id = "range-1:upper"
    crossed_at = _clock(22)
    generation_id = _crossing_generation_id(
        level_id=level_id,
        timeframe=acceptance_timeframe,
        crossed_at=crossed_at,
    )
    candidate = _event(
        "range-candidate",
        21,
        canonical=True,
        kind=EventKind.LIQUIDITY_LEVEL_CREATED,
        timeframe=acceptance_timeframe,
        price=110.0,
        details={
            "level_id": level_id,
            "range_id": "range-1",
            "candidate_only": True,
            "source_kind": "mature_range_boundary",
        },
        # A boundary level descends from the interval that froze it. The
        # Structural Range no longer matures, so the parent is its creation.
        source_event_ids=(created.event_id,),
        source_entity_ids=(level_id, "range-1"),
        zone=(110.0, 110.0),
        sequence_no=1,
    )
    touch = _event(
        "range-touch",
        22,
        canonical=True,
        kind=EventKind.LEVEL_TOUCHED,
        timeframe=acceptance_timeframe,
        price=110.0,
        details={
            "level_id": level_id,
        },
        source_event_ids=(candidate.event_id, acceptance_bar.event_id),
        sequence_no=(1 if acceptance_timeframe is Timeframe.H1 else 2),
    )
    penetration = _event(
        "range-penetration",
        22,
        canonical=True,
        kind=EventKind.LEVEL_PENETRATED,
        timeframe=acceptance_timeframe,
        direction=Direction.LONG,
        price=110.0,
        details={
            "level_id": level_id,
            "crossing_generation_id": generation_id,
            "crossed_at": crossed_at.isoformat(),
        },
        source_event_ids=(
            candidate.event_id,
            touch.event_id,
            acceptance_bar.event_id,
        ),
        sequence_no=(2 if acceptance_timeframe is Timeframe.H1 else 3),
    )
    acceptance = _event(
        "range-acceptance",
        22,
        canonical=True,
        kind=EventKind.ACCEPTANCE_CONFIRMED,
        timeframe=acceptance_timeframe,
        direction=Direction.LONG,
        price=110.0,
        details={
            "range_id": acceptance_range_id,
            "accepted_close": close,
            "level_id": level_id,
            "crossing_generation_id": generation_id,
            "crossed_at": crossed_at.isoformat(),
            "resolved_at": crossed_at.isoformat(),
        },
        source_event_ids=(penetration.event_id, acceptance_bar.event_id),
        source_entity_ids=(level_id,),
        sequence_no=(3 if acceptance_timeframe is Timeframe.H1 else 4),
    )
    invalidated = _event(
        "range-invalidated",
        24,
        canonical=True,
        kind=EventKind.DEALING_RANGE_INVALIDATED,
        timeframe=Timeframe.H1,
        details={
            "range_id": "range-1",
            "transition_reason": "close_beyond_frozen_range",
        },
        zone=(90.0, 110.0),
        source_event_ids=(
            created.event_id,
            activated.event_id,
            bar.event_id,
            acceptance.event_id,
        ),
    )
    crossing_events = (
        (bar,)
        if acceptance_bar is bar
        else (bar, acceptance_bar)
    )
    return (
        created,
        activated,
        candidate,
        *crossing_events,
        touch,
        penetration,
        acceptance,
        invalidated,
    )


def _forming_range_invalidation_chain(
    *,
    close: float = 111.0,
) -> tuple[MarketEvent, ...]:
    created = _event(
        "forming-range-created",
        20,
        canonical=True,
        kind=EventKind.DEALING_RANGE_CREATED,
        timeframe=Timeframe.H1,
        details={"range_id": "forming-range-1"},
        zone=(90.0, 110.0),
    )
    bar = _normalized_bar(
        "forming-range-break-bar",
        21,
        timeframe=Timeframe.H1,
        close=close,
    )
    invalidated = _event(
        "forming-range-invalidated",
        22,
        canonical=True,
        kind=EventKind.DEALING_RANGE_INVALIDATED,
        timeframe=Timeframe.H1,
        details={
            "range_id": "forming-range-1",
            "transition_reason": (
                "close_beyond_frozen_range_before_activation"
            ),
        },
        zone=(90.0, 110.0),
        source_event_ids=(created.event_id, bar.event_id),
    )
    return created, bar, invalidated


def test_canonical_event_normalizes_only_unambiguous_legacy_alias() -> None:
    legacy_spelling = _event(
        "legacy-spelling",
        1,
        canonical=True,
        source_ids=("parent",),
    )
    explicit_spelling = _event(
        "explicit-spelling",
        1,
        canonical=True,
        source_event_ids=("parent",),
    )

    assert legacy_spelling.source_ids == ("parent",)
    assert legacy_spelling.source_event_ids == ("parent",)
    assert explicit_spelling.source_ids == ("parent",)
    assert explicit_spelling.source_event_ids == ("parent",)

    with pytest.raises(ValueError, match="must equal source_event_ids"):
        _event(
            "ambiguous",
            1,
            canonical=True,
            source_ids=("legacy-parent",),
            source_event_ids=("explicit-parent",),
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("source_event_ids", ("duplicate", "duplicate")),
        ("source_data_ids", ("duplicate", "duplicate")),
        ("source_entity_ids", ("duplicate", "duplicate")),
        ("context_event_ids", ("duplicate", "duplicate")),
        ("source_event_ids", ("",)),
        ("source_data_ids", ("   ",)),
    ),
)
def test_source_namespaces_reject_duplicates_and_empty_text(
    field_name: str,
    field_value: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="unique|non-empty text"):
        _event("invalid", 0, **{field_name: field_value})


def test_legacy_source_ids_are_stably_deduplicated_for_compatibility() -> None:
    event = _event(
        "legacy-duplicate",
        0,
        source_ids=("same-role", "same-role", "other-role"),
    )

    assert event.source_ids == ("same-role", "other-role")
    assert event.source_event_ids == event.source_ids


@pytest.mark.parametrize(
    "overlap",
    (
        {"source_event_ids": ("same",), "source_data_ids": ("same",)},
        {
            "source_event_ids": ("same",),
            "source_entity_ids": ("same",),
        },
        {"source_data_ids": ("same",), "context_event_ids": ("same",)},
        {
            "source_entity_ids": ("same",),
            "context_event_ids": ("same",),
        },
    ),
)
def test_explicit_source_namespaces_are_mutually_exclusive(
    overlap: dict[str, tuple[str, ...]],
) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _event("overlap", 0, **overlap)


def test_store_accepts_closed_sources_and_context_from_same_batch() -> None:
    parent = _event("parent", 0, canonical=True)
    context = _event("context", 1, canonical=True)
    child = _event(
        "child",
        2,
        canonical=True,
        source_event_ids=(parent.event_id,),
        source_data_ids=("ohlcv:NQ:2025-01-06T10:02",),
        source_entity_ids=("candidate-level:1",),
        context_event_ids=(context.event_id,),
    )
    store = EventStore()

    assert store.append_batch((parent, context, child)) == 3
    assert store.events() == (parent, context, child)


def test_append_batch_uses_overlay_without_copying_lifetime_history() -> None:
    class NonCopyableHistory(dict[str, MarketEvent]):
        def __iter__(self):
            raise AssertionError("append_batch copied lifetime event history")

        def keys(self):
            raise AssertionError("append_batch copied lifetime event history")

    parent = _event("overlay-parent", 0, canonical=True)
    child = _event(
        "overlay-child",
        1,
        canonical=True,
        source_event_ids=(parent.event_id,),
    )
    store = EventStore()
    assert store.append(parent)
    guarded = NonCopyableHistory()
    dict.update(guarded, store._by_id)
    store._by_id = guarded

    assert store.append_batch((child,)) == 1
    assert store.get(child.event_id) == child


def test_append_batch_reads_the_unresolved_reference_set_by_membership_only() -> None:
    """The batch overlay never copies the committed forward-reference set.

    Every batch began with ``set(self._unresolved_forward_reference_ids)``
    -- a copy of every forward identity the lifetime journal still holds
    open, 54,963 of them at bar 20,000 of 2022-02, on each of the ~1.4
    batches a completed minute flushes.  The overlay now records only this
    batch's additions and removals against the committed set and commits
    them by ``add``/``discard``.
    """

    class MembershipOnlyReferences:
        def __init__(self, items):
            self._items = set(items)

        def __contains__(self, item):
            return item in self._items

        def add(self, item):
            self._items.add(item)

        def discard(self, item):
            self._items.discard(item)

    legacy_parent = _event("legacy-parent", 0, source_ids=("later-child",))
    store = EventStore()
    assert store.append(legacy_parent) is True
    assert "later-child" in store._unresolved_forward_reference_ids
    guarded = MembershipOnlyReferences(store._unresolved_forward_reference_ids)
    store._unresolved_forward_reference_ids = guarded

    resolving = _event("later-child", 1, canonical=True)
    another_legacy = _event("legacy-two", 2, source_ids=("much-later",))
    assert store.append_batch((resolving, another_legacy)) == 2
    assert "later-child" not in guarded
    assert "much-later" in guarded
    assert store._unresolved_forward_reference_ids is guarded


@pytest.mark.parametrize("namespace", ("source_event_ids", "context_event_ids"))
def test_store_rejects_dangling_canonical_event_references_atomically(
    namespace: str,
) -> None:
    root = _event("root", 0, canonical=True)
    dangling = _event(
        "dangling",
        1,
        canonical=True,
        **{namespace: ("missing",)},
    )
    store = EventStore()

    with pytest.raises(ValueError, match="unavailable earlier event"):
        store.append_batch((root, dangling))
    assert len(store) == 0


def test_store_rejects_forward_reference_self_reference_and_cycle() -> None:
    parent = _event("parent", 0, canonical=True)
    forward_child = _event(
        "child",
        1,
        canonical=True,
        source_event_ids=(parent.event_id,),
    )
    store = EventStore()
    with pytest.raises(ValueError, match="unavailable earlier event"):
        store.append_batch((forward_child, parent))
    assert len(store) == 0

    self_referencing = _event(
        "self",
        0,
        canonical=True,
        source_event_ids=("self",),
    )
    with pytest.raises(ValueError, match="self-reference"):
        store.append(self_referencing)

    # Legacy transports may retain an opaque forward identity.  A canonical
    # child is still forbidden from closing that pre-existing edge into a
    # cycle.
    legacy_parent = _event(
        "legacy-parent",
        0,
        source_ids=("cycle-child",),
    )
    cycle_child = _event(
        "cycle-child",
        1,
        canonical=True,
        source_event_ids=(legacy_parent.event_id,),
    )
    assert store.append(legacy_parent) is True
    with pytest.raises(ValueError, match="cycle"):
        store.append(cycle_child)


def test_store_rejects_future_parent_and_cross_version_batch() -> None:
    future_parent = _event("future-parent", 5)
    child = _event(
        "older-child",
        4,
        canonical=True,
        source_event_ids=(future_parent.event_id,),
    )
    store = EventStore()
    assert store.append(future_parent) is True
    with pytest.raises(ValueError, match="causally ordered by known_at"):
        store.append(child)

    other_version = _event(
        "other-version",
        6,
        semantic_version="smc_semantics_v2.0",
    )
    with pytest.raises(ValueError, match="cannot mix semantic versions"):
        store.append_batch((other_version,))


def test_legacy_and_projection_events_retain_compatibility_boundary() -> None:
    legacy = _event(
        "legacy",
        0,
        source_ids=("opaque-candle-or-entity",),
    )
    projection = _event(
        "projection",
        1,
        canonical=True,
        projection=True,
        source_event_ids=("rebuildable-missing-parent",),
    )
    store = EventStore()

    assert legacy.source_event_ids == legacy.source_ids
    assert store.append_batch((legacy, projection)) == 2


def test_details_markers_cannot_spoof_explicit_event_origin() -> None:
    legacy = replace(
        _event(
            "legacy-spoof",
            0,
            source_ids=("opaque-missing",),
        ),
        details={"canonical_semantic": True, "projection_only": False},
    )
    assert legacy.is_canonical_semantic is False
    assert EventStore.from_events((legacy,)).events() == (legacy,)

    atomic = replace(
        _event(
            "atomic-cannot-downgrade",
            1,
            canonical=True,
            source_event_ids=("missing",),
        ),
        details={"canonical_semantic": False, "projection_only": True},
    )
    assert atomic.is_canonical_semantic is True
    assert atomic.is_projection is False
    with pytest.raises(ValueError, match="unavailable earlier event"):
        EventStore.from_events((atomic,))


def test_projection_digest_requires_bound_payload_and_cannot_be_spoofed() -> None:
    projection = _event(
        "projection",
        0,
        canonical=True,
        projection=True,
    )
    store = EventStore()
    assert store.append(projection) is True

    tampered_details = dict(projection.details)
    tampered_details["projection_state"] = {
        "state_id": projection.event_id,
        "value": 2,
    }
    with pytest.raises(ValueError, match="does not bind its payload"):
        store.append(replace(projection, details=tampered_details))

    spoofed = _event("spoofed", 1, canonical=True)
    spoofed_details = dict(spoofed.details)
    spoofed_details["projection_sha256"] = "0" * 64
    spoofed = replace(spoofed, details=spoofed_details)
    assert store.append(spoofed) is True
    with pytest.raises(ValueError, match="immutable history"):
        store.append(
            replace(
                spoofed,
                details={**spoofed_details, "unbound_change": True},
            )
        )


def test_store_accepts_complete_authoritative_phase23_parent_chain() -> None:
    events = _authoritative_phase23_chain()
    store = EventStore()

    assert store.append_batch(events) == len(events)
    assert store.events() == events


@pytest.mark.parametrize(
    ("kind", "resolved_minutes"),
    (
        (EventKind.SWEEP_CONFIRMED, 2),
        (EventKind.SWEEP_CONFIRMED, 3),
        (EventKind.ACCEPTANCE_CONFIRMED, 2),
        (EventKind.ACCEPTANCE_CONFIRMED, 3),
    ),
)
def test_store_accepts_closed_authoritative_crossing_chain(
    kind: EventKind,
    resolved_minutes: int,
) -> None:
    events = _crossing_chain(
        prefix=f"valid-{kind.value}-{resolved_minutes}",
        kind=kind,
        resolved_minutes=resolved_minutes,
    )
    store = EventStore()

    assert store.append_batch(events) == len(events)
    terminal = events[-1]
    direct_kinds = {
        store.get(event_id).kind
        for event_id in terminal.source_event_ids
        if store.get(event_id) is not None
    }
    assert direct_kinds == {
        EventKind.LEVEL_PENETRATED,
        EventKind.BAR_COMPLETED,
    }


def test_store_rejects_orphan_canonical_level_touch() -> None:
    orphan = _event(
        "orphan-level-touch",
        0,
        canonical=True,
        kind=EventKind.LEVEL_TOUCHED,
        details={"level_id": "orphan-level"},
        source_event_ids=(),
    )

    with pytest.raises(
        ValueError,
        match="authoritative parent contract failed for level_touched",
    ):
        EventStore.from_events((orphan,))


def test_m1_crossing_binds_higher_timeframe_candidate_explicitly() -> None:
    events = _crossing_chain(
        prefix="cross-scale-candidate",
        timeframe=Timeframe.M1,
        candidate_timeframe=Timeframe.M5,
    )

    assert EventStore.from_events(events).events() == events

    touch_index = next(
        index
        for index, event in enumerate(events)
        if event.kind is EventKind.LEVEL_TOUCHED
    )
    unbound_touch = _with_evidence(
        events[touch_index],
        source_timeframe=None,
    )
    with pytest.raises(ValueError, match="level touch does not bind"):
        EventStore.from_events(
            (*events[:touch_index], unbound_touch, *events[touch_index + 1 :])
        )


def test_crossing_terminal_generation_is_global_and_rebuilt() -> None:
    events = _crossing_chain(prefix="terminal-global")
    terminal = events[-1]
    duplicate = replace(
        terminal,
        event_id="terminal-global-conflict",
        sequence_no=terminal.sequence_no + 1,
    )

    store = EventStore.from_events(events)
    with pytest.raises(ValueError, match="already has an immutable terminal"):
        store.append(duplicate)

    empty = EventStore()
    with pytest.raises(ValueError, match="already has an immutable terminal"):
        empty.append_batch((*events, duplicate))
    assert len(empty) == 0
    with pytest.raises(ValueError, match="already has an immutable terminal"):
        EventStore.from_events((*events, duplicate))
    with pytest.raises(ValueError, match="already has an immutable terminal"):
        EventStore.from_checkpoint(
            (*events, duplicate),
            store.checkpoint_metadata(),
        )

    restored = EventStore.from_checkpoint(
        events,
        store.checkpoint_metadata(),
    )
    with pytest.raises(ValueError, match="already has an immutable terminal"):
        restored.append(duplicate)

    unpickled = pickle.loads(pickle.dumps(store))
    with pytest.raises(ValueError, match="already has an immutable terminal"):
        unpickled.append(duplicate)

    store._events.append(duplicate)
    with pytest.raises(ValueError, match="mutated committed evidence"):
        pickle.loads(pickle.dumps(store))


@pytest.mark.parametrize("bar_role", ("crossing", "resolution"))
def test_crossing_requires_real_definitional_bar_roots(
    bar_role: str,
) -> None:
    events = list(
        _crossing_chain(
            prefix=f"synthetic-{bar_role}",
            timeframe=Timeframe.M1,
            resolved_minutes=3,
        )
    )
    target_clock = _clock(2 if bar_role == "crossing" else 3)
    index = next(
        index
        for index, event in enumerate(events)
        if event.kind is EventKind.BAR_COMPLETED
        and event.known_at == target_clock
    )
    events[index] = _with_evidence(
        events[index],
        real_completed=False,
        clock_only=True,
    )

    with pytest.raises(ValueError, match="exact real normalized BAR"):
        EventStore.from_events(events)


def test_synthetic_displacement_context_is_exact_open_closed_m1_subset() -> None:
    events = _synthetic_displacement_chain()
    store = EventStore.from_events(events)

    assert store.events() == events
    assert events[-1].context_event_ids == (events[4].event_id,)

    missing_context = replace(events[-1], context_event_ids=())
    with pytest.raises(ValueError, match="exact clock-only M1 subset"):
        EventStore.from_events((*events[:-1], missing_context))

    extra_context = replace(
        events[-1],
        context_event_ids=(events[3].event_id, events[4].event_id),
    )
    with pytest.raises(ValueError, match="exact clock-only M1 subset"):
        EventStore.from_events((*events[:-1], extra_context))

    with pytest.raises(ValueError, match="five contiguous unique M1"):
        EventStore.from_events((*events[:5], *events[6:]))

    stale_source = replace(
        events[1],
        observed_at=_clock(4),
        event_time=_clock(4),
        known_at=_clock(4),
    )
    stale_payload = {
        **dict(events[-1].evidence),
        "prefix_last_admitted_at": _clock(4).isoformat(),
    }
    stale_terminal = replace(
        events[-1],
        details=stale_payload,
        evidence=stale_payload,
    )
    with pytest.raises(ValueError, match="immediately preceding real M5"):
        EventStore.from_events(
            (events[0], stale_source, *events[2:-1], stale_terminal)
        )


def test_normalized_bar_index_retry_batch_pickle_and_checkpoint_are_atomic(
) -> None:
    bar = _normalized_bar("indexed-bar", 0)
    store = EventStore.from_events((bar,))
    initial_index = dict(store._normalized_bar_event_ids)

    assert store.append(replace(bar, sequence_no=999)) is False
    assert store._normalized_bar_event_ids == initial_index

    duplicate = replace(bar, event_id="indexed-bar-conflict", sequence_no=1)
    with pytest.raises(ValueError, match="root clock already"):
        store.append(duplicate)
    assert store._normalized_bar_event_ids == initial_index

    staged = _normalized_bar("staged-indexed-bar", 1)
    staged_conflict = replace(
        staged,
        event_id="staged-indexed-bar-conflict",
        sequence_no=1,
    )
    before_events = store.events()
    with pytest.raises(ValueError, match="root clock already"):
        store.append_batch((staged, staged_conflict))
    assert store.events() == before_events
    assert store._normalized_bar_event_ids == initial_index

    for restored in (
        pickle.loads(pickle.dumps(store)),
        EventStore.from_checkpoint(
            store.events(),
            store.checkpoint_metadata(),
        ),
    ):
        assert restored._normalized_bar_event_ids == initial_index
        with pytest.raises(ValueError, match="root clock already"):
            restored.append(duplicate)


def test_same_protected_assignment_may_resolve_distinct_crossing_generations(
) -> None:
    phase_events = _authoritative_phase23_chain()
    protected_index = next(
        index
        for index, event in enumerate(phase_events)
        if event.kind is EventKind.PROTECTED_SWING_ASSIGNED
    )
    prefix = phase_events[: protected_index + 1]
    protected = prefix[-1]

    def protected_crossing(prefix_text: str, crossed: int) -> tuple[MarketEvent, ...]:
        crossing = list(
            _crossing_chain(
                prefix=prefix_text,
                side="below",
                level_id="swing:swing-a",
                candidate_minutes=crossed - 1,
                crossed_minutes=crossed,
                resolved_minutes=crossed + 1,
            )
        )
        crossing[-1] = replace(
            _with_evidence(
                crossing[-1],
                protected_swing_id="swing-a",
                protected_swing_event_id=protected.event_id,
            ),
            context_event_ids=(protected.event_id,),
        )
        return tuple(crossing)

    first = protected_crossing("protected-generation-one", 11)
    second = protected_crossing("protected-generation-two", 14)
    store = EventStore.from_events((*prefix, *first, *second))

    assert store._latest_protected_assignment_event_ids == {
        "swing-a": protected.event_id
    }
    assert {
        first[-1].evidence["crossing_generation_id"],
        second[-1].evidence["crossing_generation_id"],
    } == {
        event.evidence["crossing_generation_id"]
        for event in store.events()
        if event.kind is EventKind.ACCEPTANCE_CONFIRMED
    }


def test_m1_acceptance_binds_higher_timeframe_protected_assignment() -> None:
    phase_events = _authoritative_phase23_chain()
    protected_index = next(
        index
        for index, event in enumerate(phase_events)
        if event.kind is EventKind.PROTECTED_SWING_ASSIGNED
    )
    prefix = phase_events[: protected_index + 1]
    protected = prefix[-1]
    crossing = list(
        _crossing_chain(
            prefix="cross-scale-protected",
            timeframe=Timeframe.M1,
            candidate_timeframe=protected.timeframe,
            side="below",
            level_id="swing:swing-a",
            candidate_minutes=10,
            crossed_minutes=11,
            resolved_minutes=12,
        )
    )
    crossing[-1] = replace(
        _with_evidence(
            crossing[-1],
            protected_swing_id="swing-a",
            protected_swing_event_id=protected.event_id,
        ),
        context_event_ids=(protected.event_id,),
    )

    store = EventStore.from_events((*prefix, *crossing))

    assert store.events()[-1] == crossing[-1]


def _protected_acceptance_chain(
    *,
    prefix: str,
    protected: MarketEvent,
    candidate_minutes: int,
) -> tuple[MarketEvent, ...]:
    crossing = list(
        _crossing_chain(
            prefix=prefix,
            side="below",
            level_id="swing:swing-a",
            candidate_minutes=candidate_minutes,
            crossed_minutes=candidate_minutes + 1,
            resolved_minutes=candidate_minutes + 2,
        )
    )
    crossing[-1] = replace(
        _with_evidence(
            crossing[-1],
            protected_swing_id="swing-a",
            protected_swing_event_id=protected.event_id,
        ),
        context_event_ids=(protected.event_id,),
    )
    return tuple(crossing)


def test_protected_acceptance_must_reference_latest_timeframe_assignment(
) -> None:
    phase_events = _authoritative_phase23_chain()
    protected_index = next(
        index
        for index, event in enumerate(phase_events)
        if event.kind is EventKind.PROTECTED_SWING_ASSIGNED
    )
    first_prefix = phase_events[: protected_index + 1]
    first_protected = first_prefix[-1]
    high_swing = next(
        event for event in first_prefix if event.event_id == "swing-right"
    )
    release = _protected_acceptance_chain(
        prefix="first-protected-release",
        protected=first_protected,
        candidate_minutes=8,
    )
    second_assignment = _second_short_protected_assignment(
        high_swing=high_swing,
    )
    stale = _protected_acceptance_chain(
        prefix="stale-old-protected-release",
        protected=first_protected,
        candidate_minutes=17,
    )
    accepted_prefix = (*first_prefix, *release, *second_assignment, *stale[:-1])

    store = EventStore.from_events(accepted_prefix)
    before = (
        store.events(),
        dict(store._latest_protected_assignment_event_ids_by_timeframe),
    )
    with pytest.raises(ValueError, match="latest protected-swing assignment"):
        store.append(stale[-1])
    assert store.events() == before[0]
    assert store._latest_protected_assignment_event_ids_by_timeframe == before[1]

    # Pickle restore is a validation boundary, not a blind index rebuild.
    store._events.append(stale[-1])
    with pytest.raises(ValueError, match="mutated committed evidence"):
        pickle.loads(pickle.dumps(store))

    reducer_store = EventStore.from_events(accepted_prefix)
    reducer = TimeframeEventReducer(
        event_store=reducer_store,
        semantic_registry_identity="protected-owner-timeframe-test"
    )
    reducer.consume_available()
    reducer_before = (dict(reducer.states), reducer.cursor)
    with pytest.raises(ValueError, match="latest protected-swing assignment"):
        reducer_store.append(stale[-1])
    assert (dict(reducer.states), reducer.cursor) == reducer_before


def _completed_day_high_candidate() -> tuple[MarketEvent, ...]:
    extreme = _normalized_bar(
        "reference-extreme-bar",
        0,
        timeframe=Timeframe.M1,
        high=110.0,
        low=99.0,
        close=100.0,
    )
    admission = _normalized_bar(
        "reference-admission-bar",
        3,
        timeframe=Timeframe.M1,
        high=102.0,
        low=98.0,
        close=101.0,
    )
    level_id = "reference:day:2025-01-06:high:NQH5:750"
    source_id = "reference_source:day:2025-01-06:high:NQH5:750"
    candidate = _event(
        "reference-day-high-candidate",
        3,
        canonical=True,
        kind=EventKind.LIQUIDITY_LEVEL_CREATED,
        timeframe=Timeframe.M1,
        side="above",
        price=110.0,
        zone=(110.0, 110.0),
        event_time_minutes=0,
        source_event_ids=(extreme.event_id, admission.event_id),
        source_entity_ids=(level_id, source_id),
        details={
            "level_id": level_id,
            "candidate_only": True,
            "source_kind": "previous_day_high",
            "source_ids": (source_id,),
            "source_confirmed_at": _clock(2).isoformat(),
            "reference_period_started_at": _clock(-60).isoformat(),
            "reference_period_last_completed_at": _clock(2).isoformat(),
            "reference_extreme_at": _clock(0).isoformat(),
            "reference_admitted_at": _clock(3).isoformat(),
            "reference_extreme_tie_rule": (
                "first_completed_m1_at_extreme"
            ),
        },
    )
    return extreme, admission, candidate


def test_completed_period_candidate_binds_exact_extreme_and_admission_bars(
) -> None:
    events = _completed_day_high_candidate()

    assert EventStore.from_events(events).events() == events


@pytest.mark.parametrize(
    "case",
    ("wrong_price", "wrong_extreme_clock", "one_parent", "wrong_timeframe"),
)
def test_completed_period_candidate_rejects_forged_reference_contract(
    case: str,
) -> None:
    extreme, admission, candidate = _completed_day_high_candidate()
    if case == "wrong_price":
        candidate = replace(candidate, price=100.0, zone=(100.0, 100.0))
    elif case == "wrong_extreme_clock":
        candidate = _with_evidence(
            candidate,
            reference_extreme_at=_clock(1).isoformat(),
        )
    elif case == "one_parent":
        candidate = replace(
            candidate,
            source_ids=(extreme.event_id,),
            source_event_ids=(extreme.event_id,),
        )
    else:
        candidate = replace(candidate, timeframe=Timeframe.H1)

    with pytest.raises(ValueError, match="completed-period candidate"):
        EventStore.from_events((extreme, admission, candidate))


def _reference_zone_candidate_with_point_parent() -> tuple[MarketEvent, ...]:
    extreme, admission, point = _completed_day_high_candidate()
    level_id = "reference-zone:day-high"
    source_id = "reference_source:day:2025-01-06:high:NQH5:750"
    context = replace(
        _event(
            "reference-zone-state",
            3,
            kind=EventKind.SUPPORT_RESISTANCE_STATE,
            timeframe=Timeframe.M1,
            side="above",
            price=110.0,
            zone=(109.75, 110.25),
            source_ids=(source_id,),
            details={
                "source_kind": "previous_day",
                "source_ids": (source_id,),
                "lower_bound": 109.75,
                "upper_bound": 110.25,
            },
        ),
        entity_id=level_id,
        lifecycle="active",
        formed_at=_clock(0),
        sequence_no=1,
    )
    candidate = _event(
        "reference-zone-candidate",
        3,
        canonical=True,
        kind=EventKind.LIQUIDITY_LEVEL_CREATED,
        timeframe=Timeframe.M1,
        side="above",
        price=110.0,
        zone=(109.75, 110.25),
        event_time_minutes=0,
        sequence_no=2,
        source_event_ids=(point.event_id,),
        source_entity_ids=(level_id, source_id),
        context_event_ids=(context.event_id,),
        details={
            "level_id": level_id,
            "candidate_only": True,
            "source_kind": "previous_day",
            "source_ids": (source_id,),
        },
    )
    return extreme, admission, point, context, candidate


def test_reference_zone_candidate_binds_exact_completed_period_point_parent(
) -> None:
    events = _reference_zone_candidate_with_point_parent()

    assert EventStore.from_events(events).events() == events


def test_legacy_v1_2_source_free_reference_zone_candidate_remains_replayable(
) -> None:
    *_, context, candidate = _reference_zone_candidate_with_point_parent()
    # The replay seam is pinned to the exact historical version, so the legacy
    # payload has to say v1.2 rather than inherit whatever is current.
    legacy_version = "smc_semantics_v1.2"
    legacy_context = replace(context, semantic_version=legacy_version)
    legacy = replace(
        candidate,
        semantic_version=legacy_version,
        source_ids=(),
        source_event_ids=(),
    )

    assert EventStore.from_events(
        (legacy_context, legacy),
        semantic_version=legacy_version,
    ).events() == (legacy_context, legacy)

    future_version = "smc_semantics_v1.4"
    future_context = replace(legacy_context, semantic_version=future_version)
    future_legacy = replace(legacy, semantic_version=future_version)
    with pytest.raises(ValueError, match="source-free compatibility"):
        EventStore.from_events(
            (future_context, future_legacy),
            semantic_version=future_version,
        )


@pytest.mark.parametrize(
    "case",
    (
        "wrong_family",
        "wrong_suffix_side",
        "wrong_price",
        "wrong_point_zone",
        "wrong_source_id",
        "wrong_point_level_id",
        "wrong_child_level_id",
        "nonpoint_parent",
        "multiple_parents",
    ),
)
def test_sourceful_reference_zone_candidate_rejects_parent_tampering(
    case: str,
) -> None:
    extreme, admission, point, context, candidate = (
        _reference_zone_candidate_with_point_parent()
    )
    if case == "wrong_family":
        context = _with_evidence(context, source_kind="previous_session")
        candidate = _with_evidence(candidate, source_kind="previous_session")
    elif case == "wrong_suffix_side":
        context = replace(context, side="below")
        candidate = replace(candidate, side="below")
    elif case == "wrong_price":
        context = replace(
            context,
            price=109.75,
            zone=(109.5, 110.0),
        )
        candidate = replace(
            candidate,
            price=109.75,
            zone=(109.5, 110.0),
        )
    elif case == "wrong_point_zone":
        point = replace(point, zone=(109.75, 110.25))
    elif case == "wrong_source_id":
        forged_source_id = (
            "reference_source:day:2025-01-05:high:NQH5:750"
        )
        context = replace(
            _with_evidence(context, source_ids=(forged_source_id,)),
            source_ids=(forged_source_id,),
            source_event_ids=(forged_source_id,),
        )
        candidate = replace(
            _with_evidence(candidate, source_ids=(forged_source_id,)),
            source_entity_ids=(
                candidate.evidence["level_id"],
                forged_source_id,
            ),
        )
    elif case == "wrong_point_level_id":
        forged_level_id = "reference:day:forged:high:NQH5:750"
        point = replace(
            _with_evidence(point, level_id=forged_level_id),
            source_entity_ids=(
                forged_level_id,
                point.evidence["source_ids"][0],
            ),
        )
    elif case == "wrong_child_level_id":
        candidate = _with_evidence(
            candidate,
            level_id="reference-zone:forged",
        )
    elif case == "nonpoint_parent":
        candidate = replace(
            candidate,
            source_ids=(admission.event_id,),
            source_event_ids=(admission.event_id,),
        )
    else:
        candidate = replace(
            candidate,
            source_ids=(point.event_id, admission.event_id),
            source_event_ids=(point.event_id, admission.event_id),
        )

    with pytest.raises(ValueError):
        EventStore.from_events(
            (extreme, admission, point, context, candidate)
        )


def test_sourceful_reference_zone_candidate_cannot_fall_back_to_legacy_seam(
) -> None:
    extreme, admission, point, context, candidate = (
        _reference_zone_candidate_with_point_parent()
    )
    forged = replace(
        candidate,
        source_ids=(admission.event_id,),
        source_event_ids=(admission.event_id,),
    )

    with pytest.raises(
        ValueError,
        match="exact completed-period point parent",
    ):
        EventStore.from_events(
            (extreme, admission, point, context, forged)
        )


def test_range_boundary_candidate_requires_exact_compatibility_or_bar_source(
) -> None:
    context, candidate = _legacy_range_boundary_candidate(
        prefix="range-boundary-contract",
        minutes=3,
        timeframe=Timeframe.M5,
        side="above",
        level_id="range-boundary-level",
    )
    assert EventStore.from_events((context, candidate)).events() == (
        context,
        candidate,
    )

    with pytest.raises(ValueError, match="exact compatibility state"):
        EventStore.from_events(
            (replace(candidate, context_event_ids=()),)
        )

    unrelated = _normalized_bar(
        "range-boundary-unrelated-bar",
        3,
        timeframe=Timeframe.M5,
        high=110.0,
    )
    forged = replace(
        candidate,
        source_ids=(unrelated.event_id,),
        source_event_ids=(unrelated.event_id,),
        context_event_ids=(),
    )
    with pytest.raises(ValueError, match="exact extreme geometry"):
        EventStore.from_events((unrelated, forged))


def test_failed_batch_does_not_publish_staged_protected_or_forward_indexes(
) -> None:
    phase_events = _authoritative_phase23_chain()
    protected_index = next(
        index
        for index, event in enumerate(phase_events)
        if event.kind is EventKind.PROTECTED_SWING_ASSIGNED
    )
    prefix = phase_events[: protected_index + 1]
    protected = prefix[-1]
    store = EventStore.from_events(prefix)
    before_latest = dict(store._latest_protected_assignment_event_ids)
    before_unresolved = set(store._unresolved_forward_reference_ids)
    later_assignment = replace(
        protected,
        event_id="staged-protected-assignment",
        sequence_no=protected.sequence_no + 1,
    )
    forward_legacy = _event(
        "staged-forward-legacy",
        19,
        source_event_ids=("never-arrives",),
    )
    bar = _normalized_bar("staged-atomic-bar", 20)
    conflict = replace(
        bar,
        event_id="staged-atomic-bar-conflict",
        sequence_no=1,
    )

    with pytest.raises(ValueError, match="root clock already"):
        store.append_batch(
            (later_assignment, forward_legacy, bar, conflict)
        )

    assert store.events() == prefix
    assert store._latest_protected_assignment_event_ids == before_latest
    assert store._unresolved_forward_reference_ids == before_unresolved


def test_normal_canonical_append_does_not_walk_complete_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_context, first = _legacy_range_boundary_candidate(
        prefix="bounded-provenance-one",
        minutes=0,
        timeframe=Timeframe.M5,
        side="above",
        level_id="bounded-one",
    )
    second_context, second = _legacy_range_boundary_candidate(
        prefix="bounded-provenance-two",
        minutes=1,
        timeframe=Timeframe.M5,
        side="above",
        level_id="bounded-two",
    )
    store = EventStore.from_events((first_context, first))

    def unexpected_walk(*args: object, **kwargs: object) -> bool:
        raise AssertionError("normal canonical append performed a DAG walk")

    monkeypatch.setattr(
        EventStore,
        "_provenance_reaches",
        staticmethod(unexpected_walk),
    )
    assert store.append(second_context) is True
    assert store.append(second) is True


def test_timeframe_reducer_validation_uses_persistent_indexes() -> None:
    context, candidate = _legacy_range_boundary_candidate(
        prefix="indexed-reducer",
        minutes=0,
        timeframe=Timeframe.M5,
        side="above",
        level_id="indexed-reducer-level",
    )
    store = EventStore.from_events((context, candidate))
    reducer = TimeframeEventReducer(
        event_store=store,
        semantic_registry_identity="indexed-reducer-test",
    )
    store.events = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("timeframe reducer scanned complete history")
    )

    assert reducer.consume_available() == (context, candidate)
    assert Timeframe.M5 in reducer.states
    assert "_events_by_id" not in reducer.__dict__
    assert "_event_ids" not in reducer.__dict__


def test_timeframe_reducer_rejects_a_second_crossing_terminal_atomically(
) -> None:
    events = _crossing_chain(prefix="reducer-terminal-unique")
    store = EventStore.from_events(events)
    reducer = TimeframeEventReducer(
        event_store=store,
        semantic_registry_identity="reducer-terminal-unique-test"
    )
    reducer.consume_available()

    penetration = next(
        event for event in events if event.kind is EventKind.LEVEL_PENETRATED
    )
    terminal = events[-1]
    later_bar = _normalized_bar(
        "reducer-terminal-second-bar",
        4,
        timeframe=terminal.timeframe,
        close=101.0,
    )
    store.append(later_bar)
    reducer.consume_available()
    duplicate = replace(
        _with_evidence(terminal, resolved_at=_clock(4).isoformat()),
        event_id="reducer-terminal-second",
        observed_at=_clock(4),
        known_at=_clock(4),
        source_ids=(penetration.event_id, later_bar.event_id),
        source_event_ids=(penetration.event_id, later_bar.event_id),
        sequence_no=1,
    )
    before = (dict(reducer.states), reducer.cursor)

    with pytest.raises(ValueError, match="already has an immutable terminal"):
        store.append(duplicate)
    assert (dict(reducer.states), reducer.cursor) == before


def test_timeframe_reducer_invalid_owner_metadata_is_failure_atomic() -> None:
    store = EventStore()
    reducer = TimeframeEventReducer(
        event_store=store,
        semantic_registry_identity="reducer-failure-atomic-test"
    )
    malformed = _with_evidence(
        _normalized_bar(
            "reducer-invalid-owner-bar",
            0,
            timeframe=Timeframe.M1,
        ),
        active_timeframes=(Timeframe.M1.value, Timeframe.M5.value),
        source_timeframe="not-a-timeframe",
    )
    store.append(malformed)
    before = (dict(reducer.states), reducer.cursor)

    with pytest.raises(ValueError, match="not-a-timeframe"):
        reducer.consume_available()
    assert (dict(reducer.states), reducer.cursor) == before


def test_timeframe_reducer_pickle_keeps_only_store_bound_compact_state() -> None:
    store = EventStore()
    reducer = TimeframeEventReducer(
        event_store=store,
        semantic_registry_identity="legacy-reducer-index-test"
    )
    bar = _normalized_bar("legacy-reducer-index-bar", 0)
    store.append(bar)
    assert reducer.consume_available() == (bar,)
    before = (dict(reducer.states), reducer.cursor)
    duplicate = replace(
        bar,
        event_id="legacy-reducer-index-bar-conflict",
        sequence_no=1,
    )
    with pytest.raises(ValueError, match="root clock already"):
        store.append(duplicate)
    assert (dict(reducer.states), reducer.cursor) == before
    assert not {
        "_event_ids",
        "_events_by_id",
        "_normalized_bar_event_ids",
        "_terminal_crossing_event_ids",
        "_latest_protected_assignment_event_ids",
        "_unresolved_forward_reference_ids",
    }.intersection(reducer.__dict__)

    restored = pickle.loads(pickle.dumps(reducer))
    context, candidate = _legacy_range_boundary_candidate(
        prefix="legacy-reducer-index",
        minutes=1,
        timeframe=Timeframe.M5,
        side="above",
        level_id="legacy-reducer-index-level",
    )

    restored.event_store.append_batch((context, candidate))
    assert restored.consume_available() == (context, candidate)
    assert Timeframe.M5 in restored.states


def test_legacy_unresolved_forward_reference_still_detects_cycle() -> None:
    legacy = _event(
        "legacy-forward-parent",
        0,
        source_event_ids=("future-canonical",),
    )
    future = _event(
        "future-canonical",
        1,
        canonical=True,
        kind=EventKind.LIQUIDITY_LEVEL_CREATED,
        details={
            "level_id": "future-level",
            "candidate_only": True,
            "source_kind": "range_boundary",
        },
        source_entity_ids=("future-level",),
        context_event_ids=(legacy.event_id,),
        zone=(100.0, 100.0),
    )
    store = EventStore.from_events((legacy,))

    with pytest.raises(ValueError, match="contains a cycle"):
        store.append(future)
    assert store.events() == (legacy,)


@pytest.mark.parametrize(
    "case",
    (
        "swing_synthetic_bar",
        "leg_forged_endpoint",
        "candidate_unknown_taxonomy",
        "displacement_synthetic_bar",
        "displacement_missing_metric",
        "raw_break_close_drift",
        "structure_formation_drift",
        "qualified_direction_drift",
        "qualified_definition_drift",
        "protected_entity_drift",
        "protected_bos_parent_drift",
        "protected_structure_parent_drift",
        "protected_definition_drift",
        "mss_prior_direction_drift",
        "mss_definition_drift",
    ),
)
def test_authority_cross_links_fail_closed(case: str) -> None:
    events = list(_authoritative_phase23_chain())
    if case == "swing_synthetic_bar":
        index = next(
            i for i, event in enumerate(events) if event.event_id == "bar-left"
        )
        events[index] = _with_evidence(
            replace(events[index], timeframe=Timeframe.M1),
            real_completed=False,
            clock_only=True,
        )
    elif case == "leg_forged_endpoint":
        index = next(
            i
            for i, event in enumerate(events)
            if event.kind is EventKind.STRUCTURAL_LEG_CREATED
        )
        events[index] = _with_evidence(
            events[index], end_swing_id="forged-swing"
        )
    elif case == "candidate_unknown_taxonomy":
        events = list(_crossing_chain(prefix="unknown-taxonomy"))
        index = next(
            i
            for i, event in enumerate(events)
            if event.kind is EventKind.LIQUIDITY_LEVEL_CREATED
        )
        events[index] = _with_evidence(
            events[index], source_kind="institutional_stops"
        )
    elif case == "displacement_synthetic_bar":
        index = next(
            i for i, event in enumerate(events) if event.event_id == "bar-five"
        )
        events[index] = _with_evidence(
            replace(events[index], timeframe=Timeframe.M1),
            real_completed=False,
            clock_only=True,
        )
    elif case == "displacement_missing_metric":
        index = next(
            i
            for i, event in enumerate(events)
            if event.kind is EventKind.DISPLACEMENT_OBSERVED
        )
        metrics = dict(events[index].evidence["state_metrics"])
        metrics.pop("real_episode_bar_count")
        events[index] = _with_evidence(events[index], state_metrics=metrics)
    elif case == "raw_break_close_drift":
        index = next(
            i for i, event in enumerate(events) if event.event_id == "raw-break"
        )
        events[index] = _with_evidence(events[index], break_close=104.0)
    elif case == "structure_formation_drift":
        index = next(
            i
            for i, event in enumerate(events)
            if event.event_id == "structure-direction"
        )
        events[index] = _with_evidence(
            events[index],
            candidate_protected_swing_id="swing-b",
        )
    elif case == "qualified_direction_drift":
        index = next(
            i
            for i, event in enumerate(events)
            if event.kind is EventKind.QUALIFIED_BOS
        )
        events[index] = replace(events[index], direction=Direction.SHORT)
    elif case == "qualified_definition_drift":
        index = next(
            i
            for i, event in enumerate(events)
            if event.kind is EventKind.QUALIFIED_BOS
        )
        events[index] = _with_evidence(events[index], qualification="score_gate")
    elif case == "protected_entity_drift":
        index = next(
            i
            for i, event in enumerate(events)
            if event.kind is EventKind.PROTECTED_SWING_ASSIGNED
        )
        events[index] = _with_evidence(
            events[index], protected_swing_id="forged-swing"
        )
    elif case in {
        "protected_bos_parent_drift",
        "protected_structure_parent_drift",
    }:
        index = next(
            i
            for i, event in enumerate(events)
            if event.kind is EventKind.PROTECTED_SWING_ASSIGNED
        )
        protected = events[index]
        entities = list(protected.source_entity_ids)
        field = (
            "bos_id"
            if case == "protected_bos_parent_drift"
            else "structure_id"
        )
        entity_index = 0 if field == "bos_id" else 1
        forged_identity = f"forged-{field}"
        entities[entity_index] = forged_identity
        events[index] = replace(
            _with_evidence(protected, **{field: forged_identity}),
            source_entity_ids=tuple(entities),
        )
    elif case == "protected_definition_drift":
        index = next(
            i
            for i, event in enumerate(events)
            if event.kind is EventKind.PROTECTED_SWING_ASSIGNED
        )
        events[index] = _with_evidence(events[index], break_standard="raw_wick")
    elif case == "mss_prior_direction_drift":
        index = next(
            i
            for i, event in enumerate(events)
            if event.event_id == "structure-direction-opposed"
        )
        events[index] = replace(
            _with_evidence(events[index], direction="long"),
            direction=Direction.LONG,
        )
    elif case == "mss_definition_drift":
        index = next(
            i
            for i, event in enumerate(events)
            if event.kind is EventKind.MSS_CORE_CONFIRMED
        )
        events[index] = _with_evidence(
            events[index],
            core_definition="sweep_plus_displacement",
        )
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(case)

    with pytest.raises(ValueError, match="authoritative"):
        EventStore.from_events(events)


def test_acceptance_binds_the_live_protected_assignment_context() -> None:
    phase_events = _authoritative_phase23_chain()
    protected_index = next(
        index
        for index, event in enumerate(phase_events)
        if event.kind is EventKind.PROTECTED_SWING_ASSIGNED
    )
    prefix = phase_events[: protected_index + 1]
    protected = prefix[-1]
    crossing = list(
        _crossing_chain(
            prefix="protected-acceptance",
            side="below",
            level_id="swing:swing-a",
            candidate_minutes=10,
            crossed_minutes=11,
            resolved_minutes=12,
        )
    )
    crossing[-1] = replace(
        _with_evidence(
            crossing[-1],
            protected_swing_id="swing-a",
            protected_swing_event_id=protected.event_id,
        ),
        context_event_ids=(protected.event_id,),
    )

    store = EventStore.from_events((*prefix, *crossing))
    assert store.events()[-1] == crossing[-1]

    missing_context = replace(crossing[-1], context_event_ids=())
    with pytest.raises(ValueError, match="protected-swing assignment context"):
        EventStore.from_events((*prefix, *crossing[:-1], missing_context))

    later_assignment = replace(
        protected,
        event_id="protected-swing-later-authority",
        sequence_no=protected.sequence_no + 1,
    )
    with pytest.raises(ValueError, match="latest protected-swing assignment"):
        EventStore.from_events(
            (*prefix, later_assignment, *crossing)
        )


@pytest.mark.parametrize("missing_kind", ("penetration", "resolution_bar"))
def test_store_rejects_terminal_with_missing_definitional_parent(
    missing_kind: str,
) -> None:
    events = _crossing_chain(prefix=f"missing-{missing_kind}")
    terminal = events[-1]
    penetration = next(
        event for event in events if event.kind is EventKind.LEVEL_PENETRATED
    )
    resolution_bar = next(
        event
        for event in reversed(events)
        if event.kind is EventKind.BAR_COMPLETED
    )
    retained_source = (
        resolution_bar.event_id
        if missing_kind == "penetration"
        else penetration.event_id
    )
    forged = replace(
        terminal,
        source_ids=(retained_source,),
        source_event_ids=(retained_source,),
    )

    with pytest.raises(ValueError, match="authoritative parent contract"):
        EventStore.from_events((*events[:-1], forged))


def test_store_rejects_raw_break_as_terminal_definitional_source() -> None:
    events = _crossing_chain(prefix="raw-source")
    terminal = events[-1]
    raw_context = _event(
        "raw-context",
        1,
        kind=EventKind.RAW_BOUNDARY_BREAK,
    )
    forged = replace(
        terminal,
        source_ids=(*terminal.source_event_ids, raw_context.event_id),
        source_event_ids=(
            *terminal.source_event_ids,
            raw_context.event_id,
        ),
    )

    with pytest.raises(ValueError, match="authoritative parent contract"):
        EventStore.from_events(
            (*events[:2], raw_context, *events[2:-1], forged)
        )


def test_store_accepts_raw_break_only_as_terminal_context() -> None:
    events = _crossing_chain(prefix="raw-context")
    terminal = events[-1]
    raw_context = _event(
        "raw-context-parent",
        1,
        kind=EventKind.RAW_BOUNDARY_BREAK,
    )
    contextual = replace(
        terminal,
        context_event_ids=(raw_context.event_id,),
    )
    store = EventStore()

    ordered = (*events[:2], raw_context, *events[2:-1], contextual)
    assert store.append_batch(ordered) == len(ordered)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("level_id", "other-level", "generation identity differ"),
        ("crossing_generation_id", "wrong-generation", "generation identity differ"),
        ("crossed_at", _clock(1).isoformat(), "generation identity differ"),
        ("resolved_at", _clock(4).isoformat(), "clocks differ"),
    ),
)
def test_store_rejects_terminal_identity_or_clock_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    events = _crossing_chain(prefix=f"drift-{field}")
    forged = _with_evidence(events[-1], **{field: value})

    with pytest.raises(ValueError, match=message):
        EventStore.from_events((*events[:-1], forged))


def test_store_rejects_terminal_timeframe_side_and_direction_drift() -> None:
    events = _crossing_chain(prefix="terminal-relations")
    terminal = events[-1]
    resolution_bar = next(
        event
        for event in reversed(events)
        if event.kind is EventKind.BAR_COMPLETED
    )
    wrong_timeframe_bar = replace(
        resolution_bar,
        timeframe=Timeframe.H1,
    )
    with pytest.raises(ValueError, match="timeframes differ"):
        EventStore.from_events(
            tuple(
                wrong_timeframe_bar if event is resolution_bar else event
                for event in events
            )
        )

    wrong_side = replace(terminal, side="below")
    with pytest.raises(ValueError, match="sides differ"):
        EventStore.from_events((*events[:-1], wrong_side))

    wrong_direction = replace(terminal, direction=Direction.SHORT)
    with pytest.raises(ValueError, match="direction does not match"):
        EventStore.from_events((*events[:-1], wrong_direction))


def test_store_rejects_incomplete_or_cross_generation_penetration_chain() -> None:
    events = _crossing_chain(prefix="penetration-chain")
    penetration_index = next(
        index
        for index, event in enumerate(events)
        if event.kind is EventKind.LEVEL_PENETRATED
    )
    penetration = events[penetration_index]
    crossing_bar = next(
        event for event in events if event.kind is EventKind.BAR_COMPLETED
    )
    touch = next(event for event in events if event.kind is EventKind.LEVEL_TOUCHED)

    missing_candidate = replace(
        penetration,
        source_ids=(touch.event_id, crossing_bar.event_id),
        source_event_ids=(touch.event_id, crossing_bar.event_id),
    )
    with pytest.raises(ValueError, match="authoritative parent contract"):
        EventStore.from_events(
            (*events[:penetration_index], missing_candidate)
        )

    wrong_generation = _with_evidence(
        penetration,
        crossing_generation_id="forged-generation",
    )
    with pytest.raises(ValueError, match="does not bind"):
        EventStore.from_events(
            (*events[:penetration_index], wrong_generation)
        )


def test_terminal_validation_fails_closed_on_missing_transitive_parent() -> None:
    events = _crossing_chain(prefix="missing-transitive")
    terminal = events[-1]
    penetration = next(
        event for event in events if event.kind is EventKind.LEVEL_PENETRATED
    )
    resolution_bar = next(
        event
        for event in reversed(events)
        if event.kind is EventKind.BAR_COMPLETED
    )

    with pytest.raises(ValueError, match="unavailable transitive source"):
        validate_canonical_event(
            terminal,
            available_events={
                penetration.event_id: penetration,
                resolution_bar.event_id: resolution_bar,
            },
        )


def test_store_rejects_touch_from_different_candidate_generation() -> None:
    events = _crossing_chain(prefix="touch-generation")
    candidate_index = next(
        index
        for index, event in enumerate(events)
        if event.kind is EventKind.LIQUIDITY_LEVEL_CREATED
    )
    candidate = events[candidate_index]
    touch_index = next(
        index
        for index, event in enumerate(events)
        if event.kind is EventKind.LEVEL_TOUCHED
    )
    penetration_index = next(
        index
        for index, event in enumerate(events)
        if event.kind is EventKind.LEVEL_PENETRATED
    )
    touch = events[touch_index]
    penetration = events[penetration_index]
    other_candidate = replace(
        candidate,
        event_id="other-candidate",
        sequence_no=candidate.sequence_no + 1,
    )
    forged_touch = replace(
        touch,
        source_ids=(other_candidate.event_id, touch.source_event_ids[1]),
        source_event_ids=(
            other_candidate.event_id,
            touch.source_event_ids[1],
        ),
    )
    forged_penetration = replace(
        penetration,
        source_ids=(
            penetration.source_event_ids[0],
            forged_touch.event_id,
            penetration.source_event_ids[2],
        ),
        source_event_ids=(
            penetration.source_event_ids[0],
            forged_touch.event_id,
            penetration.source_event_ids[2],
        ),
    )
    prefix = (
        *events[: candidate_index + 1],
        other_candidate,
        *events[candidate_index + 1 : touch_index],
        forged_touch,
        forged_penetration,
    )

    with pytest.raises(ValueError, match="same candidate"):
        EventStore.from_events(prefix)


@pytest.mark.parametrize(
    "kind",
    (EventKind.SWEEP_CONFIRMED, EventKind.ACCEPTANCE_CONFIRMED),
)
def test_crossing_parent_contract_does_not_reclassify_legacy_terminal(
    kind: EventKind,
) -> None:
    legacy = _event(
        f"legacy-{kind.value}",
        0,
        kind=kind,
        source_ids=("opaque-crossing",),
    )

    assert EventStore.from_events((legacy,)).events() == (legacy,)


@pytest.mark.parametrize(
    ("kind", "source_event_ids"),
    (
        (EventKind.STRUCTURAL_LEG_CREATED, ("swing-left",)),
        (
            EventKind.RAW_BOUNDARY_BREAK,
            ("swing-left", "swing-right"),
        ),
        (EventKind.QUALIFIED_BOS, ("raw-break",)),
        (
            EventKind.PROTECTED_SWING_ASSIGNED,
            ("qualified-bos", "structural-leg"),
        ),
        (
            EventKind.MSS_CORE_CONFIRMED,
            ("raw-break", "qualified-bos"),
        ),
        (EventKind.FVG_CREATED, ("bar-left", "bar-middle")),
        (
            EventKind.QUALIFIED_ORIGIN_ZONE_CREATED,
            ("displacement", "raw-break"),
        ),
    ),
)
def test_store_rejects_missing_or_mistyped_authoritative_parents(
    kind: EventKind,
    source_event_ids: tuple[str, ...],
) -> None:
    store = EventStore.from_events(_authoritative_phase23_chain())
    bad = _event(
        f"bad-{kind.value}",
        30,
        canonical=True,
        kind=kind,
        source_event_ids=source_event_ids,
    )
    before = store.events()

    with pytest.raises(ValueError, match="authoritative parent contract"):
        store.append(bad)
    assert store.events() == before


@pytest.mark.parametrize(
    "kind",
    (
        EventKind.ORIGIN_ZONE_MITIGATED,
        EventKind.ORIGIN_ZONE_INVALIDATED,
    ),
)
def test_origin_zone_terminal_production_shape_replays(
    kind: EventKind,
) -> None:
    events = _origin_zone_terminal_chain(kind)

    first = EventStore.from_events(events)
    replayed = EventStore.from_events(first.events())

    assert replayed.events() == first.events()
    assert replayed.events()[-1].kind is kind
    assert replayed.events()[-1].source_event_ids == (
        events[-3].event_id,
        events[-2].event_id,
    )


@pytest.mark.parametrize(
    "case",
    (
        "missing_created_parent",
        "unknown_zone_identity",
        "foreign_symbol_scope",
        "wrong_transition_clock",
        "changed_frozen_zone",
        "synthetic_transition_bar",
        "mitigation_with_failed_close",
    ),
)
def test_origin_zone_terminal_contract_fails_closed(case: str) -> None:
    events = list(
        _origin_zone_terminal_chain(EventKind.ORIGIN_ZONE_MITIGATED)
    )
    transition_bar = events[-2]
    terminal = events[-1]
    if case == "missing_created_parent":
        terminal = replace(
            terminal,
            source_ids=(transition_bar.event_id,),
            source_event_ids=(transition_bar.event_id,),
        )
    elif case == "unknown_zone_identity":
        payload = {
            **dict(terminal.evidence),
            "origin_zone_id": "unknown-zone",
        }
        terminal = replace(
            terminal,
            details=payload,
            evidence=payload,
            source_entity_ids=("unknown-zone",),
        )
    elif case == "foreign_symbol_scope":
        payload = {
            **dict(transition_bar.evidence),
            "symbol": "ESH5",
        }
        events[-2] = replace(
            transition_bar,
            details=payload,
            evidence=payload,
        )
    elif case == "wrong_transition_clock":
        terminal = replace(
            terminal,
            observed_at=_clock(15),
            known_at=_clock(15),
            event_time=_clock(15),
        )
    elif case == "changed_frozen_zone":
        terminal = replace(terminal, zone=(98.0, 102.0))
    elif case == "synthetic_transition_bar":
        events[-2] = _with_evidence(
            replace(transition_bar, timeframe=Timeframe.M1),
            real_completed=False,
            clock_only=True,
        )
    elif case == "mitigation_with_failed_close":
        payload = {
            **dict(transition_bar.evidence),
            "low": 98.0,
            "close": 98.5,
        }
        events[-2] = replace(
            transition_bar,
            details=payload,
            evidence=payload,
            price=98.5,
        )
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(case)
    events[-1] = terminal

    with pytest.raises(
        ValueError,
        match=(
            "authoritative (origin-zone terminal|parent) contract|"
            "authoritative origin_zone_mitigated parent requires an exact real"
        ),
    ):
        EventStore.from_events(events)


def test_origin_zone_invalidation_requires_close_through_geometry() -> None:
    events = list(
        _origin_zone_terminal_chain(EventKind.ORIGIN_ZONE_INVALIDATED)
    )
    transition_bar = events[-2]
    payload = {
        **dict(transition_bar.evidence),
        "low": 99.5,
        "close": 100.25,
    }
    events[-2] = replace(
        transition_bar,
        details=payload,
        evidence=payload,
        price=100.25,
    )

    with pytest.raises(
        ValueError,
        match="origin-zone terminal contract conflicts",
    ):
        EventStore.from_events(events)


def test_origin_zone_terminal_state_replay_and_unknown_identity_fail_closed(
) -> None:
    events = _origin_zone_terminal_chain(
        EventKind.ORIGIN_ZONE_MITIGATED
    )[-3:]

    def replay(sequence: tuple[MarketEvent, ...]):
        state = None
        for event in sequence:
            state = reduce_timeframe_state(
                state,
                event,
                semantic_registry_identity="origin-zone-terminal-test",
            )
        return state

    first = replay(events)
    assert first is not None and first.zones.active_ob == ()
    assert replay(events) == first

    created, _, terminal = events
    created_state = replay((created,))
    unknown_payload = {
        **dict(terminal.evidence),
        "origin_zone_id": "unknown-zone",
    }
    unknown = replace(
        terminal,
        details=unknown_payload,
        evidence=unknown_payload,
        source_entity_ids=("unknown-zone",),
    )
    with pytest.raises(ValueError, match="unknown active zone"):
        reduce_timeframe_state(
            created_state,
            unknown,
            semantic_registry_identity="origin-zone-terminal-test",
        )


def test_two_unrelated_facts_cannot_masquerade_as_qualified_bos() -> None:
    unrelated_one = _event("unrelated-one", 0, canonical=True)
    unrelated_two = _event("unrelated-two", 1, canonical=True)
    counterfeit_bos = _event(
        "counterfeit-bos",
        2,
        canonical=True,
        kind=EventKind.QUALIFIED_BOS,
        source_event_ids=(unrelated_one.event_id, unrelated_two.event_id),
    )
    store = EventStore()

    with pytest.raises(
        ValueError,
        match="authoritative parent contract failed for qualified_bos",
    ):
        store.append_batch((unrelated_one, unrelated_two, counterfeit_bos))
    assert len(store) == 0


def test_authoritative_bar_parent_must_be_normalized_data() -> None:
    legacy_bar = _event(
        "legacy-bar",
        0,
        kind=EventKind.BAR_COMPLETED,
        details={"close": 100.0},
    )
    bar_middle = _normalized_bar("normalized-middle", 1)
    bar_right = _normalized_bar("normalized-right", 2)
    fvg = _event(
        "fvg-with-legacy-bar",
        3,
        canonical=True,
        kind=EventKind.FVG_CREATED,
        source_event_ids=(
            legacy_bar.event_id,
            bar_middle.event_id,
            bar_right.event_id,
        ),
    )
    store = EventStore()

    with pytest.raises(ValueError, match="must be normalized_data"):
        store.append_batch((legacy_bar, bar_middle, bar_right, fvg))
    assert len(store) == 0


def test_authoritative_semantic_parent_must_itself_be_atomic() -> None:
    phase_events = _authoritative_phase23_chain()
    direction_index = next(
        index
        for index, event in enumerate(phase_events)
        if event.event_id == "structure-direction"
    )
    direction = phase_events[direction_index]
    legacy_raw_break = _event(
        "legacy-raw-break",
        20,
        kind=EventKind.RAW_BOUNDARY_BREAK,
    )
    qualified_bos = _event(
        "bos-with-legacy-parent",
        21,
        canonical=True,
        kind=EventKind.QUALIFIED_BOS,
        source_event_ids=(legacy_raw_break.event_id, direction.event_id),
    )
    store = EventStore()

    with pytest.raises(ValueError, match="must be semantic_atomic"):
        store.append_batch(
            (
                *phase_events[: direction_index + 1],
                legacy_raw_break,
                qualified_bos,
            )
        )
    assert len(store) == 0


def test_store_accepts_complete_external_range_invalidation_chain() -> None:
    events = _external_range_chain()
    store = EventStore()

    assert store.append_batch(events) == len(events)
    assert store.events() == events


def test_store_accepts_forming_range_failure_without_active_ancestry() -> None:
    events = _forming_range_invalidation_chain()
    store = EventStore()

    assert store.append_batch(events) == len(events)
    assert store.events() == events


@pytest.mark.parametrize("close", (100.0, 110.0))
def test_forming_range_failure_requires_strictly_outside_h1_close(
    close: float,
) -> None:
    events = _forming_range_invalidation_chain(close=close)

    with pytest.raises(ValueError, match="strictly outside"):
        EventStore.from_events(events)


def test_external_range_invalidation_requires_complete_parent_chain() -> None:
    events = _external_range_chain()
    created = next(
        event for event in events if event.kind is EventKind.DEALING_RANGE_CREATED
    )
    activated = next(
        event
        for event in events
        if event.kind is EventKind.BALANCE_RANGE_MATURED
    )
    bar = next(event for event in events if event.event_id == "range-break-bar")
    invalidated = next(
        event
        for event in events
        if event.kind is EventKind.DEALING_RANGE_INVALIDATED
    )
    missing_acceptance = replace(
        invalidated,
        source_ids=(created.event_id, activated.event_id, bar.event_id),
        source_event_ids=(
            created.event_id,
            activated.event_id,
            bar.event_id,
        ),
    )
    store = EventStore()

    with pytest.raises(ValueError, match="authoritative parent contract"):
        store.append_batch((*events[:-1], missing_acceptance))
    assert len(store) == 0


@pytest.mark.parametrize(
    ("acceptance_range_id", "acceptance_timeframe", "message"),
    (
        ("range-other", Timeframe.H1, "parent range_id differs"),
        ("range-1", Timeframe.M5, "parent is on another scale"),
    ),
)
def test_external_range_acceptance_must_share_scale_and_range(
    acceptance_range_id: str,
    acceptance_timeframe: Timeframe,
    message: str,
) -> None:
    events = _external_range_chain(
        acceptance_range_id=acceptance_range_id,
        acceptance_timeframe=acceptance_timeframe,
    )

    with pytest.raises(ValueError, match=message):
        EventStore.from_events(events)


@pytest.mark.parametrize("close", (100.0, 110.0))
def test_external_range_bar_must_close_strictly_outside_frozen_zone(
    close: float,
) -> None:
    events = _external_range_chain(close=close)

    with pytest.raises(
        ValueError,
        match="strictly outside|terminal kind conflicts",
    ):
        EventStore.from_events(events)


@pytest.mark.parametrize(
    "event",
    (
        _event(
            "legacy-qualified-bos",
            0,
            kind=EventKind.QUALIFIED_BOS,
        ),
        _event(
            "normalized-qualified-bos",
            1,
            kind=EventKind.QUALIFIED_BOS,
            origin=EventOrigin.NORMALIZED_DATA,
        ),
        replace(
            _event(
                "projected-fvg",
                2,
                canonical=True,
                projection=True,
            ),
            kind=EventKind.FVG_CREATED,
        ),
    ),
)
def test_parent_contract_does_not_reclassify_non_atomic_origins(
    event: MarketEvent,
) -> None:
    assert EventStore.from_events((event,)).events() == (event,)


def test_explicit_source_namespaces_roundtrip_through_bound_journal(
    tmp_path: Path,
) -> None:
    registry = SemanticRegistry.from_file()
    parent = _event(
        "bar-parent",
        0,
        source_data_ids=("ohlcv-row:1",),
    )
    context = _event("context", 1, canonical=True)
    child = _event(
        "child",
        2,
        canonical=True,
        source_event_ids=(parent.event_id,),
        source_data_ids=("ohlcv-row:2",),
        source_entity_ids=("candidate-level:1",),
        context_event_ids=(context.event_id,),
    )
    store = EventStore(
        definition_identity=registry.definition_identity,
    )
    assert store.append_batch((parent, context, child)) == 3

    write_event_journal(tmp_path / "journal", store)
    restored = read_event_journal(
        tmp_path / "journal",
        expected_definition_identity=registry.definition_identity,
    ).store

    assert restored == store
    restored_child = restored.get(child.event_id)
    assert restored_child is not None
    assert restored_child.source_event_ids == (parent.event_id,)
    assert restored_child.source_data_ids == ("ohlcv-row:2",)
    assert restored_child.source_entity_ids == ("candidate-level:1",)
    assert restored_child.context_event_ids == (context.event_id,)


def _swap_bar(
    leg: MarketEvent,
    available: dict[str, MarketEvent],
    *,
    seed: str,
    replacement: MarketEvent,
) -> MarketEvent:
    """Replace one cited BAR root in place, keeping every other citation."""

    detector_id = f"detector:{seed}"
    old = next(
        event
        for event in available.values()
        if event.kind is EventKind.BAR_COMPLETED
        and event.evidence.get("detector_candle_id") == detector_id
    )
    del available[old.event_id]
    available[replacement.event_id] = replacement
    return replace(
        leg,
        context_event_ids=tuple(
            replacement.event_id if value == old.event_id else value
            for value in leg.context_event_ids
        ),
    )


def test_foundation_structural_leg_path_admits_a_densified_no_trade_bar() -> None:
    leg, available = _foundation_structural_leg_contract()
    densified = _normalized_bar(
        "foundation-path-16",
        16,
        timeframe=Timeframe.M1,
        open_=100.0,
        high=102.0,
        low=98.5,
        close=99.5,
        real_completed=False,
    )
    admitted = _swap_bar(
        leg, available, seed="foundation-path-16", replacement=densified
    )

    validate_canonical_event(admitted, available_events=available)


def test_foundation_structural_leg_atr_ancestry_rejects_a_densified_bar() -> None:
    leg, available = _foundation_structural_leg_contract()
    densified = _normalized_bar(
        "foundation-prior-7",
        7,
        timeframe=Timeframe.M1,
        open_=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        real_completed=False,
    )
    forged = _swap_bar(
        leg, available, seed="foundation-prior-7", replacement=densified
    )

    with pytest.raises(ValueError, match="real normalized BAR root"):
        validate_canonical_event(forged, available_events=available)


def test_foundation_structural_leg_atr_window_skips_a_densified_prior_bar() -> None:
    """The ATR window reaches past a densified bar, exactly as the producer does."""

    leg, available = _foundation_structural_leg_contract()
    # one extra real bar so a real 14-bar window still exists once minute 7 is
    # densified, mirroring the producer reaching one bar further back
    extra = _normalized_bar(
        "foundation-prior-0",
        0,
        timeframe=Timeframe.M1,
        open_=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
    )
    available[extra.event_id] = extra
    densified = _normalized_bar(
        "foundation-prior-7",
        7,
        timeframe=Timeframe.M1,
        open_=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        real_completed=False,
    )
    stale = next(
        event
        for event in available.values()
        if event.evidence.get("detector_candle_id") == "detector:foundation-prior-7"
    )
    del available[stale.event_id]
    available[densified.event_id] = densified

    atr_bars = tuple(
        sorted(
            (
                event
                for event in available.values()
                if event.kind is EventKind.BAR_COMPLETED
                and event.evidence.get("real_completed") is True
                and event.known_at <= _clock(14)
            ),
            key=lambda event: event.known_at,
        )
    )[-14:]
    assert len(atr_bars) == 14
    path_ids = tuple(leg.context_event_ids[14:])
    admitted = _with_evidence(
        replace(
            leg,
            context_event_ids=(
                *(bar.event_id for bar in atr_bars),
                *path_ids,
            ),
            source_data_ids=(
                *(str(bar.evidence["detector_candle_id"]) for bar in atr_bars),
                *leg.source_data_ids[14:],
            ),
        ),
        atr_source_candle_ids=tuple(
            str(bar.evidence["detector_candle_id"]) for bar in atr_bars
        ),
    )

    validate_canonical_event(admitted, available_events=available)
