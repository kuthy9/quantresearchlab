from __future__ import annotations

import pandas as pd

from eyes.core.event_store import EventStore, event_order_key
from eyes.core.market_state import (
    DeliveryPhase,
    HierarchicalReplayState,
    MarketSnapshotPublisher,
    RelationRole,
    TimeframeDeliveryState,
    TimeframeLiquidityState,
    TimeframeQualityState,
    TimeframeRangeState,
    TimeframeState,
    TimeframeStructureState,
    TimeframeZoneState,
    build_structural_legs,
    reduce_hierarchical_state,
)
from shares.core.model import (
    Candle,
    Direction,
    EventKind,
    EventOrigin,
    FrameObservation,
    MarketEvent,
    SMC_SEMANTIC_VERSION,
    SwingLifecycle,
    SwingPoint,
    SwingRelation,
    SwingSide,
    Timeframe,
    to_primitive,
)


TZ = "America/New_York"


def _clock(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz=TZ)


def _candle(start: pd.Timestamp, close: float) -> Candle:
    return Candle(
        timeframe=Timeframe.M5,
        start=start,
        end=start + pd.Timedelta(minutes=5),
        open=close - 0.25,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=100.0,
        symbol="NQH5",
        instrument_id=1,
        observed_minutes=5,
        expected_minutes=5,
        complete=True,
    )


def _swing(
    swing_id: str,
    side: SwingSide,
    price: float,
    pivot_start: pd.Timestamp,
    confirmed_at: pd.Timestamp,
) -> SwingPoint:
    return SwingPoint(
        swing_id=swing_id,
        timeframe=Timeframe.M5,
        symbol="NQH5",
        instrument_id=1,
        side=side,
        price=price,
        price_ticks=int(price * 4),
        pivot_start=pivot_start,
        pivot_end=pivot_start + pd.Timedelta(minutes=5),
        observed_at=confirmed_at,
        confirmed_at=confirmed_at,
        lifecycle=SwingLifecycle.CONFIRMED,
        relation=(
            SwingRelation.HH
            if side is SwingSide.HIGH
            else SwingRelation.HL
        ),
        magnitude_atr=1.5,
    )


def _empty_state(
    timeframe: Timeframe,
    *,
    direction: Direction | None,
    internal: Direction | None,
    protected_low: float | None = None,
    last_mss: Direction | None = None,
) -> TimeframeState:
    return TimeframeState(
        timeframe=timeframe,
        structure=TimeframeStructureState(
            external_direction=direction,
            internal_direction=internal,
            protected_low=protected_low,
            protected_high=None,
            protected_low_id=("protected-low" if protected_low else None),
            protected_high_id=None,
            protected_swing_intact=(True if direction else None),
            last_bos=None,
            last_bos_direction=None,
            last_mss=(101.0 if last_mss else None),
            last_mss_direction=last_mss,
            protected_low_event_id=(
                "protected-low-assignment" if protected_low else None
            ),
        ),
        delivery=TimeframeDeliveryState(
            phase=DeliveryPhase.RETRACEMENT,
            active_leg_direction=internal,
            displacement_score=None,
        ),
        range=TimeframeRangeState(
            "range",
            90.0,
            110.0,
            0.45,
            "discount",
            "active",
            ("low", "high"),
        ),
        liquidity=TimeframeLiquidityState((), (), (), ()),
        zones=TimeframeZoneState(),
        quality=TimeframeQualityState(
            data_complete=True,
            semantic_version=SMC_SEMANTIC_VERSION,
            known_at=_clock("2025-01-06 10:00"),
        ),
    )


def test_structural_leg_uses_opposite_confirmed_swings_and_later_known_at() -> None:
    start = _clock("2025-01-06 09:30")
    candles = tuple(
        _candle(start + pd.Timedelta(minutes=5 * index), close)
        for index, close in enumerate((100.0, 101.0, 102.0, 104.0))
    )
    low = _swing(
        "low",
        SwingSide.LOW,
        99.5,
        start,
        start + pd.Timedelta(minutes=10),
    )
    high = _swing(
        "high",
        SwingSide.HIGH,
        104.5,
        start + pd.Timedelta(minutes=15),
        start + pd.Timedelta(minutes=25),
    )

    legs = build_structural_legs(
        Timeframe.M5,
        (low, high),
        candles,
        atr=2.0,
    )

    assert len(legs) == 1
    leg = legs[0]
    assert leg.direction is Direction.LONG
    assert leg.known_at == high.confirmed_at
    assert leg.end_event_time == high.pivot_start
    assert leg.source_swing_ids == (low.swing_id, high.swing_id)
    assert leg.efficiency == 1.0


def test_child_break_cannot_formally_invalidate_parent_structure() -> None:
    parent = _empty_state(
        Timeframe.H1,
        direction=Direction.LONG,
        internal=Direction.LONG,
        protected_low=100.0,
    )
    child = _empty_state(
        Timeframe.M5,
        direction=Direction.SHORT,
        internal=Direction.SHORT,
    )

    relation = MarketSnapshotPublisher._relation(
        parent,
        child,
        price=99.0,
    )

    assert relation.role is RelationRole.PARENT_RETRACEMENT
    assert relation.parent_state_invalidated is False
    assert relation.parent_protected_swing_under_attack is True
    assert relation.reversal_warning is True


def test_parent_acceptance_persists_formal_invalidation_until_replacement() -> None:
    publisher = MarketSnapshotPublisher(
        event_store=EventStore(),
        semantic_registry_identity="registry-test"
    )
    prior = _empty_state(
        Timeframe.H1,
        direction=Direction.LONG,
        internal=Direction.LONG,
        protected_low=100.0,
    ).structure
    publisher._formalize_structure(Timeframe.H1, prior, ())
    transition = TimeframeStructureState(
        external_direction=None,
        internal_direction=Direction.SHORT,
        protected_low=None,
        protected_high=None,
        protected_low_id=None,
        protected_high_id=None,
        protected_swing_intact=None,
        last_bos=None,
        last_bos_direction=None,
        last_mss=100.0,
        last_mss_direction=Direction.SHORT,
    )
    clock = _clock("2025-01-06 11:00")
    accepted = MarketEvent(
        event_id="parent-acceptance",
        kind=EventKind.ACCEPTANCE_CONFIRMED,
        observed_at=clock,
        timeframe=Timeframe.H1,
        side="below",
        price=100.0,
        strength=1.0,
        source_ids=(),
        details={
            "protected_swing_id": "protected-low",
            "protected_swing_event_id": "protected-low-assignment",
        },
        direction=Direction.SHORT,
        event_time=clock,
        known_at=clock,
        source_event_ids=(),
        context_event_ids=("protected-low-assignment",),
        origin=EventOrigin.SEMANTIC_ATOMIC,
    )

    invalidated = publisher._formalize_structure(
        Timeframe.H1,
        transition,
        (accepted,),
    )
    persisted = publisher._formalize_structure(
        Timeframe.H1,
        transition,
        (),
    )

    assert invalidated.external_direction is None
    assert invalidated.protected_low == 100.0
    assert invalidated.protected_swing_intact is False
    assert persisted == invalidated


def test_delivery_phase_uses_one_rule_for_invalidated_active_range() -> None:
    structure = _empty_state(
        Timeframe.H1,
        direction=None,
        internal=None,
    ).structure
    invalidated_range = TimeframeRangeState(
        range_id="range",
        low=90.0,
        high=110.0,
        normalized_location=0.5,
        location_label="equilibrium",
        lifecycle="invalidated",
        source_ids=("low", "high"),
        range_kind="active_dealing_range",
    )

    assert (
        MarketSnapshotPublisher._phase(
            structure,
            invalidated_range,
            (),
        )
        is DeliveryPhase.TRANSITION
    )


def test_stale_opposed_mss_does_not_warn_after_child_realigns() -> None:
    parent = _empty_state(
        Timeframe.H1,
        direction=Direction.LONG,
        internal=Direction.LONG,
        protected_low=100.0,
    )
    child = _empty_state(
        Timeframe.M5,
        direction=Direction.LONG,
        internal=Direction.LONG,
        last_mss=Direction.SHORT,
    )

    relation = MarketSnapshotPublisher._relation(
        parent,
        child,
        price=105.0,
    )

    assert relation.role is RelationRole.ALIGNED_EXPANSION
    assert relation.child_mss_against_parent is False
    assert relation.reversal_warning is False
    assert (
        MarketSnapshotPublisher._phase(
            child.structure,
            child.range,
            (),
        )
        is DeliveryPhase.EXPANSION
    )


def test_snapshot_projection_replays_to_same_hierarchical_state() -> None:
    publisher = MarketSnapshotPublisher(
        event_store=EventStore(),
        semantic_registry_identity="registry-test"
    )
    first_start = _clock("2025-01-06 09:30")
    frames = {
        timeframe: FrameObservation(
            timeframe=timeframe,
            cutoff=first_start + pd.Timedelta(minutes=1),
            bars=1,
            metrics={},
            ready=True,
        )
        for timeframe in (
            Timeframe.H4,
            Timeframe.H1,
            Timeframe.M15,
            Timeframe.M5,
            Timeframe.M1,
        )
    }

    def m1(start: pd.Timestamp, price: float) -> Candle:
        return Candle(
            timeframe=Timeframe.M1,
            start=start,
            end=start + pd.Timedelta(minutes=1),
            open=price,
            high=price + 0.25,
            low=price - 0.25,
            close=price,
            volume=100.0,
            symbol="NQH5",
            instrument_id=1,
            observed_minutes=1,
            expected_minutes=1,
            complete=True,
        )

    first_bar = m1(first_start, 100.0)
    _, first_events, _ = publisher.publish(
        asof=first_bar.end,
        symbol="NQH5",
        instrument_id=1,
        price=first_bar.close,
        completed_1m=first_bar,
        frames=frames,
        inventory=(),
        displacement=None,
        anomalies=(),
    )
    second_bar = m1(first_bar.end, 100.25)
    snapshot, second_events, _ = publisher.publish(
        asof=second_bar.end,
        symbol="NQH5",
        instrument_id=1,
        price=second_bar.close,
        completed_1m=second_bar,
        frames=frames,
        inventory=(),
        displacement=None,
        anomalies=(),
    )
    store = EventStore.from_events(
        sorted((*first_events, *second_events), key=event_order_key)
    )
    replay = store.replay(
        HierarchicalReplayState(),
        reduce_hierarchical_state,
    ).state
    payload = snapshot.replay_payload()

    assert replay.timeframe_states == payload["timeframes"]
    assert replay.relations == payload["relations"]
    assert replay.session == payload["session"]
    assert snapshot.session is not None
    assert "session" not in to_primitive(next(iter(frames.values())))
