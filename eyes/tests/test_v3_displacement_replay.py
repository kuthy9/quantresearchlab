from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import pickle
from eyes.core.causal import CausalMarketReader, ReaderUpdate
from eyes.core.displacement import (
    CausalDisplacementTracker,
    DisplacementLifecycle,
    DisplacementProtocol,
)
from eyes.core.displacement_observer import CausalDisplacementEye, READER_ANOMALY_WHITELIST
from contract.market import (
    Bar,
    Candle,
    Timeframe,
)
from contract.eye import (
    EventKind,
    MarketEvent,
    OrderBlockAttemptOutcome,
)
from eyes.core.observation import CausalObserver, EventMemory, ObserverConfig

from shares.tests.helpers import (
    CORE_TEST_SCALE_REGISTRY_ID,
    CORE_TEST_SCALE_SPECS,
    MODEL_SCALE_SPECS,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/primitives_displacement.json"
GROUP12_PROTOCOL_PATH = ROOT / "configs/primitives_structure_liquidity.json"
GROUP3_PROTOCOL_PATH = ROOT / "configs/primitives_zones.json"
BASE = pd.Timestamp("2025-01-06T09:30:00-05:00")
WHITELIST = frozenset((
    "contract_change_history_reset", "data_gap_history_reset",
    "scheduled_market_closure", "scheduled_weekend_closure",
    "registered_full_session_closure", "registered_abbreviated_good_friday_closure",
    "registered_special_session_closure", "historical_settlement_pause",
    "registered_exchange_closure",
))
BOUNDARIES = (
    (("scheduled_weekend_closure", "contract_change_history_reset",
      "data_gap_history_reset"), "contract_change_history_reset"),
    (("registered_exchange_closure", "contract_change_history_reset"),
     "contract_change_history_reset"),
    (("scheduled_market_closure", "registered_special_session_closure"),
     "registered_session_reset"),
)


def _protocol() -> DisplacementProtocol:
    return DisplacementProtocol.from_file(PROTOCOL_PATH)


def _m5(index: int, values=(100.0, 101.0, 100.0, 101.0)) -> Candle:
    start = BASE + pd.Timedelta(minutes=5 * index)
    return Candle(
        Timeframe.M5, start, start + pd.Timedelta(minutes=5), *values, 100.0,
        "NQH5", 1, 5, 5, True, 5, 0,
    )


def _update(asof: pd.Timestamp, *, m5=(), anomalies=()) -> ReaderUpdate:
    start = asof - pd.Timedelta(minutes=1)
    minute = Candle(
        Timeframe.M1, start, asof, 100.0, 100.25, 99.75, 100.0, 100.0,
        "NQH5", 1, 1, 1, True, 1, 0,
    )
    active_timeframes = tuple(
        spec.native_timeframe
        for spec in CORE_TEST_SCALE_SPECS
        if spec.enabled and spec.native_timeframe is not None
    )
    newly = {timeframe: () for timeframe in active_timeframes}
    histories = {timeframe: () for timeframe in active_timeframes}
    newly[Timeframe.M1], newly[Timeframe.M5] = (minute,), m5
    histories[Timeframe.M1], histories[Timeframe.M5] = (minute,), m5
    return ReaderUpdate(
        asof=asof,
        completed_1m=minute,
        newly_completed=newly,
        histories=histories,
        anomalies=anomalies,
        active_timeframes=active_timeframes,
        scale_specs=CORE_TEST_SCALE_SPECS,
        scale_registry_id=CORE_TEST_SCALE_REGISTRY_ID,
    )


def _send(eye: CausalDisplacementEye, candle: Candle):
    return eye.on_update(_update(candle.end, m5=(candle,)))


def _warm_eye(eye: CausalDisplacementEye) -> int:
    for index in range(15):
        observation = _send(eye, _m5(index, (100.0, 101.0, 100.0, 100.0)))
        assert observation.lifecycle == "idle"
    return 15


def _started_eye():
    eye = CausalDisplacementEye(_protocol())
    index = _warm_eye(eye)
    observation = _send(eye, _m5(index))
    assert observation.lifecycle == "started"
    return eye, index + 1, observation


def _bar(index: int) -> Bar:
    return Bar(
        BASE + pd.Timedelta(minutes=index), 100.0, 100.25, 99.75, 100.0,
        100.0, "NQH5", 1,
    )


def test_exp013_no_advance_before_completed_m5() -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    eye = CausalDisplacementEye(_protocol())
    for index in range(4):
        update = reader.on_bar(_bar(index))
        observation = eye.on_update(update)
        assert update.newly_completed[Timeframe.M5] == ()
        assert (observation.lifecycle, observation.recent_transitions) == ("idle", ())
    assert eye.tracker._prior_close is None
    update = reader.on_bar(_bar(4))
    observation = eye.on_update(update)
    assert (len(update.newly_completed[Timeframe.M5]), observation.lifecycle) == (1, "idle")
    assert eye.tracker._prior_close == update.newly_completed[Timeframe.M5][0].close


def test_exp013_exactly_once_per_completed_m5() -> None:
    eye = CausalDisplacementEye(_protocol())
    first, second = _m5(0, (100.0, 100.5, 99.5, 100.0)), _m5(1)
    update = _update(second.end, m5=(first, second))
    histories = {**update.histories, Timeframe.M5: ()}
    observation = eye.on_update(replace(update, histories=histories))
    assert (observation.lifecycle, observation.recent_transitions) == ("idle", ())
    assert (eye.tracker._prior_close, tuple(eye.tracker._trs),
            eye.tracker._last_clock) == (second.close, (1.0,), second.end)


def test_exp013_last_batch_preserves_each_completed_m5_update() -> None:
    eye = CausalDisplacementEye(_protocol())
    assert eye.last_batch == ()
    index = _warm_eye(eye)
    seed = _m5(index)
    continuation = _m5(index + 1, (101.0, 102.0, 101.0, 102.0))

    observation = eye.on_update(
        _update(continuation.end, m5=(seed, continuation))
    )

    assert tuple(candle for candle, _ in eye.last_batch) == (
        seed,
        continuation,
    )
    seed_update, continuation_update = (
        raw for _, raw in eye.last_batch
    )
    assert (
        seed_update.state.lifecycle,
        continuation_update.state.lifecycle,
    ) == (
        DisplacementLifecycle.STARTED,
        DisplacementLifecycle.ACTIVE,
    )
    assert tuple(
        transition
        for _, raw in eye.last_batch
        for transition in raw.transitions
    ) == eye.last_update.transitions
    assert eye.last_update.state == continuation_update.state
    assert observation.lifecycle == "active"

    eye.on_update(_update(continuation.end + pd.Timedelta(minutes=1)))
    assert eye.last_batch == ()
    assert eye.last_update.state == continuation_update.state
    assert eye.last_update.transitions == ()


@pytest.mark.parametrize(("anomalies", "reason"), BOUNDARIES,
                         ids=["contract-over-gap", "contract", "session"])
def test_exp013_boundary_priority(anomalies: tuple[str, ...], reason: str) -> None:
    eye, index, _ = _started_eye()
    observation = eye.on_update(_update(_m5(index).end, anomalies=anomalies))
    assert READER_ANOMALY_WHITELIST == WHITELIST
    assert (observation.lifecycle, observation.reader_anomalies) == ("idle", anomalies)
    assert observation.latest_transition is not None
    assert (observation.latest_transition.lifecycle,
            observation.latest_transition.reason) == ("censored", reason)
    assert eye.tracker.snapshot() is None


def test_exp013_unknown_anomaly_fails_before_mutation() -> None:
    eye, index, prior_observation = _started_eye()
    before = (
        repr(eye.tracker.__dict__),
        tuple(eye._transitions),
        eye.last_observation,
        eye.last_batch,
    )
    candidate = _m5(index, (101.0, 102.0, 101.0, 102.0))
    with pytest.raises(ValueError, match="unknown"):
        eye.on_update(_update(candidate.end, m5=(candidate,),
                              anomalies=("unregistered_boundary",)))
    after = (
        repr(eye.tracker.__dict__),
        tuple(eye._transitions),
        eye.last_observation,
        eye.last_batch,
    )
    assert before == after
    assert eye.last_observation == prior_observation


def test_exp013_boundary_m5_xor() -> None:
    eye, index, _ = _started_eye()
    candidate = _m5(index, (101.0, 101.0, 99.0, 99.0))
    observation = eye.on_update(
        _update(candidate.end, m5=(candidate,),
                anomalies=("contract_change_history_reset",))
    )
    assert (observation.lifecycle, observation.recent_transitions[-1].lifecycle) == (
        "idle", "censored",
    )
    assert (eye.tracker.snapshot(), eye.tracker._prior_close,
            tuple(eye.tracker._trs)) == (None, None, ())
    assert eye.last_batch == ()


def test_displacement_eye_rejects_off_grid_m5_before_boundary_mutation() -> None:
    eye, index, _ = _started_eye()
    candidate = _m5(index, (101.0, 102.0, 101.0, 101.1))
    before = (
        repr(eye.tracker.__dict__),
        tuple(eye._transitions),
        eye.last_observation,
        eye.last_update,
        eye.last_batch,
    )

    with pytest.raises(ValueError, match="off-grid"):
        eye.on_update(
            _update(
                candidate.end,
                m5=(candidate,),
                anomalies=("contract_change_history_reset",),
            )
        )

    assert (
        repr(eye.tracker.__dict__),
        tuple(eye._transitions),
        eye.last_observation,
        eye.last_update,
        eye.last_batch,
    ) == before


@pytest.mark.parametrize(
    "case",
    ("real", "gap", "synthetic"),
    ids=("real-candle", "gap-boundary", "synthetic-boundary"),
)
def test_displacement_rejects_off_grid_before_any_state_change(
    case: str,
) -> None:
    tracker = CausalDisplacementTracker(_protocol())
    for index in range(15):
        tracker.on_completed_5m(
            _m5(index, (100.0, 101.0, 100.0, 100.0))
        )
    started = tracker.on_completed_5m(_m5(15))
    assert started.state is not None
    assert started.state.lifecycle is DisplacementLifecycle.STARTED
    before = repr(tracker.__dict__)
    malformed = _m5(
        17 if case == "gap" else 16,
        (101.0, 102.0, 101.0, 101.1),
    )
    if case == "synthetic":
        malformed = replace(
            malformed,
            real_minutes=0,
            synthetic_minutes=5,
        )

    with pytest.raises(ValueError, match="off-grid"):
        tracker.on_completed_5m(malformed)

    assert repr(tracker.__dict__) == before
    assert tracker._failed is False


def test_displacement_rejects_retry_from_different_stored_grid() -> None:
    tracker = CausalDisplacementTracker(_protocol())
    candle = _m5(0, (100.0, 101.0, 100.0, 100.0))
    tracker.on_completed_5m(candle)
    retry = replace(
        candle,
        price_tick_size=0.5,
        normalized_ohlc_ticks=None,
    )
    assert retry == candle
    before = repr(tracker.__dict__)

    with pytest.raises(ValueError, match="grid disagrees"):
        tracker.on_completed_5m(retry)

    assert repr(tracker.__dict__) == before
    assert tracker._failed is False


def test_observer_reuses_displacement_on_exact_boundary_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = CausalMarketReader(scale_specs=MODEL_SCALE_SPECS)
    observer = CausalObserver(
        ObserverConfig(
            scale_specs=MODEL_SCALE_SPECS,
            displacement_protocol=str(PROTOCOL_PATH),
        )
    )
    observer.observe(reader.on_bar(_bar(0)))
    update = reader.on_bar(
        replace(
            _bar(10),
            data_gap_before_minutes=9,
        )
    )
    original = observer._observe_frame

    def fail_once(*args, **kwargs):
        raise RuntimeError("frame failure after displacement")

    monkeypatch.setattr(observer, "_observe_frame", fail_once)
    with pytest.raises(
        RuntimeError,
        match="frame failure after displacement",
    ):
        observer.observe(update)
    cached = observer._last_displacement_inputs[Timeframe.M5][1]
    monkeypatch.setattr(observer, "_observe_frame", original)
    retried = observer.observe(update)
    assert retried.displacement == cached
    assert retried.displacement is not None
    assert retried.displacement.asof == update.asof


def test_observer_synthetic_displacement_terminal_uses_context_only_clock_root() -> None:
    observer = CausalObserver(
        ObserverConfig(
            scale_specs=CORE_TEST_SCALE_SPECS,
            displacement_protocol=str(PROTOCOL_PATH),
        )
    )
    for index in range(15):
        observer.observe(
            _update(
                _m5(index).end,
                m5=(_m5(index, (100.0, 101.0, 100.0, 100.0)),),
            )
        )
    observer.observe(_update(_m5(15).end, m5=(_m5(15),)))
    active = _m5(16, (101.0, 102.0, 101.0, 102.0))
    active_observation = observer.observe(
        _update(active.end, m5=(active,))
    )
    assert active_observation.displacement is not None
    assert active_observation.displacement.lifecycle == "active"

    synthetic_m5 = replace(
        _m5(17, (102.0, 102.0, 102.0, 102.0)),
        real_minutes=4,
        synthetic_minutes=1,
    )
    base_update = _update(synthetic_m5.end, m5=(synthetic_m5,))
    for minute_offset in range(4, 0, -1):
        observer.observe(
            _update(
                synthetic_m5.end
                - pd.Timedelta(minutes=minute_offset)
            )
        )
    synthetic_m1 = Candle(
        timeframe=Timeframe.M1,
        start=synthetic_m5.end - pd.Timedelta(minutes=1),
        end=synthetic_m5.end,
        open=102.0,
        high=102.0,
        low=102.0,
        close=102.0,
        volume=0.0,
        symbol="NQH5",
        instrument_id=1,
        observed_minutes=1,
        expected_minutes=1,
        complete=True,
        real_minutes=0,
        synthetic_minutes=1,
    )
    newly = {**base_update.newly_completed, Timeframe.M1: (synthetic_m1,)}
    histories = {**base_update.histories, Timeframe.M1: (synthetic_m1,)}
    observation = observer.observe(
        replace(
            base_update,
            completed_1m=synthetic_m1,
            newly_completed=newly,
            histories=histories,
        )
    )
    terminal = next(
        event
        for event in observation.semantic_events_this_update
        if (
            event.kind is EventKind.DISPLACEMENT_OBSERVED
            and event.evidence.get("lifecycle") == "censored"
            and event.evidence.get("terminal_reason")
            == "synthetic_interruption"
        )
    )
    assert len(terminal.context_event_ids) == 1
    context = observer.audit_store.get(terminal.context_event_ids[0])
    assert context is not None
    assert context.kind is EventKind.BAR_COMPLETED
    assert context.timeframe is Timeframe.M1
    assert context.known_at == terminal.known_at == synthetic_m1.end
    assert context.evidence["clock_only"] is True
    assert context.evidence["real_completed"] is False
    assert context.event_id not in terminal.source_event_ids
    source_events = tuple(
        observer.audit_store.get(event_id)
        for event_id in terminal.source_event_ids
    )
    assert source_events
    assert all(event is not None for event in source_events)
    assert all(event.evidence["real_completed"] is True for event in source_events)
    expected_detector_ids = tuple(
        sorted(event.evidence["detector_candle_id"] for event in source_events)
    )
    assert terminal.source_data_ids == tuple(
        terminal.evidence["admitted_candle_ids"]
    )
    assert tuple(sorted(terminal.source_data_ids)) == expected_detector_ids
    assert len(terminal.source_data_ids) == len(set(terminal.source_data_ids))
    assert not set(terminal.source_data_ids).intersection(context.source_data_ids)
    assert context.evidence["detector_candle_id"] not in terminal.source_data_ids
    assert max(event.known_at for event in source_events) < terminal.known_at


def test_synthetic_displacement_terminal_uses_all_interior_m1_roots() -> None:
    observer = CausalObserver(
        ObserverConfig(
            scale_specs=CORE_TEST_SCALE_SPECS,
            displacement_protocol=str(PROTOCOL_PATH),
        )
    )
    for index in range(15):
        observer.observe(
            _update(
                _m5(index).end,
                m5=(_m5(index, (100.0, 101.0, 100.0, 100.0)),),
            )
        )
    observer.observe(_update(_m5(15).end, m5=(_m5(15),)))
    active = _m5(16, (101.0, 102.0, 101.0, 102.0))
    observer.observe(_update(active.end, m5=(active,)))

    incomplete_m5 = replace(
        _m5(17, (102.0, 102.0, 102.0, 102.0)),
        real_minutes=3,
        synthetic_minutes=2,
    )
    synthetic_ends = (
        incomplete_m5.end - pd.Timedelta(minutes=3),
        incomplete_m5.end - pd.Timedelta(minutes=2),
    )
    for minute_end in (
        incomplete_m5.end - pd.Timedelta(minutes=4),
        *synthetic_ends,
        incomplete_m5.end - pd.Timedelta(minutes=1),
    ):
        update = _update(minute_end)
        if minute_end in synthetic_ends:
            synthetic_m1 = replace(
                update.completed_1m,
                volume=0.0,
                real_minutes=0,
                synthetic_minutes=1,
            )
            update = replace(
                update,
                completed_1m=synthetic_m1,
                newly_completed={
                    **update.newly_completed,
                    Timeframe.M1: (synthetic_m1,),
                },
                histories={
                    **update.histories,
                    Timeframe.M1: (synthetic_m1,),
                },
            )
        observer.observe(update)

    observation = observer.observe(
        _update(incomplete_m5.end, m5=(incomplete_m5,))
    )
    terminal = next(
        event
        for event in observation.semantic_events_this_update
        if (
            event.kind is EventKind.DISPLACEMENT_OBSERVED
            and event.evidence.get("lifecycle") == "censored"
            and event.evidence.get("terminal_reason")
            == "synthetic_interruption"
        )
    )
    roots = tuple(
        observer.audit_store.get(event_id)
        for event_id in terminal.context_event_ids
    )

    assert all(root is not None for root in roots)
    assert tuple(root.known_at for root in roots if root is not None) == (
        synthetic_ends
    )
    assert all(
        root.evidence["clock_only"] is True
        and root.evidence["real_completed"] is False
        for root in roots
        if root is not None
    )
    assert terminal.known_at == incomplete_m5.end
    assert terminal.known_at not in synthetic_ends


def test_synthetic_terminal_rejects_incomplete_m1_constituent_index() -> None:
    observer = CausalObserver(
        ObserverConfig(
            scale_specs=CORE_TEST_SCALE_SPECS,
            displacement_protocol=str(PROTOCOL_PATH),
        )
    )
    terminal_clock = BASE + pd.Timedelta(minutes=5)
    for minute_offset in (4, 3, 1, 0):
        observer.observe(
            _update(
                terminal_clock
                - pd.Timedelta(minutes=minute_offset)
            )
        )

    with pytest.raises(ValueError, match="five contiguous unique M1 roots"):
        observer._emitter._synthetic_m1_context_event_ids_for_m5_terminal(
            terminal_clock
        )


def test_off_grid_detector_candle_fails_closed_before_group3_projection() -> None:
    observer = CausalObserver(
        ObserverConfig(
            scale_specs=CORE_TEST_SCALE_SPECS,
            structure_protocol=str(GROUP12_PROTOCOL_PATH),
            liquidity_protocol=str(GROUP12_PROTOCOL_PATH),
            displacement_protocol=str(PROTOCOL_PATH),
            zone_protocol=str(GROUP3_PROTOCOL_PATH),
        )
    )
    index = 0
    warm_observation = None
    for _ in range(15):
        candle = _m5(
            index,
            (100.0, 101.0, 100.0, 100.0),
        )
        warm_observation = observer.observe(
            _update(candle.end, m5=(candle,))
        )
        index += 1
    assert warm_observation is not None
    assert len(warm_observation.group3_order_block_funnel) == 1
    assert (
        warm_observation.group3_order_block_funnel[0].outcome
        is OrderBlockAttemptOutcome.NO_ACTIVE_DISPLACEMENT
    )
    for values in (
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 101.5, 100.0, 101.5),
        (101.5, 102.25, 101.5, 102.25),
    ):
        candle = _m5(index, values)
        before_boundary = observer.observe(
            _update(candle.end, m5=(candle,))
        )
        index += 1
    created = before_boundary.frames[Timeframe.M5].fair_value_gaps
    assert len(created) == 1
    created_id = created[0].fvg_id

    off_grid = _m5(
        index,
        (102.25, 102.5, 102.0, 102.1),
    )
    before_displacement = repr(observer._displacement_eye.__dict__)
    before_group3 = repr(observer._zone_tracker.__dict__)
    with pytest.raises(ValueError, match="off-grid"):
        observer.observe(_update(off_grid.end, m5=(off_grid,)))

    # This path intentionally bypasses the production Reader.  Exact grid
    # admission still precedes every detector boundary/cache path, so the
    # corrected same-clock input remains replayable without rounded ancestry.
    assert repr(observer._displacement_eye.__dict__) == before_displacement
    assert repr(observer._zone_tracker.__dict__) == before_group3
    assert observer._terminal_failure is None
    corrected = replace(off_grid, close=102.25)
    recovered = observer.observe(
        _update(corrected.end, m5=(corrected,))
    )
    assert any(
        state.fvg_id == created_id
        for state in recovered.frames[Timeframe.M5].fair_value_gaps
    )


def test_exp013_terminal_projection_keeps_current_and_history_separate() -> None:
    eye, index, _ = _started_eye()
    first_pause = _m5(index, (101.0, 101.25, 100.75, 101.0))
    first = _send(eye, first_pause)
    assert first.lifecycle == "started"
    evidence = _m5(index + 1, (101.0, 101.25, 100.75, 101.0))
    observation = _send(eye, evidence)
    current = (
        observation.current_entity_id, observation.current_direction,
        observation.current_state_observed_at, observation.current_started_at,
        observation.current_active_at, observation.current_last_admitted_at,
        observation.current_metrics,
    )
    assert (observation.lifecycle, current) == ("idle", (None,) * 7)
    assert observation.latest_transition == observation.recent_transitions[-1]
    assert (observation.latest_transition.lifecycle,
            observation.latest_transition.reason,
            observation.latest_transition.observed_at) == (
        "exhausted", "confirmed_progress_loss", evidence.end,
    )


def test_exp013_same_clock_exhausted_then_started_projection() -> None:
    eye, index, prior = _started_eye()
    candidate = _m5(index, (101.0, 101.0, 99.0, 99.0))
    observation = _send(eye, candidate)
    exhausted, started = observation.recent_transitions[-2:]
    assert (exhausted.lifecycle, started.lifecycle) == ("exhausted", "started")
    assert exhausted.entity_id == prior.current_entity_id
    assert started.entity_id == observation.current_entity_id != exhausted.entity_id
    assert exhausted.observed_at == started.observed_at == candidate.end
    assert (observation.lifecycle, observation.latest_transition) == ("started", started)
    assert tuple(eye.tracker._trs).count(2.0) == 1


def test_exp013_continuous_view_freezes_state_clocks() -> None:
    eye, _, started = _started_eye()
    first = eye.on_update(_update(started.asof + pd.Timedelta(minutes=1)))
    second = eye.on_update(_update(started.asof + pd.Timedelta(minutes=2)))
    assert replace(
        started,
        asof=first.asof,
        transitions_this_update=(),
    ) == first
    assert replace(
        started,
        asof=second.asof,
        transitions_this_update=(),
    ) == second
    assert first.asof < second.asof
    assert second.current_state_observed_at == started.current_state_observed_at


def test_exp013_isolated_transition_capacity_noninterference() -> None:
    memory, eye = EventMemory(1), CausalDisplacementEye(_protocol())
    before = memory.recent()
    index = _warm_eye(eye)
    seeded = _send(eye, _m5(index))
    seed_id = seeded.recent_transitions[-1].transition_id
    observation = _send(
        eye,
        _m5(index + 1, (101.0, 102.0, 101.0, 102.0)),
    )
    price = 102.0
    candle_index = index + 2
    direction = -1
    for _ in range(24):
        first_close = price + direction * 1.0
        first = (
            (price, price + 1.25, price, first_close)
            if direction > 0
            else (price, price, price - 1.25, first_close)
        )
        pending = _send(eye, _m5(candle_index, first))
        assert pending.transitions_this_update == ()
        candle_index += 1
        second_close = first_close + direction * 1.0
        second = (
            (first_close, first_close + 1.25, first_close, second_close)
            if direction > 0
            else (first_close, first_close, first_close - 1.25, second_close)
        )
        observation = _send(eye, _m5(candle_index, second))
        assert tuple(
            item.lifecycle for item in observation.transitions_this_update
        ) == ("exhausted", "started", "active")
        candle_index += 1
        price = second_close
        direction *= -1
    ids = {item.transition_id for item in observation.recent_transitions}
    assert (len(observation.recent_transitions), len(ids)) == (64, 64)
    assert seed_id not in ids
    assert not any(isinstance(item, MarketEvent) for item in observation.recent_transitions)
    # Displacement transitions remain isolated from this EventMemory even
    # though the public phase-2 semantic contract now registers an atomic
    # ``displacement_observed`` event kind for the full Observer pipeline.
    assert EventKind.DISPLACEMENT_OBSERVED.value == "displacement_observed"
    assert not hasattr(eye, "memory")
    assert memory.recent() == before


def test_exp013_prefix_invariance_under_different_future_suffixes() -> None:
    left, right = CausalDisplacementEye(_protocol()), CausalDisplacementEye(_protocol())
    common = tuple(_m5(i, (100.0, 101.0, 100.0, 100.0)) for i in range(15))
    common += (_m5(15),)
    left_prefix = tuple(_send(left, candle) for candle in common)
    right_prefix = tuple(_send(right, candle) for candle in common)
    frozen_prefix = left_prefix
    assert left_prefix == right_prefix
    _send(left, _m5(16, (101.0, 102.0, 101.0, 102.0)))
    _send(right, _m5(16, (101.0, 101.0, 99.0, 99.0)))
    assert left_prefix == right_prefix == frozen_prefix
    assert left_prefix[-1].current_entity_id == right_prefix[-1].current_entity_id


def test_exp013_checkpoint_resume_observation_equivalence(tmp_path: Path) -> None:
    """A pickled displacement eye resumes with the observations it would have seen.

    The typed Brain's ``ReplayCheckpointStore`` that once wrapped this pickle was
    retired; the Eye-side property it exercised is the round trip itself.
    """
    uninterrupted, index, checkpoint_observation = _started_eye()
    checkpoint = tmp_path / "trusted-local" / "checkpoint.pkl"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(pickle.dumps(uninterrupted))
    resumed = pickle.loads(checkpoint.read_bytes())
    assert isinstance(resumed, CausalDisplacementEye)
    assert resumed.last_observation == checkpoint_observation
    assert resumed.last_batch == uninterrupted.last_batch
    continuation = _m5(index, (101.0, 102.0, 101.0, 102.0))
    assert _send(uninterrupted, continuation) == _send(resumed, continuation)
    assert uninterrupted.last_batch == resumed.last_batch
    assert uninterrupted.tracker.snapshot() == resumed.tracker.snapshot()


def test_synthetic_terminal_on_a_15m_scale_reads_fifteen_m1_constituents() -> None:
    # A secondary displacement scale's synthetic-interruption terminal cites
    # the clock-only minutes of its own incomplete bar: fifteen constituents
    # on 15m, not the five of the 5m scale the check was first written for.
    # The 2022-01-03 23:13 missing minute on the NQ tape raised
    # "lacks five contiguous unique M1 roots" from a 15m terminal.
    observer = CausalObserver(
        ObserverConfig(
            scale_specs=CORE_TEST_SCALE_SPECS,
            displacement_protocol=str(PROTOCOL_PATH),
        )
    )
    terminal_clock = BASE + pd.Timedelta(minutes=15)
    synthetic_end = terminal_clock - pd.Timedelta(minutes=2)
    for minute_offset in range(14, -1, -1):
        update = _update(terminal_clock - pd.Timedelta(minutes=minute_offset))
        if update.asof == synthetic_end:
            synthetic_m1 = replace(
                update.completed_1m,
                volume=0.0,
                real_minutes=0,
                synthetic_minutes=1,
            )
            update = replace(
                update,
                completed_1m=synthetic_m1,
                newly_completed={**update.newly_completed, Timeframe.M1: (synthetic_m1,)},
                histories={**update.histories, Timeframe.M1: (synthetic_m1,)},
            )
        observer.observe(update)

    context = observer._emitter._synthetic_m1_context_event_ids_for_terminal(
        terminal_clock, timeframe=Timeframe.M15
    )
    roots = tuple(observer.audit_store.get(event_id) for event_id in context)
    assert tuple(root.known_at for root in roots) == (synthetic_end,)
    assert roots[0].evidence["clock_only"] is True

    # Fourteen of the fifteen constituents is not a covered 15m interval.
    gapped = CausalObserver(
        ObserverConfig(
            scale_specs=CORE_TEST_SCALE_SPECS,
            displacement_protocol=str(PROTOCOL_PATH),
        )
    )
    for minute_offset in range(14, -1, -1):
        if minute_offset == 7:
            continue
        gapped.observe(_update(terminal_clock - pd.Timedelta(minutes=minute_offset)))
    with pytest.raises(ValueError, match="fifteen contiguous unique M1 roots"):
        gapped._emitter._synthetic_m1_context_event_ids_for_terminal(
            terminal_clock, timeframe=Timeframe.M15
        )
