from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from smc_trader.calibration_replay import ReplayCheckpointStore
from smc_trader.causal import CausalMarketReader, ReaderUpdate
from smc_trader.displacement import DisplacementLifecycle, DisplacementProtocol
from smc_trader.displacement_observer import CausalDisplacementEye, READER_ANOMALY_WHITELIST
from smc_trader.model import Bar, Candle, EventKind, MarketEvent, Timeframe
from smc_trader.observation import CausalObserver, EventMemory, ObserverConfig

from .helpers import (
    CORE_TEST_SCALE_REGISTRY_ID,
    CORE_TEST_SCALE_SPECS,
    MODEL_SCALE_SPECS,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/primitives_displacement.json"
GROUP12_PROTOCOL_PATH = ROOT / "configs/primitives_structure_liquidity.json"
GROUP3_PROTOCOL_PATH = ROOT / "configs/primitives_zones.json"
EXPERIMENT_ID = "EXP-SMC-3.0.2-013-CAUSAL-5M-DISPLACEMENT-RECOVERY-IDENTITY-CLOSURE"
SEMANTIC_BASE_SHA = "0d7844635ee77a679da87fd47f284719adcca9afc71850ecd0da76addf62455b"
PREREGISTRATION_SHA = "220ddd88eb6decf2d23af7b4368d5b9ea3867588c3f7310564f6160349a6de75"
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
    cached = observer._last_displacement_observation
    monkeypatch.setattr(observer, "_observe_frame", original)
    retried = observer.observe(update)
    assert retried.displacement == cached
    assert retried.displacement is not None
    assert retried.displacement.asof == update.asof


def test_group3_derived_data_anomaly_is_auditable_without_memory_leak() -> None:
    observer = CausalObserver(
        ObserverConfig(
            scale_specs=CORE_TEST_SCALE_SPECS,
            structure_protocol=str(GROUP12_PROTOCOL_PATH),
            liquidity_protocol=str(GROUP12_PROTOCOL_PATH),
            displacement_protocol=str(PROTOCOL_PATH),
            group3_protocol=str(GROUP3_PROTOCOL_PATH),
        )
    )
    index = 0
    for _ in range(15):
        candle = _m5(
            index,
            (100.0, 101.0, 100.0, 100.0),
        )
        observer.observe(_update(candle.end, m5=(candle,)))
        index += 1
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
    boundary = observer.observe(
        _update(off_grid.end, m5=(off_grid,))
    )
    assert boundary.displacement is not None
    assert boundary.displacement.latest_transition is not None
    assert boundary.displacement.latest_transition.reason == "data_anomaly"
    assert boundary.frames[Timeframe.M5].fair_value_gaps == ()
    assert len(boundary.group3_boundary_fvg_transitions) == 1
    invalidated = boundary.group3_boundary_fvg_transitions[0]
    assert invalidated.fvg_id == created_id
    assert invalidated.transition_reason == "data_anomaly"
    assert all(
        not key.startswith("fvg:")
        for key in boundary.retained_entity_timelines
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
    assert all("displacement" not in item.value for item in EventKind)
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


def _bindings() -> dict[str, str]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "semantic_base_sha256": SEMANTIC_BASE_SHA,
        "preregistration_sha256": PREREGISTRATION_SHA,
        "protocol_sha256": _protocol().protocol_hash,
    }


def test_exp013_checkpoint_resume_observation_equivalence(tmp_path: Path) -> None:
    uninterrupted, index, checkpoint_observation = _started_eye()
    store, bindings = ReplayCheckpointStore(tmp_path / "trusted-local"), _bindings()
    state = {
        "replay": uninterrupted, "last_source_start": None, "processed_bars": index,
        "decision_rows": 0, "next_shard_index": 0, "committed_shards": [],
    }
    store.save(state, bindings=bindings)
    assert store.exists
    with pytest.raises(ValueError, match="bindings"):
        store.load(expected_bindings={**bindings, "experiment_id": "EXP-SMC-3.0.2-011-CAUSAL-5M-DISPLACEMENT-ISOLATED-SHADOW"},
                   expected_replay_type=CausalDisplacementEye)
    loaded = store.load(expected_bindings=bindings,
                        expected_replay_type=CausalDisplacementEye)
    resumed = loaded["replay"]
    assert isinstance(resumed, CausalDisplacementEye)
    assert resumed.last_observation == checkpoint_observation
    assert resumed.last_batch == uninterrupted.last_batch
    continuation = _m5(index, (101.0, 102.0, 101.0, 102.0))
    assert _send(uninterrupted, continuation) == _send(resumed, continuation)
    assert uninterrupted.last_batch == resumed.last_batch
    assert uninterrupted.tracker.snapshot() == resumed.tracker.snapshot()
