from pathlib import Path

import pandas as pd

from smc_trader.causal import ReaderUpdate
from smc_trader.displacement import (
    CausalDisplacementTracker,
    DisplacementLifecycle,
    DisplacementProtocol,
    EPISODE_PROTOCOL_VERSION,
)
from smc_trader.displacement_observer import CausalDisplacementEye
from smc_trader.group3 import (
    CausalGroup3Tracker,
    Group3BOSSource,
    Group3Protocol,
)
from smc_trader.model import (
    BOSLifecycle,
    BOSScope,
    BreakOfStructureState,
    Candle,
    Direction,
    FairValueGapLifecycle,
    OrderBlockLifecycle,
    Timeframe,
)
from smc_trader.observation import CausalObserver, ObserverConfig


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/smc_primitives_v3_displacement_episode.json"
GROUP12_PATH = ROOT / "configs/smc_primitives_v3_group12.json"
GROUP3_PATH = ROOT / "configs/smc_primitives_v3_group3.json"
BASE = pd.Timestamp("2025-01-06T09:30:00-05:00")
STRUCTURE_HASH = "1" * 64


def _protocol() -> DisplacementProtocol:
    return DisplacementProtocol.from_file(PROTOCOL_PATH)


def _candle(
    index: int,
    values: tuple[float, float, float, float],
    *,
    symbol: str = "NQH5",
    instrument_id: int = 1,
    real_minutes: int = 5,
    synthetic_minutes: int = 0,
) -> Candle:
    start = BASE + pd.Timedelta(minutes=5 * index)
    return Candle(
        timeframe=Timeframe.M5,
        start=start,
        end=start + pd.Timedelta(minutes=5),
        open=values[0],
        high=values[1],
        low=values[2],
        close=values[3],
        volume=100.0,
        symbol=symbol,
        instrument_id=instrument_id,
        observed_minutes=5,
        expected_minutes=5,
        complete=True,
        real_minutes=real_minutes,
        synthetic_minutes=synthetic_minutes,
    )


def _warm(
    tracker: CausalDisplacementTracker,
    *,
    count: int = 15,
    atr: float = 1.0,
) -> int:
    for index in range(count):
        update = tracker.on_completed_5m(
            _candle(index, (100.0, 100.0 + atr, 100.0, 100.0))
        )
        assert update.state is None
        assert update.transitions == ()
    return count


def _active() -> tuple[CausalDisplacementTracker, int, str]:
    tracker = CausalDisplacementTracker(_protocol())
    index = _warm(tracker)
    started = tracker.on_completed_5m(
        _candle(index, (100.0, 100.75, 99.75, 100.5))
    )
    assert started.state is not None
    entity_id = started.state.entity_id
    active = tracker.on_completed_5m(
        _candle(index + 1, (100.5, 101.5, 100.5, 101.5))
    )
    assert active.state is not None
    assert active.state.lifecycle is DisplacementLifecycle.ACTIVE
    assert active.state.entity_id == entity_id
    return tracker, index + 2, entity_id


def _reader_update(candle: Candle) -> ReaderUpdate:
    minute = Candle(
        timeframe=Timeframe.M1,
        start=candle.end - pd.Timedelta(minutes=1),
        end=candle.end,
        open=candle.close,
        high=candle.close,
        low=candle.close,
        close=candle.close,
        volume=1.0,
        symbol=candle.symbol,
        instrument_id=candle.instrument_id,
        observed_minutes=1,
        expected_minutes=1,
        complete=True,
        real_minutes=1,
        synthetic_minutes=0,
    )
    newly = {timeframe: () for timeframe in Timeframe}
    histories = {timeframe: () for timeframe in Timeframe}
    newly[Timeframe.M1] = (minute,)
    newly[Timeframe.M5] = (candle,)
    histories[Timeframe.M1] = (minute,)
    histories[Timeframe.M5] = (candle,)
    return ReaderUpdate(candle.end, minute, newly, histories, ())


def test_episode_protocol_is_authoritative_after_real_blind_audit() -> None:
    protocol = _protocol()
    assert protocol.protocol_version == EPISODE_PROTOCOL_VERSION
    assert protocol.activation_max_bar == 0
    assert protocol.downstream_authoritative is True

    observer = CausalObserver(
        ObserverConfig(
            structure_protocol=str(GROUP12_PATH),
            displacement_protocol=str(PROTOCOL_PATH),
            group3_protocol=str(GROUP3_PATH),
        )
    )
    assert observer._displacement_downstream_authoritative is True
    assert observer._group3_tracker is not None


def test_first_candidate_starts_before_a_later_large_bar_and_keeps_identity() -> None:
    tracker = CausalDisplacementTracker(_protocol())
    index = _warm(tracker)
    candidate = tracker.on_completed_5m(
        _candle(index, (100.0, 100.75, 99.75, 100.5))
    )
    assert candidate.state is not None
    assert candidate.state.lifecycle is DisplacementLifecycle.STARTED
    assert candidate.state.protection_price == 99.75
    entity_id = candidate.state.entity_id

    large_bar = tracker.on_completed_5m(
        _candle(index + 1, (100.5, 101.5, 100.5, 101.5))
    )
    assert large_bar.state is not None
    assert large_bar.state.lifecycle is DisplacementLifecycle.ACTIVE
    assert large_bar.state.entity_id == entity_id
    assert large_bar.state.started_at == candidate.state.started_at
    assert large_bar.state.protection_price == candidate.state.protection_price


def test_active_episode_survives_one_doji_then_resumes() -> None:
    tracker, index, entity_id = _active()
    pause = tracker.on_completed_5m(
        _candle(index, (101.5, 101.75, 101.25, 101.5))
    )
    assert pause.transitions == ()
    assert pause.state is not None
    assert pause.state.entity_id == entity_id
    assert pause.state.interruption_run == 1
    assert pause.state.neutral_bar_count == 1

    resumed = tracker.on_completed_5m(
        _candle(index + 1, (101.5, 102.25, 101.5, 102.25))
    )
    assert resumed.transitions == ()
    assert resumed.state is not None
    assert resumed.state.entity_id == entity_id
    assert resumed.state.lifecycle is DisplacementLifecycle.ACTIVE
    assert resumed.state.interruption_run == 0
    assert resumed.state.last_favorable_close == 102.25


def test_active_episode_survives_one_small_opposite_bar_then_resumes() -> None:
    tracker, index, entity_id = _active()
    pullback = tracker.on_completed_5m(
        _candle(index, (101.5, 101.75, 101.0, 101.25))
    )
    assert pullback.transitions == ()
    assert pullback.state is not None
    assert pullback.state.entity_id == entity_id
    assert pullback.state.opposite_bar_count == 1
    assert 0.0 < pullback.state.body_continuity < 1.0

    resumed = tracker.on_completed_5m(
        _candle(index + 1, (101.25, 102.0, 101.25, 102.0))
    )
    assert resumed.state is not None
    assert resumed.state.entity_id == entity_id
    assert resumed.state.interruption_run == 0


def test_started_episode_has_no_fixed_activation_deadline() -> None:
    tracker = CausalDisplacementTracker(_protocol())
    index = _warm(tracker, atr=2.0)
    updates = [
        tracker.on_completed_5m(
            _candle(index, (100.0, 100.5, 100.0, 100.5))
        )
    ]
    close = 100.5
    for offset in range(1, 5):
        updates.append(
            tracker.on_completed_5m(
                _candle(
                    index + offset,
                    (close, close + 0.25, close, close + 0.25),
                )
            )
        )
        close += 0.25
    assert all(update.state is not None for update in updates)
    assert all(
        update.state.lifecycle is DisplacementLifecycle.STARTED
        for update in updates
        if update.state is not None
    )

    activated = tracker.on_completed_5m(
        _candle(index + 5, (close, close + 2.25, close, close + 2.25))
    )
    assert activated.state is not None
    assert activated.state.lifecycle is DisplacementLifecycle.ACTIVE
    assert activated.state.real_episode_bar_count == 6


def test_second_interruption_confirms_progress_loss_without_rewriting_prefix() -> None:
    tracker, index, entity_id = _active()
    first = _candle(index, (101.5, 101.75, 101.25, 101.5))
    admitted = tracker.on_completed_5m(first)
    assert admitted.state is not None
    admitted_prefix = admitted.state.prefix_commitment
    admitted_count = admitted.state.real_episode_bar_count

    second = _candle(index + 1, (101.5, 101.75, 101.25, 101.5))
    terminal_update = tracker.on_completed_5m(second)
    assert terminal_update.state is None
    assert len(terminal_update.transitions) == 1
    terminal = terminal_update.transitions[0].state
    assert terminal.entity_id == entity_id
    assert terminal.lifecycle is DisplacementLifecycle.EXHAUSTED
    assert terminal.terminal_reason == "confirmed_progress_loss"
    assert terminal.prefix_commitment == admitted_prefix
    assert terminal.real_episode_bar_count == admitted_count
    assert terminal.prefix_last_admitted_at == first.end
    assert terminal.terminal_evidence_candle_id == tracker._candle_id(second)


def test_protection_break_exhausts_without_admitting_evidence() -> None:
    tracker, index, entity_id = _active()
    prior = tracker.snapshot()
    assert prior is not None
    evidence = _candle(index, (99.5, 99.75, 99.25, 99.5))
    update = tracker.on_completed_5m(evidence)
    terminal = update.transitions[0].state
    assert update.state is None
    assert terminal.entity_id == entity_id
    assert terminal.terminal_reason == "protection_broken"
    assert terminal.prefix_commitment == prior.prefix_commitment
    assert terminal.last_valid_candle_id == prior.last_valid_candle_id


def test_single_strong_opposite_bar_is_only_a_pending_candidate() -> None:
    tracker, index, old_entity_id = _active()
    reverse = _candle(index, (101.5, 101.5, 100.25, 100.5))
    update = tracker.on_completed_5m(reverse)
    assert update.transitions == ()
    assert update.state is not None
    assert update.state.entity_id == old_entity_id
    assert update.state.lifecycle is DisplacementLifecycle.ACTIVE
    assert update.state.interruption_run == 1
    assert tracker._reverse_probe is not None


def test_qualified_reverse_commits_exhausted_started_active_on_same_bar() -> None:
    tracker, index, old_entity_id = _active()
    first = _candle(index, (101.5, 101.5, 100.25, 100.5))
    assert tracker.on_completed_5m(first).transitions == ()
    reverse = _candle(index + 1, (100.5, 100.5, 99.5, 99.5))
    update = tracker.on_completed_5m(reverse)
    assert len(update.transitions) == 3
    exhausted, started, active = update.transitions
    assert exhausted.state.entity_id == old_entity_id
    assert exhausted.state.lifecycle is DisplacementLifecycle.EXHAUSTED
    assert exhausted.state.terminal_reason == "qualified_opposite_displacement"
    assert started.state.lifecycle is DisplacementLifecycle.STARTED
    assert started.state.direction is Direction.SHORT
    assert started.state.entity_id != old_entity_id
    assert active.state.lifecycle is DisplacementLifecycle.ACTIVE
    assert active.state.entity_id == started.state.entity_id
    assert (
        exhausted.state.observed_at
        == started.state.observed_at
        == active.state.observed_at
        == reverse.end
    )
    assert started.state.started_at == first.end
    assert update.state == active.state
    assert started.state.atr0 == exhausted.state.atr0
    assert tuple(tracker._trs)[-1] == 1.0


def test_progress_loss_cannot_same_clock_reseed_same_direction() -> None:
    tracker, index, old_entity_id = _active()
    first = _candle(index, (101.0, 101.25, 100.75, 101.0))
    admitted = tracker.on_completed_5m(first)
    assert admitted.state is not None
    assert admitted.state.interruption_run == 1
    recovery_below_favorable = _candle(
        index + 1,
        (101.0, 101.5, 101.0, 101.25),
    )
    update = tracker.on_completed_5m(recovery_below_favorable)
    assert update.state is None
    assert len(update.transitions) == 1
    assert update.transitions[0].state.entity_id == old_entity_id
    assert (
        update.transitions[0].state.terminal_reason
        == "confirmed_progress_loss"
    )


def test_observer_preserves_complete_same_update_transition_order_and_state() -> None:
    eye = CausalDisplacementEye(_protocol())
    for index in range(15):
        eye.on_update(
            _reader_update(_candle(index, (100.0, 101.0, 100.0, 100.0)))
        )
    eye.on_update(
        _reader_update(_candle(15, (100.0, 100.75, 99.75, 100.5)))
    )
    active = eye.on_update(
        _reader_update(_candle(16, (100.5, 101.5, 100.5, 101.5)))
    )
    old_entity_id = active.current_entity_id

    first_reverse = _candle(17, (101.5, 101.5, 100.25, 100.5))
    pending = eye.on_update(_reader_update(first_reverse))
    assert pending.transitions_this_update == ()
    assert pending.current_entity_id == old_entity_id
    observation = eye.on_update(
        _reader_update(_candle(18, (100.5, 100.5, 99.5, 99.5)))
    )
    exhausted, started, active = observation.transitions_this_update
    assert (exhausted.ordinal, started.ordinal, active.ordinal) == (0, 1, 2)
    assert (exhausted.lifecycle, started.lifecycle, active.lifecycle) == (
        "exhausted",
        "started",
        "active",
    )
    assert exhausted.entity_id == old_entity_id
    assert started.entity_id == observation.current_entity_id
    assert active.entity_id == started.entity_id
    assert observation.recent_transitions[-3:] == (
        exhausted,
        started,
        active,
    )
    assert observation.latest_transition == active
    assert exhausted.terminal_at == observation.asof
    assert dict(exhausted.state_metrics)["protection_price"] == 99.75
    assert dict(started.state_metrics)["real_episode_bar_count"] == 2.0
    assert len(started.admitted_candle_ids) == 2


def test_contract_data_and_synthetic_boundaries_censor_and_reset() -> None:
    for boundary_kind in ("contract", "gap", "synthetic"):
        tracker, index, entity_id = _active()
        if boundary_kind == "contract":
            candle = _candle(
                index,
                (101.5, 102.0, 101.5, 102.0),
                symbol="NQM5",
                instrument_id=2,
            )
            reason = "contract_change_history_reset"
        elif boundary_kind == "gap":
            candle = _candle(
                index + 1,
                (101.5, 102.0, 101.5, 102.0),
            )
            reason = "data_gap_history_reset"
        else:
            candle = _candle(
                index,
                (101.5, 102.0, 101.5, 102.0),
                real_minutes=0,
                synthetic_minutes=5,
            )
            reason = "synthetic_interruption"
        update = tracker.on_completed_5m(candle)
        assert update.state is None
        assert len(update.transitions) == 1
        terminal = update.transitions[0].state
        assert terminal.entity_id == entity_id
        assert terminal.lifecycle is DisplacementLifecycle.CENSORED
        assert terminal.terminal_reason == reason
        assert tracker.snapshot() is None
        assert tuple(tracker._trs) == ()


def _bos(candle: Candle) -> Group3BOSSource:
    state = BreakOfStructureState(
        bos_id="bos-after-interruption",
        timeframe=Timeframe.M5,
        direction=Direction.LONG,
        lifecycle=BOSLifecycle.CONFIRMED,
        scope=BOSScope.LOCAL,
        target_swing_id="swing-before-episode",
        source_structure_id=None,
        target_price=102.0,
        target_ticks=408,
        pending_at=candle.end - pd.Timedelta(minutes=5),
        resolved_at=candle.end,
        age_bars=1,
        failure_reason=None,
        strength=0.8,
    )
    return Group3BOSSource(
        state=state,
        symbol=candle.symbol,
        instrument_id=candle.instrument_id,
        protocol_hash=STRUCTURE_HASH,
        tick_size=0.25,
    )


def test_group3_zones_keep_exact_episode_identity_across_interruption() -> None:
    displacement_protocol = _protocol()
    displacement = CausalDisplacementTracker(displacement_protocol)
    group3 = CausalGroup3Tracker(
        Group3Protocol.from_file(GROUP3_PATH),
        displacement_protocol_hash=displacement_protocol.protocol_hash,
        structure_protocol_hash=STRUCTURE_HASH,
    )

    def send(
        index: int,
        values: tuple[float, float, float, float],
        bos_sources: tuple[Group3BOSSource, ...] = (),
    ):
        candle = _candle(index, values)
        update = displacement.on_completed_5m(candle)
        output = group3.on_completed_5m(candle, update, bos_sources)
        return candle, update, output

    for index in range(64):
        values = (
            (100.25, 100.5, 99.75, 100.0)
            if index == 10
            else (100.0, 101.0, 100.0, 100.0)
        )
        _, update, output = send(index, values)
        assert update.state is None
        assert output.fair_value_gaps == ()
        assert output.order_blocks == ()

    _, started, _ = send(64, (100.0, 100.75, 99.75, 100.5))
    assert started.state is not None
    entity_id = started.state.entity_id
    _, active, _ = send(65, (100.5, 101.5, 100.5, 101.5))
    assert active.state is not None
    assert active.state.lifecycle is DisplacementLifecycle.ACTIVE
    _, pause, _ = send(66, (101.5, 101.75, 101.25, 101.5))
    assert pause.state is not None
    assert pause.state.entity_id == entity_id
    assert pause.state.interruption_run == 1

    trigger_candle = _candle(
        67,
        (101.75, 102.5, 101.75, 102.5),
    )
    _, resumed, output = send(
        67,
        (101.75, 102.5, 101.75, 102.5),
        (_bos(trigger_candle),),
    )
    assert resumed.state is not None
    assert resumed.state.entity_id == entity_id
    assert resumed.state.interruption_run == 0
    assert len(output.fvg_transitions) == 1
    assert len(output.order_block_transitions) == 1
    fvg = output.fvg_transitions[0]
    order_block = output.order_block_transitions[0]
    assert fvg.lifecycle is FairValueGapLifecycle.OPEN
    assert order_block.lifecycle is OrderBlockLifecycle.CREATED
    assert fvg.source_displacement_id == entity_id
    assert order_block.source_displacement_id == entity_id
    assert (
        fvg.source_active_transition_id
        == order_block.source_active_transition_id
    )
