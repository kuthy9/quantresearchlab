from pathlib import Path

import pandas as pd
import pytest

from smc_trader.displacement import CausalDisplacementTracker
from smc_trader.displacement import DisplacementLifecycle, DisplacementProtocol
from smc_trader.model import Candle, Direction, Timeframe


pytestmark = pytest.mark.skip(
    reason="archived EXP013 semantics; replaced by test_displacement_episode.py"
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/experiments/EXP-SMC-3.0.2-004-CAUSAL-5M-DISPLACEMENT-DUAL-CLOCK.json"
PROTOCOL_SHA = "850a97998feda2e2c2556fd79da0bf5b5f1975feb63543824f5a3a218f8e47a5"
BASE = pd.Timestamp("2025-01-06T09:30:00-05:00")
SEED_FAILURES = (
    (6.25, (101.0, 105.0, 100.0, 103.75)),
    (6.25, (100.5, 105.0, 100.0, 103.5)),
    (6.50, (100.75, 105.0, 100.0, 103.75)),
)
ACTIVATIONS = (
    (1.25, (100.0, 101.0, 100.0, 100.75), ((100.75, 101.25, 100.75, 101.25),)),
    (2.50, (100.0, 101.5, 99.5, 101.25), (
        (101.25, 101.5, 101.25, 101.5),
        (101.5, 101.75, 101.5, 101.75),
        (101.75, 103.0, 101.75, 103.0),
    )),
)
CONTINUATIONS = (
    ((101.75, 102.0, 101.75, 102.0), True, None),
    ((101.5, 102.0, 101.5, 101.75), False, "no_directional_close_progress"),
)
TERMINALS = (
    ((101.75, 102.0, 101.5, 101.75), "doji_body"),
    ((101.75, 101.75, 101.25, 101.25), "opposite_body"),
    ((101.5, 102.0, 101.5, 101.75), "no_directional_close_progress"),
)
CENSORSHIP = (
    ("synthetic", "synthetic_interruption"),
    ("boundary", "registered_session_reset"),
    ("data-anomaly", "data_anomaly"),
)


def _protocol() -> DisplacementProtocol:
    return DisplacementProtocol.from_file(PROTOCOL_PATH)


def _candle(index: int, values=(100.0, 101.0, 100.0, 101.0), *,
            volume: float = 100.0, real: int = 5, synthetic: int = 0) -> Candle:
    start = BASE + pd.Timedelta(minutes=5 * index)
    return Candle(
        Timeframe.M5, start, start + pd.Timedelta(minutes=5), *values, volume,
        "NQH5", 1, 5, 5, True, real, synthetic,
    )


def _tracker() -> CausalDisplacementTracker:
    return CausalDisplacementTracker(_protocol())


def _warm(tracker: CausalDisplacementTracker, *, atr=1.0, count=15, volume=100.0) -> int:
    for index in range(count):
        update = tracker.on_completed_5m(
            _candle(index, (100.0, 100.0 + atr, 100.0, 100.0), volume=volume)
        )
        assert (update.state, update.transitions) == (None, ())
    return count


def _started(*, atr=1.0, seed=(100.0, 101.0, 100.0, 101.0), volume=100.0):
    tracker = _tracker()
    index = _warm(tracker, atr=atr, volume=volume)
    candle = _candle(index, seed, volume=volume)
    update = tracker.on_completed_5m(candle)
    assert update.state is not None
    assert update.state.lifecycle is DisplacementLifecycle.STARTED
    return tracker, index + 1, candle


def _active():
    tracker, index, _ = _started()
    update = tracker.on_completed_5m(_candle(index, (101.0, 101.75, 101.0, 101.75)))
    assert update.state is not None
    assert update.state.lifecycle is DisplacementLifecycle.ACTIVE
    return tracker, index + 1


def test_exp013_protocol_constants() -> None:
    protocol = _protocol()
    actual = (
        protocol.atr_baseline_bars, protocol.seed_body_fraction,
        protocol.seed_directional_clv, protocol.seed_tr_atr,
        protocol.activation_min_bar, protocol.activation_max_bar,
        protocol.activation_relative_atr, protocol.activation_efficiency,
        protocol.activation_speed, protocol.activation_mean_body_fraction,
        protocol.activation_min_directional_clv, protocol.continuation_progress_ticks,
    )
    assert protocol.protocol_hash == PROTOCOL_SHA
    assert actual == (14, .60, .75, .80, 2, 4, 1., .70, .30, .60, .70, 1)


def test_exp013_first_m5_seeds_prior_close_only() -> None:
    tracker = _tracker()
    first = _candle(0)
    update = tracker.on_completed_5m(first)
    assert (update.state, update.transitions, tracker.snapshot()) == (None, (), None)
    assert (tracker._prior_close, tuple(tracker._trs)) == (first.close, ())


def test_exp013_strict_prior_baseline_excludes_candidate() -> None:
    tracker = _tracker()
    index = _warm(tracker, count=14)
    assert len(tracker._trs) == 13
    candidate = tracker.on_completed_5m(_candle(index))
    assert (candidate.state, candidate.transitions, len(tracker._trs)) == (None, (), 14)
    assert tracker.on_completed_5m(_candle(index + 1)).state is not None


def test_exp013_seed_equal_boundaries() -> None:
    tracker = _tracker()
    index = _warm(tracker, atr=6.25)
    state = tracker.on_completed_5m(
        _candle(index, (100.75, 105.0, 100.0, 103.75))
    ).state
    assert state is not None
    assert state.lifecycle is DisplacementLifecycle.STARTED
    assert (state.mean_body_fraction, state.min_directional_clv) == pytest.approx((.60, .75))
    assert 5.0 / state.atr0 == pytest.approx(.80)


@pytest.mark.parametrize(("atr", "values"), SEED_FAILURES,
                         ids=["body-fraction", "directional-clv", "tr-atr"])
def test_exp013_seed_just_below(atr: float, values: tuple[float, ...]) -> None:
    tracker = _tracker()
    index = _warm(tracker, atr=atr)
    update = tracker.on_completed_5m(_candle(index, values))
    assert (update.state, update.transitions) == (None, ())


def test_exp013_volume_unready_is_not_gate() -> None:
    tracker = _tracker()
    index = _warm(tracker, volume=0.0)
    state = tracker.on_completed_5m(_candle(index, volume=0.0)).state
    assert state is not None
    assert (state.lifecycle, state.volume_ready, state.v0, state.volume_ratio) == (
        DisplacementLifecycle.STARTED, False, None, None,
    )


def test_exp013_started_fields_and_clocks() -> None:
    tracker, _, seed = _started()
    state = tracker.snapshot()
    assert state is not None
    assert (state.direction, state.timeframe, state.lifecycle) == (
        Direction.LONG, Timeframe.M5, DisplacementLifecycle.STARTED,
    )
    assert state.seed_candle_id == state.last_valid_candle_id == tracker._candle_id(seed)
    assert (
        state.started_at, state.state_started_at, state.prefix_last_admitted_at,
        state.last_updated_at, state.observed_at,
    ) == (seed.end,) * 5
    assert (state.active_at, state.terminal_at, state.real_episode_bar_count,
            state.age_minutes_at_last_admitted, state.net_ticks) == (None, None, 1, 0, 4)


@pytest.mark.parametrize(("atr", "seed", "continuations"), ACTIVATIONS,
                         ids=["bar-2", "bar-4"])
def test_exp013_activation_equal_boundary(atr, seed, continuations) -> None:
    tracker, index, _ = _started(atr=atr, seed=seed)
    states = [
        tracker.on_completed_5m(_candle(index + offset, values)).state
        for offset, values in enumerate(continuations)
    ]
    assert all(state is not None for state in states)
    assert all(state.lifecycle is DisplacementLifecycle.STARTED for state in states[:-1])
    state = states[-1]
    assert state.lifecycle is DisplacementLifecycle.ACTIVE
    assert state.real_episode_bar_count == len(continuations) + 1
    boundary = state.relative_atr if len(continuations) == 1 else state.speed_atr_per_bar
    assert boundary == pytest.approx(1.0 if len(continuations) == 1 else .30)


def test_exp013_activation_window_elapsed() -> None:
    tracker, index, _ = _started(atr=4.0, seed=(100.0, 103.25, 100.0, 103.25))
    values = (
        (103.25, 103.5, 103.25, 103.5),
        (103.5, 103.75, 103.5, 103.75),
        (103.75, 104.0, 103.75, 104.0),
    )
    updates = [tracker.on_completed_5m(_candle(index + i, value))
               for i, value in enumerate(values)]
    assert all(item.state is not None for item in updates[:-1])
    terminal = updates[-1].transitions[0].state
    assert updates[-1].state is None
    assert (terminal.lifecycle, terminal.terminal_reason, terminal.real_episode_bar_count) == (
        DisplacementLifecycle.EXHAUSTED, "activation_window_elapsed", 4,
    )


@pytest.mark.parametrize(("values", "survives", "reason"), CONTINUATIONS,
                         ids=["one-tick", "zero-progress"])
def test_exp013_continuation_progress(values, survives: bool, reason: str | None) -> None:
    tracker, index = _active()
    update = tracker.on_completed_5m(_candle(index, values))
    if survives:
        assert update.state is not None
        assert (update.state.lifecycle, update.state.real_episode_bar_count) == (
            DisplacementLifecycle.ACTIVE, 3,
        )
        assert update.transitions == ()
    else:
        assert update.state is None
        assert update.transitions[0].state.terminal_reason == reason


@pytest.mark.parametrize(("values", "reason"), TERMINALS,
                         ids=["doji", "opposite-body", "no-progress"])
def test_exp013_terminal_reason(values, reason: str) -> None:
    tracker, index = _active()
    update = tracker.on_completed_5m(_candle(index, values))
    terminal = update.transitions[0].state
    assert update.state is None
    assert len(update.transitions) == 1
    assert (terminal.lifecycle, terminal.terminal_reason) == (
        DisplacementLifecycle.EXHAUSTED, reason,
    )


def test_exp013_terminal_evidence_does_not_rewrite_geometry() -> None:
    tracker, index = _active()
    prior = tracker.snapshot()
    evidence = _candle(index, (101.5, 102.0, 101.5, 101.75))
    terminal = tracker.on_completed_5m(evidence).transitions[0].state
    fields = (
        "last_valid_candle_id", "prefix_last_admitted_at", "real_episode_bar_count",
        "age_minutes_at_last_admitted", "net_points", "relative_atr", "travel_points",
        "efficiency", "mean_body_fraction", "min_directional_clv", "favorable_extreme",
        "favorable_extreme_first_observed_at", "prefix_commitment",
    )
    assert prior is not None
    assert tuple(getattr(terminal, name) for name in fields) == tuple(
        getattr(prior, name) for name in fields
    )
    assert terminal.terminal_evidence_candle_id == tracker._candle_id(evidence)
    assert (terminal.terminal_at, terminal.observed_at) == (evidence.end, evidence.end)


def test_exp013_nested_seed_retains_entity() -> None:
    tracker, index, _ = _started()
    prior = tracker.snapshot()
    state = tracker.on_completed_5m(
        _candle(index, (101.0, 102.0, 101.0, 102.0))
    ).state
    assert prior is not None and state is not None
    assert (state.entity_id, state.seed_candle_id, state.nested_seed_observed) == (
        prior.entity_id, prior.seed_candle_id, True,
    )


def test_exp013_opposite_reseed_order_and_single_baseline_append() -> None:
    tracker, index, _ = _started()
    prior = tracker.snapshot()
    update = tracker.on_completed_5m(_candle(index, (101.0, 101.0, 99.0, 99.0)))
    exhausted, started = (item.state for item in update.transitions)
    assert prior is not None
    assert (exhausted.lifecycle, started.lifecycle) == (
        DisplacementLifecycle.EXHAUSTED, DisplacementLifecycle.STARTED,
    )
    assert (exhausted.entity_id, update.state) == (prior.entity_id, started)
    assert started.entity_id != prior.entity_id
    assert exhausted.observed_at == started.observed_at
    assert tuple(tracker._trs).count(2.0) == 1


@pytest.mark.parametrize(("mode", "reason"), CENSORSHIP,
                         ids=["synthetic", "boundary", "data-anomaly"])
def test_exp013_censorship(mode: str, reason: str) -> None:
    tracker, index, _ = _started()
    if mode == "synthetic":
        update = tracker.on_completed_5m(
            _candle(index, (101.0, 101.0, 101.0, 101.0), real=4, synthetic=1)
        )
    elif mode == "boundary":
        update = tracker.on_boundary(reason, _candle(index).end)
    else:
        update = tracker.on_completed_5m(_candle(index, (100.0, 101.0, 100.0, 100.1)))
    terminal = update.transitions[0].state
    assert (update.state, len(update.transitions), tracker.snapshot()) == (None, 1, None)
    assert (terminal.lifecycle, terminal.terminal_reason,
            terminal.terminal_evidence_candle_id) == (
        DisplacementLifecycle.CENSORED, reason, None,
    )
