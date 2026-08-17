from __future__ import annotations

from dataclasses import replace
import pickle

import pandas as pd
import pytest

import smc_trader.engine as engine_module
import smc_trader.playbooks as playbooks_module
from smc_trader.engine import (
    ContinuousSMCEngine,
    NEUTRAL_ENGINE_CHECKPOINT_SCHEMA_VERSION,
)
from smc_trader.model import (
    AccountState,
    EngineSnapshot,
    NeutralEngineSnapshot,
    Playbook,
    to_primitive,
)

from .helpers import session_bars


def _grid_bars(count: int):
    output = []
    for bar in session_bars(1)[:count]:
        open_price = round(bar.open / 0.25) * 0.25
        close = round(bar.close / 0.25) * 0.25
        output.append(
            replace(
                bar,
                open=open_price,
                high=max(open_price, close) + 0.5,
                low=min(open_price, close) - 0.5,
                close=close,
            )
        )
    return output


def _legacy_engine_step(
    engine: ContinuousSMCEngine,
    bar,
) -> EngineSnapshot:
    """The pre-neutral Engine ordering, retained only as a parity oracle."""

    account = AccountState(equity=100_000.0)
    update = engine.reader.on_bar(bar)
    observation = engine.observer.observe(update, None)
    if {
        "contract_change_history_reset",
        "data_gap_history_reset",
    }.intersection(observation.anomalies):
        engine.brain.reset()
    belief = engine.brain.update(
        observation,
        position=account.position,
        scene_graph=engine.observer.scene_graph,
        scene_delta=engine.observer.last_scene_delta,
    )
    decision = engine.decision.decide(observation, belief, account)
    risk = engine.risk.review(decision, observation, account)
    return EngineSnapshot(
        observation=observation,
        belief=belief,
        decision=decision,
        risk=risk,
    )


def test_engine_neutral_integration_preserves_old_outputs_each_clock() -> None:
    integrated = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    legacy = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    for bar in _grid_bars(100):
        actual = integrated.on_bar(bar)
        expected = _legacy_engine_step(legacy, bar)
        assert to_primitive(actual.observation) == to_primitive(expected.observation)
        assert to_primitive(actual.belief) == to_primitive(expected.belief)
        assert to_primitive(actual.belief.global_context) == to_primitive(
            expected.belief.global_context
        )
        assert to_primitive(actual.decision) == to_primitive(expected.decision)
        assert to_primitive(actual.risk) == to_primitive(expected.risk)
        assert actual.neutral_market_state is not None
        assert (
            actual.neutral_market_state.open_market_theses
            == actual.belief.global_context.open_market_theses
        )
        assert (
            actual.neutral_market_state.asof
            == actual.observation.asof
            == actual.belief.asof
        )


def test_engine_calls_global_context_reducer_once_per_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    real_reducer = engine_module.update_global_market_context
    calls = []

    def counted_reducer(*args, **kwargs):
        calls.append((args, kwargs))
        return real_reducer(*args, **kwargs)

    def forbidden_brain_fallback(*_args, **_kwargs):
        raise AssertionError("Brain recomputed the Engine raw global context")

    monkeypatch.setattr(
        engine_module,
        "update_global_market_context",
        counted_reducer,
    )
    monkeypatch.setattr(
        playbooks_module,
        "update_global_market_context",
        forbidden_brain_fallback,
    )
    first_bar, second_bar = _grid_bars(2)
    first = engine.on_bar(first_bar)
    assert first.neutral_market_state is not None
    assert len(calls) == 1
    assert calls[0][0][0] is None

    previous_neutral_context = first.neutral_market_state.global_context
    second = engine.on_bar(second_bar)
    assert second.neutral_market_state is not None
    assert len(calls) == 2
    assert calls[1][0][0] is previous_neutral_context


def test_action_policy_cannot_change_neutral_output_each_clock() -> None:
    action_enabled = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    action_disabled = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
        action_disabled_playbooks=tuple(Playbook),
    )
    for bar in _grid_bars(120):
        enabled = action_enabled.on_bar(bar)
        disabled = action_disabled.on_bar(bar)
        assert enabled.neutral_market_state is not None
        assert disabled.neutral_market_state is not None
        assert to_primitive(enabled.neutral_market_state) == to_primitive(
            disabled.neutral_market_state
        )


def test_same_epoch_brain_reset_or_replacement_cannot_reseed_neutral_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bars = _grid_bars(90)
    control = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    disturbed = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    for bar in bars[:60]:
        control.on_bar(bar)
        disturbed.on_bar(bar)
    prior_neutral = disturbed.neutral_market_state
    assert prior_neutral is not None

    real_reducer = engine_module.update_global_market_context
    reducer_priors = []

    def captured_reducer(previous, *args, **kwargs):
        reducer_priors.append(previous)
        return real_reducer(previous, *args, **kwargs)

    monkeypatch.setattr(
        engine_module,
        "update_global_market_context",
        captured_reducer,
    )

    disturbed.brain.reset()
    expected_after_reset = control.on_bar(bars[60])
    actual_after_reset = disturbed.on_bar(bars[60])
    assert actual_after_reset.neutral_market_state is not None
    assert reducer_priors[-1] is prior_neutral.global_context
    assert to_primitive(actual_after_reset.neutral_market_state) == to_primitive(
        expected_after_reset.neutral_market_state
    )
    assert actual_after_reset.neutral_market_state.market_epoch_id == (
        prior_neutral.market_epoch_id
    )

    brain_current = disturbed.brain.current
    assert brain_current is not None
    assert brain_current.global_context is not None
    poisoned_context = replace(
        brain_current.global_context,
        unknown_evidence=(
            *brain_current.global_context.unknown_evidence,
            "brain-only:poison",
        ),
    )
    disturbed.brain._belief = replace(
        brain_current,
        global_context=poisoned_context,
    )
    expected_after_replacement = control.on_bar(bars[61])
    actual_after_replacement = disturbed.on_bar(bars[61])
    assert actual_after_replacement.neutral_market_state is not None
    assert (
        reducer_priors[-1]
        is actual_after_reset.neutral_market_state.global_context
    )
    assert to_primitive(actual_after_replacement.neutral_market_state) == to_primitive(
        expected_after_replacement.neutral_market_state
    )
    neutral_unknown_evidence = (
        actual_after_replacement.neutral_market_state.global_context.unknown_evidence
    )
    assert "brain-only:poison" not in neutral_unknown_evidence


def test_engine_neutral_state_is_pickle_checkpoint_ready() -> None:
    bars = _grid_bars(45)
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    for bar in bars[:35]:
        engine.on_bar(bar)
    encoded = pickle.dumps(engine, protocol=pickle.HIGHEST_PROTOCOL)
    resumed = pickle.loads(encoded)
    assert NEUTRAL_ENGINE_CHECKPOINT_SCHEMA_VERSION == 2
    assert engine.__getstate__()["_neutral_checkpoint_schema_version"] == 2
    assert resumed.neutral_market_state == engine.neutral_market_state
    assert resumed.last_snapshot == engine.last_snapshot

    for bar in bars[35:]:
        expected = engine.on_bar(bar)
        actual = resumed.on_bar(bar)
        assert to_primitive(actual) == to_primitive(expected)


def test_neutral_only_entry_matches_normal_eye_and_neutral_projection() -> None:
    normal = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    neutral_only = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )

    for bar in _grid_bars(60):
        expected = normal.on_bar(bar)
        actual = neutral_only.on_bar_neutral_input(bar)
        assert isinstance(actual, NeutralEngineSnapshot)
        assert to_primitive(actual.observation) == to_primitive(
            expected.observation
        )
        assert to_primitive(actual.neutral_market_state) == to_primitive(
            expected.neutral_market_state
        )
        assert neutral_only.last_snapshot is actual
        assert neutral_only.neutral_market_state is actual.neutral_market_state


def test_neutral_only_entry_never_calls_brain_decision_or_risk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("neutral-only Engine invoked an action layer")

    monkeypatch.setattr(engine.brain, "reset", forbidden)
    monkeypatch.setattr(engine.brain, "update", forbidden)
    monkeypatch.setattr(engine.decision, "decide", forbidden)
    monkeypatch.setattr(engine.risk, "review", forbidden)

    first, second = _grid_bars(2)
    first_snapshot = engine.on_bar_neutral_input(first)
    reset_bar = replace(
        second,
        start=first.end + pd.Timedelta(minutes=6),
        data_gap_before_minutes=6,
    )
    reset_snapshot = engine.on_bar_neutral_input(reset_bar)

    assert first_snapshot.neutral_market_state is not None
    assert reset_snapshot.neutral_market_state is not None
    assert "data_gap_history_reset" in reset_snapshot.observation.anomalies
    assert engine.brain.current is None
    assert engine._last_belief_position is None


def test_neutral_only_checkpoint_continuation_matches_uninterrupted() -> None:
    bars = _grid_bars(70)
    control = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    checkpointed = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    for bar in bars[:45]:
        expected = control.on_bar_neutral_input(bar)
        actual = checkpointed.on_bar_neutral_input(bar)
        assert to_primitive(actual) == to_primitive(expected)

    state_fields = set(checkpointed.__dict__)
    resumed = pickle.loads(
        pickle.dumps(checkpointed, protocol=pickle.HIGHEST_PROTOCOL)
    )
    assert set(resumed.__dict__) == state_fields
    assert isinstance(resumed.last_snapshot, NeutralEngineSnapshot)
    assert resumed.last_snapshot == checkpointed.last_snapshot

    for bar in bars[45:]:
        expected = control.on_bar_neutral_input(bar)
        actual = resumed.on_bar_neutral_input(bar)
        assert to_primitive(actual) == to_primitive(expected)


def test_neutral_checkpoint_rejects_snapshot_type_or_state_mismatch() -> None:
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    first, second = _grid_bars(2)
    first_snapshot = engine.on_bar_neutral_input(first)
    engine.on_bar_neutral_input(second)
    state = engine.__getstate__()

    wrong_type = dict(state)
    wrong_type["_last_snapshot"] = object()
    with pytest.raises(ValueError, match="checkpoint neutral market state"):
        object.__new__(ContinuousSMCEngine).__setstate__(wrong_type)

    mismatched_state = dict(state)
    mismatched_state["_neutral_market_state"] = (
        first_snapshot.neutral_market_state
    )
    with pytest.raises(ValueError, match="checkpoint neutral market state"):
        object.__new__(ContinuousSMCEngine).__setstate__(mismatched_state)


def test_engine_entry_mode_cannot_change_after_first_committed_bar() -> None:
    first, second = _grid_bars(2)
    normal = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    normal.on_bar(first)
    with pytest.raises(RuntimeError, match="after full evaluation"):
        normal.on_bar_neutral_input(second)

    neutral_only = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    neutral_only.on_bar_neutral_input(first)
    with pytest.raises(RuntimeError, match="after neutral-only input"):
        neutral_only.on_bar(second)


def test_neutral_only_entry_requires_scene_graph_before_reader_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    engine.observer.config = replace(
        engine.observer.config,
        project_scene_graph=False,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("neutral-only preflight advanced the reader")

    monkeypatch.setattr(engine.reader, "on_bar", forbidden)
    with pytest.raises(RuntimeError, match="requires Scene Graph projection"):
        engine.on_bar_neutral_input(_grid_bars(1)[0])


def test_neutral_only_scene_compaction_preserves_continuation() -> None:
    bars = _grid_bars(90)
    baseline = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    compacted = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )

    for index, bar in enumerate(bars):
        expected = baseline.on_bar_neutral_input(bar)
        actual = compacted.on_bar_neutral_input(bar)
        assert to_primitive(actual) == to_primitive(expected)
        if index in {44, 69}:
            result = compacted.compact_scene_graph_runtime()
            assert result["after"]["nodes"] <= result["before"]["nodes"]


@pytest.mark.parametrize("legacy_version", (None, 1))
def test_old_engine_checkpoint_without_neutral_schema_fails_closed(
    legacy_version: int | None,
) -> None:
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    legacy_state = engine.__getstate__()
    if legacy_version is None:
        legacy_state.pop("_neutral_checkpoint_schema_version")
    else:
        legacy_state["_neutral_checkpoint_schema_version"] = legacy_version
    candidate = object.__new__(ContinuousSMCEngine)
    with pytest.raises(
        ValueError,
        match="checkpoint neutral market state schema changed",
    ):
        candidate.__setstate__(legacy_state)


def test_schema_one_engine_pickle_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    engine.on_bar_neutral_input(_grid_bars(1)[0])
    with monkeypatch.context() as legacy:
        legacy.setattr(
            engine_module,
            "NEUTRAL_ENGINE_CHECKPOINT_SCHEMA_VERSION",
            1,
        )
        encoded = pickle.dumps(engine, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(
        ValueError,
        match="checkpoint neutral market state schema changed",
    ):
        pickle.loads(encoded)


@pytest.mark.parametrize(
    ("reset_kind", "expected_anomaly"),
    (
        ("contract", "contract_change_history_reset"),
        ("data_gap", "data_gap_history_reset"),
    ),
)
def test_hard_reset_advances_epoch_and_clears_neutral_context(
    reset_kind: str,
    expected_anomaly: str,
) -> None:
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    bars = _grid_bars(101)
    first = None
    for bar in bars[:100]:
        first = engine.on_bar(bar)
    assert first is not None and first.neutral_market_state is not None
    prior_thesis_ids = {
        thesis.thesis_id
        for thesis in first.neutral_market_state.open_market_theses
    }
    assert prior_thesis_ids
    reset_bar = (
        replace(bars[100], symbol="NQM5", instrument_id=2)
        if reset_kind == "contract"
        else replace(
            bars[100],
            start=bars[99].end + pd.Timedelta(minutes=6),
            data_gap_before_minutes=6,
        )
    )
    reset = engine.on_bar(reset_bar)
    assert expected_anomaly in reset.observation.anomalies
    assert reset.neutral_market_state is not None
    assert reset.belief.global_context is not None
    assert (
        reset.neutral_market_state.market_epoch_id
        == reset.belief.global_context.market_epoch_id
    )
    assert (
        reset.neutral_market_state.market_epoch_id
        != first.neutral_market_state.market_epoch_id
    )
    assert prior_thesis_ids.isdisjoint(
        thesis.thesis_id
        for thesis in reset.neutral_market_state.open_market_theses
    )
    assert all(
        thesis.market_epoch_id == reset.neutral_market_state.market_epoch_id
        for thesis in reset.neutral_market_state.open_market_theses
    )
    assert all(
        episode.market_epoch_id == reset.neutral_market_state.market_epoch_id
        for episode in reset.neutral_market_state.market_episodes
    )
