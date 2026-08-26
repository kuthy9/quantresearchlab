from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import pickle

import pandas as pd
import pytest

import smc_trader.brain_entry_sequence as brain_entry_module
import smc_trader.engine as engine_module
import smc_trader.playbooks as playbooks_module
from smc_trader.engine import (
    ContinuousSMCEngine,
    NEUTRAL_ENGINE_CHECKPOINT_SCHEMA_VERSION,
)
from smc_trader.market_state import replay_atomic_market_snapshot
from smc_trader.model import (
    AccountState,
    Bar,
    EngineSnapshot,
    EventKind,
    NeutralEngineSnapshot,
    OpenMarketThesis,
    Playbook,
    Timeframe,
    to_primitive,
)
from smc_trader.observation import ExecutionRealityInput
from smc_trader.scene_graph import foundation_dol_inventory

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


def _foundation_bars(count: int):
    return [
        replace(
            bar,
            open=(price := 20_000.0 + 0.25 * index),
            high=price + 0.50,
            low=price - 0.25,
            close=price + 0.25,
        )
        for index, bar in enumerate(_grid_bars(count))
    ]


def _unsafe_clone(value, **changes):
    """Build malformed exact-type contracts for fail-closed boundary tests."""

    clone = object.__new__(type(value))
    clone.__dict__.update(value.__dict__)
    for name, item in changes.items():
        object.__setattr__(clone, name, item)
    return clone


def _june_2024_protected_role_and_pool_anchor_bars() -> list[Bar]:
    """The exact 31-clock production prefix that exposed both DOL joins."""

    rows = (
        (18590.25, 18598.50, 18576.00, 18595.25, 876.0),
        (18594.25, 18597.00, 18587.75, 18587.75, 350.0),
        (18588.50, 18588.50, 18562.25, 18564.50, 594.0),
        (18563.75, 18567.50, 18557.25, 18560.00, 573.0),
        (18560.50, 18564.50, 18553.75, 18553.75, 209.0),
        (18553.75, 18553.75, 18539.25, 18544.25, 494.0),
        (18542.00, 18547.75, 18539.50, 18546.75, 159.0),
        (18545.75, 18550.00, 18544.25, 18549.00, 154.0),
        (18548.00, 18548.25, 18545.75, 18548.25, 80.0),
        (18548.00, 18549.25, 18545.75, 18547.00, 90.0),
        (18547.25, 18552.00, 18544.50, 18550.50, 177.0),
        (18550.00, 18550.00, 18545.75, 18547.25, 96.0),
        (18547.50, 18554.00, 18546.25, 18551.25, 169.0),
        (18551.75, 18558.25, 18551.25, 18557.00, 103.0),
        (18556.50, 18559.75, 18554.00, 18558.75, 123.0),
        (18560.00, 18571.25, 18559.75, 18570.25, 285.0),
        (18570.00, 18573.50, 18568.25, 18571.50, 171.0),
        (18571.00, 18580.00, 18570.00, 18574.25, 230.0),
        (18574.25, 18576.25, 18572.50, 18575.00, 103.0),
        (18574.25, 18578.25, 18572.75, 18577.00, 68.0),
        (18578.00, 18578.50, 18574.00, 18575.50, 64.0),
        (18575.00, 18575.25, 18571.00, 18572.25, 113.0),
        (18572.75, 18574.00, 18572.00, 18573.00, 15.0),
        (18573.50, 18575.25, 18573.00, 18573.75, 37.0),
        (18572.75, 18576.25, 18572.00, 18576.00, 80.0),
        (18576.50, 18577.75, 18573.75, 18575.25, 75.0),
        (18575.75, 18584.25, 18575.75, 18583.75, 158.0),
        (18583.75, 18584.00, 18581.00, 18581.25, 102.0),
        (18581.25, 18586.00, 18571.50, 18572.25, 275.0),
        (18572.00, 18575.50, 18571.25, 18573.25, 55.0),
        (18572.50, 18575.75, 18572.50, 18574.00, 42.0),
    )
    start = pd.Timestamp("2024-06-02T18:00:00-04:00")
    return [
        Bar(
            start=start + pd.Timedelta(minutes=index),
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            symbol="NQM4",
            instrument_id=13743,
        )
        for index, (open_price, high, low, close, volume) in enumerate(rows)
    ]


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


def test_engine_neutral_integration_preserves_old_outputs_each_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    standalone_build_calls = 0
    standalone_builder = playbooks_module.build_open_market_theses

    def counted_standalone_builder(*args, **kwargs):
        nonlocal standalone_build_calls
        standalone_build_calls += 1
        return standalone_builder(*args, **kwargs)

    monkeypatch.setattr(
        playbooks_module,
        "build_open_market_theses",
        counted_standalone_builder,
    )
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
            actual.neutral_market_state.open_market_theses
            is actual.belief.global_context.open_market_theses
        )
        assert (
            actual.neutral_market_state.asof
            == actual.observation.asof
            == actual.belief.asof
        )
    assert standalone_build_calls == 100


def test_full_engine_builds_one_authoritative_thesis_tuple_per_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine_build_calls = 0
    engine_builder = engine_module.build_open_market_theses

    def counted_engine_builder(*args, **kwargs):
        nonlocal engine_build_calls
        engine_build_calls += 1
        return engine_builder(*args, **kwargs)

    def reject_brain_rebuild(*args, **kwargs):
        raise AssertionError("full Engine Brain rebuilt neutral theses")

    monkeypatch.setattr(
        engine_module,
        "build_open_market_theses",
        counted_engine_builder,
    )
    monkeypatch.setattr(
        playbooks_module,
        "build_open_market_theses",
        reject_brain_rebuild,
    )
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    bars = _grid_bars(12)
    for index, bar in enumerate(bars, start=1):
        snapshot = engine.on_bar(bar)
        assert snapshot.neutral_market_state is not None
        assert snapshot.belief.global_context is not None
        assert engine_build_calls == index
        assert (
            snapshot.neutral_market_state.open_market_theses
            is snapshot.belief.global_context.open_market_theses
        )


def test_engine_shares_one_brain_observation_view_with_neutral_and_brain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    real_adapter = engine_module.brain_observation_view
    real_interpreter = brain_entry_module.interpret_micro_break_facts
    real_neutral_builder = engine_module.build_neutral_market_state
    real_brain_update = engine.brain.update
    real_risk_review = engine.risk.review
    constructed = []
    neutral_inputs = []
    brain_inputs = []
    risk_inputs = []
    interpretation_calls = 0

    def counted_interpreter(facts):
        nonlocal interpretation_calls
        interpretation_calls += 1
        return real_interpreter(facts)

    def counted_adapter(observation):
        view = real_adapter(observation)
        constructed.append(view)
        return view

    def capture_neutral(previous, observation, context):
        neutral_inputs.append(observation)
        return real_neutral_builder(previous, observation, context)

    def capture_brain(observation, *args, **kwargs):
        brain_inputs.append(observation)
        return real_brain_update(observation, *args, **kwargs)

    def capture_risk(decision, observation, *args, **kwargs):
        risk_inputs.append(observation)
        return real_risk_review(decision, observation, *args, **kwargs)

    monkeypatch.setattr(
        engine_module,
        "brain_observation_view",
        counted_adapter,
    )
    monkeypatch.setattr(
        brain_entry_module,
        "interpret_micro_break_facts",
        counted_interpreter,
    )
    monkeypatch.setattr(
        engine_module,
        "build_neutral_market_state",
        capture_neutral,
    )
    monkeypatch.setattr(engine.brain, "update", capture_brain)
    monkeypatch.setattr(engine.risk, "review", capture_risk)

    engine.on_bar(_grid_bars(1)[0])

    assert len(constructed) == 1
    assert interpretation_calls == 1
    assert len(neutral_inputs) == len(brain_inputs) == 1
    assert neutral_inputs[0] is constructed[0]
    assert brain_inputs[0] is constructed[0]
    assert len(risk_inputs) == 1
    assert risk_inputs[0] is constructed[0]._observation


@pytest.mark.parametrize(
    "malformation",
    (
        "type",
        "clock",
        "revision",
        "epoch",
        "context_type",
        "root",
        "order",
    ),
)
def test_brain_rejects_malformed_precomputed_neutral_state(
    malformation: str,
) -> None:
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    observation = engine._observe_bar(_grid_bars(1)[0], execution=None)
    context, neutral = engine._project_neutral(observation)
    assert context is not None and neutral is not None
    if malformation == "type":
        malformed = object()
    elif malformation == "clock":
        malformed = _unsafe_clone(
            neutral,
            asof=neutral.asof + pd.Timedelta(minutes=1),
        )
    elif malformation == "revision":
        malformed = _unsafe_clone(
            neutral,
            scene_revision_id="scene:forged",
        )
    elif malformation == "epoch":
        malformed = _unsafe_clone(
            neutral,
            market_epoch_id="epoch:forged",
        )
    elif malformation == "context_type":
        malformed = _unsafe_clone(neutral, global_context=object())
    else:
        thesis_a = OpenMarketThesis(
            thesis_id="market-thesis:a",
            root_id="root:a",
            market_epoch_id=neutral.market_epoch_id,
            formed_at=neutral.asof,
            updated_at=neutral.asof,
            direction=None,
            source_timeframe=Timeframe.M1,
            structural_scale="internal",
            mechanism="test",
            authority_relation="unknown",
            mechanism_event_ids=("root:a",),
        )
        thesis_b = replace(
            thesis_a,
            thesis_id="market-thesis:b",
            root_id="root:b",
            mechanism_event_ids=("root:b",),
        )
        if malformation == "root":
            theses = (
                thesis_a,
                replace(
                    thesis_b,
                    root_id="root:a",
                    mechanism_event_ids=("root:a",),
                ),
            )
        else:
            theses = (thesis_b, thesis_a)
        malformed = _unsafe_clone(
            neutral,
            global_context=_unsafe_clone(
                context,
                open_market_theses=theses,
            ),
        )

    with pytest.raises(ValueError, match="precomputed neutral"):
        engine.brain.update(
            observation,
            scene_graph=engine.observer.scene_graph,
            scene_delta=engine.observer.last_scene_delta,
            _precomputed_neutral_state=malformed,  # type: ignore[arg-type]
            _neutral_authority_capability=(
                playbooks_module._NEUTRAL_AUTHORITY_CAPABILITY
            ),
        )


@pytest.mark.parametrize("case", ("missing", "orphan", "forged"))
def test_brain_neutral_capability_is_private_and_failure_atomic(
    case: str,
) -> None:
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    first, second = _grid_bars(2)
    engine.on_bar(first)
    observation = engine._observe_bar(second, execution=None)
    _, neutral = engine._project_neutral(observation)
    assert neutral is not None
    before = pickle.dumps(engine.brain, protocol=pickle.HIGHEST_PROTOCOL)
    before_manager = engine.brain.hypothesis_manager
    kwargs = {
        "scene_graph": engine.observer.scene_graph,
        "scene_delta": engine.observer.last_scene_delta,
    }
    if case == "missing":
        kwargs["_precomputed_neutral_state"] = neutral
    elif case == "orphan":
        kwargs["_neutral_authority_capability"] = (
            playbooks_module._NEUTRAL_AUTHORITY_CAPABILITY
        )
    else:
        kwargs["_precomputed_neutral_state"] = neutral
        kwargs["_neutral_authority_capability"] = object()

    with pytest.raises(ValueError, match="precomputed neutral"):
        engine.brain.update(observation, **kwargs)

    assert pickle.dumps(
        engine.brain,
        protocol=pickle.HIGHEST_PROTOCOL,
    ) == before
    assert engine.brain.hypothesis_manager is before_manager
    assert "_NEUTRAL_AUTHORITY_CAPABILITY" not in playbooks_module.__all__
    assert all(
        value is not playbooks_module._NEUTRAL_AUTHORITY_CAPABILITY
        for value in engine.brain.__dict__.values()
    )


def test_clock_only_close_cannot_reprice_scene_foundation_or_brain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    first_bar, next_bar = _grid_bars(2)
    first = engine.on_bar(first_bar)
    prior_price = first.observation.price
    raw_synthetic_price = prior_price - 100.0
    synthetic = replace(
        next_bar,
        open=raw_synthetic_price,
        high=raw_synthetic_price,
        low=raw_synthetic_price,
        close=raw_synthetic_price,
        volume=0.0,
        synthetic_no_trade=True,
    )
    brain_prices: list[float] = []
    original_update = engine.brain.update

    def capture_brain_price(observation, *args, **kwargs):
        brain_prices.append(float(observation.price))
        return original_update(observation, *args, **kwargs)

    monkeypatch.setattr(engine.brain, "update", capture_brain_price)
    clock_only = engine.on_bar(synthetic)

    assert synthetic.close == raw_synthetic_price != prior_price
    assert clock_only.observation.price == prior_price
    assert clock_only.market_snapshot.price == prior_price
    assert engine.observer.scene_graph._last_price == prior_price
    assert brain_prices == [prior_price]
    assert (
        clock_only.market_snapshot.foundation
        == first.market_snapshot.foundation
    )
    assert (
        clock_only.market_snapshot.foundation_range_locations
        == first.market_snapshot.foundation_range_locations
    )
    for timeframe, prior_state in first.market_snapshot.timeframe_states.items():
        assert (
            clock_only.market_snapshot.timeframe_states[timeframe].liquidity
            == prior_state.liquidity
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
    assert engine_module.MODEL_SCHEMA_VERSION == 4
    assert json.loads(Path("configs/model.json").read_text())["schema_version"] == 4
    bars = _foundation_bars(45)
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    for bar in bars[:35]:
        snapshot = engine.on_bar(
            bar,
            execution=ExecutionRealityInput(
                spread_points=0.25,
                deadline=bar.end + pd.Timedelta(minutes=90),
                source="checkpoint-test-feed",
            ),
        )
    assert snapshot.observation.execution.source == "checkpoint-test-feed"
    assert snapshot.observation.execution.spread_points == 0.25
    assert engine.last_snapshot is not None
    foundation = engine.last_snapshot.observation.market_snapshot.foundation
    assert foundation is None
    market_snapshot = engine.last_snapshot.observation.market_snapshot
    missing_market_schema = market_snapshot.__getstate__()
    missing_market_schema.pop("schema_version")
    with pytest.raises(ValueError, match="market snapshot pickle schema"):
        object.__new__(type(market_snapshot)).__setstate__(
            missing_market_schema
        )
    observation_state = snapshot.observation.__getstate__()
    assert observation_state["schema_version"] == 5
    previous_observation = dict(observation_state)
    previous_observation["schema_version"] = 4
    with pytest.raises(ValueError, match="MarketObservation pickle schema"):
        object.__new__(type(snapshot.observation)).__setstate__(
            previous_observation
        )
    snapshot_state = snapshot.__getstate__()
    assert snapshot_state["schema_version"] == 4
    previous_snapshot = dict(snapshot_state)
    previous_snapshot["schema_version"] = 3
    with pytest.raises(ValueError, match="EngineSnapshot pickle schema"):
        object.__new__(type(snapshot)).__setstate__(previous_snapshot)
    neutral_snapshot = NeutralEngineSnapshot(
        observation=snapshot.observation,
        neutral_market_state=snapshot.neutral_market_state,
    )
    neutral_snapshot_state = neutral_snapshot.__getstate__()
    assert neutral_snapshot_state["schema_version"] == 4
    previous_neutral_snapshot = dict(neutral_snapshot_state)
    previous_neutral_snapshot["schema_version"] = 3
    with pytest.raises(
        ValueError,
        match="NeutralEngineSnapshot pickle schema",
    ):
        object.__new__(NeutralEngineSnapshot).__setstate__(
            previous_neutral_snapshot
        )
    publisher_state = engine.observer.market_snapshot_publisher.__getstate__()
    assert publisher_state["_publisher_state_schema_version"] == 4
    previous_publisher = dict(publisher_state)
    previous_publisher["_publisher_state_schema_version"] = 3
    with pytest.raises(ValueError, match="publisher checkpoint schema"):
        object.__new__(type(engine.observer.market_snapshot_publisher)).__setstate__(
            previous_publisher
        )
    replayed = replay_atomic_market_snapshot(
        engine.observer.audit_store.events(),
        semantic_registry_identity=engine.observer.semantic_registry.identity,
        foundation_records=engine.observer.materialize_foundation_history(),
    )
    assert (
        replayed.replay_payload()
        == engine.last_snapshot.observation.market_snapshot.replay_payload()
    )
    encoded = pickle.dumps(engine, protocol=pickle.HIGHEST_PROTOCOL)
    resumed = pickle.loads(encoded)
    assert NEUTRAL_ENGINE_CHECKPOINT_SCHEMA_VERSION == 10
    assert engine.__getstate__()["_neutral_checkpoint_schema_version"] == 10
    previous_engine = engine.__getstate__()
    previous_engine["_neutral_checkpoint_schema_version"] = 9
    with pytest.raises(
        ValueError,
        match="checkpoint neutral market state schema",
    ):
        object.__new__(ContinuousSMCEngine).__setstate__(previous_engine)
    assert resumed.neutral_market_state == engine.neutral_market_state
    assert resumed.last_snapshot == engine.last_snapshot
    assert "market_snapshot" not in snapshot.__dict__
    assert snapshot.market_snapshot is snapshot.observation.market_snapshot
    assert {
        "asof",
        "symbol",
        "instrument_id",
        "price",
        "semantic_events_this_update",
    }.isdisjoint(snapshot.observation.__dict__)
    assert isinstance(resumed.last_snapshot, EngineSnapshot)
    assert resumed.last_snapshot.belief.global_context is not None
    assert (
        resumed.neutral_market_state.open_market_theses
        is resumed.last_snapshot.belief.global_context.open_market_theses
    )
    assert (
        resumed.last_snapshot.observation.market_snapshot.foundation
        == foundation
    )
    drifted = pickle.loads(encoded)
    drifted._foundation_registry_identity = "0" * 64
    with pytest.raises(ValueError, match="checkpoint neutral market state schema"):
        pickle.loads(pickle.dumps(drifted, protocol=pickle.HIGHEST_PROTOCOL))

    commitment_tamper = pickle.loads(encoded)
    committed_market = (
        commitment_tamper.last_snapshot.observation.market_snapshot
    )
    assert committed_market.event_count > 0
    object.__setattr__(
        committed_market,
        "event_count",
        committed_market.event_count - 1,
    )
    with pytest.raises(
        ValueError,
        match="checkpoint neutral market state schema",
    ):
        pickle.loads(
            pickle.dumps(
                commitment_tamper,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )

    equal_delta_tamper = pickle.loads(encoded)
    equal_delta_market = (
        equal_delta_tamper.last_snapshot.observation.market_snapshot
    )
    assert equal_delta_market.events_this_update
    equal_delta = copy.deepcopy(equal_delta_market.events_this_update[0])
    assert equal_delta == equal_delta_market.events_this_update[0]
    assert equal_delta is not equal_delta_market.events_this_update[0]
    object.__setattr__(
        equal_delta_market,
        "events_this_update",
        (equal_delta, *equal_delta_market.events_this_update[1:]),
    )
    with pytest.raises(
        ValueError,
        match="checkpoint neutral market state schema",
    ):
        pickle.loads(
            pickle.dumps(
                equal_delta_tamper,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        )

    exact_engine = pickle.loads(encoded)
    exact_engine_before = dict(exact_engine.__dict__)
    extra_engine_state = dict(exact_engine.__getstate__())
    extra_engine_state["legacy_history"] = ()
    with pytest.raises(ValueError, match="schema changed"):
        exact_engine.__setstate__(extra_engine_state)
    assert exact_engine.__dict__ == exact_engine_before
    exact_engine.__dict__["legacy_history"] = ()
    with pytest.raises(ValueError, match="Engine state is not exact"):
        pickle.dumps(exact_engine, protocol=pickle.HIGHEST_PROTOCOL)

    def invalid_field_state(value, name: str, replacement):
        state = dict(value.__getstate__())
        serialized = list(state["fields"])
        serialized = [
            (field_name, replacement if field_name == name else field_value)
            for field_name, field_value in serialized
        ]
        state["fields"] = tuple(serialized)
        return state

    observation = resumed.last_snapshot.observation
    observation_before = dict(observation.__dict__)
    with pytest.raises(TypeError, match="exact MarketSnapshot"):
        observation.__setstate__(
            invalid_field_state(observation, "market_snapshot", "forged")
        )
    assert observation.__dict__ == observation_before

    engine_snapshot = resumed.last_snapshot
    engine_snapshot_before = dict(engine_snapshot.__dict__)
    with pytest.raises(ValueError, match="observation and neutral state differ"):
        engine_snapshot.__setstate__(
            invalid_field_state(
                engine_snapshot,
                "neutral_market_state",
                "forged",
            )
        )
    assert engine_snapshot.__dict__ == engine_snapshot_before

    neutral_snapshot = NeutralEngineSnapshot(
        observation=observation,
        neutral_market_state=resumed.neutral_market_state,
    )
    neutral_snapshot_before = dict(neutral_snapshot.__dict__)
    with pytest.raises(ValueError, match="observation and state differ"):
        neutral_snapshot.__setstate__(
            invalid_field_state(
                neutral_snapshot,
                "neutral_market_state",
                "forged",
            )
        )
    assert neutral_snapshot.__dict__ == neutral_snapshot_before

    duplicate = pickle.loads(pickle.dumps(snapshot))
    duplicate.__dict__["market_snapshot"] = "forged-duplicate"
    with pytest.raises(ValueError, match="EngineSnapshot pickle state is not exact"):
        pickle.dumps(duplicate)

    duplicate_observation = pickle.loads(pickle.dumps(snapshot.observation))
    duplicate_observation.__dict__["asof"] = snapshot.observation.asof
    with pytest.raises(
        ValueError,
        match="MarketObservation pickle state is not exact",
    ):
        pickle.dumps(duplicate_observation)

    for bar in bars[35:]:
        reality = ExecutionRealityInput(
            spread_points=0.25,
            deadline=bar.end + pd.Timedelta(minutes=90),
            source="checkpoint-test-feed",
        )
        expected = engine.on_bar(bar, execution=reality)
        actual = resumed.on_bar(bar, execution=reality)
        assert to_primitive(actual) == to_primitive(expected)
        assert actual.neutral_market_state is not None
        assert actual.belief.global_context is not None
        assert (
            actual.neutral_market_state.open_market_theses
            is actual.belief.global_context.open_market_theses
        )

    assert resumed.last_snapshot is not None
    resumed_replay = replay_atomic_market_snapshot(
        resumed.observer.audit_store.events(),
        semantic_registry_identity=resumed.observer.semantic_registry.identity,
        foundation_records=resumed.observer.materialize_foundation_history(),
    )
    assert (
        resumed_replay.replay_payload()
        == resumed.last_snapshot.observation.market_snapshot.replay_payload()
    )


def test_2024_06_engine_foundation_dol_role_and_pool_anchor_replay_exact() -> None:
    bars = _june_2024_protected_role_and_pool_anchor_bars()
    engine = ContinuousSMCEngine.from_config(
        "configs/model.json",
        runtime_mode="development",
    )
    for bar in bars[:26]:
        engine.on_bar(bar)
    resumed = pickle.loads(pickle.dumps(engine, protocol=pickle.HIGHEST_PROTOCOL))

    final = None
    for ordinal, bar in enumerate(bars[26:], start=27):
        expected = engine.on_bar(bar)
        actual = resumed.on_bar(bar)
        assert to_primitive(actual) == to_primitive(expected)
        final = expected
        if ordinal != 28:
            continue

        observation = expected.observation
        tracker_protected = next(
            item
            for item in observation.liquidity_inventory
            if item.kind == "swing"
            and item.is_protected_swing
            and item.price == 18_572.0
        )
        foundation_view = next(
            item
            for item in foundation_dol_inventory(observation)
            if item.source_identity == tracker_protected.item_id
        )
        assert tracker_protected.structural_rank == "external"
        # The tracker role appears with the initial structure snapshot.  It
        # must not rewrite the canonical creation-time DOL rank or masquerade
        # as a protected assignment before that exact semantic event exists.
        assert foundation_view.structural_rank == "internal"
        assert foundation_view.is_protected_swing is False
        assert (
            engine.observer._foundation_dol_templates[
                tracker_protected.item_id
            ].rank
            == "internal"
        )
        assert not any(
            record.status.value == "active"
            and record.payload.get("protected_swing_id")
            == tracker_protected.item_id.removeprefix("swing:")
            for record in observation.market_snapshot.foundation.latest_records
            if record.object_type.value == "structure_generation"
        )

    assert final is not None
    observation = final.observation
    projection = observation.market_snapshot.foundation
    records = {record.object_id: record for record in projection.latest_records}
    pool_view = next(
        item
        for item in foundation_dol_inventory(observation)
        if records[item.item_id].payload.get("price_anchor_rule")
        == "near_side_tradable_zone_boundary_for_nontradable_midpoint"
    )
    pool_record = records[pool_view.item_id]
    anchor = (
        int(pool_record.payload["price_ticks"])
        * float(pool_record.payload["tick_size"])
    )
    published = next(
        candidate
        for state in observation.market_snapshot.timeframe_states.values()
        for candidate in state.liquidity.candidates
        if candidate.candidate_id == pool_view.item_id
    )
    assert pool_view.foundation_source_kind == "formed_liquidity_pool"
    assert pool_view.price == published.price == anchor

    replayed = replay_atomic_market_snapshot(
        engine.observer.audit_store.events(),
        semantic_registry_identity=engine.observer.semantic_registry.identity,
        foundation_records=engine.observer.materialize_foundation_history(),
    )
    assert (
        replayed.replay_payload()
        == observation.market_snapshot.replay_payload()
    )


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


@pytest.mark.parametrize("legacy_version", (None, 1, 2, 9))
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
    boundary_events = reset.observation.semantic_events_this_update
    epoch_reset = next(
        event
        for event in boundary_events
        if event.kind is EventKind.MARKET_EPOCH_RESET
    )
    displacement_terminals = tuple(
        event
        for event in boundary_events
        if event.kind is EventKind.DISPLACEMENT_OBSERVED
        and event.evidence.get("lifecycle") != "active"
    )
    assert displacement_terminals
    assert all(
        engine.observer.audit_store.get(event.event_id) == event
        for event in displacement_terminals
    )
    reset_order = (
        epoch_reset.known_at,
        epoch_reset.sequence_no,
        epoch_reset.event_id,
    )
    assert all(
        (event.known_at, event.sequence_no, event.event_id) < reset_order
        for event in displacement_terminals
    )
    assert reset.neutral_market_state is not None
    assert reset.belief.global_context is not None
    assert (
        reset.neutral_market_state.open_market_theses
        is reset.belief.global_context.open_market_theses
    )
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
