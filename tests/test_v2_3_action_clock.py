from __future__ import annotations

from dataclasses import replace
import math

import pandas as pd
import pyarrow.dataset as ds
import pytest

from smc_trader.action_clock import (
    FEATURE_NAMES,
    FLAT_ACTIONS,
    ActionClockProtocol,
    ActionClockReplay,
    PlanLineageStore,
    build_action_clock_engine,
    build_v2_3_action_clock_engine,
    candidate_action_row,
    executable_candidate_groups,
)
from smc_trader.action_equivalence import ActionEquivalenceProtocol
from smc_trader.calibration import CalibrationError
from smc_trader.action_value import (
    ActionClockValueDecisionLayer,
    FrozenActionValueArtifact,
    FrozenLinearModel,
)
from smc_trader.artifact_stream import (
    new_stream_state,
    verify_stream_shards,
    write_stream_shards_bounded,
)
from smc_trader.model import (
    Action,
    ActionUtility,
    Bar,
    Decision,
    EngineSnapshot,
    HypothesisSequenceState,
    LiquidityLevel,
    MarketBelief,
    PlaybookPhase,
    RiskAssessment,
    SequenceStepState,
    Timeframe,
)
from smc_trader.shadow_replay import FrozenShadowReplay
from smc_trader.observation import ExecutionRealityInput

from .helpers import engine_snapshot, executable_belief, market_observation


def _candidate_snapshot():
    base = engine_snapshot()
    hypothesis = next(iter(base.belief.hypotheses.values()))
    sequence_clock = base.observation.asof - pd.Timedelta(minutes=5)
    sequence = HypothesisSequenceState(
        protocol_version="test-v2.3",
        protocol_hash="b" * 64,
        setup_id="setup-1",
        steps=(
            SequenceStepState(
                "setup",
                True,
                1.0,
                sequence_clock,
            ),
        ),
        started_at=sequence_clock,
    )
    hypothesis = replace(
        hypothesis,
        phase=PlaybookPhase.EXECUTABLE,
        sequence=sequence,
    )
    belief = MarketBelief(
        base.observation.asof,
        {hypothesis.key: hypothesis},
    )
    abstain = ActionUtility(
        Action.ABSTAIN,
        0.0,
        {"capital_preservation": 0.0},
        None,
        "test intentionally suppresses the behavior-policy entry",
    )
    decision = Decision(
        base.observation.asof,
        Action.ABSTAIN,
        (abstain,),
        None,
        0.0,
        ("suppressed behavior action",),
        None,
    )
    snapshot = replace(
        base,
        belief=belief,
        decision=decision,
        risk=RiskAssessment(
            Action.ABSTAIN,
            Action.ABSTAIN,
            True,
            (),
            ("flat",),
        ),
    )
    engine = build_action_clock_engine(
        "configs/model_v2_1_belief_identity.json"
    )
    equivalence = ActionEquivalenceProtocol.from_file(
        "configs/action_equivalence_v2_2.json"
    )
    lineage = PlanLineageStore()
    candidates = executable_candidate_groups(
        snapshot,
        equivalence,
        lineage,
        engine=engine,
    )
    assert len(candidates) == 1
    return candidates[0], snapshot


def _later_snapshot(
    candidate,
    *,
    asof: pd.Timestamp,
    price: float,
    protection: LiquidityLevel | None = None,
) -> EngineSnapshot:
    observation = market_observation(asof=asof, price=price)
    if protection is not None:
        frame = observation.frame(Timeframe.M1)
        frames = dict(observation.frames)
        frames[Timeframe.M1] = replace(
            frame,
            liquidity=(*frame.liquidity, protection),
        )
        observation = replace(observation, frames=frames)
    hypothesis = replace(candidate.representative)
    belief = MarketBelief(asof, {hypothesis.key: hypothesis})
    abstain = ActionUtility(
        Action.ABSTAIN,
        0.0,
        {"capital_preservation": 0.0},
        None,
        "shadow state only",
    )
    decision = Decision(
        asof,
        Action.ABSTAIN,
        (abstain,),
        None,
        0.0,
        ("shadow state only",),
        None,
    )
    risk = RiskAssessment(
        Action.ABSTAIN,
        Action.ABSTAIN,
        True,
        (),
        ("shadow state only",),
    )
    return EngineSnapshot(
        observation,
        belief,
        decision,
        risk,
        "c" * 64,
    )


def _constant_model(
    names: tuple[str, ...],
    *,
    kind: str,
    value: float,
    residual_q10: float | None = None,
) -> FrozenLinearModel:
    intercept = (
        math.log(value / (1.0 - value))
        if kind == "logistic"
        else value
    )
    return FrozenLinearModel(
        kind=kind,
        feature_names=names,
        means=(0.0,) * len(names),
        scales=(1.0,) * len(names),
        coefficients=(0.0,) * len(names),
        intercept=intercept,
        samples=100,
        residual_q10=residual_q10,
    )


def test_v2_3_protocol_and_all_candidate_actions_are_frozen() -> None:
    protocol = ActionClockProtocol.from_file()
    candidate, snapshot = _candidate_snapshot()
    rows = [
        candidate_action_row(candidate, snapshot, action_id=action)
        for action in FLAT_ACTIONS
    ]
    assert protocol.status == "preregistered_before_v2_3_sample_generation"
    assert {row["action_id"] for row in rows} == set(FLAT_ACTIONS)
    assert len({row["action_key"] for row in rows}) == len(FLAT_ACTIONS)
    assert all(tuple(row[name] for name in FEATURE_NAMES) for row in rows)
    # The behavior policy abstained and the current cost vetoed entry, yet the
    # structurally valid candidate and every alternative are still retained.
    assert snapshot.decision.selected_action is Action.ABSTAIN
    assert not candidate.risk.passed
    assert all(row["structural_plan_valid"] for row in rows)


def test_stale_v2_3_belief_artifact_remains_fail_closed() -> None:
    with pytest.raises(CalibrationError, match="model-code hash is stale"):
        build_v2_3_action_clock_engine()


def test_prevalid_calibration_warmup_does_not_seed_recursive_belief() -> None:
    replay = ActionClockReplay(
        build_action_clock_engine(
            "configs/model_v2_1_belief_identity.json"
        )
    )
    start = pd.Timestamp("2022-12-30 10:00", tz="America/New_York")
    execution = ExecutionRealityInput(
        spread_points=None,
        expected_slippage_points=0.0,
        commission_per_contract_per_side=0.0,
        deadline=start + pd.Timedelta(hours=6),
        data_age_seconds=61.0,
        source="missing_execution_authority",
    )
    warmup = Bar(
        start,
        100.0,
        100.5,
        99.5,
        100.25,
        10.0,
        "NQH5",
        1,
    )
    snapshot = replay.on_bar(
        warmup,
        execution=execution,
        belief_enabled=False,
    )
    assert snapshot.belief.hypotheses == {}
    assert replay.engine.brain.current is None


def test_decomposed_cost_uses_current_mbo_only_for_orders_beginning_now() -> None:
    candidate, snapshot = _candidate_snapshot()
    features = candidate_action_row(
        candidate,
        snapshot,
        action_id="enter_now",
    )
    feature_values = {name: float(features[name]) for name in FEATURE_NAMES}
    feature_values["mbo_available"] = 1.0
    feature_values["expected_round_trip_cost_R"] = 0.40
    flat_models = {
        action: {
            "fill": _constant_model(
                FEATURE_NAMES,
                kind="logistic",
                value=0.8,
            ),
            "conditional_gross": _constant_model(
                FEATURE_NAMES,
                kind="ridge",
                value=1.0,
                residual_q10=-0.5,
            ),
            "conditional_cost": _constant_model(
                FEATURE_NAMES,
                kind="ridge",
                value=0.10,
            ),
            "conditional_loss": _constant_model(
                FEATURE_NAMES,
                kind="logistic",
                value=0.2,
            ),
        }
        for action in FLAT_ACTIONS
        if action != "abstain"
    }
    delta = {
        action: _constant_model(
            FEATURE_NAMES,
            kind="ridge",
            value=0.2,
        )
        for action in FLAT_ACTIONS
        if action != "enter_now"
    }
    artifact = FrozenActionValueArtifact(
        version="test",
        fingerprint="d" * 64,
        protocol_hash="e" * 64,
        runtime_semantics_hash="f" * 64,
        status="ready",
        flat_action_models=flat_models,
        gross_delta_models=delta,
        net_delta_models=delta,
        position_models={},
        release_gates={"test": True},
    )
    immediate = artifact.estimate(feature_values, action_id="enter_now")
    delayed = artifact.estimate(feature_values, action_id="wait_one_bar")
    assert immediate.conditional_cost_R == 0.40
    assert immediate.cost_source == "current_causal_mbo"
    assert delayed.conditional_cost_R == 0.10
    assert delayed.cost_source == "frozen_future_submission_cost_model"
    assert delayed.expiry_probability == pytest.approx(0.2)
    assert delayed.expiry_utility_R == 0.0


def test_runtime_compares_one_best_wait_instruction_per_candidate() -> None:
    _, snapshot = _candidate_snapshot()
    gross_by_action = {
        "enter_now": 1.0,
        "wait_one_bar": 0.7,
        "wait_better_price": 0.9,
        "wait_reacceptance": 0.8,
    }
    flat_models = {
        action: {
            "fill": _constant_model(
                FEATURE_NAMES,
                kind="logistic",
                value=0.8,
            ),
            "conditional_gross": _constant_model(
                FEATURE_NAMES,
                kind="ridge",
                value=gross,
                residual_q10=-0.5,
            ),
            "conditional_cost": _constant_model(
                FEATURE_NAMES,
                kind="ridge",
                value=0.1,
            ),
            "conditional_loss": _constant_model(
                FEATURE_NAMES,
                kind="logistic",
                value=0.2,
            ),
        }
        for action, gross in gross_by_action.items()
    }
    deltas = {
        action: _constant_model(
            FEATURE_NAMES,
            kind="ridge",
            value=0.2,
        )
        for action in FLAT_ACTIONS
        if action != "enter_now"
    }
    artifact = FrozenActionValueArtifact(
        version="test",
        fingerprint="d" * 64,
        protocol_hash="e" * 64,
        runtime_semantics_hash="f" * 64,
        status="ready",
        flat_action_models=flat_models,
        gross_delta_models=deltas,
        net_delta_models=deltas,
        position_models={},
        release_gates={"test": True},
    )
    layer = ActionClockValueDecisionLayer(
        ActionEquivalenceProtocol.from_file(
            "configs/action_equivalence_v2_2.json"
        ),
        artifact,
    )
    utilities = layer._flat_utilities(
        snapshot.observation,
        snapshot.belief,
    )
    waits = [item for item in utilities if item.action is Action.WAIT]
    assert len(waits) == 1
    assert "wait_better_price" in waits[0].reason
    assert {
        "wait_one_bar_Q_R",
        "wait_better_price_Q_R",
        "wait_reacceptance_Q_R",
    }.issubset(waits[0].components)


def test_shadow_replay_conservatively_charges_ambiguous_entry_bar_stop() -> None:
    candidate, snapshot = _candidate_snapshot()
    shadow = FrozenShadowReplay()
    shadow.register(candidate, snapshot)
    bar = Bar(
        snapshot.observation.asof,
        100.0,
        103.5,
        97.5,
        100.5,
        10.0,
        "NQH5",
        1,
    )
    shadow.advance(
        bar,
        _later_snapshot(candidate, asof=bar.end, price=bar.close),
    )
    flat, _, _ = shadow.drain()
    enter = next(row for row in flat if row["action_id"] == "enter_now")
    assert enter["outcome"] == "same_bar_ambiguous_stop_first"
    assert enter["gross_R"] == -1.0
    assert enter["ambiguous_same_bar"]


def test_protected_position_state_is_sampled_only_while_causally_alive() -> None:
    candidate, snapshot = _candidate_snapshot()
    shadow = FrozenShadowReplay()
    shadow.register(candidate, snapshot)
    protection = LiquidityLevel(
        "post-entry-protection",
        Timeframe.M1,
        "below",
        100.5,
        snapshot.observation.asof,
        snapshot.observation.asof + pd.Timedelta(seconds=30),
        0,
    )
    entry_bar = Bar(
        snapshot.observation.asof,
        100.0,
        101.25,
        99.75,
        101.0,
        10.0,
        "NQH5",
        1,
    )
    shadow.advance(
        entry_bar,
        _later_snapshot(
            candidate,
            asof=entry_bar.end,
            price=entry_bar.close,
            protection=protection,
        ),
    )
    parent = next(
        action
        for action in shadow.active.values()
        if action.action_id == "enter_now"
    )
    assert len(parent.position_states) == 2
    _, samples, _ = shadow.drain()
    assert {row["action_id"] for row in samples} == {
        "hold",
        "protect",
        "exit",
    }

    stop_bar = Bar(
        entry_bar.end,
        101.0,
        101.25,
        100.25,
        100.75,
        10.0,
        "NQH5",
        1,
    )
    shadow.advance(
        stop_bar,
        _later_snapshot(
            candidate,
            asof=stop_bar.end,
            price=stop_bar.close,
            protection=protection,
        ),
    )
    parent = next(
        action
        for action in shadow.active.values()
        if action.action_id == "enter_now"
    )
    assert len(parent.position_states) == 1


def test_bounded_shard_writer_never_exceeds_registered_cap(tmp_path) -> None:
    state = new_stream_state()
    buffer = [{"key": index, "value": index * 2} for index in range(11)]
    write_stream_shards_bounded(
        tmp_path,
        "bounded",
        buffer,
        state,
        key_column="key",
        maximum_rows=4,
    )
    assert buffer == []
    assert [item["rows"] for item in state["committed_shards"]] == [4, 4, 3]
    assert verify_stream_shards(tmp_path, state) == 11


def test_registered_shard_schema_is_stable_when_a_chunk_is_all_null(
    tmp_path,
) -> None:
    state = new_stream_state()
    field_types = {
        "key": "large_string",
        "observed_at": "timestamp_ny",
        "value": "float64",
    }
    first = [
        {"key": "a", "observed_at": None, "value": None},
        {"key": "b", "observed_at": None, "value": None},
    ]
    second = [
        {
            "key": "c",
            "observed_at": pd.Timestamp(
                "2025-01-02 10:00",
                tz="America/New_York",
            ),
            "value": 1.5,
        }
    ]
    for rows in (first, second):
        write_stream_shards_bounded(
            tmp_path,
            "stable",
            rows,
            state,
            key_column="key",
            maximum_rows=2,
            field_types=field_types,
        )
    assert verify_stream_shards(tmp_path, state) == 3
    table = ds.dataset(tmp_path / "stable", format="parquet").to_table()
    assert table.num_rows == 3
    assert table.schema.field("observed_at").type.tz == "America/New_York"
    assert table.schema.field("value").type.bit_width == 64
