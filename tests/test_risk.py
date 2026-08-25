from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from smc_trader.model import (
    Action,
    ActionUtility,
    Bar,
    Decision,
    EntryLocationLifecycle,
    EntryLocationState,
    EventKind,
    InteractionUpdate,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    LiquidityLevel,
    LiquidityPoolLifecycle,
    LiquidityPoolState,
    MarketObservation,
    PathSequenceLifecycle,
    PathSequenceState,
    PathSequenceStep,
    PositionSnapshot,
    StructuralLevel,
    Timeframe,
    TradePlan,
    VetoCode,
)
from smc_trader.risk import (
    RiskLimits,
    StructuralRiskEngine,
    _valid_stop,
    _valid_targets,
    conservative_entry_bar,
    conservative_position_bar,
)

from .helpers import flat_account, long_plan, market_observation


def _physical_entry_fixture(
    *, observation: MarketObservation | None = None
) -> tuple[MarketObservation, TradePlan]:
    observation = observation or market_observation()
    unbound = long_plan(observation)
    location_id = "test-entry-location"
    path_id = "test-entry-path"
    zone_id = "test-entry-zone"
    protocol_hash = "test-interaction-physical-v1"
    formed_at = unbound.invalidation.observed_at
    lower_bound = unbound.invalidation.price
    upper_bound = observation.price
    age = int((observation.asof - formed_at) / pd.Timedelta(minutes=1))
    location = EntryLocationState(
        location_id=location_id,
        protocol_hash=protocol_hash,
        source_zone_detector_protocol_hash="test-zone-physical-v1",
        symbol=observation.symbol,
        instrument_id=observation.instrument_id,
        direction=unbound.direction,
        source_zone_kind="fvg",
        source_zone_id=zone_id,
        source_zone_protocol_hash="test-zone-physical-v1",
        source_displacement_id="test-entry-displacement",
        source_bos_id=None,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        midpoint=(lower_bound + upper_bound) / 2.0,
        near_edge=upper_bound,
        far_edge=lower_bound,
        failure_boundary=lower_bound,
        formed_at=formed_at,
        lifecycle=EntryLocationLifecycle.APPROACHING,
        state_started_at=formed_at,
        last_updated_at=observation.asof,
        age_real_1m_bars=age,
        state_duration_real_1m_bars=age,
        current_price=observation.price,
        distance_to_zone_points=0.0,
        distance_to_failure_points=observation.price - lower_bound,
        transition_reason="source_registered",
    )
    visible_step = PathSequenceStep(
        step_id="test-entry-step:zone-visible",
        kind="zone_visible",
        observed_at=formed_at,
        source_event_id=zone_id,
        source_entity_id=zone_id,
        predecessor_step_ids=(),
        same_clock_relation="origin",
        direction=unbound.direction,
        strength=1.0,
        reason="typed_entry_zone_registered",
    )
    path = PathSequenceState(
        sequence_id=path_id,
        protocol_hash=protocol_hash,
        symbol=observation.symbol,
        instrument_id=observation.instrument_id,
        context_kind="zone_return",
        context_id=location_id,
        direction=unbound.direction,
        lifecycle=PathSequenceLifecycle.ACTIVE,
        formed_at=formed_at,
        state_started_at=formed_at,
        last_updated_at=observation.asof,
        age_real_1m_bars=age,
        state_duration_real_1m_bars=age,
        steps=(visible_step,),
        transition_reason="context_registered",
    )
    interaction = InteractionUpdate(
        zone_interactions=(location,),
        reacceptance_interactions=(),
        micro_break_facts=(),
        interaction_paths=(path,),
    )
    observation = replace(observation, interaction_update=interaction)
    target = replace(
        unbound.targets[0],
        level_id="h4-above-level",
        timeframe=Timeframe.H4,
    )
    plan = replace(
        unbound,
        invalidation=replace(
            unbound.invalidation,
            source_level_id=location_id,
        ),
        targets=(target,),
        setup_id=path_id,
        entry_location_id=location_id,
        entry_path_id=path_id,
        entry_zone_lower=lower_bound,
        entry_zone_upper=upper_bound,
        selected_draw_id=target.level_id,
    )
    return observation, plan


def _decision(observation, plan) -> Decision:
    utility = ActionUtility(Action.ENTER, 1.0, {}, "key", "reason")
    return Decision(
        observation.asof,
        Action.ENTER,
        (utility,),
        "key",
        1.0,
        ("reason",),
        plan,
    )


def test_entry_freezes_original_structural_thesis() -> None:
    observation, plan = _physical_entry_fixture()
    result = StructuralRiskEngine(RiskLimits(maximum_cost_R=0.50)).review(
        _decision(observation, plan), observation, flat_account()
    )
    assert result.passed
    assert result.frozen_thesis is not None
    assert result.frozen_thesis.original_invalidation == plan.invalidation
    with pytest.raises(FrozenInstanceError):
        result.frozen_thesis.original_invalidation = replace(
            plan.invalidation, price=99.0
        )


def test_entry_without_physical_custody_fails_closed() -> None:
    observation = market_observation()
    plan = long_plan(observation)
    result = StructuralRiskEngine(
        RiskLimits(maximum_cost_R=0.50)
    ).review(
        _decision(observation, plan),
        observation,
        flat_account(),
    )
    assert not result.passed
    assert result.final_action is Action.ABSTAIN
    assert result.vetoes == (VetoCode.NO_PLAN,)


def test_invisible_target_and_missing_deadline_have_hard_veto() -> None:
    observation = market_observation(deadline_missing=True)
    plan = long_plan(observation)
    invisible = replace(
        plan,
        targets=(replace(plan.targets[0], level_id="not-visible"),),
    )
    result = StructuralRiskEngine().review(
        _decision(observation, invisible), observation, flat_account()
    )
    assert not result.passed
    assert VetoCode.INVALID_TARGET in result.vetoes
    assert VetoCode.DEADLINE in result.vetoes
    assert result.final_action is Action.ABSTAIN


def test_frozen_plan_deadline_must_outlive_the_next_completed_bar() -> None:
    observation = market_observation()
    plan = replace(
        long_plan(observation),
        deadline=observation.asof + pd.Timedelta(minutes=1),
    )
    result = StructuralRiskEngine(
        RiskLimits(maximum_cost_R=0.50)
    ).review(
        _decision(observation, plan),
        observation,
        flat_account(),
    )
    assert not result.passed
    assert result.final_action is Action.ABSTAIN
    assert VetoCode.DEADLINE in result.vetoes


@pytest.mark.parametrize("anomaly", ("data_anomaly", "tick_size_mismatch"))
def test_observation_integrity_anomaly_hard_vetoes_flat_entry(
    anomaly: str,
) -> None:
    observation, plan = _physical_entry_fixture(
        observation=replace(
            market_observation(),
            anomalies=(anomaly,),
        )
    )
    result = StructuralRiskEngine(
        RiskLimits(maximum_cost_R=0.50)
    ).review(
        _decision(observation, plan),
        observation,
        flat_account(),
    )
    assert not result.passed
    assert result.final_action is Action.ABSTAIN
    assert VetoCode.DATA_ANOMALY in result.vetoes


@pytest.mark.parametrize(
    ("source", "execution_anomalies"),
    (
        ("missing", ()),
        ("unknown", ()),
        ("constant", ()),
        (
            "synthetic_observed_execution",
            ("spread_missing_used_one_tick",),
        ),
    ),
)
def test_non_observed_execution_or_missing_spread_hard_vetoes_entry(
    source: str,
    execution_anomalies: tuple[str, ...],
) -> None:
    observation = market_observation()
    observation = replace(
        observation,
        execution=replace(
            observation.execution,
            source=source,
            anomalies=execution_anomalies,
        ),
    )
    observation, plan = _physical_entry_fixture(observation=observation)

    result = StructuralRiskEngine(
        RiskLimits(maximum_cost_R=0.50)
    ).review(
        _decision(observation, plan),
        observation,
        flat_account(),
    )

    assert not result.passed
    assert result.final_action is Action.ABSTAIN
    assert result.vetoes == (VetoCode.DATA_ANOMALY,)


@pytest.mark.parametrize("anomaly", ("data_anomaly", "tick_size_mismatch"))
def test_observation_integrity_anomaly_forces_open_position_exit(
    anomaly: str,
) -> None:
    observation = replace(
        market_observation(),
        anomalies=(anomaly,),
    )
    plan = long_plan(observation)
    position = PositionSnapshot(
        thesis_hash="a" * 64,
        symbol=observation.symbol,
        instrument_id=observation.instrument_id,
        playbook=plan.playbook,
        direction=plan.direction,
        entry_price=plan.planned_entry,
        original_invalidation=plan.invalidation,
        current_stop=plan.invalidation.price,
        primary_target=plan.targets[0],
        opened_at=observation.asof - pd.Timedelta(minutes=1),
        deadline=plan.deadline,
        quantity=1,
        unrealized_R=0.0,
        elapsed_minutes=1,
    )
    hold = Decision(
        observation.asof,
        Action.HOLD,
        (ActionUtility(Action.HOLD, 0.5, {}, None, "hold"),),
        None,
        0.5,
        ("hold",),
        None,
    )
    result = StructuralRiskEngine().review(
        hold,
        observation,
        replace(flat_account(), position=position),
    )
    assert not result.passed
    assert result.requested_action is Action.HOLD
    assert result.final_action is Action.EXIT
    assert VetoCode.DATA_ANOMALY in result.vetoes


def test_open_position_uses_frozen_deadline_and_rejects_earlier_execution_cutoff() -> None:
    observation = market_observation()
    plan = long_plan(observation)
    position = PositionSnapshot(
        thesis_hash="a" * 64,
        symbol=observation.symbol,
        instrument_id=observation.instrument_id,
        playbook=plan.playbook,
        direction=plan.direction,
        entry_price=plan.planned_entry,
        original_invalidation=plan.invalidation,
        current_stop=plan.invalidation.price,
        primary_target=plan.targets[0],
        opened_at=observation.asof - pd.Timedelta(minutes=1),
        deadline=plan.deadline,
        quantity=1,
        unrealized_R=0.0,
        elapsed_minutes=1,
    )
    drifted = replace(
        observation,
        execution=replace(
            observation.execution,
            minutes_to_deadline=59,
        ),
    )
    hold = Decision(
        observation.asof,
        Action.HOLD,
        (ActionUtility(Action.HOLD, 0.5, {}, None, "hold"),),
        None,
        0.5,
        ("hold",),
        None,
    )
    result = StructuralRiskEngine().review(
        hold,
        drifted,
        replace(flat_account(), position=position),
    )
    assert not result.passed
    assert result.final_action is Action.EXIT
    assert VetoCode.DEADLINE in result.vetoes
    assert VetoCode.DATA_ANOMALY in result.vetoes


def test_stop_and_target_ids_are_bound_to_their_canonical_values() -> None:
    observation = market_observation()
    plan = long_plan(observation)
    assert not _valid_stop(
        replace(
            plan,
            invalidation=replace(
                plan.invalidation,
                price=plan.invalidation.price - 0.25,
            ),
        ),
        observation,
    )
    assert not _valid_targets(
        replace(
            plan,
            targets=(
                replace(
                    plan.targets[0],
                    price=plan.targets[0].price + 0.25,
                ),
            ),
        ),
        observation,
    )


def test_consumption_event_cannot_source_an_invalidation() -> None:
    observation = market_observation()
    event = replace(
        observation.recent_events[0],
        event_id="consumed-liquidity-event",
        kind=EventKind.LIQUIDITY_CONSUMED,
    )
    observation = replace(observation, recent_events=(event,))
    plan = long_plan(observation)
    plan = replace(
        plan,
        invalidation=StructuralLevel(
            price=float(event.price),
            side="below",
            source_level_id=event.event_id,
            observed_at=event.observed_at,
            rationale="consumption event must not register as a stop",
        ),
    )
    assert not _valid_stop(plan, observation)


def test_actual_contract_risk_has_hard_veto_even_if_requested_fraction_is_small() -> None:
    observation = market_observation()
    plan = long_plan(observation)
    account = replace(
        flat_account(),
        equity=1_000.0,
        quantity=10,
        requested_risk_fraction=0.001,
    )
    result = StructuralRiskEngine().review(
        _decision(observation, plan), observation, account
    )
    assert VetoCode.ACCOUNT_RISK in result.vetoes
    assert result.final_action is Action.ABSTAIN


def test_same_bar_stop_and_target_resolve_to_stop() -> None:
    bar = Bar(
        start=market_observation().asof,
        open=100.0,
        high=104.0,
        low=97.0,
        close=101.0,
        volume=100,
        symbol="NQH5",
        instrument_id=1,
    )
    result = conservative_position_bar(
        long_plan().direction,
        bar,
        current_stop=98.0,
        target=103.0,
    )
    assert result.closed and result.stop_touched and result.target_touched
    assert result.reason == "same_bar_ambiguous_stop_first"
    assert result.exit_price == 98.0


def test_entry_bar_never_credits_favorable_target_ordering() -> None:
    plan = long_plan()
    bar = Bar(
        start=market_observation().asof,
        open=101.0,
        high=104.0,
        low=99.5,
        close=103.5,
        volume=100,
        symbol="NQH5",
        instrument_id=1,
    )
    result = conservative_entry_bar(plan, bar)
    assert result.filled
    assert not result.closed
    assert result.target_touched


def test_protect_uses_structure_confirmed_on_the_current_action_clock() -> None:
    observation = market_observation()
    plan = long_plan(observation)
    opened_at = observation.asof - pd.Timedelta(minutes=1)
    current_m1 = LiquidityLevel(
        "current-m1-protection",
        Timeframe.M1,
        "below",
        99.5,
        opened_at,
        observation.asof,
        0,
    )
    ignored_h4 = LiquidityLevel(
        "current-h4-protection",
        Timeframe.H4,
        "below",
        99.75,
        opened_at - pd.Timedelta(hours=4),
        opened_at,
        0,
    )

    def typed_source(level: LiquidityLevel):
        source_id = f"source:{level.level_id}"
        template = observation.frame(Timeframe.H1).swings[0]
        swing = replace(
            template,
            swing_id=source_id,
            timeframe=level.timeframe,
            price=level.price,
            price_ticks=int(round(level.price / 0.25)),
            pivot_start=level.formed_at - pd.Timedelta(minutes=1),
            pivot_end=level.formed_at,
            observed_at=level.confirmed_at,
            confirmed_at=level.confirmed_at,
            broken_at=None,
        )
        item = LiquidityInventoryItem(
            item_id=level.level_id,
            timeframe=level.timeframe,
            side=level.side,
            kind="swing",
            price=level.price,
            lower_bound=level.price,
            upper_bound=level.price,
            formed_at=level.formed_at,
            confirmed_at=level.confirmed_at,
            lifecycle=LiquidityInventoryLifecycle.VISIBLE,
            source_ids=(source_id,),
            age_bars=0,
            strength=0.5,
        )
        return swing, item

    m1_swing, m1_item = typed_source(current_m1)
    h4_swing, h4_item = typed_source(ignored_h4)
    frames = dict(observation.frames)
    frames[Timeframe.M1] = replace(
        frames[Timeframe.M1],
        swings=(*frames[Timeframe.M1].swings, m1_swing),
    )
    frames[Timeframe.H4] = replace(
        frames[Timeframe.H4],
        swings=(*frames[Timeframe.H4].swings, h4_swing),
    )
    observation = replace(
        observation,
        frames=frames,
        liquidity_inventory=(
            *observation.liquidity_inventory,
            m1_item,
            h4_item,
        ),
    )
    position = PositionSnapshot(
        thesis_hash="a" * 64,
        symbol=observation.symbol,
        instrument_id=observation.instrument_id,
        playbook=plan.playbook,
        direction=plan.direction,
        entry_price=plan.planned_entry,
        original_invalidation=plan.invalidation,
        current_stop=plan.invalidation.price,
        primary_target=plan.targets[0],
        opened_at=opened_at,
        deadline=plan.deadline,
        quantity=1,
        unrealized_R=0.0,
        elapsed_minutes=1,
    )
    account = replace(flat_account(), position=position)
    utility = ActionUtility(Action.PROTECT, 0.5, {}, None, "protect")
    decision = Decision(
        observation.asof,
        Action.PROTECT,
        (utility,),
        None,
        0.5,
        ("protect",),
        None,
    )
    result = StructuralRiskEngine().review(decision, observation, account)
    assert result.passed
    assert result.final_action is Action.PROTECT
    assert result.protected_stop == current_m1.price


def test_protect_does_not_use_equal_liquidity_as_structure() -> None:
    observation = market_observation()
    plan = long_plan(observation)
    opened_at = observation.asof - pd.Timedelta(minutes=2)
    equal_low = LiquidityInventoryItem(
        item_id="pool:equal-low-not-a-stop",
        timeframe=Timeframe.M1,
        side="below",
        kind="equal_lows",
        price=99.25,
        lower_bound=99.25,
        upper_bound=99.5,
        formed_at=opened_at,
        confirmed_at=observation.asof - pd.Timedelta(minutes=1),
        lifecycle=LiquidityInventoryLifecycle.VISIBLE,
        source_ids=("low-a", "low-b"),
        age_bars=1,
        strength=0.8,
    )
    equal_low_pool = LiquidityPoolState(
        pool_id="equal-low-not-a-stop",
        timeframe=Timeframe.M1,
        side="below",
        lower_bound=equal_low.lower_bound,
        upper_bound=equal_low.upper_bound,
        midpoint=(equal_low.lower_bound + equal_low.upper_bound) / 2.0,
        formed_at=equal_low.formed_at,
        confirmed_at=equal_low.confirmed_at,
        lifecycle=LiquidityPoolLifecycle.FORMED,
        member_swing_ids=equal_low.source_ids,
        touch_times=(equal_low.formed_at, equal_low.confirmed_at),
        age_bars=equal_low.age_bars,
        strength=equal_low.strength,
        total_touch_count=2,
    )
    frames = dict(observation.frames)
    frames[Timeframe.M1] = replace(
        frames[Timeframe.M1],
        liquidity_pools=(equal_low_pool,),
    )
    observation = replace(
        observation,
        frames=frames,
        liquidity_inventory=(equal_low,),
        liquidity_pool_states=(equal_low_pool,),
    )
    position = PositionSnapshot(
        thesis_hash="a" * 64,
        symbol=observation.symbol,
        instrument_id=observation.instrument_id,
        playbook=plan.playbook,
        direction=plan.direction,
        entry_price=plan.planned_entry,
        original_invalidation=plan.invalidation,
        current_stop=plan.invalidation.price,
        primary_target=plan.targets[0],
        opened_at=opened_at,
        deadline=plan.deadline,
        quantity=1,
        unrealized_R=0.0,
        elapsed_minutes=2,
    )
    decision = Decision(
        observation.asof,
        Action.PROTECT,
        (ActionUtility(Action.PROTECT, 0.5, {}, None, "protect"),),
        None,
        0.5,
        ("protect",),
        None,
    )
    result = StructuralRiskEngine().review(
        decision,
        observation,
        replace(flat_account(), position=position),
    )
    assert not result.passed
    assert result.final_action is Action.ABSTAIN
    assert VetoCode.PROTECTION_NOT_TIGHTER in result.vetoes
