from __future__ import annotations

from dataclasses import replace
import math

import pandas as pd

from smc_trader.model import (
    AccountState,
    Action,
    ActionUtility,
    Bar,
    Candle,
    Decision,
    Direction,
    EngineSnapshot,
    EventKind,
    ExecutionObservation,
    FrameObservation,
    HypothesisBelief,
    LiquidityLevel,
    MarketBelief,
    MarketEvent,
    MarketObservation,
    Playbook,
    PlaybookPhase,
    RiskAssessment,
    StructuralLevel,
    Timeframe,
    TradePlan,
)


TZ = "America/New_York"


def session_bars(
    sessions: int = 5,
    *,
    first_trade_date: str = "2025-01-06",
) -> list[Bar]:
    trade_dates = pd.bdate_range(first_trade_date, periods=sessions)
    output: list[Bar] = []
    counter = 0
    for trade_date in trade_dates:
        start = (trade_date - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).tz_localize(TZ)
        end = (trade_date + pd.Timedelta(hours=17)).tz_localize(TZ)
        for timestamp in pd.date_range(start, end, freq="1min", inclusive="left"):
            center = 20_000.0 + 0.012 * counter + 7.0 * math.sin(counter / 47.0)
            open_price = center - 0.12 * math.sin(counter / 9.0)
            close = center + 0.12 * math.sin(counter / 9.0)
            output.append(
                Bar(
                    start=timestamp,
                    open=open_price,
                    high=max(open_price, close) + 0.5,
                    low=min(open_price, close) - 0.5,
                    close=close,
                    volume=100 + counter % 31,
                    symbol="NQH5",
                    instrument_id=1,
                )
            )
            counter += 1
    return output


def candle(timeframe: Timeframe, start: str, price: float) -> Candle:
    minutes = {
        Timeframe.H4: 240,
        Timeframe.H1: 60,
        Timeframe.M5: 5,
        Timeframe.M1: 1,
    }[timeframe]
    timestamp = pd.Timestamp(start, tz=TZ)
    return Candle(
        timeframe=timeframe,
        start=timestamp,
        end=timestamp + pd.Timedelta(minutes=minutes),
        open=price - 0.2,
        high=price + 0.6,
        low=price - 0.6,
        close=price + 0.2,
        volume=100,
        symbol="NQH5",
        instrument_id=1,
        observed_minutes=minutes,
        expected_minutes=minutes,
        complete=True,
    )


def market_observation(
    *,
    asof: pd.Timestamp | None = None,
    price: float = 100.0,
    deadline_missing: bool = False,
) -> MarketObservation:
    asof = asof or pd.Timestamp("2025-01-06 10:00", tz=TZ)
    formed = asof - pd.Timedelta(hours=4)
    confirmed = asof - pd.Timedelta(hours=2)
    below = LiquidityLevel(
        "below-level",
        Timeframe.H1,
        "below",
        98.0,
        formed,
        confirmed,
        0,
    )
    above = LiquidityLevel(
        "above-level",
        Timeframe.H1,
        "above",
        103.0,
        formed,
        confirmed,
        0,
    )
    h4_below = replace(
        below,
        level_id="h4-below-level",
        timeframe=Timeframe.H4,
    )
    h4_above = replace(
        above,
        level_id="h4-above-level",
        timeframe=Timeframe.H4,
    )
    m5_below = replace(
        below,
        level_id="m5-below-level",
        timeframe=Timeframe.M5,
    )
    m5_above = replace(
        above,
        level_id="m5-above-level",
        timeframe=Timeframe.M5,
    )
    common = {
        "atr": 2.0,
    }
    frames = {
        Timeframe.H4: FrameObservation(
            Timeframe.H4,
            asof,
            20,
            {
                **common,
                "directional_displacement": 0.7,
                "path_efficiency": 0.75,
                "structure_direction": 0.6,
                "structure_age_bars": 2.0,
                "range_position": 0.25,
                "dealing_range_low": 96.0,
                "dealing_range_high": 108.0,
                "external_above_distance_atr": 1.5,
                "external_below_distance_atr": 1.0,
                "external_above_count": 1.0,
                "external_below_count": 1.0,
            },
            (h4_below, h4_above),
            True,
        ),
        Timeframe.H1: FrameObservation(
            Timeframe.H1,
            asof,
            30,
            {
                **common,
                "swing_high_progression": 0.5,
                "swing_low_progression": 0.6,
                "swing_progression": 0.55,
                "acceptance_direction": 0.4,
                "rejection_direction": 0.65,
                "dealing_range_low": 98.0,
                "dealing_range_high": 103.0,
                "dealing_range_position": 0.3,
                "up_path_obstruction_atr": 1.5,
                "down_path_obstruction_atr": 1.0,
            },
            (below, above),
            True,
        ),
        Timeframe.M5: FrameObservation(
            Timeframe.M5,
            asof,
            30,
            {
                **common,
                "impulse_direction": 1.0,
                "impulse_strength": 0.8,
                "impulse_age_bars": 2.0,
                "impulse_extension_atr": 1.0,
                "pullback_depth": 0.45,
                "pullback_completeness": 0.9,
                "reacceptance_direction": 0.7,
                "compression": 0.2,
            },
            (m5_below, m5_above),
            True,
        ),
        Timeframe.M1: FrameObservation(
            Timeframe.M1,
            asof,
            60,
            {
                **common,
                "path_sequence": 0.7,
                "acceleration": 0.5,
                "counter_pressure": 0.4,
                "trigger_hold_direction": 1.0,
                "trigger_high": 99.5,
                "trigger_low": 98.5,
                "trigger_age_bars": 3.0,
            },
            (),
            True,
        ),
    }
    sweep = MarketEvent(
        "sweep",
        EventKind.LIQUIDITY_SWEEP,
        asof - pd.Timedelta(minutes=3),
        Timeframe.M1,
        "below",
        97.75,
        0.9,
        ("old-below",),
    )
    anomalies = ("deadline_missing",) if deadline_missing else ()
    execution = ExecutionObservation(
        spread_points=0.25,
        expected_slippage_points=0.25,
        expected_round_trip_cost_points=0.975,
        minutes_to_deadline=60,
        fillability=0.9,
        data_age_seconds=0.0,
        size_available=10,
        anomalies=anomalies,
        source="synthetic_observed_execution",
    )
    return MarketObservation(
        asof=asof,
        symbol="NQH5",
        instrument_id=1,
        price=price,
        frames=frames,
        recent_events=(sweep,),
        event_durations_minutes={"liquidity_sweep:below:1m": 3},
        execution=execution,
        anomalies=anomalies,
    )


def long_plan(observation: MarketObservation | None = None) -> TradePlan:
    observation = observation or market_observation()
    below = observation.frame(Timeframe.H1).liquidity[0]
    above = observation.frame(Timeframe.H1).liquidity[1]
    invalidation = StructuralLevel(
        below.price,
        "below",
        below.level_id,
        below.confirmed_at,
        "confirmed swing low",
    )
    risk = observation.price - invalidation.price
    return TradePlan(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        observation.price,
        invalidation,
        (above,),
        risk,
        (above.price - observation.price) / risk,
        (above.price - observation.price) / risk,
        observation.asof + pd.Timedelta(minutes=60),
    )


def executable_belief(
    observation: MarketObservation | None = None,
    *,
    probability: float = 0.85,
    uncertainty: float = 0.1,
) -> MarketBelief:
    observation = observation or market_observation()
    plan = long_plan(observation)
    hypothesis = HypothesisBelief(
        Playbook.DISPLACEMENT_FIRST_PULLBACK,
        Direction.LONG,
        probability,
        PlaybookPhase.EXECUTABLE,
        observation.asof - pd.Timedelta(minutes=3),
        (),
        (),
        plan.invalidation,
        plan.targets,
        plan.remaining_path_R,
        uncertainty,
        plan,
    )
    return MarketBelief(observation.asof, {hypothesis.key: hypothesis})


def engine_snapshot() -> EngineSnapshot:
    observation = market_observation()
    belief = executable_belief(observation)
    plan = long_plan(observation)
    utility = ActionUtility(
        Action.ENTER,
        0.5,
        {
            "expected_gross_R": 0.5,
            "cost_R": 0.0,
            "uncertainty": 0.0,
            "deadline": 0.0,
            "fillability": 0.0,
            "phase_readiness": 0.0,
        },
        next(iter(belief.hypotheses)),
        "reason",
    )
    decision = Decision(
        observation.asof,
        Action.ENTER,
        (utility,),
        next(iter(belief.hypotheses)),
        0.5,
        ("reason",),
        plan,
    )
    risk = RiskAssessment(Action.ENTER, Action.ENTER, True, (), ("passed",))
    return EngineSnapshot(observation, belief, decision, risk, "a" * 64)


def flat_account() -> AccountState:
    return AccountState(100_000.0)
