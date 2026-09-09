from __future__ import annotations

from dataclasses import fields, replace
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from pathlib import Path

import pandas as pd

from contract.market import (
    Bar,
    Candle,
    Direction,
    LiquidityLevel,
    Playbook,
    PlaybookPhase,
    StructuralLevel,
    Timeframe,
    ticks_to_price,
)
from contract.execution import (
    AccountState,
    ExecutionObservation,
)
from contract.eye import (
    EventKind,
    FrameObservation,
    LiquidityInventoryItem,
    LiquidityInventoryLifecycle,
    MarketEvent,
    MarketObservation,
    SwingLifecycle,
    SwingPoint,
    SwingRelation,
    SwingSide,
)
from contract.brain import (
    HypothesisBelief,
    MarketBelief,
    TradePlan,
)
from contract.decision import (
    Action,
    ActionUtility,
    Decision,
)
from contract.risk import RiskAssessment
from contract.research import EngineSnapshot
from shares.core.scale_registry import parse_scale_specs, scale_registry_id


TZ = "America/New_York"
ROOT = Path(__file__).resolve().parents[2]
MODEL_SCALE_SPECS = parse_scale_specs(
    json.loads((ROOT / "configs/model.json").read_text(encoding="utf-8"))["scales"]
)
CORE_TEST_SCALE_SPECS = tuple(
    spec
    for spec in MODEL_SCALE_SPECS
    if spec.native_timeframe
    in {Timeframe.H4, Timeframe.H1, Timeframe.M5, Timeframe.M1}
)
CORE_TEST_SCALE_REGISTRY_ID = scale_registry_id(CORE_TEST_SCALE_SPECS)


class GraphFreeActionBelief(MarketBelief):
    """Test-only adapter for evaluator/Decision tests without a Scene Graph."""

    def action_candidate_items(self):
        if self.global_context is not None:
            return super().action_candidate_items()
        return tuple(self.hypotheses.items())

    def owns_actionable_entry_episode(self, candidate_id, hypothesis):
        if self.global_context is not None:
            return super().owns_actionable_entry_episode(
                candidate_id,
                hypothesis,
            )
        plan = hypothesis.plan
        return bool(
            self.hypotheses.get(candidate_id) is hypothesis
            and hypothesis.context_thesis_id
            and hypothesis.episode_id
            and hypothesis.parent_context_thesis_id
            == hypothesis.context_thesis_id
            and plan is not None
            and plan.setup_id == hypothesis.episode_id
            and hypothesis.setup_context_id == hypothesis.episode_id
        )


def graph_free_action_belief(belief: MarketBelief) -> MarketBelief:
    """Explicitly grant summary-slot action authority in a unit-test fixture.

    Runtime ``MarketBelief`` is intentionally fail-closed without root-specific
    candidates.  This adapter keeps graph-free component tests focused on the
    downstream behavior they were built to exercise.
    """

    return GraphFreeActionBelief(
        **{
            item.name: getattr(belief, item.name)
            for item in fields(MarketBelief)
        }
    )


def session_bars(
    sessions: int = 5,
    *,
    first_trade_date: str = "2025-01-06",
) -> list[Bar]:
    trade_dates = pd.bdate_range(first_trade_date, periods=sessions)
    output: list[Bar] = []
    counter = 0

    def test_grid_price(value: float) -> float:
        coordinate = (
            Decimal(str(value)) / Decimal("0.25")
        ).to_integral_value(rounding=ROUND_HALF_UP)
        return ticks_to_price(int(coordinate), 0.25)

    for trade_date in trade_dates:
        start = (trade_date - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).tz_localize(TZ)
        end = (trade_date + pd.Timedelta(hours=17)).tz_localize(TZ)
        for timestamp in pd.date_range(start, end, freq="1min", inclusive="left"):
            center = 20_000.0 + 0.012 * counter + 7.0 * math.sin(counter / 47.0)
            open_price = test_grid_price(
                center - 0.12 * math.sin(counter / 9.0)
            )
            close = test_grid_price(
                center + 0.12 * math.sin(counter / 9.0)
            )
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
        Timeframe.M15: 15,
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

    def typed_swing(
        level: LiquidityLevel,
    ) -> tuple[SwingPoint, LiquidityInventoryItem]:
        source_id = f"source:{level.level_id}"
        side = (
            SwingSide.HIGH if level.side == "above" else SwingSide.LOW
        )
        swing = SwingPoint(
            swing_id=source_id,
            timeframe=level.timeframe,
            symbol="NQH5",
            instrument_id=1,
            side=side,
            price=level.price,
            price_ticks=int(round(level.price / 0.25)),
            pivot_start=level.formed_at,
            pivot_end=level.formed_at + pd.Timedelta(minutes=1),
            observed_at=level.confirmed_at,
            confirmed_at=level.confirmed_at,
            lifecycle=SwingLifecycle.CONFIRMED,
            relation=(
                SwingRelation.HH
                if side is SwingSide.HIGH
                else SwingRelation.HL
            ),
            delta_ticks=1,
            delta_points=0.25,
            magnitude_atr=0.25,
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
            visibility_strength=0.5,
        )
        return swing, item

    h4_swing_items = tuple(typed_swing(level) for level in (h4_below, h4_above))
    h1_swing_items = tuple(typed_swing(level) for level in (below, above))
    m5_swing_items = tuple(typed_swing(level) for level in (m5_below, m5_above))
    common = {
        "atr": 2.0,
    }
    frames = {
        Timeframe.H4: FrameObservation(
            timeframe=Timeframe.H4,
            cutoff=asof,
            bars=20,
            metrics={
                **common,
                "directional_displacement": 0.7,
                "path_efficiency": 0.75,
                "structure_direction": 0.6,
                "structure_age_bars": 2.0,
                "range_position": 0.25,
                "rolling_range_low": 96.0,
                "rolling_range_high": 108.0,
                "external_above_distance_atr": 1.5,
                "external_below_distance_atr": 1.0,
                "external_above_count": 1.0,
                "external_below_count": 1.0,
            },
            ready=True,
            swings=tuple(item[0] for item in h4_swing_items),
        ),
        Timeframe.H1: FrameObservation(
            timeframe=Timeframe.H1,
            cutoff=asof,
            bars=30,
            metrics={
                **common,
                "swing_high_progression": 0.5,
                "swing_low_progression": 0.6,
                "swing_progression": 0.55,
                "acceptance_direction": 0.4,
                "rejection_direction": 0.65,
                "rolling_range_low": 98.0,
                "rolling_range_high": 103.0,
                "rolling_range_position": 0.3,
                "up_path_obstruction_atr": 1.5,
                "down_path_obstruction_atr": 1.0,
            },
            ready=True,
            swings=tuple(item[0] for item in h1_swing_items),
        ),
        Timeframe.M5: FrameObservation(
            timeframe=Timeframe.M5,
            cutoff=asof,
            bars=30,
            metrics={
                **common,
                "compression": 0.2,
            },
            ready=True,
            swings=tuple(item[0] for item in m5_swing_items),
        ),
        Timeframe.M1: FrameObservation(
            timeframe=Timeframe.M1,
            cutoff=asof,
            bars=60,
            metrics={
                **common,
                "acceleration": 0.5,
                "counter_pressure": 0.4,
            },
            ready=True,
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
        market_snapshot=None,
        _snapshot_free_identity=(asof, "NQH5", 1, price, ()),
        frames=frames,
        recent_events=(sweep,),
        event_durations_minutes={"liquidity_sweep:below:1m": 3},
        execution=execution,
        anomalies=anomalies,
        liquidity_inventory=tuple(
            item[1]
            for pair in (h4_swing_items, h1_swing_items, m5_swing_items)
            for item in pair
        ),
        active_timeframes=tuple(frames),
        scale_registry_id=CORE_TEST_SCALE_REGISTRY_ID,
    )


def replace_market_observation(
    observation: MarketObservation,
    /,
    **changes: object,
) -> MarketObservation:
    """Replace an observation while keeping its sole snapshot identity aligned."""

    identity_names = {
        "asof",
        "symbol",
        "instrument_id",
        "price",
        "semantic_events_this_update",
    }
    identity_changes = {
        name: changes.pop(name)
        for name in tuple(changes)
        if name in identity_names
    }
    if "market_snapshot" in changes and identity_changes:
        raise ValueError(
            "replace one observation identity through either snapshot or aliases"
        )
    snapshot_replacement = changes.get("market_snapshot", ...)
    snapshot = observation.market_snapshot
    if snapshot_replacement is not ... and snapshot_replacement is not None:
        changes["_snapshot_free_identity"] = None
    elif snapshot is None:
        current = {
            "asof": observation.asof,
            "symbol": observation.symbol,
            "instrument_id": observation.instrument_id,
            "price": observation.price,
            "semantic_events_this_update": (
                observation.semantic_events_this_update
            ),
        }
        current.update(identity_changes)
        changes["_snapshot_free_identity"] = (
            current["asof"],
            current["symbol"],
            current["instrument_id"],
            current["price"],
            tuple(current["semantic_events_this_update"]),
        )
    elif identity_changes:
        snapshot_changes: dict[str, object] = {}
        for name, value in identity_changes.items():
            snapshot_changes[
                "events_this_update"
                if name == "semantic_events_this_update"
                else name
            ] = value
        if "asof" in identity_changes:
            snapshot_changes["session"] = replace(
                snapshot.session,
                known_at=identity_changes["asof"],
            )
        changes["market_snapshot"] = replace(snapshot, **snapshot_changes)
    return replace(observation, **changes)


def long_plan(observation: MarketObservation | None = None) -> TradePlan:
    observation = observation or market_observation()
    levels = {
        item.item_id: LiquidityLevel(
            level_id=item.item_id,
            timeframe=item.timeframe,
            side=item.side,
            price=item.price,
            formed_at=item.formed_at,
            confirmed_at=item.confirmed_at,
            touches=max(0, len(item.source_ids) - 1),
        )
        for item in observation.liquidity_inventory
    }
    below = levels["below-level"]
    above = levels["above-level"]
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
        thesis_strength=probability,
        sequence_progress=1.0,
        location_quality=1.0,
        entry_readiness=1.0,
        delivery_quality=1.0,
        evidence_group_scores={
            "structure": probability,
            "displacement": 1.0,
            "location": 1.0,
            "liquidity": probability,
            "trigger": 1.0,
            "execution": 1.0,
        },
        hard_gate_results={"synthetic_current_gate": True},
        raw_quality_dimensions={
            "thesis_strength": probability,
            "sequence_progress": 1.0,
            "location_quality": 1.0,
            "entry_readiness": 1.0,
            "delivery_quality": 1.0,
            "uncertainty": uncertainty,
        },
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
    return EngineSnapshot(observation, belief, decision, risk)


def flat_account() -> AccountState:
    return AccountState(100_000.0)
