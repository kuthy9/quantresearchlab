"""Frozen causal shadow replay for v2.3 flat and position actions."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .action_clock import CandidatePlan, FLAT_ACTIONS, freeze_candidate
from .model import (
    Bar,
    Direction,
    EngineSnapshot,
    FrozenThesis,
    PlaybookPhase,
    StructuralLevel,
    Timeframe,
    TradePlan,
    to_primitive,
)
from .risk import conservative_entry_bar, conservative_position_bar


POSITION_ACTIONS = ("hold", "protect", "exit")
POSITION_SAMPLING_STRIDE_MINUTES = 5
POSITION_FEATURE_NAMES = (
    "position_belief_probability",
    "position_uncertainty",
    "position_mark_R",
    "position_parent_mfe_R",
    "position_parent_mae_R",
    "position_elapsed_scaled",
    "position_remaining_target_R",
    "position_stop_distance_R",
    "position_minutes_to_deadline_scaled",
    "position_m5_reacceptance_aligned",
    "position_m1_path_aligned",
    "position_m1_counter_pressure_against",
    "position_m1_trigger_hold_aligned",
    "position_protection_available",
    "position_applied_stop_R",
    "position_action_hold",
    "position_action_protect",
    "position_action_exit",
)


def shadow_action_key(candidate_id: str, action_id: str) -> str:
    if action_id not in FLAT_ACTIONS:
        raise ValueError(f"unregistered flat action: {action_id}")
    return hashlib.sha256(f"{candidate_id}|{action_id}".encode("utf-8")).hexdigest()


def _gross_R(
    direction: Direction,
    entry: float,
    exit_price: float,
    original_risk: float,
) -> float:
    if original_risk <= 0:
        raise ValueError("shadow outcome has invalid original risk")
    return float(direction.sign * (float(exit_price) - float(entry)) / original_risk)


def _improved_plan(plan: TradePlan) -> TradePlan:
    improved_entry = plan.planned_entry - plan.direction.sign * 0.25 * plan.risk_points
    new_risk = abs(improved_entry - plan.invalidation.price)
    if new_risk <= 0:
        raise ValueError("better-price wait crossed the frozen invalidation")
    primary_R = (
        plan.direction.sign * (plan.targets[0].price - improved_entry) / new_risk
    )
    return replace(
        plan,
        planned_entry=float(improved_entry),
        risk_points=float(new_risk),
        primary_target_R=float(primary_R),
        remaining_path_R=float(primary_R),
    )


def _pre_entry_terminal(
    thesis: FrozenThesis,
    bar: Bar,
) -> str | None:
    if bar.start >= thesis.deadline:
        return "deadline_before_fill"
    if thesis.direction is Direction.LONG:
        if bar.low <= thesis.original_invalidation.price:
            return "invalidation_before_fill"
        if bar.high >= thesis.original_targets[0].price:
            return "target_consumed_before_fill"
    else:
        if bar.high >= thesis.original_invalidation.price:
            return "invalidation_before_fill"
        if bar.low <= thesis.original_targets[0].price:
            return "target_consumed_before_fill"
    return None


@dataclass
class _ShadowFlatAction:
    candidate: CandidatePlan
    action_id: str
    action_key: str
    thesis: FrozenThesis
    order_plan: TradePlan
    symbol: str
    instrument_id: int
    decision_cost_R: float | None
    cost_source: str | None
    cost_observed_at: pd.Timestamp | None
    status: str
    bars_waited: int = 0
    trigger_lost: bool = False
    filled_at: pd.Timestamp | None = None
    entry_price: float | None = None
    mfe_R: float = 0.0
    mae_R: float = 0.0
    last_bar_end: pd.Timestamp | None = None
    last_position_sample_elapsed: int = -POSITION_SAMPLING_STRIDE_MINUTES
    position_states: dict[str, "_PositionClockState"] = field(
        default_factory=dict
    )
    retired_position_state_ids: set[str] = field(default_factory=set)

    @property
    def original_risk(self) -> float:
        return float(self.candidate.plan.risk_points)


@dataclass
class _PositionTrial:
    position_action_key: str
    candidate_id: str
    parent_action_key: str
    action_id: str
    decision_time: pd.Timestamp
    thesis: FrozenThesis
    symbol: str
    instrument_id: int
    entry_price: float
    original_risk: float
    current_stop: float
    applied_stop: float
    target: float
    cost_R: float | None
    mark_R: float
    parent_mfe_R: float
    parent_mae_R: float
    elapsed_minutes: int
    protection_source_id: str | None
    status: str = "pending"
    mfe_R: float = 0.0
    mae_R: float = 0.0


@dataclass
class _PositionClockState:
    """One causally reachable current-stop state on the shared price path."""

    state_id: str
    current_stop: float
    protection_source_id: str | None
    activated_at: pd.Timestamp


def _terminal_row(
    action: _ShadowFlatAction,
    *,
    resolved_at: pd.Timestamp,
    outcome: str,
    filled: bool,
    gross_R: float | None,
    ambiguous_same_bar: bool = False,
    right_censored: bool = False,
) -> dict[str, Any]:
    cost_R = (
        action.decision_cost_R
        if filled
        else 0.0
    )
    net_R = (
        None
        if gross_R is None or cost_R is None
        else float(gross_R - cost_R)
    )
    return {
        "candidate_id": action.candidate.candidate_id,
        "action_key": action.action_key,
        "action_id": action.action_id,
        "decision_time": action.candidate.decision_time,
        "resolved_at": resolved_at,
        "status": "right_censored" if right_censored else "resolved",
        "outcome": outcome,
        "filled": bool(filled),
        "filled_at": action.filled_at,
        "entry_price": action.entry_price,
        "gross_R": None if gross_R is None else float(gross_R),
        "cost_R": cost_R,
        "cost_source": action.cost_source if filled else None,
        "cost_observed_at": action.cost_observed_at if filled else None,
        "net_R": net_R,
        "mfe_R": float(action.mfe_R),
        "mae_R": float(action.mae_R),
        "bars_waited": int(action.bars_waited),
        "ambiguous_same_bar": bool(ambiguous_same_bar),
        "right_censored": bool(right_censored),
    }


def _position_terminal_row(
    trial: _PositionTrial,
    *,
    resolved_at: pd.Timestamp,
    outcome: str,
    gross_R: float | None,
    right_censored: bool = False,
    ambiguous_same_bar: bool = False,
) -> dict[str, Any]:
    net_R = (
        None
        if gross_R is None or trial.cost_R is None
        else float(gross_R - trial.cost_R)
    )
    return {
        "position_action_key": trial.position_action_key,
        "candidate_id": trial.candidate_id,
        "parent_action_key": trial.parent_action_key,
        "action_id": trial.action_id,
        "decision_time": trial.decision_time,
        "resolved_at": resolved_at,
        "status": "right_censored" if right_censored else "resolved",
        "outcome": outcome,
        "gross_R": None if gross_R is None else float(gross_R),
        "cost_R": trial.cost_R,
        "net_R": net_R,
        "mfe_R": float(trial.mfe_R),
        "mae_R": float(trial.mae_R),
        "ambiguous_same_bar": bool(ambiguous_same_bar),
        "right_censored": bool(right_censored),
    }


def _contract_matches(symbol: str, instrument_id: int, bar: Bar) -> bool:
    return (symbol, int(instrument_id)) == (bar.symbol, int(bar.instrument_id))


def _mark_extremes(
    direction: Direction,
    entry: float,
    original_risk: float,
    bar: Bar,
    current_mfe: float,
    current_mae: float,
) -> tuple[float, float]:
    favorable_price = bar.high if direction is Direction.LONG else bar.low
    adverse_price = bar.low if direction is Direction.LONG else bar.high
    favorable = _gross_R(direction, entry, favorable_price, original_risk)
    adverse = _gross_R(direction, entry, adverse_price, original_risk)
    return max(current_mfe, favorable), min(current_mae, adverse)


def _setup_still_executable(
    action: _ShadowFlatAction,
    snapshot: EngineSnapshot,
) -> bool:
    hypothesis = snapshot.belief.hypotheses.get(
        action.candidate.representative.key
    )
    if hypothesis is None or hypothesis.phase is not PlaybookPhase.EXECUTABLE:
        return False
    original_sequence = action.candidate.representative.sequence
    current_sequence = hypothesis.sequence
    return bool(
        original_sequence is not None
        and current_sequence is not None
        and original_sequence.setup_id is not None
        and current_sequence.setup_id == original_sequence.setup_id
    )


def _protection_candidate(
    action: _ShadowFlatAction,
    snapshot: EngineSnapshot,
    *,
    current_stop: float,
) -> StructuralLevel | None:
    if action.entry_price is None or action.filled_at is None:
        return None
    direction = action.thesis.direction
    current_price = snapshot.observation.price
    consumed = {
        source_id
        for event in snapshot.observation.recent_events
        for source_id in event.source_ids
    }
    candidates = [
        level
        for timeframe in (Timeframe.H1, Timeframe.M5, Timeframe.M1)
        for level in snapshot.observation.frame(timeframe).liquidity
        if (
            not level.swept
            and level.level_id not in consumed
            and level.confirmed_at > action.filled_at
            and level.confirmed_at <= snapshot.observation.asof
            and (
                (
                    direction is Direction.LONG
                    and current_stop < level.price < current_price
                    and level.side == "below"
                )
                or (
                    direction is Direction.SHORT
                    and current_price < level.price < current_stop
                    and level.side == "above"
                )
            )
        )
    ]
    if not candidates:
        return None
    source = (
        max(candidates, key=lambda level: level.price)
        if direction is Direction.LONG
        else min(candidates, key=lambda level: level.price)
    )
    return StructuralLevel(
        price=source.price,
        side=source.side,
        source_level_id=source.level_id,
        observed_at=source.confirmed_at,
        rationale="v2.3 shadow protection uses later confirmed structure",
    )


def _position_state_id(
    action_key: str,
    *,
    current_stop: float,
    source_id: str | None,
) -> str:
    identity = (
        f"{action_key}|{float(current_stop):.10f}|"
        f"{source_id or 'original_invalidation'}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _ensure_baseline_position_state(
    action: _ShadowFlatAction,
    *,
    activated_at: pd.Timestamp,
) -> None:
    if action.position_states:
        return
    stop = action.thesis.original_invalidation.price
    state_id = _position_state_id(
        action.action_key,
        current_stop=stop,
        source_id=None,
    )
    action.position_states[state_id] = _PositionClockState(
        state_id=state_id,
        current_stop=float(stop),
        protection_source_id=None,
        activated_at=activated_at,
    )


class FrozenShadowReplay:
    """Resolve many overlapping frozen actions with bounded active state."""

    def __init__(self) -> None:
        self.active: dict[str, _ShadowFlatAction] = {}
        self.position_trials: dict[str, _PositionTrial] = {}
        self._resolved_flat: list[dict[str, Any]] = []
        self._position_samples: list[dict[str, Any]] = []
        self._resolved_positions: list[dict[str, Any]] = []
        self.registered_candidates = 0
        self.registered_actions = 0
        self.resolved_actions = 0
        self.resolved_position_actions = 0

    def register(
        self,
        candidate: CandidatePlan,
        snapshot: EngineSnapshot,
    ) -> None:
        if candidate.decision_time != snapshot.observation.asof:
            raise ValueError("candidate registration clock differs from snapshot")
        self.registered_candidates += 1
        execution = snapshot.observation.execution
        observed_cost = (
            execution.expected_round_trip_cost_points / candidate.plan.risk_points
            if execution.source == "mbo_reconstructed"
            else None
        )
        for action_id in FLAT_ACTIONS:
            action_key = shadow_action_key(candidate.candidate_id, action_id)
            if action_key in self.active:
                raise ValueError("duplicate shadow action key")
            self.registered_actions += 1
            if action_id == "abstain":
                dummy = _ShadowFlatAction(
                    candidate=candidate,
                    action_id=action_id,
                    action_key=action_key,
                    thesis=freeze_candidate(candidate),
                    order_plan=candidate.plan,
                    symbol=snapshot.observation.symbol,
                    instrument_id=snapshot.observation.instrument_id,
                    decision_cost_R=0.0,
                    cost_source=None,
                    cost_observed_at=None,
                    status="resolved",
                )
                self._resolved_flat.append(
                    _terminal_row(
                        dummy,
                        resolved_at=candidate.decision_time,
                        outcome="abstained",
                        filled=False,
                        gross_R=0.0,
                    )
                )
                self.resolved_actions += 1
                continue
            order_plan = (
                _improved_plan(candidate.plan)
                if action_id == "wait_better_price"
                else candidate.plan
            )
            self.active[action_key] = _ShadowFlatAction(
                candidate=candidate,
                action_id=action_id,
                action_key=action_key,
                thesis=freeze_candidate(candidate),
                order_plan=order_plan,
                symbol=snapshot.observation.symbol,
                instrument_id=snapshot.observation.instrument_id,
                decision_cost_R=observed_cost,
                cost_source=(
                    execution.source if observed_cost is not None else None
                ),
                cost_observed_at=(
                    candidate.decision_time
                    if observed_cost is not None
                    else None
                ),
                status=(
                    "awaiting_entry"
                    if action_id in {"enter_now", "wait_better_price"}
                    else "waiting"
                ),
            )

    def _resolve_flat(
        self,
        action: _ShadowFlatAction,
        row: dict[str, Any],
    ) -> None:
        self._resolved_flat.append(row)
        self.active.pop(action.action_key, None)
        self.resolved_actions += 1

    def _fill_or_expire_one_bar(
        self,
        action: _ShadowFlatAction,
        bar: Bar,
    ) -> bool:
        result = conservative_entry_bar(action.order_plan, bar)
        action.bars_waited += 1
        if not result.filled:
            if action.action_id == "wait_better_price" and action.bars_waited < 5:
                terminal = _pre_entry_terminal(action.thesis, bar)
                if terminal is None:
                    return False
                self._resolve_flat(
                    action,
                    _terminal_row(
                        action,
                        resolved_at=bar.end,
                        outcome=terminal,
                        filled=False,
                        gross_R=0.0,
                    ),
                )
                return True
            self._resolve_flat(
                action,
                _terminal_row(
                    action,
                    resolved_at=bar.end,
                    outcome="order_expired_unfilled",
                    filled=False,
                    gross_R=0.0,
                ),
            )
            return True
        action.filled_at = bar.start
        action.entry_price = float(result.entry_price)
        action.mfe_R, action.mae_R = _mark_extremes(
            action.thesis.direction,
            action.entry_price,
            action.original_risk,
            bar,
            action.mfe_R,
            action.mae_R,
        )
        if result.closed:
            if result.exit_price is None:
                raise RuntimeError("closed entry bar has no exit")
            gross = _gross_R(
                action.thesis.direction,
                action.entry_price,
                result.exit_price,
                action.original_risk,
            )
            self._resolve_flat(
                action,
                _terminal_row(
                    action,
                    resolved_at=bar.end,
                    outcome=result.reason,
                    filled=True,
                    gross_R=gross,
                    ambiguous_same_bar=result.ambiguous_same_bar,
                ),
            )
            return True
        action.status = "open"
        action.last_bar_end = bar.end
        _ensure_baseline_position_state(
            action,
            activated_at=bar.end,
        )
        return False

    def _advance_wait(
        self,
        action: _ShadowFlatAction,
        bar: Bar,
        snapshot: EngineSnapshot,
    ) -> None:
        if not _contract_matches(action.symbol, action.instrument_id, bar):
            self._resolve_flat(
                action,
                _terminal_row(
                    action,
                    resolved_at=bar.start,
                    outcome="contract_change_before_fill",
                    filled=False,
                    gross_R=0.0,
                ),
            )
            return
        terminal = _pre_entry_terminal(action.thesis, bar)
        if terminal is not None:
            self._resolve_flat(
                action,
                _terminal_row(
                    action,
                    resolved_at=bar.end,
                    outcome=terminal,
                    filled=False,
                    gross_R=0.0,
                ),
            )
            return
        action.bars_waited += 1
        if action.action_id == "wait_one_bar":
            if action.bars_waited != 1:
                raise RuntimeError("wait-one-bar state advanced more than once")
            if not _setup_still_executable(action, snapshot):
                self._resolve_flat(
                    action,
                    _terminal_row(
                        action,
                        resolved_at=bar.end,
                        outcome="setup_not_executable_after_wait",
                        filled=False,
                        gross_R=0.0,
                    ),
                )
                return
            action.decision_cost_R = (
                snapshot.observation.execution.expected_round_trip_cost_points
                / action.original_risk
                if snapshot.observation.execution.source == "mbo_reconstructed"
                else None
            )
            action.cost_source = (
                snapshot.observation.execution.source
                if action.decision_cost_R is not None
                else None
            )
            action.cost_observed_at = (
                snapshot.observation.asof
                if action.decision_cost_R is not None
                else None
            )
            action.status = "awaiting_entry"
            return
        if action.action_id != "wait_reacceptance":
            raise RuntimeError("unknown waiting shadow action")
        m5 = snapshot.observation.frame(Timeframe.M5).metrics
        m1 = snapshot.observation.frame(Timeframe.M1).metrics
        sign = action.thesis.direction.sign
        trigger = sign * float(m1["trigger_hold_direction"])
        reacceptance = sign * float(m5["reacceptance_direction"])
        if trigger <= 0:
            action.trigger_lost = True
        if action.trigger_lost and trigger > 0 and reacceptance > 0:
            action.decision_cost_R = (
                snapshot.observation.execution.expected_round_trip_cost_points
                / action.original_risk
                if snapshot.observation.execution.source == "mbo_reconstructed"
                else None
            )
            action.cost_source = (
                snapshot.observation.execution.source
                if action.decision_cost_R is not None
                else None
            )
            action.cost_observed_at = (
                snapshot.observation.asof
                if action.decision_cost_R is not None
                else None
            )
            action.status = "awaiting_entry"
            return
        if action.bars_waited >= 15:
            self._resolve_flat(
                action,
                _terminal_row(
                    action,
                    resolved_at=bar.end,
                    outcome="reacceptance_wait_expired",
                    filled=False,
                    gross_R=0.0,
                ),
            )

    def _advance_open(
        self,
        action: _ShadowFlatAction,
        bar: Bar,
    ) -> None:
        if action.entry_price is None or action.filled_at is None:
            raise RuntimeError("open shadow action lacks fill state")
        if not _contract_matches(action.symbol, action.instrument_id, bar):
            gross = _gross_R(
                action.thesis.direction,
                action.entry_price,
                bar.open,
                action.original_risk,
            )
            self._resolve_flat(
                action,
                _terminal_row(
                    action,
                    resolved_at=bar.start,
                    outcome="contract_change_gap_exit",
                    filled=True,
                    gross_R=gross,
                ),
            )
            return
        if bar.start >= action.thesis.deadline:
            gross = _gross_R(
                action.thesis.direction,
                action.entry_price,
                bar.open,
                action.original_risk,
            )
            self._resolve_flat(
                action,
                _terminal_row(
                    action,
                    resolved_at=bar.start,
                    outcome="hard_deadline",
                    filled=True,
                    gross_R=gross,
                ),
            )
            return
        result = conservative_position_bar(
            action.thesis.direction,
            bar,
            current_stop=action.thesis.original_invalidation.price,
            target=action.thesis.original_targets[0].price,
        )
        action.mfe_R, action.mae_R = _mark_extremes(
            action.thesis.direction,
            action.entry_price,
            action.original_risk,
            bar,
            action.mfe_R,
            action.mae_R,
        )
        action.last_bar_end = bar.end
        if result.closed:
            if result.exit_price is None:
                raise RuntimeError("closed shadow position has no exit")
            gross = _gross_R(
                action.thesis.direction,
                action.entry_price,
                result.exit_price,
                action.original_risk,
            )
            self._resolve_flat(
                action,
                _terminal_row(
                    action,
                    resolved_at=bar.end,
                    outcome=result.reason,
                    filled=True,
                    gross_R=gross,
                    ambiguous_same_bar=result.ambiguous_same_bar,
                ),
            )

    def _advance_position_clock_states(
        self,
        action: _ShadowFlatAction,
        bar: Bar,
    ) -> None:
        """Drop protected-stop states that could not reach this action clock."""

        original_stop = action.thesis.original_invalidation.price
        for state_id, state in list(action.position_states.items()):
            if math.isclose(
                state.current_stop,
                original_stop,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                # The parent flat action owns the original-stop trajectory.
                continue
            if (
                not _contract_matches(action.symbol, action.instrument_id, bar)
                or bar.start >= action.thesis.deadline
            ):
                action.position_states.pop(state_id, None)
                action.retired_position_state_ids.add(state_id)
                continue
            result = conservative_position_bar(
                action.thesis.direction,
                bar,
                current_stop=state.current_stop,
                target=action.thesis.original_targets[0].price,
            )
            if result.closed:
                action.position_states.pop(state_id, None)
                # A protected trajectory that has touched its stop is no longer
                # causally reachable.  Keep a tombstone so an unchanged
                # structural level in the next snapshot cannot resurrect it.
                action.retired_position_state_ids.add(state_id)

    def _advance_position_trial(
        self,
        trial: _PositionTrial,
        bar: Bar,
    ) -> None:
        if not _contract_matches(trial.symbol, trial.instrument_id, bar):
            gross = _gross_R(
                trial.thesis.direction,
                trial.entry_price,
                bar.open,
                trial.original_risk,
            )
            self._resolved_positions.append(
                _position_terminal_row(
                    trial,
                    resolved_at=bar.start,
                    outcome="contract_change_gap_exit",
                    gross_R=gross,
                )
            )
            self.position_trials.pop(trial.position_action_key, None)
            self.resolved_position_actions += 1
            return
        if trial.action_id == "exit":
            gross = _gross_R(
                trial.thesis.direction,
                trial.entry_price,
                bar.open,
                trial.original_risk,
            )
            self._resolved_positions.append(
                _position_terminal_row(
                    trial,
                    resolved_at=bar.start,
                    outcome="shadow_exit_next_open",
                    gross_R=gross,
                )
            )
            self.position_trials.pop(trial.position_action_key, None)
            self.resolved_position_actions += 1
            return
        if bar.start >= trial.thesis.deadline:
            gross = _gross_R(
                trial.thesis.direction,
                trial.entry_price,
                bar.open,
                trial.original_risk,
            )
            self._resolved_positions.append(
                _position_terminal_row(
                    trial,
                    resolved_at=bar.start,
                    outcome="hard_deadline",
                    gross_R=gross,
                )
            )
            self.position_trials.pop(trial.position_action_key, None)
            self.resolved_position_actions += 1
            return
        result = conservative_position_bar(
            trial.thesis.direction,
            bar,
            current_stop=trial.applied_stop,
            target=trial.target,
        )
        trial.mfe_R, trial.mae_R = _mark_extremes(
            trial.thesis.direction,
            trial.entry_price,
            trial.original_risk,
            bar,
            trial.mfe_R,
            trial.mae_R,
        )
        if not result.closed:
            return
        if result.exit_price is None:
            raise RuntimeError("closed position trial has no exit")
        gross = _gross_R(
            trial.thesis.direction,
            trial.entry_price,
            result.exit_price,
            trial.original_risk,
        )
        self._resolved_positions.append(
            _position_terminal_row(
                trial,
                resolved_at=bar.end,
                outcome=result.reason,
                gross_R=gross,
                ambiguous_same_bar=result.ambiguous_same_bar,
            )
        )
        self.position_trials.pop(trial.position_action_key, None)
        self.resolved_position_actions += 1

    def _sample_position_actions(
        self,
        action: _ShadowFlatAction,
        snapshot: EngineSnapshot,
    ) -> None:
        if (
            action.action_id != "enter_now"
            or action.status != "open"
            or action.entry_price is None
            or action.filled_at is None
        ):
            return
        _ensure_baseline_position_state(
            action,
            activated_at=action.filled_at,
        )
        elapsed = max(
            0,
            int(
                (
                    snapshot.observation.asof - action.filled_at
                ).total_seconds()
                // 60
            ),
        )
        sample_due = (
            elapsed - action.last_position_sample_elapsed
            >= POSITION_SAMPLING_STRIDE_MINUTES
        )
        if sample_due:
            action.last_position_sample_elapsed = elapsed
        mark_R = _gross_R(
            action.thesis.direction,
            action.entry_price,
            snapshot.observation.price,
            action.original_risk,
        )
        hypothesis = snapshot.belief.hypotheses.get(
            action.candidate.representative.key
        )
        belief_probability = (
            0.5 if hypothesis is None else float(hypothesis.probability)
        )
        uncertainty = (
            1.0 if hypothesis is None else float(hypothesis.uncertainty)
        )
        sign = action.thesis.direction.sign
        m5 = snapshot.observation.frame(Timeframe.M5).metrics
        m1 = snapshot.observation.frame(Timeframe.M1).metrics
        minutes_to_deadline = max(
            0.0,
            (
                action.thesis.deadline - snapshot.observation.asof
            ).total_seconds()
            / 60.0,
        )
        discovered_states: dict[str, _PositionClockState] = {}
        for state in list(action.position_states.values()):
            protection = _protection_candidate(
                action,
                snapshot,
                current_stop=state.current_stop,
            )
            if protection is not None:
                discovered_id = _position_state_id(
                    action.action_key,
                    current_stop=protection.price,
                    source_id=protection.source_level_id,
                )
                if (
                    discovered_id not in action.position_states
                    and discovered_id not in discovered_states
                    and discovered_id
                    not in action.retired_position_state_ids
                ):
                    discovered_states[discovered_id] = _PositionClockState(
                        state_id=discovered_id,
                        current_stop=float(protection.price),
                        protection_source_id=protection.source_level_id,
                        activated_at=snapshot.observation.asof,
                    )
            if not sample_due:
                continue
            for position_action in POSITION_ACTIONS:
                if position_action == "protect" and protection is None:
                    continue
                position_key = hashlib.sha256(
                    (
                        f"{action.action_key}|{state.state_id}|"
                        f"{snapshot.observation.asof.isoformat()}|"
                        f"{position_action}"
                    ).encode("utf-8")
                ).hexdigest()
                applied_stop = (
                    protection.price
                    if position_action == "protect" and protection is not None
                    else state.current_stop
                )
                applied_source_id = (
                    protection.source_level_id
                    if position_action == "protect" and protection is not None
                    else state.protection_source_id
                )
                trial = _PositionTrial(
                    position_action_key=position_key,
                    candidate_id=action.candidate.candidate_id,
                    parent_action_key=action.action_key,
                    action_id=position_action,
                    decision_time=snapshot.observation.asof,
                    thesis=action.thesis,
                    symbol=action.symbol,
                    instrument_id=action.instrument_id,
                    entry_price=action.entry_price,
                    original_risk=action.original_risk,
                    current_stop=state.current_stop,
                    applied_stop=applied_stop,
                    target=action.thesis.original_targets[0].price,
                    cost_R=(
                        snapshot.observation.execution.expected_round_trip_cost_points
                        / (2.0 * action.original_risk)
                        if snapshot.observation.execution.source
                        == "mbo_reconstructed"
                        else None
                    ),
                    mark_R=mark_R,
                    parent_mfe_R=action.mfe_R,
                    parent_mae_R=action.mae_R,
                    elapsed_minutes=elapsed,
                    protection_source_id=applied_source_id,
                    mfe_R=mark_R,
                    mae_R=mark_R,
                )
                self.position_trials[position_key] = trial
                position_features = {
                    "position_belief_probability": belief_probability,
                    "position_uncertainty": uncertainty,
                    "position_mark_R": mark_R,
                    "position_parent_mfe_R": action.mfe_R,
                    "position_parent_mae_R": action.mae_R,
                    "position_elapsed_scaled": min(1.0, elapsed / 240.0),
                    "position_remaining_target_R": max(
                        0.0,
                        _gross_R(
                            action.thesis.direction,
                            snapshot.observation.price,
                            action.thesis.original_targets[0].price,
                            action.original_risk,
                        ),
                    ),
                    "position_stop_distance_R": (
                        abs(snapshot.observation.price - applied_stop)
                        / action.original_risk
                    ),
                    "position_minutes_to_deadline_scaled": min(
                        1.0,
                        minutes_to_deadline / 240.0,
                    ),
                    "position_m5_reacceptance_aligned": sign
                    * float(m5["reacceptance_direction"]),
                    "position_m1_path_aligned": sign
                    * float(m1["path_sequence"]),
                    "position_m1_counter_pressure_against": max(
                        0.0,
                        -sign * float(m1["counter_pressure"]),
                    ),
                    "position_m1_trigger_hold_aligned": sign
                    * float(m1["trigger_hold_direction"]),
                    "position_protection_available": (
                        1.0 if protection is not None else 0.0
                    ),
                    "position_applied_stop_R": _gross_R(
                        action.thesis.direction,
                        action.entry_price,
                        applied_stop,
                        action.original_risk,
                    ),
                    "position_action_hold": (
                        1.0 if position_action == "hold" else 0.0
                    ),
                    "position_action_protect": (
                        1.0 if position_action == "protect" else 0.0
                    ),
                    "position_action_exit": (
                        1.0 if position_action == "exit" else 0.0
                    ),
                }
                if (
                    tuple(position_features) != POSITION_FEATURE_NAMES
                    or not all(
                        math.isfinite(float(value))
                        for value in position_features.values()
                    )
                ):
                    raise ValueError("position action features violate protocol")
                self._position_samples.append(
                    {
                        "position_action_key": position_key,
                        "position_state_id": state.state_id,
                        "candidate_id": action.candidate.candidate_id,
                        "parent_action_key": action.action_key,
                        "action_id": position_action,
                        "decision_time": snapshot.observation.asof,
                        "representative_playbook": (
                            action.candidate.representative.playbook.value
                        ),
                        "direction": action.thesis.direction.value,
                        "entry_price": action.entry_price,
                        "original_invalidation": (
                            action.thesis.original_invalidation.price
                        ),
                        "current_stop": state.current_stop,
                        "current_stop_source_id": (
                            state.protection_source_id
                        ),
                        "current_stop_activated_at": state.activated_at,
                        "applied_stop": applied_stop,
                        "primary_target": (
                            action.thesis.original_targets[0].price
                        ),
                        "deadline": action.thesis.deadline,
                        "mark_R": mark_R,
                        "parent_mfe_R": action.mfe_R,
                        "parent_mae_R": action.mae_R,
                        "elapsed_minutes": elapsed,
                        "remaining_target_R": max(
                            0.0,
                            _gross_R(
                                action.thesis.direction,
                                snapshot.observation.price,
                                action.thesis.original_targets[0].price,
                                action.original_risk,
                            ),
                        ),
                        "stop_distance_R": abs(
                            snapshot.observation.price - applied_stop
                        )
                        / action.original_risk,
                        "protection_available": protection is not None,
                        "protection_source_id": applied_source_id,
                        "belief_state_json": json.dumps(
                            {
                                key: {
                                    "probability": value.probability,
                                    "uncertainty": value.uncertainty,
                                    "phase": value.phase.value,
                                }
                                for key, value in (
                                    snapshot.belief.hypotheses.items()
                                )
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "observation_state_json": json.dumps(
                            {
                                timeframe.value: dict(
                                    snapshot.observation.frame(
                                        timeframe
                                    ).metrics
                                )
                                for timeframe in (
                                    snapshot.observation.active_timeframes
                                )
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "execution_source": (
                            snapshot.observation.execution.source
                        ),
                        **position_features,
                    }
                )
        action.position_states.update(discovered_states)

    def advance(
        self,
        bar: Bar,
        snapshot: EngineSnapshot,
    ) -> None:
        """Advance prior actions through this bar, then sample new position clocks."""

        if snapshot.observation.asof != bar.end:
            raise ValueError("shadow replay snapshot does not match completed bar")
        for trial in list(self.position_trials.values()):
            self._advance_position_trial(trial, bar)
        for action in list(self.active.values()):
            if action.status == "open":
                self._advance_position_clock_states(action, bar)
                self._advance_open(action, bar)
                continue
            if not _contract_matches(action.symbol, action.instrument_id, bar):
                self._resolve_flat(
                    action,
                    _terminal_row(
                        action,
                        resolved_at=bar.start,
                        outcome="contract_change_before_fill",
                        filled=False,
                        gross_R=0.0,
                    ),
                )
                continue
            if action.status == "waiting":
                self._advance_wait(action, bar, snapshot)
                continue
            if action.status == "awaiting_entry":
                if bar.start >= action.thesis.deadline:
                    self._resolve_flat(
                        action,
                        _terminal_row(
                            action,
                            resolved_at=bar.start,
                            outcome="deadline_before_fill",
                            filled=False,
                            gross_R=0.0,
                        ),
                    )
                    continue
                self._fill_or_expire_one_bar(action, bar)
                continue
            raise RuntimeError(f"unknown shadow action state: {action.status}")
        for action in list(self.active.values()):
            self._sample_position_actions(action, snapshot)

    def drain(self) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        resolved = self._resolved_flat
        samples = self._position_samples
        positions = self._resolved_positions
        self._resolved_flat = []
        self._position_samples = []
        self._resolved_positions = []
        return resolved, samples, positions

    def finalize_boundary(
        self,
        boundary: pd.Timestamp,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        for action in list(self.active.values()):
            self._resolve_flat(
                action,
                _terminal_row(
                    action,
                    resolved_at=boundary,
                    outcome="right_boundary_unresolved",
                    filled=action.entry_price is not None,
                    gross_R=None,
                    right_censored=True,
                ),
            )
        for trial in list(self.position_trials.values()):
            self._resolved_positions.append(
                _position_terminal_row(
                    trial,
                    resolved_at=boundary,
                    outcome="right_boundary_unresolved",
                    gross_R=None,
                    right_censored=True,
                )
            )
            self.position_trials.pop(trial.position_action_key, None)
            self.resolved_position_actions += 1
        return self.drain()

    def finalize_data_gap(
        self,
        boundary: pd.Timestamp,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """Censor every trajectory whose intervening market path is unknown."""

        for action in list(self.active.values()):
            self._resolve_flat(
                action,
                _terminal_row(
                    action,
                    resolved_at=boundary,
                    outcome="data_gap_censored",
                    filled=action.entry_price is not None,
                    gross_R=None,
                    right_censored=True,
                ),
            )
        for trial in list(self.position_trials.values()):
            self._resolved_positions.append(
                _position_terminal_row(
                    trial,
                    resolved_at=boundary,
                    outcome="data_gap_censored",
                    gross_R=None,
                    right_censored=True,
                )
            )
            self.position_trials.pop(trial.position_action_key, None)
            self.resolved_position_actions += 1
        return self.drain()


def shadow_replay_code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


__all__ = [
    "FrozenShadowReplay",
    "POSITION_ACTIONS",
    "POSITION_FEATURE_NAMES",
    "POSITION_SAMPLING_STRIDE_MINUTES",
    "shadow_action_key",
    "shadow_replay_code_fingerprint",
]
