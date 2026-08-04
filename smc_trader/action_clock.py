"""Causal action-clock candidates, timing primitives and flat replay.

This module contains feature-time contracts only.  Future bars are intentionally
absent; frozen shadow outcomes live in :mod:`smc_trader.shadow_replay`.

The action family remains the frozen v2.3 contract during
EXP-SMC-3.0.0-001.  A v3 engine may therefore reuse the action-clock machinery
only through :func:`build_action_clock_engine`; the legacy v2.3 builder keeps
its stricter version guard for backward compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .action_equivalence import (
    ActionEquivalenceDecisionLayer,
    ActionEquivalenceProtocol,
    ActionPlanIdentity,
    EquivalentActionGroup,
    action_plan_identity,
)
from .engine import ContinuousSMCEngine
from .model import (
    AccountState,
    Action,
    ActionUtility,
    Bar,
    Decision,
    Direction,
    EngineSnapshot,
    FrozenThesis,
    HypothesisBelief,
    MarketBelief,
    MarketObservation,
    Playbook,
    PlaybookPhase,
    RiskAssessment,
    Timeframe,
    TradePlan,
    VetoCode,
    content_hash,
    to_primitive,
)
from .observation import ExecutionRealityInput


FLAT_ACTIONS = (
    "enter_now",
    "wait_one_bar",
    "wait_better_price",
    "wait_reacceptance",
    "abstain",
)

CONTINUOUS_FEATURE_NAMES = (
    "belief_max",
    "belief_second",
    "belief_dispersion",
    "equivalent_evidence_count",
    "representative_raw_probability",
    "calibration_gap",
    "uncertainty",
    "phase_age_scaled",
    "sequence_completion_fraction",
    "minutes_since_sequence_complete_scaled",
    "entry_drift_R",
    "invalidation_drift_R",
    "target_drift_R",
    "primary_target_R_capped",
    "remaining_path_R_capped",
    "minutes_to_deadline_scaled",
    "h4_directional_displacement_aligned",
    "h4_path_efficiency",
    "h4_structure_direction_aligned",
    "h4_structure_age_scaled",
    "h4_range_position_aligned",
    "h4_external_draw_distance_atr",
    "h1_swing_progression_aligned",
    "h1_acceptance_aligned",
    "h1_rejection_aligned",
    "h1_dealing_range_position_aligned",
    "h1_path_obstruction_atr",
    "m5_impulse_direction_aligned",
    "m5_impulse_strength",
    "m5_impulse_age_scaled",
    "m5_impulse_extension_atr",
    "extension_to_remaining_draw",
    "m5_pullback_depth",
    "m5_pullback_completeness",
    "m5_reacceptance_aligned",
    "m5_compression",
    "first_pullback_quality",
    "m1_path_sequence_aligned",
    "m1_acceleration_aligned",
    "m1_counter_pressure_against",
    "m1_trigger_hold_aligned",
    "m1_trigger_age_scaled",
    "swing_trigger_consistency",
    "spread_ticks",
    "expected_round_trip_cost_R",
    "fillability",
    "execution_data_age_scaled",
    "depth_imbalance_aligned",
    "mbo_available",
)

CATEGORICAL_FEATURE_NAMES = (
    "playbook_dfp",
    "playbook_lsr",
    "playbook_favr",
    "direction_long",
    "action_enter_now",
    "action_wait_one_bar",
    "action_wait_better_price",
    "action_wait_reacceptance",
    "action_abstain",
)

FEATURE_NAMES = CONTINUOUS_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES


@dataclass(frozen=True)
class ActionClockProtocol:
    version: str
    fingerprint: str
    status: str
    regularization_lambda: float

    @classmethod
    def from_file(
        cls,
        path: str | Path = "configs/action_clock_value_protocol_v2_3.json",
    ) -> "ActionClockProtocol":
        source = Path(path)
        if not source.is_absolute() and not source.exists():
            source = Path(__file__).resolve().parents[1] / source
        raw = source.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("action-clock protocol root must be an object")
        version = str(payload.get("protocol_version", ""))
        status = str(payload.get("status", ""))
        if (
            not version.startswith("2.3.")
            or status != "preregistered_before_v2_3_sample_generation"
        ):
            raise ValueError("action-clock protocol is not frozen for v2.3")
        feature_payload = payload.get("features")
        if not isinstance(feature_payload, Mapping):
            raise ValueError("action-clock protocol omits feature contract")
        if tuple(feature_payload.get("continuous_fixed_order", ())) != (
            CONTINUOUS_FEATURE_NAMES
        ):
            raise ValueError("action-clock continuous feature order changed")
        if tuple(feature_payload.get("categorical_fixed_order", ())) != (
            CATEGORICAL_FEATURE_NAMES
        ):
            raise ValueError("action-clock categorical feature order changed")
        from .shadow_replay import POSITION_FEATURE_NAMES

        if tuple(feature_payload.get("position_fixed_order", ())) != (
            POSITION_FEATURE_NAMES
        ):
            raise ValueError("position action feature order changed")
        flat_payload = payload.get("flat_actions")
        if not isinstance(flat_payload, Mapping):
            raise ValueError("action-clock protocol omits flat actions")
        actions = tuple(
            str(item.get("id", ""))
            for item in flat_payload.get("actions", ())
            if isinstance(item, Mapping)
        )
        if actions != FLAT_ACTIONS:
            raise ValueError("action-clock flat action family changed")
        fit = payload.get("fit")
        if not isinstance(fit, Mapping):
            raise ValueError("action-clock protocol omits fit contract")
        penalty = float(fit.get("regularization_lambda", 0.0))
        if not math.isfinite(penalty) or penalty <= 0:
            raise ValueError("action-clock regularization must be positive")
        return cls(
            version=version,
            fingerprint=hashlib.sha256(raw).hexdigest(),
            status=status,
            regularization_lambda=penalty,
        )


@dataclass(frozen=True)
class InitialPlanState:
    lineage_key: str
    observed_at: pd.Timestamp
    plan: TradePlan


class PlanLineageStore:
    """Remember the first causal plan for each hypothesis setup."""

    def __init__(self) -> None:
        self._initial: dict[str, InitialPlanState] = {}
        self._total_observed = 0

    @staticmethod
    def lineage_key(hypothesis: HypothesisBelief) -> str | None:
        sequence = hypothesis.sequence
        if sequence is None or sequence.setup_id is None:
            return None
        return f"{hypothesis.key}|{sequence.setup_id}"

    def observe(self, belief: MarketBelief) -> None:
        active_keys: set[str] = set()
        for hypothesis in belief.hypotheses.values():
            key = self.lineage_key(hypothesis)
            if key is None:
                continue
            active_keys.add(key)
            if hypothesis.plan is None or key in self._initial:
                continue
            self._initial[key] = InitialPlanState(
                lineage_key=key,
                observed_at=belief.asof,
                plan=hypothesis.plan,
            )
            self._total_observed += 1
        for stale in set(self._initial).difference(active_keys):
            self._initial.pop(stale)

    def reset(self) -> None:
        self._initial.clear()

    def initial_for(self, hypothesis: HypothesisBelief) -> InitialPlanState:
        key = self.lineage_key(hypothesis)
        if key is None or hypothesis.plan is None:
            raise ValueError("candidate has no causal setup lineage and plan")
        initial = self._initial.get(key)
        if initial is None:
            raise RuntimeError(
                "plan lineage must be observed before candidate enumeration"
            )
        return initial

    @property
    def size(self) -> int:
        return len(self._initial)

    @property
    def total_observed(self) -> int:
        return self._total_observed


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    decision_time: pd.Timestamp
    action_identity: ActionPlanIdentity
    group: EquivalentActionGroup
    representative: HypothesisBelief
    initial_plan: InitialPlanState
    risk: RiskAssessment
    features: Mapping[str, float]

    @property
    def plan(self) -> TradePlan:
        plan = self.representative.plan
        if plan is None:
            raise RuntimeError("candidate representative lost its plan")
        return plan


def _finite(value: Any, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _bounded(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, _finite(value))))


def _mapped_support(value: float) -> float:
    return _bounded((value + 1.0) / 2.0, 0.0, 1.0)


def _minutes(left: pd.Timestamp, right: pd.Timestamp) -> float:
    return max(0.0, (left - right).total_seconds() / 60.0)


def _sequence_fields(
    hypothesis: HypothesisBelief,
    asof: pd.Timestamp,
) -> tuple[float, float]:
    sequence = hypothesis.sequence
    if sequence is None or not sequence.steps:
        return 0.0, 0.0
    fraction = sequence.completed_steps / len(sequence.steps)
    if not sequence.complete:
        return float(fraction), 0.0
    clocks = [
        step.observed_at for step in sequence.steps if step.observed_at is not None
    ]
    completed_at = max(clocks) if clocks else asof
    return float(fraction), min(1.0, _minutes(asof, completed_at) / 240.0)


def _evidence_probabilities(
    group: EquivalentActionGroup,
    belief: MarketBelief,
) -> tuple[float, float, float]:
    probabilities = sorted(
        (float(belief.hypotheses[key].probability) for key in group.hypothesis_keys),
        reverse=True,
    )
    maximum = probabilities[0]
    second = probabilities[1] if len(probabilities) > 1 else 0.0
    minimum = probabilities[-1]
    return maximum, second, maximum - minimum


def _playbook_indicators(
    group: EquivalentActionGroup,
    belief: MarketBelief,
) -> tuple[float, float, float]:
    values = {belief.hypotheses[key].playbook for key in group.hypothesis_keys}
    return (
        1.0 if Playbook.DISPLACEMENT_FIRST_PULLBACK in values else 0.0,
        1.0 if Playbook.LIQUIDITY_SWEEP_REVERSAL in values else 0.0,
        1.0 if Playbook.FAILED_AUCTION_VALUE_RETURN in values else 0.0,
    )


def action_clock_features(
    group: EquivalentActionGroup,
    belief: MarketBelief,
    observation: MarketObservation,
    initial: InitialPlanState,
    *,
    action_id: str,
    tick_size: float = 0.25,
) -> dict[str, float]:
    """Extract the preregistered causal feature vector for one action."""

    if action_id not in FLAT_ACTIONS:
        raise ValueError(f"unregistered flat action: {action_id}")
    representative_key = group.representative.hypothesis_key
    if representative_key is None:
        raise ValueError("candidate group lacks representative hypothesis")
    hypothesis = belief.hypotheses.get(representative_key)
    if hypothesis is None or hypothesis.plan is None:
        raise ValueError("candidate representative has no plan")
    plan = hypothesis.plan
    if plan.direction is not initial.plan.direction:
        raise ValueError("plan lineage changed direction")
    sign = plan.direction.sign
    risk = float(initial.plan.risk_points)
    if not math.isfinite(risk) or risk <= 0:
        raise ValueError("initial plan has invalid risk")
    h4 = observation.frame(Timeframe.H4).metrics
    h1 = observation.frame(Timeframe.H1).metrics
    m5 = observation.frame(Timeframe.M5).metrics
    m1 = observation.frame(Timeframe.M1).metrics
    maximum, second, dispersion = _evidence_probabilities(group, belief)
    raw = float(
        hypothesis.probability
        if hypothesis.raw_probability is None
        else hypothesis.raw_probability
    )
    sequence_fraction, sequence_age = _sequence_fields(
        hypothesis,
        observation.asof,
    )
    target_distance = max(
        0.0,
        sign * (plan.targets[0].price - observation.price),
    )
    true_remaining_R = target_distance / risk
    m5_atr = max(1e-9, _finite(m5.get("atr"), default=1.0))
    remaining_draw_atr = max(0.25, target_distance / m5_atr)
    impulse_extension = max(0.0, _finite(m5.get("impulse_extension_atr")))
    extension_ratio = min(10.0, impulse_extension / remaining_draw_atr)
    swing_aligned = sign * _finite(h1.get("swing_progression"))
    path_aligned = sign * _finite(m1.get("path_sequence"))
    trigger_aligned = sign * _finite(m1.get("trigger_hold_direction"))
    consistency = min(
        _mapped_support(swing_aligned),
        _mapped_support(path_aligned),
        _mapped_support(trigger_aligned),
    )
    reacceptance_aligned = sign * _finite(m5.get("reacceptance_direction"))
    counter_aligned = sign * _finite(m1.get("counter_pressure"))
    counter_against = max(0.0, -counter_aligned)
    depth = _bounded(_finite(m5.get("pullback_depth")), 0.0, 1.0)
    depth_quality = _bounded(1.0 - abs(depth - 0.5) / 0.5, 0.0, 1.0)
    completeness = _bounded(
        _finite(m5.get("pullback_completeness")),
        0.0,
        1.0,
    )
    reacceptance_quality = _mapped_support(reacceptance_aligned)
    pressure_quality = _bounded(
        (1.0 - _bounded(_finite(m5.get("compression")), 0.0, 1.0))
        * (1.0 - counter_against),
        0.0,
        1.0,
    )
    pullback_quality = float(
        (
            depth_quality
            * completeness
            * reacceptance_quality
            * pressure_quality
        )
        ** 0.25
    )
    execution = observation.execution
    mbo_available = 1.0 if execution.source == "mbo_reconstructed" else 0.0
    if mbo_available:
        spread_ticks = _finite(execution.spread_points) / tick_size
        cost_R = _finite(execution.expected_round_trip_cost_points) / risk
        fillability = _bounded(_finite(execution.fillability), 0.0, 1.0)
        depth_imbalance = _finite(execution.depth_imbalance)
    else:
        # Missing execution authority is represented by its explicit indicator
        # and stale age. Observer fallbacks must not become pseudo-MBO values.
        spread_ticks = 0.0
        cost_R = 0.0
        fillability = 0.0
        depth_imbalance = 0.0
    directional_range_h4 = sign * (
        2.0 * _bounded(_finite(h4.get("range_position"), default=0.5), 0.0, 1.0)
        - 1.0
    )
    directional_range_h1 = sign * (
        2.0
        * _bounded(
            _finite(h1.get("dealing_range_position"), default=0.5),
            0.0,
            1.0,
        )
        - 1.0
    )
    h4_draw = (
        _finite(h4.get("external_above_distance_atr"), default=99.0)
        if sign > 0
        else _finite(h4.get("external_below_distance_atr"), default=99.0)
    )
    h1_obstruction = (
        _finite(h1.get("up_path_obstruction_atr"), default=99.0)
        if sign > 0
        else _finite(h1.get("down_path_obstruction_atr"), default=99.0)
    )
    playbook_dfp, playbook_lsr, playbook_favr = _playbook_indicators(
        group,
        belief,
    )
    continuous = (
        maximum,
        second,
        dispersion,
        float(len(group.hypothesis_keys)),
        raw,
        float(hypothesis.probability) - raw,
        float(hypothesis.uncertainty),
        min(1.0, _minutes(observation.asof, hypothesis.phase_started_at) / 240.0),
        sequence_fraction,
        sequence_age,
        sign * (plan.planned_entry - initial.plan.planned_entry) / risk,
        sign
        * (plan.invalidation.price - initial.plan.invalidation.price)
        / risk,
        sign * (plan.targets[0].price - initial.plan.targets[0].price) / risk,
        min(5.0, max(0.0, float(plan.primary_target_R))),
        min(5.0, true_remaining_R),
        min(1.0, _minutes(plan.deadline, observation.asof) / 240.0),
        sign * _finite(h4.get("directional_displacement")),
        _bounded(_finite(h4.get("path_efficiency")), 0.0, 1.0),
        sign * _finite(h4.get("structure_direction")),
        min(1.0, max(0.0, _finite(h4.get("structure_age_bars"))) / 80.0),
        directional_range_h4,
        min(10.0, max(0.0, h4_draw)),
        swing_aligned,
        sign * _finite(h1.get("acceptance_direction")),
        sign * _finite(h1.get("rejection_direction")),
        directional_range_h1,
        min(10.0, max(0.0, h1_obstruction)),
        sign * _finite(m5.get("impulse_direction")),
        _bounded(_finite(m5.get("impulse_strength")), 0.0, 1.0),
        min(1.0, max(0.0, _finite(m5.get("impulse_age_bars"))) / 12.0),
        min(10.0, impulse_extension),
        extension_ratio,
        depth,
        completeness,
        reacceptance_aligned,
        _bounded(_finite(m5.get("compression")), 0.0, 1.0),
        pullback_quality,
        path_aligned,
        sign * _finite(m1.get("acceleration")),
        counter_against,
        trigger_aligned,
        min(1.0, max(0.0, _finite(m1.get("trigger_age_bars"))) / 30.0),
        consistency,
        min(20.0, max(0.0, spread_ticks)),
        min(5.0, max(0.0, cost_R)),
        fillability,
        min(1.0, max(0.0, _finite(execution.data_age_seconds)) / 61.0),
        sign * _bounded(depth_imbalance, -1.0, 1.0),
        mbo_available,
    )
    categorical = (
        playbook_dfp,
        playbook_lsr,
        playbook_favr,
        1.0 if plan.direction is Direction.LONG else 0.0,
        *(1.0 if action_id == registered else 0.0 for registered in FLAT_ACTIONS),
    )
    values = continuous + categorical
    if len(values) != len(FEATURE_NAMES):
        raise AssertionError("action-clock feature width is inconsistent")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("action-clock features contain non-finite values")
    return dict(zip(FEATURE_NAMES, (float(value) for value in values)))


def _candidate_risk(
    engine: ContinuousSMCEngine,
    observation: MarketObservation,
    hypothesis: HypothesisBelief,
    utility: ActionUtility,
    *,
    account: AccountState,
) -> RiskAssessment:
    if hypothesis.plan is None:
        raise ValueError("candidate risk review requires a plan")
    decision = Decision(
        asof=observation.asof,
        selected_action=Action.ENTER,
        utilities=(utility,),
        best_hypothesis_key=hypothesis.key,
        advantage=max(0.0, float(utility.utility)),
        reasons=("v2.3 independent candidate hard-risk review",),
        plan=hypothesis.plan,
    )
    return engine.risk.review(decision, observation, account)


def executable_candidate_groups(
    snapshot: EngineSnapshot,
    protocol: ActionEquivalenceProtocol,
    lineage: PlanLineageStore,
    *,
    engine: ContinuousSMCEngine,
    disabled_playbooks: Sequence[Playbook] = (),
    account: AccountState | None = None,
) -> tuple[CandidatePlan, ...]:
    """Enumerate all executable plans, independent of the selected action."""

    disabled = frozenset(disabled_playbooks)
    lineage.observe(snapshot.belief)
    grouped: dict[ActionPlanIdentity, list[str]] = {}
    for key, hypothesis in snapshot.belief.hypotheses.items():
        if (
            hypothesis.playbook in disabled
            or hypothesis.phase is not PlaybookPhase.EXECUTABLE
            or hypothesis.plan is None
        ):
            continue
        identity = action_plan_identity(
            Action.ENTER,
            hypothesis.plan,
            tick_size=protocol.tick_size,
        )
        grouped.setdefault(identity, []).append(key)
    utility_by_key = {
        utility.hypothesis_key: utility
        for utility in snapshot.decision.utilities
        if utility.action is Action.ENTER and utility.hypothesis_key is not None
    }
    flat_account = account or AccountState(equity=100_000.0)
    output: list[CandidatePlan] = []
    for identity, keys in sorted(grouped.items(), key=lambda item: item[0].key):
        ordered_keys = sorted(
            keys,
            key=lambda key: (
                -float(snapshot.belief.hypotheses[key].probability),
                float(snapshot.belief.hypotheses[key].uncertainty),
                key,
            ),
        )
        representative_key = next(
            (key for key in ordered_keys if key in utility_by_key),
            ordered_keys[0],
        )
        hypothesis = snapshot.belief.hypotheses[representative_key]
        utility = utility_by_key.get(
            representative_key,
            ActionUtility(
                action=Action.ENTER,
                utility=0.0,
                components={"candidate_enumeration": 0.0},
                hypothesis_key=representative_key,
                reason="all-plan candidate enumeration independent of live selection",
            ),
        )
        probabilities = tuple(
            sorted(
                (float(snapshot.belief.hypotheses[key].probability) for key in keys),
                reverse=True,
            )
        )
        group = EquivalentActionGroup(
            identity=identity,
            representative=utility,
            hypothesis_keys=tuple(sorted(keys)),
            probabilities=probabilities,
        )
        initial = lineage.initial_for(hypothesis)
        features = action_clock_features(
            group,
            snapshot.belief,
            snapshot.observation,
            initial,
            action_id="enter_now",
            tick_size=protocol.tick_size,
        )
        risk = _candidate_risk(
            engine,
            snapshot.observation,
            hypothesis,
            utility,
            account=flat_account,
        )
        candidate_id = content_hash(
            {
                "protocol": "v2.3-action-clock",
                "decision_time": snapshot.observation.asof,
                "plan_identity": identity.key,
            }
        )
        output.append(
            CandidatePlan(
                candidate_id=candidate_id,
                decision_time=snapshot.observation.asof,
                action_identity=identity,
                group=group,
                representative=hypothesis,
                initial_plan=initial,
                risk=risk,
                features=features,
            )
        )
    return tuple(output)


def freeze_candidate(candidate: CandidatePlan) -> FrozenThesis:
    plan = candidate.plan
    payload: dict[str, Any] = {
        "created_at": candidate.decision_time,
        "playbook": plan.playbook,
        "direction": plan.direction,
        "entry": plan.planned_entry,
        "original_invalidation": plan.invalidation,
        "original_targets": plan.targets,
        "deadline": plan.deadline,
        "setup_id": plan.setup_id,
        "entry_location_id": plan.entry_location_id,
        "entry_path_id": plan.entry_path_id,
        "draw_selection": plan.draw_selection,
        "range_auction": plan.range_auction,
        "liquidity_route": plan.liquidity_route,
    }
    return FrozenThesis(
        thesis_hash=content_hash(payload),
        created_at=candidate.decision_time,
        playbook=plan.playbook,
        direction=plan.direction,
        entry=plan.planned_entry,
        original_invalidation=plan.invalidation,
        original_targets=plan.targets,
        deadline=plan.deadline,
        setup_id=plan.setup_id,
        entry_location_id=plan.entry_location_id,
        entry_path_id=plan.entry_path_id,
        draw_selection=plan.draw_selection,
        range_auction=plan.range_auction,
        liquidity_route=plan.liquidity_route,
    )


def candidate_row(
    candidate: CandidatePlan,
    snapshot: EngineSnapshot,
    *,
    action_id: str,
    tick_size: float = 0.25,
) -> dict[str, Any]:
    """Serialize one action alternative without future or outcome fields."""

    hypothesis = candidate.representative
    plan = candidate.plan
    features = action_clock_features(
        candidate.group,
        snapshot.belief,
        snapshot.observation,
        candidate.initial_plan,
        action_id=action_id,
        tick_size=tick_size,
    )
    sequence = hypothesis.sequence
    evidence_state = {
        key: {
            "playbook": snapshot.belief.hypotheses[key].playbook.value,
            "direction": snapshot.belief.hypotheses[key].direction.value,
            "probability": float(snapshot.belief.hypotheses[key].probability),
            "raw_probability": float(
                snapshot.belief.hypotheses[key].probability
                if snapshot.belief.hypotheses[key].raw_probability is None
                else snapshot.belief.hypotheses[key].raw_probability
            ),
            "uncertainty": float(snapshot.belief.hypotheses[key].uncertainty),
            "phase": snapshot.belief.hypotheses[key].phase.value,
        }
        for key in candidate.group.hypothesis_keys
    }
    frame_state = {
        timeframe.value: {
            "cutoff": snapshot.observation.frame(timeframe).cutoff.isoformat(),
            "ready": snapshot.observation.frame(timeframe).ready,
            "metrics": dict(snapshot.observation.frame(timeframe).metrics),
        }
        for timeframe in snapshot.observation.active_timeframes
    }
    recent_events = [
        to_primitive(event)
        for event in snapshot.observation.recent_events[-32:]
        if event.observed_at <= snapshot.observation.asof
    ]
    action_key = hashlib.sha256(
        f"{candidate.candidate_id}|{action_id}".encode("utf-8")
    ).hexdigest()
    return {
        "candidate_id": candidate.candidate_id,
        "action_key": action_key,
        "action_id": action_id,
        "decision_time": candidate.decision_time,
        "plan_identity": candidate.action_identity.key,
        "representative_hypothesis_key": hypothesis.key,
        "representative_playbook": hypothesis.playbook.value,
        "direction": plan.direction.value,
        "lineage_key": candidate.initial_plan.lineage_key,
        "setup_id": None if sequence is None else sequence.setup_id,
        "phase": hypothesis.phase.value,
        "phase_started_at": hypothesis.phase_started_at,
        "sequence_state_json": json.dumps(
            None if sequence is None else to_primitive(sequence),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "evidence_state_json": json.dumps(
            evidence_state,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "observation_state_json": json.dumps(
            frame_state,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "event_memory_json": json.dumps(
            {
                "recent_events": recent_events,
                "durations_minutes": dict(
                    snapshot.observation.event_durations_minutes
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "planned_entry": plan.planned_entry,
        "original_invalidation": plan.invalidation.price,
        "invalidation_source_id": plan.invalidation.source_level_id,
        "primary_target": plan.targets[0].price,
        "primary_target_id": plan.targets[0].level_id,
        "deadline": plan.deadline,
        "initial_plan_observed_at": candidate.initial_plan.observed_at,
        "initial_entry": candidate.initial_plan.plan.planned_entry,
        "initial_invalidation": candidate.initial_plan.plan.invalidation.price,
        "initial_target": candidate.initial_plan.plan.targets[0].price,
        "risk_points": plan.risk_points,
        "risk_passed": bool(candidate.risk.passed),
        "risk_final_action": candidate.risk.final_action.value,
        "risk_vetoes_json": json.dumps(
            [item.value for item in candidate.risk.vetoes],
            sort_keys=True,
        ),
        "risk_reasons_json": json.dumps(list(candidate.risk.reasons)),
        "execution_source": snapshot.observation.execution.source,
        **features,
    }


def candidate_state_row(
    candidate: CandidatePlan,
    snapshot: EngineSnapshot,
    *,
    tick_size: float = 0.25,
) -> dict[str, Any]:
    """One full causal state row per candidate plan."""

    row = candidate_row(
        candidate,
        snapshot,
        action_id="enter_now",
        tick_size=tick_size,
    )
    row.pop("action_key")
    row.pop("action_id")
    for name in FEATURE_NAMES:
        row.pop(name)
    row["available_actions_json"] = json.dumps(list(FLAT_ACTIONS))
    return row


def candidate_action_row(
    candidate: CandidatePlan,
    snapshot: EngineSnapshot,
    *,
    action_id: str,
    tick_size: float = 0.25,
) -> dict[str, Any]:
    """Lean model row; full observations remain in the candidate-state stream."""

    row = candidate_row(
        candidate,
        snapshot,
        action_id=action_id,
        tick_size=tick_size,
    )
    structural_vetoes = {
        VetoCode.INVALID_STOP,
        VetoCode.INVALID_TARGET,
        VetoCode.NO_PLAN,
    }
    structural_plan_valid = not any(
        veto in structural_vetoes for veto in candidate.risk.vetoes
    )
    return {
        "candidate_id": row["candidate_id"],
        "action_key": row["action_key"],
        "action_id": row["action_id"],
        "decision_time": row["decision_time"],
        "representative_playbook": row["representative_playbook"],
        "direction": row["direction"],
        "risk_passed": row["risk_passed"],
        "structural_plan_valid": structural_plan_valid,
        "risk_vetoes_json": row["risk_vetoes_json"],
        "execution_source": row["execution_source"],
        **{name: row[name] for name in FEATURE_NAMES},
    }


class ActionClockReplay:
    """Always-flat causal replay with a lightweight rolling commitment."""

    def __init__(
        self,
        engine: ContinuousSMCEngine,
        *,
        rolling_commitment: str = "0" * 64,
    ) -> None:
        if len(rolling_commitment) != 64:
            raise ValueError("rolling commitment must be a 64-character digest")
        self.engine = engine
        self.rolling_commitment = rolling_commitment

    def on_bar(
        self,
        bar: Bar,
        *,
        execution: ExecutionRealityInput,
        belief_enabled: bool = True,
    ) -> EngineSnapshot:
        update = self.engine.reader.on_bar(bar)
        observation = self.engine.observer.observe(update, execution)
        if {
            "contract_change_history_reset",
            "data_gap_history_reset",
        }.intersection(observation.anomalies):
            self.engine.brain.reset()
        if belief_enabled:
            belief = self.engine.brain.update(
                observation,
                position=None,
                scene_graph=self.engine.observer.scene_graph,
                scene_delta=self.engine.observer.last_scene_delta,
            )
        else:
            # Reader/observer warm-up before a fitted calibrator becomes
            # temporally available must not seed the recursive belief state.
            self.engine.brain.reset()
            belief = MarketBelief(asof=observation.asof, hypotheses={})
        account = AccountState(equity=100_000.0)
        decision = self.engine.decision.decide(observation, belief, account)
        risk = self.engine.risk.review(decision, observation, account)
        digest = hashlib.sha256()
        for field in (
            self.rolling_commitment,
            observation.asof.isoformat(),
            observation.symbol,
            str(observation.instrument_id),
            decision.selected_action.value,
            risk.final_action.value,
            decision.best_hypothesis_key or "",
        ):
            encoded = field.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        self.rolling_commitment = digest.hexdigest()
        snapshot = EngineSnapshot(
            observation=observation,
            belief=belief,
            decision=decision,
            risk=risk,
            snapshot_hash=self.rolling_commitment,
        )
        self.engine._last_snapshot = snapshot
        return snapshot


def build_action_clock_engine(
    config_path: str | Path,
    *,
    disabled_playbooks: Sequence[Playbook] = (),
) -> ContinuousSMCEngine:
    """Build an unfitted action-clock engine from a governed model config.

    The model version is deliberately not hard-coded here.  The referenced
    action-clock protocol still validates the exact frozen action and feature
    contract, while the caller is responsible for enforcing the experiment
    and data-window policy appropriate to the model version.
    """

    source = Path(config_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not str(payload.get("version", "")).strip():
        raise ValueError("action-clock engine requires a versioned config")
    if payload.get("action_clock_value_artifact") is not None:
        raise ValueError("candidate generation requires an unfitted base config")
    protocol = ActionEquivalenceProtocol.from_file(
        payload.get(
            "action_equivalence_protocol",
            "configs/action_equivalence_v2_2.json",
        )
    )
    ActionClockProtocol.from_file(
        payload.get(
            "action_clock_value_protocol",
            "configs/action_clock_value_protocol_v2_3.json",
        )
    )
    engine = ContinuousSMCEngine.from_config(source)
    engine.decision = ActionEquivalenceDecisionLayer(
        protocol,
        engine.decision.config,
        disabled_playbooks=disabled_playbooks,
    )
    return engine


def build_v2_3_action_clock_engine(
    config_path: str | Path = "configs/model_v2_3_action_clock_base.json",
    *,
    disabled_playbooks: Sequence[Playbook] = (),
) -> ContinuousSMCEngine:
    """Backward-compatible v2.3 builder with the original version guard."""

    source = Path(config_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not str(payload.get("version", "")).startswith("2.3."):
        raise ValueError("action-clock engine requires a v2.3 config")
    return build_action_clock_engine(
        source,
        disabled_playbooks=disabled_playbooks,
    )


def action_clock_code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


__all__ = [
    "ActionClockProtocol",
    "ActionClockReplay",
    "CATEGORICAL_FEATURE_NAMES",
    "CONTINUOUS_FEATURE_NAMES",
    "CandidatePlan",
    "FEATURE_NAMES",
    "FLAT_ACTIONS",
    "InitialPlanState",
    "PlanLineageStore",
    "action_clock_code_fingerprint",
    "action_clock_features",
    "build_action_clock_engine",
    "build_v2_3_action_clock_engine",
    "candidate_action_row",
    "candidate_row",
    "candidate_state_row",
    "executable_candidate_groups",
    "freeze_candidate",
]
