"""Stable physical schemas for v2.3 action-clock and episode Parquet streams."""
from __future__ import annotations

from smc_trader.action_clock import FEATURE_NAMES, FLAT_ACTIONS
from smc_trader.shadow_replay import POSITION_FEATURE_NAMES


S = "large_string"
F = "float64"
I = "int64"
B = "bool"
TNY = "timestamp_ny"
TUTC = "timestamp_utc"


DECISION_FIELDS = {
    "asof": TNY,
    "calibration_state_commitment": S,
    "symbol": S,
    "instrument_id": I,
    "model_action": S,
    "risk_action": S,
    "best_variant_action": S,
    "best_variant_utility_R": F,
    "best_variant_hypothesis_key": S,
    "unique_executable_candidate_plans": I,
    "active_flat_shadow_actions": I,
    "active_position_trials": I,
    "execution_source": S,
    "snapshot_full_hash_computed": B,
}

CANDIDATE_STATE_FIELDS = {
    "candidate_id": S,
    "decision_time": TNY,
    "plan_identity": S,
    "representative_hypothesis_key": S,
    "representative_playbook": S,
    "direction": S,
    "lineage_key": S,
    "setup_id": S,
    "phase": S,
    "phase_started_at": TNY,
    "sequence_state_json": S,
    "evidence_state_json": S,
    "observation_state_json": S,
    "event_memory_json": S,
    "planned_entry": F,
    "original_invalidation": F,
    "invalidation_source_id": S,
    "primary_target": F,
    "primary_target_id": S,
    "deadline": TNY,
    "initial_plan_observed_at": TNY,
    "initial_entry": F,
    "initial_invalidation": F,
    "initial_target": F,
    "risk_points": F,
    "risk_passed": B,
    "risk_final_action": S,
    "risk_vetoes_json": S,
    "risk_reasons_json": S,
    "execution_source": S,
    "available_actions_json": S,
}

CANDIDATE_ACTION_FIELDS = {
    "candidate_id": S,
    "action_key": S,
    "action_id": S,
    "decision_time": TNY,
    "representative_playbook": S,
    "direction": S,
    "risk_passed": B,
    "structural_plan_valid": B,
    "risk_vetoes_json": S,
    "execution_source": S,
    **{name: F for name in FEATURE_NAMES},
}

FLAT_OUTCOME_FIELDS = {
    "candidate_id": S,
    "action_key": S,
    "action_id": S,
    "decision_time": TNY,
    "resolved_at": TNY,
    "status": S,
    "outcome": S,
    "filled": B,
    "filled_at": TNY,
    "entry_price": F,
    "gross_R": F,
    "cost_R": F,
    "cost_source": S,
    "cost_observed_at": TNY,
    "net_R": F,
    "mfe_R": F,
    "mae_R": F,
    "bars_waited": I,
    "ambiguous_same_bar": B,
    "right_censored": B,
}

POSITION_ACTION_FIELDS = {
    "position_action_key": S,
    "position_state_id": S,
    "candidate_id": S,
    "parent_action_key": S,
    "action_id": S,
    "decision_time": TNY,
    "representative_playbook": S,
    "direction": S,
    "entry_price": F,
    "original_invalidation": F,
    "current_stop": F,
    "current_stop_source_id": S,
    "current_stop_activated_at": TNY,
    "applied_stop": F,
    "primary_target": F,
    "deadline": TNY,
    "mark_R": F,
    "parent_mfe_R": F,
    "parent_mae_R": F,
    "elapsed_minutes": I,
    "remaining_target_R": F,
    "stop_distance_R": F,
    "protection_available": B,
    "protection_source_id": S,
    "belief_state_json": S,
    "observation_state_json": S,
    "execution_source": S,
    **{name: F for name in POSITION_FEATURE_NAMES},
}

POSITION_OUTCOME_FIELDS = {
    "position_action_key": S,
    "candidate_id": S,
    "parent_action_key": S,
    "action_id": S,
    "decision_time": TNY,
    "resolved_at": TNY,
    "status": S,
    "outcome": S,
    "gross_R": F,
    "cost_R": F,
    "net_R": F,
    "mfe_R": F,
    "mae_R": F,
    "ambiguous_same_bar": B,
    "right_censored": B,
}

CALIBRATION_STREAM_FIELD_TYPES = {
    "decision_shards": DECISION_FIELDS,
    "candidate_state_shards": CANDIDATE_STATE_FIELDS,
    "candidate_action_shards": CANDIDATE_ACTION_FIELDS,
    "flat_outcome_shards": FLAT_OUTCOME_FIELDS,
    "position_action_shards": POSITION_ACTION_FIELDS,
    "position_outcome_shards": POSITION_OUTCOME_FIELDS,
}

_FLAT_OUTCOME_JOIN_FIELDS = {
    name: field_type
    for name, field_type in FLAT_OUTCOME_FIELDS.items()
    if name not in {"candidate_id", "action_key", "action_id", "decision_time"}
}
FLAT_EPISODE_FIELDS = {
    **CANDIDATE_ACTION_FIELDS,
    **_FLAT_OUTCOME_JOIN_FIELDS,
    "fill_label": B,
    "conditional_gross_R": F,
    "loss_label": B,
    "gross_action_utility_R": F,
    "net_action_utility_R": F,
    "fit_eligible_fill": B,
    "fit_eligible_conditional_gross": B,
    "fit_eligible_net": B,
    "fit_eligible_cost": B,
}

_SORTED_FLAT_ACTIONS = tuple(sorted(FLAT_ACTIONS))
DELTA_EPISODE_FIELDS = {
    "candidate_id": S,
    "decision_time": TNY,
    "representative_playbook": S,
    "direction": S,
    "risk_passed": B,
    "structural_plan_valid": B,
    "execution_source": S,
    **{name: F for name in FEATURE_NAMES},
    **{
        f"{family}__{action}": F
        for family in ("gross_action_utility_R", "net_action_utility_R")
        for action in _SORTED_FLAT_ACTIONS
    },
    "label_resolved_at": TUTC,
    **{
        f"{family}_delta_enter_vs_{alternative}_R": F
        for alternative in (
            "wait_one_bar",
            "wait_better_price",
            "wait_reacceptance",
            "abstain",
        )
        for family in ("gross", "net")
    },
}

_POSITION_OUTCOME_JOIN_FIELDS = {
    name: field_type
    for name, field_type in POSITION_OUTCOME_FIELDS.items()
    if name
    not in {
        "position_action_key",
        "candidate_id",
        "parent_action_key",
        "action_id",
        "decision_time",
    }
}
POSITION_EPISODE_FIELDS = {
    **POSITION_ACTION_FIELDS,
    **_POSITION_OUTCOME_JOIN_FIELDS,
    "fit_eligible_gross": B,
    "fit_eligible_net": B,
}

EPISODE_STREAM_FIELD_TYPES = {
    "flat_episode_shards": FLAT_EPISODE_FIELDS,
    "delta_episode_shards": DELTA_EPISODE_FIELDS,
    "position_episode_shards": POSITION_EPISODE_FIELDS,
}


__all__ = [
    "CALIBRATION_STREAM_FIELD_TYPES",
    "EPISODE_STREAM_FIELD_TYPES",
]
