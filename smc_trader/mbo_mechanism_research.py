"""Preregistered Phase-6 MBO mechanism research primitives.

The module deliberately sits between the Trading Eye and later Brain work:
it may test whether immutable OHLCV semantics have a repeatable order-flow
signature, but it does not create a semantic event, update a market state, or
authorize a trading belief.  Post-event features are explicitly retrospective
and can never be consumed as evidence at the semantic event's ``known_at``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .signal_research import (
    ControlDirectionPolicy,
    MatchResult,
    MatchSpec,
    MatchedControlPair,
    ResearchContractError,
    canonical_treatment_episodes,
    deterministic_maximum_cardinality_match,
    holm_adjust_fixed_family,
    sha256_file,
)


PHASE6_PROTOCOL_VERSION = 1
PHASE6_EXECUTABLE_STATUS = (
    "frozen_mbo_development_mechanism_validation_not_oos_not_trading_authority"
)
PHASE6_FIXED_FAMILY = (
    "sweep_rejection",
    "acceptance_continuation",
    "displacement_impact",
    "mss_flow_shift",
    "fvg_retest_response",
)
STRICT_PRIOR_CONTEXT_UNAVAILABLE_FIELD_POLICY = MappingProxyType(
    {
        "atr": "censor_if_none_nonfinite_or_less_than_or_equal_to_zero",
        "relative_volume": (
            "censor_if_none_nonfinite_or_less_than_zero_zero_is_valid"
        ),
        "trend_direction": "none_maps_to_trend_neutral",
    }
)
STRICT_PRIOR_CONTEXT_CENSOR_REASONS = (
    "strict_prior_real_completed_m1_context_missing",
    "strict_prior_real_completed_m1_context_atr_unavailable",
    "strict_prior_real_completed_m1_context_relative_volume_unavailable",
)
LEGACY_SYNTHETIC_SEMANTIC_EXCEPTION_POLICY = {
    "default_policy": "forbid_semantic_atomic_on_synthetic_m1_clock",
    "registered_allowlist": [
        {
            "kind": "displacement_observed",
            "timeframe": "5m",
            "lifecycle": "censored",
            "terminal_reason": "synthetic_interruption",
        }
    ],
    "registered_clock_policy": "known_at_must_equal_registered_synthetic_clock",
    "provenance_policy": {
        "current_synthetic_m1_root": "exact_clock_normalized_BAR_clock_only",
        "synthetic_root_namespace": "context_event_ids_only",
        "source_event_ids": "recursive_normalized_BAR_roots_all_real_completed",
        "time_order": "event_time_lte_last_real_source_known_at_lt_known_at",
    },
    "source_data_policy": {
        "terminal_source_data_ids": (
            "producer_order_unique_detector_candle_ids_set_equal_canonical_sorted_"
            "recursive_real_M5_detector_union"
        ),
        "synthetic_context_identifiers": (
            "clock_root_source_data_ids_and_detector_candle_id_must_be_nonempty_"
            "and_disjoint_from_terminal"
        ),
        "missing_extra_or_duplicate_terminal_detector_ids": "fail_closed",
    },
    "descendant_sample_exclusion": {
        "blocked_identities": (
            "allowed_terminal_event_ids_plus_their_synthetic_context_root_ids"
        ),
        "lineage_scan": (
            "recursive_explicit_semantic_atomic_source_plus_context;"
            "non_atomic_origins_terminal"
        ),
        "apply_before": [
            "fvg_creation",
            "treatment",
            "raw_control",
            "M5_candidate_base",
        ],
        "formation_clock_lineage": "source_only_M5_ancestry_unchanged",
        "disposition": "exclude_with_reason_and_blocked_event_ids",
    },
    "analysis_policy": (
        "always_exclude_allowed_terminal_from_treatment_control_and_fvg_creation"
    ),
    "audit_policy": (
        "record_allowed_and_rejected_ids_plus_recursive_DAG_sha256_and_node_count"
    ),
}
SYNTHETIC_SEMANTIC_EXCEPTION_POLICY = {
    **LEGACY_SYNTHETIC_SEMANTIC_EXCEPTION_POLICY,
    "registered_clock_policy": (
        "context_root_clocks_must_exactly_equal_all_registered_synthetic_"
        "clocks_inside_the_open_closed_terminal_M5_interval"
    ),
    "provenance_policy": {
        "synthetic_m1_roots": (
            "all_exact_clock_only_M1_BAR_roots_in_open_closed_terminal_M5_interval"
        ),
        "constituent_clock_contract": (
            "exactly_five_unique_contiguous_M1_roots_in_the_terminal_M5_interval"
        ),
        "synthetic_root_namespace": "context_event_ids_only",
        "source_event_ids": "recursive_normalized_BAR_roots_all_real_completed",
        "time_order": (
            "event_time_lte_last_real_source_known_at_lt_terminal_known_at;"
            "context_root_known_at_lte_terminal_known_at"
        ),
    },
    "source_data_policy": {
        "terminal_source_data_ids": (
            "producer_order_unique_detector_candle_ids_set_equal_canonical_sorted_"
            "recursive_real_M5_detector_union"
        ),
        "synthetic_context_identifiers": (
            "every_context_root_source_data_ids_and_detector_candle_id_must_be_"
            "nonempty_and_disjoint_from_terminal"
        ),
        "missing_extra_or_duplicate_terminal_detector_ids": "fail_closed",
    },
    "audit_policy": (
        "record_allowed_and_rejected_ids_plus_all_synthetic_context_roots_"
        "and_recursive_DAG_sha256_and_node_count"
    ),
}
EXTENSION_WARMUP_SYNTHETIC_EXCEPTION_POLICY = {
    "source": "hash_bound_week1_gate_result_only",
    "prior_clock_registration": (
        "exact_prior_reader_synthetic_clock_with_allowed_audit"
    ),
    "replay_verification": (
        "current_event_and_DAG_legacy_projection_must_equal_hash_bound_prior_audit"
    ),
    "blocked_lineage_import": (
        "prior_allowed_terminal_and_synthetic_context_root_ids"
    ),
    "active_reader_census_scope": "active_extension_window_only",
    "unregistered_or_missing_prior_exception": "fail_closed",
}
EXTENSION_DISPLACEMENT_MONOTONICITY_POLICY = (
    "report_hash_bound_prior_week1_and_current_week2_separately;"
    "combined_null_because_compact_ledgers_do_not_bind_all_episode_responses;"
    "excluded_from_holm_and_phase7"
)
STABILITY_POLICY = {
    "half_week_definition": (
        "within_each_registered_week_before_Wednesday_00_ET_vs_from_Wednesday_00_ET"
    ),
    "extension_half_week_strata": (
        "study_week_x_half_week_no_cross_week_pooling"
    ),
    "session_field": "registered_independent_SessionState.phase",
    "systematic_reversal": (
        "negative_mean_primary_paired_effect_in_any_stratum_with_at_least_5_pairs"
    ),
}
REQUIRED_PHASE6_IDENTITY_BINDINGS = frozenset(
    {
        "ohlcv_artifact",
        "ohlcv_manifest",
        "split_registry",
        "semantic_registry",
        "semantic_parameters",
        "model_config",
        "structure_protocol",
        "liquidity_protocol",
        "displacement_protocol",
        "group3_protocol",
        "group4_protocol",
        "group5_protocol",
        "mbo_feature_artifact",
        "mbo_feature_manifest",
        "raw_mbo_partition_manifest",
        "runner",
        "feature_materializer",
        "runtime_package_init",
        "runtime_artifact_stream",
        "runtime_mbo",
        "runtime_market_clock",
        "runtime_research",
        "runtime_feature",
        "runtime_signal_research",
        "runtime_causal",
        "runtime_io",
        "runtime_model",
        "runtime_observation",
        "runtime_event_store",
        "runtime_market_state",
        "runtime_semantics",
        "runtime_structure",
        "runtime_liquidity",
        "runtime_displacement",
        "runtime_displacement_observer",
        "runtime_group3",
        "runtime_group4",
        "runtime_group5",
        "runtime_scene_graph",
        "runtime_validation",
        "pyproject",
        "lockfile",
    }
)
PHASE6_EVENT_KINDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "sweep_rejection": ("sweep_confirmed",),
        "acceptance_continuation": ("acceptance_confirmed",),
        "displacement_impact": ("displacement_observed",),
        "mss_flow_shift": ("mss_core_confirmed",),
        "fvg_retest_response": (
            "fvg_partially_filled",
            "fvg_midpoint_touched",
            "fvg_fully_filled",
            "fvg_invalidated",
        ),
    }
)
FORBIDDEN_FVG_ALIAS = "fvg_touched"

PRIMARY_METRIC_BY_HYPOTHESIS: Mapping[str, str] = MappingProxyType(
    {
        "sweep_rejection": "response_aggressor_reversal",
        "acceptance_continuation": (
            "formation_directional_aggressor_imbalance"
        ),
        "displacement_impact": (
            "formation_directional_mid_impact_ticks_per_contract"
        ),
        "mss_flow_shift": "response_directional_ofi_shift",
        "fvg_retest_response": (
            "post_directional_aggressor_imbalance"
        ),
    }
)
SECONDARY_METRICS_BY_HYPOTHESIS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "sweep_rejection": (
            "post_displayed_defense_net_add_per_contract",
            "post_directional_best_level_ofi_per_contract",
        ),
        "acceptance_continuation": (
            "formation_directional_best_level_ofi_per_contract",
            "formation_directional_mid_impact_ticks_per_contract",
        ),
        "displacement_impact": (
            "formation_directional_aggressor_imbalance",
            "formation_directional_best_level_ofi_per_contract",
        ),
        "mss_flow_shift": (
            "post_directional_aggressor_imbalance",
            "post_directional_best_level_ofi_per_contract",
        ),
        "fvg_retest_response": (
            "post_displayed_defense_net_add_per_contract",
            "post_directional_best_level_ofi_per_contract",
        ),
    }
)

REQUIRED_MINUTE_FEATURE_COLUMNS = frozenset(
    {
        "decision_time",
        "symbol",
        "instrument_id",
        "book_observed_at",
        "book_valid",
        "book_change_valid",
        "aggressor_buy_volume",
        "aggressor_sell_volume",
        "aggressor_unknown_volume",
        "aggressor_buy_trade_count",
        "aggressor_sell_trade_count",
        "aggressor_unknown_trade_count",
        "passive_bid_fill_volume",
        "passive_ask_fill_volume",
        "passive_unknown_fill_volume",
        "passive_bid_fill_count",
        "passive_ask_fill_count",
        "passive_unknown_fill_count",
        "displayed_bid_add_volume",
        "displayed_ask_add_volume",
        "displayed_bid_cancel_volume",
        "displayed_ask_cancel_volume",
        "best_level_ofi_contracts",
        "mid_change_ticks",
        "spread_ticks",
        "book_valid_clock_fraction",
    }
)

_SHA256_CHARACTERS = frozenset("0123456789abcdef")


class Phase6ResearchError(ResearchContractError):
    """Raised before analysis when the frozen Phase-6 contract is invalid."""


class EpisodeWindowUnavailable(RuntimeError):
    """Expected episode censoring with a stable machine-readable reason."""

    def __init__(self, reason: str, *, episode_id: str) -> None:
        super().__init__(f"{episode_id}: {reason}")
        self.reason = reason
        self.episode_id = episode_id


class EpisodeMatchContextUnavailable(RuntimeError):
    """Expected censoring when strict-prior M1 matching context is absent."""

    _REASONS = frozenset(STRICT_PRIOR_CONTEXT_CENSOR_REASONS)

    def __init__(
        self,
        *,
        episode_id: str,
        context_clock: pd.Timestamp,
        reason: str = "strict_prior_real_completed_m1_context_missing",
    ) -> None:
        if reason not in self._REASONS:
            raise Phase6ResearchError("strict-prior context censor reason is invalid")
        super().__init__(f"{episode_id}: {reason}")
        self.reason = reason
        self.episode_id = episode_id
        self.context_clock = context_clock


def _aware_utc(value: Any, *, name: str) -> pd.Timestamp:
    clock = pd.Timestamp(value)
    if clock.tzinfo is None:
        raise Phase6ResearchError(f"{name} must be timezone aware")
    return clock.tz_convert("UTC")


def _sha256(value: Any, *, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        character not in _SHA256_CHARACTERS for character in normalized
    ):
        raise Phase6ResearchError(f"{name} must be a lowercase SHA-256")
    return normalized


def _bound_regular_file(root: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Phase6ResearchError(f"{name}.path is required")
    candidate = (root / value).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise Phase6ResearchError(f"{name}.path escapes the repository") from error
    if not candidate.is_file() or candidate.is_symlink():
        raise Phase6ResearchError(f"{name}.path is not a regular file")
    return candidate


@dataclass(frozen=True)
class Phase6Window:
    window_id: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    expected_rows: int

    def __post_init__(self) -> None:
        start = _aware_utc(self.start, name=f"{self.window_id}.start")
        end = _aware_utc(
            self.end_exclusive,
            name=f"{self.window_id}.end_exclusive",
        )
        if end <= start:
            raise Phase6ResearchError("Phase-6 window end must follow start")
        if (
            isinstance(self.expected_rows, bool)
            or not isinstance(self.expected_rows, Integral)
            or int(self.expected_rows) <= 0
        ):
            raise Phase6ResearchError("Phase-6 expected_rows must be positive")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end_exclusive", end)
        object.__setattr__(self, "expected_rows", int(self.expected_rows))


@dataclass(frozen=True)
class FrozenPhase6Contract:
    manifest_path: Path
    manifest_sha256: str
    payload: Mapping[str, Any]
    identity_paths: Mapping[str, Path]
    feature_artifact_path: Path
    feature_manifest_path: Path
    ohlcv_path: Path
    model_path: Path
    primary_window: Phase6Window
    extension_window: Phase6Window
    active_window: Phase6Window
    warmup_start: pd.Timestamp
    symbol: str
    instrument_id: int
    prior_week1_result_path: Path | None = None
    prior_week1_result: Mapping[str, Any] | None = None
    prior_week1_ledger_paths: Mapping[str, Path] | None = None


def _window_from_payload(value: Any, *, name: str) -> Phase6Window:
    if not isinstance(value, Mapping):
        raise Phase6ResearchError(f"{name} must be an object")
    return Phase6Window(
        window_id=str(value.get("id", "")),
        start=value.get("start"),
        end_exclusive=value.get("end_exclusive"),
        expected_rows=value.get("expected_rows"),
    )


def validate_phase6_design(payload: Mapping[str, Any]) -> None:
    """Validate the non-negotiable, threshold-free Phase-6 design."""

    if payload.get("schema_version") != 1:
        raise Phase6ResearchError("Phase-6 manifest schema_version must be 1")
    if payload.get("research_protocol_version") != PHASE6_PROTOCOL_VERSION:
        raise Phase6ResearchError("Phase-6 research protocol version changed")
    if tuple(payload.get("hypothesis_family", ())) != PHASE6_FIXED_FAMILY:
        raise Phase6ResearchError("Phase-6 fixed five-hypothesis family changed")
    if payload.get("parameter_search_space") != {}:
        raise Phase6ResearchError("Phase-6 threshold search is forbidden")
    if int(payload.get("minimum_matched_episodes", 0)) != 30:
        raise Phase6ResearchError("Phase-6 minimum matched sample must remain 30")

    inference = payload.get("inference")
    if not isinstance(inference, Mapping):
        raise Phase6ResearchError("Phase-6 inference contract is required")
    expected_inference = {
        "primary_test": "two_sided_exact_paired_sign_test",
        "multiple_testing": "holm_fixed_family",
        "family_order": list(PHASE6_FIXED_FAMILY),
        "alpha": 0.05,
        "paired_ci": "deterministic_paired_percentile_bootstrap",
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20240602,
        "missing_or_underpowered_p_value": 1.0,
    }
    if dict(inference) != expected_inference:
        raise Phase6ResearchError("Phase-6 inference contract changed")

    windows = payload.get("windows")
    if not isinstance(windows, Mapping):
        raise Phase6ResearchError("Phase-6 windows are required")
    primary = _window_from_payload(windows.get("primary"), name="windows.primary")
    extension = _window_from_payload(
        windows.get("underpowered_extension"),
        name="windows.underpowered_extension",
    )
    if (
        primary.window_id != "2024-06-week-1"
        or primary.start != pd.Timestamp("2024-06-02T22:00:00Z")
        or primary.end_exclusive != pd.Timestamp("2024-06-07T21:01:00Z")
        or primary.expected_rows != 6900
        or extension.window_id != "2024-06-week-2"
        or extension.start != pd.Timestamp("2024-06-09T22:00:00Z")
        or extension.end_exclusive != pd.Timestamp("2024-06-14T21:01:00Z")
        or extension.expected_rows != 6900
    ):
        raise Phase6ResearchError("Phase-6 June windows changed")
    if windows.get("extension_rule") != (
        "open_week_2_only_if_any_fixed_hypothesis_has_primary_matched_n_below_30;"
        "never_open_from_effect_direction_or_p_value"
    ):
        raise Phase6ResearchError("Phase-6 extension rule changed")

    reader_census = payload.get("reader_census_contract")
    if not isinstance(reader_census, Mapping):
        raise Phase6ResearchError("Phase-6 Reader census contract is required")
    expected_primary_census = {
        "window_id": "2024-06-week-1",
        "completed_clocks": 6900,
        "real_completed": 6899,
        "synthetic_no_trade": 1,
        "synthetic_decision_clocks": ["2024-06-07T03:10:00Z"],
        "synthetic_event_policy": (
            "normalized_M1_clock_root_with_only_registered_terminal_censor_semantic_exception"
        ),
        "synthetic_m5_source_or_control_eligible": False,
        "mbo_response_window_eligible": True,
    }
    if payload.get("study_mode") == "primary_week_only":
        if dict(reader_census) != expected_primary_census:
            raise Phase6ResearchError("Phase-6 primary Reader census changed")
    else:
        completed = reader_census.get("completed_clocks")
        real = reader_census.get("real_completed")
        synthetic = reader_census.get("synthetic_no_trade")
        synthetic_clocks = reader_census.get("synthetic_decision_clocks")
        if (
            reader_census.get("window_id") != "2024-06-week-2"
            or isinstance(completed, bool)
            or not isinstance(completed, Integral)
            or int(completed) != extension.expected_rows
            or isinstance(real, bool)
            or not isinstance(real, Integral)
            or isinstance(synthetic, bool)
            or not isinstance(synthetic, Integral)
            or int(real) + int(synthetic) != int(completed)
            or not isinstance(synthetic_clocks, list)
            or len(synthetic_clocks) != int(synthetic)
            or reader_census.get("synthetic_event_policy")
            != "normalized_M1_clock_root_with_only_registered_terminal_censor_semantic_exception"
            or reader_census.get("synthetic_m5_source_or_control_eligible") is not False
            or reader_census.get("mbo_response_window_eligible") is not True
        ):
            raise Phase6ResearchError("Phase-6 extension Reader census is invalid")
        clocks = tuple(
            _aware_utc(value, name="reader synthetic decision clock")
            for value in synthetic_clocks
        )
        if len(clocks) != len(set(clocks)) or any(
            not (extension.start <= value < extension.end_exclusive)
            for value in clocks
        ):
            raise Phase6ResearchError(
                "Phase-6 extension synthetic clock census is invalid"
            )

    feature_windows = payload.get("feature_windows")
    if feature_windows != {
        "pre_context_completed_minutes": 5,
        "formation": "exact_M5_source_BAR_clocks_expanded_to_completed_M1",
        "post_response_completed_minutes": 5,
        "formation_available_at": "semantic_known_at",
        "post_response_authority": "retrospective_mechanism_validation_only",
    }:
        raise Phase6ResearchError("Phase-6 feature-window contract changed")
    if payload.get("minute_feature_availability") != {
        "event_clock_assignment": (
            "ceil_ts_recv_to_completed_minute_including_exact_decision_clock"
        ),
        "book_change_valid_required_for_ofi_and_mid_delta": True,
        "invalid_book_change_window_policy": (
            "censor_episode_with_exact_reason_never_fill_zero"
        ),
        "displayed_defense_net_add_semantics": (
            "all_book_A_minus_C_minus_passive_F_proxy_not_same_level_queue_replenishment_or_absorption"
        ),
        "synthetic_clock_mbo_flow_policy": (
            "aggressor_buy_sell_unknown_trade_volume_and_count_plus_passive_bid_ask_unknown_fill_volume_and_count_must_all_equal_zero_displayed_add_cancel_may_be_nonzero"
        ),
    }:
        raise Phase6ResearchError("Phase-6 minute feature availability changed")

    hypothesis_design = payload.get("hypothesis_design")
    if not isinstance(hypothesis_design, Mapping) or set(hypothesis_design) != set(
        PHASE6_FIXED_FAMILY
    ):
        raise Phase6ResearchError("Phase-6 hypothesis definitions are incomplete")
    for hypothesis in PHASE6_FIXED_FAMILY:
        definition = hypothesis_design[hypothesis]
        if not isinstance(definition, Mapping):
            raise Phase6ResearchError(f"invalid hypothesis definition: {hypothesis}")
        if definition.get("primary_metric") != PRIMARY_METRIC_BY_HYPOTHESIS[hypothesis]:
            raise Phase6ResearchError(f"primary metric changed: {hypothesis}")
        if tuple(definition.get("secondary_metrics", ())) != (
            SECONDARY_METRICS_BY_HYPOTHESIS[hypothesis]
        ):
            raise Phase6ResearchError(f"secondary metrics changed: {hypothesis}")
        if definition.get("expected_effect_direction") != "greater_than_control":
            raise Phase6ResearchError(f"effect direction changed: {hypothesis}")
        if definition.get("causal_claim") is not False:
            raise Phase6ResearchError("Phase-6 cannot claim causality")
    fvg_variants = hypothesis_design["fvg_retest_response"].get("event_variants")
    if fvg_variants != {
        "successful_retest": [
            "fvg_partially_filled",
            "fvg_midpoint_touched",
            "fvg_fully_filled",
        ],
        "failed_retest": ["fvg_invalidated"],
        "pooling": "forbidden",
    }:
        raise Phase6ResearchError("FVG successful/failed lifecycle variants changed")
    if (
        hypothesis_design["fvg_retest_response"].get("primary_control_variant")
        != "failed_retest"
        or hypothesis_design["fvg_retest_response"].get("pseudo_zone_role")
        != "descriptive_sensitivity_only_excluded_from_holm_and_phase7"
    ):
        raise Phase6ResearchError(
            "FVG primary and pseudo-zone control roles must remain disjoint"
        )
    if hypothesis_design["fvg_retest_response"].get("pseudo_zone_protocol") != {
        "construction": (
            "same_width_shifted_farther_on_the_same_directional_side_at_creation_known_at"
        ),
        "exclude_overlap_with_then_known_real_fvg": True,
        "first_touch_only": True,
        "maximum_future_m5_bars": 120,
        "eye_publication": False,
    }:
        raise Phase6ResearchError("FVG pseudo-zone protocol changed")

    support = payload.get("support_rule")
    if support != {
        "minimum_primary_matched_n": 30,
        "holm_adjusted_p_below": 0.05,
        "paired_primary_ci_excludes_zero_expected_direction": True,
        "one_secondary_mean_effect_expected_direction": True,
        "half_week_and_session_no_systematic_sign_reversal": True,
        "stability_stratum_minimum_n": 5,
        "unsupported_or_underpowered_excluded_from_phase7": True,
    }:
        raise Phase6ResearchError("Phase-6 support/admission rule changed")
    if payload.get("matching_context") != {
        "context_clock": (
            "earliest_formation_source_M5_end_minus_5_completed_minutes"
        ),
        "source": "cached_real_completed_M1_at_strict_prior_formation_clock",
        "strict_prior_fields": [
            "volatility_bucket",
            "trend_relation",
            "relative_volume_bucket",
        ],
        "event_clock_field": "session_phase_only",
        "missing_context_policy": (
            "censor_with_strict_prior_real_completed_m1_context_missing"
        ),
        "unavailable_field_policy": dict(
            STRICT_PRIOR_CONTEXT_UNAVAILABLE_FIELD_POLICY
        ),
        "censor_reasons": list(STRICT_PRIOR_CONTEXT_CENSOR_REASONS),
        "same_clock_or_post_event_fallback": False,
    }:
        raise Phase6ResearchError("Phase-6 strict-prior matching context changed")
    semantic_exception_policy = payload.get(
        "synthetic_semantic_exception_policy"
    )
    if payload.get("study_mode") == (
        "primary_plus_registered_underpowered_extension"
    ):
        valid_semantic_exception_policy = (
            semantic_exception_policy
            == SYNTHETIC_SEMANTIC_EXCEPTION_POLICY
        )
    else:
        valid_semantic_exception_policy = semantic_exception_policy in (
            LEGACY_SYNTHETIC_SEMANTIC_EXCEPTION_POLICY,
            SYNTHETIC_SEMANTIC_EXCEPTION_POLICY,
        )
    if not valid_semantic_exception_policy:
        raise Phase6ResearchError(
            "Phase-6 synthetic semantic exception policy changed"
        )
    extension_warmup_policy = payload.get(
        "extension_warmup_synthetic_exception_policy"
    )
    if payload.get("study_mode") == "primary_plus_registered_underpowered_extension":
        if extension_warmup_policy != EXTENSION_WARMUP_SYNTHETIC_EXCEPTION_POLICY:
            raise Phase6ResearchError(
                "Phase-6 extension warmup synthetic exception policy changed"
            )
        if payload.get("extension_displacement_monotonicity_policy") != (
            EXTENSION_DISPLACEMENT_MONOTONICITY_POLICY
        ):
            raise Phase6ResearchError(
                "Phase-6 extension displacement monotonicity policy changed"
            )
    elif extension_warmup_policy is not None and (
        extension_warmup_policy != EXTENSION_WARMUP_SYNTHETIC_EXCEPTION_POLICY
    ):
        raise Phase6ResearchError(
            "Phase-6 extension warmup synthetic exception policy changed"
        )
    stability = payload.get("stability")
    legacy_primary_stability = {
        key: value
        for key, value in STABILITY_POLICY.items()
        if key != "extension_half_week_strata"
    }
    if stability != STABILITY_POLICY and not (
        payload.get("study_mode") == "primary_week_only"
        and stability == legacy_primary_stability
    ):
        raise Phase6ResearchError("Phase-6 stability stratification changed")
    if payload.get("analysis_unit") != {
        "entity_first_rules": {
            "fvg_retest_response": (
                "first_concrete_lifecycle_transition_per_fvg_entity"
            ),
            "displacement_impact": (
                "first_ACTIVE_observation_per_displacement_entity_by_known_at_then_event_id"
            ),
        },
        "clock_canonical_identity_fields": [
            "symbol",
            "instrument_id",
            "timeframe",
            "hypothesis",
            "event_variant",
            "control_kind",
            "known_at",
            "direction",
        ],
        "opposite_direction_same_clock_policy": (
            "exclude_as_ambiguous_before_matching"
        ),
        "constituent_audit": "sorted_event_ids_entities_and_count",
        "displacement_score_aggregation": "maximum_constituent_score",
        "candidate_clock_capacity": 1,
        "source_m5_definition": (
            "transitive_source_only_ancestry_intersected_with_closed_episode_interval_event_time_to_known_at"
        ),
        "source_m5_audit": (
            "event_ids_sorted_by_clock_then_event_id_plus_sha256_count_and_episode_local_formation_clocks"
        ),
    }:
        raise Phase6ResearchError("Phase-6 canonical analysis unit changed")
    if payload.get("matching") != {
        "algorithm": (
            "deterministic_earliest_first_global_nonoverlap_greedy_within_hypothesis"
        ),
        "forward_only": True,
        "exact_fields": [
            "symbol",
            "instrument_id",
            "study_week",
            "half_week",
            "timeframe",
            "session_phase",
            "direction",
            "volatility_bucket",
            "trend_relation",
            "relative_volume_bucket",
        ],
        "maximum_completed_minute_offset": 3000,
        "post_response_minutes": 5,
        "embargo_minutes": 5,
        "replacement_limit": 1,
        "candidate_direction": "candidate_local_known_at",
        "same_clock_treatment_exclusion": True,
        "inference_clock_set": "formation_plus_post",
        "within_pair_embargo_completed_minutes": 5,
        "cross_pair_clock_reuse_within_hypothesis": False,
        "cross_hypothesis_clock_reuse_restricted": False,
    }:
        raise Phase6ResearchError("Phase-6 non-overlap matching contract changed")
    if payload.get("engineering_pass_rule") != {
        "identity_and_lineage_verified": True,
        "raw_partition_hashes_verified": True,
        "exact_completed_clock_census": True,
        "real_completed_ohlcv_clock_coverage": 1.0,
        "registered_synthetic_mbo_response_clock_allowed": True,
        "registered_synthetic_semantic_exception_verified": True,
        "unique_event_and_episode_identity": True,
        "no_future_book_or_source_bar_clock": True,
        "fixed_family_inference_completed": True,
        "individual_mechanism_support_not_required": True,
    }:
        raise Phase6ResearchError("Phase-6 engineering pass contract changed")


def load_frozen_phase6_contract(
    manifest_path: str | Path,
    *,
    root: str | Path,
    verify_raw_partition_hashes: bool = True,
    comparison_validation_only: bool = False,
) -> FrozenPhase6Contract:
    """Verify all authority and identity bindings before opening market data."""

    if type(comparison_validation_only) is not bool:
        raise Phase6ResearchError("comparison_validation_only must be boolean")

    repository = Path(root).resolve()
    source = Path(manifest_path).resolve()
    try:
        source.relative_to(repository)
    except ValueError as error:
        raise Phase6ResearchError("manifest must live inside the repository") from error
    if not source.is_file() or source.is_symlink():
        raise Phase6ResearchError("Phase-6 manifest is not a regular file")
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase6ResearchError(
            "Phase-6 manifest must use the JSON subset of YAML"
        ) from error
    if not isinstance(payload, Mapping):
        raise Phase6ResearchError("Phase-6 manifest root must be an object")
    validate_phase6_design(payload)
    if (
        payload.get("status") != PHASE6_EXECUTABLE_STATUS
        or payload.get("frozen_before_run") is not True
    ):
        raise Phase6ResearchError(
            "Phase-6 template is incomplete/unfrozen and cannot be executed"
        )
    authority = payload.get("authority")
    expected_authority = {
        "mbo_development_only": True,
        "mechanism_association_only": True,
        "artifact_fit_allowed": False,
        "phase7_evidence_allowed_only_if_supported": (
            not comparison_validation_only
        ),
        "trading_authority": False,
        "sealed_holdout_opened": False,
    }
    if authority != expected_authority:
        raise Phase6ResearchError("Phase-6 authority must fail closed")
    comparison_contract = payload.get("comparison_contract")
    if comparison_validation_only:
        if not isinstance(comparison_contract, Mapping) or dict(
            comparison_contract
        ).get("execution_authority") != {
            "comparison_only": True,
            "rolling_oof": False,
            "sealed_oos": False,
            "model_admission": False,
            "trading": False,
        }:
            raise Phase6ResearchError(
                "Phase-6 comparison-only authority is absent"
            )
    elif comparison_contract is not None:
        raise Phase6ResearchError(
            "comparison contract requires the explicit validation-only seam"
        )
    experiment_id = payload.get("experiment_id")
    frozen_at = pd.Timestamp(payload.get("frozen_at"))
    if not isinstance(experiment_id, str) or not experiment_id or frozen_at.tzinfo is None:
        raise Phase6ResearchError("experiment_id and aware frozen_at are required")

    contract = payload.get("contract")
    if contract != {"symbol": "NQM4", "instrument_id": 13743}:
        raise Phase6ResearchError("Phase-6 contract identity must be NQM4/13743")

    bindings = payload.get("identity_bindings")
    if (
        not isinstance(bindings, Mapping)
        or set(bindings) != REQUIRED_PHASE6_IDENTITY_BINDINGS
    ):
        raise Phase6ResearchError(
            "Phase-6 identity bindings must contain the exact runtime/input set"
        )
    paths: dict[str, Path] = {}
    for name, binding in bindings.items():
        if not isinstance(name, str) or not isinstance(binding, Mapping):
            raise Phase6ResearchError("invalid Phase-6 identity binding")
        path = _bound_regular_file(repository, binding.get("path"), name=name)
        expected = _sha256(binding.get("sha256"), name=f"{name}.sha256")
        if sha256_file(path) != expected:
            raise Phase6ResearchError(f"Phase-6 identity binding changed: {name}")
        paths[name] = path
    expected_runtime = {
        "ohlcv_artifact": repository
        / "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.parquet",
        "ohlcv_manifest": repository
        / "data/processed/nq_1m_previous_session_front_v2_3_2017_2026.manifest.json",
        "split_registry": repository / "configs/data_splits.json",
        "semantic_registry": repository / "semantics/registry_v1_2.yaml",
        "semantic_parameters": repository / "semantics/parameters_v1_2.yaml",
        "model_config": repository / "configs/model.json",
        "raw_mbo_partition_manifest": repository
        / "data/raw/nq_mbo/legacy_parquet/manifest.json",
        "runner": repository / "scripts/run_mbo_mechanism_research.py",
        "runtime_research": repository / "smc_trader/mbo_mechanism_research.py",
        "runtime_feature": repository / "smc_trader/mbo_mechanism.py",
        "feature_materializer": repository / "scripts/materialize_mbo_mechanism.py",
        "runtime_package_init": repository / "smc_trader/__init__.py",
        "runtime_artifact_stream": repository / "smc_trader/artifact_stream.py",
        "runtime_mbo": repository / "smc_trader/mbo.py",
        "runtime_market_clock": repository / "smc_trader/market_clock.py",
        "runtime_signal_research": repository / "smc_trader/signal_research.py",
        "runtime_causal": repository / "smc_trader/causal.py",
        "runtime_io": repository / "smc_trader/io.py",
        "runtime_model": repository / "smc_trader/model.py",
        "runtime_observation": repository / "smc_trader/observation.py",
        "runtime_event_store": repository / "smc_trader/event_store.py",
        "runtime_market_state": repository / "smc_trader/market_state.py",
        "runtime_semantics": repository / "smc_trader/semantics.py",
        "runtime_structure": repository / "smc_trader/structure.py",
        "runtime_liquidity": repository / "smc_trader/liquidity.py",
        "runtime_displacement": repository / "smc_trader/displacement.py",
        "runtime_displacement_observer": repository / "smc_trader/displacement_observer.py",
        "runtime_group3": repository / "smc_trader/group3.py",
        "runtime_group4": repository / "smc_trader/group4.py",
        "runtime_group5": repository / "smc_trader/group5.py",
        "runtime_scene_graph": repository / "smc_trader/scene_graph.py",
        "runtime_validation": repository / "smc_trader/validation.py",
        "pyproject": repository / "pyproject.toml",
        "lockfile": repository / "uv.lock",
    }
    for name, expected in expected_runtime.items():
        if paths[name] != expected.resolve():
            raise Phase6ResearchError(f"{name} does not bind the executed project")
    expected_protocol_paths = {
        "structure_protocol": repository / "configs/primitives_structure_liquidity.json",
        "liquidity_protocol": repository / "configs/primitives_structure_liquidity.json",
        "displacement_protocol": repository / "configs/primitives_displacement.json",
        "group3_protocol": repository / "configs/primitives_zones.json",
        "group4_protocol": repository / "configs/primitives_range.json",
        "group5_protocol": repository / "configs/primitives_entry.json",
    }
    for name, expected in expected_protocol_paths.items():
        if paths[name] != expected.resolve():
            raise Phase6ResearchError(f"{name} does not bind its production protocol")
    model = json.loads(paths["model_config"].read_text(encoding="utf-8"))
    observer = model.get("observer") if isinstance(model, Mapping) else None
    if not isinstance(observer, Mapping):
        raise Phase6ResearchError("bound model observer configuration is invalid")
    for field, binding in (
        ("semantic_registry", "semantic_registry"),
        ("structure_protocol", "structure_protocol"),
        ("liquidity_protocol", "liquidity_protocol"),
        ("displacement_protocol", "displacement_protocol"),
        ("group3_protocol", "group3_protocol"),
        ("group4_protocol", "group4_protocol"),
        ("group5_protocol", "group5_protocol"),
    ):
        label = observer.get(field)
        if (
            not isinstance(label, str)
            or (repository / label).resolve(strict=False) != paths[binding]
        ):
            raise Phase6ResearchError(
                f"model {field} disagrees with the exact identity binding"
            )
    registry = json.loads(paths["semantic_registry"].read_text(encoding="utf-8"))
    parameters_file = registry.get("parameters_file") if isinstance(registry, Mapping) else None
    if (
        not isinstance(parameters_file, str)
        or (repository / parameters_file).resolve(strict=False)
        != paths["semantic_parameters"]
    ):
        raise Phase6ResearchError(
            "semantic registry parameters disagree with identity binding"
        )
    feature_manifest = json.loads(
        paths["mbo_feature_manifest"].read_text(encoding="utf-8")
    )
    if not isinstance(feature_manifest, Mapping):
        raise Phase6ResearchError("MBO feature manifest root is invalid")
    feature_output = feature_manifest.get("output")
    if not isinstance(feature_output, Mapping):
        raise Phase6ResearchError("MBO feature manifest output binding is absent")
    feature_output_path = Path(str(feature_output.get("path", "")))
    if not feature_output_path.is_absolute():
        feature_output_path = repository / feature_output_path
    if (
        feature_output_path.resolve(strict=False) != paths["mbo_feature_artifact"]
        or feature_output.get("sha256")
        != sha256_file(paths["mbo_feature_artifact"])
    ):
        raise Phase6ResearchError(
            "MBO feature manifest does not bind the registered feature artifact"
        )

    raw_partitions = payload.get("raw_mbo_partitions")
    study_mode = payload.get("study_mode")
    expected_partition_count = 6
    if (
        not isinstance(raw_partitions, list)
        or len(raw_partitions) != expected_partition_count
    ):
        raise Phase6ResearchError(
            "Phase-6 run binds an unexpected raw-partition census"
        )
    registered_partition_manifest = _sha256(
        payload.get("raw_mbo_partition_manifest_sha256"),
        name="raw_mbo_partition_manifest_sha256",
    )
    if sha256_file(paths["raw_mbo_partition_manifest"]) != registered_partition_manifest:
        raise Phase6ResearchError("raw MBO partition manifest identity changed")
    registered_raw_partitions: dict[Path, str] = {}
    for index, item in enumerate(raw_partitions):
        if not isinstance(item, Mapping):
            raise Phase6ResearchError("invalid raw MBO partition binding")
        path = _bound_regular_file(
            repository,
            item.get("path"),
            name=f"raw_mbo_partitions[{index}]",
        )
        expected = _sha256(
            item.get("sha256"),
            name=f"raw_mbo_partitions[{index}].sha256",
        )
        if verify_raw_partition_hashes and sha256_file(path) != expected:
            raise Phase6ResearchError(f"raw MBO partition changed: {path}")
        registered_raw_partitions[path] = expected

    windows = payload["windows"]
    primary = _window_from_payload(windows["primary"], name="windows.primary")
    extension = _window_from_payload(
        windows["underpowered_extension"],
        name="windows.underpowered_extension",
    )
    if study_mode == "primary_week_only":
        active = primary
        gate_path = None
        gate_result = None
        gate_ledgers = None
        if payload.get("week1_gate_result") is not None:
            raise Phase6ResearchError("primary run cannot bind a prior gate result")
    elif study_mode == "primary_plus_registered_underpowered_extension":
        gate = payload.get("week1_gate_result")
        if not isinstance(gate, Mapping):
            raise Phase6ResearchError("extension run requires the week-1 gate result")
        gate_path = _bound_regular_file(repository, gate.get("path"), name="week1_gate")
        if sha256_file(gate_path) != _sha256(
            gate.get("sha256"), name="week1_gate.sha256"
        ):
            raise Phase6ResearchError("week-1 gate result identity changed")
        gate_result = json.loads(gate_path.read_text(encoding="utf-8"))
        if not isinstance(gate_result, Mapping):
            raise Phase6ResearchError("week-1 gate result root is invalid")
        validate_underpowered_extension_gate(
            gate_result,
            expected_manifest_sha256=str(gate.get("manifest_sha256")),
        )
        gate_ledgers = validate_prior_result_ledgers(
            gate_result,
            root=repository,
        )
        # Week 2 is materialized and replayed independently. The runner then
        # combines its compact pairs with the hash-bound week-1 pair ledger.
        # This avoids reading week 2 before the sample-size gate opens it.
        active = extension
    else:
        raise Phase6ResearchError("Phase-6 study_mode is invalid")
    warmup_start = _aware_utc(payload.get("warmup_start"), name="warmup_start")
    if warmup_start >= primary.start:
        raise Phase6ResearchError("Phase-6 warmup must precede the primary window")
    feature_raw = feature_manifest.get("raw_partitions")
    if not isinstance(feature_raw, list):
        raise Phase6ResearchError("MBO feature manifest raw partitions are absent")
    feature_raw_partitions: dict[Path, str] = {}
    for item in feature_raw:
        if not isinstance(item, Mapping):
            raise Phase6ResearchError("MBO feature raw partition binding is invalid")
        path = Path(str(item.get("path", "")))
        if not path.is_absolute():
            path = repository / path
        feature_raw_partitions[path.resolve(strict=False)] = str(item.get("sha256", ""))
    if feature_raw_partitions != registered_raw_partitions:
        raise Phase6ResearchError(
            "research and MBO feature manifests bind different raw partitions"
        )
    try:
        feature_start = _aware_utc(feature_manifest.get("start"), name="feature.start")
        feature_end = _aware_utc(
            feature_manifest.get("end_exclusive"), name="feature.end_exclusive"
        )
    except Phase6ResearchError:
        raise
    if (
        feature_start != active.start
        or feature_end != active.end_exclusive
        or feature_manifest.get("symbol") != "NQM4"
        or feature_manifest.get("instrument_id") != 13743
        or feature_output.get("rows") != active.expected_rows
    ):
        raise Phase6ResearchError(
            "MBO feature manifest disagrees with the active registered window"
        )

    return FrozenPhase6Contract(
        manifest_path=source,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        payload=MappingProxyType(dict(payload)),
        identity_paths=MappingProxyType(paths),
        feature_artifact_path=paths["mbo_feature_artifact"],
        feature_manifest_path=paths["mbo_feature_manifest"],
        ohlcv_path=paths["ohlcv_artifact"],
        model_path=paths["model_config"],
        primary_window=primary,
        extension_window=extension,
        active_window=active,
        warmup_start=warmup_start,
        symbol="NQM4",
        instrument_id=13743,
        prior_week1_result_path=gate_path,
        prior_week1_result=(
            None if gate_result is None else MappingProxyType(dict(gate_result))
        ),
        prior_week1_ledger_paths=gate_ledgers,
    )


def validate_underpowered_extension_gate(
    week1_result: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
) -> None:
    """Admit week 2 only from week-1 primary sample size, never effect size."""

    if week1_result.get("manifest_sha256") != expected_manifest_sha256:
        raise Phase6ResearchError("extension gate binds a different week-1 manifest")
    if week1_result.get("study_mode") != "primary_week_only":
        raise Phase6ResearchError("extension gate is not a primary-week result")
    if week1_result.get("raw_partition_hashes_verified") is not True:
        raise Phase6ResearchError(
            "extension gate requires verified week-1 raw partition hashes"
        )
    mechanisms = week1_result.get("mechanisms")
    if not isinstance(mechanisms, Mapping) or set(mechanisms) != set(
        PHASE6_FIXED_FAMILY
    ):
        raise Phase6ResearchError("extension gate has an incomplete fixed family")
    sample_sizes: list[int] = []
    for name in PHASE6_FIXED_FAMILY:
        value = mechanisms[name]
        if not isinstance(value, Mapping):
            raise Phase6ResearchError("extension gate mechanism is invalid")
        sample = value.get("primary_matched_n")
        if isinstance(sample, bool) or not isinstance(sample, Integral) or sample < 0:
            raise Phase6ResearchError("extension gate sample size is invalid")
        sample_sizes.append(int(sample))
    if not any(sample < 30 for sample in sample_sizes):
        raise Phase6ResearchError(
            "week 2 may open only because a fixed hypothesis is underpowered"
        )


def validate_prior_result_ledgers(
    week1_result: Mapping[str, Any],
    *,
    root: str | Path,
) -> Mapping[str, Path]:
    """Verify compact week-1 ledgers before they enter an extension result."""

    if (
        week1_result.get("engineering_status") != "pass"
        or week1_result.get("result_identity") != canonical_identity(week1_result)
    ):
        raise Phase6ResearchError("week-1 result identity/engineering gate is invalid")
    ledgers = week1_result.get("ledgers")
    if not isinstance(ledgers, Mapping) or set(ledgers) != {
        "episodes",
        "matched_pairs",
        "unmatched",
    }:
        raise Phase6ResearchError("week-1 compact ledger bindings are incomplete")
    repository = Path(root).resolve()
    paths: dict[str, Path] = {}
    for name, binding in ledgers.items():
        if not isinstance(binding, Mapping):
            raise Phase6ResearchError("week-1 ledger binding is invalid")
        path = _bound_regular_file(repository, binding.get("path"), name=f"week1.{name}")
        if sha256_file(path) != _sha256(
            binding.get("sha256"), name=f"week1.{name}.sha256"
        ):
            raise Phase6ResearchError(f"week-1 ledger identity changed: {name}")
        rows = binding.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, Integral) or rows < 0:
            raise Phase6ResearchError(f"week-1 ledger row count is invalid: {name}")
        with path.open("rb") as handle:
            actual_rows = sum(1 for line in handle if line.strip())
        if actual_rows != int(rows):
            raise Phase6ResearchError(f"week-1 ledger row count changed: {name}")
        paths[name] = path
    return MappingProxyType(paths)


@dataclass(frozen=True)
class MechanismEpisode:
    episode_id: str
    hypothesis: str
    event_kind: str
    entity_id: str | None
    known_at: pd.Timestamp
    event_time: pd.Timestamp
    source_bar_clocks: tuple[pd.Timestamp, ...]
    source_m5_bar_event_ids: tuple[str, ...]
    symbol: str
    instrument_id: int
    timeframe: str
    direction: str
    session_phase: str
    half_week: str
    match_fields: Mapping[str, Any]
    score: float | None = None
    control_kind: str | None = None
    event_variant: str | None = None
    match_context_clock: pd.Timestamp | None = None
    match_context_source: str = (
        "cached_real_completed_M1_at_strict_prior_formation_clock"
    )
    source_m5_lineage_definition: str = (
        "transitive_source_only_ancestry_intersected_with_closed_episode_interval_event_time_to_known_at"
    )
    constituent_event_ids: tuple[str, ...] = ()
    constituent_entity_ids: tuple[str, ...] = ()
    statistical_unit: str = "atomic_semantic_event"

    def __post_init__(self) -> None:
        if self.hypothesis not in PHASE6_FIXED_FAMILY:
            raise Phase6ResearchError("episode hypothesis is not registered")
        if self.event_kind == FORBIDDEN_FVG_ALIAS:
            raise Phase6ResearchError("FVG_TOUCHED alias is forbidden in Phase 6")
        allowed = PHASE6_EVENT_KINDS[self.hypothesis]
        if self.control_kind is None and self.event_kind not in allowed:
            raise Phase6ResearchError(
                f"event kind is invalid for {self.hypothesis}: {self.event_kind}"
            )
        if self.hypothesis == "fvg_retest_response":
            expected_variant = (
                "failed_retest"
                if self.event_kind == "fvg_invalidated"
                else "successful_retest"
            )
            if self.control_kind == "pseudo_zone_retest":
                expected_variant = "pseudo_zone"
            if self.event_variant != expected_variant:
                raise Phase6ResearchError(
                    "FVG lifecycle event_variant must separate successful, failed "
                    "and pseudo-zone retests"
                )
        if not self.episode_id or self.direction not in {"long", "short"}:
            raise Phase6ResearchError("episode identity/direction is invalid")
        known_at = _aware_utc(self.known_at, name="episode.known_at")
        event_time = _aware_utc(self.event_time, name="episode.event_time")
        if event_time > known_at:
            raise Phase6ResearchError("episode event_time cannot follow known_at")
        source_clocks = tuple(
            sorted(
                {
                    _aware_utc(value, name="episode.source_bar_clock")
                    for value in self.source_bar_clocks
                }
            )
        )
        if not source_clocks or source_clocks[-1] > known_at:
            raise Phase6ResearchError(
                "episode source BAR clocks must be nonempty and known by known_at"
            )
        if known_at not in source_clocks:
            raise Phase6ResearchError(
                "episode terminal M5 source BAR clock must equal known_at"
            )
        source_ids = tuple(self.source_m5_bar_event_ids)
        if (
            len(source_ids) != len(source_clocks)
            or len(source_ids) != len(set(source_ids))
            or any(not isinstance(value, str) or not value for value in source_ids)
        ):
            raise Phase6ResearchError(
                "episode source M5 BAR identities must map one-to-one to clocks"
            )
        if (
            not isinstance(self.symbol, str)
            or not self.symbol
            or isinstance(self.instrument_id, bool)
            or not isinstance(self.instrument_id, Integral)
            or self.timeframe != "5m"
            or not self.session_phase
            or not self.half_week
        ):
            raise Phase6ResearchError("episode market identity is invalid")
        if not isinstance(self.match_fields, Mapping):
            raise Phase6ResearchError("episode match_fields must be a mapping")
        if self.match_context_clock is None:
            raise Phase6ResearchError("episode strict-prior match context is required")
        context_clock = _aware_utc(
            self.match_context_clock,
            name="episode.match_context_clock",
        )
        expected_context_clock = strict_prior_match_context_clock(source_clocks)
        if (
            context_clock != expected_context_clock
            or self.match_context_source
            != "cached_real_completed_M1_at_strict_prior_formation_clock"
        ):
            raise Phase6ResearchError(
                "episode matching context is not the frozen strict-prior M1 fact"
            )
        if self.source_m5_lineage_definition != (
            "transitive_source_only_ancestry_intersected_with_closed_episode_interval_event_time_to_known_at"
        ):
            raise Phase6ResearchError("episode source M5 lineage definition changed")
        constituent_event_ids = tuple(self.constituent_event_ids) or (self.episode_id,)
        if (
            tuple(sorted(set(constituent_event_ids))) != constituent_event_ids
            or any(not isinstance(value, str) or not value for value in constituent_event_ids)
        ):
            raise Phase6ResearchError("episode constituent event identities are invalid")
        constituent_entity_ids = tuple(self.constituent_entity_ids)
        if not constituent_entity_ids and self.entity_id:
            constituent_entity_ids = (str(self.entity_id),)
        constituent_entity_ids = tuple(sorted(set(constituent_entity_ids)))
        if any(
            not isinstance(value, str) or not value
            for value in constituent_entity_ids
        ):
            raise Phase6ResearchError("episode constituent entity identities are invalid")
        if self.statistical_unit not in {
            "atomic_semantic_event",
            "first_active_per_displacement_entity",
            "first_fvg_lifecycle_per_entity",
            "canonical_phase6_clock_episode",
        }:
            raise Phase6ResearchError("episode statistical unit is invalid")
        object.__setattr__(self, "known_at", known_at)
        object.__setattr__(self, "event_time", event_time)
        object.__setattr__(self, "source_bar_clocks", source_clocks)
        object.__setattr__(self, "source_m5_bar_event_ids", source_ids)
        object.__setattr__(self, "instrument_id", int(self.instrument_id))
        object.__setattr__(self, "match_fields", MappingProxyType(dict(self.match_fields)))
        object.__setattr__(self, "match_context_clock", context_clock)
        object.__setattr__(self, "constituent_event_ids", constituent_event_ids)
        object.__setattr__(self, "constituent_entity_ids", constituent_entity_ids)


def strict_prior_match_context_clock(
    source_bar_clocks: Sequence[pd.Timestamp],
) -> pd.Timestamp:
    """Return the real M1 clock immediately before earliest M5 formation."""

    clocks = tuple(
        sorted(_aware_utc(value, name="source M5 BAR clock") for value in source_bar_clocks)
    )
    if not clocks:
        raise Phase6ResearchError("source BAR clocks are required for match context")
    return clocks[0] - pd.Timedelta(minutes=5)


def first_fvg_lifecycle_episodes(
    episodes: Sequence[MechanismEpisode],
) -> tuple[MechanismEpisode, ...]:
    """Freeze the historical first concrete lifecycle transition per entity.

    This is the already-registered Phase 6 estimand, not a geometric
    ``first_retest_event``.  In particular, an invalidation alone does not
    prove that price first retested the zone; a future first-retest study must
    derive that treatment from creation-time geometry and completed BAR facts.
    """

    valid_kinds = set(PHASE6_EVENT_KINDS["fvg_retest_response"])
    by_entity: dict[str, list[MechanismEpisode]] = {}
    for episode in episodes:
        if episode.event_kind == FORBIDDEN_FVG_ALIAS:
            raise Phase6ResearchError("FVG_TOUCHED alias is forbidden in Phase 6")
        if episode.hypothesis != "fvg_retest_response":
            continue
        if episode.event_kind not in valid_kinds or not episode.entity_id:
            raise Phase6ResearchError("FVG lifecycle episode lacks concrete identity")
        by_entity.setdefault(episode.entity_id, []).append(episode)

    selected: list[MechanismEpisode] = []
    for entity_id, members in by_entity.items():
        earliest_known_at = min(item.known_at for item in members)
        earliest = tuple(
            item for item in members if item.known_at == earliest_known_at
        )
        lifecycle_signatures = {
            (item.event_kind, item.event_variant) for item in earliest
        }
        if len(lifecycle_signatures) != 1:
            raise Phase6ResearchError(
                "FVG entity earliest known_at has conflicting lifecycle "
                f"transitions: {entity_id} at {earliest_known_at.isoformat()}"
            )
        if any(item != earliest[0] for item in earliest[1:]):
            raise Phase6ResearchError(
                "FVG entity earliest known_at has conflicting duplicate "
                f"lifecycle payloads: {entity_id} at "
                f"{earliest_known_at.isoformat()}"
            )
        selected.append(min(earliest, key=lambda item: item.episode_id))
    return tuple(
        replace(item, statistical_unit="first_fvg_lifecycle_per_entity")
        for item in sorted(
            selected, key=lambda item: (item.known_at, item.episode_id)
        )
    )


def first_active_displacement_episodes(
    episodes: Sequence[MechanismEpisode],
) -> tuple[MechanismEpisode, ...]:
    """Keep one preregistered first-ACTIVE unit per displacement entity."""

    selected: dict[str, MechanismEpisode] = {}
    for episode in episodes:
        if (
            episode.hypothesis != "displacement_impact"
            or episode.event_kind != "displacement_observed"
            or episode.control_kind is not None
            or not episode.entity_id
        ):
            raise Phase6ResearchError(
                "active displacement episode lacks a stable entity identity"
            )
        prior = selected.get(episode.entity_id)
        if prior is None or (episode.known_at, episode.episode_id) < (
            prior.known_at,
            prior.episode_id,
        ):
            selected[episode.entity_id] = episode
    return tuple(
        replace(item, statistical_unit="first_active_per_displacement_entity")
        for item in sorted(
            selected.values(), key=lambda item: (item.known_at, item.episode_id)
        )
    )


def canonicalize_mechanism_episodes(
    episodes: Sequence[MechanismEpisode],
    *,
    analysis_hypothesis: str,
    control_kind: str | None,
) -> tuple[tuple[MechanismEpisode, ...], tuple[dict[str, Any], ...]]:
    """Collapse same-clock windows and exclude opposite-direction ambiguity.

    This is a research projection only. Immutable Trading-Eye semantic events
    remain auditable through the sorted constituent identities.
    """

    if analysis_hypothesis not in PHASE6_FIXED_FAMILY:
        raise Phase6ResearchError("canonical analysis hypothesis is invalid")
    base_groups: dict[tuple[Any, ...], list[MechanismEpisode]] = {}
    for episode in episodes:
        base_key = (
            episode.symbol,
            episode.instrument_id,
            episode.timeframe,
            analysis_hypothesis,
            episode.event_variant or "none",
            control_kind or "treatment",
            episode.known_at,
        )
        base_groups.setdefault(base_key, []).append(episode)

    excluded_ids: set[str] = set()
    exclusions: list[dict[str, Any]] = []
    for key, members in sorted(base_groups.items(), key=lambda item: repr(item[0])):
        directions = tuple(sorted({item.direction for item in members}))
        if len(directions) <= 1:
            continue
        member_ids = tuple(sorted(item.episode_id for item in members))
        excluded_ids.update(member_ids)
        exclusions.append(
            {
                "hypothesis": analysis_hypothesis,
                "control_kind": control_kind,
                "known_at": key[-1],
                "directions": directions,
                "constituent_event_ids": member_ids,
                "reason": "ambiguous_opposite_direction_same_clock",
            }
        )

    eligible = [item for item in episodes if item.episode_id not in excluded_ids]
    records = [
        {
            "event_id": item.episode_id,
            "symbol": item.symbol,
            "instrument_id": item.instrument_id,
            "timeframe": item.timeframe,
            "hypothesis": analysis_hypothesis,
            "event_variant": item.event_variant or "none",
            "control_kind": control_kind or "treatment",
            "known_at": item.known_at,
            "direction": item.direction,
        }
        for item in eligible
    ]
    projections = canonical_treatment_episodes(
        records,
        identity_fields=(
            "symbol",
            "instrument_id",
            "timeframe",
            "hypothesis",
            "event_variant",
            "control_kind",
            "known_at",
            "direction",
        ),
    )
    by_id = {item.episode_id: item for item in eligible}
    canonical: list[MechanismEpisode] = []
    for projection in projections:
        member_ids = tuple(str(value) for value in projection["constituent_event_ids"])
        members = tuple(by_id[value] for value in member_ids)
        representative = min(members, key=lambda item: item.episode_id)
        source_by_id: dict[str, pd.Timestamp] = {}
        for member in members:
            for source_id, source_clock in zip(
                member.source_m5_bar_event_ids,
                member.source_bar_clocks,
                strict=True,
            ):
                prior_clock = source_by_id.setdefault(source_id, source_clock)
                if prior_clock != source_clock:
                    raise Phase6ResearchError(
                        "canonical source M5 identity maps to multiple clocks"
                    )
        source_pairs = tuple(
            sorted(source_by_id.items(), key=lambda item: (item[1], item[0]))
        )
        source_ids = tuple(item[0] for item in source_pairs)
        source_clocks = tuple(item[1] for item in source_pairs)
        context_clock = strict_prior_match_context_clock(source_clocks)
        context_members = tuple(
            item for item in members if item.match_context_clock == context_clock
        )
        if not context_members:
            raise Phase6ResearchError(
                "canonical episode lacks earliest-formation match context"
            )
        context_fields = dict(context_members[0].match_fields)
        if any(
            dict(item.match_fields) != context_fields
            for item in context_members[1:]
        ):
            raise Phase6ResearchError(
                "canonical earliest-formation match contexts disagree"
            )
        if len({item.session_phase for item in members}) != 1:
            raise Phase6ResearchError("same-clock canonical session phases disagree")
        constituent_ids = tuple(
            sorted(
                {
                    value
                    for member in members
                    for value in member.constituent_event_ids
                }
            )
        )
        entity_ids = tuple(
            sorted(
                {
                    value
                    for member in members
                    for value in member.constituent_entity_ids
                }
            )
        )
        scores = tuple(
            float(item.score) for item in members if item.score is not None
        )
        canonical.append(
            replace(
                representative,
                episode_id=str(projection["event_id"]),
                hypothesis=analysis_hypothesis,
                control_kind=control_kind,
                entity_id=entity_ids[0] if len(entity_ids) == 1 else None,
                event_time=min(item.event_time for item in members),
                source_bar_clocks=source_clocks,
                source_m5_bar_event_ids=source_ids,
                match_context_clock=context_clock,
                match_fields=context_fields,
                score=max(scores) if scores else None,
                constituent_event_ids=constituent_ids,
                constituent_entity_ids=entity_ids,
                statistical_unit="canonical_phase6_clock_episode",
            )
        )
    canonical.sort(key=lambda item: (item.known_at, item.episode_id))
    return tuple(canonical), tuple(exclusions)


def validate_minute_feature_frame(
    frame: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    symbol: str,
    instrument_id: int,
    expected_rows: int,
) -> pd.DataFrame:
    """Validate exact clocks, identity and as-of causality for minute features."""

    missing = REQUIRED_MINUTE_FEATURE_COLUMNS - set(frame.columns)
    if missing:
        raise Phase6ResearchError(
            f"MBO mechanism artifact lacks columns: {sorted(missing)}"
        )
    value = frame.copy()
    for name in ("decision_time", "book_observed_at"):
        value[name] = pd.to_datetime(value[name], utc=True, errors="raise")
    start_utc = _aware_utc(start, name="feature.start")
    end_utc = _aware_utc(end_exclusive, name="feature.end_exclusive")
    value = value[
        (value["decision_time"] >= start_utc)
        & (value["decision_time"] < end_utc)
    ].copy()
    value.sort_values("decision_time", inplace=True, kind="mergesort")
    value.reset_index(drop=True, inplace=True)
    if len(value) != int(expected_rows):
        raise Phase6ResearchError(
            f"MBO minute clock census changed: {len(value)} != {expected_rows}"
        )
    if value["decision_time"].duplicated().any():
        raise Phase6ResearchError("MBO minute artifact has duplicate clocks")
    if set(value["symbol"].astype(str)) != {symbol} or set(
        value["instrument_id"].astype(int)
    ) != {int(instrument_id)}:
        raise Phase6ResearchError("MBO minute artifact contract identity changed")
    for name in ("book_valid", "book_change_valid"):
        if not value[name].map(lambda item: type(item) in {bool, np.bool_}).all():
            raise Phase6ResearchError(f"{name} must be boolean")
    if not bool(value["book_valid"].all()):
        raise Phase6ResearchError("registered Phase-6 minute contains invalid book")
    if value["book_observed_at"].isna().any() or bool(
        (value["book_observed_at"] > value["decision_time"]).any()
    ):
        raise Phase6ResearchError("MBO book observation uses a future clock")
    numeric = REQUIRED_MINUTE_FEATURE_COLUMNS - {
        "decision_time",
        "symbol",
        "instrument_id",
        "book_observed_at",
        "book_valid",
        "book_change_valid",
    }
    change_metrics = {
        "best_level_ofi_contracts",
        "mid_change_ticks",
    }
    for name in numeric:
        values = pd.to_numeric(value[name], errors="coerce")
        invalid_change = ~value["book_change_valid"] if name in change_metrics else False
        required_values = values[~invalid_change] if name in change_metrics else values
        if required_values.isna().any() or not np.isfinite(
            required_values.to_numpy(dtype=float)
        ).all():
            raise Phase6ResearchError(f"MBO minute feature is non-finite: {name}")
        if name in change_metrics and values[invalid_change].notna().any():
            raise Phase6ResearchError(
                f"invalid first-clock BBO delta must remain null: {name}"
            )
        value[name] = values.astype(float)
    if not value["book_valid_clock_fraction"].between(0.0, 1.0).all():
        raise Phase6ResearchError("book_valid_clock_fraction is outside [0,1]")
    return value


def validate_registered_synthetic_mbo_flow(
    frame: pd.DataFrame,
    synthetic_decision_clocks: Sequence[pd.Timestamp],
) -> None:
    """Fail closed unless registered clock-only OHLCV rows have zero T/F flow."""

    clocks = tuple(
        sorted(
            {
                _aware_utc(value, name="synthetic MBO decision clock")
                for value in synthetic_decision_clocks
            }
        )
    )
    if len(clocks) != len(tuple(synthetic_decision_clocks)):
        raise Phase6ResearchError("synthetic MBO clock identities are duplicated")
    if not clocks:
        return
    rows = frame.loc[frame["decision_time"].isin(clocks)]
    if tuple(rows["decision_time"]) != clocks:
        raise Phase6ResearchError("registered synthetic MBO clock row is missing")
    zero_fields = (
        "aggressor_buy_volume",
        "aggressor_sell_volume",
        "aggressor_unknown_volume",
        "aggressor_buy_trade_count",
        "aggressor_sell_trade_count",
        "aggressor_unknown_trade_count",
        "passive_bid_fill_volume",
        "passive_ask_fill_volume",
        "passive_unknown_fill_volume",
        "passive_bid_fill_count",
        "passive_ask_fill_count",
        "passive_unknown_fill_count",
    )
    if any(not bool((rows[name] == 0.0).all()) for name in zero_fields):
        raise Phase6ResearchError(
            "registered synthetic OHLCV clock contains nonzero MBO trade/fill flow"
        )


@dataclass(frozen=True)
class EpisodeFeatureWindow:
    episode_id: str
    formation_clocks: tuple[pd.Timestamp, ...]
    pre_clocks: tuple[pd.Timestamp, ...]
    post_clocks: tuple[pd.Timestamp, ...]
    formation_available_at: pd.Timestamp
    post_available_at: pd.Timestamp
    post_retrospective_only: bool
    metrics: Mapping[str, float]


def _expanded_source_m1_clocks(
    source_bar_clocks: Sequence[pd.Timestamp],
) -> tuple[pd.Timestamp, ...]:
    clocks: set[pd.Timestamp] = set()
    for raw in source_bar_clocks:
        end = _aware_utc(raw, name="source M5 BAR clock")
        for offset in range(4, -1, -1):
            clocks.add(end - pd.Timedelta(minutes=offset))
    return tuple(sorted(clocks))


def _window_metrics(frame: pd.DataFrame, *, direction: str) -> dict[str, float]:
    sign = 1.0 if direction == "long" else -1.0
    buy = float(frame["aggressor_buy_volume"].sum())
    sell = float(frame["aggressor_sell_volume"].sum())
    unknown = float(frame["aggressor_unknown_volume"].sum())
    known_trade = buy + sell
    total_trade = known_trade + unknown
    imbalance = 0.0 if known_trade <= 0.0 else (buy - sell) / known_trade
    ofi = float(frame["best_level_ofi_contracts"].sum())
    mid_change = float(frame["mid_change_ticks"].sum())
    if direction == "long":
        defense_add = float(frame["displayed_bid_add_volume"].sum())
        defense_cancel = float(frame["displayed_bid_cancel_volume"].sum())
        defense_fill = float(frame["passive_bid_fill_volume"].sum())
    else:
        defense_add = float(frame["displayed_ask_add_volume"].sum())
        defense_cancel = float(frame["displayed_ask_cancel_volume"].sum())
        defense_fill = float(frame["passive_ask_fill_volume"].sum())
    denominator = max(total_trade, 1.0)
    return {
        "aggressor_trade_volume": total_trade,
        "directional_aggressor_imbalance": sign * imbalance,
        "directional_best_level_ofi_per_contract": sign * ofi / denominator,
        "directional_mid_impact_ticks_per_contract": (
            sign * mid_change / denominator
        ),
        # This is an all-book displayed-flow proxy. It is intentionally not
        # named queue replenishment or absorption: A/C/F aggregates cannot
        # prove that the same best-level queue was replenished.
        "displayed_defense_net_add_per_contract": (
            defense_add - defense_cancel - defense_fill
        ) / denominator,
        "defensive_passive_fill_fraction": defense_fill / denominator,
        "mean_spread_ticks": float(frame["spread_ticks"].mean()),
        "book_valid_clock_fraction": float(
            frame["book_valid_clock_fraction"].mean()
        ),
    }


def aggregate_episode_feature_window(
    feature_frame: pd.DataFrame,
    episode: MechanismEpisode,
    *,
    pre_context_minutes: int = 5,
    post_response_minutes: int = 5,
) -> EpisodeFeatureWindow:
    """Aggregate exact source clocks without making post-event data causal."""

    if pre_context_minutes != 5 or post_response_minutes != 5:
        raise Phase6ResearchError("Phase-6 feature windows are frozen at five minutes")
    if tuple(feature_frame["decision_time"]) != tuple(
        sorted(feature_frame["decision_time"])
    ):
        raise Phase6ResearchError("feature frame must be in completed-clock order")
    by_clock = feature_frame.set_index("decision_time", drop=False)
    formation = _expanded_source_m1_clocks(episode.source_bar_clocks)
    first_formation = formation[0]
    pre = tuple(
        first_formation - pd.Timedelta(minutes=offset)
        for offset in range(pre_context_minutes, 0, -1)
    )
    post = tuple(
        episode.known_at + pd.Timedelta(minutes=offset)
        for offset in range(1, post_response_minutes + 1)
    )
    if formation[-1] > episode.known_at or any(
        clock > episode.known_at for clock in pre
    ):
        raise Phase6ResearchError("formation/pre window leaks beyond known_at")
    required = (*pre, *formation, *post)
    missing = tuple(clock for clock in required if clock not in by_clock.index)
    if missing:
        raise EpisodeWindowUnavailable(
            "missing_completed_feature_clock",
            episode_id=episode.episode_id,
        )
    identity = by_clock.loc[list(required), ["symbol", "instrument_id"]]
    if set(identity["symbol"].astype(str)) != {episode.symbol} or set(
        identity["instrument_id"].astype(int)
    ) != {episode.instrument_id}:
        raise Phase6ResearchError("episode feature window crosses contract identity")
    groups = {
        "pre": by_clock.loc[list(pre)],
        "formation": by_clock.loc[list(formation)],
        "post": by_clock.loc[list(post)],
    }
    if any(not bool(window["book_change_valid"].all()) for window in groups.values()):
        raise EpisodeWindowUnavailable(
            "book_change_invalid_in_required_window",
            episode_id=episode.episode_id,
        )
    metrics: dict[str, float] = {}
    for prefix, window in groups.items():
        values = _window_metrics(window, direction=episode.direction)
        metrics.update({f"{prefix}_{name}": value for name, value in values.items()})
    metrics["response_aggressor_reversal"] = (
        metrics["post_directional_aggressor_imbalance"]
        - metrics["formation_directional_aggressor_imbalance"]
    )
    metrics["response_directional_ofi_shift"] = (
        metrics["post_directional_best_level_ofi_per_contract"]
        - metrics["formation_directional_best_level_ofi_per_contract"]
    )
    return EpisodeFeatureWindow(
        episode_id=episode.episode_id,
        formation_clocks=formation,
        pre_clocks=pre,
        post_clocks=post,
        formation_available_at=episode.known_at,
        post_available_at=post[-1],
        post_retrospective_only=True,
        metrics=MappingProxyType(metrics),
    )


def match_mechanism_controls(
    treatments: Sequence[MechanismEpisode],
    candidates: Sequence[MechanismEpisode],
    *,
    completed_index: Mapping[pd.Timestamp, int],
    exact_fields: Sequence[str],
    maximum_completed_minute_offset: int,
    post_response_minutes: int = 5,
    embargo_minutes: int = 5,
) -> MatchResult:
    """Deterministically match non-overlapping, forward causal controls."""

    fields = tuple(exact_fields)
    if not {
        "symbol",
        "instrument_id",
        "study_week",
        "half_week",
        "timeframe",
        "session_phase",
    }.issubset(fields):
        raise Phase6ResearchError(
            "Phase-6 controls must match contract, week/half, timeframe and session"
        )

    def record(episode: MechanismEpisode, *, candidate: bool) -> dict[str, Any]:
        value = {
            "event_id": episode.episode_id,
            "candidate_id": episode.episode_id,
            "known_at": episode.known_at,
            "symbol": episode.symbol,
            "instrument_id": episode.instrument_id,
            "timeframe": episode.timeframe,
            "session_phase": episode.session_phase,
            "direction": episode.direction,
            **dict(episode.match_fields),
        }
        if candidate:
            value["direction_known_at"] = episode.known_at
        return value

    return deterministic_maximum_cardinality_match(
        [record(item, candidate=False) for item in treatments],
        [record(item, candidate=True) for item in candidates],
        completed_index=completed_index,
        spec=MatchSpec(
            exact_fields=fields,
            maximum_completed_bar_offset=maximum_completed_minute_offset,
            outcome_horizon_completed_bars=post_response_minutes,
            embargo_completed_bars=embargo_minutes,
            forward_only=True,
            replacement_limit=1,
            direction_policy=ControlDirectionPolicy.CANDIDATE_LOCAL,
        ),
    )


@dataclass(frozen=True)
class PackedMechanismMatch:
    """Stable earliest-first matches with hypothesis-local clock packing."""

    pairs: tuple[MatchedControlPair, ...]
    unmatched: Mapping[str, str]
    requested: int
    window_eligible: int
    eligible_candidates: int
    overlap_exclusions: int

    @property
    def matched(self) -> int:
        return len(self.pairs)


def pack_nonoverlapping_mechanism_controls(
    treatments: Sequence[MechanismEpisode],
    candidates: Sequence[MechanismEpisode],
    *,
    windows: Mapping[str, EpisodeFeatureWindow],
    completed_index: Mapping[pd.Timestamp, int],
    exact_fields: Sequence[str],
    maximum_completed_minute_offset: int,
    embargo_minutes: int = 5,
) -> PackedMechanismMatch:
    """Greedily pack pairs whose formation/post clocks are never reused.

    Packing is deterministic and earliest-treatment/earliest-control first. It
    is deliberately not described as maximum-cardinality matching. The clock
    occupancy scope is one hypothesis invocation; Holm-family hypotheses may
    share clocks with each other.
    """

    fields = tuple(exact_fields)
    required = {
        "symbol",
        "instrument_id",
        "study_week",
        "half_week",
        "timeframe",
        "session_phase",
        "direction",
        "volatility_bucket",
        "trend_relation",
        "relative_volume_bucket",
    }
    if set(fields) != required or len(fields) != len(required):
        raise Phase6ResearchError("Phase-6 packed exact-match fields changed")
    if maximum_completed_minute_offset != 3000 or embargo_minutes != 5:
        raise Phase6ResearchError("Phase-6 packed matching caliper changed")
    positions = sorted(int(value) for value in completed_index.values())
    if (
        not positions
        or len(positions) != len(set(positions))
        or positions != list(range(positions[0], positions[-1] + 1))
    ):
        raise Phase6ResearchError("packed matching completed index is invalid")

    def field_value(episode: MechanismEpisode, field: str) -> Any:
        if field in {
            "symbol",
            "instrument_id",
            "timeframe",
            "session_phase",
            "direction",
        }:
            return getattr(episode, field)
        if field not in episode.match_fields:
            raise Phase6ResearchError(f"episode lacks packed match field: {field}")
        return episode.match_fields[field]

    def stratum(episode: MechanismEpisode) -> tuple[Any, ...]:
        return tuple(field_value(episode, field) for field in fields)

    def used_clocks(episode: MechanismEpisode) -> frozenset[pd.Timestamp]:
        window = windows.get(episode.episode_id)
        if window is None:
            raise Phase6ResearchError("packed episode lacks a feature window")
        clocks = frozenset((*window.formation_clocks, *window.post_clocks))
        if not clocks or any(clock not in completed_index for clock in clocks):
            raise Phase6ResearchError(
                "packed inference window is outside the completed index"
            )
        return clocks

    ordered_treatments = tuple(
        sorted(treatments, key=lambda item: (item.known_at, item.episode_id))
    )
    ordered_candidates = tuple(
        sorted(candidates, key=lambda item: (item.known_at, item.episode_id))
    )
    candidate_strata: dict[tuple[Any, ...], list[MechanismEpisode]] = {}
    for candidate in ordered_candidates:
        candidate_strata.setdefault(stratum(candidate), []).append(candidate)

    occupied: set[pd.Timestamp] = set()
    used_candidate_ids: set[str] = set()
    pairs: list[MatchedControlPair] = []
    unmatched: dict[str, str] = {}
    window_eligible = 0
    overlap_exclusions = 0
    for treatment in ordered_treatments:
        treatment_index = completed_index.get(treatment.known_at)
        if treatment_index is None:
            unmatched[treatment.episode_id] = "treatment_clock_not_completed"
            continue
        treatment_clocks = used_clocks(treatment)
        treatment_last = max(completed_index[clock] for clock in treatment_clocks)
        same_stratum = candidate_strata.get(stratum(treatment), ())
        edges: list[tuple[int, pd.Timestamp, str, MechanismEpisode, frozenset[pd.Timestamp]]] = []
        for candidate in same_stratum:
            candidate_index = completed_index.get(candidate.known_at)
            if candidate_index is None:
                continue
            offset = int(candidate_index - treatment_index)
            if offset <= 0 or offset > maximum_completed_minute_offset:
                continue
            candidate_clocks = used_clocks(candidate)
            candidate_first = min(
                completed_index[clock] for clock in candidate_clocks
            )
            # Exact formation/post windows must be disjoint with five unused
            # completed clocks between treatment response and control formation.
            if (
                treatment_clocks.intersection(candidate_clocks)
                or candidate_first - treatment_last - 1 < embargo_minutes
            ):
                continue
            edges.append(
                (
                    offset,
                    candidate.known_at,
                    candidate.episode_id,
                    candidate,
                    candidate_clocks,
                )
            )
        edges.sort(key=lambda item: (item[0], item[1], item[2]))
        if not edges:
            unmatched[treatment.episode_id] = (
                "no_exact_stratum"
                if not same_stratum
                else "no_exact_window_nonoverlap_embargo_edge"
            )
            continue
        window_eligible += 1
        selected: tuple[
            int,
            pd.Timestamp,
            str,
            MechanismEpisode,
            frozenset[pd.Timestamp],
        ] | None = None
        for edge in edges:
            candidate = edge[3]
            pair_clocks = treatment_clocks.union(edge[4])
            if (
                candidate.episode_id in used_candidate_ids
                or pair_clocks.intersection(occupied)
            ):
                continue
            selected = edge
            break
        if selected is None:
            unmatched[treatment.episode_id] = (
                "global_inference_clock_overlap_or_candidate_capacity"
            )
            overlap_exclusions += 1
            continue
        offset, _, candidate_id, candidate, candidate_clocks = selected
        occupied.update(treatment_clocks)
        occupied.update(candidate_clocks)
        used_candidate_ids.add(candidate_id)
        pairs.append(
            MatchedControlPair(
                treatment_id=treatment.episode_id,
                candidate_id=candidate_id,
                treatment=MappingProxyType({"event_id": treatment.episode_id}),
                candidate=MappingProxyType({"candidate_id": candidate_id}),
                completed_bar_offset=offset,
                control_direction=candidate.direction,
                direction_known_at=candidate.known_at,
            )
        )
    return PackedMechanismMatch(
        pairs=tuple(pairs),
        unmatched=MappingProxyType(unmatched),
        requested=len(ordered_treatments),
        window_eligible=window_eligible,
        eligible_candidates=len(ordered_candidates),
        overlap_exclusions=overlap_exclusions,
    )


@dataclass(frozen=True)
class PairedEffect:
    paired_n: int
    treatment_mean: float | None
    control_mean: float | None
    mean_effect: float | None
    ci_low: float | None
    ci_high: float | None
    positive: int
    negative: int
    ties: int
    exact_sign_p_value: float | None


def paired_effect(
    treatment: Sequence[float | None],
    control: Sequence[float | None],
    *,
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 20240602,
) -> PairedEffect:
    """Compute a deterministic paired effect, CI and exact two-sided sign test."""

    if len(treatment) != len(control):
        raise Phase6ResearchError("paired effect inputs have different lengths")
    pairs: list[tuple[float, float]] = []
    for left, right in zip(treatment, control):
        if left is None or right is None:
            continue
        left_value, right_value = float(left), float(right)
        if math.isfinite(left_value) and math.isfinite(right_value):
            pairs.append((left_value, right_value))
    if not pairs:
        return PairedEffect(0, None, None, None, None, None, 0, 0, 0, None)
    left = np.asarray([item[0] for item in pairs], dtype=float)
    right = np.asarray([item[1] for item in pairs], dtype=float)
    difference = left - right
    positive = int(np.sum(difference > 0.0))
    negative = int(np.sum(difference < 0.0))
    ties = int(np.sum(difference == 0.0))
    discordant = positive + negative
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(positive, negative)
        numerator = sum(math.comb(discordant, index) for index in range(tail + 1))
        p_value = min(1.0, 2.0 * numerator / (1 << discordant))
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, Integral)
        or int(bootstrap_replicates) < 1000
    ):
        raise Phase6ResearchError("paired bootstrap requires at least 1000 replicates")
    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(0, len(difference), size=(int(bootstrap_replicates), len(difference)))
    bootstrap = difference[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return PairedEffect(
        paired_n=len(pairs),
        treatment_mean=float(left.mean()),
        control_mean=float(right.mean()),
        mean_effect=float(difference.mean()),
        ci_low=float(low),
        ci_high=float(high),
        positive=positive,
        negative=negative,
        ties=ties,
        exact_sign_p_value=float(p_value),
    )


def spearman_monotonicity(
    score: Sequence[float | None],
    response: Sequence[float | None],
) -> dict[str, Any]:
    """Return descriptive rank monotonicity without selecting a threshold."""

    pairs = [
        (float(left), float(right))
        for left, right in zip(score, response)
        if left is not None
        and right is not None
        and math.isfinite(float(left))
        and math.isfinite(float(right))
    ]
    if len(pairs) < 2:
        return {"n": len(pairs), "spearman_rho": None, "threshold_selected": False}
    left_rank = pd.Series([item[0] for item in pairs]).rank(method="average").to_numpy()
    right_rank = pd.Series([item[1] for item in pairs]).rank(method="average").to_numpy()
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        rho = None
    else:
        rho = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return {"n": len(pairs), "spearman_rho": rho, "threshold_selected": False}


def _effect_to_dict(value: PairedEffect) -> dict[str, Any]:
    return {
        "paired_n": value.paired_n,
        "treatment_mean": value.treatment_mean,
        "control_mean": value.control_mean,
        "mean_effect": value.mean_effect,
        "ci_low": value.ci_low,
        "ci_high": value.ci_high,
        "positive": value.positive,
        "negative": value.negative,
        "ties": value.ties,
        "exact_sign_p_value": value.exact_sign_p_value,
    }


def evaluate_descriptive_sensitivity(
    rows: Sequence[Mapping[str, Any]],
    *,
    hypothesis: str,
    comparison: str,
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 20240602,
) -> dict[str, Any]:
    """Summarize a separately matched sensitivity comparison without admission.

    The result deliberately omits the exact-sign p-value. It is outside the
    fixed Holm family and can neither support a mechanism nor enter the Phase-7
    evidence allowlist.
    """

    if hypothesis not in PHASE6_FIXED_FAMILY:
        raise Phase6ResearchError("descriptive sensitivity hypothesis is invalid")
    primary_name = PRIMARY_METRIC_BY_HYPOTHESIS[hypothesis]
    effect = paired_effect(
        [item.get("treatment_metrics", {}).get(primary_name) for item in rows],
        [item.get("control_metrics", {}).get(primary_name) for item in rows],
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "comparison": comparison,
        "hypothesis": hypothesis,
        "primary_metric": primary_name,
        "paired_n": effect.paired_n,
        "treatment_mean": effect.treatment_mean,
        "control_mean": effect.control_mean,
        "mean_effect": effect.mean_effect,
        "descriptive_bootstrap_ci_low": effect.ci_low,
        "descriptive_bootstrap_ci_high": effect.ci_high,
        "holm_included": False,
        "phase7_evidence_admission": False,
        "support_verdict": "not_applicable_descriptive_only",
        "causal_claim": False,
    }


def evaluate_fixed_mechanism_family(
    paired_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    minimum_matched: int = 30,
    alpha: float = 0.05,
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 20240602,
    stability_stratum_minimum_n: int = 5,
) -> dict[str, Any]:
    """Evaluate all five hypotheses and return a strict Phase-7 allowlist."""

    if set(paired_rows) != set(PHASE6_FIXED_FAMILY):
        raise Phase6ResearchError("paired rows must contain the fixed five-hypothesis family")
    preliminary: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float | None] = {}
    for family_index, hypothesis in enumerate(PHASE6_FIXED_FAMILY):
        rows = tuple(paired_rows[hypothesis])
        primary_name = PRIMARY_METRIC_BY_HYPOTHESIS[hypothesis]
        treatment = [
            item.get("treatment_metrics", {}).get(primary_name) for item in rows
        ]
        control = [item.get("control_metrics", {}).get(primary_name) for item in rows]
        primary = paired_effect(
            treatment,
            control,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + family_index,
        )
        secondary: dict[str, dict[str, Any]] = {}
        for metric_index, metric in enumerate(
            SECONDARY_METRICS_BY_HYPOTHESIS[hypothesis]
        ):
            effect = paired_effect(
                [item.get("treatment_metrics", {}).get(metric) for item in rows],
                [item.get("control_metrics", {}).get(metric) for item in rows],
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + 100 + family_index * 10 + metric_index,
            )
            secondary[metric] = _effect_to_dict(effect)

        stratum_effects: dict[str, dict[str, dict[str, Any]]] = {}
        stability_complete = True
        systematic_reversal = False
        for field in ("study_week_x_half_week", "session_phase"):
            by_stratum: dict[str, list[Mapping[str, Any]]] = {}
            for item in rows:
                if field == "study_week_x_half_week":
                    treatment_week = item.get("treatment_study_week")
                    control_week = item.get("control_study_week")
                    treatment_half = item.get("treatment_half_week")
                    control_half = item.get("control_half_week")
                    if (
                        treatment_week != control_week
                        or treatment_half != control_half
                        or not isinstance(treatment_week, str)
                        or not treatment_week
                        or not isinstance(treatment_half, str)
                        or not treatment_half
                    ):
                        raise Phase6ResearchError(
                            "paired stratum mismatch for study_week_x_half_week"
                        )
                    label = f"{treatment_week}:{treatment_half}"
                else:
                    treatment_label = item.get(
                        f"treatment_{field}", item.get(field)
                    )
                    control_label = item.get(
                        f"control_{field}", treatment_label
                    )
                    if treatment_label != control_label:
                        raise Phase6ResearchError(
                            f"paired stratum mismatch for {field}"
                        )
                    label = treatment_label
                if isinstance(label, str) and label:
                    by_stratum.setdefault(label, []).append(item)
            values: dict[str, dict[str, Any]] = {}
            eligible = 0
            for label, subset in sorted(by_stratum.items()):
                effect = paired_effect(
                    [
                        item.get("treatment_metrics", {}).get(primary_name)
                        for item in subset
                    ],
                    [
                        item.get("control_metrics", {}).get(primary_name)
                        for item in subset
                    ],
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=(
                        bootstrap_seed
                        + 1000
                        + family_index * 100
                        + len(values)
                    ),
                )
                is_eligible = effect.paired_n >= stability_stratum_minimum_n
                if is_eligible:
                    eligible += 1
                    systematic_reversal = systematic_reversal or bool(
                        effect.mean_effect is not None and effect.mean_effect < 0.0
                    )
                values[label] = {
                    **_effect_to_dict(effect),
                    "stability_eligible": is_eligible,
                }
            # Half-week labels are week-qualified so an extension cannot hide
            # a reversal by pooling identically named halves across weeks.
            # Session checks retain the registered cross-week phase summary.
            required_eligible = 2 if field == "study_week_x_half_week" else 1
            if eligible < required_eligible:
                stability_complete = False
            stratum_effects[field] = values

        p_values[hypothesis] = (
            primary.exact_sign_p_value
            if primary.paired_n >= minimum_matched
            else None
        )
        preliminary[hypothesis] = {
            "primary_metric": primary_name,
            "primary": _effect_to_dict(primary),
            "secondary": secondary,
            "strata": stratum_effects,
            "stability_complete": stability_complete,
            "systematic_sign_reversal": systematic_reversal,
        }

    holm = holm_adjust_fixed_family(
        p_values,
        family_order=PHASE6_FIXED_FAMILY,
        alpha=alpha,
    )
    mechanisms: dict[str, dict[str, Any]] = {}
    allowlist: list[str] = []
    for hypothesis in PHASE6_FIXED_FAMILY:
        value = preliminary[hypothesis]
        primary = value["primary"]
        n = int(primary["paired_n"])
        secondary_positive = any(
            item["mean_effect"] is not None and item["mean_effect"] > 0.0
            for item in value["secondary"].values()
        )
        ci_expected = bool(
            primary["ci_low"] is not None and primary["ci_low"] > 0.0
        )
        supported = bool(
            n >= minimum_matched
            and holm.rejected[hypothesis]
            and ci_expected
            and secondary_positive
            and value["stability_complete"]
            and not value["systematic_sign_reversal"]
        )
        status = "supported" if supported else (
            "underpowered" if n < minimum_matched else "unsupported"
        )
        mechanisms[hypothesis] = {
            "status": status,
            "primary_matched_n": n,
            "primary_metric": value["primary_metric"],
            "primary_effect": primary,
            "secondary_effects": value["secondary"],
            "holm_adjusted_p_value": holm.adjusted_p_values[hypothesis],
            "holm_rejected": holm.rejected[hypothesis],
            "primary_ci_expected_direction": ci_expected,
            "secondary_expected_direction": secondary_positive,
            "stability_complete": value["stability_complete"],
            "systematic_sign_reversal": value["systematic_sign_reversal"],
            "strata": value["strata"],
            "causal_claim": False,
        }
        if supported:
            allowlist.append(hypothesis)
    return {
        "engineering_status": "pass",
        "mechanisms": mechanisms,
        "holm": {
            "family_order": list(holm.family_order),
            "raw_p_values": dict(holm.raw_p_values),
            "adjusted_p_values": dict(holm.adjusted_p_values),
            "rejected": dict(holm.rejected),
            "missing_as_one": list(holm.missing_as_one),
        },
        "extension_required": any(
            mechanisms[name]["status"] == "underpowered"
            for name in PHASE6_FIXED_FAMILY
        ),
        "phase7_evidence_allowlist": allowlist,
        "phase7_excluded_mechanisms": [
            name for name in PHASE6_FIXED_FAMILY if name not in allowlist
        ],
        "causal_claim": False,
    }


def canonical_identity(value: Mapping[str, Any]) -> str:
    """Return a stable hash while excluding runtime duration and self hash."""

    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): normalize(child)
                for key, child in item.items()
                if key not in {"result_identity", "elapsed_seconds"}
            }
        if isinstance(item, (tuple, list)):
            return [normalize(child) for child in item]
        if isinstance(item, pd.Timestamp):
            return item.isoformat()
        if isinstance(item, np.generic):
            return item.item()
        return item

    payload = json.dumps(
        normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "EpisodeFeatureWindow",
    "EpisodeMatchContextUnavailable",
    "EpisodeWindowUnavailable",
    "EXTENSION_DISPLACEMENT_MONOTONICITY_POLICY",
    "EXTENSION_WARMUP_SYNTHETIC_EXCEPTION_POLICY",
    "FORBIDDEN_FVG_ALIAS",
    "FrozenPhase6Contract",
    "LEGACY_SYNTHETIC_SEMANTIC_EXCEPTION_POLICY",
    "MechanismEpisode",
    "PackedMechanismMatch",
    "PHASE6_EVENT_KINDS",
    "PHASE6_EXECUTABLE_STATUS",
    "PHASE6_FIXED_FAMILY",
    "PHASE6_PROTOCOL_VERSION",
    "PRIMARY_METRIC_BY_HYPOTHESIS",
    "PairedEffect",
    "Phase6ResearchError",
    "Phase6Window",
    "REQUIRED_MINUTE_FEATURE_COLUMNS",
    "REQUIRED_PHASE6_IDENTITY_BINDINGS",
    "SECONDARY_METRICS_BY_HYPOTHESIS",
    "STABILITY_POLICY",
    "STRICT_PRIOR_CONTEXT_CENSOR_REASONS",
    "STRICT_PRIOR_CONTEXT_UNAVAILABLE_FIELD_POLICY",
    "SYNTHETIC_SEMANTIC_EXCEPTION_POLICY",
    "aggregate_episode_feature_window",
    "canonical_identity",
    "canonicalize_mechanism_episodes",
    "evaluate_fixed_mechanism_family",
    "evaluate_descriptive_sensitivity",
    "first_active_displacement_episodes",
    "first_fvg_lifecycle_episodes",
    "load_frozen_phase6_contract",
    "match_mechanism_controls",
    "pack_nonoverlapping_mechanism_controls",
    "paired_effect",
    "spearman_monotonicity",
    "strict_prior_match_context_clock",
    "validate_minute_feature_frame",
    "validate_phase6_design",
    "validate_prior_result_ledgers",
    "validate_registered_synthetic_mbo_flow",
    "validate_underpowered_extension_gate",
]
