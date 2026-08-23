#!/usr/bin/env python3
"""Validate the frozen design-only 2024-06 Week4 branching diagnostic.

This preflight never opens OHLCV, MBO, BBO, event ledgers, or result files.  It
exists to keep the sensitivity grid and claim limits reviewable while the
executable artifact/runtime bindings remain intentionally absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smc_trader.signal_research import (  # noqa: E402
    ResearchContractError,
    ResearchLinkMode,
    TypedLinkSpec,
)


DEFAULT_MANIFEST = ROOT / (
    "experiments/manifests/"
    "smc_semantics_v1_2_2024_06_week4_temporal_branching_construct_v1.yaml"
)
DESIGN_STATUS = (
    "preregistered_development_design_execution_bindings_pending_"
    "not_authorized_to_run"
)
EXPECTED_AUTHORITY = {
    "development_reuse_only": True,
    "diagnostic_association_only": True,
    "causal_proof": False,
    "artifact_fit_allowed": False,
    "production_threshold_selection_allowed": False,
    "trading_authority": False,
    "oos_claim_allowed": False,
    "sealed_holdout_opened": False,
}
EXPECTED_WINDOW = {
    "window_id": "2024-06-week-4",
    "start": "2024-06-23T22:00:00Z",
    "end_exclusive": "2024-06-28T21:01:00Z",
    "expected_completed_m1_clocks": 6900,
    "symbol": "NQU4",
    "instrument_id": 4358,
    "contract_selection": (
        "highest_total_volume_from_strictly_prior_completed_Globex_session"
    ),
}
EXPECTED_GRID = (1, 2, 3)
EXPECTED_ROOT_KINDS = ("acceptance_confirmed", "sweep_confirmed")
EXPECTED_TRANSITION_KINDS = (
    "qualified_bos",
    "mss_core_confirmed",
    "raw_boundary_break",
)
EXPECTED_MASS_CATEGORIES = (
    "unresolved_interaction",
    "response_without_registered_transition",
    "registered_transition_observed",
)
EXPECTED_ROOT_FIELDS = frozenset(
    {
        "analysis_population",
        "authority",
        "branch_graph",
        "claim_scope",
        "excluded_estimands",
        "execution_bindings",
        "experiment_id",
        "frozen_at",
        "frozen_before_outcome_run",
        "input_authority",
        "mbo_response",
        "notes",
        "parameter_search_space",
        "reporting",
        "research_protocol_version",
        "schema_version",
        "selection_policy",
        "semantic_version",
        "status",
        "temporal_sensitivity",
        "window",
    }
)
EXPECTED_DESIGN_IDENTITY = (
    "90b2b87de47f4b03f0fc3bb3f998db28e5c81d4a87f6b985ec0065ea6c80c320"
)


def _object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResearchContractError(f"{name} must be an object")
    return value


def _canonical_identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relation_specs(
    rows: Any,
    *,
    mode: ResearchLinkMode,
) -> tuple[dict[str, TypedLinkSpec], dict[str, str]]:
    if not isinstance(rows, list) or not rows:
        raise ResearchContractError("branch relation family must be nonempty")
    specs: dict[str, TypedLinkSpec] = {}
    stages: dict[str, str] = {}
    for row in rows:
        item = _object(row, name="branch relation")
        relation_id = item.get("relation_id")
        stage = item.get("stage")
        if (
            not isinstance(relation_id, str)
            or not relation_id
            or relation_id in specs
            or stage not in {"interaction_to_response", "response_to_transition"}
        ):
            raise ResearchContractError("branch relation identity/stage is invalid")
        if item.get("mode") != mode.value:
            raise ResearchContractError(f"branch relation mode changed: {relation_id}")
        if mode is ResearchLinkMode.REGISTERED_TEMPORAL_EPISODE:
            if "same_clock_only" in item:
                raise ResearchContractError(
                    f"temporal relation cannot admit same clock: {relation_id}"
                )
        elif item.get("same_clock_only") is not True:
            raise ResearchContractError(
                f"same-clock composition must be exact: {relation_id}"
            )
        spec = TypedLinkSpec(
            previous_kind=str(item.get("previous_kind", "")),
            previous_timeframe=str(item.get("previous_timeframe", "")),
            current_kind=str(item.get("current_kind", "")),
            current_timeframe=str(item.get("current_timeframe", "")),
            maximum_completed_bars=EXPECTED_GRID[-1],
            mode=mode,
            constituent_bar_timeframe=item.get("constituent_bar_timeframe"),
            registered_episode_definition=item.get(
                "registered_episode_definition"
            ),
        )
        expected_fields = {
            "relation_id",
            "stage",
            "previous_kind",
            "current_kind",
            "previous_timeframe",
            "current_timeframe",
            "mode",
            *(
                {"registered_episode_definition"}
                if mode is ResearchLinkMode.REGISTERED_TEMPORAL_EPISODE
                else {"constituent_bar_timeframe", "same_clock_only"}
            ),
        }
        if set(item) != expected_fields:
            raise ResearchContractError(
                f"branch relation fields changed: {relation_id}"
            )
        specs[relation_id] = spec
        stages[relation_id] = str(stage)
    return specs, stages


def validate_design(payload: Mapping[str, Any]) -> None:
    """Validate the outcome-blind design without authorizing execution."""

    if set(payload) != EXPECTED_ROOT_FIELDS:
        raise ResearchContractError("Week4 preregistration root fields changed")
    if (
        payload.get("schema_version") != 1
        or payload.get("research_protocol_version")
        != "temporal_branching_construct_v1"
        or payload.get("status") != DESIGN_STATUS
        or payload.get("frozen_before_outcome_run") is not True
        or payload.get("semantic_version") != "smc_semantics_v1.2"
    ):
        raise ResearchContractError("Week4 preregistration identity is invalid")
    if dict(_object(payload.get("authority"), name="authority")) != EXPECTED_AUTHORITY:
        raise ResearchContractError("Week4 authority must remain development-only")
    if dict(_object(payload.get("window"), name="window")) != EXPECTED_WINDOW:
        raise ResearchContractError("Week4 time/contract census changed")

    input_authority = _object(
        payload.get("input_authority"), name="input_authority"
    )
    if input_authority != {
        "ohlcv_canonical_split_label": "rolling_oof",
        "ohlcv_experiment_role": "previously_opened_reused_development_not_oof",
        "mbo_canonical_split_label": "development",
        "joint_claim_role": "development_diagnostic_not_oof",
        "reason": (
            "2024-06 OHLCV and MBO have already been opened for iterative "
            "semantic and mechanism diagnostics"
        ),
        "rolling_oof_contribution_allowed": False,
        "sealed_oos_contribution_allowed": False,
    }:
        raise ResearchContractError("Week4 data role may not be upgraded to OOF/OOS")

    bindings = _object(payload.get("execution_bindings"), name="execution_bindings")
    if (
        bindings.get("status") != "pending_before_execution"
        or bindings.get("execution_authorized") is not False
        or bindings.get("week4_mbo_mechanism_artifact") is not None
        or bindings.get("week4_mbo_mechanism_manifest") is not None
        or bindings.get("runtime_identity_sha256") is not None
        or bindings.get("raw_partition_sha256_bindings") is not None
        or bindings.get("reader_census") is not None
    ):
        raise ResearchContractError(
            "design-only preregistration cannot contain executable bindings"
        )

    population = _object(payload.get("analysis_population"), name="population")
    if (
        tuple(population.get("root_interaction_kinds", ())) != EXPECTED_ROOT_KINDS
        or population.get("response_kind") != "displacement_observed"
        or tuple(population.get("transition_kinds", ()))
        != EXPECTED_TRANSITION_KINDS
        or population.get("timeframe") != "5m"
        or "already-frozen semantic threshold"
        not in str(population.get("response_predicate"))
    ):
        raise ResearchContractError("Week4 analysis population changed")

    sensitivity = _object(
        payload.get("temporal_sensitivity"), name="temporal_sensitivity"
    )
    if (
        sensitivity.get("clock") != "completed_M5_decision_clock"
        or sensitivity.get("minimum_completed_bars") != 1
        or tuple(sensitivity.get("maximum_completed_bars_grid", ()))
        != EXPECTED_GRID
        or sensitivity.get("same_clock_policy")
        != "separate_exact_constituent_M5_BAR_composition_never_temporal_predecessor"
        or sensitivity.get("window_selection_policy")
        != "report_every_grid_member_no_selection_from_development_results"
        or sensitivity.get("claim")
        != "candidate_temporal_association_not_ancestry_not_causal_proof"
    ):
        raise ResearchContractError("Week4 temporal sensitivity contract changed")

    graph = _object(payload.get("branch_graph"), name="branch_graph")
    temporal, temporal_stages = _relation_specs(
        graph.get("temporal_relations"),
        mode=ResearchLinkMode.REGISTERED_TEMPORAL_EPISODE,
    )
    composition, composition_stages = _relation_specs(
        graph.get("same_clock_composition_relations"),
        mode=ResearchLinkMode.CROSS_TIMEFRAME_CONSTITUENT_BAR,
    )
    if set(temporal).intersection(composition):
        raise ResearchContractError("temporal/composition relation IDs overlap")
    for relation_id, spec in {**temporal, **composition}.items():
        stage = {**temporal_stages, **composition_stages}[relation_id]
        if spec.previous_timeframe != "5m" or spec.current_timeframe != "5m":
            raise ResearchContractError("Week4 relation timeframe changed")
        if stage == "interaction_to_response" and (
            spec.previous_kind not in EXPECTED_ROOT_KINDS
            or spec.current_kind != "displacement_observed"
        ):
            raise ResearchContractError("Week4 interaction/response edge changed")
        if stage == "response_to_transition" and (
            spec.previous_kind != "displacement_observed"
            or spec.current_kind not in EXPECTED_TRANSITION_KINDS
        ):
            raise ResearchContractError("Week4 response/transition edge changed")
    edge_signatures = {
        (spec.previous_kind, spec.current_kind) for spec in temporal.values()
    }
    composition_signatures = {
        (spec.previous_kind, spec.current_kind) for spec in composition.values()
    }
    if edge_signatures != composition_signatures or len(edge_signatures) != 5:
        raise ResearchContractError(
            "every temporal edge requires a separately named same-clock composition"
        )
    if tuple(graph.get("mass_categories", ())) != EXPECTED_MASS_CATEGORIES:
        raise ResearchContractError("Week4 unresolved/no-transition mass changed")
    if graph.get("denominator_policy") != (
        "retain_every_root_in_exactly_one_mass_category_at_every_grid_member"
    ) or graph.get("sibling_branch_policy") != (
        "retain_all_eligible_predecessors_and_all_terminal_siblings_no_single_chain_selection"
    ):
        raise ResearchContractError("Week4 branch retention policy changed")

    mbo = _object(payload.get("mbo_response"), name="mbo_response")
    if (
        tuple(mbo.get("post_response_completed_m5_bars_grid", ())) != EXPECTED_GRID
        or tuple(mbo.get("post_response_completed_m1_minutes_grid", ()))
        != (5, 10, 15)
        or mbo.get("classification_policy")
        != "raw_values_are_primary_any_sign_label_is_descriptive_only_and_never_a_gate"
        or mbo.get("displayed_flow_semantics")
        != "all_book_proxy_not_same_level_queue_replenishment_or_absorption"
        or mbo.get("future_feature_authority")
        != "retrospective_construct_diagnostic_only"
    ):
        raise ResearchContractError("Week4 MBO response contract changed")

    excluded = _object(payload.get("excluded_estimands"), name="excluded_estimands")
    if "true_geometry_derived_first_retest_event" not in str(
        excluded.get("fvg_first_retest")
    ) or any(
        "fvg" in spec.previous_kind.lower() or "fvg" in spec.current_kind.lower()
        for spec in (*temporal.values(), *composition.values())
    ):
        raise ResearchContractError("FVG proxy cannot enter the Week4 construct")
    if payload.get("parameter_search_space") != {
        "maximum_completed_m5_bars": list(EXPECTED_GRID)
    } or payload.get("selection_policy") != (
        "all_preregistered_grid_members_are_reported_no_result_based_window_selection"
    ):
        raise ResearchContractError("Week4 parameter grid/selection policy changed")
    if _canonical_identity(payload) != EXPECTED_DESIGN_IDENTITY:
        raise ResearchContractError("Week4 frozen design identity changed")


def load_and_validate_design(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path).resolve()
    try:
        source.relative_to(ROOT)
    except ValueError as error:
        raise ResearchContractError("manifest must live inside the repository") from error
    if not source.is_file() or source.is_symlink():
        raise ResearchContractError("manifest must be a regular repository file")
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResearchContractError("manifest must use the JSON subset of YAML") from error
    if not isinstance(payload, dict):
        raise ResearchContractError("manifest root must be an object")
    validate_design(payload)
    return payload, hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()
    payload, identity = load_and_validate_design(args.manifest)
    print(
        json.dumps(
            {
                "experiment_id": payload["experiment_id"],
                "manifest_sha256": identity,
                "status": payload["status"],
                "execution_authorized": False,
                "market_data_opened": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
