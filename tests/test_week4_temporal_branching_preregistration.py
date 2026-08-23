from __future__ import annotations

from copy import deepcopy
import json

import pytest

from scripts.validate_week4_temporal_branching_preregistration import (
    DEFAULT_MANIFEST,
    DESIGN_STATUS,
    EXPECTED_GRID,
    load_and_validate_design,
    validate_design,
)
from smc_trader.signal_research import ResearchContractError


def _manifest() -> dict[str, object]:
    value = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_week4_design_is_frozen_but_execution_remains_closed() -> None:
    payload, identity = load_and_validate_design(DEFAULT_MANIFEST)

    assert payload["status"] == DESIGN_STATUS
    assert payload["window"] == {
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
    assert payload["temporal_sensitivity"]["maximum_completed_bars_grid"] == list(
        EXPECTED_GRID
    )
    assert payload["execution_bindings"]["execution_authorized"] is False
    assert payload["execution_bindings"]["week4_mbo_mechanism_artifact"] is None
    assert payload["input_authority"]["joint_claim_role"] == (
        "development_diagnostic_not_oof"
    )
    assert len(identity) == 64


def test_week4_design_rejects_oos_upgrade_or_result_selected_window() -> None:
    oos = deepcopy(_manifest())
    oos["authority"]["oos_claim_allowed"] = True
    with pytest.raises(ResearchContractError, match="development-only"):
        validate_design(oos)

    selected = deepcopy(_manifest())
    selected["temporal_sensitivity"]["maximum_completed_bars_grid"] = [3]
    with pytest.raises(ResearchContractError, match="sensitivity contract"):
        validate_design(selected)


def test_week4_design_keeps_same_clock_composition_out_of_temporal_family() -> None:
    manifest = _manifest()
    graph = manifest["branch_graph"]
    temporal = graph["temporal_relations"]
    composition = graph["same_clock_composition_relations"]

    assert len(temporal) == len(composition) == 5
    assert {item["mode"] for item in temporal} == {"registered_temporal_episode"}
    assert {item["mode"] for item in composition} == {
        "cross_timeframe_constituent_bar_composition"
    }
    assert all(item["same_clock_only"] is True for item in composition)

    conflated = deepcopy(manifest)
    conflated["branch_graph"]["temporal_relations"][0]["mode"] = (
        "cross_timeframe_constituent_bar_composition"
    )
    with pytest.raises(ResearchContractError, match="mode changed"):
        validate_design(conflated)


def test_week4_design_excludes_historical_fvg_lifecycle_proxy() -> None:
    manifest = _manifest()
    assert "true_geometry_derived_first_retest_event" in manifest[
        "excluded_estimands"
    ]["fvg_first_retest"]
    assert all(
        "fvg" not in item["previous_kind"]
        and "fvg" not in item["current_kind"]
        for family in (
            manifest["branch_graph"]["temporal_relations"],
            manifest["branch_graph"]["same_clock_composition_relations"],
        )
        for item in family
    )

    contaminated = deepcopy(manifest)
    contaminated["branch_graph"]["temporal_relations"][0]["current_kind"] = (
        "fvg_midpoint_touched"
    )
    with pytest.raises(ResearchContractError, match="interaction/response"):
        validate_design(contaminated)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__("experiment_id", "renamed-after-freeze"),
        lambda value: value.pop("frozen_at"),
        lambda value: value.__setitem__(
            "claim_scope", "causal proof production threshold"
        ),
        lambda value: value["reporting"].__setitem__(
            "probability_role", "fitted_calibrated_artifact"
        ),
        lambda value: value["reporting"].__setitem__("no_primary_window", False),
        lambda value: value["reporting"].__setitem__(
            "no_effect_driven_extension", False
        ),
        lambda value: value["analysis_population"].__setitem__(
            "direction_policy", "ignore_direction"
        ),
        lambda value: value["analysis_population"].__setitem__(
            "response_predicate", "future_outcome_label_is_true"
        ),
        lambda value: value["mbo_response"].__setitem__(
            "role", "production_known_at_evidence"
        ),
        lambda value: value["mbo_response"].__setitem__(
            "invalid_book_change_policy", "fill_zero"
        ),
        lambda value: value["excluded_estimands"].__setitem__(
            "path_or_dol_artifact_fit", "allowed"
        ),
        lambda value: value.__setitem__("result_selected_threshold", 0.5),
        lambda value: value["execution_bindings"].__setitem__(
            "materialization_allowed", True
        ),
    ),
)
def test_every_frozen_claim_and_authority_field_is_identity_bound(mutation) -> None:
    changed = deepcopy(_manifest())
    mutation(changed)

    with pytest.raises(ResearchContractError):
        validate_design(changed)
