from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import brain.scripts.evaluate_typed_brain_calibration as oos_evaluator
from brain.scripts.fit_typed_brain_calibration import (
    FITTED_DIMENSIONS,
    UNCERTAINTY_FORMULA_VERSION,
    _dimension_fit_diagnostics,
    _fit_admission_funnel,
    _validate_identity_columns,
    fit_typed_brain_calibration,
    resolve_model_bindings,
)
from brain.scripts.evaluate_typed_brain_calibration import (
    evaluate_frozen_typed_brain_calibration,
)
from shares.scripts.run_continuous_replay import BRAIN_CALIBRATION_FIELD_TYPES
from brain.core.calibration import (
    CalibrationError,
    DimensionReliabilityMap,
    DimensionReliabilityPoint,
    TypedBrainCalibrator,
    monotone_reliability_points,
)
from brain.core.brain_calibration import RECORDER_SCHEMA_VERSION
from shares.core.artifact_stream import (
    atomic_parquet,
    new_stream_state,
    sha256_file,
    write_stream_manifest,
)
from shares.core.model import Playbook


ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = ROOT / "configs/model.json"
VALIDATION_PROTOCOL = ROOT / "configs/data_splits.json"
DFP = Playbook.DISPLACEMENT_FIRST_PULLBACK
LSR = Playbook.LIQUIDITY_SWEEP_REVERSAL
FAVR = Playbook.FAILED_AUCTION_VALUE_RETURN


def _identity_fields(
    token: str,
    *,
    playbook: Playbook,
    dimension: str,
    runtime_candidate_token: str | None = None,
) -> dict[str, object]:
    setup_id = f"setup:{token}"
    episode_id = f"episode:{runtime_candidate_token or token}"
    context_id = f"context:{token}"
    entry_path_id = f"path:{token}"
    market_thesis_id = (
        None
        if runtime_candidate_token is None
        else f"market-thesis:{runtime_candidate_token}"
    )
    market_thesis_root_id = (
        None
        if runtime_candidate_token is None
        else f"root:{runtime_candidate_token}"
    )
    if dimension == "thesis_strength":
        calibration_unit_id = context_id
        calibration_unit_kind = (
            "dfp_context_thesis"
            if playbook is DFP
            else "lsr_context_thesis"
        )
        target_deadline_kind = "thesis_deadline"
    elif dimension == "location_quality":
        calibration_unit_id = entry_path_id
        calibration_unit_kind = "entry_path_location"
        target_deadline_kind = "entry_deadline"
    elif dimension in {"entry_readiness", "delivery_quality"}:
        calibration_unit_id = entry_path_id
        calibration_unit_kind = "trigger_entry_path"
        target_deadline_kind = (
            "entry_deadline"
            if dimension == "entry_readiness"
            else "plan_deadline"
        )
    elif dimension == "sequence_progress":
        calibration_unit_id = setup_id
        calibration_unit_kind = "hypothesis_sequence"
        target_deadline_kind = "entry_deadline"
    else:
        calibration_unit_id = episode_id
        calibration_unit_kind = "decision_hypothesis"
        target_deadline_kind = "entry_deadline"
    lsr_context_thesis = bool(
        playbook is LSR and dimension == "thesis_strength"
    )
    return {
        "hypothesis_key": token,
        "setup_id": setup_id,
        "calibration_unit_id": calibration_unit_id,
        "calibration_unit_kind": calibration_unit_kind,
        "episode_id": episode_id,
        "context_id": context_id,
        "context_thesis_id": context_id,
        "parent_context_thesis_id": context_id,
        "evidence_revision_id": f"evidence:{token}",
        "entry_location_id": f"location:{token}",
        "entry_path_id": entry_path_id,
        "selected_trigger_id": None,
        "selected_trigger_kind": None,
        "selected_trigger_at": None,
        "available_trigger_kinds": "[]",
        "market_thesis_id": market_thesis_id,
        "bound_market_thesis_id": market_thesis_id,
        "market_thesis_root_id": market_thesis_root_id,
        "market_thesis_mechanism": (
            None
            if runtime_candidate_token is None
            else (
                "directional_displacement_continuation"
                if playbook is DFP
                else "liquidity_sweep_reversal"
            )
        ),
        "market_thesis_authority_relation": (
            None if runtime_candidate_token is None else "aligned"
        ),
        "playbook_match_strength": (
            0.0 if runtime_candidate_token is None else 0.8
        ),
        "market_thesis_binding_required": runtime_candidate_token is not None,
        "market_thesis_action_bound": runtime_candidate_token is not None,
        "market_thesis_match_status": (
            "not_required"
            if runtime_candidate_token is None
            else "exact_root_bound"
        ),
        "playbook_first_failed_hard_gate_id": None,
        "playbook_plan_delivery_valid": runtime_candidate_token is not None,
        "global_market_mode": "directional",
        "authority_relation": "unrelated",
        "authority_rank_gap": 0,
        "conflict_role": "none",
        "conflict_scope": "none",
        "acceptance_state": "unknown",
        "obstruction_distance_R": None,
        "free_path_R": None,
        "soft_obstruction_count": 0,
        "hard_barrier_before_target": False,
        "ambiguity_count": 0,
        "uncertainty_conflict": 0.0,
        "uncertainty_required_evidence_missing": 0.0,
        "uncertainty_authority_missing": 0.0,
        "uncertainty_graph_ambiguity": 0.0,
        "uncertainty_total": 0.0,
        "phase": "executable",
        "origin_price": 100.0,
        "trigger_bar_high": 100.5,
        "trigger_bar_low": 99.5,
        "invalidation_price": 99.0,
        "invalidation_source_id": f"invalidation:{token}",
        "draw_price": None if lsr_context_thesis else 103.0,
        "draw_id": None if lsr_context_thesis else f"draw:{token}",
        "liquidity_route_id": (
            None if lsr_context_thesis else f"route:{token}"
        ),
        "context_draw_id": (
            None if lsr_context_thesis else f"context-draw:{token}"
        ),
        "intermediate_liquidity_ids": "[]",
        "primary_deliverable_target_id": (
            None if lsr_context_thesis else f"target:{token}"
        ),
        "terminal_draw_id": (
            None if lsr_context_thesis else f"terminal:{token}"
        ),
        "authority_barrier_id": None,
        "authority_barrier_price": None,
        "path_blocker_ids": "[]",
        "source_path_ids": (
            "[]"
            if lsr_context_thesis
            else json.dumps(
                [f"target:{token}"],
                separators=(",", ":"),
            )
        ),
        "dfp_structure_id": None,
        "dfp_structure_confirmed_at": None,
        "dfp_h1_bos_id": None,
        "target_deadline_kind": target_deadline_kind,
        "deadline": pd.Timestamp("2022-02-20T16:00:00-05:00"),
        "symbol": "NQH2",
        "instrument_id": 1,
    }


def _rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2022-02-01T09:31:00-05:00")
    for playbook in (DFP, LSR):
        for dimension_index, dimension in enumerate(FITTED_DIMENSIONS):
            for index in range(8):
                sampled_at = start + pd.Timedelta(
                    days=dimension_index,
                    minutes=index,
                )
                rows.append(
                    {
                        **_identity_fields(
                            f"{playbook.value}:{dimension}:{index}",
                            playbook=playbook,
                            dimension=dimension,
                            runtime_candidate_token=(
                                f"{playbook.value}:{index}"
                            ),
                        ),
                        "sample_id": f"{playbook.value}:{dimension}:{index}",
                        "playbook": playbook.value,
                        "direction": "long" if index % 2 == 0 else "short",
                        "dimension": dimension,
                        "sampled_at": sampled_at,
                        "resolved_at": sampled_at + pd.Timedelta(minutes=5),
                        "raw_value": (index + 1) / 10.0,
                        "outcome_value": float(index >= 4),
                        "resolution": "causal_target_resolved",
                        "censored": False,
                        "fit_eligible": True,
                    }
                )
            # This future-unresolved row proves censorship is excluded instead
            # of being silently treated as a zero outcome.
            rows.append(
                {
                    **_identity_fields(
                        f"{playbook.value}:{dimension}:censored",
                        playbook=playbook,
                        dimension=dimension,
                    ),
                    "sample_id": f"{playbook.value}:{dimension}:censored",
                    "playbook": playbook.value,
                    "direction": "long",
                    "dimension": dimension,
                    "sampled_at": start + pd.Timedelta(days=10),
                    "resolved_at": start + pd.Timedelta(days=11),
                    "raw_value": 0.99,
                    "outcome_value": None,
                    "resolution": "window_end_censored",
                    "censored": True,
                    "fit_eligible": False,
                }
            )
        for dimension in ("sequence_progress", "uncertainty"):
            rows.append(
                {
                    **_identity_fields(
                        f"{playbook.value}:{dimension}:observed",
                        playbook=playbook,
                        dimension=dimension,
                    ),
                    "sample_id": f"{playbook.value}:{dimension}:observed",
                    "playbook": playbook.value,
                    "direction": "short",
                    "dimension": dimension,
                    "sampled_at": start,
                    "resolved_at": start,
                    "raw_value": 0.0 if dimension == "uncertainty" else 0.6,
                    "outcome_value": None,
                    "resolution": f"{dimension}_observed",
                    "censored": False,
                    "fit_eligible": False,
                }
            )
    return pd.DataFrame(rows)


def _write_run_manifest(
    tmp_path: Path,
    *,
    row_paths: tuple[Path, ...],
    field_types: dict[str, str] | None = None,
    identity_override: dict[str, object] | None = None,
    fit_admission_override: dict[str, int] | None = None,
    window_start: str = "2022-01-01T00:00:00-05:00",
    window_end_exclusive: str = "2023-01-01T00:00:00-05:00",
    window_role: str = "calibration",
) -> Path:
    registered_field_types = (
        BRAIN_CALIBRATION_FIELD_TYPES
        if field_types is None
        else field_types
    )
    bindings = resolve_model_bindings(MODEL_CONFIG)
    identity: dict[str, object] = {
        "recorder_schema_version": RECORDER_SCHEMA_VERSION,
        "registry_fingerprint": bindings["registry_hash"],
        "registry_schema_version": bindings["registry_schema_version"],
        "playbook_schema_versions": {
            key: value
            for key, value in sorted(
                bindings["playbook_schema_versions"].items()
            )
        },
    }
    if identity_override is not None:
        identity.update(identity_override)
    path = tmp_path / "run_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runner": "continuous_replay",
                "model_config": {
                    "identity": bindings["model_config_identity"],
                },
                "brain_calibration_identity": identity,
                "brain_calibration_fit_admission": (
                    fit_admission_override
                    if fit_admission_override is not None
                    else {
                        "minimum_dimension_units": 8,
                        "minimum_plan_valid_roots": 8,
                        "minimum_executable_episodes": 8,
                    }
                ),
                "window": {
                    "start": window_start,
                    "end_exclusive": window_end_exclusive,
                    "role": window_role,
                    "warmup_days": 3,
                },
                "output": {
                    "brain_calibration": True,
                    "brain_calibration_schema_version": (
                        RECORDER_SCHEMA_VERSION
                    ),
                    "stream_families": [
                        "brain_calibration_shards",
                        "decision_shards",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    shards: list[dict[str, object]] = []
    rows_total = 0
    for index, row_path in enumerate(row_paths):
        frame = pd.read_parquet(row_path)
        atomic_parquet(
            frame,
            row_path,
            field_types=registered_field_types,
        )
        relative = row_path.resolve().relative_to(tmp_path.resolve())
        rows_total += len(frame)
        shards.append(
            {
                "index": index,
                "path": str(relative),
                "rows": len(frame),
                "first_key": str(frame["sample_id"].iloc[0]),
                "last_key": str(frame["sample_id"].iloc[-1]),
                "sha256": sha256_file(row_path),
            }
        )
    stream_state = new_stream_state(registered_field_types)
    stream_state.update(
        {
            "rows": rows_total,
            "next_shard_index": len(shards),
            "committed_shards": shards,
        }
    )
    write_stream_manifest(
        tmp_path,
        "brain_calibration_shards",
        stream_state,
        artifact="continuous_development_brain_calibration_shards",
        bindings={"run_manifest": path.name},
    )
    return path


def _fit(tmp_path: Path, frame: pd.DataFrame) -> tuple[Path, dict[str, object]]:
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))
    output = tmp_path / "typed-calibration.json"
    payload = fit_typed_brain_calibration(
        row_paths=[rows],
        run_manifest=run_manifest,
        output=output,
        model_config=MODEL_CONFIG,
        validation_protocol=VALIDATION_PROTOCOL,
        bins=4,
        minimum_bin_samples=2,
        calibration_version="typed-test-fit",
    )
    return output, payload


def _validation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.copy()
    shift = (
        pd.Timestamp("2023-02-01T09:31:00-05:00")
        - values["sampled_at"].min()
    )
    values["sampled_at"] = values["sampled_at"] + shift
    values["resolved_at"] = values["resolved_at"] + shift
    values["deadline"] = values["deadline"] + shift
    return values


def _write_validation_run(tmp_path: Path, frame: pd.DataFrame) -> tuple[Path, Path]:
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(
        tmp_path,
        row_paths=(rows,),
        window_start="2023-01-01T00:00:00-05:00",
        window_end_exclusive="2024-01-01T00:00:00-05:00",
        window_role="brain_validation",
    )
    return rows, run_manifest


def test_fitter_builds_loader_valid_typed_artifact_without_pnl(tmp_path: Path) -> None:
    output, payload = _fit(tmp_path, _rows())
    bindings = resolve_model_bindings(MODEL_CONFIG)

    calibrator = TypedBrainCalibrator.from_file(
        output,
        expected_registry_hash=bindings["registry_hash"],
    )

    assert calibrator.status == "ready"
    assert payload["method"]["pnl_labels_used"] is False
    assert payload["method"]["threshold_search"] is False
    assert payload["method"]["sequence_progress"] == (
        "deterministic_passthrough_not_fitted"
    )
    assert payload["method"]["hard_rules_calibrated"] is False
    assert payload["method"]["context_strata_fitted"] is False
    assert payload["method"]["uncertainty_formula_version"] == (
        UNCERTAINTY_FORMULA_VERSION
    )
    assert payload["fit_admission"]["fit_allowed"] is True
    assert payload["fit_admission"]["phase_filter"] == (
        "none_dimension_specific_causal_capture"
    )
    assert payload["playbook_registry_schema_version"] == bindings[
        "registry_schema_version"
    ]
    assert payload["brain_target_protocol_versions"] == bindings[
        "playbook_schema_versions"
    ]
    for playbook in (DFP, LSR):
        dimensions = payload["playbooks"][playbook.value]["dimensions"]
        assert set(dimensions) == {*FITTED_DIMENSIONS, "uncertainty"}
        assert all(dimensions[name]["episodes"] == 8 for name in FITTED_DIMENSIONS)
        assert dimensions["thesis_strength"]["status"] == (
            "fitted_regularized_monotone_logistic"
        )
        assert dimensions["delivery_quality"]["status"] == (
            "fitted_regularized_monotone_logistic"
        )
        assert dimensions["location_quality"]["status"] == (
            "fitted_causal_target"
        )
        assert dimensions["entry_readiness"]["status"] == (
            "fitted_causal_target"
        )
        assert payload["playbooks"][playbook.value][
            "context_conditioning"
        ]["status"] == "calibration_pending_no_stratified_model"
        assert dimensions["uncertainty"]["status"] == (
            "authorized_formula_passthrough"
        )
        assert calibrator.apply(playbook, "uncertainty", 0.37) == 0.37
        assert calibrator.apply(playbook, "sequence_progress", 0.61) == 0.61
    assert payload["playbooks"][FAVR.value] == {
        "status": "parked_missing_natural_authority",
        "dimensions": {},
    }


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        (
            {"market_thesis_match_status": "unknown_status"},
            "market_thesis_match_status is invalid",
        ),
        (
            {
                "market_thesis_id": None,
                "bound_market_thesis_id": None,
                "market_thesis_root_id": None,
                "market_thesis_mechanism": None,
                "market_thesis_authority_relation": None,
                "market_thesis_binding_required": True,
                "market_thesis_action_bound": True,
                "market_thesis_match_status": "exact_root_bound",
            },
            "market thesis binding diagnostics are inconsistent",
        ),
        (
            {
                "market_thesis_id": "market-thesis:stale",
                "market_thesis_root_id": "root:stale",
                "market_thesis_mechanism": "directional_displacement",
                "market_thesis_authority_relation": "aligned",
                "playbook_match_strength": 0.8,
                "market_thesis_binding_required": True,
                "market_thesis_match_status": "no_open_thesis",
            },
            "market thesis binding diagnostics are inconsistent",
        ),
    ),
)
def test_fitter_rejects_inconsistent_market_thesis_diagnostics(
    updates: dict[str, object],
    message: str,
) -> None:
    frame = _rows().iloc[[0]].copy()
    for field, value in updates.items():
        frame.loc[:, field] = value

    with pytest.raises(ValueError, match=message):
        _validate_identity_columns(frame)


def test_fitter_requires_independent_calibration_units_not_only_revision_rows(
    tmp_path: Path,
) -> None:
    frame = _rows()
    mask = (
        frame["playbook"].eq(DFP.value)
        & frame["dimension"].eq("thesis_strength")
        & frame["fit_eligible"]
    )
    frame.loc[mask, "calibration_unit_id"] = "one-long-lived-unit"
    frame.loc[mask, "context_id"] = "one-long-lived-unit"
    frame.loc[mask, "context_thesis_id"] = "one-long-lived-unit"
    frame.loc[
        mask,
        "parent_context_thesis_id",
    ] = "one-long-lived-unit"
    frame.loc[mask, "direction"] = "long"
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))

    with pytest.raises(
        CalibrationError,
        match=r"1 unique calibration units values across 8 eligible rows",
    ):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
        )


def test_lsr_zone_children_do_not_expand_context_thesis_unit_count() -> None:
    values = _rows()
    values = values.loc[
        values["playbook"].eq(LSR.value)
        & values["dimension"].eq("thesis_strength")
        & values["fit_eligible"]
    ].iloc[:2].copy()
    values.loc[:, "context_thesis_id"] = "lsr-context-shared"
    values.loc[:, "parent_context_thesis_id"] = "lsr-context-shared"
    values.loc[:, "calibration_unit_id"] = "lsr-context-shared"
    values.loc[:, "episode_id"] = ["lsr-zone-a", "lsr-zone-b"]
    values.loc[:, "direction"] = "long"
    values.loc[:, "outcome_value"] = 1.0

    _raw, _outcome, unique_units, failures = _dimension_fit_diagnostics(
        values,
        playbook=LSR,
        dimension="thesis_strength",
        minimum_dimension_units=2,
    )

    assert unique_units == 1
    assert any("1 unique calibration units" in failure for failure in failures)


def test_fitter_rejects_lsr_child_episode_as_thesis_owner() -> None:
    frame = _rows().iloc[[0]].copy()
    frame.loc[:, "playbook"] = LSR.value
    frame.loc[:, "dimension"] = "thesis_strength"
    frame.loc[:, "calibration_unit_kind"] = "lsr_context_thesis"
    frame.loc[:, "calibration_unit_id"] = frame["episode_id"].astype(str)

    with pytest.raises(
        ValueError,
        match="calibration_unit_id disagrees with its causal dimension owner",
    ):
        _validate_identity_columns(frame)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("draw_id", "child-draw"),
        ("liquidity_route_id", "child-route"),
        ("source_path_ids", '["child-target"]'),
        ("invalidation_price", None),
        ("invalidation_source_id", None),
        ("deadline", None),
    ),
)
def test_fitter_rejects_lsr_context_thesis_child_target_custody(
    field: str,
    value: object,
) -> None:
    frame = _rows()
    lsr_thesis = frame["playbook"].eq(LSR.value) & frame["dimension"].eq(
        "thesis_strength"
    )
    frame.loc[lsr_thesis, field] = value

    with pytest.raises(
        ValueError,
        match="LSR Context thesis target custody is invalid",
    ):
        _validate_identity_columns(frame)


def test_fitter_rejects_conflicting_outcomes_for_one_causal_unit() -> None:
    values = _rows()
    values = values.loc[
        values["playbook"].eq(DFP.value)
        & values["dimension"].eq("thesis_strength")
        & values["fit_eligible"]
    ].copy()
    conflicting = values.iloc[[0]].copy()
    conflicting.loc[:, "outcome_value"] = (
        1.0 - float(conflicting.iloc[0]["outcome_value"])
    )
    values = pd.concat((values, conflicting), ignore_index=True)

    _raw, _outcome, unique_units, failures = _dimension_fit_diagnostics(
        values,
        playbook=DFP,
        dimension="thesis_strength",
        minimum_dimension_units=8,
    )

    assert unique_units == 8
    assert any(
        "conflicting outcomes for 1 independent calibration units" in failure
        for failure in failures
    )


@pytest.mark.parametrize("stale_schema", (14, 15, 16, 17))
def test_fitter_rejects_stale_calibration_rows_manifest(
    tmp_path: Path,
    stale_schema: int,
) -> None:
    rows = tmp_path / "rows.parquet"
    _rows().to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    payload["brain_calibration_identity"][
        "recorder_schema_version"
    ] = stale_schema
    payload["output"]["brain_calibration_schema_version"] = stale_schema
    run_manifest.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="current Brain calibration stream",
    ):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
        )


def test_fit_admission_keeps_dimension_specific_non_executable_phases(
    tmp_path: Path,
) -> None:
    frame = _rows()
    phases = {
        "thesis_strength": "forming",
        "location_quality": "waiting_location",
        "entry_readiness": "waiting_trigger",
        "delivery_quality": "executable",
    }
    for dimension, phase in phases.items():
        frame.loc[frame["dimension"].eq(dimension), "phase"] = phase

    _, payload = _fit(tmp_path, frame)
    admission = payload["fit_admission"]

    assert admission["fit_allowed"] is True
    assert admission["phase_filter"] == (
        "none_dimension_specific_causal_capture"
    )
    for playbook in (DFP, LSR):
        playbook_admission = admission["playbooks"][playbook.value]
        assert playbook_admission["unique_plan_delivery_valid_units"] == 8
        assert playbook_admission["unique_executable_units"] == 8
        for dimension, phase in phases.items():
            dimension_admission = playbook_admission["dimensions"][dimension]
            assert dimension_admission["status"] == "admitted"
            assert dimension_admission["fit_eligible_rows"] == 8
            assert phase in dimension_admission["phase_distribution"]
            assert payload["playbooks"][playbook.value]["dimensions"][
                dimension
            ]["episodes"] == 8


def test_formal_fit_uses_three_independent_manifest_limits(
    tmp_path: Path,
) -> None:
    frame = _rows()
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(
        tmp_path,
        row_paths=(rows,),
        fit_admission_override={
            "minimum_dimension_units": 8,
            "minimum_plan_valid_roots": 7,
            "minimum_executable_episodes": 6,
        },
    )

    payload = fit_typed_brain_calibration(
        row_paths=[rows],
        run_manifest=run_manifest,
        output=tmp_path / "artifact.json",
        model_config=MODEL_CONFIG,
        validation_protocol=VALIDATION_PROTOCOL,
        bins=4,
        minimum_bin_samples=2,
    )

    assert payload["fit_admission"]["fit_allowed"] is True
    assert payload["fit_admission"]["thresholds_enforced"] is True
    assert {
        "minimum_dimension_units",
        "minimum_plan_valid_roots",
        "minimum_executable_episodes",
    }.isdisjoint(payload["fit_admission"])


def test_fit_admission_requires_unique_plan_roots_per_playbook() -> None:
    frame = _rows()
    dfp_plan = (
        frame["playbook"].eq(DFP.value)
        & frame["playbook_plan_delivery_valid"]
    )
    frame.loc[dfp_plan, "market_thesis_root_id"] = "root:collapsed"

    admission = _fit_admission_funnel(
        frame,
        minimum_dimension_units=8,
        minimum_plan_valid_roots=8,
        minimum_executable_episodes=8,
        window_role="calibration",
    )
    dfp = admission["playbooks"][DFP.value]

    assert admission["fit_allowed"] is False
    assert dfp["unique_plan_delivery_valid_units"] == 1
    assert dfp["unique_executable_units"] == 8
    assert all(
        dimension["status"] == "admitted"
        for dimension in dfp["dimensions"].values()
    )
    assert any(
        "1 unique plan_delivery_valid roots" in reason
        for reason in dfp["runtime_funnel_failure_reasons"]
    )
    assert admission["playbooks"][LSR.value]["status"] == "admitted"


def test_fit_admission_excludes_unbound_plan_rows() -> None:
    frame = _rows()
    dfp_plan = (
        frame["playbook"].eq(DFP.value)
        & frame["playbook_plan_delivery_valid"]
    )
    frame.loc[dfp_plan, "bound_market_thesis_id"] = None
    frame.loc[dfp_plan, "market_thesis_action_bound"] = False
    frame.loc[dfp_plan, "market_thesis_match_status"] = (
        "root_identity_unbound"
    )

    admission = _fit_admission_funnel(
        frame,
        minimum_dimension_units=8,
        minimum_plan_valid_roots=8,
        minimum_executable_episodes=8,
        window_role="calibration",
    )
    dfp = admission["playbooks"][DFP.value]

    assert dfp["plan_delivery_valid_rows"] == 0
    assert dfp["unique_plan_delivery_valid_units"] == 0
    assert dfp["unique_executable_units"] == 0
    assert all(
        dimension["status"] == "admitted"
        for dimension in dfp["dimensions"].values()
    )
    assert admission["playbooks"][LSR.value]["status"] == "admitted"


def test_fit_admission_requires_unique_executable_episodes_per_playbook() -> None:
    frame = _rows()
    dfp_plan = (
        frame["playbook"].eq(DFP.value)
        & frame["playbook_plan_delivery_valid"]
    )
    frame.loc[dfp_plan, "episode_id"] = "episode:collapsed"

    admission = _fit_admission_funnel(
        frame,
        minimum_dimension_units=8,
        minimum_plan_valid_roots=8,
        minimum_executable_episodes=8,
        window_role="calibration",
    )
    dfp = admission["playbooks"][DFP.value]

    assert admission["fit_allowed"] is False
    assert dfp["unique_plan_delivery_valid_units"] == 8
    assert dfp["unique_executable_units"] == 1
    assert all(
        dimension["status"] == "admitted"
        for dimension in dfp["dimensions"].values()
    )
    assert any(
        "1 unique executable episodes" in reason
        for reason in dfp["runtime_funnel_failure_reasons"]
    )


def test_fit_admission_blocks_before_any_dimension_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _rows()
    mask = (
        frame["playbook"].eq(DFP.value)
        & frame["dimension"].eq("delivery_quality")
        & frame["fit_eligible"]
    )
    frame = frame.drop(index=frame.index[mask][1:])
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))

    def unexpected_fit(*args: object, **kwargs: object) -> object:
        raise AssertionError("a dimension fitter ran before admission")

    monkeypatch.setattr(
        "brain.scripts.fit_typed_brain_calibration._regularized_monotone_logistic_payload",
        unexpected_fit,
    )
    monkeypatch.setattr(
        "brain.scripts.fit_typed_brain_calibration._isotonic_dimension_payload",
        unexpected_fit,
    )
    with pytest.raises(CalibrationError, match="fit admission blocked"):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
        )


def test_runtime_fit_admission_blocks_before_any_dimension_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _rows()
    dfp_plan = (
        frame["playbook"].eq(DFP.value)
        & frame["playbook_plan_delivery_valid"]
    )
    frame.loc[dfp_plan, "phase"] = "waiting_trigger"
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))

    def unexpected_fit(*args: object, **kwargs: object) -> object:
        raise AssertionError("a dimension fitter ran before runtime admission")

    monkeypatch.setattr(
        "brain.scripts.fit_typed_brain_calibration._regularized_monotone_logistic_payload",
        unexpected_fit,
    )
    monkeypatch.setattr(
        "brain.scripts.fit_typed_brain_calibration._isotonic_dimension_payload",
        unexpected_fit,
    )
    with pytest.raises(
        CalibrationError,
        match=r"0 unique executable episodes with a valid plan",
    ):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
        )


def test_fit_admission_separates_delivery_barriers_and_revision_rows() -> None:
    frame = _rows()
    delivery_index = frame.index[
        frame["playbook"].eq(DFP.value)
        & frame["dimension"].eq("delivery_quality")
        & frame["fit_eligible"]
    ][0]
    frame.loc[delivery_index, "fit_eligible"] = False
    frame.loc[delivery_index, "hard_barrier_before_target"] = True

    source = frame.loc[
        frame["playbook"].eq(DFP.value)
        & frame["dimension"].eq("thesis_strength")
        & frame["fit_eligible"]
    ].iloc[0]
    revisions: list[pd.Series] = []
    for index in range(5):
        revision = source.copy()
        revision["sample_id"] = f"admission-revision:{index}"
        revision["evidence_revision_id"] = f"admission-evidence:{index}"
        revisions.append(revision)
    frame = pd.concat([frame, pd.DataFrame(revisions)], ignore_index=True)

    admission = _fit_admission_funnel(
        frame,
        minimum_dimension_units=8,
        minimum_plan_valid_roots=8,
        minimum_executable_episodes=8,
        window_role="calibration",
    )
    dfp_admission = admission["playbooks"][DFP.value]
    dfp = dfp_admission["dimensions"]

    assert dfp["delivery_quality"]["status"] == "blocked"
    assert dfp["delivery_quality"]["delivery_hard_barrier_rows"] == 1
    assert dfp["delivery_quality"]["delivery_hard_barrier_units"] == 1
    for dimension in (
        "thesis_strength",
        "location_quality",
        "entry_readiness",
    ):
        assert dfp[dimension]["status"] == "admitted"
        assert dfp[dimension]["delivery_hard_barrier_rows"] == 0
    assert dfp["thesis_strength"]["fit_eligible_rows"] == 13
    assert (
        dfp["thesis_strength"]["fit_eligible_unique_causal_units"]
        == 8
    )
    assert dfp_admission["plan_delivery_valid_rows"] == 37
    assert dfp_admission["unique_plan_delivery_valid_units"] == 8
    assert dfp_admission["unique_executable_units"] == 8


def test_fitter_gives_each_calibration_unit_equal_total_revision_weight(
    tmp_path: Path,
) -> None:
    def with_revisions(copies: int) -> pd.DataFrame:
        frame = _rows()
        additions: list[pd.Series] = []
        for dimension in ("thesis_strength", "location_quality"):
            source = frame.loc[
                frame["playbook"].eq(DFP.value)
                & frame["dimension"].eq(dimension)
                & frame["fit_eligible"]
            ].iloc[0]
            revisions = (
                (0.1, "low"),
                (0.9, "high"),
            )
            for raw_value, label in revisions:
                for revision in range(copies):
                    row = source.copy()
                    row["sample_id"] = (
                        f"weighted:{dimension}:{label}:{revision}"
                    )
                    row["evidence_revision_id"] = (
                        f"weighted-evidence:{dimension}:{label}:{revision}"
                    )
                    row["raw_value"] = raw_value
                    additions.append(row)
            frame = frame.drop(index=source.name)
        return pd.concat(
            [frame, pd.DataFrame(additions)],
            ignore_index=True,
        )

    compact_dir = tmp_path / "compact"
    repeated_dir = tmp_path / "repeated"
    compact_dir.mkdir()
    repeated_dir.mkdir()
    _, compact = _fit(compact_dir, with_revisions(1))
    _, repeated = _fit(repeated_dir, with_revisions(10))

    compact_dimensions = compact["playbooks"][DFP.value]["dimensions"]
    repeated_dimensions = repeated["playbooks"][DFP.value]["dimensions"]
    for dimension in ("thesis_strength", "location_quality"):
        left = compact_dimensions[dimension]
        right = repeated_dimensions[dimension]
        assert left["episodes"] == right["episodes"] == 8
        assert left["rows"] == 9
        assert right["rows"] == 27
        assert left["calibration_unit_weighting"] == (
            "equal_total_weight_per_causal_owner"
        )
        assert right["calibration_unit_weighting"] == left[
            "calibration_unit_weighting"
        ]
        assert left["outcome_mean"] == pytest.approx(right["outcome_mean"])
        assert left["raw_brier_descriptive_only"] == pytest.approx(
            right["raw_brier_descriptive_only"]
        )
        assert [point["raw_value"] for point in left["points"]] == pytest.approx(
            [point["raw_value"] for point in right["points"]]
        )
        assert [
            point["calibrated_value"] for point in left["points"]
        ] == pytest.approx(
            [point["calibrated_value"] for point in right["points"]]
        )
        assert [point["setup_weight"] for point in left["points"]] == pytest.approx(
            [point["setup_weight"] for point in right["points"]]
        )
    assert compact_dimensions["thesis_strength"]["coefficients"] == pytest.approx(
        repeated_dimensions["thesis_strength"]["coefficients"]
    )
    assert compact["method"]["fit_admission_threshold_source"] == (
        "source_run_manifest.brain_calibration_fit_admission"
    )
    assert {
        "minimum_dimension_units",
        "minimum_plan_valid_roots",
        "minimum_executable_episodes",
    }.isdisjoint(compact["method"])
    assert compact["method"]["revision_weighting"] == (
        "equal_total_weight_per_causal_owner"
    )


def test_fitter_fails_closed_when_one_dimension_has_one_raw_level(tmp_path: Path) -> None:
    frame = _rows()
    mask = frame["playbook"].eq(DFP.value) & frame["dimension"].eq(
        "thesis_strength"
    ) & frame["fit_eligible"]
    frame.loc[mask, "raw_value"] = 0.5
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))

    with pytest.raises(CalibrationError, match="fewer than two unique raw values"):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
        )


def test_fitter_keeps_large_equal_raw_ties_in_one_support_level(
    tmp_path: Path,
) -> None:
    frame = _rows()
    mask = (
        frame["playbook"].eq(LSR.value)
        & frame["dimension"].eq("thesis_strength")
        & frame["fit_eligible"]
    )
    indices = frame.index[mask].tolist()
    frame.loc[indices[:4], "raw_value"] = 0.0
    frame.loc[indices[4:], "raw_value"] = [0.2, 0.4, 0.6, 0.8]

    _, payload = _fit(tmp_path, frame)
    points = payload["playbooks"][LSR.value]["dimensions"][
        "thesis_strength"
    ]["points"]

    raw_values = [point["raw_value"] for point in points]
    assert len(raw_values) >= 2
    assert all(
        right > left
        for left, right in zip(raw_values[:-1], raw_values[1:])
    )
    assert any(point["episodes"] >= 4 for point in points)


def test_fitter_accepts_valid_identity_lists_without_byte_canonicalization(
    tmp_path: Path,
) -> None:
    frame = _rows()
    lsr_thesis = frame["playbook"].eq(LSR.value) & frame["dimension"].eq(
        "thesis_strength"
    )
    frame.loc[~lsr_thesis, "intermediate_liquidity_ids"] = (
        '[ "liquidity-1" ]'
    )
    descriptive = frame.index[
        frame["dimension"].eq("sequence_progress")
    ][0]
    frame.loc[descriptive, "path_blocker_ids"] = (
        "[\n  \"blocker-1\"\n]"
    )
    frame.loc[descriptive, "hard_barrier_before_target"] = True
    frame.loc[descriptive, "obstruction_distance_R"] = 0.5
    frame.loc[descriptive, "free_path_R"] = 0.75
    frame.loc[~lsr_thesis, "source_path_ids"] = (
        '[ "source-1", "target-1" ]'
    )

    _, payload = _fit(tmp_path, frame)

    assert payload["status"] == "ready"


def test_fitter_rejects_invalid_context_covariates(tmp_path: Path) -> None:
    frame = _rows()
    frame.loc[frame.index[0], "soft_obstruction_count"] = -1
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))

    with pytest.raises(
        ValueError,
        match="soft_obstruction_count must contain non-negative integers",
    ):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
        )


def test_coverage_only_reports_units_without_enforcing_fit_minimum(
    tmp_path: Path,
) -> None:
    frame = _rows()
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))
    output = tmp_path / "coverage.json"

    payload = fit_typed_brain_calibration(
        row_paths=[rows],
        run_manifest=run_manifest,
        output=output,
        model_config=MODEL_CONFIG,
        validation_protocol=VALIDATION_PROTOCOL,
        coverage_only=True,
    )

    assert payload["status"] == "descriptive_coverage_only"
    assert payload["fitted"] is False
    assert payload["fit_admission"]["status"] == "coverage_observed"
    assert payload["fit_admission"]["data_ready"] is None
    assert payload["fit_admission"]["window_fit_authorized"] is True
    assert payload["fit_admission"]["fit_allowed"] is False
    assert payload["fit_admission"]["thresholds_enforced"] is False
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    for playbook in (DFP, LSR):
        coverage = payload["playbooks"][playbook.value]
        assert coverage["rows"] == 38
        assert coverage["unique_calibration_units"] == 38
        assert coverage["fit_eligible_calibration_units"] == 32
        assert coverage["strata"]
        assert all(
            {
                "rows",
                "unique_calibration_units",
                "fit_eligible_calibration_units",
                "positive_outcome_units",
                "negative_outcome_units",
                "censored_units",
            }.issubset(stratum)
            for stratum in coverage["strata"]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("obstruction_distance_R", False),
        ("free_path_R", True),
    ),
)
def test_fitter_context_validation_rejects_boolean_distance_values(
    field: str,
    value: bool,
) -> None:
    frame = _rows()
    frame.at[frame.index[0], field] = value

    with pytest.raises(
        ValueError,
        match=rf"{field} must be null or a non-negative finite number",
    ):
        _validate_identity_columns(frame)


@pytest.mark.parametrize(
    ("hard_barrier", "blocker_ids"),
    (
        (True, "[]"),
        (False, '["blocker:1"]'),
    ),
)
def test_fitter_rejects_hard_barrier_blocker_identity_mismatch(
    tmp_path: Path,
    hard_barrier: bool,
    blocker_ids: str,
) -> None:
    frame = _rows()
    lsr_thesis = frame["playbook"].eq(LSR.value) & frame["dimension"].eq(
        "thesis_strength"
    )
    frame.loc[~lsr_thesis, "hard_barrier_before_target"] = hard_barrier
    frame.loc[~lsr_thesis, "path_blocker_ids"] = blocker_ids
    frame.loc[~lsr_thesis, "free_path_R"] = 0.75
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))

    with pytest.raises(
        ValueError,
        match="hard barrier state disagrees with its frozen blocker geometry",
    ):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
        )


@pytest.mark.parametrize(
    ("field", "wrong_type", "value"),
    (
        ("obstruction_distance_R", "large_string", "0.75"),
        ("soft_obstruction_count", "large_string", "0"),
        ("path_blocker_ids", "bool", False),
    ),
)
def test_fitter_rejects_self_consistent_noncanonical_obstruction_field_type(
    tmp_path: Path,
    field: str,
    wrong_type: str,
    value: object,
) -> None:
    frame = _rows()
    frame[field] = value
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    field_types = dict(BRAIN_CALIBRATION_FIELD_TYPES)
    field_types[field] = wrong_type
    run_manifest = _write_run_manifest(
        tmp_path,
        row_paths=(rows,),
        field_types=field_types,
    )

    with pytest.raises(
        ValueError,
        match="shard manifest omits the current typed schema",
    ):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            coverage_only=True,
        )


def test_fitter_rejects_hard_barrier_without_obstruction_distance(
    tmp_path: Path,
) -> None:
    frame = _rows()
    lsr_thesis = frame["playbook"].eq(LSR.value) & frame["dimension"].eq(
        "thesis_strength"
    )
    frame.loc[~lsr_thesis, "path_blocker_ids"] = '["blocker:1"]'
    frame.loc[~lsr_thesis, "hard_barrier_before_target"] = True
    frame.loc[~lsr_thesis, "free_path_R"] = 0.75
    frame.loc[~lsr_thesis, "obstruction_distance_R"] = None
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))

    with pytest.raises(
        ValueError,
        match="hard barrier state disagrees with its frozen blocker geometry",
    ):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            coverage_only=True,
        )


def test_fitter_rejects_fit_eligible_hard_barrier_delivery(
    tmp_path: Path,
) -> None:
    frame = _rows()
    index = frame.index[
        frame["dimension"].eq("delivery_quality")
        & frame["fit_eligible"]
    ][0]
    frame.loc[index, "path_blocker_ids"] = '["blocker:authority"]'
    frame.loc[index, "hard_barrier_before_target"] = True
    frame.loc[index, "obstruction_distance_R"] = 0.5
    frame.loc[index, "free_path_R"] = 0.4
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))

    with pytest.raises(
        ValueError,
        match="hard-barrier delivery rows are descriptive",
    ):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            coverage_only=True,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            {"deadline": pd.NaT},
            "fitted calibration targets require a frozen deadline",
        ),
        (
            {"target_deadline_kind": "plan_deadline"},
            "fitted calibration target uses the wrong causal deadline kind",
        ),
        (
            {
                "deadline": pd.Timestamp("2022-02-01T09:34:00-05:00"),
            },
            "fitted calibration targets cannot resolve after their deadline",
        ),
    ),
)
def test_fitter_rejects_invalid_fitted_deadline_contract(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    frame = _rows()
    thesis = frame.index[
        frame["dimension"].eq("thesis_strength")
        & frame["fit_eligible"]
    ][0]
    for field, value in mutation.items():
        frame.loc[thesis, field] = value
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))

    with pytest.raises(ValueError, match=message):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            coverage_only=True,
        )


def test_fixed_2024_trial_is_coverage_only_and_cannot_fit_artifact(
    tmp_path: Path,
) -> None:
    frame = _rows()
    shift = (
        pd.Timestamp("2024-01-02T09:31:00-05:00")
        - frame["sampled_at"].min()
    )
    frame["sampled_at"] = frame["sampled_at"] + shift
    frame["resolved_at"] = frame["resolved_at"] + shift
    frame["deadline"] = pd.Timestamp("2024-01-31T23:59:00-05:00")
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(
        tmp_path,
        row_paths=(rows,),
        window_start="2024-01-01T00:00:00-05:00",
        window_end_exclusive="2024-02-01T00:00:00-05:00",
        window_role="brain_calibration_trial",
    )

    coverage = fit_typed_brain_calibration(
        row_paths=[rows],
        run_manifest=run_manifest,
        output=tmp_path / "coverage.json",
        model_config=MODEL_CONFIG,
        validation_protocol=VALIDATION_PROTOCOL,
        coverage_only=True,
    )
    assert coverage["status"] == "descriptive_coverage_only"
    assert coverage["training_window_role"] == "brain_calibration_trial"
    assert coverage["fit_admission"]["window_fit_authorized"] is False
    assert coverage["fit_admission"]["fit_allowed"] is False

    with pytest.raises(
        ValueError,
        match="fitting may use only 2022 calibration rows",
    ):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
        )


def test_frozen_2022_artifact_is_applied_without_refit_to_2023_strata(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training"
    training.mkdir()
    artifact, _ = _fit(training, _rows())
    artifact_before = artifact.read_bytes()

    validation = tmp_path / "validation"
    validation.mkdir()
    frame = _rows()
    shift = (
        pd.Timestamp("2023-02-01T09:31:00-05:00")
        - frame["sampled_at"].min()
    )
    frame["sampled_at"] = frame["sampled_at"] + shift
    frame["resolved_at"] = frame["resolved_at"] + shift
    frame["deadline"] = frame["deadline"] + shift
    frame.loc[frame["direction"].eq("short"), "global_market_mode"] = (
        "transition"
    )
    rows = validation / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(
        validation,
        row_paths=(rows,),
        window_start="2023-01-01T00:00:00-05:00",
        window_end_exclusive="2024-01-01T00:00:00-05:00",
        window_role="brain_validation",
    )

    output = validation / "frozen-oos.json"
    payload = evaluate_frozen_typed_brain_calibration(
        row_paths=[rows],
        run_manifest=run_manifest,
        artifact=artifact,
        output=output,
        model_config=MODEL_CONFIG,
        validation_protocol=VALIDATION_PROTOCOL,
    )

    assert payload["status"] == "frozen_oos_evaluation_complete"
    assert payload["method"]["mapping_fitted"] is False
    assert payload["fitted"] is False
    assert artifact.read_bytes() == artifact_before
    result = payload["playbooks"][DFP.value]["dimensions"][
        "thesis_strength"
    ]
    assert result["overall"]["status"] == "evaluated"
    assert result["overall"]["weighted_auc"] is not None
    assert {
        item["value"]
        for item in result["strata"]["global_market_mode"]
    } == {"directional", "transition"}
    assert result["strata"]["calendar_month"][0]["value"] == "2023-02"


def test_frozen_oos_evaluator_rejects_non_validation_rows(tmp_path: Path) -> None:
    training = tmp_path / "training"
    training.mkdir()
    artifact, _ = _fit(training, _rows())
    rows = training / "rows.parquet"

    with pytest.raises(ValueError, match="requires brain_validation rows"):
        evaluate_frozen_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=training / "run_manifest.json",
            artifact=artifact,
            output=tmp_path / "invalid.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
        )


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("unknown_dimension", "unsupported typed validation dimensions"),
        ("reverse_clock", "resolve before they were sampled"),
        ("missing_outcome", "require resolved outcomes"),
        ("out_of_range_outcome", "must lie in \\[0, 1\\]"),
    ),
)
def test_frozen_oos_evaluator_rejects_invalid_row_contracts(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    training = tmp_path / "training"
    validation = tmp_path / "validation"
    training.mkdir()
    validation.mkdir()
    artifact, _ = _fit(training, _rows())
    frame = _validation_rows(_rows())
    if failure == "unknown_dimension":
        frame.loc[frame.index[0], "dimension"] = "unknown_dimension"
    elif failure == "reverse_clock":
        frame.loc[frame.index[-1], "resolved_at"] = (
            frame.loc[frame.index[-1], "sampled_at"]
            - pd.Timedelta(minutes=1)
        )
    elif failure == "missing_outcome":
        frame.loc[frame.index[0], "outcome_value"] = None
    else:
        frame.loc[frame.index[0], "outcome_value"] = 1.1
    rows, run_manifest = _write_validation_run(validation, frame)

    with pytest.raises(ValueError, match=message):
        evaluate_frozen_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            artifact=artifact,
            output=validation / "invalid.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("not_ready", "not a current frozen 2022 typed calibration mapping"),
        ("missing_dimension", "exactly the five calibrated dimensions"),
    ),
)
def test_frozen_oos_evaluator_requires_ready_complete_artifact(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    training = tmp_path / "training"
    validation = tmp_path / "validation"
    training.mkdir()
    validation.mkdir()
    artifact, _ = _fit(training, _rows())
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if mutation == "not_ready":
        payload["status"] = "identity_unvalidated"
    else:
        del payload["playbooks"][DFP.value]["dimensions"][
            "delivery_quality"
        ]
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    rows, run_manifest = _write_validation_run(
        validation,
        _validation_rows(_rows()),
    )

    with pytest.raises((ValueError, CalibrationError), match=message):
        evaluate_frozen_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            artifact=artifact,
            output=validation / "invalid.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
        )


@pytest.mark.parametrize(
    "protected_name",
    ("row", "run_manifest", "artifact", "model", "protocol"),
)
def test_frozen_oos_evaluator_refuses_to_overwrite_any_input(
    tmp_path: Path,
    protected_name: str,
) -> None:
    inputs = {
        "row": tmp_path / "row.parquet",
        "run_manifest": tmp_path / "run_manifest.json",
        "artifact": tmp_path / "artifact.json",
        "model": tmp_path / "model.json",
        "protocol": tmp_path / "protocol.json",
    }
    for name, path in inputs.items():
        path.write_bytes(f"protected:{name}".encode("utf-8"))
    protected = inputs[protected_name]
    before = protected.read_bytes()

    with pytest.raises(ValueError, match="must not overwrite an input"):
        evaluate_frozen_typed_brain_calibration(
            row_paths=[inputs["row"]],
            run_manifest=inputs["run_manifest"],
            artifact=inputs["artifact"],
            output=protected,
            model_config=inputs["model"],
            validation_protocol=inputs["protocol"],
        )

    assert protected.read_bytes() == before
    assert not protected.with_name(f".{protected.name}.tmp").exists()


def test_frozen_oos_evaluator_detects_artifact_change_before_atomic_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = tmp_path / "training"
    validation = tmp_path / "validation"
    training.mkdir()
    validation.mkdir()
    artifact, _ = _fit(training, _rows())
    rows, run_manifest = _write_validation_run(
        validation,
        _validation_rows(_rows()),
    )
    output = validation / "frozen-oos.json"
    actual_hash = sha256_file(artifact)
    calls = 0

    def changing_hash(path: str | Path) -> str:
        nonlocal calls
        assert Path(path).resolve() == artifact.resolve()
        calls += 1
        return actual_hash if calls == 1 else "0" * 64

    monkeypatch.setattr(oos_evaluator, "sha256_file", changing_hash)

    with pytest.raises(RuntimeError, match="changed before OOS output commit"):
        evaluate_frozen_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            artifact=artifact,
            output=output,
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
        )

    assert not output.exists()
    assert not output.with_name(f".{output.name}.tmp").exists()


def test_frozen_oos_evaluator_atomic_success_leaves_no_temporary_file(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training"
    validation = tmp_path / "validation"
    training.mkdir()
    validation.mkdir()
    artifact, _ = _fit(training, _rows())
    rows, run_manifest = _write_validation_run(
        validation,
        _validation_rows(_rows()),
    )
    output = validation / "frozen-oos.json"

    payload = evaluate_frozen_typed_brain_calibration(
        row_paths=[rows],
        run_manifest=run_manifest,
        artifact=artifact,
        output=output,
        model_config=MODEL_CONFIG,
        validation_protocol=VALIDATION_PROTOCOL,
    )

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert not output.with_name(f".{output.name}.tmp").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "calibration_unit_kind",
            "trigger_entry_path",
            "calibration_unit_kind disagrees with its causal dimension owner",
        ),
        (
            "calibration_unit_id",
            "tampered-owner",
            "calibration_unit_id disagrees with its causal dimension owner",
        ),
    ),
)
def test_fitter_rejects_tampered_calibration_unit_contract(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    frame = _rows()
    mask = (
        frame["playbook"].eq(DFP.value)
        & frame["dimension"].eq("thesis_strength")
    )
    frame.loc[mask, field] = value
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))

    with pytest.raises(ValueError, match=message):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
        )


def test_reliability_bins_choose_a_legal_boundary_before_a_trailing_tie() -> None:
    points = monotone_reliability_points(
        [0.1] * 3 + [0.9] * 7,
        [0.0] * 3 + [1.0] * 7,
        bins=2,
        minimum_bin_episodes=2,
    )

    assert [point.raw_probability for point in points] == pytest.approx([0.1, 0.9])
    assert [point.episodes for point in points] == [3, 7]


def test_weighted_pava_preserves_a_flat_map_across_pooled_bins() -> None:
    points = monotone_reliability_points(
        [0.1, 0.1, 0.3, 0.3, 0.7, 0.7],
        [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        bins=3,
        minimum_bin_episodes=2,
    )
    mapping = DimensionReliabilityMap(
        playbook=LSR,
        dimension="thesis_strength",
        episodes=6,
        points=tuple(
            DimensionReliabilityPoint(
                raw_value=point.raw_probability,
                calibrated_value=point.calibrated_probability,
                episodes=point.episodes,
            )
            for point in points
        ),
    )

    assert [point.raw_probability for point in points] == pytest.approx(
        [0.1, 0.3, 0.7]
    )
    assert points[0].calibrated_probability == pytest.approx(
        points[1].calibrated_probability
    )
    assert mapping.apply(0.2) == pytest.approx(points[0].calibrated_probability)
    assert mapping.apply(0.3) == pytest.approx(points[1].calibrated_probability)


def test_fitter_rejects_stale_registry_identity_in_run_manifest(
    tmp_path: Path,
) -> None:
    frame = _rows()
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(
        tmp_path,
        row_paths=(rows,),
        identity_override={"registry_fingerprint": "0" * 64},
    )

    with pytest.raises(ValueError, match="registry identity is stale"):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
        )


def test_fitter_rejects_incomplete_manifest_fit_admission(
    tmp_path: Path,
) -> None:
    frame = _rows()
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    payload["brain_calibration_fit_admission"].pop(
        "minimum_executable_episodes"
    )
    run_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fit admission is invalid"):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
        )


def test_fitter_rejects_run_manifest_without_registry_identity(
    tmp_path: Path,
) -> None:
    frame = _rows()
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))
    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    payload.pop("brain_calibration_identity")
    run_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="registry identity is stale"):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("status", "incomplete"),
        ("stream", "decision_shards"),
        ("binding", "another_run_manifest.json"),
    ),
)
def test_fitter_rejects_invalid_bound_stream_manifest(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    rows = tmp_path / "rows.parquet"
    _rows().to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))
    stream_manifest = tmp_path / "brain_calibration_shards.manifest.json"
    payload = json.loads(stream_manifest.read_text(encoding="utf-8"))
    if field == "binding":
        payload["bindings"]["run_manifest"] = value
    else:
        payload[field] = value
    stream_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="shard manifest is invalid"):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
        )


def test_fitter_rejects_shards_from_another_run(tmp_path: Path) -> None:
    first_run = tmp_path / "first"
    second_run = tmp_path / "second"
    first_run.mkdir()
    second_run.mkdir()
    first_rows = first_run / "rows.parquet"
    second_rows = second_run / "rows.parquet"
    _rows().to_parquet(first_rows, index=False)
    _rows().to_parquet(second_rows, index=False)
    first_manifest = _write_run_manifest(first_run, row_paths=(first_rows,))
    _write_run_manifest(second_run, row_paths=(second_rows,))

    with pytest.raises(ValueError, match="do not exactly match the bound run"):
        fit_typed_brain_calibration(
            row_paths=[second_rows],
            run_manifest=first_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
        )


@pytest.mark.parametrize(
    ("field", "outside_clock"),
    (
        ("sampled_at", pd.Timestamp("2021-12-31T23:59:00-05:00")),
        ("resolved_at", pd.Timestamp("2023-01-01T00:00:00-05:00")),
    ),
)
def test_fitter_rejects_rows_outside_bound_run_window(
    tmp_path: Path,
    field: str,
    outside_clock: pd.Timestamp,
) -> None:
    frame = _rows()
    frame.loc[frame.index[0], field] = outside_clock
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))

    with pytest.raises(ValueError, match="outside the bound run window"):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
        )


def test_fitter_rejects_manifest_window_outside_registered_split(
    tmp_path: Path,
) -> None:
    rows = tmp_path / "rows.parquet"
    _rows().to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(
        tmp_path,
        row_paths=(rows,),
        window_start="2021-12-31T00:00:00-05:00",
        window_end_exclusive="2024-01-02T00:00:00-05:00",
    )

    with pytest.raises(ValueError, match="preregistered windows"):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
        )


def test_fitter_verifies_bound_shard_hash_and_rows(tmp_path: Path) -> None:
    rows = tmp_path / "rows.parquet"
    _rows().to_parquet(rows, index=False)
    run_manifest = _write_run_manifest(tmp_path, row_paths=(rows,))
    stream_manifest = tmp_path / "brain_calibration_shards.manifest.json"
    payload = json.loads(stream_manifest.read_text(encoding="utf-8"))
    payload["shards"][0]["sha256"] = "0" * 64
    stream_manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="shard hash is invalid"):
        fit_typed_brain_calibration(
            row_paths=[rows],
            run_manifest=run_manifest,
            output=tmp_path / "artifact.json",
            model_config=MODEL_CONFIG,
            validation_protocol=VALIDATION_PROTOCOL,
            bins=4,
            minimum_bin_samples=2,
        )


def test_fitter_rejects_economic_labels_in_recorder_rows(tmp_path: Path) -> None:
    frame = _rows()
    frame["net_R"] = 0.0
    rows = tmp_path / "rows.parquet"
    frame.to_parquet(rows, index=False)
    with pytest.raises(
        ValueError,
        match="Parquet shard columns differ from the registered schema",
    ):
        _write_run_manifest(tmp_path, row_paths=(rows,))
