from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_continuous_replay import (
    _load_mbo_execution,
    _load_shadow_diagnostic_profile,
)
from smc_trader.model import Playbook
from smc_trader.shadow_outcome import (
    RECORDER_SCHEMA_VERSION as SHADOW_OUTCOME_RECORDER_SCHEMA_VERSION,
    SHADOW_DERIVED_SCHEMA_VERSION,
    SHADOW_OUTCOME_PROTOCOL,
)
from smc_trader.validation import ValidationProtocolError, load_validation_protocol


ROOT = Path(__file__).resolve().parents[1]
DATA_SPLITS = ROOT / "configs/data_splits.json"


def test_current_data_splits_preserve_causal_and_mbo_identities() -> None:
    protocol = load_validation_protocol(DATA_SPLITS)

    assert protocol.schema_version == 1
    assert protocol.causal_source.path.endswith(
        "nq_1m_previous_session_front_v2_3_2017_2026.parquet"
    )
    assert protocol.causal_source.sha256 == (
        "84c9ed4d1de379382bdc41e0fe02e3611373ba182cfeebe09e3832d29cbafd7b"
    )
    assert protocol.causal_source.manifest_sha256 == (
        "391415fa23abfde5281c60ad4d882b32ff284b1aedb6298e398ed72036a9042f"
    )
    assert protocol.mbo_identity.development_partition_manifest_sha256 == (
        "9fbaef324de51cdbc60e13ea07acb7015cce6c409545761d7148ec5f7e8045cd"
    )
    assert len(protocol.mbo_identity.development_execution_artifacts) == 2
    assert protocol.mbo_identity.sealed_sha256 == (
        "8927a904876d3a22af8dce7f3fe05d2ba574f2083435ba9bb477faafc45477b8"
    )
    assert protocol.mbo_identity.sealed_manifest_sha256 == (
        "cb04c2706a21bc4c293e6900ef7d34b0fecf13bd304a13dfbebf162278e17c6f"
    )
    assert protocol.brain_calibration_fit_admission.as_dict() == {
        "minimum_dimension_units": 200,
        "minimum_plan_valid_roots": 30,
        "minimum_executable_episodes": 20,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload["brain_calibration_fit_admission"].pop(
                "minimum_plan_valid_roots"
            ),
            "must explicitly contain",
        ),
        (
            lambda payload: payload["brain_calibration_fit_admission"].update(
                {"minimum_dimension_units": True}
            ),
            "minimum_dimension_units must be an integer",
        ),
        (
            lambda payload: payload["brain_calibration_fit_admission"].update(
                {"minimum_executable_episodes": 0}
            ),
            "minimum_executable_episodes must be at least 1",
        ),
    ),
)
def test_brain_fit_admission_requires_three_independent_explicit_limits(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    payload = json.loads(DATA_SPLITS.read_text(encoding="utf-8"))
    mutation(payload)
    candidate = tmp_path / "data_splits.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationProtocolError, match=message):
        load_validation_protocol(candidate)


def test_current_data_splits_keep_stage_boundaries_and_group4_windows() -> None:
    protocol = load_validation_protocol()

    assert tuple(protocol.ohlcv_windows) == (
        "development",
        "calibration",
        "brain_validation",
        "brain_calibration_trial",
        "rolling_oof",
        "sealed_holdout",
    )
    assert (
        protocol.classify_ohlcv(
            pd.Timestamp("2022-02-01", tz="America/New_York"),
            pd.Timestamp("2022-03-01", tz="America/New_York"),
        ).role
        == "calibration"
    )
    assert (
        protocol.classify_ohlcv(
            pd.Timestamp("2023-01-01", tz="America/New_York"),
            pd.Timestamp("2024-01-01", tz="America/New_York"),
        ).role
        == "brain_validation"
    )
    assert (
        protocol.classify_ohlcv(
            pd.Timestamp("2024-01-01", tz="America/New_York"),
            pd.Timestamp("2024-02-01", tz="America/New_York"),
        ).role
        == "brain_calibration_trial"
    )
    assert (
        protocol.classify_ohlcv(
            pd.Timestamp("2024-02-01", tz="America/New_York"),
            pd.Timestamp("2024-03-01", tz="America/New_York"),
        ).role
        == "rolling_oof"
    )
    assert (
        protocol.classify_mbo(
            pd.Timestamp("2024-06-02", tz="UTC"),
            pd.Timestamp("2024-06-03", tz="UTC"),
        ).role
        == "development"
    )
    assert tuple(
        window.role
        for window in protocol.fixed_development_windows["mature_range_coverage"]
    ) == (
        "2017-june",
        "2018-june",
        "2019-june",
        "2020-june",
        "2021-june",
    )
    with pytest.raises(ValidationProtocolError):
        protocol.classify_ohlcv(
            pd.Timestamp("2026-03-31", tz="America/New_York"),
            pd.Timestamp("2026-04-02", tz="America/New_York"),
        )


def test_january_2024_shadow_profile_is_fixed_and_outcome_blind() -> None:
    payload = json.loads(DATA_SPLITS.read_text(encoding="utf-8"))
    profile = payload["shadow_diagnostic_profiles"][
        "brain_playbook_reverse_validation_2024_01"
    ]

    assert profile["start"] == "2024-01-01T00:00:00-05:00"
    assert profile["end_exclusive"] == "2024-02-01T00:00:00-05:00"
    assert profile["warmup_calendar_days"] == 14
    assert profile["playbooks_frozen_during_run"] is True
    assert profile["action_disabled_playbooks"] == [
        Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    ]
    assert profile["recorder_schema_version"] == (
        SHADOW_OUTCOME_RECORDER_SCHEMA_VERSION
    )
    assert profile["derived_schema_version"] == SHADOW_DERIVED_SCHEMA_VERSION
    assert profile["protocol_version"] == (
        SHADOW_OUTCOME_PROTOCOL["protocol_version"]
    )
    for field in (
        "threshold_search",
        "calibration_fit_allowed",
        "pnl_used",
        "mbo_used",
        "future_path_visible_to_model",
        "shadow_output_affects_action",
    ):
        assert profile[field] is False
    assert set(profile["candidate_sources"]) == {
        "playbook_executable",
        "open_market_thesis_revision",
        "confirmed_bos",
        "displacement_active",
        "eligible_entry_fvg",
        "eligible_entry_order_block",
        "first_entry_fvg",
        "first_entry_order_block",
        "liquidity_sweep",
        "mature_range_reentry",
        "qualified_micro_bos",
        "qualified_reacceptance",
    }
    with pytest.raises(ValueError, match="runtime action policy"):
        _load_shadow_diagnostic_profile(
            DATA_SPLITS,
            start=pd.Timestamp(profile["start"]),
            end=pd.Timestamp(profile["end_exclusive"]),
            warmup_days=profile["warmup_calendar_days"],
        )
    selected_name, selected = _load_shadow_diagnostic_profile(
        DATA_SPLITS,
        start=pd.Timestamp(profile["start"]),
        end=pd.Timestamp(profile["end_exclusive"]),
        warmup_days=profile["warmup_calendar_days"],
        action_disabled_playbooks=(Playbook.LIQUIDITY_SWEEP_REVERSAL,),
    )
    assert selected_name == "brain_playbook_reverse_validation_2024_01"
    assert selected == profile


def test_exact_five_day_shadow_profile_is_registered_and_selected() -> None:
    payload = json.loads(DATA_SPLITS.read_text(encoding="utf-8"))
    profile_name = "brain_playbook_reverse_validation_2024_01_07_12"
    profile = payload["shadow_diagnostic_profiles"][profile_name]

    assert profile["start"] == "2024-01-07T18:00:00-05:00"
    assert profile["end_exclusive"] == "2024-01-12T18:00:00-05:00"
    assert profile["warmup_calendar_days"] == 7
    assert profile["action_disabled_playbooks"] == [
        Playbook.LIQUIDITY_SWEEP_REVERSAL.value
    ]
    with pytest.raises(ValueError, match="runtime action policy"):
        _load_shadow_diagnostic_profile(
            DATA_SPLITS,
            start=pd.Timestamp(profile["start"]),
            end=pd.Timestamp(profile["end_exclusive"]),
            warmup_days=7,
        )
    selected_name, selected = _load_shadow_diagnostic_profile(
        DATA_SPLITS,
        start=pd.Timestamp(profile["start"]),
        end=pd.Timestamp(profile["end_exclusive"]),
        warmup_days=7,
        action_disabled_playbooks=(Playbook.LIQUIDITY_SWEEP_REVERSAL,),
    )

    assert selected_name == profile_name
    assert selected == profile


def test_exact_2022_seven_session_census_profile_is_registered_and_non_actionable() -> None:
    payload = json.loads(DATA_SPLITS.read_text(encoding="utf-8"))
    profile_name = "brain_episode_sequence_census_2022_01_18_27"
    profile = payload["shadow_diagnostic_profiles"][profile_name]
    start = pd.Timestamp("2022-01-18T18:00:00-05:00")
    end = pd.Timestamp("2022-01-27T18:00:00-05:00")

    assert profile["start"] == start.isoformat()
    assert profile["end_exclusive"] == end.isoformat()
    assert profile["warmup_calendar_days"] == 7
    assert profile["allowed_ohlcv_role"] == "calibration"
    assert profile["playbooks_frozen_during_run"] is True
    for field in (
        "threshold_search",
        "calibration_fit_allowed",
        "pnl_used",
        "mbo_used",
        "future_path_visible_to_model",
        "shadow_output_affects_action",
    ):
        assert profile[field] is False

    selected_name, selected = _load_shadow_diagnostic_profile(
        DATA_SPLITS,
        start=start,
        end=end,
        warmup_days=7,
    )
    assert selected_name == profile_name
    assert selected == profile
    assert load_validation_protocol(DATA_SPLITS).classify_ohlcv(
        start,
        end,
    ).role == "calibration"

    with pytest.raises(ValueError, match="exact registered window"):
        _load_shadow_diagnostic_profile(
            DATA_SPLITS,
            start=start,
            end=end,
            warmup_days=14,
        )


def test_january_2022_census_extension_is_preregistered_and_non_actionable() -> None:
    payload = json.loads(DATA_SPLITS.read_text(encoding="utf-8"))
    profile_name = "brain_episode_sequence_census_2022_01"
    profile = payload["shadow_diagnostic_profiles"][profile_name]
    start = pd.Timestamp("2022-01-01T00:00:00-05:00")
    end = pd.Timestamp("2022-02-01T00:00:00-05:00")

    assert profile["start"] == start.isoformat()
    assert profile["end_exclusive"] == end.isoformat()
    assert profile["warmup_calendar_days"] == 14
    assert profile["allowed_ohlcv_role"] == "calibration"
    assert profile["playbooks_frozen_during_run"] is True
    for field in (
        "threshold_search",
        "calibration_fit_allowed",
        "pnl_used",
        "mbo_used",
        "future_path_visible_to_model",
        "shadow_output_affects_action",
    ):
        assert profile[field] is False

    selected_name, selected = _load_shadow_diagnostic_profile(
        DATA_SPLITS,
        start=start,
        end=end,
        warmup_days=14,
    )
    assert selected_name == profile_name
    assert selected == profile
    assert load_validation_protocol(DATA_SPLITS).classify_ohlcv(
        start,
        end,
    ).role == "calibration"


def test_lsr_multi_zone_five_session_profile_is_fixed_and_non_actionable() -> None:
    payload = json.loads(DATA_SPLITS.read_text(encoding="utf-8"))
    profile_name = "brain_lsr_multi_zone_2022_01_02_07"
    profile = payload["shadow_diagnostic_profiles"][profile_name]
    start = pd.Timestamp("2022-01-02T18:00:00-05:00")
    end = pd.Timestamp("2022-01-07T18:00:00-05:00")

    assert profile["start"] == start.isoformat()
    assert profile["end_exclusive"] == end.isoformat()
    assert profile["warmup_calendar_days"] == 14
    assert profile["allowed_ohlcv_role"] == "calibration"
    assert "eligible-zone to first-pullback recovery" in profile["purpose"]
    assert "neither OOS nor edge validation" in profile["purpose"]
    assert profile["registration_at"] == "2026-08-16"
    assert profile["playbooks_frozen_during_run"] is True
    for field in (
        "threshold_search",
        "calibration_fit_allowed",
        "pnl_used",
        "mbo_used",
        "future_path_visible_to_model",
        "shadow_output_affects_action",
    ):
        assert profile[field] is False

    selected_name, selected = _load_shadow_diagnostic_profile(
        DATA_SPLITS,
        start=start,
        end=end,
        warmup_days=14,
    )
    assert selected_name == profile_name
    assert selected == profile
    assert load_validation_protocol(DATA_SPLITS).classify_ohlcv(
        start,
        end,
    ).role == "calibration"


def test_shadow_profile_rejects_unregistered_subwindow() -> None:
    with pytest.raises(ValueError, match="exact registered window"):
        _load_shadow_diagnostic_profile(
            DATA_SPLITS,
            start=pd.Timestamp("2024-01-08T18:00:00-05:00"),
            end=pd.Timestamp("2024-01-12T18:00:00-05:00"),
            warmup_days=7,
        )


def test_shadow_profile_candidate_sources_must_match_recorder_protocol(
    tmp_path: Path,
) -> None:
    payload = json.loads(DATA_SPLITS.read_text(encoding="utf-8"))
    profile = payload["shadow_diagnostic_profiles"][
        "brain_playbook_reverse_validation_2024_01"
    ]
    profile["candidate_sources"].remove("open_market_thesis_revision")
    candidate = tmp_path / "data_splits.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate_sources disagree"):
        _load_shadow_diagnostic_profile(
            candidate,
            start=pd.Timestamp(profile["start"]),
            end=pd.Timestamp(profile["end_exclusive"]),
            warmup_days=profile["warmup_calendar_days"],
            action_disabled_playbooks=(
                Playbook.LIQUIDITY_SWEEP_REVERSAL,
            ),
        )


def test_shadow_profile_rejects_legacy_custody_contract(
    tmp_path: Path,
) -> None:
    payload = json.loads(DATA_SPLITS.read_text(encoding="utf-8"))
    profile = payload["shadow_diagnostic_profiles"][
        "brain_lsr_multi_zone_2022_01_02_07"
    ]
    profile.update(
        {
            "recorder_schema_version": 7,
            "derived_schema_version": 8,
            "protocol_version": "shadow-candidate-outcome-1.6.0",
        }
    )
    candidate = tmp_path / "legacy-shadow-data-splits.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="profile identity is incompatible"):
        _load_shadow_diagnostic_profile(
            candidate,
            start=pd.Timestamp(profile["start"]),
            end=pd.Timestamp(profile["end_exclusive"]),
            warmup_days=profile["warmup_calendar_days"],
        )


def test_current_schema_rejects_changed_mbo_manifest_binding(tmp_path: Path) -> None:
    payload = json.loads(DATA_SPLITS.read_text(encoding="utf-8"))
    payload["sources"]["mbo"]["development"]["partition_manifest_sha256"] = "0" * 64
    candidate = tmp_path / "data_splits.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationProtocolError, match="SHA-256 mismatch"):
        load_validation_protocol(candidate)


def test_archived_protocol_shape_is_rejected(tmp_path: Path) -> None:
    candidate = tmp_path / "archived_protocol.json"
    candidate.write_text(json.dumps({"protocol_version": "archived-test"}))

    with pytest.raises(ValidationProtocolError, match="schema_version must be 1"):
        load_validation_protocol(candidate)


def test_registered_development_mbo_artifacts_load_by_exact_identity() -> None:
    protocol = load_validation_protocol(DATA_SPLITS)
    for artifact in protocol.mbo_identity.development_execution_artifacts:
        source = ROOT / artifact.path
        manifest_path = ROOT / artifact.manifest_path
        if not source.is_file() or not manifest_path.is_file():
            pytest.skip("registered development MBO artifact is not present")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        store, loaded_manifest = _load_mbo_execution(
            source,
            validation=protocol,
            start=pd.Timestamp(manifest["start"]),
            end=pd.Timestamp(manifest["end_exclusive"]),
            reveal_sealed_holdout=False,
        )
        assert len(store.frame) > 0
        assert loaded_manifest["output_sha256"] == artifact.sha256
